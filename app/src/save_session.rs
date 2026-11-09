//! Managed tree lifecycle. The chooser owns no user document until an explicit
//! create/open/import succeeds; background saves temporarily own the whole tree.
use super::*;
use crate::save_library::{Library, SaveEntry};

pub(super) struct SaveSession {
    pub enabled: bool,
    pub home: bool,
    pub settings: bool,
    pub catalog: Result<Library,String>,
    pub entries: Vec<SaveEntry>,
    pub warnings: Vec<String>,
    pub active: Option<SaveEntry>,
    pub lease:Option<crate::save_library::SaveLease>,
    pub dirty: bool,
    pub pending_save: bool,
    pub pending_leave:Option<crate::manual_save::PendingLeave>,
    pub approved_close:bool,
    pub error: Option<String>,
    pub ui: crate::save_library_ui::SaveLibraryUi,
}
impl SaveSession {
    pub fn new(enabled:bool)->Self{
        let args=std::env::args().collect::<Vec<_>>();
        let catalog=args.iter().position(|a|a=="--library-smoke-output"||a=="--redraw-smoke-output"||a=="--mode-switch-smoke-output"||a=="--manual-save-smoke-output"||a=="--reference-smoke-output")
            .and_then(|i|args.get(i+1)).map(|p|Library::at(PathBuf::from(p).with_extension("library")))
            .map(Ok).unwrap_or_else(Library::system);
        let mut s=Self{enabled,home:enabled,settings:false,catalog,entries:Vec::new(),warnings:Vec::new(),
            active:None,lease:None,dirty:false,
            pending_save:false,pending_leave:None,approved_close:false,error:None,ui:Default::default()};
        if enabled{s.refresh();} s
    }
    pub fn refresh(&mut self){
        match self.catalog.as_ref().map_err(Clone::clone).and_then(|l|l.list()){
            Ok(list)=>{self.entries=list.entries;self.warnings=list.warnings;},
            Err(error)=>self.error=Some(error),
        }
    }
    pub fn mark_dirty(&mut self){if self.enabled&&self.active.is_some(){self.dirty=true;}}
    pub fn mark_workspace_dirty(&mut self){self.mark_dirty();}

}

pub(super) enum LibraryWork {
    Create{library:Library,name:String},
    Open{library:Library,id:String},
    Import{library:Library,path:PathBuf,workspace:crate::workspace::WorkspaceState},
    Save{library:Library,entry:SaveEntry,document:Document,histories:RawHistoryStore},
    Rename{library:Library,id:String,name:String},
}
pub(super) enum LibraryOutcome {
    Opened(Result<(SaveEntry,Document,RawHistoryStore,crate::save_library::SaveLease),String>),
    Saved{document:Document,histories:RawHistoryStore,result:Result<SaveEntry,String>},
    Renamed(Result<SaveEntry,String>),
}
pub(super) fn perform(work:LibraryWork)->LibraryOutcome{
    match work{
        LibraryWork::Create{library,name}=>LibraryOutcome::Opened(file_attempt(||{
            let doc=Document::new();let history=RawHistoryStore::new();
            let entry=library.create(&name,&doc,&history)?;let lease=library.claim(&entry.id)?;Ok((entry,doc,history,lease))
        })),
        LibraryWork::Open{library,id}=>LibraryOutcome::Opened(file_attempt(||{
            let entry=library.list()?.entries.into_iter().find(|e|e.id==id).ok_or("存档已不存在，请刷新列表")?;
            let lease=library.claim(&id)?;let (doc,history)=project_file::load_project(&library.project_path(&id)?)?;Ok((entry,doc,history,lease))
        })),
        LibraryWork::Import{library,path,workspace}=>LibraryOutcome::Opened(file_attempt(||{
            let (doc,history)=project_file::load_project(&path)?;
            let name=path.file_stem().and_then(|s|s.to_str()).filter(|s|!s.trim().is_empty()).unwrap_or("导入的状态树");
            let entry=library.create_workspace(name,&workspace,&doc,&history)?;let lease=library.claim(&entry.id)?;Ok((entry,doc,history,lease))
        })),
        LibraryWork::Save{library,entry,document,histories}=>{
            let result=file_attempt(||library.save_workspace(&entry.id,&entry.display_name,&entry.workspace,&document,&histories));
            LibraryOutcome::Saved{document,histories,result}
        },
        LibraryWork::Rename{library,id,name}=>LibraryOutcome::Renamed(file_attempt(||library.rename(&id,&name))),
    }
}

impl PainterApp {
    pub(super) fn library_action(&mut self,action:crate::save_library_ui::Action){
        use crate::save_library_ui::Action;
        if self.file_busy_message.is_some()||!self.save_session.home{return;}
        if matches!(action,Action::Settings){self.save_session.settings=true;return;}
        if matches!(action,Action::OpenLogs){self.action(ui::UiAction::OpenLogs);return;}
        let library=match &self.save_session.catalog{Ok(l)=>l.clone(),Err(e)=>{self.save_session.error=Some(e.clone());return;}};
        let work=match action{
            Action::Create{name}=>LibraryWork::Create{library,name},
            Action::Open{id}=>LibraryWork::Open{library,id},
            Action::Rename{id,name}=>LibraryWork::Rename{library,id,name},
            Action::Import=>match ui::file_dialogs::choose_import(None){
                Ok(Some(path))=>LibraryWork::Import{library,path,workspace:Default::default()},Ok(None)=>return,
                Err(e)=>{self.save_session.error=Some(e);return;}
            },
            _=>return,
        };
        self.save_session.error=None;
        self.start_file_job("正在处理存档…",move|_|FileWork::Library(work));
    }
    pub(super) fn begin_library_save(&mut self){
        let Some(entry)=self.save_session.active.clone()else{return;};
        let library=match &self.save_session.catalog{Ok(l)=>l.clone(),Err(e)=>{self.library_failure(e.clone());return;}};
        self.save_session.pending_save=false;
        self.start_file_job("正在保存存档和预览…",move|app|FileWork::Library(LibraryWork::Save{
            library,entry,document:std::mem::take(&mut app.document),histories:std::mem::take(&mut app.histories),
        }));
    }
    pub(super) fn request_library_save(&mut self){
        if self.save_session.active.is_none(){return;}
        if self.running{
            self.save_session.pending_save=true;self.runtime.stop();self.stopping=true;
            self.set_file_busy(Some("正在停止生成并保存存档…".into()));
        }else{self.begin_library_save();}
    }
    pub(super) fn return_to_library(&mut self){
        self.request_leave(crate::manual_save::LeaveAction::Home);
    }
    pub(super) fn leave_saved_document(&mut self){
        self.cancel_reference_session();
        self.document=Document::new();self.histories=RawHistoryStore::new();self.parallel=None;
        self.pending_canvas_edit=None;self.cancel_global_edit();self.stage_map_node=None;self.redraw_view_key=None;
        self.editor.invalidate_history_after_deletion();self.editor.set_project_path(None);
        self.project_path=None;self.drawing_region=DrawingRegion::default();
        self.save_session.active=None;self.save_session.lease=None;self.save_session.home=true;self.save_session.settings=false;
        self.save_session.pending_leave=None;self.save_session.dirty=false;self.save_session.refresh();
        self.error=None;self.status="请选择或新建存档".into();
    }
    pub(super) fn library_failure(&mut self,error:String){
        self.save_session.error=Some(error.clone());self.error=Some(error.clone());
        self.save_session.pending_save=false;self.save_session.pending_leave=None;self.save_session.approved_close=false;self.editor.clear_unsaved_changes();
        self.close_after_file=false;self.set_file_busy(None);
        diagnostics::error("save_library_failed",&anyhow::anyhow!(error));
    }
    pub(super) fn poll_library_outcome(&mut self,outcome:LibraryOutcome){
        match outcome{
            LibraryOutcome::Opened(result)=>match result{
                Ok((entry,document,histories,lease))=>{
                    let path=match &self.save_session.catalog{Ok(l)=>l.project_path(&entry.id),Err(e)=>Err(e.clone())};
                    match path{
                        Ok(path)=>{
                            self.apply_imported_project(path,document,histories);self.editor.show_canvas();
                            self.save_session.active=Some(entry);self.save_session.lease=Some(lease);self.redraw_view_key=None;self.save_session.home=false;self.save_session.settings=false;
                            self.save_session.dirty=false;self.save_session.error=None;
                            self.auto_load_pending=!self.loaded&&!self.loading&&!self.compiling;
                            if std::env::args().any(|a|a=="--library-smoke-output"||a=="--redraw-smoke-output"||a=="--mode-switch-smoke-output"||a=="--manual-save-smoke-output"||a=="--reference-smoke-output"){self.auto_load_pending=false;}
                            self.save_session.refresh();
                        },Err(e)=>self.library_failure(e),
                    }
                },Err(e)=>self.library_failure(e),
            },
            LibraryOutcome::Saved{document,histories,result}=>{
                self.document=document;self.histories=histories;self.set_file_busy(None);
                match result{
                    Ok(entry)=>{
                        self.save_session.active=Some(entry);self.save_session.dirty=false;
                        self.save_session.error=None;self.error=None;self.status="存档已保存".into();
                        diagnostics::event("save_library_saved",serde_json::json!({"nodes":self.document.nodes.len(),"current":self.document.current}));
                        self.finish_saved_leave();
                    },Err(e)=>{
                        let leave=self.save_session.pending_leave.as_ref()
                            .filter(|p|p.choice==Some(ui::UnsavedChoice::Save)).map(|p|p.action.clone());
                        let message=format!("保存失败，当前状态树仍保留在窗口中：{e}");
                        self.library_failure(message.clone());
                        if let Some(action)=leave{self.recover_failed_save_leave(action,&message);}
                    },
                }
            },
            LibraryOutcome::Renamed(result)=>{self.set_file_busy(None);match result{
                Ok(_)=>{self.save_session.error=None;self.save_session.refresh();},Err(e)=>self.library_failure(e),
            }},
        }
    }
    pub(super) fn begin_library_import(&mut self,path:PathBuf){
        let library=match &self.save_session.catalog{Ok(l)=>l.clone(),Err(e)=>{self.library_failure(e.clone());return;}};
        let workspace=if self.is_redraw(){crate::workspace::WorkspaceState::redraw()}else{Default::default()};
        self.start_file_job("正在导入新的存档…",move|_|FileWork::Library(LibraryWork::Import{library,path,workspace}));
    }
    pub(super) fn render_library(&mut self,ctx:&egui::Context){
        let views=self.save_session.entries.iter().map(|entry|{
            let path=self.save_session.catalog.as_ref().ok().and_then(|l|l.preview_path(&entry.id).ok());
            (entry,path,format!("{} 个图元",entry.primitive_count))
        }).collect::<Vec<_>>();
        let entries=views.iter().map(|(e,path,label)|crate::save_library_ui::SaveEntryView{
            id:&e.id,name:&e.display_name,updated_label:label,preview_path:path.as_deref(),node_count:e.node_count,
        }).collect::<Vec<_>>();
        let error=self.save_session.error.clone().or_else(||(!self.save_session.warnings.is_empty()).then(||self.save_session.warnings.join("\n")));
        let actions=self.save_session.ui.show(ctx,&entries,self.file_busy_message.is_some(),error.as_deref());
        if self.file_busy_message.is_some(){ctx.request_repaint_after(std::time::Duration::from_millis(50));}
        for action in actions{self.library_action(action);}
    }
}
