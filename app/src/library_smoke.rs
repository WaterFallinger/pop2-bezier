//! Opt-in installed native acceptance using a dedicated, initially absent library.
use super::*;

impl PainterApp{
    pub(super) fn drive_library_smoke(&mut self){
        let args=std::env::args().collect::<Vec<_>>();
        let Some(path)=args.iter().position(|a|a=="--library-smoke-output").and_then(|i|args.get(i+1)).map(PathBuf::from)else{return;};
        self.auto_load_pending=false;
        if self.frames<2||self.library_smoke_done{return;}
        let result=self.library_smoke_step();
        if let Err(error)=result{self.library_smoke_done=true;self.save_session.error=Some(error);}
        if self.library_smoke_done{
            let report=serde_json::json!({"status":if self.save_session.error.is_none(){"passed"}else{"failed"},
                "error":self.save_session.error,"checks":self.library_smoke_checks,"home":self.save_session.home,
                "entries":self.save_session.entries,"model_loaded":self.loaded,"auto_load_pending":self.auto_load_pending,
                "scope":"installed native Main and asynchronous managed project workers; no inference"});
            if let Some(parent)=path.parent(){let _=std::fs::create_dir_all(parent);}
            let _=std::fs::write(path,serde_json::to_vec_pretty(&report).unwrap());
        }
        self.file_repaint.request_repaint_after(std::time::Duration::from_millis(30));
    }
    fn library_smoke_step(&mut self)->Result<(),String>{
        use save_library_ui::Action;
        if let Some(error)=self.save_session.error.clone(){return Err(error);}
        if self.file_busy_message.is_some(){return Ok(());}
        match self.library_smoke_stage{
            0=>{
                if !self.save_session.home||!self.save_session.entries.is_empty()||self.save_session.catalog.as_ref().unwrap().root().exists(){return Err("空启动必须停在存档页，不得创建目录或存档".into());}
                self.library_smoke_checks.push("empty_startup_has_no_implicit_save_or_model_load".into());
                self.library_action(Action::Create{name:"创作实验".into()});self.library_smoke_stage=1;
            },
            1=>{
                if self.save_session.home||self.save_session.active.is_none(){return Err("手动创建后未进入存档".into());}
                self.library_smoke_checks.push("explicit_create_opens_new_saved_tree".into());
                self.action(ui::UiAction::CommitCanvas{label:"背景".into(),primitives:vec![Primitive::background([235,241,250])]});
                let root=self.document.current;let mut canvas=self.document.primitives.clone();
                canvas.push(Primitive::new(ShapeKind::Ellipse,230.,240.,260.,300.,[250,181,96]));
                self.action(ui::UiAction::CommitCanvas{label:"暖色分支".into(),primitives:canvas.clone()});
                self.action(ui::UiAction::Jump(root as u64));canvas[1].color=[70,136,210];
                self.action(ui::UiAction::CommitCanvas{label:"蓝色分支".into(),primitives:canvas});
                self.action(ui::UiAction::SetPrompt("两个创作分支".into()));
                self.action(ui::UiAction::ReturnToLibrary);self.resolve_unsaved_choice(ui::UnsavedChoice::Save);self.library_smoke_stage=2;
            },
            2=>{
                if !self.save_session.home||self.save_session.active.is_some(){return Err("保存后未返回存档列表".into());}
                self.library_smoke_checks.push("return_home_saves_complete_branch_tree_and_releases_view".into());
                let entry=self.save_session.entries.first().ok_or("保存后列表为空")?;
                if entry.node_count!=4||entry.primitive_count!=2{return Err("存档统计不正确".into());}
                let image=image::open(self.save_session.catalog.as_ref().unwrap().preview_path(&entry.id)?).map_err(|e|e.to_string())?;
                if image.width()!=720||image.height()!=280{return Err("双栏预览尺寸错误".into());}
                self.library_smoke_checks.push("composite_preview_canvas_left_branch_tree_right_is_decodable".into());
                self.library_action(Action::Rename{id:entry.id.clone(),name:"双分支 · 已重命名".into()});self.library_smoke_stage=3;
            },
            3=>{
                let entry=self.save_session.entries.first().ok_or("重命名后列表为空")?;
                if entry.display_name!="双分支 · 已重命名"{return Err("重命名未持久化".into());}
                self.library_smoke_checks.push("rename_keeps_stable_id_and_tree".into());
                self.library_action(Action::Open{id:entry.id.clone()});self.library_smoke_stage=4;
            },
            4=>{
                if self.save_session.home||self.document.nodes.len()!=4||self.document.nodes[1].children.len()!=2
                    ||self.document.primitives[1].color!=[70,136,210]||self.document.prompt!="两个创作分支"{
                    return Err("重新打开未恢复完整树、当前画布或提示词".into());
                }
                self.library_smoke_checks.push("reopen_restores_tree_branches_current_canvas_and_prompt".into());
                self.action(ui::UiAction::ReturnToLibrary);self.resolve_unsaved_choice(ui::UnsavedChoice::Save);self.library_smoke_stage=5;
            },
            5=>{
                if !self.save_session.home||self.loaded||self.loading||self.auto_load_pending{return Err("存档页不应加载模型".into());}
                self.library_smoke_checks.push("home_navigation_and_acceptance_never_load_a_model".into());
                self.library_smoke_done=true;
            },_=>{},
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn directory()->PathBuf{static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!("painter-session-{}-{}",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)))}
    #[test]
    fn managed_worker_create_save_rename_open_and_invalid_import_preserve_existing(){
        let root=directory();let library=save_library::Library::at(&root);
        assert!(library.list().unwrap().entries.is_empty());assert!(!root.exists());
        let (entry,mut doc,history,lease)=match save_session::perform(save_session::LibraryWork::Create{library:library.clone(),name:"初始".into()}){
            save_session::LibraryOutcome::Opened(Ok(data))=>data,_=>panic!("create failed"),
        };
        doc.prompt="尚未创建节点的草稿".into();doc.settings.cfg=2.5;
        let entry=match save_session::perform(save_session::LibraryWork::Save{library:library.clone(),entry,document:doc,histories:history}){
            save_session::LibraryOutcome::Saved{document,result:Ok(entry),..}=>{assert_eq!(document.prompt,"尚未创建节点的草稿");entry},_=>panic!("save failed"),
        };
        let id=entry.id.clone();drop(lease);library.rename(&id,"重命名").unwrap();
        match save_session::perform(save_session::LibraryWork::Open{library:library.clone(),id}){
            save_session::LibraryOutcome::Opened(Ok((entry,doc,_,_lease)))=>{assert_eq!(entry.display_name,"重命名");assert_eq!(doc.prompt,"尚未创建节点的草稿");assert_eq!(doc.settings.cfg,2.5);},_=>panic!("open failed"),
        }
        let bad=root.join("bad.pproj");std::fs::write(&bad,b"bad").unwrap();
        assert!(matches!(save_session::perform(save_session::LibraryWork::Import{library:library.clone(),path:bad,workspace:Default::default()}),save_session::LibraryOutcome::Opened(Err(_))));
        assert_eq!(library.list().unwrap().entries.len(),1);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn failed_managed_save_returns_live_tree_and_old_saved_project_remains_loadable(){
        let root=directory();let library=save_library::Library::at(&root);
        let mut doc=Document::new();let history=RawHistoryStore::new();
        let entry=library.create("已有作品",&doc,&history).unwrap();
        doc.prompt="尚未存好的草稿".into();doc.settings.cfg=3.;
        let mut renamed=entry.clone();renamed.display_name="\n".into();
        match save_session::perform(save_session::LibraryWork::Save{library:library.clone(),entry:renamed,document:doc,histories:history}){
            save_session::LibraryOutcome::Saved{document,histories,result}=>{
                assert!(result.is_err());assert_eq!(document.prompt,"尚未存好的草稿");assert_eq!(document.settings.cfg,3.);
                assert!(histories.effective_history(&document,document.current).is_ok());
            },_=>panic!("a failed save must return document ownership"),
        }
        let (saved,_)=project_file::load_project(&library.project_path(&entry.id).unwrap()).unwrap();
        assert!(saved.prompt.is_empty());assert_eq!(library.list().unwrap().entries[0].display_name,"已有作品");
        std::fs::remove_dir_all(root).unwrap();
    }
}
