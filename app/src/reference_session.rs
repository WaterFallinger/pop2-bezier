//! Reference wizard orchestration. No result enters the document until Import.
use super::*;
use crate::document::NodeId;
use crate::ui::reference_ui::{ReferenceAction,ReferenceImageView,ReferenceResultView,ReferenceStage,ReferenceView};
use crate::{reference_image::SourceImage,reference_serialize::{SerializeProgress,SerializeResult}};
use std::sync::{Arc,atomic::{AtomicBool,AtomicU64,Ordering},mpsc::{self,Receiver,SyncSender}};

#[derive(Clone)]
struct Origin{node:NodeId,save:Option<String>,nodes:usize}
impl Origin{
    fn capture(app:&PainterApp)->Self{Self{node:app.document.current,save:app.save_session.active.as_ref().map(|e|e.id.clone()),nodes:app.document.nodes.len()}}
    fn matches(&self,app:&PainterApp)->bool{
        app.document.current==self.node&&app.document.nodes.len()==self.nodes&&app.document.primitives.is_empty()
            &&app.save_session.active.as_ref().map(|e|e.id.clone())==self.save
    }
}
enum Event{
    Loaded(u64,Result<(Arc<SourceImage>,ReferenceImageView),String>),
    Cropped(u64,Result<(Arc<image::RgbImage>,ReferenceImageView),String>),
    Progress(u64,SerializeProgress),
    Finished(u64,Result<SerializeResult,String>),
    Failed(u64,String),
}
pub(super) struct ReferenceSession{
    origin:Origin,view:ReferenceView,source:Option<Arc<SourceImage>>,target:Option<Arc<image::RgbImage>>,
    result:Option<Arc<SerializeResult>>,receiver:Option<Receiver<Event>>,cancel:Arc<AtomicBool>,
    job:u64,resume_auto_load:bool,
}
impl Drop for ReferenceSession{fn drop(&mut self){self.cancel.store(true,Ordering::Relaxed);}}
fn revision()->u64{static NEXT:AtomicU64=AtomicU64::new(1);NEXT.fetch_add(1,Ordering::Relaxed)}
fn rgb_view(image:&image::RgbImage,name:&str)->ReferenceImageView{
    crate::reference_image::rgb_to_view(image,revision(),name.into())
}
fn source_view(source:&SourceImage,name:&str)->ReferenceImageView{
    crate::reference_image::source_to_view(source,revision(),name.into())
}

#[path="reference_smoke.rs"]
mod smoke;
impl ReferenceSession{
    fn spawn(&mut self,repaint:egui::Context,task:impl FnOnce(u64,SyncSender<Event>,Arc<AtomicBool>)->Event+Send+'static)->Result<(),String>{
        self.cancel.store(true,Ordering::Relaxed);self.cancel=Arc::new(AtomicBool::new(false));
        let job=revision();self.job=job;let (tx,rx)=mpsc::sync_channel(3);let cancel=self.cancel.clone();
        std::thread::Builder::new().name("reference-preparation".into()).spawn(move||{
            let event=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||task(job,tx.clone(),cancel)))
                .unwrap_or_else(|_|Event::Failed(job,"参考制作任务异常终止，画布未修改".into()));
            let _=tx.send(event);repaint.request_repaint();
        }).map_err(|e|format!("无法启动参考制作任务：{e}"))?;
        self.receiver=Some(rx);self.view.busy=true;self.view.error=None;self.view.can_import=false;Ok(())
    }
    fn load(&mut self,path:PathBuf,repaint:egui::Context)->Result<(),String>{
        let name=path.file_name().map(|n|n.to_string_lossy().into_owned()).unwrap_or_else(||"参考图像".into());
        self.source=None;self.target=None;self.result=None;self.view.source=None;self.view.target=None;self.view.result=None;
        self.view.stage=ReferenceStage::Loading;self.view.status="正在读取参考图像…".into();self.view.progress=None;
        self.spawn(repaint,move|id,_,cancel|Event::Loaded(id,(||{
            let source=crate::reference_image::decode(&path)?;
            if cancel.load(Ordering::Relaxed){return Err("已取消读取参考图像".into());}
            let view=source_view(&source,&name);Ok((Arc::new(source),view))
        })()))
    }
}

impl PainterApp{
    fn reference_can_enter(&self)->bool{
        !self.running&&!self.loading&&!self.compiling&&!self.stopping&&self.file_busy_message.is_none()
            &&self.pending_canvas_edit.is_none()&&self.document.primitives.is_empty()
            &&(!self.save_session.enabled||!self.save_session.home&&self.save_session.active.is_some())
    }
    pub(super) fn open_reference_import(&mut self){
        if !self.reference_can_enter(){self.editor.notify("参考图像只能导入到停止生成后的空白画布");return;}
        let path=match ui::file_dialogs::choose_reference_image(None){Ok(Some(path))=>path,Ok(None)=>return,Err(e)=>{self.editor.notify(e);return;}};
        self.cancel_reference_session();
        let mut session=ReferenceSession{origin:Origin::capture(self),view:ReferenceView::default(),source:None,target:None,result:None,
            receiver:None,cancel:Arc::new(AtomicBool::new(false)),job:0,resume_auto_load:self.auto_load_pending};
        self.auto_load_pending=false;
        if let Err(e)=session.load(path,self.file_repaint.clone()){session.view.busy=false;session.view.stage=ReferenceStage::Crop;session.view.error=Some(e);}
        self.editor.set_reference_view(Some(session.view.clone()));self.reference_session=Some(session);
    }
    pub(super) fn cancel_reference_session(&mut self){
        if let Some(session)=self.reference_session.take(){
            session.cancel.store(true,Ordering::Relaxed);
            if session.resume_auto_load&&!self.loaded&&!self.loading&&!self.compiling{self.auto_load_pending=true;}
        }
        self.editor.set_reference_view(None);
    }
    pub(super) fn reference_action(&mut self,action:ReferenceAction){
        if action==ReferenceAction::Close{self.cancel_reference_session();return;}
        let Some(mut session)=self.reference_session.take()else{return;};
        let mut imported=false;
        let result=(||->Result<(),String>{
            match action{
                ReferenceAction::ChooseImage if !session.view.busy=>{
                    if let Some(path)=ui::file_dialogs::choose_reference_image(None)?{session.load(path,self.file_repaint.clone())?;}
                },
                ReferenceAction::ConfirmCrop{crop} if !session.view.busy=>{
                    let source=session.source.clone().ok_or("请先选择参考图像")?;
                    session.target=None;session.result=None;session.view.target=None;session.view.result=None;
                    session.view.stage=ReferenceStage::Prepare;session.view.status="正在截取参考区域…".into();session.view.progress=None;
                    session.spawn(self.file_repaint.clone(),move|id,_,cancel|Event::Cropped(id,(||{
                        let target=crate::reference_image::crop_to_target(&source,&crop)?;
                        if cancel.load(Ordering::Relaxed){return Err("已取消截取参考区域".into());}
                        let view=rgb_view(&target,"截取的参考图像");Ok((Arc::new(target),view))
                    })()))?;
                },
                ReferenceAction::Serialize{stop_rmse} if !session.view.busy=>{
                    crate::reference_serialize::validate_stop_rmse(stop_rmse)?;
                    let target=session.target.clone().ok_or("请先确认截取范围")?;
                    session.result=None;session.view.result=None;session.view.stop_rmse=stop_rmse;
                    session.view.status="正在初始化图像序列化…".into();session.view.progress=None;
                    diagnostics::event("reference_serialization_requested",serde_json::json!({"target_rmse":stop_rmse,"width":512,"height":512,
                        "minimum_rmse":crate::reference_serialize::MIN_STOP_RMSE,"maximum_rmse":crate::reference_serialize::MAX_STOP_RMSE,
                        "threshold_policy":"whole_image_only_in_global_stage_then_independent_cells","automatic_import":false}));
                    let repaint=self.file_repaint.clone();
                    session.spawn(self.file_repaint.clone(),move|id,tx,cancel|{
                        let result=crate::reference_serialize::serialize(&target,stop_rmse,&cancel,|progress|{
                            if tx.try_send(Event::Progress(id,progress)).is_ok(){repaint.request_repaint();}
                        }).map_err(|e|format!("{e:#}"));Event::Finished(id,result)
                    })?;
                },
                ReferenceAction::CancelSerialization if session.view.busy=>{
                    session.cancel.store(true,Ordering::Relaxed);session.view.status="正在取消序列化…".into();session.view.can_import=false;
                },
                ReferenceAction::BackToCrop if !session.view.busy=>{
                    session.view.stage=ReferenceStage::Crop;session.result=None;session.view.result=None;session.view.can_import=false;
                    session.view.error=None;
                },
                ReferenceAction::Import if !session.view.busy=>{
                    if !session.origin.matches(self)||!self.reference_can_enter(){return Err("原来的空白画布已经变化，请关闭参考制作后重新开始；当前画布未修改".into());}
                    let result=session.result.as_ref().ok_or("请先完成序列化，再手动导入参考")?;
                    let current=self.editor.reference_stop_rmse().ok_or("参考制作界面已关闭")?;
                    crate::reference_serialize::validate_stop_rmse(current)?;
                    crate::reference_serialize::validate_stop_rmse(result.target_rmse)?;
                    if (current-result.target_rmse).abs()>1e-9{return Err("截止RMSE已修改，请先重新序列化再导入".into());}
                    let outcome=crate::reference_import::import_reference(&mut self.document,&mut self.histories,session.origin.node,&result.primitives,&result.records)?;
                    if let Some(entry)=&mut self.save_session.active{
                        entry.workspace.mode=crate::workspace::WorkspaceMode::Redraw;
                        entry.workspace.selection=crate::redraw::RedrawSelection::from_reference_regions(&outcome.resume_regions)?;
                    }
                    self.parallel=None;self.stage_map_node=None;self.redraw_view_key=None;self.pending_canvas_edit=None;self.cancel_global_edit();
                    self.save_session.mark_dirty();self.editor.invalidate_history_after_deletion();self.editor.show_canvas();
                    self.editor.set_workflow_mode(true);
                    self.refresh_drawing_region();self.refresh_redraw_ui();self.error=None;
                    let deferred_fine=self.histories.stage_floor(&self.document,self.document.current)?.unwrap_or(1)<8
                        &&self.document.primitives.iter().any(|p|p.stage_assignment.is_some_and(|a|a.stage==document::StageTier::Fine));
                    if deferred_fine{self.editor.notify("细网格图形已保留。模型先续画未完成的粗网格，粗阶段结束后再读取细阶段数据。");}
                    self.status=format!("参考已导入修改模式 · {}个图元 · 全图 RMSE {:.3}",outcome.primitives,result.rmse);
                    diagnostics::event("reference_imported",serde_json::json!({"node":outcome.node,"shapes":outcome.primitives,
                        "actual_rmse":result.rmse,"target_rmse":result.target_rmse,"threshold_reached":result.threshold_reached,
                        "reset_boundary":outcome.reset_boundary,"manual_import":true,"mode":"redraw",
                        "resume_regions":outcome.resume_regions,"fine_context_is_deferred":deferred_fine}));
                    if session.resume_auto_load&&!self.loaded&&!self.loading&&!self.compiling{self.auto_load_pending=true;}
                    self.editor.set_reference_view(None);imported=true;return Ok(());
                },_=>{},
            }Ok(())
        })();
        if let Err(error)=result{diagnostics::error("reference_preparation_failed",&anyhow::anyhow!(error.clone()));session.view.error=Some(error);
            if session.view.stage==ReferenceStage::Loading&&!session.view.busy{session.view.stage=ReferenceStage::Crop;}}
        // Successful import moved current away from the frozen blank origin.
        if imported{return;}
        session.view.can_import=!session.view.busy&&session.result.is_some()&&session.origin.matches(self)&&self.reference_can_enter();
        self.editor.set_reference_view(Some(session.view.clone()));self.reference_session=Some(session);
    }
    pub(super) fn poll_reference_session(&mut self){
        let Some(mut session)=self.reference_session.take()else{return;};
        let mut events=Vec::new();let mut disconnected=false;
        if let Some(receiver)=&session.receiver{for _ in 0..16{match receiver.try_recv(){
            Ok(event)=>events.push(event),Err(mpsc::TryRecvError::Empty)=>break,Err(mpsc::TryRecvError::Disconnected)=>{disconnected=true;break;}
        }}}
        let mut terminal=false;
        for event in events{
            let id=match &event{Event::Loaded(id,_)|Event::Cropped(id,_)|Event::Progress(id,_)|Event::Finished(id,_)|Event::Failed(id,_)=>*id};
            if id!=session.job{continue;}
            match event{
                Event::Progress(_,progress)=>{
                    session.view.status=progress.stage_label;session.view.progress=Some((progress.records.len()as f32/897.).clamp(0.,1.));
                    session.view.result=Some(ReferenceResultView{preview:rgb_view(&progress.preview,"序列化预览"),actual_rmse:progress.rmse,
                        shape_count:progress.shape_count,requested_rmse:session.view.stop_rmse,stop_reason:"正在序列化，尚未导入".into()});
                },
                Event::Loaded(_,result)=>{terminal=true;match result{
                    Ok((source,view))=>{session.source=Some(source);session.view.source=Some(view);session.view.stage=ReferenceStage::Crop;session.view.status="请框选参考范围".into();},
                    Err(e)=>{diagnostics::error("reference_image_failed",&anyhow::anyhow!(e.clone()));session.view.error=Some(e);session.view.stage=ReferenceStage::Crop;session.view.status="读取失败，请重新选择参考图像".into();},
                }},
                Event::Cropped(_,result)=>{terminal=true;match result{
                    Ok((target,view))=>{session.target=Some(target);session.view.target=Some(view);session.view.stage=ReferenceStage::Prepare;session.view.status="设置截止RMSE，然后点击序列化".into();},
                    Err(e)=>{diagnostics::error("reference_crop_failed",&anyhow::anyhow!(e.clone()));session.view.error=Some(e);session.view.stage=ReferenceStage::Crop;},
                }},
                Event::Finished(_,result)=>{terminal=true;
                    if session.cancel.load(Ordering::Relaxed){session.view.status="已取消，画布未修改".into();session.result=None;}
                    else{match result{Ok(result)=>{
                        diagnostics::event("reference_serialization_completed",serde_json::json!({"actual_rmse":result.rmse,"target_rmse":result.target_rmse,
                            "rmse_scope":"whole_image_diagnostic","threshold_policy":"whole_image_only_in_global_stage_then_independent_cells",
                            "shapes":result.shape_count,"threshold_reached":result.threshold_reached,"automatic_import":false}));
                        session.view.result=Some(ReferenceResultView{preview:rgb_view(&result.preview,"序列化结果"),actual_rmse:result.rmse,
                            shape_count:result.shape_count,requested_rmse:result.target_rmse,stop_reason:result.stop_reason.clone()});
                        session.view.status=format!("序列化已完成 · {}。请检查后手动点击“导入参考”。",result.stage_label);
                        session.view.progress=Some(1.);session.result=Some(Arc::new(result));
                    },Err(e)=>{diagnostics::error("reference_serialization_failed",&anyhow::anyhow!(e.clone()));session.view.error=Some(e);session.result=None;}}}
                },Event::Failed(_,error)=>{terminal=true;diagnostics::error("reference_worker_failed",&anyhow::anyhow!(error.clone()));session.view.error=Some(error);session.result=None;
                    if session.view.stage==ReferenceStage::Loading{session.view.stage=ReferenceStage::Crop;}},
            }
        }
        if terminal||disconnected{session.view.busy=false;session.receiver=None;
            if disconnected&&!terminal{session.view.error=Some("参考制作任务意外结束，画布未修改".into());}}
        session.view.can_import=!session.view.busy&&session.result.is_some()&&session.origin.matches(self)&&self.reference_can_enter();
        self.editor.set_reference_view(Some(session.view.clone()));self.reference_session=Some(session);
    }
}
