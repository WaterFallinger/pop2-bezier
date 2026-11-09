use super::UiAction;
use eframe::egui::{self,Align,Color32,Layout,Pos2,Rect,Stroke,Vec2};
use std::collections::VecDeque;

const LIFETIME:f64=8.;
struct Toast {id:u64,text:String,remaining:f64,last_time:Option<f64>,hovered:bool,rect:Rect}
impl Toast {fn new(id:u64,text:String)->Self{Self{id,text,remaining:LIFETIME,last_time:None,hovered:false,rect:Rect::NOTHING}}}

#[derive(Default)]
pub(super) struct Toasts {
    notices:VecDeque<Toast>,transition:Option<Toast>,next_id:u64,
    #[cfg(test)] pub(super) confirm_rect:Option<Rect>,
    #[cfg(test)] pub(super) close_rect:Option<Rect>,
}

impl Toasts {
    fn make(&mut self,text:String)->Toast{self.next_id=self.next_id.wrapping_add(1);Toast::new(self.next_id,text)}
    pub(super) fn notify(&mut self,text:String){let toast=self.make(text);self.notices.push_back(toast);while self.notices.len()>3{self.notices.pop_front();}}
    pub(super) fn request_transition(&mut self,text:String){self.transition=Some(self.make(text));}
    /// Root already invalidated its pending edit. This is deliberately silent.
    pub(super) fn cancel_transition(&mut self){self.transition=None;}
    pub(super) fn show(&mut self,ctx:&egui::Context,allow_confirmation:bool)->Vec<UiAction>{
        let mut actions=Vec::new();let screen=ctx.screen_rect();let width=(screen.width()-32.).clamp(160.,560.);
        let mut top=Pos2::new(screen.center().x-width/2.,screen.top()+12.);
        #[cfg(test)]{self.confirm_rect=None;self.close_rect=None;}
        if let Some(toast)=&mut self.transition{
            let result=paint(ctx,toast,top,width,true,allow_confirmation);top.y+=result.height+8.;
            #[cfg(test)]{self.confirm_rect=result.confirm_rect;self.close_rect=result.close_rect;}
            if result.confirm {self.transition=None;actions.push(UiAction::ConfirmGlobalTransition);ctx.request_repaint();}
            else if result.close||result.expired{self.transition=None;actions.push(UiAction::CancelGlobalTransition);ctx.request_repaint();}
        }
        let count=self.notices.len();
        self.notices.retain_mut(|toast|{let result=paint(ctx,toast,top,width,false,false);top.y+=result.height+8.;!(result.close||result.expired)});
        if count!=self.notices.len(){ctx.request_repaint();}
        actions
    }
}

#[derive(Default)]
struct Painted {close:bool,confirm:bool,expired:bool,height:f32,
    #[cfg(test)]confirm_rect:Option<Rect>,#[cfg(test)]close_rect:Option<Rect>}

fn paint(ctx:&egui::Context,toast:&mut Toast,position:Pos2,width:f32,critical:bool,allow_confirmation:bool)->Painted{
    let now=ctx.input(|input|input.time);
    if let Some(last)=toast.last_time {if !toast.hovered {toast.remaining-=(now-last).max(0.);}}
    toast.last_time=Some(now);
    let hovering=ctx.input(|input|input.pointer.hover_pos().is_some_and(|p|toast.rect.contains(p)));
    if toast.remaining<=0.&&!hovering{return Painted{expired:true,..Default::default()};}
    let mut result=Painted::default();
    let area=egui::Area::new(egui::Id::new(("painter_notice",toast.id))).order(egui::Order::Foreground)
        .fixed_pos(position).movable(false).show(ctx,|ui|{
            egui::Frame::none().fill(Color32::from_rgb(242,244,247)).stroke(Stroke::new(1.,Color32::from_rgb(216,222,230)))
                .rounding(8.).inner_margin(12.).show(ui,|ui|{
                    ui.set_width(width-24.);
                    ui.horizontal_top(|ui|{
                        ui.allocate_ui_with_layout(Vec2::new(width-62.,0.),Layout::top_down(Align::Min),|ui|{ui.label(&toast.text);});
                        let close=ui.small_button("×").on_hover_text("关闭");result.close=close.clicked();
                        #[cfg(test)]{result.close_rect=Some(close.rect);}
                    });
                    if critical {ui.add_space(6.);let confirm=ui.add_enabled(allow_confirmation,egui::Button::new("进入粗网格"));result.confirm=confirm.clicked();
                        #[cfg(test)]{result.confirm_rect=Some(confirm.rect);}
                    }
                });
        });
    toast.rect=area.response.rect;result.height=toast.rect.height();
    toast.hovered=ctx.input(|input|input.pointer.hover_pos().is_some_and(|p|toast.rect.contains(p)));
    if !toast.hovered {ctx.request_repaint_after(std::time::Duration::from_secs_f64(toast.remaining.max(0.01)));}
    result
}

#[cfg(test)]
mod tests{
    use super::*;
    fn frame(ctx:&egui::Context,toasts:&mut Toasts,time:f64,events:Vec<egui::Event>)->Vec<UiAction>{
        let mut actions=Vec::new();ctx.run(egui::RawInput{time:Some(time),screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(900.,700.))),events,..Default::default()},|ctx|{
            egui::CentralPanel::default().show(ctx,|_|{});actions=toasts.show(ctx,true);
        });actions
    }
    fn warm(ctx:&egui::Context,toasts:&mut Toasts){for time in [0.,0.02,0.04]{assert!(frame(ctx,toasts,time,vec![]).is_empty());}}
    fn click(ctx:&egui::Context,toasts:&mut Toasts,point:Pos2)->Vec<UiAction>{
        let mut actions=Vec::new();for(time,pressed)in[(0.06,true),(0.08,false)]{actions.extend(frame(ctx,toasts,time,vec![egui::Event::PointerMoved(point),egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()}]));}actions
    }
    #[test]fn critical_confirmation_is_explicit_and_never_fires_from_timeout_or_close(){
        let ctx=egui::Context::default();let mut toasts=Toasts::default();toasts.request_transition("全图阶段已满".into());warm(&ctx,&mut toasts);
        let point=toasts.confirm_rect.unwrap().center();assert!(matches!(click(&ctx,&mut toasts,point).as_slice(),[UiAction::ConfirmGlobalTransition]));
        assert!(frame(&ctx,&mut toasts,10.,vec![egui::Event::PointerGone]).is_empty());
        let ctx=egui::Context::default();let mut toasts=Toasts::default();toasts.request_transition("确认后才加入图元".into());warm(&ctx,&mut toasts);
        let point=toasts.close_rect.unwrap().center();assert!(matches!(click(&ctx,&mut toasts,point).as_slice(),[UiAction::CancelGlobalTransition]));
        let ctx=egui::Context::default();let mut toasts=Toasts::default();toasts.request_transition("确认后才加入图元".into());warm(&ctx,&mut toasts);
        assert!(matches!(frame(&ctx,&mut toasts,8.1,vec![]).as_slice(),[UiAction::CancelGlobalTransition]));
    }
    #[test]fn hover_pauses_remaining_time_instead_of_resetting_or_confirming(){
        let ctx=egui::Context::default();let mut toasts=Toasts::default();toasts.request_transition("等待明确确认".into());warm(&ctx,&mut toasts);
        let point=toasts.transition.as_ref().unwrap().rect.center();
        assert!(frame(&ctx,&mut toasts,2.,vec![egui::Event::PointerMoved(point)]).is_empty());
        assert!(frame(&ctx,&mut toasts,60.,vec![]).is_empty());
        assert!(frame(&ctx,&mut toasts,60.1,vec![egui::Event::PointerGone]).is_empty());
        assert!(frame(&ctx,&mut toasts,65.9,vec![]).is_empty());
        assert!(matches!(frame(&ctx,&mut toasts,66.2,vec![]).as_slice(),[UiAction::CancelGlobalTransition]));
    }
    #[test]fn ordinary_notices_are_bounded_expire_and_silent_programmatic_cancel_emits_nothing(){
        let ctx=egui::Context::default();let mut toasts=Toasts::default();for n in 0..8{toasts.notify(format!("提醒{n}"));}
        assert_eq!(toasts.notices.len(),3);warm(&ctx,&mut toasts);assert!(frame(&ctx,&mut toasts,8.1,vec![]).is_empty());assert!(toasts.notices.is_empty());
        toasts.request_transition("失效的旧操作".into());toasts.cancel_transition();assert!(frame(&ctx,&mut toasts,20.,vec![]).is_empty());
    }
}
