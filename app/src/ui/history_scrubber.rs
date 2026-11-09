//! A spring-return browsing rate, independent of model generation and history filters.
use crate::document::Document;
use eframe::egui::{self,Pos2,Rect,Vec2};

const DEADZONE:f32=0.05;
const MAX_RATE:f64=240.;

pub(super) struct ScrubOutput {pub target:Option<u64>,pub freeze_view:bool}
impl ScrubOutput {fn none()->Self{Self{target:None,freeze_view:false}}}

pub(super) struct HistoryScrubber {
    value:f32,grabbed:bool,froze:bool,latched:bool,direction:i8,
    cursor:Option<usize>,expected:Option<usize>,budget:f64,last_time:Option<f64>,
    pub(super) rect:Rect,
}
impl Default for HistoryScrubber {
    fn default()->Self{Self{value:0.,grabbed:false,froze:false,latched:false,direction:0,
        cursor:None,expected:None,budget:0.,last_time:None,rect:Rect::NOTHING}}
}

impl HistoryScrubber {
    pub(super) fn reset(&mut self){let rect=self.rect;*self=Self::default();self.rect=rect;}
    pub(super) fn show(&mut self,ui:&mut egui::Ui,doc:&Document,current:u64,enabled:bool)->ScrubOutput{
        let enabled=enabled&&ui.is_enabled();let screen=ui.ctx().screen_rect();
        let(now,focused,position,down,pressed)=ui.input(|input|(input.time,input.raw.focused,
            input.pointer.hover_pos().filter(|position|screen.contains(*position)),input.pointer.primary_down(),input.pointer.button_pressed(egui::PointerButton::Primary)));
        if !enabled||!focused||position.is_none()||!down{self.reset();}
        let side_width=20.;let gap=ui.spacing().item_spacing.x;
        let width=(ui.available_width()-2.*side_width-2.*gap).clamp(32.,780.);
        let row_width=width+2.*side_width+2.*gap;
        let response=ui.scope(|ui|{
            ui.spacing_mut().item_spacing.y=2.;
            ui.vertical_centered(|ui|{
                ui.label("前后翻阅");
                let padding=((ui.available_width()-row_width)/2.).max(0.);
                ui.horizontal(|ui|{
                    ui.add_space(padding);
                    ui.add_sized([side_width,22.],egui::Label::new(egui::RichText::new("根").small()));
                    let response=ui.scope(|ui|{
                ui.spacing_mut().slider_width=width;
                ui.spacing_mut().slider_rail_height=4.;
                ui.visuals_mut().widgets.inactive.bg_fill=egui::Color32::from_rgb(205,214,228);
                ui.visuals_mut().widgets.inactive.fg_stroke.color=egui::Color32::from_rgb(111,133,168);
                ui.add_enabled_ui(enabled,|ui|ui.add_sized([width,22.],egui::Slider::new(&mut self.value,-1.0..=1.0)
                    .show_value(false).smart_aim(false).trailing_fill(false))).inner
                    }).inner;
                    ui.add_sized([side_width,22.],egui::Label::new(egui::RichText::new("枝").small()));
                    response
                }).inner
            }).inner
        }).inner;
        self.rect=response.rect;
        response.clone().on_hover_text("按住拖动：左侧向父节点，右侧沿唯一子节点前进。离中心越远越快；遇到岔点或端点即停，回到中间或松手后可再次翻阅。");
        if !enabled||!focused||position.is_none()||!down{if self.value!=0.{self.reset();ui.ctx().request_repaint();}return ScrubOutput::none();}
        if pressed&&(response.hovered()||response.is_pointer_button_down_on())&&position.is_some_and(|p|response.rect.contains(p)){
            self.grabbed=true;self.last_time=Some(now);
        }
        if !self.grabbed{self.value=0.;return ScrubOutput::none();}
        let output=self.advance(doc,current,self.value,now);
        if self.value.abs()>DEADZONE&&!self.latched{ui.ctx().request_repaint_after(std::time::Duration::from_millis(16));}
        output
    }

    fn advance(&mut self,doc:&Document,current:u64,value:f32,now:f64)->ScrubOutput{
        let Ok(current)=usize::try_from(current)else{self.reset();return ScrubOutput::none();};
        if doc.nodes.get(current).is_none_or(|node|node.id!=current){self.reset();return ScrubOutput::none();}
        if self.expected.is_some_and(|expected|expected!=current){self.reset();return ScrubOutput::none();}
        let dt=self.last_time.map_or(0.,|last|(now-last).clamp(0.,0.1));self.last_time=Some(now);
        if !value.is_finite()||value.abs()<=DEADZONE{
            self.budget=0.;self.direction=0;self.latched=false;self.cursor=None;self.expected=None;return ScrubOutput::none();
        }
        let value=value.clamp(-1.,1.);
        let direction=if value<0.{-1}else{1};let mut output=ScrubOutput::none();
        if self.cursor.is_none(){self.cursor=Some(current);self.expected=Some(current);output.freeze_view=!self.froze;self.froze=true;}
        if self.direction!=direction {self.direction=direction;self.budget=0.;return output;}
        if self.latched{return output;}
        let cursor=self.cursor.unwrap();
        if one_step(doc,cursor,direction).is_none(){self.latched=true;self.budget=0.;return output;}
        let displacement=((value.abs()-DEADZONE)/(1.-DEADZONE))as f64;
        self.budget+=dt*(1.+(MAX_RATE-1.)*displacement*displacement);
        let steps=(self.budget.floor()as usize).min(24);self.budget-=steps as f64;
        let mut target=cursor;
        for _ in 0..steps{
            let Some((next,boundary))=one_step(doc,target,direction)else{self.latched=true;self.budget=0.;break;};
            target=next;if boundary{self.latched=true;self.budget=0.;break;}
        }
        if target!=current{self.cursor=Some(target);self.expected=Some(target);output.target=Some(target as u64);}
        output
    }
}

/// Validate each edge, including reciprocal links/depth, before walking it.
/// A fork is observable even when it is hidden by the tree's display mode.
fn one_step(doc:&Document,current:usize,direction:i8)->Option<(usize,bool)>{
    let node=doc.nodes.get(current).filter(|node|node.id==current)?;
    let next=if direction<0{node.parent?}else{if node.children.len()!=1{return None;}node.children[0]};
    let other=doc.nodes.get(next).filter(|node|node.id==next)?;
    if direction<0{
        if !other.children.contains(&current)||other.depth.checked_add(1)!=Some(node.depth){return None;}
        Some((next,other.parent.is_none()||other.children.len()>1))
    }else{
        if other.parent!=Some(current)||node.depth.checked_add(1)!=Some(other.depth){return None;}
        Some((next,other.children.len()!=1))
    }
}

#[cfg(test)]
mod tests{
    use super::*;use crate::document::{Primitive,ShapeKind};
    fn chain(length:usize)->Document{let mut doc=Document::new();for _ in 0..length{doc.commit_generated(Primitive::new(ShapeKind::Rectangle,20.,20.,10.,10.,[40,60,80])).unwrap();}doc}
    fn armed()->HistoryScrubber{HistoryScrubber{grabbed:true,last_time:Some(0.),..Default::default()}}
    #[test]fn bounded_walk_stops_at_real_forks_in_both_directions_and_left_can_leave_a_fork(){
        let mut doc=chain(6);doc.jump(3).unwrap();doc.commit_generated(Primitive::new(ShapeKind::Rectangle,40.,40.,10.,10.,[200;3])).unwrap();
        let mut state=armed();assert!(state.advance(&doc,0,1.,0.).freeze_view);
        assert_eq!(state.advance(&doc,0,1.,5.).target,Some(3));assert!(state.latched);
        assert!(state.advance(&doc,3,1.,6.).target.is_none());
        let mut state=armed();state.advance(&doc,6,-1.,0.);assert_eq!(state.advance(&doc,6,-1.,0.1).target,Some(3));
        let mut state=armed();state.advance(&doc,3,-1.,0.);assert_eq!(state.advance(&doc,3,-1.,0.1).target,Some(0));
        let mut state=armed();state.advance(&doc,3,1.,0.);assert!(state.advance(&doc,3,1.,0.1).target.is_none());assert!(state.latched);
    }
    #[test]fn latched_leaf_stays_put_when_model_appends_until_the_control_reenters_deadzone(){
        let mut doc=chain(3);let mut state=armed();state.advance(&doc,0,1.,0.);assert_eq!(state.advance(&doc,0,1.,0.1).target,Some(3));
        doc.commit_generated(Primitive::new(ShapeKind::Rectangle,40.,40.,10.,10.,[80,60,40])).unwrap();
        assert!(state.advance(&doc,3,1.,0.2).target.is_none());assert!(state.advance(&doc,3,-1.,0.3).target.is_none(),"direction changes cannot bypass a boundary latch");
        state.advance(&doc,3,0.,0.4);state.advance(&doc,3,1.,0.5);assert_eq!(state.advance(&doc,3,1.,0.6).target,Some(4));
    }
    #[test]fn reversal_clears_fraction_debt_and_bad_ids_or_external_jumps_reset_safely(){
        let mut doc=chain(100);let mut state=armed();state.advance(&doc,5,-0.2,0.);
        assert!(state.advance(&doc,5,-0.2,0.1).target.is_none());assert!(state.budget>0.5);
        state.advance(&doc,5,0.2,0.2);assert_eq!(state.budget,0.);assert!(state.advance(&doc,5,0.2,0.21).target.is_none());
        assert_eq!(state.advance(&doc,5,1.,100.).target,Some(29),"a stalled frame may traverse at most 24 validated edges");
        state.value=1.;assert!(state.advance(&doc,7,1.,101.).target.is_none());assert!(!state.grabbed);assert_eq!(state.value,0.);
        doc.nodes[5].children=vec![9999];let mut state=armed();state.advance(&doc,5,1.,0.);assert!(state.advance(&doc,5,1.,0.1).target.is_none());
        assert!(state.advance(&doc,u64::MAX,1.,0.2).target.is_none());
    }
    fn frame(ctx:&egui::Context,state:&mut HistoryScrubber,doc:&Document,current:u64,time:f64,focused:bool,events:Vec<egui::Event>)->(ScrubOutput,egui::FullOutput){
        let mut action=ScrubOutput::none();let output=ctx.run(egui::RawInput{time:Some(time),focused,screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(600.,150.))),events,..Default::default()},|ctx|{
            egui::CentralPanel::default().show(ctx,|ui|action=state.show(ui,doc,current,true));
        });(action,output)
    }
    #[test]fn real_pointer_hold_keeps_moving_without_pointer_events_and_release_or_window_loss_springs_home(){
        let ctx=egui::Context::default();let mut state=HistoryScrubber::default();let doc=chain(100);
        frame(&ctx,&mut state,&doc,0,0.,true,vec![]);let point=Pos2::new(state.rect.right()-2.,state.rect.center().y);
        let press=|pressed|egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()};
        assert!(frame(&ctx,&mut state,&doc,0,0.01,true,vec![egui::Event::PointerMoved(point),press(true)]).0.freeze_view);
        let(first,output)=frame(&ctx,&mut state,&doc,0,0.11,true,vec![]);let target=first.target.unwrap();
        assert!(output.viewport_output[&egui::ViewportId::ROOT].repaint_delay<=std::time::Duration::from_millis(16));
        let(second,_)=frame(&ctx,&mut state,&doc,target,0.21,true,vec![]);assert!(second.target.unwrap()>target);assert!(!second.freeze_view);
        frame(&ctx,&mut state,&doc,second.target.unwrap(),0.22,true,vec![press(false)]);assert_eq!(state.value,0.);assert!(!state.grabbed);
        frame(&ctx,&mut state,&doc,0,0.3,true,vec![egui::Event::PointerMoved(point),press(true)]);
        frame(&ctx,&mut state,&doc,0,0.4,false,vec![]);assert_eq!(state.value,0.);assert!(!state.grabbed);
        frame(&ctx,&mut state,&doc,0,0.5,true,vec![egui::Event::PointerGone]);assert_eq!(state.value,0.);
    }
}
