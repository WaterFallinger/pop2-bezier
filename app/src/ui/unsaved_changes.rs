//! An explicit leave decision. This component never saves or discards data.
use eframe::egui::{self,Align,Align2,Color32,Layout,Rect,RichText,Vec2};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum UnsavedChoice{Save,Discard,Cancel}

pub(super) struct UnsavedChanges{
    action_label:Option<String>,
    failure:Option<String>,
    #[cfg(test)]pub(super) save_rect:Rect,
    #[cfg(test)]pub(super) discard_rect:Rect,
    #[cfg(test)]pub(super) cancel_rect:Rect,
}
impl Default for UnsavedChanges{
    fn default()->Self{Self{action_label:None,failure:None,
        #[cfg(test)]save_rect:Rect::NOTHING,#[cfg(test)]discard_rect:Rect::NOTHING,#[cfg(test)]cancel_rect:Rect::NOTHING}}
}
impl UnsavedChanges{
    pub(super) fn open(&mut self,action_label:&str){
        self.action_label=Some(if action_label.trim().is_empty(){"离开当前存档".into()}else{action_label.trim().into()});
        self.failure=None;
    }
    pub(super) fn save_failed(&mut self,action_label:&str,error:&str){
        self.open(action_label);self.failure=Some(error.into());
    }
    pub(super) fn clear(&mut self){self.action_label=None;self.failure=None;}
    pub(super) fn is_open(&self)->bool{self.action_label.is_some()}
    pub(super) fn show(&mut self,ctx:&egui::Context)->Option<UnsavedChoice>{
        let Some(label)=self.action_label.as_deref()else{return None;};
        egui::CentralPanel::default().frame(egui::Frame::none().fill(Color32::from_rgb(242,245,250))).show(ctx,|_|{});
        let mut open=true;let mut choice=ctx.input(|input|input.key_pressed(egui::Key::Escape)).then_some(UnsavedChoice::Cancel);
        #[cfg(test)]let(mut save_rect,mut discard_rect,mut cancel_rect)=(Rect::NOTHING,Rect::NOTHING,Rect::NOTHING);
        egui::Window::new(if self.failure.is_some(){"保存未完成"}else{"有尚未保存的更改"}).id(egui::Id::new("unsaved_leave_prompt"))
            .open(&mut open).anchor(Align2::CENTER_CENTER,Vec2::ZERO).collapsible(false).resizable(false)
            .default_width(470.).show(ctx,|ui|{
                ui.label(RichText::new(if self.failure.is_some(){format!("保存失败，仍可选择不保存并{label}。")}else{format!("要先保存，再{label}吗？")}).size(15.).strong());
                if let Some(error)=&self.failure{
                    ui.add_space(9.);
                    egui::ScrollArea::vertical().max_height(150.).show(ui,|ui|{ui.label(RichText::new(error).color(Color32::from_rgb(177,65,59)));});
                    ui.add_space(9.);ui.label("当前状态树仍在窗口中。取消可返回继续编辑；程序不会自动重试保存。");
                }
                ui.add_space(9.);ui.label(RichText::new("不保存将放弃本次尚未保存的修改，磁盘中已有存档保持不变。").color(Color32::from_rgb(111,122,143)));
                ui.add_space(19.);ui.with_layout(Layout::right_to_left(Align::Center),|ui|{
                    let cancel=ui.add(egui::Button::new("取消").min_size(Vec2::new(84.,34.)));
                    #[cfg(test)]{cancel_rect=cancel.rect;}
                    if cancel.clicked(){choice=Some(UnsavedChoice::Cancel);}
                    let discard_label=if self.failure.is_some(){format!("不保存并{label}")}else{"不保存".into()};
                    let discard=ui.add(egui::Button::new(RichText::new(discard_label).color(Color32::from_rgb(177,65,59))).min_size(Vec2::new(92.,34.)));
                    #[cfg(test)]{discard_rect=discard.rect;}
                    if discard.clicked(){choice=Some(UnsavedChoice::Discard);}
                    let save=ui.add(egui::Button::new(RichText::new(if self.failure.is_some(){"重试保存"}else{"保存"}).color(Color32::WHITE)).fill(Color32::from_rgb(58,110,242)).min_size(Vec2::new(92.,34.)));
                    #[cfg(test)]{save_rect=save.rect;}
                    if save.clicked(){choice=Some(UnsavedChoice::Save);}
                });
            });
        #[cfg(test)]{self.save_rect=save_rect;self.discard_rect=discard_rect;self.cancel_rect=cancel_rect;}
        if !open{choice=Some(UnsavedChoice::Cancel);}
        if choice.is_some(){self.clear();ctx.request_repaint();}
        choice
    }
}

#[cfg(test)]
mod tests{
    use super::*;use eframe::egui::Pos2;
    fn frame(ctx:&egui::Context,dialog:&mut UnsavedChanges,events:Vec<egui::Event>)->Option<UnsavedChoice>{
        let mut choice=None;let _=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(1000.,720.))),events,..Default::default()},|ctx|choice=dialog.show(ctx));choice
    }
    #[test]fn no_prompt_is_created_by_rendering_and_each_explicit_button_returns_one_decision(){
        let ctx=egui::Context::default();let mut dialog=UnsavedChanges::default();
        assert_eq!(frame(&ctx,&mut dialog,vec![]),None);assert!(!dialog.is_open());
        for wanted in [UnsavedChoice::Save,UnsavedChoice::Discard,UnsavedChoice::Cancel]{
            dialog.open("返回存档页");for _ in 0..3{assert_eq!(frame(&ctx,&mut dialog,vec![]),None);assert!(dialog.is_open());}
            let point=match wanted{UnsavedChoice::Save=>dialog.save_rect.center(),UnsavedChoice::Discard=>dialog.discard_rect.center(),UnsavedChoice::Cancel=>dialog.cancel_rect.center()};
            let event=|pressed|egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()};
            assert_eq!(frame(&ctx,&mut dialog,vec![egui::Event::PointerMoved(point),event(true)]),None);
            assert_eq!(frame(&ctx,&mut dialog,vec![event(false)]),Some(wanted));assert!(!dialog.is_open());
            assert_eq!(frame(&ctx,&mut dialog,vec![]),None);
        }
    }
    #[test]fn escape_is_cancel_and_clear_never_returns_a_file_action(){
        let ctx=egui::Context::default();let mut dialog=UnsavedChanges::default();dialog.open("关闭程序");frame(&ctx,&mut dialog,vec![]);
        assert_eq!(frame(&ctx,&mut dialog,vec![egui::Event::Key{key:egui::Key::Escape,physical_key:None,pressed:true,repeat:false,modifiers:Default::default()}]),Some(UnsavedChoice::Cancel));
        dialog.open("导入状态树");dialog.clear();assert_eq!(frame(&ctx,&mut dialog,vec![]),None);
    }
    #[test]fn failed_save_waits_for_explicit_retry_or_discard_and_cancel_keeps_edits(){
        let ctx=egui::Context::default();let mut dialog=UnsavedChanges::default();
        for wanted in [UnsavedChoice::Discard,UnsavedChoice::Cancel,UnsavedChoice::Save]{
            dialog.save_failed("关闭窗口","状态树结构验证失败");
            for _ in 0..5{assert_eq!(frame(&ctx,&mut dialog,vec![]),None);assert!(dialog.is_open());assert!(dialog.failure.is_some());}
            let point=match wanted{UnsavedChoice::Save=>dialog.save_rect.center(),UnsavedChoice::Discard=>dialog.discard_rect.center(),UnsavedChoice::Cancel=>dialog.cancel_rect.center()};
            let event=|pressed|egui::Event::PointerButton{pos:point,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()};
            assert_eq!(frame(&ctx,&mut dialog,vec![egui::Event::PointerMoved(point),event(true)]),None);
            assert_eq!(frame(&ctx,&mut dialog,vec![event(false)]),Some(wanted));assert!(!dialog.is_open());assert!(dialog.failure.is_none());
        }
    }
}
