//! Startup compatibility notice. The owner gates editor actions while open;
//! dismissing this presentation never sends a model or document command.
use crate::compatibility_notice::CompatibilityNotice;
use eframe::egui::{self, Align, Align2, Color32, Id, Layout, Order, Rect, RichText, Sense, Vec2};

const INK: Color32 = Color32::from_rgb(30, 41, 59);
const MUTED: Color32 = Color32::from_rgb(100, 116, 139);
const BLUE: Color32 = Color32::from_rgb(56, 106, 255);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DialogActions {
    pub dismissed: bool,
    pub copied: bool,
    pub open_logs: bool,
}

pub struct CompatibilityDialog {
    open: bool,
    copied: bool,
    /// Recorded geometry also lets bounded UI acceptance click the actual UI.
    pub dialog_rect: Rect,
    pub copy_rect: Rect,
    pub logs_rect: Rect,
    pub continue_rect: Rect,
    pub close_rect: Rect,
    feature_text_rects: Vec<(Rect, Rect)>,
}
impl Default for CompatibilityDialog {
    fn default() -> Self {
        Self { open: false, copied: false, dialog_rect: Rect::NOTHING,
            copy_rect: Rect::NOTHING, logs_rect: Rect::NOTHING,
            continue_rect: Rect::NOTHING, close_rect: Rect::NOTHING,
            feature_text_rects: Vec::new() }
    }
}
impl CompatibilityDialog {
    pub fn open(&mut self) { self.open = true; self.copied = false; }
    pub fn is_open(&self) -> bool { self.open }

    pub fn show(&mut self, ctx: &egui::Context, notice: &CompatibilityNotice) -> DialogActions {
        self.dialog_rect = Rect::NOTHING;
        self.copy_rect = Rect::NOTHING;
        self.logs_rect = Rect::NOTHING;
        self.continue_rect = Rect::NOTHING;
        self.close_rect = Rect::NOTHING;
        self.feature_text_rects.clear();
        let mut actions = DialogActions::default();
        if !self.open { return actions; }
        let screen = ctx.screen_rect();
        egui::Area::new(Id::new("compatibility_notice_scrim"))
            .order(Order::Foreground).fixed_pos(screen.min).movable(false).constrain(false)
            .show(ctx, |ui| {
                let (rect, _) = ui.allocate_exact_size(screen.size(), Sense::click_and_drag());
                ui.painter().rect_filled(rect, 0.0, Color32::from_black_alpha(90));
            });
        let id = Id::new("compatibility_notice_dialog");
        let width = (screen.width() - 104.0).clamp(260.0, 720.0);
        let body_height = (screen.height() - 300.0).clamp(120.0, 400.0);
        let mut close = ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        let area = egui::Area::new(id).order(Order::Foreground)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO).movable(false)
            .show(ctx, |ui| {
                egui::Frame::window(ui.style()).fill(Color32::WHITE)
                    .inner_margin(22.0).rounding(12.0).show(ui, |ui| {
                        ui.set_width(width);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("当前正在使用兼容配置").size(19.0).strong().color(INK));
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                let response = ui.add_sized([28.0, 28.0], egui::Button::new("×").frame(false));
                                self.close_rect = response.rect;
                                if response.on_hover_text("关闭提示，继续使用软件").clicked() { close = true; }
                            });
                        });
                        ui.add_space(7.0);
                        ui.label(RichText::new("关闭此提示后，可以继续使用当前可用的功能。").size(13.0).color(MUTED));
                        ui.add_space(12.0);
                        egui::ScrollArea::vertical().id_source("compatibility_notice_body")
                            .max_height(body_height).auto_shrink([false, true]).show(ui, |ui| {
                                ui.set_width(width);
                                ui.label(RichText::new(format!("当前方案：{}", notice.current_scheme)).strong().color(INK));
                                ui.add(egui::Label::new(RichText::new(format!("实际计算：{}", notice.current_backend)).size(13.0).color(MUTED)).wrap(true));
                                ui.add_space(8.0);
                                ui.add(egui::Label::new(RichText::new(&notice.reason).size(13.0).color(INK)).wrap(true));
                                ui.add_space(14.0);
                                ui.label(RichText::new("未启用的功能").strong().color(INK));
                                ui.add_space(5.0);
                                if notice.disabled_features.is_empty() {
                                    ui.label(RichText::new("当前没有额外的GPU加速功能启用。").size(13.0).color(MUTED));
                                } else {
                                    for (index, feature) in notice.disabled_features.iter().enumerate() {
                                        let fill = if index % 2 == 0 { Color32::from_rgb(245, 247, 251) } else { Color32::WHITE };
                                        let texts = egui::Frame::none().fill(fill).inner_margin(8.0)
                                            .show(ui, |ui| {
                                                ui.set_width(width - 16.0);
                                                let name = ui.add(egui::Label::new(RichText::new(&feature.name)
                                                    .size(13.0).strong().color(INK)).wrap(true)).rect;
                                                ui.add_space(3.0);
                                                let reason = ui.add(egui::Label::new(RichText::new(&feature.reason)
                                                    .size(13.0).color(MUTED)).wrap(true)).rect;
                                                (name, reason)
                                            }).inner;
                                        self.feature_text_rects.push(texts);
                                        ui.add_space(3.0);
                                    }
                                }
                                ui.add_space(12.0);
                                egui::CollapsingHeader::new("交给其他AI的诊断摘要")
                                    .id_source("compatibility_notice_summary").default_open(false)
                                    .show(ui, |ui| {
                                        ui.label(RichText::new("可以使用下方按钮完整复制。这里的文本只读，不包含原始提示词或权重内容。").size(12.0).color(MUTED));
                                        let mut text = notice.ai_summary.as_str();
                                        ui.add(egui::TextEdit::multiline(&mut text).font(egui::TextStyle::Monospace)
                                            .desired_width(width - 24.0).desired_rows(5).interactive(false));
                                    });
                                if let Some(directory) = &notice.log_directory {
                                    ui.add_space(8.0);
                                    ui.add(egui::Label::new(RichText::new(format!("详细日志：{}", directory.display())).size(11.0).color(MUTED)).wrap(true));
                                }
                            });
                        ui.add_space(13.0);
                        ui.separator();
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            let label = if self.copied { "已复制诊断信息" } else { "复制给AI的诊断信息" };
                            let response = ui.add_sized([184.0, 34.0], egui::Button::new(label));
                            self.copy_rect = response.rect;
                            if response.clicked() {
                                ctx.output_mut(|output| output.copied_text = notice.ai_summary.clone());
                                self.copied = true; actions.copied = true;
                            }
                            let response = ui.add_enabled_ui(notice.log_directory.is_some(), |ui| {
                                ui.add_sized([112.0, 34.0], egui::Button::new("打开日志目录"))
                            }).inner;
                            self.logs_rect = response.rect;
                            if response.clicked() { actions.open_logs = true; }
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                let response = ui.add_sized([156.0, 34.0], egui::Button::new(
                                    RichText::new("关闭，继续使用").color(Color32::WHITE)).fill(BLUE));
                                self.continue_rect = response.rect;
                                if response.clicked() { close = true; }
                            });
                        });
                    });
            });
        self.dialog_rect = area.response.rect;
        ctx.move_to_top(egui::LayerId::new(Order::Foreground, id));
        if close { self.open = false; actions.dismissed = true; }
        actions
    }
}

#[cfg(test)] mod tests {
    use super::*;
    use crate::compatibility_notice::DisabledFeature;
    use std::path::PathBuf;

    fn notice() -> CompatibilityNotice {
        CompatibilityNotice {
            current_backend: "CPU F32".into(), current_scheme: "CPU兼容计算".into(),
            reason: "GPU基础组件缺失，已使用可用的兼容计算。".into(),
            disabled_features: vec![DisabledFeature { name: "GPU采样".into(), reason: "当前模型由CPU计算。".into() }],
            ai_summary: "Primitive Painter 0.9.1\nbackend=cpu\nGPU sampling=false\nreason=missing CUDA component\n请先查看实际后端与运行库。".into(),
            log_directory: Some(PathBuf::from("C:/owned/logs/session")),
        }
    }
    fn frame(ctx: &egui::Context, dialog: &mut CompatibilityDialog, notice: &CompatibilityNotice,
        events: Vec<egui::Event>, disable_underlay: bool) -> (DialogActions, egui::FullOutput, bool, Rect) {
        let blocked = disable_underlay && dialog.is_open();
        let mut actions = DialogActions::default(); let mut clicked = false; let mut button = Rect::NOTHING;
        let out = ctx.run(egui::RawInput {
            screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(1100.0,720.0))), events,
            ..Default::default()
        }, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let response = ui.add_enabled(!blocked, egui::Button::new("underlying application action"));
                button=response.rect; clicked=response.clicked();
            });
            actions=dialog.show(ctx,notice);
        });
        (actions,out,clicked,button)
    }
    fn pointer(pos:egui::Pos2,pressed:bool)->Vec<egui::Event>{vec![
        egui::Event::PointerMoved(pos), egui::Event::PointerButton{pos,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()}
    ]}
    fn settle(ctx:&egui::Context, dialog:&mut CompatibilityDialog, notice:&CompatibilityNotice){
        for _ in 0..3 {frame(ctx,dialog,notice,vec![],true);}
    }
    #[test] fn compatibility_dialog_copy_contains_the_complete_readonly_diagnostic(){
        let ctx=egui::Context::default();let mut dialog=CompatibilityDialog::default();let notice=notice();
        dialog.open();settle(&ctx,&mut dialog,&notice);let at=dialog.copy_rect.center();
        frame(&ctx,&mut dialog,&notice,pointer(at,true),true);
        let (action,out,_,_)=frame(&ctx,&mut dialog,&notice,pointer(at,false),true);
        assert!(action.copied);assert!(!action.dismissed);assert!(!action.open_logs);
        assert_eq!(out.platform_output.copied_text,notice.ai_summary);assert!(dialog.is_open());
    }
    #[test] fn compatibility_dialog_close_releases_application_without_side_effect_commands(){
        let ctx=egui::Context::default();let mut dialog=CompatibilityDialog::default();let notice=notice();
        dialog.open();settle(&ctx,&mut dialog,&notice);let at=dialog.continue_rect.center();
        frame(&ctx,&mut dialog,&notice,pointer(at,true),true);
        let (action,_,clicked,_)=frame(&ctx,&mut dialog,&notice,pointer(at,false),true);
        assert_eq!(action,DialogActions{dismissed:true,..Default::default()});assert!(!clicked);assert!(!dialog.is_open());
        let (_,_,_,button)=frame(&ctx,&mut dialog,&notice,vec![],true);
        frame(&ctx,&mut dialog,&notice,vec![],true);
        frame(&ctx,&mut dialog,&notice,pointer(button.center(),true),true);
        assert!(frame(&ctx,&mut dialog,&notice,pointer(button.center(),false),true).2);
    }
    #[test] fn compatibility_dialog_scrim_blocks_outside_pointer_and_x_dismisses(){
        let ctx=egui::Context::default();let mut dialog=CompatibilityDialog::default();let notice=notice();
        dialog.open();settle(&ctx,&mut dialog,&notice);
        let (_,_,_,button)=frame(&ctx,&mut dialog,&notice,vec![],false);
        frame(&ctx,&mut dialog,&notice,pointer(button.center(),true),false);
        let (action,_,clicked,_)=frame(&ctx,&mut dialog,&notice,pointer(button.center(),false),false);
        assert!(!clicked);assert_eq!(action,DialogActions::default());assert!(dialog.is_open());
        let x=dialog.close_rect.center();frame(&ctx,&mut dialog,&notice,pointer(x,true),true);
        assert!(frame(&ctx,&mut dialog,&notice,pointer(x,false),true).0.dismissed);assert!(!dialog.is_open());
    }
    #[test] fn compatibility_dialog_long_details_keep_all_actions_on_screen(){
        let ctx=egui::Context::default();let mut dialog=CompatibilityDialog::default();let mut notice=notice();
        notice.reason="很长的可用性说明。".repeat(80);notice.disabled_features=(0..12).map(|i|DisabledFeature{name:format!("功能{i}"),reason:"原因说明。".repeat(20)}).collect();
        notice.ai_summary="diagnostic detail\n".repeat(100);dialog.open();settle(&ctx,&mut dialog,&notice);
        let screen=Rect::from_min_size(egui::Pos2::ZERO,Vec2::new(1100.0,720.0));
        assert!(screen.contains_rect(dialog.dialog_rect));
        for rect in [dialog.copy_rect,dialog.logs_rect,dialog.continue_rect,dialog.close_rect]{assert!(rect.is_positive());assert!(screen.contains_rect(rect));}
        assert_eq!(dialog.feature_text_rects.len(),notice.disabled_features.len());
        for (index,(name,reason)) in dialog.feature_text_rects.iter().enumerate(){
            assert!(name.bottom() < reason.top(), "name/reason overlap for feature {index}");
            if index>0{assert!(dialog.feature_text_rects[index-1].1.bottom()<name.top(),"adjacent feature rows overlap");}
        }
        let at=dialog.logs_rect.center();frame(&ctx,&mut dialog,&notice,pointer(at,true),true);
        assert!(frame(&ctx,&mut dialog,&notice,pointer(at,false),true).0.open_logs);assert!(dialog.is_open());
    }
}
