use crate::canvas::{
    drawing_stage_label, selection_bounds, show_region_indicator, CanvasEdit, CanvasState, Tool,
};
use crate::document::{Document, Primitive, SamplingBackend, Settings, ShapeKind};
use crate::acceleration::{AccelerationMode, FeaturePreference};
use crate::region_state::{DrawingRegion,Stage};
use crate::thumbnails::ThumbnailCache;
use crate::tree_view::TreeViewState;
use eframe::egui::{
    self, Align, Color32, FontData, FontDefinitions, FontFamily, FontId, Layout, Pos2, Rect,
    RichText, Stroke, Vec2,
};
use std::path::{Path, PathBuf};
use std::collections::{BTreeMap,HashSet};
mod prompt_editor;
use prompt_editor::PromptEditor;
mod prompt_templates;
use prompt_templates::PromptTemplates;
mod hsv_picker;
use hsv_picker::HsvPicker;
#[path = "file_dialogs.rs"]
pub(crate) mod file_dialogs;
mod file_menu;
use file_menu::FileMenu;
mod toast;
use toast::Toasts;
mod history_scrubber;
use history_scrubber::HistoryScrubber;
mod unsaved_changes;
pub use unsaved_changes::UnsavedChoice;
#[path="redraw_ui.rs"]
pub(crate) mod redraw_ui;
pub use redraw_ui::RedrawUiModel;
#[path="reference_ui.rs"]
pub(crate) mod reference_ui;
use reference_ui::{ReferenceAction,ReferenceUi,ReferenceView};

const INK: Color32 = Color32::from_rgb(34, 43, 62);
const MUTED: Color32 = Color32::from_rgb(111, 122, 143);
const BLUE: Color32 = Color32::from_rgb(58, 110, 242);
const LINE: Color32 = Color32::from_rgb(228, 233, 241);

#[derive(Clone, Debug)]
pub struct NodeView {
    pub id: u64,
    pub parent: Option<u64>,
    pub title: String,
    pub source: String,
    pub primitive_count: usize,
}

pub struct UiModel<'a> {
    pub document: &'a Document,
    pub drawing_region: &'a DrawingRegion,
    pub primitives: &'a [Primitive],
    pub stage_by_id: Option<&'a BTreeMap<u64,Stage>>,
    pub prompt: &'a str,
    pub settings: &'a Settings,
    pub current_node: u64,
    pub nodes: &'a [NodeView],
    pub running: bool,
    pub loading: bool,
    pub compiling: bool,
    pub stopping: bool,
    pub model_loaded: bool,
    pub model_status: &'a str,
    pub model_path: &'a str,
    pub tokenizer_path: &'a str,
    pub error: Option<&'a str>,
    pub status: &'a str,
}

#[derive(Clone, Debug)]
pub enum UiAction {
    Generate {
        prompt: String,
        settings: Settings,
    },
    Stop,
    ReturnToLibrary,
    FreezeGenerationView,
    Jump(u64),
    BrowseHistory(u64),
    DeleteHistory(Vec<u64>),
    SaveProject(PathBuf),
    ImportProject(PathBuf),
    OpenReference,
    Reference(ReferenceAction),
    ExportPng { path:PathBuf, resolution:u32 },
    ExportSvg(PathBuf),
    AssignStage { ids:Vec<u64>, stage:Stage },
    ToggleRedrawRegion { stage:Stage, index:usize },
    DeleteRedrawRegions,
    DeleteLockedRedrawRegions,
    SwitchWorkspaceMode { redraw:bool },
    ResolveUnsaved { choice:UnsavedChoice },
    ConfirmGlobalTransition,
    CancelGlobalTransition,
    CommitCanvas {
        label: String,
        primitives: Vec<Primitive>,
    },
    PasteCanvas {
        label: String,
        primitives: Vec<Primitive>,
    },
    SetPrompt(String),
    SetSettings(Settings),
    ReorderByContribution,
    OpenLogs,
    ExportDiagnostics,
    LoadModel {
        checkpoint_path: String,
        tokenizer_path: String,
    },
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Page {
    Canvas,
    History,
    Settings,
}

struct PropertyDraft {
    node: u64,
    ids: Vec<u64>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    angle: f32,
    geometry_dirty: bool,
}

pub struct EditorUi {
    model_download:crate::model_download::DownloadUi,
    save_name:Option<String>,
    pub canvas: CanvasState,
    page: Page,
    node: Option<u64>,
    prompt: String,
    settings: Settings,
    last_seen_actual_seed: Option<u64>,
    manual_seed_text: String,
    seed_display_value: u64,
    model_path: String,
    tokenizer_path: String,
    paths_initialized: bool,
    show_shortcuts: bool,
    files: FileMenu,
    toasts: Toasts,
    history_scrubber: HistoryScrubber,
    redraw_view: Option<RedrawUiModel>,
    redraw_panel: redraw_ui::RedrawUi,
    workflow_redraw:bool,
    mode_issues:Option<Vec<String>>,
    unsaved:bool,
    unsaved_prompt:unsaved_changes::UnsavedChanges,
    workflow_mode_rects:[Rect;2],
    #[cfg(test)]mode_issues_close_rect:Rect,
    toast_context: egui::Context,
    tree: TreeViewState,
    thumbnails: ThumbnailCache,
    history_thumbnails_invalidated: bool,
    property: Option<PropertyDraft>,
    edit_canvas_rect: Rect,
    region_indicator_rect: Rect,
    parallel_control_rect: Rect,
    parallel_warning_rect: Rect,
    pause_stage_end_control_rect: Rect,
    half_region_switch_control_rect: Rect,
    auto_seed_control_rect: Rect,
    seed_control_rect: Rect,
    sampling_header_rect: Rect,
    seed_control_enabled: bool,
    stage_control_rects: [Rect;3],
    stage_control_active: [bool;3],
    auto_reorder_control_rect: Rect,
    manual_reorder_rect: Rect,
    log_directory_rect: Rect,
    diagnostics_export_rect: Rect,
    file_selection_error: Option<String>,
    prompt_editor: PromptEditor,
    prompt_templates: PromptTemplates,
    prompt_templates_button_rect: Rect,
    navigation_rects: [Rect; 3],
    open_settings_rect: Rect,
    generation_button_rect: Rect,
    preparation_modal_rect: Rect,
    shape_color_picker: HsvPicker,
    background_color_picker: HsvPicker,
    background_color: [u8; 3],
    color_context: Option<(u64, Vec<u64>)>,
    tools_panel_rect: Rect,
    rasterize_control_rect: Rect,
    reference_entry_rect:Rect,
    reference:ReferenceUi,
    properties_panel_rect: Rect,
    freeze_requested_for_node: Option<u64>,
}

impl EditorUi {
    pub fn new(ctx: &egui::Context) -> Self {
        configure_style(ctx);
        Self {
            model_download:Default::default(),
            save_name:None,
            canvas: CanvasState::default(),
            page: Page::Canvas,
            node: None,
            prompt: String::new(),
            settings: Settings::default(),
            last_seen_actual_seed: None,
            manual_seed_text: Settings::default().seed.to_string(),
            seed_display_value: Settings::default().seed,
            model_path: String::new(),
            tokenizer_path: String::new(),
            paths_initialized: false,
            show_shortcuts: false,
            files: FileMenu::default(),
            toasts: Toasts::default(),
            history_scrubber: HistoryScrubber::default(),
            redraw_view: None,
            redraw_panel: redraw_ui::RedrawUi::default(),
            workflow_redraw:false,
            mode_issues:None,
            unsaved:false,
            unsaved_prompt:unsaved_changes::UnsavedChanges::default(),
            workflow_mode_rects:[Rect::NOTHING;2],
            #[cfg(test)]mode_issues_close_rect:Rect::NOTHING,
            toast_context: ctx.clone(),
            tree: TreeViewState::default(),
            thumbnails: ThumbnailCache::new(),
            history_thumbnails_invalidated: false,
            property: None,
            edit_canvas_rect: Rect::NOTHING,
            region_indicator_rect: Rect::NOTHING,
            parallel_control_rect: Rect::NOTHING,
            parallel_warning_rect: Rect::NOTHING,
            pause_stage_end_control_rect: Rect::NOTHING,
            half_region_switch_control_rect: Rect::NOTHING,
            auto_seed_control_rect: Rect::NOTHING,
            seed_control_rect: Rect::NOTHING,
            sampling_header_rect: Rect::NOTHING,
            seed_control_enabled: false,
            stage_control_rects: [Rect::NOTHING;3],
            stage_control_active: [false;3],
            auto_reorder_control_rect: Rect::NOTHING,
            manual_reorder_rect: Rect::NOTHING,
            log_directory_rect: Rect::NOTHING,
            diagnostics_export_rect: Rect::NOTHING,
            file_selection_error: None,
            prompt_editor: PromptEditor::default(),
            prompt_templates: PromptTemplates::default(),
            prompt_templates_button_rect: Rect::NOTHING,
            navigation_rects: [Rect::NOTHING; 3],
            open_settings_rect: Rect::NOTHING,
            generation_button_rect: Rect::NOTHING,
            preparation_modal_rect: Rect::NOTHING,
            shape_color_picker: HsvPicker::default(),
            background_color_picker: HsvPicker::default(),
            background_color: [255; 3],
            color_context: None,
            tools_panel_rect: Rect::NOTHING,
            rasterize_control_rect: Rect::NOTHING,
            reference_entry_rect:Rect::NOTHING,
            reference:ReferenceUi::default(),
            properties_panel_rect: Rect::NOTHING,
            freeze_requested_for_node: None,
        }
    }

    pub fn set_save_name(&mut self,name:Option<String>){self.files.set_managed(name.is_some());self.save_name=name;}
    pub fn set_unsaved(&mut self,unsaved:bool){if self.unsaved!=unsaved{self.unsaved=unsaved;self.toast_context.request_repaint();}}
    pub fn show_unsaved_changes(&mut self,action_label:&str){self.unsaved_prompt.open(action_label);self.history_scrubber.reset();self.toast_context.request_repaint();}
    pub fn show_failed_save_leave(&mut self,action_label:&str,error:&str){self.unsaved_prompt.save_failed(action_label,error);self.history_scrubber.reset();self.toast_context.request_repaint();}
    pub fn clear_unsaved_changes(&mut self){self.unsaved_prompt.clear();self.toast_context.request_repaint();}
    pub fn set_reference_view(&mut self,view:Option<ReferenceView>){self.reference.set_view(view);self.history_scrubber.reset();self.toast_context.request_repaint();}
    pub fn reference_is_open(&self)->bool{self.reference.is_open()}
    pub fn reference_stop_rmse(&self)->Option<f64>{self.reference.stop_rmse()}
    pub fn set_workflow_mode(&mut self,redraw:bool){
        if self.workflow_redraw!=redraw{self.history_scrubber.reset();self.redraw_panel.clear_focus();}
        self.workflow_redraw=redraw;
    }
    pub fn dismiss_mode_issues(&mut self){self.mode_issues=None;}
    pub fn show_mode_issues(&mut self,issues:Vec<String>){
        let mut issues=issues.into_iter().filter(|issue|!issue.trim().is_empty()).collect::<Vec<_>>();
        if issues.is_empty(){issues.push("当前状态暂时不能切换到标准模式，请检查画布后重试。".into());}
        self.mode_issues=Some(issues);self.history_scrubber.reset();self.redraw_panel.clear_focus();self.toast_context.request_repaint();
    }
    fn render_mode_issues(&mut self,ctx:&egui::Context){
        let Some(issues)=self.mode_issues.as_ref()else{return;};
        egui::CentralPanel::default().frame(egui::Frame::none().fill(Color32::from_rgb(242,245,250))).show(ctx,|_|{});
        let width=(ctx.screen_rect().width()-80.).clamp(320.,680.);
        let height=(ctx.screen_rect().height()-240.).clamp(120.,440.);
        let mut open=true;let mut close=ctx.input(|input|input.key_pressed(egui::Key::Escape));
        #[cfg(test)]let mut close_rect=Rect::NOTHING;
        egui::Window::new("暂时无法切换到标准模式").id(egui::Id::new("workflow_mode_issues"))
            .open(&mut open).anchor(egui::Align2::CENTER_CENTER,Vec2::ZERO).collapsible(false).resizable(true)
            .default_width(width).min_width(300.).show(ctx,|ui|{
                ui.label(RichText::new(format!("请先修正以下 {} 个问题，再尝试切换。",issues.len())).color(MUTED));
                ui.add_space(12.);
                egui::ScrollArea::vertical().id_source("workflow_mode_issues_list").max_height(height)
                    .auto_shrink([false,true]).show(ui,|ui|{
                        for(index,issue)in issues.iter().enumerate(){
                            egui::Frame::none().fill(Color32::from_rgb(247,249,252)).rounding(5.).inner_margin(10.).show(ui,|ui|{
                                ui.add(egui::Label::new(format!("{}. {}",index+1,issue)).wrap(true));
                            });ui.add_space(6.);
                        }
                    });
                ui.add_space(12.);ui.with_layout(Layout::right_to_left(Align::Center),|ui|{
                    let response=ui.add(egui::Button::new("关闭").min_size(Vec2::new(86.,32.)));
                    #[cfg(test)]{close_rect=response.rect;}
                    close|=response.clicked();
                });
            });
        #[cfg(test)]{self.mode_issues_close_rect=close_rect;}
        if close||!open{self.mode_issues=None;ctx.request_repaint();}
    }
    pub fn set_redraw_model(&mut self,model:Option<RedrawUiModel>){
        if self.redraw_view.is_some()!=model.is_some(){self.redraw_panel.reset();}
        self.redraw_view=model;
    }
    pub fn show_canvas(&mut self){self.page=Page::Canvas;self.node=None;self.history_scrubber.reset();self.redraw_panel.clear_focus();}
    pub fn render_library_settings(&mut self,ctx:&egui::Context,model:&UiModel<'_>)->(Vec<UiAction>,bool){
        if !self.paths_initialized{self.model_path=model.model_path.to_owned();self.tokenizer_path=model.tokenizer_path.to_owned();self.paths_initialized=true;}
        let mut actions=Vec::new();let mut back=false;
        egui::TopBottomPanel::top("library_settings_header").show(ctx,|ui|{
            ui.add_space(12.);ui.horizontal(|ui|{back=ui.button("← 返回存档").clicked();ui.heading("设置");});ui.add_space(12.);
        });
        egui::CentralPanel::default().show(ctx,|ui|self.settings_page(ui,model,&mut actions));
        if model.compiling{self.preparation_modal_rect=acceleration_preparation_modal(ctx,model.status);}
        (actions,back)
    }
    pub fn show_history(&mut self) {
        self.history_scrubber.reset();
        self.page = Page::History;
    }

    pub fn set_project_path(&mut self,path:Option<PathBuf>) { self.files.set_project_path(path); }
    pub fn set_file_busy(&mut self,busy:bool) { if busy{self.history_scrubber.reset();}self.files.set_busy(busy); }
    pub fn locate_history_current(&mut self) { self.tree.request_locate_current(); }
    pub fn notify(&mut self,text:impl Into<String>) { self.toasts.notify(text.into());self.toast_context.request_repaint(); }
    pub fn request_global_transition(&mut self,text:impl Into<String>) {self.toasts.request_transition(text.into());self.toast_context.request_repaint();}
    pub fn cancel_global_transition(&mut self) {self.toasts.cancel_transition();self.toast_context.request_repaint();}

    pub fn invalidate_history_after_deletion(&mut self) {
        self.redraw_panel.clear_focus();
        self.history_scrubber.reset();
        self.cancel_global_transition();
        self.tree.invalidate_after_deletion();
        self.history_thumbnails_invalidated = true;
        self.canvas.invalidate_history_after_deletion();
        // Compaction may reuse an old numeric ID for a different state.
        self.node = None;
        self.last_seen_actual_seed = None;
        self.property = None;
        self.color_context = None;
        self.reset_generation_view_freeze();
    }

    pub fn reset_generation_view_freeze(&mut self) {
        self.freeze_requested_for_node = None;
    }

    pub fn show_settings(&mut self) {
        self.history_scrubber.reset();
        self.page = Page::Settings;
    }

    pub fn show_prompt_templates(&mut self) {
        self.history_scrubber.reset();
        self.page = Page::Canvas;
        self.prompt_templates.open();
    }

    pub fn configure_history(&mut self, compact: bool, fit: bool) {
        self.history_scrubber.reset();
        self.page = Page::History;
        self.tree.configure(compact, fit);
    }

    pub fn configure_history_mode(&mut self, mode: crate::tree_layout::TreeDisplayMode, fit: bool) {
        self.history_scrubber.reset();
        self.page = Page::History;
        self.tree.configure_mode(mode, fit);
    }

    pub fn render(&mut self, ctx: &egui::Context, model: &UiModel<'_>) -> Vec<UiAction> {
        // A background save may temporarily own the real document. Never
        // synchronize editor fields from the main thread's placeholder.
        if self.page!=Page::Canvas||self.show_shortcuts||self.prompt_templates.is_open(){self.history_scrubber.reset();}
        if self.files.is_busy() {self.history_scrubber.reset();let mut actions=self.files.show_busy(ctx,model);actions.extend(self.toasts.show(ctx,false));return actions;}
        if self.unsaved_prompt.is_open(){
            return self.unsaved_prompt.show(ctx).map(|choice|vec![UiAction::ResolveUnsaved{choice}]).unwrap_or_default();
        }
        if self.files.has_modal() {self.history_scrubber.reset();let mut actions=self.files.show_modal(ctx,true);actions.extend(self.toasts.show(ctx,false));return actions;}
        if self.mode_issues.is_some(){self.render_mode_issues(ctx);return Vec::new();}
        if self.reference.is_open(){return self.reference.show(ctx).into_iter().map(UiAction::Reference).collect();}
        if self.history_thumbnails_invalidated {
            self.thumbnails = ThumbnailCache::new();
            self.history_thumbnails_invalidated = false;
        }
        let mut actions = Vec::new();
        if self.node != Some(model.current_node) {
            self.reset_generation_view_freeze();
            self.node = Some(model.current_node);
            self.prompt = model.prompt.to_owned();
            self.settings = model.settings.clone();
            self.last_seen_actual_seed = Some(model.settings.seed);
            self.manual_seed_text = model.settings.seed.to_string();
            self.property = None;
        } else if self.last_seen_actual_seed != Some(model.settings.seed) {
            // Main can freeze a new real seed without creating a node (for
            // example an empty region). Only auto mode follows that external
            // update; never overwrite a staged manual edit.
            if self.settings.auto_seed {
                self.settings.seed = model.settings.seed;
                self.manual_seed_text = model.settings.seed.to_string();
            }
            self.last_seen_actual_seed = Some(model.settings.seed);
        }
        if !self.paths_initialized {
            self.model_path = model.model_path.to_owned();
            self.tokenizer_path = model.tokenizer_path.to_owned();
            self.paths_initialized = true;
        }
        let hidden=self.redraw_view.as_ref().map(|view|self.redraw_panel.hidden_ids(view)).unwrap_or_default();
        self.canvas.set_hidden_ids(&hidden);
        self.canvas.sync_node(model.current_node, model.primitives);
        self.sync_color_controls(model);
        if !model.running { self.freeze_requested_for_node = None; }
        if self.page == Page::Canvas && !self.prompt_templates.is_open() {
            if self.redraw_view.is_some(){
                ctx.input(|input|{if input.pointer.button_pressed(egui::PointerButton::Primary){
                    if let Some(point)=input.pointer.interact_pos(){
                        if self.redraw_panel.contains_grid(point){self.redraw_panel.focused=true;}
                        else if self.tools_panel_rect.contains(point)||self.properties_panel_rect.contains(point)
                            ||self.edit_canvas_rect.contains(point)&&!input.key_down(egui::Key::Space){self.redraw_panel.clear_focus();}
                    }
                }});
            }
            let interacting = ctx.input(|input| input.pointer.interact_pos().is_some_and(|point| {
                let down = input.pointer.button_pressed(egui::PointerButton::Primary)
                    || input.pointer.button_pressed(egui::PointerButton::Middle);
                (down && !self.rasterize_control_rect.contains(point)&&!self.reference_entry_rect.contains(point)
                    && [self.edit_canvas_rect, self.tools_panel_rect, self.properties_panel_rect]
                    .iter().any(|rect| rect.contains(point)))
                    || (self.edit_canvas_rect.contains(point) && input.raw_scroll_delta != Vec2::ZERO)
            }));
            if interacting { self.request_view_freeze(model, &mut actions); }
        }
        self.keyboard(ctx, model, &mut actions);

        egui::TopBottomPanel::top("app_header")
            .frame(
                egui::Frame::none()
                    .fill(Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(22.0, 14.0)),
            )
            .show(ctx, |ui| {
                ui.set_enabled(!self.prompt_templates.is_open() && !model.compiling);
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(32.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect.shrink(1.0), 9.0, BLUE);
                    ui.painter().rect_filled(
                        Rect::from_center_size(
                            rect.center() - Vec2::new(3.0, 3.0),
                            Vec2::new(14.0, 12.0),
                        ),
                        2.0,
                        Color32::from_white_alpha(135),
                    );
                    ui.painter().circle_filled(
                        rect.center() + Vec2::new(4.0, 4.0),
                        7.0,
                        Color32::WHITE,
                    );
                    ui.add_space(7.0);
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new("Primitive Painter")
                                .size(20.0)
                                .strong()
                                .color(INK),
                        );
                        let name=self.save_name.as_deref().unwrap_or("图元绘画工作台");
                        let name=if self.unsaved{format!("{name} *")}else{name.to_owned()};
                        ui.label(RichText::new(name).size(11.0).color(MUTED));
                    });
                    ui.add_space(24.0);
                    if self.save_name.is_some(){
                        if ui.add_enabled(!model.loading&&!model.compiling&&!model.stopping,egui::Button::new("← 存档"))
                            .on_hover_text("停止生成、保存当前状态树并返回存档页").clicked(){actions.push(UiAction::ReturnToLibrary);}
                    }
                    ui.add_enabled_ui(!self.show_shortcuts,|ui|self.files.menu(ui,model,&mut actions));
                    ui.add_space(16.0);
                    let settings_tab = tab(ui, "设置", self.page == Page::Settings);
                    self.navigation_rects[2] = settings_tab.rect;
                    if settings_tab.clicked() {
                        self.page = Page::Settings;
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("快捷键").clicked() {
                            self.show_shortcuts = !self.show_shortcuts;
                        }
                        ui.add_space(12.0);
                        let color = if model.model_loaded {
                            Color32::from_rgb(47, 153, 111)
                        } else {
                            Color32::from_rgb(185, 135, 51)
                        };
                        ui.label(
                            RichText::new(if model.model_loaded {
                                "● 模型已加载"
                            } else {
                                "● 模型未加载"
                            })
                            .size(12.0)
                            .color(color),
                        );
                    });
                });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let canvas_tab = tab(ui, "画布", self.page == Page::Canvas);
                    self.navigation_rects[0] = canvas_tab.rect;
                    if canvas_tab.clicked() { self.page = Page::Canvas; }
                    let history_tab = tab(ui, "状态树", self.page == Page::History);
                    self.navigation_rects[1] = history_tab.rect;
                    if history_tab.clicked() { self.page = Page::History; }
                    ui.add_space(12.0);ui.separator();ui.add_space(8.0);
                    let can_switch=!model.running&&!model.loading&&!model.compiling&&!model.stopping;
                    for(index,redraw,label)in [(0,false,"标准模式"),(1,true,"修改模式")]{
                        let response=ui.add_enabled_ui(can_switch,|ui|tab(ui,label,self.workflow_redraw==redraw)).inner;
                        self.workflow_mode_rects[index]=response.rect;
                        if response.clicked()&&self.workflow_redraw!=redraw{actions.push(UiAction::SwitchWorkspaceMode{redraw});}
                        response.on_hover_text(if redraw{"切换到区域选择与局部重绘，保留当前画布和状态树。"}else{"检查当前画布后，切换到标准绘制流程。"})
                            .on_disabled_hover_text("请先停止生成，并等待模型加载或准备完成，再切换模式。");
                    }
                });
            });

        egui::TopBottomPanel::bottom("status_bar")
            .frame(
                egui::Frame::none()
                    .fill(Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(18.0, 7.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if model.running || model.loading {
                        paint_busy_indicator(ui);
                    }
                    ui.label(RichText::new(model.status).size(11.0).color(MUTED));
                    if self.unsaved{ui.add_space(9.);ui.label(RichText::new("未保存 · Ctrl+S").size(11.).color(Color32::from_rgb(155,111,32)));}
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!(
                                "节点 {}  ·  {} 个图元  ·  {} 个状态",
                                model.current_node,
                                model.primitives.len(),
                                model.nodes.len()
                            ))
                            .size(11.0)
                            .color(MUTED),
                        );
                    });
                });
            });

        if self.page != Page::Settings {
        egui::SidePanel::right("inspector")
            .exact_width(310.0)
            .resizable(false)
            .frame(egui::Frame::none().fill(Color32::WHITE))
            .show(ctx, |ui| {
                ui.set_enabled(!self.prompt_templates.is_open() && !model.compiling);
                // SidePanel applies its minimum width inside its own Frame.
                // Keep the padding in a nested frame, otherwise its 36 px is
                // added beyond the panel's reserved width in egui 0.27.
                egui::Frame::none().inner_margin(18.0).show(ui, |ui| {
                    let content_width = ui.available_width();
                    egui::ScrollArea::vertical()
                        .id_source("inspector_scroll")
                        .max_width(content_width)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.set_max_width(content_width);
                            self.generation_panel(ui, model, &mut actions);
                            ui.add_space(16.0);
                            ui.separator();
                            ui.add_space(12.0);
                            if self.page == Page::Canvas {
                                let top = ui.cursor().min;
                                self.properties_panel(ui, model, &mut actions);
                                self.properties_panel_rect = Rect::from_min_max(top,
                                    Pos2::new(ui.max_rect().right(), ui.min_rect().bottom()));
                            } else {
                                self.node_details(ui, model);
                            }
                        });
                });
            });
        }

        if self.page == Page::Canvas {
            let tools_panel = egui::SidePanel::left("tool_rail")
                .exact_width(196.0)
                .resizable(false)
                .frame(
                    egui::Frame::none()
                        .fill(Color32::from_rgb(246, 248, 252))
                        .inner_margin(0.0),
                )
                .show(ctx, |ui| {
                    ui.set_enabled(!self.prompt_templates.is_open() && !model.compiling);
                    egui::Frame::none().inner_margin(12.0).show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .id_source("drawing_tools_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                self.drawing_tools(ui, model, &mut actions);
                            });
                    });
                });
            self.tools_panel_rect = tools_panel.response.rect;
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Color32::from_rgb(242, 245, 250)))
            .show(ctx, |ui| {
                ui.set_enabled(!self.prompt_templates.is_open() && !model.compiling);
                match self.page {
                    Page::Canvas => self.canvas_pair(ui, model, &mut actions),
                    Page::History => self.history(ui, model, &mut actions),
                    Page::Settings => self.settings_page(ui, model, &mut actions),
                }
            });

        if self.show_shortcuts {
            egui::Window::new("快捷键与画布操作")
                .collapsible(false)
                .resizable(false)
                .open(&mut self.show_shortcuts)
                .show(ctx, |ui| {
                    egui::Grid::new("keys")
                        .spacing([28.0, 12.0])
                        .show(ui, |ui| {
                            for (key, description) in [
                                ("V / R / E", "选择 / 矩形 / 椭圆"),
                                ("Shift + 点击", "添加或移除选中对象"),
                                ("空白区域拖动", "框选多个图元"),
                                ("拖动四角 / 顶部圆点", "缩放 / 旋转选中对象"),
                                ("Shift + 旋转", "以 15° 为步长旋转"),
                                ("滚轮", "缩放画布"),
                                ("鼠标中键 / 空格 + 拖动", "平移画布"),
                                ("Ctrl + C / V / D", "复制 / 粘贴 / 创建副本"),
                                ("Ctrl + S / O", "保存状态树 / 导入状态树"),
                                ("Delete", "删除选中图元"),
                                ("Ctrl + Z / Y", "回到父状态 / 前往最新子状态"),
                                ("F", "画布适应窗口"),
                                ("Esc", "取消当前拖动"),
                            ] {
                                ui.label(RichText::new(key).strong());
                                ui.label(description);
                                ui.end_row();
                            }
                        });
                });
        }
        if !model.compiling {
            if let Some(text) = self.prompt_templates.show(ctx, !model.running && !model.loading && !model.stopping) {
                self.prompt = text;
                actions.push(UiAction::SetPrompt(self.prompt.clone()));
            }
        }
        if actions.iter().any(|action| matches!(action,
            UiAction::CommitCanvas { .. } | UiAction::PasteCanvas { .. } | UiAction::ReorderByContribution)) {
            self.redraw_panel.clear_focus();
            self.request_view_freeze(model, &mut actions);
        }
        self.preparation_modal_rect = Rect::NOTHING;
        if model.compiling {
            self.preparation_modal_rect = acceleration_preparation_modal(ctx, model.status);
        }
        // Run file shortcuts after text widgets have committed this frame's
        // edits. Main applies ordinary staged edits before the file action.
        if !self.show_shortcuts && !self.prompt_templates.is_open() {
            self.files.shortcuts(ctx,model,&mut actions);
        }
        self.files.confirm_same_frame_import_drafts(model,&self.prompt,&self.settings,&mut actions);
        if self.files.has_modal() { actions.extend(self.files.show_modal(ctx,false)); }
        // A fresh operation takes precedence over an older confirmation in
        // the same frame. Main cancels that old pending edit with the action.
        if actions.iter().any(|action|matches!(action,UiAction::Generate{..}|UiAction::Jump(_)|UiAction::BrowseHistory(_)|UiAction::DeleteHistory(_)
            |UiAction::ToggleRedrawRegion{..}|UiAction::DeleteRedrawRegions
            |UiAction::SwitchWorkspaceMode{..}
            |UiAction::CommitCanvas{..}|UiAction::PasteCanvas{..}|UiAction::AssignStage{..}|UiAction::ReorderByContribution
            |UiAction::SetPrompt(_)|UiAction::SetSettings(_)|UiAction::SaveProject(_)|UiAction::ImportProject(_)
            |UiAction::ExportPng{..}|UiAction::ExportSvg(_)|UiAction::Stop)) {self.toasts.cancel_transition();}
        actions.extend(self.toasts.show(ctx,!model.loading&&!model.compiling&&!model.stopping
            &&!self.show_shortcuts&&!self.prompt_templates.is_open()&&!self.files.has_modal()));
        if self.page!=Page::Canvas||self.show_shortcuts||self.prompt_templates.is_open()||self.files.has_modal(){self.history_scrubber.reset();}
        actions
    }

    fn request_view_freeze(&mut self, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        if model.running && !model.loading && !model.compiling && !model.stopping
            && self.freeze_requested_for_node != Some(model.current_node) {
            actions.insert(0, UiAction::FreezeGenerationView);
            self.freeze_requested_for_node = Some(model.current_node);
        }
    }

    fn sync_color_controls(&mut self, model: &UiModel<'_>) {
        let selected: Vec<_> = self.canvas.selected(model.primitives).into_iter()
            .filter(|primitive| primitive.shape != ShapeKind::Background).collect();
        let context = (model.current_node, selected.iter().map(|primitive| primitive.id).collect());
        let node_changed = self.color_context.as_ref().map(|previous| previous.0) != Some(model.current_node);
        if self.color_context.as_ref() != Some(&context) {
            if let Some(primitive) = selected.first() {
                self.canvas.new_color = primitive.color;
            }
            self.shape_color_picker.cancel_interaction();
            self.color_context = Some(context);
        }
        if node_changed {
            self.background_color = model.primitives.iter()
                .find(|primitive| primitive.shape == ShapeKind::Background)
                .map(|background| background.color).unwrap_or([255; 3]);
            self.background_color_picker.cancel_interaction();
        }
    }

    fn drawing_tools(&mut self, ui: &mut egui::Ui, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        let editable = ui.is_enabled() && !model.loading && !model.stopping;
        ui.add_enabled_ui(editable, |ui| {
            ui.horizontal(|ui| {
                for (tool, label, key) in [
                    (Tool::Select, "选择", "V"),
                    (Tool::Rectangle, "矩形", "R"),
                    (Tool::Ellipse, "椭圆", "E"),
                ] {
                    if tool_button(ui, label, key, self.canvas.tool == tool).clicked() {
                        self.canvas.tool = tool;
                    }
                }
            });
            ui.add_space(8.0);
            let raster=ui.checkbox(&mut self.canvas.rasterize_canvas,"栅格化画布");
            self.rasterize_control_rect=raster.rect;
            if raster.changed(){ui.ctx().request_repaint();}
            raster.on_hover_text("以512×512抗锯齿渲染画布，缩放时保持512像素分辨率。");
            ui.add_space(8.);
            let can_reference=model.primitives.is_empty()&&!model.running&&!model.loading&&!model.compiling&&!model.stopping;
            let reference=ui.add_enabled(can_reference,egui::Button::new("导入参考图像").min_size(Vec2::new(ui.available_width(),28.)));
            self.reference_entry_rect=reference.rect;
            if reference.clicked(){actions.push(UiAction::OpenReference);}
            reference.on_hover_text("选择图像、框选范围并序列化，确认预览后再手动导入。").on_disabled_hover_text("仅可在空白画布且模型停止生成时导入参考图像。");
            ui.add_space(14.0);
            section_title(ui, "图形颜色", None);
            ui.add_space(4.0);
            let selected_shapes = self.canvas.selected(model.primitives).into_iter()
                .filter(|primitive| primitive.shape != ShapeKind::Background).count();
            ui.label(RichText::new(if selected_shapes == 0 { "用于新图元".to_owned() }
                else { format!("新图元 · 已选 {selected_shapes} 个图形") }).size(10.0).color(MUTED));
            ui.add_space(7.0);
            let width = ui.available_width();
            let shape = ui.push_id("shape_color_hsv", |ui| {
                self.shape_color_picker.show(ui, &mut self.canvas.new_color, width)
            }).inner;
            if shape.changed { ui.ctx().request_repaint(); }
            if editable && shape.finished {
                self.commit_shape_color(model, actions);
            }
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(13.0);
            section_title(ui, "背景颜色", None);
            ui.add_space(4.0);
            ui.label(RichText::new("独立的画布底色").size(10.0).color(MUTED));
            ui.add_space(7.0);
            let width = ui.available_width();
            let background = ui.push_id("background_color_hsv", |ui| {
                self.background_color_picker.show(ui, &mut self.background_color, width)
            }).inner;
            if background.changed { ui.ctx().request_repaint(); }
            if editable && background.finished {
                self.commit_background_color(model, actions);
            }
            ui.add_space(10.0);
        });
    }

    fn commit_shape_color(&self, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        if model.loading || model.stopping { return; }
        let color = native_color(self.canvas.new_color);
        let mut next = model.primitives.to_vec();
        let mut changed = 0;
        for primitive in &mut next {
            if primitive.shape != ShapeKind::Background && self.canvas.selection.contains(&primitive.id)
                && primitive.color != color {
                primitive.color = color;
                changed += 1;
            }
        }
        if changed > 0 {
            actions.push(UiAction::CommitCanvas {
                label: format!("修改 {changed} 个图形的颜色"), primitives: next,
            });
        }
    }

    fn commit_background_color(&self, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        if model.loading || model.stopping { return; }
        let color = native_color(self.background_color);
        let mut next = model.primitives.to_vec();
        let label = if let Some(background) = next.iter_mut()
            .find(|primitive| primitive.shape == ShapeKind::Background) {
            if background.color == color { return; }
            background.color = color;
            "修改背景颜色"
        } else {
            next.insert(0, Primitive::background(color));
            "添加背景"
        };
        actions.push(UiAction::CommitCanvas { label: label.into(), primitives: next });
    }

    fn generation_panel(
        &mut self,
        ui: &mut egui::Ui,
        model: &UiModel<'_>,
        actions: &mut Vec<UiAction>,
    ) {
        ui.horizontal(|ui| {
            section_title(ui, "创作", None);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let response = ui.add_enabled(!model.running && !model.loading && !model.stopping,
                    egui::Button::new(RichText::new("文本模板").size(12.0)));
                self.prompt_templates_button_rect = response.rect;
                if response.clicked() {
                    self.prompt_templates.open();
                }
                response.on_hover_text("选择一条示例文字填入提示词，不会开始生成")
                    .on_disabled_hover_text("请先等待模型加载或生成停止，再选择文本模板");
            });
        });
        ui.add_space(4.0);
        ui.label(RichText::new("文字与已有图元均可作为输入").size(11.0).color(MUTED));
        ui.add_space(10.0);
        let mut prompt_error = None;
        ui.add_enabled_ui(!model.running && !model.loading && !model.compiling && !model.stopping, |ui| {
            let prompt_width = (ui.available_width() - 20.0).max(40.0);
            let (response, validation) = self.prompt_editor.show(
                ui, &mut self.prompt, model.tokenizer_path, prompt_width,
            );
            prompt_error = validation.error;
            if response.changed() {
                actions.push(UiAction::SetPrompt(self.prompt.clone()));
            }
            ui.add_space(14.0);
            let mut changed = false;
            self.parallel_warning_rect = Rect::NOTHING;
            // Reserve the warning's UI scope even while hidden, so inserting it
            // cannot change the input's ID or interrupt keyboard editing.
            ui.push_id("parallel_inference_warning", |ui| {
                if self.settings.parallel_sequences > 16 {
                    let warning = egui::Frame::none()
                        .fill(Color32::from_rgb(255, 242, 184))
                        .inner_margin(egui::Margin::symmetric(7.0, 5.0))
                        .rounding(4.0)
                        .show(ui, |ui| {
                            ui.set_width((ui.available_width() - 14.0).max(30.0));
                            ui.add(egui::Label::new(
                                RichText::new("并行数过大可能导致爆显存")
                                    .size(11.0).color(Color32::from_rgb(119, 75, 0)),
                            ).wrap(true));
                        });
                    self.parallel_warning_rect = warning.response.rect;
                    ui.add_space(6.0);
                }
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("并行推理数").color(MUTED));
                ui.add_space((ui.available_width() - 72.0).max(0.0));
                let response = ui.add_sized(
                    [72.0, 26.0],
                    egui::DragValue::new(&mut self.settings.parallel_sequences)
                        .clamp_range(1..=usize::MAX)
                        .speed(1),
                );
                self.parallel_control_rect = response.rect;
                changed |= response.changed();
                response
                    .on_hover_text("1 为单分支；大于 1 时，从当前状态同时创建对应数量的独立分支。");
            });
            if self.settings.parallel_sequences > 1 {
                ui.add(
                    egui::Label::new(
                        RichText::new("默认显示第 1 分支；可在状态树切换查看其他画布。")
                            .size(11.0)
                            .color(MUTED),
                    )
                    .wrap(true),
                );
            }
            changed |= ui.checkbox(&mut self.settings.context_overclock, "上下文超频")
                .on_hover_text("用满1024总上下文；优先保留全局、粗网格图元，再按距离保留细网格。始终按完整区域保留，并为当前区域生成留足空间。不会删除画布中的图元。")
                .changed();
            ui.horizontal(|ui|{
                let regional=self.redraw_view.is_some();
                let stage_pause = ui.add_enabled(!regional,egui::Checkbox::new(&mut self.settings.pause_at_stage_end, RichText::new("阶段结束时提醒及中断模型").size(11.)));
                self.pause_stage_end_control_rect = stage_pause.rect;
                changed |= stage_pause.changed();
                stage_pause.on_hover_text("全局或整个粗网格阶段完成时提醒并中断模型，手动继续才进入下一阶段。手工修改的块已达上限但模型尚未结束时，也会中断并提示先调整图形；普通单块结束不会暂停。")
                    .on_disabled_hover_text("局部修改会在所选区域完成后停止；此设置保留用于普通绘制。");
                let half_switch=ui.add_enabled(!regional,egui::Checkbox::new(&mut self.settings.half_region_switch,RichText::new("半程自动换块").size(11.)));
                self.half_region_switch_control_rect=half_switch.rect;
                changed|=half_switch.changed();
                half_switch.on_hover_text("仅影响粗网格和细网格：一个块累计达到粗网格10个、细网格5个图元时，暂时换到其他块；再次回访继续绘制到粗网格20个、细网格10个的完整限额。全局绘制阶段不受影响。")
                    .on_disabled_hover_text("局部修改会依次完成所选区域；此设置保留用于普通绘制。");
            });
            ui.horizontal(|ui| {
                let automatic = ui.add_enabled(self.redraw_view.is_none(),egui::Checkbox::new(
                    &mut self.settings.auto_reorder_before_generation,
                    "生成前自动重排序",
                ));
                self.auto_reorder_control_rect = automatic.rect;
                changed |= automatic.changed();
                automatic.on_hover_text("每次开始或继续生成前，仅在原绘制区域内按贡献从大到小排序：全局阶段内部、每个粗网格内部、每个细网格内部各自排序，区域顺序不变。贡献为RGB与背景色的差值平方和乘以图形面积；背景始终保留在最底层。")
                    .on_disabled_hover_text("局部修改保留已有图元顺序，不执行自动重排序；此设置保留用于普通绘制。");
            });
            let sampling = egui::CollapsingHeader::new("高级推理设置")
                .id_source("sampling")
                .show(ui, |ui| {
                    egui::Grid::new("sampling_fields")
                        .num_columns(2)
                        .spacing([10.0, 12.0])
                        .show(ui, |ui| {
                            ui.label("推理模式");
                            egui::ComboBox::from_id_source("acceleration_mode")
                                .selected_text(self.settings.acceleration.mode.label())
                                .width(110.0)
                                .show_ui(ui, |ui| {
                                    for mode in [AccelerationMode::Auto, AccelerationMode::Compatible, AccelerationMode::Optimized] {
                                        changed |= ui.selectable_value(&mut self.settings.acceleration.mode, mode, mode.label()).changed();
                                    }
                                }).response.on_hover_text("自动：按当前后端能力选择。普通兼容：保留GPU模型执行，关闭可选加速并使用CPU采样。优先加速：尝试当前后端支持的加速能力，不支持时保留普通路径。");
                            ui.end_row();
                            let optional = self.settings.acceleration.mode != AccelerationMode::Compatible;
                            for (id, label, value) in [
                                ("acceleration_fusion", "算子融合", &mut self.settings.acceleration.fusion),
                                ("acceleration_split_k", "Split-K注意力", &mut self.settings.acceleration.split_k),
                                ("acceleration_graph", "整图元Graph", &mut self.settings.acceleration.primitive_graph),
                            ] {
                                ui.label(label).on_hover_text(if id == "acceleration_graph" {
                                    "整图元Graph需要融合解码器和GPU采样。自动模式仅用于全局阶段，避免网格切换时频繁捕获；优先加速或明确开启可用于区域阶段，但捕获成本可能抵消收益。"
                                } else {
                                    "按当前后端能力选择；普通兼容模式不使用这些可选加速。"
                                });
                                changed |= acceleration_feature_picker(ui, id, value, optional);
                                ui.end_row();
                            }
                            ui.label("KV INT8").on_hover_text("自动或开启时尝试压缩生成缓存，减少显存占用；不支持时保留浮点缓存并记录原因。普通兼容模式始终使用浮点缓存。");
                            changed |= acceleration_feature_picker(ui, "acceleration_kv_int8", &mut self.settings.acceleration.kv_int8, optional);
                            ui.end_row();
                            ui.label("CFG");
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut self.settings.cfg)
                                        .clamp_range(0.0..=32.0)
                                        .speed(0.1)
                                        .max_decimals(2),
                                )
                                .changed();
                            ui.end_row();
                            ui.label("Rényi α");
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut self.settings.renyi_alpha)
                                        .clamp_range(1.01..=32.0)
                                        .speed(0.1)
                                        .max_decimals(2),
                                )
                                .changed();
                            ui.end_row();
                            ui.label("种子模式");
                            let was_auto=self.settings.auto_seed;
                            let automatic=ui.checkbox(&mut self.settings.auto_seed,"自动随机种子");
                            self.auto_seed_control_rect=automatic.rect;
                            changed|=automatic.changed();
                            if automatic.changed(){
                                if was_auto&&!self.settings.auto_seed {self.settings.seed=model.settings.seed;}
                                self.manual_seed_text=self.settings.seed.to_string();
                            }
                            automatic.on_hover_text(format!("预览值持续随时间变化，点击生成时才确定本次实际种子并固定保存。当前状态实际种子：{}。关闭后可手动填写种子复现。",model.settings.seed));
                            ui.end_row();
                            ui.label("随机种子");
                            let response=if self.settings.auto_seed {
                                let preview=crate::auto_seed::preview_seed();
                                self.seed_display_value=preview;
                                let mut text=preview.to_string();
                                ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
                                ui.add_enabled_ui(false,|ui|ui.add(egui::TextEdit::singleline(&mut text)
                                    .id_source("random_seed_value").desired_width(148.).font(egui::TextStyle::Monospace))).inner
                            }else{
                                self.seed_display_value=self.settings.seed;
                                let invalid=self.manual_seed_text.parse::<u64>().is_err();
                                let mut input=egui::TextEdit::singleline(&mut self.manual_seed_text)
                                    .id_source("random_seed_value").desired_width(148.).font(egui::TextStyle::Monospace);
                                if invalid{input=input.text_color(Color32::from_rgb(184,57,70));}
                                let response=ui.add(input);
                                if response.changed(){if let Ok(seed)=self.manual_seed_text.parse::<u64>(){
                                    self.settings.seed=seed;self.seed_display_value=seed;changed=true;
                                }}
                                response
                            };
                            self.seed_control_rect=response.rect;self.seed_control_enabled=response.enabled();
                            response.on_hover_text("手动种子需为0到18446744073709551615之间的整数。")
                                .on_disabled_hover_text(format!("自动模式显示下一次种子的时间预览；当前状态实际种子：{}。真正生成种子固定保存在状态节点，关闭自动后可手动修改。",model.settings.seed));
                            ui.end_row();
                            ui.label("采样后端");
                            let sampler = ui.add_enabled_ui(optional, |ui| {
                                egui::ComboBox::from_id_source("sampling_backend")
                                    .selected_text(if optional { self.settings.sampling_backend.label() } else { "CPU（兼容）" })
                                    .width(110.0)
                                    .show_ui(ui, |ui| {
                                        for backend in [SamplingBackend::Auto, SamplingBackend::Gpu, SamplingBackend::Cpu] {
                                            changed |= ui.selectable_value(
                                                &mut self.settings.sampling_backend,
                                                backend,
                                                backend.label(),
                                            ).changed();
                                        }
                                    }).response
                            }).inner;
                            sampler
                                .on_hover_text("只选择采样运算，不更换模型执行设备。自动模式跟随当前执行路径；整图元Graph需要GPU采样与融合解码器。GPU使用FP32，与CPU同一种子时结果可能不同。")
                                .on_disabled_hover_text("普通兼容模式固定使用CPU采样，模型仍在GPU。原采样选择已保留，离开兼容模式后恢复。");
                            ui.end_row();
                        });
                });
            self.sampling_header_rect=sampling.header_response.rect;
            if changed {
                actions.push(UiAction::SetSettings(self.settings.clone()));
            }
        });
        ui.add_space(5.0);
        ui.horizontal(|ui| {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let manual = ui.add_enabled(!model.loading && !model.stopping, egui::Button::new("重排序"));
                self.manual_reorder_rect = manual.rect;
                if manual.clicked() { actions.push(UiAction::ReorderByContribution); }
                manual.on_hover_text("立即仅在原绘制区域内按贡献值重排序，区域之间的顺序不变，保留背景和所有图元。生成期间编辑会先安全停止生成。");
            });
        });
        ui.add_space(12.0);
        if model.loading {
            ui.add_enabled(
                false,
                egui::Button::new("正在加载模型…")
                    .min_size(Vec2::new(ui.available_width(), 42.0))
                    .rounding(7.0),
            );
        } else if model.running {
            let button = egui::Button::new(
                RichText::new(if model.stopping {
                    "正在完成停止…"
                } else if model.settings.parallel_sequences > 1 {
                    "停止全部分支"
                } else {
                    "停止生成"
                })
                .strong()
                .color(Color32::from_rgb(184, 57, 70)),
            )
            .fill(Color32::from_rgb(255, 239, 241))
            .stroke(Stroke::NONE)
            .rounding(7.0);
            let response = ui.add_enabled(
                    !model.stopping,
                    button.min_size(Vec2::new(ui.available_width(), 42.0)),
                );
            self.generation_button_rect = response.rect;
            if response.clicked() {
                actions.push(UiAction::Stop);
            }
            ui.add_space(7.0);
            ui.label(
                RichText::new(if model.settings.parallel_sequences > 1 {
                    "停止时保留各分支已完成图元"
                } else {
                    "停止时保留已完成图元"
                })
                .size(11.0)
                .color(MUTED),
            );
        } else if let Some(reason) = model.drawing_region.generation_block_reason().filter(|_|self.redraw_view.is_none()) {
            prompt_editor::generation_button(ui, "继续生成", Some(reason));
        } else if !model.model_loaded {
            let selected_exists = Path::new(&self.model_path).is_file();
            let button = egui::Button::new(RichText::new("打开设置").strong().color(Color32::WHITE))
                .fill(BLUE)
                .stroke(Stroke::NONE)
                .rounding(7.0)
                .min_size(Vec2::new(ui.available_width(), 42.0));
            let response = ui.add(button);
            self.open_settings_rect = response.rect;
            if response.clicked() {
                self.page = Page::Settings;
            }
            ui.add_space(7.0);
            if !selected_exists {
                ui.add(egui::Label::new(RichText::new("未找到权重，可在设置中下载官方模型或选择本地文件。").size(11.0).color(Color32::from_rgb(171, 66, 53))).wrap(true));
            } else {
                ui.label(
                    RichText::new("本地权重尚未加载。可在设置页面查看状态或重试。")
                        .size(11.0)
                        .color(MUTED),
                );
            }
        } else {
            let text=if self.redraw_view.as_ref().is_some_and(|view|view.can_continue_global){
                    if self.settings.parallel_sequences>1{format!("继续全图绘制 · {} 个分支",self.settings.parallel_sequences)}else{"继续全图绘制".into()}}
                else if self.redraw_view.is_some(){if self.settings.parallel_sequences>1{format!("生成所选区域 · {} 个分支",self.settings.parallel_sequences)}else{"生成所选区域".into()}}
                else{generation_button_caption(&self.settings,model.drawing_region,model.primitives.is_empty())};
            let response = prompt_editor::generation_button(
                ui, &text, model.drawing_region.generation_block_reason().filter(|_|self.redraw_view.is_none()).or(prompt_error.as_deref())
                    .or_else(||(!self.settings.auto_seed&&self.manual_seed_text.parse::<u64>().is_err()).then_some("随机种子必须为0到18446744073709551615之间的整数")),
            );
            self.generation_button_rect = response.rect;
            if response.clicked() {
                if let Some(view)=self.redraw_view.as_ref().filter(|view|!view.can_generate){
                    let reason=view.reason.clone().unwrap_or_else(||"请先选择要重画的区域，并手动删除这些区域已有的图元。".into());self.notify(reason);
                }else{actions.push(UiAction::Generate {prompt:self.prompt.clone(),settings:self.settings.clone()});}
            }
        }
        if let Some(error) = self.file_selection_error.as_deref().or(model.error) {
            ui.add_space(9.0);
            egui::Frame::none()
                .fill(Color32::from_rgb(255, 244, 242))
                .rounding(6.0)
                .inner_margin(9.0)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(error)
                            .size(11.0)
                            .color(Color32::from_rgb(171, 66, 53)),
                    );
                });
        }
    }

    pub fn poll_model_download(&mut self,idle:bool)->Option<UiAction>{
        let paths=self.model_download.poll(idle)?;
        self.model_path=paths.model.display().to_string();
        self.tokenizer_path=paths.tokenizer.display().to_string();
        self.paths_initialized=true;self.file_selection_error=None;
        Some(UiAction::LoadModel{checkpoint_path:self.model_path.clone(),tokenizer_path:self.tokenizer_path.clone()})
    }

    fn model_panel(&mut self, ui: &mut egui::Ui, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        egui::CollapsingHeader::new("模型与运行环境")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Label::new(RichText::new(model.model_status).size(11.0).color(MUTED))
                        .wrap(true),
                );
                ui.add_space(8.0);
                self.model_download.show(ui,!model.running&&!model.loading&&!model.compiling&&!model.stopping);
                ui.separator();
                ui.add_enabled_ui(!model.running && !model.loading && !model.stopping, |ui| {
                    let filename = Path::new(&self.model_path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("尚未选择权重");
                    ui.add(
                        egui::Label::new(
                            RichText::new(format!("权重：{filename}"))
                                .size(11.0)
                                .color(MUTED),
                        )
                        .wrap(true),
                    );
                    if ui
                        .add_sized(
                            [ui.available_width(), 30.0],
                            egui::Button::new(if model.model_loaded {
                                "更换模型权重…"
                            } else {
                                "浏览模型权重…"
                            }),
                        )
                        .clicked()
                    {
                        self.browse_model(actions);
                    }
                    ui.add_space(7.0);
                    ui.add(
                        egui::Label::new(
                            RichText::new(if Path::new(&self.tokenizer_path).is_file() {
                                "分词器：安装包已提供"
                            } else {
                                "分词器缺失，请修复安装"
                            })
                            .size(11.0)
                            .color(MUTED),
                        )
                        .wrap(true),
                    );
                    egui::CollapsingHeader::new("高级路径设置")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(RichText::new("模型权重").size(11.0).color(MUTED));
                            ui.horizontal(|ui| {
                                let width = (ui.available_width() - 62.0).max(40.0);
                                ui.add_sized(
                                    [width, 26.0],
                                    egui::TextEdit::singleline(&mut self.model_path)
                                        .desired_width((width - 8.0).max(24.0))
                                        .min_size(Vec2::ZERO)
                                        .clip_text(true)
                                        .hint_text("model.pt"),
                                );
                                if ui
                                    .add_sized([54.0, 26.0], egui::Button::new("浏览…"))
                                    .clicked()
                                {
                                    self.browse_model(actions);
                                }
                            });
                            ui.add_space(7.0);
                            ui.label(
                                RichText::new("分词器（随应用提供）")
                                    .size(11.0)
                                    .color(MUTED),
                            );
                            ui.horizontal(|ui| {
                                let width = (ui.available_width() - 62.0).max(40.0);
                                ui.add_sized(
                                    [width, 26.0],
                                    egui::TextEdit::singleline(&mut self.tokenizer_path)
                                        .desired_width((width - 8.0).max(24.0))
                                        .min_size(Vec2::ZERO)
                                        .clip_text(true)
                                        .hint_text("caption_spm_1500.model"),
                                );
                                if ui
                                    .add_sized([54.0, 26.0], egui::Button::new("浏览…"))
                                    .clicked()
                                {
                                    self.browse_tokenizer();
                                }
                            });
                            ui.add_space(8.0);
                            if ui
                                .add_enabled(
                                    Path::new(&self.model_path).is_file(),
                                    egui::Button::new("加载所选文件")
                                        .min_size(Vec2::new(ui.available_width(), 30.0)),
                                )
                                .clicked()
                            {
                                self.file_selection_error = None;
                                actions.push(UiAction::LoadModel {
                                    checkpoint_path: self.model_path.clone(),
                                    tokenizer_path: self.tokenizer_path.clone(),
                                });
                            }
                        });
                });
            });
    }

    fn settings_page(&mut self, ui: &mut egui::Ui, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        egui::ScrollArea::vertical()
            .id_source("settings_page_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(24.0);
                ui.horizontal(|ui| {
                    ui.add_space(26.0);
                    ui.vertical(|ui| {
                        ui.set_max_width((ui.available_width() - 26.0).min(840.0).max(100.0));
                        section_title(ui, "设置", Some("下载官方模型，或选择已有本地权重；启动时自动使用已保存的位置"));
                        ui.add_space(18.0);
                        egui::Frame::none().fill(Color32::WHITE).rounding(8.0)
                            .inner_margin(18.0).show(ui, |ui| {
                                ui.set_max_width(ui.available_width());
                                self.model_panel(ui, model, actions);
                            });
                        ui.add_space(16.0);
                        egui::Frame::none().fill(Color32::WHITE).rounding(8.0)
                            .inner_margin(18.0).show(ui, |ui| {
                                section_title(ui,"运行日志",None);
                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    let open=ui.button("打开日志目录");
                                    self.log_directory_rect=open.rect;
                                    if open.clicked(){actions.push(UiAction::OpenLogs);}
                                    let export=ui.button("导出诊断").on_hover_text("导出运行环境、执行路径与错误记录；默认不包含提示词或权重内容。");
                                    self.diagnostics_export_rect=export.rect;
                                    if export.clicked(){actions.push(UiAction::ExportDiagnostics);}
                                });
                                if let Some(path)=crate::diagnostics::directory(){
                                    ui.add_space(6.0);
                                    ui.add(egui::Label::new(RichText::new(path.to_string_lossy()).size(11.0).color(MUTED)).wrap(true));
                                }
                            });
                        if model.running {
                            ui.add_space(10.0);
                            ui.label(RichText::new("生成期间不能更换模型，请先停止生成。").size(12.0).color(MUTED));
                            if ui.add_enabled(!model.stopping, egui::Button::new("停止生成")).clicked() {
                                actions.push(UiAction::Stop);
                            }
                        }
                        if let Some(error) = self.file_selection_error.as_deref().or(model.error) {
                            ui.add_space(12.0);
                            ui.add(egui::Label::new(RichText::new(error).size(12.0)
                                .color(Color32::from_rgb(171, 66, 53))).wrap(true));
                        }
                    });
                });
            });
    }

    fn apply_model_selection(
        &mut self,
        selected: Option<PathBuf>,
    ) -> Result<Option<UiAction>, String> {
        let Some(path) = selected_file_text(selected.as_deref(), LocalFileKind::Model)? else {
            return Ok(None);
        };
        self.model_path = path;
        self.file_selection_error = None;
        Ok(Some(UiAction::LoadModel {
            checkpoint_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
        }))
    }

    fn browse_model(&mut self, actions: &mut Vec<UiAction>) {
        self.file_selection_error = None;
        match choose_local_file(&self.model_path, LocalFileKind::Model)
            .and_then(|path| self.apply_model_selection(path))
        {
            Ok(Some(action)) => actions.push(action),
            Ok(None) => {}
            Err(error) => self.file_selection_error = Some(error),
        }
    }

    fn browse_tokenizer(&mut self) {
        self.file_selection_error = None;
        match choose_local_file(&self.tokenizer_path, LocalFileKind::Tokenizer)
            .and_then(|path| selected_file_text(path.as_deref(), LocalFileKind::Tokenizer))
        {
            Ok(Some(path)) => self.tokenizer_path = path,
            Ok(None) => {}
            Err(error) => self.file_selection_error = Some(error),
        }
    }

    fn canvas_header(&mut self, ui: &mut egui::Ui, model: &UiModel<'_>, actions:&mut Vec<UiAction>) {
        egui::Frame::none()
            .fill(Color32::from_rgb(247, 249, 252))
            .inner_margin(egui::Margin::symmetric(18.0, 11.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("当前画布").strong().color(INK));
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(if model.loading {
                            "模型加载中 · 画布只读".to_owned()
                        } else if model.stopping {
                            "正在停止 · 画布只读".to_owned()
                        } else if model.running {
                            "生成中 · 可浏览或编辑画布".to_owned()
                        } else if self.canvas.selection.is_empty() {
                            "512 × 512".to_owned()
                        } else {
                            format!("已选择 {} 个图元", self.canvas.selection.len())
                        })
                        .size(11.0)
                        .color(MUTED),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.small_button("适应窗口  F").clicked() {
                            self.canvas.fit();
                        }
                        ui.label(
                            RichText::new(format!("{:.0}%", self.canvas.zoom * 100.0))
                                .size(11.0)
                                .color(MUTED),
                        );
                    });
                });
                let selected_ids=self.canvas.selection.iter().copied().collect::<HashSet<_>>();
                let selected=model.primitives.iter().filter(|p|selected_ids.contains(&p.id)).collect::<Vec<_>>();
                let mut ids=selected.iter().filter(|p|p.shape!=ShapeKind::Background).map(|p|p.id).collect::<Vec<_>>();ids.sort_unstable();ids.dedup();
                self.stage_control_active=[false;3];let mut complete=true;
                for primitive in &selected {
                    let stage=if primitive.shape==ShapeKind::Background{Some(Stage::Global)}else{model.stage_by_id.and_then(|map|map.get(&primitive.id).copied())};
                    match stage{Some(Stage::Global)=>self.stage_control_active[0]=true,Some(Stage::Coarse)=>self.stage_control_active[1]=true,
                        Some(Stage::Fine)=>self.stage_control_active[2]=true,_=>complete=false}
                }
                ui.add_space(6.);ui.horizontal(|ui|{
                    ui.label(RichText::new("图元阶段").size(11.).color(MUTED));
                    let enabled=!ids.is_empty()&&complete&&!model.loading&&!model.compiling&&!model.stopping;
                    for (index,(stage,label)) in [(Stage::Global,"全"),(Stage::Coarse,"粗"),(Stage::Fine,"细")].into_iter().enumerate(){
                        let response=ui.add_enabled_ui(enabled,|ui|ui.add_sized([64.,24.],egui::SelectableLabel::new(self.stage_control_active[index],label))).inner;
                        self.stage_control_rects[index]=response.rect;
                        if response.clicked(){actions.push(UiAction::AssignStage{ids:ids.clone(),stage});}
                        response.on_hover_text("调整选中图元所属的绘制阶段，画面保持不变。背景固定属于全图阶段。")
                            .on_disabled_hover_text("请先选择普通图元；背景固定属于全图阶段，模型加载或停止期间不能修改。");
                    }
                });
            });
    }

    fn canvas_pair(&mut self, ui: &mut egui::Ui, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        let area = ui.available_rect_before_wrap();
        let (edit, indicator) = split_canvas_pair(area);
        ui.allocate_ui_at_rect(edit, |ui| {
            self.canvas_header(ui, model, actions);
            let remaining = ui.available_rect_before_wrap();
            let footer_height = 76.0_f32.min(remaining.height() * 0.4);
            let viewport =
                Rect::from_min_max(remaining.min, remaining.max - Vec2::new(0.0, footer_height));
            self.edit_canvas_rect = viewport;
            ui.allocate_ui_at_rect(viewport, |ui| {
                if let Some(edit) = self.canvas.show(
                    ui,
                    model.primitives,
                    ui.is_enabled() && !model.loading && !model.stopping,
                    model.current_node,
                ) {
                    push_edit(actions, edit);
                }
            });
            let footer = Rect::from_min_max(
                Pos2::new(remaining.left(), viewport.bottom()),
                remaining.max,
            );
            ui.painter()
                .rect_filled(footer, 0.0, Color32::from_rgb(247, 249, 252));
            ui.painter().text(
                footer.left_top() + Vec2::new(16.0, 14.0),
                egui::Align2::LEFT_CENTER,
                drawing_stage_label(model.drawing_region),
                FontId::proportional(11.0),
                MUTED,
            );
            let scrubber=Rect::from_min_max(footer.min+Vec2::new(10.,28.),footer.max-Vec2::new(10.,4.));
            ui.allocate_ui_at_rect(scrubber,|ui|{
                let enabled=ui.is_enabled()&&!model.loading&&!model.compiling&&!model.stopping
                    &&!self.show_shortcuts&&!self.prompt_templates.is_open();
                let output=self.history_scrubber.show(ui,model.document,model.current_node,enabled);
                if output.freeze_view&&model.running {actions.push(UiAction::FreezeGenerationView);}
                if let Some(node)=output.target {actions.push(UiAction::BrowseHistory(node));}
            });
        });
        ui.allocate_ui_at_rect(indicator, |ui| {
            if let Some(redraw)=&self.redraw_view{
                let enabled=ui.is_enabled()&&!model.running&&!model.loading&&!model.compiling&&!model.stopping;
                let output=self.redraw_panel.show(ui,redraw,enabled);
                self.region_indicator_rect=output.panel;
                if model.running{ui.interact(output.panel,egui::Id::new("redraw_busy_hint"),egui::Sense::hover()).on_hover_text("阶段可视化可随时调整；选择或清空重绘区域前，请先停止生成。");}
                if output.clear_locked{actions.push(UiAction::DeleteLockedRedrawRegions);}
                for(stage,index)in output.toggles{actions.push(UiAction::ToggleRedrawRegion{stage,index});}
                if let Some(hint)=output.locked_hint{self.notify(hint);}
            }else{
                let geometry = show_region_indicator(ui, model.primitives, model.drawing_region);
                self.region_indicator_rect = geometry.panel;
            }
        });
    }

    fn properties_panel(
        &mut self,
        ui: &mut egui::Ui,
        model: &UiModel<'_>,
        actions: &mut Vec<UiAction>,
    ) {
        section_title(ui, "图元属性", None);
        let selected = self.canvas.selected(model.primitives);
        if selected.is_empty() {
            ui.add_space(10.0);
            ui.label(
                RichText::new("选择画布中的图元，即可编辑属性。\n拖动空白区域可以框选。")
                    .size(12.0)
                    .color(MUTED),
            );
            self.property = None;
            return;
        }
        let bounds = selection_bounds(&selected).unwrap();
        let ids: Vec<u64> = selected.iter().map(|p| p.id).collect();
        if self
            .property
            .as_ref()
            .map(|p| p.node != model.current_node || p.ids != ids)
            .unwrap_or(true)
        {
            self.property = Some(PropertyDraft {
                node: model.current_node,
                ids: ids.clone(),
                x: bounds.center().x,
                y: bounds.center().y,
                width: if selected.len() == 1 {
                    selected[0].width
                } else {
                    bounds.width()
                },
                height: if selected.len() == 1 {
                    selected[0].height
                } else {
                    bounds.height()
                },
                angle: if selected.len() == 1 {
                    selected[0].angle_deg
                } else {
                    0.0
                },
                geometry_dirty: false,
            });
        }
        let draft = self.property.as_mut().unwrap();
        let only_background = selected.iter().all(|p| p.shape == ShapeKind::Background);
        ui.add_space(8.0);
        ui.label(
            RichText::new(if selected.len() == 1 {
                format!("{}  ·  #{}", selected[0].shape.label(), selected[0].id)
            } else {
                format!("多选  ·  {} 个图元", selected.len())
            })
            .size(12.0)
            .color(MUTED),
        );
        ui.add_space(10.0);
        ui.add_enabled_ui(!model.loading && !model.stopping, |ui| {
            if !only_background {
                egui::Grid::new("geometry")
                    .num_columns(4)
                    .spacing([8.0, 10.0])
                    .show(ui, |ui| {
                        ui.label("X");
                        draft.geometry_dirty |= ui
                            .add(
                                egui::DragValue::new(&mut draft.x)
                                    .clamp_range(0.0..=511.5)
                                    .speed(0.5)
                                    .max_decimals(1),
                            )
                            .changed();
                        ui.label("Y");
                        draft.geometry_dirty |= ui
                            .add(
                                egui::DragValue::new(&mut draft.y)
                                    .clamp_range(0.0..=511.5)
                                    .speed(0.5)
                                    .max_decimals(1),
                            )
                            .changed();
                        ui.end_row();
                        let old_width = draft.width;
                        let old_height = draft.height;
                        ui.label("宽");
                        let width_changed = ui
                            .add(
                                egui::DragValue::new(&mut draft.width)
                                    .clamp_range(0.25..=512.0)
                                    .speed(0.5)
                                    .max_decimals(2),
                            )
                            .changed();
                        if width_changed && selected.len() > 1 {
                            draft.height = old_height * draft.width / old_width.max(0.25);
                        }
                        draft.geometry_dirty |= width_changed;
                        ui.label("高");
                        let height_changed = ui
                            .add(
                                egui::DragValue::new(&mut draft.height)
                                    .clamp_range(0.25..=512.0)
                                    .speed(0.5)
                                    .max_decimals(2),
                            )
                            .changed();
                        if height_changed && selected.len() > 1 {
                            draft.width = old_width * draft.height / old_height.max(0.25);
                        }
                        draft.geometry_dirty |= height_changed;
                        ui.end_row();
                        ui.label("旋转");
                        draft.geometry_dirty |= ui
                            .add(
                                egui::DragValue::new(&mut draft.angle)
                                    .speed(0.5)
                                    .suffix("°")
                                    .max_decimals(1),
                            )
                            .changed();
                        ui.end_row();
                    });
                ui.add_space(10.0);
            }
            ui.horizontal(|ui| {
                ui.label("颜色");
                let color = selected[0].color;
                let (swatch, _) = ui.allocate_exact_size(Vec2::new(20.0, 16.0), egui::Sense::hover());
                ui.painter().rect_filled(swatch, 3.0, Color32::from_rgb(color[0], color[1], color[2]));
                ui.label(
                    RichText::new(format!(
                        "#{:02X}{:02X}{:02X}",
                        color[0], color[1], color[2]
                    ))
                    .size(11.0)
                    .color(MUTED),
                );
            });
            ui.label(RichText::new(if only_background { "在左侧“背景颜色”中调整" }
                else { "在左侧“图形颜色”中调整" }).size(10.0).color(MUTED));
            ui.add_space(10.0);
            if ui
                .add_enabled(
                    draft.geometry_dirty,
                    egui::Button::new("应用属性").min_size(Vec2::new(ui.available_width(), 29.0)),
                )
                .clicked()
            {
                let mut next = model.primitives.to_vec();
                let sx = draft.width
                    / if selected.len() == 1 {
                        selected[0].width.max(0.25)
                    } else {
                        bounds.width().max(0.25)
                    };
                let sy = draft.height
                    / if selected.len() == 1 {
                        selected[0].height.max(0.25)
                    } else {
                        bounds.height().max(0.25)
                    };
                let angle_delta = if selected.len() == 1 {
                    draft.angle - selected[0].angle_deg
                } else {
                    draft.angle
                };
                let (sin, cos) = angle_delta.to_radians().sin_cos();
                for p in &mut next {
                    if ids.contains(&p.id) {
                        if draft.geometry_dirty && p.shape != ShapeKind::Background {
                            let dx = (p.x - bounds.center().x) * sx;
                            let dy = (p.y - bounds.center().y) * sy;
                            p.x = draft.x + dx * cos - dy * sin;
                            p.y = draft.y + dx * sin + dy * cos;
                            p.width *= sx;
                            p.height *= sy;
                            p.angle_deg += angle_delta;
                        }
                    }
                }
                actions.push(UiAction::CommitCanvas {
                    label: format!("修改 {} 个图元的属性", ids.len()),
                    primitives: next,
                });
            }
            ui.add_space(8.0);
            ui.columns(2, |columns| {
                let left_width = columns[0].available_width();
                let right_width = columns[1].available_width();
                if columns[0]
                    .add_sized([left_width, 27.0], egui::Button::new("上移一层"))
                    .clicked()
                {
                    if let Some(edit) = self.canvas.reorder(model.primitives, true, false) {
                        push_edit(actions, edit);
                    }
                }
                if columns[1]
                    .add_sized([right_width, 27.0], egui::Button::new("下移一层"))
                    .clicked()
                {
                    if let Some(edit) = self.canvas.reorder(model.primitives, false, false) {
                        push_edit(actions, edit);
                    }
                }
            });
            ui.add_space(5.0);
            ui.columns(2, |columns| {
                let left_width = columns[0].available_width();
                let right_width = columns[1].available_width();
                if columns[0]
                    .add_sized([left_width, 27.0], egui::Button::new("置于顶层"))
                    .clicked()
                {
                    if let Some(edit) = self.canvas.reorder(model.primitives, true, true) {
                        push_edit(actions, edit);
                    }
                }
                if columns[1]
                    .add_sized([right_width, 27.0], egui::Button::new("置于底层"))
                    .clicked()
                {
                    if let Some(edit) = self.canvas.reorder(model.primitives, false, true) {
                        push_edit(actions, edit);
                    }
                }
            });
            ui.add_space(5.0);
            ui.columns(2, |columns| {
                let left_width = columns[0].available_width();
                let right_width = columns[1].available_width();
                if columns[0]
                    .add_enabled(
                        !only_background,
                        egui::Button::new("复制").min_size(Vec2::new(left_width, 27.0)),
                    )
                    .on_hover_text(if only_background {
                        "背景是唯一底色记录"
                    } else {
                        "创建所选图元的副本"
                    })
                    .clicked()
                {
                    if let Some(edit) = self.canvas.duplicate(model.primitives) {
                        push_edit(actions, edit);
                    }
                }
                if columns[1]
                    .add_sized(
                        [right_width, 27.0],
                        egui::Button::new(
                            RichText::new("删除").color(Color32::from_rgb(188, 69, 76)),
                        ),
                    )
                    .clicked()
                {
                    if let Some(edit) = self.canvas.delete(model.primitives) {
                        push_edit(actions, edit);
                    }
                }
            });
        });
        if self.property.as_ref().is_some_and(|draft| draft.geometry_dirty) {
            self.request_view_freeze(model, actions);
        }
    }

    fn node_details(&self, ui: &mut egui::Ui, model: &UiModel<'_>) {
        section_title(ui, "当前状态", None);
        if let Some(node) = model.nodes.iter().find(|n| n.id == model.current_node) {
            ui.add_space(8.0);
            ui.label(RichText::new(&node.title).strong().color(INK));
            ui.add_space(6.0);
            ui.label(
                RichText::new(format!("节点 #{}  ·  {}", node.id, node.source))
                    .size(12.0)
                    .color(MUTED),
            );
            ui.label(
                RichText::new(format!("{} 个图元", node.primitive_count))
                    .size(12.0)
                    .color(MUTED),
            );
            ui.add_space(14.0);
            ui.label(
                RichText::new("点击任意历史节点即可回到该状态；继续编辑或生成会形成新分支。")
                    .size(12.0)
                    .color(MUTED),
            );
        }
    }

    fn history(&mut self, ui: &mut egui::Ui, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        if let Some(action) = self.tree.show(ui, model, &mut self.thumbnails) {
            match action {
                crate::tree_view::TreeAction::Jump(id) => actions.push(UiAction::Jump(id)),
                crate::tree_view::TreeAction::DeleteSubtrees(ids) if !self.show_shortcuts
                    && !self.prompt_templates.is_open() => {
                        actions.retain(|action|!matches!(action,UiAction::Jump(_)));
                        actions.push(UiAction::DeleteHistory(ids));
                    },
                crate::tree_view::TreeAction::DeleteSubtrees(_) => {},
            }
        }
    }

    fn keyboard(&mut self, ctx: &egui::Context, model: &UiModel<'_>, actions: &mut Vec<UiAction>) {
        if model.loading || model.compiling || model.stopping || self.page == Page::Settings
            || self.prompt_templates.is_open() || ctx.wants_keyboard_input() {
            return;
        }
        if self.page == Page::Canvas && self.canvas.tool == Tool::Select {
            let (copy, pasted, raw_paste) = ctx.input_mut(|input| {
                let mut copy = false;
                let mut pasted = None;
                let mut raw_paste = false;
                input.events.retain(|event| {
                    match event {
                        egui::Event::Copy => copy = true,
                        egui::Event::Paste(text) => {
                            if pasted.is_none() { pasted = Some(text.clone()); }
                        }
                        egui::Event::Key { key, pressed, modifiers, .. }
                            if (modifiers.command || modifiers.ctrl)
                                && matches!(key, egui::Key::C | egui::Key::V) => {
                            if *pressed {
                                if *key == egui::Key::C { copy = true; }
                                else { raw_paste = true; }
                            }
                        }
                        _ => return true,
                    }
                    false
                });
                (copy, pasted, raw_paste)
            });
            if copy {
                if let Some(payload) = self.canvas.copy_payload(model.primitives) {
                    ctx.output_mut(|output| output.copied_text = payload);
                }
            }
            let edit = if let Some(payload) = pasted {
                // A real OS paste takes precedence over raw-key compatibility.
                // Unrelated clipboard text must never paste old internal shapes.
                self.canvas.paste_payload(model.primitives, &payload)
            } else if raw_paste {
                self.canvas.paste(model.primitives)
            } else { None };
            if let Some(edit) = edit {
                actions.push(UiAction::PasteCanvas { label: edit.label, primitives: edit.primitives });
            }
        }
        ctx.input(|input| {
            let command = input.modifiers.command;
            if command && input.key_pressed(egui::Key::Z) {
                if let Some(parent) = model
                    .nodes
                    .iter()
                    .find(|n| n.id == model.current_node)
                    .and_then(|n| n.parent)
                {
                    actions.push(UiAction::Jump(parent));
                }
            }
            if command && input.key_pressed(egui::Key::Y) {
                if let Some(child) = model
                    .nodes
                    .iter()
                    .rev()
                    .find(|n| n.parent == Some(model.current_node))
                {
                    actions.push(UiAction::Jump(child.id));
                }
            }
            if self.page != Page::Canvas {
                return;
            }
            if command && input.key_pressed(egui::Key::D) {
                self.redraw_panel.clear_focus();
                if let Some(edit) = self.canvas.duplicate(model.primitives) {
                    push_edit(actions, edit);
                }
            }
            if command && input.key_pressed(egui::Key::A) {
                self.redraw_panel.clear_focus();
                self.canvas.selection = model
                    .primitives
                    .iter()
                    .filter(|p| p.shape != ShapeKind::Background&&!self.canvas.is_hidden(p.id))
                    .map(|p| p.id)
                    .collect();
            }
            if input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace) {
                if self.redraw_view.is_some()&&self.redraw_panel.focused{
                    actions.push(UiAction::DeleteRedrawRegions);
                }else if let Some(edit) = self.canvas.delete(model.primitives) {
                    push_edit(actions, edit);
                }
            }
            if !command {
                let plain_key = |key| input.events.iter().any(|event| matches!(event,
                    egui::Event::Key { key: event_key, pressed: true, modifiers, .. }
                        if *event_key == key && !modifiers.command && !modifiers.ctrl
                            && !modifiers.mac_cmd && !modifiers.alt));
                if plain_key(egui::Key::V) {
                    self.redraw_panel.clear_focus();
                    self.canvas.tool = Tool::Select;
                }
                if plain_key(egui::Key::R) {
                    self.redraw_panel.clear_focus();
                    self.canvas.tool = Tool::Rectangle;
                }
                if plain_key(egui::Key::E) {
                    self.redraw_panel.clear_focus();
                    self.canvas.tool = Tool::Ellipse;
                }
                if plain_key(egui::Key::F) {
                    self.canvas.fit();
                }
            }
        });
    }
}

fn push_edit(actions: &mut Vec<UiAction>, edit: CanvasEdit) {
    actions.push(UiAction::CommitCanvas {
        label: edit.label,
        primitives: edit.primitives,
    });
}

fn generation_button_caption(settings:&Settings,region:&DrawingRegion,empty:bool)->String{
    let next=if settings.pause_at_stage_end{match region.stage{
        Stage::Global if region.progress.global_complete=>Some("粗网格"),
        Stage::Coarse if region.completed_coarse==16=>Some("细网格"),_=>None,
    }}else{None};
    if let Some(stage)=next{return if settings.parallel_sequences>1{
        format!("进入{stage}，生成 {} 条分支",settings.parallel_sequences)
    }else{format!("进入{stage}并生成")};}
    if settings.parallel_sequences>1{format!("同时生成 {} 条分支",settings.parallel_sequences)}
    else if empty{"开始生成".into()}else{"从当前状态继续生成".into()}
}

fn acceleration_preparation_modal(ctx: &egui::Context, message: &str) -> Rect {
    let screen = ctx.screen_rect();
    egui::Area::new(egui::Id::new("acceleration_preparation_scrim"))
        .order(egui::Order::Foreground).fixed_pos(screen.min).movable(false).constrain(false)
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(screen.size(), egui::Sense::click_and_drag());
            ui.painter().rect_filled(rect, 0.0, Color32::from_black_alpha(90));
        });
    let id = egui::Id::new("acceleration_preparation_modal");
    let dialog = egui::Area::new(id).order(egui::Order::Foreground)
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO).movable(false)
        .show(ctx, |ui| {
            egui::Frame::window(ui.style()).inner_margin(24.0).rounding(12.0)
                .show(ui, |ui| {
                    ui.set_width(420.0);
                    ui.label(RichText::new("正在准备推理加速").size(19.0).strong().color(INK));
                    ui.add_space(18.0);
                    ui.horizontal(|ui| {
                        paint_busy_indicator(ui);
                        ui.add_space(6.0);
                        let waiting = if message.contains("ATen") { "请等待加速准备完成" } else { "请等待编译完成" };
                        ui.label(RichText::new(waiting).size(14.0).strong().color(INK));
                    });
                    ui.add_space(12.0);
                    ui.add(egui::Label::new(RichText::new(message).size(13.0).color(MUTED)).wrap(true));
                    ui.add_space(12.0);
                    ui.add(egui::Label::new(RichText::new("Graph 已启用且输入适用时，会在首次生成时按需准备。")
                        .size(11.0).color(MUTED)).wrap(true));
                });
        });
    ctx.move_to_top(egui::LayerId::new(egui::Order::Foreground, id));
    dialog.response.rect
}

fn native_color(color: [u8; 3]) -> [u8; 3] {
    color.map(|channel| channel / 2 * 2)
}

fn split_canvas_pair(area: Rect) -> (Rect, Rect) {
    let gap = 10.0_f32.min(area.width() * 0.03);
    let left_width = (area.width() - gap) * (2.0 / 3.0);
    let left = Rect::from_min_max(area.min, Pos2::new(area.left() + left_width, area.bottom()));
    let right = Rect::from_min_max(Pos2::new(left.right() + gap, area.top()), area.max);
    (left, right)
}

fn acceleration_feature_picker(ui: &mut egui::Ui, id: &str, value: &mut FeaturePreference, enabled: bool) -> bool {
    let mut changed = false;
    let kv_cache = id == "acceleration_kv_int8";
    let response = ui.add_enabled_ui(enabled, |ui| {
        egui::ComboBox::from_id_source(id)
            .selected_text(if kv_cache && !enabled { "浮点（兼容）" } else { feature_preference_label(*value) })
            .width(110.0)
            .show_ui(ui, |ui| {
                for preference in [FeaturePreference::Auto, FeaturePreference::Disabled, FeaturePreference::Enabled] {
                    changed |= ui.selectable_value(value, preference, feature_preference_label(preference)).changed();
                }
            }).response
    }).inner;
    response.on_disabled_hover_text(if kv_cache {
        "普通兼容模式使用原浮点缓存；切换模式后仍保留KV INT8设置。"
    } else {
        "普通兼容模式不使用可选加速；切换模式后仍保留这些设置。"
    });
    changed
}

fn feature_preference_label(value: FeaturePreference) -> &'static str {
    match value { FeaturePreference::Auto => "自动", FeaturePreference::Disabled => "关闭", FeaturePreference::Enabled => "开启" }
}

#[cfg(test)]
mod kv_int8_picker_tests {
    use super::*;

    fn text_in_shapes(shape: &egui::epaint::Shape) -> String {
        match shape {
            egui::epaint::Shape::Text(text) => text.galley.job.text.clone(),
            egui::epaint::Shape::Vec(shapes) => shapes.iter().map(text_in_shapes).collect::<Vec<_>>().join(" "),
            _ => String::new(),
        }
    }

    #[test]
    fn compatible_int8_picker_shows_float_and_preserves_each_saved_preference() {
        for saved in [FeaturePreference::Auto, FeaturePreference::Disabled, FeaturePreference::Enabled] {
            let ctx = egui::Context::default();
            let mut value = saved;
            for events in [Vec::new(), vec![egui::Event::PointerMoved(Pos2::new(35.0, 18.0)),
                egui::Event::PointerButton { pos:Pos2::new(35.0,18.0),button:egui::PointerButton::Primary,pressed:true,modifiers:Default::default() }],
                vec![egui::Event::PointerButton { pos:Pos2::new(35.0,18.0),button:egui::PointerButton::Primary,pressed:false,modifiers:Default::default() }]] {
                let output = ctx.run(egui::RawInput { events, ..Default::default() }, |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        assert!(!acceleration_feature_picker(ui, "acceleration_kv_int8", &mut value, false));
                    });
                });
                let text = output.shapes.iter().map(|s|text_in_shapes(&s.shape)).collect::<Vec<_>>().join(" ");
                assert!(text.contains("浮点（兼容）"), "effective float mode must remain visible: {text}");
                assert_eq!(value, saved, "compatible mode must not erase the user's saved INT8 preference");
            }
        }
    }
}

fn paint_busy_indicator(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(12.0), egui::Sense::hover());
    let phase = ui.input(|input| input.time) as f32 * 4.0;
    let points = (0..=20).map(|i| {
        let angle = phase + i as f32 / 20.0 * std::f32::consts::TAU * 0.72;
        rect.center() + Vec2::new(angle.cos(), angle.sin()) * 4.5
    }).collect();
    ui.painter().add(egui::Shape::line(points, Stroke::new(1.6, BLUE)));
    // The application's active-task timer supplies frames. egui::Spinner
    // requests zero-delay repaint on every frame and defeats that timer.
}

fn section_title(ui: &mut egui::Ui, title: &str, subtitle: Option<&str>) {
    ui.label(RichText::new(title).size(15.0).strong().color(INK));
    if let Some(subtitle) = subtitle {
        ui.add_space(4.0);
        ui.label(RichText::new(subtitle).size(11.0).color(MUTED));
    }
}

fn tab(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).size(14.0).color(if selected {
            BLUE
        } else {
            MUTED
        }))
        .fill(if selected {
            Color32::from_rgb(235, 241, 255)
        } else {
            Color32::TRANSPARENT
        })
        .stroke(Stroke::NONE)
        .rounding(7.0)
        .min_size(Vec2::new(78.0, 34.0)),
    )
}

fn tool_button(ui: &mut egui::Ui, label: &str, key: &str, selected: bool) -> egui::Response {
    ui.add(
        egui::Button::new(
            RichText::new(format!("{label}\n{key}"))
                .size(12.0)
                .color(if selected { BLUE } else { MUTED }),
        )
        .fill(if selected {
            Color32::from_rgb(228, 236, 255)
        } else {
            Color32::TRANSPARENT
        })
        .stroke(Stroke::NONE)
        .rounding(7.0)
        .min_size(Vec2::new(48.0, 48.0)),
    )
}

fn elide(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_owned()
    } else {
        text.chars()
            .take(max_chars - 1)
            .chain(std::iter::once('…'))
            .collect()
    }
}

#[derive(Clone, Copy)]
enum LocalFileKind {
    Model,
    Tokenizer,
}

fn selected_file_text(
    selected: Option<&Path>,
    kind: LocalFileKind,
) -> Result<Option<String>, String> {
    let Some(path) = selected else {
        return Ok(None);
    };
    let text = path
        .to_str()
        .ok_or("所选路径无法表示为 Unicode，请选择其他路径")?;
    if text.is_empty() || text.contains('\0') {
        return Err("所选文件路径无效".into());
    }
    let expected = match kind {
        LocalFileKind::Model => "pt",
        LocalFileKind::Tokenizer => "model",
    };
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("");
    if !extension.eq_ignore_ascii_case(expected) {
        return Err(match kind {
            LocalFileKind::Model => "请选择 .pt 模型权重文件",
            LocalFileKind::Tokenizer => "请选择 .model 分词器文件",
        }
        .into());
    }
    Ok(Some(text.to_owned()))
}

#[cfg(target_os = "windows")]
fn choose_local_file(initial_path: &str, kind: LocalFileKind) -> Result<Option<PathBuf>, String> {
    use std::{
        ffi::{c_void, OsString},
        os::windows::ffi::{OsStrExt, OsStringExt},
        ptr,
    };
    #[repr(C)]
    struct OpenFileNameW {
        size: u32,
        owner: *mut c_void,
        instance: *mut c_void,
        filter: *const u16,
        custom_filter: *mut u16,
        max_custom_filter: u32,
        filter_index: u32,
        file: *mut u16,
        max_file: u32,
        file_title: *mut u16,
        max_file_title: u32,
        initial_dir: *const u16,
        title: *const u16,
        flags: u32,
        file_offset: u16,
        file_extension: u16,
        default_extension: *const u16,
        custom_data: isize,
        hook: Option<unsafe extern "system" fn(*mut c_void, u32, usize, isize) -> usize>,
        template_name: *const u16,
        reserved: *mut c_void,
        reserved_word: u32,
        flags_ex: u32,
    }
    #[link(name = "comdlg32")]
    extern "system" {
        fn GetOpenFileNameW(data: *mut OpenFileNameW) -> i32;
        fn CommDlgExtendedError() -> u32;
    }
    #[link(name = "user32")]
    extern "system" {
        fn GetForegroundWindow() -> *mut c_void;
    }
    let (title, filter, extension) = match kind {
        LocalFileKind::Model => (
            "选择已下载的模型权重",
            "PyTorch 模型权重 (*.pt)\0*.pt\0\0",
            "pt",
        ),
        LocalFileKind::Tokenizer => (
            "选择分词器文件",
            "SentencePiece 分词器 (*.model)\0*.model\0\0",
            "model",
        ),
    };
    let wide = |text: &str| {
        text.encode_utf16()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>()
    };
    let title = wide(title);
    let filter: Vec<u16> = filter.encode_utf16().collect();
    let extension = wide(extension);
    let initial = Path::new(initial_path);
    let directory = initial
        .parent()
        .filter(|path| path.is_dir())
        .map(Path::to_path_buf)
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .map(|profile| PathBuf::from(profile).join("Downloads"))
                .filter(|path| path.is_dir())
        });
    let directory = directory.map(|path| {
        path.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>()
    });
    let mut file = vec![0u16; 32_768];
    if let Some(name) = initial.file_name() {
        for (slot, value) in file.iter_mut().take(32_767).zip(name.encode_wide()) {
            *slot = value;
        }
    }
    let mut data = OpenFileNameW {
        size: std::mem::size_of::<OpenFileNameW>() as u32,
        owner: unsafe { GetForegroundWindow() },
        instance: ptr::null_mut(),
        filter: filter.as_ptr(),
        custom_filter: ptr::null_mut(),
        max_custom_filter: 0,
        filter_index: 1,
        file: file.as_mut_ptr(),
        max_file: file.len() as u32,
        file_title: ptr::null_mut(),
        max_file_title: 0,
        initial_dir: directory
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null()),
        title: title.as_ptr(),
        // Explorer, must-exist file/path, no process working-directory change,
        // and no automatic addition of the model path to Windows Recent Files.
        flags: 0x0008_0000 | 0x0000_1000 | 0x0000_0800 | 0x0000_0008 | 0x0200_0000,
        file_offset: 0,
        file_extension: 0,
        default_extension: extension.as_ptr(),
        custom_data: 0,
        hook: None,
        template_name: ptr::null(),
        reserved: ptr::null_mut(),
        reserved_word: 0,
        flags_ex: 0,
    };
    if unsafe { GetOpenFileNameW(&mut data) } != 0 {
        let end = file
            .iter()
            .position(|&value| value == 0)
            .ok_or("Windows 返回了不完整的文件路径")?;
        Ok(Some(PathBuf::from(OsString::from_wide(&file[..end]))))
    } else {
        let error = unsafe { CommDlgExtendedError() };
        if error == 0 {
            Ok(None)
        } else {
            Err(format!(
                "无法打开文件选择窗口（Windows 错误 0x{error:04X}）。可在高级路径设置中选择文件。"
            ))
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn choose_local_file(_initial_path: &str, _kind: LocalFileKind) -> Result<Option<PathBuf>, String> {
    Err("本版本的文件选择器仅支持 Windows".into())
}

fn configure_style(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    for (name, path) in [
        ("windows-ui", "C:/Windows/Fonts/segoeui.ttf"),
        ("chinese-ui", "C:/Windows/Fonts/msyh.ttc"),
    ] {
        if let Ok(data) = std::fs::read(path) {
            fonts
                .font_data
                .insert(name.to_owned(), FontData::from_owned(data));
            fonts
                .families
                .entry(FontFamily::Proportional)
                .or_default()
                .push(name.to_owned());
            fonts
                .families
                .entry(FontFamily::Monospace)
                .or_default()
                .push(name.to_owned());
        }
    }
    ctx.set_fonts(fonts);
    let mut visuals = egui::Visuals::light();
    visuals.override_text_color = Some(INK);
    visuals.panel_fill = Color32::WHITE;
    visuals.window_fill = Color32::WHITE;
    visuals.extreme_bg_color = Color32::from_rgb(247, 249, 252);
    visuals.faint_bg_color = Color32::from_rgb(245, 247, 251);
    visuals.selection.bg_fill = Color32::from_rgb(220, 231, 255);
    visuals.selection.stroke = Stroke::new(1.0, BLUE);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(245, 247, 251);
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, LINE);
    visuals.widgets.inactive.rounding = egui::Rounding::same(5.0);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(233, 239, 250);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(181, 201, 243));
    visuals.widgets.active.bg_fill = Color32::from_rgb(223, 234, 255);
    ctx.set_visuals(visuals);
    let mut style = (*ctx.style()).clone();
    style
        .text_styles
        .insert(egui::TextStyle::Body, FontId::proportional(13.0));
    style
        .text_styles
        .insert(egui::TextStyle::Button, FontId::proportional(13.0));
    style
        .text_styles
        .insert(egui::TextStyle::Small, FontId::proportional(11.0));
    style.spacing.item_spacing = Vec2::new(8.0, 7.0);
    style.spacing.button_padding = Vec2::new(10.0, 6.0);
    ctx.set_style(style);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dirty_state_only_marks_header_and_status_without_implicit_save_while_ctrl_s_still_requests_save(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let doc=Document::new();let region=DrawingRegion::default();
        editor.set_save_name(Some("测试存档".into()));editor.set_unsaved(true);
        fn text(shape:&egui::epaint::Shape)->String{match shape{egui::epaint::Shape::Text(value)=>value.galley.job.text.clone(),egui::epaint::Shape::Vec(values)=>values.iter().map(text).collect::<Vec<_>>().join(" "),_=>String::new()}}
        for _ in 0..4{let(actions,output)=render_frame_output(&ctx,&mut editor,&doc,&region,false,false,Vec2::new(1440.,920.),vec![]);
            assert!(actions.is_empty());assert!(!editor.unsaved_prompt.is_open());
            let labels=output.shapes.iter().map(|shape|text(&shape.shape)).collect::<Vec<_>>().join(" ");
            assert!(labels.contains("测试存档 *")&&labels.contains("未保存 · Ctrl+S"));}
        editor.set_project_path(Some(PathBuf::from("C:/test/manual-save.pproj")));
        let actions=render_frame(&ctx,&mut editor,&doc,&region,vec![egui::Event::Key{key:egui::Key::S,physical_key:None,pressed:true,repeat:false,
            modifiers:egui::Modifiers{ctrl:true,command:true,..Default::default()}}]);
        assert!(matches!(actions.as_slice(),[UiAction::SaveProject(path)]if path==&PathBuf::from("C:/test/manual-save.pproj")));
        assert_eq!(doc.nodes.len(),1);assert!(doc.primitives.is_empty());
        editor.set_unsaved(false);let(_,output)=render_frame_output(&ctx,&mut editor,&doc,&region,false,false,Vec2::new(1440.,920.),vec![]);
        assert!(!output.shapes.iter().any(|shape|text(&shape.shape).contains("未保存 · Ctrl+S")));
    }
    #[test]
    fn explicit_unsaved_leave_prompt_overrides_mode_report_and_returns_choices_without_touching_document(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let mut doc=Document::new();
        doc.commit_generated(Primitive::new(ShapeKind::Rectangle,100.,100.,20.,20.,[80,120,160])).unwrap();
        let canvas=doc.primitives.clone();let nodes=doc.nodes.len();editor.canvas.selection=vec![canvas[0].id];
        editor.set_unsaved(true);editor.show_mode_issues(vec!["请修正区域配额".into()]);let region=DrawingRegion::default();
        editor.show_unsaved_changes("返回存档页");editor.set_file_busy(true);
        assert!(render_frame(&ctx,&mut editor,&doc,&region,vec![]).is_empty());assert!(editor.unsaved_prompt.is_open());
        editor.set_file_busy(false);editor.clear_unsaved_changes();
        for choice in [UnsavedChoice::Save,UnsavedChoice::Discard,UnsavedChoice::Cancel]{
            editor.show_unsaved_changes("返回存档页");for _ in 0..3{assert!(render_frame(&ctx,&mut editor,&doc,&region,vec![]).is_empty());}
            let point=match choice{UnsavedChoice::Save=>editor.unsaved_prompt.save_rect.center(),UnsavedChoice::Discard=>editor.unsaved_prompt.discard_rect.center(),UnsavedChoice::Cancel=>editor.unsaved_prompt.cancel_rect.center()};
            let actions=click_at(&ctx,&mut editor,&doc,&region,point);
            assert!(matches!(actions.as_slice(),[UiAction::ResolveUnsaved{choice:actual}]if *actual==choice));
            assert!(!editor.unsaved_prompt.is_open());assert!(editor.unsaved);assert!(editor.mode_issues.is_some());
            assert_eq!(doc.primitives,canvas);assert_eq!(doc.nodes.len(),nodes);assert_eq!(editor.canvas.selection,vec![canvas[0].id]);
        }
        editor.show_unsaved_changes("关闭程序");editor.clear_unsaved_changes();assert!(!editor.unsaved_prompt.is_open());assert!(editor.unsaved);
    }
    #[test]
    fn workspace_mode_buttons_request_switches_without_optimistically_changing_the_real_mode(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let doc=Document::new();let region=DrawingRegion::default();
        editor.set_workflow_mode(false);render_frame(&ctx,&mut editor,&doc,&region,vec![]);
        let active=editor.workflow_mode_rects[0].center();assert!(click_at(&ctx,&mut editor,&doc,&region,active).is_empty());
        let target=editor.workflow_mode_rects[1].center();let actions=click_at(&ctx,&mut editor,&doc,&region,target);
        assert!(matches!(actions.as_slice(),[UiAction::SwitchWorkspaceMode{redraw:true}]));assert!(!editor.workflow_redraw);
        editor.set_workflow_mode(true);render_frame(&ctx,&mut editor,&doc,&region,vec![]);
        let target=editor.workflow_mode_rects[0].center();let actions=click_at(&ctx,&mut editor,&doc,&region,target);
        assert!(matches!(actions.as_slice(),[UiAction::SwitchWorkspaceMode{redraw:false}]));assert!(editor.workflow_redraw);
        assert_eq!(doc.nodes.len(),1);assert!(doc.primitives.is_empty());
    }
    #[test]
    fn workspace_mode_buttons_are_disabled_during_generation_loading_preparation_and_stopping(){
        for(running,loading,compiling,stopping)in [(true,false,false,false),(false,true,false,false),(false,false,true,false),(false,false,false,true)]{
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let doc=Document::new();let region=DrawingRegion::default();
            let nodes=vec![NodeView{id:0,parent:None,title:"初始".into(),source:"初始".into(),primitive_count:0}];
            let model=UiModel{document:&doc,drawing_region:&region,primitives:&doc.primitives,stage_by_id:None,prompt:&doc.prompt,settings:&doc.settings,
                current_node:0,nodes:&nodes,running,loading,compiling,stopping,model_loaded:false,model_status:"就绪",model_path:"",tokenizer_path:"",error:None,status:"就绪"};
            let render=|editor:&mut EditorUi,events|{let mut actions=Vec::new();let _=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(1440.,920.))),events,..Default::default()},|ctx|actions=editor.render(ctx,&model));actions};
            render(&mut editor,vec![]);let target=editor.workflow_mode_rects[1].center();
            for pressed in [true,false]{let actions=render(&mut editor,vec![egui::Event::PointerMoved(target),pointer_event(target,pressed)]);
                assert!(actions.iter().all(|action|!matches!(action,UiAction::SwitchWorkspaceMode{..}|UiAction::Stop|UiAction::Generate{..})));}
            assert!(!editor.workflow_redraw);
        }
    }
    #[test]
    fn mode_issue_dialog_retains_every_issue_and_blocks_edits_until_closed_without_changing_canvas_or_mode(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let mut doc=Document::new();
        doc.commit_generated(Primitive::new(ShapeKind::Rectangle,100.,100.,20.,20.,[80,120,160])).unwrap();
        let old_canvas=doc.primitives.clone();let old_nodes=doc.nodes.len();let id=doc.primitives[0].id;
        editor.canvas.selection=vec![id];editor.set_workflow_mode(true);
        let issues=(0..80).map(|index|format!("问题{index}：请调整对应区域中的图元后重试。")).collect::<Vec<_>>();
        editor.show_mode_issues(issues.clone());let region=DrawingRegion::default();
        for _ in 0..3{assert!(render_frame(&ctx,&mut editor,&doc,&region,vec![]).is_empty());assert_eq!(editor.mode_issues.as_ref(),Some(&issues));}
        let actions=render_frame(&ctx,&mut editor,&doc,&region,vec![egui::Event::Key{key:egui::Key::Delete,physical_key:None,pressed:true,repeat:false,modifiers:Default::default()}]);
        assert!(actions.is_empty());assert!(editor.workflow_redraw);assert_eq!(editor.canvas.selection,vec![id]);
        assert_eq!(doc.primitives,old_canvas);assert_eq!(doc.nodes.len(),old_nodes);
        let close=editor.mode_issues_close_rect.center();assert!(editor.mode_issues_close_rect.is_positive());
        assert!(click_at(&ctx,&mut editor,&doc,&region,close).is_empty());assert!(editor.mode_issues.is_none());assert!(editor.workflow_redraw);
        editor.show_mode_issues(vec!["仍需修正".into()]);render_frame(&ctx,&mut editor,&doc,&region,vec![]);
        assert!(render_frame(&ctx,&mut editor,&doc,&region,vec![egui::Event::Key{key:egui::Key::Escape,physical_key:None,pressed:true,repeat:false,modifiers:Default::default()}]).is_empty());
        assert!(editor.mode_issues.is_none());assert_eq!(doc.primitives,old_canvas);assert_eq!(doc.nodes.len(),old_nodes);
    }
    #[test]
    fn canvas_and_history_navigation_occupy_the_second_header_row(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let doc=Document::new();
        render_frame(&ctx,&mut editor,&doc,&DrawingRegion::default(),vec![]);
        assert!(editor.navigation_rects[0].top()>editor.navigation_rects[2].bottom());
        assert!((editor.navigation_rects[0].top()-editor.navigation_rects[1].top()).abs()<2.);
    }

    #[test]
    fn redraw_region_delete_does_not_delete_the_old_canvas_selection_and_canvas_click_reclaims_focus(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let mut doc=Document::new();
        doc.commit_generated(Primitive::new(ShapeKind::Rectangle,256.,256.,60.,60.,[80,120,160])).unwrap();
        let id=doc.primitives[0].id;editor.canvas.selection=vec![id];
        editor.set_redraw_model(Some(RedrawUiModel{selected_coarse:vec![0],..Default::default()}));
        let region=DrawingRegion::default();render_frame(&ctx,&mut editor,&doc,&region,vec![]);
        let board=editor.redraw_panel.board_rect(Stage::Coarse);assert!(board.is_positive());
        let point=board.min+Vec2::splat(board.width()/8.);
        let actions=click_at(&ctx,&mut editor,&doc,&region,point);
        assert!(actions.iter().any(|action|matches!(action,UiAction::ToggleRedrawRegion{stage:Stage::Coarse,index:0})));
        assert!(editor.redraw_panel.focused);
        let key=|pressed|egui::Event::Key{key:egui::Key::Delete,physical_key:None,pressed,repeat:false,modifiers:Default::default()};
        let actions=render_frame(&ctx,&mut editor,&doc,&region,vec![key(true)]);
        assert!(actions.iter().any(|action|matches!(action,UiAction::DeleteRedrawRegions)));
        assert!(!actions.iter().any(|action|matches!(action,UiAction::CommitCanvas{..})));
        assert_eq!(editor.canvas.selection,vec![id]);
        render_frame(&ctx,&mut editor,&doc,&region,vec![key(false)]);
        let point=editor.edit_canvas_rect.center();click_at(&ctx,&mut editor,&doc,&region,point);
        assert!(!editor.redraw_panel.focused);editor.canvas.selection=vec![id];
        let actions=render_frame(&ctx,&mut editor,&doc,&region,vec![key(true)]);
        assert!(actions.iter().any(|action|matches!(action,UiAction::CommitCanvas{..})));
        assert!(!actions.iter().any(|action|matches!(action,UiAction::DeleteRedrawRegions)));
    }

    #[test]
    fn redraw_generation_explains_nonempty_selection_without_generating_or_deleting_and_ignores_normal_completion(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=Document::new();let region=DrawingRegion::complete();
        let nodes=vec![NodeView{id:0,parent:None,title:"初始".into(),source:"初始".into(),primitive_count:0}];
        let model=UiModel{document:&document,drawing_region:&region,primitives:&document.primitives,stage_by_id:None,
            prompt:&document.prompt,settings:&document.settings,current_node:0,nodes:&nodes,running:false,loading:false,compiling:false,stopping:false,
            model_loaded:true,model_status:"模型已加载",model_path:"",tokenizer_path:concat!(env!("CARGO_MANIFEST_DIR"),"/assets/caption_spm_1500.model"),error:None,status:"就绪"};
        let render=|editor:&mut EditorUi,events|{let mut actions=Vec::new();let output=ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,Vec2::new(1440.,920.))),events,..Default::default()},|ctx|actions=editor.render(ctx,&model));(actions,output)};
        let reason="请先手动删除所选区域中的图元";
        editor.set_redraw_model(Some(RedrawUiModel{selected_fine:vec![0],can_generate:false,reason:Some(reason.into()),..Default::default()}));
        render(&mut editor,vec![]);let point=editor.generation_button_rect.center();assert!(editor.generation_button_rect.is_positive());
        let mut actions=Vec::new();for pressed in [true,false]{actions.extend(render(&mut editor,vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)]).0);}
        assert!(actions.iter().all(|action|!matches!(action,UiAction::Generate{..}|UiAction::DeleteRedrawRegions|UiAction::CommitCanvas{..})));
        fn text(shape:&egui::epaint::Shape)->String{match shape{egui::epaint::Shape::Text(value)=>value.galley.job.text.clone(),egui::epaint::Shape::Vec(values)=>values.iter().map(text).collect::<Vec<_>>().join(" "),_=>String::new()}}
        let output=render(&mut editor,vec![]).1;assert!(output.shapes.iter().any(|shape|text(&shape.shape).contains(reason)));
        editor.set_redraw_model(Some(RedrawUiModel{selected_fine:vec![0],can_generate:true,..Default::default()}));
        render(&mut editor,vec![]);let point=editor.generation_button_rect.center();let mut actions=Vec::new();
        for pressed in [true,false]{actions.extend(render(&mut editor,vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)]).0);}
        assert!(matches!(actions.as_slice(),[UiAction::Generate{..}]));
    }
    #[test]
    fn generation_caption_names_the_next_guarded_stage_without_changing_other_cases(){
        let mut settings=Settings{pause_at_stage_end:true,parallel_sequences:1,..Default::default()};
        let mut region=DrawingRegion::default();assert_eq!(generation_button_caption(&settings,&region,true),"开始生成");
        region.progress.global_complete=true;assert_eq!(generation_button_caption(&settings,&region,false),"进入粗网格并生成");
        settings.parallel_sequences=8;assert_eq!(generation_button_caption(&settings,&region,false),"进入粗网格，生成 8 条分支");
        region.stage=Stage::Coarse;region.completed_coarse=16;assert_eq!(generation_button_caption(&settings,&region,false),"进入细网格，生成 8 条分支");
        settings.parallel_sequences=1;assert_eq!(generation_button_caption(&settings,&region,false),"进入细网格并生成");
        region.completed_coarse=15;assert_eq!(generation_button_caption(&settings,&region,false),"从当前状态继续生成");
        settings.pause_at_stage_end=false;region.completed_coarse=16;assert_eq!(generation_button_caption(&settings,&region,false),"从当前状态继续生成");
    }
    use crate::document::OperationSource;
    use crate::region_state::Stage;

    #[test]
    fn selected_unicode_model_path_loads_with_bundled_tokenizer_without_generation() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        editor.model_path = "C:/old/model.pt".into();
        editor.tokenizer_path =
            "C:/Program Files/Primitive Painter/assets/caption_spm_1500.model".into();
        let selected = PathBuf::from("C:/Users/测试用户/Downloads/最新 权重/model.pt");
        let action = editor
            .apply_model_selection(Some(selected.clone()))
            .unwrap()
            .unwrap();
        match action {
            UiAction::LoadModel {
                checkpoint_path,
                tokenizer_path,
            } => {
                assert_eq!(checkpoint_path, selected.to_str().unwrap());
                assert_eq!(tokenizer_path, editor.tokenizer_path);
                assert_eq!(editor.model_path, checkpoint_path);
            }
            _ => panic!(
                "Choosing a weight must load it, never start image generation or create history"
            ),
        }
    }

    #[test]
    fn cancelled_or_unsupported_file_selection_preserves_existing_model_path() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        editor.model_path = "C:/existing/model.pt".into();
        let original = editor.model_path.clone();
        assert!(editor.apply_model_selection(None).unwrap().is_none());
        assert_eq!(editor.model_path, original);
        assert!(editor
            .apply_model_selection(Some(PathBuf::from("C:/downloads/unsupported.safetensors")))
            .is_err());
        assert_eq!(editor.model_path, original);
        assert!(selected_file_text(
            Some(Path::new("C:/模型/分词器 空格.MODEL")),
            LocalFileKind::Tokenizer
        )
        .unwrap()
        .is_some());
        assert!(
            selected_file_text(Some(Path::new("C:/模型/正确权重.PT")), LocalFileKind::Model)
                .unwrap()
                .is_some()
        );
        assert!(selected_file_text(
            Some(Path::new("C:/bad\0path/model.pt")),
            LocalFileKind::Model
        )
        .is_err());
    }

    fn render_frame(
        ctx: &egui::Context,
        editor: &mut EditorUi,
        document: &Document,
        region: &DrawingRegion,
        events: Vec<egui::Event>,
    ) -> Vec<UiAction> {
        render_frame_flags(ctx, editor, document, region, false, false, events)
    }

    fn render_frame_flags(
        ctx: &egui::Context,
        editor: &mut EditorUi,
        document: &Document,
        region: &DrawingRegion,
        running: bool,
        loading: bool,
        events: Vec<egui::Event>,
    ) -> Vec<UiAction> {
        render_frame_sized(ctx, editor, document, region, running, loading,
            Vec2::new(1440.0, 920.0), events)
    }

    fn render_frame_sized(
        ctx: &egui::Context,
        editor: &mut EditorUi,
        document: &Document,
        region: &DrawingRegion,
        running: bool,
        loading: bool,
        size: Vec2,
        events: Vec<egui::Event>,
    ) -> Vec<UiAction> {
        render_frame_output(ctx, editor, document, region, running, loading, size, events).0
    }

    fn render_frame_output(
        ctx: &egui::Context,
        editor: &mut EditorUi,
        document: &Document,
        region: &DrawingRegion,
        running: bool,
        loading: bool,
        size: Vec2,
        events: Vec<egui::Event>,
    ) -> (Vec<UiAction>, egui::FullOutput) {
        render_frame_state_output(ctx, editor, document, region, running, loading, false, size, events)
    }

    fn render_frame_state_output(
        ctx: &egui::Context,
        editor: &mut EditorUi,
        document: &Document,
        region: &DrawingRegion,
        running: bool,
        loading: bool,
        stopping: bool,
        size: Vec2,
        events: Vec<egui::Event>,
    ) -> (Vec<UiAction>, egui::FullOutput) {
        render_frame_state_output_stages(ctx,editor,document,region,running,loading,stopping,size,events,None)
    }

    fn render_frame_state_output_stages(
        ctx:&egui::Context,editor:&mut EditorUi,document:&Document,region:&DrawingRegion,
        running:bool,loading:bool,stopping:bool,size:Vec2,events:Vec<egui::Event>,stage_by_id:Option<&BTreeMap<u64,Stage>>,
    )->(Vec<UiAction>,egui::FullOutput){
        let nodes: Vec<NodeView> = document
            .nodes
            .iter()
            .map(|node| NodeView {
                id: node.id as u64,
                parent: node.parent.map(|id| id as u64),
                title: node.title.clone(),
                source: "用户".into(),
                primitive_count: node.primitive_count,
            })
            .collect();
        let model = UiModel {
            document,
            drawing_region: region,
            primitives: &document.primitives,
            stage_by_id,
            prompt: &document.prompt,
            settings: &document.settings,
            current_node: document.current as u64,
            nodes: &nodes,
            running,
            loading,
            compiling: false,
            stopping,
            model_loaded: false,
            model_status: "模型未加载",
            model_path: "",
            tokenizer_path: "",
            error: None,
            status: "就绪",
        };
        let mut actions = Vec::new();
        let modifiers = events.iter().rev().find_map(|event| match event {
            egui::Event::Key { modifiers, .. } => Some(*modifiers), _ => None,
        }).unwrap_or_default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)),
            modifiers,
            events,
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            actions = editor.render(ctx, &model);
        });
        (actions, output)
    }

    fn click_at(
        ctx: &egui::Context,
        editor: &mut EditorUi,
        document: &Document,
        region: &DrawingRegion,
        point: Pos2,
    ) -> Vec<UiAction> {
        let mut actions = Vec::new();
        for pressed in [true, false] {
            actions.extend(render_frame(ctx, editor, document, region, vec![
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton { pos: point, button: egui::PointerButton::Primary,
                    pressed, modifiers: Default::default() },
            ]));
        }
        actions
    }

    #[test]
    fn raster_display_toggle_is_visual_only_and_does_not_pause_or_freeze_generation(){
        for running in [false,true]{
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);
            let document=color_document();let region=DrawingRegion::default();
            let before=serde_json::to_vec(&document.project_ref()).unwrap();
            for _ in 0..3{assert!(render_frame_flags(&ctx,&mut editor,&document,&region,running,false,vec![]).is_empty());}
            assert!(!editor.canvas.rasterize_canvas);
            let point=editor.rasterize_control_rect.center();
            assert!(editor.tools_panel_rect.contains(point));
            let mut uploaded=false;
            for pressed in [true,false]{
                let (actions,output)=render_frame_output(&ctx,&mut editor,&document,&region,running,false,Vec2::new(1440.,920.),
                    vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)]);
                assert!(actions.is_empty(),"a display toggle must not edit, save, stop or freeze generation: {actions:?}");
                for (_,delta) in output.textures_delta.set{
                    if delta.image.size()==[512,512]{assert_eq!(delta.options,egui::TextureOptions::NEAREST);uploaded=true;}
                }
            }
            assert!(editor.canvas.rasterize_canvas);assert!(uploaded);
            let (actions,output)=render_frame_output(&ctx,&mut editor,&document,&region,running,false,Vec2::new(1440.,920.),vec![]);
            assert!(actions.is_empty());assert!(!output.textures_delta.set.iter().any(|(_,d)|d.image.size()==[512,512]),"idle frames reuse the raster texture");
            for pressed in [true,false]{assert!(render_frame_flags(&ctx,&mut editor,&document,&region,running,false,
                vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)]).is_empty());}
            assert!(!editor.canvas.rasterize_canvas);
            assert_eq!(serde_json::to_vec(&document.project_ref()).unwrap(),before);
        }
    }

    #[test]
    fn modification_stage_visibility_filters_only_display_and_selection_and_standard_mode_restores_all(){
        use crate::document::{PrimitiveStageAssignment,StageTier};
        for raster in [false,true]{for running in [false,true]{
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let region=DrawingRegion::default();
            let mut document=Document::new();let mut rows=vec![Primitive::background([220,230,240])];
            for(stage,x)in [(StageTier::Global,80.),(StageTier::Coarse,200.),(StageTier::Fine,350.)]{
                let mut p=Primitive::new(ShapeKind::Rectangle,x,180.,24.,20.,[60,100,180]);
                p.stage_assignment=Some(PrimitiveStageAssignment{stage,cell:0,queue_order:1});rows.push(p);
            }
            document.commit_canvas("三个独立阶段",OperationSource::User,rows).unwrap();
            let stages=crate::redraw::stage_canvases(&document.primitives,&BTreeMap::new()).unwrap();
            let redraw=RedrawUiModel{global:stages.global,coarse:stages.coarse,fine:stages.fine,
                coarse_counts:stages.coarse_counts,fine_counts:stages.fine_counts,selected_coarse:vec![0],..Default::default()};
            let before=serde_json::to_vec(&document.project_ref()).unwrap();
            let coarse=document.primitives[2].id;let fine=document.primitives[3].id;
            editor.canvas.rasterize_canvas=raster;editor.set_redraw_model(Some(redraw.clone()));
            for _ in 0..3{assert!(render_frame_flags(&ctx,&mut editor,&document,&region,running,false,vec![]).is_empty());}
            editor.canvas.selection=vec![coarse,fine];
            let point=editor.redraw_panel.visibility_rect(Stage::Coarse).center();
            for pressed in [true,false]{assert!(render_frame_flags(&ctx,&mut editor,&document,&region,running,false,
                vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)]).is_empty(),"display toggles must not issue model/edit actions");}
            assert!(render_frame_flags(&ctx,&mut editor,&document,&region,running,false,vec![]).is_empty());
            assert!(editor.canvas.is_hidden(coarse));assert!(!editor.canvas.is_hidden(fine));
            assert!(!editor.canvas.is_hidden(document.primitives[0].id));assert!(!editor.canvas.is_hidden(document.primitives[1].id));
            assert_eq!(editor.canvas.selection,vec![fine]);
            assert_eq!(editor.redraw_view.as_ref().unwrap().selected_coarse,vec![0]);
            editor.set_redraw_model(None);render_frame_flags(&ctx,&mut editor,&document,&region,running,false,vec![]);
            assert!(!editor.canvas.is_hidden(coarse),"standard mode ignores modification visibility");
            editor.set_redraw_model(Some(redraw));render_frame_flags(&ctx,&mut editor,&document,&region,running,false,vec![]);
            assert!(editor.canvas.is_hidden(coarse),"switching views preserves visibility preferences");
            assert_eq!(serde_json::to_vec(&document.project_ref()).unwrap(),before);
        }}
    }

    #[test]
    fn reference_entry_only_opens_on_an_idle_blank_canvas_without_edit_or_stop_actions(){
        for(nonempty,running,loading)in[(false,false,false),(true,false,false),(false,true,false),(false,false,true)]{
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);
            let document=if nonempty{color_document()}else{Document::new()};let region=DrawingRegion::default();
            let before=serde_json::to_vec(&document.project_ref()).unwrap();
            for _ in 0..3{assert!(render_frame_flags(&ctx,&mut editor,&document,&region,running,loading,vec![]).is_empty());}
            let point=editor.reference_entry_rect.center();let mut actions=Vec::new();
            for pressed in [true,false]{actions.extend(render_frame_flags(&ctx,&mut editor,&document,&region,running,loading,
                vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)]));}
            if !nonempty&&!running&&!loading{assert!(matches!(actions.as_slice(),[UiAction::OpenReference]));}
            else{assert!(actions.is_empty(),"disabled reference entry must not stop or freeze generation");}
            assert_eq!(serde_json::to_vec(&document.project_ref()).unwrap(),before);
        }
    }

    fn color_document() -> Document {
        let mut document = Document::new();
        document.commit_canvas("彩色画布", OperationSource::User, vec![
            Primitive::background([230, 240, 250]),
            Primitive::new(ShapeKind::Rectangle, 150.0, 170.0, 80.0, 60.0, [40, 100, 180]),
            Primitive::new(ShapeKind::Ellipse, 300.0, 250.0, 80.0, 60.0, [180, 80, 40]),
        ]).unwrap();
        document
    }

    fn pointer_event(point: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton { pos: point, button: egui::PointerButton::Primary,
            pressed, modifiers: Default::default() }
    }

    fn history_root_label(shapes:&[egui::epaint::ClippedShape])->Pos2 {
        fn find(shape:&egui::epaint::Shape)->Option<Pos2>{match shape {
            egui::epaint::Shape::Text(text) if text.galley.job.text=="初始画布" => Some(text.pos+text.galley.size()*0.5),
            egui::epaint::Shape::Vec(shapes)=>shapes.iter().find_map(find),_=>None,
        }}
        shapes.iter().find_map(|shape|find(&shape.shape)).expect("root history card label must be visible")
    }

    fn select_history_root(ctx:&egui::Context,editor:&mut EditorUi,document:&Document,region:&DrawingRegion) {
        editor.show_history();
        let (_,output)=render_frame_output(ctx,editor,document,region,false,false,Vec2::new(1440.,920.),vec![]);
        let point=history_root_label(&output.shapes);
        assert!(render_frame(ctx,editor,document,region,vec![egui::Event::PointerMoved(point),pointer_event(point,true)]).is_empty());
        assert!(matches!(render_frame(ctx,editor,document,region,vec![pointer_event(point,false)]).as_slice(),[UiAction::Jump(0)]));
    }

    fn history_delete_key(pressed:bool)->egui::Event{
        egui::Event::Key{key:egui::Key::Delete,physical_key:None,pressed,repeat:false,modifiers:Default::default()}
    }

    #[test]
    fn history_delete_after_card_click_is_allowed_while_running_and_help_preserves_selection() {
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=color_document();let region=DrawingRegion::default();
        select_history_root(&ctx,&mut editor,&document,&region);
        let actions=render_frame_flags(&ctx,&mut editor,&document,&region,true,false,vec![history_delete_key(true)]);
        assert!(matches!(actions.as_slice(),[UiAction::DeleteHistory(ids)] if ids==&vec![0]),"UI must request deletion without stopping a running model itself");
        render_frame(&ctx,&mut editor,&document,&region,vec![history_delete_key(false)]);
        editor.show_shortcuts=true;
        assert!(render_frame(&ctx,&mut editor,&document,&region,vec![history_delete_key(true)]).is_empty());
        render_frame(&ctx,&mut editor,&document,&region,vec![history_delete_key(false)]);
        editor.show_shortcuts=false;
        assert!(matches!(render_frame(&ctx,&mut editor,&document,&region,vec![history_delete_key(true)]).as_slice(),[UiAction::DeleteHistory(ids)] if ids==&vec![0]));
        assert_eq!(document.nodes.len(),2,"UI must never mutate/delete history directly");
    }

    #[test]
    fn history_invalidation_defers_thumbnail_release_and_reloads_same_numeric_ids_next_frame() {
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=color_document();let region=DrawingRegion::default();
        select_history_root(&ctx,&mut editor,&document,&region);
        render_frame(&ctx,&mut editor,&document,&region,vec![]);
        editor.canvas.selection=vec![document.primitives[1].id];
        editor.freeze_requested_for_node=Some(document.current as u64);
        editor.color_context=Some((document.current as u64,editor.canvas.selection.clone()));
        editor.invalidate_history_after_deletion();
        assert!(editor.history_thumbnails_invalidated&&editor.node.is_none()&&editor.property.is_none()
            &&editor.color_context.is_none()&&editor.freeze_requested_for_node.is_none()&&editor.canvas.selection.is_empty());
        let (actions,output)=render_frame_output(&ctx,&mut editor,&document,&region,false,false,Vec2::new(1440.,920.),vec![]);
        assert!(actions.is_empty());assert!(!editor.history_thumbnails_invalidated);
        assert!(output.textures_delta.set.iter().any(|(_,delta)|delta.image.size()==[96,96]),
            "unchanged numeric IDs and node count still need freshly uploaded previews after compaction");
        assert!(render_frame(&ctx,&mut editor,&document,&region,vec![history_delete_key(true)]).is_empty(),"invalidated selection cannot target remapped IDs");
    }

    #[test]
    fn file_save_shortcut_keeps_the_last_text_edit_from_the_same_frame() {
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=color_document();let region=DrawingRegion::default();
        render_frame(&ctx,&mut editor,&document,&region,vec![]);editor.set_project_path(Some(PathBuf::from("保存测试.pproj")));
        ctx.memory_mut(|m|m.request_focus(egui::Id::new("creation_prompt_text")));
        let actions=render_frame(&ctx,&mut editor,&document,&region,vec![egui::Event::Text("红发。".into()),command_key(egui::Key::S)]);
        assert!(actions.iter().any(|a|matches!(a,UiAction::SetPrompt(text)if text.contains("红发。"))));
        assert_eq!(actions.iter().filter(|a|matches!(a,UiAction::SaveProject(_))).count(),1);
        assert!(!actions.iter().any(|a|matches!(a,UiAction::Stop|UiAction::Generate{..})));
    }

    #[test]
    fn file_busy_freezes_placeholder_document_sync_but_keeps_stop_available() {
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=color_document();let region=DrawingRegion::default();
        render_frame(&ctx,&mut editor,&document,&region,vec![]);
        editor.prompt="尚未提交的编辑".into();editor.settings.cfg=2.75;editor.canvas.selection=vec![document.primitives[1].id];
        let old_node=editor.node;let old_selection=editor.canvas.selection.clone();editor.set_file_busy(true);
        let placeholder=Document::new();
        let actions=render_frame_flags(&ctx,&mut editor,&placeholder,&region,true,false,vec![command_key(egui::Key::S),command_key(egui::Key::O),command_key(egui::Key::Z),history_delete_key(true)]);
        assert!(actions.is_empty());assert_eq!(editor.prompt,"尚未提交的编辑");assert_eq!(editor.settings.cfg,2.75);
        assert_eq!(editor.node,old_node);assert_eq!(editor.canvas.selection,old_selection);
        let stop=egui::Event::Key{key:egui::Key::Escape,physical_key:None,pressed:true,repeat:false,modifiers:Default::default()};
        assert!(matches!(render_frame_flags(&ctx,&mut editor,&placeholder,&region,true,false,vec![stop]).as_slice(),[UiAction::Stop]));
        editor.set_file_busy(false);render_frame(&ctx,&mut editor,&document,&region,vec![]);
        assert_eq!(editor.prompt,"尚未提交的编辑");assert_eq!(editor.canvas.selection,old_selection);
    }

    #[test]
    fn file_import_preserves_same_frame_prompt_and_staged_settings_until_confirmation() {
        for settings_draft in [false,true]{
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=Document::new();let region=DrawingRegion::default();
            render_frame(&ctx,&mut editor,&document,&region,vec![]);
            editor.files.dialog_results.push_back(Ok(Some(PathBuf::from("其他项目.pproj"))));
            let mut events=Vec::new();
            if settings_draft{editor.settings.cfg+=0.5;}else{
                ctx.memory_mut(|m|m.request_focus(egui::Id::new("creation_prompt_text")));events.push(egui::Event::Text("红发。".into()));
            }
            events.push(command_key(egui::Key::O));
            let actions=render_frame(&ctx,&mut editor,&document,&region,events);
            assert!(!actions.iter().any(|action|matches!(action,UiAction::ImportProject(_)|UiAction::Stop)));
            assert!(editor.files.has_modal(),"uncommitted editor drafts need replacement confirmation too");
            if !settings_draft{assert!(actions.iter().any(|a|matches!(a,UiAction::SetPrompt(text)if text.contains("红发。"))));}
        }
    }

    fn clipboard_frame(
        ctx: &egui::Context, editor: &mut EditorUi, document: &Document,
        region: &DrawingRegion, events: Vec<egui::Event>,
    ) -> (Vec<UiAction>, egui::FullOutput) {
        render_frame_output(ctx, editor, document, region, false, false,
            Vec2::new(1440.0, 920.0), events)
    }

    fn command_key(key: egui::Key) -> egui::Event {
        egui::Event::Key { key, physical_key: None, pressed: true, repeat: false,
            modifiers: egui::Modifiers { ctrl: true, command: true, ..Default::default() } }
    }

    #[test]
    fn windows_copy_paste_events_use_a_real_clipboard_payload_and_one_paste_action() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[0].id, document.primitives[1].id];
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let (actions, output) = clipboard_frame(&ctx, &mut editor, &document, &region, vec![egui::Event::Copy]);
        assert!(actions.is_empty(), "copying must not change history");
        let payload = output.platform_output.copied_text;
        assert!(payload.starts_with("PrimitivePainter.Primitives.v1\n"), "the OS clipboard needs actual native data");
        assert!(!ctx.input(|input| input.events.iter().any(|event| matches!(event, egui::Event::Copy))));
        // Clear the internal buffer so a semantic paste must reconstruct the OS payload.
        editor.canvas.selection.clear();
        editor.canvas.copy(&document.primitives);
        let (actions, _) = clipboard_frame(&ctx, &mut editor, &document, &region, vec![egui::Event::Paste(payload)]);
        let [UiAction::PasteCanvas { primitives, .. }] = actions.as_slice() else { panic!("one semantic paste expected") };
        assert_eq!(&primitives[..document.primitives.len()], document.primitives.as_slice());
        assert_eq!(primitives.len(), document.primitives.len() + 1, "background must never be duplicated");
        let pasted = primitives.last().unwrap();
        assert_eq!(pasted.id, 0, "document must allocate a fresh ID");
        assert_eq!(pasted.x, document.primitives[1].x + 16.0);
        assert_eq!(pasted.y, document.primitives[1].y + 16.0);
        assert_eq!(pasted.color, document.primitives[1].color);
        assert!(!ctx.input(|input| input.events.iter().any(|event| matches!(event, egui::Event::Paste(_)))));
        assert_eq!(document.nodes.len(), 2);
        assert_eq!(document.primitives.len(), 3);
    }

    #[test]
    fn raw_and_semantic_clipboard_events_are_deduplicated_and_foreign_text_never_pastes_old_shapes() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[1].id, document.primitives[2].id];
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let (actions, output) = clipboard_frame(&ctx, &mut editor, &document, &region,
            vec![command_key(egui::Key::C), egui::Event::Copy]);
        assert!(actions.is_empty());
        let payload = output.platform_output.copied_text;
        assert!(!payload.is_empty());
        let (actions, _) = clipboard_frame(&ctx, &mut editor, &document, &region,
            vec![egui::Event::Paste(payload), command_key(egui::Key::V)]);
        let [UiAction::PasteCanvas { primitives, .. }] = actions.as_slice() else { panic!("two event types must still produce one paste") };
        assert_eq!(primitives.len(), document.primitives.len() + 2);
        let (actions, _) = clipboard_frame(&ctx, &mut editor, &document, &region,
            vec![command_key(egui::Key::V), egui::Event::Paste("普通文本，不是图元".into())]);
        assert!(actions.is_empty(), "a semantic paste cannot fall back to stale internal shapes");
        let (actions, _) = clipboard_frame(&ctx, &mut editor, &document, &region,
            vec![command_key(egui::Key::V)]);
        assert!(matches!(actions.as_slice(), [UiAction::PasteCanvas { .. }]), "raw-key compatibility remains available");
    }

    #[test]
    fn raw_clipboard_shortcuts_work_without_semantic_events() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[1].id];
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let (actions, output) = clipboard_frame(&ctx, &mut editor, &document, &region,
            vec![command_key(egui::Key::C)]);
        assert!(actions.is_empty());
        assert!(!output.platform_output.copied_text.is_empty());
        let (actions, _) = clipboard_frame(&ctx, &mut editor, &document, &region,
            vec![command_key(egui::Key::V)]);
        assert!(matches!(actions.as_slice(), [UiAction::PasteCanvas { .. }]));
    }

    #[test]
    fn focused_prompt_keeps_ordinary_text_copy_and_paste() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = color_document();
        document.prompt = "原始提示词".into();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[1].id];
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        ctx.memory_mut(|memory| memory.request_focus(egui::Id::new("creation_prompt_text")));
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        assert!(ctx.wants_keyboard_input());
        let (actions, _) = clipboard_frame(&ctx, &mut editor, &document, &region,
            vec![egui::Event::Paste("文本粘贴。".into())]);
        assert!(actions.iter().any(|action| matches!(action, UiAction::SetPrompt(text) if text.contains("文本粘贴。"))));
        assert!(actions.iter().all(|action| matches!(action, UiAction::SetPrompt(_))));
        let text = editor.prompt.clone();
        let (actions, output) = clipboard_frame(&ctx, &mut editor, &document, &region,
            vec![command_key(egui::Key::A), egui::Event::Copy]);
        assert!(actions.is_empty());
        assert_eq!(output.platform_output.copied_text, text);
        assert_eq!(document.prompt, "原始提示词");
        assert_eq!(document.nodes.len(), 2);
    }

    #[test]
    fn clipboard_canvas_routing_does_not_run_in_other_tools_pages_loading_or_stopping() {
        for case in 0..6 {
            let ctx = egui::Context::default();
            let mut editor = EditorUi::new(&ctx);
            let document = color_document();
            let region = DrawingRegion::default();
            editor.canvas.selection = vec![document.primitives[1].id];
            render_frame(&ctx, &mut editor, &document, &region, vec![]);
            let payload = editor.canvas.copy_payload(&document.primitives).unwrap();
            match case {
                0 => editor.canvas.tool = Tool::Rectangle,
                1 => editor.page = Page::History,
                2 => editor.show_settings(),
                3 => editor.show_prompt_templates(),
                _ => {},
            }
            render_frame_state_output(&ctx, &mut editor, &document, &region,
                case == 5, case == 4, case == 5, Vec2::new(1440.0, 920.0), vec![]);
            let (actions, output) = render_frame_state_output(&ctx, &mut editor, &document, &region,
                case == 5, case == 4, case == 5, Vec2::new(1440.0, 920.0),
                vec![egui::Event::Copy, command_key(egui::Key::C),
                    egui::Event::Paste(payload), command_key(egui::Key::V)]);
            assert!(!actions.iter().any(|action| matches!(action, UiAction::PasteCanvas { .. } | UiAction::CommitCanvas { .. })), "case {case}");
            assert!(output.platform_output.copied_text.is_empty(), "case {case} must not copy canvas data");
            if case == 0 { assert!(editor.canvas.tool == Tool::Rectangle); }
            assert_eq!(document.nodes.len(), 2);
        }
    }

    #[test]
    fn hsv_shape_drag_commits_once_without_background_or_geometry_changes() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[0].id, document.primitives[1].id];
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        editor.property.as_mut().unwrap().x += 100.0;
        editor.property.as_mut().unwrap().geometry_dirty = true;
        let rect = editor.shape_color_picker.sv_rect;
        let start = rect.min + rect.size() * 0.3;
        let end = rect.min + rect.size() * 0.7;
        assert!(render_frame(&ctx, &mut editor, &document, &region,
            vec![egui::Event::PointerMoved(start), pointer_event(start, true)]).is_empty());
        for fraction in [0.4, 0.5, 0.6, 0.7] {
            let point = rect.min + rect.size() * fraction;
            assert!(render_frame(&ctx, &mut editor, &document, &region,
                vec![egui::Event::PointerMoved(point)]).is_empty(), "drag samples cannot create states");
        }
        let actions = render_frame(&ctx, &mut editor, &document, &region,
            vec![pointer_event(end, false)]);
        assert_eq!(actions.len(), 1);
        let UiAction::CommitCanvas { label, primitives } = &actions[0] else { panic!("color gesture must only commit canvas") };
        assert_eq!(primitives[0], document.primitives[0], "background is controlled separately");
        assert_ne!(primitives[1].color, document.primitives[1].color);
        assert_eq!(primitives[1].x, document.primitives[1].x, "pending geometry must not be applied with color");
        assert_eq!(primitives[2], document.primitives[2], "unselected shape must stay unchanged");
        let old_count = document.nodes.len();
        document.commit_canvas(label, OperationSource::User, primitives.clone()).unwrap();
        assert_eq!(document.nodes.len(), old_count + 1);
        assert!(render_frame(&ctx, &mut editor, &document, &region, vec![]).is_empty());
    }

    #[test]
    fn hsv_background_is_independent_and_white_click_can_create_it() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[1].id, document.primitives[2].id];
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let brush = editor.canvas.new_color;
        let point = editor.background_color_picker.sv_rect.center();
        let actions = click_at(&ctx, &mut editor, &document, &region, point);
        assert_eq!(actions.len(), 1);
        let UiAction::CommitCanvas { primitives, .. } = &actions[0] else { panic!("background gesture must only commit canvas") };
        assert_ne!(primitives[0].color, document.primitives[0].color);
        assert_eq!(primitives[1..], document.primitives[1..]);
        assert_eq!(editor.canvas.new_color, brush);

        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut empty = Document::new();
        render_frame(&ctx, &mut editor, &empty, &region, vec![]);
        // Changing hue on a white selection keeps RGB white but is an explicit gesture.
        let point = editor.background_color_picker.hue_rect.center();
        let actions = click_at(&ctx, &mut editor, &empty, &region, point);
        assert_eq!(actions.len(), 1);
        let UiAction::CommitCanvas { label, primitives } = &actions[0] else { panic!("white gesture must create background") };
        assert_eq!(primitives.len(), 1);
        assert_eq!(primitives[0].shape, ShapeKind::Background);
        assert_eq!(primitives[0].color, [254; 3]);
        empty.commit_canvas(label, OperationSource::User, primitives.clone()).unwrap();
        render_frame(&ctx, &mut editor, &empty, &region, vec![]);
        let point = editor.background_color_picker.hue_rect.center();
        assert!(click_at(&ctx, &mut editor, &empty, &region, point).is_empty(), "same color cannot add redundant states");
    }

    #[test]
    fn hsv_controls_are_read_only_during_loading_or_stopping() {
        for (running, loading, stopping) in [(true, false, true), (false, true, false)] {
            let ctx = egui::Context::default();
            let mut editor = EditorUi::new(&ctx);
            let document = color_document();
            let region = DrawingRegion::default();
            editor.canvas.selection = vec![document.primitives[1].id];
            render_frame_state_output(&ctx, &mut editor, &document, &region,
                running, loading, stopping, Vec2::new(1440.0, 920.0), vec![]);
            let brush = editor.canvas.new_color;
            let background = editor.background_color;
            for point in [editor.shape_color_picker.sv_rect.center(), editor.background_color_picker.sv_rect.center()] {
                assert!(click_state_control(&ctx, &mut editor, &document, &region,
                    running, loading, stopping, point).is_empty());
            }
            assert_eq!(editor.canvas.new_color, brush);
            assert_eq!(editor.background_color, background);
        }
    }

    #[test]
    fn running_canvas_browsing_and_copying_do_not_request_a_stop() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[1].id];
        assert!(render_frame_flags(&ctx, &mut editor, &document, &region, true, false, vec![]).is_empty(),
            "watching live generation without an interaction must not freeze its view");
        let (actions, output) = render_frame_output(&ctx, &mut editor, &document, &region,
            true, false, Vec2::new(1440.0, 920.0), vec![egui::Event::Copy]);
        assert!(actions.is_empty());
        assert!(!output.platform_output.copied_text.is_empty());
        let actions = render_frame_flags(&ctx, &mut editor, &document, &region,
            true, false, vec![command_key(egui::Key::Z)]);
        assert!(matches!(actions.as_slice(), [UiAction::Jump(0)]), "history browsing must only request a jump");
        let history_tab = editor.navigation_rects[1].center();
        assert!(click_state_control(&ctx, &mut editor, &document, &region,
            true, false, false, history_tab).is_empty());
        assert!(editor.page == Page::History);
        let canvas_tab = editor.navigation_rects[0].center();
        assert!(click_state_control(&ctx, &mut editor, &document, &region,
            true, false, false, canvas_tab).is_empty());
        let point = editor.edit_canvas_rect.center();
        let zoom = editor.canvas.zoom;
        let actions = render_frame_flags(&ctx, &mut editor, &document, &region,
            true, false, vec![egui::Event::PointerMoved(point), egui::Event::Scroll(Vec2::new(0.0, 80.0))]);
        assert!(matches!(actions.as_slice(), [UiAction::FreezeGenerationView]));
        assert!(editor.canvas.zoom > zoom);
        assert_eq!(document.nodes.len(), 2);
    }

    #[test]
    fn running_canvas_draw_releases_one_edit_for_the_parent_to_queue() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        editor.canvas.tool = Tool::Rectangle;
        let document = color_document();
        let region = DrawingRegion::default();
        render_frame_flags(&ctx, &mut editor, &document, &region, true, false, vec![]);
        let start = editor.edit_canvas_rect.center() - Vec2::new(25.0, 20.0);
        let end = start + Vec2::new(55.0, 40.0);
        assert!(matches!(render_frame_flags(&ctx, &mut editor, &document, &region, true, false,
            vec![egui::Event::PointerMoved(start), pointer_event(start, true)]).as_slice(),
            [UiAction::FreezeGenerationView]), "starting a gesture must freeze the view without stopping inference");
        assert!(render_frame_flags(&ctx, &mut editor, &document, &region, true, false,
            vec![egui::Event::PointerMoved(end)]).is_empty());
        let actions = render_frame_flags(&ctx, &mut editor, &document, &region, true, false,
            vec![pointer_event(end, false)]);
        let [UiAction::CommitCanvas { primitives, .. }] = actions.as_slice() else { panic!("one completed manual edit expected") };
        assert_eq!(primitives.len(), document.primitives.len() + 1);
        assert_eq!(document.nodes.len(), 2);
    }

    #[test]
    fn returning_to_a_live_history_node_can_freeze_again_with_real_pointer_input() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = color_document();
        let region = DrawingRegion::default();
        let live_head = document.current;
        render_frame_flags(&ctx, &mut editor, &document, &region, true, false, vec![]);
        let point = editor.edit_canvas_rect.center();
        assert!(matches!(click_state_control(&ctx, &mut editor, &document, &region,
            true, false, false, point).as_slice(), [UiAction::FreezeGenerationView]));
        assert!(click_state_control(&ctx, &mut editor, &document, &region,
            true, false, false, point).is_empty(), "one view needs only one freeze request");

        // These model updates are the result of main applying A -> X -> A history browsing.
        document.jump(0).unwrap();
        render_frame_flags(&ctx, &mut editor, &document, &region, true, false, vec![]);
        document.jump(live_head).unwrap();
        render_frame_flags(&ctx, &mut editor, &document, &region, true, false, vec![]);
        let point = editor.edit_canvas_rect.center();
        assert!(matches!(click_state_control(&ctx, &mut editor, &document, &region,
            true, false, false, point).as_slice(), [UiAction::FreezeGenerationView]),
            "returning to the same live head must not reuse its old freeze suppression");

        // Main can explicitly resume live following by selecting the same head again.
        editor.reset_generation_view_freeze();
        assert!(matches!(click_state_control(&ctx, &mut editor, &document, &region,
            true, false, false, point).as_slice(), [UiAction::FreezeGenerationView]));
        assert_eq!(document.nodes.len(), 2);
        assert_eq!(document.current, live_head);
    }

    #[test]
    fn running_keyboard_edit_freezes_the_current_view_before_sending_the_edit() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[1].id];
        render_frame_flags(&ctx, &mut editor, &document, &region, true, false, vec![]);
        let actions = render_frame_flags(&ctx, &mut editor, &document, &region, true, false,
            vec![egui::Event::Key { key: egui::Key::Delete, physical_key: None,
                pressed: true, repeat: false, modifiers: Default::default() }]);
        assert!(matches!(actions.as_slice(), [UiAction::FreezeGenerationView, UiAction::CommitCanvas { .. }]));
        assert_eq!(document.nodes.len(), 2);
    }

    #[test]
    fn running_hsv_delete_and_paste_emit_edits_while_generation_settings_remain_disabled() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[1].id];
        render_frame_flags(&ctx, &mut editor, &document, &region, true, false, vec![]);
        let payload = editor.canvas.copy_payload(&document.primitives).unwrap();
        let point = editor.shape_color_picker.sv_rect.center();
        let actions = click_state_control(&ctx, &mut editor, &document, &region, true, false, false, point);
        assert!(matches!(actions.as_slice(), [UiAction::FreezeGenerationView, UiAction::CommitCanvas { .. }]));
        let actions = render_frame_flags(&ctx, &mut editor, &document, &region, true, false,
            vec![egui::Event::Key { key: egui::Key::Delete, physical_key: None, pressed: true,
                repeat: false, modifiers: Default::default() }]);
        assert!(matches!(actions.as_slice(), [UiAction::CommitCanvas { .. }]));
        let actions = render_frame_flags(&ctx, &mut editor, &document, &region, true, false,
            vec![egui::Event::Paste(payload)]);
        assert!(matches!(actions.as_slice(), [UiAction::PasteCanvas { .. }]));
        let pause = editor.pause_stage_end_control_rect.center();
        assert!(click_state_control(&ctx, &mut editor, &document, &region,
            true, false, false, pause).is_empty());
        assert!(!editor.settings.pause_at_stage_end);
        let automatic = editor.auto_reorder_control_rect.center();
        assert!(click_state_control(&ctx, &mut editor, &document, &region,
            true, false, false, automatic).is_empty());
        assert!(!editor.settings.auto_reorder_before_generation);
        assert_eq!(document.nodes.len(), 2);
    }

    #[test]
    fn stopping_prevents_additional_edit_actions_and_color_gestures() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = color_document();
        let region = DrawingRegion::default();
        editor.canvas.selection = vec![document.primitives[1].id];
        let size = Vec2::new(1440.0, 920.0);
        render_frame_state_output(&ctx, &mut editor, &document, &region, true, false, true, size, vec![]);
        let payload = editor.canvas.copy_payload(&document.primitives).unwrap();
        let (actions, _) = render_frame_state_output(&ctx, &mut editor, &document, &region,
            true, false, true, size, vec![egui::Event::Paste(payload),
                egui::Event::Key { key: egui::Key::Delete, physical_key: None, pressed: true,
                    repeat: false, modifiers: Default::default() }]);
        assert!(actions.is_empty());
        for point in [editor.shape_color_picker.sv_rect.center(), editor.background_color_picker.sv_rect.center(),
            editor.manual_reorder_rect.center(), editor.pause_stage_end_control_rect.center()] {
            assert!(click_state_control(&ctx, &mut editor, &document, &region,
                true, false, true, point).is_empty());
        }
        editor.canvas.tool = Tool::Rectangle;
        let start = editor.edit_canvas_rect.center();
        let end = start + Vec2::new(40.0, 30.0);
        for events in [vec![egui::Event::PointerMoved(start), pointer_event(start, true)],
            vec![egui::Event::PointerMoved(end)], vec![pointer_event(end, false)]] {
            assert!(render_frame_state_output(&ctx, &mut editor, &document, &region,
                true, false, true, size, events).0.is_empty());
        }
        assert_eq!(document.nodes.len(), 2);
    }

    #[test]
    fn hsv_hue_survives_a_black_color_commit_and_history_node_change() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = Document::new();
        document.commit_canvas("蓝色图形", OperationSource::User, vec![
            Primitive::new(ShapeKind::Rectangle, 150.0, 170.0, 80.0, 60.0, [0, 0, 200]),
        ]).unwrap();
        editor.canvas.selection = vec![document.primitives[0].id];
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let sv = editor.shape_color_picker.sv_rect;
        let black = Pos2::new(sv.left() + sv.width() * 0.8, sv.bottom() - 0.01);
        let actions = click_at(&ctx, &mut editor, &document, &region, black);
        let [UiAction::CommitCanvas { label, primitives }] = actions.as_slice() else { panic!("one black-color commit expected") };
        assert_eq!(primitives[0].color, [0; 3]);
        document.commit_canvas(label, OperationSource::User, primitives.clone()).unwrap();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let sv = editor.shape_color_picker.sv_rect;
        let vivid = sv.min + sv.size() * 0.2 + Vec2::new(sv.width() * 0.6, 0.0);
        let actions = click_at(&ctx, &mut editor, &document, &region, vivid);
        let [UiAction::CommitCanvas { primitives, .. }] = actions.as_slice() else { panic!("one restored-color commit expected") };
        let color = primitives[0].color;
        assert!(color[2] as u16 > color[0] as u16 * 2 && color[2] as u16 > color[1] as u16 * 2,
            "restoring brightness after black should preserve blue hue: {color:?}");
    }

    #[test]
    fn hsv_tool_rail_and_drawing_progress_fit_a_small_window() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = Document::new();
        let region = DrawingRegion::default();
        let size = Vec2::new(1100.0, 720.0);
        for _ in 0..2 {
            render_frame_sized(&ctx, &mut editor, &document, &region, false, false, size, vec![]);
        }
        let rail = egui::containers::panel::PanelState::load(&ctx, egui::Id::new("tool_rail")).unwrap().rect;
        assert!(rail.width() <= 196.5);
        assert!(editor.shape_color_picker.sv_rect.left() >= rail.left());
        assert!(editor.background_color_picker.sv_rect.right() <= rail.right());
        assert!(!editor.shape_color_picker.sv_rect.intersects(editor.background_color_picker.sv_rect));
        assert!(editor.edit_canvas_rect.right() < editor.region_indicator_rect.left());
        assert!(editor.region_indicator_rect.bottom() < size.y);
        assert!(editor.background_color_picker.sv_rect.height() > 100.0);
    }

    #[test]
    fn settings_navigation_preserves_canvas_history_and_staged_generation_inputs() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = Document::new();
        document.prompt = "原提示词".into();
        document.commit_canvas("已有画布", OperationSource::User,
            vec![Primitive::background([240, 230, 220])]).unwrap();
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        editor.prompt = "尚未生成的提示词草稿".into();
        editor.settings.cfg = 2.5;
        let canvas = document.primitives.clone();
        let node = document.current;
        let count = document.nodes.len();
        let open = editor.navigation_rects[2].center();
        let actions = click_at(&ctx, &mut editor, &document, &region, open);
        assert!(editor.page == Page::Settings);
        assert!(actions.is_empty(), "Opening settings must not load or generate");
        let back = editor.navigation_rects[0].center();
        assert!(click_at(&ctx, &mut editor, &document, &region, back).is_empty());
        assert!(editor.page == Page::Canvas);
        assert_eq!(editor.prompt, "尚未生成的提示词草稿");
        assert_eq!(editor.settings.cfg, 2.5);
        assert_eq!(document.current, node);
        assert_eq!(document.nodes.len(), count);
        assert_eq!(document.primitives, canvas);
        assert_eq!(document.prompt, "原提示词");
    }

    #[test]
    fn missing_model_button_opens_settings_without_requesting_a_file_or_loading() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = Document::new();
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        assert!(editor.open_settings_rect.is_positive());
        let open = editor.open_settings_rect.center();
        let actions = click_at(&ctx, &mut editor, &document, &region, open);
        assert!(actions.is_empty());
        assert!(editor.page == Page::Settings);
        assert!(editor.file_selection_error.is_none());
        assert_eq!(document.nodes.len(), 1);
    }

    #[test]
    fn clicking_a_searched_template_replaces_only_the_prompt_draft() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = Document::new();
        document.prompt = "已有提示词".into();
        document.commit_canvas("已有画布", OperationSource::User,
            vec![Primitive::background([220, 230, 240])]).unwrap();
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let node = document.current;
        let nodes = document.nodes.len();
        let canvas = document.primitives.clone();
        let button = editor.prompt_templates_button_rect.center();
        assert!(click_at(&ctx, &mut editor, &document, &region, button).is_empty());
        assert!(editor.prompt_templates.is_open());
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        assert!(render_frame(&ctx, &mut editor, &document, &region,
            vec![egui::Event::Text("100".into())]).is_empty());
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let point = editor.prompt_templates.visible_rows.iter()
            .find(|(index, _)| *index == 99).expect("search must reveal template 100").1.center();
        let expected = prompt_templates::templates().unwrap()[99].text.clone();
        let actions = click_at(&ctx, &mut editor, &document, &region, point);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], UiAction::SetPrompt(text) if text == &expected));
        assert_eq!(editor.prompt, expected);
        assert!(!editor.prompt_templates.is_open());
        assert_eq!(document.prompt, "已有提示词");
        assert_eq!(document.current, node);
        assert_eq!(document.nodes.len(), nodes);
        assert_eq!(document.primitives, canvas);
    }

    #[test]
    fn templates_are_disabled_while_busy_and_escape_preserves_the_draft() {
        for (running, loading) in [(true, false), (false, true)] {
            let ctx = egui::Context::default();
            let mut editor = EditorUi::new(&ctx);
            let mut document = Document::new();
            document.prompt = "保留提示词".into();
            let region = DrawingRegion::default();
            render_frame_flags(&ctx, &mut editor, &document, &region, running, loading, vec![]);
            let button = editor.prompt_templates_button_rect.center();
            assert!(click_reorder_control(&ctx, &mut editor, &document, &region,
                running, loading, button).is_empty());
            assert!(!editor.prompt_templates.is_open());
            editor.show_prompt_templates();
            render_frame_flags(&ctx, &mut editor, &document, &region, running, loading, vec![]);
            render_frame_flags(&ctx, &mut editor, &document, &region, running, loading, vec![]);
            let row = editor.prompt_templates.visible_rows.first().unwrap().1.center();
            assert!(click_reorder_control(&ctx, &mut editor, &document, &region,
                running, loading, row).is_empty());
            assert!(editor.prompt_templates.is_open());
            assert_eq!(editor.prompt, "保留提示词");
        }
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = Document::new();
        document.prompt = "关闭后保留原文".into();
        document.commit_canvas("已有画布", OperationSource::User,
            vec![Primitive::background([200, 220, 240])]).unwrap();
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        editor.show_prompt_templates();
        let key = |key| egui::Event::Key { key, physical_key: None, pressed: true,
            repeat: false, modifiers: Default::default() };
        assert!(render_frame(&ctx, &mut editor, &document, &region,
            vec![key(egui::Key::Delete), key(egui::Key::Space)]).is_empty());
        assert!(render_frame(&ctx, &mut editor, &document, &region,
            vec![key(egui::Key::Escape)]).is_empty());
        assert!(!editor.prompt_templates.is_open());
        assert_eq!(editor.prompt, "关闭后保留原文");
        assert_eq!(document.nodes.len(), 2);
        assert_eq!(document.primitives.len(), 1);
    }

    #[test]
    fn parallel_control_range_emits_only_parameter_changes_without_history() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = Document::new();
        document.settings.parallel_sequences = 4;
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let press = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut actions = Vec::new();
        for delta in [30.0, -40.0] {
            let start = editor.parallel_control_rect.center();
            let end = start + Vec2::new(delta, 0.0);
            actions.extend(render_frame(
                &ctx,
                &mut editor,
                &document,
                &region,
                vec![egui::Event::PointerMoved(start), press(start, true)],
            ));
            actions.extend(render_frame(
                &ctx,
                &mut editor,
                &document,
                &region,
                vec![egui::Event::PointerMoved(end)],
            ));
            actions.extend(render_frame(
                &ctx,
                &mut editor,
                &document,
                &region,
                vec![press(end, false)],
            ));
        }
        assert!(actions.iter().any(|action| matches!(action, UiAction::SetSettings(settings) if settings.parallel_sequences > 16)));
        assert!(actions.iter().any(|action| matches!(action, UiAction::SetSettings(settings) if settings.parallel_sequences == 1)));
        assert!(actions
            .iter()
            .all(|action| matches!(action, UiAction::SetSettings(_))));
        for action in &actions {
            if let UiAction::SetSettings(settings) = action {
                assert!(settings.parallel_sequences>=1);
                assert!(settings.validate().is_ok());
            }
        }
        assert_eq!(document.nodes.len(), 1);
        assert_eq!(document.current, 0);
        assert_eq!(document.settings.parallel_sequences, 4);
    }

    #[test]
    fn parallel_count_keeps_keyboard_focus_when_warning_appears() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = Document::new();
        document.settings.parallel_sequences = 1;
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let position = editor.parallel_control_rect.center();
        let button = |pressed| egui::Event::PointerButton {
            pos: position, button: egui::PointerButton::Primary,
            pressed, modifiers: Default::default(),
        };
        render_frame(&ctx, &mut editor, &document, &region,
            vec![egui::Event::PointerMoved(position), button(true)]);
        render_frame(&ctx, &mut editor, &document, &region, vec![button(false)]);
        let focused = ctx.memory(|memory| memory.focused());
        assert!(focused.is_some());
        render_frame(&ctx, &mut editor, &document, &region,
            vec![egui::Event::Text("32".into())]);
        assert_eq!(editor.settings.parallel_sequences, 32);
        render_frame(&ctx, &mut editor, &document, &region,
            vec![egui::Event::Text("0".into())]);
        assert_eq!(editor.settings.parallel_sequences, 320);
        assert_eq!(ctx.memory(|memory| memory.focused()), focused);
        assert!(editor.parallel_warning_rect.is_positive());
        assert_eq!(document.nodes.len(), 1);
    }

    #[test]
    fn large_parallel_count_is_retained_and_warning_is_above_the_input(){
        for count in [16,17,32,1000] {
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);
            let mut document=Document::new();document.settings.parallel_sequences=count;
            let region=DrawingRegion::default();
            let actions=render_frame(&ctx,&mut editor,&document,&region,vec![]);
            assert_eq!(editor.settings.parallel_sequences,count);
            assert!(actions.iter().all(|a|!matches!(a,UiAction::Generate{..}|UiAction::Stop)));
            if count>16 {
                assert!(editor.parallel_warning_rect.is_positive());
                assert!(editor.parallel_warning_rect.max.y<editor.parallel_control_rect.min.y);
            }else{assert_eq!(editor.parallel_warning_rect,Rect::NOTHING);}
        }
    }

    #[test]
    fn loading_and_generation_disable_parallel_parameter_edits() {
        for (running, loading) in [(true, false), (false, true)] {
            let ctx = egui::Context::default();
            let mut editor = EditorUi::new(&ctx);
            let mut document = Document::new();
            document.settings.parallel_sequences = 8;
            let region = DrawingRegion::default();
            render_frame_flags(
                &ctx,
                &mut editor,
                &document,
                &region,
                running,
                loading,
                vec![],
            );
            let start = editor.parallel_control_rect.center();
            let end = start + Vec2::new(30.0, 0.0);
            let press = |pos, pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            let mut actions = render_frame_flags(
                &ctx,
                &mut editor,
                &document,
                &region,
                running,
                loading,
                vec![egui::Event::PointerMoved(start), press(start, true)],
            );
            actions.extend(render_frame_flags(
                &ctx,
                &mut editor,
                &document,
                &region,
                running,
                loading,
                vec![egui::Event::PointerMoved(end)],
            ));
            actions.extend(render_frame_flags(
                &ctx,
                &mut editor,
                &document,
                &region,
                running,
                loading,
                vec![press(end, false)],
            ));
            assert_eq!(editor.settings.parallel_sequences, 8);
            assert!(!actions.iter().any(|action| matches!(
                action,
                UiAction::SetSettings(_)
                    | UiAction::CommitCanvas { .. }
                    | UiAction::Generate { .. }
            )));
        }
    }

    fn click_reorder_control(
        ctx: &egui::Context,
        editor: &mut EditorUi,
        document: &Document,
        region: &DrawingRegion,
        running: bool,
        loading: bool,
        point: Pos2,
    ) -> Vec<UiAction> {
        click_state_control(ctx, editor, document, region, running, loading, false, point)
    }

    fn click_state_control(
        ctx: &egui::Context,
        editor: &mut EditorUi,
        document: &Document,
        region: &DrawingRegion,
        running: bool,
        loading: bool,
        stopping: bool,
        point: Pos2,
    ) -> Vec<UiAction> {
        let mut actions = Vec::new();
        for pressed in [true, false] {
            actions.extend(render_frame_state_output(
                ctx, editor, document, region, running, loading, stopping, Vec2::new(1440.0, 920.0),
                vec![egui::Event::PointerMoved(point), egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                }],
            ).0);
        }
        actions
    }

    #[test]
    fn diagnostic_controls_remain_available_without_stopping_generation_or_loading() {
        for (running,loading,stopping) in [(false,false,false),(true,false,false),(true,false,true),(false,true,false)] {
            let ctx=egui::Context::default();
            let mut editor=EditorUi::new(&ctx);
            editor.show_settings();
            let document=Document::new();
            let region=DrawingRegion::default();
            render_frame_state_output(&ctx,&mut editor,&document,&region,running,loading,stopping,Vec2::new(1440.0,920.0),vec![]);
            let open=editor.log_directory_rect.center();
            assert!(editor.log_directory_rect.is_positive());
            assert!(matches!(click_state_control(&ctx,&mut editor,&document,&region,running,loading,stopping,open).as_slice(),[UiAction::OpenLogs]));
            let export=editor.diagnostics_export_rect.center();
            assert!(matches!(click_state_control(&ctx,&mut editor,&document,&region,running,loading,stopping,export).as_slice(),[UiAction::ExportDiagnostics]));
            assert_eq!(document.nodes.len(),1);
        }
    }

    #[test]
    fn stage_end_pause_checkbox_only_stages_settings_and_recovers_with_history_nodes() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = Document::new();
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        assert!(!editor.settings.pause_at_stage_end);
        let point = editor.pause_stage_end_control_rect.center();
        let actions = click_reorder_control(&ctx, &mut editor, &document, &region, false, false, point);
        let [UiAction::SetSettings(settings)] = actions.as_slice() else { panic!("only a settings change is expected") };
        assert!(settings.pause_at_stage_end);
        assert!(editor.settings.pause_at_stage_end);
        assert!(!document.settings.pause_at_stage_end);
        assert_eq!(document.nodes.len(), 1);
        assert_eq!(document.current, 0);

        document.settings = settings.clone();
        let enabled = document.commit_generated(Primitive::background([220, 230, 240])).unwrap();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        assert!(editor.settings.pause_at_stage_end);
        document.jump(0).unwrap();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        assert!(!editor.settings.pause_at_stage_end);
        document.jump(enabled).unwrap();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        assert!(editor.settings.pause_at_stage_end);
        let point = editor.pause_stage_end_control_rect.center();
        let actions = click_reorder_control(&ctx, &mut editor, &document, &region, false, false, point);
        assert!(matches!(actions.as_slice(), [UiAction::SetSettings(settings)] if !settings.pause_at_stage_end));
        assert_eq!(document.nodes.len(), 2);
        assert_eq!(document.current, enabled);
        assert!(document.nodes[enabled].settings.pause_at_stage_end);
    }

    #[test]
    fn loading_and_generation_disable_stage_end_pause_changes() {
        for (running, loading) in [(true, false), (false, true)] {
            let ctx = egui::Context::default();
            let mut editor = EditorUi::new(&ctx);
            let mut document = Document::new();
            document.settings.pause_at_stage_end = true;
            let region = DrawingRegion::default();
            render_frame_flags(&ctx, &mut editor, &document, &region, running, loading, vec![]);
            let point = editor.pause_stage_end_control_rect.center();
            assert!(click_reorder_control(&ctx, &mut editor, &document, &region,
                running, loading, point).is_empty());
            assert!(editor.settings.pause_at_stage_end);
            assert_eq!(document.nodes.len(), 1);
        }
    }

    #[test]
    fn half_region_switch_checkbox_defaults_off_only_stages_settings_and_follows_history() {
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let mut document=Document::new();let region=DrawingRegion::default();
        render_frame(&ctx,&mut editor,&document,&region,vec![]);assert!(!editor.settings.half_region_switch);
        assert!(editor.half_region_switch_control_rect.right()<=editor.properties_panel_rect.right(),"the compact checkbox row must stay inside the inspector");
        let point=editor.half_region_switch_control_rect.center();
        let actions=click_reorder_control(&ctx,&mut editor,&document,&region,false,false,point);
        let [UiAction::SetSettings(settings)]=actions.as_slice()else{panic!("checkbox must only stage settings");};
        assert!(settings.half_region_switch);assert_eq!(document.nodes.len(),1);assert!(!document.settings.half_region_switch);
        document.settings=settings.clone();let enabled=document.commit_generated(Primitive::background([220,230,240])).unwrap();
        render_frame(&ctx,&mut editor,&document,&region,vec![]);assert!(editor.settings.half_region_switch);
        document.jump(0).unwrap();render_frame(&ctx,&mut editor,&document,&region,vec![]);assert!(!editor.settings.half_region_switch);
        document.jump(enabled).unwrap();render_frame(&ctx,&mut editor,&document,&region,vec![]);assert!(editor.settings.half_region_switch);
        assert!(document.nodes[enabled].settings.half_region_switch);
    }

    #[test]
    fn half_region_switch_cannot_change_while_loading_or_generating() {
        for (running,loading) in [(true,false),(false,true)]{
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=Document::new();let region=DrawingRegion::default();
            render_frame_flags(&ctx,&mut editor,&document,&region,running,loading,vec![]);
            let point=editor.half_region_switch_control_rect.center();
            assert!(click_reorder_control(&ctx,&mut editor,&document,&region,running,loading,point).is_empty());
            assert!(!editor.settings.half_region_switch);assert_eq!(document.nodes.len(),1);
        }
    }

    #[test]
    fn stage_buttons_highlight_mixed_selection_and_only_request_metadata_assignment(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=color_document();let region=DrawingRegion::default();
        let ids=[document.primitives[1].id,document.primitives[2].id];let map=BTreeMap::from([(ids[0],Stage::Global),(ids[1],Stage::Coarse)]);
        editor.request_global_transition("旧的待确认编辑");
        editor.canvas.selection=ids.to_vec();
        let frame=|editor:&mut EditorUi,events|render_frame_state_output_stages(&ctx,editor,&document,&region,true,false,false,Vec2::new(1440.,920.),events,Some(&map)).0;
        assert!(frame(&mut editor,vec![]).is_empty());assert_eq!(editor.stage_control_active,[true,true,false]);
        let point=editor.stage_control_rects[2].center();let mut actions=frame(&mut editor,vec![egui::Event::PointerMoved(point),pointer_event(point,true)]);
        actions.extend(frame(&mut editor,vec![pointer_event(point,false)]));
        assert!(matches!(actions.as_slice(),[UiAction::AssignStage{ids:selected,stage:Stage::Fine}]if selected==&ids.to_vec()));
        assert_eq!(document.nodes.len(),2);assert_eq!(document.primitives[1].color,[40,100,180]);
    }

    #[test]
    fn stage_buttons_never_assign_background_and_disable_empty_or_unresolved_selections(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let document=color_document();let region=DrawingRegion::default();
        let background=document.primitives[0].id;let shape=document.primitives[1].id;let map=BTreeMap::from([(shape,Stage::Fine)]);
        for selection in [vec![],vec![background],vec![shape]]{
            editor.canvas.selection=selection.clone();
            render_frame_state_output_stages(&ctx,&mut editor,&document,&region,false,false,false,Vec2::new(1440.,920.),vec![],None);
            let point=editor.stage_control_rects[1].center();
            for pressed in [true,false]{let actions=render_frame_state_output_stages(&ctx,&mut editor,&document,&region,false,false,false,Vec2::new(1440.,920.),vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)],None).0;assert!(actions.is_empty());}
        }
        editor.canvas.selection=vec![background,shape];
        render_frame_state_output_stages(&ctx,&mut editor,&document,&region,false,false,false,Vec2::new(1440.,920.),vec![],Some(&map));
        assert_eq!(editor.stage_control_active,[true,false,true]);let point=editor.stage_control_rects[1].center();let mut actions=Vec::new();
        for pressed in [true,false]{actions.extend(render_frame_state_output_stages(&ctx,&mut editor,&document,&region,false,false,false,Vec2::new(1440.,920.),vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)],Some(&map)).0);}
        assert!(matches!(actions.as_slice(),[UiAction::AssignStage{ids,stage:Stage::Coarse}]if ids==&vec![shape]));
    }

    fn seed_frame(ctx:&egui::Context,editor:&mut EditorUi,doc:&Document,running:bool,loading:bool,events:Vec<egui::Event>)->(Vec<UiAction>,egui::FullOutput){
        render_frame_state_output(ctx,editor,doc,&DrawingRegion::default(),running,loading,false,Vec2::new(1440.,1500.),events)
    }
    fn seed_click(ctx:&egui::Context,editor:&mut EditorUi,doc:&Document,running:bool,loading:bool,point:Pos2)->Vec<UiAction>{
        let mut actions=Vec::new();for pressed in [true,false]{actions.extend(seed_frame(ctx,editor,doc,running,loading,vec![egui::Event::PointerMoved(point),pointer_event(point,pressed)]).0);}actions
    }
    fn open_seed_controls(ctx:&egui::Context,editor:&mut EditorUi,doc:&Document){
        seed_frame(ctx,editor,doc,false,false,vec![]);let point=editor.sampling_header_rect.center();
        assert!(seed_click(ctx,editor,doc,false,false,point).is_empty());
        for _ in 0..20{assert!(seed_frame(ctx,editor,doc,false,false,vec![]).0.is_empty());}
        assert!(editor.auto_seed_control_rect.is_positive()&&editor.seed_control_rect.is_positive());
        assert!(editor.seed_control_rect.right()<=editor.properties_panel_rect.right());
    }
    #[test]
    fn auto_seed_preview_changes_without_settings_or_history_mutation(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let doc=Document::new();open_seed_controls(&ctx,&mut editor,&doc);
        assert!(editor.settings.auto_seed);assert!(!editor.seed_control_enabled);let first=editor.seed_display_value;let seed=doc.settings.seed;
        for _ in 0..4{let(actions,output)=seed_frame(&ctx,&mut editor,&doc,false,false,vec![]);assert!(actions.is_empty());
            assert!(output.viewport_output[&egui::ViewportId::ROOT].repaint_delay<=std::time::Duration::from_millis(200));
            assert_eq!(editor.settings.seed,seed);assert_eq!(doc.settings.seed,seed);assert_eq!(doc.nodes.len(),1);}
        assert_ne!(editor.seed_display_value,first);
    }
    #[test]
    fn auto_seed_toggle_restores_latest_real_seed_after_zero_output_same_node(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let mut doc=Document::new();open_seed_controls(&ctx,&mut editor,&doc);
        doc.settings.seed=987654321;let node=doc.current;
        assert!(seed_frame(&ctx,&mut editor,&doc,false,false,vec![]).0.is_empty());assert_eq!(editor.settings.seed,987654321);
        let point=editor.auto_seed_control_rect.center();let actions=seed_click(&ctx,&mut editor,&doc,false,false,point);
        assert!(matches!(actions.as_slice(),[UiAction::SetSettings(settings)]if !settings.auto_seed&&settings.seed==987654321));
        assert_eq!(editor.manual_seed_text,"987654321");assert!(editor.seed_control_enabled);assert_eq!(doc.current,node);assert_eq!(doc.nodes.len(),1);
    }
    #[test]
    fn manual_seed_text_is_exact_u64_and_auto_mode_clears_invalid_draft(){
        let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let mut doc=Document::new();open_seed_controls(&ctx,&mut editor,&doc);
        let point=editor.auto_seed_control_rect.center();let actions=seed_click(&ctx,&mut editor,&doc,false,false,point);
        let [UiAction::SetSettings(settings)]=actions.as_slice()else{panic!("toggle must only stage settings")};doc.settings=settings.clone();
        let point=editor.seed_control_rect.center();seed_click(&ctx,&mut editor,&doc,false,false,point);
        let exact=u64::MAX-2;let actions=seed_frame(&ctx,&mut editor,&doc,false,false,vec![command_key(egui::Key::A),egui::Event::Text(exact.to_string())]).0;
        assert!(matches!(actions.as_slice(),[UiAction::SetSettings(settings)]if settings.seed==exact&&!settings.auto_seed));
        doc.settings.seed=777;seed_frame(&ctx,&mut editor,&doc,false,false,vec![]);assert_eq!(editor.settings.seed,exact,"external updates cannot overwrite staged manual input");
        seed_frame(&ctx,&mut editor,&doc,false,false,vec![command_key(egui::Key::A),egui::Event::Text("invalid".into())]);assert!(editor.manual_seed_text.parse::<u64>().is_err());
        let point=editor.auto_seed_control_rect.center();seed_click(&ctx,&mut editor,&doc,false,false,point);
        assert!(editor.settings.auto_seed);assert!(editor.manual_seed_text.parse::<u64>().is_ok());assert!(!editor.seed_control_enabled);
    }
    #[test]
    fn auto_seed_and_input_are_readonly_during_generation_or_loading(){
        for(running,loading)in[(true,false),(false,true)]{
            let ctx=egui::Context::default();let mut editor=EditorUi::new(&ctx);let doc=Document::new();open_seed_controls(&ctx,&mut editor,&doc);let first=editor.seed_display_value;
            seed_frame(&ctx,&mut editor,&doc,running,loading,vec![]);let point=editor.auto_seed_control_rect.center();
            assert!(seed_click(&ctx,&mut editor,&doc,running,loading,point).is_empty());
            assert!(editor.settings.auto_seed);assert!(!editor.seed_control_enabled);assert_ne!(editor.seed_display_value,first);assert_eq!(editor.settings.seed,doc.settings.seed);
        }
    }

    #[test]
    fn auto_reorder_is_a_staged_setting_and_manual_reorder_is_a_separate_action() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = Document::new();
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let point = editor.auto_reorder_control_rect.center();
        let actions = click_reorder_control(&ctx, &mut editor, &document, &region, false, false, point);
        assert!(editor.settings.auto_reorder_before_generation);
        assert!(actions.iter().any(|action| matches!(action,
            UiAction::SetSettings(settings) if settings.auto_reorder_before_generation)));
        assert!(actions.iter().all(|action| matches!(action, UiAction::SetSettings(_))));
        assert_eq!(document.nodes.len(), 1);
        assert!(!document.settings.auto_reorder_before_generation);

        let point = editor.manual_reorder_rect.center();
        let actions = click_reorder_control(&ctx, &mut editor, &document, &region, false, false, point);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], UiAction::ReorderByContribution));
        assert_eq!(document.nodes.len(), 1);
    }

    #[test]
    fn manual_reorder_is_available_on_completed_canvas_without_loaded_model() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = Document::new();
        let region = DrawingRegion::complete();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        let point = editor.manual_reorder_rect.center();
        let actions = click_reorder_control(&ctx, &mut editor, &document, &region, false, false, point);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], UiAction::ReorderByContribution));
    }

    #[test]
    fn running_keeps_manual_reordering_available_but_disables_automatic_settings() {
        for (running, loading) in [(true, false), (false, true)] {
            let ctx = egui::Context::default();
            let mut editor = EditorUi::new(&ctx);
            let document = Document::new();
            let region = DrawingRegion::default();
            render_frame_flags(&ctx, &mut editor, &document, &region, running, loading, vec![]);
            let point = editor.auto_reorder_control_rect.center();
            assert!(click_reorder_control(&ctx, &mut editor, &document, &region, running, loading, point).is_empty());
            let point = editor.manual_reorder_rect.center();
            let actions = click_reorder_control(&ctx, &mut editor, &document, &region, running, loading, point);
            if loading { assert!(actions.is_empty()); }
            else { assert!(matches!(actions.as_slice(), [UiAction::FreezeGenerationView, UiAction::ReorderByContribution])); }
            assert!(!editor.settings.auto_reorder_before_generation);
        }
    }

    #[test]
    fn paired_canvas_rectangles_are_disjoint_and_two_to_one() {
        let area = Rect::from_min_size(Pos2::new(92.0, 70.0), Vec2::new(1038.0, 816.0));
        let (editable, indicator) = split_canvas_pair(area);
        assert!(!editable.intersects(indicator));
        assert!(editable.right() < indicator.left());
        assert!((editable.width() / indicator.width() - 2.0).abs() < 0.001);
        assert_eq!(editable.top(), indicator.top());
        assert_eq!(indicator.right(), area.right());
    }

    #[test]
    fn first_run_without_weights_still_allows_native_canvas_editing() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = Document::new();
        let region = DrawingRegion::default();
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        assert!(editor.model_path.is_empty());
        editor.canvas.tool = Tool::Rectangle;
        let start = editor.edit_canvas_rect.center() - Vec2::new(30.0, 20.0);
        let end = start + Vec2::new(65.0, 55.0);
        let press = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut actions = render_frame(
            &ctx,
            &mut editor,
            &document,
            &region,
            vec![egui::Event::PointerMoved(start), press(start, true)],
        );
        actions.extend(render_frame(
            &ctx,
            &mut editor,
            &document,
            &region,
            vec![egui::Event::PointerMoved(end)],
        ));
        actions.extend(render_frame(
            &ctx,
            &mut editor,
            &document,
            &region,
            vec![press(end, false)],
        ));
        assert_eq!(
            actions
                .iter()
                .filter(|action| matches!(action, UiAction::CommitCanvas { .. }))
                .count(),
            1
        );
        assert!(!actions.iter().any(|action| matches!(
            action,
            UiAction::LoadModel { .. } | UiAction::Generate { .. }
        )));
    }

    #[test]
    fn region_indicator_click_drag_and_scroll_never_edit_main_canvas() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let mut document = Document::new();
        document
            .commit_canvas(
                "已有作品",
                OperationSource::User,
                vec![
                    Primitive::background([230, 240, 250]),
                    Primitive::new(ShapeKind::Ellipse, 200.0, 200.0, 80.0, 50.0, [220, 120, 80]),
                ],
            )
            .unwrap();
        let region = DrawingRegion::active(Stage::Coarse, 5, 0, 0, 0);
        render_frame(&ctx, &mut editor, &document, &region, vec![]);
        editor.canvas.tool = Tool::Rectangle;
        editor.canvas.selection = vec![document.primitives[1].id];
        let old_selection = editor.canvas.selection.clone();
        let old_zoom = editor.canvas.zoom;
        let original = document.primitives.clone();
        assert!(editor.edit_canvas_rect.right() < editor.region_indicator_rect.left());
        assert!(editor.region_indicator_rect.right() <= 1131.0);
        let start = editor.region_indicator_rect.center();
        let end = start + Vec2::new(35.0, 30.0);
        let press = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let mut all_actions = render_frame(
            &ctx,
            &mut editor,
            &document,
            &region,
            vec![egui::Event::PointerMoved(start), press(start, true)],
        );
        all_actions.extend(render_frame(
            &ctx,
            &mut editor,
            &document,
            &region,
            vec![egui::Event::PointerMoved(end)],
        ));
        all_actions.extend(render_frame(
            &ctx,
            &mut editor,
            &document,
            &region,
            vec![press(end, false)],
        ));
        all_actions.extend(render_frame(
            &ctx,
            &mut editor,
            &document,
            &region,
            vec![egui::Event::Scroll(Vec2::new(0.0, 90.0))],
        ));
        assert!(!all_actions
            .iter()
            .any(|action| matches!(action, UiAction::CommitCanvas { .. })));
        assert_eq!(editor.canvas.selection, old_selection);
        assert_eq!(editor.canvas.zoom, old_zoom);
        assert_eq!(document.primitives, original);
    }

    #[test]
    fn inspector_primary_buttons_remain_visible_without_scrolling_in_supported_window_sizes() {
        for size in [Vec2::new(1440.0, 920.0), Vec2::new(1100.0, 720.0)] {
            for (model_loaded, running) in [(false, false), (true, false), (true, true)] {
                let ctx = egui::Context::default();
                let mut editor = EditorUi::new(&ctx);
                let document = Document::new();
                let region = DrawingRegion::default();
                let nodes = vec![NodeView { id: 0, parent: None, title: "初始画布".into(),
                    source: "初始".into(), primitive_count: 0 }];
                let model = UiModel {
                    document: &document, drawing_region: &region, primitives: &document.primitives,
            stage_by_id: None,
                    prompt: &document.prompt, settings: &document.settings,
                    current_node: 0, nodes: &nodes, running, loading: false, compiling: false, stopping: false,
                    model_loaded, model_status: "就绪", model_path: "",
                    tokenizer_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets/caption_spm_1500.model"),
                    error: None, status: "就绪",
                };
                for _ in 0..2 {
                    let _ = ctx.run(egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)), ..Default::default()
                    }, |ctx| { assert!(editor.render(ctx, &model).is_empty()); });
                }
                let gap = editor.manual_reorder_rect.top() - editor.auto_reorder_control_rect.bottom();
                assert!((0.0..80.0).contains(&gap),
                    "manual reorder must occupy one nearby row, size={size:?}, gap={gap}");
                let primary = if model_loaded { editor.generation_button_rect } else { editor.open_settings_rect };
                assert!(primary.is_positive());
                assert!(primary.bottom() < size.y - 20.0,
                    "initial primary button must remain above the status bar, size={size:?}, loaded={model_loaded}, running={running}, button={primary:?}");
                assert!(primary.top() > editor.manual_reorder_rect.bottom());
            }
        }
    }

    #[test]
    fn acceleration_waiting_modal_blocks_input_until_preparation_is_released() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let document = Document::new();
        let region = DrawingRegion::default();
        let nodes = vec![NodeView { id: 0, parent: None, title: "初始画布".into(),
            source: "初始".into(), primitive_count: 0 }];
        let mut model = UiModel {
            document: &document, drawing_region: &region, primitives: &document.primitives,
            stage_by_id: None,
            prompt: &document.prompt, settings: &document.settings, current_node: 0, nodes: &nodes,
            running: false, loading: false, compiling: false, stopping: false, model_loaded: true,
            model_status: "mock GPU已加载", model_path: "",
            tokenizer_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets/caption_spm_1500.model"),
            error: None, status: "正在编译GPU采样算子…",
        };
        let render = |editor: &mut EditorUi, model: &UiModel<'_>, events| {
            let mut actions = Vec::new();
            let _ = ctx.run(egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1100.0, 720.0))),
                events, ..Default::default()
            }, |ctx| { actions = editor.render(ctx, model); });
            actions
        };
        render(&mut editor, &model, vec![]);
        let generate = editor.generation_button_rect.center();
        model.compiling = true;
        model.loading = true;
        for _ in 0..2 { assert!(render(&mut editor, &model, vec![]).is_empty()); }
        assert!(editor.preparation_modal_rect.is_positive());
        assert!(editor.preparation_modal_rect.contains(Pos2::new(550.0, 360.0)));
        for pressed in [true, false] {
            assert!(render(&mut editor, &model,
                vec![egui::Event::PointerMoved(generate), pointer_event(generate, pressed)]).is_empty());
        }
        assert!(render(&mut editor, &model, vec![egui::Event::Key {
            key: egui::Key::Escape, physical_key: None, pressed: true,
            repeat: false, modifiers: Default::default(),
        }]).is_empty());
        assert!(editor.preparation_modal_rect.is_positive(), "waiting has no Escape or close affordance");
        editor.canvas.tool = Tool::Rectangle;
        let start = editor.edit_canvas_rect.center();
        let end = start + Vec2::new(40.0, 25.0);
        for events in [vec![egui::Event::PointerMoved(start), pointer_event(start, true)],
            vec![egui::Event::PointerMoved(end)], vec![pointer_event(end, false)]] {
            assert!(render(&mut editor, &model, events).is_empty(), "modal must prevent background edits");
        }
        // AccelerationPrepared alone is not modeled as release; main waits for Loaded.
        model.status = "加速准备完成，正在确认模型就绪…";
        model.loading = false;
        assert!(render(&mut editor, &model, vec![]).is_empty());
        assert!(editor.preparation_modal_rect.is_positive());
        for pressed in [true, false] {
            assert!(render(&mut editor, &model,
                vec![egui::Event::PointerMoved(generate), pointer_event(generate, pressed)]).is_empty(),
                "compilation waiting must block clicks independently of the loading flag");
        }
        model.compiling = false;
        model.loading = false;
        editor.canvas.tool = Tool::Select;
        for _ in 0..2 { render(&mut editor, &model, vec![]); }
        assert!(!editor.preparation_modal_rect.is_positive());
        let generate = editor.generation_button_rect.center();
        let mut actions = Vec::new();
        for pressed in [true, false] {
            actions.extend(render(&mut editor, &model,
                vec![egui::Event::PointerMoved(generate), pointer_event(generate, pressed)]));
        }
        assert!(matches!(actions.as_slice(), [UiAction::Generate { .. }]), "Loaded must restore the normal generation button");
        assert_eq!(document.nodes.len(), 1);
    }

    #[test]
    fn long_model_paths_do_not_expand_inspector_past_window_edge() {
        let ctx = egui::Context::default();
        let mut editor = EditorUi::new(&ctx);
        let settings = Settings::default();
        let document = Document::new();
        let drawing_region = DrawingRegion::default();
        let paths = format!(
            "C:/models/{}/model.pt",
            "very_long_local_directory/".repeat(30)
        );
        let nodes = vec![NodeView {
            id: 0,
            parent: None,
            title: "初始画布".into(),
            source: "初始".into(),
            primitive_count: 0,
        }];
        let model = UiModel {
            document: &document,
            drawing_region: &drawing_region,
            primitives: &[],
            stage_by_id: None,
            prompt: "这里是一段用于验证自动换行的提示词，不应把右侧面板撑出窗口。",
            settings: &settings,
            current_node: 0,
            nodes: &nodes,
            running: false,
            loading: false,
            compiling: false,
            stopping: false,
            model_loaded: false,
            model_status: "模型未加载",
            model_path: &paths,
            tokenizer_path: &paths,
            error: None,
            status: "就绪",
        };
        for _ in 0..3 {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 920.0))),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                editor.render(ctx, &model);
            });
        }
        let panel =
            egui::containers::panel::PanelState::load(&ctx, egui::Id::new("inspector")).unwrap();
        assert!(
            panel.rect.right() <= 1440.5,
            "inspector overflows screen: {:?}",
            panel.rect
        );
        assert!(
            panel.rect.width() <= 310.5,
            "inspector ignores fixed width: {:?}",
            panel.rect
        );
    }
}
    #[test]
    fn busy_indicator_does_not_override_the_active_task_frame_deadline() {
        let ctx = egui::Context::default();
        let mut last = None;
        for frame in 0..12 {
            let output = ctx.run(egui::RawInput { time: Some(frame as f64 * 0.1), ..Default::default() }, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| { paint_busy_indicator(ui); });
                ctx.request_repaint_after(std::time::Duration::from_millis(40));
            });
            last = Some(output.viewport_output[&egui::ViewportId::ROOT].repaint_delay);
        }
        assert!(last.unwrap() >= std::time::Duration::from_millis(40));
    }
