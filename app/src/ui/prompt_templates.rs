use eframe::egui::{self, Color32, RichText, Vec2};
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct TemplateFile {
    schema_version: u32,
    templates: Vec<PromptTemplate>,
}

#[derive(Deserialize)]
pub(super) struct PromptTemplate {
    pub(super) text: String,
}

pub(super) fn templates() -> Result<&'static [PromptTemplate], &'static str> {
    static TEMPLATES: OnceLock<Result<Vec<PromptTemplate>, String>> = OnceLock::new();
    TEMPLATES.get_or_init(|| {
        let file: TemplateFile = serde_json::from_str(include_str!("../../assets/prompt_templates.json"))
            .map_err(|error| format!("无法读取文本模板：{error}"))?;
        if file.schema_version != 1 || file.templates.len() != 100 {
            return Err("文本模板文件格式不匹配".into());
        }
        Ok(file.templates)
    }).as_deref().map_err(String::as_str)
}

#[derive(Default)]
pub(super) struct PromptTemplates {
    open: bool,
    search: String,
    focus_search: bool,
    #[cfg(test)]
    pub(super) visible_rows: Vec<(usize, egui::Rect)>,
}

impl PromptTemplates {
    pub(super) fn open(&mut self) {
        self.open = true;
        self.search.clear();
        self.focus_search = true;
    }

    pub(super) fn is_open(&self) -> bool {
        self.open
    }

    pub(super) fn show(&mut self, ctx: &egui::Context, can_select: bool) -> Option<String> {
        #[cfg(test)]
        self.visible_rows.clear();
        if !self.open {
            return None;
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.open = false;
            return None;
        }
        let mut open = self.open;
        let mut selected = None;
        egui::Window::new("文本模板")
            .id(egui::Id::new("prompt_templates_window"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(560.0)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                let response = ui.add(egui::TextEdit::singleline(&mut self.search)
                    .id_source("prompt_templates_search")
                    .hint_text("搜索文本或编号…")
                    .desired_width(f32::INFINITY));
                if self.focus_search {
                    response.request_focus();
                    self.focus_search = false;
                }
                ui.add_space(10.0);
                match templates() {
                    Ok(items) => {
                        let query = self.search.trim().to_lowercase();
                        let mut visible = 0;
                        egui::ScrollArea::vertical()
                            .id_source("prompt_templates_list")
                            .max_height(420.0)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.add_enabled_ui(can_select, |ui| {
                                    for (index, template) in items.iter().enumerate() {
                                        if !matches_search(index + 1, &template.text, &query) {
                                            continue;
                                        }
                                        visible += 1;
                                        ui.push_id(index, |ui| {
                                            let response = ui.add(
                                                egui::Button::new(RichText::new(format!("{:02}  {}", index + 1, template.text))
                                                    .size(13.0).color(super::INK))
                                                    .wrap(true)
                                                    .min_size(Vec2::new(ui.available_width(), 48.0))
                                                    .fill(Color32::from_rgb(246, 248, 252))
                                                    .stroke(egui::Stroke::NONE)
                                                    .rounding(6.0),
                                            );
                                            #[cfg(test)]
                                            if ui.is_rect_visible(response.rect) {
                                                self.visible_rows.push((index, response.rect));
                                            }
                                            if response.clicked() {
                                                selected = Some(template.text.clone());
                                            }
                                            ui.add_space(6.0);
                                        });
                                    }
                                });
                                if visible == 0 {
                                    ui.label(RichText::new("没有匹配的模板").color(super::MUTED));
                                }
                            });
                        if !can_select {
                            ui.add_space(8.0);
                            ui.label(RichText::new("请先等待模型加载或生成停止，再选择模板").size(12.0).color(super::MUTED));
                        }
                    }
                    Err(error) => { ui.colored_label(Color32::from_rgb(171, 66, 53), error); }
                }
            });
        self.open = open && selected.is_none();
        selected
    }
}

fn matches_search(number: usize, text: &str, query: &str) -> bool {
    query.is_empty() || text.to_lowercase().contains(query)
        || format!("{number:02}").contains(query)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::prompt_validation::{PromptValidator, MAX_TEXT_TOKENS};
    use std::collections::HashSet;

    #[test]
    fn bundled_templates_are_unique_and_valid_for_the_real_model_tokenizer() {
        let items = templates().expect("bundled template data must load");
        assert_eq!(items.len(), 100);
        let validator = PromptValidator::open(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/caption_spm_1500.model"))
            .expect("the actual 1500-piece tokenizer must load");
        let mut unique = HashSet::new();
        for (index, item) in items.iter().enumerate() {
            assert!(!item.text.trim().is_empty(), "template {} is empty", index + 1);
            assert!(unique.insert(item.text.trim()), "template {} is duplicated", index + 1);
            let validation = validator.validate(&item.text);
            assert!(validation.error.is_none(), "template {}: {:?}", index + 1, validation.error);
            assert!(validation.unknown_spans.is_empty());
            assert!(validation.token_ids.len() <= MAX_TEXT_TOKENS);
        }
    }

    #[test]
    fn search_supports_text_and_stable_display_number() {
        assert!(matches_search(32, "金发蓝眼的少女。", "蓝眼"));
        assert!(matches_search(32, "金发蓝眼的少女。", "32"));
        assert!(!matches_search(32, "金发蓝眼的少女。", "红发"));
        assert!(matches_search(1, "一条模板", "01"));
    }
}
