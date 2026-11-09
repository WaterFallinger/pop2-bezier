use crate::inference::prompt_validation::{PromptValidation, PromptValidator};
use eframe::egui::{self, Color32, FontId, Stroke, TextFormat, Vec2};
use egui::text::LayoutJob;

const UNKNOWN: Color32 = Color32::from_rgb(204, 47, 62);

pub(super) fn generation_button(ui: &mut egui::Ui, label: &str, error: Option<&str>) -> egui::Response {
    let valid = error.is_none();
    let button = egui::Button::new(egui::RichText::new(label).strong().color(
        if valid { Color32::WHITE } else { super::MUTED }
    )).fill(if valid { super::BLUE } else { Color32::from_rgb(230, 233, 239) })
        .stroke(Stroke::NONE).rounding(7.0)
        .min_size(Vec2::new(ui.available_width(), 42.0));
    let response = ui.add_enabled(valid, button);
    match error {
        Some(error) => response.on_disabled_hover_text(error),
        None => response,
    }
}

#[derive(Default)]
pub(super) struct PromptEditor {
    composing: bool,
    validation: ValidationCache,
}

#[derive(Default)]
struct ValidationCache {
    path: Option<String>,
    validator: Option<Result<PromptValidator, String>>,
    checked: Option<(String, PromptValidation)>,
}

impl ValidationCache {
    fn check(&mut self, text: &str, path: &str) -> PromptValidation {
        if self.path.as_deref() != Some(path) {
            self.path = Some(path.to_owned());
            self.validator = Some(PromptValidator::open(path));
            self.checked = None;
        }
        if let Some((previous, result)) = &self.checked {
            if previous == text {
                return result.clone();
            }
        }
        let result = match self.validator.as_ref().expect("validator initialized") {
            Ok(validator) => validator.validate(text),
            Err(error) => PromptValidation { error: Some(error.clone()), ..Default::default() },
        };
        self.checked = Some((text.to_owned(), result.clone()));
        result
    }
}

impl PromptEditor {
    pub(super) fn show(
        &mut self,
        ui: &mut egui::Ui,
        text: &mut String,
        tokenizer: &str,
        width: f32,
    ) -> (egui::Response, PromptValidation) {
        let id = egui::Id::new("creation_prompt_text");
        let focused = ui.is_enabled() && ui.memory(|memory| memory.has_focus(id));
        if focused {
            ui.input_mut(|input| repair_standalone_ime_commits(&mut input.events, &mut self.composing));
        } else {
            self.composing = false;
        }
        let normal_color = ui.visuals().text_color();
        let validation = &mut self.validation;
        let mut layouter = |ui: &egui::Ui, text: &str, width: f32| {
            let result = validation.check(text, tokenizer);
            ui.fonts(|fonts| fonts.layout_job(prompt_layout(text, &result, width, normal_color)))
        };
        let response = ui.add(
            egui::TextEdit::multiline(text)
                .id(id)
                .hint_text("描述你希望绘制的画面…\n也可以先在画布上添加图元，再让模型续画。")
                .desired_rows(5)
                .desired_width(width)
                .min_size(Vec2::ZERO)
                .margin(Vec2::splat(10.0))
                .layouter(&mut layouter),
        );
        let result = self.validation.check(text, tokenizer);
        (response, result)
    }
}

// egui 0.27 compares CompositionEnd with an old composition cursor even when
// the IME submitted punctuation without preedit. Ordinary Text events already
// have the correct selection/cursor semantics, including undo and replacement.
fn repair_standalone_ime_commits(events: &mut [egui::Event], composing: &mut bool) {
    for event in events {
        match event {
            egui::Event::CompositionStart | egui::Event::CompositionUpdate(_) => *composing = true,
            egui::Event::CompositionEnd(text) => {
                if !*composing && !text.is_empty() && text != "\r" && text != "\n" {
                    *event = egui::Event::Text(text.clone());
                }
                *composing = false;
            }
            _ => {}
        }
    }
}

fn prompt_layout(text: &str, validation: &PromptValidation, width: f32, normal: Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.wrap.max_width = width;
    let mut cursor = 0;
    for span in &validation.unknown_spans {
        if span.start < cursor || !text.is_char_boundary(span.start) || !text.is_char_boundary(span.end) {
            continue;
        }
        job.append(&text[cursor..span.start], 0.0, TextFormat {
            font_id: FontId::proportional(14.0), color: normal, ..Default::default()
        });
        job.append(&text[span.clone()], 0.0, TextFormat {
            font_id: FontId::proportional(14.0), color: UNKNOWN,
            underline: Stroke::new(1.0, UNKNOWN), ..Default::default()
        });
        cursor = span.end;
    }
    job.append(&text[cursor..], 0.0, TextFormat {
        font_id: FontId::proportional(14.0), color: normal, ..Default::default()
    });
    job
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Pos2, Rect};

    fn frame(ctx: &egui::Context, editor: &mut PromptEditor, text: &mut String, events: Vec<egui::Event>) -> PromptValidation {
        let mut result = PromptValidation::default();
        let _ = ctx.run(egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(600.0, 400.0))),
            events, ..Default::default()
        }, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                result = editor.show(ui, text, "assets/caption_spm_1500.model", 400.0).1;
            });
        });
        result
    }

    fn focused() -> (egui::Context, PromptEditor, String) {
        let ctx = egui::Context::default();
        let mut editor = PromptEditor::default();
        let mut text = String::new();
        frame(&ctx, &mut editor, &mut text, vec![]);
        ctx.memory_mut(|memory| memory.request_focus(egui::Id::new("creation_prompt_text")));
        (ctx, editor, text)
    }

    #[test]
    fn direct_ime_punctuation_after_text_is_inserted_and_preserved() {
        let (ctx, mut editor, mut text) = focused();
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::Text("金发蓝眼".into())]);
        for punctuation in ["，", "。", "！", "？", ",.!?;:()"] {
            frame(&ctx, &mut editor, &mut text, vec![egui::Event::CompositionEnd(punctuation.into())]);
        }
        assert_eq!(text, "金发蓝眼，。！？,.!?;:()");
    }

    #[test]
    fn punctuation_replaces_selection_at_nonzero_cursor() {
        let (ctx, mut editor, mut text) = focused();
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::Text("金发蓝眼".into())]);
        let id = egui::Id::new("creation_prompt_text");
        let mut state = egui::TextEdit::load_state(&ctx, id).unwrap();
        state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
            egui::text::CCursor::new(1), egui::text::CCursor::new(3),
        )));
        state.store(&ctx, id);
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::CompositionEnd("，".into())]);
        assert_eq!(text, "金，眼");
    }

    #[test]
    fn real_composition_across_frames_commits_once_without_pinyin() {
        let (ctx, mut editor, mut text) = focused();
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::Text("金发".into())]);
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::CompositionStart, egui::Event::CompositionUpdate("lan".into())]);
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::CompositionUpdate("蓝".into())]);
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::CompositionEnd("蓝".into())]);
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::CompositionEnd("，".into())]);
        assert_eq!(text, "金发蓝，");
    }

    #[test]
    fn unfocused_prompt_does_not_rewrite_other_inputs() {
        let (ctx, mut editor, mut text) = focused();
        ctx.memory_mut(|memory| memory.surrender_focus(egui::Id::new("creation_prompt_text")));
        frame(&ctx, &mut editor, &mut text, vec![egui::Event::CompositionEnd("，".into())]);
        assert_eq!(text, "");
        ctx.input(|input| assert!(matches!(&input.events[0], egui::Event::CompositionEnd(s) if s == "，")));
    }

    #[test]
    fn actual_unknown_utf8_is_red_and_correction_clears_it() {
        let (ctx, mut editor, mut text) = focused();
        let invalid = frame(&ctx, &mut editor, &mut text, vec![egui::Event::Text("金发🦄蓝眼".into())]);
        assert!(!invalid.is_valid());
        let layout = prompt_layout(&text, &invalid, 400.0, Color32::BLACK);
        let red: Vec<_> = layout.sections.iter().filter(|s| s.format.color == UNKNOWN).map(|s| &layout.text[s.byte_range.clone()]).collect();
        assert_eq!(red, vec!["🦄"]);
        text = "金发蓝眼，粉色背景。".into();
        assert!(frame(&ctx, &mut editor, &mut text, vec![]).is_valid());
    }

    #[test]
    fn generate_button_disables_clicks_and_reenables_after_correction() {
        let ctx = egui::Context::default();
        let mut rect = Rect::NOTHING;
        for error in [Some("提示词中有模型文字词表不支持的内容：🦄"), None] {
            let mut enabled = false;
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let response = generation_button(ui, "开始生成", error);
                    enabled = response.enabled();
                    rect = response.rect;
                });
            });
            assert_eq!(enabled, error.is_none());
            let pos = rect.center();
            let mut clicked = false;
            for pressed in [true, false] {
                let _ = ctx.run(egui::RawInput {
                    events: vec![egui::Event::PointerMoved(pos), egui::Event::PointerButton {
                        pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default(),
                    }], ..Default::default()
                }, |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        clicked |= generation_button(ui, "开始生成", error).clicked();
                    });
                });
            }
            assert_eq!(clicked, error.is_none());
        }
    }

    #[test]
    fn completed_canvas_disables_generation_and_old_unfinished_state_reenables_it() {
        use crate::region_state::{DrawingRegion, Stage};
        let ctx = egui::Context::default();
        let complete = DrawingRegion::complete();
        let unfinished = DrawingRegion::active(Stage::Fine, 63, 4, 16, 63);
        for region in [&complete, &unfinished] {
            let reason = region.generation_block_reason();
            assert_eq!(reason.is_some(), region.completed);
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let response = generation_button(ui, "继续生成", reason);
                    assert_eq!(response.enabled(), !region.completed);
                });
            });
        }
        assert!(complete.generation_block_reason().unwrap().contains("已绘制完毕"));
        let awaiting = DrawingRegion::waiting(Stage::Fine, 16, 63);
        assert!(awaiting.generation_block_reason().is_none());
    }
}
