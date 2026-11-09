//! Prompt validation shared by the editor and inference worker.
//!
//! A word assembled from ordinary SentencePiece pieces is supported. A byte
//! fallback is not a learned text piece: this model has all 256 fallback bytes,
//! so testing only `unk_id` would silently accept every emoji and rare character.

use sentencepiece_rs::SentencePieceProcessor;
use std::{ops::Range, path::Path};

pub const MAX_TEXT_TOKENS: usize = 224;
const TEXT_VOCABULARY_SIZE: usize = 1500;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PromptValidation {
    /// Includes the model's text BOS and EOS, even for an empty prompt.
    pub token_ids: Vec<u32>,
    /// UTF-8 byte ranges in the original, unmodified editor string.
    pub unknown_spans: Vec<Range<usize>>,
    pub error: Option<String>,
}

impl PromptValidation {
    pub fn is_valid(&self) -> bool {
        self.error.is_none()
    }

    pub fn into_token_ids(self) -> Result<Vec<u32>, String> {
        match self.error {
            Some(error) => Err(error),
            None => Ok(self.token_ids),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PromptValidator {
    processor: SentencePieceProcessor,
}

impl PromptValidator {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let processor = SentencePieceProcessor::open(path)
            .map_err(|error| format!("无法加载提示词分词器：{error}"))?;
        Ok(Self { processor })
    }

    pub fn validate(&self, text: &str) -> PromptValidation {
        validate_with_processor(&self.processor, text)
    }
}

pub fn validate_with_processor(processor: &SentencePieceProcessor, text: &str) -> PromptValidation {
    let mut validation = PromptValidation {
        token_ids: vec![2],
        ..Default::default()
    };
    let result = inspect(processor, text, &mut validation);
    validation.token_ids.push(3);
    if let Err(error) = result {
        validation.error = Some(error);
    } else if !validation.unknown_spans.is_empty() {
        let mut examples = Vec::new();
        for span in &validation.unknown_spans {
            let unknown = &text[span.clone()];
            if !examples.contains(&unknown) {
                examples.push(unknown);
            }
            if examples.len() == 8 {
                break;
            }
        }
        validation.error = Some(format!(
            "提示词中有模型文字词表不支持的内容：{}。请修改红色标记后再生成。",
            examples.join("、")
        ));
    } else if validation.token_ids.len() > MAX_TEXT_TOKENS {
        validation.error = Some(format!(
            "提示词包含 {} 个 token，超过模型的 {} 上限；请缩短文字（不会静默截断）",
            validation.token_ids.len(),
            MAX_TEXT_TOKENS
        ));
    }
    validation
}

fn inspect(
    processor: &SentencePieceProcessor,
    text: &str,
    validation: &mut PromptValidation,
) -> Result<(), String> {
    if processor.model().vocab_size() != TEXT_VOCABULARY_SIZE
        || processor.bos_id() != Some(2)
        || processor.eos_id() != Some(3)
    {
        return Err("分词器与模型的 1500 文字词表不匹配".into());
    }
    if text.trim().is_empty() {
        return Ok(());
    }
    let ids = processor
        .encode_to_ids(text)
        .map_err(|error| format!("文字分词失败：{error}"))?;
    validation.token_ids.extend(ids.iter().map(|&id| id as u32));
    if ids.iter().any(|&id| id >= TEXT_VOCABULARY_SIZE) {
        return Err("分词器与模型的 1500 文字词表不匹配".into());
    }
    if !ids
        .iter()
        .any(|&id| processor.model().is_byte(id) || processor.model().is_unknown(id))
    {
        return Ok(());
    }

    // `encode` preserves a true UNK's normalized surface and exposes byte
    // fallback as <0xXX>. Reconstruct their exact spans in the normalized text.
    let normalized = processor
        .normalize(text)
        .map_err(|error| format!("文字规范化失败：{error}"))?;
    let pieces = processor
        .encode(text)
        .map_err(|error| format!("文字分词失败：{error}"))?;
    if pieces.len() != ids.len() {
        return Err("分词器返回了不一致的文字结果".into());
    }
    let mut cursor = 0;
    let mut normalized_unknown = Vec::new();
    for (&id, piece) in ids.iter().zip(&pieces) {
        let byte_piece = processor.model().is_byte(id);
        let length = if byte_piece { 1 } else { piece.len() };
        let end = cursor + length;
        if end > normalized.len() {
            return Err("无法定位提示词中的未支持文字".into());
        }
        if byte_piece || processor.model().is_unknown(id) {
            normalized_unknown.push(cursor..end);
        }
        cursor = end;
    }
    if cursor != normalized.len() {
        return Err("无法定位提示词中的未支持文字".into());
    }
    let origins = normalization_origins(processor, text)?;
    for unknown in normalized_unknown {
        for (normalized_span, source_span) in &origins {
            if normalized_span.start < unknown.end && unknown.start < normalized_span.end {
                let mut source = source_span.clone();
                // Collapsed whitespace can share an origin with the following
                // character; do not paint that whitespace as an unknown word.
                let content = &text[source.clone()];
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    continue;
                }
                source.start += content.len() - content.trim_start().len();
                source.end = source.start + trimmed.len();
                validation.unknown_spans.push(source);
            }
        }
    }
    validation.unknown_spans.sort_by_key(|range| range.start);
    let mut merged: Vec<Range<usize>> = Vec::new();
    for span in validation.unknown_spans.drain(..) {
        match merged.last_mut() {
            Some(previous) if span.start <= previous.end => {
                previous.end = previous.end.max(span.end);
            }
            _ => merged.push(span),
        }
    }
    validation.unknown_spans = merged;
    Ok(())
}

/// Align normalization edits back to source characters. Prefix normalization
/// also handles multi-character rules, e.g. a combining mark replacing the
/// preceding character. The work is done only when actual fallback is present;
/// ordinary valid edits need a single tokenization and no alignment pass.
fn normalization_origins(
    processor: &SentencePieceProcessor,
    text: &str,
) -> Result<Vec<(Range<usize>, Range<usize>)>, String> {
    let mut previous = String::new();
    let mut origins: Vec<(Range<usize>, Range<usize>)> = Vec::new();
    let mut pending_start = 0;
    for (source_start, character) in text.char_indices() {
        let source_end = source_start + character.len_utf8();
        let current = processor
            .normalize(&text[..source_end])
            .map_err(|error| format!("文字规范化失败：{error}"))?;
        let common = previous
            .chars()
            .zip(current.chars())
            .take_while(|(left, right)| left == right)
            .map(|(character, _)| character.len_utf8())
            .sum::<usize>();
        if current == previous {
            continue;
        }
        let replaced_start = origins
            .iter()
            .filter(|(normalized, _)| normalized.end > common)
            .map(|(_, source)| source.start)
            .min()
            .unwrap_or(pending_start);
        let source_origin = replaced_start.min(pending_start)..source_end;
        origins.retain(|(normalized, _)| normalized.end <= common);
        for (offset, character) in current[common..].char_indices() {
            let start = common + offset;
            origins.push((start..start + character.len_utf8(), source_origin.clone()));
        }
        previous = current;
        pending_start = source_end;
    }
    Ok(origins)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> SentencePieceProcessor {
        SentencePieceProcessor::from_serialized_model(include_bytes!(
            "../assets/caption_spm_1500.model"
        ))
        .unwrap()
    }

    #[test]
    fn known_prompt_preserves_training_tokenization_and_punctuation() {
        let processor = processor();
        let prompt =
            "金发蓝眼的白皙圆脸女性以明亮动画风呈现，张嘴露舌，粉色背景可见，取景包含头部和肩颈。";
        let checked = validate_with_processor(&processor, prompt);
        assert!(checked.is_valid(), "{:?}", checked.error);
        let mut expected = vec![2];
        expected.extend(
            processor
                .encode_to_ids(prompt)
                .unwrap()
                .into_iter()
                .map(|id| id as u32),
        );
        expected.push(3);
        assert_eq!(checked.token_ids, expected);
    }

    #[test]
    fn empty_and_whitespace_are_valid_unconditional_prompts() {
        for prompt in ["", " \t\n　"] {
            let checked = validate_with_processor(&processor(), prompt);
            assert!(checked.is_valid());
            assert_eq!(checked.token_ids, vec![2, 3]);
        }
    }

    #[test]
    fn byte_fallback_and_rare_chars_are_flagged_at_original_utf8_spans() {
        let prompt = "金发 🦄蓝眼龘。";
        let checked = validate_with_processor(&processor(), prompt);
        assert!(!checked.is_valid());
        assert_eq!(
            checked
                .unknown_spans
                .iter()
                .map(|span| &prompt[span.clone()])
                .collect::<Vec<_>>(),
            vec!["🦄", "龘"]
        );
        assert!(!checked.token_ids.contains(&(processor().unk_id() as u32)));
    }

    #[test]
    fn supported_subwords_do_not_require_a_whole_word_vocabulary_entry() {
        let processor = processor();
        let prompt = "金发蓝眼";
        assert_eq!(processor.piece_to_id(prompt), processor.unk_id());
        assert!(validate_with_processor(&processor, prompt).is_valid());
    }

    #[test]
    fn fullwidth_normalization_keeps_original_unknown_offsets() {
        let prompt = "　金发，🦄。　";
        let checked = validate_with_processor(&processor(), prompt);
        assert_eq!(
            checked
                .unknown_spans
                .iter()
                .map(|span| &prompt[span.clone()])
                .collect::<Vec<_>>(),
            vec!["🦄"]
        );
    }

    #[test]
    fn combining_normalization_includes_both_source_characters() {
        let processor = processor();
        let prompt = "金发 e\u{301}🦄";
        let normalized = processor.normalize(prompt).unwrap();
        let origins = normalization_origins(&processor, prompt).unwrap();
        assert_eq!(origins.last().unwrap().0.end, normalized.len());
        let emoji = origins.last().unwrap();
        assert_eq!(&prompt[emoji.1.clone()], "🦄");
        let accented = origins
            .iter()
            .find(|(range, _)| &normalized[range.clone()] == "é")
            .unwrap();
        assert_eq!(prompt[accented.1.clone()].trim(), "e\u{301}");
    }

    #[test]
    fn overlong_text_is_rejected_without_silent_truncation() {
        let checked = validate_with_processor(&processor(), &"金发蓝眼".repeat(225));
        assert!(!checked.is_valid());
        assert!(checked.unknown_spans.is_empty());
        assert!(checked.token_ids.len() > MAX_TEXT_TOKENS);
        assert!(checked.error.unwrap().contains("224"));
    }
}
