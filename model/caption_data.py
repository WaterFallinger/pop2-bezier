"""Build and read a compact, random-access text sidecar for frontal training."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from config import TrainConfig


def variant_count(config):
    count = int(getattr(config, "CAPTION_VARIANTS", 20))
    if count < 1:
        raise ValueError("CAPTION_VARIANTS must be positive")
    return count


def selected_rows(path, expected_count=None):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError("Caption selection is empty")
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(f"Expected {expected_count} caption images, found {len(rows)}")
    names = []
    for priority, row in enumerate(rows):
        if int(row["priority"]) != priority:
            raise ValueError(f"提示词 priority 在 {priority} 处不连续。")
        names.append(Path(row["filename"]).stem)
    if len(set(names)) != len(names):
        raise ValueError("提示词样本名称不唯一。")
    return rows, names


def build_caption_cache(config=None):
    import sentencepiece as spm

    config = config or TrainConfig()
    rows, _ = selected_rows(config.CAPTION_SELECTION_PATH, getattr(config, "CAPTION_EXPECTED_IMAGES", None))
    variants_count = variant_count(config)
    root = Path(config.CAPTION_CACHE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    token_path = root / "captions.tokens.bin"
    offset_path = root / "captions.offsets.npy"
    manifest_path = root / "manifest.json"
    temporary_tokens = root / f".captions.tokens.bin.tmp.{os.getpid()}"
    temporary_offsets = root / f".captions.offsets.npy.tmp.{os.getpid()}"
    temporary_manifest = root / f".manifest.json.tmp.{os.getpid()}"
    sp = spm.SentencePieceProcessor(model_file=config.CAPTION_TOKENIZER_PATH)
    if sp.vocab_size() != config.TEXT_VOCAB_SIZE:
        raise ValueError("文字分词器词表大小与模型配置不一致。")
    if (sp.pad_id(), sp.bos_id(), sp.eos_id()) != (0, 2, 3):
        raise ValueError("文字分词器的 PAD/BOS/EOS ID 与训练协议不一致。")
    offsets = np.empty((len(rows), variants_count, 2), dtype=np.int64)
    position = 0
    try:
        with temporary_tokens.open("wb") as token_stream:
            for priority, row in enumerate(rows):
                image_id = int(row["image_id"])
                source = Path(config.CAPTION_PROMPTS_DIR) / f"{(image_id - 1) // 1000:03d}" / f"{image_id:06d}.json"
                with source.open("r", encoding="utf-8") as stream:
                    record = json.load(stream)
                if record.get("filename") != row["filename"] or len(record.get("prompts", [])) != variants_count:
                    raise ValueError(f"提示词图片名或版本数不一致：{source}")
                variants = sorted(record["prompts"], key=lambda item: int(item["variant"]))
                if [int(item["variant"]) for item in variants] != list(range(1, variants_count + 1)):
                    raise ValueError(f"提示词版本号不完整：{source}")
                for index, item in enumerate(variants):
                    value = item.get("text", "")
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"提示词为空：{source} version={index + 1}")
                    ids = [sp.bos_id(), *sp.encode(value), sp.eos_id()]
                    if len(ids) > config.TEXT_MAX_TOKENS:
                        raise ValueError(f"提示词超过 {config.TEXT_MAX_TOKENS} token：{source} version={index + 1}")
                    array = np.asarray(ids, dtype="<u2")
                    offsets[priority, index] = (position, len(array))
                    array.tofile(token_stream)
                    position += len(array)
                if (priority + 1) % 10_000 == 0:
                    print(f"已编码 {priority + 1}/{len(rows)} 张图片", flush=True)
            token_stream.flush()
            os.fsync(token_stream.fileno())
        with temporary_offsets.open("wb") as stream:
            np.save(stream, offsets)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary_manifest.open("w", encoding="utf-8") as stream:
            json.dump({
                "schema_version": 1,
                "image_count": len(rows),
                "variants_per_image": variants_count,
                "token_count": position,
                "vocab_size": sp.vocab_size(),
                "max_text_tokens": config.TEXT_MAX_TOKENS,
            }, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_tokens, token_path)
        os.replace(temporary_offsets, offset_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        for path in (temporary_tokens, temporary_offsets, temporary_manifest):
            path.unlink(missing_ok=True)
    return manifest_path


def _mix64(value):
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


class CaptionStore:
    def __init__(self, config=None):
        self.config = config or TrainConfig()
        _, names = selected_rows(self.config.CAPTION_SELECTION_PATH, getattr(self.config, "CAPTION_EXPECTED_IMAGES", None))
        self.variants = variant_count(self.config)
        self.name_to_row = {name: index for index, name in enumerate(names)}
        root = Path(self.config.CAPTION_CACHE_DIR)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"文字索引尚未构建：运行 train_caption_cache.py 或 caption_data.py build；缺少 {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("image_count") != len(names) or manifest.get("vocab_size") != self.config.TEXT_VOCAB_SIZE or manifest.get("variants_per_image") != self.variants:
            raise ValueError("文字索引与当前提示词或模型词表不兼容。")
        self.offsets = np.load(root / "captions.offsets.npy", mmap_mode="r")
        self.tokens = np.memmap(root / "captions.tokens.bin", dtype="<u2", mode="r")
        if self.offsets.shape != (len(names), self.variants, 2) or len(self.tokens) != manifest["token_count"]:
            raise ValueError("文字索引文件形状或长度不正确。")

    def batch(self, names, epoch, seed):
        selected = []
        for name in names:
            row = self.name_to_row[name]
            random_bits = _mix64(int(seed) ^ (int(epoch) << 32) ^ row)
            if random_bits % 10_000 < round(self.config.TEXT_DROPOUT_PROBABILITY * 10_000):
                selected.append(np.asarray([2, 3], dtype=np.int64))  # Empty prompt still has EOS.
            else:
                variant = _mix64(random_bits) % self.variants
                offset, length = self.offsets[row, variant]
                selected.append(self.tokens[int(offset):int(offset + length)].astype(np.int64))
        width = max(map(len, selected))
        ids = np.zeros((len(selected), width), dtype=np.int64)
        mask = np.zeros((len(selected), width), dtype=np.bool_)
        for index, value in enumerate(selected):
            ids[index, :len(value)] = value
            mask[index, :len(value)] = True
        return ids, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    args = parser.parse_args()
    print(build_caption_cache())
