"""POP 2 checkpoint + tokenizer verification (no full model build)."""

import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def find_root(start):
    current = Path(start).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "primitive-operation-painter").is_dir():
            return candidate
    raise SystemExit("workspace root not found (need a dir containing primitive-operation-painter/)")


BASE = find_root(Path(__file__).parent)
POP2 = BASE / "weights" / "primitive-operation-painter-2-1b"

import torch  # noqa: E402

ckpt = POP2 / "model.pt"
print("[1] size:", ckpt.stat().st_size, "bytes")

obj = None
for weights_only in (True, False):
    try:
        obj = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=weights_only)
        print(f"[2] torch.load OK (weights_only={weights_only})")
        break
    except Exception as exc:  # noqa: BLE001
        print(f"[2] load failed weights_only={weights_only}: {type(exc).__name__}")
if obj is None:
    sys.exit("checkpoint unreadable")

state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else None
if isinstance(state, dict):
    params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    dtypes = {str(v.dtype) for v in state.values() if hasattr(v, "dtype")}
    print("[3] tensors:", len(state), "| params: %.4f B" % (params / 1e9), "| dtypes:", dtypes)

try:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(POP2 / "caption_spm_1500.model"))
    pieces = sp.encode("金发蓝眼的白皙圆脸女性", out_type=str)
    print("[4] sentencepiece OK | vocab:", sp.get_piece_size(), "| encode:", pieces[:8])
except Exception as exc:  # noqa: BLE001
    print("[4] tokenizer check failed:", exc)

print("[done]")
