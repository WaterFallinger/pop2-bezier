"""POP 1 checkpoint smoke test: strict load + one forward pass."""

import sys
import time
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


ROOT = find_root(Path(__file__).parent)
sys.path.insert(0, str(ROOT / "primitive-operation-painter"))

import torch  # noqa: E402
from pretrained import load_pretrained  # noqa: E402

t0 = time.time()
model, config, model_dir = load_pretrained(
    ROOT / "weights" / "primitive-operation-painter-weight",
    device="cpu",
)
print("[OK] load_pretrained 严格加载成功")
print("   模型目录 :", model_dir)
print("   参数量   : %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
print("   上下文   :", config["sequence"])
print("   画布     :", config["canvas"])
print("   词表     :", config["architecture"]["vocab_size"])
print("   tied wte :", model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr())
print("   耗时     :", round(time.time() - t0, 1), "s")

model.eval()
with torch.no_grad():
    vocab = config["architecture"]["vocab_size"]
    ids = torch.randint(0, vocab, (1, 32))
    logits = model(ids)
    out = logits[0] if isinstance(logits, (tuple, list)) else logits
    print("[OK] 前向冒烟通过, 输出形状:", tuple(out.shape))
