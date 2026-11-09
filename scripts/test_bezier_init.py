"""Unit test for load_bezier_initialization (7069 -> 8097 vocab migration)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ex" / "model"
sys.path.insert(0, str(MODEL_DIR))

import torch  # noqa: E402

from token_layout import TOKEN_LAYOUT as L  # noqa: E402
from model_gpt import GeometrizeGPT  # noqa: E402
from checkpoint_compat import load_bezier_initialization  # noqa: E402

D = 64
DECIMAL = 7054
PAD, MARK = 7054, 7057
EMPTY0, EMPTY_LAST = 7058, 7065
TRUE_END, INTERRUPT, BOS = 7066, 7067, 7068


def build_model():
    return GeometrizeGPT(
        vocab_size=L.vocab_size, d_model=D, n_layer=2, n_head=4, ffn_hidden=128,
        max_context_len=1024, max_block_positions=1024,
        text_vocab_size=1500, max_text_tokens=224, norm_eps=1e-6, rope_base=10000.0,
    )


def main():
    model = build_model()
    baseline = {k: v.detach().clone() for k, v in model.state_dict().items()}

    legacy = {k: v.detach().clone() for k, v in baseline.items()}
    # 每一行填一个可识别的常数，便于验证"哪一行去了哪里"。
    legacy["wte.weight"] = torch.arange(7069 * D, dtype=torch.float32).reshape(7069, D) / 1e6

    load_bezier_initialization(model, {"model_state_dict": legacy})
    got = model.state_dict()["wte.weight"]
    src = legacy["wte.weight"]

    checks = []
    checks.append(("attr rows copied", torch.equal(got[:DECIMAL], src[:DECIMAL])))
    checks.append(("edge rows zero", torch.count_nonzero(got[7054:8078]).item() == 0))
    checks.append(("special relocated", torch.equal(got[8078:8082], src[PAD:PAD + 4])))
    checks.append(("empty relocated", torch.equal(got[8082:8090], src[EMPTY0:EMPTY_LAST + 1])))
    checks.append(("edge empty zero", torch.count_nonzero(got[8090:8094]).item() == 0))
    checks.append(("true_end", torch.equal(got[8094], src[TRUE_END])))
    checks.append(("interrupt_end", torch.equal(got[8095], src[INTERRUPT])))
    checks.append(("bos", torch.equal(got[8096], src[BOS])))
    checks.append(("text_wte untouched", torch.equal(
        model.state_dict()["text_wte.weight"], baseline["text_wte.weight"])))
    checks.append(("blocks untouched", torch.equal(
        model.state_dict()["h.0.attn.c_attn.weight"], baseline["h.0.attn.c_attn.weight"])))

    ok = True
    for name, passed in checks:
        print(("  OK  " if passed else "  FAIL") + f"  {name}")
        ok = ok and passed

    print("\nvte shape:", tuple(got.shape), "| expected rows:", L.vocab_size)
    print("edge offset:", L.edge_offset, "| special:", L.special_offset,
          "| empty:", L.empty_offset, "| controls:", L.marker_end_token,
          L.interrupt_end_token, L.bos_token)
    if not ok:
        raise SystemExit("bezier init test FAILED")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
