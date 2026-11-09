"""Run the official POP2 infer.py with args from a UTF-8 JSON file.

Why: passing Chinese prompts through the Windows shell risks ANSI/UTF-8 mixups.
Keeping argv in a JSON file makes the prompt bytes unambiguous.

The workspace root is found by walking up until a directory containing
primitive-operation-painter/ is located, so this script can live in _pop2/tools/.

Usage: python _pop2/tools/run_infer.py <args.json>
"""
import json
import sys
from pathlib import Path


def find_root(start):
    current = Path(start).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "primitive-operation-painter").is_dir():
            return candidate
    raise SystemExit("workspace root not found (need a dir containing primitive-operation-painter/)")


ROOT = find_root(Path(__file__).parent)
MODEL_DIR = ROOT / "primitive-operation-painter" / "model"

if len(sys.argv) != 2:
    sys.exit("usage: python run_infer.py <args.json>")

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    args = json.load(fh)
if not isinstance(args, list):
    sys.exit("args.json must be a JSON array of CLI arguments")

sys.path.insert(0, str(MODEL_DIR))
sys.argv = ["infer.py"] + [str(a) for a in args]

import infer  # noqa: E402

infer.main()
