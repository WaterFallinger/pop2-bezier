"""Download the public POP 2 inference package without a Hugging Face account."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

REPO_ID = "doubixz/primitive-operation-painter-2-1b"
REVISION = "ca575269a461c90ecf630dcf70c4beaaf319b3a1"
FILES = {
    "model.pt": (2_008_472_552, "143f447aaecf9be75c6d14f3a3170002e94e49ff70a758f66d568d27ff65e9a0"),
    "caption_spm_1500.model": (261326, "8dc18fe4e13f2058807ed658bfed93461c296da9a7818f0224452b73a797497e"),
}

def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def verify(directory: Path) -> dict:
    result = {}
    for name, (size, digest) in FILES.items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"Missing file: {path}")
        if path.stat().st_size != size or sha256(path) != digest:
            raise RuntimeError(f"Size or SHA-256 mismatch: {path}")
        result[name] = {"bytes": size, "sha256": digest}
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "weights/pop2-1b-epoch60")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify an existing model package without network access")
    args = parser.parse_args()
    directory = args.output.expanduser().resolve()
    if not args.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise SystemExit("Install the downloader dependency: python -m pip install 'huggingface_hub>=0.34,<2'") from error
        directory.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, revision=REVISION,
                          local_dir=str(directory), token=False,
                          allow_patterns=[*FILES, "config.json", "checkpoint_metadata.json", "README.md", "LICENSE"])
    files = verify(directory)
    receipt = {"repository": REPO_ID, "revision": REVISION, "epoch": 60,
               "global_step": 10020, "verified": files}
    if not args.verify_only:
        (directory / "download_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "directory": str(directory), **receipt},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
