"""Vendor all Cargo.lock crates into ./vendor using Python's OpenSSL.

Why: cargo's libcurl uses Windows schannel, which cannot acquire credentials in
this sandbox, so `cargo build` cannot reach the registry. Python's urllib works,
so we download .crate files, extract them, and write .cargo-checksum.json so the
project builds fully offline via source replacement.
"""

import hashlib
import io
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "ex" / "data_tools" / "image_to_sequence"
LOCK = ROOT / "Cargo.lock"
VENDOR = ROOT / "vendor"
DL_BASE = "https://rsproxy.cn/api/v1/crates"

try:
    import tomllib
except ImportError:  # pragma: no cover
    sys.exit("Python 3.11+ required (tomllib)")


def fetch(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "cargo-vendor-script"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main():
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    packages = [p for p in lock.get("package", []) if p.get("source", "").startswith("registry+")]
    VENDOR.mkdir(parents=True, exist_ok=True)
    print(f"packages to vendor: {len(packages)}")

    ok = skipped = failed = 0
    t0 = time.time()
    for index, pkg in enumerate(packages, 1):
        name, version, checksum = pkg["name"], pkg["version"], pkg.get("checksum")
        dest = VENDOR / f"{name}-{version}"
        if (dest / ".cargo-checksum.json").is_file():
            skipped += 1
            continue
        url = f"{DL_BASE}/{name}/{version}/download"
        try:
            blob = fetch(url)
            actual = hashlib.sha256(blob).hexdigest()
            if checksum and actual != checksum:
                raise RuntimeError(f"checksum mismatch: {actual} != {checksum}")
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                tar.extractall(VENDOR)
            if not dest.is_dir():
                raise RuntimeError(f"unexpected archive layout, {dest} missing")
            files = {}
            for path in sorted(dest.rglob("*")):
                if path.is_file() and path.name != ".cargo-checksum.json":
                    rel = path.relative_to(dest).as_posix()
                    files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            (dest / ".cargo-checksum.json").write_text(
                json.dumps({"files": files, "package": checksum}, separators=(",", ":")),
                encoding="utf-8",
            )
            ok += 1
            if index % 10 == 0 or index == len(packages):
                print(f"  [{index}/{len(packages)}] {name}-{version} "
                      f"({ok} ok / {skipped} skip / {failed} fail, {time.time()-t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}-{version}: {type(exc).__name__}: {exc}")

    print(f"\nDONE ok={ok} skipped={skipped} failed={failed} in {time.time()-t0:.0f}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
