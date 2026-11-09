"""Build edge-on / edge-off variants of the generator and record their hashes."""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

PROJ = Path(r"D:\DS_workspaceNew\DS_Studynew\ex\data_tools\image_to_sequence")
MAIN = PROJ / "src" / "main.rs"
ROOT = Path(r"D:\DS_workspaceNew\DS_Studynew\.toolchain")
MSYS = r"D:\DS_workspaceNew\DS_Studynew\.toolchain\msys2\mingw64\bin"
BIN_DIR = ROOT / "bin"
BIN_DIR.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PATH"] = MSYS + ";" + env["PATH"]
env["RUSTUP_HOME"] = str(ROOT / "rustup")
env["CARGO_HOME"] = str(ROOT / "cargo")
env["CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER"] = "gcc"
env["CC_x86_64_pc_windows_gnu"] = "gcc"
env["AR_x86_64_pc_windows_gnu"] = "ar"
env["RUSTFLAGS"] = ""

CARGO = ["cargo", "+stable-x86_64-pc-windows-gnu"]


def set_weight(weight: int) -> None:
    text = MAIN.read_text(encoding="utf-8")
    new = re.sub(r"const EDGE_MUTATION_WEIGHT: u32 = \d+;",
                 f"const EDGE_MUTATION_WEIGHT: u32 = {weight};", text)
    if new == text:
        raise SystemExit("weight replacement did not change the file")
    MAIN.write_text(new, encoding="utf-8", newline="")


def build(weight: int, name: str) -> None:
    set_weight(weight)
    subprocess.run(CARGO + ["clean", "-p", "gpu_anime_vectorizer"],
                   cwd=PROJ, env=env, capture_output=True)
    res = subprocess.run(CARGO + ["build", "--locked", "--offline", "--release"],
                         cwd=PROJ, env=env, capture_output=True, text=True)
    blob = res.stdout + res.stderr
    if res.returncode != 0 or "Finished" not in blob:
        print(f"{name}: BUILD FAILED\n{blob[-900:]}")
        return
    src = PROJ / "target" / "release" / "gpu_anime_vectorizer.exe"
    dst = BIN_DIR / f"{name}.exe"
    shutil.copy2(src, dst)
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()[:12]
    print(f"{name}: weight={weight} hash={digest} size={dst.stat().st_size}")


if __name__ == "__main__":
    build(0, "edge-off")
    build(2, "edge-on")
    set_weight(2)
    print("restored EDGE_MUTATION_WEIGHT=2")
