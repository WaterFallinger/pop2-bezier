"""Fetch a matching MSYS2 mingw-w64 GCC toolchain (gcc + crt/headers/libs)."""

import pathlib
import re
import subprocess
import urllib.request

BASE = "https://mirrors.tuna.tsinghua.edu.cn/msys2/mingw/mingw64/"
DEST = pathlib.Path(r"D:\DS_workspaceNew\DS_Studynew\.toolchain\msys2")
TMP = pathlib.Path(r"D:\DS_workspaceNew\DS_Studynew\.toolchain\msys2_pkgs")
DEST.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

PREFIXES = [
    "mingw-w64-x86_64-gcc-",
    "mingw-w64-x86_64-gcc-libs-",
    "mingw-w64-x86_64-crt-",
    "mingw-w64-x86_64-headers-",
    "mingw-w64-x86_64-mpfr-",
    "mingw-w64-x86_64-gmp-",
    "mingw-w64-x86_64-mpc-",
    "mingw-w64-x86_64-isl-",
    "mingw-w64-x86_64-windows-default-manifest-",
]


def version_key(name):
    return [int(n) for n in re.findall(r"\d+", name)]


def main():
    html = urllib.request.urlopen(BASE, timeout=60).read().decode("utf-8", "ignore")
    links = re.findall('href="([^"]+\.pkg\.tar\.zst)"', html)
    for prefix in PREFIXES:
        cands = [
            l for l in links
            if l.startswith(prefix) and "arm" not in l and "aarch64" not in l
        ]
        if not cands:
            print("MISSING", prefix)
            continue
        link = max(cands, key=version_key)
        out = TMP / link
        if not out.is_file():
            blob = urllib.request.urlopen(BASE + link, timeout=240).read()
            out.write_bytes(blob)
            print(f"downloaded {link} ({len(blob)//1024} KB)")
        res = subprocess.run(["tar", "-xf", str(out), "-C", str(DEST)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            print("extract failed", link, res.stderr[:200])
    gcc = DEST / "mingw64" / "bin" / "gcc.exe"
    print("gcc exists:", gcc.is_file())
    if gcc.is_file():
        res = subprocess.run([str(gcc), "--version"], capture_output=True, text=True)
        print(res.stdout.splitlines()[:1])


if __name__ == "__main__":
    main()
