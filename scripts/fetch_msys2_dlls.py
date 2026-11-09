"""Fetch extra MSYS2 mingw-w64 runtime DLLs needed by binutils 2.47."""

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
    "mingw-w64-x86_64-zlib-1",
    "mingw-w64-x86_64-zstd-",
    "mingw-w64-x86_64-libtre-",
    "mingw-w64-x86_64-libgcc-",
]


def version_key(name):
    return [int(n) for n in re.findall(r"\d+", name)]


def main():
    html = urllib.request.urlopen(BASE, timeout=40).read().decode("utf-8", "ignore")
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
            blob = urllib.request.urlopen(BASE + link, timeout=180).read()
            out.write_bytes(blob)
            print(f"downloaded {link} ({len(blob)//1024} KB)")
        res = subprocess.run(["tar", "-xf", str(out), "-C", str(DEST)],
                             capture_output=True, text=True)
        print("extract", link, "rc=", res.returncode)
    print("DLLs now:", sorted(p.name for p in (DEST / "mingw64" / "bin").glob("*.dll")))


if __name__ == "__main__":
    main()
