"""Download MSYS2 mingw-w64 binutils (and DLL deps) via Python, extract with bsdtar."""

import io
import pathlib
import re
import subprocess
import sys
import urllib.request

BASE = "https://mirrors.tuna.tsinghua.edu.cn/msys2/mingw/mingw64/"
DEST = pathlib.Path(r"D:\DS_workspaceNew\DS_Studynew\.toolchain\msys2")
TMP = pathlib.Path(r"D:\DS_workspaceNew\DS_Studynew\.toolchain\msys2_pkgs")
DEST.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

PREFIXES = [
    "mingw-w64-x86_64-binutils-",
    "mingw-w64-x86_64-libwinpthread-",
    "mingw-w64-x86_64-libiconv-",
    "mingw-w64-x86_64-zlib-",
    "mingw-w64-x86_64-gettext-runtime-",
]


def version_key(name):
    nums = re.findall(r"\d+", name)
    return [int(n) for n in nums]


def main():
    html = urllib.request.urlopen(BASE, timeout=40).read().decode("utf-8", "ignore")
    links = re.findall('href="([^"]+\.pkg\.tar\.zst)"', html)
    print("index entries:", len(links))

    picks = []
    for prefix in PREFIXES:
        cands = [
            l for l in links
            if l.startswith(prefix)
            and "arm" not in l
            and "aarch64" not in l
            and ".sig" not in l
        ]
        if not cands:
            print("MISSING", prefix)
            continue
        picks.append(max(cands, key=version_key))
        print("pick", picks[-1])

    for link in picks:
        out = TMP / link
        if not out.is_file():
            blob = urllib.request.urlopen(BASE + link, timeout=180).read()
            out.write_bytes(blob)
            print(f"downloaded {link} ({len(blob)//1024} KB)")
        result = subprocess.run(["tar", "-xf", str(out), "-C", str(DEST)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print("extract failed", link, result.stderr[:200])
            sys.exit(1)
    print("extracted to", DEST)


if __name__ == "__main__":
    main()
