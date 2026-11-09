# Contributing

The research pipeline is `data_tools/` → `model/`. The Windows interactive application is `app/`. Keep model token/geometry protocols explicit when changing any of those layers.

## Small validation first

- Use the procedural synthetic example before processing a large dataset.
- Use `model/train.py --tiny` to check a new training pipeline. Do not treat its output as a quality result or load it as the released 1B model.
- Run `python -m unittest discover -s model/tests -v` after model, data preparation or checkpoint changes.
- Run the image converter's documented CPU tests; GPU tests are opt-in and require actual hardware.
- Use `app/scripts/build.ps1` and the App CPU test target for native code. Hardware-specific acceleration needs real-device numerical and execution checks in addition to compilation.

Changes to graph capture, KV reuse, quantization, background handling, region transitions and checkpoint retention should include a targeted regression. A device name, successful DLL load or build is not evidence of successful model execution.

## Data and artifacts

Do not commit weights, optimizer state, raw training images, bulk prompt corpora, generated caches, personal projects, tokens or private deployment files. Tiny upstream parser fixtures and the self-owned synthetic example are deliberate source test assets. Dependency lockfiles and source-required build scripts remain tracked.

Place distributable Windows installers on GitHub Releases. Update the source/model/application version descriptions separately, and keep third-party licenses with binary distributions. Do not silently replace a previously published attachment or relabel a development placeholder as a runnable installer.
