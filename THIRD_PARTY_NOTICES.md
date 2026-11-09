# Third party notices

The root MIT license applies to first-party Primitive Operation Painter code. It does not relicense third-party source, fonts, GPU runtimes, build tools, model packages or training data.

## Included source and Rust dependencies

- The patched Candle core source is in `app/vendor/candle-core/`, with its upstream license and the project's linkage-only patch description.
- `third_party/rust-dependencies.json` records the actual Windows/CUDA App and data-tool dependency versions, declared SPDX licenses and accompanying text files. It covers 333 registry packages; target-specific dependencies for other builds remain governed by their own distributions.
- `third_party/rust_licenses/` contains original license/notices from published crates or their recorded upstream commits. Where a published crate explicitly declares an SPDX license but contains no standalone text, a canonical SPDX copy is supplied with `LICENSE-DECLARATION.json` explaining that source. No third-party package is asserted to be MIT merely because this project is MIT.
- Font terms include the Ubuntu Font License, SIL Open Font License, Hack notices and the emoji-icon-font MIT notice accompanying the egui/epaint default fonts. Proprietary Windows system fonts are not copied into this repository.

## Python and GPU libraries

Python dependencies are installed from their own distributions. Choose a PyTorch build appropriate for your hardware. PyTorch, NumPy, pandas, SentencePiece and other packages retain their own licenses.

AMD native bridge/runtime notices and source provenance are in `app/native/aten_hip/licenses/` and its README. The native runtime manifest pins official package URLs and individual file hashes. The collected notices describe exact package files where available and corresponding official versioned sources for other components; their provenance explicitly identifies remaining source-build uncertainty. Hashes establish file identity, not licensing scope.

NVIDIA runtime package sources and file hashes are in `app/installer/runtime-manifest.json`. Available CUDA redistribution terms are included under `third_party/nvidia_cuda/`; the official selected packages' own terms continue to apply.

Microsoft VC runtime redistributables and NSIS are external developer packaging prerequisites, not first-party source. They are not included as development binaries in this repository and are not relicensed under MIT.

## Installer attachment and new source builds

The local `distribution/PrimitivePainter-0.9.29-Setup.exe` is the previously tested 0.9.29 installer. It has not been silently rebuilt or relabeled during source curation. New source packaging scripts include the accompanying App notices collected for this source release. Keep this notice set with any binary distribution, and consult the component-specific source records when preparing a public release.

The model weight release has its own model card and license on Hugging Face. Original training images and the full prompt corpus are not redistributed here. The included synthetic examples were drawn by a small program for protocol testing.
