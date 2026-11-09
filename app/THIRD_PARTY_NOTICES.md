# Third-party components

The MIT license at the application/project root covers first-party Primitive Painter code. It does not relicense third-party packages, GPU drivers, runtime libraries, fonts or training data.

- **Candle 0.11.0**: vendored source under `vendor/candle-core/`; preserve its upstream `LICENSE`, Cargo licensing metadata and notices. The local manifest selects dynamic cudarc loading so optional CUDA compilation is not a mandatory Windows PE import.
- **egui / eframe / WGPU, serde, image, sentencepiece-rs, Candle dependencies and other Rust crates**: exact versions are recorded in Cargo lockfiles. Their upstream licenses apply; source and license files are available through the normal Cargo package distribution. Do not replace this notice with a claim that every dependency is MIT.
- **PyTorch / ATen**: the C++ bridge uses its public headers and links its ROCm libraries. PyTorch and included subcomponents have their own license/notice files; collect the files from the exact official distribution used to build or ship the bridge.
- **AMD ROCm/HIP** and **NVIDIA CUDA redistribution components**: fixed official sources are recorded in the runtime manifests. These libraries are not stored in this source tree. Their runtime/SDK licenses apply separately. File hashes do not replace license texts.
- **Microsoft Visual C++ runtime and Windows system fonts**: not included in this source tree. Packaging expects a licensed developer-provided VC redistributable directory. The application uses installed system fonts; do not copy proprietary fonts into the repository.
- **NSIS**: an external installer build tool; its executable and bundled plugins are not part of this source tree.

The small SentencePiece tokenizer and curated prompt-template asset are first-party project resources, not the full training dataset. Large model weights are distributed separately on Hugging Face with their model card and license.

## Collected materials and packaging distinction

This source candidate includes **21 license/notice text files** under `native/aten_hip/licenses/`, with per-file hashes, official URLs, package entries and source refs in `PROVENANCE.json`. These include exact installed official PyTorch wheel LICENSE/NOTICE and COMGR documentation, official ROCm `rocm-7.1.1` project licenses, and the OpenBLAS 0.3.30 upstream base/LAPACK-Netlib texts.

The project-level [Rust dependency inventory](../third_party/rust-dependencies.json) and accompanying text files cover the actual reachable Rust dependency set collected for this source candidate. Consult those records rather than this short overview for crate-by-crate license declarations.

The previously built **0.9.29 Setup binary is byte-for-byte unchanged** and predates these newly collected notice documents. The release folder supplies the collected documents alongside that binary. A **new installer built from this source candidate** copies the first-party, Candle and collected AMD/ATen texts into `licenses/`; pass `-AdditionalLicenseDirectory ../third_party` to include the project-level dependency material as well. Do not describe the old installer as already containing the new documents.

Specific remaining checks are recorded in `native/aten_hip/licenses/PROVENANCE.json`: identify the exact AMD fork source commit/modifications for `rocm-openblas.dll`; confirm distribution terms and transitive notice coverage for the two `0.1.dev0` ROCm SDK wheels whose metadata declares no License fields; compare AOTriton images, liblzma/libomp and MIOpen dependencies against the consolidated exact PyTorch notices and official ROCm source texts. Microsoft VC redistributables must retain the terms from the developer-supplied licensed redist. The collected files are useful provenance and attribution material, not a claim of exhaustive vendor-binary legal clearance.
