# Source bundle validation

This directory is a curated source candidate, not a copy of the development workspace. Generated test outputs, device logs, compiler caches and tiny training checkpoints are stored outside it by the maintainer.

## Data generator

- Nine CPU Rust tests passed; two GPU tests are opt-in by default.
- Two Python CSV replay tests passed.
- CLI help and invalid-argument paths were checked without unnecessarily creating a GPU device.
- A self-owned procedural 512×512 image was serialized on one NVIDIA/DX12 drawing worker with the formal record budget. The output had 394 records: one background, 233 shapes and 160 region markers. All three stages completed; the packed CSV was independently replayed to PNG.
- Core search, shader, scheduling and geometry files retain the formal algorithm. CLI defaults and input/output handling were made portable.

## Model, training and inference

- Nine CPU tests cover shared-history gradients and causal prefix behavior, the real decoder's shape rules, CFG residual arithmetic, captions, checkpoint safety, archive gating and resume-position contracts.
- The actual generated synthetic CSV was linked to caption data and read by the trainer's mmap dataset.
- A real public CLI run trained a 630,850-parameter model for Epochs 1 and 2, saved complete optimizer checkpoints, then resumed Epoch 2 to Epoch 3. Global steps and Adam counters advanced exactly 1 → 2 → 3; parameters changed on the resumed optimizer step.
- The same Epoch 3 checkpoint was then loaded by the public inference CLI and produced eight geometry records plus PNG/NPY/JSON output, exercising train → save → resume → infer.
- The tiny CUDA validation used PyTorch 2.11.0+cu130 on the NVIDIA development laptop, a pre-existing triton-windows 3.6.0.post26 package and existing MSVC/Windows SDK, with process-only settings. Compile level 0 disables the outer model compilation; the original fixed-grid FlexAttention still JIT-compiles its GPU kernel. This does not certify arbitrary bare Windows Python environments. Linux CUDA/ROCm remains the intended production fused-training platform.
- The 1B checkpoint's metadata and tensor shapes were inspected using read-only mmap; 1B training and Python full-image inference were not rerun for source curation.

## Native App source

- The cleaned CPU application passed 663 tests, with zero failures and two opt-in tests ignored.
- All registered root binaries and all targets, including test configurations, passed type checking.
- The public CPU, tools and AMD Rust build-script check paths were invoked successfully.
- PowerShell source parsing and NSIS syntax checks passed. Generated syntax-only placeholder files were removed.
- The AMD C++ bridge was not rebuilt on the NVIDIA development laptop; its matching Windows HIP/ATen SDK is an external developer dependency. The parameterized build script preserves the production compiler flags.
- No new all-backend installer was built from the cleaned candidate. See [binary provenance](binary_provenance.md) for the distinction from the supplied existing 0.9.29 installer.

## Package boundaries

The package is checked for developer-machine paths, known private endpoint identifiers, credential-shaped tokens, large weight files and build-cache directories. The only supplied images are the self-owned synthetic input and replay. Small upstream parser tensor fixtures remain tracked; generated model checkpoints do not.

Dependency license texts, exact package metadata and versioned upstream provenance are included as materials for review. Specific third-party binary source/redistribution questions are recorded in the App's component provenance instead of being represented as a complete legal audit.
