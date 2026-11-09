# AMD / ATen runtime license materials

These are license and notice texts, not runtime DLLs or a development SDK.

- `pytorch-LICENSE.txt` and `pytorch-NOTICE.txt` were copied from the exact installed official AMD `torch-2.9.0+rocmsdk20251116` wheel metadata.
- `amd-comgr-*` were copied from documentation in the installed official ROCm core wheel.
- HIP, CLR, MIOpen and the BLAS/FFT/random/solver/sparse project texts use the official ROCm `rocm-7.1.1` source tag.
- The installed `rocm-openblas.dll` identifies `OpenBLAS 0.3.30 NO_AFFINITY generic`; matching upstream base and LAPACK-Netlib license texts are included. This does not identify the exact AMD fork commit.

`PROVENANCE.json` records per-file URLs, refs, hashes, sources and specific remaining checks. No development-machine paths, IP addresses or credentials are published.

The strict native runtime manifest remains unchanged. Source-built installers now copy these documents under `licenses/amd/`; this does **not** mean the already-built 0.9.29 executable has been modified or that every transitive vendor notice has been exhaustively audited.
