---
mode: maintenance
commit: 15cda6c
generated_at: 2026-11-23
ignore: See Scope and Exclusions
---

## Summary

Rust audio execution/local-gradient primitives, prepared native training, Intel/NVIDIA adapters, and reproducible performance comparisons.

## Scope and Exclusions

The base snapshot is `15cda6c` on `main`. Maps inventory maintained text
sources, build configuration and documentation. Maps themselves are navigation artifacts.

| Path / pattern | Treatment | Reason / basis |
| --- | --- | --- |
| `.git/`, `target/`, `.venv/`, `.project-maintenance/`, `.serena/`, `**/__pycache__/`, `**/*.pyc` | Excluded | VCS, build, environment, review-state, external-agent cache and interpreter artifacts; `.gitignore` |
| `nas_access/` | Excluded | Private, user-supplied connection credentials; `.gitignore`; never include in source archives |
| `Cargo.lock` | Excluded | Generated dependency lock; still part of build and benchmark source fingerprints |
| `benchmarks/results/` | Boundary only | Raw measurements, source archives, compressed measured binaries/traces and archived one-off analysis scripts; indexed only as a reproducibility boundary |
| `models/gtcrn/gtcrn.py`, `models/gtcrn/upstream_infer.py`, `models/gtcrn/LICENSE` | Included, immutable vendor snapshot | Preserve upstream bytes and MIT attribution; hash-checked by `upstream.json`, adapters live in `tools/` |

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Maintain the project website | Website | `website/CODEMAP.md` | `README.md` and the relevant runtime contract when publishing examples or measurements |
| Continue the project objective | Direction | `target.md`, `docs/roadmap.md`, `docs/plan-phases-a-d.md`, `docs/progress.analysis.md` | `docs/competitive-analysis-v1.md`, indexed progress ranges and `reference.md` for original environment notes |
| Change tensor contracts or audio geometry | Storage | `src/CODEMAP.md` | `tests/` and `docs/architecture.md` for public contract changes |
| Reconstruct audio or validate full spectra | Audio | `examples/CODEMAP.md`, `src/CODEMAP.md` | `tools/check_audio.py`, `tests/`, `docs/architecture.md` |
| Change 2D/grouped/transposed convolution | Spatial | `src/CODEMAP.md`, `native/CODEMAP.md` | `tests/conv2d.rs`, actual GTCRN stage captures |
| Build 1D/2D and transposed convolution local VJPs and tape training | Spatial / Gradients | `docs/conv-backward-v1.md`, `src/CODEMAP.md` | Selective operand dependencies, bounded tiled/vendor routes, exact mathematical duality for `ConvTranspose1d`/`ConvTranspose2d`, independent FP64/Torch controls and CPU/CUDA resource checks |
| Build normalization, stable softmax/log-softmax and nonlinear reductions | Numerical blocks / Gradients | `docs/norm-v1.md`, `docs/softmax-v1.md`, `docs/reduction-v1.md`, `src/CODEMAP.md` | `LayerNorm`/`RmsNorm`/`GroupNorm`, `Softmax`/`LogSoftmax`, `Max`/`Min`/`LogSumExp` forward and local VJPs on CPU/CUDA, NaN/inf/tie parity and tape/fusion integration |
| Build broadcast batched matmul and fused Flash SDPA | Sequence / Dense algebra | `docs/bmm-v1.md`, `docs/sdpa-v1.md`, `src/CODEMAP.md`, `native/CODEMAP.md` | Arbitrary-rank broadcast `Bmm` (`Collapsed2d`/`CublasStridedBatched`/`TiledCuda`) and strict-FP32 O(T)-memory Flash `Sdpa` (MHA/GQA, causal/non-causal, zero-atomic `dQ`/`dK`/`dV`) |
| Build differentiable structural padding and concatenation | Tensor algebra / Gradients | `docs/structural-v1.md`, `src/CODEMAP.md`, `native/CODEMAP.md` | `Pad` (`Constant`/`Reflect`/`Replicate`, zero-atomic backward gather) and `Cat`/`cat_all` on CPU/CUDA |
| Build native fused AdamW, global L2 grad clipping and checkpoints | Optimizers / Checkpointing | `docs/optim-v1.md`, `src/CODEMAP.md`, `native/CODEMAP.md` | `cpu::AdamW`, `cuda::AdamW`, deterministic FP64 global L2 grad clipping, single-CUDA-Graph dynamic-LR `step_scalar_adamw_captured` and checksummed cross-device `Checkpoint` |
| Validate the existing complete-graph regression | Model fixture | `models/CODEMAP.md`, `docs/gtcrn.md` | `tools/CODEMAP.md`, `src/gru.rs`, captured shapes and profiles; no new application work in year one |
| Plan foundational tensor, autodiff and runtime work | Foundation | `docs/roadmap.md`, `docs/architecture.md` | `src/CODEMAP.md`, operator/backward coverage and reproducible CPU/GPU evidence |
| Build general reductions and their local derivatives | Reduction | `docs/reduction-v1.md`, `src/CODEMAP.md` | Storage/view-adjoint contracts, independent numerical/gradient oracles and CPU/CUDA resource checks |
| Build pointwise derivatives and broadcast contraction | Gradients | `docs/pointwise-backward-v1.md`, `src/CODEMAP.md` | Pointwise/reduction/saved-value contracts, full operation/operand VJPs, independent references and CPU/CUDA resource checks |
| Build native reverse-mode composition and training | Training | `docs/autodiff-v1.md`, `docs/matmul-v1.md`, `src/CODEMAP.md` | Saved versions, view/local-VJP contracts, tied/branched gradients, parameter updates and independent native training/resource checks |
| Change complete native model execution | Graph / Model | `src/CODEMAP.md`, `src/models/CODEMAP.md` | `examples/gtcrn-infer.rs`, `tools/check_gtcrn.py`, full model traces |
| Improve Intel/NVIDIA operator latency | Execution | `src/CODEMAP.md`, `native/CODEMAP.md` | `benchmarks/`, `tools/`, `docs/benchmarking.md` |
| Optimize CPU backward traversal | CPU optimization | `docs/cpu-prepared-backward-results-v1.md`, `docs/cpu-prepared-backward-v1.md`, `src/CODEMAP.md` | Qualified native CPU control plus original cumulative anchor, both negative predecessors, explicit header budget and complete-step/local-gradient/resource gates |
| Reproduce or inspect a measured result | Measurement | `tools/CODEMAP.md`, `benchmarks/CODEMAP.md` | `docs/benchmarking.md` and recorded source hashes |
| Characterize whole-model performance or process memory | Model measurement | `docs/gtcrn-characterization-v1.md`, `docs/gtcrn-characterization-results-v1.md`, `tools/CODEMAP.md` | Fixed model protocol, frozen binary, raw profiles and full numerical references |
| Characterize pointwise operator/layout performance | Tensor measurement | `docs/pointwise-characterization-v2.md`, `docs/pointwise-characterization-results-v2.md`, `tools/CODEMAP.md`, `src/bin/CODEMAP.md` | Fixed pointwise protocol, complete backing checks, producer/transcript/profile recovery and per-layout losses; v1 failures remain archived |
| Characterize complete native training steps | Training measurement | `docs/training-characterization-v1.md`, `docs/training-characterization-results-v1.md`, `tools/CODEMAP.md`, `src/bin/CODEMAP.md` | Fixed protocol, independent evolving controls, complete forward/backward/SGD modes, reset/repeat gates, explicit diagnostic recovery and archive/event audits |
| Adjust SDK/build integration | Build | `Cargo.toml`, `build.rs` | `native/`, `README.md` |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
| --- | --- | --- | --- |
| [website/](website/CODEMAP.md) | Website | React, TypeScript, Vite | Multilingual installation, runnable tutorials, API index and benchmark explorer |
| [src/](src/CODEMAP.md) | Storage / Execution / Training | RustFFT; native adapters | Public Rust API, prepared inference/training execution, model lowering, CPU/CUDA plans and operator benchmark workers |
| [native/](native/CODEMAP.md) | Execution | Intel oneMKL/oneDNN; CUDA/cuBLAS/cuFFT/cuDNN | Header-checked vendor ABI, planned convolution and CUDA kernels |
| [tests/](tests/CODEMAP.md) | Correctness | Rust public API | Independent oracles, geometry, reuse, allocation and ownership checks |
| [tools/](tools/CODEMAP.md) | Measurement | PyTorch; standard-library archive tooling | Cross-framework measurement and source recovery |
| [benchmarks/](benchmarks/CODEMAP.md) | Measurement | Benchmark binary and comparison harness | Authoritative workloads, Python environment pins, result boundary |
| [examples/](examples/CODEMAP.md) | Audio / Model | Public library API | Complete-output audio and real-model stage replay fixtures |
| [models/](models/CODEMAP.md) | Model | Pinned upstream reference and PyTorch | Model contracts, source/license identities and reference requirements |
| [docs/](docs/CODEMAP.md) | Direction / Contracts | Current implementation and evidence | Architecture, measurement rules, progress and roadmap |

## Files

| File | Domain | Function |
| --- | --- | --- |
| `Cargo.toml` | Build | Package, explicit vendor features, dependencies and release/lint settings |
| `LICENSE` | Distribution | MIT license for fast-audio project code |
| `CHANGELOG.md` | Distribution | User-visible changes and public preview history |
| `.github/workflows/ci.yml` | CI | Change-scoped portable Rust, Cargo package and website checks with one merge status |
| `build.rs` | Build | Compile/link selected native adapters and configure SDK/architecture overrides |
| `.gitignore` | Build | Keep private access material, environments and generated artifacts out of Git |
| `.gitattributes` | Reproduction | Preserve byte-identical vendor snapshots, frozen protocols and checksummed raw/derived measurement artifacts, including emitted whitespace |
| `README.md` | Direction | Project status, build/API examples and measurement entry points |
| `AGENTS.md` | Navigation / Environment | English documentation rule, map navigation protocol, and Colab CLI CPU/GPU instance-separation and Google Drive I/O rules |
| `target.md` | Direction | Original user-authored long-term objective |
| `reference.md` | Environment | Original user-authored dependency/hardware setup notes |
