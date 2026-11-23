---
mode: maintenance
generated_at: 2026-11-23
---

## Summary

Execution contracts, reproducible measurement rules, current evidence and long-term direction.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Extend the tensor/operator API | Contracts | `architecture.md` | `../src/`, public consumers and tests |
| Design or assess a performance experiment | Measurement | `benchmarking.md` | Workload suite, raw report and `colab-hardware-evaluation.md` for hosted cloud constraints |
| Evaluate Google Colab as an experiment host | Environment / Measurement | `colab-hardware-evaluation.md` | `benchmarking.md`, `roadmap.md`, `../build.rs` and `../tools/CODEMAP.md` |
| Continue implementation priorities | Direction | `roadmap.md`, `plan-phases-a-d.md`, `progress.analysis.md` | `competitive-analysis-v1.md`, indexed progress ranges, `../target.md` and current source/runtime state |
| Assess PyTorch positioning or vendor vs custom boundaries | Strategy / Positioning | `competitive-analysis-v1.md` | `plan-phases-a-d.md`, `roadmap.md`, `operator-coverage.md`, `progress.md` |

## Files

| File | Domain | Function |
| --- | --- | --- |
| `competitive-analysis-v1.md` | Strategy / Positioning | PyTorch `2.14+` competitive analysis covering GPU/CPU training and inference modes, audio workload physical bottlenecks, structural PyTorch audio weaknesses, core `fast-audio` systems/kernel levers, PyTorch-only benchmark anchor matrix, and explicitly deferred non-PyTorch comparisons |
| `plan-phases-a-d.md` | Execution plan | Active PyTorch-first 25-ticket Work Breakdown Structure (`A0–A6`, `B1–B7`, `C1–C6`, `D1–D5`), dependency DAG, unified `TrainingTape` + `FrozenInferencePlan` architecture, mandatory per-step PyTorch measurement protocol, and deferred post-PyTorch scope |
| `training-characterization-v1.md` | Training measurement | Predeclared twelve-case complete forward/backward/SGD baseline, matched eager/compiled/captured Torch modes, reset and post-update snapshot semantics, independent controls, fixed windows and no promotion |
| `training-characterization-results-v1.md` | Training evidence | Complete twelve-case CPU/GPU baseline, per-case compiled/captured losses, CPU samples and GPU event audits, preserved exporter/mutation failures, exact archive recovery and bounded next hypotheses |
| `cpu-backward-traversal-v1.md` | CPU optimization contract | Predeclared coalesced-row backward experiment, fixed native control, exact numerical/resource constraints, primary/per-case gates and bounded fresh confirmation |
| `cpu-backward-traversal-results-v1.md` | CPU optimization evidence | Completed two-screen/five-context experiment, large-case gains with unresolved tiny guard, full functional/resource/byte audits, restored scalar default, sampled/codegen diagnostics and fixed next hypotheses |
| `cpu-backward-isolation-v1.md` | CPU optimization contract | Scalar/row execution isolation, preserved unexecuted declaration and pre-timing worker-cap correction, original control, one candidate/screen and twelve-context fixed confirmation |
| `cpu-backward-isolation-results-v1.md` | CPU optimization evidence | Sole-screen rejection despite large-case gains, preserved unexecuted correction, full functional/exceptional/resource checks, entry-code observations and restored scalar control without confirmation |
| `cpu-prepared-backward-v1.md` | CPU optimization contract | Prepared operation/strategy dispatch, separate one-pointer fixed-header budget, original scalar anchor, one candidate without timing screening and twelve fixed fresh contexts |
| `cpu-prepared-backward-results-v1.md` | CPU optimization evidence | Qualified twelve-context native CPU gain, all case guards, explicit fixed-header cost, exact functional/bit/resource recovery, order/model limits and retained original cumulative anchor |
| `architecture.md` | Contracts | Storage, numerical/layout, preparation, FFI and stream ownership rules |
| `api-stability.md` | Public releases | Cargo compatibility lines, numerical and checkpoint contracts, deprecation and release review |
| `releasing.md` | Release engineering | Manual version promotion, CI responsibilities and registry checks |
| `benchmarking.md` | Measurement | PyTorch-only training and inference comparison rules, workloads, correctness gates, timing/fairness limits, reproduction and promotion protocol |
| `colab-hardware-evaluation.md` | Environment / Measurement | Live-verified Colab GPU/CPU evaluation, strict CPU/GPU instance-separation rule, `colab` CLI and Google Drive `/content/drive/MyDrive/fast-audio-colab` I/O cache workflow, and compatibility audit |
| `roadmap.md` | Direction | Foundation-only first-year admission and dependency gates, training/inference closure, operator-to-suite performance qualification, separate readiness statuses and deferred audio applications |
| `progress.md` | Evidence | Verified work, measured results, remaining gaps and next action |
| `progress.analysis.md` | Navigation | Feature/section ranges for the long chronological progress record; verification evidence stays in progress and linked reports |
| `intel-long-window-v1.md` | Measurement | Predeclared follow-up after the unresolved Intel short-STFT guard |
| `gtcrn.md` | Model | Pinned reference/parameters, complete native model and stage evidence, GPU algorithm diagnostics, numerical limits and remaining acceptance work |
| `gtcrn-characterization-v1.md` | Model measurement | Declared complete-model contexts, exact sampling/stopping, memory/profiling scopes, frozen binary copies and fairness limits |
| `gtcrn-characterization-results-v1.md` | Model evidence | Completed three-context comparison, observed process memory, CPU/CUDA attribution, exact artifact recovery and next experiment rationale |
| `mapped-access-screen-v1.md` | Graph measurement | Declared mapped-address mechanism, integer proof, Intel screen, preserved GPU failure/repeatability diagnosis and recovery limits |
| `dac.md` | Deferred research | Preserved pinned decoder meta/source inventory, recovery and limits; integration excluded from year-one foundations |
| `cudnn-repeatability-v1.md` | Runtime correctness | Explicit convolution engine policy, fixed twelve-process validation, raw-output audit, selected engines, recovery and claim limits |
| `tensor-layout-v1.md` | Storage contracts | Signed-stride/scalar/empty layout, borrowed aliases/copying, independent oracles, Miri/model/recovery evidence and future training/memory requirements |
| `storage-versions-v1.md` | Training storage contracts | Opt-in identity/generations, saved-version checks, CUDA ordering limits, retained interruption/resumption evidence and tested resource budgets |
| `saved-values-v1.md` | Training lifetimes | Shared/retained ownership, borrow/version/poison rules, verified CPU/CUDA lifetime/resource evidence and future tape view-adjoint requirements |
| `view-backward-v1.md` | Gradient contracts | Prepared view adjoints, bounded overlap indexing, analytical broadcasts, fixed reduction order, verified gradient/resource checks and GPU instrumentation evidence |
| `pointwise-v1.md` | Tensor algebra contracts | Strided unary/binary preparation, explicit broadcasting, FP32 math rules, traversal/resource contracts, verified independent forward checks and GPU instrumentation evidence |
| `pointwise-backward-v1.md` | Gradient contracts | Verified unary/binary local VJPs, selective payload dependencies, fused ordered broadcast contraction, stable/empty/bit rules, 524-case CPU/CUDA/Miri/tool evidence and preserved predecessor/recovery identities |
| `autodiff-v1.md` | Training contracts | Verified initial prepared tape API, branded values, view/tied gradients, selective saves, failure/epoch validity, checked plain SGD, procedural audio/Torch controls, frozen functional campaign and separate runtime/performance gaps |
| `matmul-v1.md` | Dense gradient contracts | Verified rank-two strided forward/local VJPs, vendor transpose/direct and bounded fallback routes, dependency/empty/accumulation rules, FP64/Torch/manual updates, frozen backend/Miri/GPU-tool gates and recoverable sources/exports/executables |
| `conv-backward-v1.md` | Spatial gradient contracts | Verified rank-three/four strided 1D/2D cross-correlation forward/local VJPs, direct vendor and bounded tiled routes, selective operand saves, tape integration and CPU/MKL/CUDA/Miri/sanitizer/PyTorch evidence |
| `stft-backward-v1.md` | Spectral gradient contracts | Prepared strided FP32 STFT, reflect-centered STFT and NOLA inverse STFT forward/local VJPs (`input_required == false`), deterministic zero-atomic CUDA kernels, `autograd::Builder` integration, whole-step fusion and CUDA Graph capture |
| `norm-v1.md` | Normalization contracts | Parameter-free strided FP32 `LayerNorm`, `RmsNorm` and zero-copy reshaped `GroupNorm` forward and local VJPs on CPU (runtime AVX2+FMA) and CUDA (Tier-1 warp, Tier-2 `float4` block, general strided) |
| `softmax-v1.md` | Softmax contracts | Arbitrary-axis strided numerically stable `Softmax` and `LogSoftmax` forward and local VJPs on CPU (runtime AVX2+FMA Cody-Waite `exp256_ps`, zero-allocation `.bss` forward-probability cache, MKL VML `vmsExp`) and CUDA (Tier-1 warp, Tier-2 register block, Tier-3 strided) |
| `bmm-v1.md` | Batched GEMM contracts | Arbitrary-rank (`rank >= 2`) right-aligned broadcast batched matrix multiplication (`BmmSpec`, `BatchedMatmulPlan`, `BatchedMatmulBackward`) across `Collapsed2d`, `CublasStridedBatched` and `TiledCuda` routes |
| `sdpa-v1.md` | Attention contracts | Strict-FP32 O(T)-memory Flash-style Scaled Dot-Product Attention (`SdpaSpec`, `SdpaPlan`, `SdpaBackward`) with MHA/GQA (`H % H_kv == 0`), causal masking, signed-stride 4D views, zero-atomic `dQ`/`dK`/`dV` and selective `dV` dependencies |
| `structural-v1.md` | Structural contracts | Arbitrary-rank strided `Pad` (`Constant`/`Reflect`/`Replicate`, zero-atomic backward gather, `input_required == false`) and `Cat`/`cat_all` (`input_dependencies == [false, false]`) on CPU and CUDA |
| `optim-v1.md` | Optimizer / Checkpoint contracts | Strict-FP32 decoupled `AdamWConfig`, `cpu::AdamW`, `cuda::AdamW`, deterministic FP64 global L2 gradient clipping, single-CUDA-Graph dynamic-LR `step_scalar_adamw_captured` and FNV-1a checksummed cross-device `Checkpoint` |
| `reduction-v1.md` | Reduction contracts | Verified arbitrary-stride `Sum`/`Mean`/`Max`/`Min`/`LogSumExp` and local adjoints, fixed-tree/empty/count/NaN/inf/tie rules, O(rank) resources and CPU/CUDA/Miri/tool evidence |
| `operator-coverage.md` | Capability inventory | Current FP32 forward/backend/layout coverage, explicit derivative gaps, retained graph consumers and verified descriptive pointwise baselines |
| `pointwise-characterization-v1.md` | Tensor measurement | Original pointwise declaration, preserved identical non-contiguous compiled-output readiness failures, sealed archive/binary/log audit and no partial performance inference |
| `pointwise-characterization-v2.md` | Tensor measurement | Fixed-scope functional compiled writeback, unchanged gates, verified CPU/GPU harness/source/binary/log evidence and completed characterization route |
| `pointwise-characterization-results-v2.md` | Tensor evidence | Complete 56-case CPU/GPU comparisons, per-layout losses, CPU samples, recovered original GPU streams, producer/transcript/event audits and no-promotion limits |
