# Progress

## 2026-11-23: audio deep learning foundation suite (Norm, Softmax, nonlinear reductions, broadcast BMM, strict-FP32 SDPA, ConvTranspose1d/2d, Pad/Cat, liveness reuse, 2D CPU fusion, AdamW, and Checkpoints)

Completed the Year-1 audio deep learning operator, runtime, optimizer, and serialization stack on
both Intel CPU (`--features mkl,onednn`) and NVIDIA CUDA (`--features cuda,cudnn`), backed by
contracts [norm-v1.md](norm-v1.md), [softmax-v1.md](softmax-v1.md),
[reduction-v1.md](reduction-v1.md), [bmm-v1.md](bmm-v1.md), [sdpa-v1.md](sdpa-v1.md),
[conv-backward-v1.md](conv-backward-v1.md), [structural-v1.md](structural-v1.md), and
[optim-v1.md](optim-v1.md):

- **Normalization, Softmax, and Nonlinear Reductions (`src/norm.rs`, `src/softmax.rs`,
  `src/reduce.rs`, `src/cuda/{norm,softmax,reduce}.rs`, `native/{norm,softmax,reduce}.{h,cu}`):**
  Added strided `LayerNorm`, `RmsNorm`, `GroupNorm` (via `Builder::group_norm`), `Softmax`,
  `LogSoftmax` (both saved-input `run` and stateless `run_from_output` matching
  `aten::_softmax_backward_data`), and `Max`/`Min`/`LogSumExp` forward + local VJPs with exact
  tie counters and NaN propagation. Runtime AVX2+FMA contiguous kernels (`exp256_ps` Cody-Waite
  degree-6, `.bss` `SOFTMAX_FWD_CACHE`) on CPU; zero-shared-memory warp/vec4 kernels (SM 8.0+
  hardware `max.NaN.f32`, streaming `ld.global.cs` on `>= 4 MiB` inputs) and WAR-free block
  reductions on CUDA.
- **Broadcast Batched Matmul & Strict-FP32 SDPA (`src/bmm.rs`, `src/sdpa.rs`,
  `src/cuda/{bmm,sdpa}.rs`, `native/{bmm,sdpa}.{h,cu}`, `native/cuda_backend.h`):** Added
  arbitrary-rank broadcast `BmmSpec`/`BatchedMatmulPlan`/`BatchedMatmulBackward` (`Collapsed2d`
  single-GEMM fast path, strided batched BLAS, and zero-atomic broadcast reduction) and
  MHA/GQA causal/non-causal `SdpaSpec`/`SdpaPlan`/`SdpaBackward`. CPU runs L1-tiled
  FlashAttention-2 with runtime AVX2+FMA `exp256_ps`/dot/axpy and zero-allocation `.bss`
  `SDPA_BWD_CACHE`; CUDA combines L2-cached double-buffered 64×64 strict-FP32 SIMT SDPA GEMM
  kernels (`qk`/`pv`/`ptdo`, 1-launch permuted & GQA support, causal upper-triangle skip,
  parallel per-query-head `dK`/`dV` scratch + `float4` GQA head reduction, and `float4` warp
  softmax) with `O(T)` register-tiled FlashAttention-2 fallback for long sequences (`T_k > 1024`).
- **Transposed Convolutions, Structural Pad/Cat, Activation Liveness Reuse, and 2D CPU Fusion
  (`src/conv.rs`, `src/structural.rs`, `src/cuda/{conv,structural}.rs`,
  `native/{conv_backward,structural}.{h,cu}`, `src/autograd/{prepare,fusion,host,engine}.rs`):**
  Added `ConvTranspose1d`/`ConvTranspose2d`, arbitrary-rank `Pad` (`Constant`/`Reflect`/`Replicate`,
  zero-atomic gather backward), and `Cat`/`cat_all`; integrated exact tape activation liveness
  slot reuse (`src/autograd/prepare.rs`) and 2D L1 cache-blocked CPU whole-step fusion
  (`src/autograd/fusion.rs`, closing the `spectrum-medium`/`spectrum-large` CPU gap vs Inductor).
- **Decoupled AdamW, Global L2 Gradient Clipping, Dynamic-LR Single-CUDA-Graph Training, and
  Checkpoints (`src/optim.rs`, `native/training.cu`, `src/autograd/host.rs`, `src/cuda/tape.rs`):**
  Added `cpu::AdamW`, `cuda::AdamW`, deterministic FP64 global L2 gradient clipping, on-device
  `d_step`/`d_lr` control state so `cuda::Tape::step_scalar_adamw_captured` replays forward +
  backward + grad-clip + AdamW with dynamic LR changes in a single CUDA Graph without recapture,
  and FNV-1a checksummed cross-device binary `Checkpoint` save/load.

Verified across all four check stacks under strict Colab CPU/GPU isolation:
- **Local portable CPU & Miri:** `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`
  (0 warnings), **130 Rust tests + 16 doctests** passed; `cargo +nightly miri test` passed all 4
  new unsafe-boundary & tape tests (`norm_softmax_nl_reduce_miri_strided_and_tape`,
  `cpu_tape_audio_transformer_attention_block_and_miri`, `cpu_tape_unet_vocoder_step_and_miri`,
  `cpu_adamw_and_checkpoint_miri`) with **0 UB**.
- **Colab Intel CPU (`fa-cpu`, `--features mkl,onednn`):** `clippy --all-targets --features
  mkl,onednn -- -D warnings` (0 warnings), **132 Rust tests + 16 doctests** passed, and **35/35**
  strict-FP32 PyTorch `2.14.1+cpu` cross-checks (`tools/check_new_ops.py cpu`, max abs diff
  `<= 1.192e-07`).
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0, `CUDA_ARCH=89 --features cuda,cudnn`):**
  `clippy --all-targets --features cuda,cudnn -- -D warnings` (0 warnings), **195 Rust tests +
  17 doctests** (212 total, 0 failed), **35/35** strict-FP32 PyTorch `2.14.1+cu130` cross-checks
  (`tools/check_new_ops.py cuda`, max abs diff `<= 2.384e-07`), and **12/12 `compute-sanitizer`
  runs** (`memcheck`, `racecheck`, `synccheck` across `norm_cuda`, `bmm_sdpa_cuda`,
  `conv_transpose_structural_cuda`, and `optim_cuda`: **0 errors, 0 race hazards, 0 warnings**).

Head-to-head strict-FP32 benchmark suite (`examples/audio-foundation-bench.rs` +
`tools/bench_foundation_vs_pytorch.py`, vs PyTorch `2.14.1`):
- **Colab Intel CPU (`fa-cpu`, MKL+oneDNN, 1 thread — 11/11 clean sweep over PyTorch Eager and
  Inductor; 2.66x geomean vs Eager, 1.70x geomean vs Best):**
  - `spectrum-medium`: **`243.3 us`** vs PT Eager `3741.7 us`, PT Inductor `971.9 us` (**4.00x**)
  - `spectrum-large`: **`853.2 us`** vs PT Eager `17059.9 us`, PT Inductor `5772.0 us` (**6.77x**)
  - `wave-multichannel`: **`502.2 us`** vs PT Eager `1199.8 us`, PT Inductor `826.8 us` (**1.65x**)
  - `wave-long`: **`491.5 us`** vs PT Eager `1171.9 us`, PT Inductor `822.4 us` (**1.67x**)
  - `norm-layernorm-256`: **`461.0 us`** vs PT Eager `694.4 us`, PT Inductor `519.1 us` (**1.13x**)
  - `norm-rmsnorm-512`: **`287.6 us`** vs PT Eager `1772.0 us`, PT Inductor `450.4 us` (**1.57x**)
  - `softmax-256`: **`1387.0 us`** vs PT Eager `1480.0 us`, PT Inductor `1395.4 us` (**1.01x**)
  - `bmm-linear`: **`3218.2 us`** vs PT Eager `3322.1 us`, PT Inductor `3297.1 us` (**1.02x**)
  - `sdpa-causal-t128-d64`: **`2384.7 us`** vs PT Eager `2967.2 us` (**1.24x**)
  - `audio-transformer-step-adamw`: **`3497.7 us`** vs PT Eager `3965.9 us` (**1.13x**)
  - `spectral-vocoder-step-adamw`: **`2112.0 us`** vs PT Eager `3866.8 us` (**1.83x**)
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0 — 9/9 sweep vs PyTorch Eager at 2.79x geomean;
  8/9 wins vs PyTorch Best at 1.84x geomean):**
  - `norm-layernorm-256`: **`54.6 us`** vs PT Eager `142.8 us`, PT Inductor `117.8 us` (**2.16x**)
  - `norm-rmsnorm-512`: **`47.1 us`** vs PT Eager `422.0 us`, PT Inductor `110.0 us` (**2.34x**)
  - `softmax-256`: **`186.7 us`** vs PT Eager `343.5 us` (**1.84x vs Eager**), PT Inductor
    (single-kernel fused Triton) `182.6 us` (`0.98x` vs Inductor)
  - `bmm-linear`: **`139.0 us`** vs PT Eager `227.7 us`, PT Inductor `226.2 us` (**1.63x**)
  - `sdpa-causal-t256-d64`: **`148.6 us`** vs PT Eager `295.0 us` (**1.99x**)
  - `sdpa-noncausal-t512-d64`: **`364.7 us`** vs PT Eager `590.1 us` (**1.62x**)
  - `sdpa-gqa-causal-t512-d64`: **`374.0 us`** vs PT Eager `385.4 us` (**1.03x**)
  - `audio-transformer-step-adamw` (single CUDA Graph): **`382.8 us`** vs PT Eager `2276.8 us`
    (**5.95x**), PT CUDA Graph `459.2 us` (**1.20x**)
  - `spectral-vocoder-step-adamw` (single CUDA Graph): **`373.8 us`** vs PT Eager `2790.9 us`
    (**7.47x**; PyTorch fails CUDA Graph capture in `torch.istft` with
    `cudaErrorStreamCaptureInvalidated`)

## 2026-11-20: differentiable strided STFT, reflect-centered STFT, and NOLA inverse STFT with whole-step fusion and CUDA Graph capture

Closed the differentiable audio spectral core gap in [operator-coverage.md](operator-coverage.md)
under contract [stft-backward-v1.md](stft-backward-v1.md) (`src/stft.rs`, `src/cuda/stft.rs`,
`native/stft_backward.h`, `native/stft_backward.cu`, `src/spec.rs`, `src/cpu.rs`, `src/cuda.rs`,
`src/autograd.rs`, `src/autograd/host.rs`, `src/cuda/tape.rs`, `src/autograd/fusion.rs`,
`tests/stft_backward.rs`, `tests/stft_backward_cuda.rs`, `tools/check_stft_backward.py`,
`examples/spectral-train.rs`):

- **CPU & CUDA Prepared Strided Spectral Forward + Local VJPs:** Added `StftPlan`,
  `StftBackward`, `CenteredStftPlan`, `CenteredStftBackward`, `IstftPlan`, and `IstftBackward`
  on both `cpu` (100% safe-Rust portable RustFFT + `Send` Intel oneMKL DFTI) and `cuda`
  (cuFFT R2C/C2R + deterministic zero-atomic CUDA framing, reflection-adjoint gather, NOLA
  overlap-add, and endpoint-scaling kernels with 8-byte `direct_r2c` alignment guard).
  All six plans accept arbitrary signed-stride borrowed views, snapshot finite explicit
  windows, enforce `input_required() == false` (zero saved forward activations), support
  `Overwrite`/`Accumulate`, and perform zero hot allocations or host synchronizations.
- **Native Autograd Tape, Whole-Step Fusion, and CUDA Graph Capture:** Added
  `autograd::Builder::{stft, centered_stft, istft}` and registered all three spectral ops as
  linear fusion barriers so post-spectral/post-iSTFT pointwise/loss/SGD tails automatically
  fuse into L1-tiled CPU loops or NVRTC strict-FP32 CUDA kernels. Because NOLA envelopes and
  cuFFT descriptors are precomputed during preparation, `cuda::Tape::step_captured` captures
  and replays full `CenteredSTFT -> spectral op -> iSTFT -> loss -> backward -> SGD` steps in
  a single CUDA Graph (where standard `torch.istft` fails CUDA Graph capture with
  `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` due to its internal `cuStreamSynchronize`).

Verified across all four check stacks under strict Colab CPU/GPU isolation:
- **Local portable CPU & Miri:** 112 Rust tests + 16 doctests + `rustfmt --check`; Miri
  passed `stft_backward_miri_strided_and_tape` (strided reversed/permuted views, centered
  STFT + iSTFT backward, and fused Tape SGD) with **0 UB**.
- **Colab Intel CPU (`fa-cpu`, `--features mkl,onednn`, `rustc 1.98.1`):** 100% test suite
  (testing both RustFFT and oneMKL DFTI) + 16 doctests + `clippy --all-targets --features
  mkl,onednn -- -D warnings` (0 warnings) + `tools/check_stft_backward.py cpu`.
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0, `CUDA_ARCH=89 --features cuda,cudnn`,
  `rustc 1.98.1`):** 100% test suite + 17 doctests + `clippy --all-targets --features
  cuda,cudnn -- -D warnings` (0 warnings) + **6/6 `compute-sanitizer` runs** (`memcheck`,
  `racecheck`, `synccheck` on both `stft_backward_cuda` tests: **0 errors, 0 race hazards,
  0 warnings**) + `tools/check_stft_backward.py cuda`.

End-to-end differentiable spectral training benchmark (`examples/spectral-train.rs`:
`CenteredSTFT -> Mask+Bias -> iSTFT -> MSE + SGD`, strict FP32, `<= 1.77e-08` max abs diff vs
PyTorch `2.14.1`):
- **Colab Intel CPU (`fa-cpu`, MKL, 1 thread):**
  - `short-block-256` (`[1, 4096], N=256, H=64`): `fast-audio` **`246.79 us`** vs PyTorch Eager
    `905.84 us` (**3.67x faster**), PyTorch Inductor `872.87 us` (**3.54x faster**, `0.283x`).
  - `stereo-16k-512` (`[2, 16000], N=512, H=128`): `fast-audio` **`1910.82 us`** vs PyTorch
    Eager `3115.91 us` (**1.63x faster**), PyTorch Inductor `3098.43 us` (**1.62x faster**,
    `0.617x`). *(PyTorch Inductor stalls >5 min on `batch4-48k-1024` CPU complex fallback.)*
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0):**
  - `short-block-256`: `FA-CG` **`34.45 us`** (`FA-Uncap` `68.61 us`) vs `TorchEager`
    `1654.68 us` (**48.0x faster**), `TorchInductor` `1462.34 us` (**42.4x faster**),
    hand-optimized CUDA-Graph `TorchManualCG` (`rfft/irfft + fold`) `93.91 us` (**2.73x faster**,
    `0.367x`).
  - `stereo-16k-512`: `FA-CG` **`52.14 us`** (`FA-Uncap` `69.34 us`) vs `TorchEager`
    `1685.52 us` (**32.3x faster**), `TorchInductor` `4435.70 us` (**85.1x faster**),
    `TorchManualCG` `175.97 us` (**3.37x faster**, `0.296x`).
  - `batch4-48k-1024` (`[4, 48000], N=1024, H=256`): `FA-CG` **`185.77 us`** (`FA-Uncap`
    `192.87 us`) vs `TorchEager` `1875.69 us` (**10.1x faster**), `TorchInductor` `8208.69 us`
    (**44.2x faster**), `TorchManualCG` `657.22 us` (**3.54x faster**, `0.283x`).

## 2026-11-20: general whole-step tape fusion compiler (CPU L1-tiled fusion + CUDA NVRTC strict-FP32 fusion + CUDA Graph replay)

Closed the multi-pass pointwise/reduction tape dispatch bottleneck exposed in the previous
milestone by adding a general reverse-mode whole-step fusion compiler shared by `cpu::Tape`
and `cuda::Tape` (`src/autograd/fusion.rs`, `native/nvrtc_fusion.cu`, `build.rs`,
`src/autograd/engine.rs`, `src/autograd/host.rs`, `src/cuda/tape.rs`,
`src/bin/fa-training.rs`):

- **General Tape DAG Analyzer (`src/autograd/fusion.rs`):** Automatically inspects any
  prepared `Core` tape and lowers eligible forward/backward/SGD regions into either
  `WholeRegion` (pure pointwise / view-alias / trailing `SmallReduce(Sum|Mean, K<=8)` tapes)
  or `BarrierWithTail` (tapes with `Matmul`/`Conv1d`/`Conv2d` barriers followed by an
  arbitrary pointwise/alias/reduction/SGD tail). Supports arbitrary `Unary`/`Binary`/`Alias`
  (including bijective inverse `Slice(±1)`/`Permute` domain pull-back), automatic
  `GroupKept`/`SmallGlobal`/`FullElementwise` parameter-gradient classification, and strict
  disjoint-owner eligibility checks with automatic unfused fallback.
- **CPU Zero-`unsafe` L1-Tiled Fused Executor (`src/autograd/fusion.rs`, `src/autograd/host.rs`):**
  100% safe Rust 256-float L1-tiled forward+backward+SGD execution (`CpuFusedScratch`,
  reusing forward `Tanh`/`Sigmoid`/`Exp`/`Sqrt` values on backward and calling oneMKL VML
  `VML_HA` on tile slices when enabled). Exposed via `cpu::Tape::{step, step_scalar,
  has_fused_plan}`.
- **CUDA NVRTC Strict-FP32 Fused Codegen + CUDA Graph Replay (`src/autograd/fusion.rs`,
  `native/nvrtc_fusion.cu`, `src/cuda/tape.rs`):** Dynamic `dlopen` loader for `libnvrtc.so`
  and `libcuda.so.1` (linking only `-ldl`, process-wide PTX cache keyed by
  `(compute_XY, source)`), compiling strict FP32 kernels (`--std=c++17`, no
  `--use_fast_math`) across four launch topologies (`BarrierTailOnly`, `SingleBlockAll`,
  `WarpPerGroup`, `SplitBlocksPerGroup`) with disjoint `__restrict__` pointers and
  `cuLaunchKernel` on `ctx->stream`. Exposed via `cuda::Tape::{step, step_scalar,
  has_fused_plan}` and integrated into `step_captured` (compiling strictly before
  `fa_cuda_capture_begin` so the fused kernel records directly into the CUDA Graph).

Verified across all four check stacks under strict Colab CPU/GPU isolation:
- **Local portable CPU & Miri:** 108 Rust tests (including
  `cpu_tape_fused_step_matches_unfused_across_dense_waveform_spectrum_and_conv`) + 16
  doctests + `rustfmt --check`; Miri (`cargo +nightly miri test`) passed 4/4 unsafe-boundary
  and fused-tape tests with **0 UB**.
- **Colab Intel CPU (`fa-cpu`, `--features mkl,onednn`, `rustc 1.98.1`):** 100% test suite +
  16 doctests + `clippy --all-targets --features mkl,onednn -- -D warnings` (0 warnings).
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0, `CUDA_ARCH=89 --features cuda,cudnn`,
  `rustc 1.98.1`):** 100% test suite (including
  `cuda_tape_fused_audio_waveform_and_spectrum_topologies_match_unfused` exercising all four
  NVRTC topologies) + 17 doctests + `clippy --all-targets --features cuda,cudnn -- -D warnings`
  (0 warnings) + 6 `compute-sanitizer` runs (`memcheck`, `racecheck`, `synccheck` on both
  fused/captured CUDA tape tests: **0 errors, 0 race hazards, 0 warnings**).

Paired 5-block BEFORE (`b1ee58f`) vs AFTER vs PyTorch (`2.14.1`) benchmarks on Colab confirm
unchanged explicit resources and full numerical agreement across all 12 training-v1 workloads:
- **Colab Intel CPU (`fa-cpu`, MKL, 1 thread):** **1.368x additional geomean speedup** over
  `b1ee58f` and **`0.237x` geomean vs strongest PyTorch Inductor** (**4.22x faster geomean**).
  Sweeps PyTorch Eager **12/12** and wins **10/12 vs PyTorch `torch.compile(inductor)`**:
  `dense-tiny` `1.38 us` vs `101.32 us` (**73.5x faster**, bit-exact), `dense-tail` `5.99 us`
  vs `108.96 us` (**18.2x**, bit-exact), `dense-projection` `575.57 us` vs `733.12 us`
  (**1.27x**, bit-exact), `dense-large` `5544.59 us` vs `5891.29 us` (**1.06x**, bit-exact),
  `wave-tiny` `3.06 us` vs `90.25 us` (**29.5x**), `wave-block` `4.39 us` vs `86.36 us`
  (**19.7x**), `wave-multichannel` `513.68 us` vs `749.36 us` (**1.46x**), `wave-long`
  `618.08 us` vs `721.74 us` (**1.17x**), `spectrum-tiny` `5.07 us` vs `87.48 us` (**17.2x**),
  `spectrum-block` `35.12 us` vs `100.66 us` (**2.87x**); `spectrum-medium` (`1556.89 vs
  883.33 us`, 1.58x over M3) and `spectrum-large` (`8716.73 vs 5586.42 us`, 2.20x over M3)
  remain the only two CPU Inductor losses (max abs diff `<= 4.8e-8` across all non-dense
  cases; 4/4 `dense-*` bit-exact).
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0):** **12/12 SHA-256 bit-exact** between
  uncaptured fused and captured CUDA Graph fused (`Base vs Cand <= 2e-8`; 4/4 `dense-*`
  bit-exact). Delivers **10.963x geomean speedup** over `b1ee58f` (**8.042x** uncaptured),
  achieving a **12/12 clean sweep over PyTorch `compiled-cuda-graph`, `compiled`, and `eager`**
  with **`0.620x` geomean vs PyTorch `compiled-cuda-graph`** (**1.61x faster geomean**):
  `dense-tiny` `8.63 vs 9.41 us` (`0.918x`), `dense-tail` `14.25 vs 14.29 us` (`0.997x`),
  `dense-projection` `14.58 vs 16.27 us` (`0.896x`), `dense-large` `48.88 vs 50.76 us`
  (`0.963x`), `wave-tiny` `2.56 vs 4.10 us` (`0.624x`), `wave-block` `3.45 vs 3.96 us`
  (`0.870x`), `wave-multichannel` `5.45 vs 12.42 us` (`0.439x`), `wave-long` `5.15 vs
  11.39 us` (`0.452x`), `spectrum-tiny` `2.56 vs 5.11 us` (`0.500x`), `spectrum-block`
  `4.08 vs 5.75 us` (`0.710x`), `spectrum-medium` `5.06 vs 13.52 us` (`0.374x`),
  `spectrum-large` `8.08 vs 28.10 us` (**35.30x over M3**, `0.287x` — **3.48x faster than
  PyTorch CUDA Graph**).

## 2026-11-20: Intel oneMKL VML transcendentals, multi-SM CUDA split reductions, and guarded CUDA Graph step capture/replay

Targeted the two remaining whole-step bottlenecks exposed by the previous optimization pass:
CPU scalar glibc `libm` transcendentals (`tanh`/`exp`/`ln`/`sqrt`) in `wave-multichannel` and
`wave-long`, and the ~50 us uncaptured CUDA host launch floor plus low-group multi-SM reduction
underutilization on GPU:

- **CPU (`native/mkl.c`, `src/pointwise.rs`, `src/pointwise_backward.rs`):** Added single-threaded
  high-accuracy Intel oneMKL VML bindings (`vmsTanh`, `vmsExp`, `vmsLn`, `vmsSqrt` under
  `VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE`, `< 1 ULP`, no subnormal flushing) with
  contiguous `inner >= 64` fast paths for forward `Sqrt`/`Exp`/`Log`/`Sigmoid`/`Tanh`/`Silu`/
  `Softplus` and chunked 256-float backward `copy_rows` (`Exp`/`Sigmoid`/`Tanh`/`Silu`/
  `Softplus`), plus slice-iterator dense binary/backward rows.
- **CUDA (`native/cuda_backend.h`, `native/cuda.cu`, `native/training.cu`, `native/reduce.cu`,
  `native/pointwise_backward.cu`, `src/autograd/engine.rs`, `src/cuda/tape.rs`,
  `src/bin/fa-training.rs`):**
  - Added explicit guarded CUDA Graph capture and zero-allocation replay on `cuda::Tape`
    (`step_captured`, `step_scalar_captured`, `invalidate_capture`, `has_captured_graph`)
    with RAII capture abort, pre-bound static 4 MiB `cublasSetWorkspace` on `Context` (so
    deterministic `cublasSgemm` records cleanly without dynamic workspace allocations), relaxed
    capture mode (`cudaStreamCaptureModeRelaxed`) for multi-threaded host safety, and full
    per-replay session/pointer/layout/shape/borrow-lock/rate validation with conservative
    `Shared<DeviceTensor>` generation advancement on every updated leaf.
  - Added 8-block x 32-lane 4-way-prefetch bit-exact `Tree256` split reductions
    (`reduce_block_split` + `reduce_block_finalize`, `block_split` + `block_finalize`) backed
    by a fixed 32 KiB `Context::reduce_scratch` (leaving per-plan `workspace_bytes()` unchanged).
  - Exposed `--captured` on `fa-training` for explicit CUDA Graph step benchmarking.

Verified across all three hardware stacks under strict Colab CPU/GPU instance separation:
- **Local portable CPU:** 100% test suite (107 tests + 16 doctests), `clippy --all-targets -- -D warnings`,
  and `rustfmt --check`.
- **Colab Intel CPU (`fa-cpu`, `--features mkl,onednn`, `rustc 1.98.1`):** 100% test suite +
  16 doctests + `clippy --all-targets --features mkl,onednn -- -D warnings`.
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0, `CUDA_ARCH=89 --features cuda,cudnn`, `rustc 1.98.1`):**
  100% test suite (including `cuda_tape_captured_step_matches_uncaptured_bits_and_enforces_safety`
  covering 8-step bit-exactness, zero hot Rust allocations on replay, `SavedValue` invalidation,
  borrow-conflict rejection, rate-change re-capture, and `invalidate_capture`) + 17 doctests +
  `clippy --all-targets --features cuda,cudnn -- -D warnings` + 9 `compute-sanitizer` runs
  (`memcheck`, `racecheck`, `synccheck` across `reduce_cuda`, `pointwise_backward_cuda`, and
  `autograd_cuda` CUDA Graph capture/replay: **0 errors, 0 race hazards, 0 warnings**).

Paired 5-block BEFORE (`e98ddb1`) vs AFTER vs PyTorch (`2.14.1`) benchmarks on Colab confirm
unchanged explicit resources and full numerical agreement across all 12 training-v1 workloads:
- **Colab Intel CPU (`fa-cpu`, MKL, 1 thread):** **1.169x additional geomean speedup** (cumulative
  ~1.92x over `09fa8f1`): `wave-long` **2.10x** (`1594.56 -> 757.92 us`, beating PyTorch Eager
  `786.51 us`), `wave-multichannel` **1.85x** (`1242.54 -> 672.03 us`, beating Eager `776.97 us`),
  `wave-block` **1.54x** (`10.00 -> 6.51 us`, 9.3x faster than Inductor), `dense-tail` **1.18x**
  (`7.69 -> 6.51 us`, 12.7x faster than Inductor), `dense-tiny` **`1.24 us`** (**55.8x faster**
  than Inductor `68.98 us`). Sweeps PyTorch Eager **12/12** and achieves **`0.296x` geomean vs
  strongest PyTorch Inductor** (**3.38x faster geomean**). `dense-*` and `spectrum-*` remain
  8/8 SHA-256 bit-exact; the 4 `waveform` VML cases stay within **`2.98e-8` max abs diff** (< 1 ULP).
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0):** **12/12 SHA-256 bit-exact** across
  Base == Uncaptured == Captured CUDA Graph. Native CUDA Graph (`--captured`) delivers a
  **1.880x geomean speedup** over `e98ddb1` (`dense-tiny` **3.36x** `49.81 -> 14.84 us`,
  `wave-block` **2.76x** `57.11 -> 20.67 us`, `spectrum-block` **2.75x** `64.33 -> 23.39 us`,
  `spectrum-tiny` **2.73x** `67.52 -> 24.76 us`, `wave-tiny` **2.63x** `60.31 -> 22.97 us`,
  `dense-tail` **2.49x** `52.86 -> 21.25 us`), beating PyTorch Eager **12/12 (4.4x–31.9x)** and
  PyTorch `torch.compile(inductor)` uncaptured **11/12 (2.65x–16.2x)**. Remaining gap vs
  PyTorch `compiled-cuda-graph` (`3.67x` geomean) and large-tensor CPU Inductor (`1.33x–1.65x`)
  is now cleanly isolated to unfused multi-pass pointwise/reduction tape dispatch (~18 separate
  kernels vs Inductor's 1–2 fused Triton/C++ kernels), motivating tape-level kernel fusion in a
  subsequent milestone.

## 2026-11-20: bit-exact CPU/CUDA small-axis reduction, broadcast-backward, and pointwise indexing optimization

Targeted the top measured complete-step bottlenecks from
[training-characterization-results-v1.md](training-characterization-results-v1.md) (CPU
forward reduction 33.14%, CPU binary backward 21–26%, GPU `reduce_warp` 44.8%, GPU
broadcast backward 40.6%) while preserving exact `Tree32`/`Tree256` arithmetic order,
zero extra dynamic workspace/shared memory, and bit-identical snapshots across all 12
declared workloads:

- **CPU (`src/reduce.rs`, `src/pointwise_backward.rs`):** Preselected forward reduction
  kernels (`pair_contiguous`/`pair_linear` for `count = 2`, exact power-of-two `small_tree`
  for `3..=32`, and sequential 256-lane chunked `tree256_contiguous`/`tree256_linear`),
  plus sequential chunked 1D/2D `contract` and mode-hoisted `copy_rows` in
  `pointwise_backward`.
- **CUDA (`native/reduce.cu`, `native/pointwise_backward.cu`, `native/pointwise.cu`):**
  Added 1-thread-per-group `reduce_small_2` (with aligned `float2` coalesced loads) and
  `reduce_small_8`, 16-lane sub-warp `reduce_halfwarp` with full-warp shuffle participation,
  zero-division incremental 2-axis indexing in `reduce_block` and `pointwise_backward::block`,
  and register-hoisted 32-bit/power-of-two fast paths in `pointwise.cu`.

Verified across all three hardware paths under strict Colab CPU/GPU isolation:
local portable tests (107 Rust tests + 16 doctests, Clippy, `rustfmt`, and Miri on both
`reduce` and `pointwise_backward` with 0 UB); Colab `fa-cpu` (`--features mkl,onednn`
full test suite + Clippy); Colab L4 `fa-gpu` (`CUDA_ARCH=89 --features cuda,cudnn` full
test suite + Clippy + 8 `compute-sanitizer` `memcheck`/`racecheck`/`synccheck` runs on
`reduce_cuda` and `pointwise_backward_cuda` with 0 errors and 0 hazards).

Paired 5-block BEFORE (`09fa8f1`) vs AFTER vs PyTorch (`2.14.1`) benchmarks on Colab
confirm 12/12 bit-exact snapshots and unchanged explicit resources:
- **Colab Intel CPU (`fa-cpu`, MKL, 1 thread):** **1.639x geomean speedup** across all 12
  cases (0 regressions; `spectrum-block` **3.30x**, `spectrum-medium` **3.23x**,
  `spectrum-large` **3.16x**, `wave-tiny` **1.60x**, `wave-multichannel` **1.45x**);
  **0.346x geomean vs strongest PyTorch Inductor** (2.89x faster on geomean; beats PyTorch
  Eager on 10/12 cases and beats PyTorch Inductor on 6/12 cases including `spectrum-block`
  at `21.39 us` vs `75.26 us`). Remaining CPU gap is scalar glibc `libm` `tanh`/`exp` in
  `wave-multichannel` and `wave-long`.
- **Colab NVIDIA L4 (`fa-gpu`, SM 8.9, CUDA 13.0):** `spectrum-large` **1.653x** faster
  (`523.05 -> 316.46 us`, 2.09x faster than PyTorch Eager), `spectrum-medium` **1.314x**
  faster (`110.65 -> 84.23 us`, 2.67x faster than PyTorch Compiled), `wave-multichannel`
  **1.275x** faster (`77.73 -> 60.96 us`, 3.60x faster than PyTorch Compiled). Native CUDA
  beats PyTorch Eager on 12/12 cases and PyTorch Compiled (no CUDA Graph) on 11/12 cases;
  remaining GPU gap vs `compiled-cuda-graph` is the ~50 us uncaptured host launch floor
  plus single/two-block memory latency in `spectrum-large`.

## 2026-11-20: prepared strided 1D/2D convolutions, local VJPs and tape training verified

Contract `a8e64a8` ([conv-backward-v1.md](conv-backward-v1.md)) declares borrowed FP32
`ConvStrategy`, `CONV_WORKSPACE_LIMIT`, `cpu::{Conv1dPlan, Conv1dBackward, Conv2dPlan,
Conv2dBackward}`, `cuda::{Conv1dPlan, Conv1dBackward, Conv2dPlan, Conv2dBackward}`, and
`autograd::Builder::{conv1d, conv2d}`. A unified 2D spatial implicit-GEMM engine embeds
1D `[N, C, L]` as `[N, C, 1, L]` so forward, `dX` (`BinaryOperand::Left`), and `dW`
(`BinaryOperand::Right`) share injective writer partitioning with zero atomics, selective
operand dependencies (`[false, true]` for `dX`, `[true, false]` for `dW`), direct
pointwise `Mkl`/`Cublas` (`0` workspace), bounded `<= 57,344`-byte tiled `PackedMkl`,
and single-launch 16x16 shared-memory `TiledCuda` (`2,048` static shared bytes, `0`
device-global workspace, 240-byte by-value descriptor).

Verification passes across all three hardware stacks under strict Colab CPU/GPU instance
isolation: portable CPU (106 Rust tests, 16 doctests, Clippy, `rustfmt`, 54 Python tests,
248 PyTorch `2.14.1+cpu` records with max abs error `7.15e-7` / `2.98e-8` for 8-step
`Tape` SGD, and local Miri signed-stride bounds with 0 UB); Colab CPU `fa-cpu` with
`--features mkl,onednn` (full suite, Clippy, 496 `reference` + `mkl` PyTorch records with
max abs error `7.15e-7` / `2.98e-8`); and Colab L4 `fa-gpu` with `--features cuda,cudnn`
(`CUDA_ARCH=89`, full suite + 17 doctests, Clippy, 248 PyTorch `2.14.1+cu130` records
with max abs error `7.15e-7` / `2.98e-8`, and `compute-sanitizer` `memcheck`/`racecheck`/
`synccheck` reporting 0 errors and 0 hazards on both tape SGD and 65,537-tile grid reuse).

Existing dense inference plans remain unchanged. Transposed 2D forward derivatives,
spectral/normalization/reduction breadth, and performance qualification (including small-axis
forward reductions on CPU/GPU) remain open year-one foundation work.

## 2026-10-20: prepared CPU backward dispatch passes fixed confirmation

Declaration `9d5056f` precedes candidate `04ce7d6`: select operation/operand and
Empty/Copy/Tree32/Tree256 CPU kernels at construction, retaining original formulas,
ordered trees, layouts and public API. Reduction adjoints preselect empty/scalar/row
and scaling paths. A separate ABI budget allows one eight-byte fixed header pointer;
all three measured CPU headers grow by exactly eight bytes, while reported dynamic
resources and warmed Rust allocations remain unchanged.

The [qualified result](cpu-prepared-backward-results-v1.md) preserves one candidate,
zero timing screens and exactly twelve fresh contexts against the original control.
Primary paired speedup is 1.5648 with simultaneous lower bound 1.5341; all twelve
mean slowdown guards pass. The tightest case, `dense-tiny`, has lower bound 0.9969,
above 0.952381 but below parity. Its own speedup is not established. Context/order
variation and post-hoc adjacent-pair sensitivity remain visible without replacing
the frozen df=11 decision or claiming distribution-free/deployment confidence.

Full portable/MKL/CUDA debug/release, Clippy, formatting, two Miri filters, 50 Python
tests, eight experiment guards and the 80-binding/2720-element long special-value
probe pass. Independent numerical/bit/update/reset/resource audits cover 2496 full
validation snapshots and 1728 timed windows; all 330 recorded processes are terminal.
Exact source/build/header executables, compiler comments, transport/payload recovery,
fixed admission, failed early test insertion and all contexts remain archived.

Keep the prepared CPU default and qualified native executable `976483c1...14ddec`,
while retaining producer `3b66357` as the original cumulative guard anchor. Both
negative predecessors and existing operator/model champions remain unchanged. No
fresh Torch or NVIDIA latency is claimed. CPU forward reduction/nonlinear costs,
GPU reduction/replay, required derivative/operator breadth and runtime/memory/stream
contracts remain open; year one continues without new applications.

## 2026-10-20: isolated CPU backward entries fail screening admission

Declaration `cd9ad8a` fixes one isolated-row candidate and its unchanged twelve-case
control. Pre-timing correction `a883707` preserves the unexecuted 32-block proposal
and respects the frozen worker's six-block cap; twelve fresh contexts would follow
only an eligible screen. Candidate `d8e5358` keeps scalar bodies and non-inlines large
row loops, with no formula, contraction-tree, public API or explicit resource change.

The [completed isolation screen](cpu-backward-isolation-results-v1.md) has primary
paired point speedup 1.5376, but `dense-tiny` and `wave-tiny` score 0.9347 and 0.9152,
below the unchanged 0.952381 guard. No selection, revision, retry or confirmation is
performed. These are one-context screening estimates, not confirmed gains or a
significance claim. Every sample, all twelve cases and the earlier unresolved family
remain visible; the original scalar source is restored.

Full portable/MKL/CUDA debug/release, Clippy, formatting, two Miri fixtures, 50 Python
checks and seven experiment guards pass. A supplemental long special-value/signed-
row probe verifies 80 bindings and 2720 gradient elements against scalar policy.
Independent preflight/screen audits check 384 complete snapshots and 144 timed
windows, full FP64/eager/update/bit/repeat/resource gates and zero hot Rust allocations.
All 53 recorded processes are terminal. Read-only disassembly confirms restored
entry stack reservations, which alone do not close the small-case performance gate.

Restoration checks 71 compiled sources against the original control and passes
24 targeted regressions. Next examine prepared operation/strategy selection and
account for plan metadata and tiny-case precision before a new intervention; do not
extend either completed family. Broader derivatives/runtime work stays required,
and year one continues without new applications.

## 2026-10-20: CPU backward gains leave the tiny guard unresolved

Declaration `c94ebaa` fixes the [CPU traversal experiment](cpu-backward-traversal-v1.md)
before implementation: retain the original native binary, all twelve cases, a 1.10
primary speedup requirement, every-case 1.05 slowdown guards, two screens and five
fresh confirmation contexts. Numerical/resource gates precede performance acceptance.

The [completed experiment](cpu-backward-traversal-results-v1.md) preserves two screens,
one size-dispatch revision and all five frozen confirmations. Primary paired speedup
is 1.5571, lower bound 1.5153. Eleven per-case guards pass, but `dense-tiny` has
estimate 0.9803 and lower bound 0.9248, below its unchanged 0.952381 guard. This is
insufficient evidence for that guard, not a demonstrated 8% regression. No extra
samples or relaxed thresholds are used, and the scalar default is restored.

Both candidates pass full portable/MKL/CUDA functional suites, Miri, 50 Python checks,
complete independent/bit/resource controls and zero warmed Rust allocations. The
audit covers 1728 validation snapshots and 1008 timed windows with exact repeats;
all 231 recorded processes are terminal. Separate bounded waveform profiles preserve
identical results after 8192 steps and show address-decoding samples disappearing.
Larger backward entry code and stack reservations remain a hypothesis for later
scalar/row isolation, not proven causal attribution. Sources, exact binaries, failed
bootstrap, raw transports, payloads and compressed traces are retained.

The default training assessor's source-dictionary shadowing bug is also fixed and
checked through complete/default-readiness regression audits. Expanded signed-row
Miri coverage remains after restoring the original three backward/copy source files.
Next, predeclare scalar/row code-generation work and enough tiny-case precision;
preserve the original anchor and this unresolved outcome. Broader foundation work
and the year-one application boundary remain unchanged.

## 2026-10-20: complete-step baseline exposes CPU and GPU foundation gaps

The [training results](training-characterization-results-v1.md) close the frozen
twelve-case characterization for producer `3b6635754637ca81e897e6431ae88695b8c51496`:
216 Intel and 600 NVIDIA warmed complete-step windows, independently evolving
FP64/eager controls, eight-update validation, actual-update ULP checks, exact
reset/re-execution and zero native warmed Rust allocations. Standalone frozen
readiness also passes all twelve cases on both platforms. Source archives match
the committed producer; exact executable, transport, payload and profile artifacts
are recoverable, without an isolated rebuild or memory-peak claim.

CPU wins five cases against the measured Torch modes, but large spectral cases
trail full-graph Inductor by about thirteen times. GPU trails complete compiled
CUDA graph execution on all twelve cases, by 1.7–11.7 times. No AOTInductor/C++ Torch
deployment was measured. These are descriptive baseline observations, not a promoted
optimization, universal language comparison or application milestone.

The original GPU report remains failed solely for Nsight export. The first recovery
detects importer mutation of its original input; exact restoration uses a pre-import
gzip while preserving the modified variant. The successful second recovery imports
copies, reuses the dense trace and runs only the two missing diagnostics, with zero
timing windows rerun. Recovered events verify complete update counts, launch
correlations, one nonoverlapping stream and successful runtime calls. Maintained
tooling now discovers a matching importer and copies its input; a real SDK check and
50 Python tests cover the fix and evidence guards. Large original streams remain
losslessly archived, including sparse patches for modified copies.

CPU profiles identify backward coordinate decoding and forward spectral reduction;
GPU profiles identify many launches, broadcast contraction and small-axis reduction.
Next, predeclare a bounded CPU backward traversal experiment against the frozen native
control, then qualify further runtime and GPU changes separately. Required operator/
backward breadth, allocator/stream contracts and liveness/fusion/replay remain open.
Year one stays dedicated to foundations; no new application is introduced.

## 2026-10-20: complete-step training characterization enters harness verification

Declaration `f023d49` fixes twelve dense/waveform/spectral regimes, complete native
forward/backward/SGD timing, compatible eager/compiled/captured Torch modes, reset and
post-update snapshot semantics, independent controls and no promotion. Resident
workers and collection/build/assessment tools now implement that contract. Raw
requests/replies, exact artifacts, full content-addressed payloads and actual update
consistency are checked separately from descriptive timing summaries.

All twelve native CPU objectives pass eight-update development comparisons against
analytic FP64 and independent eager controls. Small real Inductor regressions cover
all three families, repeated updates, reset/snapshot semantics and corruption guards.
Full frozen Intel/NVIDIA readiness and characterization remain required; no training
latency improvement, memory-peak claim or application follows from these preflights.

## 2026-10-20: native tape and procedural training pass frozen acceptance

The [assessment](../benchmarks/results/2026-10-20-autograd-assessment.json) closes the
declared initial reverse-mode tape/plain-SGD functional and resource campaign for
producer `160070d3332fd90ce88b5e09928d5289a862331a`. Portable, Intel and NVIDIA each
pass complete debug/release regressions: respectively 103/105/152 Rust tests and
16/16/17 doctests per build, plus 41 Python guards per context. All-target Clippy,
formatting, targeted signed/overlapping-view Miri and 24 separate CUDA memcheck/
racecheck/synccheck commands pass.

Independent pinned Torch checks cover 570 raw records in 32 files across both profiles:
all sixteen local-operation compositions under two changed bindings, tied/detached/
overlapping branches with ordered duplicate/ancestor roots, and eight native SGD
updates per dense, multichannel-waveform and spectral-pair objective. Full predictions,
losses, gradients and pre/post-update parameters satisfy the original strict FP32
policy and `2e-5` absolute/relative gates. Reference trajectories evolve independently;
they never rebase to native weights. The spectral fixture does not claim FFT gradients.

All three 155-file source archives match the same committed producer. Exact release
executables, terminal logs, compressed raw exports and process-state receipts pass
hash/recovery audits. This does not include an isolated archive rebuild or exact memory
peaks; targeted instrumentation is not proof for every input or global-memory race.

The [roadmap](roadmap.md#immediate-sequence) now advances to complete-step profiling,
allocator/stream lifetimes and measured liveness/fusion/replay work, alongside required
operator/backward breadth and the existing CPU nonlinear/GPU broadcast losses. Native
training composition is verified for its declared initial scope; full training,
runtime memory optimization and performance qualification remain open. Applications
stay deferred in year one.

## 2026-10-20: native reverse-mode tapes and checked SGD enter composed verification

Following API declaration `862f2b2`, implement shared CPU/CUDA prepared execution for
all current unary/binary local VJPs, Sum/Mean, metadata/detach views and rank-two
matmul. Canonical storage gradients combine tied leaves and overlapping views; ordered
roots and reverse edges support repeated/accumulating backward calls. Selective saves,
forward/update epochs and checked writers guard against stale/partial gradients.
Structural zero contributions preserve existing gradient bits and need no view index
table or logical scratch. Inference graph routes remain in place.

Eleven CPU/MKL and eight RTX 3060 integration tests pass in development. Shared fixtures
cover sixteen operation compositions, changed payloads, independent FP64 analytic/
finite-difference controls, exact repeats and accumulation; dense training checks eight
native updates. Additional cases cover physical aliases/detach, signed seeds, duplicate/
ancestor roots, ownership/errors, real host threads, strided matrix VJPs, virtual-root
resource bounds, zero bits and SGD grid reuse. Four real-plan fault-injection unit
checks verify interleaved forward rejection, partial backward/SGD invalidation and
unwind/overwrite/zero recovery. The executable
CPU example and output-borrow compile-fail check pass. Procedural multichannel waveforms
and five-dimensional spectral-pair tensors each train for eight updates with full
prediction/gradient/parameter checks. Independent pinned Torch composition, branch,
dense and audio-shaped trajectory comparisons pass in development; six checker guards
reject corruption and incomplete/reordered scope.

The [contract](autodiff-v1.md#frozen-functional-campaign) retains complete frozen
backend regressions and both-profile Torch exports, GPU instrumentation and source/executable/log recovery as open
gates. Development checks do not establish full training support, timing superiority,
peak memory acceptance or a new application. Continue this verification before runtime
optimization or widening the operator surface.

## 2026-10-20: strided matrices, both local VJPs and primitive updates verified

The [assessment](../benchmarks/results/2026-10-19-matmul-assessment.json) verifies
producer `b786aaf` after declaration `a64faf9`. Portable/Intel/NVIDIA debug and release
builds pass 88/90/129 Rust tests and 14/14/15 doctests each. Every platform passes
35 Python checks, formatting, relevant all-target Clippy and strict pinned Torch
comparison. Targeted Miri passes; its nightly retains the preexisting graph naming
warning. Eighteen separate GPU memcheck/racecheck/synccheck commands pass with zero
reported errors/hazards. Every recorded collector and command process is terminal.

The 389-case shared catalogue checks forward, both local gradients, accumulation,
changed bindings, affine FP64/finite-difference/adjoint controls, source/hole bits,
repeats, saved/aliased composition, rejected bindings/sessions and bounded resources.
Sixteen recovered raw-bit export archives contain 7,860 matrix/VJP/manual-update
records checked by independent Torch processes, including eight evolving primitive
SGD steps per backend/profile. Three 138-file source archives match the committed
producer; exact compressed executables, command logs and export hashes pass the
independent audit. Development/preflight artifacts are retained separately.

The [matrix contract](matmul-v1.md#verified-implementation) records resource scope,
numerical observations and recovery limits. No isolated rebuild, relative latency,
exact process/device peak, tape, public optimizer or performance promotion is claimed.
The roadmap now advances to native reverse-mode composition, with application work
still deferred and existing measured graph/model routes retained.

## 2026-10-19: prepared strided matrices and local VJPs enter frozen verification

The [matrix contract](matmul-v1.md) has CPU/CUDA implementations for rank-two forward
and either-operand VJPs. Compatible layouts use vendor transpose/leading dimensions;
CPU fallback packs at most 57,344 bytes, while CUDA uses 16x16 shared tiles and no
explicit global scratch. Exact binding, selective payload and zero/accumulation
contracts remain explicit. Existing dense-owned/graph consumers retain their routes.

Seven CPU/MKL tests and six RTX 3060 CUDA tests pass across the 389-case catalogue,
changed bindings, independent FP64/finite-difference/adjoint checks, resource bounds,
source/hole bits, saved/aliased composition and CUDA grid reuse. Independent pinned
Torch processes verify exported forward/both-VJP/accumulation values and eight evolving
primitive SGD updates at the declared gates. Thirty-five Python checks, portable/Intel
all-target Clippy and format pass. Frozen complete backend regressions, Miri, GPU
instrumentation and source/executable/log/export recovery are the next gate. No tape,
public optimizer, latency improvement, performance promotion or application is claimed.

## 2026-10-19: year-one admission and performance qualification made explicit

The [roadmap](roadmap.md) admits first-year work only for reusable framework
capabilities, correctness/resource gaps and measured performance bottlenecks. Early
inference or training success does not reopen model acquisition, task datasets or
application delivery. Functional verification, runtime integration and performance
qualification have separate statuses; profiling begins as each functional gate passes.

Performance qualification advances through operators, reusable subgraphs, whole
inference/training steps and the required suite. Every claim remains tied to hardware,
precision, layouts and the fastest compatible measured reference. Whole-step comparisons
include updates and synchronization consistently; losses and unmeasured regimes stay
visible. This is a planning clarification, with no new runtime verification or speedup
claim. Music, separation and synthesis remain the first later application priorities.

## 2026-10-19: full pointwise VJPs and expanded layout gates verified

The [expanded assessment](../benchmarks/results/2026-10-19-pointwise-backward-layout-assessment.json)
verifies producer `50b16b8` for all twelve unary operations and both binary operands
on the 524-case shared geometry suite. Portable/Intel/NVIDIA debug and release builds
pass 81/83/116 Rust tests and 14/14/15 doctests each. Every platform passes thirty
Python checks, relevant all-target Clippy and formatting. Extended targeted Miri
passes; eighteen separate GPU memcheck/racecheck/synccheck commands pass with zero
reported errors/hazards. All collecting processes and recorded commands are terminal.

Three 129-file source archives match the committed producer; exact compressed test
executables and all command logs pass the independent audit. The assessment binds
the complete initial 496-case predecessor and proves that only four test/map files
changed for the expanded campaign. Native implementation and numerical gates are
unchanged; initial artifacts remain immutable.

The [contract](pointwise-backward-v1.md#verified-implementation) records payload
dependencies, stable derivatives, fused contraction, saved versions, alias composition,
empty/bit/tree policies and tested resources. No relative latency, promotion, exact
memory peak or isolated archive rebuild is claimed. The reverse-mode tape, dense/
convolution/spectral derivatives and native training remain the next foundation work;
new applications remain deferred.

## 2026-10-19: pointwise gradient layout coverage expanded after initial verification

The [initial assessment](../benchmarks/results/2026-10-19-pointwise-backward-assessment.json)
verifies `40c928e` across portable/Intel/NVIDIA debug and release builds: 81/83/116
Rust tests and 14/14/15 doctests per build, thirty Python checks per platform,
Clippy/format, targeted Miri and eighteen separate GPU memcheck/racecheck/synccheck
commands with zero reported errors/hazards. Three 129-file source archives match
the committed producer; exact compressed test executables and command logs pass
the independent audit. All collection processes and recorded commands are terminal.

A coverage review identified a missing combination: read-overlap upstream gradients
together with axis-permuted gradient destinations. Add 28 operation/layout cases,
including short/block-sized broadcast contractions, and extend the targeted Miri
fixture. This expands the shared matrix from 496 to 524 while retaining every
original case and numerical gate. Native implementation bytes are unchanged.
The expanded CPU target and portable Clippy pass. Frozen three-platform regression,
instrumentation and recovery for the expanded matrix are pending; retain the initial
assessment and raw artifacts as predecessor evidence. No performance or training
completion is claimed.

## 2026-10-19: full pointwise local gradients enter frozen verification

The [pointwise-backward contract](pointwise-backward-v1.md) now has CPU/CUDA
implementations for twelve unary operations and both operands of Add/Sub/Mul/Div.
Plans expose selective forward-value dependencies, retain only geometry and fuse
local derivatives with fixed-order broadcast contraction. Stable nonlinear
derivatives retain representable saturated-tail gradients; empty accumulation
preserves all bits. Existing forward kernels and measured graph/model routes stay
in place.

Seven CPU development tests and six RTX 3060 CUDA tests pass. The shared independent
oracle covers 496 operation/geometry combinations with changed bindings, FP64 VJPs,
central differences, both modes, exact repeats and full read/hole preservation.
Separate cases check special/tree/tail rules, errors before writes, saved-version
guards, overlap/aliased-operand view-adjoint composition, real host thread sharing,
bounded metadata and CUDA grid reuse. Portable all-target Clippy and format pass.
Frozen three-platform regressions, Miri, GPU instrumentation and source/executable
recovery are the next gate. No timing advantage, tape or complete training is claimed.

## 2026-10-19: foundation priorities and performance readiness clarified

The user reaffirmed a foundation-only first year. The [roadmap](roadmap.md) now
orders work around a complete single-device forward/backward/update loop, followed
by required audio operator coverage and whole-step runtime optimization. Allocator,
stream and performance work begins alongside the storage and gradient contracts;
quarter labels do not postpone that work or imply completion.

The performance plan identifies short-block launch overhead, waveform/spectral memory
traffic, dense/convolution dispatch, training-step storage and chunked state as
separate targets. Compatible compiled/captured Torch modes remain required controls.
Functional readiness and competitive readiness have separate exit statuses, with
losing and unmeasured regimes visible. This records planning changes, not new runtime
verification or a performance result. New model and application work remains deferred
until year one ends and the foundation gates pass.

## 2026-10-19: strided Sum/Mean and local adjoints verified on CPU/CUDA

Producer `1c87db4`, following declaration `907b931`, completes the
[reduction contract](reduction-v1.md#verified-implementation). Portable/Intel/NVIDIA
debug and release builds pass 74/76/103 Rust tests and 14/14/15 doctests each.
Every platform passes thirty Python checks, formatting and relevant all-target
Clippy with warnings denied. Targeted Miri passes the new unsafe signed-stride paths.
Its nightly retains the preexisting graph naming deprecation; stable Clippy is clean.

The new targets check 80 geometry/keep-policy fixtures under Sum/Mean and two payload
bindings, independent FP64 forward/local-gradient references, adjoint/finite-difference
identities, CPU/CUDA bit agreement, source/holes/repeats, empty rules and both gradient
modes. Separate cases verify cancellation/axis order, one-contribution special bits,
the rounded count above `2^24`, actual zero-storage inputs, host sharing, zero hot Rust
allocation, bounded large preparation and view-adjoint composition for both overlap
and broadcast. GPU tails reuse warp/block grids.

All four GPU tests pass separately under memcheck, racecheck and synccheck: twelve
commands with zero reported errors/hazards. The original NVIDIA collection's offline
Clippy configuration failure is retained. A separately recorded continuation fixes
argument placement and executes the missing checks using the already frozen binaries;
completed Rust regressions are reused, with unchanged gates and original deadline.
All collecting processes and recorded child commands are terminal.

The [assessment](../benchmarks/results/2026-10-19-reduction-assessment.json) verifies
three 122-file committed source inventories, command logs, failed/continued lineage
and exact compressed executables. No isolated rebuild, fresh model collection,
relative latency, exact memory-peak acceptance or promotion is claimed. Pointwise
local derivatives/broadcast contraction come next, followed by the tape and native
training. Broader reductions and dtype/runtime gaps remain; applications stay deferred.

## 2026-10-19: ordered reductions and local derivatives enter frozen verification

Implementation follows declaration `907b931`: explicit-axis FP32 Sum/Mean forward
plans and payload-free local backward plans on CPU/CUDA. Prepared metadata preserves
signed/read-overlap geometry and reduced order, with fixed 32/256-lane trees, exact
one-contribution copies, empty-contraction rules and injective gradient accumulation.
No existing graph/model kernels or pointwise baseline artifacts are replaced.

The new six-test CPU target passes, including 80 geometry/policy fixtures under both
operations, independent FP64/adjoint/finite-difference checks, special/tree/count
boundaries, actual sharing and zero hot Rust allocations. The initial four-test CUDA
preflight passes on the RTX 3060. Relevant portable Clippy/format checks pass. These
are development checks; final frozen backend regressions, Miri, GPU instrumentation,
source/executable recovery and their evidence assessment are pending.

The [reduction contract](reduction-v1.md) keeps that verification boundary explicit.
Pointwise/dense/convolution/spectral derivatives, the reverse-mode tape, broader
reduction families and native training remain required foundation work. There is
no timing claim, performance promotion or new application in this increment.

## 2026-10-19: complete pointwise baselines expose nonlinear and layout gaps

Both frozen `5d42e35` contexts finish without restart: 56 cases, 1,008 Intel and
2,800 NVIDIA timing windows. The [results](pointwise-characterization-results-v2.md)
retain every original FP64/input/hole/repeat gate, all execution modes and per-case
losses. All controllers/workers are terminal. The independent assessor verifies
complete scope, artifacts and recorded gates; a separate audit matches both
115-file committed source inventories and every worker transcript to the receipts.
The complete 336-row table and all-case figure are archived and visually checked.

Descriptive equal-case Torch/native wall ratios are 1.33895 / 3.91698 for Intel
eager/compiled, and 1.58719 / 9.11930 / 1.07640 / 1.04702 for NVIDIA
eager/compiled/eager-graph/compiled-graph. These averages hide material losses:
CPU bulk-log is 0.15082 against eager, transposed-write sigmoid is 0.10802 against
compiled, and GPU channel broadcast is 0.25395 against compiled CUDA Graph.
The observations do not establish independent confidence or a framework-wide win.

All five CPU perf commands complete with zero reported lost samples; bulk Softplus
spends about 79.08% self cycles in `logp1f32`. The five GPU attempts emit original
QDSTRM but automatic import fails because the multiarch CLI misses the installed
host importer. Same-version postprocessing recovers all five existing captures,
without rerunning a workload. Compressed streams/reports/SQLite and command logs are
sealed; an independent query verifies 1,000 correlated kernels/launches per capture.
Instrumentation overhead remains separate from uninstrumented latency.

This increment changes no runtime sources, accepts no optimization and leaves all
champions/older guards intact. No raw-array recomputation, isolated rebuild, exact
memory peak, training or application claim follows. Next work is strided reductions
and local derivatives, with bounded CPU math/GPU broadcast-layout mechanism tests
alongside the foundation sequence. Applications remain deferred for year one.

## 2026-10-19: compiled strided writeback passes both hardware harnesses

Producer `5d42e35`, following declaration `ecea37d`, replaces the unsupported direct
compiled out-parameter path with functional math and caller-view `copy_` in one full
graph. Eager remains an out-parameter route; compiled capture uses the same prepared
writeback. All 56 cases, layouts, numerical gates, windows and mode requirements stay
fixed. V1 failures remain preserved, and the active runner accepts v2 only.

Portable regression passes 68 Rust tests, thirteen doctests, thirty Python checks,
Clippy and formatting. Both release builds pass the same 24-case transport/readiness
validation: 72 Intel and 120 NVIDIA validation-only windows retain input bits, output
holes and exact same-mode repeats under the original FP64 gate. Both controllers and
workers are terminal. The [v2 harness evidence](pointwise-characterization-v2.md#verified-harness-and-completed-collection)
verifies 115-file committed snapshots, exact executables and sealed logs. CPU/GPU
capability probes also cover all eight gapped/transposed writeback cases; GPU captured
repeats pass. No eager fallback or reduced suite was admitted.

The full 56-case Intel/NVIDIA characterization contexts are now running from the frozen
producer. Validation windows provide no performance result, and characterization,
profiling and assessment remain pending. Existing champions stay unchanged; no speed,
memory-peak or training claim follows from this frontend/harness milestone.

## 2026-10-19: pointwise baseline validation preserves a compiler capability failure

The pointwise measurement workers in `a93acc4`, declared in `a9d9d35`, pass portable
68-test / thirteen-doctest regression, 26 Python checks and Clippy/format checks.
Both hardware-specific release builds complete. During validation-only transport and
readiness checks, both hosts stop at `gapped-add`: Dynamo rejects the non-contiguous
out-parameter in the required compiled path. Native readiness passes; the Torch worker
exits with the original exception. Both controllers and workers are terminal.

The [v1 outcome](pointwise-characterization-v1.md#validation-outcome-direct-compiled-out-parameter-path-unavailable)
retains nineteen completed validation cases per host and the same failed twentieth
case, along with both 114-file producer archives, exact binaries and sealed logs.
No partial-suite speed summary, formal performance collection, removed case, relaxed
gate or eager fallback is admitted. Existing performance champions remain unchanged.
The next step is a separately declared compiled writeback route that can preserve
strided outputs, followed by full-scope characterization; training and applications
remain outside this measurement increment.

## 2026-10-18: general strided unary/binary forward operators verified

Implementation `7a637c7`, following declaration `12e1d2a`, adds prepared CPU/CUDA
pointwise plans: twelve unary operations and ordered add/subtract/multiply/divide.
Binary broadcasting is explicit and fixed in preparation. Output-directed axis order,
negative-axis normalization and joint coalescing preserve signed-stride/read-overlap
geometry while keeping dense/broadcast CPU rows separate from general traversal.
CUDA specializes the operation and flat path outside the element loop, passing flat
geometry as parameters and retaining only additional O(rank) coordinate descriptors.

Portable/Intel/NVIDIA suites pass 68/70/93 Rust tests and 13/13/14 doctests; both
hardware hosts pass nineteen Python checks. Relevant Clippy/format checks pass.
Independent coordinate/FP64 references cover 79 CPU and 27 GPU geometry cases across
all sixteen operations, two changed bindings, source/hole preservation and exact
same-backend repeats. Original `2e-6` absolute / `3e-6` relative forward gates remain.
Separate cases verify stable extremes, exceptional categories, signed zeros and exact
Neg/Abs/Relu bits. Miri passes negative/gapped loops with overlapping read inputs.

All four CUDA tests pass separately under memcheck, racecheck and synccheck, with
zero reported errors/hazards. They include a grid-stride tail beyond the launch cap.
Tested prepared execution allocates zero Rust heap storage; flat CUDA coordinate
workspace is zero and large logical preparation has no element-sized index table.

The [pointwise evidence](pointwise-v1.md#verified-implementation) verifies both
106-file producer archives, all command logs and the exact compressed instrumented
binary. Both archived backend collections completed without retry. The published
[coverage matrix](operator-coverage.md) keeps missing derivatives visible. There is
no isolated rebuild, fresh model collection, timing comparison, exact memory-peak
acceptance or performance promotion, and the measured legacy graph kernels remain.

Next work is matched pointwise performance characterization and profiling, strided
reductions/local derivatives, then reverse-mode tape and training execution.
Application work remains deferred under the foundation-first roadmap.

## 2026-10-18: first prepared CPU/CUDA backward primitive verified

Implementation `af7fdc2`, following declaration `88c2500`, adds prepared view
adjoints with explicit overwrite and accumulation modes. Signed strides, permutations,
broadcasts and arbitrary read overlap map gradients back to dense base storage.
Analytical broadcasts use geometry proportional to rank; general overlap uses bounded
prepared indices. CPU/CUDA reductions use the same fixed 32/256-lane tree, with one
writer per selected target and no floating-point atomics or per-run index preparation.

Portable/Intel/NVIDIA suites pass 62/64/83 Rust tests and 12/12/13 doctests. Both
hardware hosts pass nineteen Python checks; relevant all-target Clippy and format
checks pass. Independent FP64 coordinate oracles, adjoint identities and central
finite differences check bounded fixtures under the original `2e-5` absolute/relative
gate. GPU cases agree with the CPU bits and repeat exactly on the tested stack.
Targeted Miri passes the signed-stride addition path.

The frozen GPU test executable passes all four CUDA cases separately under memcheck,
racecheck and synccheck, with zero reported errors/hazards. These include warp/block
reductions and grid-stride reuse. Tested warmed execution allocates no Rust heap
storage; index-budget boundaries and billion-element broadcast preparation without
element index tables pass.

The [view-adjoint evidence](view-backward-v1.md#verified-implementation) verifies
both 101-file source archives, command logs and the exact compressed instrumented
binary against the producer. Both collections completed without retry. This closes
the preceding retained-value entry's executable view-adjoint gap, not the full autodiff
foundation. No isolated archive rebuild, fresh model collection, latency comparison,
exact memory peak or performance promotion is claimed.

Next work is general strided math, followed by the native reverse-mode tape and
training execution, with explicit forward/backward coverage and measured dispatch.
Application work remains deferred under the foundation-first roadmap.

## 2026-10-18: retained host/device values and guarded shared storage verified

Implementation `490737b`, following declaration `45e980d`, adds `Shared<T>`,
`SavedValue<T>` and nonblocking read/write guards. Saves retain a strong owner,
immutable prepared layout and version without a payload copy or persistent lock.
Later writes invalidate them. Active readers/writers enforce borrow conflicts before
exposing data, while successful writers advance the generation once. Write guards
do not expose a new current-version key before their remaining mutations finish.

Portable/Intel/NVIDIA checks pass 56/58/73 Rust tests, 11/11/12 doctests, and nineteen
Python checks per hardware host. Relevant all-target Clippy and format checks pass.
Actual concurrent host borrowers, poison recovery, unique extraction, initialized
bounds, prepared CPU/CUDA consumers, queued device lifetime and thread restrictions
are covered. Captures/clones/guards and tested warmed consumers allocate no Rust heap
storage; ordinary tensor/error size budgets remain unchanged.

The [retained-value evidence](saved-values-v1.md#verified-implementation) preserves
both 96-file source archives and all command logs, checked against the producer.
Both detached collections completed without retry. These are functional/resource
checks, with no new model integration, isolated archive rebuild, latency comparison,
exact memory peak or performance acceptance.

The [view-adjoint contract](saved-values-v1.md#view-gradients-required-by-a-future-tape)
requires repeated-address accumulation, untouched/empty semantics and separate
metadata-only dependencies. Its backward kernels and gradient checks are not yet
implemented. Next work is executable view adjoints and strided math, then the native
reverse-mode tape and training execution. Application work remains deferred.

## 2026-10-18: opt-in tracked ownership and mutation versions verified

Implementation `1bc77b7`, following contract declaration `74ddd0c`, adds
`Tracked<Tensor>` and `Tracked<DeviceTensor>`. Each owner has a non-recycled identity
and advances its generation before mutable access. Saved checks reject both foreign
owners and modified generations, including potential writes that fail, touch no bytes
or unwind. Ordinary inference tensors have no new tracking fields or hot-path atomics.

Portable/Intel/NVIDIA checks pass 47/49/62 Rust tests, seven doctests per configuration,
and nineteen Python checks per hardware host. Relevant all-target Clippy checks pass
with warnings denied. Independent access histories, concurrent IDs, layouts, actual
queued GPU writes, snapshots, hot allocations and size/exhaustion budgets are covered.
A 40-byte draft error layout failed its resource regression and was replaced by compact
structured variants that preserve the original 32-byte budget.

The [tracked-storage evidence](storage-versions-v1.md#verified-implementation) retains
the interrupted Intel compile and NVIDIA observer pipe failure in `feec4d6`, alongside
the original runner. Both initial workers were confirmed terminal before continuation.
NVIDIA reuses its verified Rust/cuDNN Clippy results; Intel completes the previously
unfinished suite. Both completed receipts verify their archived 95-file producer
snapshots and logs. There is no new latency, peak-memory or performance acceptance.

Next foundation work is retained saved values and view-gradient contracts, followed
by strided math consumers and reverse-mode execution. Shared owning aliases, allocator
and stream reuse, automatic parameter refresh and actual training remain incomplete.
Year-one application work stays deferred.

## 2026-10-18: borrowed-layout evidence and source recovery completed

Implementation `8b89df6` passes the existing complete-graph regression on both
backends: four final cases and sixty intermediate outputs per host, with all 144
recorded numerical checks inside the original tolerance. Both collections freeze
the same 93 source hashes and preserve the exact compressed executables. The GPU
check also verifies the requested deterministic convolution plan metadata.

Restored isolated trees pass 37/48 Intel/NVIDIA release tests, three doctests and
nineteen Python tests per host. Intel's first evidence runner resolved the virtualenv
entry point to base Python and failed to import Torch; the failed receipt/logs and
original runner are preserved in `610054c`. After correcting that runner path, the
specified virtualenv completes Python recovery; its receipt retains the verified
Rust log by hash. No numerical tolerance changed or model collection was repeated.

The [layout evidence](tensor-layout-v1.md#verified-implementation) and
[assessment](../benchmarks/results/2026-10-18-layout-views-assessment.json) link both
archives, binaries and recovery chains. The assessment verifies metadata and recorded
checker results; omitted NAS raw outputs are not recomputed locally. Existing
champions remain unchanged, with no timing or memory-peak acceptance. Next work is
allocation identity, modification generations and training-aware storage ownership;
application work remains deferred under the foundation-first roadmap.

## 2026-10-17: signed-stride layouts and borrowed host/device views

Implemented the [layout/storage contract](tensor-layout-v1.md), declared in `6a9176a`.
Logical geometry is prepared separately from dense owned storage. Host/device view
binding borrows that metadata without allocation or payload movement; read broadcasts
may overlap, while mutable binding requires an injectivity proof and exclusive storage.
Permutation, counted signed-step slicing and compatible chunk reshape support ranks
beyond the legacy graph ABI. Explicit copy plans coalesce host rows or prepare CUDA
coordinate metadata once, preserving all FP32 bits and untouched destination gaps.

Scalar and empty shapes are now distinct. Empty CUDA tensors own no allocation;
empty transfers/copies skip payload work. Dense empty GEMM avoids vendor calls and
zero contractions overwrite outputs with zeros. Existing positive convolution,
spectral and recurrent contracts remain; prepared graphs explicitly reject unsupported
scalar/empty values and normalization suffixes.

Portable/Intel/NVIDIA tests pass 35/37/48, plus three doctests and nineteen Python
tests per host. Relevant all-target Clippy checks pass. Independent coordinate-set
and affine-feasibility tests cover address/reshape behavior; targeted Miri runs pass
mutable reference iteration and negative/broadcast copy paths. The installed nightly
warns about the existing graph identity `fetch_update` name; stable checks are clean.
The subsequent entry above closes complete-graph/source-recovery verification.
Training generations, shared owning views, allocators/pools and strided math operators
remain subsequent foundation work. No application integration or performance
promotion is implied.

## 2026-10-17: explicit cuDNN convolution selection policy

Implemented an immutable per-session convolution policy with a permissive default
and explicit deterministic requirement. Both fused and unfused cuDNN searches keep
the strict-FP32 exclusions and enforce the requested determinism note. Conv1d,
spatial/transposed convolution and graph-contained convolutions inherit it. Actual
selected engine/notes/workspace/fusion metadata is exposed without hot-loop queries;
absence of a compatible plan is an explicit unsupported capability.

Portable/Intel/NVIDIA tests pass 24/26/35 and both hosts pass 19 Python checks. The
GPU oracle/rebinding/allocation cases exercise both policies and same-plan exact
repeats. Relevant portable/Intel/CUDA/cuDNN all-target Clippy passes. The bounded
[cross-process validation](cudnn-repeatability-v1.md), declared in `4d088b5` before
implementation `343b3be`, completes all twelve commands. Its 864 original numerical
checks pass; deterministic mode passes all 288 exact trace/final-output comparisons
at zero and twenty warm-ups. Default mode retains tiny differences in both regimes,
with two flagged convolution plans per case; deterministic mode selects none.

Engine metadata remains stable within each context and planned graph storage is
equal across modes. The independent audit recomputes all original numerical checks
and exact output comparisons from raw bytes and hash-verified references. All
thirteen archives verify against the same 85-file source inventory. Isolated offline
recovery passes 26/35 Intel/NVIDIA release tests and 19 Python tests per host. The
exact collected binary is preserved compressed.
This establishes the tested repeatability context, without performance inference,
exact memory peaks or cross-stack guarantees. The prior failed GPU screen remains
failed and champions are unchanged. Tensor/view/alias and training-aware memory
contracts are the next foundation work; no new application has been added.

## 2026-10-17: first year reserved for framework foundations

The user's latest direction supersedes the application-led sequence below. The
[roadmap](roadmap.md) now reserves year one for tensor/storage/alias and stream
contracts, forward/backward operators, autodiff, graph/memory planning, training
mechanics and Intel/NVIDIA performance engineering. Gradient and parameter-version
contracts begin early rather than being added after inference planning is fixed.

No new model integration, task pipeline or application demo is scheduled in year one.
Music processing, separation and synthesis remain the first later application wave;
other audio domains remain important. The existing GTCRN implementation stays as a
regression fixture. The completed DAC inventory is retained as deferred research,
without downloading a checkpoint or implementing its decoder.

Performance acceptance now explicitly spans operator/subgraph suites and the existing
complete graph, matched execution modes, forward/backward correctness, resource limits
and fresh confirmation. The ambition to outperform PyTorch remains; performance
claims require measured results. The first concrete work is GPU repeatability
policy and tensor/memory/autodiff contracts, with the Intel guard still unresolved.
The original `target.md`, historical protocols, failed GPU result and champions stay
unchanged. This entry records a direction change, not completion of those foundations.

## 2026-10-17: prepared mapped addressing and synthesis inventory

Implemented coalesced CPU rows and checked u32 reciprocal CUDA addressing for graph
layout/broadcast multiplication. The generic 64-bit CUDA route remains for larger
mappings. Floating-point equations and planned tensor storage stay fixed. Independent
coordinate-sort/broadcast oracles cover all permutations, singleton/sliced/tail shapes,
NaN/signed-zero bit preservation, rebinding and warmed allocation behavior. Reciprocal
extremes/random inputs and large-index fallback selection are checked separately.

The protocol was declared in `6cb3ec3`, before implementation `e176704`. Intel completes
all 28 pairs with bit-identical final outputs and unchanged planned storage. The paired
four-case geometric mean is 1.286, approximately 22.2% less latency against the retained
native control, with exploratory block interval [1.261,1.311]. This screen does not
provide independent confirmation or a new matched Torch result.

The GPU context stops at its second pair: two enhanced-spectrum components differ by
one ULP, at most `1.863e-9`, while waveform bits and planned storage match. The original
numerical gate passes, but the declared exact-output gate fails. Preserve that failure;
no GPU aggregate, memory replay, replacement run or relaxed gate was admitted.
A separately declared four-command correctness probe finds tiny fresh-process output
differences in both the unchanged old binary and the candidate, first visible at the
same decoder transposed-convolution/PReLU block. It establishes repeatability limits in
that diagnostic regime, not an identified kernel bug or repeatability at all warm-ups.
The next GPU campaign must explicitly address that policy. Existing champions remain.

Portable/Intel/NVIDIA tests pass 24/26/34, with 16 Python tests on each host. Relevant
Clippy configurations pass. Both hosts validate all four final pipelines and 60 model
traces against the original tolerance from identical 81-file source inventories.
Six candidate/control archives verify; restored candidate snapshots rebuild offline
and pass 26/34 release tests plus 16 Python tests each. Both compressed binary pairs
recover their exact measured hashes. [Screen/diagnostic evidence](mapped-access-screen-v1.md)
and the [audit](../benchmarks/results/2026-10-17-mapped-access-assessment.json) retain
the successful Intel screen and failed GPU context together.

Before the foundation-only direction, pinned DAC source at
`c7cfc5d2647e26471dc394f95846a0830e7bec34` and verified its default decoder with real
Torch meta-device modules: 26 Conv1d, four ConvTranspose1d, 29 Snake1d and 54,104,162
parameter elements. Four meta forwards verify the `[B,1024,F]` to `[B,1,512F]` geometry.
The [DAC inventory](dac.md) records per-channel Snake, effective weight export and
checked transpose lowering as possible later requirements. It is shape/source evidence;
no checkpoint, real waveform, stereo-generation quality or native DAC inference has
been validated. All further DAC/model integration is now deferred; low-level state
and training mechanics remain foundation work.

## 2026-10-17: audio application priorities confirmed

The user initially confirmed a general audio framework with music processing, source
separation and audio synthesis as first delivery priorities. The first-year sequence
was subsequently superseded by the foundation-only direction above; these remain
long-term priorities. Other audio domains remain important and non-audio workloads
remain outside the main focus. GTCRN remains the numerical/performance regression anchor.

## 2026-10-17: complete-model latency and memory characterization

Completed the protocol declared in `a7caf75`: Intel eager, NVIDIA eager and the
explicitly prepared Torch CUDA Graph baseline, each with seven counterbalanced blocks
over four complete waveform pipelines. All 84 pairs and 16 separate memory workers
pass the original full-output numerical gate. Their 804 recorded tensor checks and
every worker's counts/features/order are independently audited. No native optimization
candidate was evaluated, no threshold changed and no operator champion was promoted.

| Context | Equally weighted paired Torch/native latency ratio | Exploratory 95% interval |
| --- | ---: | --- |
| Intel / eager | 0.833 | [0.809, 0.857] |
| NVIDIA / eager | 4.217 | [4.094, 4.396] |
| NVIDIA / prepared Torch CUDA Graph | 0.520 | [0.511, 0.532] |

The GPU eager advantage does not carry over to the prepared capture baseline. The
unmodified `torch.istft` capture probe failed and remains recorded; the separately
declared prepared inverse caches/checks its fixed NOLA envelope before capture and
passes independent inverse tests. These are fixed-fixture characterization contexts,
not framework promotion or speech-quality acceptance. Intel's absolute timings vary
substantially despite fixed affinity and threads; frequencies/load were observed,
not continuously controlled. Keep all blocks and require fresh paired confirmation.

GNU time records whole-command host RSS separately from native planned workspace.
For the four-second input, sampled GPU process high-water values are 454 MiB native,
372 MiB eager Torch and 260 MiB prepared Torch Graph. Actual median NVML intervals
are approximately 50–59 ms; the values include preparation/context and are lower
bounds that can miss allocation peaks, not exact peak-memory acceptance evidence.

Separate one-control/one-profile diagnostics pass numerical checks on both hosts.
Intel cycle sampling assigns 72.91% self-period weight to `Host::execute`, including
generic tensor kernels. NVIDIA's last 100 pipelines contain 264 kernels per call;
generic tensor kernels account for 50.0% of summed kernel time and persistent GRU
for 30.6%. Unioned GPU activity occupies about 84.9% of the traced device envelope.
There are no host/device transfers, memsets or CUDA allocation APIs in that warmed
envelope. Nsight's missing importer was supplied from the checksum-verified matching
official package; the existing trace was parsed without repeating collection.

The next mechanism test should specialize generic indexing/layout execution before
another latency campaign. Removing all observed GPU gaps alone leaves more recorded
GPU activity than the prepared Torch Graph latency; launch reduction cannot explain
the entire gap under unchanged kernel work. Buffer lifetime planning remains a separate
memory opportunity. Licensed speech quality, exact peaks, additional compiled/vendor
baselines, streaming and training remain open under the original framework objective.

All five source archives verify against the same 78-file inventory. Restored Intel and
NVIDIA snapshots rebuild offline and pass 22/29 Rust release tests plus 15 Python tests
each. Compressed copies preserve both exact measured binaries and raw CPU/CUDA traces;
hashes verify after decompression. This recovery claim remains functional rather than
bit-identical rebuilding. Complete results, raw evidence and analysis commands are in
[the characterization report](gtcrn-characterization-results-v1.md) and its
[recomputed audit](../benchmarks/results/2026-10-17-gtcrn-characterization-assessment.json).

## 2026-10-17: complete native GTCRN inference

Connected the full offline waveform path on Intel and NVIDIA: exact-window centered
STFT, complete GTCRN and inverse STFT. A prepared fixed-shape graph validates builder-owned
value identities, edges and geometry; snapshots parameters; and retains reusable buffers.
CPU/CUDA backends cover layouts, explicit broadcasting, linear projections, padding,
frequency neighborhoods, mean-square reductions, layer normalization and complex audio
operations alongside the existing convolution/GRU plans. CUDA activations stay on its
owned stream/device. Graph input/output remain caller-owned and recurrent states are
explicit. GTCRN initializes those states to zero for every offline call.

The architecture adapter lowers ERB analysis/synthesis, SFE, temporal attention, both
dual-path GRNNs, shuffles, decoder skips and complex masking. Inference BatchNorm is
folded into prepared forward/grouped-transposed weights and bias. A hash-locked export
provides named FP32 parameters without linking PyTorch into runtime execution.

Both hosts pass all four complete waveform cases and all 60 captured model-block
outputs under the unchanged numerical gate. Maximum enhanced-spectrum absolute error
is `2.01e-5` on Intel and `4.47e-4` on RTX 3060; final-waveform error is at most `1.57e-7`
and `2.38e-6`. The largest error/tolerance ratios are 0.056 and 0.753, respectively.
Unlike the earlier isolated checks, intermediate model stages consume native outputs.
Changed parameter bytes are rejected before native execution; source and binary digests
are frozen during each complete validation.

- [Complete Intel validation](../benchmarks/results/2026-10-17-gtcrn-native-mkl/validation.json).
- [Complete NVIDIA validation](../benchmarks/results/2026-10-17-gtcrn-native-cudnn/validation.json).
- [CPU parameters/traces](../benchmarks/results/2026-10-17-gtcrn-model-cpu-reference/reference.json)
  and [CUDA parameters/traces](../benchmarks/results/2026-10-17-gtcrn-model-cuda-reference/reference.json).

Portable tests pass (20), Intel release tests pass (22) and NVIDIA release tests pass
(29). Fifteen independent FP64 kernel cases plus explicit-state, immutable-parameter,
value/session rejection, rebinding and allocation checks cover the graph contract.
Clippy passes for portable, MKL, oneDNN, combined Intel, CUDA and cuDNN builds.
Eight Python regression tests pass. All four new source archives pass hash/inventory
verification; restored native Intel/CUDA snapshots rebuild and pass 22/29 tests plus
all eight Python tests on their respective hosts. Maps now cover the graph/model
adapters and shared kernels. Earlier numerical and performance controls remain active.
Combined Intel/NVIDIA SDKs on one host remain untested.

The later NVIDIA regression build changed the executable digest with all recorded source
hashes unchanged; its environment companion preserves that observation. The collected
report/archive remain immutable. Byte-identical CUDA rebuilds are not established, and
any future latency campaign must preserve copies of its exact measured binaries.

Native timing is exploratory (two warm-ups, three samples of five calls), with no matched
Torch measurement or fresh confirmation. Reference captures use one sample for fixtures.
The executor retains every intermediate for tracing: reported graph workspace reaches
about 81.5 MB on Intel and 249.9 MB on GPU for the four-second input, excluding caller
tensors and unreported vendor allocations. The spectral field counts inverse storage
only; complete observed peak memory remains open. The next work is a declared paired
full-model performance/quality protocol, profiling and memory/transfer accounting,
followed by justified fusion or buffer reuse. Licensed speech/noise quality, streaming
state and training remain part of the original framework objective.

## 2026-10-17: exact model windows and spatial convolution

Added explicit snapshot windows to CPU/CUDA STFT, centered STFT and inverse STFT.
GTCRN now uses its exact captured FP32 square-root Hann coefficients, including the
matching inverse NOLA envelope. Prepared NCHW forward/transposed 2D convolution covers
groups, dilation, bias/ReLU and output padding. CPU has reusable GEMM/column scratch,
oneDNN forward/deconvolution and direct MKL pointwise plans. CUDA snapshots parameters,
uses strict-FP32 cuDNN graphs or pointwise cuBLAS, and keeps execution on its session stream.
Valid output padding at least as large as stride uses reusable high-edge input zeros;
the CUDA workspace limit includes that storage and graph scratch together.

Fresh references capture every convolution without changing earlier artifacts.
All 88 actual layer/shape instances pass on each host under the original full-output
tolerance. Maximum absolute error is `3.82e-5` on Intel and `2.39e-6` on NVIDIA.
All four exact-window spectral cases also pass; enhanced inverse waveform error is
at most `1.50e-8` and `2.99e-8`, respectively. Each layer consumes its captured input,
and inverse synthesis consumes the captured enhanced spectrum. The model middle is
still a PyTorch reference; this evidence establishes isolated native stages.

- [Intel stage validation](../benchmarks/results/2026-10-17-gtcrn-stages-mkl/validation.json).
- [NVIDIA stage validation](../benchmarks/results/2026-10-17-gtcrn-stages-cudnn/validation.json).
- [CPU captures](../benchmarks/results/2026-10-17-gtcrn-conv-cpu-reference/reference.json) and
  [CUDA captures](../benchmarks/results/2026-10-17-gtcrn-conv-cuda-reference/reference.json).

Independent FP64 gather/DFT and analytic inverse tests cover rectangular, grouped,
depthwise, dilated and transposed convolution; signed/asymmetric windows; parameter
snapshots; exact axes; NOLA; rebinding/session rejection; workspace bounds; and warmed
Rust allocation reuse. Portable tests pass (17), combined Intel release tests pass (19),
and RTX 3060 release tests pass (25). Clippy passes for portable, MKL, oneDNN, combined
Intel, CUDA and cuDNN builds. Eight Python tests pass. All four new source archives
pass hash/inventory verification. Restored Intel and CUDA stage snapshots rebuild
and pass 19 and 25 tests plus all eight Python tests on their respective hosts.
Shared replay tensor I/O now has one owner under `examples/support`. Combined
Intel/NVIDIA SDKs on one host remain untested.

Reference capture timing uses one sample solely for additional fixtures. Native
layer timings are exploratory, with no matched Torch measurements or fresh confirmation;
the earlier champion identities remain unchanged. ERB projections, normalization,
activations, reductions, layouts/shuffles and complex masking must now connect these
stages into the full native model. Speech-quality evaluation, end-to-end latency/memory,
streaming behavior and training remain open parts of the original objective.

## 2026-10-17: real-model reference and prepared GRU

Pinned the official GTCRN MIT source at `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`
and its DNS3 checkpoint SHA-256. The complete PyTorch waveform/STFT/model/inverse path
runs on both recorded hosts. Four deterministic tone/noise cases preserve full outputs,
104 leaf-module shapes and 14 actual recurrent layer inputs/parameters/states each.
These are reproducible execution fixtures, not a speech-quality corpus.

Separate CPU profiling attributes approximately 73–81% of recorded CPU time to GRU
calls. Added one-layer reset-after FP32 GRU with explicit state, bidirectional output,
immutable prepared weights and reusable scratch. CPU batches input projections and
uses reference/MKL recurrent projections. CUDA packs owned cuDNN weights and exposes
standard/automatic algorithms plus batch/sequence internal layouts. Public tensors stay
batch-first, and execution includes layout conversion. The default uses persistent
small-hidden inference for SM 6+ and `H <= 192`, with sequence-major internal storage.

All 56 real layer/shape cases pass against captured outputs/final states and newly
replayed PyTorch modules on both hosts. Maximum absolute discrepancies across all
checks are about `8.4e-7` for MKL and `9.4e-6` for CUDA. Independent FP64 tests also
cover nonzero state, reverse indexing, odd dimensions, source-weight mutation, explicit
chunk continuation, session rejection and no Rust allocations after warm-up.

The first standard GPU route was much slower than PyTorch on long sequences. Two
planned screens isolated persistent small-hidden selection and then sequence-major
layout. For the 64,000-sample attention GRU, native median latency dropped from roughly
1.7 ms to 0.10 ms. Full-model advantage remains unproven: these sequential layer timings
are diagnostic, with no paired confirmation or framework promotion. All initial results
and source archives are preserved in [GTCRN evidence](gtcrn.md).

Current Intel release tests pass (16 total), NVIDIA release tests pass (20 total),
and both feature builds pass Clippy. Eight Python tests pass. The selected sequence-major
source archive passes checksum/inventory verification and was restored on both hosts:
its Intel build passes 16 tests, its CUDA/cuDNN build passes 19, and both pass all eight
Python tests. The current extra GPU test covers explicit controls and the vendor sequence
limit added after collection. A corrupted checkpoint is rejected before deserialization.
Combined Intel/NVIDIA SDKs on one host remain untested. Shared allocation/fixture helpers
now have one owner under `tests/support`; earlier benchmark controls remain active.

The native model still needs its exact sqrt-Hann frontend, ERB and 2D convolution/
transpose paths, normalization, activations, shuffles and complex mask. The next step
uses that measured inventory. Licensed speech/noise evaluation, model-level quality,
end-to-end latency/memory and training remain required by the original objective.

## 2026-10-17: centered audio reconstruction

Added validated `CenteredStftSpec`, reflect-centered forward plans and fixed-length
`Istft` on the portable, oneMKL and CUDA backends. Preparation caches the periodic
Hann window, inverse FFT plans and reciprocal overlap envelope. Invalid reflection
geometry, uncovered samples and near-zero envelopes are rejected. The inverse
preserves the borrowed spectrum, including when cuFFT uses destructive C2R execution.
CUDA overlap-add uses one thread per output sample with ordered frame accumulation.
Both paths reuse explicit scratch; the existing non-centered STFT API stays available.

Independent FP64 direct forward/inverse DFT checks cover even/odd FFTs, minimal
reflection lengths, multiple batches, changed spectra and real-only DC/Nyquist handling.
Tests also cover wrong axes, NOLA/tail rejection, host-allocation-free repeated execution,
foreign CUDA sessions and session lifetime. Long stereo reconstruction includes 44,101
samples and incomplete final hops. Portable tests pass (11), the combined Intel release
build passes (13), and RTX 3060 CUDA/cuDNN debug and release builds pass (15).
Clippy passes for portable, combined Intel and cuDNN configurations; combined Intel/NVIDIA
SDKs on one host remain untested.

The [runnable example](../examples/audio-roundtrip.rs) and `tools/check_audio.py` compare
all spectrum components, reconstructed waveforms and arbitrary modified-spectrum
inverses against PyTorch. All six cases pass on each device under the original
`2e-4 + 2e-4*abs(expected)` gate. They include FFT sizes 2, 7, 15, 511, 512 and 2048.
These are correctness checks, with no new transform-latency or model-quality claim.

Reproducible evidence:

- [Intel complete-output validation](../benchmarks/results/2026-10-16-audio-mkl-v2/validation.json).
- [NVIDIA complete-output validation](../benchmarks/results/2026-10-16-audio-cuda-v2/validation.json).

Each report preserves its binary digest, source manifest and archive. Archives now include
runnable examples and protocol JSON fixtures required by the Python tests. Older
long-window snapshots recover the already-hashed companion profile at its canonical
fixture path. A restore regression check covers that behavior. Eight Python tests pass
in the current tree and both recovered audio snapshots. The initial local-only audio
checks predate this fixture-inventory fix; the linked v2 reports are the recovery artifacts.
Recovered Intel and NVIDIA snapshots also pass their release tests (13 and 15).
Raw FP32 outputs remain generated files and can be regenerated by the example.

The next M2 step is a licensed real-model reference and its measured operator requirements.
Streaming overlap state, model import, end-to-end audio quality and latency, mixed precision,
autodiff and training remain open parts of the original framework goal.

## 2026-10-16: Intel long-window follow-up

The [timing profile](intel-long-window-v1.md) was committed as `741bdbd` before
collection. Both binaries were copied and hash-checked; three fresh executions used
the immutable checkout, CPU affinity `{0}`, one thread and seven paired blocks.
Every candidate/anchor output passed the unchanged PyTorch gate. No run was added,
discarded or resized after collection began. All six candidate/anchor source archives
and all three saved protocol files pass verification.

The same candidate's seven-case anchor/candidate ratio is `1.1401`, with 95% interval
`[1.1140, 1.1668]`. The short-STFT guard now passes: `1.0142 [0.9925, 1.0363]`.
The projection GEMM guard remains unresolved: `1.0140 [0.8948, 1.1491]`.
Absolute projection timings also vary between executions. The data do not establish
a GEMM regression or identify its cause; longer STFT windows alone do not control
all machine variability. The full candidate stays inconclusive and the original MKL
artifact remains the Intel control. Prepared oneDNN convolution stays available explicitly.
Further promotion needs a new, justified context with better control of CPU frequency
and contention; repeating confirmation until acceptance is not a valid next step.

Evidence: [assessment](../benchmarks/results/2026-10-16-intel-long-v1-assessment.json),
[execution 1](../benchmarks/results/2026-10-16-intel-long-v1-1/comparison.json),
[execution 2](../benchmarks/results/2026-10-16-intel-long-v1-2/comparison.json), and
[execution 3](../benchmarks/results/2026-10-16-intel-long-v1-3/comparison.json).
The prior short-window campaign and its result remain unchanged.

## 2026-10-16: prepared convolution campaign

Added packed, constant-parameter oneDNN plans, `cpu::PreparedConv1d`, and strict-FP32
cuDNN graph plans. Shape-based plans preserve direct GEMM for pointwise convolution.
The initial all-oneDNN screen regressed pointwise latency by roughly 12%, so it was
rejected. The retained im2col/GEMM routes still support dynamic parameters, provide
the comparison anchor, and cover SDK builds without the new optional features.

Three fresh confirmation executions per host remeasure the original `201026f` binaries
and the frozen candidate under affinity `{0}` and one CPU thread. Each execution includes
seven workloads, seven paired blocks, 20 warm-ups and 50 timed iterations. Every candidate
and anchor output passes the independent PyTorch numerical gate. Source/binary hashes,
hardware identity, original anchor reports and both source archives are preserved.

| Metric: original anchor / candidate | Intel | NVIDIA |
| --- | ---: | ---: |
| Seven-case geometric mean | 1.141 | 1.125 |
| Primary 95% interval | [1.121, 1.162] | [1.096, 1.155] |
| Temporal convolution | 1.267 | 1.421 |
| Grouped convolution | 1.964 | 1.591 |
| Pointwise convolution | 0.999 | 1.000 |

The NVIDIA candidate passes the primary latency gain and every regression guard. Its
seven-case speedup is about 12.5% against this project's original CUDA implementation.
Use `cuda::Conv1d::planned` / benchmark backend `nvidia` for this validated prepared route.
These are operator results; they do not establish a complete-model advantage over PyTorch.

The Intel primary gain passes, but the unchanged short-STFT case is unresolved: its
speedup interval is `[0.952155, 1.064411]`, just below the required lower bound
`1/1.05 = 0.952381`. This does not establish a real STFT regression. It does prevent
promotion under the frozen contract. The oneDNN/Intel plans remain explicit opt-in;
the original `mkl` route remains the control. No extra confirmation runs or relaxed
thresholds were used. The next Intel experiment must declare longer timing windows
before gathering new data and remeasure both artifacts under that protocol.

CPU release tests pass with `mkl,onednn` (9 tests); portable tests pass (7 tests).
CUDA/cuDNN debug and release tests pass (9 tests), including graph rebinding, changed parameter values,
fixed bias presence, workspace bounds, and pointwise dispatch. Clippy passes for oneDNN,
the combined Intel feature build and cuDNN. Four Python tests validate the assessment's
primary/regression logic, candidate identity and use of raw measurements. Combined
Intel/NVIDIA SDKs on one machine remain untested.

Every recorded candidate/anchor source archive passes its checksum and inventory checks.
The first Intel and NVIDIA confirmation archives were restored into separate empty
directories, rebuilt, and passed their release tests (9 each). The restored assessment
tests also pass. Later documentation/navigation and anchor-argument validation updates
do not change the measured library/binary. [Champion identities](../benchmarks/champions.json)
retain the accepted NVIDIA artifact and the original Intel control for future campaigns.

Evidence:

- [Intel assessment](../benchmarks/results/2026-10-16-intel-assessment.json) and
  [first confirmation report](../benchmarks/results/2026-10-16-intel-confirm-1/comparison.json).
- [NVIDIA assessment](../benchmarks/results/2026-10-16-nvidia-assessment.json) and
  [first confirmation report](../benchmarks/results/2026-10-16-nvidia-confirm-1/comparison.json).
- [Rejected all-oneDNN screen](../benchmarks/results/2026-10-16-onednn-screen1/comparison.json).
  Confirmation reports 2/3 and their source archives are beside report 1 on each host.

M1 continues with Intel measurement precision, cuBLASLt and mixed-precision policies;
model-level inference and training remain part of the original framework goal.

## 2026-10-16: initial foundation

The workspace began with `target.md`, `reference.md`, and a private NAS access bundle.
A Rust library now implements explicit FP32 host/device storage, Intel oneMKL operations,
CUDA operations, and reusable audio operator plans. Git was initialized on `main` at the
user's request; access credentials, environments, and build artifacts are ignored.

CPU debug and release tests pass for both the portable control and oneMKL. The tests use
independent FP64/DFT oracles, rectangular shapes, grouped/strided/dilated convolution,
pointwise paths, odd/even FFTs, repeated execution, invalid geometry, and warmed Rust
heap-allocation checks. Both CPU feature configurations pass Clippy with warnings denied.

RTX 3060 debug tests and release tests both pass, including CUDA output checks against
independent oracles, initialized storage, session identity rejection, and lifetime checks.
The default, `mkl`, and `cuda` feature builds have been verified individually; a combined
`mkl,cuda` SDK build has not been tested on one machine. CPU and CUDA Clippy checks pass
with warnings denied.

Both hardware comparisons pass full-output correctness on every case in all seven blocks.
The following values are medians of warmed wall latency, in microseconds. They are
exploratory operator measurements against eager PyTorch; see the fairness limits in
[benchmarking.md](benchmarking.md).

| Case | Intel Rust | Intel PyTorch | RTX 3060 Rust | RTX 3060 PyTorch |
| --- | ---: | ---: | ---: | ---: |
| gemm-stream | 310.81 | 311.13 | 12.06 | 12.78 |
| gemm-projection | 19841.26 | 19726.52 | 288.33 | 288.77 |
| conv-audio | 5205.74 | 4272.30 | 85.26 | 64.52 |
| conv-pointwise | 10270.71 | 10430.04 | 112.03 | 151.49 |
| conv-grouped | 369.65 | 403.08 | 22.48 | 22.22 |
| stft-stereo | 472.11 | 507.01 | 17.24 | 28.76 |
| stft-stream | 42.99 | 30.30 | 4.62 | 29.45 |

The equally weighted seven-case paired geometric-mean ratios (`PyTorch/Rust`) are 0.962
on Intel and 1.455 on NVIDIA. Those aggregates are specific to these probes. The short
GPU STFT advantage includes dispatch/allocation differences on the NAS's Pentium host;
it is not evidence of a sevenfold faster FFT kernel. Initial CUDA vendor stacks also
differ: system Toolkit 12.4 versus PyTorch's bundled CUDA 13.0. Several cases show noisy
block ratios. No framework-level or language-level performance claim is established.

Temporal convolution is the clearest gap: median latency is about 22% higher than
PyTorch on Intel and 32% higher on NVIDIA. Prioritize prepared oneDNN/cuDNN convolution
plans against the retained im2col/GEMM anchor. Short CPU STFT needs controlled profiling
before algorithm changes. Pointwise and STFT wins must survive model-level confirmation.

Evidence:

- [Intel raw report](../benchmarks/results/2026-10-16-cpu/comparison.json),
  [source archive](../benchmarks/results/2026-10-16-cpu/source.tar.gz), and
  [native library environment](../benchmarks/results/2026-10-16-cpu/native-environment.json).
- [NVIDIA raw report](../benchmarks/results/2026-10-16-cuda/comparison.json) and
  [source archive](../benchmarks/results/2026-10-16-cuda/source.tar.gz).

Archive SHA-256, member hashes and inventories were verified. The Intel archive was
restored into an empty directory and rebuilt/tested with `--features mkl` (6 tests).
The NVIDIA archive was restored separately on the NAS and rebuilt/tested with
`--release --features cuda` (7 tests). Raw output files are generated artifacts; the
reports, archived source and deterministic fixtures are sufficient to regenerate them.

M0 is complete. The full framework objective remains active; M1 begins with the measured
convolution gaps, followed by a real audio-model inference path and training support.

## 2026-11-23 — A0 prototype screen and cross-entropy stabilization

The A0 freeze remains open. This screen used the worktree prototype based on
`1843e4235518df80cfaa789d99a6260ddc7215be`. The [evidence directory](../benchmarks/results/2026-11-23-a0-prototype-screen/)
contains the original source, GPU delta, verification logs and raw timing blocks.

**CPU.** The original implementation returned zero loss for uniform logits of
`1e20`. The shifted formula retains `ln(vocab)` on CPU and CUDA. Reusing the
overwrite gradient buffer for exponentials improved the median paired wall ratio by
**1.40×** on the prepared FP32 `[512, 4096]` forward/backward probe. Reusing the
existing AVX2/FMA max and exponent implementations improved the ratio by another
**4.23×** in a fresh block against the scalar candidate. The SIMD block median was
**11.31 ms**. Each screen used eight alternating pairs of fresh processes pinned to
one P-core on a Core i5-13420H. The guard operators had wide paired ranges, so their
regression bounds are unresolved.

An adjacent one-thread PyTorch 2.14.1 operator screen on the same core recorded median
**18.22 ms eager** and **8.11 ms compiled**, with p95/p99 of **19.40/19.74 ms** and
**10.68/11.54 ms** respectively. The native results are batch means and the Torch
tails are individual calls. These probes use fixed logits and resident targets.
Full-step and multicore Intel qualification remains open.

The original prototype passed 154 tests and doctests on a dedicated Colab EPYC 7B12
with MKL/oneDNN and passed eight independent PyTorch CPU output checks. The local SIMD
candidate passed the same eight output checks and added coverage of large common
offsets in the forward loss and overwrite gradient. The FP16 conversion oracle now
uses the exact bit representation of the minimum subnormal.

**CUDA.** The original L4 build failed because the BF16 attention wrapper called
nonexistent `SdpaSpec` accessors. Using `seq_q` and `seq_k` restored compilation.
The focused L4 tests verified FP32 audio operators, BF16 attention and the BF16
cross-entropy boundary, including rejection of short buffers. The prototype's original
Transformer smoke test required an arbitrary 50% loss drop for its sine-initialized
fixture. Its replacement compares every replayed loss to native eager through a
learning-rate change and checks that loss decreases. The independent six-layer
PyTorch control retained its existing thresholds. It passed 17 output comparisons
and all 30 training steps with a maximum absolute loss difference of **0.1288**.

The comparison harness now shares the six-layer model's initial weights and batch,
records first-call and warmed wall/event samples, uses synchronized batch wall time
for speed ratios and retains native samples if the Torch comparison fails.

The original Torch capture failed because autograd retained the default warmup
stream. Using the same side stream for warmup and capture produced the following
L4 screen with PyTorch 2.14.1+cu130. The release build used CUDA 13.0. The
Transformer used six layers, batch 4, sequence length 512 and vocabulary 4096,
with forward, backward, clipping and fused AdamW in each timed step.

| Mode | Batch wall per step | Serial median / p95 / p99 | Tokens/s |
| --- | ---: | ---: | ---: |
| fast-audio captured | 12.52 ms | 12.72 / 13.23 / 13.31 ms | 163,641 |
| PyTorch BF16 eager | 17.55 ms | 20.56 / 21.17 / 21.32 ms | 116,720 |
| PyTorch compiled model and autograd | 13.93 ms | 14.44 / 14.80 / 14.88 ms | 146,991 |
| PyTorch compiled model and captured step | 14.99 ms | 15.36 / 15.60 / 15.66 ms | 136,609 |

The captured Rust step was **1.11×** as fast as the best observed Torch mode.
Seven of fourteen probes won against the measured Torch modes. Long attention
forward/backward remains a gap at **0.71×** for causal GQA with T=2048 and
**0.69×** for causal attention with T=1024 and D=128. This sequential performance
screen used one runtime and the same starting Transformer weights/tokens. The
operator comparisons retain the framework-specific dtype and scratch behavior,
and compilation timing reflects a warmed Inductor disk cache on the retry.

The first sanitizer pass found a shared-memory race in the arbitrary-head-width
attention forward fallback and reproduced an output mismatch. Publishing the score in
its own shared slot removed the race without adding a block barrier. Eight alternating
pairs on the same L4 timed the causal GQA fallback at `B=1, Hq=4, Hkv=2, Tq=64,
Tk=80, D=48`. The debug-build medians were **13.805 ms before** and **13.807 ms after**
with a **0.9998×** median paired wall ratio. Focused L4 memcheck, racecheck and
synccheck reported zero errors or hazards after the fix. The fallback remains a performance gap.

The local oneDNN candidate suite passed 155 tests and doctests and the focused Miri
suite passed seven. The L4 CUDA candidate suite passed 229 tests and doctests. Drive
authorization prevented cloud-cache persistence, so the Colab CLI transferred the
small source archives and reports and the evidence is stored locally.

A0 still requires full sanitizer coverage of the prototype, launch and allocation
counts, bandwidth and peak memory accounting and a counterbalanced release confirmation.
The training and C++ inference parity objectives remain open.

## 2026-11-23 — BF16 SDPA fallback and GTCRN C++ inference screen

### BF16 training and forward attention on L4

The general signed-stride SDPA path now uses one warp per query or key row,
online output accumulation and a fused backward row reduction. A warp shares the
score and its derivative among output channels. Heads wider than 128 channels
reuse a bounded register tile. The public strided bindings reject out-of-range
reads and out-of-range or potentially overlapping writes before CUDA launch.
General head widths skip the per-query-head GQA scratch reserved for the
Tensor Core path.

Four counterbalanced fresh-process blocks used the same L4, release flags,
deterministic input values and retained native predecessor. The estimated
paired speedups below use synchronized batch wall time. Full serial
distributions, cold calls, geometry and producer hashes remain in the
[raw report](../benchmarks/results/2026-11-23-bf16-sdpa-fallback/screen.json).

| Workload | Previous, µs | Candidate, µs | Paired native speedup | PyTorch, µs |
| --- | ---: | ---: | ---: | ---: |
| D48 causal GQA forward | 118.45 | 52.62 | 2.25× | 46.72 |
| D48 causal GQA forward and backward | 543.84 | 296.13 | 1.84× | 298.60 |
| D80 noncausal MHA forward and backward | 2362.51 | 349.46 | 6.76× | 254.12 |
| D129 causal GQA forward and backward | 939.19 | 219.16 | 4.29× | 1388.00 |
| D64 Tensor Core control forward and backward | 57.98 | 57.97 | 1.00× | 275.77 |
| Two-layer D48 captured training step | 1382.44 | 892.66 | 1.55× | 301.71 |
| Two-layer D64 captured training step | 389.44 | 389.65 | 1.00× | 326.20 |

PyTorch columns use eager BF16 for attention operations and compiled-model
CUDA Graph replay with fused AdamW for complete steps. The native D48 step's
serial p99 fell from 1.450 ms to 0.910 ms. The Tensor Core control's forward
batch time held steady with a single-call p99 change from 23.61 to 29.85 µs.
The first five D48 optimizer losses stayed within 0.00330 absolute of PyTorch
eager. Independent FP32 PyTorch attention checks passed for D48 and D129,
alongside the Tensor Core controls. The six-layer BF16 trajectory check also
passed. L4 memcheck, racecheck and synccheck each covered the targeted SDPA
suite, including signed channel strides, accumulation and rejected bindings,
with zero reported errors or hazards.

D48 attention scratch dropped from 124,928 to 2,048 bytes. Separate Nsight
ranges saw 4 → 3 device kernels for forward/backward and 104 → 102 for each
D48 optimizer replay. CUDA allocation API counts were zero in both warmed
ranges. Separate D48 training replays sampled a high-water value of 282 MiB
for each native binary and 394 MiB for the Torch graph process. Those
process readings include framework context and vendor memory and have MiB
sampling granularity. [Profiling and recovery](../benchmarks/results/2026-11-23-bf16-sdpa-fallback/README.md)

### GTCRN inference on Intel against ONNX Runtime

The first direct C API screen used the existing GTCRN checkpoint and all four
frozen FP32 spectra, preallocated output buffers, an i5-13420H performance
thread and ONNX Runtime 1.32.0 CPU EP. The initial median paired native/ORT
speed ratios ranged from 0.44× to 0.48×. Spectrum comparisons used the existing
absolute and relative FP32 envelope. This campaign measured the prepared
network, with waveform DSP outside the timing boundary.

A longest-input profile placed 64.11% of user-space samples in the native CPU
backend's generic execution symbol. The structural kernels now concatenate,
pad and unfold frequencies through contiguous row slices. The three-way
release screen verified identical native output hashes and planned resource
bytes in every block. All candidate outputs passed the reference and ORT
checks. The maximum paired absolute difference was 1.63e-5.

| Frozen input | Previous native median, ms | Candidate median, ms | ORT median, ms | Paired native gain | Candidate/ORT speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 / 4,096 | 8.39 | 4.23 | 3.84 | 1.88× | 0.93× |
| B1 / 16,000 | 29.13 | 14.55 | 13.19 | 1.97× | 0.87× |
| B1 / 64,000 | 114.01 | 61.99 | 51.67 | 1.83× | 0.81× |
| B2 / 16,000 | 57.15 | 30.29 | 25.22 | 1.91× | 0.84× |

The longest shape's candidate/ORT p99 was 125.98/61.54 ms and median process
maximum RSS was 122,828/75,276 KiB. The follow-up user-clock profile placed
26.17% of samples in generic execution and 17.20% in scalar tanh. The
[recorded comparison](../benchmarks/results/2026-11-23-gtcrn-ort/candidate.json)
preserves per-call timings and output hashes. The
[recovery notes](../benchmarks/results/2026-11-23-gtcrn-ort/README.md)
identify SDKs, same-input layout conversions, the C API driver and node graph.

The local graph oracle passed with MKL/oneDNN, including an unfold width larger
than the frequency dimension. Complete waveform parity, other engine/hardware
strata, exact trainer buffer accounting and independent benchmark confirmation
remain open.

## 2026-11-23 — Objective refocused on comprehensive PyTorch leadership

By user direction, the active competitive objective is refocused strictly on
**comprehensively beating PyTorch `2.14+`** across Intel CPUs and NVIDIA GPUs for both
training and inference (strict FP32 and mixed precision `BF16`/`FP16`/`TF32`), removing
non-PyTorch framework and standalone C++ inference engine parity gates from the active
roadmap.

- Updated [roadmap.md](roadmap.md#next-stage-plan-comprehensive-pytorch-leadership),
  [plan-phases-a-d.md](plan-phases-a-d.md),
  [competitive-analysis-v1.md](competitive-analysis-v1.md),
  [operator-coverage.md](operator-coverage.md), and [CODEMAP.md](CODEMAP.md) to anchor
  all 25 active tickets (`A0–A6`, `B1–B7`, `C1–C6`, `D1–D5`) and every Step Measurement
  Receipt strictly against PyTorch `2.14+` strongest compatible training and inference modes.
- Deferred cross-framework and standalone C++ inference engine comparisons, post-training
  weight-only quantized inference (`W8A16`/`W4A16`/INT8), and training-time `FP8` until
  after comprehensive PyTorch leadership passes in Phase D.
- Preserved the original `target.md`, all historical baselines, and the exploratory
  2026-11-23 GTCRN/ORT screen above as immutable records. This entry records a direction
  adjustment, not new verification or performance results.
