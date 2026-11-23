# Operator and differentiation coverage

## Reading the matrix

This is the current FP32 execution surface, not a claim that any operator family or
the foundation is finished. CPU means the portable route unless a vendor feature is
specified. CUDA means the single-session NVIDIA route; cuDNN is an additional feature.
Only an explicit backward API counts as a derivative. A transposed convolution or an
internal cuDNN backward-data implementation does not establish autograd coverage.
The public API has no implicit fallback or dtype conversion.

| Primitive | CPU forward | NVIDIA forward | Public backward | Layout and capability limits |
| --- | --- | --- | --- | --- |
| Layout transforms/copy | Borrowed metadata/copy plan | Borrowed metadata/copy plan | Prepared read-view adjoint on CPU/CUDA | Signed strides, scalar/empty, arbitrary rank; mutable views need injectivity proof; overlapping adjoints use analytical or bounded indexed reduction |
| Unary pointwise | Common Rust plan | Custom CUDA plan | Explicit local VJP on CPU/CUDA | Neg, Abs, Square, Reciprocal, Sqrt, Exp, Log, Relu, Sigmoid, Tanh, Silu, Softplus; signed-stride/aliased upstream, permuted logical gradients and selective original input; verified native tape composition |
| Binary pointwise | Common Rust plan | Custom CUDA plan | Explicit local VJP for each operand on CPU/CUDA | Add, Sub, Mul, Div; right-aligned broadcasting and fused ordered gradient contraction; read overlap and injective/permuted logical gradients |
| Sum/Mean/Max/Min/LogSumExp reduction | Prepared fixed-tree Rust plan (+ MKL VML `vmsExp`) | Prepared Tier-1 warp (`count<=128`) / block / `Tree256` CUDA plan | Explicit local derivative (`ReduceBackward`, selective `input_required`) on CPU/CUDA | Verified signed-stride/read-overlap inputs, arbitrary explicit axes, both keep-dimension policies, empty contractions, exact `uint64_t` tie division and PyTorch-matched NaN/±inf backward rules; verified native tape composition |
| Normalization (`LayerNorm`, `RmsNorm`, `GroupNorm`) | Prepared strided `NormPlan` (runtime AVX2+FMA contiguous rows) | Prepared Tier-1 warp (`D<=128`), Tier-2 `float4` register-cached block (`128<D<=1024`) and general strided CUDA `NormPlan` | Verified local VJP (`NormBackward`) on CPU/CUDA and `autograd::Builder::{layer_norm, rms_norm, group_norm}` | Parameter-free arbitrary-rank tail normalization over last `normalized_dims` axes (`GroupNorm` via zero-copy reshape composition); affine scale/bias compose via fused post-barrier pointwise tail |
| Stable `Softmax` & `LogSoftmax` | Prepared strided `SoftmaxPlan` (runtime AVX2+FMA Cody-Waite `exp256_ps` + `.bss` forward cache + MKL VML `vmsExp`) | Prepared Tier-1B `float4` warp (`D<=512`, `D%4==0`, SM 8.0+ `max.NaN.f32`, `ld.global.cs` streaming `>=4 MiB`), Tier-1 warp, Tier-2 register block and Tier-3 arbitrary-axis strided CUDA `SoftmaxPlan` | Verified local VJP (`SoftmaxBackward::run` and stateless `run_from_output`) on CPU/CUDA and `autograd::Builder::{softmax, log_softmax}` | Arbitrary rank and axis `0..rank`, signed strides, two-pass max-shifted online stability and NaN/±inf safety |
| GEMM & Broadcast Batched GEMM (`Bmm`) | Reference / oneMKL, prepared strided `MatmulPlan` & `BatchedMatmulPlan` (`Collapsed2d` single SGEMM & general broadcast) | cuBLAS / prepared strided tiled `MatmulPlan` & `BatchedMatmulPlan` (`Collapsed2d` single `cublasSgemm`, `CublasStridedBatched`, 16x16 `TiledCuda`) | Verified either-operand local VJPs (`MatmulBackward`, `BatchedMatmulBackward`) on CPU/CUDA and `autograd::Builder::{matmul, bmm}` | Rank-two and arbitrary-rank (`rank>=2`) right-aligned broadcast batched FP32 signed/read-overlap inputs, zero-atomic broadcast gradient contractions and selective operand saves |
| Fused Flash Scaled Dot-Product Attention (`Sdpa`) | Prepared L1-tiled (`BC=64`) online-softmax `SdpaPlan` (runtime AVX2+FMA `exp256_ps`/dots/axpy + `.bss` backward tile cache) | Prepared strict-FP32 L2-cached double-buffered 64×64 SIMT SDPA GEMM + `float4` warp softmax + parallel GQA reduction (`T_k<=1024`) and `O(T)` register-tiled FlashAttention-2 (`BR=32, BC=32`) fallback | Verified zero-atomic local VJPs (`SdpaBackward` for `Query`, `Key`, `Value`) on CPU/CUDA and `autograd::Builder::sdpa` | Strict-FP32 MHA and GQA (`H % H_kv == 0`), causal and non-causal masking, arbitrary signed-stride 4D views (`[B,T,H,D].permute([0,2,1,3])`), selective `dV` saves (`[true, true, false]`) and CUDA Graph capture; arbitrary additive/bool attention masks and Tensor Core BF16/FP16 FlashAttention remain open |
| Conv1d & ConvTranspose1d | Reference / oneMKL / oneDNN; prepared strided `Conv1dPlan` & `ConvTranspose1dPlan` (`Reference` / `Mkl` / `PackedMkl`) | Custom GEMM / cuDNN; prepared strided `Conv1dPlan` & `ConvTranspose1dPlan` (`Cublas` / `TiledCuda`) | Verified either-operand local VJPs (`Conv1dBackward`, `ConvTranspose1dBackward`) on CPU/CUDA and `autograd::Builder::{conv1d, conv_transpose1d}` | Rank-three FP32 signed/read-overlap inputs and injective outputs/gradients, groups/stride/padding/output_padding/dilation, exact operator duality over `SpatialCore`, selective operand saves, bounded 64 KiB CPU packing / zero GPU global workspace |
| Conv2d & ConvTranspose2d | Reference / oneMKL / oneDNN; prepared strided `Conv2dPlan` & `ConvTranspose2dPlan` (`Reference` / `Mkl` / `PackedMkl`) | cuBLAS pointwise / cuDNN; prepared strided `Conv2dPlan` & `ConvTranspose2dPlan` (`Cublas` / `TiledCuda`) | Verified either-operand local VJPs (`Conv2dBackward`, `ConvTranspose2dBackward`) on CPU/CUDA and `autograd::Builder::{conv2d, conv_transpose2d}` | Rank-four FP32 signed/read-overlap inputs and injective outputs/gradients, groups/per-axis stride/dilation/padding/output_padding, selective operand saves, bounded 64 KiB CPU packing / zero GPU global workspace |
| Structural `Pad` & `Cat` | Prepared arbitrary-rank strided `PadPlan` (`Constant`/`Reflect`/`Replicate`) & `CatPlan` | Prepared zero-atomic CUDA `PadPlan` & `CatPlan` | Verified payload-free local VJPs (`PadBackward`, `CatBackward`, `input_required == false`) on CPU/CUDA and `autograd::Builder::{pad, cat, cat_all}` | Arbitrary rank (`1..=64` CPU, `1..=16` CUDA), signed strides, zero-atomic recursive reflect/replicate backward gather and zero saved forward activations |
| STFT/centered STFT/inverse STFT | RustFFT / oneMKL; prepared strided `StftPlan` / `CenteredStftPlan` / `IstftPlan` | cuFFT; prepared strided `StftPlan` / `CenteredStftPlan` / `IstftPlan` | Verified strided local VJPs (`StftBackward` / `CenteredStftBackward` / `IstftBackward`) on CPU/CUDA and `autograd::Builder::{stft, centered_stft, istft}` | Arbitrary signed-stride real audio and one-sided interleaved FP32 spectra, explicit finite snapshotted windows, reflection adjoint and NOLA rules, zero saved forward activations (`input_required == false`), deterministic zero-atomic GPU kernels, full CUDA Graph capture/replay and post-barrier tape fusion; dense inference wrappers retained |
| GRU | Reference / oneMKL | cuDNN | Missing | Dense one-layer reset-after recurrence; explicit direction/state and immutable prepared parameters |
| Fixed inference graph tensor kernels | CPU graph kernels | CUDA graph kernels | Missing | Rank-four, positive fixed shapes; limited add/multiply/unary, layout, reduction, normalization, linear and complex helpers; not the arbitrary-stride API |
| Native reverse-mode tape, liveness reuse, fused SGD/AdamW & Checkpointing | Verified implementation (+ activation liveness storage reuse, runtime AVX2+FMA + 2D L1 cache-blocked whole-step fusion, `cpu::AdamW` + global L2 grad clip, `Checkpoint`) | Verified implementation (+ NVRTC strict-FP32 whole-step fusion, `cuda::AdamW` + deterministic two-stage FP64 global L2 grad clip, single-CUDA-Graph dynamic-LR `step_scalar_adamw_captured`) | Composed arity-3 VJPs, residual-skip pre-barrier leaf protection, selected-leaf SGD/AdamW updates and FNV-1a checksummed cross-device `Checkpoint` save/restore | Fixed-shape arbitrary-rank supported tensor operations, `Bmm`, Flash `Sdpa`, `Norm`, `Softmax`, `Conv1d`/`2d`, `ConvTranspose1d`/`2d`, `Pad`, `Cat` and `Stft`/`CenteredStft`/`Istft`; exact dead-activation buffer reuse (`free_after < pos`), zero-recapture dynamic-LR CUDA Graph training and cross-device CPU<->CUDA checkpointing |

Exact shapes, equations, algorithm policy and errors remain specified by the linked
[execution contract](architecture.md), [layout](tensor-layout-v1.md),
[view adjoint](view-backward-v1.md), [pointwise](pointwise-v1.md) and
[pointwise local VJP](pointwise-backward-v1.md),
[reduction](reduction-v1.md), [normalization](norm-v1.md), [softmax](softmax-v1.md),
[prepared matmul](matmul-v1.md), [batched matmul](bmm-v1.md), [Flash SDPA](sdpa-v1.md),
[prepared 1D/2D convolution](conv-backward-v1.md), [structural Pad/Cat](structural-v1.md),
[differentiable STFT/iSTFT](stft-backward-v1.md) and [AdamW/checkpointing](optim-v1.md) contracts. Older
graph/model specializations retain current consumers; no migration or performance
equivalence is inferred from availability of a newer primitive.

## Required gaps

The strict-FP32 single-device MVP core (pointwise, `Sum`/`Mean`/`Max`/`Min`/`LogSumExp`,
`LayerNorm`/`RmsNorm`/`GroupNorm`, `Softmax`/`LogSoftmax`, `Matmul`/`Bmm`, causal/non-causal
MHA/GQA `Sdpa`, `Conv1d`/`2d` & `ConvTranspose1d`/`2d`, `Pad`/`Cat`, `Stft`/`CenteredStft`/`Istft`,
activation liveness reuse, 2D CPU & NVRTC CUDA whole-step fusion, `AdamW` + global L2 grad clip,
single-CUDA-Graph dynamic-LR training, and `Checkpoint`) is implemented and verified.

The following major foundation gaps remain open and govern the next roadmap phases
([roadmap.md](roadmap.md#next-stage-plan-comprehensive-pytorch-leadership)):

- **Mixed precision & Tensor Core execution (Phase A):** Explicit `BF16`/`FP16`/`TF32` storage,
  casts, `cublasLt`/cuDNN Tensor Core GEMM/Conv, Tensor Core `BF16`/`FP16` FlashAttention-2
  (short and long context), dynamic FP16 `GradScaler`, and master-weight mixed-precision `AdamW`.
- **Missing dtypes, operators, and differentiable DSP (Phase B):** General `I64`/`U32`/`Bool`/
  `Complex64` tensor dtypes; `Embedding`, `Gather`, `ScatterAdd`; `GRU`/`LSTM` backward;
  arbitrary additive/boolean attention masks and `RoPE`; deterministic counter-based `RNG`
  (Philox) and `Dropout`; `AvgPool`/`MaxPool`; `CrossEntropy` and `MultiResolutionStftLoss`;
  product reductions; and stateful chunked DSP (`process`/`flush`/`reset` resampling/filterbanks).
- **System scale & parallelism (Phase C):** Variable-length audio shape bucketing, pooled GPU
  VRAM allocator with exact peak reporting, activation checkpointing (rematerialization),
  multi-core CPU thread-pool scaling (`threads > 1`), high-throughput long-sequence (`T > 1024`)
  GPU FlashAttention-2, and pinned async H2D staging streams.
- **Full-model convergence, ergonomics, and multi-GPU (Phase D):** Multi-epoch convergence and
  throughput/VRAM qualification on complete multi-layer audio models against PyTorch
  `bf16 + compile + cudagraph`, ergonomic Rust module/parameter APIs, and NCCL `DDP`.

Historical performance baselines remain preserved as explicit regression anchors:
[prepared CPU backward dispatch](cpu-prepared-backward-results-v1.md) (1.5648 primary speedup),
[complete pointwise characterization](pointwise-characterization-results-v2.md) (56 cases with
documented CPU nonlinear/layout and GPU channel-broadcast gaps), and the fresh strict-FP32
foundation suite in [progress.md](progress.md). The [performance protocol](benchmarking.md) and
[roadmap](roadmap.md) require matched execution modes, frozen sources, per-stratum
resource/correctness gates, and independent confirmation before any promotion.
