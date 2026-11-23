---
mode: maintenance
generated_at: 2026-10-19
---

## Summary

Private C ABI adapters compiled against installed vendor headers and owned by the Rust API.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Change Intel ABI, thread settings or prepared kernels | Intel | `mkl.c`, `onednn.cpp`, `conv.h` | Rust CPU/oneDNN consumers, build and numerical tests |
| Change CUDA kernels, graph planning or lifetimes | NVIDIA | `cuda.cu`, `cudnn.cu`, `cuda_backend.h` | `../src/cuda.rs`, `../build.rs`, actual GPU tests |
| Change BF16 kernels or attention dispatch | Mixed precision | `bf16_tc.analysis.md`, `sdpa_bf16.analysis.md` | `../src/cuda/mixed.analysis.md`, `../tests/bf16_tc_cuda.rs` |
| Change recurrent inference or parameter packing | NVIDIA | `gru.cu`, `gru.h` | `../src/gru.rs`, `../src/cuda.rs`, actual GRU layer/oracle tests |
| Change 2D convolution/deconvolution | Spatial | `conv2d.h`, `onednn.cpp`, `cudnn.cu` | `../src/conv2d.rs`, `../src/cuda/`, independent and real-model stage checks |
| Change graph tensor transforms/reductions/normalization | Graph | `tensor_ops.cu`, `tensor_ops.h` | `../src/graph.rs`, `../src/cuda/graph.rs`, FP64 and complete-model gates |
| Change general strided pointwise kernels | Tensor algebra | `pointwise.cu`, `pointwise.h` | `../src/pointwise.rs`, `../src/cuda/pointwise.rs`, `../tests/pointwise.rs` |
| Change pointwise gradients or broadcast contraction | Gradients | `pointwise_backward.cu`, `pointwise_backward.h` | Shared Rust derivative geometry, `../src/cuda/pointwise_backward.rs`, independent GPU VJP/empty/tree/session/grid tests |
| Change ordered strided reductions/local derivatives | Reductions | `reduce.cu`, `reduce.h` | `../src/reduce.rs`, `../src/cuda/reduce.rs`, `../tests/reduce_cuda.rs`, `../tests/norm_cuda.rs` |
| Change LayerNorm/RmsNorm or Softmax/LogSoftmax CUDA kernels | Numerical blocks / Gradients | `norm.cu`, `norm.h`, `softmax.cu`, `softmax.h` | `../src/cuda/norm.rs`, `../src/cuda/softmax.rs`, `../tests/norm_cuda.rs`, `compute-sanitizer` checks |
| Change strided matrix forward/backward kernels | Dense algebra / Gradients | `matmul.cu`, `matmul.h`, `mkl.c` | `../src/matmul.rs`, `../src/cuda/matmul.rs`, transpose/leading-dimension and tile/grid/zero tests |
| Change broadcast batched matmul or Flash SDPA CUDA kernels | Sequence / Dense algebra | `bmm.cu`, `bmm.h`, `sdpa.cu`, `sdpa.h` | `../src/cuda/bmm.rs`, `../src/cuda/sdpa.rs`, `../tests/bmm_sdpa_cuda.rs`, `compute-sanitizer` checks |
| Change strided 1D/2D or transposed convolution forward/backward kernels | Spatial / Gradients | `conv_backward.cu`, `conv_backward.h` | `../src/conv.rs`, `../src/cuda/conv.rs`, `../tests/conv_backward_cuda.rs`, `../tests/conv_transpose_structural_cuda.rs`, `compute-sanitizer` checks |
| Change strided Pad or Cat structural CUDA kernels | Tensor algebra / Gradients | `structural.cu`, `structural.h` | `../src/structural.rs`, `../src/cuda/structural.rs`, `../tests/conv_transpose_structural_cuda.rs`, `compute-sanitizer` checks |
| Change strided STFT/CenteredSTFT/iSTFT forward/backward kernels | Audio / Gradients | `stft_backward.cu`, `stft_backward.h` | `../src/stft.rs`, `../src/cuda/stft.rs`, `../tests/stft_backward_cuda.rs`, `compute-sanitizer` checks |
| Change canonical gradient zeroing, SGD, AdamW, global L2 grad clipping or CUDA Graph capture | Training | `training.cu` | `../src/cuda/tape.rs`, `../src/optim.rs`, `../src/autograd/update.rs`, `../tests/optim_cuda.rs` |
| Change runtime NVRTC whole-step fusion compilation or launch | Training / Fusion | `nvrtc_fusion.cu` | `../src/autograd/fusion.rs`, `../src/cuda/tape.rs`, `../build.rs`, `../tests/autograd_cuda.rs` |

## Files

| File | Domain | Deps | Function |
| --- | --- | --- | --- |
| `audio_ops.h` | ABI | → `../src/cuda/audio_ops.rs` | FP32 embedding, RoPE, cross-entropy and dropout entry points |
| `audio_ops.cu` | Audio / Training | ← session and `audio_ops.h` | FP32 CUDA operator kernels including shifted cross-entropy and deterministic embedding backward |
| `bf16_tc.h` | ABI | → `../src/cuda/mixed.rs` | BF16 operator, TF32 GEMM, AdamW and device loss-scaling entry points |
| `bf16_tc.cu` | Mixed precision | ← session, cuBLAS and `bf16_tc.h` | BF16/TF32 GEMM and BF16 CUDA kernels including fused cross-entropy |
| `bf16_tc.analysis.md` | Navigation | ← `bf16_tc.cu` | Feature ranges for CUDA mixed-precision operators |
| `sdpa_bf16.h` | ABI | → `../src/cuda/mixed.rs` | BF16 attention strides, preparation and forward/backward entry points |
| `sdpa_bf16.cu` | Mixed-precision attention | ← session and `sdpa_bf16.h` | Tensor Core attention, warp-cooperative signed-stride fallback, deterministic GQA reduction and launch dispatch |
| `sdpa_bf16.analysis.md` | Navigation | ← `sdpa_bf16.cu` | Feature ranges for BF16 attention |
| `mkl.c` | Intel | → `../src/cpu.rs`, `../src/matmul.rs`, `../src/pointwise.rs`; Intel headers/runtime | Scoped-thread dense and explicit transpose/leading-dimension SGEMM; single-threaded `VML_HA` (`vmsTanh`/`vmsExp`/`vmsLn`/`vmsSqrt`) wrappers; owned batched forward/inverse DFTI descriptors with direction-specific distances |
| `onednn.cpp` | Intel | ← `conv.h`, `conv2d.h`; → `../src/onednn.rs` | Shared 1D/2D forward/deconvolution preparation, packed weights/bias, cached layout reorders, user scratchpad and scoped OpenMP policy |
| `cuda.cu` | NVIDIA | → `../src/cuda.rs`; CUDA/cuBLAS/cuFFT | Context/transfers with static 4 MiB `cublasSetWorkspace` and 32 KiB `reduce_scratch`, FP32 GEMM, convolution/framing kernels, R2C/C2R plans, spectrum preservation, deterministic overlap-add and events |
| `cudnn.cu` | NVIDIA | ← `cuda_backend.h`, `conv2d.h`, `cudnn_plan.h`; → `../src/cuda.rs`, `../src/cuda/` | Strict-FP32 forward/backward-data graphs, inherited determinism filter in both searches, selected notes/engine metadata, bounded workspace, optional fusion and immutable bindings |
| `cudnn_plan.h` | ABI | → cuDNN graph adapter, Rust `cuda/cudnn.rs` | 32-byte selected-engine metadata and negative no-compatible-plan status, distinct from vendor error domains |
| `conv.h` | ABI | → native convolution adapters, Rust `spec.rs` | Shared ten-i32 convolution geometry layout |
| `conv2d.h` | ABI | → native spatial adapters, Rust `conv2d.rs` | Shared nineteen-i32 forward/transpose geometry with a 76-byte layout assertion |
| `tensor_ops.h` | ABI | → tensor CUDA kernels, Rust graph descriptors | Checked 112-byte generic and 60-byte bounded mapped-index descriptors with native and Rust size assertions |
| `tensor_ops.cu` | Graph / NVIDIA | ← shared session/geometry headers; → CUDA graph backend | Same-stream tensor kernels, exact reciprocal mapped layouts/broadcasts, retained 64-bit fallback, reductions and complex math without host staging |
| `tensor_copy.h` | ABI | → CUDA copy plan, Rust `cuda/views.rs` | 32-byte count/origin/rank descriptor with signed element offsets and asserted field alignment |
| `tensor_copy.cu` | Storage / NVIDIA | ← session/copy headers; → Rust view copy plan | Prepared device coordinate metadata, same-stream contiguous transfers or bit-preserving signed-stride kernels, synchronized metadata lifetime |
| `view_backward.h` | Gradient ABI | → native adjoint kernels, Rust `cuda/adjoint.rs` | Asserted 80-byte base/view/group/span/stride-kind descriptor with explicit reduction lanes |
| `view_backward.cu` | Gradients / NVIDIA | ← session/gradient headers; → Rust adjoint plan | Unique writes, bounded indexed/grouped gather and analytical broadcast reductions, fixed warp/block trees, initialized output spans and synchronized metadata ownership |
| `pointwise.h` | Tensor algebra ABI | → native pointwise kernels, Rust `cuda/pointwise.rs` | Asserted 80-byte operation/flat geometry and 32-byte outer-axis descriptors |
| `pointwise.cu` | Tensor algebra / NVIDIA | ← session/pointwise headers; → Rust unary/binary plans | Operation-specialized flat, register-hoisted 32-bit/pow2 rank-1/2 and 64-bit coordinate kernels, stable nonlinear equations, explicit rounded arithmetic and synchronized O(rank) metadata ownership |
| `pointwise_backward.h` | Gradient ABI | → fused pointwise backward kernels, Rust `cuda/pointwise_backward.rs` | Asserted 80-byte dependency/geometry header and 40/32-byte kept/reduced-axis descriptors |
| `pointwise_backward.cu` | Gradients / NVIDIA | ← session/gradient headers; → Rust local-VJP plans | Preparation-selected operation/operand kernels, stable derivatives, 1-thread small-count, zero-division incremental 2-axis block and 8-block multi-SM `Tree256` split broadcast contractions, unique writes and synchronized descriptor ownership |
| `reduce.h` | Reduction ABI | → ordered reduction kernels, Rust `cuda/reduce.rs` | Asserted 56-byte count/operation descriptor and 24/16-byte kept/reduced-axis layouts |
| `reduce.cu` | Reductions / Gradients | ← shared session/reduction headers; → Rust forward/local-backward plans | Empty/copy, vectorized 1-thread small-axis (`count<=8`), 16-lane sub-warp (`9..=16`), warp (`count<=128` for nonlinear `Max`/`Min`/`LogSumExp`), zero-division 2-axis block and 8-block multi-SM `Tree256` split reductions, exact `uint64_t` tie counters, WAR-free shared reductions and synchronized descriptor ownership |
| `norm.h` | Normalization ABI | → CUDA norm kernels, Rust `cuda/norm.rs` | Asserted 64-byte group/inner-length/eps/kind descriptor and 32-byte outer/inner axis descriptors |
| `norm.cu` | Numerical blocks / Gradients | ← session and `norm.h`; → Rust CUDA norm plans | Tier-1 warp-per-group (`D<=128`, 0 smem, 0 barriers), Tier-2 `float4` register-cached block (`128<D<=1024`), WAR-free `block_reduce_sum`/`block_reduce_sum2` and general strided `LayerNorm`/`RmsNorm` forward and local VJPs |
| `softmax.h` | Softmax ABI | → CUDA softmax kernels, Rust `cuda/softmax.rs` | Asserted 80-byte group/axis-length/stride/kind descriptor and 32-byte outer-group axis descriptor |
| `softmax.cu` | Numerical blocks / Gradients | ← session and `softmax.h`; → Rust CUDA softmax plans | Tier-1B `float4` warp-per-row (`D<=512`, `D%4==0`, SM 8.0+ `max.NaN.f32`, `ld.global.cs` streaming for `>=4 MiB`), Tier-1 warp-per-row (`D<=128`), Tier-2 register-cached block (`128<D<=1024`) and Tier-3 arbitrary-axis strided numerically stable `Softmax`/`LogSoftmax` forward, `run` and `run_from_output` local VJPs |
| `matmul.h` | Dense algebra ABI | → CUDA matrix kernels, Rust `cuda/matmul.rs` | Asserted 128-byte by-value dimensions, signed origins/strides, leading dimensions, transpose flags and route descriptor |
| `matmul.cu` | Dense algebra / Gradients | ← session/matrix headers; → Rust matrix plans | Strict FP32 cuBLAS transpose/leading-dimension dispatch, shared-memory 16x16 signed-stride tiles, grid reuse and positive-zero overwrite without global scratch |
| `bmm.h` | Batched GEMM ABI | → CUDA batched GEMM kernels, Rust `cuda/bmm.rs` | Asserted 192-byte by-value dimensions/origins/leading-dims/batch-strides/route descriptor and 32-byte broadcast batch-axis descriptor |
| `bmm.cu` | Dense algebra / Gradients | ← session, cuBLAS and `bmm.h`; → Rust CUDA `BatchedMatmulPlan`/`BatchedMatmulBackward` | Strict-FP32 `Collapsed2d` single `cublasSgemm`, `CublasStridedBatched` and single-launch 16x16 shared-memory `TiledCuda` broadcast batched GEMM forward and zero-atomic `dA`/`dB` local VJPs |
| `sdpa.h` | Flash SDPA ABI | → CUDA Flash SDPA kernels, Rust `cuda/sdpa.rs` | Asserted 264-byte by-value 4D strided Q/K/V/O/dO/dX geometry (with five 40-byte `FaSdpaView4` descriptors), scale and causal-mask descriptor |
| `sdpa.cu` | Sequence / Gradients | ← session and `sdpa.h`; → Rust CUDA `SdpaPlan`/`SdpaBackward` | Strict-FP32 L2-cached double-buffered 64×64 SIMT SDPA GEMM (`qk`/`pv`/`ptdo`, 1-launch permuted/GQA, causal tile skip, parallel GQA scratch + `float4` reduce) + `float4` warp softmax + cuBLAS fallback, and O(T)-memory register-tiled FlashAttention-2 (`BR=32, BC=32`, 8 lanes/row, zero-atomic `dQ`/`dK`/`dV`) |
| `conv_backward.h` | Spatial gradient ABI | → CUDA 1D/2D conv kernels, Rust `cuda/conv.rs` | Asserted 240-byte by-value spatial dimensions, signed origins/strides, kernel/stride/padding/dilation, leading dimensions, transpose flags, route and mode descriptor |
| `conv_backward.cu` | Spatial / Gradients | ← session and `conv_backward.h`; → Rust 1D/2D & transposed conv plans | Strict FP32 pointwise cuBLAS dispatch and single-launch 16x16 shared-memory tiled implicit-GEMM across all groups for 1D/2D and transposed 1D/2D forward, `dX` and `dW` with grid-stride tile reuse, zero global scratch and no atomics |
| `structural.h` | Structural ABI | → CUDA Pad/Cat kernels, Rust `cuda/structural.rs` | Asserted 48-byte `FaPadDims`/`FaPadAxis`, 56-byte `FaCatDims` and 32-byte `FaCatAxis` descriptors |
| `structural.cu` | Tensor algebra / Gradients | ← session and `structural.h`; → Rust CUDA `PadPlan`/`PadBackward`/`CatPlan`/`CatBackward` | Zero-atomic arbitrary-rank strided `Pad` (`Constant`/`Reflect`/`Replicate` recursive backward gather) and `Cat` forward and local VJPs |
| `stft_backward.h` | Spectral gradient ABI | → CUDA STFT/iSTFT kernels, Rust `cuda/stft.rs` | Asserted 32-byte rank-2 (`FaStftView2`) and 48-byte rank-4 (`FaStftView4`) by-value strided view and direct-R2C alignment descriptors |
| `stft_backward.cu` | Audio / Gradients | ← session, cuFFT and `stft_backward.h`; → Rust CUDA spectral plans | Strided R2C/C2R cuFFT plans and deterministic zero-atomic framing/reflection-adjoint/overlap-add/endpoint-scaling CUDA kernels for `Stft`, `CenteredStft` and `Istft` forward and local VJPs; CUDA Graph capture-safe |
| `training.cu` | Training / NVIDIA | ← session header; → Rust CUDA tape and `cuda::AdamW` | Same-stream positive-zero clearing, dense FP32 SGD writes, deterministic two-stage FP64 global L2 gradient-norm partial/finalize kernels, on-device `d_step`/`d_lr` AdamW control block and single-pass decoupled `adamw_param_kernel`, and guarded CUDA Graph stream capture/instantiate/launch/destroy |
| `nvrtc_fusion.cu` | Training / Fusion | ← session header, `dlopen` (`libnvrtc.so`, `libcuda.so.1`); → Rust CUDA tape backend | Dynamic NVRTC/driver loader, process-wide PTX cache keyed by `(compute_XY, source)`, strict-FP32 kernel compilation without `--use_fast_math`, RAII module/function handle and same-stream `cuLaunchKernel` execution |
| `cuda_backend.h` | ABI | → native CUDA adapters | Shared CUDA session layout (including static cuBLAS workspace and split-reduction scratch) and status/device helpers, including training kernels |
| `gru.cu` | NVIDIA | ← `cuda_backend.h`, `gru.h`; → `../src/cuda.rs` | FP32 cuDNN GRU preparation, gate-verified owned weights, standard/small-hidden algorithms, same-stream execution and destruction |
| `gru.h` | ABI | → `gru.cu`, Rust CUDA adapter | Five-i32 recurrent geometry with a 20-byte native layout assertion |

The `ConvDims` ABI is ten 32-bit C integers in the same order as Rust's `#[repr(C)]`
structure. Native handles stay private; Rust retains the session until every buffer,
plan and event is destroyed. Native code uses status returns across the C boundary.
