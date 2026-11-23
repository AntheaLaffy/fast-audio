---
mode: maintenance
generated_at: 2026-10-19
---

## Summary

Prepared CUDA tensor, gradient, spatial and graph plans built on the parent session/storage contract.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Change CUDA 2D convolution or transpose geometry | Spatial | `conv2d.rs` | `../conv2d.rs`, `../../native/cudnn.cu`, `../../tests/conv2d.rs` |
| Change ownership, pointer rebinding or graph lifetime | NVIDIA | `conv2d.rs`, `../cuda.rs` | Parent `CudnnPlan`, session checks and destruction in native adapter |
| Change complete device graph execution or tensor kernels | Graph | `graph.rs`, `../graph.rs` | `../../native/tensor_ops.cu`, independent CUDA graph tests and model waveform gate |
| Change view-gradient plans or reductions | Gradients | `adjoint.rs` | `../adjoint.rs`, `views.rs`, `../../native/view_backward.cu`, `../../tests/adjoint.rs` |
| Change unary/binary strided math | Tensor algebra | `pointwise.rs` | `../pointwise.rs`, `views.rs`, `../../native/pointwise.cu`, `../../tests/pointwise.rs` |
| Change pointwise local gradients or broadcast contraction | Gradients | `pointwise_backward.rs` | `../pointwise_backward.rs`, `views.rs`, `../../native/pointwise_backward.cu`, `../../tests/pointwise_backward_cuda.rs` |
| Change general Sum/Mean/Max/Min/LogSumExp and local adjoints | Reductions | `reduce.rs` | `../reduce.rs`, `views.rs`, `../../native/reduce.cu`, `../../tests/reduce_cuda.rs`, `../../tests/norm_cuda.rs` |
| Change LayerNorm/RmsNorm or Softmax/LogSoftmax | Numerical blocks / Gradients | `norm.rs`, `softmax.rs` | `../norm.rs`, `../softmax.rs`, `views.rs`, `../../native/norm.cu`, `../../native/softmax.cu`, `../../tests/norm_cuda.rs` |
| Change strided matmul or either matrix gradient | Dense algebra / Gradients | `matmul.rs` | `../matmul.rs`, `views.rs`, `../../native/matmul.cu`, `../../tests/matmul_cuda.rs` |
| Change broadcast batched matmul or Flash SDPA | Sequence / Dense algebra | `bmm.rs`, `sdpa.rs` | `../bmm.rs`, `../sdpa.rs`, `views.rs`, `../../native/bmm.cu`, `../../native/sdpa.cu`, `../../tests/bmm_sdpa_cuda.rs` |
| Audit BF16 CUDA training or loss and typed operands | Mixed precision | `mixed.analysis.md`, `audio_ops.rs` | `../../native/CODEMAP.md`, `../../tests/bf16_tc_cuda.rs`, `../../tests/audio_ops_cuda.rs` |
| Change strided 1D/2D or transposed convolution or either local gradient | Spatial / Gradients | `conv.rs` | `../conv.rs`, `views.rs`, `../../native/conv_backward.cu`, `../../native/conv_backward.h`, `../../tests/conv_backward_cuda.rs`, `../../tests/conv_transpose_structural_cuda.rs` |
| Change differentiable Pad or Cat structural plans | Tensor algebra / Gradients | `structural.rs` | `../structural.rs`, `views.rs`, `../../native/structural.cu`, `../../native/structural.h`, `../../tests/conv_transpose_structural_cuda.rs` |
| Change strided STFT/CenteredSTFT/iSTFT or local gradients | Audio / Gradients | `stft.rs` | `../stft.rs`, `views.rs`, `../../native/stft_backward.cu`, `../../native/stft_backward.h`, `../../tests/stft_backward_cuda.rs` |
| Change native training execution, fusion, zeroing, SGD, AdamW or CUDA Graph capture | Training | `tape.rs` | `../autograd/CODEMAP.md`, `../optim.rs`, `../../native/training.cu`, `../../native/nvrtc_fusion.cu`, `../../tests/autograd_cuda.rs`, `../../tests/optim_cuda.rs` |

## Files

| File | Domain | Deps | Function |
| --- | --- | --- | --- |
| `audio_ops.rs` | Audio / Training | ← shared session, operator specs and typed indices; → `../../native/audio_ops.h` | Prepared FP32 embedding, RoPE, cross-entropy and dropout with their local VJPs |
| `mixed.rs` | Mixed precision | ← session, dtype and operator specs; → BF16/TF32 native ABIs | Contiguous typed device storage, prepared BF16 and TF32 operators and a prototype captured Transformer trainer |
| `mixed.analysis.md` | Navigation | ← `mixed.rs` | Feature ranges for the mixed-precision prototype |
| `conv2d.rs` | Spatial | ← parent CUDA API, shared `Conv2dSpec`, native graph/padding entry points | Immutable uploaded parameters, direct pointwise GEMM or cuDNN graph, reusable output-padding extension and total workspace bound |
| `graph.rs` | Graph | ← shared executor/mapping, parent session/storage, cuBLAS/cuDNN, native kernels | Public CUDA graph wrapper, prepared uploads/bounded address maps, retained 64-bit fallback, explicit recurrent states and resource metadata |
| `cudnn.rs` | NVIDIA policy | ← parent session, native cuDNN metadata ABI | Explicit convolution engine policy, actual selected engine/numerical-note metadata, preparation errors and shared native plan ownership |
| `views.rs` | Storage / Data movement | ← shared layouts/copy geometry, parent ownership; → native tensor-copy ABI | Borrowed device views, injective mutable bindings and prepared arbitrary-rank signed-stride copies without host payload staging |
| `adjoint.rs` | Gradients | ← shared bounded preparation, device view/session contracts; → native view-backward ABI | Prepared adjoints, synchronized descriptor/index uploads, strict input/output ownership, immutable execution and device/shared metadata accounting |
| `pointwise.rs` | Tensor algebra | ← shared normalized/coalesced geometry, device view/session contracts; → native pointwise ABI | Immutable unary/binary plans, asserted descriptors, flat parameter/outer-axis preparation, strict rebinding/session checks and persistent metadata accounting |
| `pointwise_backward.rs` | Gradients | ← shared pointwise derivative/reduction geometry, device views; → native local-VJP ABI | Selective optional payload/session checks, asserted 80/40/32-byte descriptors, synchronized O(rank) ownership and actual static shared-memory/resource reporting |
| `reduce.rs` | Reductions / Gradients | ← shared ordered reduction/broadcast geometry, device views; → native reduction ABI | Prepared `Sum`/`Mean`/`Max`/`Min`/`LogSumExp` forward and local adjoints, exact bindings/sessions, synchronized O(rank) descriptors, actual static shared-memory accounting and retained native ownership |
| `norm.rs` | Numerical blocks / Gradients | ← shared `NormSpec`, device views; → native norm ABI | Asserted 64/32-byte descriptors, Tier-1 warp-per-group (`D<=128`), Tier-2 `float4` register-cached block (`128<D<=1024`) and general strided CUDA `LayerNorm`/`RmsNorm` forward and local VJPs |
| `softmax.rs` | Numerical blocks / Gradients | ← shared `SoftmaxSpec`, device views; → native softmax ABI | Asserted 80/32-byte descriptors, Tier-1B `float4` warp-per-row, Tier-1 warp-per-row, Tier-2 register-cached block and Tier-3 arbitrary-axis strided CUDA `Softmax`/`LogSoftmax` forward and local VJPs (`run` and `run_from_output`) |
| `matmul.rs` | Dense algebra / Gradients | ← shared matrix geometry and parent device/session views; → native matrix ABI | Asserted by-value 128-byte descriptor, direct cuBLAS/strided tiled routes, exact optional payload/session validation, no explicit global scratch and actual static shared-memory queries |
| `bmm.rs` | Dense algebra / Gradients | ← shared `BmmSpec` and parent device/session views; → native bmm ABI | Asserted 192/32-byte descriptors, `Collapsed2d` single `cublasSgemm`, `CublasStridedBatched` and single-launch 16x16 `TiledCuda` broadcast batched GEMM forward and local VJPs |
| `sdpa.rs` | Sequence / Gradients | ← shared `SdpaSpec` and parent device/session views; → native sdpa ABI | Asserted 264/40-byte descriptors, strict-FP32 L2-cached 64×64 SIMT SDPA GEMM + `float4` softmax + cuBLAS fallback and O(T)-memory register-tiled FlashAttention-2 (`BR=32, BC=32`) forward and zero-atomic `dQ`/`dK`/`dV` |
| `conv.rs` | Spatial / Gradients | ← shared 1D/2D spatial core and parent device/session views; → native conv-backward ABI | Asserted by-value 240-byte descriptor, direct pointwise cuBLAS and single-launch 16x16 tiled implicit-GEMM CUDA routes for 1D/2D and transposed 1D/2D forward, `dX` and `dW`, exact optional payload/session checks, zero global workspace and static shared-memory queries |
| `structural.rs` | Tensor algebra / Gradients | ← shared `PadSpec`/`CatSpec` and parent device/session views; → native structural ABI | Asserted 48/56-byte descriptors, zero-atomic arbitrary-rank strided CUDA `Pad` (`Constant`/`Reflect`/`Replicate`) and `Cat` forward and local VJPs |
| `stft.rs` | Audio / Gradients | ← shared spectral view geometry and parent device/session views; → native stft-backward ABI | Asserted by-value 32/48-byte strided descriptors, cuFFT R2C/C2R and deterministic zero-atomic CUDA forward/local-VJP plans for `Stft`, `CenteredStft` and `Istft`, CUDA Graph capture-safe execution and explicit workspace/session accounting |
| `tape.rs` | Training | ← shared native tape engine, fusion analyzer, parent session/storage and prepared primitive plans; → zero/SGD/AdamW/NVRTC/CUDA-Graph native ABI | Public CUDA tape, same-stream forward/VJP/projection dispatch across all operators, NVRTC strict-FP32 whole-step fusion (`step`, `step_scalar`, `step_adamw`, `step_scalar_adamw`, `has_fused_plan`), guarded zero-allocation CUDA Graph step capture/replay for both SGD (`step_captured`, `step_scalar_captured`) and dynamic-LR AdamW (`step_adamw_captured`, `step_scalar_adamw_captured`), exact session checks and payload/plan resource queries |
