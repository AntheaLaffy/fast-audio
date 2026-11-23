---
mode: maintenance
generated_at: 2026-10-20
---

## Summary

Checked host/device storage and prepared CPU/CUDA operators with explicit output ownership.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Change shape validation or tensor storage | Storage | `tensor.rs`, `layout.rs`, `view.rs`, `spec.rs` | `copy.rs`, `cpu.rs`, `cuda.rs`, `../docs/tensor-layout-v1.md`, relevant tests |
| Change tracked/shared ownership or saved payload access | Training storage | `storage.rs`, `error.rs` | `../docs/storage-versions-v1.md`, `../docs/saved-values-v1.md`, `tensor.rs`, `cuda.rs`, `../tests/storage.rs`, `../tests/saved.rs` |
| Change view adjoints or gradient accumulation | Gradients | `adjoint.rs`, `cuda/adjoint.rs` | `copy.rs`, `layout.rs`, `../native/view_backward.cu`, `../docs/view-backward-v1.md`, `../tests/adjoint.rs` |
| Change general unary/binary math or broadcasting | Tensor algebra | `pointwise.rs`, `cuda/pointwise.rs` | `layout.rs`, `view.rs`, `../native/pointwise.cu`, `../docs/pointwise-v1.md`, `../tests/pointwise.rs` |
| Change unary/binary local VJPs or broadcast contraction | Gradients | `pointwise_backward.rs`, `cuda/pointwise_backward.rs` | `pointwise.rs`, `reduce.rs`, `storage.rs`, `../docs/pointwise-backward-v1.md`, independent CPU/CUDA gradient tests |
| Change general Sum/Mean/Max/Min/LogSumExp or local derivatives | Reductions | `reduce.rs`, `cuda/reduce.rs` | `copy.rs`, `layout.rs`, `../native/reduce.cu`, `../docs/reduction-v1.md`, `../tests/reduce.rs`, `../tests/reduce_cuda.rs`, `../tests/norm.rs`, `../tests/norm_cuda.rs` |
| Change LayerNorm/RmsNorm/GroupNorm or Softmax/LogSoftmax | Numerical blocks / Gradients | `norm.rs`, `softmax.rs`, `cuda/norm.rs`, `cuda/softmax.rs` | `../native/norm.cu`, `../native/softmax.cu`, `../docs/norm-v1.md`, `../docs/softmax-v1.md`, `../tests/norm.rs`, `../tests/norm_cuda.rs` |
| Change broadcast batched matmul or Flash SDPA | Sequence / Dense algebra | `bmm.rs`, `sdpa.rs`, `cuda/bmm.rs`, `cuda/sdpa.rs` | `../native/bmm.cu`, `../native/sdpa.cu`, `../docs/bmm-v1.md`, `../docs/sdpa-v1.md`, `../tests/bmm_sdpa.rs`, `../tests/bmm_sdpa_cuda.rs` |
| Change strided matrix multiplication or either local VJP | Dense algebra / Gradients | `matmul.rs`, `cuda/matmul.rs` | `../native/mkl.c`, `../native/matmul.cu`, `../docs/matmul-v1.md`, independent matrix/training oracles and explicit workspace/ownership checks |
| Change strided 1D/2D or transposed convolution or either local VJP | Spatial / Gradients | `conv.rs`, `cuda/conv.rs`, `spec.rs` | `../native/conv_backward.cu`, `../native/conv_backward.h`, `../docs/conv-backward-v1.md`, `../tests/conv_backward.rs`, `../tests/conv_transpose_structural.rs`, `../tests/conv_transpose_structural_cuda.rs` |
| Change differentiable Pad or Cat structural primitives | Tensor algebra / Gradients | `structural.rs`, `cuda/structural.rs` | `../native/structural.cu`, `../native/structural.h`, `../docs/structural-v1.md`, `../tests/conv_transpose_structural.rs`, `../tests/conv_transpose_structural_cuda.rs` |
| Change strided STFT/CenteredSTFT/iSTFT or local VJPs | Audio / Gradients | `stft.rs`, `cuda/stft.rs`, `spec.rs` | `../native/stft_backward.cu`, `../native/stft_backward.h`, `../docs/stft-backward-v1.md`, `../tests/stft_backward.rs`, `../tests/stft_backward_cuda.rs`, `../tools/check_stft_backward.py` |
| Change native reverse-mode programs, tapes, SGD, AdamW or checkpoints | Training | `autograd.rs`, `autograd/CODEMAP.md`, `optim.rs` | `storage.rs`, `cuda/tape.rs`, `../native/training.cu`, `../docs/autodiff-v1.md`, `../docs/optim-v1.md`, `../tests/optim.rs`, `../tests/optim_cuda.rs` |
| Optimize CPU GEMM, convolution or FFT | Intel | `cpu.rs`, `onednn.rs` | `../native/onednn.cpp`, `../native/mkl.c`, workloads and tests |
| Change CUDA ownership, stream execution or timing | NVIDIA | `cuda.rs` | `../native/cuda.cu`, `../tests/` |
| Change GRU equations, layout or prepared parameters | Sequence | `gru.rs`, `cpu.rs`, `cuda.rs` | `../native/gru.cu`, `../tests/gru.rs`, actual GTCRN captures |
| Change spatial forward/transpose convolution | Spatial | `conv2d.rs`, `onednn.rs`, `cuda/CODEMAP.md` | Native adapters, `../tests/conv2d.rs` and actual model layers |
| Change explicit spectral windows | Audio | `spec.rs`, `cpu.rs`, `cuda.rs` | `../tests/window.rs`, exact GTCRN coefficients and inverse NOLA |
| Change graph validation, lifetime or tensor transforms | Graph | `graph.rs`, `graph/CODEMAP.md`, `cuda/graph.rs` | `../native/tensor_ops.cu`, `../tests/graph.rs`, architecture contract |
| Change complete GTCRN lowering or weight folding | Model | `models/CODEMAP.md` | Immutable upstream source, complete model traces and waveform checker |
| Change benchmark CLI/workload execution | Measurement | `bin/CODEMAP.md` | `../tools/compare_torch.py` and workload schema |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
| --- | --- | --- | --- |
| [bin/](bin/CODEMAP.md) | Measurement | Public library API; serde | Standalone benchmark and output serialization |
| [cuda/](cuda/CODEMAP.md) | NVIDIA | Parent CUDA session/storage API; cuDNN | Prepared tensor, gradient, spatial and graph plans sharing parent session/storage ownership |
| [graph/](graph/CODEMAP.md) | Graph / addressing | Shared graph descriptors; public CPU plans | Prepared CPU graph backend, tensor kernels and shared coalesced/32-bit mapping preparation |
| [models/](models/CODEMAP.md) | Model | Public graph and primitive contracts | Architecture adapters for complete native inference |
| [autograd/](autograd/CODEMAP.md) | Training | Shared owners/layouts and verified local plans | Prepared forward/reverse/update execution and CPU adapter, with CUDA supplied by the sibling backend |

## Files

| File | Domain | Deps | Function |
| --- | --- | --- | --- |
| `lib.rs` | — | → `../tests/`, `bin/` | Module routing and public type re-exports |
| `error.rs` | Storage | ← storage identity/version types | Argument/allocation/capability/native and structured mutation/identity/exhaustion/borrow errors, status handling and integer limits |
| `storage.rs` | Training storage | ← owned host/device tensors and layouts | Opt-in identities/generations, shared/retained ownership, nonblocking guards, nonmutating writer preflight, checked versioned writers, poison recovery and saved views without payload copying |
| `autograd.rs` | Training topology / API | ← layouts, shapes, primitive specifications and shared owners; → `autograd/`, CPU/CUDA tapes | Branded checked DAG, metadata/detach views, forward/gradient reachability, bindings, seeds, guarded outputs, gradient validity and resource API |
| `tensor.rs` | Storage | ← layout/view contracts | Initialized dense host tensors, scalar/empty-aware bounded shapes, borrowed view binding and matmul dimensions |
| `layout.rs` | Geometry / Storage | ← `Shape` | Signed strides, cached reachable spans/injectivity, arbitrary-rank permutation/slice/broadcast and no-copy chunk reshape |
| `view.rs` | Storage | ← `Layout`, initialized slices | Shared/exclusive borrowed views, checked bindings, exact logical iteration, proved unique mutable references and explicit materialization |
| `copy.rs` | Data movement | ← layout/view contracts; → copy/adjoint plans | Shared immutable coalesced mappings, proved copy destinations and allocation-free row copy/add kernels with exact rebinding checks |
| `adjoint.rs` | Gradients | ← layouts/mapping/views; → CPU and CUDA adjoints | Explicit gradient modes/strategies, analytical broadcasts, bounded ordered index tables, interval coverage proof, fixed CPU lane trees and resource metadata |
| `pointwise.rs` | Tensor algebra | ← layouts/borrowed views, `../native/mkl.c` (mkl); → CPU/CUDA plans | Explicit broadcast inference, output-ordered normalized/coalesced geometry, twelve stable/bit-defined unary and four binary operations, optional single-threaded oneMKL VML (`VML_HA`) contiguous fast paths, dense/broadcast CPU rows and signed-stride traversal |
| `pointwise_backward.rs` | Gradients | ← pointwise/reduction/layout/view contracts; → CPU/CUDA local VJPs | Operand selection, selective dependencies, ordered broadcast contraction, stable derivatives and exact empty/bit rules; CPU preselects operation/strategy kernels with mode-hoisted slice/VML rows and bit-exact chunked 1D/2D contraction |
| `reduce.rs` | Reductions / Gradients | ← shapes/layouts/mapping/borrowed views; → CPU/CUDA plans | Explicit `Sum`/`Mean`/`Max`/`Min`/`LogSumExp` and empty/NaN/inf/tie rules, ordered coalesced mappings, fixed lane trees and MKL VML `vmsExp` fast paths |
| `norm.rs` | Numerical blocks / Gradients | ← shapes/layouts/views; → CPU/CUDA plans | Parameter-free strided `LayerNorm` and `RmsNorm` forward and local VJPs with runtime AVX2+FMA contiguous row kernels (`GroupNorm` composed via `Builder::group_norm`) |
| `softmax.rs` | Numerical blocks / Gradients | ← shapes/layouts/views; → CPU/CUDA plans | Arbitrary-axis strided numerically stable `Softmax` and `LogSoftmax` forward and local VJPs (`run` and stateless `run_from_output`) with runtime AVX2+FMA Cody-Waite `exp256_ps`, zero-allocation `.bss` forward-probability cache (`SOFTMAX_FWD_CACHE`), MKL VML `vmsExp` and 8-way accumulators |
| `loss.rs` | Training losses | ← `tensor.rs`, `dtype.rs`, `softmax.rs`, gradient modes; → CPU and CUDA loss plans | Cross-entropy geometry, ignore indices, label smoothing, shifted mean loss, AVX2/FMA row operations and destination reuse in the overwrite VJP |
| `matmul.rs` | Dense algebra / Gradients | ← layouts/borrowed views, CPU and gradient contracts; → CPU/CUDA plans | Rank-two forward/either-operand VJPs, exact geometry and selective payloads, transpose/leading-dimension routing, bounded MKL packing, zero/accumulation rules and resource queries |
| `bmm.rs` | Dense algebra / Gradients | ← layouts/borrowed views, CPU/MKL contracts; → CPU/CUDA plans | Arbitrary-rank (`rank >= 2`) broadcast batched matrix multiplication (`BmmSpec`, `BatchedMatmulPlan`, `BatchedMatmulBackward`) with `Collapsed2d` single-GEMM fast path and zero-atomic broadcast contraction |
| `sdpa.rs` | Sequence / Gradients | ← layouts/borrowed views, MKL VML; → CPU/CUDA plans | Strict-FP32 O(T)-memory Flash-style Scaled Dot-Product Attention (`SdpaSpec`, `SdpaPlan`, `SdpaBackward`) supporting MHA/GQA (`H % H_kv == 0`), causal masking, arbitrary signed-stride 4D views, runtime AVX2+FMA `exp256_ps`/dots/axpy, zero-allocation `.bss` backward-tile cache (`SDPA_BWD_CACHE`) and selective `dV` saves (`[true, true, false]`) |
| `conv.rs` | Spatial / Gradients | ← `spec.rs`, `conv2d.rs`, `matmul.rs`, layouts/borrowed views, CPU and gradient contracts; → CPU/CUDA plans | Unified rank-three/four strided 1D/2D cross-correlation (`Conv1dPlan`/`Conv2dPlan`) and transposed convolution (`ConvTranspose1dPlan`/`ConvTranspose2dPlan`) forward and either-operand local VJPs (`dX`/`dW`), pointwise BLAS compatibility, bounded 64 KiB tiled MKL packing and selective operand dependencies |
| `structural.rs` | Tensor algebra / Gradients | ← layouts/borrowed views; → CPU/CUDA plans | Arbitrary-rank strided `Pad` (`Constant`/`Reflect`/`Replicate`, zero-atomic backward gather, `input_required == false`) and `Cat` (`input_dependencies == [false, false]`) forward and local VJPs |
| `optim.rs` | Optimizers / Checkpointing | ← `autograd.rs`, `storage.rs`, `tensor.rs`, `cuda.rs`; → `cpu::AdamW`, `cuda::AdamW`, `Checkpoint` | Decoupled `AdamWConfig`, preallocated CPU/CUDA `AdamW` with deterministic FP64 global L2 gradient clipping, dynamic device-scalar LR for zero-recapture CUDA Graph training, and FNV-1a checksummed cross-device binary `Checkpoint` |
| `spec.rs` | Geometry | — | `Conv1dSpec`, `ConvTranspose1dSpec`, `StftSpec` geometry, validated reflection centering, explicit finite windows and inverse NOLA envelope |
| `stft.rs` | Audio / Gradients | ← `spec.rs`, `cpu.rs` (mkl), layouts/borrowed views, RustFFT; → CPU/CUDA spectral plans | Shared strided rank-2/rank-4 spectral view geometry and 100% safe-Rust CPU `StftPlan`/`StftBackward`/`CenteredStftPlan`/`CenteredStftBackward`/`IstftPlan`/`IstftBackward` with deterministic zero-allocation RustFFT/oneMKL execution |
| `cpu.rs` | Intel | ← `../native/mkl.c`, `onednn.rs`, `autograd/`; RustFFT | Reference/MKL GEMM, convolution, explicit-window spectral plans and public prepared-operator/tape re-exports |
| `conv2d.rs` | Spatial | ← `cpu.rs`, `onednn.rs`, `tensor.rs` | Checked NCHW geometry and 76-byte native ABI, immutable CPU parameters, pointwise/GEMM/oneDNN forward and transpose execution |
| `gru.rs` | Sequence | ← `cpu.rs`, `tensor.rs` | Checked one-layer reset-after geometry, snapshot parameters, batched input projections and allocation-free CPU recurrent execution |
| `graph.rs` | Graph | ← tensor/primitive contracts; → CPU/CUDA graph executors and model lowering | Builder-owned value identities, checked DAG/shape inference, 112-byte tensor-kernel ABI, prepared buffers and explicit state/output execution |
| `models.rs` | Model | → `models/` | Public architecture-adapter module route |
| `onednn.rs` | Intel | ← `../native/onednn.cpp`, `conv2d.rs` | Packed immutable parameter snapshots, 1D/2D convolution/deconvolution, output-padding extension and resource metadata |
| `cuda.rs` | NVIDIA | ← `../native/cuda.cu`, `../native/cudnn.cu`, `../native/gru.cu` | Session lifetime and immutable convolution policy, Conv1d/GRU/spectral plans, prepared-operator/tape re-exports and event timing |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
| --- | --- | --- |
| `error.rs` | `storage.rs` | `tensor.rs`, `storage.rs`, `spec.rs`, `cpu.rs`, `cuda.rs` |
| `storage.rs` | `tensor.rs`, `layout.rs`, `error.rs`, `cuda.rs` (feature-gated) | `lib.rs`, `error.rs`, `autograd/` |
| `autograd.rs` | `layout.rs`, `tensor.rs`, `storage.rs`, `pointwise.rs`, `reduce.rs` | `lib.rs`, `cpu.rs`, `autograd/`, `cuda/tape.rs` |
| `adjoint.rs` | `layout.rs`, `copy.rs`, `tensor.rs`, `view.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/adjoint.rs` |
| `pointwise.rs` | `layout.rs`, `view.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/pointwise.rs`, `pointwise_backward.rs` |
| `pointwise_backward.rs` | `pointwise.rs`, `reduce.rs`, `layout.rs`, `view.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/pointwise_backward.rs` |
| `reduce.rs` | `copy.rs`, `layout.rs`, `tensor.rs`, `view.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/reduce.rs`, `pointwise_backward.rs` |
| `norm.rs` | `layout.rs`, `tensor.rs`, `view.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/norm.rs`, `autograd.rs` |
| `softmax.rs` | `layout.rs`, `tensor.rs`, `view.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/softmax.rs`, `autograd.rs` |
| `matmul.rs` | `layout.rs`, `view.rs`, `cpu.rs`, `pointwise_backward.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/matmul.rs`, `conv.rs` |
| `bmm.rs` | `layout.rs`, `view.rs`, `cpu.rs`, `matmul.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/bmm.rs`, `autograd.rs` |
| `sdpa.rs` | `layout.rs`, `view.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/sdpa.rs`, `autograd.rs` |
| `conv.rs` | `spec.rs`, `conv2d.rs`, `matmul.rs`, `layout.rs`, `view.rs`, `cpu.rs`, `pointwise_backward.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/conv.rs` |
| `structural.rs` | `layout.rs`, `view.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/structural.rs`, `autograd.rs` |
| `optim.rs` | `autograd.rs`, `storage.rs`, `tensor.rs`, `layout.rs`, `error.rs`, `cuda.rs` (feature-gated) | `lib.rs`, `cpu.rs`, `cuda.rs`, `autograd/host.rs`, `cuda/tape.rs` |
| `stft.rs` | `spec.rs`, `layout.rs`, `view.rs`, `tensor.rs`, `cpu.rs`, `adjoint.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda/stft.rs` |
| `tensor.rs` | `error.rs` | `spec.rs`, `cpu.rs`, `cuda.rs` |
| `spec.rs` | `error.rs`, `tensor.rs` | `cpu.rs`, `cuda.rs` |
| `cpu.rs` | `spec.rs`, `tensor.rs`, `error.rs`, `onednn.rs`, `gru.rs`, `conv2d.rs`, `graph/`, `autograd/` | `lib.rs`, `gru.rs`, `conv2d.rs`, `graph/`, `autograd/` |
| `conv2d.rs` | `cpu.rs`, `tensor.rs`, `error.rs`, `onednn.rs` | `lib.rs`, `cpu.rs`, `onednn.rs`, `cuda/` |
| `gru.rs` | `cpu.rs`, `tensor.rs`, `error.rs` | `lib.rs`, `cpu.rs`, `cuda.rs` |
| `onednn.rs` | `spec.rs`, `conv2d.rs`, `tensor.rs`, `error.rs` | `cpu.rs`, `conv2d.rs`, `lib.rs` |
| `cuda.rs` | `spec.rs`, `tensor.rs`, `error.rs`, `cuda/` | `lib.rs`, `cuda/` |
| `graph.rs` | `tensor.rs`, `error.rs`, `conv2d.rs`, `gru.rs`, `graph/` | `lib.rs`, `models/`, `cuda/` |
| `models.rs` | `models/` | `lib.rs` |
