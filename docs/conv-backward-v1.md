# Prepared strided 1D/2D convolutions and local derivatives

## Contract declared before implementation

Add `ConvStrategy`, `CONV_WORKSPACE_LIMIT`, `cpu::{Conv1dPlan, Conv1dBackward, Conv2dPlan, Conv2dBackward}`, and `cuda::{Conv1dPlan, Conv1dBackward, Conv2dPlan, Conv2dBackward}` for borrowed FP32 layouts, and extend `autograd::Builder` with `conv1d` and `conv2d`. This closes the 1D and 2D cross-correlation local VJP and reverse-mode tape gaps in [the coverage matrix](operator-coverage.md) and [the foundation roadmap](roadmap.md). Existing dense-owned inference plans (`cpu::Conv1d`, `cpu::PreparedConv1d`, `cuda::Conv1d`, `cpu::Conv2d`, `cuda::Conv2d`, `onednn::Conv1d`, `onednn::Conv2d`) retain their current signatures and consumers.

For 1D cross-correlation (`Conv1dSpec`), input `X[N, C_in, L_in]` and weight `W[C_out, C_in / G, K]` produce `Y[N, C_out, L_out]` with symmetric zero padding, stride, dilation, and groups `G`. For 2D cross-correlation (`Conv2dSpec` with `transposed == false` and `output_padding == [0, 0]`), input `X[N, C_in, H_in, W_in]` and weight `W[C_out, C_in / G, K_h, K_w]` produce `Y[N, C_out, H_out, W_out]`. Transposed 2D forward specs are rejected by `Conv2dPlan` and `Conv2dBackward` (`Error::Unsupported`); the input local VJP (`BinaryOperand::Left`) of forward cross-correlation computes the transposed spatial adjoint `dX`, and the weight local VJP (`BinaryOperand::Right`) computes the filter gradient `dW`. Bias addition and nonlinear activations compose explicitly through pointwise and reduction plans or tape nodes rather than hidden fused side effects.

Read layouts (`X`, `W`, `dY`) may use offsets, signed strides, permutations, padded leading dimensions, or read-overlapped broadcasts. Every output (`Y`) and local gradient (`dX`, `dW`) requires an exact logical shape and a proved injective layout (`Layout::is_non_overlapping()`). No implicit layout materialization, dtype conversion, host fallback, output allocation, or in-place read/write aliasing is permitted. Equal element counts with mismatched axis extents fail before writing.

## Preparation, selective dependencies, and execution

CPU preparation takes `Cpu`, `Conv1dSpec` or `Conv2dSpec`, and exact layouts (`[input, weight, destination]` for forward; `operand`, `[input, weight, upstream, gradient]` for backward). CUDA preparation takes `&Cuda` and the same specifications and layouts. Forward `run` always overwrites `destination`. Backward `run` takes optional `input` (`X`) and `weight` (`W`) views, required `upstream` (`dY`) and `destination` (`dX` or `dW`) views, and `GradientMode::{Overwrite, Accumulate}`.

`input_dependencies()` reports `[false, true]` for `BinaryOperand::Left` (`dX` depends only on `W`) and `[true, false]` for `BinaryOperand::Right` (`dW` depends only on `X`). No forward output `Y` is retained or required. Omitting the required operand view fails before writing; supplying an optional unneeded view still validates its exact prepared layout and originating CUDA session before any write.

Expose `spec`, `operand` (on backward plans), layout getters, `input_dependencies`, `strategy`, `workspace_bytes`, `host_metadata_bytes`, and (on CUDA) `shared_memory_bytes_per_block`. CPU execution takes `&mut self` to use its bounded per-instance packing scratch; CUDA execution takes `&self` and enqueues on the session stream with a by-value descriptor.

## Equations, accumulation, and numerical policy

For group $g \in [0, G)$, group channel sizes $C_{i,g} = C_{\text{in}} / G$ and $C_{o,g} = C_{\text{out}} / G$, and 2D spatial coordinates (where 1D cross-correlation embeds as $H_{\text{in}} = H_{\text{out}} = K_h = s_h = d_h = 1, p_h = 0$):

1. **Forward ($Y$, `GradientMode::Overwrite`):**
   $$Y[n, g C_{o,g} + c_o, h_o, w_o] = \sum_{c_i = 0}^{C_{i,g} - 1} \sum_{k_h = 0}^{K_h - 1} \sum_{k_w = 0}^{K_w - 1} W[g C_{o,g} + c_o, c_i, k_h, k_w] \cdot \tilde{X}(n, g C_{i,g} + c_i, h_o s_h + k_h d_h - p_h, w_o s_w + k_w d_w - p_w)$$
2. **Input local VJP ($dX$, `BinaryOperand::Left`):**
   $$dX[n, g C_{i,g} + c_i, h_i, w_i] = \sum_{c_o = 0}^{C_{o,g} - 1} \sum_{k_h = 0}^{K_h - 1} \sum_{k_w = 0}^{K_w - 1} W[g C_{o,g} + c_o, c_i, k_h, k_w] \cdot \widetilde{dY}(n, g C_{o,g} + c_o, h_i, w_i, k_h, k_w)$$
   where $\widetilde{dY}$ reads $dY[n, g C_{o,g} + c_o, (h_i + p_h - k_h d_h)/s_h, (w_i + p_w - k_w d_w)/s_w]$ when $h_i + p_h \ge k_h d_h$, $w_i + p_w \ge k_w d_w$, both differences are divisible by $(s_h, s_w)$, and the quotient coordinates lie in $[0, H_{\text{out}}) \times [0, W_{\text{out}})$, and $0$ otherwise.
3. **Weight local VJP ($dW$, `BinaryOperand::Right`):**
   $$dW[g C_{o,g} + c_o, c_i, k_h, k_w] = \sum_{n = 0}^{N - 1} \sum_{h_o = 0}^{H_{\text{out}} - 1} \sum_{w_o = 0}^{W_{\text{out}} - 1} dY[n, g C_{o,g} + c_o, h_o, w_o] \cdot \tilde{X}(n, g C_{i,g} + c_i, h_o s_h + k_h d_h - p_h, w_o s_w + k_w d_w - p_w)$$

Every operation partitions its injective destination so each logical output coordinate has a single writer; no floating-point atomic additions are used on CPU or CUDA. Preserve all read bits and destination holes. Overwrite mode ignores prior destination values at logical coordinates, including NaN-initialized destinations; Accumulate mode adds the convolution derivative to prior destination values. Strict FP32 precision applies throughout (TF32 disabled, `CUBLAS_PEDANTIC_MATH`); vendor GEMM FMA/reassociation and tiled accumulation satisfy the finite-input `atol = rtol = 2e-5` gate against independent FP64 and strict-FP32 PyTorch references, with exact bit repeatability across repeated executions of a prepared plan on the same stack.

## Routes and bounded resources

Publish `ConvStrategy::{Reference, Mkl, PackedMkl, Cublas, TiledCuda}` and `CONV_WORKSPACE_LIMIT = 64 * 1024`:

1. **Direct vendor GEMM (`Mkl` on `Cpu::Mkl`, `Cublas` on `Cuda`):** Selected when $K_h = K_w = s_h = s_w = 1$, $p_h = p_w = 0$, and per-`(batch, group)` spatial slices collapse into rank-two matrices compatible with vendor transpose and positive leading-dimension rules. Uses `0` explicit workspace bytes and `0` static shared memory bytes.
2. **Bounded tiled packed MKL (`PackedMkl` on `Cpu::Mkl`):** Selected for general kernels, strides, paddings, dilations, or non-BLAS strided layouts on `Cpu::Mkl`. Packs implicit-GEMM tiles of at most $32 \times 128$ ($A_g$), $128 \times 64$ ($B_g$), and $32 \times 64$ ($C_g$): at most `57,344` bytes (`<= CONV_WORKSPACE_LIMIT`), avoiding full-sequence `im2col`/`col2im` allocations. `with_workspace_limit` rejects smaller caller budgets before allocating.
3. **Portable reference (`Reference` on `Cpu::Reference`):** Evaluates the implicit-GEMM contraction in ascending index order with `0` workspace bytes.
4. **Shared-memory tiled CUDA (`TiledCuda` on `Cuda`):** Selected for general kernels, strides, paddings, dilations, or strided layouts on CUDA. Uses a single-launch $16 \times 16$ shared-memory tiled implicit-GEMM kernel across all groups with grid-stride tile reuse, `2,048` static shared memory bytes per block, `0` device-global workspace bytes, and a 240-byte by-value descriptor (no execution-time allocation or metadata upload).

Retained dynamic `host_metadata_bytes()` are `144` (`Conv1dPlan`), `192` (`Conv1dBackward` and `Conv2dPlan`), and `256` (`Conv2dBackward`) on 64-bit targets. Warmed `run` calls perform zero Rust heap allocations.

## Native reverse-mode tape composition

`autograd::Builder::conv1d(input, weight, stride, padding, dilation, groups)` and `Builder::conv2d(input, weight, stride, padding, dilation, groups)` validate operand ownership, ranks (`3` for `conv1d`, `4` for `conv2d`), channel/group divisibility, and positive output geometry before appending `Operation::Conv1d(Conv1dSpec)` or `Operation::Conv2d(Conv2dSpec)`. `cpu::Tape` and `cuda::Tape` prepare forward and reachable operand backward plans, selectively capture `SavedValue` versions according to `input_dependencies()`, project non-contiguous view gradients through `ViewBackward`, and update trainable filter/input leaves with `sgd`.

## Verification gates

Verify all routes on portable CPU, Intel MKL/oneDNN, and NVIDIA CUDA/cuDNN:

- Independent FP64 coordinate oracle, adjoint dot-product identity $\langle \text{Conv}(X, W), dY \rangle = \langle X, dX \rangle = \langle W, dW \rangle$, central finite differences (`step = 1e-4`, `atol = rtol = 2e-5`), and separate-process PyTorch comparisons (`tools/check_conv.py`).
- Coverage of pointwise ($K = 1, s = 1, p = 0$) direct BLAS and general strided/dilated/padded/grouped/depthwise 1D and 2D geometries, non-unit batches, tile tails ($15, 16, 17, 31, 32, 33, 65$), negative/permuted/gapped/broadcast read layouts, injective permuted destinations, `Overwrite` (including NaN-initialized targets) and `Accumulate` modes, changed payload bindings, bit preservation of inputs and output holes, and exact same-plan repeats.
- Resource and error checks: workspace budget rejection, `0` hot Rust allocations, exact metadata/workspace/shared-memory reports, missing dependency rejection, mismatched layout/session rejection before writes, `SavedValue` mutation guards, `ViewBackward` composition for overlapping forward views, multi-thread CPU plan independence, CUDA grid-stride tile reuse, and end-to-end 8-step `Tape` SGD trajectories against evolving FP64 and PyTorch controls.
- Full portable/MKL/CUDA test suites, Clippy, formatting, targeted signed-layout Miri, and GPU `compute-sanitizer` checks.

## Verified implementation

Implemented on 2026-11-20 across [src/conv.rs](../src/conv.rs), [src/cuda/conv.rs](../src/cuda/conv.rs), [native/conv_backward.h](../native/conv_backward.h), [native/conv_backward.cu](../native/conv_backward.cu), [src/autograd.rs](../src/autograd.rs), [src/autograd/host.rs](../src/autograd/host.rs), and [src/cuda/tape.rs](../src/cuda/tape.rs), with test/oracle suites in [tests/support/conv.rs](../tests/support/conv.rs), [tests/conv_backward.rs](../tests/conv_backward.rs), [tests/conv_backward_cuda.rs](../tests/conv_backward_cuda.rs), [tools/check_conv.py](../tools/check_conv.py), and [tests/test_conv_reference.py](../tests/test_conv_reference.py):

- **Portable CPU + Miri (local):** `cargo test --offline`, `cargo clippy --offline --all-targets -- -D warnings`, `cargo fmt --check`, 54/54 Python unit tests (`unittest discover -s tests`), `tools/check_conv.py --platform portable` (248 records vs PyTorch `2.14.1+cpu`: max abs error `7.15e-7` for 1D/2D forward/VJP/accumulate, `2.98e-8` for 8-step `Tape` SGD, `loss` error `0.0`), and `cargo +nightly miri test --offline --test conv_backward cpu_prepared_conv_miri_signed_overlap_and_bounds` (0 UB).
- **Intel oneMKL + oneDNN (Colab CPU `fa-cpu`, strictly isolated from GPU):** Full `cargo test --offline --features mkl,onednn`, release `conv_backward`, `tools/check_conv.py --platform intel` (496 records across both `reference` and `mkl` backends vs PyTorch `2.14.1+cpu`: max abs error `7.15e-7` for 1D/2D forward/VJP/accumulate, `2.98e-8` for 8-step `Tape` SGD, `loss` error `0.0`), 54/54 Python unit tests, `cargo clippy --offline --all-targets --features mkl,onednn -- -D warnings`, and `cargo fmt --check`.
- **NVIDIA CUDA + cuDNN (Colab L4 SM 8.9 `fa-gpu`, Driver 580.82.07, CUDA 13.0, PyTorch `2.14.1+cu130`):** Full `CUDA_ARCH=89 cargo test --offline --features cuda,cudnn` (debug + release `conv_backward_cuda` and all 17 doctests), `tools/check_conv.py --platform nvidia` (248 records vs strict-FP32 PyTorch CUDA: max abs error `7.15e-7` for 1D/2D forward/VJP/accumulate, `2.98e-8` for 8-step `Tape` SGD, `b1`/`b2` error `0.0`), `CUDA_ARCH=89 cargo clippy --offline --all-targets --features cuda,cudnn -- -D warnings`, and `compute-sanitizer` (`memcheck`: 0 errors, `racecheck`: 0 hazards/errors/warnings, `synccheck`: 0 errors) on both the 8-step `Tape` SGD test and the `SavedValue`/`ViewBackward`/65,537-tile grid-reuse test.

This closes the functional 1D/2D cross-correlation forward, local-VJP, and tape-training gate on CPU and CUDA. Transposed 2D forward differentiation, cuDNN/oneDNN backward performance tuning, and latency qualification against compatible PyTorch modes remain separate open items.
