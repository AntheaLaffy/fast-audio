# Prepared Differentiable Softmax and LogSoftmax (`v1`)

## Scope

This contract specifies borrowed FP32 strided forward plans, exact local reverse-mode VJPs, and
native `autograd::Builder` integration for numerically stable `Softmax` and `LogSoftmax` along
any single axis `axis in 0..rank` on CPU and NVIDIA CUDA:

- Shared metadata (`fast_audio`): `SoftmaxKind::{Softmax, LogSoftmax}`, `SoftmaxSpec`
- CPU plans (`fast_audio::cpu`): `SoftmaxPlan`, `SoftmaxBackward`
- CUDA plans (`fast_audio::cuda`): `SoftmaxPlan`, `SoftmaxBackward`
- Native tape builder (`fast_audio::autograd::Builder`):
  - `Builder::softmax(input, axis)`
  - `Builder::log_softmax(input, axis)`

## Geometry and Equations

Let input `x` have shape `[d_0, ..., d_{r-1}]` (`r >= 1`), `axis in 0..r`,
`D = d_axis`, and `G = prod_{k != axis} d_k` independent slices. Empty tensors (`G == 0` or
`D == 0` when `x.is_empty()`) are valid no-ops; a positive `G > 0` with `D == 0` is rejected.

For each slice `x_0, ..., x_{D-1}` along `axis`:

1. Let `m = max_i x_i` (NaN-propagating) and `Z = sum_i exp(x_i - m)`.
2. **Softmax**:
   - Forward: `y_i = exp(x_i - m) / Z`.
   - Backward (`input_required() == true`, given saved `x` and upstream `dy`):
     Recompute `p_i = exp(x_i - m) / Z`, `dot = sum_j dy_j * p_j`,
     `dx_i = p_i * (dy_i - dot)`.
3. **LogSoftmax**:
   - Forward: `y_i = (x_i - m) - ln(Z)`.
   - Backward (`input_required() == true`, given saved `x` and upstream `dy`):
     Recompute `p_i = exp(x_i - m) / Z`, `sum_dy = sum_j dy_j`,
     `dx_i = dy_i - p_i * sum_dy`.

## Layout, Resource, and CUDA Graph Rules

- Inputs accept arbitrary signed-stride borrowed `TensorView`/`DeviceView`; outputs require
  proved non-overlapping `TensorViewMut`/`DeviceViewMut`.
- `SoftmaxBackward` supports both `GradientMode::Overwrite` and `GradientMode::Accumulate`, and
  provides both `run(&input, &upstream, &mut grad_input, mode)` (recomputing from saved `x` when
  activation liveness reuse frees `y` early) and stateless `run_from_output(&output, &upstream,
  &mut grad_input, mode)` (consuming forward output `y` directly without recomputing `exp(x - m)`
  for `Softmax` or `max`/`log_sum_exp` for `LogSoftmax`, matching PyTorch's
  `aten::_softmax_backward_data` contract).
- Steady-state execution performs zero host/device allocations and zero host synchronizations.
- Contiguous last-axis CPU routes execute runtime AVX2+FMA degree-6 Cody-Waite `exp256_ps`
  with an L1 register-direct write buffer, a process-global `.bss` `Mutex`-backed (`try_lock()`)
  zero-allocation forward-probability cache (`SOFTMAX_FWD_CACHE`), and two-pass AVX2 FMA
  `run_from_output`.
- Contiguous last-axis CUDA routes use zero-shared-memory vectorized `float4` warp-per-row kernels
  (`D <= 512`, `D % 4 == 0`, hardware `max.NaN.f32` on SM 8.0+, streaming `ld.global.cs` on inputs
  `>= 4 MiB` to preserve L2 residency of outputs), scalar warp-per-row kernels (`D <= 128`), and
  single-read register-cached block kernels (`128 < D <= 1024`), with zero atomics and 100% CUDA
  Graph capture compatibility.
