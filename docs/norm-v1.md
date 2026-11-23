# Prepared Differentiable LayerNorm and RmsNorm (`v1`)

## Scope

This contract specifies borrowed FP32 strided forward plans, exact local reverse-mode VJPs, and
native `autograd::Builder` integration for parameter-free `LayerNorm` and `RmsNorm` over the
trailing `normalized_dims` axes on CPU and NVIDIA CUDA:

- Shared metadata (`fast_audio`): `NormKind::{LayerNorm, RmsNorm}`, `NormSpec`
- CPU plans (`fast_audio::cpu`): `NormPlan`, `NormBackward`
- CUDA plans (`fast_audio::cuda`): `NormPlan`, `NormBackward`
- Native tape builder (`fast_audio::autograd::Builder`):
  - `Builder::layer_norm(input, normalized_dims, eps)`
  - `Builder::rms_norm(input, normalized_dims, eps)`
  - `Builder::group_norm(input, num_groups, eps)` (zero-copy `[N, C, ...] -> [N, G, C/G, ...]` reshape composition over `layer_norm`)

Optional learnable affine parameters (`gamma` / `beta`) compose directly via `Builder::binary`
(`Mul` / `Add`), enabling arbitrary broadcast layouts and automatic whole-step tail fusion.

## Geometry and Equations

Let input `x` have shape `[d_0, ..., d_{r-1}]` (`r >= 1`), `normalized_dims in 1..=r`,
`eps > 0` finite FP32, `G = prod(d_0..d_{r - normalized_dims})` outer groups, and
`D = prod(d_{r - normalized_dims}..d_r)` normalized elements per group (`D > 0` for non-empty
tensors; empty tensors with `G == 0` are valid no-ops).

For each group `g in 0..G` with elements `x_0, ..., x_{D-1}`:

1. **LayerNorm**:
   - Forward:
     - `mean = (1 / D) * sum_i x_i`
     - `var = (1 / D) * sum_i (x_i - mean)^2`
     - `rstd = 1 / sqrt(var + eps)`
     - `y_i = (x_i - mean) * rstd`
   - Backward (`input_required() == true`, given saved `x` and upstream `dy`):
     - Recompute `mean`, `rstd` from `x`; let `x_hat_i = (x_i - mean) * rstd`.
     - `s1 = (1 / D) * sum_i dy_i`
     - `s2 = (1 / D) * sum_i dy_i * x_hat_i`
     - `dx_i = rstd * (dy_i - s1 - x_hat_i * s2)`

2. **RmsNorm**:
   - Forward:
     - `ms = (1 / D) * sum_i x_i^2`
     - `rstd = 1 / sqrt(ms + eps)`
     - `y_i = x_i * rstd`
   - Backward (`input_required() == true`, given saved `x` and upstream `dy`):
     - Recompute `rstd` from `x`; let `x_hat_i = x_i * rstd`.
     - `s2 = (1 / D) * sum_i dy_i * x_hat_i`
     - `dx_i = rstd * (dy_i - x_hat_i * s2)`

## Layout, Resource, and CUDA Graph Rules

- Inputs accept arbitrary signed-stride borrowed `TensorView`/`DeviceView`; outputs require
  proved non-overlapping `TensorViewMut`/`DeviceViewMut`.
- `NormBackward` supports both `GradientMode::Overwrite` and `GradientMode::Accumulate`.
- Steady-state execution performs zero host/device allocations and zero host synchronizations.
- CUDA kernels use zero atomics and static shared memory (64 bytes per block for warp-reduced
  statistics), register-cache `D <= 1024` contiguous groups to read `x` and `dy` once, and are
  100% CUDA Graph capture-compatible.
