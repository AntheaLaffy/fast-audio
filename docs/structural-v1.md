# Prepared Differentiable Pad and Cat Structural Operators (`v1`)

## Scope

This contract specifies arbitrary-rank borrowed FP32 forward plans, exact zero-atomic local
reverse-mode VJPs (zero saved forward activations), and native `autograd::Builder` integration
for `Pad` (`Constant` / `Reflect` / `Replicate`) and binary `Cat` (plus `cat_all` helper) on CPU
and NVIDIA CUDA:

- Shared metadata (`fast_audio`): `PadMode::{Constant(f32), Reflect, Replicate}`, `PadSpec`, `CatSpec`
- CPU plans (`fast_audio::cpu`): `PadPlan`, `PadBackward`, `CatPlan`, `CatBackward`
- CUDA plans (`fast_audio::cuda`): `PadPlan`, `PadBackward`, `CatPlan`, `CatBackward`
- Native tape builder (`fast_audio::autograd::Builder`):
  - `Builder::pad(input, pads, mode)`
  - `Builder::cat(left, right, axis)` / `Builder::cat_all(inputs, axis)`

## Semantics and Resource Rules

- **Pad**: per-axis `(pad_before, pad_after)` on arbitrary rank `>= 1`, supporting asymmetric
  causal audio padding `(dilation * (kernel - 1), 0)` as well as symmetric/reflect/replicate
  boundary extension. `PadBackward` (`input_required() == false`) gathers all contributing output
  coordinates per input coordinate with zero atomics.
- **Cat**: concatenates `left` and `right` along `axis in 0..rank` with identical non-axis
  dimensions. `CatBackward` (`input_dependencies() == [false, false]`) slices upstream gradients
  directly into `dLeft` or `dRight` in `Overwrite` or `Accumulate` mode.
- Zero hot allocations, zero host syncs, zero atomics, 100% CUDA Graph capture-compatible.
