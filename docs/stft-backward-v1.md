# Prepared Differentiable STFT, Centered STFT, and Inverse STFT (`v1`)

## Scope

This contract defines borrowed FP32 strided forward plans, local reverse-mode VJPs, and native
`autograd::Builder` integration for one-sided real STFT, reflect-centered STFT, and
reflect-centered inverse STFT on CPU (portable RustFFT and Intel oneMKL DFTI) and NVIDIA CUDA
(cuFFT and deterministic zero-atomic CUDA kernels):

- CPU plans (`fast_audio::cpu`):
  - `StftPlan`, `StftBackward` (`StftSpec`, uncentered `padding = 0`)
  - `CenteredStftPlan`, `CenteredStftBackward` (`CenteredStftSpec`, reflect-padded `padding = fft_size / 2`)
  - `IstftPlan`, `IstftBackward` (`CenteredStftSpec`, NOLA-normalized overlap-add inverse)
- CUDA plans (`fast_audio::cuda`):
  - `StftPlan`, `StftBackward`
  - `CenteredStftPlan`, `CenteredStftBackward`
  - `IstftPlan`, `IstftBackward`
- Native tape builder (`fast_audio::autograd::Builder`):
  - `Builder::stft(input, fft_size, hop, window)`
  - `Builder::centered_stft(input, fft_size, hop, window)`
  - `Builder::istft(spectrum, output_len, fft_size, hop, window)`

Existing dense inference wrappers (`cpu::{Stft, CenteredStft, Istft}` and
`cuda::{Stft, CenteredStft, Istft}`) remain unchanged.

## Geometry, Windows, and Layout Rules

- Waveform shape: `[batch, input_len]` (rank 2).
- Spectrum shape: `[batch, frames, bins, 2]` (rank 4, `bins = fft_size / 2 + 1`, last axis `[real, imag]`).
- Window: finite FP32 coefficients of exact length `fft_size`, snapshotted at plan/builder
  creation (mutating caller window tensors later has no effect). Default window when omitted is
  the periodic Hann window `w[t] = 0.5 - 0.5 * cos(2 * pi * t / fft_size)`.
- All plans accept arbitrary signed-stride borrowed `TensorView`/`DeviceView` inputs and
  proved-injective `TensorViewMut`/`DeviceViewMut` outputs matching the prepared layouts.
- All three transforms are linear in their tensor input with fixed snapshotted windows; every
  local VJP has `input_required() == false` (zero saved forward activations in `autograd::Tape`).

## Forward and Exact Local VJP Equations

Let `N = fft_size`, `H = hop`, `K = N / 2 + 1`, `P = 0` (uncentered) or `P = N / 2` (centered),
and `F` be the validated frame count.

1. **STFT / Centered STFT Forward**:
   - For frame `f in 0..F`, `t in 0..N`, `p = f * H + t`:
     - `src(p) = p` when `P = 0`.
     - When `P > 0`, `q = p - P`: if `q < 0`, `src(p) = -q`; else if `q < L`, `src(p) = q`;
       else `src(p) = 2 * (L - 1) - q`.
   - `u[b, f, t] = w[t] * x[b, src(f * H + t)]`.
   - `Y[b, f, k] = sum_{t=0}^{N-1} u[b, f, t] * exp(-i * 2 * pi * k * t / N)` for `k in 0..K`.

2. **STFT / Centered STFT Backward (`dY -> dx`)**:
   - Build Hermitian one-sided spectrum `Z[b, f, 0..K]`:
     - `Z[b, f, 0] = (dY_re[b, f, 0], 0)`
     - If `N` is even: `Z[b, f, N/2] = (dY_re[b, f, N/2], 0)`
     - For `0 < k < N/2`: `Z[b, f, k] = (0.5 * dY_re[b, f, k], 0.5 * dY_im[b, f, k])`
   - Compute unscaled C2R inverse DFT `du[b, f, 0..N] = IDFT_unscaled(Z[b, f])`.
   - For each sample `(b, s)`, gather deterministically (zero atomics on GPU):
     - `sum_at(pos) = sum_{f = first(pos)..=last(pos)} du[b, f, pos - f * H] * w[pos - f * H]`
       where `first(pos) = (pos >= N) ? (pos - N) / H + 1 : 0` and
       `last(pos) = min(pos / H, F - 1)`.
     - `acc = sum_at(s + P)`
     - If `P > 0` and `1 <= s <= P`: `acc += sum_at(P - s)`
     - If `P > 0` and `L - 1 - P <= s <= L - 2`: `acc += sum_at(P + 2 * (L - 1) - s)`
   - Write `dx[b, s] = acc` (`GradientMode::Overwrite`) or `dx[b, s] += acc`
     (`GradientMode::Accumulate`).

3. **Centered Inverse STFT Forward (`S -> y`)**:
   - Precompute NOLA normalization `alpha[s] = 1 / (N * sum_{f, t: f*H+t == s+P} w[t]^2)` in FP64
     (rejecting any sample where `sum w[t]^2 < 1e-11`).
   - Sanitize `S` into Hermitian `S_tilde` with zero imaginary parts at DC (`k = 0`) and even-`N`
     Nyquist (`k = N / 2`); caller input `S` is never mutated.
   - Compute unscaled C2R inverse DFT `v[b, f, 0..N] = IDFT_unscaled(S_tilde[b, f])`.
   - Overlap-add: `y[b, s] = alpha[s] * sum_{f = first(s+P)..=last(s+P)} v[b, f, s+P - f*H] * w[s+P - f*H]`.

4. **Centered Inverse STFT Backward (`dy -> dS`)**:
   - Frame weighted upstream gradient: for `(b, f, t)`, `p = f * H + t`:
     - If `P <= p < P + L`: `dv[b, f, t] = w[t] * alpha[p - P] * dy[b, p - P]`
     - Else: `dv[b, f, t] = 0`.
   - Compute unnormalized R2C DFT `G[b, f, 0..K] = RFFT(dv[b, f])`.
   - Scale and enforce real endpoints:
     - `k = 0` or (`N` even and `k = N / 2`): `(dS_re, dS_im) = (G_re, 0)`
     - `0 < k < N / 2`: `(dS_re, dS_im) = (2 * G_re, 2 * G_im)`
   - Write (`GradientMode::Overwrite`) or add (`GradientMode::Accumulate`) into `dS`.

## Resource, Threading, Fusion, and CUDA Graph Guarantees

- All FFT descriptors and scratch buffers are allocated once during plan construction; steady-state
  `run` performs zero host/device allocations and zero hidden host synchronizations.
- CUDA kernels use zero atomics and zero dynamic shared memory (`shared_memory_bytes_per_block() == 0`)
  and enqueue strictly on the session stream, making `Stft`, `CenteredStft`, and `Istft` forward
  and backward fully compatible with `cuda::Tape::step_captured` CUDA Graph capture/replay.
- In whole-step tape fusion (`src/autograd/fusion.rs`), spectral transforms act as linear barriers
  so arbitrary post-spectral or post-iSTFT pointwise/reduction/loss/SGD tails automatically fuse
  into a single L1-tiled CPU loop or a single NVRTC strict-FP32 CUDA kernel.
