# Prepared Broadcast Batched Matrix Multiplication and Local VJPs (`v1`)

## Scope

This contract specifies arbitrary-rank (`rank >= 2`) borrowed FP32 broadcast batched matrix
multiplication (`A @ B`) and exact local reverse-mode VJPs (`dA`, `dB`) on CPU and NVIDIA CUDA:

- Shared metadata (`fast_audio`): `BmmSpec`
- CPU plans (`fast_audio::cpu`): `BatchedMatmulPlan`, `BatchedMatmulBackward`
- CUDA plans (`fast_audio::cuda`): `BatchedMatmulPlan`, `BatchedMatmulBackward`
- Native tape builder (`fast_audio::autograd::Builder`): `Builder::bmm(left, right)`

## Geometry and Broadcasting

Let `A` have shape `[...batch_a..., M, K]` and `B` have shape `[...batch_b..., K, N]` with
`rank(A) >= 2`, `rank(B) >= 2`, and matching contraction extent `K`.

1. Right-align `batch_a` and `batch_b`; each axis pairs `(d_a, d_b)` where `d_a == d_b`,
   `d_a == 1`, or `d_b == 1`, producing `batch_out` and output shape `[...batch_out..., M, N]`.
2. **Forward:** For every batch multi-index `g` in `batch_out` with projected source batch indices
   `g_a`, `g_b`: `C[g] = A[g_a] @ B[g_b]`.
3. **Local VJPs (`BinaryOperand::Left` / `BinaryOperand::Right`):**
   - `dA` (`input_dependencies() == [false, true]`): for each unique target batch `g_a`,
     `dA[g_a] = sum_{g -> g_a} dC[g] @ B[g_b]^T` (zero-atomic broadcast contraction).
   - `dB` (`input_dependencies() == [true, false]`): for each unique target batch `g_b`,
     `dB[g_b] = sum_{g -> g_b} A[g_a]^T @ dC[g]` (zero-atomic broadcast contraction).

## Dispatch, Resource, and CUDA Graph Rules

- **Collapsed 2D BLAS route:** When one operand is broadcast across all batch axes and the other
  and output are contiguous across batch/rows (e.g., batched linear `[B, ..., T, K] @ [K, N]` and
  its `dX` / `dW`), collapses into a single `cblas_sgemm` / `cublasSgemm` call.
- **Strided-batched BLAS route:** When batch axes flatten to constant strides and inner matrices
  are BLAS-compatible (with no target broadcast reduction on backward), dispatches a single
  `cublasSgemmStridedBatched` (CUDA, strict FP32 `CUBLAS_PEDANTIC_MATH`) or direct `cblas_sgemm`
  loop (oneMKL) with zero workspace.
- **General tiled route:** Arbitrary signed strides and arbitrary broadcast reductions execute via
  a single-launch 16x16 shared-memory tiled CUDA kernel (`2048` static shared bytes, zero atomics)
  or safe-Rust blocked CPU kernel.
- Zero hot allocations, zero host syncs, 100% CUDA Graph capture-compatible.
