# Prepared Differentiable Flash-Style Scaled Dot-Product Attention (`v1`)

## Scope

This contract specifies strict-FP32 O(T)-memory FlashAttention-2 forward plans, exact zero-atomic
local reverse-mode VJPs (`dQ`, `dK`, `dV`), and native `autograd::Builder::sdpa` integration on
CPU and NVIDIA CUDA with multi-head (MHA) and grouped-query (GQA) support:

- Shared metadata (`fast_audio`): `SdpaOperand::{Query, Key, Value}`, `SdpaSpec`
- CPU plans (`fast_audio::cpu`): `SdpaPlan`, `SdpaBackward`
- CUDA plans (`fast_audio::cuda`): `SdpaPlan`, `SdpaBackward`
- Native tape builder (`fast_audio::autograd::Builder`):
  - `Builder::sdpa(query, key, value, scale, is_causal)`

## Geometry and Equations

Let `Q` have shape `[B, H, T_q, D]`, `K` and `V` have shape `[B, H_kv, T_k, D]` (`H % H_kv == 0`,
group size `G_q = H / H_kv`), `O` have shape `[B, H, T_q, D]`, finite positive `scale` (default
`1 / sqrt(D)`), and boolean `is_causal`.

For each batch `b`, head `h` (with KV head `h_kv = h / G_q`), and query index `i in 0..T_q`:
- Valid key range is `0..min(T_k, i + 1)` when `is_causal == true`, else `0..T_k`.
- `S_{i,j} = scale * dot(Q_{b,h,i}, K_{b,h_kv,j})`
- `P_{i,j} = softmax_j(S_{i,j})` (with `0` on masked keys `j > i`)
- `O_{b,h,i} = sum_j P_{i,j} * V_{b,h_kv,j}`

Local VJPs (`Overwrite` and `Accumulate`):
- `dQ` (`input_dependencies() == [true, true, true]`): `dQ_i = scale * sum_j P_{i,j} * (dot(dO_i, V_j) - D_i) * K_j`, where `D_i = dot(O_i, dO_i)`.
- `dK` (`input_dependencies() == [true, true, true]`): `dK_{h_kv,j} = sum_{h in group(h_kv)} sum_i dS_{h,i,j} * Q_{h,i}`.
- `dV` (`input_dependencies() == [true, true, false]`): `dV_{h_kv,j} = sum_{h in group(h_kv)} sum_i P_{h,i,j} * dO_{h,i}`.

## Resource, Layout, and CUDA Graph Rules

- All tensors accept arbitrary 4D signed-stride borrowed views (including zero-copy permuted
  `[B, T, H, D].permute(&[0, 2, 1, 3])`).
- Never materializes the `O(T_q * T_k)` attention score matrix in memory.
- CPU executes L1-tiled online-softmax FlashAttention-2 with zero hot heap allocations, runtime
  AVX2+FMA degree-6 Cody-Waite `exp256_ps`/dot/axpy kernels, and a process-global `.bss`
  `Mutex`-backed (`try_lock()`) zero-allocation backward tile cache (`SDPA_BWD_CACHE`) that
  eliminates redundant `Q @ K^T` / `dO @ V^T` / `softmax` recomputation across `dQ`, `dK`, and
  `dV`.
- CUDA executes two complementary strict-FP32 zero-atomic paths, both 100% CUDA Graph capture
  compatible with zero hot allocations:
  1. For L2-resident sequence regimes (`T_k <= 1024`, `B * H * T_q * T_k <= 8_388_608`, positive
     aligned views): context-owned preallocated `P`/`dS` buffers (eliminating redundant forward
     recomputation across `dQ`, `dK`, and `dV`), vectorized `float4` warp-per-row causal/non-causal
     softmax and softmax-backward, double-buffered 64×64 strict-FP32 SIMT GEMM kernels (`BM=64,
     BN=64, BK=16`, 128 threads, `float4` coalesced loads, single launch across all `(b, h)` even
     for permuted `[B, T, H, D].permute([0, 2, 1, 3])` views, causal upper-triangle tile skip and
     longest-tile-first scheduling, plus parallel per-query-head `dK`/`dV` scratch + 1-pass `float4`
     GQA head reduction) with strict-FP32 `cublasSgemmStridedBatched` fallback for non-64-aligned
     shapes.
  2. For long sequences (`T_k > 1024` or larger score grids) and arbitrary signed strides:
     register-tiled bank-conflict-free FlashAttention-2 kernels (`BR=32, BC=32`, `D <= 32 / 64 /
     128` plus arbitrary-`D` fallback) with `O(B * H * T_q)` preallocated backward scratch for
     `dK`/`dV` and zero workspace for forward and `dQ`.
