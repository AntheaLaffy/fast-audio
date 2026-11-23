#include "cuda_backend.h"
#include "sdpa.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

namespace {

struct SdpaPlanCuda {
    Context *ctx;
    FaSdpaDims dims;
    float *lse_buf = nullptr;   // B * H * T_q (for dK and dV)
    float *drow_buf = nullptr;  // B * H * T_q (for dK)
    SdpaPlanCuda(Context *c, FaSdpaDims d) : ctx(c), dims(d) {}
    ~SdpaPlanCuda() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (lse_buf) cudaFree(lse_buf);
        if (drow_buf) cudaFree(drow_buf);
    }
};

__host__ __device__ __forceinline__ uint64_t umin64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

__device__ __forceinline__ int64_t row_base(const FaSdpaView4 &v, uint64_t b, uint64_t h, uint64_t t) {
    return v.offset + static_cast<int64_t>(b) * v.strides[0]
                    + static_cast<int64_t>(h) * v.strides[1]
                    + static_cast<int64_t>(t) * v.strides[2];
}

__device__ __forceinline__ float subwarp8_sum(float v) {
    const unsigned mask8 = 0xffu << (threadIdx.x & 24);
    v += __shfl_xor_sync(mask8, v, 4);
    v += __shfl_xor_sync(mask8, v, 2);
    v += __shfl_xor_sync(mask8, v, 1);
    return v;
}

__device__ __forceinline__ float warp_sum_sdpa(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_sum_sdpa(float v, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum_sdpa(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < 8) ? smem[lane] : 0.0f;
        x = warp_sum_sdpa(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    const float res = smem[0];
    __syncthreads();
    return res;
}

// =========================================================================
// Fast Register-Tiled FlashAttention-2 Forward (BR=32, BC=32, 256 threads/block)
// 8 lanes per query row -> Q and O live 100% in registers! Zero T_q*T_k VRAM!
// =========================================================================
template <int D_MAX>
__global__ void sdpa_fwd_flash(FaSdpaDims d,
                               const float *__restrict__ Q,
                               const float *__restrict__ K,
                               const float *__restrict__ V,
                               float *__restrict__ O) {
    constexpr int BR = 32;
    constexpr int BC = 32;
    constexpr int D_PER_T = D_MAX / 8;
    constexpr int D_PAD = D_MAX + 1;
    __shared__ float sK[BC][D_PAD];
    __shared__ float sV[BC][D_PAD];

    const uint64_t q_blocks = (d.seq_q + BR - 1) / BR;
    const uint64_t bh_total = d.batch * d.num_heads;
    const uint64_t gqa = d.num_heads / d.num_kv_heads;
    const uint64_t total_tiles = bh_total * q_blocks;

    const int row_local = threadIdx.x >> 3; // 0..31
    const int lane8 = threadIdx.x & 7;      // 0..7
    const uint64_t D = d.head_dim;

    for (uint64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const uint64_t bh = tile / q_blocks;
        const uint64_t qb_idx = tile % q_blocks;
        const uint64_t b = bh / d.num_heads;
        const uint64_t h = bh % d.num_heads;
        const uint64_t h_kv = h / gqa;
        const uint64_t q_start = qb_idx * BR;
        const uint64_t q_row = q_start + row_local;
        const bool q_valid = (q_row < d.seq_q);

        float q_reg[D_PER_T];
        float o_reg[D_PER_T];
        const int64_t q_off = q_valid ? row_base(d.q, b, h, q_row) : 0;
        #pragma unroll
        for (int r = 0; r < D_PER_T; ++r) {
            const uint64_t dim = lane8 + (r << 3);
            q_reg[r] = (q_valid && dim < D) ? Q[q_off + dim] : 0.0f;
            o_reg[r] = 0.0f;
        }

        float m_prev = -INFINITY;
        float l_acc = 0.0f;

        const uint64_t k_max_tile = d.is_causal
            ? umin64(d.seq_k, q_start + BR)
            : d.seq_k;

        for (uint64_t j0 = 0; j0 < k_max_tile; j0 += BC) {
            const uint64_t tile_k = umin64(BC, d.seq_k - j0);
            // Collaborative load of K and V tiles into padded shared memory
            const uint64_t k_row = j0 + row_local;
            const bool k_valid = (row_local < tile_k);
            const int64_t k_off = k_valid ? row_base(d.k, b, h_kv, k_row) : 0;
            const int64_t v_off = k_valid ? row_base(d.v, b, h_kv, k_row) : 0;
            #pragma unroll
            for (int r = 0; r < D_PER_T; ++r) {
                const uint64_t dim = lane8 + (r << 3);
                sK[row_local][dim] = (k_valid && dim < D) ? K[k_off + dim] : 0.0f;
                sV[row_local][dim] = (k_valid && dim < D) ? V[v_off + dim] : 0.0f;
            }
            __syncthreads();

            // Compute BC scores for this query row
            float s_tile[BC];
            float m_tile = -INFINITY;
            for (uint64_t c = 0; c < tile_k; ++c) {
                const uint64_t k_idx = j0 + c;
                if (d.is_causal && k_idx > q_row) {
                    s_tile[c] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
                #pragma unroll
                for (int r = 0; r < D_PER_T; ++r) {
                    const uint64_t dim = lane8 + (r << 3);
                    dot = fmaf(q_reg[r], sK[c][dim], dot);
                }
                dot = subwarp8_sum(dot) * d.scale;
                s_tile[c] = dot;
                if (isnan(dot) || dot > m_tile) m_tile = dot;
            }

            if (q_valid && m_tile > -INFINITY) {
                const float m_new = (isnan(m_prev) || isnan(m_tile))
                    ? __uint_as_float(0x7fc00000u)
                    : fmaxf(m_prev, m_tile);
                const float alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
                if (alpha != 1.0f) {
                    #pragma unroll
                    for (int r = 0; r < D_PER_T; ++r) o_reg[r] *= alpha;
                    l_acc *= alpha;
                }
                for (uint64_t c = 0; c < tile_k; ++c) {
                    const uint64_t k_idx = j0 + c;
                    if (d.is_causal && k_idx > q_row) continue;
                    const float p = expf(s_tile[c] - m_new);
                    l_acc += p;
                    #pragma unroll
                    for (int r = 0; r < D_PER_T; ++r) {
                        const uint64_t dim = lane8 + (r << 3);
                        o_reg[r] = fmaf(p, sV[c][dim], o_reg[r]);
                    }
                }
                m_prev = m_new;
            }
            __syncthreads();
        }

        if (q_valid) {
            const float inv_l = 1.0f / l_acc;
            const int64_t o_off = row_base(d.dy_or_o, b, h, q_row);
            #pragma unroll
            for (int r = 0; r < D_PER_T; ++r) {
                const uint64_t dim = lane8 + (r << 3);
                if (dim < D) O[o_off + dim] = o_reg[r] * inv_l;
            }
        }
    }
}

// =========================================================================
// Fast Register-Tiled FlashAttention-2 Backward: dQ (Single Kernel, 0 Workspace, 0 Atomics!)
// =========================================================================
template <int D_MAX, bool accumulate>
__global__ void sdpa_bwd_dq_flash(FaSdpaDims d,
                                  const float *__restrict__ Q,
                                  const float *__restrict__ K,
                                  const float *__restrict__ V,
                                  const float *__restrict__ dO,
                                  float *__restrict__ dQ) {
    constexpr int BR = 32;
    constexpr int BC = 32;
    constexpr int D_PER_T = D_MAX / 8;
    constexpr int D_PAD = D_MAX + 1;
    __shared__ float sK[BC][D_PAD];
    __shared__ float sV[BC][D_PAD];

    const uint64_t q_blocks = (d.seq_q + BR - 1) / BR;
    const uint64_t bh_total = d.batch * d.num_heads;
    const uint64_t gqa = d.num_heads / d.num_kv_heads;
    const uint64_t total_tiles = bh_total * q_blocks;

    const int row_local = threadIdx.x >> 3;
    const int lane8 = threadIdx.x & 7;
    const uint64_t D = d.head_dim;

    for (uint64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const uint64_t bh = tile / q_blocks;
        const uint64_t qb_idx = tile % q_blocks;
        const uint64_t b = bh / d.num_heads;
        const uint64_t h = bh % d.num_heads;
        const uint64_t h_kv = h / gqa;
        const uint64_t q_start = qb_idx * BR;
        const uint64_t q_row = q_start + row_local;
        const bool q_valid = (q_row < d.seq_q);

        float q_reg[D_PER_T];
        float do_reg[D_PER_T];
        float dq_reg[D_PER_T];
        const int64_t q_off = q_valid ? row_base(d.q, b, h, q_row) : 0;
        const int64_t do_off = q_valid ? row_base(d.dy_or_o, b, h, q_row) : 0;
        #pragma unroll
        for (int r = 0; r < D_PER_T; ++r) {
            const uint64_t dim = lane8 + (r << 3);
            q_reg[r] = (q_valid && dim < D) ? Q[q_off + dim] : 0.0f;
            do_reg[r] = (q_valid && dim < D) ? dO[do_off + dim] : 0.0f;
            dq_reg[r] = 0.0f;
        }

        // Pass 1: compute LSE_i and D_i = sum_j P_{i,j} * dot(dO_i, V_j) in registers!
        float m_prev = -INFINITY;
        float l_acc = 0.0f;
        float d_num = 0.0f;
        const uint64_t k_max_tile = d.is_causal
            ? umin64(d.seq_k, q_start + BR)
            : d.seq_k;

        for (uint64_t j0 = 0; j0 < k_max_tile; j0 += BC) {
            const uint64_t tile_k = umin64(BC, d.seq_k - j0);
            const uint64_t k_row = j0 + row_local;
            const bool k_valid = (row_local < tile_k);
            const int64_t k_off = k_valid ? row_base(d.k, b, h_kv, k_row) : 0;
            const int64_t v_off = k_valid ? row_base(d.v, b, h_kv, k_row) : 0;
            #pragma unroll
            for (int r = 0; r < D_PER_T; ++r) {
                const uint64_t dim = lane8 + (r << 3);
                sK[row_local][dim] = (k_valid && dim < D) ? K[k_off + dim] : 0.0f;
                sV[row_local][dim] = (k_valid && dim < D) ? V[v_off + dim] : 0.0f;
            }
            __syncthreads();

            float s_tile[BC];
            float dp_tile[BC];
            float m_tile = -INFINITY;
            for (uint64_t c = 0; c < tile_k; ++c) {
                const uint64_t k_idx = j0 + c;
                if (d.is_causal && k_idx > q_row) {
                    s_tile[c] = -INFINITY;
                    dp_tile[c] = 0.0f;
                    continue;
                }
                float dot_qk = 0.0f;
                float dot_dov = 0.0f;
                #pragma unroll
                for (int r = 0; r < D_PER_T; ++r) {
                    const uint64_t dim = lane8 + (r << 3);
                    dot_qk = fmaf(q_reg[r], sK[c][dim], dot_qk);
                    dot_dov = fmaf(do_reg[r], sV[c][dim], dot_dov);
                }
                dot_qk = subwarp8_sum(dot_qk) * d.scale;
                dot_dov = subwarp8_sum(dot_dov);
                s_tile[c] = dot_qk;
                dp_tile[c] = dot_dov;
                if (isnan(dot_qk) || dot_qk > m_tile) m_tile = dot_qk;
            }

            if (q_valid && m_tile > -INFINITY) {
                const float m_new = (isnan(m_prev) || isnan(m_tile))
                    ? __uint_as_float(0x7fc00000u)
                    : fmaxf(m_prev, m_tile);
                const float alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
                l_acc *= alpha;
                d_num *= alpha;
                for (uint64_t c = 0; c < tile_k; ++c) {
                    const uint64_t k_idx = j0 + c;
                    if (d.is_causal && k_idx > q_row) continue;
                    const float p = expf(s_tile[c] - m_new);
                    l_acc += p;
                    d_num += p * dp_tile[c];
                }
                m_prev = m_new;
            }
            __syncthreads();
        }

        const float lse = m_prev + logf(l_acc);
        const float d_row = d_num / l_acc;

        // Pass 2: accumulate dQ in registers!
        for (uint64_t j0 = 0; j0 < k_max_tile; j0 += BC) {
            const uint64_t tile_k = umin64(BC, d.seq_k - j0);
            const uint64_t k_row = j0 + row_local;
            const bool k_valid = (row_local < tile_k);
            const int64_t k_off = k_valid ? row_base(d.k, b, h_kv, k_row) : 0;
            const int64_t v_off = k_valid ? row_base(d.v, b, h_kv, k_row) : 0;
            #pragma unroll
            for (int r = 0; r < D_PER_T; ++r) {
                const uint64_t dim = lane8 + (r << 3);
                sK[row_local][dim] = (k_valid && dim < D) ? K[k_off + dim] : 0.0f;
                sV[row_local][dim] = (k_valid && dim < D) ? V[v_off + dim] : 0.0f;
            }
            __syncthreads();

            if (q_valid) {
                for (uint64_t c = 0; c < tile_k; ++c) {
                    const uint64_t k_idx = j0 + c;
                    if (d.is_causal && k_idx > q_row) continue;
                    float dot_qk = 0.0f;
                    float dot_dov = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < D_PER_T; ++r) {
                        const uint64_t dim = lane8 + (r << 3);
                        dot_qk = fmaf(q_reg[r], sK[c][dim], dot_qk);
                        dot_dov = fmaf(do_reg[r], sV[c][dim], dot_dov);
                    }
                    const float p = expf(subwarp8_sum(dot_qk) * d.scale - lse);
                    const float dp = subwarp8_sum(dot_dov);
                    const float ds = p * (dp - d_row) * d.scale;
                    #pragma unroll
                    for (int r = 0; r < D_PER_T; ++r) {
                        const uint64_t dim = lane8 + (r << 3);
                        dq_reg[r] = fmaf(ds, sK[c][dim], dq_reg[r]);
                    }
                }
            }
            __syncthreads();
        }

        if (q_valid) {
            const int64_t out_off = row_base(d.out, b, h, q_row);
            #pragma unroll
            for (int r = 0; r < D_PER_T; ++r) {
                const uint64_t dim = lane8 + (r << 3);
                if (dim < D) {
                    if constexpr (accumulate) dQ[out_off + dim] += dq_reg[r];
                    else dQ[out_off + dim] = dq_reg[r];
                }
            }
        }
    }
}

// =========================================================================
// Prepass for dK (computes L and D_row) / dV (computes L only)
// =========================================================================
template <int D_MAX, bool need_drow>
__global__ void sdpa_bwd_prep_flash(FaSdpaDims d,
                                    const float *__restrict__ Q,
                                    const float *__restrict__ K,
                                    const float *__restrict__ V,
                                    const float *__restrict__ dO,
                                    float *__restrict__ L_out,
                                    float *__restrict__ D_out) {
    constexpr int BR = 32;
    constexpr int BC = 32;
    constexpr int D_PER_T = D_MAX / 8;
    constexpr int D_PAD = D_MAX + 1;
    __shared__ float sK[BC][D_PAD];
    __shared__ float sV[BC][D_PAD];

    const uint64_t q_blocks = (d.seq_q + BR - 1) / BR;
    const uint64_t bh_total = d.batch * d.num_heads;
    const uint64_t gqa = d.num_heads / d.num_kv_heads;
    const uint64_t total_tiles = bh_total * q_blocks;
    const int row_local = threadIdx.x >> 3;
    const int lane8 = threadIdx.x & 7;
    const uint64_t D = d.head_dim;

    for (uint64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const uint64_t bh = tile / q_blocks;
        const uint64_t qb_idx = tile % q_blocks;
        const uint64_t b = bh / d.num_heads;
        const uint64_t h = bh % d.num_heads;
        const uint64_t h_kv = h / gqa;
        const uint64_t q_start = qb_idx * BR;
        const uint64_t q_row = q_start + row_local;
        const bool q_valid = (q_row < d.seq_q);

        float q_reg[D_PER_T];
        float do_reg[D_PER_T];
        const int64_t q_off = q_valid ? row_base(d.q, b, h, q_row) : 0;
        const int64_t do_off = (need_drow && q_valid) ? row_base(d.dy_or_o, b, h, q_row) : 0;
        #pragma unroll
        for (int r = 0; r < D_PER_T; ++r) {
            const uint64_t dim = lane8 + (r << 3);
            q_reg[r] = (q_valid && dim < D) ? Q[q_off + dim] : 0.0f;
            if constexpr (need_drow) {
                do_reg[r] = (q_valid && dim < D) ? dO[do_off + dim] : 0.0f;
            }
        }

        float m_prev = -INFINITY;
        float l_acc = 0.0f;
        float d_num = 0.0f;
        const uint64_t k_max_tile = d.is_causal
            ? umin64(d.seq_k, q_start + BR)
            : d.seq_k;

        for (uint64_t j0 = 0; j0 < k_max_tile; j0 += BC) {
            const uint64_t tile_k = umin64(BC, d.seq_k - j0);
            const uint64_t k_row = j0 + row_local;
            const bool k_valid = (row_local < tile_k);
            const int64_t k_off = k_valid ? row_base(d.k, b, h_kv, k_row) : 0;
            const int64_t v_off = (need_drow && k_valid) ? row_base(d.v, b, h_kv, k_row) : 0;
            #pragma unroll
            for (int r = 0; r < D_PER_T; ++r) {
                const uint64_t dim = lane8 + (r << 3);
                sK[row_local][dim] = (k_valid && dim < D) ? K[k_off + dim] : 0.0f;
                if constexpr (need_drow) {
                    sV[row_local][dim] = (k_valid && dim < D) ? V[v_off + dim] : 0.0f;
                }
            }
            __syncthreads();

            float s_tile[BC];
            float dp_tile[BC];
            float m_tile = -INFINITY;
            for (uint64_t c = 0; c < tile_k; ++c) {
                const uint64_t k_idx = j0 + c;
                if (d.is_causal && k_idx > q_row) {
                    s_tile[c] = -INFINITY;
                    dp_tile[c] = 0.0f;
                    continue;
                }
                float dot_qk = 0.0f;
                float dot_dov = 0.0f;
                #pragma unroll
                for (int r = 0; r < D_PER_T; ++r) {
                    const uint64_t dim = lane8 + (r << 3);
                    dot_qk = fmaf(q_reg[r], sK[c][dim], dot_qk);
                    if constexpr (need_drow) {
                        dot_dov = fmaf(do_reg[r], sV[c][dim], dot_dov);
                    }
                }
                dot_qk = subwarp8_sum(dot_qk) * d.scale;
                s_tile[c] = dot_qk;
                if constexpr (need_drow) dp_tile[c] = subwarp8_sum(dot_dov);
                if (isnan(dot_qk) || dot_qk > m_tile) m_tile = dot_qk;
            }

            if (q_valid && m_tile > -INFINITY) {
                const float m_new = (isnan(m_prev) || isnan(m_tile))
                    ? __uint_as_float(0x7fc00000u)
                    : fmaxf(m_prev, m_tile);
                const float alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
                l_acc *= alpha;
                if constexpr (need_drow) d_num *= alpha;
                for (uint64_t c = 0; c < tile_k; ++c) {
                    const uint64_t k_idx = j0 + c;
                    if (d.is_causal && k_idx > q_row) continue;
                    const float p = expf(s_tile[c] - m_new);
                    l_acc += p;
                    if constexpr (need_drow) d_num += p * dp_tile[c];
                }
                m_prev = m_new;
            }
            __syncthreads();
        }

        if (q_valid && lane8 == 0) {
            const uint64_t idx = bh * d.seq_q + q_row;
            L_out[idx] = m_prev + logf(l_acc);
            if constexpr (need_drow) D_out[idx] = d_num / l_acc;
        }
    }
}

// =========================================================================
// Fast Register-Tiled FlashAttention-2 Backward: dK and dV (KV-Parallel, 0 Atomics!)
// =========================================================================
template <int D_MAX, bool is_dk, bool accumulate>
__global__ void sdpa_bwd_kv_flash(FaSdpaDims d,
                                  const float *__restrict__ Q,
                                  const float *__restrict__ K,
                                  const float *__restrict__ V,
                                  const float *__restrict__ dO,
                                  const float *__restrict__ L_in,
                                  const float *__restrict__ D_in,
                                  float *__restrict__ dKV) {
    constexpr int BR = 32;
    constexpr int BC = 32;
    constexpr int D_PER_T = D_MAX / 8;
    constexpr int D_PAD = D_MAX + 1;
    __shared__ float sQ[BR][D_PAD];
    __shared__ float sdO[BR][D_PAD];
    __shared__ float sL[BR];
    __shared__ float sD[BR];

    const uint64_t k_blocks = (d.seq_k + BC - 1) / BC;
    const uint64_t bkv_total = d.batch * d.num_kv_heads;
    const uint64_t gqa = d.num_heads / d.num_kv_heads;
    const uint64_t total_tiles = bkv_total * k_blocks;

    const int row_local = threadIdx.x >> 3;
    const int lane8 = threadIdx.x & 7;
    const uint64_t D = d.head_dim;

    for (uint64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const uint64_t bkv = tile / k_blocks;
        const uint64_t kb_idx = tile % k_blocks;
        const uint64_t b = bkv / d.num_kv_heads;
        const uint64_t h_kv = bkv % d.num_kv_heads;
        const uint64_t k_start = kb_idx * BC;
        const uint64_t k_row = k_start + row_local;
        const bool k_valid = (k_row < d.seq_k);

        float k_reg[D_PER_T];
        float v_reg[D_PER_T];
        float acc_reg[D_PER_T];
        const int64_t k_off = k_valid ? row_base(d.k, b, h_kv, k_row) : 0;
        const int64_t v_off = (is_dk && k_valid) ? row_base(d.v, b, h_kv, k_row) : 0;
        #pragma unroll
        for (int r = 0; r < D_PER_T; ++r) {
            const uint64_t dim = lane8 + (r << 3);
            k_reg[r] = (k_valid && dim < D) ? K[k_off + dim] : 0.0f;
            if constexpr (is_dk) {
                v_reg[r] = (k_valid && dim < D) ? V[v_off + dim] : 0.0f;
            }
            acc_reg[r] = 0.0f;
        }

        const uint64_t q_begin = d.is_causal ? (k_start / BR) * BR : 0;
        for (uint64_t hg = 0; hg < gqa; ++hg) {
            const uint64_t h = h_kv * gqa + hg;
            const uint64_t bh = b * d.num_heads + h;
            for (uint64_t i0 = q_begin; i0 < d.seq_q; i0 += BR) {
                const uint64_t tile_q = umin64(BR, d.seq_q - i0);
                const uint64_t q_row = i0 + row_local;
                const bool q_valid = (row_local < tile_q);
                const int64_t q_off = q_valid ? row_base(d.q, b, h, q_row) : 0;
                const int64_t do_off = q_valid ? row_base(d.dy_or_o, b, h, q_row) : 0;
                #pragma unroll
                for (int r = 0; r < D_PER_T; ++r) {
                    const uint64_t dim = lane8 + (r << 3);
                    sQ[row_local][dim] = (q_valid && dim < D) ? Q[q_off + dim] : 0.0f;
                    sdO[row_local][dim] = (q_valid && dim < D) ? dO[do_off + dim] : 0.0f;
                }
                if (lane8 == 0) {
                    sL[row_local] = q_valid ? L_in[bh * d.seq_q + q_row] : 0.0f;
                    if constexpr (is_dk) {
                        sD[row_local] = q_valid ? D_in[bh * d.seq_q + q_row] : 0.0f;
                    }
                }
                __syncthreads();

                if (k_valid) {
                    for (uint64_t r = 0; r < tile_q; ++r) {
                        const uint64_t qi = i0 + r;
                        if (d.is_causal && k_row > qi) continue;
                        float dot_qk = 0.0f;
                        float dot_dov = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < D_PER_T; ++k) {
                            const uint64_t dim = lane8 + (k << 3);
                            dot_qk = fmaf(k_reg[k], sQ[r][dim], dot_qk);
                            if constexpr (is_dk) {
                                dot_dov = fmaf(v_reg[k], sdO[r][dim], dot_dov);
                            }
                        }
                        const float p = expf(subwarp8_sum(dot_qk) * d.scale - sL[r]);
                        if constexpr (is_dk) {
                            const float ds = p * (subwarp8_sum(dot_dov) - sD[r]) * d.scale;
                            #pragma unroll
                            for (int k = 0; k < D_PER_T; ++k) {
                                const uint64_t dim = lane8 + (k << 3);
                                acc_reg[k] = fmaf(ds, sQ[r][dim], acc_reg[k]);
                            }
                        } else {
                            #pragma unroll
                            for (int k = 0; k < D_PER_T; ++k) {
                                const uint64_t dim = lane8 + (k << 3);
                                acc_reg[k] = fmaf(p, sdO[r][dim], acc_reg[k]);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }

        if (k_valid) {
            const int64_t out_off = row_base(d.out, b, h_kv, k_row);
            #pragma unroll
            for (int r = 0; r < D_PER_T; ++r) {
                const uint64_t dim = lane8 + (r << 3);
                if (dim < D) {
                    if constexpr (accumulate) dKV[out_off + dim] += acc_reg[r];
                    else dKV[out_off + dim] = acc_reg[r];
                }
            }
        }
    }
}

// =========================================================================
// General Arbitrary-D / Arbitrary-Stride Online-Softmax Fallback Kernels
// =========================================================================
__global__ void sdpa_fwd_general(FaSdpaDims d,
                                 const float *__restrict__ Q,
                                 const float *__restrict__ K,
                                 const float *__restrict__ V,
                                 float *__restrict__ O) {
    __shared__ float smem[8];
    const uint64_t total_rows = d.batch * d.num_heads * d.seq_q;
    const uint64_t gqa = d.num_heads / d.num_kv_heads;
    const uint64_t D = d.head_dim;

    for (uint64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
        const uint64_t i = row % d.seq_q;
        const uint64_t bh = row / d.seq_q;
        const uint64_t b = bh / d.num_heads;
        const uint64_t h = bh % d.num_heads;
        const uint64_t h_kv = h / gqa;
        const uint64_t k_lim = d.is_causal ? umin64(d.seq_k, i + 1) : d.seq_k;

        const int64_t qb = row_base(d.q, b, h, i);
        const int64_t ob = row_base(d.dy_or_o, b, h, i);

        // Pass 1: find max and sum_exp
        float m = -INFINITY;
        for (uint64_t j = 0; j < k_lim; ++j) {
            const int64_t kb = row_base(d.k, b, h_kv, j);
            float part = 0.0f;
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                part += Q[qb + static_cast<int64_t>(dim) * d.q.strides[3]] *
                        K[kb + static_cast<int64_t>(dim) * d.k.strides[3]];
            }
            const float sc = block_sum_sdpa(part, smem) * d.scale;
            if (isnan(sc) || sc > m) m = sc;
        }
        float l_sum = 0.0f;
        for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
            O[ob + static_cast<int64_t>(dim) * d.dy_or_o.strides[3]] = 0.0f;
        }
        for (uint64_t j = 0; j < k_lim; ++j) {
            const int64_t kb = row_base(d.k, b, h_kv, j);
            const int64_t vb = row_base(d.v, b, h_kv, j);
            float part = 0.0f;
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                part += Q[qb + static_cast<int64_t>(dim) * d.q.strides[3]] *
                        K[kb + static_cast<int64_t>(dim) * d.k.strides[3]];
            }
            const float p = expf(block_sum_sdpa(part, smem) * d.scale - m);
            l_sum += p;
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                O[ob + static_cast<int64_t>(dim) * d.dy_or_o.strides[3]] +=
                    p * V[vb + static_cast<int64_t>(dim) * d.v.strides[3]];
            }
        }
        const float inv_l = 1.0f / l_sum;
        for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
            O[ob + static_cast<int64_t>(dim) * d.dy_or_o.strides[3]] *= inv_l;
        }
    }
}

__global__ void sdpa_bwd_prep_general(FaSdpaDims d,
                                      const float *__restrict__ Q,
                                      const float *__restrict__ K,
                                      const float *__restrict__ V,
                                      const float *__restrict__ dO,
                                      float *__restrict__ L_out,
                                      float *__restrict__ D_out,
                                      int need_drow) {
    __shared__ float smem[8];
    const uint64_t total_rows = d.batch * d.num_heads * d.seq_q;
    const uint64_t gqa = d.num_heads / d.num_kv_heads;
    const uint64_t D = d.head_dim;

    for (uint64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
        const uint64_t i = row % d.seq_q;
        const uint64_t bh = row / d.seq_q;
        const uint64_t b = bh / d.num_heads;
        const uint64_t h = bh % d.num_heads;
        const uint64_t h_kv = h / gqa;
        const uint64_t k_lim = d.is_causal ? umin64(d.seq_k, i + 1) : d.seq_k;

        const int64_t qb = row_base(d.q, b, h, i);
        const int64_t dob = need_drow ? row_base(d.dy_or_o, b, h, i) : 0;

        float m = -INFINITY;
        for (uint64_t j = 0; j < k_lim; ++j) {
            const int64_t kb = row_base(d.k, b, h_kv, j);
            float part = 0.0f;
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                part += Q[qb + static_cast<int64_t>(dim) * d.q.strides[3]] *
                        K[kb + static_cast<int64_t>(dim) * d.k.strides[3]];
            }
            const float sc = block_sum_sdpa(part, smem) * d.scale;
            if (isnan(sc) || sc > m) m = sc;
        }
        float l_sum = 0.0f, d_num = 0.0f;
        for (uint64_t j = 0; j < k_lim; ++j) {
            const int64_t kb = row_base(d.k, b, h_kv, j);
            const int64_t vb = need_drow ? row_base(d.v, b, h_kv, j) : 0;
            float pqk = 0.0f, pdov = 0.0f;
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                pqk += Q[qb + static_cast<int64_t>(dim) * d.q.strides[3]] *
                       K[kb + static_cast<int64_t>(dim) * d.k.strides[3]];
                if (need_drow) {
                    pdov += dO[dob + static_cast<int64_t>(dim) * d.dy_or_o.strides[3]] *
                            V[vb + static_cast<int64_t>(dim) * d.v.strides[3]];
                }
            }
            const float p = expf(block_sum_sdpa(pqk, smem) * d.scale - m);
            l_sum += p;
            if (need_drow) d_num += p * block_sum_sdpa(pdov, smem);
        }
        if (threadIdx.x == 0) {
            L_out[row] = m + logf(l_sum);
            if (need_drow) D_out[row] = d_num / l_sum;
        }
    }
}

__global__ void sdpa_bwd_dq_general(FaSdpaDims d,
                                    const float *__restrict__ Q,
                                    const float *__restrict__ K,
                                    const float *__restrict__ V,
                                    const float *__restrict__ dO,
                                    const float *__restrict__ L_in,
                                    const float *__restrict__ D_in,
                                    float *__restrict__ dQ,
                                    int accumulate) {
    __shared__ float smem[8];
    const uint64_t total_rows = d.batch * d.num_heads * d.seq_q;
    const uint64_t gqa = d.num_heads / d.num_kv_heads;
    const uint64_t D = d.head_dim;

    for (uint64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
        const uint64_t i = row % d.seq_q;
        const uint64_t bh = row / d.seq_q;
        const uint64_t b = bh / d.num_heads;
        const uint64_t h = bh % d.num_heads;
        const uint64_t h_kv = h / gqa;
        const uint64_t k_lim = d.is_causal ? umin64(d.seq_k, i + 1) : d.seq_k;

        const int64_t qb = row_base(d.q, b, h, i);
        const int64_t dob = row_base(d.dy_or_o, b, h, i);
        const int64_t outb = row_base(d.out, b, h, i);
        const float lse = L_in[row];
        const float d_row = D_in[row];

        if (!accumulate) {
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                dQ[outb + static_cast<int64_t>(dim) * d.out.strides[3]] = 0.0f;
            }
        }
        for (uint64_t j = 0; j < k_lim; ++j) {
            const int64_t kb = row_base(d.k, b, h_kv, j);
            const int64_t vb = row_base(d.v, b, h_kv, j);
            float pqk = 0.0f, pdov = 0.0f;
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                pqk += Q[qb + static_cast<int64_t>(dim) * d.q.strides[3]] *
                       K[kb + static_cast<int64_t>(dim) * d.k.strides[3]];
                pdov += dO[dob + static_cast<int64_t>(dim) * d.dy_or_o.strides[3]] *
                        V[vb + static_cast<int64_t>(dim) * d.v.strides[3]];
            }
            const float p = expf(block_sum_sdpa(pqk, smem) * d.scale - lse);
            const float ds = p * (block_sum_sdpa(pdov, smem) - d_row) * d.scale;
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                dQ[outb + static_cast<int64_t>(dim) * d.out.strides[3]] +=
                    ds * K[kb + static_cast<int64_t>(dim) * d.k.strides[3]];
            }
        }
    }
}

template <bool is_dk>
__global__ void sdpa_bwd_kv_general(FaSdpaDims d,
                                    const float *__restrict__ Q,
                                    const float *__restrict__ K,
                                    const float *__restrict__ V,
                                    const float *__restrict__ dO,
                                    const float *__restrict__ L_in,
                                    const float *__restrict__ D_in,
                                    float *__restrict__ dKV,
                                    int accumulate) {
    __shared__ float smem[8];
    const uint64_t total_krows = d.batch * d.num_kv_heads * d.seq_k;
    const uint64_t gqa = d.num_heads / d.num_kv_heads;
    const uint64_t D = d.head_dim;

    for (uint64_t row = blockIdx.x; row < total_krows; row += gridDim.x) {
        const uint64_t j = row % d.seq_k;
        const uint64_t bkv = row / d.seq_k;
        const uint64_t b = bkv / d.num_kv_heads;
        const uint64_t h_kv = bkv % d.num_kv_heads;

        const int64_t kb = row_base(d.k, b, h_kv, j);
        const int64_t vb = is_dk ? row_base(d.v, b, h_kv, j) : 0;
        const int64_t outb = row_base(d.out, b, h_kv, j);

        if (!accumulate) {
            for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                dKV[outb + static_cast<int64_t>(dim) * d.out.strides[3]] = 0.0f;
            }
        }
        const uint64_t i_start = d.is_causal ? j : 0;
        for (uint64_t hg = 0; hg < gqa; ++hg) {
            const uint64_t h = h_kv * gqa + hg;
            const uint64_t bh = b * d.num_heads + h;
            for (uint64_t i = i_start; i < d.seq_q; ++i) {
                const int64_t qb = row_base(d.q, b, h, i);
                const int64_t dob = row_base(d.dy_or_o, b, h, i);
                float pqk = 0.0f, pdov = 0.0f;
                for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                    pqk += Q[qb + static_cast<int64_t>(dim) * d.q.strides[3]] *
                           K[kb + static_cast<int64_t>(dim) * d.k.strides[3]];
                    if constexpr (is_dk) {
                        pdov += dO[dob + static_cast<int64_t>(dim) * d.dy_or_o.strides[3]] *
                                V[vb + static_cast<int64_t>(dim) * d.v.strides[3]];
                    }
                }
                const float lse = L_in[bh * d.seq_q + i];
                const float p = expf(block_sum_sdpa(pqk, smem) * d.scale - lse);
                if constexpr (is_dk) {
                    const float d_row = D_in[bh * d.seq_q + i];
                    const float ds = p * (block_sum_sdpa(pdov, smem) - d_row) * d.scale;
                    for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                        dKV[outb + static_cast<int64_t>(dim) * d.out.strides[3]] +=
                            ds * Q[qb + static_cast<int64_t>(dim) * d.q.strides[3]];
                    }
                } else {
                    for (uint64_t dim = threadIdx.x; dim < D; dim += 256) {
                        dKV[outb + static_cast<int64_t>(dim) * d.out.strides[3]] +=
                            p * dO[dob + static_cast<int64_t>(dim) * d.dy_or_o.strides[3]];
                    }
                }
            }
        }
    }
}

unsigned sdpa_grid(uint64_t tiles) {
    return static_cast<unsigned>(std::min<uint64_t>(tiles, 65535ULL));
}

static int blas_sdpa(cublasStatus_t s) {
    return s == CUBLAS_STATUS_SUCCESS ? 0 : 2000 + static_cast<int>(s);
}

__device__ __forceinline__ float max_nan_sdpa(float a, float b) {
    if (isnan(a) || isnan(b)) return __uint_as_float(0x7fc00000u);
    return a > b ? a : b;
}

__device__ __forceinline__ float warp_max_sdpa(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = max_nan_sdpa(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float block_max_sdpa(float v, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_max_sdpa(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < 8) ? smem[lane] : -INFINITY;
        x = warp_max_sdpa(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    const float res = smem[0];
    __syncthreads();
    return res;
}

// Vectorized float4 warp-per-row in-place scale + causal mask + softmax on L2-resident P ([B*H*T_q, T_k]).
template <int NV4, bool exact, bool tile64_causal>
__global__ void sdpa_scale_mask_softmax_warp_vec4(float *__restrict__ P,
                                                  uint64_t total_rows,
                                                  uint64_t T_q,
                                                  uint64_t T_k,
                                                  float scale,
                                                  int is_causal) {
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint64_t T_k4 = T_k >> 2;
    float4 *P4 = reinterpret_cast<float4 *>(P);
    for (uint64_t row = warp_id; row < total_rows; row += total_warps) {
        const uint64_t q_idx = row % T_q;
        const uint64_t k_lim = is_causal ? umin64(T_k, q_idx + 1ULL) : T_k;
        const uint64_t k_tile_lim = (is_causal && tile64_causal)
            ? umin64(T_k, ((q_idx >> 6) + 1ULL) << 6)
            : T_k;
        float4 *pg4 = P4 + row * T_k4;
        float4 reg[NV4];
        float m = -INFINITY;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            const uint64_t j0 = idx4 << 2;
            if (j0 + 3ULL < k_lim) {
                float4 v = pg4[idx4];
                v.x *= scale;
                v.y *= scale;
                v.z *= scale;
                v.w *= scale;
                reg[k] = v;
                m = max_nan_sdpa(m, max_nan_sdpa(max_nan_sdpa(v.x, v.y), max_nan_sdpa(v.z, v.w)));
            } else if (j0 < k_lim) {
                const float4 v = pg4[idx4];
                const float x0 = v.x * scale;
                const float x1 = (j0 + 1ULL < k_lim) ? (v.y * scale) : -INFINITY;
                const float x2 = (j0 + 2ULL < k_lim) ? (v.z * scale) : -INFINITY;
                const float x3 = (j0 + 3ULL < k_lim) ? (v.w * scale) : -INFINITY;
                reg[k] = make_float4(x0, x1, x2, x3);
                m = max_nan_sdpa(m, x0);
                if (j0 + 1ULL < k_lim) m = max_nan_sdpa(m, x1);
                if (j0 + 2ULL < k_lim) m = max_nan_sdpa(m, x2);
                if (j0 + 3ULL < k_lim) m = max_nan_sdpa(m, x3);
            } else {
                reg[k] = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            }
        }
        m = warp_max_sdpa(m);
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            const uint64_t j0 = idx4 << 2;
            if (j0 + 3ULL < k_lim) {
                const float4 v = reg[k];
                const float e0 = expf(v.x - m);
                const float e1 = expf(v.y - m);
                const float e2 = expf(v.z - m);
                const float e3 = expf(v.w - m);
                reg[k] = make_float4(e0, e1, e2, e3);
                sum += (e0 + e1) + (e2 + e3);
            } else if (j0 < k_lim) {
                const float4 v = reg[k];
                const float e0 = expf(v.x - m);
                const float e1 = (j0 + 1ULL < k_lim) ? expf(v.y - m) : 0.0f;
                const float e2 = (j0 + 2ULL < k_lim) ? expf(v.z - m) : 0.0f;
                const float e3 = (j0 + 3ULL < k_lim) ? expf(v.w - m) : 0.0f;
                reg[k] = make_float4(e0, e1, e2, e3);
                sum += (e0 + e1) + (e2 + e3);
            } else {
                reg[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        const float inv_sum = 1.0f / warp_sum_sdpa(sum);
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            const uint64_t j0 = idx4 << 2;
            if ((exact || j0 < T_k) && (!tile64_causal || j0 < k_tile_lim)) {
                if (j0 + 3ULL < k_lim) {
                    const float4 e = reg[k];
                    pg4[idx4] = make_float4(e.x * inv_sum, e.y * inv_sum, e.z * inv_sum, e.w * inv_sum);
                } else if (j0 < k_lim) {
                    const float4 e = reg[k];
                    pg4[idx4] = make_float4(
                        e.x * inv_sum,
                        (j0 + 1ULL < k_lim) ? (e.y * inv_sum) : 0.0f,
                        (j0 + 2ULL < k_lim) ? (e.z * inv_sum) : 0.0f,
                        (j0 + 3ULL < k_lim) ? (e.w * inv_sum) : 0.0f);
                } else {
                    pg4[idx4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
    }
}

// Vectorized float4 warp-per-row in-place softmax backward: dS = scale * P * (dP - dot(P, dP)).
template <int NV4, bool exact, bool tile64_causal>
__global__ void sdpa_softmax_bwd_warp_vec4(const float *__restrict__ P,
                                           float *__restrict__ dS,
                                           uint64_t total_rows,
                                           uint64_t T_q,
                                           uint64_t T_k,
                                           float scale,
                                           int is_causal) {
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint64_t T_k4 = T_k >> 2;
    const float4 *P4 = reinterpret_cast<const float4 *>(P);
    float4 *dS4 = reinterpret_cast<float4 *>(dS);
    for (uint64_t row = warp_id; row < total_rows; row += total_warps) {
        const uint64_t q_idx = row % T_q;
        const uint64_t k_lim = is_causal ? umin64(T_k, q_idx + 1ULL) : T_k;
        const uint64_t k_tile_lim = (is_causal && tile64_causal)
            ? umin64(T_k, ((q_idx >> 6) + 1ULL) << 6)
            : T_k;
        const float4 *pg4 = P4 + row * T_k4;
        float4 *dsg4 = dS4 + row * T_k4;
        float4 rp[NV4];
        float4 rdp[NV4];
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            const uint64_t j0 = idx4 << 2;
            if (j0 + 3ULL < k_lim) {
                const float4 p = pg4[idx4];
                const float4 dp = dsg4[idx4];
                rp[k] = p;
                rdp[k] = dp;
                dot = fmaf(p.x, dp.x, fmaf(p.y, dp.y, fmaf(p.z, dp.z, fmaf(p.w, dp.w, dot))));
            } else if (j0 < k_lim) {
                const float4 p = pg4[idx4];
                const float4 dp = dsg4[idx4];
                const float p0 = p.x, dp0 = dp.x;
                const float p1 = (j0 + 1ULL < k_lim) ? p.y : 0.0f;
                const float dp1 = (j0 + 1ULL < k_lim) ? dp.y : 0.0f;
                const float p2 = (j0 + 2ULL < k_lim) ? p.z : 0.0f;
                const float dp2 = (j0 + 2ULL < k_lim) ? dp.z : 0.0f;
                const float p3 = (j0 + 3ULL < k_lim) ? p.w : 0.0f;
                const float dp3 = (j0 + 3ULL < k_lim) ? dp.w : 0.0f;
                rp[k] = make_float4(p0, p1, p2, p3);
                rdp[k] = make_float4(dp0, dp1, dp2, dp3);
                dot = fmaf(p0, dp0, fmaf(p1, dp1, fmaf(p2, dp2, fmaf(p3, dp3, dot))));
            } else {
                rp[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                rdp[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        dot = warp_sum_sdpa(dot);
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            const uint64_t j0 = idx4 << 2;
            if ((exact || j0 < T_k) && (!tile64_causal || j0 < k_tile_lim)) {
                if (j0 + 3ULL < k_lim) {
                    const float4 p = rp[k];
                    const float4 dp = rdp[k];
                    dsg4[idx4] = make_float4(
                        p.x * (dp.x - dot) * scale,
                        p.y * (dp.y - dot) * scale,
                        p.z * (dp.z - dot) * scale,
                        p.w * (dp.w - dot) * scale);
                } else if (j0 < k_lim) {
                    const float4 p = rp[k];
                    const float4 dp = rdp[k];
                    dsg4[idx4] = make_float4(
                        p.x * (dp.x - dot) * scale,
                        (j0 + 1ULL < k_lim) ? (p.y * (dp.y - dot) * scale) : 0.0f,
                        (j0 + 2ULL < k_lim) ? (p.z * (dp.z - dot) * scale) : 0.0f,
                        (j0 + 3ULL < k_lim) ? (p.w * (dp.w - dot) * scale) : 0.0f);
                } else {
                    dsg4[idx4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
    }
}

// =========================================================================
// Double-Buffered 64x64 Strict-FP32 SIMT SDPA GEMM Kernels (128 threads, 8x4 micro-tile)
// Eliminates cuBLAS 128x128 tile waste (D=64) and permuted/GQA serial launches in 1 launch!
// =========================================================================
template <bool causal>
__global__ void __launch_bounds__(128, 3) sdpa_sgemm_qk_64x64(
    const float *__restrict__ Q_base,
    FaSdpaView4 q_view,
    const float *__restrict__ K_base,
    FaSdpaView4 k_view,
    float *__restrict__ S_base,
    uint64_t H,
    uint64_t gqa,
    uint64_t T_q,
    uint64_t T_k,
    uint64_t D) {
    const uint64_t n_blk = blockIdx.x;
    const uint64_t m_blk = blockIdx.y;
    if constexpr (causal) {
        if (n_blk > m_blk) return;
    }
    const uint64_t bh = blockIdx.z;
    const uint64_t b = bh / H;
    const uint64_t h = bh % H;
    const uint64_t h_kv = h / gqa;
    const uint64_t m0 = m_blk << 6;
    const uint64_t n0 = n_blk << 6;

    const float *Q_bh = Q_base + q_view.offset + static_cast<int64_t>(b) * q_view.strides[0]
                        + static_cast<int64_t>(h) * q_view.strides[1];
    const float *K_bh = K_base + k_view.offset + static_cast<int64_t>(b) * k_view.strides[0]
                        + static_cast<int64_t>(h_kv) * k_view.strides[1];
    float *S_bh = S_base + bh * T_q * T_k;
    const int64_t ldq = q_view.strides[2];
    const int64_t ldk = k_view.strides[2];

    __shared__ float sA[2][16][68];
    __shared__ float sB[2][16][68];

    const int tid = threadIdx.x;
    const int row_l = tid >> 2;        // 0..31
    const int col4_l = (tid & 3) << 2; // 0, 4, 8, 12
    const int ty = tid >> 4;           // 0..7
    const int tx = tid & 15;           // 0..15

    float acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    const uint64_t num_tiles = D >> 4;
    // Preload tile 0 into buffer 0
    {
        const float4 qa0 = *reinterpret_cast<const float4 *>(Q_bh + static_cast<int64_t>(m0 + row_l) * ldq + col4_l);
        const float4 qa1 = *reinterpret_cast<const float4 *>(Q_bh + static_cast<int64_t>(m0 + row_l + 32) * ldq + col4_l);
        const float4 kb0 = *reinterpret_cast<const float4 *>(K_bh + static_cast<int64_t>(n0 + row_l) * ldk + col4_l);
        const float4 kb1 = *reinterpret_cast<const float4 *>(K_bh + static_cast<int64_t>(n0 + row_l + 32) * ldk + col4_l);
        sA[0][col4_l + 0][row_l] = qa0.x;
        sA[0][col4_l + 1][row_l] = qa0.y;
        sA[0][col4_l + 2][row_l] = qa0.z;
        sA[0][col4_l + 3][row_l] = qa0.w;
        sA[0][col4_l + 0][row_l + 32] = qa1.x;
        sA[0][col4_l + 1][row_l + 32] = qa1.y;
        sA[0][col4_l + 2][row_l + 32] = qa1.z;
        sA[0][col4_l + 3][row_l + 32] = qa1.w;
        sB[0][col4_l + 0][row_l] = kb0.x;
        sB[0][col4_l + 1][row_l] = kb0.y;
        sB[0][col4_l + 2][row_l] = kb0.z;
        sB[0][col4_l + 3][row_l] = kb0.w;
        sB[0][col4_l + 0][row_l + 32] = kb1.x;
        sB[0][col4_l + 1][row_l + 32] = kb1.y;
        sB[0][col4_l + 2][row_l + 32] = kb1.z;
        sB[0][col4_l + 3][row_l + 32] = kb1.w;
    }
    __syncthreads();

    for (uint64_t t = 0; t < num_tiles; ++t) {
        const int cur = static_cast<int>(t & 1ULL);
        const int nxt = cur ^ 1;
        float4 qa0, qa1, kb0, kb1;
        if (t + 1ULL < num_tiles) {
            const uint64_t k_next = ((t + 1ULL) << 4) + static_cast<uint64_t>(col4_l);
            qa0 = *reinterpret_cast<const float4 *>(Q_bh + static_cast<int64_t>(m0 + row_l) * ldq + k_next);
            qa1 = *reinterpret_cast<const float4 *>(Q_bh + static_cast<int64_t>(m0 + row_l + 32) * ldq + k_next);
            kb0 = *reinterpret_cast<const float4 *>(K_bh + static_cast<int64_t>(n0 + row_l) * ldk + k_next);
            kb1 = *reinterpret_cast<const float4 *>(K_bh + static_cast<int64_t>(n0 + row_l + 32) * ldk + k_next);
        }
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float4 a0 = *reinterpret_cast<const float4 *>(&sA[cur][k][ty * 8]);
            const float4 a1 = *reinterpret_cast<const float4 *>(&sA[cur][k][ty * 8 + 4]);
            const float4 b0 = *reinterpret_cast<const float4 *>(&sB[cur][k][tx * 4]);
            acc[0][0] = fmaf(a0.x, b0.x, acc[0][0]); acc[0][1] = fmaf(a0.x, b0.y, acc[0][1]);
            acc[0][2] = fmaf(a0.x, b0.z, acc[0][2]); acc[0][3] = fmaf(a0.x, b0.w, acc[0][3]);
            acc[1][0] = fmaf(a0.y, b0.x, acc[1][0]); acc[1][1] = fmaf(a0.y, b0.y, acc[1][1]);
            acc[1][2] = fmaf(a0.y, b0.z, acc[1][2]); acc[1][3] = fmaf(a0.y, b0.w, acc[1][3]);
            acc[2][0] = fmaf(a0.z, b0.x, acc[2][0]); acc[2][1] = fmaf(a0.z, b0.y, acc[2][1]);
            acc[2][2] = fmaf(a0.z, b0.z, acc[2][2]); acc[2][3] = fmaf(a0.z, b0.w, acc[2][3]);
            acc[3][0] = fmaf(a0.w, b0.x, acc[3][0]); acc[3][1] = fmaf(a0.w, b0.y, acc[3][1]);
            acc[3][2] = fmaf(a0.w, b0.z, acc[3][2]); acc[3][3] = fmaf(a0.w, b0.w, acc[3][3]);
            acc[4][0] = fmaf(a1.x, b0.x, acc[4][0]); acc[4][1] = fmaf(a1.x, b0.y, acc[4][1]);
            acc[4][2] = fmaf(a1.x, b0.z, acc[4][2]); acc[4][3] = fmaf(a1.x, b0.w, acc[4][3]);
            acc[5][0] = fmaf(a1.y, b0.x, acc[5][0]); acc[5][1] = fmaf(a1.y, b0.y, acc[5][1]);
            acc[5][2] = fmaf(a1.y, b0.z, acc[5][2]); acc[5][3] = fmaf(a1.y, b0.w, acc[5][3]);
            acc[6][0] = fmaf(a1.z, b0.x, acc[6][0]); acc[6][1] = fmaf(a1.z, b0.y, acc[6][1]);
            acc[6][2] = fmaf(a1.z, b0.z, acc[6][2]); acc[6][3] = fmaf(a1.z, b0.w, acc[6][3]);
            acc[7][0] = fmaf(a1.w, b0.x, acc[7][0]); acc[7][1] = fmaf(a1.w, b0.y, acc[7][1]);
            acc[7][2] = fmaf(a1.w, b0.z, acc[7][2]); acc[7][3] = fmaf(a1.w, b0.w, acc[7][3]);
        }
        if (t + 1ULL < num_tiles) {
            sA[nxt][col4_l + 0][row_l] = qa0.x;
            sA[nxt][col4_l + 1][row_l] = qa0.y;
            sA[nxt][col4_l + 2][row_l] = qa0.z;
            sA[nxt][col4_l + 3][row_l] = qa0.w;
            sA[nxt][col4_l + 0][row_l + 32] = qa1.x;
            sA[nxt][col4_l + 1][row_l + 32] = qa1.y;
            sA[nxt][col4_l + 2][row_l + 32] = qa1.z;
            sA[nxt][col4_l + 3][row_l + 32] = qa1.w;
            sB[nxt][col4_l + 0][row_l] = kb0.x;
            sB[nxt][col4_l + 1][row_l] = kb0.y;
            sB[nxt][col4_l + 2][row_l] = kb0.z;
            sB[nxt][col4_l + 3][row_l] = kb0.w;
            sB[nxt][col4_l + 0][row_l + 32] = kb1.x;
            sB[nxt][col4_l + 1][row_l + 32] = kb1.y;
            sB[nxt][col4_l + 2][row_l + 32] = kb1.z;
            sB[nxt][col4_l + 3][row_l + 32] = kb1.w;
            __syncthreads();
        }
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        *reinterpret_cast<float4 *>(S_bh + (m0 + ty * 8 + r) * T_k + n0 + tx * 4) =
            make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    }
}

template <bool causal, bool accumulate>
__global__ void __launch_bounds__(128, 3) sdpa_sgemm_pv_64x64(
    const float *__restrict__ P_base,
    const float *__restrict__ V_base,
    FaSdpaView4 v_view,
    float *__restrict__ O_base,
    FaSdpaView4 o_view,
    uint64_t H,
    uint64_t gqa,
    uint64_t T_q,
    uint64_t T_k) {
    const uint64_t n0 = static_cast<uint64_t>(blockIdx.x) << 6;
    const uint64_t m_blk = causal ? (static_cast<uint64_t>(gridDim.y) - 1ULL - blockIdx.y) : blockIdx.y;
    const uint64_t m0 = m_blk << 6;
    const uint64_t bh = blockIdx.z;
    const uint64_t b = bh / H;
    const uint64_t h = bh % H;
    const uint64_t h_kv = h / gqa;
    const uint64_t k_limit = causal ? umin64(T_k, m0 + 64ULL) : T_k;
    const uint64_t num_tiles = k_limit >> 4;

    const float *P_bh = P_base + bh * T_q * T_k;
    const float *V_bh = V_base + v_view.offset + static_cast<int64_t>(b) * v_view.strides[0]
                        + static_cast<int64_t>(h_kv) * v_view.strides[1];
    float *O_bh = O_base + o_view.offset + static_cast<int64_t>(b) * o_view.strides[0]
                  + static_cast<int64_t>(h) * o_view.strides[1];
    const int64_t ldv = v_view.strides[2];
    const int64_t ldo = o_view.strides[2];

    __shared__ float sA[2][16][68];
    __shared__ float sB[2][16][68];

    const int tid = threadIdx.x;
    const int p_row = tid >> 2;        // 0..31
    const int p_col4 = (tid & 3) << 2; // 0, 4, 8, 12
    const int v_row = tid >> 4;        // 0..7
    const int v_col4 = (tid & 15) << 2;
    const int ty = tid >> 4;
    const int tx = tid & 15;

    float acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    if (num_tiles > 0) {
        {
            const float4 pa0 = *reinterpret_cast<const float4 *>(P_bh + (m0 + p_row) * T_k + p_col4);
            const float4 pa1 = *reinterpret_cast<const float4 *>(P_bh + (m0 + p_row + 32) * T_k + p_col4);
            const float4 vb0 = *reinterpret_cast<const float4 *>(V_bh + static_cast<int64_t>(v_row) * ldv + n0 + v_col4);
            const float4 vb1 = *reinterpret_cast<const float4 *>(V_bh + static_cast<int64_t>(v_row + 8) * ldv + n0 + v_col4);
            sA[0][p_col4 + 0][p_row] = pa0.x;
            sA[0][p_col4 + 1][p_row] = pa0.y;
            sA[0][p_col4 + 2][p_row] = pa0.z;
            sA[0][p_col4 + 3][p_row] = pa0.w;
            sA[0][p_col4 + 0][p_row + 32] = pa1.x;
            sA[0][p_col4 + 1][p_row + 32] = pa1.y;
            sA[0][p_col4 + 2][p_row + 32] = pa1.z;
            sA[0][p_col4 + 3][p_row + 32] = pa1.w;
            *reinterpret_cast<float4 *>(&sB[0][v_row][v_col4]) = vb0;
            *reinterpret_cast<float4 *>(&sB[0][v_row + 8][v_col4]) = vb1;
        }
        __syncthreads();

        for (uint64_t t = 0; t < num_tiles; ++t) {
            const int cur = static_cast<int>(t & 1ULL);
            const int nxt = cur ^ 1;
            float4 pa0, pa1, vb0, vb1;
            if (t + 1ULL < num_tiles) {
                const uint64_t k_next = (t + 1ULL) << 4;
                pa0 = *reinterpret_cast<const float4 *>(P_bh + (m0 + p_row) * T_k + k_next + p_col4);
                pa1 = *reinterpret_cast<const float4 *>(P_bh + (m0 + p_row + 32) * T_k + k_next + p_col4);
                vb0 = *reinterpret_cast<const float4 *>(V_bh + static_cast<int64_t>(k_next + v_row) * ldv + n0 + v_col4);
                vb1 = *reinterpret_cast<const float4 *>(V_bh + static_cast<int64_t>(k_next + v_row + 8) * ldv + n0 + v_col4);
            }
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                const float4 a0 = *reinterpret_cast<const float4 *>(&sA[cur][k][ty * 8]);
                const float4 a1 = *reinterpret_cast<const float4 *>(&sA[cur][k][ty * 8 + 4]);
                const float4 b0 = *reinterpret_cast<const float4 *>(&sB[cur][k][tx * 4]);
                acc[0][0] = fmaf(a0.x, b0.x, acc[0][0]); acc[0][1] = fmaf(a0.x, b0.y, acc[0][1]);
                acc[0][2] = fmaf(a0.x, b0.z, acc[0][2]); acc[0][3] = fmaf(a0.x, b0.w, acc[0][3]);
                acc[1][0] = fmaf(a0.y, b0.x, acc[1][0]); acc[1][1] = fmaf(a0.y, b0.y, acc[1][1]);
                acc[1][2] = fmaf(a0.y, b0.z, acc[1][2]); acc[1][3] = fmaf(a0.y, b0.w, acc[1][3]);
                acc[2][0] = fmaf(a0.z, b0.x, acc[2][0]); acc[2][1] = fmaf(a0.z, b0.y, acc[2][1]);
                acc[2][2] = fmaf(a0.z, b0.z, acc[2][2]); acc[2][3] = fmaf(a0.z, b0.w, acc[2][3]);
                acc[3][0] = fmaf(a0.w, b0.x, acc[3][0]); acc[3][1] = fmaf(a0.w, b0.y, acc[3][1]);
                acc[3][2] = fmaf(a0.w, b0.z, acc[3][2]); acc[3][3] = fmaf(a0.w, b0.w, acc[3][3]);
                acc[4][0] = fmaf(a1.x, b0.x, acc[4][0]); acc[4][1] = fmaf(a1.x, b0.y, acc[4][1]);
                acc[4][2] = fmaf(a1.x, b0.z, acc[4][2]); acc[4][3] = fmaf(a1.x, b0.w, acc[4][3]);
                acc[5][0] = fmaf(a1.y, b0.x, acc[5][0]); acc[5][1] = fmaf(a1.y, b0.y, acc[5][1]);
                acc[5][2] = fmaf(a1.y, b0.z, acc[5][2]); acc[5][3] = fmaf(a1.y, b0.w, acc[5][3]);
                acc[6][0] = fmaf(a1.z, b0.x, acc[6][0]); acc[6][1] = fmaf(a1.z, b0.y, acc[6][1]);
                acc[6][2] = fmaf(a1.z, b0.z, acc[6][2]); acc[6][3] = fmaf(a1.z, b0.w, acc[6][3]);
                acc[7][0] = fmaf(a1.w, b0.x, acc[7][0]); acc[7][1] = fmaf(a1.w, b0.y, acc[7][1]);
                acc[7][2] = fmaf(a1.w, b0.z, acc[7][2]); acc[7][3] = fmaf(a1.w, b0.w, acc[7][3]);
            }
            if (t + 1ULL < num_tiles) {
                sA[nxt][p_col4 + 0][p_row] = pa0.x;
                sA[nxt][p_col4 + 1][p_row] = pa0.y;
                sA[nxt][p_col4 + 2][p_row] = pa0.z;
                sA[nxt][p_col4 + 3][p_row] = pa0.w;
                sA[nxt][p_col4 + 0][p_row + 32] = pa1.x;
                sA[nxt][p_col4 + 1][p_row + 32] = pa1.y;
                sA[nxt][p_col4 + 2][p_row + 32] = pa1.z;
                sA[nxt][p_col4 + 3][p_row + 32] = pa1.w;
                *reinterpret_cast<float4 *>(&sB[nxt][v_row][v_col4]) = vb0;
                *reinterpret_cast<float4 *>(&sB[nxt][v_row + 8][v_col4]) = vb1;
                __syncthreads();
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        float4 *dst = reinterpret_cast<float4 *>(
            O_bh + static_cast<int64_t>(m0 + ty * 8 + r) * ldo + n0 + tx * 4);
        if constexpr (accumulate) {
            const float4 prev = *dst;
            *dst = make_float4(prev.x + acc[r][0], prev.y + acc[r][1], prev.z + acc[r][2], prev.w + acc[r][3]);
        } else {
            *dst = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
        }
    }
}

template <bool causal, bool accumulate>
__global__ void __launch_bounds__(128, 3) sdpa_sgemm_ptdo_64x64(
    const float *__restrict__ P_base,
    const float *__restrict__ X_base,
    FaSdpaView4 x_view,
    float *__restrict__ Out_base,
    FaSdpaView4 out_view,
    uint64_t H,
    uint64_t T_q,
    uint64_t T_k) {
    const uint64_t n0 = static_cast<uint64_t>(blockIdx.x) << 6;
    const uint64_t m0 = static_cast<uint64_t>(blockIdx.y) << 6;
    const uint64_t bh = blockIdx.z;
    const uint64_t b = bh / H;
    const uint64_t h = bh % H;
    const uint64_t k_start = causal ? umin64(T_q, m0) : 0ULL;
    const uint64_t num_tiles = (T_q - k_start) >> 4;

    const float *P_bh = P_base + bh * T_q * T_k;
    const float *X_bh = X_base + x_view.offset + static_cast<int64_t>(b) * x_view.strides[0]
                        + static_cast<int64_t>(h) * x_view.strides[1];
    float *Out_bh = Out_base + out_view.offset + static_cast<int64_t>(b) * out_view.strides[0]
                    + static_cast<int64_t>(h) * out_view.strides[1];
    const int64_t ldx = x_view.strides[2];
    const int64_t ldout = out_view.strides[2];

    __shared__ float sA[2][16][68];
    __shared__ float sB[2][16][68];

    const int tid = threadIdx.x;
    const int k_row = tid >> 4;        // 0..7
    const int c4_loc = (tid & 15) << 2;
    const int ty = tid >> 4;
    const int tx = tid & 15;

    float acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    if (num_tiles > 0) {
        {
            const uint64_t k0 = k_start;
            const float4 pa0 = *reinterpret_cast<const float4 *>(P_bh + (k0 + k_row) * T_k + m0 + c4_loc);
            const float4 pa1 = *reinterpret_cast<const float4 *>(P_bh + (k0 + k_row + 8) * T_k + m0 + c4_loc);
            const float4 xb0 = *reinterpret_cast<const float4 *>(X_bh + static_cast<int64_t>(k0 + k_row) * ldx + n0 + c4_loc);
            const float4 xb1 = *reinterpret_cast<const float4 *>(X_bh + static_cast<int64_t>(k0 + k_row + 8) * ldx + n0 + c4_loc);
            *reinterpret_cast<float4 *>(&sA[0][k_row][c4_loc]) = pa0;
            *reinterpret_cast<float4 *>(&sA[0][k_row + 8][c4_loc]) = pa1;
            *reinterpret_cast<float4 *>(&sB[0][k_row][c4_loc]) = xb0;
            *reinterpret_cast<float4 *>(&sB[0][k_row + 8][c4_loc]) = xb1;
        }
        __syncthreads();

        for (uint64_t t = 0; t < num_tiles; ++t) {
            const int cur = static_cast<int>(t & 1ULL);
            const int nxt = cur ^ 1;
            float4 pa0, pa1, xb0, xb1;
            if (t + 1ULL < num_tiles) {
                const uint64_t k_next = k_start + ((t + 1ULL) << 4);
                pa0 = *reinterpret_cast<const float4 *>(P_bh + (k_next + k_row) * T_k + m0 + c4_loc);
                pa1 = *reinterpret_cast<const float4 *>(P_bh + (k_next + k_row + 8) * T_k + m0 + c4_loc);
                xb0 = *reinterpret_cast<const float4 *>(X_bh + static_cast<int64_t>(k_next + k_row) * ldx + n0 + c4_loc);
                xb1 = *reinterpret_cast<const float4 *>(X_bh + static_cast<int64_t>(k_next + k_row + 8) * ldx + n0 + c4_loc);
            }
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                const float4 a0 = *reinterpret_cast<const float4 *>(&sA[cur][k][ty * 8]);
                const float4 a1 = *reinterpret_cast<const float4 *>(&sA[cur][k][ty * 8 + 4]);
                const float4 b0 = *reinterpret_cast<const float4 *>(&sB[cur][k][tx * 4]);
                acc[0][0] = fmaf(a0.x, b0.x, acc[0][0]); acc[0][1] = fmaf(a0.x, b0.y, acc[0][1]);
                acc[0][2] = fmaf(a0.x, b0.z, acc[0][2]); acc[0][3] = fmaf(a0.x, b0.w, acc[0][3]);
                acc[1][0] = fmaf(a0.y, b0.x, acc[1][0]); acc[1][1] = fmaf(a0.y, b0.y, acc[1][1]);
                acc[1][2] = fmaf(a0.y, b0.z, acc[1][2]); acc[1][3] = fmaf(a0.y, b0.w, acc[1][3]);
                acc[2][0] = fmaf(a0.z, b0.x, acc[2][0]); acc[2][1] = fmaf(a0.z, b0.y, acc[2][1]);
                acc[2][2] = fmaf(a0.z, b0.z, acc[2][2]); acc[2][3] = fmaf(a0.z, b0.w, acc[2][3]);
                acc[3][0] = fmaf(a0.w, b0.x, acc[3][0]); acc[3][1] = fmaf(a0.w, b0.y, acc[3][1]);
                acc[3][2] = fmaf(a0.w, b0.z, acc[3][2]); acc[3][3] = fmaf(a0.w, b0.w, acc[3][3]);
                acc[4][0] = fmaf(a1.x, b0.x, acc[4][0]); acc[4][1] = fmaf(a1.x, b0.y, acc[4][1]);
                acc[4][2] = fmaf(a1.x, b0.z, acc[4][2]); acc[4][3] = fmaf(a1.x, b0.w, acc[4][3]);
                acc[5][0] = fmaf(a1.y, b0.x, acc[5][0]); acc[5][1] = fmaf(a1.y, b0.y, acc[5][1]);
                acc[5][2] = fmaf(a1.y, b0.z, acc[5][2]); acc[5][3] = fmaf(a1.y, b0.w, acc[5][3]);
                acc[6][0] = fmaf(a1.z, b0.x, acc[6][0]); acc[6][1] = fmaf(a1.z, b0.y, acc[6][1]);
                acc[6][2] = fmaf(a1.z, b0.z, acc[6][2]); acc[6][3] = fmaf(a1.z, b0.w, acc[6][3]);
                acc[7][0] = fmaf(a1.w, b0.x, acc[7][0]); acc[7][1] = fmaf(a1.w, b0.y, acc[7][1]);
                acc[7][2] = fmaf(a1.w, b0.z, acc[7][2]); acc[7][3] = fmaf(a1.w, b0.w, acc[7][3]);
            }
            if (t + 1ULL < num_tiles) {
                *reinterpret_cast<float4 *>(&sA[nxt][k_row][c4_loc]) = pa0;
                *reinterpret_cast<float4 *>(&sA[nxt][k_row + 8][c4_loc]) = pa1;
                *reinterpret_cast<float4 *>(&sB[nxt][k_row][c4_loc]) = xb0;
                *reinterpret_cast<float4 *>(&sB[nxt][k_row + 8][c4_loc]) = xb1;
                __syncthreads();
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        float4 *dst = reinterpret_cast<float4 *>(
            Out_bh + static_cast<int64_t>(m0 + ty * 8 + r) * ldout + n0 + tx * 4);
        if constexpr (accumulate) {
            const float4 prev = *dst;
            *dst = make_float4(prev.x + acc[r][0], prev.y + acc[r][1], prev.z + acc[r][2], prev.w + acc[r][3]);
        } else {
            *dst = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
        }
    }
}

template <bool accumulate>
__global__ void sdpa_gqa_reduce_vec4(const float *__restrict__ scratch,
                                     float *__restrict__ out,
                                     FaSdpaView4 out_view,
                                     uint64_t B,
                                     uint64_t H,
                                     uint64_t H_kv,
                                     uint64_t T_k,
                                     uint64_t D) {
    const uint64_t D4 = D >> 2;
    const uint64_t gqa = H / H_kv;
    const uint64_t total = B * H_kv * T_k * D4;
    const float4 *s4 = reinterpret_cast<const float4 *>(scratch);
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {
        const uint64_t d4 = idx % D4;
        const uint64_t rem0 = idx / D4;
        const uint64_t t = rem0 % T_k;
        const uint64_t bkv = rem0 / T_k;
        const uint64_t h_kv = bkv % H_kv;
        const uint64_t b = bkv / H_kv;
        const uint64_t bh_base = b * H + h_kv * gqa;
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint64_t hg = 0; hg < gqa; ++hg) {
            const float4 v = s4[((bh_base + hg) * T_k + t) * D4 + d4];
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        float4 *dst = reinterpret_cast<float4 *>(out + row_base(out_view, b, h_kv, t) + (d4 << 2));
        if constexpr (accumulate) {
            const float4 prev = *dst;
            *dst = make_float4(prev.x + sum.x, prev.y + sum.y, prev.z + sum.z, prev.w + sum.w);
        } else {
            *dst = sum;
        }
    }
}

// Warp-per-row in-place scale + causal mask + softmax on L2-resident P ([B*H*T_q, T_k], T_k <= 32 * MAX_K).
template <int MAX_K>
__global__ void sdpa_scale_mask_softmax_warp(float *__restrict__ P,
                                             uint64_t total_rows,
                                             uint64_t T_q,
                                             uint64_t T_k,
                                             float scale,
                                             int is_causal) {
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    for (uint64_t row = warp_id; row < total_rows; row += total_warps) {
        const uint64_t q_idx = row % T_q;
        const uint64_t k_lim = is_causal ? umin64(T_k, q_idx + 1ULL) : T_k;
        float *pg = P + row * T_k;
        float reg[MAX_K];
        float m = -INFINITY;
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            const uint64_t j = static_cast<uint64_t>(lane + (k << 5));
            if (j < k_lim) {
                const float v = pg[j] * scale;
                reg[k] = v;
                m = max_nan_sdpa(m, v);
            } else {
                reg[k] = -INFINITY;
            }
        }
        m = warp_max_sdpa(m);
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            const uint64_t j = static_cast<uint64_t>(lane + (k << 5));
            if (j < k_lim) {
                const float e = expf(reg[k] - m);
                reg[k] = e;
                sum += e;
            } else {
                reg[k] = 0.0f;
            }
        }
        const float inv_sum = 1.0f / warp_sum_sdpa(sum);
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            const uint64_t j = static_cast<uint64_t>(lane + (k << 5));
            if (j < T_k) {
                pg[j] = (j < k_lim) ? (reg[k] * inv_sum) : 0.0f;
            }
        }
    }
}

// Block-per-row in-place scale + causal mask + softmax for 512 < T_k <= 1024.
__global__ void sdpa_scale_mask_softmax_block(float *__restrict__ P,
                                              uint64_t total_rows,
                                              uint64_t T_q,
                                              uint64_t T_k,
                                              float scale,
                                              int is_causal) {
    __shared__ float smem[8];
    for (uint64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
        const uint64_t q_idx = row % T_q;
        const uint64_t k_lim = is_causal ? umin64(T_k, q_idx + 1ULL) : T_k;
        float *pg = P + row * T_k;
        float reg[4];
        float m = -INFINITY;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t j = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (j < k_lim) {
                const float v = pg[j] * scale;
                reg[k] = v;
                m = max_nan_sdpa(m, v);
            } else {
                reg[k] = -INFINITY;
            }
        }
        m = block_max_sdpa(m, smem);
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t j = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (j < k_lim) {
                const float e = expf(reg[k] - m);
                reg[k] = e;
                sum += e;
            } else {
                reg[k] = 0.0f;
            }
        }
        const float inv_sum = 1.0f / block_sum_sdpa(sum, smem);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t j = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (j < T_k) {
                pg[j] = (j < k_lim) ? (reg[k] * inv_sum) : 0.0f;
            }
        }
    }
}

// Warp-per-row in-place softmax backward: dS[i,j] = scale * P[i,j] * (dP[i,j] - dot(P_i, dP_i)).
template <int MAX_K>
__global__ void sdpa_softmax_bwd_warp(const float *__restrict__ P,
                                      float *__restrict__ dS,
                                      uint64_t total_rows,
                                      uint64_t T_q,
                                      uint64_t T_k,
                                      float scale,
                                      int is_causal) {
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    for (uint64_t row = warp_id; row < total_rows; row += total_warps) {
        const uint64_t q_idx = row % T_q;
        const uint64_t k_lim = is_causal ? umin64(T_k, q_idx + 1ULL) : T_k;
        const float *pg = P + row * T_k;
        float *dsg = dS + row * T_k;
        float rp[MAX_K];
        float rdp[MAX_K];
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            const uint64_t j = static_cast<uint64_t>(lane + (k << 5));
            if (j < k_lim) {
                const float p = pg[j];
                const float dp = dsg[j];
                rp[k] = p;
                rdp[k] = dp;
                dot = fmaf(p, dp, dot);
            } else {
                rp[k] = 0.0f;
                rdp[k] = 0.0f;
            }
        }
        dot = warp_sum_sdpa(dot);
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            const uint64_t j = static_cast<uint64_t>(lane + (k << 5));
            if (j < T_k) {
                dsg[j] = (j < k_lim) ? (rp[k] * (rdp[k] - dot) * scale) : 0.0f;
            }
        }
    }
}

__global__ void sdpa_softmax_bwd_block(const float *__restrict__ P,
                                       float *__restrict__ dS,
                                       uint64_t total_rows,
                                       uint64_t T_q,
                                       uint64_t T_k,
                                       float scale,
                                       int is_causal) {
    __shared__ float smem[8];
    for (uint64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
        const uint64_t q_idx = row % T_q;
        const uint64_t k_lim = is_causal ? umin64(T_k, q_idx + 1ULL) : T_k;
        const float *pg = P + row * T_k;
        float *dsg = dS + row * T_k;
        float rp[4];
        float rdp[4];
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t j = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (j < k_lim) {
                const float p = pg[j];
                const float dp = dsg[j];
                rp[k] = p;
                rdp[k] = dp;
                dot = fmaf(p, dp, dot);
            } else {
                rp[k] = 0.0f;
                rdp[k] = 0.0f;
            }
        }
        dot = block_sum_sdpa(dot, smem);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t j = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (j < T_k) {
                dsg[j] = (j < k_lim) ? (rp[k] * (rdp[k] - dot) * scale) : 0.0f;
            }
        }
    }
}

bool view_blas_ok(const FaSdpaView4 &v, uint64_t b, uint64_t h, uint64_t d) {
    if (v.offset < 0 || v.strides[3] != 1) return false;
    if (v.strides[2] < static_cast<int64_t>(d)) return false;
    if (h > 1 && v.strides[1] <= 0) return false;
    if (b > 1 && v.strides[0] <= 0) return false;
    return true;
}

bool can_use_cublas_sdpa(const FaSdpaDims &d) {
    if (!d.batch || !d.num_heads || !d.num_kv_heads || !d.seq_q || !d.seq_k || !d.head_dim) {
        return false;
    }
    if (d.seq_k > 1024ULL) return false;
    const uint64_t p_elems = d.batch * d.num_heads * d.seq_q * d.seq_k;
    if (p_elems > 8388608ULL) return false;
    if (!view_blas_ok(d.q, d.batch, d.num_heads, d.head_dim)) return false;
    if (!view_blas_ok(d.k, d.batch, d.num_kv_heads, d.head_dim)) return false;
    if (d.operand != 2 && !view_blas_ok(d.v, d.batch, d.num_kv_heads, d.head_dim)) return false;
    if (!view_blas_ok(d.dy_or_o, d.batch, d.num_heads, d.head_dim)) return false;
    if (d.operand >= 0) {
        const uint64_t oh = (d.operand == 0) ? d.num_heads : d.num_kv_heads;
        if (!view_blas_ok(d.out, d.batch, oh, d.head_dim)) return false;
    }
    return true;
}

bool sdpa_gemm64_shape_ok(const FaSdpaDims &d) {
    return d.batch > 0 && d.num_heads > 0 && d.num_kv_heads > 0 &&
           d.seq_q > 0 && d.seq_k > 0 && d.head_dim > 0 &&
           (d.seq_q & 63ULL) == 0 && (d.seq_k & 63ULL) == 0 && (d.head_dim & 63ULL) == 0;
}

bool sdpa_gemm64_view_ok(const FaSdpaView4 &v, uint64_t b, uint64_t h, const float *base) {
    if (!base || v.offset < 0 || v.strides[3] != 1) return false;
    if ((v.offset & 3LL) != 0 || (v.strides[2] & 3LL) != 0) return false;
    if (h > 1 && (v.strides[1] & 3LL) != 0) return false;
    if (b > 1 && (v.strides[0] & 3LL) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(base) & 15ULL) != 0) return false;
    return true;
}

// Launch one family of SDPA strided-batched GEMMs across all batches, KV heads, and GQA groups.
// Mode 0: S = Q @ K^T       (P_cm[T_k, T_q]  = K_cm^T[T_k, D] @ Q_cm[D, T_q])
// Mode 1: O = P @ V         (O_cm[D, T_q]    = V_cm[D, T_k]   @ P_cm[T_k, T_q])
// Mode 2: dP = dO @ V^T     (dS_cm[T_k, T_q] = V_cm^T[T_k, D] @ dO_cm[D, T_q])
// Mode 3: dQ = dS @ K       (dQ_cm[D, T_q]   = K_cm[D, T_k]   @ dS_cm[T_k, T_q])
// Mode 4: dK = dS^T @ Q     (dK_cm[D, T_k]   = Q_cm[D, T_q]   @ dS_cm^T[T_q, T_k])
// Mode 5: dV = P^T @ dO     (dV_cm[D, T_k]   = dO_cm[D, T_q]  @ P_cm^T[T_q, T_k])
int sdpa_launch_gemm(Context *ctx,
                     const FaSdpaDims &d,
                     int mode,
                     const float *q_base,
                     const FaSdpaView4 &q_view,
                     const float *kv_base,
                     const FaSdpaView4 &kv_view,
                     float *p_base,
                     float *out_base,
                     const FaSdpaView4 &out_view,
                     float init_beta,
                     bool use_tile64_causal) {
    const uint64_t B = d.batch;
    const uint64_t H = d.num_heads;
    const uint64_t H_kv = d.num_kv_heads;
    const uint64_t gqa = H / H_kv;
    const bool p_aligned = (reinterpret_cast<uintptr_t>(p_base) & 15ULL) == 0;
    const bool acc = (init_beta != 0.0f);

    // Fast path: custom double-buffered 64x64 strict-FP32 SIMT GEMM (zero tile waste, 1-launch permuted/GQA, causal skip).
    if (sdpa_gemm64_shape_ok(d) && p_aligned) {
        if ((mode == 0 || mode == 2) &&
            sdpa_gemm64_view_ok(q_view, B, H, q_base) &&
            sdpa_gemm64_view_ok(kv_view, B, H_kv, kv_base)) {
            const dim3 grid(static_cast<unsigned>(d.seq_k >> 6),
                            static_cast<unsigned>(d.seq_q >> 6),
                            static_cast<unsigned>(B * H));
            if (use_tile64_causal) {
                sdpa_sgemm_qk_64x64<true><<<grid, 128, 0, ctx->stream>>>(
                    q_base, q_view, kv_base, kv_view, p_base, H, gqa, d.seq_q, d.seq_k, d.head_dim);
            } else {
                sdpa_sgemm_qk_64x64<false><<<grid, 128, 0, ctx->stream>>>(
                    q_base, q_view, kv_base, kv_view, p_base, H, gqa, d.seq_q, d.seq_k, d.head_dim);
            }
            return rt(cudaGetLastError());
        }
        if ((mode == 1 || mode == 3) &&
            sdpa_gemm64_view_ok(kv_view, B, H_kv, kv_base) &&
            sdpa_gemm64_view_ok(out_view, B, H, out_base)) {
            const dim3 grid(static_cast<unsigned>(d.head_dim >> 6),
                            static_cast<unsigned>(d.seq_q >> 6),
                            static_cast<unsigned>(B * H));
            if (d.is_causal) {
                if (acc) {
                    sdpa_sgemm_pv_64x64<true, true><<<grid, 128, 0, ctx->stream>>>(
                        p_base, kv_base, kv_view, out_base, out_view, H, gqa, d.seq_q, d.seq_k);
                } else {
                    sdpa_sgemm_pv_64x64<true, false><<<grid, 128, 0, ctx->stream>>>(
                        p_base, kv_base, kv_view, out_base, out_view, H, gqa, d.seq_q, d.seq_k);
                }
            } else {
                if (acc) {
                    sdpa_sgemm_pv_64x64<false, true><<<grid, 128, 0, ctx->stream>>>(
                        p_base, kv_base, kv_view, out_base, out_view, H, gqa, d.seq_q, d.seq_k);
                } else {
                    sdpa_sgemm_pv_64x64<false, false><<<grid, 128, 0, ctx->stream>>>(
                        p_base, kv_base, kv_view, out_base, out_view, H, gqa, d.seq_q, d.seq_k);
                }
            }
            return rt(cudaGetLastError());
        }
        if ((mode == 4 || mode == 5) &&
            sdpa_gemm64_view_ok(q_view, B, H, q_base) &&
            sdpa_gemm64_view_ok(out_view, B, H_kv, out_base)) {
            const dim3 grid(static_cast<unsigned>(d.head_dim >> 6),
                            static_cast<unsigned>(d.seq_k >> 6),
                            static_cast<unsigned>(B * H));
            if (gqa == 1ULL) {
                if (d.is_causal) {
                    if (acc) {
                        sdpa_sgemm_ptdo_64x64<true, true><<<grid, 128, 0, ctx->stream>>>(
                            p_base, q_base, q_view, out_base, out_view, H, d.seq_q, d.seq_k);
                    } else {
                        sdpa_sgemm_ptdo_64x64<true, false><<<grid, 128, 0, ctx->stream>>>(
                            p_base, q_base, q_view, out_base, out_view, H, d.seq_q, d.seq_k);
                    }
                } else {
                    if (acc) {
                        sdpa_sgemm_ptdo_64x64<false, true><<<grid, 128, 0, ctx->stream>>>(
                            p_base, q_base, q_view, out_base, out_view, H, d.seq_q, d.seq_k);
                    } else {
                        sdpa_sgemm_ptdo_64x64<false, false><<<grid, 128, 0, ctx->stream>>>(
                            p_base, q_base, q_view, out_base, out_view, H, d.seq_q, d.seq_k);
                    }
                }
                return rt(cudaGetLastError());
            } else if (ctx->sdpa_gqa_scratch &&
                       ctx->sdpa_gqa_scratch_floats >= static_cast<size_t>(B * H * d.seq_k * d.head_dim)) {
                FaSdpaView4 scratch_view{};
                scratch_view.offset = 0;
                scratch_view.strides[3] = 1;
                scratch_view.strides[2] = static_cast<int64_t>(d.head_dim);
                scratch_view.strides[1] = static_cast<int64_t>(d.seq_k * d.head_dim);
                scratch_view.strides[0] = static_cast<int64_t>(H * d.seq_k * d.head_dim);
                if (d.is_causal) {
                    sdpa_sgemm_ptdo_64x64<true, false><<<grid, 128, 0, ctx->stream>>>(
                        p_base, q_base, q_view, ctx->sdpa_gqa_scratch, scratch_view, H, d.seq_q, d.seq_k);
                } else {
                    sdpa_sgemm_ptdo_64x64<false, false><<<grid, 128, 0, ctx->stream>>>(
                        p_base, q_base, q_view, ctx->sdpa_gqa_scratch, scratch_view, H, d.seq_q, d.seq_k);
                }
                const uint64_t total_vec4 = B * H_kv * d.seq_k * (d.head_dim >> 2);
                const unsigned rg = sdpa_grid((total_vec4 + 255ULL) / 256ULL);
                if (acc) {
                    sdpa_gqa_reduce_vec4<true><<<rg, 256, 0, ctx->stream>>>(
                        ctx->sdpa_gqa_scratch, out_base, out_view, B, H, H_kv, d.seq_k, d.head_dim);
                } else {
                    sdpa_gqa_reduce_vec4<false><<<rg, 256, 0, ctx->stream>>>(
                        ctx->sdpa_gqa_scratch, out_base, out_view, B, H, H_kv, d.seq_k, d.head_dim);
                }
                return rt(cudaGetLastError());
            }
        }
    }

    const int T_q = static_cast<int>(d.seq_q);
    const int T_k = static_cast<int>(d.seq_k);
    const int D = static_cast<int>(d.head_dim);
    const int ldq = static_cast<int>(q_view.strides[2]);
    const int ldkv = static_cast<int>(kv_view.strides[2]);
    const int ldout = static_cast<int>(out_view.strides[2]);
    const int ldp = T_k;

    const int64_t q_s1 = (H > 1) ? q_view.strides[1] : static_cast<int64_t>(d.seq_q * d.head_dim);
    const int64_t kv_s1 = (H_kv > 1) ? kv_view.strides[1] : static_cast<int64_t>(d.seq_k * d.head_dim);
    const int64_t out_s1 = ((mode >= 4 ? H_kv : H) > 1)
        ? out_view.strides[1]
        : static_cast<int64_t>((mode >= 4 ? d.seq_k : d.seq_q) * d.head_dim);
    const int64_t p_s1 = static_cast<int64_t>(d.seq_q * d.seq_k);
    const int64_t p_s0 = static_cast<int64_t>(H) * p_s1;

    const bool q_merge = (B == 1) || (q_view.strides[0] == static_cast<int64_t>(H) * q_s1);
    const bool kv_merge = (B == 1) || (kv_view.strides[0] == static_cast<int64_t>(H_kv) * kv_s1);
    const bool out_merge = (B == 1) || (out_view.strides[0] == static_cast<int64_t>(mode >= 4 ? H_kv : H) * out_s1);
    const bool merge_bh = (mode == 0 || mode == 2)
        ? (q_merge && kv_merge)
        : (mode == 1 || mode == 3)
            ? (kv_merge && out_merge)
            : (q_merge && out_merge);

    const uint64_t outer_b = merge_bh ? 1ULL : B;
    const int batch_count = static_cast<int>(merge_bh ? (B * H_kv) : H_kv);
    const long long stride_q = static_cast<long long>(gqa) * q_s1;
    const long long stride_kv = static_cast<long long>(kv_s1);
    const long long stride_p = static_cast<long long>(gqa) * p_s1;
    const long long stride_out_q = static_cast<long long>(gqa) * out_s1;
    const long long stride_out_kv = static_cast<long long>(out_s1);
    const float alpha = 1.0f;

    for (uint64_t b = 0; b < outer_b; ++b) {
        for (uint64_t hg = 0; hg < gqa; ++hg) {
            const float *qp = q_base + q_view.offset + static_cast<int64_t>(b) * q_view.strides[0]
                              + static_cast<int64_t>(hg) * q_s1;
            const float *kvp = kv_base + kv_view.offset + static_cast<int64_t>(b) * kv_view.strides[0];
            float *pp = p_base + static_cast<int64_t>(b) * p_s0 + static_cast<int64_t>(hg) * p_s1;
            int status = 0;
            if (mode == 0 || mode == 2) {
                const float zero = 0.0f;
                status = blas_sdpa(cublasSgemmStridedBatched(
                    ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    T_k, T_q, D,
                    &alpha,
                    kvp, ldkv, stride_kv,
                    qp, ldq, stride_q,
                    &zero,
                    pp, ldp, stride_p,
                    batch_count));
            } else if (mode == 1 || mode == 3) {
                float *op = out_base + out_view.offset + static_cast<int64_t>(b) * out_view.strides[0]
                            + static_cast<int64_t>(hg) * out_s1;
                const float beta = init_beta;
                status = blas_sdpa(cublasSgemmStridedBatched(
                    ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    D, T_q, T_k,
                    &alpha,
                    kvp, ldkv, stride_kv,
                    pp, ldp, stride_p,
                    &beta,
                    op, ldout, stride_out_q,
                    batch_count));
            } else {
                float *op = out_base + out_view.offset + static_cast<int64_t>(b) * out_view.strides[0];
                const float beta = (hg == 0) ? init_beta : 1.0f;
                status = blas_sdpa(cublasSgemmStridedBatched(
                    ctx->blas, CUBLAS_OP_N, CUBLAS_OP_T,
                    D, T_k, T_q,
                    &alpha,
                    qp, ldq, stride_q,
                    pp, ldp, stride_p,
                    &beta,
                    op, ldout, stride_out_kv,
                    batch_count));
            }
            if (status) return status;
        }
    }
    return 0;
}

template <bool tile64_causal>
void launch_sdpa_softmax_vec4(cudaStream_t stream, float *p_cache, uint64_t total_rows,
                              uint64_t T_q, uint64_t T_k, float scale, int is_causal) {
    const unsigned g = sdpa_grid((total_rows + 7ULL) / 8ULL);
    if (T_k == 128ULL) {
        sdpa_scale_mask_softmax_warp_vec4<1, true, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, total_rows, T_q, T_k, scale, is_causal);
    } else if (T_k == 256ULL) {
        sdpa_scale_mask_softmax_warp_vec4<2, true, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, total_rows, T_q, T_k, scale, is_causal);
    } else if (T_k == 512ULL) {
        sdpa_scale_mask_softmax_warp_vec4<4, true, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, total_rows, T_q, T_k, scale, is_causal);
    } else if (T_k <= 128ULL) {
        sdpa_scale_mask_softmax_warp_vec4<1, false, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, total_rows, T_q, T_k, scale, is_causal);
    } else if (T_k <= 256ULL) {
        sdpa_scale_mask_softmax_warp_vec4<2, false, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, total_rows, T_q, T_k, scale, is_causal);
    } else {
        sdpa_scale_mask_softmax_warp_vec4<4, false, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, total_rows, T_q, T_k, scale, is_causal);
    }
}

template <bool tile64_causal>
void launch_sdpa_softmax_bwd_vec4(cudaStream_t stream, const float *p_cache, float *ds_cache,
                                  uint64_t total_rows, uint64_t T_q, uint64_t T_k, float scale, int is_causal) {
    const unsigned g = sdpa_grid((total_rows + 7ULL) / 8ULL);
    if (T_k == 128ULL) {
        sdpa_softmax_bwd_warp_vec4<1, true, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, ds_cache, total_rows, T_q, T_k, scale, is_causal);
    } else if (T_k == 256ULL) {
        sdpa_softmax_bwd_warp_vec4<2, true, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, ds_cache, total_rows, T_q, T_k, scale, is_causal);
    } else if (T_k == 512ULL) {
        sdpa_softmax_bwd_warp_vec4<4, true, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, ds_cache, total_rows, T_q, T_k, scale, is_causal);
    } else if (T_k <= 128ULL) {
        sdpa_softmax_bwd_warp_vec4<1, false, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, ds_cache, total_rows, T_q, T_k, scale, is_causal);
    } else if (T_k <= 256ULL) {
        sdpa_softmax_bwd_warp_vec4<2, false, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, ds_cache, total_rows, T_q, T_k, scale, is_causal);
    } else {
        sdpa_softmax_bwd_warp_vec4<4, false, tile64_causal><<<g, 256, 0, stream>>>(
            p_cache, ds_cache, total_rows, T_q, T_k, scale, is_causal);
    }
}

int sdpa_cublas_compute_p(Context *ctx, const FaSdpaDims &d, const float *q, const float *k,
                          bool use_tile64_causal) {
    int code = sdpa_launch_gemm(
        ctx, d, 0, q, d.q, k, d.k, ctx->sdpa_p_cache, nullptr, d.q, 0.0f, use_tile64_causal);
    if (code) return code;
    const uint64_t total_rows = d.batch * d.num_heads * d.seq_q;
    if (d.seq_k <= 512ULL && (d.seq_k & 3ULL) == 0 &&
        (reinterpret_cast<uintptr_t>(ctx->sdpa_p_cache) & 15ULL) == 0) {
        if (use_tile64_causal) {
            launch_sdpa_softmax_vec4<true>(
                ctx->stream, ctx->sdpa_p_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
        } else {
            launch_sdpa_softmax_vec4<false>(
                ctx->stream, ctx->sdpa_p_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
        }
    } else if (d.seq_k <= 512ULL) {
        const unsigned g = sdpa_grid((total_rows + 7ULL) / 8ULL);
        if (d.seq_k <= 128ULL) {
            sdpa_scale_mask_softmax_warp<4><<<g, 256, 0, ctx->stream>>>(
                ctx->sdpa_p_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
        } else if (d.seq_k <= 256ULL) {
            sdpa_scale_mask_softmax_warp<8><<<g, 256, 0, ctx->stream>>>(
                ctx->sdpa_p_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
        } else {
            sdpa_scale_mask_softmax_warp<16><<<g, 256, 0, ctx->stream>>>(
                ctx->sdpa_p_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
        }
    } else {
        const unsigned g = sdpa_grid(total_rows);
        sdpa_scale_mask_softmax_block<<<g, 256, 0, ctx->stream>>>(
            ctx->sdpa_p_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
    }
    return rt(cudaGetLastError());
}

int sdpa_cublas_compute_ds(Context *ctx, const FaSdpaDims &d, const float *v, const float *dy,
                           bool use_tile64_causal) {
    int code = sdpa_launch_gemm(
        ctx, d, 2, dy, d.dy_or_o, v, d.v, ctx->sdpa_ds_cache, nullptr, d.dy_or_o, 0.0f, use_tile64_causal);
    if (code) return code;
    const uint64_t total_rows = d.batch * d.num_heads * d.seq_q;
    if (d.seq_k <= 512ULL && (d.seq_k & 3ULL) == 0 &&
        (reinterpret_cast<uintptr_t>(ctx->sdpa_p_cache) & 15ULL) == 0 &&
        (reinterpret_cast<uintptr_t>(ctx->sdpa_ds_cache) & 15ULL) == 0) {
        if (use_tile64_causal) {
            launch_sdpa_softmax_bwd_vec4<true>(
                ctx->stream, ctx->sdpa_p_cache, ctx->sdpa_ds_cache, total_rows,
                d.seq_q, d.seq_k, d.scale, d.is_causal);
        } else {
            launch_sdpa_softmax_bwd_vec4<false>(
                ctx->stream, ctx->sdpa_p_cache, ctx->sdpa_ds_cache, total_rows,
                d.seq_q, d.seq_k, d.scale, d.is_causal);
        }
    } else if (d.seq_k <= 512ULL) {
        const unsigned g = sdpa_grid((total_rows + 7ULL) / 8ULL);
        if (d.seq_k <= 128ULL) {
            sdpa_softmax_bwd_warp<4><<<g, 256, 0, ctx->stream>>>(
                ctx->sdpa_p_cache, ctx->sdpa_ds_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
        } else if (d.seq_k <= 256ULL) {
            sdpa_softmax_bwd_warp<8><<<g, 256, 0, ctx->stream>>>(
                ctx->sdpa_p_cache, ctx->sdpa_ds_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
        } else {
            sdpa_softmax_bwd_warp<16><<<g, 256, 0, ctx->stream>>>(
                ctx->sdpa_p_cache, ctx->sdpa_ds_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
        }
    } else {
        const unsigned g = sdpa_grid(total_rows);
        sdpa_softmax_bwd_block<<<g, 256, 0, ctx->stream>>>(
            ctx->sdpa_p_cache, ctx->sdpa_ds_cache, total_rows, d.seq_q, d.seq_k, d.scale, d.is_causal);
    }
    return rt(cudaGetLastError());
}

} // namespace

#define TRY_SDPA(expr) do { const int code = (expr); if (code) return code; } while (0)

extern "C" int fa_cuda_sdpa_create(Context *ctx, FaSdpaDims d,
                                   size_t *workspace, size_t *shared, void **out) {
    if (!ctx || !workspace || !shared || !out || d.operand < -1 || d.operand > 2 ||
        !(d.scale > 0.0f) || !std::isfinite(d.scale)) {
        return rt(cudaErrorInvalidValue);
    }
    TRY_SDPA(activate(ctx));
    auto p = std::unique_ptr<SdpaPlanCuda>(new (std::nothrow) SdpaPlanCuda(ctx, d));
    if (!p) return 1002;

    if (can_use_cublas_sdpa(d)) {
        const size_t needed = static_cast<size_t>(d.batch * d.num_heads * d.seq_q * d.seq_k);
        if (needed > ctx->sdpa_cache_floats) {
            TRY_SDPA(rt(cudaStreamSynchronize(ctx->stream)));
            if (ctx->sdpa_p_cache) {
                cudaFree(ctx->sdpa_p_cache);
                ctx->sdpa_p_cache = nullptr;
            }
            if (ctx->sdpa_ds_cache) {
                cudaFree(ctx->sdpa_ds_cache);
                ctx->sdpa_ds_cache = nullptr;
            }
            ctx->sdpa_cache_floats = 0;
            TRY_SDPA(rt(cudaMalloc(&ctx->sdpa_p_cache, needed * sizeof(float))));
            TRY_SDPA(rt(cudaMalloc(&ctx->sdpa_ds_cache, needed * sizeof(float))));
            ctx->sdpa_cache_floats = needed;
            ctx->sdpa_p_mask = 0;
            ctx->sdpa_ds_mask = 0;
            ctx->sdpa_p_tile64_causal = 0;
        }
        if (d.num_heads > d.num_kv_heads) {
            const size_t gqa_needed = static_cast<size_t>(d.batch * d.num_heads * d.seq_k * d.head_dim);
            if (gqa_needed > ctx->sdpa_gqa_scratch_floats) {
                TRY_SDPA(rt(cudaStreamSynchronize(ctx->stream)));
                if (ctx->sdpa_gqa_scratch) {
                    cudaFree(ctx->sdpa_gqa_scratch);
                    ctx->sdpa_gqa_scratch = nullptr;
                }
                ctx->sdpa_gqa_scratch_floats = 0;
                TRY_SDPA(rt(cudaMalloc(&ctx->sdpa_gqa_scratch, gqa_needed * sizeof(float))));
                ctx->sdpa_gqa_scratch_floats = gqa_needed;
            }
        }
    }

    const bool non_empty = d.batch && d.num_heads && d.seq_q && d.head_dim;
    size_t ws = 0;
    const bool need_lse = non_empty && (d.operand == 1 || d.operand == 2 || (d.operand == 0 && !d.fast_inner));
    const bool need_drow = non_empty && (d.operand == 1 || (d.operand == 0 && !d.fast_inner));
    if (need_lse) {
        const size_t bytes = static_cast<size_t>(d.batch * d.num_heads * d.seq_q) * sizeof(float);
        TRY_SDPA(rt(cudaMalloc(&p->lse_buf, bytes)));
        ws += bytes;
    }
    if (need_drow) {
        const size_t bytes = static_cast<size_t>(d.batch * d.num_heads * d.seq_q) * sizeof(float);
        TRY_SDPA(rt(cudaMalloc(&p->drow_buf, bytes)));
        ws += bytes;
    }

    cudaFuncAttributes attr{};
    if (!d.fast_inner) {
        TRY_SDPA(rt(cudaFuncGetAttributes(&attr, sdpa_fwd_general)));
    } else if (d.operand == 1 || d.operand == 2) {
        if (d.head_dim <= 32) {
            TRY_SDPA(rt(cudaFuncGetAttributes(&attr, sdpa_bwd_kv_flash<32, true, false>)));
        } else if (d.head_dim <= 64) {
            TRY_SDPA(rt(cudaFuncGetAttributes(&attr, sdpa_bwd_kv_flash<64, true, false>)));
        } else {
            TRY_SDPA(rt(cudaFuncGetAttributes(&attr, sdpa_bwd_kv_flash<128, true, false>)));
        }
    } else if (d.head_dim <= 32) {
        TRY_SDPA(rt(cudaFuncGetAttributes(&attr, sdpa_fwd_flash<32>)));
    } else if (d.head_dim <= 64) {
        TRY_SDPA(rt(cudaFuncGetAttributes(&attr, sdpa_fwd_flash<64>)));
    } else {
        TRY_SDPA(rt(cudaFuncGetAttributes(&attr, sdpa_fwd_flash<128>)));
    }
    *shared = attr.sharedSizeBytes;
    *workspace = ws;
    *out = p.release();
    return 0;
}

extern "C" int fa_cuda_sdpa_forward_run(const SdpaPlanCuda *p,
                                        const float *q, const float *k, const float *v, float *o) {
    if (!p || p->dims.operand != -1) return rt(cudaErrorInvalidValue);
    const auto d = p->dims;
    if (!d.batch || !d.num_heads || !d.seq_q || !d.head_dim) return 0;
    if (!q || !k || !v || !o) return rt(cudaErrorInvalidValue);
    TRY_SDPA(activate(p->ctx));
    if (can_use_cublas_sdpa(d) && p->ctx->sdpa_cache_floats >= d.batch * d.num_heads * d.seq_q * d.seq_k) {
        const bool tile64_causal = (d.is_causal != 0) &&
                                   sdpa_gemm64_shape_ok(d) &&
                                   sdpa_gemm64_view_ok(d.q, d.batch, d.num_heads, q) &&
                                   sdpa_gemm64_view_ok(d.k, d.batch, d.num_kv_heads, k) &&
                                   sdpa_gemm64_view_ok(d.v, d.batch, d.num_kv_heads, v) &&
                                   sdpa_gemm64_view_ok(d.dy_or_o, d.batch, d.num_heads, o) &&
                                   ((reinterpret_cast<uintptr_t>(p->ctx->sdpa_p_cache) & 15ULL) == 0);
        TRY_SDPA(sdpa_cublas_compute_p(p->ctx, d, q, k, tile64_causal));
        TRY_SDPA(sdpa_launch_gemm(
            p->ctx, d, 1, q, d.q, v, d.v, p->ctx->sdpa_p_cache, o, d.dy_or_o, 0.0f, tile64_causal));
        uint32_t scale_bits = 0;
        std::memcpy(&scale_bits, &d.scale, sizeof(uint32_t));
        p->ctx->sdpa_last_q = q + d.q.offset;
        p->ctx->sdpa_last_k = k + d.k.offset;
        p->ctx->sdpa_last_bh = d.batch * d.num_heads;
        p->ctx->sdpa_last_tq = d.seq_q;
        p->ctx->sdpa_last_tk = d.seq_k;
        p->ctx->sdpa_last_d = d.head_dim;
        p->ctx->sdpa_last_scale_bits = scale_bits;
        p->ctx->sdpa_last_causal = d.is_causal;
        p->ctx->sdpa_p_mask = 7;
        p->ctx->sdpa_ds_mask = 0;
        p->ctx->sdpa_p_tile64_causal = tile64_causal ? 1 : 0;
        return 0;
    }
    if (d.fast_inner) {
        const uint64_t tiles = d.batch * d.num_heads * ((d.seq_q + 31ULL) / 32ULL);
        const unsigned g = sdpa_grid(tiles);
        if (d.head_dim <= 32) sdpa_fwd_flash<32><<<g, 256, 0, p->ctx->stream>>>(d, q, k, v, o);
        else if (d.head_dim <= 64) sdpa_fwd_flash<64><<<g, 256, 0, p->ctx->stream>>>(d, q, k, v, o);
        else sdpa_fwd_flash<128><<<g, 256, 0, p->ctx->stream>>>(d, q, k, v, o);
    } else {
        const unsigned g = sdpa_grid(d.batch * d.num_heads * d.seq_q);
        sdpa_fwd_general<<<g, 256, 0, p->ctx->stream>>>(d, q, k, v, o);
    }
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_sdpa_backward_run(const SdpaPlanCuda *p,
                                         const float *q, const float *k, const float *v,
                                         const float *dy, float *out, int accumulate) {
    if (!p || p->dims.operand < 0 || p->dims.operand > 2 || (accumulate != 0 && accumulate != 1)) {
        return rt(cudaErrorInvalidValue);
    }
    const auto d = p->dims;
    if (!d.batch || !d.num_heads || !d.seq_q || !d.head_dim) return 0;
    if (!q || !k || !dy || !out || (d.operand != 2 && !v)) return rt(cudaErrorInvalidValue);
    TRY_SDPA(activate(p->ctx));
    const bool acc = (accumulate != 0);

    if (can_use_cublas_sdpa(d) && p->ctx->sdpa_cache_floats >= d.batch * d.num_heads * d.seq_q * d.seq_k) {
        Context *ctx = p->ctx;
        const bool gqa_ok = (d.num_heads == d.num_kv_heads) ||
                            (ctx->sdpa_gqa_scratch &&
                             ctx->sdpa_gqa_scratch_floats >=
                                 static_cast<size_t>(d.batch * d.num_heads * d.seq_k * d.head_dim));
        const bool out_gemm64_ok =
            sdpa_gemm64_shape_ok(d) &&
            ((reinterpret_cast<uintptr_t>(ctx->sdpa_p_cache) & 15ULL) == 0) &&
            ((reinterpret_cast<uintptr_t>(ctx->sdpa_ds_cache) & 15ULL) == 0) &&
            (d.operand == 0
                 ? (sdpa_gemm64_view_ok(d.k, d.batch, d.num_kv_heads, k) &&
                    sdpa_gemm64_view_ok(d.out, d.batch, d.num_heads, out))
                 : (d.operand == 1
                        ? (gqa_ok && sdpa_gemm64_view_ok(d.q, d.batch, d.num_heads, q) &&
                           sdpa_gemm64_view_ok(d.out, d.batch, d.num_kv_heads, out))
                        : (gqa_ok && sdpa_gemm64_view_ok(d.dy_or_o, d.batch, d.num_heads, dy) &&
                           sdpa_gemm64_view_ok(d.out, d.batch, d.num_kv_heads, out))));
        const bool p_tile64 = (d.is_causal != 0) && out_gemm64_ok &&
                              sdpa_gemm64_view_ok(d.q, d.batch, d.num_heads, q) &&
                              sdpa_gemm64_view_ok(d.k, d.batch, d.num_kv_heads, k);
        const bool ds_tile64 = (d.is_causal != 0) && out_gemm64_ok &&
                               (d.operand != 2) &&
                               sdpa_gemm64_view_ok(d.v, d.batch, d.num_kv_heads, v) &&
                               sdpa_gemm64_view_ok(d.dy_or_o, d.batch, d.num_heads, dy);

        uint32_t scale_bits = 0;
        std::memcpy(&scale_bits, &d.scale, sizeof(uint32_t));
        const int bit = 1 << d.operand;
        const bool p_hit = (ctx->sdpa_p_mask & bit) != 0 &&
                           ctx->sdpa_last_q == (q + d.q.offset) &&
                           ctx->sdpa_last_k == (k + d.k.offset) &&
                           ctx->sdpa_last_bh == d.batch * d.num_heads &&
                           ctx->sdpa_last_tq == d.seq_q &&
                           ctx->sdpa_last_tk == d.seq_k &&
                           ctx->sdpa_last_d == d.head_dim &&
                           ctx->sdpa_last_scale_bits == scale_bits &&
                           ctx->sdpa_last_causal == d.is_causal &&
                           (!ctx->sdpa_p_tile64_causal || out_gemm64_ok);
        if (!p_hit) {
            TRY_SDPA(sdpa_cublas_compute_p(ctx, d, q, k, p_tile64));
            ctx->sdpa_last_q = q + d.q.offset;
            ctx->sdpa_last_k = k + d.k.offset;
            ctx->sdpa_last_bh = d.batch * d.num_heads;
            ctx->sdpa_last_tq = d.seq_q;
            ctx->sdpa_last_tk = d.seq_k;
            ctx->sdpa_last_d = d.head_dim;
            ctx->sdpa_last_scale_bits = scale_bits;
            ctx->sdpa_last_causal = d.is_causal;
            ctx->sdpa_p_mask = 7 & ~bit;
            ctx->sdpa_ds_mask = 0;
            ctx->sdpa_p_tile64_causal = p_tile64 ? 1 : 0;
        } else {
            ctx->sdpa_p_mask &= ~bit;
        }

        if (d.operand == 0 || d.operand == 1) {
            const bool ds_hit = (ctx->sdpa_ds_mask & bit) != 0 &&
                                ctx->sdpa_last_v == (v + d.v.offset) &&
                                ctx->sdpa_last_dy == (dy + d.dy_or_o.offset) &&
                                (!ctx->sdpa_p_tile64_causal || out_gemm64_ok);
            if (!ds_hit) {
                TRY_SDPA(sdpa_cublas_compute_ds(ctx, d, v, dy, ds_tile64));
                ctx->sdpa_last_v = v + d.v.offset;
                ctx->sdpa_last_dy = dy + d.dy_or_o.offset;
                ctx->sdpa_ds_mask = 3 & ~bit;
            } else {
                ctx->sdpa_ds_mask &= ~bit;
            }
        }

        const float init_beta = acc ? 1.0f : 0.0f;
        if (d.operand == 0) {
            return sdpa_launch_gemm(ctx, d, 3, q, d.q, k, d.k, ctx->sdpa_ds_cache, out, d.out, init_beta, ds_tile64);
        } else if (d.operand == 1) {
            return sdpa_launch_gemm(ctx, d, 4, q, d.q, k, d.k, ctx->sdpa_ds_cache, out, d.out, init_beta, ds_tile64);
        } else {
            return sdpa_launch_gemm(ctx, d, 5, dy, d.dy_or_o, k, d.k, ctx->sdpa_p_cache, out, d.out, init_beta, p_tile64);
        }
    }

    if (d.fast_inner) {
        const uint64_t q_tiles = d.batch * d.num_heads * ((d.seq_q + 31ULL) / 32ULL);
        const uint64_t k_tiles = d.batch * d.num_kv_heads * ((d.seq_k + 31ULL) / 32ULL);
        const unsigned gq = sdpa_grid(q_tiles);
        const unsigned gk = sdpa_grid(k_tiles);
        if (d.operand == 0) {
            if (d.head_dim <= 32) {
                if (acc) sdpa_bwd_dq_flash<32, true><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, out);
                else sdpa_bwd_dq_flash<32, false><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, out);
            } else if (d.head_dim <= 64) {
                if (acc) sdpa_bwd_dq_flash<64, true><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, out);
                else sdpa_bwd_dq_flash<64, false><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, out);
            } else {
                if (acc) sdpa_bwd_dq_flash<128, true><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, out);
                else sdpa_bwd_dq_flash<128, false><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, out);
            }
        } else if (d.operand == 1) {
            if (d.head_dim <= 32) {
                sdpa_bwd_prep_flash<32, true><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf);
                if (acc) sdpa_bwd_kv_flash<32, true, true><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, out);
                else sdpa_bwd_kv_flash<32, true, false><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, out);
            } else if (d.head_dim <= 64) {
                sdpa_bwd_prep_flash<64, true><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf);
                if (acc) sdpa_bwd_kv_flash<64, true, true><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, out);
                else sdpa_bwd_kv_flash<64, true, false><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, out);
            } else {
                sdpa_bwd_prep_flash<128, true><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf);
                if (acc) sdpa_bwd_kv_flash<128, true, true><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, out);
                else sdpa_bwd_kv_flash<128, true, false><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, out);
            }
        } else {
            if (d.head_dim <= 32) {
                sdpa_bwd_prep_flash<32, false><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, nullptr, p->lse_buf, nullptr);
                if (acc) sdpa_bwd_kv_flash<32, false, true><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, dy, p->lse_buf, nullptr, out);
                else sdpa_bwd_kv_flash<32, false, false><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, dy, p->lse_buf, nullptr, out);
            } else if (d.head_dim <= 64) {
                sdpa_bwd_prep_flash<64, false><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, nullptr, p->lse_buf, nullptr);
                if (acc) sdpa_bwd_kv_flash<64, false, true><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, dy, p->lse_buf, nullptr, out);
                else sdpa_bwd_kv_flash<64, false, false><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, dy, p->lse_buf, nullptr, out);
            } else {
                sdpa_bwd_prep_flash<128, false><<<gq, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, nullptr, p->lse_buf, nullptr);
                if (acc) sdpa_bwd_kv_flash<128, false, true><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, dy, p->lse_buf, nullptr, out);
                else sdpa_bwd_kv_flash<128, false, false><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, dy, p->lse_buf, nullptr, out);
            }
        }
        return rt(cudaGetLastError());
    }

    const unsigned gq = sdpa_grid(d.batch * d.num_heads * d.seq_q);
    const unsigned gk = sdpa_grid(d.batch * d.num_kv_heads * d.seq_k);
    if (d.operand == 0) {
        sdpa_bwd_prep_general<<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, 1);
        sdpa_bwd_dq_general<<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, out, accumulate);
    } else if (d.operand == 1) {
        sdpa_bwd_prep_general<<<gq, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, 1);
        sdpa_bwd_kv_general<true><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, v, dy, p->lse_buf, p->drow_buf, out, accumulate);
    } else {
        sdpa_bwd_prep_general<<<gq, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, nullptr, p->lse_buf, nullptr, 0);
        sdpa_bwd_kv_general<false><<<gk, 256, 0, p->ctx->stream>>>(d, q, k, nullptr, dy, p->lse_buf, nullptr, out, accumulate);
    }
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_sdpa_destroy(SdpaPlanCuda *p) { delete p; }
