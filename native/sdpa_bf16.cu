#include "sdpa_bf16.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr float LOG2E_F = 1.44269504088896340736f;
constexpr float LN2_F = 0.69314718055994530942f;

__host__ __device__ __forceinline__ uint64_t umin64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

__device__ __forceinline__ uint32_t pack_bf16_u32(float a, float b) {
    __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&p);
}

__device__ __forceinline__ float2 unpack_bf16_u32(uint32_t p) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t smem_addr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
#else
    (void)smem_addr; r0 = r1 = r2 = r3 = 0u;
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t smem_addr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
#else
    (void)smem_addr; r0 = r1 = r2 = r3 = 0u;
#endif
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

// Bank-conflict-free XOR swizzle for 16-byte (8 BF16) chunks.
template<int D8>
__device__ __forceinline__ uint4 *smem_ptr(uint4 *base, int row, int c8) {
    return base + row * D8 + (c8 ^ (row & 7));
}

// Coalesced 128-thread load of [64, D] BF16 tile from strided global memory into swizzled SMEM.
template<int D>
__device__ __forceinline__ void load_tile_64xD(
    uint4 *smem,
    const __nv_bfloat16 *gmem_head_base,
    int64_t stride_t,
    int row_start,
    int max_rows)
{
    constexpr int D8 = D >> 3;
    constexpr int TOTAL = 64 * D8;
    for (int idx = threadIdx.x; idx < TOTAL; idx += 128) {
        int r = idx / D8;
        int c8 = idx - r * D8;
        int gr = row_start + r;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (gr < max_rows) {
            val = reinterpret_cast<const uint4 *>(gmem_head_base + static_cast<int64_t>(gr) * stride_t)[c8];
        }
        *smem_ptr<D8>(smem, r, c8) = val;
    }
}

// Load [16, 16] A-fragment from swizzled [64, D] SMEM at (row_off, k_off).
template<int D8>
__device__ __forceinline__ void load_frag_A(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    uint4 *smem, int row_off, int k_off, int lane)
{
    int mat = lane >> 3;
    int r = row_off + ((mat & 1) << 3) + (lane & 7);
    int c8 = (k_off >> 3) + (mat >> 1);
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr<D8>(smem, r, c8)));
    ldmatrix_x4(a0, a1, a2, a3, addr);
}

// Load two [8, 16] row-major B^T fragments (n_off..n_off+15, k_off..k_off+15) from [64, D] SMEM.
template<int D8>
__device__ __forceinline__ void load_frag_B_nt(
    uint32_t &b0_0, uint32_t &b1_0, uint32_t &b0_1, uint32_t &b1_1,
    uint4 *smem, int n_off, int k_off, int lane)
{
    int mat = lane >> 3;
    int n = n_off + ((mat >> 1) << 3) + (lane & 7);
    int c8 = (k_off >> 3) + (mat & 1);
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr<D8>(smem, n, c8)));
    ldmatrix_x4(b0_0, b1_0, b0_1, b1_1, addr);
}

// Load two [16, 8] transposed B fragments (k_off..k_off+15, d_off..d_off+15) from [64, D] SMEM.
template<int D8>
__device__ __forceinline__ void load_frag_B_trans(
    uint32_t &b0_0, uint32_t &b1_0, uint32_t &b0_1, uint32_t &b1_1,
    uint4 *smem, int k_off, int d_off, int lane)
{
    int mat = lane >> 3;
    int k = k_off + ((mat & 1) << 3) + (lane & 7);
    int c8 = (d_off >> 3) + (mat >> 1);
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr<D8>(smem, k, c8)));
    ldmatrix_x4_trans(b0_0, b1_0, b0_1, b1_1, addr);
}

// 4-lane XOR reduction across t = lane & 3 (all threads in same row g get the exact result).
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

// ============================================================================
// BF16 Tensor Core FlashAttention-2 FORWARD (BR=64, BC=64, 4 warps = 128 thr)
// ============================================================================
template<int D, bool CAUSAL>
__global__ void __launch_bounds__(128, 2) sdpa_bf16_tc_fwd_kernel(
    uint64_t h_q,
    uint64_t h_kv,
    uint64_t t_q,
    uint64_t t_k,
    float scale,
    const __nv_bfloat16 *__restrict__ Q, Bf16SdpaStrides sq,
    const __nv_bfloat16 *__restrict__ K, Bf16SdpaStrides sk,
    const __nv_bfloat16 *__restrict__ V, Bf16SdpaStrides sv,
    __nv_bfloat16 *__restrict__ O, Bf16SdpaStrides so,
    float *__restrict__ LSE)
{
    constexpr int D8 = D >> 3;
    constexpr int DT = D >> 3; // number of N=8 output tiles along D
    constexpr int KT = D >> 4; // number of K=16 inner tiles along D

    __shared__ uint4 s_K[64 * D8];
    __shared__ uint4 s_V[64 * D8];

    int q_block = static_cast<int>(blockIdx.x) * 64;
    if (q_block >= static_cast<int>(t_q)) return;

    uint64_t bh = blockIdx.y;
    uint64_t b = bh / h_q;
    uint64_t hq = bh - b * h_q;
    uint64_t gqa = h_q / h_kv;
    uint64_t hkv = hq / gqa;

    const __nv_bfloat16 *Q_head = Q + static_cast<int64_t>(b) * sq.sb + static_cast<int64_t>(hq) * sq.sh;
    const __nv_bfloat16 *K_head = K + static_cast<int64_t>(b) * sk.sb + static_cast<int64_t>(hkv) * sk.sh;
    const __nv_bfloat16 *V_head = V + static_cast<int64_t>(b) * sv.sb + static_cast<int64_t>(hkv) * sv.sh;
    __nv_bfloat16 *O_head = O + static_cast<int64_t>(b) * so.sb + static_cast<int64_t>(hq) * so.sh;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    int r0 = q_block + warp * 16 + g;
    int r1 = r0 + 8;

    // Load Q[64, D] once via s_K and keep in registers across the entire KV loop!
    load_tile_64xD<D>(s_K, Q_head, sq.st, q_block, static_cast<int>(t_q));
    __syncthreads();

    uint32_t q_reg[KT][4];
    #pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
        load_frag_A<D8>(q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                        s_K, warp * 16, kt * 16, lane);
    }
    __syncthreads();

    float O_acc[DT][4];
    #pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
        O_acc[dt][0] = 0.0f; O_acc[dt][1] = 0.0f;
        O_acc[dt][2] = 0.0f; O_acc[dt][3] = 0.0f;
    }
    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;
    const float scale_log2 = scale * LOG2E_F;

    int k_limit = static_cast<int>(t_k);
    if constexpr (CAUSAL) {
        int max_k = q_block + 64;
        if (max_k < k_limit) k_limit = max_k;
    }

    for (int k_start = 0; k_start < k_limit; k_start += 64) {
        load_tile_64xD<D>(s_K, K_head, sk.st, k_start, static_cast<int>(t_k));
        load_tile_64xD<D>(s_V, V_head, sv.st, k_start, static_cast<int>(t_k));
        __syncthreads();

        // 1. S[16, 64] = Q_reg[16, D] @ K_smem[64, D]^T
        float S[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            S[nt][0] = 0.0f; S[nt][1] = 0.0f; S[nt][2] = 0.0f; S[nt][3] = 0.0f;
        }
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            uint32_t a0 = q_reg[kt][0], a1 = q_reg[kt][1], a2 = q_reg[kt][2], a3 = q_reg[kt][3];
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_nt<D8>(b0_0, b1_0, b0_1, b1_1, s_K, nb * 16, kt * 16, lane);
                int n0 = nb * 2;
                int n1 = n0 + 1;
                mma_m16n8k16_bf16(
                    S[n0][0], S[n0][1], S[n0][2], S[n0][3],
                    a0, a1, a2, a3, b0_0, b1_0,
                    S[n0][0], S[n0][1], S[n0][2], S[n0][3]);
                mma_m16n8k16_bf16(
                    S[n1][0], S[n1][1], S[n1][2], S[n1][3],
                    a0, a1, a2, a3, b0_1, b1_1,
                    S[n1][0], S[n1][1], S[n1][2], S[n1][3]);
            }
        }

        // 2. Scale, mask, and online softmax (base-2 exp2f)
        bool need_mask = (k_start + 64 > static_cast<int>(t_k)) ||
                         (CAUSAL && (k_start + 63 > q_block + warp * 16));
        float row_m0 = -INFINITY;
        float row_m1 = -INFINITY;
        if (need_mask) {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                int c0 = k_start + nt * 8 + 2 * t;
                int c1 = c0 + 1;
                float s0 = S[nt][0] * scale_log2;
                float s1 = S[nt][1] * scale_log2;
                float s2 = S[nt][2] * scale_log2;
                float s3 = S[nt][3] * scale_log2;
                if (c0 >= static_cast<int>(t_k) || (CAUSAL && c0 > r0)) s0 = -INFINITY;
                if (c1 >= static_cast<int>(t_k) || (CAUSAL && c1 > r0)) s1 = -INFINITY;
                if (c0 >= static_cast<int>(t_k) || (CAUSAL && c0 > r1)) s2 = -INFINITY;
                if (c1 >= static_cast<int>(t_k) || (CAUSAL && c1 > r1)) s3 = -INFINITY;
                S[nt][0] = s0; S[nt][1] = s1; S[nt][2] = s2; S[nt][3] = s3;
                row_m0 = fmaxf(row_m0, fmaxf(s0, s1));
                row_m1 = fmaxf(row_m1, fmaxf(s2, s3));
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                S[nt][0] *= scale_log2;
                S[nt][1] *= scale_log2;
                S[nt][2] *= scale_log2;
                S[nt][3] *= scale_log2;
                row_m0 = fmaxf(row_m0, fmaxf(S[nt][0], S[nt][1]));
                row_m1 = fmaxf(row_m1, fmaxf(S[nt][2], S[nt][3]));
            }
        }
        row_m0 = quad_max(row_m0);
        row_m1 = quad_max(row_m1);

        float m0_new = fmaxf(m0, row_m0);
        float m1_new = fmaxf(m1, row_m1);
        float alpha0 = (m0 == -INFINITY || m0_new == -INFINITY) ? 0.0f : exp2f(m0 - m0_new);
        float alpha1 = (m1 == -INFINITY || m1_new == -INFINITY) ? 0.0f : exp2f(m1 - m1_new);

        l0 *= alpha0;
        l1 *= alpha1;
        #pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            O_acc[dt][0] *= alpha0;
            O_acc[dt][1] *= alpha0;
            O_acc[dt][2] *= alpha1;
            O_acc[dt][3] *= alpha1;
        }

        float p_sum0 = 0.0f, p_sum1 = 0.0f;
        uint32_t P_frag[4][4];
        #pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            int n0 = kb * 2;
            int n1 = n0 + 1;
            float p00 = (S[n0][0] == -INFINITY) ? 0.0f : exp2f(S[n0][0] - m0_new);
            float p01 = (S[n0][1] == -INFINITY) ? 0.0f : exp2f(S[n0][1] - m0_new);
            float p02 = (S[n0][2] == -INFINITY) ? 0.0f : exp2f(S[n0][2] - m1_new);
            float p03 = (S[n0][3] == -INFINITY) ? 0.0f : exp2f(S[n0][3] - m1_new);
            float p10 = (S[n1][0] == -INFINITY) ? 0.0f : exp2f(S[n1][0] - m0_new);
            float p11 = (S[n1][1] == -INFINITY) ? 0.0f : exp2f(S[n1][1] - m0_new);
            float p12 = (S[n1][2] == -INFINITY) ? 0.0f : exp2f(S[n1][2] - m1_new);
            float p13 = (S[n1][3] == -INFINITY) ? 0.0f : exp2f(S[n1][3] - m1_new);
            p_sum0 += (p00 + p01) + (p10 + p11);
            p_sum1 += (p02 + p03) + (p12 + p13);
            P_frag[kb][0] = pack_bf16_u32(p00, p01);
            P_frag[kb][1] = pack_bf16_u32(p02, p03);
            P_frag[kb][2] = pack_bf16_u32(p10, p11);
            P_frag[kb][3] = pack_bf16_u32(p12, p13);
        }
        l0 += quad_sum(p_sum0);
        l1 += quad_sum(p_sum1);
        m0 = m0_new;
        m1 = m1_new;

        // 3. O_acc[16, D] += P_frag[16, 64] @ V_smem[64, D]
        #pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            uint32_t a0 = P_frag[kb][0], a1 = P_frag[kb][1], a2 = P_frag[kb][2], a3 = P_frag[kb][3];
            #pragma unroll
            for (int db = 0; db < DT / 2; ++db) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_trans<D8>(b0_0, b1_0, b0_1, b1_1, s_V, kb * 16, db * 16, lane);
                int d0 = db * 2;
                int d1 = d0 + 1;
                mma_m16n8k16_bf16(
                    O_acc[d0][0], O_acc[d0][1], O_acc[d0][2], O_acc[d0][3],
                    a0, a1, a2, a3, b0_0, b1_0,
                    O_acc[d0][0], O_acc[d0][1], O_acc[d0][2], O_acc[d0][3]);
                mma_m16n8k16_bf16(
                    O_acc[d1][0], O_acc[d1][1], O_acc[d1][2], O_acc[d1][3],
                    a0, a1, a2, a3, b0_1, b1_1,
                    O_acc[d1][0], O_acc[d1][1], O_acc[d1][2], O_acc[d1][3]);
            }
        }
        __syncthreads();
    }

    // Normalize and write O_bf16 + LSE_f32 (in natural-log space)
    float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    if (r0 < static_cast<int>(t_q)) {
        __nv_bfloat16 *o_row = O_head + static_cast<int64_t>(r0) * so.st;
        #pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            uint32_t packed = pack_bf16_u32(O_acc[dt][0] * inv_l0, O_acc[dt][1] * inv_l0);
            *reinterpret_cast<uint32_t *>(o_row + dt * 8 + 2 * t) = packed;
        }
        if (LSE && t == 0) {
            LSE[bh * t_q + static_cast<uint64_t>(r0)] =
                (l0 > 0.0f) ? ((m0 + log2f(l0)) * LN2_F) : -INFINITY;
        }
    }
    if (r1 < static_cast<int>(t_q)) {
        __nv_bfloat16 *o_row = O_head + static_cast<int64_t>(r1) * so.st;
        #pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            uint32_t packed = pack_bf16_u32(O_acc[dt][2] * inv_l1, O_acc[dt][3] * inv_l1);
            *reinterpret_cast<uint32_t *>(o_row + dt * 8 + 2 * t) = packed;
        }
        if (LSE && t == 0) {
            LSE[bh * t_q + static_cast<uint64_t>(r1)] =
                (l1 > 0.0f) ? ((m1 + log2f(l1)) * LN2_F) : -INFINITY;
        }
    }
}

// ============================================================================
// BF16 Tensor Core FlashAttention-2 BACKWARD dQ (Q-parallel, ZERO atomics)
// ============================================================================
template<int D, bool CAUSAL>
__global__ void __launch_bounds__(128, 1) sdpa_bf16_tc_bwd_dq_kernel(
    uint64_t h_q,
    uint64_t h_kv,
    uint64_t t_q,
    uint64_t t_k,
    float scale,
    const __nv_bfloat16 *__restrict__ Q, Bf16SdpaStrides sq,
    const __nv_bfloat16 *__restrict__ K, Bf16SdpaStrides sk,
    const __nv_bfloat16 *__restrict__ V, Bf16SdpaStrides sv,
    const __nv_bfloat16 *__restrict__ dO, Bf16SdpaStrides sdo,
    const float *__restrict__ LSE,
    const float *__restrict__ D_row,
    __nv_bfloat16 *__restrict__ dQ, Bf16SdpaStrides sdq,
    int accumulate_dq)
{
    constexpr int D8 = D >> 3;
    constexpr int DT = D >> 3;
    constexpr int KT = D >> 4;

    extern __shared__ uint4 dyn_smem[];
    uint4 *s_Q  = dyn_smem;
    uint4 *s_dO = s_Q  + 64 * D8;
    uint4 *s_K  = s_dO + 64 * D8;
    uint4 *s_V  = s_K  + 64 * D8;

    int q_block = static_cast<int>(blockIdx.x) * 64;
    if (q_block >= static_cast<int>(t_q)) return;

    uint64_t bh = blockIdx.y;
    uint64_t b = bh / h_q;
    uint64_t hq = bh - b * h_q;
    uint64_t hkv = hq / (h_q / h_kv);

    const __nv_bfloat16 *Q_head  = Q  + static_cast<int64_t>(b) * sq.sb  + static_cast<int64_t>(hq)  * sq.sh;
    const __nv_bfloat16 *K_head  = K  + static_cast<int64_t>(b) * sk.sb  + static_cast<int64_t>(hkv) * sk.sh;
    const __nv_bfloat16 *V_head  = V  + static_cast<int64_t>(b) * sv.sb  + static_cast<int64_t>(hkv) * sv.sh;
    const __nv_bfloat16 *dO_head = dO + static_cast<int64_t>(b) * sdo.sb + static_cast<int64_t>(hq)  * sdo.sh;
    __nv_bfloat16 *dQ_head = dQ + static_cast<int64_t>(b) * sdq.sb + static_cast<int64_t>(hq) * sdq.sh;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    int r0 = q_block + warp * 16 + g;
    int r1 = r0 + 8;

    load_tile_64xD<D>(s_Q,  Q_head,  sq.st,  q_block, static_cast<int>(t_q));
    load_tile_64xD<D>(s_dO, dO_head, sdo.st, q_block, static_cast<int>(t_q));
    __syncthreads();

    float lse0 = (r0 < static_cast<int>(t_q)) ? (LSE[bh * t_q + static_cast<uint64_t>(r0)] * LOG2E_F) : 0.0f;
    float lse1 = (r1 < static_cast<int>(t_q)) ? (LSE[bh * t_q + static_cast<uint64_t>(r1)] * LOG2E_F) : 0.0f;
    float dr0  = (r0 < static_cast<int>(t_q)) ? D_row[bh * t_q + static_cast<uint64_t>(r0)] : 0.0f;
    float dr1  = (r1 < static_cast<int>(t_q)) ? D_row[bh * t_q + static_cast<uint64_t>(r1)] : 0.0f;
    const float scale_log2 = scale * LOG2E_F;

    float dQ_acc[DT][4];
    #pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
        dQ_acc[dt][0] = 0.0f; dQ_acc[dt][1] = 0.0f;
        dQ_acc[dt][2] = 0.0f; dQ_acc[dt][3] = 0.0f;
    }

    int k_limit = static_cast<int>(t_k);
    if constexpr (CAUSAL) {
        int max_k = q_block + 64;
        if (max_k < k_limit) k_limit = max_k;
    }

    for (int k_start = 0; k_start < k_limit; k_start += 64) {
        load_tile_64xD<D>(s_K, K_head, sk.st, k_start, static_cast<int>(t_k));
        load_tile_64xD<D>(s_V, V_head, sv.st, k_start, static_cast<int>(t_k));
        __syncthreads();

        // 1. S = Q @ K^T
        float S[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            S[nt][0] = 0.0f; S[nt][1] = 0.0f; S[nt][2] = 0.0f; S[nt][3] = 0.0f;
        }
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            uint32_t a0, a1, a2, a3;
            load_frag_A<D8>(a0, a1, a2, a3, s_Q, warp * 16, kt * 16, lane);
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_nt<D8>(b0_0, b1_0, b0_1, b1_1, s_K, nb * 16, kt * 16, lane);
                int n0 = nb * 2, n1 = n0 + 1;
                mma_m16n8k16_bf16(S[n0][0], S[n0][1], S[n0][2], S[n0][3], a0, a1, a2, a3, b0_0, b1_0, S[n0][0], S[n0][1], S[n0][2], S[n0][3]);
                mma_m16n8k16_bf16(S[n1][0], S[n1][1], S[n1][2], S[n1][3], a0, a1, a2, a3, b0_1, b1_1, S[n1][0], S[n1][1], S[n1][2], S[n1][3]);
            }
        }

        // Convert S -> P in registers (compact BF16 is NOT needed here, keep P in FP32 inside S!)
        // Actually compute dP = dO @ V^T into dP[8][4] while P stays in S? Wait: pack P into 16 u32 or keep FP32:
        // Keeping P in S[8][4] (32 regs) + dP[8][4] (32 regs) + dQ_acc (64 regs for D=128) = 128 regs -> fits cleanly in 255 regs!
        bool need_mask = (k_start + 64 > static_cast<int>(t_k)) ||
                         (CAUSAL && (k_start + 63 > q_block + warp * 16));
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            int c0 = k_start + nt * 8 + 2 * t;
            int c1 = c0 + 1;
            float p0 = (need_mask && (c0 >= static_cast<int>(t_k) || (CAUSAL && c0 > r0)))
                ? 0.0f : exp2f(S[nt][0] * scale_log2 - lse0);
            float p1 = (need_mask && (c1 >= static_cast<int>(t_k) || (CAUSAL && c1 > r0)))
                ? 0.0f : exp2f(S[nt][1] * scale_log2 - lse0);
            float p2 = (need_mask && (c0 >= static_cast<int>(t_k) || (CAUSAL && c0 > r1)))
                ? 0.0f : exp2f(S[nt][2] * scale_log2 - lse1);
            float p3 = (need_mask && (c1 >= static_cast<int>(t_k) || (CAUSAL && c1 > r1)))
                ? 0.0f : exp2f(S[nt][3] * scale_log2 - lse1);
            S[nt][0] = p0; S[nt][1] = p1; S[nt][2] = p2; S[nt][3] = p3;
        }

        // 2. dP = dO @ V^T
        float dP[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            dP[nt][0] = 0.0f; dP[nt][1] = 0.0f; dP[nt][2] = 0.0f; dP[nt][3] = 0.0f;
        }
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            uint32_t a0, a1, a2, a3;
            load_frag_A<D8>(a0, a1, a2, a3, s_dO, warp * 16, kt * 16, lane);
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_nt<D8>(b0_0, b1_0, b0_1, b1_1, s_V, nb * 16, kt * 16, lane);
                int n0 = nb * 2, n1 = n0 + 1;
                mma_m16n8k16_bf16(dP[n0][0], dP[n0][1], dP[n0][2], dP[n0][3], a0, a1, a2, a3, b0_0, b1_0, dP[n0][0], dP[n0][1], dP[n0][2], dP[n0][3]);
                mma_m16n8k16_bf16(dP[n1][0], dP[n1][1], dP[n1][2], dP[n1][3], a0, a1, a2, a3, b0_1, b1_1, dP[n1][0], dP[n1][1], dP[n1][2], dP[n1][3]);
            }
        }

        // 3. dS = P * (dP - D_row) * scale -> pack into A-fragments and accumulate dQ += dS @ K
        #pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            int n0 = kb * 2, n1 = n0 + 1;
            uint32_t a0 = pack_bf16_u32(S[n0][0] * (dP[n0][0] - dr0) * scale, S[n0][1] * (dP[n0][1] - dr0) * scale);
            uint32_t a1 = pack_bf16_u32(S[n0][2] * (dP[n0][2] - dr1) * scale, S[n0][3] * (dP[n0][3] - dr1) * scale);
            uint32_t a2 = pack_bf16_u32(S[n1][0] * (dP[n1][0] - dr0) * scale, S[n1][1] * (dP[n1][1] - dr0) * scale);
            uint32_t a3 = pack_bf16_u32(S[n1][2] * (dP[n1][2] - dr1) * scale, S[n1][3] * (dP[n1][3] - dr1) * scale);
            #pragma unroll
            for (int db = 0; db < DT / 2; ++db) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_trans<D8>(b0_0, b1_0, b0_1, b1_1, s_K, kb * 16, db * 16, lane);
                int d0 = db * 2, d1 = d0 + 1;
                mma_m16n8k16_bf16(dQ_acc[d0][0], dQ_acc[d0][1], dQ_acc[d0][2], dQ_acc[d0][3], a0, a1, a2, a3, b0_0, b1_0, dQ_acc[d0][0], dQ_acc[d0][1], dQ_acc[d0][2], dQ_acc[d0][3]);
                mma_m16n8k16_bf16(dQ_acc[d1][0], dQ_acc[d1][1], dQ_acc[d1][2], dQ_acc[d1][3], a0, a1, a2, a3, b0_1, b1_1, dQ_acc[d1][0], dQ_acc[d1][1], dQ_acc[d1][2], dQ_acc[d1][3]);
            }
        }
        __syncthreads();
    }

    if (r0 < static_cast<int>(t_q)) {
        __nv_bfloat16 *row = dQ_head + static_cast<int64_t>(r0) * sdq.st;
        #pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            float v0 = dQ_acc[dt][0], v1 = dQ_acc[dt][1];
            uint32_t *dst = reinterpret_cast<uint32_t *>(row + dt * 8 + 2 * t);
            if (accumulate_dq) {
                float2 prev = unpack_bf16_u32(*dst);
                v0 += prev.x; v1 += prev.y;
            }
            *dst = pack_bf16_u32(v0, v1);
        }
    }
    if (r1 < static_cast<int>(t_q)) {
        __nv_bfloat16 *row = dQ_head + static_cast<int64_t>(r1) * sdq.st;
        #pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            float v0 = dQ_acc[dt][2], v1 = dQ_acc[dt][3];
            uint32_t *dst = reinterpret_cast<uint32_t *>(row + dt * 8 + 2 * t);
            if (accumulate_dq) {
                float2 prev = unpack_bf16_u32(*dst);
                v0 += prev.x; v1 += prev.y;
            }
            *dst = pack_bf16_u32(v0, v1);
        }
    }
}

// ============================================================================
// BF16 Tensor Core FlashAttention-2 BACKWARD dK + dV (KV-parallel, ZERO atomics!)
// Computes S^T = K@Q^T, P^T -> dV += P^T@dO, dP^T = V@dO^T -> dK += dS^T@Q
// Zero SMEM round-trip for S^T, P^T, dP^T, dS^T!
// ============================================================================
template<int D, bool CAUSAL, bool WRITE_F32_PARTIAL>
__global__ void __launch_bounds__(128, 1) sdpa_bf16_tc_bwd_dkdv_kernel(
    uint64_t h_q,
    uint64_t h_kv,
    uint64_t t_q,
    uint64_t t_k,
    float scale,
    const __nv_bfloat16 *__restrict__ Q, Bf16SdpaStrides sq,
    const __nv_bfloat16 *__restrict__ K, Bf16SdpaStrides sk,
    const __nv_bfloat16 *__restrict__ V, Bf16SdpaStrides sv,
    const __nv_bfloat16 *__restrict__ dO, Bf16SdpaStrides sdo,
    const float *__restrict__ LSE,
    const float *__restrict__ D_row,
    __nv_bfloat16 *__restrict__ dK_bf16, Bf16SdpaStrides sdk,
    __nv_bfloat16 *__restrict__ dV_bf16, Bf16SdpaStrides sdv,
    float *__restrict__ gqa_dk_f32,
    float *__restrict__ gqa_dv_f32,
    int accumulate_dk,
    int accumulate_dv)
{
    constexpr int D8 = D >> 3;
    constexpr int DT = D >> 3;
    constexpr int KT = D >> 4;

    extern __shared__ uint4 dyn_smem[];
    uint4 *s_K  = dyn_smem;
    uint4 *s_V  = s_K  + 64 * D8;
    uint4 *s_Q  = s_V  + 64 * D8;
    uint4 *s_dO = s_Q  + 64 * D8;
    float *s_lse  = reinterpret_cast<float *>(s_dO + 64 * D8);
    float *s_drow = s_lse + 64;

    int k_block = static_cast<int>(blockIdx.x) * 64;
    if (k_block >= static_cast<int>(t_k)) return;

    uint64_t bh = blockIdx.y; // 0 .. B*H_q - 1
    uint64_t b = bh / h_q;
    uint64_t hq = bh - b * h_q;
    uint64_t hkv = hq / (h_q / h_kv);

    const __nv_bfloat16 *Q_head  = Q  + static_cast<int64_t>(b) * sq.sb  + static_cast<int64_t>(hq)  * sq.sh;
    const __nv_bfloat16 *K_head  = K  + static_cast<int64_t>(b) * sk.sb  + static_cast<int64_t>(hkv) * sk.sh;
    const __nv_bfloat16 *V_head  = V  + static_cast<int64_t>(b) * sv.sb  + static_cast<int64_t>(hkv) * sv.sh;
    const __nv_bfloat16 *dO_head = dO + static_cast<int64_t>(b) * sdo.sb + static_cast<int64_t>(hq)  * sdo.sh;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    int k0 = k_block + warp * 16 + g;
    int k1 = k0 + 8;

    load_tile_64xD<D>(s_K, K_head, sk.st, k_block, static_cast<int>(t_k));
    load_tile_64xD<D>(s_V, V_head, sv.st, k_block, static_cast<int>(t_k));
    __syncthreads();

    float dK_acc[DT][4];
    float dV_acc[DT][4];
    #pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
        dK_acc[dt][0] = 0.0f; dK_acc[dt][1] = 0.0f; dK_acc[dt][2] = 0.0f; dK_acc[dt][3] = 0.0f;
        dV_acc[dt][0] = 0.0f; dV_acc[dt][1] = 0.0f; dV_acc[dt][2] = 0.0f; dV_acc[dt][3] = 0.0f;
    }

    const float scale_log2 = scale * LOG2E_F;
    int q_begin = CAUSAL ? k_block : 0;

    for (int q_start = q_begin; q_start < static_cast<int>(t_q); q_start += 64) {
        load_tile_64xD<D>(s_Q,  Q_head,  sq.st,  q_start, static_cast<int>(t_q));
        load_tile_64xD<D>(s_dO, dO_head, sdo.st, q_start, static_cast<int>(t_q));
        if (threadIdx.x < 64) {
            int qr = q_start + threadIdx.x;
            s_lse[threadIdx.x]  = (qr < static_cast<int>(t_q)) ? (LSE[bh * t_q + static_cast<uint64_t>(qr)] * LOG2E_F) : 0.0f;
            s_drow[threadIdx.x] = (qr < static_cast<int>(t_q)) ? D_row[bh * t_q + static_cast<uint64_t>(qr)] : 0.0f;
        }
        __syncthreads();

        // 1. S^T[16, 64] = K[16, D] @ Q[64, D]^T
        float ST[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            ST[nt][0] = 0.0f; ST[nt][1] = 0.0f; ST[nt][2] = 0.0f; ST[nt][3] = 0.0f;
        }
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            uint32_t a0, a1, a2, a3;
            load_frag_A<D8>(a0, a1, a2, a3, s_K, warp * 16, kt * 16, lane);
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_nt<D8>(b0_0, b1_0, b0_1, b1_1, s_Q, nb * 16, kt * 16, lane);
                int n0 = nb * 2, n1 = n0 + 1;
                mma_m16n8k16_bf16(ST[n0][0], ST[n0][1], ST[n0][2], ST[n0][3], a0, a1, a2, a3, b0_0, b1_0, ST[n0][0], ST[n0][1], ST[n0][2], ST[n0][3]);
                mma_m16n8k16_bf16(ST[n1][0], ST[n1][1], ST[n1][2], ST[n1][3], a0, a1, a2, a3, b0_1, b1_1, ST[n1][0], ST[n1][1], ST[n1][2], ST[n1][3]);
            }
        }

        // 2. Compute P^T and pack directly into 16 u32 registers (PT_frag) for BOTH dV and dS^T!
        bool need_mask = (q_start + 64 > static_cast<int>(t_q)) ||
                         (CAUSAL && (q_start < k_block + warp * 16 + 16));
        uint32_t PT_frag[4][4];
        #pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            int n0 = kb * 2, n1 = n0 + 1;
            int q00 = q_start + n0 * 8 + 2 * t;
            int q01 = q00 + 1;
            int q10 = q_start + n1 * 8 + 2 * t;
            int q11 = q10 + 1;
            float lse00 = s_lse[n0 * 8 + 2 * t];
            float lse01 = s_lse[n0 * 8 + 2 * t + 1];
            float lse10 = s_lse[n1 * 8 + 2 * t];
            float lse11 = s_lse[n1 * 8 + 2 * t + 1];

            float p00 = (need_mask && (q00 >= static_cast<int>(t_q) || (CAUSAL && k0 > q00))) ? 0.0f : exp2f(ST[n0][0] * scale_log2 - lse00);
            float p01 = (need_mask && (q01 >= static_cast<int>(t_q) || (CAUSAL && k0 > q01))) ? 0.0f : exp2f(ST[n0][1] * scale_log2 - lse01);
            float p02 = (need_mask && (q00 >= static_cast<int>(t_q) || (CAUSAL && k1 > q00))) ? 0.0f : exp2f(ST[n0][2] * scale_log2 - lse00);
            float p03 = (need_mask && (q01 >= static_cast<int>(t_q) || (CAUSAL && k1 > q01))) ? 0.0f : exp2f(ST[n0][3] * scale_log2 - lse01);
            float p10 = (need_mask && (q10 >= static_cast<int>(t_q) || (CAUSAL && k0 > q10))) ? 0.0f : exp2f(ST[n1][0] * scale_log2 - lse10);
            float p11 = (need_mask && (q11 >= static_cast<int>(t_q) || (CAUSAL && k0 > q11))) ? 0.0f : exp2f(ST[n1][1] * scale_log2 - lse11);
            float p12 = (need_mask && (q10 >= static_cast<int>(t_q) || (CAUSAL && k1 > q10))) ? 0.0f : exp2f(ST[n1][2] * scale_log2 - lse10);
            float p13 = (need_mask && (q11 >= static_cast<int>(t_q) || (CAUSAL && k1 > q11))) ? 0.0f : exp2f(ST[n1][3] * scale_log2 - lse11);

            PT_frag[kb][0] = pack_bf16_u32(p00, p01);
            PT_frag[kb][1] = pack_bf16_u32(p02, p03);
            PT_frag[kb][2] = pack_bf16_u32(p10, p11);
            PT_frag[kb][3] = pack_bf16_u32(p12, p13);
        }

        // 3. dV_acc[16, D] += P^T[16, 64] @ dO[64, D]
        #pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            uint32_t a0 = PT_frag[kb][0], a1 = PT_frag[kb][1], a2 = PT_frag[kb][2], a3 = PT_frag[kb][3];
            #pragma unroll
            for (int db = 0; db < DT / 2; ++db) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_trans<D8>(b0_0, b1_0, b0_1, b1_1, s_dO, kb * 16, db * 16, lane);
                int d0 = db * 2, d1 = d0 + 1;
                mma_m16n8k16_bf16(dV_acc[d0][0], dV_acc[d0][1], dV_acc[d0][2], dV_acc[d0][3], a0, a1, a2, a3, b0_0, b1_0, dV_acc[d0][0], dV_acc[d0][1], dV_acc[d0][2], dV_acc[d0][3]);
                mma_m16n8k16_bf16(dV_acc[d1][0], dV_acc[d1][1], dV_acc[d1][2], dV_acc[d1][3], a0, a1, a2, a3, b0_1, b1_1, dV_acc[d1][0], dV_acc[d1][1], dV_acc[d1][2], dV_acc[d1][3]);
            }
        }

        // 4. dP^T[16, 64] = V[16, D] @ dO[64, D]^T (reuse ST register buffer!)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            ST[nt][0] = 0.0f; ST[nt][1] = 0.0f; ST[nt][2] = 0.0f; ST[nt][3] = 0.0f;
        }
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            uint32_t a0, a1, a2, a3;
            load_frag_A<D8>(a0, a1, a2, a3, s_V, warp * 16, kt * 16, lane);
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_nt<D8>(b0_0, b1_0, b0_1, b1_1, s_dO, nb * 16, kt * 16, lane);
                int n0 = nb * 2, n1 = n0 + 1;
                mma_m16n8k16_bf16(ST[n0][0], ST[n0][1], ST[n0][2], ST[n0][3], a0, a1, a2, a3, b0_0, b1_0, ST[n0][0], ST[n0][1], ST[n0][2], ST[n0][3]);
                mma_m16n8k16_bf16(ST[n1][0], ST[n1][1], ST[n1][2], ST[n1][3], a0, a1, a2, a3, b0_1, b1_1, ST[n1][0], ST[n1][1], ST[n1][2], ST[n1][3]);
            }
        }

        // 5. dS^T = P^T * (dP^T - D_row) * scale -> dK_acc += dS^T @ Q
        #pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            int n0 = kb * 2, n1 = n0 + 1;
            float2 p0_lo = unpack_bf16_u32(PT_frag[kb][0]);
            float2 p0_hi = unpack_bf16_u32(PT_frag[kb][1]);
            float2 p1_lo = unpack_bf16_u32(PT_frag[kb][2]);
            float2 p1_hi = unpack_bf16_u32(PT_frag[kb][3]);
            float dr00 = s_drow[n0 * 8 + 2 * t];
            float dr01 = s_drow[n0 * 8 + 2 * t + 1];
            float dr10 = s_drow[n1 * 8 + 2 * t];
            float dr11 = s_drow[n1 * 8 + 2 * t + 1];

            uint32_t a0 = pack_bf16_u32(p0_lo.x * (ST[n0][0] - dr00) * scale, p0_lo.y * (ST[n0][1] - dr01) * scale);
            uint32_t a1 = pack_bf16_u32(p0_hi.x * (ST[n0][2] - dr00) * scale, p0_hi.y * (ST[n0][3] - dr01) * scale);
            uint32_t a2 = pack_bf16_u32(p1_lo.x * (ST[n1][0] - dr10) * scale, p1_lo.y * (ST[n1][1] - dr11) * scale);
            uint32_t a3 = pack_bf16_u32(p1_hi.x * (ST[n1][2] - dr10) * scale, p1_hi.y * (ST[n1][3] - dr11) * scale);

            #pragma unroll
            for (int db = 0; db < DT / 2; ++db) {
                uint32_t b0_0, b1_0, b0_1, b1_1;
                load_frag_B_trans<D8>(b0_0, b1_0, b0_1, b1_1, s_Q, kb * 16, db * 16, lane);
                int d0 = db * 2, d1 = d0 + 1;
                mma_m16n8k16_bf16(dK_acc[d0][0], dK_acc[d0][1], dK_acc[d0][2], dK_acc[d0][3], a0, a1, a2, a3, b0_0, b1_0, dK_acc[d0][0], dK_acc[d0][1], dK_acc[d0][2], dK_acc[d0][3]);
                mma_m16n8k16_bf16(dK_acc[d1][0], dK_acc[d1][1], dK_acc[d1][2], dK_acc[d1][3], a0, a1, a2, a3, b0_1, b1_1, dK_acc[d1][0], dK_acc[d1][1], dK_acc[d1][2], dK_acc[d1][3]);
            }
        }
        __syncthreads();
    }

    if constexpr (WRITE_F32_PARTIAL) {
        float *dk_base = gqa_dk_f32 + bh * t_k * D;
        float *dv_base = gqa_dv_f32 + bh * t_k * D;
        if (k0 < static_cast<int>(t_k)) {
            float *dk_r = dk_base + static_cast<uint64_t>(k0) * D;
            float *dv_r = dv_base + static_cast<uint64_t>(k0) * D;
            #pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                *reinterpret_cast<float2 *>(dk_r + dt * 8 + 2 * t) = make_float2(dK_acc[dt][0], dK_acc[dt][1]);
                *reinterpret_cast<float2 *>(dv_r + dt * 8 + 2 * t) = make_float2(dV_acc[dt][0], dV_acc[dt][1]);
            }
        }
        if (k1 < static_cast<int>(t_k)) {
            float *dk_r = dk_base + static_cast<uint64_t>(k1) * D;
            float *dv_r = dv_base + static_cast<uint64_t>(k1) * D;
            #pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                *reinterpret_cast<float2 *>(dk_r + dt * 8 + 2 * t) = make_float2(dK_acc[dt][2], dK_acc[dt][3]);
                *reinterpret_cast<float2 *>(dv_r + dt * 8 + 2 * t) = make_float2(dV_acc[dt][2], dV_acc[dt][3]);
            }
        }
    } else {
        __nv_bfloat16 *dK_head = dK_bf16 + static_cast<int64_t>(b) * sdk.sb + static_cast<int64_t>(hkv) * sdk.sh;
        __nv_bfloat16 *dV_head = dV_bf16 + static_cast<int64_t>(b) * sdv.sb + static_cast<int64_t>(hkv) * sdv.sh;
        if (k0 < static_cast<int>(t_k)) {
            __nv_bfloat16 *kr = dK_head + static_cast<int64_t>(k0) * sdk.st;
            __nv_bfloat16 *vr = dV_head + static_cast<int64_t>(k0) * sdv.st;
            #pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                float kv0 = dK_acc[dt][0], kv1 = dK_acc[dt][1];
                float vv0 = dV_acc[dt][0], vv1 = dV_acc[dt][1];
                uint32_t *kdst = reinterpret_cast<uint32_t *>(kr + dt * 8 + 2 * t);
                uint32_t *vdst = reinterpret_cast<uint32_t *>(vr + dt * 8 + 2 * t);
                if (accumulate_dk) { float2 p = unpack_bf16_u32(*kdst); kv0 += p.x; kv1 += p.y; }
                if (accumulate_dv) { float2 p = unpack_bf16_u32(*vdst); vv0 += p.x; vv1 += p.y; }
                *kdst = pack_bf16_u32(kv0, kv1);
                *vdst = pack_bf16_u32(vv0, vv1);
            }
        }
        if (k1 < static_cast<int>(t_k)) {
            __nv_bfloat16 *kr = dK_head + static_cast<int64_t>(k1) * sdk.st;
            __nv_bfloat16 *vr = dV_head + static_cast<int64_t>(k1) * sdv.st;
            #pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                float kv0 = dK_acc[dt][2], kv1 = dK_acc[dt][3];
                float vv0 = dV_acc[dt][2], vv1 = dV_acc[dt][3];
                uint32_t *kdst = reinterpret_cast<uint32_t *>(kr + dt * 8 + 2 * t);
                uint32_t *vdst = reinterpret_cast<uint32_t *>(vr + dt * 8 + 2 * t);
                if (accumulate_dk) { float2 p = unpack_bf16_u32(*kdst); kv0 += p.x; kv1 += p.y; }
                if (accumulate_dv) { float2 p = unpack_bf16_u32(*vdst); vv0 += p.x; vv1 += p.y; }
                *kdst = pack_bf16_u32(kv0, kv1);
                *vdst = pack_bf16_u32(vv0, vv1);
            }
        }
    }
}

// --- Backward prep: D_row[b, h, q] = sum_d O[q, d] * dO[q, d] (1 warp per row) ---

__global__ void sdpa_bf16_bwd_prep_kernel(
    uint64_t h_q, uint64_t t_q, uint64_t d,
    const __nv_bfloat16 *__restrict__ O, Bf16SdpaStrides so,
    const __nv_bfloat16 *__restrict__ dO, Bf16SdpaStrides sdo,
    float *__restrict__ D_row)
{
    uint64_t row_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    uint64_t bh = blockIdx.y;
    uint64_t q_idx = row_id;
    if (q_idx >= t_q) return;
    uint64_t b = bh / h_q;
    uint64_t h = bh - b * h_q;
    const __nv_bfloat16 *o_r  = O  + static_cast<int64_t>(b) * so.sb  + static_cast<int64_t>(h) * so.sh  + static_cast<int64_t>(q_idx) * so.st;
    const __nv_bfloat16 *do_r = dO + static_cast<int64_t>(b) * sdo.sb + static_cast<int64_t>(h) * sdo.sh + static_cast<int64_t>(q_idx) * sdo.st;
    unsigned lane = threadIdx.x & 31u;
    float acc = 0.0f;
    for (uint64_t i = lane; i < d; i += 32) {
        acc += __bfloat162float(o_r[static_cast<int64_t>(i) * so.sd]) * __bfloat162float(do_r[static_cast<int64_t>(i) * sdo.sd]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        D_row[bh * t_q + q_idx] = acc;
    }
}

// --- Deterministic GQA reduction from [B, H_q, T_k, D] FP32 -> [B, H_kv, T_k, D] BF16 ---

__global__ void sdpa_bf16_gqa_reduce_kernel(
    uint64_t batch, uint64_t h_q, uint64_t h_kv, uint64_t t_k, uint64_t d,
    const float *__restrict__ src_f32,
    __nv_bfloat16 *__restrict__ dst_bf16, Bf16SdpaStrides sdst,
    int accumulate)
{
    uint64_t gqa = h_q / h_kv;
    uint64_t total = batch * h_kv * t_k * d;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        uint64_t di = idx % d;
        uint64_t tmp = idx / d;
        uint64_t ti = tmp % t_k;
        tmp /= t_k;
        uint64_t hkv = tmp % h_kv;
        uint64_t bi = tmp / h_kv;
        float acc = 0.0f;
        uint64_t hq_base = hkv * gqa;
        for (uint64_t g = 0; g < gqa; ++g) {
            uint64_t bh = bi * h_q + (hq_base + g);
            acc += src_f32[(bh * t_k + ti) * d + di];
        }
        __nv_bfloat16 *dst = dst_bf16 + static_cast<int64_t>(bi) * sdst.sb
                                      + static_cast<int64_t>(hkv) * sdst.sh
                                      + static_cast<int64_t>(ti) * sdst.st
                                      + static_cast<int64_t>(di) * sdst.sd;
        if (accumulate) acc += __bfloat162float(*dst);
        *dst = __float2bfloat16_rn(acc);
    }
}

// General signed-stride fallback. Each warp owns one output row, so the dot
// products are shared by all its output lanes. Wide heads repeat one dot per
// channel tile, with bounded register use and no score matrix.
constexpr unsigned FALLBACK_ROWS_PER_BLOCK = 4;
constexpr unsigned FALLBACK_CHANNELS_PER_LANE = 4;
constexpr unsigned FALLBACK_TILE = 32 * FALLBACK_CHANNELS_PER_LANE;

template<class T>
__device__ __forceinline__ T *strided_row(
    T *ptr, Bf16SdpaStrides s, uint64_t b, uint64_t h, uint64_t t)
{
    return ptr + (static_cast<int64_t>(b) * s.sb + static_cast<int64_t>(h) * s.sh
                  + static_cast<int64_t>(t) * s.st);
}

__device__ __forceinline__ float load_channel(
    const __nv_bfloat16 *row, uint64_t col, int64_t stride)
{
    return __bfloat162float(row[static_cast<int64_t>(col) * stride]);
}

__device__ __forceinline__ float fallback_dot(
    const __nv_bfloat16 *x, int64_t sx, const __nv_bfloat16 *y, int64_t sy,
    uint64_t d, unsigned lane)
{
    float sum = 0.0f;
    for (uint64_t col = lane; col < d; col += 32) {
        sum += load_channel(x, col, sx) * load_channel(y, col, sy);
    }
    for (int shift = 16; shift > 0; shift >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, shift);
    }
    return __shfl_sync(0xffffffffu, sum, 0);
}

__global__ void sdpa_bf16_generic_fwd_kernel(
    uint64_t h_q, uint64_t h_kv, uint64_t t_q, uint64_t t_k, uint64_t d,
    float scale, int is_causal,
    const __nv_bfloat16 *__restrict__ Q, Bf16SdpaStrides sq,
    const __nv_bfloat16 *__restrict__ K, Bf16SdpaStrides sk,
    const __nv_bfloat16 *__restrict__ V, Bf16SdpaStrides sv,
    __nv_bfloat16 *__restrict__ O, Bf16SdpaStrides so,
    float *__restrict__ LSE)
{
    const unsigned lane = threadIdx.x & 31u;
    const uint64_t qi = static_cast<uint64_t>(blockIdx.x) * FALLBACK_ROWS_PER_BLOCK
                        + (threadIdx.x >> 5);
    if (qi >= t_q) return;
    const uint64_t bh = blockIdx.y;
    const uint64_t b = bh / h_q;
    const uint64_t hq = bh % h_q;
    const uint64_t hkv = hq / (h_q / h_kv);
    const auto *q = strided_row(Q, sq, b, hq, qi);
    auto *out = strided_row(O, so, b, hq, qi);
    const uint64_t key_end = is_causal ? umin64(t_k, qi + 1) : t_k;

    for (uint64_t tile = 0; tile < d; tile += FALLBACK_TILE) {
        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float sums[FALLBACK_CHANNELS_PER_LANE] = {};
        for (uint64_t ki = 0; ki < key_end; ++ki) {
            const auto *k = strided_row(K, sk, b, hkv, ki);
            const auto *v = strided_row(V, sv, b, hkv, ki);
            const float score = scale * fallback_dot(q, sq.sd, k, sk.sd, d, lane);
            const float next_max = fmaxf(max_score, score);
            const float alpha = max_score == -INFINITY ? 0.0f : expf(max_score - next_max);
            const float beta = expf(score - next_max);
            sum_exp = alpha * sum_exp + beta;
            max_score = next_max;
            #pragma unroll
            for (unsigned i = 0; i < FALLBACK_CHANNELS_PER_LANE; ++i) {
                const uint64_t col = tile + i * 32 + lane;
                if (col < d) sums[i] = alpha * sums[i] + beta * load_channel(v, col, sv.sd);
            }
        }
        if (LSE && tile == 0 && lane == 0) {
            LSE[bh * t_q + qi] = max_score + logf(sum_exp);
        }
        const float inv_sum = 1.0f / sum_exp;
        #pragma unroll
        for (unsigned i = 0; i < FALLBACK_CHANNELS_PER_LANE; ++i) {
            const uint64_t col = tile + i * 32 + lane;
            if (col < d) {
                out[static_cast<int64_t>(col) * so.sd] = __float2bfloat16_rn(sums[i] * inv_sum);
            }
        }
    }
}

__global__ void sdpa_bf16_generic_bwd_dq_kernel(
    uint64_t h_q, uint64_t h_kv, uint64_t t_q, uint64_t t_k, uint64_t d,
    float scale, int is_causal,
    const __nv_bfloat16 *__restrict__ Q, Bf16SdpaStrides sq,
    const __nv_bfloat16 *__restrict__ K, Bf16SdpaStrides sk,
    const __nv_bfloat16 *__restrict__ V, Bf16SdpaStrides sv,
    const __nv_bfloat16 *__restrict__ O, Bf16SdpaStrides so,
    const __nv_bfloat16 *__restrict__ dO, Bf16SdpaStrides sdo,
    const float *__restrict__ LSE, float *__restrict__ D_row,
    __nv_bfloat16 *__restrict__ dQ, Bf16SdpaStrides sdq, int accumulate_dq)
{
    const unsigned lane = threadIdx.x & 31u;
    const uint64_t qi = static_cast<uint64_t>(blockIdx.x) * FALLBACK_ROWS_PER_BLOCK
                        + (threadIdx.x >> 5);
    if (qi >= t_q) return;
    const uint64_t bh = blockIdx.y;
    const uint64_t b = bh / h_q;
    const uint64_t hq = bh % h_q;
    const uint64_t hkv = hq / (h_q / h_kv);
    const auto *q = strided_row(Q, sq, b, hq, qi);
    const auto *out = strided_row(O, so, b, hq, qi);
    const auto *dout = strided_row(dO, sdo, b, hq, qi);
    auto *dq = strided_row(dQ, sdq, b, hq, qi);
    const uint64_t key_end = is_causal ? umin64(t_k, qi + 1) : t_k;
    const float lse = LSE[bh * t_q + qi];
    const float row_delta = fallback_dot(out, so.sd, dout, sdo.sd, d, lane);
    if (lane == 0) D_row[bh * t_q + qi] = row_delta;

    for (uint64_t tile = 0; tile < d; tile += FALLBACK_TILE) {
        float sums[FALLBACK_CHANNELS_PER_LANE] = {};
        for (uint64_t ki = 0; ki < key_end; ++ki) {
            const auto *k = strided_row(K, sk, b, hkv, ki);
            const auto *v = strided_row(V, sv, b, hkv, ki);
            const float s = fallback_dot(q, sq.sd, k, sk.sd, d, lane);
            const float dp = fallback_dot(dout, sdo.sd, v, sv.sd, d, lane);
            const float ds = expf(s * scale - lse) * (dp - row_delta) * scale;
            #pragma unroll
            for (unsigned i = 0; i < FALLBACK_CHANNELS_PER_LANE; ++i) {
                const uint64_t col = tile + i * 32 + lane;
                if (col < d) sums[i] += ds * load_channel(k, col, sk.sd);
            }
        }
        #pragma unroll
        for (unsigned i = 0; i < FALLBACK_CHANNELS_PER_LANE; ++i) {
            const uint64_t col = tile + i * 32 + lane;
            if (col < d) {
                float value = sums[i];
                if (accumulate_dq) value += load_channel(dq, col, sdq.sd);
                dq[static_cast<int64_t>(col) * sdq.sd] = __float2bfloat16_rn(value);
            }
        }
    }
}

__global__ void sdpa_bf16_generic_bwd_dkdv_kernel(
    uint64_t h_q, uint64_t h_kv, uint64_t t_q, uint64_t t_k, uint64_t d,
    float scale, int is_causal,
    const __nv_bfloat16 *__restrict__ Q, Bf16SdpaStrides sq,
    const __nv_bfloat16 *__restrict__ K, Bf16SdpaStrides sk,
    const __nv_bfloat16 *__restrict__ V, Bf16SdpaStrides sv,
    const __nv_bfloat16 *__restrict__ dO, Bf16SdpaStrides sdo,
    const float *__restrict__ LSE, const float *__restrict__ D_row,
    __nv_bfloat16 *__restrict__ dK, Bf16SdpaStrides sdk,
    __nv_bfloat16 *__restrict__ dV, Bf16SdpaStrides sdv,
    int accumulate_dk, int accumulate_dv)
{
    const unsigned lane = threadIdx.x & 31u;
    const uint64_t ki = static_cast<uint64_t>(blockIdx.x) * FALLBACK_ROWS_PER_BLOCK
                        + (threadIdx.x >> 5);
    if (ki >= t_k) return;
    const uint64_t b = blockIdx.y / h_kv;
    const uint64_t hkv = blockIdx.y % h_kv;
    const uint64_t group = h_q / h_kv;
    const auto *k = strided_row(K, sk, b, hkv, ki);
    const auto *v = strided_row(V, sv, b, hkv, ki);
    auto *dk = strided_row(dK, sdk, b, hkv, ki);
    auto *dv = strided_row(dV, sdv, b, hkv, ki);
    const uint64_t query_start = is_causal ? ki : 0;

    for (uint64_t tile = 0; tile < d; tile += FALLBACK_TILE) {
        float sum_k[FALLBACK_CHANNELS_PER_LANE] = {};
        float sum_v[FALLBACK_CHANNELS_PER_LANE] = {};
        for (uint64_t hq = hkv * group; hq < (hkv + 1) * group; ++hq) {
            const uint64_t bh = b * h_q + hq;
            for (uint64_t qi = query_start; qi < t_q; ++qi) {
                const auto *q = strided_row(Q, sq, b, hq, qi);
                const auto *dout = strided_row(dO, sdo, b, hq, qi);
                const float s = fallback_dot(q, sq.sd, k, sk.sd, d, lane);
                const float dp = fallback_dot(dout, sdo.sd, v, sv.sd, d, lane);
                const float p = expf(s * scale - LSE[bh * t_q + qi]);
                const float ds = p * (dp - D_row[bh * t_q + qi]) * scale;
                #pragma unroll
                for (unsigned i = 0; i < FALLBACK_CHANNELS_PER_LANE; ++i) {
                    const uint64_t col = tile + i * 32 + lane;
                    if (col < d) {
                        sum_k[i] += ds * load_channel(q, col, sq.sd);
                        sum_v[i] += p * load_channel(dout, col, sdo.sd);
                    }
                }
            }
        }
        #pragma unroll
        for (unsigned i = 0; i < FALLBACK_CHANNELS_PER_LANE; ++i) {
            const uint64_t col = tile + i * 32 + lane;
            if (col < d) {
                float value_k = sum_k[i], value_v = sum_v[i];
                if (accumulate_dk) value_k += load_channel(dk, col, sdk.sd);
                if (accumulate_dv) value_v += load_channel(dv, col, sdv.sd);
                dk[static_cast<int64_t>(col) * sdk.sd] = __float2bfloat16_rn(value_k);
                dv[static_cast<int64_t>(col) * sdv.sd] = __float2bfloat16_rn(value_v);
            }
        }
    }
}

static bool stride_aligned8(const void *ptr, Bf16SdpaStrides s) {
    return s.sd == 1 &&
           ((reinterpret_cast<uintptr_t>(ptr) & 15ULL) == 0ULL) &&
           ((s.sb & 7LL) == 0LL) &&
           ((s.sh & 7LL) == 0LL) &&
           ((s.st & 7LL) == 0LL);
}

static int ensure_smem(const void *func, size_t bytes) {
    if (bytes <= 48 * 1024) return 0;
    return rt(cudaFuncSetAttribute(
        func, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
}

} // namespace

extern "C" int fa_cuda_sdpa_bf16_prepare(Context *ctx) {
    if (!ctx) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    if (ctx->cc_major < 8) return 0;
    const size_t dq128 = 4ULL * 64ULL * 128ULL * sizeof(uint16_t);
    const size_t kv128 = dq128 + 128ULL * sizeof(float);
    if ((code = ensure_smem((const void *)sdpa_bf16_tc_bwd_dq_kernel<128, true>, dq128))) return code;
    if ((code = ensure_smem((const void *)sdpa_bf16_tc_bwd_dq_kernel<128, false>, dq128))) return code;
    if ((code = ensure_smem((const void *)sdpa_bf16_tc_bwd_dkdv_kernel<128, true, true>, kv128))) return code;
    if ((code = ensure_smem((const void *)sdpa_bf16_tc_bwd_dkdv_kernel<128, true, false>, kv128))) return code;
    if ((code = ensure_smem((const void *)sdpa_bf16_tc_bwd_dkdv_kernel<128, false, true>, kv128))) return code;
    if ((code = ensure_smem((const void *)sdpa_bf16_tc_bwd_dkdv_kernel<128, false, false>, kv128))) return code;
    return 0;
}

extern "C" int fa_cuda_sdpa_bf16_fwd(
    Context *ctx,
    uint64_t batch, uint64_t h_q, uint64_t h_kv, uint64_t t_q, uint64_t t_k, uint64_t d,
    float scale, int is_causal,
    const uint16_t *q, Bf16SdpaStrides sq,
    const uint16_t *k, Bf16SdpaStrides sk,
    const uint16_t *v, Bf16SdpaStrides sv,
    uint16_t *out, Bf16SdpaStrides so,
    float *lse_f32_opt)
{
    if (!ctx || h_q == 0 || h_kv == 0 || (h_q % h_kv) != 0 || d == 0) {
        return rt(cudaErrorInvalidValue);
    }
    if (batch == 0 || t_q == 0) return 0;
    if (!q || !k || !v || !out) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;

    const __nv_bfloat16 *qp = reinterpret_cast<const __nv_bfloat16 *>(q);
    const __nv_bfloat16 *kp = reinterpret_cast<const __nv_bfloat16 *>(k);
    const __nv_bfloat16 *vp = reinterpret_cast<const __nv_bfloat16 *>(v);
    __nv_bfloat16 *op = reinterpret_cast<__nv_bfloat16 *>(out);

    bool tc_ok = (ctx->cc_major >= 8) && (d == 64 || d == 128) && (t_k > 0) &&
                 stride_aligned8(q, sq) && stride_aligned8(k, sk) &&
                 stride_aligned8(v, sv) && stride_aligned8(out, so);

    if (tc_ok) {
        dim3 grid(static_cast<unsigned>((t_q + 63ULL) / 64ULL), static_cast<unsigned>(batch * h_q));
        if (d == 64) {
            if (is_causal) {
                sdpa_bf16_tc_fwd_kernel<64, true><<<grid, 128, 0, ctx->stream>>>(
                    h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, op, so, lse_f32_opt);
            } else {
                sdpa_bf16_tc_fwd_kernel<64, false><<<grid, 128, 0, ctx->stream>>>(
                    h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, op, so, lse_f32_opt);
            }
        } else {
            if (is_causal) {
                sdpa_bf16_tc_fwd_kernel<128, true><<<grid, 128, 0, ctx->stream>>>(
                    h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, op, so, lse_f32_opt);
            } else {
                sdpa_bf16_tc_fwd_kernel<128, false><<<grid, 128, 0, ctx->stream>>>(
                    h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, op, so, lse_f32_opt);
            }
        }
        return rt(cudaGetLastError());
    }

    dim3 grid(static_cast<unsigned>((t_q + FALLBACK_ROWS_PER_BLOCK - 1) / FALLBACK_ROWS_PER_BLOCK),
              static_cast<unsigned>(batch * h_q));
    sdpa_bf16_generic_fwd_kernel<<<grid, 32 * FALLBACK_ROWS_PER_BLOCK, 0, ctx->stream>>>(
        h_q, h_kv, t_q, t_k, d, scale, is_causal, qp, sq, kp, sk, vp, sv, op, so, lse_f32_opt);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_sdpa_bf16_bwd(
    Context *ctx,
    uint64_t batch, uint64_t h_q, uint64_t h_kv, uint64_t t_q, uint64_t t_k, uint64_t d,
    float scale, int is_causal,
    const uint16_t *q, Bf16SdpaStrides sq,
    const uint16_t *k, Bf16SdpaStrides sk,
    const uint16_t *v, Bf16SdpaStrides sv,
    const uint16_t *o, Bf16SdpaStrides so,
    const uint16_t *do_in, Bf16SdpaStrides sdo,
    const float *lse_f32, float *d_row_f32,
    float *gqa_dk_f32, float *gqa_dv_f32,
    uint16_t *dq, Bf16SdpaStrides sdq,
    uint16_t *dk, Bf16SdpaStrides sdk,
    uint16_t *dv, Bf16SdpaStrides sdv,
    int accumulate_dq, int accumulate_dk, int accumulate_dv)
{
    if (!ctx || h_q == 0 || h_kv == 0 || (h_q % h_kv) != 0 || d == 0) {
        return rt(cudaErrorInvalidValue);
    }
    if (batch == 0 || t_q == 0 || t_k == 0) return 0;
    if (!q || !k || !v || !o || !do_in || !lse_f32 || !d_row_f32 || !dq || !dk || !dv) {
        return rt(cudaErrorInvalidValue);
    }
    bool is_gqa = (h_q > h_kv);
    int code = activate(ctx);
    if (code) return code;

    const __nv_bfloat16 *qp  = reinterpret_cast<const __nv_bfloat16 *>(q);
    const __nv_bfloat16 *kp  = reinterpret_cast<const __nv_bfloat16 *>(k);
    const __nv_bfloat16 *vp  = reinterpret_cast<const __nv_bfloat16 *>(v);
    const __nv_bfloat16 *op  = reinterpret_cast<const __nv_bfloat16 *>(o);
    const __nv_bfloat16 *dop = reinterpret_cast<const __nv_bfloat16 *>(do_in);
    __nv_bfloat16 *dqp = reinterpret_cast<__nv_bfloat16 *>(dq);
    __nv_bfloat16 *dkp = reinterpret_cast<__nv_bfloat16 *>(dk);
    __nv_bfloat16 *dvp = reinterpret_cast<__nv_bfloat16 *>(dv);

    bool tc_ok = (ctx->cc_major >= 8) && (d == 64 || d == 128) &&
                 stride_aligned8(q, sq) && stride_aligned8(k, sk) && stride_aligned8(v, sv) &&
                 stride_aligned8(do_in, sdo) && stride_aligned8(dq, sdq) &&
                 stride_aligned8(dk, sdk) && stride_aligned8(dv, sdv);

    if (tc_ok) {
        if (is_gqa && (!gqa_dk_f32 || !gqa_dv_f32)) {
            return rt(cudaErrorInvalidValue);
        }
        dim3 prep_grid(static_cast<unsigned>((t_q + 3ULL) / 4ULL),
                       static_cast<unsigned>(batch * h_q));
        sdpa_bf16_bwd_prep_kernel<<<prep_grid, 128, 0, ctx->stream>>>(
            h_q, t_q, d, op, so, dop, sdo, d_row_f32);
        code = rt(cudaGetLastError());
        if (code) return code;
        size_t dq_smem = 4ULL * 64ULL * d * sizeof(uint16_t);
        size_t kv_smem = dq_smem + 128ULL * sizeof(float);
        dim3 dq_grid(static_cast<unsigned>((t_q + 63ULL) / 64ULL), static_cast<unsigned>(batch * h_q));
        dim3 kv_grid(static_cast<unsigned>((t_k + 63ULL) / 64ULL), static_cast<unsigned>(batch * h_q));

        if (d == 64) {
            if (is_causal) {
                sdpa_bf16_tc_bwd_dq_kernel<64, true><<<dq_grid, 128, dq_smem, ctx->stream>>>(
                    h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                    lse_f32, d_row_f32, dqp, sdq, accumulate_dq);
                code = rt(cudaGetLastError()); if (code) return code;
                if (is_gqa) {
                    sdpa_bf16_tc_bwd_dkdv_kernel<64, true, true><<<kv_grid, 128, kv_smem, ctx->stream>>>(
                        h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                        lse_f32, d_row_f32, dkp, sdk, dvp, sdv, gqa_dk_f32, gqa_dv_f32,
                        accumulate_dk, accumulate_dv);
                } else {
                    sdpa_bf16_tc_bwd_dkdv_kernel<64, true, false><<<kv_grid, 128, kv_smem, ctx->stream>>>(
                        h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                        lse_f32, d_row_f32, dkp, sdk, dvp, sdv, nullptr, nullptr,
                        accumulate_dk, accumulate_dv);
                }
            } else {
                sdpa_bf16_tc_bwd_dq_kernel<64, false><<<dq_grid, 128, dq_smem, ctx->stream>>>(
                    h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                    lse_f32, d_row_f32, dqp, sdq, accumulate_dq);
                code = rt(cudaGetLastError()); if (code) return code;
                if (is_gqa) {
                    sdpa_bf16_tc_bwd_dkdv_kernel<64, false, true><<<kv_grid, 128, kv_smem, ctx->stream>>>(
                        h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                        lse_f32, d_row_f32, dkp, sdk, dvp, sdv, gqa_dk_f32, gqa_dv_f32,
                        accumulate_dk, accumulate_dv);
                } else {
                    sdpa_bf16_tc_bwd_dkdv_kernel<64, false, false><<<kv_grid, 128, kv_smem, ctx->stream>>>(
                        h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                        lse_f32, d_row_f32, dkp, sdk, dvp, sdv, nullptr, nullptr,
                        accumulate_dk, accumulate_dv);
                }
            }
        } else {
            if (is_causal) {
                sdpa_bf16_tc_bwd_dq_kernel<128, true><<<dq_grid, 128, dq_smem, ctx->stream>>>(
                    h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                    lse_f32, d_row_f32, dqp, sdq, accumulate_dq);
                code = rt(cudaGetLastError()); if (code) return code;
                if (is_gqa) {
                    sdpa_bf16_tc_bwd_dkdv_kernel<128, true, true><<<kv_grid, 128, kv_smem, ctx->stream>>>(
                        h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                        lse_f32, d_row_f32, dkp, sdk, dvp, sdv, gqa_dk_f32, gqa_dv_f32,
                        accumulate_dk, accumulate_dv);
                } else {
                    sdpa_bf16_tc_bwd_dkdv_kernel<128, true, false><<<kv_grid, 128, kv_smem, ctx->stream>>>(
                        h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                        lse_f32, d_row_f32, dkp, sdk, dvp, sdv, nullptr, nullptr,
                        accumulate_dk, accumulate_dv);
                }
            } else {
                sdpa_bf16_tc_bwd_dq_kernel<128, false><<<dq_grid, 128, dq_smem, ctx->stream>>>(
                    h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                    lse_f32, d_row_f32, dqp, sdq, accumulate_dq);
                code = rt(cudaGetLastError()); if (code) return code;
                if (is_gqa) {
                    sdpa_bf16_tc_bwd_dkdv_kernel<128, false, true><<<kv_grid, 128, kv_smem, ctx->stream>>>(
                        h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                        lse_f32, d_row_f32, dkp, sdk, dvp, sdv, gqa_dk_f32, gqa_dv_f32,
                        accumulate_dk, accumulate_dv);
                } else {
                    sdpa_bf16_tc_bwd_dkdv_kernel<128, false, false><<<kv_grid, 128, kv_smem, ctx->stream>>>(
                        h_q, h_kv, t_q, t_k, scale, qp, sq, kp, sk, vp, sv, dop, sdo,
                        lse_f32, d_row_f32, dkp, sdk, dvp, sdv, nullptr, nullptr,
                        accumulate_dk, accumulate_dv);
                }
            }
        }
        code = rt(cudaGetLastError());
        if (code) return code;

        if (is_gqa) {
            uint64_t total_kv = batch * h_kv * t_k * d;
            unsigned g = static_cast<unsigned>(umin64((total_kv + 255ULL) / 256ULL, 65535ULL));
            sdpa_bf16_gqa_reduce_kernel<<<g, 256, 0, ctx->stream>>>(
                batch, h_q, h_kv, t_k, d, gqa_dk_f32, dkp, sdk, accumulate_dk);
            code = rt(cudaGetLastError()); if (code) return code;
            sdpa_bf16_gqa_reduce_kernel<<<g, 256, 0, ctx->stream>>>(
                batch, h_q, h_kv, t_k, d, gqa_dv_f32, dvp, sdv, accumulate_dv);
            code = rt(cudaGetLastError());
        }
        return code;
    }

    dim3 dq_grid(static_cast<unsigned>((t_q + FALLBACK_ROWS_PER_BLOCK - 1) / FALLBACK_ROWS_PER_BLOCK),
                 static_cast<unsigned>(batch * h_q));
    sdpa_bf16_generic_bwd_dq_kernel<<<dq_grid, 32 * FALLBACK_ROWS_PER_BLOCK, 0, ctx->stream>>>(
        h_q, h_kv, t_q, t_k, d, scale, is_causal,
        qp, sq, kp, sk, vp, sv, op, so, dop, sdo, lse_f32, d_row_f32, dqp, sdq, accumulate_dq);
    code = rt(cudaGetLastError());
    if (code) return code;
    dim3 kv_grid(static_cast<unsigned>((t_k + FALLBACK_ROWS_PER_BLOCK - 1) / FALLBACK_ROWS_PER_BLOCK),
                 static_cast<unsigned>(batch * h_kv));
    sdpa_bf16_generic_bwd_dkdv_kernel<<<kv_grid, 32 * FALLBACK_ROWS_PER_BLOCK, 0, ctx->stream>>>(
        h_q, h_kv, t_q, t_k, d, scale, is_causal,
        qp, sq, kp, sk, vp, sv, dop, sdo, lse_f32, d_row_f32,
        dkp, sdk, dvp, sdv, accumulate_dk, accumulate_dv);
    return rt(cudaGetLastError());
}
