#include "bf16_tc.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

static int blas_status(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : 2000 + (int)s; }

__host__ __device__ __forceinline__ uint64_t umin64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max_f32(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(0xffffffffu, v, off);
        v = (isnan(v) || isnan(o)) ? NAN : fmaxf(v, o);
    }
    return v;
}

__device__ __forceinline__ double warp_reduce_sum_f64(double v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ int32_t warp_reduce_sum_i32(int32_t v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum_f32(float v) {
    __shared__ float smem[8];
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    v = warp_reduce_sum_f32(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        t = warp_reduce_sum_f32(t);
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ double block_reduce_sum_f64(double v) {
    __shared__ double smem[8];
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    v = warp_reduce_sum_f64(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        double t = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0;
        t = warp_reduce_sum_f64(t);
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_max_f32(float v) {
    __shared__ float smem[8];
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    v = warp_reduce_max_f32(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < (blockDim.x >> 5)) ? smem[lane] : -INFINITY;
        t = warp_reduce_max_f32(t);
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    return smem[0];
}

// Helper: unpack 8 BF16s from uint4 into 8 FP32s.
__device__ __forceinline__ void unpack_bf16x8(uint4 p, float out[8]) {
    float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.x));
    float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.y));
    float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.z));
    float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.w));
    out[0] = f0.x; out[1] = f0.y;
    out[2] = f1.x; out[3] = f1.y;
    out[4] = f2.x; out[5] = f2.y;
    out[6] = f3.x; out[7] = f3.y;
}

__device__ __forceinline__ uint4 pack_bf16x8(const float in[8]) {
    __nv_bfloat162 b0 = __floats2bfloat162_rn(in[0], in[1]);
    __nv_bfloat162 b1 = __floats2bfloat162_rn(in[2], in[3]);
    __nv_bfloat162 b2 = __floats2bfloat162_rn(in[4], in[5]);
    __nv_bfloat162 b3 = __floats2bfloat162_rn(in[6], in[7]);
    return make_uint4(
        *reinterpret_cast<const uint32_t *>(&b0),
        *reinterpret_cast<const uint32_t *>(&b1),
        *reinterpret_cast<const uint32_t *>(&b2),
        *reinterpret_cast<const uint32_t *>(&b3));
}

__device__ __forceinline__ uint4 philox4x32_10_dev(uint4 ctr, uint2 key) {
    constexpr uint64_t M0 = 0xD2511F53ULL;
    constexpr uint64_t M1 = 0xCD9E8D57ULL;
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        uint64_t p0 = M0 * static_cast<uint64_t>(ctr.x);
        uint64_t p1 = M1 * static_cast<uint64_t>(ctr.z);
        uint32_t hi0 = static_cast<uint32_t>(p0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(p0);
        uint32_t hi1 = static_cast<uint32_t>(p1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(p1);
        ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += W0;
        key.y += W1;
    }
    return ctr;
}

__device__ __forceinline__ float u32_to_unit_f32_dev(uint32_t word) {
    return static_cast<float>(word >> 8) * (1.0f / 16777216.0f);
}

// --- Vectorized BF16 RmsNorm (and fused Residual + RmsNorm) ---

template<bool FUSED_RESIDUAL>
__global__ void bf16_rmsnorm_fwd_vec8_kernel(
    uint64_t rows,
    uint64_t cols,
    float eps,
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ y,
    float *__restrict__ inv_rms_out)
{
    uint64_t r = blockIdx.x;
    if (r >= rows) return;
    uint64_t cols8 = cols >> 3;
    const uint4 *a8 = reinterpret_cast<const uint4 *>(a + r * cols);
    const uint4 *b8 = FUSED_RESIDUAL ? reinterpret_cast<const uint4 *>(b + r * cols) : nullptr;
    const uint4 *w8 = weight ? reinterpret_cast<const uint4 *>(weight) : nullptr;
    uint4 *res8 = FUSED_RESIDUAL ? reinterpret_cast<uint4 *>(residual_out + r * cols) : nullptr;
    uint4 *y8 = reinterpret_cast<uint4 *>(y + r * cols);

    // Single-pass register cache when cols8 <= blockDim.x (covers cols <= 2048 at 256 threads!).
    if (cols8 <= blockDim.x) {
        float x_reg[8];
        float sum_sq = 0.0f;
        if (threadIdx.x < cols8) {
            unpack_bf16x8(a8[threadIdx.x], x_reg);
            if constexpr (FUSED_RESIDUAL) {
                float b_reg[8];
                unpack_bf16x8(b8[threadIdx.x], b_reg);
                #pragma unroll
                for (int k = 0; k < 8; ++k) x_reg[k] += b_reg[k];
                uint4 packed = pack_bf16x8(x_reg);
                res8[threadIdx.x] = packed;
                unpack_bf16x8(packed, x_reg);
            }
            #pragma unroll
            for (int k = 0; k < 8; ++k) sum_sq += x_reg[k] * x_reg[k];
        }
        float total_sq = block_reduce_sum_f32(sum_sq);
        float inv_rms = rsqrtf(total_sq / static_cast<float>(cols) + eps);
        if (inv_rms_out && threadIdx.x == 0) inv_rms_out[r] = inv_rms;
        if (threadIdx.x < cols8) {
            if (w8) {
                float w_reg[8];
                unpack_bf16x8(w8[threadIdx.x], w_reg);
                #pragma unroll
                for (int k = 0; k < 8; ++k) x_reg[k] = x_reg[k] * inv_rms * w_reg[k];
            } else {
                #pragma unroll
                for (int k = 0; k < 8; ++k) x_reg[k] *= inv_rms;
            }
            y8[threadIdx.x] = pack_bf16x8(x_reg);
        }
        return;
    }

    // General multi-iteration cols8 > blockDim.x
    float sum_sq = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols8; j += blockDim.x) {
        float x_reg[8];
        unpack_bf16x8(a8[j], x_reg);
        if constexpr (FUSED_RESIDUAL) {
            float b_reg[8];
            unpack_bf16x8(b8[j], b_reg);
            #pragma unroll
            for (int k = 0; k < 8; ++k) x_reg[k] += b_reg[k];
            uint4 packed = pack_bf16x8(x_reg);
            res8[j] = packed;
            unpack_bf16x8(packed, x_reg);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) sum_sq += x_reg[k] * x_reg[k];
    }
    float total_sq = block_reduce_sum_f32(sum_sq);
    float inv_rms = rsqrtf(total_sq / static_cast<float>(cols) + eps);
    if (inv_rms_out && threadIdx.x == 0) inv_rms_out[r] = inv_rms;

    const uint4 *src8 = FUSED_RESIDUAL ? res8 : a8;
    for (uint64_t j = threadIdx.x; j < cols8; j += blockDim.x) {
        float x_reg[8];
        unpack_bf16x8(src8[j], x_reg);
        if (w8) {
            float w_reg[8];
            unpack_bf16x8(w8[j], w_reg);
            #pragma unroll
            for (int k = 0; k < 8; ++k) x_reg[k] = x_reg[k] * inv_rms * w_reg[k];
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) x_reg[k] *= inv_rms;
        }
        y8[j] = pack_bf16x8(x_reg);
    }
}

template<bool FUSED_RESIDUAL>
__global__ void bf16_rmsnorm_fwd_scalar_kernel(
    uint64_t rows,
    uint64_t cols,
    float eps,
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ y,
    float *__restrict__ inv_rms_out)
{
    uint64_t r = blockIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16 *ar = a + r * cols;
    const __nv_bfloat16 *br = FUSED_RESIDUAL ? (b + r * cols) : nullptr;
    __nv_bfloat16 *rr = FUSED_RESIDUAL ? (residual_out + r * cols) : nullptr;
    __nv_bfloat16 *yr = y + r * cols;

    float sum_sq = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = __bfloat162float(ar[j]);
        if constexpr (FUSED_RESIDUAL) {
            __nv_bfloat16 s = __float2bfloat16_rn(v + __bfloat162float(br[j]));
            rr[j] = s;
            v = __bfloat162float(s);
        }
        sum_sq += v * v;
    }
    float total_sq = block_reduce_sum_f32(sum_sq);
    float inv_rms = rsqrtf(total_sq / static_cast<float>(cols) + eps);
    if (inv_rms_out && threadIdx.x == 0) inv_rms_out[r] = inv_rms;

    const __nv_bfloat16 *sr = FUSED_RESIDUAL ? rr : ar;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = __bfloat162float(sr[j]) * inv_rms;
        if (weight) v *= __bfloat162float(weight[j]);
        yr[j] = __float2bfloat16_rn(v);
    }
}

__global__ void bf16_rmsnorm_bwd_dx_vec8_kernel(
    uint64_t rows,
    uint64_t cols,
    float eps,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    const float *__restrict__ inv_rms_in,
    const __nv_bfloat16 *__restrict__ dy,
    __nv_bfloat16 *__restrict__ dx,
    int accumulate_dx)
{
    uint64_t r = blockIdx.x;
    if (r >= rows) return;
    uint64_t cols8 = cols >> 3;
    const uint4 *x8 = reinterpret_cast<const uint4 *>(x + r * cols);
    const uint4 *dy8 = reinterpret_cast<const uint4 *>(dy + r * cols);
    const uint4 *w8 = weight ? reinterpret_cast<const uint4 *>(weight) : nullptr;
    uint4 *dx8 = reinterpret_cast<uint4 *>(dx + r * cols);

    if (cols8 <= blockDim.x) {
        float xv[8], dyv[8], wv[8];
        float sum_sq = 0.0f;
        float dot = 0.0f;
        if (threadIdx.x < cols8) {
            unpack_bf16x8(x8[threadIdx.x], xv);
            unpack_bf16x8(dy8[threadIdx.x], dyv);
            if (w8) {
                unpack_bf16x8(w8[threadIdx.x], wv);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    dyv[k] *= wv[k];
                    dot += dyv[k] * xv[k];
                    if (!inv_rms_in) sum_sq += xv[k] * xv[k];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    dot += dyv[k] * xv[k];
                    if (!inv_rms_in) sum_sq += xv[k] * xv[k];
                }
            }
        }
        float inv_rms = inv_rms_in ? inv_rms_in[r]
            : rsqrtf(block_reduce_sum_f32(sum_sq) / static_cast<float>(cols) + eps);
        float total_dot = block_reduce_sum_f32(dot);
        float coeff = total_dot * (inv_rms * inv_rms) / static_cast<float>(cols);
        if (threadIdx.x < cols8) {
            float outv[8];
            if (accumulate_dx) {
                float prev[8];
                unpack_bf16x8(dx8[threadIdx.x], prev);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    outv[k] = prev[k] + inv_rms * (dyv[k] - xv[k] * coeff);
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    outv[k] = inv_rms * (dyv[k] - xv[k] * coeff);
                }
            }
            dx8[threadIdx.x] = pack_bf16x8(outv);
        }
        return;
    }

    float sum_sq = 0.0f;
    float dot = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols8; j += blockDim.x) {
        float xv[8], dyv[8];
        unpack_bf16x8(x8[j], xv);
        unpack_bf16x8(dy8[j], dyv);
        if (w8) {
            float wv[8];
            unpack_bf16x8(w8[j], wv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                dot += (dyv[k] * wv[k]) * xv[k];
                if (!inv_rms_in) sum_sq += xv[k] * xv[k];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                dot += dyv[k] * xv[k];
                if (!inv_rms_in) sum_sq += xv[k] * xv[k];
            }
        }
    }
    float inv_rms = inv_rms_in ? inv_rms_in[r]
        : rsqrtf(block_reduce_sum_f32(sum_sq) / static_cast<float>(cols) + eps);
    float total_dot = block_reduce_sum_f32(dot);
    float coeff = total_dot * (inv_rms * inv_rms) / static_cast<float>(cols);
    for (uint64_t j = threadIdx.x; j < cols8; j += blockDim.x) {
        float xv[8], dyv[8], outv[8];
        unpack_bf16x8(x8[j], xv);
        unpack_bf16x8(dy8[j], dyv);
        if (w8) {
            float wv[8];
            unpack_bf16x8(w8[j], wv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) dyv[k] *= wv[k];
        }
        if (accumulate_dx) {
            float prev[8];
            unpack_bf16x8(dx8[j], prev);
            #pragma unroll
            for (int k = 0; k < 8; ++k) outv[k] = prev[k] + inv_rms * (dyv[k] - xv[k] * coeff);
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) outv[k] = inv_rms * (dyv[k] - xv[k] * coeff);
        }
        dx8[j] = pack_bf16x8(outv);
    }
}

__global__ void bf16_rmsnorm_bwd_dx_scalar_kernel(
    uint64_t rows,
    uint64_t cols,
    float eps,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    const float *__restrict__ inv_rms_in,
    const __nv_bfloat16 *__restrict__ dy,
    __nv_bfloat16 *__restrict__ dx,
    int accumulate_dx)
{
    uint64_t r = blockIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16 *xr = x + r * cols;
    const __nv_bfloat16 *dyr = dy + r * cols;
    __nv_bfloat16 *dxr = dx + r * cols;

    float sum_sq = 0.0f;
    float dot = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float xv = __bfloat162float(xr[j]);
        float g = __bfloat162float(dyr[j]) * (weight ? __bfloat162float(weight[j]) : 1.0f);
        dot += g * xv;
        if (!inv_rms_in) sum_sq += xv * xv;
    }
    float inv_rms = inv_rms_in ? inv_rms_in[r]
        : rsqrtf(block_reduce_sum_f32(sum_sq) / static_cast<float>(cols) + eps);
    float total_dot = block_reduce_sum_f32(dot);
    float coeff = total_dot * (inv_rms * inv_rms) / static_cast<float>(cols);
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float xv = __bfloat162float(xr[j]);
        float g = __bfloat162float(dyr[j]) * (weight ? __bfloat162float(weight[j]) : 1.0f);
        float d = inv_rms * (g - xv * coeff);
        if (accumulate_dx) d += __bfloat162float(dxr[j]);
        dxr[j] = __float2bfloat16_rn(d);
    }
}

// Deterministic zero-atomic 2-stage reduction for RmsNorm dWeight (FP32 master grad).
__global__ void bf16_rmsnorm_bwd_dw_partial_kernel(
    uint64_t rows,
    uint64_t cols,
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ inv_rms,
    const __nv_bfloat16 *__restrict__ dy,
    float *__restrict__ partials)
{
    uint64_t chunk = blockIdx.y;
    uint64_t chunks = gridDim.y;
    uint64_t col = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    uint64_t r_begin = (rows * chunk) / chunks;
    uint64_t r_end = (rows * (chunk + 1)) / chunks;
    float acc = 0.0f;
    for (uint64_t r = r_begin; r < r_end; ++r) {
        float xv = __bfloat162float(x[r * cols + col]);
        float dyv = __bfloat162float(dy[r * cols + col]);
        acc += dyv * xv * inv_rms[r];
    }
    partials[chunk * cols + col] = acc;
}

__global__ void bf16_rmsnorm_bwd_dw_finalize_kernel(
    uint64_t chunks,
    uint64_t cols,
    const float *__restrict__ partials,
    float *__restrict__ dweight,
    int accumulate_dw)
{
    uint64_t col = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float acc = accumulate_dw ? dweight[col] : 0.0f;
    for (uint64_t c = 0; c < chunks; ++c) {
        acc += partials[c * cols + col];
    }
    dweight[col] = acc;
}

// --- Vectorized BF16 LayerNorm ---

__global__ void bf16_layernorm_fwd_kernel(
    uint64_t rows, uint64_t cols, float eps,
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ y)
{
    uint64_t r = blockIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16 *xr = x + r * cols;
    __nv_bfloat16 *yr = y + r * cols;
    float s = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) s += __bfloat162float(xr[j]);
    float mean = block_reduce_sum_f32(s) / static_cast<float>(cols);
    float v = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float d = __bfloat162float(xr[j]) - mean;
        v += d * d;
    }
    float inv_std = rsqrtf(block_reduce_sum_f32(v) / static_cast<float>(cols) + eps);
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        yr[j] = __float2bfloat16_rn((__bfloat162float(xr[j]) - mean) * inv_std);
    }
}

__global__ void bf16_layernorm_bwd_kernel(
    uint64_t rows, uint64_t cols, float eps,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ dy,
    __nv_bfloat16 *__restrict__ dx,
    int accumulate)
{
    uint64_t r = blockIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16 *xr = x + r * cols;
    const __nv_bfloat16 *dyr = dy + r * cols;
    __nv_bfloat16 *dxr = dx + r * cols;
    float s = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) s += __bfloat162float(xr[j]);
    float mean = block_reduce_sum_f32(s) / static_cast<float>(cols);
    float v = 0.0f, sum_dy = 0.0f, sum_dy_x = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float d = __bfloat162float(xr[j]) - mean;
        float g = __bfloat162float(dyr[j]);
        v += d * d;
        sum_dy += g;
        sum_dy_x += g * d;
    }
    float inv_std = rsqrtf(block_reduce_sum_f32(v) / static_cast<float>(cols) + eps);
    float mean_dy = block_reduce_sum_f32(sum_dy) / static_cast<float>(cols);
    float dot = block_reduce_sum_f32(sum_dy_x) * (inv_std * inv_std) / static_cast<float>(cols);
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float d = __bfloat162float(xr[j]) - mean;
        float g = __bfloat162float(dyr[j]);
        float out = inv_std * (g - mean_dy - d * dot);
        if (accumulate) out += __bfloat162float(dxr[j]);
        dxr[j] = __float2bfloat16_rn(out);
    }
}

// --- Vectorized BF16 Softmax ---

__global__ void bf16_softmax_fwd_kernel(
    uint64_t rows, uint64_t cols,
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ y)
{
    uint64_t r = blockIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16 *xr = x + r * cols;
    __nv_bfloat16 *yr = y + r * cols;
    float m = -INFINITY;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = __bfloat162float(xr[j]);
        m = (isnan(m) || isnan(v)) ? NAN : fmaxf(m, v);
    }
    m = block_reduce_max_f32(m);
    float s = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        s += expf(__bfloat162float(xr[j]) - m);
    }
    float inv_s = 1.0f / block_reduce_sum_f32(s);
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        yr[j] = __float2bfloat16_rn(expf(__bfloat162float(xr[j]) - m) * inv_s);
    }
}

__global__ void bf16_softmax_bwd_kernel(
    uint64_t rows, uint64_t cols,
    const __nv_bfloat16 *__restrict__ y,
    const __nv_bfloat16 *__restrict__ dy,
    __nv_bfloat16 *__restrict__ dx,
    int accumulate)
{
    uint64_t r = blockIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16 *yr = y + r * cols;
    const __nv_bfloat16 *dyr = dy + r * cols;
    __nv_bfloat16 *dxr = dx + r * cols;
    float dot = 0.0f;
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        dot += __bfloat162float(yr[j]) * __bfloat162float(dyr[j]);
    }
    dot = block_reduce_sum_f32(dot);
    for (uint64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float yv = __bfloat162float(yr[j]);
        float g = yv * (__bfloat162float(dyr[j]) - dot);
        if (accumulate) g += __bfloat162float(dxr[j]);
        dxr[j] = __float2bfloat16_rn(g);
    }
}

// --- Vectorized BF16 SwiGLU (fused [rows, 2*hidden] -> [rows, hidden]) ---

__global__ void bf16_swiglu_fwd_vec8_kernel(
    uint64_t rows, uint64_t hidden,
    const __nv_bfloat16 *__restrict__ gate_up,
    __nv_bfloat16 *__restrict__ out)
{
    uint64_t h8 = hidden >> 3;
    uint64_t total8 = rows * h8;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint4 *gu8 = reinterpret_cast<const uint4 *>(gate_up);
    uint4 *o8 = reinterpret_cast<uint4 *>(out);
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total8; idx += stride) {
        uint64_t r = idx / h8;
        uint64_t c = idx - r * h8;
        float g[8], u[8], y[8];
        unpack_bf16x8(gu8[r * (2 * h8) + c], g);
        unpack_bf16x8(gu8[r * (2 * h8) + h8 + c], u);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float sig = 1.0f / (1.0f + expf(-g[k]));
            y[k] = (g[k] * sig) * u[k];
        }
        o8[idx] = pack_bf16x8(y);
    }
}

__global__ void bf16_swiglu_fwd_scalar_kernel(
    uint64_t rows, uint64_t hidden,
    const __nv_bfloat16 *__restrict__ gate_up,
    __nv_bfloat16 *__restrict__ out)
{
    uint64_t total = rows * hidden;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        uint64_t r = idx / hidden;
        uint64_t c = idx - r * hidden;
        float g = __bfloat162float(gate_up[r * (2 * hidden) + c]);
        float u = __bfloat162float(gate_up[r * (2 * hidden) + hidden + c]);
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = __float2bfloat16_rn((g * sig) * u);
    }
}

__global__ void bf16_swiglu_bwd_vec8_kernel(
    uint64_t rows, uint64_t hidden,
    const __nv_bfloat16 *__restrict__ gate_up,
    const __nv_bfloat16 *__restrict__ dout,
    __nv_bfloat16 *__restrict__ dgate_up)
{
    uint64_t h8 = hidden >> 3;
    uint64_t total8 = rows * h8;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint4 *gu8 = reinterpret_cast<const uint4 *>(gate_up);
    const uint4 *do8 = reinterpret_cast<const uint4 *>(dout);
    uint4 *dgu8 = reinterpret_cast<uint4 *>(dgate_up);
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total8; idx += stride) {
        uint64_t r = idx / h8;
        uint64_t c = idx - r * h8;
        float g[8], u[8], dy[8], dg[8], du[8];
        unpack_bf16x8(gu8[r * (2 * h8) + c], g);
        unpack_bf16x8(gu8[r * (2 * h8) + h8 + c], u);
        unpack_bf16x8(do8[idx], dy);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float sig = 1.0f / (1.0f + expf(-g[k]));
            float silu = g[k] * sig;
            float dsilu = sig * (1.0f + g[k] * (1.0f - sig));
            dg[k] = dy[k] * u[k] * dsilu;
            du[k] = dy[k] * silu;
        }
        dgu8[r * (2 * h8) + c] = pack_bf16x8(dg);
        dgu8[r * (2 * h8) + h8 + c] = pack_bf16x8(du);
    }
}

__global__ void bf16_swiglu_bwd_scalar_kernel(
    uint64_t rows, uint64_t hidden,
    const __nv_bfloat16 *__restrict__ gate_up,
    const __nv_bfloat16 *__restrict__ dout,
    __nv_bfloat16 *__restrict__ dgate_up)
{
    uint64_t total = rows * hidden;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        uint64_t r = idx / hidden;
        uint64_t c = idx - r * hidden;
        float g = __bfloat162float(gate_up[r * (2 * hidden) + c]);
        float u = __bfloat162float(gate_up[r * (2 * hidden) + hidden + c]);
        float dy = __bfloat162float(dout[idx]);
        float sig = 1.0f / (1.0f + expf(-g));
        float silu = g * sig;
        float dsilu = sig * (1.0f + g * (1.0f - sig));
        dgate_up[r * (2 * hidden) + c] = __float2bfloat16_rn(dy * u * dsilu);
        dgate_up[r * (2 * hidden) + hidden + c] = __float2bfloat16_rn(dy * silu);
    }
}

// --- Vectorized BF16 Residual Add ---

__global__ void bf16_add_kernel(
    uint64_t n,
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out)
{
    uint64_t n8 = n >> 3;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    if (((reinterpret_cast<uintptr_t>(a) & 15ULL) == 0ULL) &&
        ((reinterpret_cast<uintptr_t>(b) & 15ULL) == 0ULL) &&
        ((reinterpret_cast<uintptr_t>(out) & 15ULL) == 0ULL)) {
        const uint4 *a8 = reinterpret_cast<const uint4 *>(a);
        const uint4 *b8 = reinterpret_cast<const uint4 *>(b);
        uint4 *o8 = reinterpret_cast<uint4 *>(out);
        for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < n8; i += stride) {
            float av[8], bv[8], cv[8];
            unpack_bf16x8(a8[i], av);
            unpack_bf16x8(b8[i], bv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) cv[k] = av[k] + bv[k];
            o8[i] = pack_bf16x8(cv);
        }
        uint64_t tail = n8 << 3;
        for (uint64_t i = tail + static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < n; i += stride) {
            out[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
        }
        return;
    }
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}

// --- Vectorized BF16 RoPE (arbitrary 4D stride_b, stride_h, stride_t with stride_d == 1; in-place safe) ---

__global__ void bf16_rope_kernel(
    uint64_t batch, uint64_t heads, uint64_t seq_len, uint64_t head_dim, uint64_t rotary_dim,
    int64_t in_sb, int64_t in_sh, int64_t in_st,
    int64_t out_sb, int64_t out_sh, int64_t out_st,
    const float *__restrict__ cos_tbl,
    const float *__restrict__ sin_tbl,
    const __nv_bfloat16 *__restrict__ inp,
    __nv_bfloat16 *__restrict__ out,
    int inverse,
    int accumulate)
{
    uint64_t half = rotary_dim >> 1;
    uint64_t pass = head_dim - rotary_dim;
    uint64_t work = half + pass;
    uint64_t rows = batch * heads * seq_len;
    uint64_t total = rows * work;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        uint64_t row = idx / work;
        uint64_t col = idx - row * work;
        uint64_t t = row % seq_len;
        uint64_t tmp = row / seq_len;
        uint64_t h = tmp % heads;
        uint64_t b = tmp / heads;
        const __nv_bfloat16 *x_row = inp + static_cast<int64_t>(b) * in_sb
                                         + static_cast<int64_t>(h) * in_sh
                                         + static_cast<int64_t>(t) * in_st;
        __nv_bfloat16 *y_row = out + static_cast<int64_t>(b) * out_sb
                                   + static_cast<int64_t>(h) * out_sh
                                   + static_cast<int64_t>(t) * out_st;
        if (col < half) {
            float a = __bfloat162float(x_row[col]);
            float bb = __bfloat162float(x_row[col + half]);
            float c = cos_tbl[t * half + col];
            float s = sin_tbl[t * half + col];
            float y0 = inverse ? (a * c + bb * s) : (a * c - bb * s);
            float y1 = inverse ? (-a * s + bb * c) : (a * s + bb * c);
            if (accumulate) {
                y0 += __bfloat162float(y_row[col]);
                y1 += __bfloat162float(y_row[col + half]);
            }
            y_row[col] = __float2bfloat16_rn(y0);
            y_row[col + half] = __float2bfloat16_rn(y1);
        } else {
            uint64_t d = rotary_dim + (col - half);
            float v = __bfloat162float(x_row[d]);
            if (accumulate) v += __bfloat162float(y_row[d]);
            y_row[d] = __float2bfloat16_rn(v);
        }
    }
}

// --- BF16 Embedding Fwd + Deterministic Zero-Atomic FP32 Master Grad Bwd ---

__global__ void bf16_embedding_fwd_kernel(
    uint64_t vocab, uint64_t dim, uint64_t num_idx,
    const __nv_bfloat16 *__restrict__ table,
    const int64_t *__restrict__ indices,
    __nv_bfloat16 *__restrict__ out)
{
    bool vec8 = ((dim & 7ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(table) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(out) & 15ULL) == 0ULL);
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    if (vec8) {
        uint64_t dim8 = dim >> 3;
        uint64_t total8 = num_idx * dim8;
        const uint4 *t8 = reinterpret_cast<const uint4 *>(table);
        uint4 *o8 = reinterpret_cast<uint4 *>(out);
        for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < total8; i += stride) {
            uint64_t pos = i / dim8;
            uint64_t d8 = i - pos * dim8;
            int64_t tok = indices[pos];
            o8[i] = (tok >= 0 && static_cast<uint64_t>(tok) < vocab)
                ? t8[static_cast<uint64_t>(tok) * dim8 + d8]
                : make_uint4(0u, 0u, 0u, 0u);
        }
        return;
    }
    uint64_t total = num_idx * dim;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        uint64_t pos = i / dim;
        uint64_t d = i - pos * dim;
        int64_t tok = indices[pos];
        out[i] = (tok >= 0 && static_cast<uint64_t>(tok) < vocab)
            ? table[static_cast<uint64_t>(tok) * dim + d]
            : __float2bfloat16_rn(0.0f);
    }
}

constexpr int BF16_EMB_ROWS = 4;
constexpr int BF16_EMB_TILE = 256;

__global__ void bf16_embedding_bwd_f32_det_kernel(
    uint64_t vocab, uint64_t dim, uint64_t num_idx,
    int64_t padding_idx, int has_pad,
    const int64_t *__restrict__ indices,
    const __nv_bfloat16 *__restrict__ upstream,
    float *__restrict__ grad_table,
    int accumulate)
{
    __shared__ int64_t s_idx[BF16_EMB_TILE];
    __shared__ uint32_t s_hits[BF16_EMB_ROWS][BF16_EMB_TILE];
    __shared__ int s_cnt[BF16_EMB_ROWS];

    uint64_t base_row = static_cast<uint64_t>(blockIdx.x) * BF16_EMB_ROWS;
    if (base_row >= vocab) return;
    int n_rows = static_cast<int>(umin64(BF16_EMB_ROWS, vocab - base_row));

    if (!accumulate) {
        for (int r = 0; r < n_rows; ++r) {
            float *dst = grad_table + (base_row + r) * dim;
            for (uint64_t d = threadIdx.x; d < dim; d += blockDim.x) dst[d] = 0.0f;
        }
    }

    for (uint64_t tile = 0; tile < num_idx; tile += BF16_EMB_TILE) {
        int tlen = static_cast<int>(umin64(BF16_EMB_TILE, num_idx - tile));
        if (threadIdx.x < tlen) s_idx[threadIdx.x] = indices[tile + threadIdx.x];
        __syncthreads();

        if (threadIdx.x < BF16_EMB_ROWS) {
            int r = threadIdx.x;
            int cnt = 0;
            if (r < n_rows) {
                int64_t target = static_cast<int64_t>(base_row + r);
                if (!has_pad || target != padding_idx) {
                    for (int k = 0; k < tlen; ++k) {
                        if (s_idx[k] == target) s_hits[r][cnt++] = static_cast<uint32_t>(tile + k);
                    }
                }
            }
            s_cnt[r] = cnt;
        }
        __syncthreads();

        for (int r = 0; r < n_rows; ++r) {
            int cnt = s_cnt[r];
            if (cnt == 0) continue;
            float *dst = grad_table + (base_row + r) * dim;
            for (uint64_t d = threadIdx.x; d < dim; d += blockDim.x) {
                float acc = dst[d];
                for (int h = 0; h < cnt; ++h) {
                    acc += __bfloat162float(upstream[static_cast<uint64_t>(s_hits[r][h]) * dim + d]);
                }
                dst[d] = acc;
            }
        }
        __syncthreads();
    }
}

__global__ void bf16_embedding_bwd_f32_atomic_kernel(
    uint64_t vocab, uint64_t dim, uint64_t num_idx,
    int64_t padding_idx, int has_pad,
    const int64_t *__restrict__ indices,
    const __nv_bfloat16 *__restrict__ upstream,
    float *__restrict__ grad_table)
{
    uint64_t total = num_idx * dim;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        uint64_t pos = i / dim;
        uint64_t d = i - pos * dim;
        int64_t tok = indices[pos];
        if (tok >= 0 && static_cast<uint64_t>(tok) < vocab && (!has_pad || tok != padding_idx)) {
            atomicAdd(&grad_table[static_cast<uint64_t>(tok) * dim + d], __bfloat162float(upstream[i]));
        }
    }
}

// --- Vectorized BF16 CrossEntropy (Fwd, Bwd, and Fused Single-Pass Fwd+Bwd) ---

__global__ void bf16_ce_count_valid_kernel(
    uint64_t num_rows, uint64_t vocab,
    int64_t ignore_index, int has_ignore,
    const int64_t *__restrict__ targets,
    int32_t *__restrict__ active_out)
{
    __shared__ int32_t s_cnt[8];
    int32_t cnt = 0;
    for (uint64_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
        int64_t t = targets[i];
        if ((!has_ignore || t != ignore_index) && t >= 0 && static_cast<uint64_t>(t) < vocab) cnt += 1;
    }
    cnt = warp_reduce_sum_i32(cnt);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    if (lane == 0) s_cnt[warp] = cnt;
    __syncthreads();
    if (warp == 0) {
        int32_t c = (lane < (blockDim.x >> 5)) ? s_cnt[lane] : 0;
        c = warp_reduce_sum_i32(c);
        if (lane == 0) active_out[0] = c;
    }
}

__global__ void bf16_ce_reduce_loss_kernel(
    uint64_t num_rows,
    const float *__restrict__ row_losses,
    const int32_t *__restrict__ row_valid,
    float *__restrict__ out_scalar)
{
    __shared__ double s_loss[8];
    __shared__ int32_t s_cnt[8];
    double acc = 0.0;
    int32_t cnt = 0;
    for (uint64_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
        acc += static_cast<double>(row_losses[i]);
        cnt += row_valid[i];
    }
    acc = warp_reduce_sum_f64(acc);
    cnt = warp_reduce_sum_i32(cnt);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    if (lane == 0) { s_loss[warp] = acc; s_cnt[warp] = cnt; }
    __syncthreads();
    if (warp == 0) {
        double l = (lane < (blockDim.x >> 5)) ? s_loss[lane] : 0.0;
        int32_t c = (lane < (blockDim.x >> 5)) ? s_cnt[lane] : 0;
        l = warp_reduce_sum_f64(l);
        c = warp_reduce_sum_i32(c);
        if (lane == 0) out_scalar[0] = (c > 0) ? static_cast<float>(l / static_cast<double>(c)) : 0.0f;
    }
}

template<bool WRITE_LOSS, bool WRITE_GRAD>
__global__ void bf16_ce_rows_kernel(
    uint64_t num_rows,
    uint64_t vocab,
    int64_t ignore_index,
    int has_ignore,
    float label_smoothing,
    float upstream_scalar,
    const __nv_bfloat16 *__restrict__ logits,
    const int64_t *__restrict__ targets,
    const int32_t *__restrict__ active_ptr,
    float *__restrict__ row_losses,
    int32_t *__restrict__ row_valid,
    __nv_bfloat16 *__restrict__ grad_logits,
    int accumulate)
{
    uint64_t r = blockIdx.x;
    if (r >= num_rows) return;
    int64_t tok = targets[r];
    bool ignored = (has_ignore && tok == ignore_index) || tok < 0 || static_cast<uint64_t>(tok) >= vocab;
    int32_t active = WRITE_GRAD ? active_ptr[0] : 1;
    if (ignored || (WRITE_GRAD && active <= 0)) {
        if constexpr (WRITE_LOSS) {
            if (threadIdx.x == 0) { row_losses[r] = 0.0f; row_valid[r] = 0; }
        }
        if constexpr (WRITE_GRAD) {
            if (!accumulate) {
                __nv_bfloat16 *dst = grad_logits + r * vocab;
                for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
                    dst[j] = __float2bfloat16_rn(0.0f);
                }
            }
        }
        return;
    }

    const __nv_bfloat16 *row = logits + r * vocab;
    bool vec8 = ((vocab & 7ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(row) & 15ULL) == 0ULL) &&
                (!WRITE_GRAD || ((reinterpret_cast<uintptr_t>(grad_logits + r * vocab) & 15ULL) == 0ULL));

    // Online 1-pass (max, sum_exp) in FP32!
    float local_m = -INFINITY;
    float local_d = 0.0f;
    double local_sum_x = 0.0;
    const bool smooth_loss = WRITE_LOSS && label_smoothing > 0.0f;
    if (vec8) {
        const uint4 *row8 = reinterpret_cast<const uint4 *>(row);
        uint64_t v8 = vocab >> 3;
        for (uint64_t j = threadIdx.x; j < v8; j += blockDim.x) {
            float vals[8];
            unpack_bf16x8(row8[j], vals);
            float tile_m = vals[0];
            #pragma unroll
            for (int k = 1; k < 8; ++k) tile_m = fmaxf(tile_m, vals[k]);
            float new_m = fmaxf(local_m, tile_m);
            local_d = (local_m == -INFINITY) ? 0.0f : local_d * expf(local_m - new_m);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                local_d += expf(vals[k] - new_m);
                if (smooth_loss) local_sum_x += vals[k];
            }
            local_m = new_m;
        }
    } else {
        for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
            float v = __bfloat162float(row[j]);
            float new_m = fmaxf(local_m, v);
            local_d = (local_m == -INFINITY) ? 1.0f : (local_d * expf(local_m - new_m) + expf(v - new_m));
            local_m = new_m;
            if (smooth_loss) local_sum_x += v;
        }
    }

    float m = block_reduce_max_f32(local_m);
    float d_scaled = (local_m == -INFINITY) ? 0.0f : local_d * expf(local_m - m);
    float sum_exp = block_reduce_sum_f32(d_scaled);
    double sum_x = smooth_loss ? block_reduce_sum_f64(local_sum_x) : 0.0;

    if constexpr (WRITE_LOSS) {
        if (threadIdx.x == 0) {
            float log_se = logf(sum_exp);
            const float target = __bfloat162float(row[static_cast<uint64_t>(tok)]);
            float loss = log_se + (m - target);
            if (smooth_loss) {
                const double target_gap = static_cast<double>(m) - target;
                const double mean_gap = static_cast<double>(m) - sum_x / static_cast<double>(vocab);
                const double eps = label_smoothing;
                loss = static_cast<float>(log_se + (1.0 - eps) * target_gap + eps * mean_gap);
            }
            row_losses[r] = loss;
            row_valid[r] = 1;
        }
    }

    if constexpr (WRITE_GRAD) {
        float inv_z = 1.0f / sum_exp;
        float scale = upstream_scalar / static_cast<float>(active);
        float smooth_off = label_smoothing / static_cast<float>(vocab);
        float target_w = 1.0f - label_smoothing;
        uint64_t t_col = static_cast<uint64_t>(tok);
        __nv_bfloat16 *dst = grad_logits + r * vocab;

        if (vec8) {
            const uint4 *row8 = reinterpret_cast<const uint4 *>(row);
            uint4 *dst8 = reinterpret_cast<uint4 *>(dst);
            uint64_t v8 = vocab >> 3;
            for (uint64_t j = threadIdx.x; j < v8; j += blockDim.x) {
                float vals[8], g[8];
                unpack_bf16x8(row8[j], vals);
                uint64_t base = j << 3;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    float prob = expf(vals[k] - m) * inv_z;
                    g[k] = (prob - smooth_off - ((base + k) == t_col ? target_w : 0.0f)) * scale;
                }
                if (accumulate) {
                    float prev[8];
                    unpack_bf16x8(dst8[j], prev);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) g[k] += prev[k];
                }
                dst8[j] = pack_bf16x8(g);
            }
        } else {
            for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
                float prob = expf(__bfloat162float(row[j]) - m) * inv_z;
                float g = (prob - smooth_off - (j == t_col ? target_w : 0.0f)) * scale;
                if (accumulate) g += __bfloat162float(dst[j]);
                dst[j] = __float2bfloat16_rn(g);
            }
        }
    }
}

// --- BF16 Philox4x32-10 Dropout ---

__global__ void bf16_dropout_kernel(
    uint64_t n, float p, float inv_keep, uint2 key,
    uint32_t step_lo, uint32_t step_hi,
    const __nv_bfloat16 *__restrict__ inp,
    __nv_bfloat16 *__restrict__ out,
    int accumulate)
{
    uint64_t n_blocks = (n + 3ULL) >> 2;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t blk = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         blk < n_blocks; blk += stride) {
        uint64_t base = blk << 2;
        if (p == 0.0f) {
            for (uint64_t k = 0; k < 4 && base + k < n; ++k) {
                float v = __bfloat162float(inp[base + k]);
                if (accumulate) v += __bfloat162float(out[base + k]);
                out[base + k] = __float2bfloat16_rn(v);
            }
            continue;
        }
        uint4 ctr = make_uint4(static_cast<uint32_t>(blk), static_cast<uint32_t>(blk >> 32), step_lo, step_hi);
        uint4 r = philox4x32_10_dev(ctr, key);
        uint32_t words[4] = {r.x, r.y, r.z, r.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (base + k < n) {
                float v = (u32_to_unit_f32_dev(words[k]) >= p)
                    ? (__bfloat162float(inp[base + k]) * inv_keep)
                    : 0.0f;
                if (accumulate) v += __bfloat162float(out[base + k]);
                out[base + k] = __float2bfloat16_rn(v);
            }
        }
    }
}

// --- Fused FP32 Master-Weight + BF16 Mirror AdamW ---

__global__ void bf16_master_adamw_param_kernel(
    uint64_t n,
    float *__restrict__ param_f32,
    __nv_bfloat16 *__restrict__ param_bf16,
    const float *__restrict__ grad_f32,
    float *__restrict__ m_f32,
    float *__restrict__ v_f32,
    const AdamwDeviceCtrl *__restrict__ ctrl,
    const uint32_t *__restrict__ found_inf_opt,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int use_grad_scale)
{
    if (found_inf_opt && *found_inf_opt != 0u) return;

    __shared__ float scalars[4];
    if (threadIdx.x == 0) {
        const float lr = ctrl->lr;
        const double t = static_cast<double>(ctrl->step);
        const double bc1 = 1.0 - pow(static_cast<double>(beta1), t);
        const double bc2 = 1.0 - pow(static_cast<double>(beta2), t);
        scalars[0] = use_grad_scale ? ctrl->grad_scale : 1.0f;
        scalars[1] = static_cast<float>(1.0 - static_cast<double>(lr) * static_cast<double>(weight_decay));
        scalars[2] = static_cast<float>(static_cast<double>(lr) / bc1);
        scalars[3] = static_cast<float>(1.0 / sqrt(bc2));
    }
    __syncthreads();

    const float scale = scalars[0];
    const float decay = scalars[1];
    const float step_size = scalars[2];
    const float inv_sqrt_bc2 = scalars[3];
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;

    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    bool vec4 = ((reinterpret_cast<uintptr_t>(param_f32) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(param_bf16) & 7ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(grad_f32) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(m_f32) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(v_f32) & 15ULL) == 0ULL);
    if (vec4) {
        uint64_t n4 = n >> 2;
        float4 *pf4 = reinterpret_cast<float4 *>(param_f32);
        uint2 *pb2 = reinterpret_cast<uint2 *>(param_bf16);
        const float4 *g4 = reinterpret_cast<const float4 *>(grad_f32);
        float4 *m4 = reinterpret_cast<float4 *>(m_f32);
        float4 *v4 = reinterpret_cast<float4 *>(v_f32);
        for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < n4; i += stride) {
            float4 p = pf4[i];
            float4 gv = g4[i];
            float4 mv = m4[i];
            float4 vv = v4[i];
            auto upd = [&](float &pi, float gi, float &mi, float &vi) {
                float g = gi * scale;
                mi = beta1 * mi + one_minus_b1 * g;
                vi = beta2 * vi + one_minus_b2 * (g * g);
                float denom = sqrtf(vi) * inv_sqrt_bc2 + eps;
                pi = pi * decay - step_size * (mi / denom);
            };
            upd(p.x, gv.x, mv.x, vv.x);
            upd(p.y, gv.y, mv.y, vv.y);
            upd(p.z, gv.z, mv.z, vv.z);
            upd(p.w, gv.w, mv.w, vv.w);
            pf4[i] = p;
            m4[i] = mv;
            v4[i] = vv;
            __nv_bfloat162 lo = __floats2bfloat162_rn(p.x, p.y);
            __nv_bfloat162 hi = __floats2bfloat162_rn(p.z, p.w);
            pb2[i] = make_uint2(
                *reinterpret_cast<const uint32_t *>(&lo),
                *reinterpret_cast<const uint32_t *>(&hi));
        }
        uint64_t tail = n4 << 2;
        for (uint64_t i = tail + static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < n; i += stride) {
            float g = grad_f32[i] * scale;
            float mi = beta1 * m_f32[i] + one_minus_b1 * g;
            float vi = beta2 * v_f32[i] + one_minus_b2 * (g * g);
            float denom = sqrtf(vi) * inv_sqrt_bc2 + eps;
            float pi = param_f32[i] * decay - step_size * (mi / denom);
            m_f32[i] = mi;
            v_f32[i] = vi;
            param_f32[i] = pi;
            param_bf16[i] = __float2bfloat16_rn(pi);
        }
        return;
    }

    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        float g = grad_f32[i] * scale;
        float mi = beta1 * m_f32[i] + one_minus_b1 * g;
        float vi = beta2 * v_f32[i] + one_minus_b2 * (g * g);
        float denom = sqrtf(vi) * inv_sqrt_bc2 + eps;
        float pi = param_f32[i] * decay - step_size * (mi / denom);
        m_f32[i] = mi;
        v_f32[i] = vi;
        param_f32[i] = pi;
        param_bf16[i] = __float2bfloat16_rn(pi);
    }
}

// --- Device-side zero-host-sync GradScaler ---

struct GradScalerDev {
    float scale;
    float growth_factor;
    float backoff_factor;
    int32_t growth_interval;
    int32_t growth_tracker;
    uint32_t found_inf;
};

__global__ void grad_scaler_reset_kernel(GradScalerDev *s) {
    if (blockIdx.x == 0 && threadIdx.x == 0) s->found_inf = 0u;
}

__global__ void grad_scaler_unscale_check_kernel(GradScalerDev *s, uint64_t n, float *grad) {
    float inv_scale = 1.0f / s->scale;
    uint32_t bad = 0u;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        float g = grad[i] * inv_scale;
        grad[i] = g;
        if (!isfinite(g)) bad = 1u;
    }
    if (__any_sync(0xffffffffu, bad != 0u)) {
        if ((threadIdx.x & 31u) == 0u) atomicOr(&s->found_inf, 1u);
    }
}

__global__ void grad_scaler_update_kernel(GradScalerDev *s) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (s->found_inf != 0u) {
            s->scale *= s->backoff_factor;
            s->growth_tracker = 0;
        } else {
            s->growth_tracker += 1;
            if (s->growth_tracker >= s->growth_interval) {
                s->scale *= s->growth_factor;
                s->growth_tracker = 0;
            }
        }
    }
}

} // namespace

// --- Extern C Implementations ---

extern "C" int fa_cuda_bf16_gemm(
    Context *ctx, int trans_a, int trans_b, int m, int n, int k,
    const uint16_t *a, int lda, const uint16_t *b, int ldb,
    void *c, int ldc, int c_is_f32, float alpha, float beta)
{
    if (!ctx || m < 0 || n < 0 || k < 0) return rt(cudaErrorInvalidValue);
    if (m == 0 || n == 0) return 0;
    if (!a || !b || !c) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cudaDataType_t c_type = c_is_f32 ? CUDA_R_32F : CUDA_R_16BF;
    return blas_status(cublasGemmEx(
        ctx->blas_tc, op_b, op_a, n, m, k,
        &alpha, b, CUDA_R_16BF, ldb,
        a, CUDA_R_16BF, lda,
        &beta, c, c_type, ldc,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

extern "C" int fa_cuda_bf16_bmm_strided(
    Context *ctx, int trans_a, int trans_b, int batch, int m, int n, int k,
    const uint16_t *a, int lda, int64_t stride_a,
    const uint16_t *b, int ldb, int64_t stride_b,
    void *c, int ldc, int64_t stride_c,
    int c_is_f32, float alpha, float beta)
{
    if (!ctx || batch < 0 || m < 0 || n < 0 || k < 0) return rt(cudaErrorInvalidValue);
    if (batch == 0 || m == 0 || n == 0) return 0;
    if (!a || !b || !c) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cudaDataType_t c_type = c_is_f32 ? CUDA_R_32F : CUDA_R_16BF;
    return blas_status(cublasGemmStridedBatchedEx(
        ctx->blas_tc, op_b, op_a, n, m, k,
        &alpha, b, CUDA_R_16BF, ldb, static_cast<long long>(stride_b),
        a, CUDA_R_16BF, lda, static_cast<long long>(stride_a),
        &beta, c, c_type, ldc, static_cast<long long>(stride_c),
        batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

extern "C" int fa_cuda_tf32_gemm(
    Context *ctx, int trans_a, int trans_b, int m, int n, int k,
    const float *a, int lda, const float *b, int ldb,
    float *c, int ldc, float alpha, float beta)
{
    if (!ctx || m < 0 || n < 0 || k < 0) return rt(cudaErrorInvalidValue);
    if (m == 0 || n == 0) return 0;
    if (!a || !b || !c) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    return blas_status(cublasGemmEx(
        ctx->blas_tc, op_b, op_a, n, m, k,
        &alpha, b, CUDA_R_32F, ldb,
        a, CUDA_R_32F, lda,
        &beta, c, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

extern "C" int fa_cuda_bf16_rmsnorm_fwd(
    Context *ctx, uint64_t rows, uint64_t cols, float eps,
    const uint16_t *x, const uint16_t *weight_opt, uint16_t *y, float *inv_rms_opt)
{
    if (!ctx || cols == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!x || !y) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bool vec8 = ((cols & 7ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(x) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(y) & 15ULL) == 0ULL) &&
                (!weight_opt || ((reinterpret_cast<uintptr_t>(weight_opt) & 15ULL) == 0ULL));
    if (vec8) {
        unsigned threads = (cols <= 512) ? 64u : ((cols <= 1024) ? 128u : 256u);
        bf16_rmsnorm_fwd_vec8_kernel<false><<<static_cast<unsigned>(rows), threads, 0, ctx->stream>>>(
            rows, cols, eps,
            reinterpret_cast<const __nv_bfloat16 *>(x), nullptr,
            reinterpret_cast<const __nv_bfloat16 *>(weight_opt),
            nullptr, reinterpret_cast<__nv_bfloat16 *>(y), inv_rms_opt);
    } else {
        bf16_rmsnorm_fwd_scalar_kernel<false><<<static_cast<unsigned>(rows), 256, 0, ctx->stream>>>(
            rows, cols, eps,
            reinterpret_cast<const __nv_bfloat16 *>(x), nullptr,
            reinterpret_cast<const __nv_bfloat16 *>(weight_opt),
            nullptr, reinterpret_cast<__nv_bfloat16 *>(y), inv_rms_opt);
    }
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_residual_rmsnorm_fwd(
    Context *ctx, uint64_t rows, uint64_t cols, float eps,
    const uint16_t *a, const uint16_t *b, const uint16_t *weight_opt,
    uint16_t *residual_out, uint16_t *y, float *inv_rms_opt)
{
    if (!ctx || cols == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!a || !b || !residual_out || !y) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bool vec8 = ((cols & 7ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(a) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(b) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(residual_out) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(y) & 15ULL) == 0ULL) &&
                (!weight_opt || ((reinterpret_cast<uintptr_t>(weight_opt) & 15ULL) == 0ULL));
    if (vec8) {
        unsigned threads = (cols <= 512) ? 64u : ((cols <= 1024) ? 128u : 256u);
        bf16_rmsnorm_fwd_vec8_kernel<true><<<static_cast<unsigned>(rows), threads, 0, ctx->stream>>>(
            rows, cols, eps,
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            reinterpret_cast<const __nv_bfloat16 *>(weight_opt),
            reinterpret_cast<__nv_bfloat16 *>(residual_out),
            reinterpret_cast<__nv_bfloat16 *>(y), inv_rms_opt);
    } else {
        bf16_rmsnorm_fwd_scalar_kernel<true><<<static_cast<unsigned>(rows), 256, 0, ctx->stream>>>(
            rows, cols, eps,
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            reinterpret_cast<const __nv_bfloat16 *>(weight_opt),
            reinterpret_cast<__nv_bfloat16 *>(residual_out),
            reinterpret_cast<__nv_bfloat16 *>(y), inv_rms_opt);
    }
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_rmsnorm_bwd(
    Context *ctx, uint64_t rows, uint64_t cols, float eps,
    const uint16_t *x, const uint16_t *weight_opt, const float *inv_rms_opt,
    const uint16_t *dy, uint16_t *dx, float *dweight_opt,
    float *dweight_partials_scratch, int accumulate_dx, int accumulate_dw)
{
    if (!ctx || cols == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!x || !dy || !dx) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bool vec8 = ((cols & 7ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(x) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(dy) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(dx) & 15ULL) == 0ULL) &&
                (!weight_opt || ((reinterpret_cast<uintptr_t>(weight_opt) & 15ULL) == 0ULL));
    if (vec8) {
        unsigned threads = (cols <= 512) ? 64u : ((cols <= 1024) ? 128u : 256u);
        bf16_rmsnorm_bwd_dx_vec8_kernel<<<static_cast<unsigned>(rows), threads, 0, ctx->stream>>>(
            rows, cols, eps,
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(weight_opt),
            inv_rms_opt,
            reinterpret_cast<const __nv_bfloat16 *>(dy),
            reinterpret_cast<__nv_bfloat16 *>(dx),
            accumulate_dx);
    } else {
        bf16_rmsnorm_bwd_dx_scalar_kernel<<<static_cast<unsigned>(rows), 256, 0, ctx->stream>>>(
            rows, cols, eps,
            reinterpret_cast<const __nv_bfloat16 *>(x),
            reinterpret_cast<const __nv_bfloat16 *>(weight_opt),
            inv_rms_opt,
            reinterpret_cast<const __nv_bfloat16 *>(dy),
            reinterpret_cast<__nv_bfloat16 *>(dx),
            accumulate_dx);
    }
    code = rt(cudaGetLastError());
    if (code) return code;

    if (dweight_opt) {
        if (!inv_rms_opt || !dweight_partials_scratch) return rt(cudaErrorInvalidValue);
        unsigned chunks = static_cast<unsigned>(umin64(rows, 64ULL));
        dim3 grid_p(static_cast<unsigned>((cols + 127ULL) / 128ULL), chunks);
        bf16_rmsnorm_bwd_dw_partial_kernel<<<grid_p, 128, 0, ctx->stream>>>(
            rows, cols,
            reinterpret_cast<const __nv_bfloat16 *>(x), inv_rms_opt,
            reinterpret_cast<const __nv_bfloat16 *>(dy),
            dweight_partials_scratch);
        code = rt(cudaGetLastError());
        if (code) return code;
        unsigned grid_f = static_cast<unsigned>((cols + 127ULL) / 128ULL);
        bf16_rmsnorm_bwd_dw_finalize_kernel<<<grid_f, 128, 0, ctx->stream>>>(
            chunks, cols, dweight_partials_scratch, dweight_opt, accumulate_dw);
        code = rt(cudaGetLastError());
    }
    return code;
}

extern "C" int fa_cuda_bf16_layernorm_fwd(
    Context *ctx, uint64_t rows, uint64_t cols, float eps, const uint16_t *x, uint16_t *y)
{
    if (!ctx || cols == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!x || !y) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bf16_layernorm_fwd_kernel<<<static_cast<unsigned>(rows), 256, 0, ctx->stream>>>(
        rows, cols, eps, reinterpret_cast<const __nv_bfloat16 *>(x), reinterpret_cast<__nv_bfloat16 *>(y));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_layernorm_bwd(
    Context *ctx, uint64_t rows, uint64_t cols, float eps,
    const uint16_t *x, const uint16_t *dy, uint16_t *dx, int accumulate)
{
    if (!ctx || cols == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!x || !dy || !dx) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bf16_layernorm_bwd_kernel<<<static_cast<unsigned>(rows), 256, 0, ctx->stream>>>(
        rows, cols, eps, reinterpret_cast<const __nv_bfloat16 *>(x),
        reinterpret_cast<const __nv_bfloat16 *>(dy), reinterpret_cast<__nv_bfloat16 *>(dx), accumulate);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_softmax_fwd(
    Context *ctx, uint64_t rows, uint64_t cols, const uint16_t *x, uint16_t *y)
{
    if (!ctx || cols == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!x || !y) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bf16_softmax_fwd_kernel<<<static_cast<unsigned>(rows), 256, 0, ctx->stream>>>(
        rows, cols, reinterpret_cast<const __nv_bfloat16 *>(x), reinterpret_cast<__nv_bfloat16 *>(y));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_softmax_bwd(
    Context *ctx, uint64_t rows, uint64_t cols,
    const uint16_t *y, const uint16_t *dy, uint16_t *dx, int accumulate)
{
    if (!ctx || cols == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!y || !dy || !dx) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bf16_softmax_bwd_kernel<<<static_cast<unsigned>(rows), 256, 0, ctx->stream>>>(
        rows, cols, reinterpret_cast<const __nv_bfloat16 *>(y),
        reinterpret_cast<const __nv_bfloat16 *>(dy), reinterpret_cast<__nv_bfloat16 *>(dx), accumulate);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_swiglu_fwd(
    Context *ctx, uint64_t rows, uint64_t hidden, const uint16_t *gate_up, uint16_t *out)
{
    if (!ctx || hidden == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!gate_up || !out) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bool vec8 = ((hidden & 7ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(gate_up) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(out) & 15ULL) == 0ULL);
    if (vec8) {
        uint64_t total8 = rows * (hidden >> 3);
        unsigned g = static_cast<unsigned>(umin64((total8 + 255ULL) / 256ULL, 65535ULL));
        bf16_swiglu_fwd_vec8_kernel<<<g, 256, 0, ctx->stream>>>(
            rows, hidden, reinterpret_cast<const __nv_bfloat16 *>(gate_up),
            reinterpret_cast<__nv_bfloat16 *>(out));
    } else {
        uint64_t total = rows * hidden;
        unsigned g = static_cast<unsigned>(umin64((total + 255ULL) / 256ULL, 65535ULL));
        bf16_swiglu_fwd_scalar_kernel<<<g, 256, 0, ctx->stream>>>(
            rows, hidden, reinterpret_cast<const __nv_bfloat16 *>(gate_up),
            reinterpret_cast<__nv_bfloat16 *>(out));
    }
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_swiglu_bwd(
    Context *ctx, uint64_t rows, uint64_t hidden,
    const uint16_t *gate_up, const uint16_t *dout, uint16_t *dgate_up)
{
    if (!ctx || hidden == 0) return rt(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    if (!gate_up || !dout || !dgate_up) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bool vec8 = ((hidden & 7ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(gate_up) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(dout) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(dgate_up) & 15ULL) == 0ULL);
    if (vec8) {
        uint64_t total8 = rows * (hidden >> 3);
        unsigned g = static_cast<unsigned>(umin64((total8 + 255ULL) / 256ULL, 65535ULL));
        bf16_swiglu_bwd_vec8_kernel<<<g, 256, 0, ctx->stream>>>(
            rows, hidden, reinterpret_cast<const __nv_bfloat16 *>(gate_up),
            reinterpret_cast<const __nv_bfloat16 *>(dout),
            reinterpret_cast<__nv_bfloat16 *>(dgate_up));
    } else {
        uint64_t total = rows * hidden;
        unsigned g = static_cast<unsigned>(umin64((total + 255ULL) / 256ULL, 65535ULL));
        bf16_swiglu_bwd_scalar_kernel<<<g, 256, 0, ctx->stream>>>(
            rows, hidden, reinterpret_cast<const __nv_bfloat16 *>(gate_up),
            reinterpret_cast<const __nv_bfloat16 *>(dout),
            reinterpret_cast<__nv_bfloat16 *>(dgate_up));
    }
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_add(
    Context *ctx, uint64_t n, const uint16_t *a, const uint16_t *b, uint16_t *out)
{
    if (!ctx) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!a || !b || !out) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t work = (n + 7ULL) >> 3;
    unsigned g = static_cast<unsigned>(umin64((work + 255ULL) / 256ULL, 65535ULL));
    bf16_add_kernel<<<g, 256, 0, ctx->stream>>>(
        n, reinterpret_cast<const __nv_bfloat16 *>(a),
        reinterpret_cast<const __nv_bfloat16 *>(b),
        reinterpret_cast<__nv_bfloat16 *>(out));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_rope_run(
    Context *ctx,
    uint64_t batch, uint64_t heads, uint64_t seq_len, uint64_t head_dim, uint64_t rotary_dim,
    int64_t in_sb, int64_t in_sh, int64_t in_st,
    int64_t out_sb, int64_t out_sh, int64_t out_st,
    const float *cos_dev, const float *sin_dev,
    const uint16_t *inp, uint16_t *out, int inverse, int accumulate)
{
    if (!ctx || head_dim == 0 || rotary_dim == 0 || (rotary_dim & 1ULL) || rotary_dim > head_dim) {
        return rt(cudaErrorInvalidValue);
    }
    uint64_t rows = batch * heads * seq_len;
    if (rows == 0) return 0;
    if (!cos_dev || !sin_dev || !inp || !out) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t work = (rotary_dim >> 1) + (head_dim - rotary_dim);
    uint64_t total = rows * work;
    unsigned g = static_cast<unsigned>(umin64((total + 255ULL) / 256ULL, 65535ULL));
    bf16_rope_kernel<<<g, 256, 0, ctx->stream>>>(
        batch, heads, seq_len, head_dim, rotary_dim,
        in_sb, in_sh, in_st, out_sb, out_sh, out_st,
        cos_dev, sin_dev,
        reinterpret_cast<const __nv_bfloat16 *>(inp),
        reinterpret_cast<__nv_bfloat16 *>(out),
        inverse, accumulate);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_embedding_fwd(
    Context *ctx, uint64_t vocab_size, uint64_t embed_dim, uint64_t num_indices,
    const uint16_t *table_bf16, const int64_t *indices, uint16_t *out_bf16)
{
    if (!ctx || vocab_size == 0 || embed_dim == 0) return rt(cudaErrorInvalidValue);
    if (num_indices == 0) return 0;
    if (!table_bf16 || !indices || !out_bf16) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t work = ((embed_dim & 7ULL) == 0ULL) ? (num_indices * (embed_dim >> 3)) : (num_indices * embed_dim);
    unsigned g = static_cast<unsigned>(umin64((work + 255ULL) / 256ULL, 65535ULL));
    bf16_embedding_fwd_kernel<<<g, 256, 0, ctx->stream>>>(
        vocab_size, embed_dim, num_indices,
        reinterpret_cast<const __nv_bfloat16 *>(table_bf16),
        indices, reinterpret_cast<__nv_bfloat16 *>(out_bf16));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_embedding_bwd_f32(
    Context *ctx, uint64_t vocab_size, uint64_t embed_dim, uint64_t num_indices,
    int64_t padding_idx, int has_padding_idx,
    const int64_t *indices, const uint16_t *upstream_bf16, float *grad_table_f32, int accumulate)
{
    if (!ctx || vocab_size == 0 || embed_dim == 0 || !grad_table_f32) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    if (num_indices == 0) {
        if (!accumulate) {
            return rt(cudaMemsetAsync(
                grad_table_f32, 0, static_cast<size_t>(vocab_size * embed_dim) * sizeof(float), ctx->stream));
        }
        return 0;
    }
    if (!indices || !upstream_bf16) return rt(cudaErrorInvalidValue);
    if (vocab_size <= 32768ULL && vocab_size * num_indices <= 134217728ULL) {
        unsigned g = static_cast<unsigned>((vocab_size + BF16_EMB_ROWS - 1) / BF16_EMB_ROWS);
        bf16_embedding_bwd_f32_det_kernel<<<g, 256, 0, ctx->stream>>>(
            vocab_size, embed_dim, num_indices, padding_idx, has_padding_idx,
            indices, reinterpret_cast<const __nv_bfloat16 *>(upstream_bf16),
            grad_table_f32, accumulate);
        return rt(cudaGetLastError());
    }
    if (!accumulate) {
        code = rt(cudaMemsetAsync(
            grad_table_f32, 0, static_cast<size_t>(vocab_size * embed_dim) * sizeof(float), ctx->stream));
        if (code) return code;
    }
    uint64_t total = num_indices * embed_dim;
    unsigned g = static_cast<unsigned>(umin64((total + 255ULL) / 256ULL, 65535ULL));
    bf16_embedding_bwd_f32_atomic_kernel<<<g, 256, 0, ctx->stream>>>(
        vocab_size, embed_dim, num_indices, padding_idx, has_padding_idx,
        indices, reinterpret_cast<const __nv_bfloat16 *>(upstream_bf16), grad_table_f32);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_cross_entropy_fwd(
    Context *ctx, uint64_t num_rows, uint64_t vocab_size,
    int64_t ignore_index, int has_ignore_index, float label_smoothing,
    const uint16_t *logits_bf16, const int64_t *targets,
    float *row_losses_scratch, int32_t *row_valid_scratch, float *out_scalar_f32)
{
    if (!ctx || vocab_size == 0 || !out_scalar_f32) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    if (num_rows == 0) return rt(cudaMemsetAsync(out_scalar_f32, 0, sizeof(float), ctx->stream));
    if (!logits_bf16 || !targets || !row_losses_scratch || !row_valid_scratch) return rt(cudaErrorInvalidValue);
    bf16_ce_rows_kernel<true, false><<<static_cast<unsigned>(num_rows), 256, 0, ctx->stream>>>(
        num_rows, vocab_size, ignore_index, has_ignore_index, label_smoothing, 1.0f,
        reinterpret_cast<const __nv_bfloat16 *>(logits_bf16), targets, nullptr,
        row_losses_scratch, row_valid_scratch, nullptr, 0);
    code = rt(cudaGetLastError());
    if (code) return code;
    bf16_ce_reduce_loss_kernel<<<1, 256, 0, ctx->stream>>>(
        num_rows, row_losses_scratch, row_valid_scratch, out_scalar_f32);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_cross_entropy_bwd(
    Context *ctx, uint64_t num_rows, uint64_t vocab_size,
    int64_t ignore_index, int has_ignore_index, float label_smoothing, float upstream_scalar,
    const uint16_t *logits_bf16, const int64_t *targets,
    int32_t *active_count_scratch, uint16_t *grad_logits_bf16, int accumulate)
{
    if (!ctx || vocab_size == 0) return rt(cudaErrorInvalidValue);
    if (num_rows == 0) return 0;
    if (!logits_bf16 || !targets || !active_count_scratch || !grad_logits_bf16) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    bf16_ce_count_valid_kernel<<<1, 256, 0, ctx->stream>>>(
        num_rows, vocab_size, ignore_index, has_ignore_index, targets, active_count_scratch);
    code = rt(cudaGetLastError());
    if (code) return code;
    bf16_ce_rows_kernel<false, true><<<static_cast<unsigned>(num_rows), 256, 0, ctx->stream>>>(
        num_rows, vocab_size, ignore_index, has_ignore_index, label_smoothing, upstream_scalar,
        reinterpret_cast<const __nv_bfloat16 *>(logits_bf16), targets, active_count_scratch,
        nullptr, nullptr, reinterpret_cast<__nv_bfloat16 *>(grad_logits_bf16), accumulate);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_cross_entropy_fwd_bwd(
    Context *ctx, uint64_t num_rows, uint64_t vocab_size,
    int64_t ignore_index, int has_ignore_index, float label_smoothing, float upstream_scalar,
    const uint16_t *logits_bf16, const int64_t *targets,
    int32_t *active_count_scratch, float *row_losses_scratch, int32_t *row_valid_scratch,
    float *out_scalar_f32, uint16_t *grad_logits_bf16)
{
    if (!ctx || vocab_size == 0 || !out_scalar_f32) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    if (num_rows == 0) return rt(cudaMemsetAsync(out_scalar_f32, 0, sizeof(float), ctx->stream));
    if (!logits_bf16 || !targets || !active_count_scratch || !row_losses_scratch ||
        !row_valid_scratch || !grad_logits_bf16) {
        return rt(cudaErrorInvalidValue);
    }
    bf16_ce_count_valid_kernel<<<1, 256, 0, ctx->stream>>>(
        num_rows, vocab_size, ignore_index, has_ignore_index, targets, active_count_scratch);
    code = rt(cudaGetLastError());
    if (code) return code;
    bf16_ce_rows_kernel<true, true><<<static_cast<unsigned>(num_rows), 256, 0, ctx->stream>>>(
        num_rows, vocab_size, ignore_index, has_ignore_index, label_smoothing, upstream_scalar,
        reinterpret_cast<const __nv_bfloat16 *>(logits_bf16), targets, active_count_scratch,
        row_losses_scratch, row_valid_scratch,
        reinterpret_cast<__nv_bfloat16 *>(grad_logits_bf16), 0);
    code = rt(cudaGetLastError());
    if (code) return code;
    bf16_ce_reduce_loss_kernel<<<1, 256, 0, ctx->stream>>>(
        num_rows, row_losses_scratch, row_valid_scratch, out_scalar_f32);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_dropout_fwd(
    Context *ctx, uint64_t n, float p, uint64_t seed, uint64_t step,
    const uint16_t *input, uint16_t *output)
{
    if (!ctx || !std::isfinite(p) || p < 0.0f || p >= 1.0f) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!input || !output) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t n_blocks = (n + 3ULL) >> 2;
    unsigned g = static_cast<unsigned>(umin64((n_blocks + 255ULL) / 256ULL, 65535ULL));
    uint2 key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
    bf16_dropout_kernel<<<g, 256, 0, ctx->stream>>>(
        n, p, 1.0f / (1.0f - p), key,
        static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32),
        reinterpret_cast<const __nv_bfloat16 *>(input),
        reinterpret_cast<__nv_bfloat16 *>(output), 0);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_dropout_bwd(
    Context *ctx, uint64_t n, float p, uint64_t seed, uint64_t step,
    const uint16_t *upstream, uint16_t *output, int accumulate)
{
    if (!ctx || !std::isfinite(p) || p < 0.0f || p >= 1.0f) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!upstream || !output) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t n_blocks = (n + 3ULL) >> 2;
    unsigned g = static_cast<unsigned>(umin64((n_blocks + 255ULL) / 256ULL, 65535ULL));
    uint2 key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
    bf16_dropout_kernel<<<g, 256, 0, ctx->stream>>>(
        n, p, 1.0f / (1.0f - p), key,
        static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32),
        reinterpret_cast<const __nv_bfloat16 *>(upstream),
        reinterpret_cast<__nv_bfloat16 *>(output), accumulate);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_bf16_adamw_param(
    Context *ctx, uint64_t n,
    float *param_f32, uint16_t *param_bf16, const float *grad_f32,
    float *m_f32, float *v_f32, void *ctrl, const uint32_t *found_inf_opt,
    float beta1, float beta2, float eps, float weight_decay, int use_grad_scale)
{
    if (!ctx || !ctrl) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!param_f32 || !param_bf16 || !grad_f32 || !m_f32 || !v_f32) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t work = (n + 3ULL) >> 2;
    unsigned g = static_cast<unsigned>(umin64((work + 255ULL) / 256ULL, 65535ULL));
    bf16_master_adamw_param_kernel<<<g, 256, 0, ctx->stream>>>(
        n, param_f32, reinterpret_cast<__nv_bfloat16 *>(param_bf16), grad_f32,
        m_f32, v_f32, static_cast<const AdamwDeviceCtrl *>(ctrl), found_inf_opt,
        beta1, beta2, eps, weight_decay, use_grad_scale);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_grad_scaler_create(
    Context *ctx, float init_scale, float growth_factor, float backoff_factor,
    int growth_interval, void **out_scaler)
{
    if (!ctx || !out_scaler || !std::isfinite(init_scale) || init_scale <= 0.0f ||
        !std::isfinite(growth_factor) || growth_factor <= 1.0f ||
        !std::isfinite(backoff_factor) || backoff_factor <= 0.0f || backoff_factor >= 1.0f ||
        growth_interval <= 0) {
        return rt(cudaErrorInvalidValue);
    }
    *out_scaler = nullptr;
    int code = activate(ctx);
    if (code) return code;
    void *dev = nullptr;
    code = rt(cudaMalloc(&dev, sizeof(GradScalerDev)));
    if (code) return code;
    GradScalerDev init{init_scale, growth_factor, backoff_factor, growth_interval, 0, 0u};
    code = rt(cudaMemcpyAsync(dev, &init, sizeof(GradScalerDev), cudaMemcpyHostToDevice, ctx->stream));
    if (!code) code = rt(cudaStreamSynchronize(ctx->stream));
    if (code) { cudaFree(dev); return code; }
    *out_scaler = dev;
    return 0;
}

extern "C" void fa_cuda_grad_scaler_destroy(Context *ctx, void *scaler) {
    if (!ctx || !scaler) return;
    cudaSetDevice(ctx->device);
    cudaStreamSynchronize(ctx->stream);
    cudaFree(scaler);
}

extern "C" int fa_cuda_grad_scaler_reset(Context *ctx, void *scaler) {
    if (!ctx || !scaler) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    grad_scaler_reset_kernel<<<1, 1, 0, ctx->stream>>>(static_cast<GradScalerDev *>(scaler));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_grad_scaler_unscale_check(Context *ctx, void *scaler, uint64_t n, float *grad_f32) {
    if (!ctx || !scaler) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!grad_f32) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    unsigned g = static_cast<unsigned>(umin64((n + 255ULL) / 256ULL, 65535ULL));
    grad_scaler_unscale_check_kernel<<<g, 256, 0, ctx->stream>>>(
        static_cast<GradScalerDev *>(scaler), n, grad_f32);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_grad_scaler_update(Context *ctx, void *scaler) {
    if (!ctx || !scaler) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    grad_scaler_update_kernel<<<1, 1, 0, ctx->stream>>>(static_cast<GradScalerDev *>(scaler));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_grad_scaler_read_state(
    Context *ctx, void *scaler, float *out_scale, uint32_t *out_found_inf)
{
    if (!ctx || !scaler || !out_scale || !out_found_inf) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    GradScalerDev host{};
    code = rt(cudaMemcpyAsync(&host, scaler, sizeof(GradScalerDev), cudaMemcpyDeviceToHost, ctx->stream));
    if (!code) code = rt(cudaStreamSynchronize(ctx->stream));
    if (code) return code;
    *out_scale = host.scale;
    *out_found_inf = host.found_inf;
    return 0;
}

extern "C" const uint32_t *fa_cuda_grad_scaler_found_inf_ptr(void *scaler) {
    if (!scaler) return nullptr;
    return &static_cast<const GradScalerDev *>(scaler)->found_inf;
}
