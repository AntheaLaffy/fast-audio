#include "cuda_backend.h"
#include "softmax.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>

namespace {

struct SoftmaxPlanCuda {
    Context *ctx;
    FaSoftmaxDims dims;
    FaSoftmaxAxis *group_axes = nullptr;
    SoftmaxPlanCuda(Context *c, FaSoftmaxDims d) : ctx(c), dims(d) {}
    ~SoftmaxPlanCuda() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (group_axes) cudaFree(group_axes);
    }
};

__device__ __forceinline__ float max_nan_sm(float a, float b) {
#if __CUDA_ARCH__ >= 800
    float r;
    asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    if (isnan(a) || isnan(b)) return __uint_as_float(0x7fc00000u);
    if (a > b) return a;
    if (b > a) return b;
    return (!signbit(a)) ? a : b;
#endif
}

__device__ __forceinline__ float4 ldcs_sm_f4(const float4 *p) {
    float4 v;
    asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = max_nan_sm(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < 8) ? smem[lane] : -INFINITY;
        x = warp_reduce_max(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    const float res = smem[0];
    __syncthreads();
    return res;
}

__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < 8) ? smem[lane] : 0.0f;
        x = warp_reduce_sum(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    const float res = smem[0];
    __syncthreads();
    return res;
}

__device__ __forceinline__ void block_reduce_sum2(float v0, float v1, float *smem,
                                                  float &out0, float &out1) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v0 = warp_reduce_sum(v0);
    v1 = warp_reduce_sum(v1);
    if (lane == 0) {
        smem[warp] = v0;
        smem[8 + warp] = v1;
    }
    __syncthreads();
    if (warp == 0) {
        float x0 = (lane < 8) ? smem[lane] : 0.0f;
        float x1 = (lane < 8) ? smem[8 + lane] : 0.0f;
        x0 = warp_reduce_sum(x0);
        x1 = warp_reduce_sum(x1);
        if (lane == 0) {
            smem[0] = x0;
            smem[8] = x1;
        }
    }
    __syncthreads();
    out0 = smem[0];
    out1 = smem[8];
    __syncthreads();
}

__device__ __forceinline__ void group_base2(
    FaSoftmaxDims d, const FaSoftmaxAxis *axes, uint64_t g, int64_t &ox, int64_t &oy) {
    ox = d.offsets[0];
    oy = d.offsets[1];
    for (int a = d.group_rank - 1; a >= 0; --a) {
        const int64_t c = static_cast<int64_t>(g % axes[a].dim);
        g /= axes[a].dim;
        ox += c * axes[a].strides[0];
        oy += c * axes[a].strides[1];
    }
}

__device__ __forceinline__ void group_base3(
    FaSoftmaxDims d, const FaSoftmaxAxis *axes, uint64_t g,
    int64_t &ox, int64_t &ody, int64_t &odx) {
    ox = d.offsets[0];
    ody = d.offsets[1];
    odx = d.offsets[2];
    for (int a = d.group_rank - 1; a >= 0; --a) {
        const int64_t c = static_cast<int64_t>(g % axes[a].dim);
        g /= axes[a].dim;
        ox += c * axes[a].strides[0];
        ody += c * axes[a].strides[1];
        odx += c * axes[a].strides[2];
    }
}

// Tier 1: Warp-per-row (8 warps = 256 threads/block, 0 shared memory, 0 __syncthreads(), D <= 128).
// Uses compile-time unrolled register indexing (`#pragma unroll`) so `reg[4]` never spills to local memory.
template <bool log_sm>
__global__ void softmax_fwd_warp(FaSoftmaxDims d, const float *__restrict__ x, float *__restrict__ y) {
    const uint64_t D = d.axis_len;
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        const float *xg = x + d.offsets[0] + g * D;
        float *yg = y + d.offsets[1] + g * D;
        float reg[4];
        float m = -INFINITY;
        bool has_v = false;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
            if (i < D) {
                const float v = xg[i];
                reg[k] = v;
                m = has_v ? max_nan_sm(m, v) : v;
                has_v = true;
            }
        }
        m = warp_reduce_max(m);
        if (isnan(m)) {
            for (uint64_t i = lane; i < D; i += 32) yg[i] = __uint_as_float(0x7fc00000u);
            continue;
        }
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
            if (i < D) {
                const float e = expf(reg[k] - m);
                if constexpr (!log_sm) reg[k] = e;
                sum += e;
            }
        }
        sum = warp_reduce_sum(sum);
        if constexpr (log_sm) {
            const float log_z = m + logf(sum);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
                if (i < D) yg[i] = reg[k] - log_z;
            }
        } else {
            const float inv_sum = 1.0f / sum;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
                if (i < D) yg[i] = reg[k] * inv_sum;
            }
        }
    }
}

// Tier 1B: Vectorized float4 Warp-per-row (8 warps = 256 threads/block, 0 smem, 0 barriers, D <= 512, D % 4 == 0).
template <bool log_sm, int NV4, bool exact, bool stream_x>
__global__ void softmax_fwd_warp_vec4(FaSoftmaxDims d, const float *__restrict__ x, float *__restrict__ y) {
    const uint64_t D = d.axis_len;
    const uint64_t D4 = D >> 2;
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const float4 *x4 = reinterpret_cast<const float4 *>(x + d.offsets[0]);
    float4 *y4 = reinterpret_cast<float4 *>(y + d.offsets[1]);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        const float4 *xg4 = x4 + g * D4;
        float4 *yg4 = y4 + g * D4;
        float4 reg[NV4];
        float m = -INFINITY;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            if (exact || idx4 < D4) {
                const float4 v = stream_x ? ldcs_sm_f4(xg4 + idx4) : xg4[idx4];
                reg[k] = v;
                m = max_nan_sm(m, max_nan_sm(max_nan_sm(v.x, v.y), max_nan_sm(v.z, v.w)));
            }
        }
        m = warp_reduce_max(m);
        if (isnan(m)) {
            const float qnan = __uint_as_float(0x7fc00000u);
            const float4 nan4 = make_float4(qnan, qnan, qnan, qnan);
            #pragma unroll
            for (int k = 0; k < NV4; ++k) {
                const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
                if (exact || idx4 < D4) yg4[idx4] = nan4;
            }
            continue;
        }
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            if (exact || idx4 < D4) {
                const float ex = expf(reg[k].x - m);
                const float ey = expf(reg[k].y - m);
                const float ez = expf(reg[k].z - m);
                const float ew = expf(reg[k].w - m);
                if constexpr (!log_sm) reg[k] = make_float4(ex, ey, ez, ew);
                sum += (ex + ey) + (ez + ew);
            }
        }
        sum = warp_reduce_sum(sum);
        if constexpr (log_sm) {
            const float log_z = m + logf(sum);
            #pragma unroll
            for (int k = 0; k < NV4; ++k) {
                const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
                if (exact || idx4 < D4) {
                    yg4[idx4] = make_float4(
                        reg[k].x - log_z, reg[k].y - log_z, reg[k].z - log_z, reg[k].w - log_z);
                }
            }
        } else {
            const float inv_sum = 1.0f / sum;
            #pragma unroll
            for (int k = 0; k < NV4; ++k) {
                const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
                if (exact || idx4 < D4) {
                    yg4[idx4] = make_float4(
                        reg[k].x * inv_sum, reg[k].y * inv_sum, reg[k].z * inv_sum, reg[k].w * inv_sum);
                }
            }
        }
    }
}

// Tier 2: Block-per-row register-cached (256 threads/block, 32B smem, 128 < D <= 1024).
template <bool log_sm>
__global__ void softmax_fwd_block_reg(FaSoftmaxDims d, const float *__restrict__ x, float *__restrict__ y) {
    __shared__ float smem[8];
    const uint64_t D = d.axis_len;
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        const float *xg = x + d.offsets[0] + g * D;
        float *yg = y + d.offsets[1] + g * D;
        float reg[4];
        float m = -INFINITY;
        bool has_v = false;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (i < D) {
                const float v = xg[i];
                reg[k] = v;
                m = has_v ? max_nan_sm(m, v) : v;
                has_v = true;
            }
        }
        m = block_reduce_max(m, smem);
        if (isnan(m)) {
            for (uint64_t i = threadIdx.x; i < D; i += 256) yg[i] = __uint_as_float(0x7fc00000u);
            continue;
        }
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (i < D) {
                const float e = expf(reg[k] - m);
                if constexpr (!log_sm) reg[k] = e;
                sum += e;
            }
        }
        sum = block_reduce_sum(sum, smem);
        if constexpr (log_sm) {
            const float log_z = m + logf(sum);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
                if (i < D) yg[i] = reg[k] - log_z;
            }
        } else {
            const float inv_sum = 1.0f / sum;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
                if (i < D) yg[i] = reg[k] * inv_sum;
            }
        }
    }
}

// Tier 3: General forward kernel (arbitrary D > 1024, arbitrary axis, arbitrary signed strides).
template <bool log_sm, bool contig_last>
__global__ void softmax_fwd_general(FaSoftmaxDims d, const FaSoftmaxAxis *axes,
                                    const float *__restrict__ x, float *__restrict__ y) {
    __shared__ float smem[8];
    const uint64_t D = d.axis_len;
    const int64_t sx = d.axis_strides[0];
    const int64_t sy = d.axis_strides[1];
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        int64_t ox, oy;
        if constexpr (contig_last) {
            ox = d.offsets[0] + static_cast<int64_t>(g * D);
            oy = d.offsets[1] + static_cast<int64_t>(g * D);
        } else {
            group_base2(d, axes, g, ox, oy);
        }
        float m = -INFINITY;
        bool has_val = false;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            const float v = contig_last ? x[ox + static_cast<int64_t>(i)]
                                        : x[ox + static_cast<int64_t>(i) * sx];
            m = has_val ? max_nan_sm(m, v) : v;
            has_val = true;
        }
        m = block_reduce_max(m, smem);
        if (isnan(m)) {
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                if constexpr (contig_last) y[oy + static_cast<int64_t>(i)] = __uint_as_float(0x7fc00000u);
                else y[oy + static_cast<int64_t>(i) * sy] = __uint_as_float(0x7fc00000u);
            }
            continue;
        }
        float sum = 0.0f;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            const float v = contig_last ? x[ox + static_cast<int64_t>(i)]
                                        : x[ox + static_cast<int64_t>(i) * sx];
            sum += expf(v - m);
        }
        sum = block_reduce_sum(sum, smem);
        if constexpr (log_sm) {
            const float log_z = m + logf(sum);
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                if constexpr (contig_last) {
                    y[oy + static_cast<int64_t>(i)] = x[ox + static_cast<int64_t>(i)] - log_z;
                } else {
                    const int64_t ii = static_cast<int64_t>(i);
                    y[oy + ii * sy] = x[ox + ii * sx] - log_z;
                }
            }
        } else {
            const float inv_sum = 1.0f / sum;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                if constexpr (contig_last) {
                    y[oy + static_cast<int64_t>(i)] = expf(x[ox + static_cast<int64_t>(i)] - m) * inv_sum;
                } else {
                    const int64_t ii = static_cast<int64_t>(i);
                    y[oy + ii * sy] = expf(x[ox + ii * sx] - m) * inv_sum;
                }
            }
        }
    }
}

// Tier 1 Backward: Warp-per-row (D <= 128, 0 smem, 0 __syncthreads(), 1 read of x and dy).
template <bool log_sm, bool accumulate>
__global__ void softmax_bwd_warp(FaSoftmaxDims d,
                                 const float *__restrict__ x,
                                 const float *__restrict__ dy,
                                 float *__restrict__ dx) {
    const uint64_t D = d.axis_len;
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        const float *xg = x + d.offsets[0] + g * D;
        const float *dyg = dy + d.offsets[1] + g * D;
        float *dxg = dx + d.offsets[2] + g * D;
        float rx[4], rdy[4];
        float m = -INFINITY;
        bool has_v = false;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
            if (i < D) {
                const float vx = xg[i];
                rx[k] = vx;
                rdy[k] = dyg[i];
                m = has_v ? max_nan_sm(m, vx) : vx;
                has_v = true;
            }
        }
        m = warp_reduce_max(m);
        if (isnan(m)) {
            for (uint64_t i = lane; i < D; i += 32) {
                if constexpr (accumulate) dxg[i] += __uint_as_float(0x7fc00000u);
                else dxg[i] = __uint_as_float(0x7fc00000u);
            }
            continue;
        }
        float sum_e = 0.0f;
        float red2 = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
            if (i < D) {
                const float e = expf(rx[k] - m);
                rx[k] = e;
                sum_e += e;
                red2 += log_sm ? rdy[k] : (rdy[k] * e);
            }
        }
        sum_e = warp_reduce_sum(sum_e);
        red2 = warp_reduce_sum(red2);
        const float inv_sum = 1.0f / sum_e;
        if constexpr (log_sm) {
            const float scale = red2 * inv_sum;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
                if (i < D) {
                    const float grad = rdy[k] - rx[k] * scale;
                    if constexpr (accumulate) dxg[i] += grad;
                    else dxg[i] = grad;
                }
            }
        } else {
            const float dot = red2 * inv_sum;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
                if (i < D) {
                    const float p = rx[k] * inv_sum;
                    const float grad = p * (rdy[k] - dot);
                    if constexpr (accumulate) dxg[i] += grad;
                    else dxg[i] = grad;
                }
            }
        }
    }
}

// Tier 1B Backward recomputing from `x` with float4 warp-per-row (D <= 512, D % 4 == 0).
template <bool log_sm, bool accumulate, int NV4, bool exact>
__global__ void softmax_bwd_warp_vec4(FaSoftmaxDims d,
                                      const float *__restrict__ x,
                                      const float *__restrict__ dy,
                                      float *__restrict__ dx) {
    const uint64_t D4 = d.axis_len >> 2;
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const float4 *x4 = reinterpret_cast<const float4 *>(x + d.offsets[0]);
    const float4 *dy4 = reinterpret_cast<const float4 *>(dy + d.offsets[1]);
    float4 *dx4 = reinterpret_cast<float4 *>(dx + d.offsets[2]);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        const float4 *xg4 = x4 + g * D4;
        const float4 *dyg4 = dy4 + g * D4;
        float4 *dxg4 = dx4 + g * D4;
        float4 rx[NV4];
        float4 rdy[NV4];
        float m = -INFINITY;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            if (exact || idx4 < D4) {
                const float4 vx = xg4[idx4];
                rx[k] = vx;
                rdy[k] = dyg4[idx4];
                m = max_nan_sm(m, max_nan_sm(max_nan_sm(vx.x, vx.y), max_nan_sm(vx.z, vx.w)));
            }
        }
        m = warp_reduce_max(m);
        if (isnan(m)) {
            const float qnan = __uint_as_float(0x7fc00000u);
            const float4 nan4 = make_float4(qnan, qnan, qnan, qnan);
            #pragma unroll
            for (int k = 0; k < NV4; ++k) {
                const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
                if (exact || idx4 < D4) {
                    if constexpr (accumulate) {
                        const float4 prev = dxg4[idx4];
                        dxg4[idx4] = make_float4(prev.x + qnan, prev.y + qnan, prev.z + qnan, prev.w + qnan);
                    } else {
                        dxg4[idx4] = nan4;
                    }
                }
            }
            continue;
        }
        float sum_e = 0.0f;
        float red2 = 0.0f;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            if (exact || idx4 < D4) {
                const float ex = expf(rx[k].x - m);
                const float ey = expf(rx[k].y - m);
                const float ez = expf(rx[k].z - m);
                const float ew = expf(rx[k].w - m);
                rx[k] = make_float4(ex, ey, ez, ew);
                sum_e += (ex + ey) + (ez + ew);
                const float4 u = rdy[k];
                if constexpr (log_sm) {
                    red2 += (u.x + u.y) + (u.z + u.w);
                } else {
                    red2 += (u.x * ex + u.y * ey) + (u.z * ez + u.w * ew);
                }
            }
        }
        sum_e = warp_reduce_sum(sum_e);
        red2 = warp_reduce_sum(red2);
        const float inv_sum = 1.0f / sum_e;
        const float c = red2 * inv_sum;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            if (exact || idx4 < D4) {
                const float4 e = rx[k];
                const float4 u = rdy[k];
                float4 g4;
                if constexpr (log_sm) {
                    g4 = make_float4(u.x - e.x * c, u.y - e.y * c, u.z - e.z * c, u.w - e.w * c);
                } else {
                    g4 = make_float4(
                        (e.x * inv_sum) * (u.x - c),
                        (e.y * inv_sum) * (u.y - c),
                        (e.z * inv_sum) * (u.z - c),
                        (e.w * inv_sum) * (u.w - c));
                }
                if constexpr (accumulate) {
                    const float4 prev = dxg4[idx4];
                    g4.x += prev.x;
                    g4.y += prev.y;
                    g4.z += prev.z;
                    g4.w += prev.w;
                }
                dxg4[idx4] = g4;
            }
        }
    }
}

// Tier 2 Backward: Block-per-row register-cached (128 < D <= 1024, 1 read of x and dy).
template <bool log_sm, bool accumulate>
__global__ void softmax_bwd_block_reg(FaSoftmaxDims d,
                                      const float *__restrict__ x,
                                      const float *__restrict__ dy,
                                      float *__restrict__ dx) {
    __shared__ float smem[16];
    const uint64_t D = d.axis_len;
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        const float *xg = x + d.offsets[0] + g * D;
        const float *dyg = dy + d.offsets[1] + g * D;
        float *dxg = dx + d.offsets[2] + g * D;
        float rx[4], rdy[4];
        float m = -INFINITY;
        bool has_v = false;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (i < D) {
                const float vx = xg[i];
                rx[k] = vx;
                rdy[k] = dyg[i];
                m = has_v ? max_nan_sm(m, vx) : vx;
                has_v = true;
            }
        }
        m = block_reduce_max(m, smem);
        if (isnan(m)) {
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                if constexpr (accumulate) dxg[i] += __uint_as_float(0x7fc00000u);
                else dxg[i] = __uint_as_float(0x7fc00000u);
            }
            continue;
        }
        float sum_e = 0.0f;
        float red2 = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (i < D) {
                const float e = expf(rx[k] - m);
                rx[k] = e;
                sum_e += e;
                red2 += log_sm ? rdy[k] : (rdy[k] * e);
            }
        }
        block_reduce_sum2(sum_e, red2, smem, sum_e, red2);
        const float inv_sum = 1.0f / sum_e;
        if constexpr (log_sm) {
            const float scale = red2 * inv_sum;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
                if (i < D) {
                    const float grad = rdy[k] - rx[k] * scale;
                    if constexpr (accumulate) dxg[i] += grad;
                    else dxg[i] = grad;
                }
            }
        } else {
            const float dot = red2 * inv_sum;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
                if (i < D) {
                    const float p = rx[k] * inv_sum;
                    const float grad = p * (rdy[k] - dot);
                    if constexpr (accumulate) dxg[i] += grad;
                    else dxg[i] = grad;
                }
            }
        }
    }
}

// Tier 3 Backward: General arbitrary D, arbitrary axis, arbitrary signed strides.
template <bool log_sm, bool accumulate, bool contig_last>
__global__ void softmax_bwd_general(FaSoftmaxDims d, const FaSoftmaxAxis *axes,
                                    const float *__restrict__ x,
                                    const float *__restrict__ dy,
                                    float *__restrict__ dx) {
    __shared__ float smem[16];
    const uint64_t D = d.axis_len;
    const int64_t sx = d.axis_strides[0];
    const int64_t sdy = d.axis_strides[1];
    const int64_t sdx = d.axis_strides[2];
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        int64_t ox, ody, odx;
        if constexpr (contig_last) {
            ox = d.offsets[0] + static_cast<int64_t>(g * D);
            ody = d.offsets[1] + static_cast<int64_t>(g * D);
            odx = d.offsets[2] + static_cast<int64_t>(g * D);
        } else {
            group_base3(d, axes, g, ox, ody, odx);
        }
        float m = -INFINITY;
        bool has_val = false;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            const float v = contig_last ? x[ox + static_cast<int64_t>(i)]
                                        : x[ox + static_cast<int64_t>(i) * sx];
            m = has_val ? max_nan_sm(m, v) : v;
            has_val = true;
        }
        m = block_reduce_max(m, smem);
        if (isnan(m)) {
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                const int64_t dst = contig_last ? (odx + static_cast<int64_t>(i))
                                                : (odx + static_cast<int64_t>(i) * sdx);
                if constexpr (accumulate) dx[dst] += __uint_as_float(0x7fc00000u);
                else dx[dst] = __uint_as_float(0x7fc00000u);
            }
            continue;
        }
        float sum_e = 0.0f;
        float red2 = 0.0f;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            const int64_t ii = static_cast<int64_t>(i);
            const float vx = contig_last ? x[ox + ii] : x[ox + ii * sx];
            const float vdy = contig_last ? dy[ody + ii] : dy[ody + ii * sdy];
            const float e = expf(vx - m);
            sum_e += e;
            red2 += log_sm ? vdy : (vdy * e);
        }
        block_reduce_sum2(sum_e, red2, smem, sum_e, red2);
        const float inv_sum = 1.0f / sum_e;
        if constexpr (log_sm) {
            const float scale = red2 * inv_sum;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                const int64_t ii = static_cast<int64_t>(i);
                const float vx = contig_last ? x[ox + ii] : x[ox + ii * sx];
                const float vdy = contig_last ? dy[ody + ii] : dy[ody + ii * sdy];
                const int64_t dst = contig_last ? (odx + ii) : (odx + ii * sdx);
                const float grad = vdy - expf(vx - m) * scale;
                if constexpr (accumulate) dx[dst] += grad;
                else dx[dst] = grad;
            }
        } else {
            const float dot = red2 * inv_sum;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                const int64_t ii = static_cast<int64_t>(i);
                const float vx = contig_last ? x[ox + ii] : x[ox + ii * sx];
                const float vdy = contig_last ? dy[ody + ii] : dy[ody + ii * sdy];
                const int64_t dst = contig_last ? (odx + ii) : (odx + ii * sdx);
                const float p = expf(vx - m) * inv_sum;
                const float grad = p * (vdy - dot);
                if constexpr (accumulate) dx[dst] += grad;
                else dx[dst] = grad;
            }
        }
    }
}

unsigned sm_grid_block(uint64_t groups) {
    return static_cast<unsigned>(std::min<uint64_t>(groups, 65535ULL));
}

unsigned sm_grid_warp(uint64_t groups) {
    const uint64_t blocks = (groups + 7ULL) / 8ULL;
    return static_cast<unsigned>(std::min<uint64_t>(blocks, 65535ULL));
}

bool sm_vec4_ok(const FaSoftmaxDims &d, const float *p0, const float *p1, const float *p2) {
    if (!d.fully_contiguous_last || (d.axis_len & 3ULL) != 0 || d.axis_len > 512ULL) return false;
    if ((d.offsets[0] & 3LL) != 0 || (d.offsets[1] & 3LL) != 0 || (d.offsets[2] & 3LL) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(p0) & 15ULL) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(p1) & 15ULL) != 0) return false;
    if (p2 && (reinterpret_cast<uintptr_t>(p2) & 15ULL) != 0) return false;
    return true;
}

// =========================================================================
// Backward-From-Output Kernels (zero expf for Softmax, zero max/sum_exp for LogSoftmax)
// =========================================================================
template <bool log_sm, bool accumulate, int NV4, bool exact, bool stream_dy>
__global__ void softmax_bwd_from_output_warp_vec4(FaSoftmaxDims d,
                                                  const float *__restrict__ y,
                                                  const float *__restrict__ dy,
                                                  float *__restrict__ dx) {
    const uint64_t D4 = d.axis_len >> 2;
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const float4 *y4 = reinterpret_cast<const float4 *>(y + d.offsets[0]);
    const float4 *dy4 = reinterpret_cast<const float4 *>(dy + d.offsets[1]);
    float4 *dx4 = reinterpret_cast<float4 *>(dx + d.offsets[2]);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        const float4 *yg4 = y4 + g * D4;
        const float4 *dyg4 = dy4 + g * D4;
        float4 *dxg4 = dx4 + g * D4;
        float4 ry[NV4];
        float4 rdy[NV4];
        float red = 0.0f;
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            if (exact || idx4 < D4) {
                const float4 vy = yg4[idx4];
                const float4 vu = stream_dy ? ldcs_sm_f4(dyg4 + idx4) : dyg4[idx4];
                ry[k] = vy;
                rdy[k] = vu;
                if constexpr (log_sm) {
                    red += (vu.x + vu.y) + (vu.z + vu.w);
                } else {
                    red += (vy.x * vu.x + vy.y * vu.y) + (vy.z * vu.z + vy.w * vu.w);
                }
            }
        }
        red = warp_reduce_sum(red);
        #pragma unroll
        for (int k = 0; k < NV4; ++k) {
            const uint64_t idx4 = static_cast<uint64_t>(lane + (k << 5));
            if (exact || idx4 < D4) {
                const float4 vy = ry[k];
                const float4 vu = rdy[k];
                float4 g4;
                if constexpr (log_sm) {
                    g4 = make_float4(
                        vu.x - expf(vy.x) * red,
                        vu.y - expf(vy.y) * red,
                        vu.z - expf(vy.z) * red,
                        vu.w - expf(vy.w) * red);
                } else {
                    g4 = make_float4(
                        fmaf(-vy.x, red, vy.x * vu.x),
                        fmaf(-vy.y, red, vy.y * vu.y),
                        fmaf(-vy.z, red, vy.z * vu.z),
                        fmaf(-vy.w, red, vy.w * vu.w));
                }
                if constexpr (accumulate) {
                    const float4 prev = dxg4[idx4];
                    g4.x += prev.x;
                    g4.y += prev.y;
                    g4.z += prev.z;
                    g4.w += prev.w;
                }
                dxg4[idx4] = g4;
            }
        }
    }
}

template <bool log_sm, bool accumulate>
__global__ void softmax_bwd_from_output_warp(FaSoftmaxDims d,
                                             const float *__restrict__ y,
                                             const float *__restrict__ dy,
                                             float *__restrict__ dx) {
    const uint64_t D = d.axis_len;
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        const float *yg = y + d.offsets[0] + g * D;
        const float *dyg = dy + d.offsets[1] + g * D;
        float *dxg = dx + d.offsets[2] + g * D;
        float ry[4], rdy[4];
        float red = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
            if (i < D) {
                const float vy = yg[i];
                const float vu = dyg[i];
                ry[k] = vy;
                rdy[k] = vu;
                red += log_sm ? vu : (vy * vu);
            }
        }
        red = warp_reduce_sum(red);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(lane + (k << 5));
            if (i < D) {
                const float grad = log_sm
                    ? (rdy[k] - expf(ry[k]) * red)
                    : fmaf(-ry[k], red, ry[k] * rdy[k]);
                if constexpr (accumulate) dxg[i] += grad;
                else dxg[i] = grad;
            }
        }
    }
}

template <bool log_sm, bool accumulate>
__global__ void softmax_bwd_from_output_block_reg(FaSoftmaxDims d,
                                                  const float *__restrict__ y,
                                                  const float *__restrict__ dy,
                                                  float *__restrict__ dx) {
    __shared__ float smem[8];
    const uint64_t D = d.axis_len;
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        const float *yg = y + d.offsets[0] + g * D;
        const float *dyg = dy + d.offsets[1] + g * D;
        float *dxg = dx + d.offsets[2] + g * D;
        float ry[4], rdy[4];
        float red = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (i < D) {
                const float vy = yg[i];
                const float vu = dyg[i];
                ry[k] = vy;
                rdy[k] = vu;
                red += log_sm ? vu : (vy * vu);
            }
        }
        red = block_reduce_sum(red, smem);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = static_cast<uint64_t>(threadIdx.x + (k << 8));
            if (i < D) {
                const float grad = log_sm
                    ? (rdy[k] - expf(ry[k]) * red)
                    : fmaf(-ry[k], red, ry[k] * rdy[k]);
                if constexpr (accumulate) dxg[i] += grad;
                else dxg[i] = grad;
            }
        }
    }
}

template <bool log_sm, bool accumulate, bool contig_last>
__global__ void softmax_bwd_from_output_general(FaSoftmaxDims d, const FaSoftmaxAxis *axes,
                                                const float *__restrict__ y,
                                                const float *__restrict__ dy,
                                                float *__restrict__ dx) {
    __shared__ float smem[8];
    const uint64_t D = d.axis_len;
    const int64_t sy = d.axis_strides[0];
    const int64_t sdy = d.axis_strides[1];
    const int64_t sdx = d.axis_strides[2];
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        int64_t oy, ody, odx;
        if constexpr (contig_last) {
            oy = d.offsets[0] + static_cast<int64_t>(g * D);
            ody = d.offsets[1] + static_cast<int64_t>(g * D);
            odx = d.offsets[2] + static_cast<int64_t>(g * D);
        } else {
            group_base3(d, axes, g, oy, ody, odx);
        }
        float red = 0.0f;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            const int64_t ii = static_cast<int64_t>(i);
            const float vu = contig_last ? dy[ody + ii] : dy[ody + ii * sdy];
            if constexpr (log_sm) {
                red += vu;
            } else {
                const float vy = contig_last ? y[oy + ii] : y[oy + ii * sy];
                red += vy * vu;
            }
        }
        red = block_reduce_sum(red, smem);
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            const int64_t ii = static_cast<int64_t>(i);
            const float vy = contig_last ? y[oy + ii] : y[oy + ii * sy];
            const float vu = contig_last ? dy[ody + ii] : dy[ody + ii * sdy];
            const int64_t dst = contig_last ? (odx + ii) : (odx + ii * sdx);
            const float grad = log_sm
                ? (vu - expf(vy) * red)
                : fmaf(-vy, red, vy * vu);
            if constexpr (accumulate) dx[dst] += grad;
            else dx[dst] = grad;
        }
    }
}

template <bool log_sm, bool stream_x>
void launch_softmax_fwd_vec4(cudaStream_t stream, unsigned g, const FaSoftmaxDims &d,
                             const float *x, float *y) {
    if (d.axis_len == 128) {
        softmax_fwd_warp_vec4<log_sm, 1, true, stream_x><<<g, 256, 0, stream>>>(d, x, y);
    } else if (d.axis_len == 256) {
        softmax_fwd_warp_vec4<log_sm, 2, true, stream_x><<<g, 256, 0, stream>>>(d, x, y);
    } else if (d.axis_len == 512) {
        softmax_fwd_warp_vec4<log_sm, 4, true, stream_x><<<g, 256, 0, stream>>>(d, x, y);
    } else if (d.axis_len <= 128) {
        softmax_fwd_warp_vec4<log_sm, 1, false, stream_x><<<g, 256, 0, stream>>>(d, x, y);
    } else if (d.axis_len <= 256) {
        softmax_fwd_warp_vec4<log_sm, 2, false, stream_x><<<g, 256, 0, stream>>>(d, x, y);
    } else {
        softmax_fwd_warp_vec4<log_sm, 4, false, stream_x><<<g, 256, 0, stream>>>(d, x, y);
    }
}

template <bool log_sm, bool acc, bool stream_dy>
void launch_softmax_bwd_from_output_vec4(cudaStream_t stream, unsigned g, const FaSoftmaxDims &d,
                                         const float *y, const float *dy, float *dx) {
    if (d.axis_len == 128) {
        softmax_bwd_from_output_warp_vec4<log_sm, acc, 1, true, stream_dy><<<g, 256, 0, stream>>>(d, y, dy, dx);
    } else if (d.axis_len == 256) {
        softmax_bwd_from_output_warp_vec4<log_sm, acc, 2, true, stream_dy><<<g, 256, 0, stream>>>(d, y, dy, dx);
    } else if (d.axis_len == 512) {
        softmax_bwd_from_output_warp_vec4<log_sm, acc, 4, true, stream_dy><<<g, 256, 0, stream>>>(d, y, dy, dx);
    } else if (d.axis_len <= 128) {
        softmax_bwd_from_output_warp_vec4<log_sm, acc, 1, false, stream_dy><<<g, 256, 0, stream>>>(d, y, dy, dx);
    } else if (d.axis_len <= 256) {
        softmax_bwd_from_output_warp_vec4<log_sm, acc, 2, false, stream_dy><<<g, 256, 0, stream>>>(d, y, dy, dx);
    } else {
        softmax_bwd_from_output_warp_vec4<log_sm, acc, 4, false, stream_dy><<<g, 256, 0, stream>>>(d, y, dy, dx);
    }
}

} // namespace

#define TRY_SM(expr) do { const int code = (expr); if (code) return code; } while (0)

extern "C" int fa_cuda_softmax_create(Context *ctx, FaSoftmaxDims d,
                                      const FaSoftmaxAxis *group_axes,
                                      size_t *workspace, size_t *shared, void **out) {
    if (!ctx || !workspace || !shared || !out || d.kind < 0 || d.kind > 1 ||
        d.group_rank < 0 || d.group_rank > 63 || d.reserved) {
        return rt(cudaErrorInvalidValue);
    }
    TRY_SM(activate(ctx));
    auto p = std::unique_ptr<SoftmaxPlanCuda>(new (std::nothrow) SoftmaxPlanCuda(ctx, d));
    if (!p) return 1002;
    size_t group_bytes = 0;
    if (!d.fully_contiguous_last && d.group_rank > 0) {
        group_bytes = static_cast<size_t>(d.group_rank) * sizeof(FaSoftmaxAxis);
        TRY_SM(rt(cudaMalloc(&p->group_axes, group_bytes)));
        TRY_SM(rt(cudaMemcpyAsync(
            p->group_axes, group_axes, group_bytes, cudaMemcpyHostToDevice, ctx->stream)));
        TRY_SM(rt(cudaStreamSynchronize(ctx->stream)));
    }
    if (d.fully_contiguous_last && (d.axis_len <= 128 || ((d.axis_len & 3ULL) == 0 && d.axis_len <= 512))) {
        cudaFuncAttributes attr{};
        TRY_SM(rt(cudaFuncGetAttributes(&attr, softmax_fwd_warp<false>)));
        *shared = attr.sharedSizeBytes;
    } else {
        cudaFuncAttributes attr{};
        TRY_SM(rt(cudaFuncGetAttributes(&attr, softmax_bwd_block_reg<false, false>)));
        *shared = attr.sharedSizeBytes;
    }
    *workspace = group_bytes;
    *out = p.release();
    return 0;
}

extern "C" int fa_cuda_softmax_forward_run(const SoftmaxPlanCuda *p, const float *x, float *y) {
    if (!p) return rt(cudaErrorInvalidValue);
    const auto d = p->dims;
    if (d.groups == 0 || d.axis_len == 0) return 0;
    if (!x || !y) return rt(cudaErrorInvalidValue);
    TRY_SM(activate(p->ctx));
    const bool log_sm = (d.kind == 1);
    if (d.fully_contiguous_last) {
        if (sm_vec4_ok(d, x, y, nullptr)) {
            const unsigned g = sm_grid_warp(d.groups);
            const bool stream_x = (d.groups * d.axis_len) >= 1048576ULL;
            if (log_sm) {
                if (stream_x) launch_softmax_fwd_vec4<true, true>(p->ctx->stream, g, d, x, y);
                else launch_softmax_fwd_vec4<true, false>(p->ctx->stream, g, d, x, y);
            } else {
                if (stream_x) launch_softmax_fwd_vec4<false, true>(p->ctx->stream, g, d, x, y);
                else launch_softmax_fwd_vec4<false, false>(p->ctx->stream, g, d, x, y);
            }
        } else if (d.axis_len <= 128) {
            const unsigned g = sm_grid_warp(d.groups);
            if (log_sm) softmax_fwd_warp<true><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
            else softmax_fwd_warp<false><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
        } else if (d.axis_len <= 1024) {
            const unsigned g = sm_grid_block(d.groups);
            if (log_sm) softmax_fwd_block_reg<true><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
            else softmax_fwd_block_reg<false><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
        } else {
            const unsigned g = sm_grid_block(d.groups);
            if (log_sm) softmax_fwd_general<true, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, x, y);
            else softmax_fwd_general<false, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, x, y);
        }
    } else {
        const unsigned g = sm_grid_block(d.groups);
        if (log_sm) softmax_fwd_general<true, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, x, y);
        else softmax_fwd_general<false, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, x, y);
    }
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_softmax_backward_run(const SoftmaxPlanCuda *p, const float *x,
                                            const float *dy, float *dx, int accumulate) {
    if (!p || (accumulate != 0 && accumulate != 1)) return rt(cudaErrorInvalidValue);
    const auto d = p->dims;
    if (d.groups == 0 || d.axis_len == 0) return 0;
    if (!x || !dy || !dx) return rt(cudaErrorInvalidValue);
    TRY_SM(activate(p->ctx));
    const bool log_sm = (d.kind == 1);
    const bool acc = (accumulate != 0);
    if (d.fully_contiguous_last) {
        if (sm_vec4_ok(d, x, dy, dx)) {
            const unsigned g = sm_grid_warp(d.groups);
            if (d.axis_len == 128) {
                if (log_sm) {
                    if (acc) softmax_bwd_warp_vec4<true, true, 1, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<true, false, 1, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                } else {
                    if (acc) softmax_bwd_warp_vec4<false, true, 1, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<false, false, 1, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                }
            } else if (d.axis_len == 256) {
                if (log_sm) {
                    if (acc) softmax_bwd_warp_vec4<true, true, 2, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<true, false, 2, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                } else {
                    if (acc) softmax_bwd_warp_vec4<false, true, 2, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<false, false, 2, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                }
            } else if (d.axis_len == 512) {
                if (log_sm) {
                    if (acc) softmax_bwd_warp_vec4<true, true, 4, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<true, false, 4, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                } else {
                    if (acc) softmax_bwd_warp_vec4<false, true, 4, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<false, false, 4, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                }
            } else if (d.axis_len <= 128) {
                if (log_sm) {
                    if (acc) softmax_bwd_warp_vec4<true, true, 1, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<true, false, 1, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                } else {
                    if (acc) softmax_bwd_warp_vec4<false, true, 1, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<false, false, 1, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                }
            } else if (d.axis_len <= 256) {
                if (log_sm) {
                    if (acc) softmax_bwd_warp_vec4<true, true, 2, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<true, false, 2, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                } else {
                    if (acc) softmax_bwd_warp_vec4<false, true, 2, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<false, false, 2, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                }
            } else {
                if (log_sm) {
                    if (acc) softmax_bwd_warp_vec4<true, true, 4, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<true, false, 4, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                } else {
                    if (acc) softmax_bwd_warp_vec4<false, true, 4, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                    else softmax_bwd_warp_vec4<false, false, 4, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                }
            }
        } else if (d.axis_len <= 128) {
            const unsigned g = sm_grid_warp(d.groups);
            if (log_sm) {
                if (acc) softmax_bwd_warp<true, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                else softmax_bwd_warp<true, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
            } else {
                if (acc) softmax_bwd_warp<false, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                else softmax_bwd_warp<false, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
            }
        } else if (d.axis_len <= 1024) {
            const unsigned g = sm_grid_block(d.groups);
            if (log_sm) {
                if (acc) softmax_bwd_block_reg<true, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                else softmax_bwd_block_reg<true, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
            } else {
                if (acc) softmax_bwd_block_reg<false, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                else softmax_bwd_block_reg<false, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
            }
        } else {
            const unsigned g = sm_grid_block(d.groups);
            if (log_sm) {
                if (acc) softmax_bwd_general<true, true, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, x, dy, dx);
                else softmax_bwd_general<true, false, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, x, dy, dx);
            } else {
                if (acc) softmax_bwd_general<false, true, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, x, dy, dx);
                else softmax_bwd_general<false, false, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, x, dy, dx);
            }
        }
    } else {
        const unsigned g = sm_grid_block(d.groups);
        if (log_sm) {
            if (acc) softmax_bwd_general<true, true, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, x, dy, dx);
            else softmax_bwd_general<true, false, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, x, dy, dx);
        } else {
            if (acc) softmax_bwd_general<false, true, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, x, dy, dx);
            else softmax_bwd_general<false, false, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, x, dy, dx);
        }
    }
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_softmax_backward_from_output_run(const SoftmaxPlanCuda *p, const float *y,
                                                        const float *dy, float *dx, int accumulate) {
    if (!p || (accumulate != 0 && accumulate != 1)) return rt(cudaErrorInvalidValue);
    const auto d = p->dims;
    if (d.groups == 0 || d.axis_len == 0) return 0;
    if (!y || !dy || !dx) return rt(cudaErrorInvalidValue);
    TRY_SM(activate(p->ctx));
    const bool log_sm = (d.kind == 1);
    const bool acc = (accumulate != 0);
    if (d.fully_contiguous_last) {
        if (sm_vec4_ok(d, y, dy, dx)) {
            const unsigned g = sm_grid_warp(d.groups);
            const bool stream_dy = (d.groups * d.axis_len) >= 1048576ULL;
            if (log_sm) {
                if (acc) {
                    if (stream_dy) launch_softmax_bwd_from_output_vec4<true, true, true>(p->ctx->stream, g, d, y, dy, dx);
                    else launch_softmax_bwd_from_output_vec4<true, true, false>(p->ctx->stream, g, d, y, dy, dx);
                } else {
                    if (stream_dy) launch_softmax_bwd_from_output_vec4<true, false, true>(p->ctx->stream, g, d, y, dy, dx);
                    else launch_softmax_bwd_from_output_vec4<true, false, false>(p->ctx->stream, g, d, y, dy, dx);
                }
            } else {
                if (acc) {
                    if (stream_dy) launch_softmax_bwd_from_output_vec4<false, true, true>(p->ctx->stream, g, d, y, dy, dx);
                    else launch_softmax_bwd_from_output_vec4<false, true, false>(p->ctx->stream, g, d, y, dy, dx);
                } else {
                    if (stream_dy) launch_softmax_bwd_from_output_vec4<false, false, true>(p->ctx->stream, g, d, y, dy, dx);
                    else launch_softmax_bwd_from_output_vec4<false, false, false>(p->ctx->stream, g, d, y, dy, dx);
                }
            }
        } else if (d.axis_len <= 128) {
            const unsigned g = sm_grid_warp(d.groups);
            if (log_sm) {
                if (acc) softmax_bwd_from_output_warp<true, true><<<g, 256, 0, p->ctx->stream>>>(d, y, dy, dx);
                else softmax_bwd_from_output_warp<true, false><<<g, 256, 0, p->ctx->stream>>>(d, y, dy, dx);
            } else {
                if (acc) softmax_bwd_from_output_warp<false, true><<<g, 256, 0, p->ctx->stream>>>(d, y, dy, dx);
                else softmax_bwd_from_output_warp<false, false><<<g, 256, 0, p->ctx->stream>>>(d, y, dy, dx);
            }
        } else if (d.axis_len <= 1024) {
            const unsigned g = sm_grid_block(d.groups);
            if (log_sm) {
                if (acc) softmax_bwd_from_output_block_reg<true, true><<<g, 256, 0, p->ctx->stream>>>(d, y, dy, dx);
                else softmax_bwd_from_output_block_reg<true, false><<<g, 256, 0, p->ctx->stream>>>(d, y, dy, dx);
            } else {
                if (acc) softmax_bwd_from_output_block_reg<false, true><<<g, 256, 0, p->ctx->stream>>>(d, y, dy, dx);
                else softmax_bwd_from_output_block_reg<false, false><<<g, 256, 0, p->ctx->stream>>>(d, y, dy, dx);
            }
        } else {
            const unsigned g = sm_grid_block(d.groups);
            if (log_sm) {
                if (acc) softmax_bwd_from_output_general<true, true, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, y, dy, dx);
                else softmax_bwd_from_output_general<true, false, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, y, dy, dx);
            } else {
                if (acc) softmax_bwd_from_output_general<false, true, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, y, dy, dx);
                else softmax_bwd_from_output_general<false, false, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, y, dy, dx);
            }
        }
    } else {
        const unsigned g = sm_grid_block(d.groups);
        if (log_sm) {
            if (acc) softmax_bwd_from_output_general<true, true, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, y, dy, dx);
            else softmax_bwd_from_output_general<true, false, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, y, dy, dx);
        } else {
            if (acc) softmax_bwd_from_output_general<false, true, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, y, dy, dx);
            else softmax_bwd_from_output_general<false, false, false><<<g, 256, 0, p->ctx->stream>>>(d, p->group_axes, y, dy, dx);
        }
    }
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_softmax_destroy(SoftmaxPlanCuda *p) { delete p; }
