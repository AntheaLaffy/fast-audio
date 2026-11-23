#include "cuda_backend.h"
#include "norm.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

namespace {

struct NormPlanCuda {
    Context *ctx;
    FaNormDims dims;
    FaNormAxis *outer_axes = nullptr;
    FaNormAxis *inner_axes = nullptr;
    NormPlanCuda(Context *c, FaNormDims d) : ctx(c), dims(d) {}
    ~NormPlanCuda() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (outer_axes) cudaFree(outer_axes);
        if (inner_axes) cudaFree(inner_axes);
    }
};

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float out = (lane < 8) ? smem[lane] : 0.0f;
        out = warp_reduce_sum(out);
        if (lane == 0) smem[0] = out;
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

__device__ __forceinline__ void strided_base2(
    FaNormDims d, const FaNormAxis *outer, uint64_t g, int64_t &ox, int64_t &oy) {
    ox = d.offsets[0];
    oy = d.offsets[1];
    for (int a = d.outer_rank - 1; a >= 0; --a) {
        const int64_t c = static_cast<int64_t>(g % outer[a].dim);
        g /= outer[a].dim;
        ox += c * outer[a].strides[0];
        oy += c * outer[a].strides[1];
    }
}

__device__ __forceinline__ void strided_base3(
    FaNormDims d, const FaNormAxis *outer, uint64_t g,
    int64_t &ox, int64_t &ody, int64_t &odx) {
    ox = d.offsets[0];
    ody = d.offsets[1];
    odx = d.offsets[2];
    for (int a = d.outer_rank - 1; a >= 0; --a) {
        const int64_t c = static_cast<int64_t>(g % outer[a].dim);
        g /= outer[a].dim;
        ox += c * outer[a].strides[0];
        ody += c * outer[a].strides[1];
        odx += c * outer[a].strides[2];
    }
}

__device__ __forceinline__ void strided_inner2(
    FaNormDims d, const FaNormAxis *inner, uint64_t i, int64_t &ix, int64_t &iy) {
    ix = 0;
    iy = 0;
    for (int a = d.inner_rank - 1; a >= 0; --a) {
        const int64_t c = static_cast<int64_t>(i % inner[a].dim);
        i /= inner[a].dim;
        ix += c * inner[a].strides[0];
        iy += c * inner[a].strides[1];
    }
}

__device__ __forceinline__ void strided_inner3(
    FaNormDims d, const FaNormAxis *inner, uint64_t i,
    int64_t &ix, int64_t &idy, int64_t &idx) {
    ix = 0;
    idy = 0;
    idx = 0;
    for (int a = d.inner_rank - 1; a >= 0; --a) {
        const int64_t c = static_cast<int64_t>(i % inner[a].dim);
        i /= inner[a].dim;
        ix += c * inner[a].strides[0];
        idy += c * inner[a].strides[1];
        idx += c * inner[a].strides[2];
    }
}

// Tier 1 Forward: Warp-per-group (8 warps = 256 threads/block, 0 shared memory, 0 __syncthreads(), D <= 128).
template <bool rms>
__global__ void norm_fwd_warp(FaNormDims d, const float *__restrict__ x, float *__restrict__ y) {
    const uint64_t D = d.inner_len;
    const float inv_d = 1.0f / static_cast<float>(D);
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        const float *xg = x + d.offsets[0] + g * D;
        float *yg = y + d.offsets[1] + g * D;
        float reg[4];
        int nreg = 0;
        float acc = 0.0f;
        for (uint64_t i = lane; i < D; i += 32) {
            float v = xg[i];
            reg[nreg++] = v;
            acc += rms ? (v * v) : v;
        }
        if constexpr (rms) {
            const float ms = warp_reduce_sum(acc) * inv_d;
            const float rstd = 1.0f / sqrtf(ms + d.eps);
            int k = 0;
            for (uint64_t i = lane; i < D; i += 32) {
                yg[i] = reg[k++] * rstd;
            }
        } else {
            const float mean = warp_reduce_sum(acc) * inv_d;
            float var_acc = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                float diff = reg[k] - mean;
                reg[k] = diff;
                var_acc += diff * diff;
            }
            const float var = warp_reduce_sum(var_acc) * inv_d;
            const float rstd = 1.0f / sqrtf(var + d.eps);
            int k = 0;
            for (uint64_t i = lane; i < D; i += 32) {
                yg[i] = reg[k++] * rstd;
            }
        }
    }
}

// Tier 2 Forward: Block-per-group register-cached for 128 < inner_len <= 1024 (1 read of x from VRAM).
template <bool rms>
__global__ void norm_fwd_contig_reg(FaNormDims d, const float *__restrict__ x, float *__restrict__ y) {
    __shared__ float smem[16];
    const uint64_t D = d.inner_len;
    const float inv_d = 1.0f / static_cast<float>(D);
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        const float *xg = x + d.offsets[0] + g * D;
        float *yg = y + d.offsets[1] + g * D;
        float reg[4];
        int nreg = 0;
        float acc = 0.0f;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            float v = xg[i];
            reg[nreg++] = v;
            acc += rms ? (v * v) : v;
        }
        if constexpr (rms) {
            const float ms = block_reduce_sum(acc, smem) * inv_d;
            const float rstd = 1.0f / sqrtf(ms + d.eps);
            int k = 0;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                yg[i] = reg[k++] * rstd;
            }
        } else {
            const float mean = block_reduce_sum(acc, smem) * inv_d;
            float var_acc = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                float diff = reg[k] - mean;
                reg[k] = diff;
                var_acc += diff * diff;
            }
            const float var = block_reduce_sum(var_acc, smem) * inv_d;
            const float rstd = 1.0f / sqrtf(var + d.eps);
            int k = 0;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                yg[i] = reg[k++] * rstd;
            }
        }
    }
}

// Tier 2b Forward: 128-bit float4 vectorized block-per-group when D % 4 == 0, D <= 1024, aligned16.
template <bool rms>
__global__ void norm_fwd_contig_vec4(FaNormDims d, const float *__restrict__ x, float *__restrict__ y) {
    __shared__ float smem[16];
    const uint64_t D = d.inner_len;
    const uint64_t D4 = D >> 2;
    const float inv_d = 1.0f / static_cast<float>(D);
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        const float4 *xg = reinterpret_cast<const float4 *>(x + d.offsets[0] + g * D);
        float4 *yg = reinterpret_cast<float4 *>(y + d.offsets[1] + g * D);
        float4 v{};
        const bool active = (threadIdx.x < D4);
        float acc = 0.0f;
        if (active) {
            v = xg[threadIdx.x];
            if constexpr (rms) {
                acc = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            } else {
                acc = v.x + v.y + v.z + v.w;
            }
        }
        if constexpr (rms) {
            const float ms = block_reduce_sum(acc, smem) * inv_d;
            const float rstd = 1.0f / sqrtf(ms + d.eps);
            if (active) {
                yg[threadIdx.x] = make_float4(v.x * rstd, v.y * rstd, v.z * rstd, v.w * rstd);
            }
        } else {
            const float mean = block_reduce_sum(acc, smem) * inv_d;
            float var_acc = 0.0f;
            if (active) {
                v.x -= mean; v.y -= mean; v.z -= mean; v.w -= mean;
                var_acc = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            const float var = block_reduce_sum(var_acc, smem) * inv_d;
            const float rstd = 1.0f / sqrtf(var + d.eps);
            if (active) {
                yg[threadIdx.x] = make_float4(v.x * rstd, v.y * rstd, v.z * rstd, v.w * rstd);
            }
        }
    }
}

// Tier 3 Forward: General forward kernel supporting arbitrary D and arbitrary signed strides.
template <bool rms, bool contig>
__global__ void norm_fwd_general(FaNormDims d, const FaNormAxis *outer, const FaNormAxis *inner,
                                 const float *__restrict__ x, float *__restrict__ y) {
    __shared__ float smem[16];
    const uint64_t D = d.inner_len;
    const float inv_d = 1.0f / static_cast<float>(D);
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        int64_t ox, oy;
        if constexpr (contig) {
            ox = d.offsets[0] + static_cast<int64_t>(g * D);
            oy = d.offsets[1] + static_cast<int64_t>(g * D);
        } else {
            strided_base2(d, outer, g, ox, oy);
        }
        float acc = 0.0f;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            float v;
            if constexpr (contig) {
                v = x[ox + static_cast<int64_t>(i)];
            } else {
                int64_t ix, iy;
                strided_inner2(d, inner, i, ix, iy);
                v = x[ox + ix];
            }
            acc += rms ? (v * v) : v;
        }
        if constexpr (rms) {
            const float ms = block_reduce_sum(acc, smem) * inv_d;
            const float rstd = 1.0f / sqrtf(ms + d.eps);
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                if constexpr (contig) {
                    y[oy + static_cast<int64_t>(i)] = x[ox + static_cast<int64_t>(i)] * rstd;
                } else {
                    int64_t ix, iy;
                    strided_inner2(d, inner, i, ix, iy);
                    y[oy + iy] = x[ox + ix] * rstd;
                }
            }
        } else {
            const float mean = block_reduce_sum(acc, smem) * inv_d;
            float var_acc = 0.0f;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                float v;
                if constexpr (contig) {
                    v = x[ox + static_cast<int64_t>(i)];
                } else {
                    int64_t ix, iy;
                    strided_inner2(d, inner, i, ix, iy);
                    v = x[ox + ix];
                }
                float diff = v - mean;
                var_acc += diff * diff;
            }
            const float var = block_reduce_sum(var_acc, smem) * inv_d;
            const float rstd = 1.0f / sqrtf(var + d.eps);
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                if constexpr (contig) {
                    y[oy + static_cast<int64_t>(i)] = (x[ox + static_cast<int64_t>(i)] - mean) * rstd;
                } else {
                    int64_t ix, iy;
                    strided_inner2(d, inner, i, ix, iy);
                    y[oy + iy] = (x[ox + ix] - mean) * rstd;
                }
            }
        }
    }
}

// Tier 1 Backward: Warp-per-group (0 shared memory, 0 __syncthreads(), D <= 128, 1 read of x and dy).
template <bool rms, bool accumulate>
__global__ void norm_bwd_warp(FaNormDims d,
                              const float *__restrict__ x,
                              const float *__restrict__ dy,
                              float *__restrict__ dx) {
    const uint64_t D = d.inner_len;
    const float inv_d = 1.0f / static_cast<float>(D);
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        const float *xg = x + d.offsets[0] + g * D;
        const float *dyg = dy + d.offsets[1] + g * D;
        float *dxg = dx + d.offsets[2] + g * D;
        float rx[4], rdy[4];
        int nreg = 0;
        float acc0 = 0.0f;
        for (uint64_t i = lane; i < D; i += 32) {
            float vx = xg[i];
            float vdy = dyg[i];
            rx[nreg] = vx;
            rdy[nreg] = vdy;
            ++nreg;
            acc0 += rms ? (vx * vx) : vx;
        }
        if constexpr (rms) {
            const float ms = warp_reduce_sum(acc0) * inv_d;
            const float rstd = 1.0f / sqrtf(ms + d.eps);
            float s2_acc = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                float xhat = rx[k] * rstd;
                rx[k] = xhat;
                s2_acc += rdy[k] * xhat;
            }
            const float s2 = warp_reduce_sum(s2_acc) * inv_d;
            int k = 0;
            for (uint64_t i = lane; i < D; i += 32) {
                float grad = rstd * (rdy[k] - rx[k] * s2);
                ++k;
                if constexpr (accumulate) dxg[i] += grad;
                else dxg[i] = grad;
            }
        } else {
            const float mean = warp_reduce_sum(acc0) * inv_d;
            float var_acc = 0.0f;
            float s1_acc = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                float diff = rx[k] - mean;
                rx[k] = diff;
                var_acc += diff * diff;
                s1_acc += rdy[k];
            }
            const float var = warp_reduce_sum(var_acc) * inv_d;
            const float s1 = warp_reduce_sum(s1_acc) * inv_d;
            const float rstd = 1.0f / sqrtf(var + d.eps);
            float s2_acc = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                float xhat = rx[k] * rstd;
                rx[k] = xhat;
                s2_acc += rdy[k] * xhat;
            }
            const float s2 = warp_reduce_sum(s2_acc) * inv_d;
            int k = 0;
            for (uint64_t i = lane; i < D; i += 32) {
                float grad = rstd * (rdy[k] - s1 - rx[k] * s2);
                ++k;
                if constexpr (accumulate) dxg[i] += grad;
                else dxg[i] = grad;
            }
        }
    }
}

// Tier 2 Backward: Block-per-group register-cached for 128 < inner_len <= 1024 (1 read of x and dy).
template <bool rms, bool accumulate>
__global__ void norm_bwd_contig_reg(FaNormDims d,
                                    const float *__restrict__ x,
                                    const float *__restrict__ dy,
                                    float *__restrict__ dx) {
    __shared__ float smem[16];
    const uint64_t D = d.inner_len;
    const float inv_d = 1.0f / static_cast<float>(D);
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        const float *xg = x + d.offsets[0] + g * D;
        const float *dyg = dy + d.offsets[1] + g * D;
        float *dxg = dx + d.offsets[2] + g * D;
        float rx[4], rdy[4];
        int nreg = 0;
        float acc0 = 0.0f;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            float vx = xg[i];
            float vdy = dyg[i];
            rx[nreg] = vx;
            rdy[nreg] = vdy;
            ++nreg;
            acc0 += rms ? (vx * vx) : vx;
        }
        if constexpr (rms) {
            const float ms = block_reduce_sum(acc0, smem) * inv_d;
            const float rstd = 1.0f / sqrtf(ms + d.eps);
            float s2_acc = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                float xhat = rx[k] * rstd;
                rx[k] = xhat;
                s2_acc += rdy[k] * xhat;
            }
            const float s2 = block_reduce_sum(s2_acc, smem) * inv_d;
            int k = 0;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                float grad = rstd * (rdy[k] - rx[k] * s2);
                ++k;
                if constexpr (accumulate) dxg[i] += grad;
                else dxg[i] = grad;
            }
        } else {
            const float mean = block_reduce_sum(acc0, smem) * inv_d;
            float var_acc = 0.0f;
            float s1_acc = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                float diff = rx[k] - mean;
                rx[k] = diff;
                var_acc += diff * diff;
                s1_acc += rdy[k];
            }
            float var_sum, s1_sum;
            block_reduce_sum2(var_acc, s1_acc, smem, var_sum, s1_sum);
            const float var = var_sum * inv_d;
            const float s1 = s1_sum * inv_d;
            const float rstd = 1.0f / sqrtf(var + d.eps);
            float s2_acc = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                float xhat = rx[k] * rstd;
                rx[k] = xhat;
                s2_acc += rdy[k] * xhat;
            }
            const float s2 = block_reduce_sum(s2_acc, smem) * inv_d;
            int k = 0;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                float grad = rstd * (rdy[k] - s1 - rx[k] * s2);
                ++k;
                if constexpr (accumulate) dxg[i] += grad;
                else dxg[i] = grad;
            }
        }
    }
}

// Tier 3 Backward: General arbitrary D and arbitrary signed strides.
template <bool rms, bool accumulate, bool contig>
__global__ void norm_bwd_general(FaNormDims d, const FaNormAxis *outer, const FaNormAxis *inner,
                                 const float *__restrict__ x,
                                 const float *__restrict__ dy,
                                 float *__restrict__ dx) {
    __shared__ float smem[16];
    const uint64_t D = d.inner_len;
    const float inv_d = 1.0f / static_cast<float>(D);
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        int64_t ox, ody, odx;
        if constexpr (contig) {
            ox = d.offsets[0] + static_cast<int64_t>(g * D);
            ody = d.offsets[1] + static_cast<int64_t>(g * D);
            odx = d.offsets[2] + static_cast<int64_t>(g * D);
        } else {
            strided_base3(d, outer, g, ox, ody, odx);
        }
        float acc0 = 0.0f;
        for (uint64_t i = threadIdx.x; i < D; i += 256) {
            float v;
            if constexpr (contig) {
                v = x[ox + static_cast<int64_t>(i)];
            } else {
                int64_t ix, idy, idx;
                strided_inner3(d, inner, i, ix, idy, idx);
                v = x[ox + ix];
            }
            acc0 += rms ? (v * v) : v;
        }
        if constexpr (rms) {
            const float ms = block_reduce_sum(acc0, smem) * inv_d;
            const float rstd = 1.0f / sqrtf(ms + d.eps);
            float s2_acc = 0.0f;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                float vx, vdy;
                if constexpr (contig) {
                    vx = x[ox + static_cast<int64_t>(i)];
                    vdy = dy[ody + static_cast<int64_t>(i)];
                } else {
                    int64_t ix, idy, idx;
                    strided_inner3(d, inner, i, ix, idy, idx);
                    vx = x[ox + ix];
                    vdy = dy[ody + idy];
                }
                s2_acc += vdy * (vx * rstd);
            }
            const float s2 = block_reduce_sum(s2_acc, smem) * inv_d;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                float vx, vdy;
                int64_t dst;
                if constexpr (contig) {
                    vx = x[ox + static_cast<int64_t>(i)];
                    vdy = dy[ody + static_cast<int64_t>(i)];
                    dst = odx + static_cast<int64_t>(i);
                } else {
                    int64_t ix, idy, idx;
                    strided_inner3(d, inner, i, ix, idy, idx);
                    vx = x[ox + ix];
                    vdy = dy[ody + idy];
                    dst = odx + idx;
                }
                float grad = rstd * (vdy - (vx * rstd) * s2);
                if constexpr (accumulate) dx[dst] += grad;
                else dx[dst] = grad;
            }
        } else {
            const float mean = block_reduce_sum(acc0, smem) * inv_d;
            float var_acc = 0.0f;
            float s1_acc = 0.0f;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                float vx, vdy;
                if constexpr (contig) {
                    vx = x[ox + static_cast<int64_t>(i)];
                    vdy = dy[ody + static_cast<int64_t>(i)];
                } else {
                    int64_t ix, idy, idx;
                    strided_inner3(d, inner, i, ix, idy, idx);
                    vx = x[ox + ix];
                    vdy = dy[ody + idy];
                }
                float diff = vx - mean;
                var_acc += diff * diff;
                s1_acc += vdy;
            }
            float var_sum, s1_sum;
            block_reduce_sum2(var_acc, s1_acc, smem, var_sum, s1_sum);
            const float var = var_sum * inv_d;
            const float s1 = s1_sum * inv_d;
            const float rstd = 1.0f / sqrtf(var + d.eps);
            float s2_acc = 0.0f;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                float vx, vdy;
                if constexpr (contig) {
                    vx = x[ox + static_cast<int64_t>(i)];
                    vdy = dy[ody + static_cast<int64_t>(i)];
                } else {
                    int64_t ix, idy, idx;
                    strided_inner3(d, inner, i, ix, idy, idx);
                    vx = x[ox + ix];
                    vdy = dy[ody + idy];
                }
                s2_acc += vdy * ((vx - mean) * rstd);
            }
            const float s2 = block_reduce_sum(s2_acc, smem) * inv_d;
            for (uint64_t i = threadIdx.x; i < D; i += 256) {
                float vx, vdy;
                int64_t dst;
                if constexpr (contig) {
                    vx = x[ox + static_cast<int64_t>(i)];
                    vdy = dy[ody + static_cast<int64_t>(i)];
                    dst = odx + static_cast<int64_t>(i);
                } else {
                    int64_t ix, idy, idx;
                    strided_inner3(d, inner, i, ix, idy, idx);
                    vx = x[ox + ix];
                    vdy = dy[ody + idy];
                    dst = odx + idx;
                }
                float grad = rstd * (vdy - s1 - ((vx - mean) * rstd) * s2);
                if constexpr (accumulate) dx[dst] += grad;
                else dx[dst] = grad;
            }
        }
    }
}

unsigned norm_grid(uint64_t groups) {
    return static_cast<unsigned>(std::min<uint64_t>(groups, 65535ULL));
}

unsigned norm_grid_warp(uint64_t groups) {
    const uint64_t blocks = (groups + 7ULL) / 8ULL;
    return static_cast<unsigned>(std::min<uint64_t>(blocks, 65535ULL));
}

inline bool is_aligned16(const void *p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

} // namespace

#define TRY_NORM(expr) do { const int code = (expr); if (code) return code; } while (0)

extern "C" int fa_cuda_norm_create(Context *ctx, FaNormDims d,
                                   const FaNormAxis *outer_axes,
                                   const FaNormAxis *inner_axes,
                                   size_t *workspace, size_t *shared, void **out) {
    if (!ctx || !workspace || !shared || !out || d.kind < 0 || d.kind > 1 ||
        d.outer_rank < 0 || d.outer_rank > 63 || d.inner_rank <= 0 || d.inner_rank > 63 ||
        !(d.eps > 0.0f) || !std::isfinite(d.eps)) {
        return rt(cudaErrorInvalidValue);
    }
    TRY_NORM(activate(ctx));
    auto p = std::unique_ptr<NormPlanCuda>(new (std::nothrow) NormPlanCuda(ctx, d));
    if (!p) return 1002;
    size_t outer_bytes = 0, inner_bytes = 0;
    if (!d.fully_contiguous) {
        outer_bytes = static_cast<size_t>(d.outer_rank) * sizeof(FaNormAxis);
        inner_bytes = static_cast<size_t>(d.inner_rank) * sizeof(FaNormAxis);
        if (outer_bytes) {
            TRY_NORM(rt(cudaMalloc(&p->outer_axes, outer_bytes)));
            TRY_NORM(rt(cudaMemcpyAsync(
                p->outer_axes, outer_axes, outer_bytes, cudaMemcpyHostToDevice, ctx->stream)));
        }
        if (inner_bytes) {
            TRY_NORM(rt(cudaMalloc(&p->inner_axes, inner_bytes)));
            TRY_NORM(rt(cudaMemcpyAsync(
                p->inner_axes, inner_axes, inner_bytes, cudaMemcpyHostToDevice, ctx->stream)));
        }
        if (outer_bytes || inner_bytes) {
            TRY_NORM(rt(cudaStreamSynchronize(ctx->stream)));
        }
    }
    if (d.fully_contiguous && d.inner_len <= 128) {
        cudaFuncAttributes attr{};
        TRY_NORM(rt(cudaFuncGetAttributes(&attr, norm_fwd_warp<false>)));
        *shared = attr.sharedSizeBytes;
    } else {
        cudaFuncAttributes attr{};
        TRY_NORM(rt(cudaFuncGetAttributes(&attr, norm_fwd_contig_reg<false>)));
        *shared = attr.sharedSizeBytes;
    }
    *workspace = outer_bytes + inner_bytes;
    *out = p.release();
    return 0;
}

extern "C" int fa_cuda_norm_forward_run(const NormPlanCuda *p, const float *x, float *y) {
    if (!p) return rt(cudaErrorInvalidValue);
    const auto d = p->dims;
    if (d.groups == 0 || d.inner_len == 0) return 0;
    if (!x || !y) return rt(cudaErrorInvalidValue);
    TRY_NORM(activate(p->ctx));
    const bool rms = (d.kind == 1);
    if (d.fully_contiguous) {
        if (d.inner_len <= 128) {
            const unsigned g = norm_grid_warp(d.groups);
            if (rms) norm_fwd_warp<true><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
            else norm_fwd_warp<false><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
        } else if (d.inner_len <= 1024) {
            const unsigned g = norm_grid(d.groups);
            const bool vec4 = (d.inner_len & 3u) == 0u &&
                              is_aligned16(x + d.offsets[0]) && is_aligned16(y + d.offsets[1]);
            if (vec4) {
                if (rms) norm_fwd_contig_vec4<true><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
                else norm_fwd_contig_vec4<false><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
            } else {
                if (rms) norm_fwd_contig_reg<true><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
                else norm_fwd_contig_reg<false><<<g, 256, 0, p->ctx->stream>>>(d, x, y);
            }
        } else {
            const unsigned g = norm_grid(d.groups);
            if (rms) norm_fwd_general<true, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, nullptr, x, y);
            else norm_fwd_general<false, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, nullptr, x, y);
        }
    } else {
        const unsigned g = norm_grid(d.groups);
        if (rms) norm_fwd_general<true, false><<<g, 256, 0, p->ctx->stream>>>(d, p->outer_axes, p->inner_axes, x, y);
        else norm_fwd_general<false, false><<<g, 256, 0, p->ctx->stream>>>(d, p->outer_axes, p->inner_axes, x, y);
    }
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_norm_backward_run(const NormPlanCuda *p, const float *x, const float *dy,
                                         float *dx, int accumulate) {
    if (!p || (accumulate != 0 && accumulate != 1)) return rt(cudaErrorInvalidValue);
    const auto d = p->dims;
    if (d.groups == 0 || d.inner_len == 0) return 0;
    if (!x || !dy || !dx) return rt(cudaErrorInvalidValue);
    TRY_NORM(activate(p->ctx));
    const bool rms = (d.kind == 1);
    const bool acc = (accumulate != 0);
    if (d.fully_contiguous) {
        if (d.inner_len <= 128) {
            const unsigned g = norm_grid_warp(d.groups);
            if (rms) {
                if (acc) norm_bwd_warp<true, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                else norm_bwd_warp<true, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
            } else {
                if (acc) norm_bwd_warp<false, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                else norm_bwd_warp<false, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
            }
        } else if (d.inner_len <= 1024) {
            const unsigned g = norm_grid(d.groups);
            if (rms) {
                if (acc) norm_bwd_contig_reg<true, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                else norm_bwd_contig_reg<true, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
            } else {
                if (acc) norm_bwd_contig_reg<false, true><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
                else norm_bwd_contig_reg<false, false><<<g, 256, 0, p->ctx->stream>>>(d, x, dy, dx);
            }
        } else {
            const unsigned g = norm_grid(d.groups);
            if (rms) {
                if (acc) norm_bwd_general<true, true, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, nullptr, x, dy, dx);
                else norm_bwd_general<true, false, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, nullptr, x, dy, dx);
            } else {
                if (acc) norm_bwd_general<false, true, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, nullptr, x, dy, dx);
                else norm_bwd_general<false, false, true><<<g, 256, 0, p->ctx->stream>>>(d, nullptr, nullptr, x, dy, dx);
            }
        }
    } else {
        const unsigned g = norm_grid(d.groups);
        if (rms) {
            if (acc) norm_bwd_general<true, true, false><<<g, 256, 0, p->ctx->stream>>>(d, p->outer_axes, p->inner_axes, x, dy, dx);
            else norm_bwd_general<true, false, false><<<g, 256, 0, p->ctx->stream>>>(d, p->outer_axes, p->inner_axes, x, dy, dx);
        } else {
            if (acc) norm_bwd_general<false, true, false><<<g, 256, 0, p->ctx->stream>>>(d, p->outer_axes, p->inner_axes, x, dy, dx);
            else norm_bwd_general<false, false, false><<<g, 256, 0, p->ctx->stream>>>(d, p->outer_axes, p->inner_axes, x, dy, dx);
        }
    }
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_norm_destroy(NormPlanCuda *p) { delete p; }
