#include "cuda_backend.h"
#include "reduce.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

namespace {
struct ReducePlan {
    Context *ctx;
    ReduceDims dims;
    ReduceGroupAxis *groups = nullptr;
    ReduceAxis *reduced = nullptr;
    ReducePlan(Context *c, ReduceDims d) : ctx(c), dims(d) {}
    ~ReducePlan() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (groups) cudaFree(groups);
        if (reduced) cudaFree(reduced);
    }
};

__device__ __forceinline__ bool is_pow2(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }
__device__ __forceinline__ unsigned log2_pow2(uint64_t v) { return static_cast<unsigned>(__ffsll(static_cast<long long>(v)) - 1); }

__device__ void group_offsets(ReduceDims d, const ReduceGroupAxis *axes, uint64_t flat,
    int64_t &x, int64_t &y) {
    x = d.source_offset; y = d.destination_offset;
    for (int a = d.rank - 1; a > 0; --a) {
        const auto c = static_cast<int64_t>(flat % axes[a].dim);
        flat /= axes[a].dim;
        x += c * axes[a].source; y += c * axes[a].destination;
    }
    if (d.rank) { x += static_cast<int64_t>(flat) * axes[0].source; y += static_cast<int64_t>(flat) * axes[0].destination; }
}
__device__ int64_t reduction_offset(ReduceDims d, const ReduceAxis *axes, uint64_t flat) {
    int64_t x = 0;
    for (int a = d.reduction_rank - 1; a > 0; --a) {
        x += static_cast<int64_t>(flat % axes[a].dim) * axes[a].stride;
        flat /= axes[a].dim;
    }
    if (d.reduction_rank) x += static_cast<int64_t>(flat) * axes[0].stride;
    return x;
}
__device__ __forceinline__ float empty_reduce_value(int operation) {
    if (operation == 0) return 0.0f;
    if (operation == 1) return __uint_as_float(0x7fc00000u);
    if (operation == 2 || operation == 4) return -INFINITY;
    return INFINITY;
}

__device__ __forceinline__ float max_nan_dev(float a, float b) {
    if (isnan(a) || isnan(b)) return __uint_as_float(0x7fc00000u);
    if (a > b) return a;
    if (b > a) return b;
    return (!signbit(a)) ? a : b;
}

__device__ __forceinline__ float min_nan_dev(float a, float b) {
    if (isnan(a) || isnan(b)) return __uint_as_float(0x7fc00000u);
    if (a < b) return a;
    if (b < a) return b;
    return signbit(a) ? a : b;
}

__device__ __forceinline__ float mean(ReduceDims d, float value) {
    return d.operation == 1 && d.count > 1 ? __fdiv_rn(value, d.divisor) : value;
}

__global__ void copy_or_fill(ReduceDims d, const ReduceGroupAxis *axes,
    const float *source, float *destination, int accumulate) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d.rank <= 1) {
        const int64_t sx = (d.rank == 1) ? axes[0].source : 0;
        const int64_t sy = (d.rank == 1) ? axes[0].destination : 0;
        for (uint64_t g = start; g < d.groups; g += stride) {
            const int64_t x = d.source_offset + static_cast<int64_t>(g) * sx;
            const int64_t y = d.destination_offset + static_cast<int64_t>(g) * sy;
            const float value = (d.kind == 1)
                ? empty_reduce_value(d.operation)
                : mean(d, source[x]);
            destination[y] = accumulate ? __fadd_rn(destination[y], value) : value;
        }
        return;
    }
    if (d.rank == 2 && d.groups <= 0xffffffffULL) {
        const uint32_t dim1 = static_cast<uint32_t>(axes[1].dim);
        const int64_t s0x = axes[0].source, s0y = axes[0].destination;
        const int64_t s1x = axes[1].source, s1y = axes[1].destination;
        const bool p2 = is_pow2(dim1);
        const uint32_t mask = dim1 - 1;
        const unsigned shift = p2 ? log2_pow2(dim1) : 0;
        for (uint64_t g = start; g < d.groups; g += stride) {
            const uint32_t gi = static_cast<uint32_t>(g);
            const uint32_t c1 = p2 ? (gi & mask) : (gi % dim1);
            const uint32_t c0 = p2 ? (gi >> shift) : (gi / dim1);
            const int64_t x = d.source_offset + static_cast<int64_t>(c0) * s0x + static_cast<int64_t>(c1) * s1x;
            const int64_t y = d.destination_offset + static_cast<int64_t>(c0) * s0y + static_cast<int64_t>(c1) * s1y;
            const float value = (d.kind == 1)
                ? empty_reduce_value(d.operation)
                : mean(d, source[x]);
            destination[y] = accumulate ? __fadd_rn(destination[y], value) : value;
        }
        return;
    }
    for (uint64_t g = start; g < d.groups; g += stride) {
        int64_t x, y;
        group_offsets(d, axes, g, x, y);
        const float value = (d.kind == 1)
            ? empty_reduce_value(d.operation)
            : mean(d, source[x]);
        destination[y] = accumulate ? __fadd_rn(destination[y], value) : value;
    }
}

__device__ __forceinline__ float nl_warp_ext(float v, bool is_min) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, offset);
        v = is_min ? min_nan_dev(v, o) : max_nan_dev(v, o);
    }
    return v;
}

__device__ __forceinline__ float nl_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ uint64_t nl_warp_sum_u64(uint64_t v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float nl_block_ext(float v, bool is_min, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = nl_warp_ext(v, is_min);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < 8) ? smem[lane] : (is_min ? INFINITY : -INFINITY);
        x = nl_warp_ext(x, is_min);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    const float res = smem[0];
    __syncthreads();
    return res;
}

__device__ __forceinline__ float nl_block_sum(float v, float *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = nl_warp_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < 8) ? smem[lane] : 0.0f;
        x = nl_warp_sum(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    const float res = smem[0];
    __syncthreads();
    return res;
}

__device__ __forceinline__ uint64_t nl_block_sum_u64(uint64_t v, uint64_t *smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = nl_warp_sum_u64(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        uint64_t x = (lane < 8) ? smem[lane] : 0ULL;
        x = nl_warp_sum_u64(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    const uint64_t res = smem[0];
    __syncthreads();
    return res;
}

__global__ void reduce_nl_forward_warp(ReduceDims d, const ReduceGroupAxis *groups, const ReduceAxis *axes,
    const float *source, float *destination) {
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const bool linear_red = (d.reduction_rank == 1);
    const int64_t r_step = linear_red ? axes[0].stride : 0;
    const bool is_min = (d.operation == 3);
    const bool is_lse = (d.operation == 4);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        int64_t x, y;
        group_offsets(d, groups, g, x, y);
        float reg[4];
        int nreg = 0;
        float acc = is_min ? INFINITY : -INFINITY;
        for (uint64_t r = lane; r < d.count; r += 32) {
            const int64_t ro = linear_red ? static_cast<int64_t>(r) * r_step : reduction_offset(d, axes, r);
            const float v = source[x + ro];
            reg[nreg++] = v;
            acc = (nreg == 1) ? v : (is_min ? min_nan_dev(acc, v) : max_nan_dev(acc, v));
        }
        const float extremum = nl_warp_ext(acc, is_min);
        if (!is_lse || isnan(extremum) || isinf(extremum)) {
            if (lane == 0) destination[y] = extremum;
            continue;
        }
        float sum = 0.0f;
        for (int k = 0; k < nreg; ++k) {
            sum = __fadd_rn(sum, expf(reg[k] - extremum));
        }
        sum = nl_warp_sum(sum);
        if (lane == 0) destination[y] = __fadd_rn(extremum, logf(sum));
    }
}

__global__ void reduce_nl_forward(ReduceDims d, const ReduceGroupAxis *groups, const ReduceAxis *axes,
    const float *source, float *destination) {
    __shared__ float smem[8];
    const bool linear_red = (d.reduction_rank == 1);
    const int64_t r_step = linear_red ? axes[0].stride : 0;
    const bool is_min = (d.operation == 3);
    const bool is_lse = (d.operation == 4);
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        int64_t x, y;
        group_offsets(d, groups, g, x, y);
        float acc = is_min ? INFINITY : -INFINITY;
        bool has_val = false;
        for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
            const int64_t ro = linear_red ? static_cast<int64_t>(r) * r_step : reduction_offset(d, axes, r);
            const float v = source[x + ro];
            if (!has_val) { acc = v; has_val = true; }
            else acc = is_min ? min_nan_dev(acc, v) : max_nan_dev(acc, v);
        }
        const float extremum = nl_block_ext(acc, is_min, smem);
        if (!is_lse || isnan(extremum) || isinf(extremum)) {
            if (threadIdx.x == 0) destination[y] = extremum;
            continue;
        }
        float sum = 0.0f;
        for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
            const int64_t ro = linear_red ? static_cast<int64_t>(r) * r_step : reduction_offset(d, axes, r);
            sum = __fadd_rn(sum, expf(source[x + ro] - extremum));
        }
        sum = nl_block_sum(sum, smem);
        if (threadIdx.x == 0) destination[y] = __fadd_rn(extremum, logf(sum));
    }
}

struct ReduceNlBwdPlan {
    Context *ctx;
    ReduceNlBwdDims dims;
    ReduceNlGroupAxis *groups = nullptr;
    ReduceNlRedAxis *reduced = nullptr;
    ReduceNlBwdPlan(Context *c, ReduceNlBwdDims d) : ctx(c), dims(d) {}
    ~ReduceNlBwdPlan() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (groups) cudaFree(groups);
        if (reduced) cudaFree(reduced);
    }
};

__device__ __forceinline__ void nl_group_offsets(
    ReduceNlBwdDims d, const ReduceNlGroupAxis *axes, uint64_t g,
    int64_t &sx, int64_t &su, int64_t &sd) {
    sx = d.source_offset;
    su = d.upstream_offset;
    sd = d.destination_offset;
    for (int a = d.rank - 1; a >= 0; --a) {
        const int64_t c = static_cast<int64_t>(g % axes[a].dim);
        g /= axes[a].dim;
        sx += c * axes[a].source;
        su += c * axes[a].upstream;
        sd += c * axes[a].destination;
    }
}

__device__ __forceinline__ void nl_red_offsets(
    ReduceNlBwdDims d, const ReduceNlRedAxis *axes, uint64_t r,
    int64_t &rx, int64_t &rd) {
    rx = 0;
    rd = 0;
    for (int a = d.reduction_rank - 1; a >= 0; --a) {
        const int64_t c = static_cast<int64_t>(r % axes[a].dim);
        r /= axes[a].dim;
        rx += c * axes[a].source;
        rd += c * axes[a].destination;
    }
}

__global__ void reduce_nl_backward_warp(ReduceNlBwdDims d,
    const ReduceNlGroupAxis *groups, const ReduceNlRedAxis *axes,
    const float *source, const float *upstream, float *destination, int accumulate) {
    const int lane = threadIdx.x & 31;
    const uint64_t total_warps = static_cast<uint64_t>(gridDim.x) * (blockDim.x >> 5);
    const uint64_t warp_id = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const bool is_min = (d.operation == 3);
    const bool is_lse = (d.operation == 4);
    for (uint64_t g = warp_id; g < d.groups; g += total_warps) {
        int64_t sx, su, sd;
        nl_group_offsets(d, groups, g, sx, su, sd);
        const float u = upstream[su];
        float reg_v[4];
        int64_t reg_dst[4];
        int nreg = 0;
        float acc = is_min ? INFINITY : -INFINITY;
        for (uint64_t r = lane; r < d.count; r += 32) {
            int64_t rx, rd;
            nl_red_offsets(d, axes, r, rx, rd);
            const float v = source[sx + rx];
            reg_v[nreg] = v;
            reg_dst[nreg] = sd + rd;
            ++nreg;
            acc = (nreg == 1) ? v : (is_min ? min_nan_dev(acc, v) : max_nan_dev(acc, v));
        }
        const float extremum = nl_warp_ext(acc, is_min);
        if (is_lse) {
            if (isnan(extremum) || isinf(extremum)) {
                for (int k = 0; k < nreg; ++k) {
                    const float grad = (isinf(extremum) && !signbit(extremum) && isfinite(reg_v[k]))
                        ? 0.0f : __uint_as_float(0x7fc00000u);
                    const int64_t dst = reg_dst[k];
                    destination[dst] = accumulate ? __fadd_rn(destination[dst], grad) : grad;
                }
                continue;
            }
            float sum = 0.0f;
            for (int k = 0; k < nreg; ++k) {
                const float e = expf(reg_v[k] - extremum);
                reg_v[k] = e;
                sum = __fadd_rn(sum, e);
            }
            const float scale = __fdiv_rn(u, nl_warp_sum(sum));
            for (int k = 0; k < nreg; ++k) {
                const float grad = reg_v[k] * scale;
                const int64_t dst = reg_dst[k];
                destination[dst] = accumulate ? __fadd_rn(destination[dst], grad) : grad;
            }
        } else {
            if (isnan(extremum)) {
                for (int k = 0; k < nreg; ++k) {
                    const int64_t dst = reg_dst[k];
                    destination[dst] = accumulate
                        ? __fadd_rn(destination[dst], __uint_as_float(0x7fc00000u))
                        : __uint_as_float(0x7fc00000u);
                }
                continue;
            }
            uint64_t ties = 0;
            for (int k = 0; k < nreg; ++k) {
                if (reg_v[k] == extremum) ++ties;
            }
            const uint64_t total_ties = nl_warp_sum_u64(ties);
            const float scale = __fdiv_rn(u, static_cast<float>(total_ties));
            for (int k = 0; k < nreg; ++k) {
                const float grad = (reg_v[k] == extremum) ? scale : 0.0f;
                const int64_t dst = reg_dst[k];
                destination[dst] = accumulate ? __fadd_rn(destination[dst], grad) : grad;
            }
        }
    }
}

__global__ void reduce_nl_backward(ReduceNlBwdDims d,
    const ReduceNlGroupAxis *groups, const ReduceNlRedAxis *axes,
    const float *source, const float *upstream, float *destination, int accumulate) {
    __shared__ float smem[8];
    __shared__ uint64_t smem_u64[8];
    const bool is_min = (d.operation == 3);
    const bool is_lse = (d.operation == 4);
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        int64_t sx, su, sd;
        nl_group_offsets(d, groups, g, sx, su, sd);
        const float u = upstream[su];

        float acc = is_min ? INFINITY : -INFINITY;
        bool has_val = false;
        for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
            int64_t rx, rd;
            nl_red_offsets(d, axes, r, rx, rd);
            const float v = source[sx + rx];
            if (!has_val) { acc = v; has_val = true; }
            else acc = is_min ? min_nan_dev(acc, v) : max_nan_dev(acc, v);
        }
        const float extremum = nl_block_ext(acc, is_min, smem);

        if (is_lse) {
            if (isnan(extremum) || isinf(extremum)) {
                for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
                    int64_t rx, rd;
                    nl_red_offsets(d, axes, r, rx, rd);
                    const float v = source[sx + rx];
                    const float grad = (isinf(extremum) && !signbit(extremum) && isfinite(v))
                        ? 0.0f : __uint_as_float(0x7fc00000u);
                    const int64_t dst = sd + rd;
                    destination[dst] = accumulate ? __fadd_rn(destination[dst], grad) : grad;
                }
                continue;
            }
            float sum = 0.0f;
            for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
                int64_t rx, rd;
                nl_red_offsets(d, axes, r, rx, rd);
                sum = __fadd_rn(sum, expf(source[sx + rx] - extremum));
            }
            const float scale = __fdiv_rn(u, nl_block_sum(sum, smem));
            for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
                int64_t rx, rd;
                nl_red_offsets(d, axes, r, rx, rd);
                const float grad = expf(source[sx + rx] - extremum) * scale;
                const int64_t dst = sd + rd;
                destination[dst] = accumulate ? __fadd_rn(destination[dst], grad) : grad;
            }
        } else {
            if (isnan(extremum)) {
                for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
                    int64_t rx, rd;
                    nl_red_offsets(d, axes, r, rx, rd);
                    const int64_t dst = sd + rd;
                    destination[dst] = accumulate
                        ? __fadd_rn(destination[dst], __uint_as_float(0x7fc00000u))
                        : __uint_as_float(0x7fc00000u);
                }
                continue;
            }
            uint64_t ties = 0;
            for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
                int64_t rx, rd;
                nl_red_offsets(d, axes, r, rx, rd);
                if (source[sx + rx] == extremum) ++ties;
            }
            const uint64_t total_ties = nl_block_sum_u64(ties, smem_u64);
            const float scale = __fdiv_rn(u, static_cast<float>(total_ties));
            for (uint64_t r = threadIdx.x; r < d.count; r += 256) {
                int64_t rx, rd;
                nl_red_offsets(d, axes, r, rx, rd);
                const float grad = (source[sx + rx] == extremum) ? scale : 0.0f;
                const int64_t dst = sd + rd;
                destination[dst] = accumulate ? __fadd_rn(destination[dst], grad) : grad;
            }
        }
    }
}

__global__ void reduce_small_2(ReduceDims d, const ReduceGroupAxis *groups, const ReduceAxis *axes,
    const float *source, float *destination) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t r_step = (d.reduction_rank == 1) ? axes[0].stride : reduction_offset(d, axes, 1);
    if (d.rank <= 1) {
        const int64_t sx = (d.rank == 1) ? groups[0].source : 0;
        const int64_t sy = (d.rank == 1) ? groups[0].destination : 0;
        if (d.rank == 1 && sx == 2 && r_step == 1
            && ((reinterpret_cast<uintptr_t>(source + d.source_offset) & 7u) == 0)) {
            const float2 *src2 = reinterpret_cast<const float2 *>(source + d.source_offset);
            for (uint64_t g = start; g < d.groups; g += stride) {
                const float2 p = src2[g];
                const float sum = __fadd_rn(__fadd_rn(0.0f, p.x), __fadd_rn(0.0f, p.y));
                destination[d.destination_offset + static_cast<int64_t>(g) * sy] = mean(d, sum);
            }
            return;
        }
        for (uint64_t g = start; g < d.groups; g += stride) {
            const int64_t x = d.source_offset + static_cast<int64_t>(g) * sx;
            const int64_t y = d.destination_offset + static_cast<int64_t>(g) * sy;
            const float sum = __fadd_rn(__fadd_rn(0.0f, source[x]), __fadd_rn(0.0f, source[x + r_step]));
            destination[y] = mean(d, sum);
        }
        return;
    }
    for (uint64_t g = start; g < d.groups; g += stride) {
        int64_t x, y;
        group_offsets(d, groups, g, x, y);
        const float sum = __fadd_rn(__fadd_rn(0.0f, source[x]), __fadd_rn(0.0f, source[x + r_step]));
        destination[y] = mean(d, sum);
    }
}

__global__ void reduce_small_8(ReduceDims d, const ReduceGroupAxis *groups, const ReduceAxis *axes,
    const float *source, float *destination) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool linear_group = (d.rank <= 1);
    const int64_t sx = (d.rank == 1) ? groups[0].source : 0;
    const int64_t sy = (d.rank == 1) ? groups[0].destination : 0;
    const bool linear_red = (d.reduction_rank == 1);
    const int64_t r_step = linear_red ? axes[0].stride : 0;
    for (uint64_t g = start; g < d.groups; g += stride) {
        int64_t x, y;
        if (linear_group) {
            x = d.source_offset + static_cast<int64_t>(g) * sx;
            y = d.destination_offset + static_cast<int64_t>(g) * sy;
        } else {
            group_offsets(d, groups, g, x, y);
        }
        float t[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (linear_red) {
            for (uint64_t r = 0; r < d.count; ++r)
                t[r] = __fadd_rn(0.0f, source[x + static_cast<int64_t>(r) * r_step]);
        } else {
            for (uint64_t r = 0; r < d.count; ++r)
                t[r] = __fadd_rn(0.0f, source[x + reduction_offset(d, axes, r)]);
        }
        for (int i = 0; i < 4; ++i) t[i] = __fadd_rn(t[i], t[i + 4]);
        for (int i = 0; i < 2; ++i) t[i] = __fadd_rn(t[i], t[i + 2]);
        destination[y] = mean(d, __fadd_rn(t[0], t[1]));
    }
}

__global__ void reduce_halfwarp(ReduceDims d, const ReduceGroupAxis *groups, const ReduceAxis *axes,
    const float *source, float *destination) {
    const unsigned lane = threadIdx.x & 15;
    const uint64_t padded_groups = (d.groups + 1ULL) & ~1ULL;
    const bool linear_group = (d.rank <= 1);
    const int64_t sx = (d.rank == 1) ? groups[0].source : 0;
    const int64_t sy = (d.rank == 1) ? groups[0].destination : 0;
    const bool linear_red = (d.reduction_rank == 1);
    const int64_t r_step = linear_red ? axes[0].stride : 0;
    for (uint64_t g = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 16;
         g < padded_groups; g += static_cast<uint64_t>(gridDim.x) * (blockDim.x / 16)) {
        float sum = 0.0f;
        int64_t x = 0, y = 0;
        if (g < d.groups) {
            if (linear_group) {
                x = d.source_offset + static_cast<int64_t>(g) * sx;
                y = d.destination_offset + static_cast<int64_t>(g) * sy;
            } else {
                group_offsets(d, groups, g, x, y);
            }
            if (lane < d.count) {
                const int64_t r_off = linear_red
                    ? static_cast<int64_t>(lane) * r_step
                    : reduction_offset(d, axes, lane);
                sum = __fadd_rn(0.0f, source[x + r_off]);
            }
        }
        // Every warp executes all 16-lane sub-warp shuffles with full 32-lane participation.
        for (unsigned step = 8; step; step /= 2)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step, 16));
        if (g < d.groups && lane == 0) destination[y] = mean(d, sum);
    }
}

__global__ void reduce_warp(ReduceDims d, const ReduceGroupAxis *groups, const ReduceAxis *axes,
    const float *source, float *destination) {
    const unsigned lane = threadIdx.x & 31;
    const bool linear_group = (d.rank <= 1);
    const int64_t sx = (d.rank == 1) ? groups[0].source : 0;
    const int64_t sy = (d.rank == 1) ? groups[0].destination : 0;
    const bool linear_red = (d.reduction_rank == 1);
    const int64_t r_step = linear_red ? axes[0].stride : 0;
    for (uint64_t g = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         g < d.groups; g += static_cast<uint64_t>(gridDim.x) * (blockDim.x / 32)) {
        int64_t x, y;
        if (linear_group) {
            x = d.source_offset + static_cast<int64_t>(g) * sx;
            y = d.destination_offset + static_cast<int64_t>(g) * sy;
        } else {
            group_offsets(d, groups, g, x, y);
        }
        float sum = 0.0f;
        if (linear_red) {
            for (uint64_t r = lane; r < d.count; r += 32)
                sum = __fadd_rn(sum, source[x + static_cast<int64_t>(r) * r_step]);
        } else {
            for (uint64_t r = lane; r < d.count; r += 32)
                sum = __fadd_rn(sum, source[x + reduction_offset(d, axes, r)]);
        }
        // Each active group is a complete warp; all named lanes reach every shuffle.
        for (unsigned step = 16; step; step /= 2)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
        if (lane == 0) destination[y] = mean(d, sum);
    }
}

__global__ void reduce_block(ReduceDims d, const ReduceGroupAxis *groups, const ReduceAxis *axes,
    const float *source, float *destination) {
    __shared__ float partial[256];
    const bool linear_group = (d.rank <= 1);
    const int64_t sx = (d.rank == 1) ? groups[0].source : 0;
    const int64_t sy = (d.rank == 1) ? groups[0].destination : 0;
    const bool linear_red = (d.reduction_rank == 1);
    const bool two_red = (d.reduction_rank == 2 && d.count <= 0xffffffffULL);
    const int64_t r0_step = (d.reduction_rank >= 1) ? axes[0].stride : 0;
    const int64_t r1_step = two_red ? axes[1].stride : 0;
    const uint32_t r1_dim = two_red ? static_cast<uint32_t>(axes[1].dim) : 1;
    const bool r1_p2 = two_red && is_pow2(r1_dim);
    const uint32_t r1_mask = r1_dim - 1;
    const unsigned r1_shift = r1_p2 ? log2_pow2(r1_dim) : 0;
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        int64_t x, y;
        if (linear_group) {
            x = d.source_offset + static_cast<int64_t>(g) * sx;
            y = d.destination_offset + static_cast<int64_t>(g) * sy;
        } else {
            group_offsets(d, groups, g, x, y);
        }
        float sum = 0.0f;
        if (linear_red) {
            for (uint64_t r = threadIdx.x; r < d.count; r += 256)
                sum = __fadd_rn(sum, source[x + static_cast<int64_t>(r) * r0_step]);
        } else if (two_red) {
            const uint32_t limit = static_cast<uint32_t>(d.count);
            const uint32_t q256 = 256u / r1_dim;
            const uint32_t rem256 = 256u - q256 * r1_dim;
            uint32_t c0 = r1_p2 ? (threadIdx.x >> r1_shift) : (threadIdx.x / r1_dim);
            uint32_t c1 = r1_p2 ? (threadIdx.x & r1_mask) : (threadIdx.x - c0 * r1_dim);
            for (uint32_t r = threadIdx.x; r < limit; r += 256) {
                sum = __fadd_rn(sum, source[x + static_cast<int64_t>(c0) * r0_step + static_cast<int64_t>(c1) * r1_step]);
                c0 += q256;
                c1 += rem256;
                if (c1 >= r1_dim) { c1 -= r1_dim; ++c0; }
            }
        } else {
            for (uint64_t r = threadIdx.x; r < d.count; r += 256)
                sum = __fadd_rn(sum, source[x + reduction_offset(d, axes, r)]);
        }
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned step = 128; step; step /= 2) {
            if (threadIdx.x < step) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + step]);
            __syncthreads();
        }
        if (threadIdx.x == 0) destination[y] = mean(d, partial[0]);
        __syncthreads(); // A later group must not overwrite a still-used partial.
    }
}

__global__ void reduce_block_split(ReduceDims d, const ReduceGroupAxis *groups, const ReduceAxis *axes,
    const float *source, float *scratch) {
    const uint64_t g = blockIdx.x >> 3;
    if (g >= d.groups) return;
    const unsigned lane = ((blockIdx.x & 7u) << 5) + threadIdx.x;
    int64_t x, y;
    if (d.rank <= 1) {
        const int64_t sx = (d.rank == 1) ? groups[0].source : 0;
        x = d.source_offset + static_cast<int64_t>(g) * sx;
    } else {
        group_offsets(d, groups, g, x, y);
    }
    float sum = 0.0f;
    if (d.reduction_rank == 1) {
        const int64_t step = axes[0].stride;
        uint64_t r = lane;
        for (; r + 768ULL < d.count; r += 1024ULL) {
            const float v0 = source[x + static_cast<int64_t>(r) * step];
            const float v1 = source[x + static_cast<int64_t>(r + 256ULL) * step];
            const float v2 = source[x + static_cast<int64_t>(r + 512ULL) * step];
            const float v3 = source[x + static_cast<int64_t>(r + 768ULL) * step];
            sum = __fadd_rn(sum, v0);
            sum = __fadd_rn(sum, v1);
            sum = __fadd_rn(sum, v2);
            sum = __fadd_rn(sum, v3);
        }
        for (; r < d.count; r += 256ULL)
            sum = __fadd_rn(sum, source[x + static_cast<int64_t>(r) * step]);
    } else if (d.reduction_rank == 2 && d.count <= 0xffffffffULL) {
        const uint32_t limit = static_cast<uint32_t>(d.count);
        const uint32_t r1_dim = static_cast<uint32_t>(axes[1].dim);
        const int64_t r0_step = axes[0].stride;
        const int64_t r1_step = axes[1].stride;
        const uint32_t q256 = 256u / r1_dim;
        const uint32_t rem256 = 256u - q256 * r1_dim;
        uint32_t c0 = lane / r1_dim;
        uint32_t c1 = lane - c0 * r1_dim;
        for (uint32_t r = lane; r < limit; r += 256u) {
            sum = __fadd_rn(sum, source[x + static_cast<int64_t>(c0) * r0_step + static_cast<int64_t>(c1) * r1_step]);
            c0 += q256;
            c1 += rem256;
            if (c1 >= r1_dim) { c1 -= r1_dim; ++c0; }
        }
    } else {
        for (uint64_t r = lane; r < d.count; r += 256ULL)
            sum = __fadd_rn(sum, source[x + reduction_offset(d, axes, r)]);
    }
    scratch[g * 256ULL + lane] = sum;
}

__global__ void reduce_block_finalize(ReduceDims d, const ReduceGroupAxis *groups,
    const float *scratch, float *destination) {
    __shared__ float partial[256];
    const uint64_t g = blockIdx.x;
    if (g >= d.groups) return;
    partial[threadIdx.x] = scratch[g * 256ULL + threadIdx.x];
    __syncthreads();
    for (unsigned step = 128; step; step /= 2) {
        if (threadIdx.x < step) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + step]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int64_t x, y;
        if (d.rank <= 1) {
            const int64_t sy = (d.rank == 1) ? groups[0].destination : 0;
            y = d.destination_offset + static_cast<int64_t>(g) * sy;
        } else {
            group_offsets(d, groups, g, x, y);
        }
        destination[y] = mean(d, partial[0]);
    }
}

unsigned blocks(uint64_t work, unsigned per_block) {
    return static_cast<unsigned>(std::min<uint64_t>((work + per_block - 1) / per_block, 65535));
}
}
#define TRY_REDUCE(expr) do { const int code = (expr); if (code) return code; } while (0)
extern "C" int fa_cuda_reduce_create(Context *ctx, ReduceDims d, const ReduceGroupAxis *groups,
    const ReduceAxis *reduced, size_t *workspace, size_t *shared, void **out) {
    if (!ctx || !workspace || !shared || !out || d.kind < 0 || d.kind > 5 || d.rank < 0 || d.rank > 63 ||
        d.reduction_rank < 0 || d.reduction_rank > 63 || d.operation < 0 || d.operation > 4 || d.reserved ||
        d.source_offset < 0 || d.destination_offset < 0 || !(d.divisor >= 1.0f) ||
        (d.rank && !groups) || (d.reduction_rank && !reduced) ||
        (d.kind == 0 && d.groups) || (d.kind == 1 && d.count) ||
        (d.kind == 2 && d.count != 1) || (d.kind == 3 && (d.count < 2 || d.count > 32)) ||
        (d.kind == 4 && d.count <= 32) || (d.kind == 5 && d.reduction_rank)) return rt(cudaErrorInvalidValue);
    TRY_REDUCE(activate(ctx));
    auto p = std::unique_ptr<ReducePlan>(new(std::nothrow) ReducePlan(ctx, d));
    if (!p) return 1002;
    const size_t group_bytes = static_cast<size_t>(d.rank) * sizeof(ReduceGroupAxis);
    const size_t reduce_bytes = static_cast<size_t>(d.reduction_rank) * sizeof(ReduceAxis);
    if (group_bytes) {
        TRY_REDUCE(rt(cudaMalloc(&p->groups, group_bytes)));
        TRY_REDUCE(rt(cudaMemcpyAsync(p->groups, groups, group_bytes, cudaMemcpyHostToDevice, ctx->stream)));
    }
    if (reduce_bytes) {
        TRY_REDUCE(rt(cudaMalloc(&p->reduced, reduce_bytes)));
        TRY_REDUCE(rt(cudaMemcpyAsync(p->reduced, reduced, reduce_bytes, cudaMemcpyHostToDevice, ctx->stream)));
    }
    if (group_bytes || reduce_bytes) TRY_REDUCE(rt(cudaStreamSynchronize(ctx->stream)));
    *shared = 0;
    if (d.operation >= 2 && (d.kind == 3 || d.kind == 4)) {
        cudaFuncAttributes attributes{};
        if (d.count <= 128) {
            TRY_REDUCE(rt(cudaFuncGetAttributes(&attributes, reduce_nl_forward_warp)));
        } else {
            TRY_REDUCE(rt(cudaFuncGetAttributes(&attributes, reduce_nl_forward)));
        }
        *shared = attributes.sharedSizeBytes;
    } else if (d.kind == 4) {
        cudaFuncAttributes attributes{};
        TRY_REDUCE(rt(cudaFuncGetAttributes(&attributes, reduce_block)));
        *shared = attributes.sharedSizeBytes;
    }
    *workspace = group_bytes + reduce_bytes;
    *out = p.release();
    return 0;
}
extern "C" int fa_cuda_reduce_run(const ReducePlan *p, const float *source, float *destination) {
    if (!p || p->dims.kind == 5 || (!source && p->dims.groups && p->dims.count) ||
        (!destination && p->dims.groups)) return rt(cudaErrorInvalidValue);
    if (!p->dims.groups) return 0;
    TRY_REDUCE(activate(p->ctx));
    const auto d = p->dims;
    if (d.kind == 1 || d.kind == 2) {
        copy_or_fill<<<blocks(d.groups, 256), 256, 0, p->ctx->stream>>>(d, p->groups, source, destination, 0);
    } else if (d.operation >= 2) {
        if (d.count <= 128) {
            reduce_nl_forward_warp<<<blocks(d.groups, 8), 256, 0, p->ctx->stream>>>(d, p->groups, p->reduced, source, destination);
        } else {
            reduce_nl_forward<<<blocks(d.groups, 1), 256, 0, p->ctx->stream>>>(d, p->groups, p->reduced, source, destination);
        }
    } else if (d.kind == 3) {
        if (d.count == 2)
            reduce_small_2<<<blocks(d.groups, 256), 256, 0, p->ctx->stream>>>(d, p->groups, p->reduced, source, destination);
        else if (d.count <= 8)
            reduce_small_8<<<blocks(d.groups, 256), 256, 0, p->ctx->stream>>>(d, p->groups, p->reduced, source, destination);
        else if (d.count <= 16)
            reduce_halfwarp<<<blocks((d.groups + 1ULL) & ~1ULL, 16), 256, 0, p->ctx->stream>>>(d, p->groups, p->reduced, source, destination);
        else
            reduce_warp<<<blocks(d.groups, 8), 256, 0, p->ctx->stream>>>(d, p->groups, p->reduced, source, destination);
    } else if (d.groups <= FA_CUDA_REDUCE_SCRATCH_GROUPS && d.count >= 4096 && p->ctx->reduce_scratch) {
        reduce_block_split<<<static_cast<unsigned>(d.groups * 8ULL), 32, 0, p->ctx->stream>>>(
            d, p->groups, p->reduced, source, p->ctx->reduce_scratch);
        reduce_block_finalize<<<static_cast<unsigned>(d.groups), 256, 0, p->ctx->stream>>>(
            d, p->groups, p->ctx->reduce_scratch, destination);
    } else {
        reduce_block<<<blocks(d.groups, 1), 256, 0, p->ctx->stream>>>(d, p->groups, p->reduced, source, destination);
    }
    return rt(cudaGetLastError());
}
extern "C" int fa_cuda_reduce_backward_run(const ReducePlan *p, const float *source, float *destination, int accumulate) {
    if (!p || p->dims.kind != 5 || (accumulate != 0 && accumulate != 1) ||
        (p->dims.groups && (!source || !destination))) return rt(cudaErrorInvalidValue);
    if (!p->dims.groups) return 0;
    TRY_REDUCE(activate(p->ctx));
    copy_or_fill<<<blocks(p->dims.groups, 256), 256, 0, p->ctx->stream>>>(p->dims, p->groups, source, destination, accumulate);
    return rt(cudaGetLastError());
}
extern "C" void fa_cuda_reduce_destroy(ReducePlan *p) { delete p; }

extern "C" int fa_cuda_reduce_nl_bwd_create(Context *ctx, ReduceNlBwdDims d,
    const ReduceNlGroupAxis *groups, const ReduceNlRedAxis *reduced,
    size_t *workspace, size_t *shared, void **out) {
    if (!ctx || !workspace || !shared || !out || d.rank < 0 || d.rank > 63 ||
        d.reduction_rank < 0 || d.reduction_rank > 63 || d.operation < 2 || d.operation > 4 ||
        d.reserved || (d.rank && !groups) || (d.reduction_rank && !reduced)) {
        return rt(cudaErrorInvalidValue);
    }
    TRY_REDUCE(activate(ctx));
    auto p = std::unique_ptr<ReduceNlBwdPlan>(new (std::nothrow) ReduceNlBwdPlan(ctx, d));
    if (!p) return 1002;
    const size_t group_bytes = static_cast<size_t>(d.rank) * sizeof(ReduceNlGroupAxis);
    const size_t reduce_bytes = static_cast<size_t>(d.reduction_rank) * sizeof(ReduceNlRedAxis);
    if (group_bytes) {
        TRY_REDUCE(rt(cudaMalloc(&p->groups, group_bytes)));
        TRY_REDUCE(rt(cudaMemcpyAsync(p->groups, groups, group_bytes, cudaMemcpyHostToDevice, ctx->stream)));
    }
    if (reduce_bytes) {
        TRY_REDUCE(rt(cudaMalloc(&p->reduced, reduce_bytes)));
        TRY_REDUCE(rt(cudaMemcpyAsync(p->reduced, reduced, reduce_bytes, cudaMemcpyHostToDevice, ctx->stream)));
    }
    if (group_bytes || reduce_bytes) TRY_REDUCE(rt(cudaStreamSynchronize(ctx->stream)));
    cudaFuncAttributes attr{};
    if (d.count <= 128) {
        TRY_REDUCE(rt(cudaFuncGetAttributes(&attr, reduce_nl_backward_warp)));
    } else {
        TRY_REDUCE(rt(cudaFuncGetAttributes(&attr, reduce_nl_backward)));
    }
    *shared = attr.sharedSizeBytes;
    *workspace = group_bytes + reduce_bytes;
    *out = p.release();
    return 0;
}

extern "C" int fa_cuda_reduce_nl_bwd_run(const ReduceNlBwdPlan *p,
    const float *source, const float *upstream, float *destination, int accumulate) {
    if (!p || (accumulate != 0 && accumulate != 1)) return rt(cudaErrorInvalidValue);
    if (!p->dims.groups || !p->dims.count) return 0;
    if (!source || !upstream || !destination) return rt(cudaErrorInvalidValue);
    TRY_REDUCE(activate(p->ctx));
    if (p->dims.count <= 128) {
        reduce_nl_backward_warp<<<blocks(p->dims.groups, 8), 256, 0, p->ctx->stream>>>(
            p->dims, p->groups, p->reduced, source, upstream, destination, accumulate);
    } else {
        reduce_nl_backward<<<blocks(p->dims.groups, 1), 256, 0, p->ctx->stream>>>(
            p->dims, p->groups, p->reduced, source, upstream, destination, accumulate);
    }
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_reduce_nl_bwd_destroy(ReduceNlBwdPlan *p) { delete p; }
