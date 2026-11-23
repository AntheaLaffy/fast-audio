#include "cuda_backend.h"
#include "view_backward.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <new>

namespace {
struct BackwardPlan {
    Context *ctx;
    ViewBackwardDims dims;
    uint64_t *metadata = nullptr;
    BackwardPlan(Context *c, ViewBackwardDims d) : ctx(c), dims(d) {}
    ~BackwardPlan() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (metadata) cudaFree(metadata);
    }
};
__device__ void offsets(const uint64_t *m, int rank, uint64_t flat,
                        int64_t &x, int64_t &y) {
    const auto sx = reinterpret_cast<const int64_t *>(m + rank);
    const auto sy = reinterpret_cast<const int64_t *>(m + 2 * rank);
    for (int axis = rank - 1; axis > 0; --axis) {
        const auto coordinate = static_cast<int64_t>(flat % m[axis]);
        flat /= m[axis]; x += coordinate * sx[axis]; y += coordinate * sy[axis];
    }
    if (rank) { x += static_cast<int64_t>(flat) * sx[0]; y += static_cast<int64_t>(flat) * sy[0]; }
}
__device__ int64_t reduction_offset(ViewBackwardDims d, const uint64_t *m, uint64_t flat) {
    const auto dims = m + 3 * d.rank;
    const auto strides = reinterpret_cast<const int64_t *>(dims + d.reduction_rank);
    int64_t out = 0;
    for (int axis = d.reduction_rank - 1; axis > 0; --axis) {
        out += static_cast<int64_t>(flat % dims[axis]) * strides[axis]; flat /= dims[axis];
    }
    if (d.reduction_rank) out += static_cast<int64_t>(flat) * strides[0];
    return out;
}
__device__ void store(float *destination, uint64_t target, float value, int accumulate) {
    destination[target] = accumulate ? __fadd_rn(destination[target], value) : value;
}
__global__ void injective(ViewBackwardDims d, const uint64_t *m,
                          const float *source, float *destination, int accumulate) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < d.logical_len; i += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        int64_t x = d.source_offset, y = d.destination_offset;
        if (d.kind == 2) offsets(m, d.rank, i, x, y);
        else { x += static_cast<int64_t>(i); y += static_cast<int64_t>(i); }
        store(destination, static_cast<uint64_t>(y), source[x], accumulate);
    }
}
__global__ void indexed(ViewBackwardDims d, const uint64_t *m,
                        const float *source, float *destination, int accumulate) {
    const auto sources = m + d.groups;
    for (uint64_t g = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         g < d.groups; g += static_cast<uint64_t>(blockDim.x) * gridDim.x)
        store(destination, m[g], source[sources[g]], accumulate);
}
template<bool GROUPED>
__device__ void group(ViewBackwardDims d, const uint64_t *m, uint64_t g,
    uint64_t &target, int64_t &base, uint64_t &begin, uint64_t &count) {
    if constexpr (GROUPED) {
        target = m[g]; begin = m[d.groups + g];
        count = m[d.groups + g + 1] - begin; base = 0;
    } else {
        base = d.source_offset;
        int64_t y = d.destination_offset;
        offsets(m, d.rank, g, base, y);
        target = static_cast<uint64_t>(y); begin = 0; count = d.reductions;
    }
}
template<bool GROUPED>
__device__ float contribution(ViewBackwardDims d, const uint64_t *m, const float *source,
    int64_t base, uint64_t begin, uint64_t r) {
    if constexpr (GROUPED) return source[m[2 * d.groups + 1 + begin + r]];
    else return source[base + reduction_offset(d, m, r)];
}
template<bool GROUPED>
__global__ void reduce_warp(ViewBackwardDims d, const uint64_t *m,
    const float *source, float *destination, int accumulate) {
    const unsigned lane = threadIdx.x & 31;
    for (uint64_t g = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         g < d.groups; g += static_cast<uint64_t>(gridDim.x) * (blockDim.x / 32)) {
        uint64_t target = 0, begin = 0, count = 0;
        int64_t base = 0;
        if (lane == 0) group<GROUPED>(d, m, g, target, base, begin, count);
        target = __shfl_sync(0xffffffff, static_cast<unsigned long long>(target), 0);
        begin = __shfl_sync(0xffffffff, static_cast<unsigned long long>(begin), 0);
        count = __shfl_sync(0xffffffff, static_cast<unsigned long long>(count), 0);
        base = __shfl_sync(0xffffffff, static_cast<long long>(base), 0);
        float sum = 0.0f;
        for (uint64_t r = lane; r < count; r += 32)
            sum = __fadd_rn(sum, contribution<GROUPED>(d, m, source, base, begin, r));
        // All lanes of each active group execute every shuffle with the same mask.
        for (unsigned step = 16; step; step /= 2)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
        if (lane == 0) store(destination, target, sum, accumulate);
    }
}
template<bool GROUPED>
__global__ void reduce_block(ViewBackwardDims d, const uint64_t *m,
    const float *source, float *destination, int accumulate) {
    __shared__ uint64_t target, begin, count;
    __shared__ int64_t base;
    __shared__ float partial[256];
    for (uint64_t g = blockIdx.x; g < d.groups; g += gridDim.x) {
        if (threadIdx.x == 0) group<GROUPED>(d, m, g, target, base, begin, count);
        __syncthreads();
        float sum = 0.0f;
        for (uint64_t r = threadIdx.x; r < count; r += 256)
            sum = __fadd_rn(sum, contribution<GROUPED>(d, m, source, base, begin, r));
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned step = 128; step; step /= 2) {
            if (threadIdx.x < step) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + step]);
            __syncthreads();
        }
        if (threadIdx.x == 0) store(destination, target, partial[0], accumulate);
        __syncthreads(); // A later group cannot overwrite shared geometry/partials early.
    }
}
unsigned blocks(uint64_t work, unsigned per_block) {
    return static_cast<unsigned>(std::min((work + per_block - 1) / per_block, uint64_t{65535}));
}
}
#define TRY_BACKWARD(expr) do { const int code = (expr); if (code) return code; } while (0)
extern "C" int fa_cuda_view_backward_create(Context *ctx, ViewBackwardDims d,
    const uint64_t *a, const uint64_t *b, const uint64_t *c, const uint64_t *e, const uint64_t *f,
    size_t *workspace, size_t *shared, void **out) {
    if (d.kind < 0 || d.kind > 5 || d.rank < 0 || d.rank > 63 || d.reduction_rank < 0 || d.reduction_rank > 63 ||
        d.written_begin > d.written_end || d.written_end > d.base_len || d.source_offset < 0 || d.destination_offset < 0 ||
        d.base_len > std::numeric_limits<size_t>::max() / sizeof(float) ||
        d.logical_len > std::numeric_limits<size_t>::max() / sizeof(float)) return rt(cudaErrorInvalidValue);
    if ((d.kind == 0 && (d.logical_len || d.groups)) ||
        (d.kind != 0 && (!d.logical_len || !d.groups)) ||
        ((d.kind == 1 || d.kind == 2 || d.kind == 4) && (d.lanes != 1 || d.groups != d.logical_len)) ||
        ((d.kind == 3 || d.kind == 5) && d.lanes != 32 && d.lanes != 256) ||
        (d.kind == 2 && !d.rank) ||
        (d.kind == 3 && (!d.reductions || !d.reduction_rank || d.logical_len / d.groups != d.reductions || d.logical_len % d.groups)) ||
        (d.kind == 1 && (static_cast<uint64_t>(d.destination_offset) > d.base_len || d.logical_len > d.base_len - d.destination_offset)) ||
        (d.kind == 5 && d.groups > d.logical_len)) return rt(cudaErrorInvalidValue);
    TRY_BACKWARD(activate(ctx));
    auto p = std::unique_ptr<BackwardPlan>(new (std::nothrow) BackwardPlan(ctx, d));
    if (!p) return 1002;
    uint64_t counts[5] = {};
    if (d.kind == 2) counts[0] = counts[1] = counts[2] = d.rank;
    if (d.kind == 3) { counts[0] = counts[1] = counts[2] = d.rank; counts[3] = counts[4] = d.reduction_rank; }
    if (d.kind == 4) counts[0] = counts[1] = d.groups;
    if (d.kind == 5) { counts[0] = d.groups; counts[1] = d.groups + 1; counts[2] = d.logical_len; }
    size_t bytes = 0;
    for (auto count : counts) {
        if (count > (std::numeric_limits<size_t>::max() - bytes) / sizeof(uint64_t)) return rt(cudaErrorInvalidValue);
        bytes += static_cast<size_t>(count) * sizeof(uint64_t);
    }
    if (bytes) {
        TRY_BACKWARD(rt(cudaMalloc(&p->metadata, bytes)));
        const uint64_t *arrays[5] = {a,b,c,e,f};
        uint64_t cursor = 0;
        for (unsigned i = 0; i < 5; ++i) {
            if (counts[i]) {
                if (!arrays[i]) return rt(cudaErrorInvalidValue);
                TRY_BACKWARD(rt(cudaMemcpyAsync(p->metadata + cursor, arrays[i], counts[i] * sizeof(uint64_t), cudaMemcpyHostToDevice, ctx->stream)));
            }
            cursor += counts[i];
        }
        TRY_BACKWARD(rt(cudaStreamSynchronize(ctx->stream)));
    }
    *shared = 0;
    if ((d.kind == 3 || d.kind == 5) && d.lanes == 256) {
        cudaFuncAttributes attributes{};
        if (d.kind == 3) TRY_BACKWARD(rt(cudaFuncGetAttributes(&attributes, reduce_block<false>)));
        else TRY_BACKWARD(rt(cudaFuncGetAttributes(&attributes, reduce_block<true>)));
        *shared = attributes.sharedSizeBytes;
    }
    *workspace = bytes;
    *out = p.release();
    return 0;
}
extern "C" int fa_cuda_view_backward_run(const BackwardPlan *p, const float *source, float *destination, int accumulate) {
    if (accumulate != 0 && accumulate != 1) return rt(cudaErrorInvalidValue);
    const auto d = p->dims;
    if (!d.base_len) return 0;
    TRY_BACKWARD(activate(p->ctx));
    if (!accumulate) {
        if (d.written_begin) TRY_BACKWARD(rt(cudaMemsetAsync(destination, 0, d.written_begin * sizeof(float), p->ctx->stream)));
        if (d.written_end < d.base_len) TRY_BACKWARD(rt(cudaMemsetAsync(destination + d.written_end, 0, (d.base_len - d.written_end) * sizeof(float), p->ctx->stream)));
    }
    if (!d.logical_len) return 0;
    if (d.kind == 1 && !accumulate) return rt(cudaMemcpyAsync(destination + d.destination_offset,
        source + d.source_offset, d.logical_len * sizeof(float), cudaMemcpyDeviceToDevice, p->ctx->stream));
    if (d.kind == 1 || d.kind == 2) injective<<<blocks(d.logical_len,256),256,0,p->ctx->stream>>>(d,p->metadata,source,destination,accumulate);
    else if (d.kind == 4) indexed<<<blocks(d.groups,256),256,0,p->ctx->stream>>>(d,p->metadata,source,destination,accumulate);
    else if (d.kind == 3 && d.lanes == 32) reduce_warp<false><<<blocks(d.groups,8),256,0,p->ctx->stream>>>(d,p->metadata,source,destination,accumulate);
    else if (d.kind == 3) reduce_block<false><<<blocks(d.groups,1),256,0,p->ctx->stream>>>(d,p->metadata,source,destination,accumulate);
    else if (d.kind == 5 && d.lanes == 32) reduce_warp<true><<<blocks(d.groups,8),256,0,p->ctx->stream>>>(d,p->metadata,source,destination,accumulate);
    else if (d.kind == 5) reduce_block<true><<<blocks(d.groups,1),256,0,p->ctx->stream>>>(d,p->metadata,source,destination,accumulate);
    return rt(cudaGetLastError());
}
extern "C" void fa_cuda_view_backward_destroy(BackwardPlan *p) { delete p; }
