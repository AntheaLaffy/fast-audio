#include "cuda_backend.h"
#include "tensor_copy.h"
#include <algorithm>
#include <memory>
#include <new>

namespace {
struct CopyPlan {
    Context *ctx;
    TensorCopyDims dims;
    uint64_t *metadata = nullptr;
    CopyPlan(Context *c, TensorCopyDims d) : ctx(c), dims(d) {}
    ~CopyPlan() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (metadata) cudaFree(metadata);
    }
};
__global__ void copy_strided(TensorCopyDims d, const uint64_t *metadata,
                            const uint32_t *source, uint32_t *destination) {
    const auto sx = reinterpret_cast<const int64_t *>(metadata + d.rank);
    const auto sy = reinterpret_cast<const int64_t *>(metadata + 2 * d.rank);
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < d.len; i += (uint64_t)blockDim.x * gridDim.x) {
        uint64_t remaining = i;
        int64_t x = d.source_offset, y = d.destination_offset;
        for (int32_t axis = d.rank - 1; axis > 0; --axis) {
            const auto coordinate = static_cast<int64_t>(remaining % metadata[axis]);
            remaining /= metadata[axis];
            x += coordinate * sx[axis]; y += coordinate * sy[axis];
        }
        if (d.rank) {
            x += static_cast<int64_t>(remaining) * sx[0];
            y += static_cast<int64_t>(remaining) * sy[0];
        }
        destination[y] = source[x];
    }
}
}
#define TRY_COPY(expr) do { const int code = (expr); if (code) return code; } while (0)
extern "C" int fa_cuda_tensor_copy_create(Context *ctx, TensorCopyDims d, const uint64_t *dims,
    const int64_t *sx, const int64_t *sy, size_t *workspace, void **out) {
    // Coalesced non-singleton axes are bounded by log2(element count), not logical rank.
    if (d.rank < 0 || d.rank > 63 || (d.contiguous != 0 && d.contiguous != 1) || (!d.contiguous && d.len && d.rank == 0))
        return rt(cudaErrorInvalidValue);
    TRY_COPY(activate(ctx));
    auto p = std::unique_ptr<CopyPlan>(new (std::nothrow) CopyPlan(ctx, d));
    if (!p) return 1002;
    *workspace = 0;
    if (d.len && !d.contiguous) {
        const size_t bytes = static_cast<size_t>(d.rank) * sizeof(uint64_t);
        TRY_COPY(rt(cudaMalloc(&p->metadata, 3 * bytes)));
        TRY_COPY(rt(cudaMemcpyAsync(p->metadata, dims, bytes, cudaMemcpyHostToDevice, ctx->stream)));
        TRY_COPY(rt(cudaMemcpyAsync(p->metadata + d.rank, sx, bytes, cudaMemcpyHostToDevice, ctx->stream)));
        TRY_COPY(rt(cudaMemcpyAsync(p->metadata + 2 * d.rank, sy, bytes, cudaMemcpyHostToDevice, ctx->stream)));
        // Host metadata borrows end on return, including any earlier error path.
        TRY_COPY(rt(cudaStreamSynchronize(ctx->stream)));
        *workspace = 3 * bytes;
    }
    *out = p.release();
    return 0;
}
extern "C" int fa_cuda_tensor_copy_run(const CopyPlan *p, const float *source, float *destination) {
    const auto d = p->dims;
    if (!d.len) return 0;
    TRY_COPY(activate(p->ctx));
    if (d.contiguous) return rt(cudaMemcpyAsync(destination + d.destination_offset,
        source + d.source_offset, d.len * sizeof(float), cudaMemcpyDeviceToDevice, p->ctx->stream));
    const unsigned blocks = static_cast<unsigned>(std::min((d.len + 255) / 256, uint64_t{65535}));
    copy_strided<<<blocks,256,0,p->ctx->stream>>>(d, p->metadata,
        reinterpret_cast<const uint32_t *>(source), reinterpret_cast<uint32_t *>(destination));
    return rt(cudaGetLastError());
}
extern "C" void fa_cuda_tensor_copy_destroy(CopyPlan *p) { delete p; }
