#include "bmm.h"
#include "cuda_backend.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <new>

namespace {

struct BmmPlanCuda {
    Context *ctx;
    FaBmmDims dims;
    FaBmmAxis *out_axes = nullptr;
    FaBmmAxis *red_axes = nullptr;
    BmmPlanCuda(Context *c, FaBmmDims d) : ctx(c), dims(d) {}
    ~BmmPlanCuda() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (out_axes) cudaFree(out_axes);
        if (red_axes) cudaFree(red_axes);
    }
};

__device__ __forceinline__ void decode_axes(
    const FaBmmAxis *axes, int rank, uint64_t idx,
    int64_t &oa, int64_t &ob, int64_t &oc) {
    oa = 0;
    ob = 0;
    oc = 0;
    for (int i = rank - 1; i >= 0; --i) {
        const int64_t c = static_cast<int64_t>(idx % axes[i].dim);
        idx /= axes[i].dim;
        oa += c * axes[i].stride_a;
        ob += c * axes[i].stride_b;
        oc += c * axes[i].stride_c;
    }
}

__global__ void bmm_zero(FaBmmDims d, const FaBmmAxis *out_axes, float *c) {
    const uint64_t mn = d.m * d.n;
    const uint64_t total = d.out_batches * mn;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint64_t g = idx / mn;
        const uint64_t rem = idx % mn;
        const uint64_t row = rem / d.n;
        const uint64_t col = rem % d.n;
        int64_t oa, ob, oc;
        decode_axes(out_axes, d.out_rank, g, oa, ob, oc);
        const int64_t dst = d.origins[2] + oc +
                            static_cast<int64_t>(row) * d.strides[4] +
                            static_cast<int64_t>(col) * d.strides[5];
        c[dst] = 0.0f;
    }
}

__global__ void bmm_tiled(FaBmmDims d,
                          const FaBmmAxis *out_axes,
                          const FaBmmAxis *red_axes,
                          const float *__restrict__ a,
                          const float *__restrict__ b,
                          float *__restrict__ c,
                          int accumulate) {
    __shared__ float aa[16][16];
    __shared__ float bb[16][16];
    const uint64_t cols = (d.n + 15ULL) / 16ULL;
    const uint64_t rows = (d.m + 15ULL) / 16ULL;
    const uint64_t tiles_per_batch = rows * cols;
    const uint64_t total_tiles = d.out_batches * tiles_per_batch;

    for (uint64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const uint64_t g = tile / tiles_per_batch;
        const uint64_t local = tile % tiles_per_batch;
        const uint64_t row = (local / cols) * 16ULL + threadIdx.y;
        const uint64_t col = (local % cols) * 16ULL + threadIdx.x;

        int64_t oa_g, ob_g, oc_g;
        decode_axes(out_axes, d.out_rank, g, oa_g, ob_g, oc_g);

        float sum = 0.0f;
        for (uint64_t r = 0; r < d.red_batches; ++r) {
            int64_t oa_r = 0, ob_r = 0, oc_r = 0;
            if (d.red_rank > 0) {
                decode_axes(red_axes, d.red_rank, r, oa_r, ob_r, oc_r);
            }
            const int64_t base_a = d.origins[0] + oa_g + oa_r;
            const int64_t base_b = d.origins[1] + ob_g + ob_r;
            for (uint64_t start = 0; start < d.k; start += 16ULL) {
                const uint64_t p = start + threadIdx.x;
                const uint64_t q = start + threadIdx.y;
                aa[threadIdx.y][threadIdx.x] = (row < d.m && p < d.k)
                    ? a[base_a + static_cast<int64_t>(row) * d.strides[0] + static_cast<int64_t>(p) * d.strides[1]]
                    : 0.0f;
                bb[threadIdx.y][threadIdx.x] = (q < d.k && col < d.n)
                    ? b[base_b + static_cast<int64_t>(q) * d.strides[2] + static_cast<int64_t>(col) * d.strides[3]]
                    : 0.0f;
                __syncthreads();
                for (int i = 0; i < 16; ++i) {
                    sum = __fadd_rn(sum, __fmul_rn(aa[threadIdx.y][i], bb[i][threadIdx.x]));
                }
                __syncthreads();
            }
        }
        if (row < d.m && col < d.n) {
            const int64_t dst = d.origins[2] + oc_g +
                                static_cast<int64_t>(row) * d.strides[4] +
                                static_cast<int64_t>(col) * d.strides[5];
            c[dst] = accumulate ? __fadd_rn(c[dst], sum) : sum;
        }
    }
}

unsigned bmm_grid(uint64_t work) {
    return static_cast<unsigned>(std::min<uint64_t>(work, 65535ULL));
}

} // namespace

#define TRY_BMM(expr) do { const int code = (expr); if (code) return code; } while (0)

extern "C" int fa_cuda_bmm_create(Context *ctx, FaBmmDims d,
                                  const FaBmmAxis *out_axes,
                                  const FaBmmAxis *red_axes,
                                  size_t *workspace, size_t *shared, void **out) {
    if (!ctx || !workspace || !shared || !out || d.route < 0 || d.route > 4 ||
        d.out_rank < 0 || d.out_rank > 63 || d.red_rank < 0 || d.red_rank > 63 ||
        d.reserved != 0) {
        return rt(cudaErrorInvalidValue);
    }
    TRY_BMM(activate(ctx));
    auto p = std::unique_ptr<BmmPlanCuda>(new (std::nothrow) BmmPlanCuda(ctx, d));
    if (!p) return 1002;
    size_t out_bytes = 0, red_bytes = 0;
    if (d.route == 1 || d.route == 4) {
        out_bytes = static_cast<size_t>(d.out_rank) * sizeof(FaBmmAxis);
        red_bytes = static_cast<size_t>(d.red_rank) * sizeof(FaBmmAxis);
        if (out_bytes) {
            TRY_BMM(rt(cudaMalloc(&p->out_axes, out_bytes)));
            TRY_BMM(rt(cudaMemcpyAsync(
                p->out_axes, out_axes, out_bytes, cudaMemcpyHostToDevice, ctx->stream)));
        }
        if (red_bytes) {
            TRY_BMM(rt(cudaMalloc(&p->red_axes, red_bytes)));
            TRY_BMM(rt(cudaMemcpyAsync(
                p->red_axes, red_axes, red_bytes, cudaMemcpyHostToDevice, ctx->stream)));
        }
        if (out_bytes || red_bytes) {
            TRY_BMM(rt(cudaStreamSynchronize(ctx->stream)));
        }
    }
    *shared = 0;
    if (d.route == 4) {
        cudaFuncAttributes attr{};
        TRY_BMM(rt(cudaFuncGetAttributes(&attr, bmm_tiled)));
        *shared = attr.sharedSizeBytes;
    }
    *workspace = out_bytes + red_bytes;
    *out = p.release();
    return 0;
}

extern "C" int fa_cuda_bmm_run(const BmmPlanCuda *p,
                               const float *a, const float *b, float *c, int accumulate) {
    if (!p || (accumulate != 0 && accumulate != 1)) return rt(cudaErrorInvalidValue);
    const auto d = p->dims;
    if (d.route == 0 || d.out_batches == 0 || d.m == 0 || d.n == 0) return 0;
    if (d.route == 1) {
        if (accumulate) return 0;
        if (!c) return rt(cudaErrorInvalidValue);
        TRY_BMM(activate(p->ctx));
        const uint64_t total = d.out_batches * d.m * d.n;
        bmm_zero<<<bmm_grid((total + 255ULL) / 256ULL), 256, 0, p->ctx->stream>>>(d, p->out_axes, c);
        return rt(cudaGetLastError());
    }
    if (!a || !b || !c) return rt(cudaErrorInvalidValue);
    TRY_BMM(activate(p->ctx));

    if (d.route == 2) {
        const float one = 1.0f, beta = accumulate ? 1.0f : 0.0f;
        const auto ta = d.transposes[0] ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto tb = d.transposes[1] ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto status = cublasSgemm(
            p->ctx->blas, tb, ta, d.blas_n, d.blas_m, d.blas_k,
            &one, b + d.origins[1], d.leading[1],
            a + d.origins[0], d.leading[0],
            &beta, c + d.origins[2], d.leading[2]);
        return status == CUBLAS_STATUS_SUCCESS ? 0 : 2000 + static_cast<int>(status);
    }

    if (d.route == 3) {
        const float one = 1.0f, beta = accumulate ? 1.0f : 0.0f;
        const auto ta = d.transposes[0] ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto tb = d.transposes[1] ? CUBLAS_OP_T : CUBLAS_OP_N;
        const float *pa = d.swapped ? (b + d.origins[1]) : (a + d.origins[0]);
        const float *pb = d.swapped ? (a + d.origins[0]) : (b + d.origins[1]);
        const auto status = cublasSgemmStridedBatched(
            p->ctx->blas, tb, ta, d.blas_n, d.blas_m, d.blas_k,
            &one, pb, d.leading[1], d.batch_strides[1],
            pa, d.leading[0], d.batch_strides[0],
            &beta, c + d.origins[2], d.leading[2], d.batch_strides[2],
            d.batch_count);
        return status == CUBLAS_STATUS_SUCCESS ? 0 : 2000 + static_cast<int>(status);
    }

    const uint64_t tiles = d.out_batches * ((d.m + 15ULL) / 16ULL) * ((d.n + 15ULL) / 16ULL);
    bmm_tiled<<<bmm_grid(tiles), dim3(16, 16), 0, p->ctx->stream>>>(
        d, p->out_axes, p->red_axes, a, b, c, accumulate);
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_bmm_destroy(BmmPlanCuda *p) { delete p; }
