#include "cuda_backend.h"
#include "structural.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <new>

namespace {

struct PadPlanCuda {
    Context *ctx;
    FaPadDims dims;
    FaPadAxis *axes = nullptr;
    PadPlanCuda(Context *c, FaPadDims d) : ctx(c), dims(d) {}
    ~PadPlanCuda() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (axes) cudaFree(axes);
    }
};

struct CatPlanCuda {
    Context *ctx;
    FaCatDims dims;
    FaCatAxis *axes = nullptr;
    CatPlanCuda(Context *c, FaCatDims d) : ctx(c), dims(d) {}
    ~CatPlanCuda() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (axes) cudaFree(axes);
    }
};

__global__ void pad_fwd_kernel(FaPadDims d, const FaPadAxis *axes,
                               const float *__restrict__ x, float *__restrict__ y) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < d.total_out; idx += stride) {
        uint64_t rem = idx;
        int64_t io = d.in_offset;
        int64_t oo = d.out_offset;
        bool outside = false;
        for (int a = d.rank - 1; a >= 0; --a) {
            const uint64_t d_out = axes[a].out_dim;
            const uint64_t yc = rem % d_out;
            rem /= d_out;
            oo += static_cast<int64_t>(yc) * axes[a].out_stride;
            const uint64_t pb = axes[a].pad_before;
            const uint64_t d_in = axes[a].in_dim;
            uint64_t xc = 0;
            if (yc >= pb && yc < pb + d_in) {
                xc = yc - pb;
            } else if (d.mode == 0) {
                outside = true;
            } else if (d.mode == 2) {
                xc = (yc < pb) ? 0ULL : (d_in - 1ULL);
            } else {
                xc = (yc < pb) ? (pb - yc) : (2ULL * (d_in - 1ULL) - (yc - pb));
            }
            io += static_cast<int64_t>(xc) * axes[a].in_stride;
        }
        y[oo] = outside ? d.const_val : x[io];
    }
}

__global__ void pad_bwd_const_kernel(FaPadDims d, const FaPadAxis *axes,
                                     const float *__restrict__ dy, float *__restrict__ dx,
                                     int accumulate) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < d.total_in; idx += stride) {
        uint64_t rem = idx;
        int64_t uo = d.out_offset;
        int64_t xo = d.in_offset;
        for (int a = d.rank - 1; a >= 0; --a) {
            const uint64_t xc = rem % axes[a].in_dim;
            rem /= axes[a].in_dim;
            xo += static_cast<int64_t>(xc) * axes[a].in_stride;
            uo += static_cast<int64_t>(xc + axes[a].pad_before) * axes[a].out_stride;
        }
        const float v = dy[uo];
        dx[xo] = accumulate ? (dx[xo] + v) : v;
    }
}

// Iterative zero-atomic gather for Reflect and Replicate backward on arbitrary rank <= 16.
__global__ void pad_bwd_gather_kernel(FaPadDims d, const FaPadAxis *axes,
                                      const float *__restrict__ dy, float *__restrict__ dx,
                                      int accumulate) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < d.total_in; idx += stride) {
        uint64_t rem = idx;
        int64_t xo = d.in_offset;
        uint64_t lo[16], hi[16], y_alt[16][3];
        int cnt[16];
        uint64_t combos = 1;
        for (int a = d.rank - 1; a >= 0; --a) {
            const uint64_t d_in = axes[a].in_dim;
            const uint64_t xc = rem % d_in;
            rem /= d_in;
            xo += static_cast<int64_t>(xc) * axes[a].in_stride;
            const uint64_t pb = axes[a].pad_before;
            const uint64_t pa = axes[a].pad_after;
            if (d.mode == 1) {
                int c = 0;
                y_alt[a][c++] = xc + pb;
                if (xc >= 1ULL && xc <= pb) y_alt[a][c++] = pb - xc;
                if (xc + 1ULL < d_in && (d_in - 1ULL - xc) <= pa) {
                    y_alt[a][c++] = pb + 2ULL * (d_in - 1ULL) - xc;
                }
                cnt[a] = c;
                combos *= static_cast<uint64_t>(c);
            } else {
                lo[a] = (xc == 0ULL) ? 0ULL : (xc + pb);
                hi[a] = (xc + 1ULL == d_in) ? axes[a].out_dim : (xc + pb + 1ULL);
                combos *= (hi[a] - lo[a]);
            }
        }
        float sum = 0.0f;
        for (uint64_t k = 0; k < combos; ++k) {
            uint64_t rk = k;
            int64_t uo = d.out_offset;
            for (int a = d.rank - 1; a >= 0; --a) {
                uint64_t yc;
                if (d.mode == 1) {
                    const uint64_t c = static_cast<uint64_t>(cnt[a]);
                    yc = y_alt[a][rk % c];
                    rk /= c;
                } else {
                    const uint64_t span = hi[a] - lo[a];
                    yc = lo[a] + (rk % span);
                    rk /= span;
                }
                uo += static_cast<int64_t>(yc) * axes[a].out_stride;
            }
            sum += dy[uo];
        }
        dx[xo] = accumulate ? (dx[xo] + sum) : sum;
    }
}

__global__ void cat_fwd_kernel(FaCatDims d, const FaCatAxis *axes,
                               const float *__restrict__ a, const float *__restrict__ b,
                               float *__restrict__ out) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < d.total; idx += stride) {
        uint64_t rem = idx;
        int64_t oa = d.offset_a;
        int64_t ob = d.offset_b;
        int64_t oo = d.offset_out;
        bool is_b = false;
        for (int i = d.rank - 1; i >= 0; --i) {
            const uint64_t c = rem % axes[i].dim_out;
            rem /= axes[i].dim_out;
            oo += static_cast<int64_t>(c) * axes[i].stride_out;
            if (i == d.axis) {
                if (c < d.split) {
                    oa += static_cast<int64_t>(c) * axes[i].stride_a;
                } else {
                    is_b = true;
                    ob += static_cast<int64_t>(c - d.split) * axes[i].stride_b;
                }
            } else {
                oa += static_cast<int64_t>(c) * axes[i].stride_a;
                ob += static_cast<int64_t>(c) * axes[i].stride_b;
            }
        }
        out[oo] = is_b ? b[ob] : a[oa];
    }
}

__global__ void cat_bwd_kernel(FaCatDims d, const FaCatAxis *axes,
                               const float *__restrict__ dy, float *__restrict__ dx,
                               int accumulate) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t shift = (d.operand == 1) ? d.split : 0ULL;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < d.total; idx += stride) {
        uint64_t rem = idx;
        int64_t uo = d.offset_out;
        int64_t xo = d.offset_a;
        for (int i = d.rank - 1; i >= 0; --i) {
            const uint64_t c = rem % axes[i].dim_out;
            rem /= axes[i].dim_out;
            xo += static_cast<int64_t>(c) * axes[i].stride_a;
            const uint64_t uc = (i == d.axis) ? (c + shift) : c;
            uo += static_cast<int64_t>(uc) * axes[i].stride_out;
        }
        const float v = dy[uo];
        dx[xo] = accumulate ? (dx[xo] + v) : v;
    }
}

unsigned struct_grid(uint64_t n) {
    const uint64_t blocks = (n + 255ULL) / 256ULL;
    return static_cast<unsigned>(std::min<uint64_t>(blocks, 65535ULL));
}

} // namespace

#define TRY_ST(expr) do { const int code = (expr); if (code) return code; } while (0)

extern "C" int fa_cuda_pad_create(Context *ctx, FaPadDims d, const FaPadAxis *axes,
                                  size_t *workspace, void **out) {
    if (!ctx || !workspace || !out || d.rank <= 0 || d.rank > 16 || d.mode < 0 || d.mode > 2 ||
        !axes) {
        return rt(cudaErrorInvalidValue);
    }
    TRY_ST(activate(ctx));
    auto p = std::unique_ptr<PadPlanCuda>(new (std::nothrow) PadPlanCuda(ctx, d));
    if (!p) return 1002;
    const size_t bytes = static_cast<size_t>(d.rank) * sizeof(FaPadAxis);
    TRY_ST(rt(cudaMalloc(&p->axes, bytes)));
    TRY_ST(rt(cudaMemcpyAsync(p->axes, axes, bytes, cudaMemcpyHostToDevice, ctx->stream)));
    TRY_ST(rt(cudaStreamSynchronize(ctx->stream)));
    *workspace = bytes;
    *out = p.release();
    return 0;
}

extern "C" int fa_cuda_pad_forward_run(const PadPlanCuda *p, const float *x, float *y) {
    if (!p) return rt(cudaErrorInvalidValue);
    if (!p->dims.total_out) return 0;
    if (!x || !y) return rt(cudaErrorInvalidValue);
    TRY_ST(activate(p->ctx));
    pad_fwd_kernel<<<struct_grid(p->dims.total_out), 256, 0, p->ctx->stream>>>(p->dims, p->axes, x, y);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_pad_backward_run(const PadPlanCuda *p, const float *dy, float *dx,
                                        int accumulate) {
    if (!p || (accumulate != 0 && accumulate != 1)) return rt(cudaErrorInvalidValue);
    if (!p->dims.total_in) return 0;
    if (!dy || !dx) return rt(cudaErrorInvalidValue);
    TRY_ST(activate(p->ctx));
    const unsigned g = struct_grid(p->dims.total_in);
    if (p->dims.mode == 0) {
        pad_bwd_const_kernel<<<g, 256, 0, p->ctx->stream>>>(p->dims, p->axes, dy, dx, accumulate);
    } else {
        pad_bwd_gather_kernel<<<g, 256, 0, p->ctx->stream>>>(p->dims, p->axes, dy, dx, accumulate);
    }
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_pad_destroy(PadPlanCuda *p) { delete p; }

extern "C" int fa_cuda_cat_create(Context *ctx, FaCatDims d, const FaCatAxis *axes,
                                  size_t *workspace, void **out) {
    if (!ctx || !workspace || !out || d.rank <= 0 || d.rank > 63 || d.axis < 0 ||
        d.axis >= d.rank || d.operand < -1 || d.operand > 1 || !axes) {
        return rt(cudaErrorInvalidValue);
    }
    TRY_ST(activate(ctx));
    auto p = std::unique_ptr<CatPlanCuda>(new (std::nothrow) CatPlanCuda(ctx, d));
    if (!p) return 1002;
    const size_t bytes = static_cast<size_t>(d.rank) * sizeof(FaCatAxis);
    TRY_ST(rt(cudaMalloc(&p->axes, bytes)));
    TRY_ST(rt(cudaMemcpyAsync(p->axes, axes, bytes, cudaMemcpyHostToDevice, ctx->stream)));
    TRY_ST(rt(cudaStreamSynchronize(ctx->stream)));
    *workspace = bytes;
    *out = p.release();
    return 0;
}

extern "C" int fa_cuda_cat_forward_run(const CatPlanCuda *p, const float *a, const float *b,
                                       float *out) {
    if (!p || p->dims.operand != -1) return rt(cudaErrorInvalidValue);
    if (!p->dims.total) return 0;
    if (!out) return rt(cudaErrorInvalidValue);
    TRY_ST(activate(p->ctx));
    cat_fwd_kernel<<<struct_grid(p->dims.total), 256, 0, p->ctx->stream>>>(p->dims, p->axes, a, b, out);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_cat_backward_run(const CatPlanCuda *p, const float *dy, float *dx,
                                        int accumulate) {
    if (!p || p->dims.operand < 0 || (accumulate != 0 && accumulate != 1)) {
        return rt(cudaErrorInvalidValue);
    }
    if (!p->dims.total) return 0;
    if (!dy || !dx) return rt(cudaErrorInvalidValue);
    TRY_ST(activate(p->ctx));
    cat_bwd_kernel<<<struct_grid(p->dims.total), 256, 0, p->ctx->stream>>>(
        p->dims, p->axes, dy, dx, accumulate);
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_cat_destroy(CatPlanCuda *p) { delete p; }
