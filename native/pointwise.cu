#include "cuda_backend.h"
#include "pointwise.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>

namespace {
struct PointwisePlan {
    Context *ctx;
    PointwiseDims dims;
    PointwiseAxis *axes = nullptr;
    PointwisePlan(Context *c, PointwiseDims d) : ctx(c), dims(d) {}
    ~PointwisePlan() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        if (axes) cudaFree(axes);
    }
};
__device__ __forceinline__ bool is_pow2(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }
__device__ __forceinline__ unsigned log2_pow2(uint64_t v) { return static_cast<unsigned>(__ffsll(static_cast<long long>(v)) - 1); }

__device__ float logistic(float x) {
    if (x >= 0.0f) return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
    const float e = expf(x);
    return __fdiv_rn(e, __fadd_rn(1.0f, e));
}
template<int OP> __device__ float unary(float x) {
    if constexpr (OP == 0) return __uint_as_float(__float_as_uint(x) ^ 0x80000000u);
    else if constexpr (OP == 1) return __uint_as_float(__float_as_uint(x) & 0x7fffffffu);
    else if constexpr (OP == 2) return __fmul_rn(x, x);
    else if constexpr (OP == 3) return __fdiv_rn(1.0f, x);
    else if constexpr (OP == 4) return sqrtf(x);
    else if constexpr (OP == 5) return expf(x);
    else if constexpr (OP == 6) return logf(x);
    else if constexpr (OP == 7) return x < 0.0f ? 0.0f : x;
    else if constexpr (OP == 8) return logistic(x);
    else if constexpr (OP == 9) return tanhf(x);
    else if constexpr (OP == 10) return x == -INFINITY ? -0.0f : __fmul_rn(x, logistic(x));
    else return __fadd_rn(fmaxf(x,0.0f), log1pf(expf(-fabsf(x))));
}
template<int OP> __device__ float binary(float x, float y) {
    if constexpr (OP == 0) return __fadd_rn(x,y);
    else if constexpr (OP == 1) return __fsub_rn(x,y);
    else if constexpr (OP == 2) return __fmul_rn(x,y);
    else return __fdiv_rn(x,y);
}
template<int OP, bool BINARY, bool FLAT>
__global__ void pointwise(PointwiseDims d, const PointwiseAxis *axes,
    const float *left, const float *right, float *__restrict__ out) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if constexpr (FLAT) {
        for (uint64_t i = start; i < d.len; i += stride) {
            const int64_t ii = static_cast<int64_t>(i);
            const int64_t x = d.offsets[0] + ii * d.strides[0];
            const int64_t z = d.offsets[2] + ii * d.strides[2];
            if constexpr (BINARY) {
                const int64_t y = d.offsets[1] + ii * d.strides[1];
                out[z] = binary<OP>(left[x], right[y]);
            } else {
                out[z] = unary<OP>(left[x]);
            }
        }
        return;
    }
    if (d.len <= 0xffffffffULL && d.rank == 1) {
        const uint32_t inner = static_cast<uint32_t>(d.inner);
        const bool p2 = is_pow2(inner);
        const uint32_t mask = inner - 1;
        const unsigned shift = p2 ? log2_pow2(inner) : 0;
        const auto a0 = axes[0];
        for (uint64_t i = start; i < d.len; i += stride) {
            const uint32_t ii = static_cast<uint32_t>(i);
            const int64_t col = static_cast<int64_t>(p2 ? (ii & mask) : (ii % inner));
            const int64_t row = static_cast<int64_t>(p2 ? (ii >> shift) : (ii / inner));
            const int64_t x = d.offsets[0] + col * d.strides[0] + row * a0.strides[0];
            const int64_t z = d.offsets[2] + col * d.strides[2] + row * a0.strides[2];
            if constexpr (BINARY) {
                const int64_t y = d.offsets[1] + col * d.strides[1] + row * a0.strides[1];
                out[z] = binary<OP>(left[x], right[y]);
            } else {
                out[z] = unary<OP>(left[x]);
            }
        }
        return;
    }
    if (d.len <= 0xffffffffULL && d.rank == 2) {
        const uint32_t inner = static_cast<uint32_t>(d.inner);
        const bool ip2 = is_pow2(inner);
        const uint32_t imask = inner - 1;
        const unsigned ishift = ip2 ? log2_pow2(inner) : 0;
        const auto a0 = axes[0];
        const auto a1 = axes[1];
        const uint32_t d1 = static_cast<uint32_t>(a1.dim);
        const bool d1p2 = is_pow2(d1);
        const uint32_t d1mask = d1 - 1;
        const unsigned d1shift = d1p2 ? log2_pow2(d1) : 0;
        for (uint64_t i = start; i < d.len; i += stride) {
            const uint32_t ii = static_cast<uint32_t>(i);
            const int64_t col = static_cast<int64_t>(ip2 ? (ii & imask) : (ii % inner));
            const uint32_t row = ip2 ? (ii >> ishift) : (ii / inner);
            const int64_t c1 = static_cast<int64_t>(d1p2 ? (row & d1mask) : (row % d1));
            const int64_t c0 = static_cast<int64_t>(d1p2 ? (row >> d1shift) : (row / d1));
            const int64_t x = d.offsets[0] + col * d.strides[0] + c1 * a1.strides[0] + c0 * a0.strides[0];
            const int64_t z = d.offsets[2] + col * d.strides[2] + c1 * a1.strides[2] + c0 * a0.strides[2];
            if constexpr (BINARY) {
                const int64_t y = d.offsets[1] + col * d.strides[1] + c1 * a1.strides[1] + c0 * a0.strides[1];
                out[z] = binary<OP>(left[x], right[y]);
            } else {
                out[z] = unary<OP>(left[x]);
            }
        }
        return;
    }
    for (uint64_t i = start; i < d.len; i += stride) {
        int64_t x = d.offsets[0], y = d.offsets[1], z = d.offsets[2];
        const auto column = static_cast<int64_t>(i % d.inner);
        auto row = i / d.inner;
        x += column * d.strides[0];
        if constexpr (BINARY) y += column * d.strides[1];
        z += column * d.strides[2];
        for (int axis = d.rank - 1; axis >= 0; --axis) {
            const auto coordinate = static_cast<int64_t>(row % axes[axis].dim);
            row /= axes[axis].dim;
            x += coordinate * axes[axis].strides[0];
            if constexpr (BINARY) y += coordinate * axes[axis].strides[1];
            z += coordinate * axes[axis].strides[2];
        }
        if constexpr (BINARY) out[z] = binary<OP>(left[x], right[y]);
        else out[z] = unary<OP>(left[x]);
    }
}
template<int OP, bool BINARY> void launch(const PointwisePlan *p, const float *x, const float *y, float *out) {
    const auto grid = static_cast<unsigned>(std::min<uint64_t>((p->dims.len + 255) / 256, 65535));
    if (!p->dims.rank) pointwise<OP,BINARY,true><<<grid,256,0,p->ctx->stream>>>(p->dims,p->axes,x,y,out);
    else pointwise<OP,BINARY,false><<<grid,256,0,p->ctx->stream>>>(p->dims,p->axes,x,y,out);
}
}
#define TRY_POINTWISE(call) do { int status = (call); if (status) return status; } while (0)
extern "C" int fa_cuda_pointwise_create(Context *ctx, PointwiseDims d, const PointwiseAxis *axes,
    size_t *workspace, void **out) {
    if (!ctx || !workspace || !out || d.rank < 0 || d.rank > 63 ||
        (d.len && (!d.inner || d.len % d.inner)) ||
        (d.arity != 1 && d.arity != 2) || d.operation < 0 ||
        d.operation >= (d.arity == 1 ? 12 : 4) || (d.rank && !axes)) return rt(cudaErrorInvalidValue);
    TRY_POINTWISE(activate(ctx));
    auto p = std::unique_ptr<PointwisePlan>(new(std::nothrow) PointwisePlan(ctx,d));
    if (!p) return 1002;
    const size_t bytes = d.rank * sizeof(PointwiseAxis);
    if (bytes) {
        TRY_POINTWISE(rt(cudaMalloc(&p->axes,bytes)));
        TRY_POINTWISE(rt(cudaMemcpyAsync(p->axes,axes,bytes,cudaMemcpyHostToDevice,ctx->stream)));
        TRY_POINTWISE(rt(cudaStreamSynchronize(ctx->stream)));
    }
    *workspace = bytes; *out = p.release();
    return 0;
}
extern "C" int fa_cuda_pointwise_run(const PointwisePlan *p, const float *x, const float *y, float *out) {
    if (!p || (!x && p->dims.len) || (!out && p->dims.len) || (p->dims.arity == 2 && !y && p->dims.len))
        return rt(cudaErrorInvalidValue);
    if (!p->dims.len) return 0;
    TRY_POINTWISE(activate(p->ctx));
    if (p->dims.arity == 1) {
        switch (p->dims.operation) {
            case 0: launch<0,false>(p,x,y,out); break;
            case 1: launch<1,false>(p,x,y,out); break;
            case 2: launch<2,false>(p,x,y,out); break;
            case 3: launch<3,false>(p,x,y,out); break;
            case 4: launch<4,false>(p,x,y,out); break;
            case 5: launch<5,false>(p,x,y,out); break;
            case 6: launch<6,false>(p,x,y,out); break;
            case 7: launch<7,false>(p,x,y,out); break;
            case 8: launch<8,false>(p,x,y,out); break;
            case 9: launch<9,false>(p,x,y,out); break;
            case 10: launch<10,false>(p,x,y,out); break;
            case 11: launch<11,false>(p,x,y,out); break;
        }
    } else {
        switch (p->dims.operation) {
            case 0: launch<0,true>(p,x,y,out); break;
            case 1: launch<1,true>(p,x,y,out); break;
            case 2: launch<2,true>(p,x,y,out); break;
            case 3: launch<3,true>(p,x,y,out); break;
        }
    }
    return rt(cudaGetLastError());
}
extern "C" void fa_cuda_pointwise_destroy(PointwisePlan *p) { delete p; }
