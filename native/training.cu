#include "cuda_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

static_assert(sizeof(float)==4,"FP32 parameter update ABI mismatch");
namespace {
__global__ void sgd(uint64_t n,float *parameter,const float *gradient,float rate) {
    for (uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;
         i+=static_cast<uint64_t>(blockDim.x)*gridDim.x)
        parameter[i]=__fsub_rn(parameter[i],__fmul_rn(rate,gradient[i]));
}
}
extern "C" int fa_cuda_tape_clear(Context *ctx,uint64_t n,float *out) {
    if (!ctx || n>std::numeric_limits<size_t>::max()/sizeof(float) || (n && !out)) return rt(cudaErrorInvalidValue);
    if (!n) return 0;
    int code=activate(ctx);if (code) return code;
    return rt(cudaMemsetAsync(out,0,static_cast<size_t>(n)*sizeof(float),ctx->stream));
}
extern "C" int fa_cuda_tape_sgd(Context *ctx,uint64_t n,float *out,const float *gradient,float rate) {
    if (!ctx || !std::isfinite(rate) || rate<0.0f || n>std::numeric_limits<size_t>::max()/sizeof(float) ||
        (n && (!out || !gradient || out==gradient))) return rt(cudaErrorInvalidValue);
    if (!n) return 0;
    int code=activate(ctx);if (code) return code;
    const unsigned grid=static_cast<unsigned>(std::min<uint64_t>((n+255)/256,65535));
    sgd<<<grid,256,0,ctx->stream>>>(n,out,gradient,rate);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_capture_begin(Context *ctx) {
    if (!ctx) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    return rt(cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeRelaxed));
}

extern "C" int fa_cuda_capture_end(Context *ctx, void **exec_out) {
    if (!ctx || !exec_out) return rt(cudaErrorInvalidValue);
    *exec_out = nullptr;
    int code = activate(ctx);
    if (code) return code;
    cudaGraph_t graph = nullptr;
    code = rt(cudaStreamEndCapture(ctx->stream, &graph));
    if (code) {
        if (graph) cudaGraphDestroy(graph);
        return code;
    }
    cudaGraphExec_t exec = nullptr;
    code = rt(cudaGraphInstantiate(&exec, graph, 0));
    cudaGraphDestroy(graph);
    if (code) {
        if (exec) cudaGraphExecDestroy(exec);
        return code;
    }
    *exec_out = exec;
    return 0;
}

extern "C" void fa_cuda_capture_abort(Context *ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->device);
    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(ctx->stream, &graph) == cudaSuccess && graph) {
        cudaGraphDestroy(graph);
    }
}

extern "C" int fa_cuda_graph_launch(Context *ctx, void *exec) {
    if (!ctx || !exec) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    return rt(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec), ctx->stream));
}

extern "C" void fa_cuda_graph_destroy(Context *ctx, void *exec) {
    if (!ctx || !exec) return;
    cudaSetDevice(ctx->device);
    cudaStreamSynchronize(ctx->stream);
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec));
}

namespace {

__device__ __forceinline__ double* ctrl_partials(AdamwDeviceCtrl *ctrl) {
    return reinterpret_cast<double*>(reinterpret_cast<char*>(ctrl) + sizeof(AdamwDeviceCtrl));
}

__device__ __forceinline__ double warp_reduce_sum_f64(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void adamw_set_lr_kernel(AdamwDeviceCtrl *ctrl, float lr) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ctrl->lr = lr;
    }
}

__global__ void adamw_set_step_kernel(AdamwDeviceCtrl *ctrl, uint64_t step) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ctrl->step = step;
    }
}

__global__ void adamw_bump_step_kernel(AdamwDeviceCtrl *ctrl) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ctrl->step += 1ULL;
    }
}

__global__ void adamw_grad_norm_partial_kernel(
    uint64_t n,
    const float *grad,
    AdamwDeviceCtrl *ctrl,
    uint64_t partials_offset
) {
    __shared__ double smem[8];
    double acc = 0.0;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double g = static_cast<double>(grad[i]);
        acc += g * g;
    }
    acc = warp_reduce_sum_f64(acc);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) {
        smem[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        double total = (lane < 8) ? smem[lane] : 0.0;
        total = warp_reduce_sum_f64(total);
        if (lane == 0) {
            ctrl_partials(ctrl)[partials_offset + blockIdx.x] = total;
        }
    }
}

__global__ void adamw_grad_norm_finalize_kernel(
    AdamwDeviceCtrl *ctrl,
    uint64_t total_partials,
    float max_grad_norm
) {
    __shared__ double smem[8];
    const double *partials = ctrl_partials(ctrl);
    double acc = 0.0;
    for (uint64_t i = threadIdx.x; i < total_partials; i += blockDim.x) {
        acc += partials[i];
    }
    acc = warp_reduce_sum_f64(acc);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) {
        smem[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        double total = (lane < 8) ? smem[lane] : 0.0;
        total = warp_reduce_sum_f64(total);
        if (lane == 0) {
            const double norm = sqrt(total);
            const double clip = static_cast<double>(max_grad_norm) / (norm + 1e-6);
            ctrl->grad_scale = (clip < 1.0) ? static_cast<float>(clip) : 1.0f;
        }
    }
}

__global__ void adamw_param_kernel(
    uint64_t n,
    float *param,
    const float *grad,
    float *m,
    float *v,
    const AdamwDeviceCtrl *ctrl,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int use_grad_scale
) {
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
    const float decay_factor = scalars[1];
    const float step_size = scalars[2];
    const float inv_sqrt_bc2 = scalars[3];
    const float one_minus_b1 = __fsub_rn(1.0f, beta1);
    const float one_minus_b2 = __fsub_rn(1.0f, beta2);

    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float g = __fmul_rn(grad[i], scale);
        const float p_decayed = __fmul_rn(param[i], decay_factor);
        const float mi = __fadd_rn(__fmul_rn(beta1, m[i]), __fmul_rn(one_minus_b1, g));
        const float vi = __fadd_rn(__fmul_rn(beta2, v[i]), __fmul_rn(one_minus_b2, __fmul_rn(g, g)));
        const float denom = __fadd_rn(__fmul_rn(sqrtf(vi), inv_sqrt_bc2), eps);
        const float step = __fmul_rn(step_size, __fdiv_rn(mi, denom));
        m[i] = mi;
        v[i] = vi;
        param[i] = __fsub_rn(p_decayed, step);
    }
}

} // namespace

extern "C" int fa_cuda_adamw_state_create(
    Context *ctx,
    uint64_t total_partials,
    float initial_lr,
    void **out_ctrl
) {
    if (!ctx || !out_ctrl || !std::isfinite(initial_lr) || initial_lr < 0.0f) {
        return rt(cudaErrorInvalidValue);
    }
    *out_ctrl = nullptr;
    int code = activate(ctx);
    if (code) return code;
    const uint64_t slots = std::max<uint64_t>(total_partials, 1);
    if (slots > (std::numeric_limits<size_t>::max() - sizeof(AdamwDeviceCtrl)) / sizeof(double)) {
        return rt(cudaErrorInvalidValue);
    }
    const size_t bytes = sizeof(AdamwDeviceCtrl) + static_cast<size_t>(slots) * sizeof(double);
    void *dev = nullptr;
    code = rt(cudaMalloc(&dev, bytes));
    if (code) return code;
    AdamwDeviceCtrl init_hdr{0ULL, initial_lr, 1.0f};
    code = rt(cudaMemcpyAsync(dev, &init_hdr, sizeof(AdamwDeviceCtrl), cudaMemcpyHostToDevice, ctx->stream));
    if (!code) {
        code = rt(cudaStreamSynchronize(ctx->stream));
    }
    if (code) {
        cudaFree(dev);
        return code;
    }
    *out_ctrl = dev;
    return 0;
}

extern "C" void fa_cuda_adamw_state_destroy(Context *ctx, void *ctrl) {
    if (!ctx || !ctrl) return;
    cudaSetDevice(ctx->device);
    cudaStreamSynchronize(ctx->stream);
    cudaFree(ctrl);
}

extern "C" int fa_cuda_adamw_set_lr(Context *ctx, void *ctrl, float lr) {
    if (!ctx || !ctrl || !std::isfinite(lr) || lr < 0.0f) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    adamw_set_lr_kernel<<<1, 1, 0, ctx->stream>>>(static_cast<AdamwDeviceCtrl*>(ctrl), lr);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_adamw_set_step(Context *ctx, void *ctrl, uint64_t step) {
    if (!ctx || !ctrl) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    adamw_set_step_kernel<<<1, 1, 0, ctx->stream>>>(static_cast<AdamwDeviceCtrl*>(ctrl), step);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_adamw_bump_step(Context *ctx, void *ctrl) {
    if (!ctx || !ctrl) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    adamw_bump_step_kernel<<<1, 1, 0, ctx->stream>>>(static_cast<AdamwDeviceCtrl*>(ctrl));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_adamw_grad_norm_partial(
    Context *ctx,
    uint64_t n,
    const float *grad,
    void *ctrl,
    uint64_t partials_offset,
    uint32_t grid
) {
    if (!ctx || !ctrl || (n && !grad) || grid == 0) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    adamw_grad_norm_partial_kernel<<<grid, 256, 0, ctx->stream>>>(
        n,
        grad,
        static_cast<AdamwDeviceCtrl*>(ctrl),
        partials_offset
    );
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_adamw_grad_norm_finalize(
    Context *ctx,
    void *ctrl,
    uint64_t total_partials,
    float max_grad_norm
) {
    if (!ctx || !ctrl || !std::isfinite(max_grad_norm) || max_grad_norm <= 0.0f) {
        return rt(cudaErrorInvalidValue);
    }
    int code = activate(ctx);
    if (code) return code;
    adamw_grad_norm_finalize_kernel<<<1, 256, 0, ctx->stream>>>(
        static_cast<AdamwDeviceCtrl*>(ctrl),
        total_partials,
        max_grad_norm
    );
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_adamw_param(
    Context *ctx,
    uint64_t n,
    float *param,
    const float *grad,
    float *m,
    float *v,
    void *ctrl,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int use_grad_scale
) {
    if (!ctx || !ctrl || (n && (!param || !grad || !m || !v || param == grad))) {
        return rt(cudaErrorInvalidValue);
    }
    if (!n) return 0;
    int code = activate(ctx);
    if (code) return code;
    const unsigned grid = static_cast<unsigned>(std::min<uint64_t>((n + 255) / 256, 65535));
    adamw_param_kernel<<<grid, 256, 0, ctx->stream>>>(
        n,
        param,
        grad,
        m,
        v,
        static_cast<const AdamwDeviceCtrl*>(ctrl),
        beta1,
        beta2,
        eps,
        weight_decay,
        use_grad_scale
    );
    return rt(cudaGetLastError());
}
