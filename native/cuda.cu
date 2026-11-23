#include "cuda_backend.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Native status domains: CUDA=1000+status, cuBLAS=2000+status, cuFFT=3000+status.
static int blas_status(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : 2000 + (int)s; }
static int fft_status(cufftResult s) { return s == CUFFT_SUCCESS ? 0 : 3000 + (int)s; }
#define TRY(expr) do { int status = (expr); if (status) return status; } while (0)

static_assert(sizeof(int) == 4, "Rust CUDA ABI requires 32-bit C int");
struct StftPlan {
    Context *ctx;
    int input_len, n, hop, padding, frames, transforms;
    float *window, *framed;
    cufftHandle fft;
};
static unsigned grid(size_t n) { return (unsigned)std::min((n+255)/256, (size_t)65535); }

extern "C" int fa_cuda_create_with_precision(int device, int policy, void **out) {
    if (policy < 0 || policy > 2 || !out) return rt(cudaErrorInvalidValue);
    TRY(rt(cudaSetDevice(device)));
    Context *ctx = (Context *)std::calloc(1, sizeof(Context));
    if (!ctx) return 1002;
    ctx->device = device;
    ctx->precision_policy = policy;
    int code = rt(cudaDeviceGetAttribute(&ctx->cc_major, cudaDevAttrComputeCapabilityMajor, device));
    if (!code) code = rt(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
    if (!code) code = blas_status(cublasCreate(&ctx->blas));
    if (!code) code = blas_status(cublasSetStream(ctx->blas, ctx->stream));
    if (!code) code = blas_status(cublasSetMathMode(ctx->blas, CUBLAS_PEDANTIC_MATH));
    if (!code) code = blas_status(cublasCreate(&ctx->blas_tc));
    if (!code) code = blas_status(cublasSetStream(ctx->blas_tc, ctx->stream));
    if (!code) code = blas_status(cublasSetMathMode(ctx->blas_tc, CUBLAS_DEFAULT_MATH));
    if (!code) code = rt(cudaMalloc(&ctx->blas_workspace, FA_CUDA_BLAS_WORKSPACE_BYTES));
    if (!code) code = blas_status(cublasSetWorkspace(ctx->blas, ctx->blas_workspace, FA_CUDA_BLAS_WORKSPACE_BYTES));
    if (!code) code = blas_status(cublasSetWorkspace(ctx->blas_tc, ctx->blas_workspace, FA_CUDA_BLAS_WORKSPACE_BYTES));
    if (!code) code = rt(cudaMalloc(&ctx->reduce_scratch, FA_CUDA_REDUCE_SCRATCH_GROUPS * 256 * sizeof(float)));
    if (code) {
        if (ctx->blas) cublasDestroy(ctx->blas);
        if (ctx->blas_tc) cublasDestroy(ctx->blas_tc);
        if (ctx->blas_workspace) cudaFree(ctx->blas_workspace);
        if (ctx->reduce_scratch) cudaFree(ctx->reduce_scratch);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        std::free(ctx);
        return code;
    }
    *out = ctx;
    return 0;
}
extern "C" int fa_cuda_create(int device, void **out) {
    return fa_cuda_create_with_precision(device, 0, out);
}
extern "C" void fa_cuda_destroy(Context *ctx) {
    cudaSetDevice(ctx->device);
    cudaStreamSynchronize(ctx->stream);
    if (ctx->blas) cublasDestroy(ctx->blas);
    if (ctx->blas_tc) cublasDestroy(ctx->blas_tc);
    if (ctx->blas_workspace) cudaFree(ctx->blas_workspace);
    if (ctx->reduce_scratch) cudaFree(ctx->reduce_scratch);
    if (ctx->sdpa_p_cache) cudaFree(ctx->sdpa_p_cache);
    if (ctx->sdpa_ds_cache) cudaFree(ctx->sdpa_ds_cache);
    if (ctx->sdpa_gqa_scratch) cudaFree(ctx->sdpa_gqa_scratch);
    cudaStreamDestroy(ctx->stream);
    std::free(ctx);
}
extern "C" int fa_cuda_name(Context *ctx, char *name) {
    cudaDeviceProp props;
    TRY(rt(cudaGetDeviceProperties(&props, ctx->device)));
    std::memcpy(name, props.name, 256);
    name[255] = '\0';
    return 0;
}
extern "C" int fa_cuda_sync(Context *ctx) {
    TRY(activate(ctx));
    return rt(cudaStreamSynchronize(ctx->stream));
}
extern "C" int fa_cuda_alloc(Context *ctx, size_t bytes, void **out) {
    TRY(activate(ctx));
    TRY(rt(cudaMalloc(out, bytes)));
    int code = rt(cudaMemsetAsync(*out, 0, bytes, ctx->stream));
    if (code) { cudaFree(*out); *out = nullptr; }
    return code;
}
extern "C" void fa_cuda_free(Context *ctx, void *ptr) {
    cudaSetDevice(ctx->device);
    cudaFree(ptr);
}
extern "C" int fa_cuda_copy(Context *ctx, void *dst, const void *src, size_t bytes, int to_device) {
    TRY(activate(ctx));
    int code = rt(cudaMemcpyAsync(dst, src, bytes,
        to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, ctx->stream));
    // Host slice lifetimes end at this function's return, including on errors.
    int sync_code = rt(cudaStreamSynchronize(ctx->stream));
    return code ? code : sync_code;
}
extern "C" int fa_cuda_matmul(Context *ctx, int m, int n, int k,
                                const float *a, const float *b, float *c) {
    TRY(activate(ctx));
    const float one = 1.0f, zero = 0.0f;
    // Row-major C=A*B is column-major C^T=B^T*A^T; no physical transpose.
    return blas_status(cublasSgemm(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, &one, b, n, a, k, &zero, c, n));
}

extern "C" int fa_cuda_zero(Context *ctx, void *destination, size_t bytes) {
    TRY(activate(ctx));
    return bytes ? rt(cudaMemsetAsync(destination, 0, bytes, ctx->stream)) : 0;
}

__global__ static void im2col(const float *x, float *columns, ConvDims s, size_t count) {
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < count; i += blockDim.x*gridDim.x) {
        int t = i % s.output_len;
        size_t row = i / s.output_len;
        int kernel = row % s.kernel;
        int channel = (row / s.kernel) % s.in_channels;
        int batch = row / ((size_t)s.kernel * s.in_channels);
        int64_t pos = (int64_t)t*s.stride + (int64_t)kernel*s.dilation - s.padding;
        columns[i] = pos >= 0 && pos < s.input_len
            ? x[((size_t)batch*s.in_channels+channel)*s.input_len+pos] : 0.0f;
    }
}
__global__ static void epilogue(float *out, const float *bias, int channels, int len,
                               int relu, size_t count) {
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < count; i += blockDim.x*gridDim.x) {
        float v = out[i] + (bias ? bias[(i/len)%channels] : 0.0f);
        if (relu && v < 0.0f) v = 0.0f;
        out[i] = v;
    }
}
extern "C" int fa_cuda_epilogue(Context *ctx, ConvDims s, float *out, const float *bias, int relu) {
    if (!bias && !relu) return 0;
    size_t count = (size_t)s.batch*s.out_channels*s.output_len;
    epilogue<<<grid(count), 256, 0, ctx->stream>>>(out, bias, s.out_channels, s.output_len, relu, count);
    return rt(cudaGetLastError());
}
extern "C" int fa_cuda_conv(Context *ctx, ConvDims s, const float *x, const float *w,
                              const float *bias, float *scratch, float *out, int relu) {
    TRY(activate(ctx));
    const bool pointwise = s.kernel == 1 && s.stride == 1 && s.padding == 0;
    const int k = (s.in_channels/s.groups)*s.kernel;
    const int cout = s.out_channels/s.groups, n = s.output_len;
    const float one = 1.0f, zero = 0.0f;
    const float *columns = x;
    if (!pointwise) {
        size_t count = (size_t)s.batch*s.in_channels*s.kernel*n;
        im2col<<<grid(count), 256, 0, ctx->stream>>>(x, scratch, s, count);
        TRY(rt(cudaGetLastError()));
        columns = scratch;
    }
    long long stride_x = pointwise ? (long long)s.in_channels*s.input_len
        : (long long)s.in_channels*s.kernel*n;
    for (int group = 0; group < s.groups; ++group) {
        TRY(blas_status(cublasSgemmStridedBatched(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N,
            n, cout, k, &one, columns+(size_t)group*k*n, n, stride_x,
            w+(size_t)group*cout*k, k, 0, &zero,
            out+(size_t)group*cout*n, n, (long long)s.out_channels*n, s.batch)));
    }
    if (bias || relu) {
        size_t count = (size_t)s.batch*s.out_channels*n;
        epilogue<<<grid(count), 256, 0, ctx->stream>>>(out, bias, s.out_channels, n, relu, count);
        TRY(rt(cudaGetLastError()));
    }
    return 0;
}

template<bool centered>
__global__ static void frame_window(const float *x, const float *window, float *out,
                                    int input_len, int n, int hop, int padding, int frames, size_t count) {
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < count; i += blockDim.x*gridDim.x) {
        int t = i%n;
        size_t frame = i/n;
        int pos = (int)(frame%frames)*hop + t;
        if constexpr (centered) {
            pos -= padding;
            if (pos < 0) pos = -pos;
            else if (pos >= input_len) pos = (input_len-1)-(pos-(input_len-1));
        }
        out[i] = x[(frame/frames)*input_len+pos]*window[t];
    }
}
extern "C" int fa_cuda_stft_create(Context *ctx, int input_len, int n, int hop,
                                     int padding, int frames, int transforms, const float *window, void **out) {
    TRY(activate(ctx));
    StftPlan *p = (StftPlan *)std::calloc(1, sizeof(StftPlan));
    if (!p) return 1002;
    p->ctx = ctx; p->input_len = input_len; p->n = n; p->hop = hop;
    p->padding = padding; p->frames = frames; p->transforms = transforms;
    int code = rt(cudaMalloc(&p->window, (size_t)n*sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->framed, (size_t)n*transforms*sizeof(float)));
    if (!code) code = rt(cudaMemcpy(p->window, window, (size_t)n*sizeof(float), cudaMemcpyHostToDevice));
    if (!code) code = fft_status(cufftPlanMany(&p->fft, 1, &n, nullptr, 1, n,
        nullptr, 1, n/2+1, CUFFT_R2C, transforms));
    if (!code) code = fft_status(cufftSetStream(p->fft, ctx->stream));
    if (code) {
        if (p->fft) cufftDestroy(p->fft);
        if (p->window) cudaFree(p->window);
        if (p->framed) cudaFree(p->framed);
        std::free(p);
        return code;
    }
    *out = p;
    return 0;
}
extern "C" int fa_cuda_stft_run(StftPlan *p, const float *x, float *out) {
    TRY(activate(p->ctx));
    size_t count = (size_t)p->n*p->transforms;
    if (p->padding) {
        frame_window<true><<<grid(count), 256, 0, p->ctx->stream>>>(x, p->window, p->framed,
            p->input_len, p->n, p->hop, p->padding, p->frames, count);
    } else {
        frame_window<false><<<grid(count), 256, 0, p->ctx->stream>>>(x, p->window, p->framed,
            p->input_len, p->n, p->hop, 0, p->frames, count);
    }
    TRY(rt(cudaGetLastError()));
    return fft_status(cufftExecR2C(p->fft, p->framed, (cufftComplex *)out));
}
extern "C" void fa_cuda_stft_destroy(StftPlan *p) {
    cudaSetDevice(p->ctx->device);
    cudaStreamSynchronize(p->ctx->stream);
    cufftDestroy(p->fft);
    cudaFree(p->window); cudaFree(p->framed);
    std::free(p);
}

struct IstftPlan {
    Context *ctx;
    int output_len, n, hop, padding, frames, transforms;
    float *window, *normalization, *framed;
    cufftComplex *spectrum;
    cufftHandle fft;
};

// cuFFT C2R may overwrite its input. Preserve the borrowed spectrum and enforce
// real DC/Nyquist endpoints even when a model has produced complex values there.
__global__ static void copy_spectrum(const float *input, cufftComplex *out,
                                    int n, size_t count) {
    int bins = n/2+1;
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < count; i += blockDim.x*gridDim.x) {
        int bin = i%bins;
        out[i].x = input[2*i];
        out[i].y = bin == 0 || (n%2 == 0 && bin == n/2) ? 0.0f : input[2*i+1];
    }
}

// One thread owns each output sample: deterministic accumulation without atomics.
__global__ static void overlap_add(const float *framed, const float *window,
                                  const float *normalization, float *out, int output_len,
                                  int n, int hop, int padding, int frames, size_t count) {
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < count; i += blockDim.x*gridDim.x) {
        int sample = i%output_len;
        int position = sample+padding;
        int first = position >= n ? (position-n)/hop+1 : 0;
        int last = min(position/hop, frames-1);
        size_t batch = i/output_len;
        float value = 0.0f;
        for (int frame = first; frame <= last; ++frame) {
            int t = position-frame*hop;
            value += framed[(batch*frames+frame)*n+t]*window[t];
        }
        out[i] = value*normalization[sample];
    }
}

extern "C" int fa_cuda_istft_create(Context *ctx, int output_len, int n, int hop,
                                      int padding, int frames, int transforms,
                                      const float *window, const float *normalization,
                                      size_t *workspace, void **out) {
    TRY(activate(ctx));
    IstftPlan *p = (IstftPlan *)std::calloc(1, sizeof(IstftPlan));
    if (!p) return 1002;
    p->ctx = ctx; p->output_len = output_len; p->n = n; p->hop = hop;
    p->padding = padding; p->frames = frames; p->transforms = transforms;
    int bins = n/2+1;
    int code = rt(cudaMalloc(&p->window, (size_t)n*sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->normalization, (size_t)output_len*sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->framed, (size_t)n*transforms*sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->spectrum, (size_t)bins*transforms*sizeof(cufftComplex)));
    if (!code) code = rt(cudaMemcpy(p->window, window, (size_t)n*sizeof(float), cudaMemcpyHostToDevice));
    if (!code) code = rt(cudaMemcpy(p->normalization, normalization, (size_t)output_len*sizeof(float), cudaMemcpyHostToDevice));
    if (!code) code = fft_status(cufftPlanMany(&p->fft, 1, &n, &bins, 1, bins,
        &n, 1, n, CUFFT_C2R, transforms));
    if (!code) code = fft_status(cufftSetStream(p->fft, ctx->stream));
    size_t fft_bytes = 0;
    if (!code) code = fft_status(cufftGetSize(p->fft, &fft_bytes));
    if (code) {
        if (p->fft) cufftDestroy(p->fft);
        if (p->window) cudaFree(p->window);
        if (p->normalization) cudaFree(p->normalization);
        if (p->framed) cudaFree(p->framed);
        if (p->spectrum) cudaFree(p->spectrum);
        std::free(p);
        return code;
    }
    *workspace = ((size_t)n+output_len+(size_t)n*transforms)*sizeof(float)
        + (size_t)bins*transforms*sizeof(cufftComplex) + fft_bytes;
    *out = p;
    return 0;
}

extern "C" int fa_cuda_istft_run(IstftPlan *p, const float *input, float *out) {
    TRY(activate(p->ctx));
    size_t spectra = (size_t)(p->n/2+1)*p->transforms;
    copy_spectrum<<<grid(spectra), 256, 0, p->ctx->stream>>>(input, p->spectrum, p->n, spectra);
    TRY(rt(cudaGetLastError()));
    TRY(fft_status(cufftExecC2R(p->fft, p->spectrum, p->framed)));
    size_t count = (size_t)(p->transforms/p->frames)*p->output_len;
    overlap_add<<<grid(count), 256, 0, p->ctx->stream>>>(p->framed, p->window, p->normalization,
        out, p->output_len, p->n, p->hop, p->padding, p->frames, count);
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_istft_destroy(IstftPlan *p) {
    cudaSetDevice(p->ctx->device);
    cudaStreamSynchronize(p->ctx->stream);
    cufftDestroy(p->fft);
    cudaFree(p->window); cudaFree(p->normalization); cudaFree(p->framed); cudaFree(p->spectrum);
    std::free(p);
}
extern "C" int fa_cuda_event_create(Context *ctx, void **out) {
    TRY(activate(ctx));
    return rt(cudaEventCreate((cudaEvent_t *)out));
}
extern "C" int fa_cuda_event_record(Context *ctx, void *event) {
    TRY(activate(ctx));
    return rt(cudaEventRecord((cudaEvent_t)event, ctx->stream));
}
extern "C" int fa_cuda_event_elapsed(Context *ctx, void *start, void *end, float *ms) {
    TRY(activate(ctx));
    TRY(rt(cudaEventSynchronize((cudaEvent_t)end)));
    return rt(cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)end));
}
extern "C" void fa_cuda_event_destroy(Context *ctx, void *event) {
    cudaSetDevice(ctx->device);
    cudaEventDestroy((cudaEvent_t)event);
}

__global__ static void cast_f32_to_bf16_kernel(uint64_t n, const float *__restrict__ src, __nv_bfloat16 *__restrict__ dst) {
    uint64_t n4 = n >> 2;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    if (((uintptr_t)src & 15ULL) == 0ULL && ((uintptr_t)dst & 7ULL) == 0ULL) {
        const float4 *s4 = reinterpret_cast<const float4 *>(src);
        uint2 *d2 = reinterpret_cast<uint2 *>(dst);
        for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
            float4 v = s4[i];
            __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
            __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
            uint2 packed;
            packed.x = *reinterpret_cast<const uint32_t *>(&lo);
            packed.y = *reinterpret_cast<const uint32_t *>(&hi);
            d2[i] = packed;
        }
        uint64_t tail = n4 << 2;
        for (uint64_t i = tail + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
            dst[i] = __float2bfloat16_rn(src[i]);
        }
        return;
    }
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = __float2bfloat16_rn(src[i]);
    }
}

__global__ static void cast_bf16_to_f32_kernel(uint64_t n, const __nv_bfloat16 *__restrict__ src, float *__restrict__ dst) {
    uint64_t n4 = n >> 2;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    if (((uintptr_t)src & 7ULL) == 0ULL && ((uintptr_t)dst & 15ULL) == 0ULL) {
        const uint2 *s2 = reinterpret_cast<const uint2 *>(src);
        float4 *d4 = reinterpret_cast<float4 *>(dst);
        for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
            uint2 p = s2[i];
            float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.x));
            float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p.y));
            d4[i] = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        uint64_t tail = n4 << 2;
        for (uint64_t i = tail + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
            dst[i] = __bfloat162float(src[i]);
        }
        return;
    }
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = __bfloat162float(src[i]);
    }
}

__global__ static void cast_f32_to_f16_kernel(uint64_t n, const float *__restrict__ src, __half *__restrict__ dst) {
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = __float2half_rn(src[i]);
    }
}

__global__ static void cast_f16_to_f32_kernel(uint64_t n, const __half *__restrict__ src, float *__restrict__ dst) {
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = __half2float(src[i]);
    }
}

extern "C" int fa_cuda_cast_f32_to_bf16(Context *ctx, uint64_t n, const float *src, uint16_t *dst) {
    if (!ctx) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!src || !dst) return rt(cudaErrorInvalidValue);
    TRY(activate(ctx));
    uint64_t work = (n + 3ULL) >> 2;
    unsigned g = (unsigned)std::min<uint64_t>((work + 255ULL) / 256ULL, 65535ULL);
    cast_f32_to_bf16_kernel<<<g, 256, 0, ctx->stream>>>(n, src, reinterpret_cast<__nv_bfloat16 *>(dst));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_cast_bf16_to_f32(Context *ctx, uint64_t n, const uint16_t *src, float *dst) {
    if (!ctx) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!src || !dst) return rt(cudaErrorInvalidValue);
    TRY(activate(ctx));
    uint64_t work = (n + 3ULL) >> 2;
    unsigned g = (unsigned)std::min<uint64_t>((work + 255ULL) / 256ULL, 65535ULL);
    cast_bf16_to_f32_kernel<<<g, 256, 0, ctx->stream>>>(n, reinterpret_cast<const __nv_bfloat16 *>(src), dst);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_cast_f32_to_f16(Context *ctx, uint64_t n, const float *src, uint16_t *dst) {
    if (!ctx) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!src || !dst) return rt(cudaErrorInvalidValue);
    TRY(activate(ctx));
    unsigned g = (unsigned)std::min<uint64_t>((n + 255ULL) / 256ULL, 65535ULL);
    cast_f32_to_f16_kernel<<<g, 256, 0, ctx->stream>>>(n, src, reinterpret_cast<__half *>(dst));
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_cast_f16_to_f32(Context *ctx, uint64_t n, const uint16_t *src, float *dst) {
    if (!ctx) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!src || !dst) return rt(cudaErrorInvalidValue);
    TRY(activate(ctx));
    unsigned g = (unsigned)std::min<uint64_t>((n + 255ULL) / 256ULL, 65535ULL);
    cast_f16_to_f32_kernel<<<g, 256, 0, ctx->stream>>>(n, reinterpret_cast<const __half *>(src), dst);
    return rt(cudaGetLastError());
}
