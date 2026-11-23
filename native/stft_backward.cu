#include "cuda_backend.h"
#include "stft_backward.h"
#include <cufft.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

// Native status domains: CUDA=1000+status, cuFFT=3000+status.
static int fft_status(cufftResult s) { return s == CUFFT_SUCCESS ? 0 : 3000 + (int)s; }
#define TRY(expr) do { int status = (expr); if (status) return status; } while (0)

static unsigned stft_grid(size_t n) {
    return (unsigned)std::min((n + 255) / 256, (size_t)65535);
}

// ============================================================================
// 1. Strided STFT / Centered STFT Forward
// ============================================================================

struct StftForwardPlan {
    Context *ctx;
    int batch, input_len, n, hop, padding, frames, transforms;
    FaStftView2 x_view;
    FaStftView4 y_view;
    float *window;
    float *framed;
    cufftComplex *spectrum; // null when y_view.direct_r2c != 0
    cufftHandle fft;
};

template <bool centered, bool x_contig>
__global__ static void stft_fwd_frame(const float *__restrict__ x,
                                      const float *__restrict__ window,
                                      float *__restrict__ framed,
                                      FaStftView2 xv,
                                      int input_len, int n, int hop, int padding,
                                      int frames, size_t count) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int t = (int)(i % (size_t)n);
        size_t ft = i / (size_t)n;
        int f = (int)(ft % (size_t)frames);
        size_t b = ft / (size_t)frames;
        int pos = f * hop + t;
        if constexpr (centered) {
            pos -= padding;
            if (pos < 0) pos = -pos;
            else if (pos >= input_len) pos = (input_len - 1) - (pos - (input_len - 1));
        }
        float val;
        if constexpr (x_contig) {
            val = x[xv.origin + (int64_t)b * input_len + pos];
        } else {
            val = x[xv.origin + (int64_t)b * xv.strides[0] + (int64_t)pos * xv.strides[1]];
        }
        framed[i] = val * window[t];
    }
}

__global__ static void stft_fwd_scatter_spectrum(const cufftComplex *__restrict__ spec,
                                                 float *__restrict__ out,
                                                 FaStftView4 yv,
                                                 int frames, int bins, size_t count) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int k = (int)(i % (size_t)bins);
        size_t bf = i / (size_t)bins;
        int f = (int)(bf % (size_t)frames);
        size_t b = bf / (size_t)frames;
        int64_t base = yv.origin
            + (int64_t)b * yv.strides[0]
            + (int64_t)f * yv.strides[1]
            + (int64_t)k * yv.strides[2];
        cufftComplex z = spec[i];
        out[base] = z.x;
        out[base + yv.strides[3]] = z.y;
    }
}

extern "C" int fa_cuda_stft_plan_create(Context *ctx,
                                        int batch, int input_len, int n, int hop,
                                        int padding, int frames,
                                        FaStftView2 x_view, FaStftView4 y_view,
                                        const float *window,
                                        size_t *workspace, void **out) {
    TRY(activate(ctx));
    auto *p = (StftForwardPlan *)std::calloc(1, sizeof(StftForwardPlan));
    if (!p) return 1002;
    p->ctx = ctx;
    p->batch = batch;
    p->input_len = input_len;
    p->n = n;
    p->hop = hop;
    p->padding = padding;
    p->frames = frames;
    p->transforms = batch * frames;
    p->x_view = x_view;
    p->y_view = y_view;

    int bins = n / 2 + 1;
    int code = rt(cudaMalloc(&p->window, (size_t)n * sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->framed, (size_t)n * p->transforms * sizeof(float)));
    if (!code && !y_view.direct_r2c) {
        code = rt(cudaMalloc(&p->spectrum, (size_t)bins * p->transforms * sizeof(cufftComplex)));
    }
    if (!code) {
        code = rt(cudaMemcpy(p->window, window, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!code) {
        code = fft_status(cufftPlanMany(
            &p->fft, 1, &n, nullptr, 1, n, nullptr, 1, bins, CUFFT_R2C, p->transforms));
    }
    if (!code) code = fft_status(cufftSetStream(p->fft, ctx->stream));
    size_t fft_bytes = 0;
    if (!code) code = fft_status(cufftGetSize(p->fft, &fft_bytes));
    if (code) {
        if (p->fft) cufftDestroy(p->fft);
        if (p->window) cudaFree(p->window);
        if (p->framed) cudaFree(p->framed);
        if (p->spectrum) cudaFree(p->spectrum);
        std::free(p);
        return code;
    }
    size_t spec_bytes = p->spectrum ? (size_t)bins * p->transforms * sizeof(cufftComplex) : 0;
    *workspace = ((size_t)n + (size_t)n * p->transforms) * sizeof(float) + spec_bytes + fft_bytes;
    *out = p;
    return 0;
}

extern "C" int fa_cuda_stft_plan_run(StftForwardPlan *p, const float *x, float *out) {
    TRY(activate(p->ctx));
    size_t count = (size_t)p->n * p->transforms;
    unsigned g = stft_grid(count);
    if (p->padding) {
        if (p->x_view.contiguous) {
            stft_fwd_frame<true, true><<<g, 256, 0, p->ctx->stream>>>(
                x, p->window, p->framed, p->x_view, p->input_len, p->n, p->hop, p->padding,
                p->frames, count);
        } else {
            stft_fwd_frame<true, false><<<g, 256, 0, p->ctx->stream>>>(
                x, p->window, p->framed, p->x_view, p->input_len, p->n, p->hop, p->padding,
                p->frames, count);
        }
    } else {
        if (p->x_view.contiguous) {
            stft_fwd_frame<false, true><<<g, 256, 0, p->ctx->stream>>>(
                x, p->window, p->framed, p->x_view, p->input_len, p->n, p->hop, 0,
                p->frames, count);
        } else {
            stft_fwd_frame<false, false><<<g, 256, 0, p->ctx->stream>>>(
                x, p->window, p->framed, p->x_view, p->input_len, p->n, p->hop, 0,
                p->frames, count);
        }
    }
    TRY(rt(cudaGetLastError()));
    if (p->y_view.direct_r2c) {
        return fft_status(
            cufftExecR2C(p->fft, p->framed, (cufftComplex *)(out + p->y_view.origin)));
    }
    TRY(fft_status(cufftExecR2C(p->fft, p->framed, p->spectrum)));
    int bins = p->n / 2 + 1;
    size_t bins_total = (size_t)bins * p->transforms;
    stft_fwd_scatter_spectrum<<<stft_grid(bins_total), 256, 0, p->ctx->stream>>>(
        p->spectrum, out, p->y_view, p->frames, bins, bins_total);
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_stft_plan_destroy(StftForwardPlan *p) {
    cudaSetDevice(p->ctx->device);
    cudaStreamSynchronize(p->ctx->stream);
    cufftDestroy(p->fft);
    cudaFree(p->window);
    cudaFree(p->framed);
    if (p->spectrum) cudaFree(p->spectrum);
    std::free(p);
}

// ============================================================================
// 2. Strided STFT / Centered STFT Backward (Zero Atomics, Deterministic)
// ============================================================================

struct StftBackwardPlan {
    Context *ctx;
    int batch, input_len, n, hop, padding, frames, transforms;
    FaStftView4 dy_view;
    FaStftView2 dx_view;
    float *window;
    cufftComplex *spectrum;
    float *framed;
    cufftHandle fft;
};

template <bool dy_contig>
__global__ static void stft_bwd_prep_spectrum(const float *__restrict__ dy,
                                              cufftComplex *__restrict__ spec,
                                              FaStftView4 dyv,
                                              int frames, int n, size_t count) {
    int bins = n / 2 + 1;
    bool even_n = (n % 2) == 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int k = (int)(i % (size_t)bins);
        float re, im;
        if constexpr (dy_contig) {
            size_t base = (size_t)dyv.origin + 2 * i;
            re = dy[base];
            im = dy[base + 1];
        } else {
            size_t bf = i / (size_t)bins;
            int f = (int)(bf % (size_t)frames);
            size_t b = bf / (size_t)frames;
            int64_t base = dyv.origin
                + (int64_t)b * dyv.strides[0]
                + (int64_t)f * dyv.strides[1]
                + (int64_t)k * dyv.strides[2];
            re = dy[base];
            im = dy[base + dyv.strides[3]];
        }
        bool endpoint = (k == 0) || (even_n && k == n / 2);
        spec[i].x = endpoint ? re : (0.5f * re);
        spec[i].y = endpoint ? 0.0f : (0.5f * im);
    }
}

__device__ static __forceinline__ float gather_framed_pos(const float *__restrict__ framed,
                                                          const float *__restrict__ window,
                                                          size_t b, int pos,
                                                          int frames, int n, int hop) {
    int first = pos >= n ? (pos - n) / hop + 1 : 0;
    int last = min(pos / hop, frames - 1);
    float acc = 0.0f;
    for (int f = first; f <= last; ++f) {
        int t = pos - f * hop;
        acc += framed[(b * (size_t)frames + (size_t)f) * (size_t)n + (size_t)t] * window[t];
    }
    return acc;
}

template <bool centered, bool accumulate, bool dx_contig>
__global__ static void stft_bwd_overlap_gather(const float *__restrict__ framed,
                                               const float *__restrict__ window,
                                               float *__restrict__ dx,
                                               FaStftView2 dxv,
                                               int input_len, int n, int hop,
                                               int padding, int frames, size_t count) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int s = (int)(i % (size_t)input_len);
        size_t b = i / (size_t)input_len;
        float acc = gather_framed_pos(framed, window, b, s + padding, frames, n, hop);
        if constexpr (centered) {
            if (s >= 1 && s <= padding) {
                acc += gather_framed_pos(framed, window, b, padding - s, frames, n, hop);
            }
            if (s >= input_len - 1 - padding && s <= input_len - 2) {
                acc += gather_framed_pos(
                    framed, window, b, padding + 2 * (input_len - 1) - s, frames, n, hop);
            }
        }
        int64_t dst;
        if constexpr (dx_contig) {
            dst = dxv.origin + (int64_t)b * input_len + s;
        } else {
            dst = dxv.origin + (int64_t)b * dxv.strides[0] + (int64_t)s * dxv.strides[1];
        }
        if constexpr (accumulate) {
            dx[dst] += acc;
        } else {
            dx[dst] = acc;
        }
    }
}

extern "C" int fa_cuda_stft_bwd_create(Context *ctx,
                                       int batch, int input_len, int n, int hop,
                                       int padding, int frames,
                                       FaStftView4 dy_view, FaStftView2 dx_view,
                                       const float *window,
                                       size_t *workspace, void **out) {
    TRY(activate(ctx));
    auto *p = (StftBackwardPlan *)std::calloc(1, sizeof(StftBackwardPlan));
    if (!p) return 1002;
    p->ctx = ctx;
    p->batch = batch;
    p->input_len = input_len;
    p->n = n;
    p->hop = hop;
    p->padding = padding;
    p->frames = frames;
    p->transforms = batch * frames;
    p->dy_view = dy_view;
    p->dx_view = dx_view;

    int bins = n / 2 + 1;
    int code = rt(cudaMalloc(&p->window, (size_t)n * sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->spectrum, (size_t)bins * p->transforms * sizeof(cufftComplex)));
    if (!code) code = rt(cudaMalloc(&p->framed, (size_t)n * p->transforms * sizeof(float)));
    if (!code) {
        code = rt(cudaMemcpy(p->window, window, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!code) {
        code = fft_status(cufftPlanMany(
            &p->fft, 1, &n, &bins, 1, bins, &n, 1, n, CUFFT_C2R, p->transforms));
    }
    if (!code) code = fft_status(cufftSetStream(p->fft, ctx->stream));
    size_t fft_bytes = 0;
    if (!code) code = fft_status(cufftGetSize(p->fft, &fft_bytes));
    if (code) {
        if (p->fft) cufftDestroy(p->fft);
        if (p->window) cudaFree(p->window);
        if (p->spectrum) cudaFree(p->spectrum);
        if (p->framed) cudaFree(p->framed);
        std::free(p);
        return code;
    }
    *workspace = ((size_t)n + (size_t)n * p->transforms) * sizeof(float)
        + (size_t)bins * p->transforms * sizeof(cufftComplex) + fft_bytes;
    *out = p;
    return 0;
}

extern "C" int fa_cuda_stft_bwd_run(StftBackwardPlan *p,
                                    const float *dy, float *dx, int accumulate) {
    TRY(activate(p->ctx));
    int bins = p->n / 2 + 1;
    size_t spectra = (size_t)bins * p->transforms;
    if (p->dy_view.contiguous) {
        stft_bwd_prep_spectrum<true><<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(
            dy, p->spectrum, p->dy_view, p->frames, p->n, spectra);
    } else {
        stft_bwd_prep_spectrum<false><<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(
            dy, p->spectrum, p->dy_view, p->frames, p->n, spectra);
    }
    TRY(rt(cudaGetLastError()));
    TRY(fft_status(cufftExecC2R(p->fft, p->spectrum, p->framed)));

    size_t samples = (size_t)p->batch * p->input_len;
    unsigned g = stft_grid(samples);
    bool centered = p->padding > 0;
    bool accum = accumulate != 0;
    bool contig = p->dx_view.contiguous != 0;
    if (centered) {
        if (accum) {
            if (contig) {
                stft_bwd_overlap_gather<true, true, true><<<g, 256, 0, p->ctx->stream>>>(
                    p->framed, p->window, dx, p->dx_view, p->input_len, p->n, p->hop,
                    p->padding, p->frames, samples);
            } else {
                stft_bwd_overlap_gather<true, true, false><<<g, 256, 0, p->ctx->stream>>>(
                    p->framed, p->window, dx, p->dx_view, p->input_len, p->n, p->hop,
                    p->padding, p->frames, samples);
            }
        } else {
            if (contig) {
                stft_bwd_overlap_gather<true, false, true><<<g, 256, 0, p->ctx->stream>>>(
                    p->framed, p->window, dx, p->dx_view, p->input_len, p->n, p->hop,
                    p->padding, p->frames, samples);
            } else {
                stft_bwd_overlap_gather<true, false, false><<<g, 256, 0, p->ctx->stream>>>(
                    p->framed, p->window, dx, p->dx_view, p->input_len, p->n, p->hop,
                    p->padding, p->frames, samples);
            }
        }
    } else {
        if (accum) {
            if (contig) {
                stft_bwd_overlap_gather<false, true, true><<<g, 256, 0, p->ctx->stream>>>(
                    p->framed, p->window, dx, p->dx_view, p->input_len, p->n, p->hop,
                    0, p->frames, samples);
            } else {
                stft_bwd_overlap_gather<false, true, false><<<g, 256, 0, p->ctx->stream>>>(
                    p->framed, p->window, dx, p->dx_view, p->input_len, p->n, p->hop,
                    0, p->frames, samples);
            }
        } else {
            if (contig) {
                stft_bwd_overlap_gather<false, false, true><<<g, 256, 0, p->ctx->stream>>>(
                    p->framed, p->window, dx, p->dx_view, p->input_len, p->n, p->hop,
                    0, p->frames, samples);
            } else {
                stft_bwd_overlap_gather<false, false, false><<<g, 256, 0, p->ctx->stream>>>(
                    p->framed, p->window, dx, p->dx_view, p->input_len, p->n, p->hop,
                    0, p->frames, samples);
            }
        }
    }
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_stft_bwd_destroy(StftBackwardPlan *p) {
    cudaSetDevice(p->ctx->device);
    cudaStreamSynchronize(p->ctx->stream);
    cufftDestroy(p->fft);
    cudaFree(p->window);
    cudaFree(p->spectrum);
    cudaFree(p->framed);
    std::free(p);
}

// ============================================================================
// 3. Strided Centered Inverse STFT Forward
// ============================================================================

struct IstftForwardPlan {
    Context *ctx;
    int batch, output_len, n, hop, padding, frames, transforms;
    FaStftView4 s_view;
    FaStftView2 y_view;
    float *window;
    float *normalization;
    float *framed;
    cufftComplex *spectrum;
    cufftHandle fft;
};

template <bool s_contig>
__global__ static void istft_fwd_prep_spectrum(const float *__restrict__ input,
                                               cufftComplex *__restrict__ spec,
                                               FaStftView4 sv,
                                               int frames, int n, size_t count) {
    int bins = n / 2 + 1;
    bool even_n = (n % 2) == 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int k = (int)(i % (size_t)bins);
        float re, im;
        if constexpr (s_contig) {
            size_t base = (size_t)sv.origin + 2 * i;
            re = input[base];
            im = input[base + 1];
        } else {
            size_t bf = i / (size_t)bins;
            int f = (int)(bf % (size_t)frames);
            size_t b = bf / (size_t)frames;
            int64_t base = sv.origin
                + (int64_t)b * sv.strides[0]
                + (int64_t)f * sv.strides[1]
                + (int64_t)k * sv.strides[2];
            re = input[base];
            im = input[base + sv.strides[3]];
        }
        bool endpoint = (k == 0) || (even_n && k == n / 2);
        spec[i].x = re;
        spec[i].y = endpoint ? 0.0f : im;
    }
}

template <bool y_contig>
__global__ static void istft_fwd_overlap_add(const float *__restrict__ framed,
                                             const float *__restrict__ window,
                                             const float *__restrict__ normalization,
                                             float *__restrict__ out,
                                             FaStftView2 yv,
                                             int output_len, int n, int hop,
                                             int padding, int frames, size_t count) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int s = (int)(i % (size_t)output_len);
        size_t b = i / (size_t)output_len;
        float val = gather_framed_pos(framed, window, b, s + padding, frames, n, hop)
            * normalization[s];
        if constexpr (y_contig) {
            out[yv.origin + (int64_t)b * output_len + s] = val;
        } else {
            out[yv.origin + (int64_t)b * yv.strides[0] + (int64_t)s * yv.strides[1]] = val;
        }
    }
}

extern "C" int fa_cuda_istft_plan_create(Context *ctx,
                                         int batch, int output_len, int n, int hop,
                                         int padding, int frames,
                                         FaStftView4 s_view, FaStftView2 y_view,
                                         const float *window, const float *normalization,
                                         size_t *workspace, void **out) {
    TRY(activate(ctx));
    auto *p = (IstftForwardPlan *)std::calloc(1, sizeof(IstftForwardPlan));
    if (!p) return 1002;
    p->ctx = ctx;
    p->batch = batch;
    p->output_len = output_len;
    p->n = n;
    p->hop = hop;
    p->padding = padding;
    p->frames = frames;
    p->transforms = batch * frames;
    p->s_view = s_view;
    p->y_view = y_view;

    int bins = n / 2 + 1;
    int code = rt(cudaMalloc(&p->window, (size_t)n * sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->normalization, (size_t)output_len * sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->framed, (size_t)n * p->transforms * sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->spectrum, (size_t)bins * p->transforms * sizeof(cufftComplex)));
    if (!code) {
        code = rt(cudaMemcpy(p->window, window, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!code) {
        code = rt(cudaMemcpy(
            p->normalization, normalization, (size_t)output_len * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!code) {
        code = fft_status(cufftPlanMany(
            &p->fft, 1, &n, &bins, 1, bins, &n, 1, n, CUFFT_C2R, p->transforms));
    }
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
    *workspace = ((size_t)n + (size_t)output_len + (size_t)n * p->transforms) * sizeof(float)
        + (size_t)bins * p->transforms * sizeof(cufftComplex) + fft_bytes;
    *out = p;
    return 0;
}

extern "C" int fa_cuda_istft_plan_run(IstftForwardPlan *p, const float *input, float *out) {
    TRY(activate(p->ctx));
    int bins = p->n / 2 + 1;
    size_t spectra = (size_t)bins * p->transforms;
    if (p->s_view.contiguous) {
        istft_fwd_prep_spectrum<true><<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(
            input, p->spectrum, p->s_view, p->frames, p->n, spectra);
    } else {
        istft_fwd_prep_spectrum<false><<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(
            input, p->spectrum, p->s_view, p->frames, p->n, spectra);
    }
    TRY(rt(cudaGetLastError()));
    TRY(fft_status(cufftExecC2R(p->fft, p->spectrum, p->framed)));

    size_t samples = (size_t)p->batch * p->output_len;
    if (p->y_view.contiguous) {
        istft_fwd_overlap_add<true><<<stft_grid(samples), 256, 0, p->ctx->stream>>>(
            p->framed, p->window, p->normalization, out, p->y_view,
            p->output_len, p->n, p->hop, p->padding, p->frames, samples);
    } else {
        istft_fwd_overlap_add<false><<<stft_grid(samples), 256, 0, p->ctx->stream>>>(
            p->framed, p->window, p->normalization, out, p->y_view,
            p->output_len, p->n, p->hop, p->padding, p->frames, samples);
    }
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_istft_plan_destroy(IstftForwardPlan *p) {
    cudaSetDevice(p->ctx->device);
    cudaStreamSynchronize(p->ctx->stream);
    cufftDestroy(p->fft);
    cudaFree(p->window);
    cudaFree(p->normalization);
    cudaFree(p->framed);
    cudaFree(p->spectrum);
    std::free(p);
}

// ============================================================================
// 4. Strided Centered Inverse STFT Backward (Zero Atomics, Deterministic)
// ============================================================================

struct IstftBackwardPlan {
    Context *ctx;
    int batch, output_len, n, hop, padding, frames, transforms;
    FaStftView2 dy_view;
    FaStftView4 ds_view;
    float *window;
    float *normalization;
    float *framed;
    cufftComplex *spectrum;
    cufftHandle fft;
};

template <bool dy_contig>
__global__ static void istft_bwd_frame(const float *__restrict__ dy,
                                       const float *__restrict__ window,
                                       const float *__restrict__ normalization,
                                       float *__restrict__ framed,
                                       FaStftView2 dyv,
                                       int output_len, int n, int hop,
                                       int padding, int frames, size_t count) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int t = (int)(i % (size_t)n);
        size_t ft = i / (size_t)n;
        int f = (int)(ft % (size_t)frames);
        size_t b = ft / (size_t)frames;
        int pos = f * hop + t;
        int s = pos - padding;
        if (s >= 0 && s < output_len) {
            float g;
            if constexpr (dy_contig) {
                g = dy[dyv.origin + (int64_t)b * output_len + s];
            } else {
                g = dy[dyv.origin + (int64_t)b * dyv.strides[0] + (int64_t)s * dyv.strides[1]];
            }
            framed[i] = g * normalization[s] * window[t];
        } else {
            framed[i] = 0.0f;
        }
    }
}

__global__ static void istft_bwd_scale_inplace(float *__restrict__ out, int n, size_t count) {
    int bins = n / 2 + 1;
    bool even_n = (n % 2) == 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int k = (int)(i % (size_t)bins);
        bool endpoint = (k == 0) || (even_n && k == n / 2);
        if (endpoint) {
            out[2 * i + 1] = 0.0f;
        } else {
            out[2 * i] *= 2.0f;
            out[2 * i + 1] *= 2.0f;
        }
    }
}

template <bool accumulate, bool ds_contig>
__global__ static void istft_bwd_write_spectrum(const cufftComplex *__restrict__ spec,
                                                float *__restrict__ out,
                                                FaStftView4 dsv,
                                                int frames, int n, size_t count) {
    int bins = n / 2 + 1;
    bool even_n = (n % 2) == 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < count;
         i += (size_t)blockDim.x * gridDim.x) {
        int k = (int)(i % (size_t)bins);
        bool endpoint = (k == 0) || (even_n && k == n / 2);
        cufftComplex z = spec[i];
        float re = endpoint ? z.x : (2.0f * z.x);
        float im = endpoint ? 0.0f : (2.0f * z.y);
        int64_t base_re, base_im;
        if constexpr (ds_contig) {
            base_re = dsv.origin + 2 * (int64_t)i;
            base_im = base_re + 1;
        } else {
            size_t bf = i / (size_t)bins;
            int f = (int)(bf % (size_t)frames);
            size_t b = bf / (size_t)frames;
            base_re = dsv.origin
                + (int64_t)b * dsv.strides[0]
                + (int64_t)f * dsv.strides[1]
                + (int64_t)k * dsv.strides[2];
            base_im = base_re + dsv.strides[3];
        }
        if constexpr (accumulate) {
            out[base_re] += re;
            out[base_im] += im;
        } else {
            out[base_re] = re;
            out[base_im] = im;
        }
    }
}

extern "C" int fa_cuda_istft_bwd_create(Context *ctx,
                                        int batch, int output_len, int n, int hop,
                                        int padding, int frames,
                                        FaStftView2 dy_view, FaStftView4 ds_view,
                                        const float *window, const float *normalization,
                                        size_t *workspace, void **out) {
    TRY(activate(ctx));
    auto *p = (IstftBackwardPlan *)std::calloc(1, sizeof(IstftBackwardPlan));
    if (!p) return 1002;
    p->ctx = ctx;
    p->batch = batch;
    p->output_len = output_len;
    p->n = n;
    p->hop = hop;
    p->padding = padding;
    p->frames = frames;
    p->transforms = batch * frames;
    p->dy_view = dy_view;
    p->ds_view = ds_view;

    int bins = n / 2 + 1;
    int code = rt(cudaMalloc(&p->window, (size_t)n * sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->normalization, (size_t)output_len * sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->framed, (size_t)n * p->transforms * sizeof(float)));
    if (!code) code = rt(cudaMalloc(&p->spectrum, (size_t)bins * p->transforms * sizeof(cufftComplex)));
    if (!code) {
        code = rt(cudaMemcpy(p->window, window, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!code) {
        code = rt(cudaMemcpy(
            p->normalization, normalization, (size_t)output_len * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!code) {
        code = fft_status(cufftPlanMany(
            &p->fft, 1, &n, nullptr, 1, n, nullptr, 1, bins, CUFFT_R2C, p->transforms));
    }
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
    *workspace = ((size_t)n + (size_t)output_len + (size_t)n * p->transforms) * sizeof(float)
        + (size_t)bins * p->transforms * sizeof(cufftComplex) + fft_bytes;
    *out = p;
    return 0;
}

extern "C" int fa_cuda_istft_bwd_run(IstftBackwardPlan *p,
                                     const float *dy, float *ds, int accumulate) {
    TRY(activate(p->ctx));
    size_t count = (size_t)p->n * p->transforms;
    if (p->dy_view.contiguous) {
        istft_bwd_frame<true><<<stft_grid(count), 256, 0, p->ctx->stream>>>(
            dy, p->window, p->normalization, p->framed, p->dy_view,
            p->output_len, p->n, p->hop, p->padding, p->frames, count);
    } else {
        istft_bwd_frame<false><<<stft_grid(count), 256, 0, p->ctx->stream>>>(
            dy, p->window, p->normalization, p->framed, p->dy_view,
            p->output_len, p->n, p->hop, p->padding, p->frames, count);
    }
    TRY(rt(cudaGetLastError()));

    int bins = p->n / 2 + 1;
    size_t spectra = (size_t)bins * p->transforms;
    if (!accumulate && p->ds_view.direct_r2c) {
        float *dst = ds + p->ds_view.origin;
        TRY(fft_status(cufftExecR2C(p->fft, p->framed, (cufftComplex *)dst)));
        istft_bwd_scale_inplace<<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(dst, p->n, spectra);
        return rt(cudaGetLastError());
    }
    TRY(fft_status(cufftExecR2C(p->fft, p->framed, p->spectrum)));
    if (accumulate) {
        if (p->ds_view.contiguous) {
            istft_bwd_write_spectrum<true, true><<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(
                p->spectrum, ds, p->ds_view, p->frames, p->n, spectra);
        } else {
            istft_bwd_write_spectrum<true, false><<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(
                p->spectrum, ds, p->ds_view, p->frames, p->n, spectra);
        }
    } else {
        if (p->ds_view.contiguous) {
            istft_bwd_write_spectrum<false, true><<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(
                p->spectrum, ds, p->ds_view, p->frames, p->n, spectra);
        } else {
            istft_bwd_write_spectrum<false, false><<<stft_grid(spectra), 256, 0, p->ctx->stream>>>(
                p->spectrum, ds, p->ds_view, p->frames, p->n, spectra);
        }
    }
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_istft_bwd_destroy(IstftBackwardPlan *p) {
    cudaSetDevice(p->ctx->device);
    cudaStreamSynchronize(p->ctx->stream);
    cufftDestroy(p->fft);
    cudaFree(p->window);
    cudaFree(p->normalization);
    cudaFree(p->framed);
    cudaFree(p->spectrum);
    std::free(p);
}
