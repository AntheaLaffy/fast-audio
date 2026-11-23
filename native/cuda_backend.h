#pragma once
#include "conv.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static constexpr size_t FA_CUDA_REDUCE_SCRATCH_GROUPS = 32;
static constexpr size_t FA_CUDA_BLAS_WORKSPACE_BYTES = 8 * 1024 * 1024;

// Shared device control block for CUDA-Graph-safe dynamic-LR AdamW (FP32 and BF16 master-weight).
struct AdamwDeviceCtrl {
    uint64_t step;
    float lr;
    float grad_scale;
    // Followed immediately by `double partials[total_partials]` at 16-byte offset.
};
static_assert(sizeof(AdamwDeviceCtrl) == 16, "AdamwDeviceCtrl ABI mismatch");

struct Context {
    int device;
    int cc_major;
    cudaStream_t stream;
    cublasHandle_t blas;       // Permanently CUBLAS_PEDANTIC_MATH (StrictFp32)
    cublasHandle_t blas_tc;    // Tensor Core handle (CUBLAS_DEFAULT_MATH for explicit TF32/BF16)
    int precision_policy;      // 0 = StrictFp32, 1 = Tf32, 2 = Bf16Mixed
    void *blas_workspace;
    float *reduce_scratch;
    // SDPA L2-cached P, dS, and GQA partial buffers (allocated at plan creation, freed at context destroy)
    float *sdpa_p_cache;
    float *sdpa_ds_cache;
    size_t sdpa_cache_floats;
    float *sdpa_gqa_scratch;
    size_t sdpa_gqa_scratch_floats;
    const float *sdpa_last_q;
    const float *sdpa_last_k;
    const float *sdpa_last_v;
    const float *sdpa_last_dy;
    uint64_t sdpa_last_bh;
    uint64_t sdpa_last_tq;
    uint64_t sdpa_last_tk;
    uint64_t sdpa_last_d;
    uint32_t sdpa_last_scale_bits;
    int sdpa_last_causal;
    int sdpa_p_mask;
    int sdpa_ds_mask;
    int sdpa_p_tile64_causal;
};
static inline int rt(cudaError_t s) { return s == cudaSuccess ? 0 : 1000 + (int)s; }
static inline int activate(Context *ctx) { return rt(cudaSetDevice(ctx->device)); }
