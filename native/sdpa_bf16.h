#pragma once
#include "cuda_backend.h"
#include <cstdint>

extern "C" {

// Strided 4D BF16 descriptor (stride_d == 1 required for fast Tensor Core path).
struct Bf16SdpaStrides {
    int64_t sb;
    int64_t sh;
    int64_t st;
    int64_t sd;
};

// Pre-configures opt-in dynamic shared memory for D=128 kernels outside CUDA Graph capture.
int fa_cuda_sdpa_bf16_prepare(Context *ctx);

// True O(T)-memory BF16 Tensor Core FlashAttention-2 forward (mma.sync.aligned.m16n8k16).
// Supports MHA and GQA, causal and non-causal, [B,H,T,D] and [B,T,H,D] (including QKV views).
// lse_f32 must have length B * H_q * T_q floats (or nullptr if backward is not needed).
int fa_cuda_sdpa_bf16_fwd(
    Context *ctx,
    uint64_t batch,
    uint64_t h_q,
    uint64_t h_kv,
    uint64_t t_q,
    uint64_t t_k,
    uint64_t d,
    float scale,
    int is_causal,
    const uint16_t *q, Bf16SdpaStrides sq,
    const uint16_t *k, Bf16SdpaStrides sk,
    const uint16_t *v, Bf16SdpaStrides sv,
    uint16_t *out, Bf16SdpaStrides so,
    float *lse_f32_opt);

// True O(T)-memory zero-atomic BF16 Tensor Core FlashAttention-2 backward (dQ + fused dK/dV).
// Workspace required:
//   d_row_f32: B * H_q * T_q floats
//   gqa_dk_f32, gqa_dv_f32: B * H_q * T_k * D floats each for Tensor Core GQA only.
int fa_cuda_sdpa_bf16_bwd(
    Context *ctx,
    uint64_t batch,
    uint64_t h_q,
    uint64_t h_kv,
    uint64_t t_q,
    uint64_t t_k,
    uint64_t d,
    float scale,
    int is_causal,
    const uint16_t *q, Bf16SdpaStrides sq,
    const uint16_t *k, Bf16SdpaStrides sk,
    const uint16_t *v, Bf16SdpaStrides sv,
    const uint16_t *o, Bf16SdpaStrides so,
    const uint16_t *do_in, Bf16SdpaStrides sdo,
    const float *lse_f32,
    float *d_row_f32,
    float *gqa_dk_f32,
    float *gqa_dv_f32,
    uint16_t *dq, Bf16SdpaStrides sdq,
    uint16_t *dk, Bf16SdpaStrides sdk,
    uint16_t *dv, Bf16SdpaStrides sdv,
    int accumulate_dq,
    int accumulate_dk,
    int accumulate_dv);

} // extern "C"
