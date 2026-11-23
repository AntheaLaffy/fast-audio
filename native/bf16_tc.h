#pragma once
#include "cuda_backend.h"
#include <cstdint>

extern "C" {

// Explicit BF16 Tensor Core GEMM via ctx->blas_tc (CUBLAS_COMPUTE_32F, TENSOR_OP).
// Row-major C[m, n] = alpha * op(A)[m, k] @ op(B)[k, n] + beta * C[m, n].
// Output C is BF16 when c_is_f32 == 0, or FP32 (direct master grad accumulation!) when c_is_f32 == 1.
int fa_cuda_bf16_gemm(
    Context *ctx,
    int trans_a,
    int trans_b,
    int m,
    int n,
    int k,
    const uint16_t *a,
    int lda,
    const uint16_t *b,
    int ldb,
    void *c,
    int ldc,
    int c_is_f32,
    float alpha,
    float beta);

int fa_cuda_bf16_bmm_strided(
    Context *ctx,
    int trans_a,
    int trans_b,
    int batch,
    int m,
    int n,
    int k,
    const uint16_t *a,
    int lda,
    int64_t stride_a,
    const uint16_t *b,
    int ldb,
    int64_t stride_b,
    void *c,
    int ldc,
    int64_t stride_c,
    int c_is_f32,
    float alpha,
    float beta);

// Explicit TF32 Tensor Core GEMM on FP32 storage via ctx->blas_tc (CUBLAS_COMPUTE_32F_FAST_TF32).
int fa_cuda_tf32_gemm(
    Context *ctx,
    int trans_a,
    int trans_b,
    int m,
    int n,
    int k,
    const float *a,
    int lda,
    const float *b,
    int ldb,
    float *c,
    int ldc,
    float alpha,
    float beta);

// Vectorized BF16 RmsNorm (optional learnable BF16 weight, optional saved FP32 inv_rms).
int fa_cuda_bf16_rmsnorm_fwd(
    Context *ctx,
    uint64_t rows,
    uint64_t cols,
    float eps,
    const uint16_t *x,
    const uint16_t *weight_opt,
    uint16_t *y,
    float *inv_rms_opt);

// Fused ResidualAdd + RmsNorm: residual_out = a + b; y = rmsnorm(residual_out, weight_opt).
int fa_cuda_bf16_residual_rmsnorm_fwd(
    Context *ctx,
    uint64_t rows,
    uint64_t cols,
    float eps,
    const uint16_t *a,
    const uint16_t *b,
    const uint16_t *weight_opt,
    uint16_t *residual_out,
    uint16_t *y,
    float *inv_rms_opt);

// BF16 RmsNorm backward with optional deterministic zero-atomic FP32 dWeight accumulation.
// When dweight_opt != nullptr, dweight_partials_scratch must hold at least min(rows, 128) * cols floats.
int fa_cuda_bf16_rmsnorm_bwd(
    Context *ctx,
    uint64_t rows,
    uint64_t cols,
    float eps,
    const uint16_t *x,
    const uint16_t *weight_opt,
    const float *inv_rms_opt,
    const uint16_t *dy,
    uint16_t *dx,
    float *dweight_opt,
    float *dweight_partials_scratch,
    int accumulate_dx,
    int accumulate_dw);

// Vectorized BF16 LayerNorm (parameter-free, FP32 internal stats).
int fa_cuda_bf16_layernorm_fwd(
    Context *ctx,
    uint64_t rows,
    uint64_t cols,
    float eps,
    const uint16_t *x,
    uint16_t *y);

int fa_cuda_bf16_layernorm_bwd(
    Context *ctx,
    uint64_t rows,
    uint64_t cols,
    float eps,
    const uint16_t *x,
    const uint16_t *dy,
    uint16_t *dx,
    int accumulate);

// Vectorized BF16 Softmax along inner axis (FP32 internal accumulation).
int fa_cuda_bf16_softmax_fwd(
    Context *ctx,
    uint64_t rows,
    uint64_t cols,
    const uint16_t *x,
    uint16_t *y);

int fa_cuda_bf16_softmax_bwd(
    Context *ctx,
    uint64_t rows,
    uint64_t cols,
    const uint16_t *y,
    const uint16_t *dy,
    uint16_t *dx,
    int accumulate);

// Vectorized BF16 SwiGLU on fused gate_up [rows, 2*hidden] -> out [rows, hidden]
// where gate = gate_up[:, :hidden] and up = gate_up[:, hidden:].
int fa_cuda_bf16_swiglu_fwd(
    Context *ctx,
    uint64_t rows,
    uint64_t hidden,
    const uint16_t *gate_up,
    uint16_t *out);

int fa_cuda_bf16_swiglu_bwd(
    Context *ctx,
    uint64_t rows,
    uint64_t hidden,
    const uint16_t *gate_up,
    const uint16_t *dout,
    uint16_t *dgate_up);

// Vectorized BF16 Residual Add: out[i] = a[i] + b[i] (safe when out == a).
int fa_cuda_bf16_add(
    Context *ctx,
    uint64_t n,
    const uint16_t *a,
    const uint16_t *b,
    uint16_t *out);

// BF16 RoPE forward/backward on 4D tensor [B, H, T, D] or strided [B, T, H_tot, D] (stride_d == 1).
// Safe for in-place execution (in == out) when stride_d == 1!
int fa_cuda_bf16_rope_run(
    Context *ctx,
    uint64_t batch,
    uint64_t heads,
    uint64_t seq_len,
    uint64_t head_dim,
    uint64_t rotary_dim,
    int64_t in_sb, int64_t in_sh, int64_t in_st,
    int64_t out_sb, int64_t out_sh, int64_t out_st,
    const float *cos_dev,
    const float *sin_dev,
    const uint16_t *inp,
    uint16_t *out,
    int inverse,
    int accumulate);

// BF16 Embedding forward + deterministic zero-atomic FP32 master gradient backward.
int fa_cuda_bf16_embedding_fwd(
    Context *ctx,
    uint64_t vocab_size,
    uint64_t embed_dim,
    uint64_t num_indices,
    const uint16_t *table_bf16,
    const int64_t *indices,
    uint16_t *out_bf16);

int fa_cuda_bf16_embedding_bwd_f32(
    Context *ctx,
    uint64_t vocab_size,
    uint64_t embed_dim,
    uint64_t num_indices,
    int64_t padding_idx,
    int has_padding_idx,
    const int64_t *indices,
    const uint16_t *upstream_bf16,
    float *grad_table_f32,
    int accumulate);

// BF16 Fused CrossEntropy forward, backward, AND single-pass fused forward+backward (zero [N,V] scratch!).
int fa_cuda_bf16_cross_entropy_fwd(
    Context *ctx,
    uint64_t num_rows,
    uint64_t vocab_size,
    int64_t ignore_index,
    int has_ignore_index,
    float label_smoothing,
    const uint16_t *logits_bf16,
    const int64_t *targets,
    float *row_losses_scratch,
    int32_t *row_valid_scratch,
    float *out_scalar_f32);

int fa_cuda_bf16_cross_entropy_bwd(
    Context *ctx,
    uint64_t num_rows,
    uint64_t vocab_size,
    int64_t ignore_index,
    int has_ignore_index,
    float label_smoothing,
    float upstream_scalar,
    const uint16_t *logits_bf16,
    const int64_t *targets,
    int32_t *active_count_scratch,
    uint16_t *grad_logits_bf16,
    int accumulate);

int fa_cuda_bf16_cross_entropy_fwd_bwd(
    Context *ctx,
    uint64_t num_rows,
    uint64_t vocab_size,
    int64_t ignore_index,
    int has_ignore_index,
    float label_smoothing,
    float upstream_scalar,
    const uint16_t *logits_bf16,
    const int64_t *targets,
    int32_t *active_count_scratch,
    float *row_losses_scratch,
    int32_t *row_valid_scratch,
    float *out_scalar_f32,
    uint16_t *grad_logits_bf16);

// BF16 Philox4x32-10 Dropout (bit-identical mask to CPU).
int fa_cuda_bf16_dropout_fwd(
    Context *ctx,
    uint64_t n,
    float p,
    uint64_t seed,
    uint64_t step,
    const uint16_t *input,
    uint16_t *output);

int fa_cuda_bf16_dropout_bwd(
    Context *ctx,
    uint64_t n,
    float p,
    uint64_t seed,
    uint64_t step,
    const uint16_t *upstream,
    uint16_t *output,
    int accumulate);

// Fused FP32-master-weight + BF16-mirror AdamW (updates param_f32, m, v AND writes param_bf16 in 1 kernel!).
// Optional found_inf_ptr (from GradScaler): if non-null and *found_inf_ptr != 0, skips update with 0 host sync.
int fa_cuda_bf16_adamw_param(
    Context *ctx,
    uint64_t n,
    float *param_f32,
    uint16_t *param_bf16,
    const float *grad_f32,
    float *m_f32,
    float *v_f32,
    void *ctrl,
    const uint32_t *found_inf_opt,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int use_grad_scale);

// Device-side zero-host-sync GradScaler (100% CUDA Graph safe).
int fa_cuda_grad_scaler_create(
    Context *ctx,
    float init_scale,
    float growth_factor,
    float backoff_factor,
    int growth_interval,
    void **out_scaler);

void fa_cuda_grad_scaler_destroy(Context *ctx, void *scaler);
int fa_cuda_grad_scaler_reset(Context *ctx, void *scaler);
int fa_cuda_grad_scaler_unscale_check(Context *ctx, void *scaler, uint64_t n, float *grad_f32);
int fa_cuda_grad_scaler_update(Context *ctx, void *scaler);
int fa_cuda_grad_scaler_read_state(Context *ctx, void *scaler, float *out_scale, uint32_t *out_found_inf);
const uint32_t *fa_cuda_grad_scaler_found_inf_ptr(void *scaler);

} // extern "C"
