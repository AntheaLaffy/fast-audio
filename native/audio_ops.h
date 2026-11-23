#pragma once
#include "cuda_backend.h"
#include <cstdint>

extern "C" {

// Embedding (FP32)
int fa_cuda_embedding_fwd_f32(
    Context *ctx,
    uint64_t vocab_size,
    uint64_t embed_dim,
    uint64_t num_indices,
    const float *table,
    const int64_t *indices,
    float *out);

int fa_cuda_embedding_bwd_f32(
    Context *ctx,
    uint64_t vocab_size,
    uint64_t embed_dim,
    uint64_t num_indices,
    int64_t padding_idx,
    int has_padding_idx,
    const int64_t *indices,
    const float *upstream,
    float *grad_table,
    int accumulate);

// RoPE (FP32, arbitrary 4D signed strides)
struct NativeRopeDims {
    uint64_t dims[4];
    int64_t in_strides[4];
    int64_t out_strides[4];
    int64_t in_offset;
    int64_t out_offset;
    int32_t seq_axis;
    int32_t rotary_dim;
    int32_t contiguous_same;
    int32_t reserved;
};

int fa_cuda_rope_create(
    Context *ctx,
    NativeRopeDims dims,
    const float *cos_host,
    const float *sin_host,
    size_t *workspace,
    size_t *shared,
    void **out);

int fa_cuda_rope_run(
    void *plan,
    const float *input,
    float *output,
    int inverse,
    int accumulate);

void fa_cuda_rope_destroy(void *plan);

// Fused CrossEntropy (FP32, zero [N, V] intermediate)
int fa_cuda_cross_entropy_fwd_f32(
    Context *ctx,
    uint64_t num_rows,
    uint64_t vocab_size,
    int64_t ignore_index,
    int has_ignore_index,
    float label_smoothing,
    const float *logits,
    const int64_t *targets,
    float *row_losses_scratch,
    int32_t *row_valid_scratch,
    float *out_scalar);

int fa_cuda_cross_entropy_bwd_f32(
    Context *ctx,
    uint64_t num_rows,
    uint64_t vocab_size,
    int64_t ignore_index,
    int has_ignore_index,
    float label_smoothing,
    float upstream_scalar,
    const float *logits,
    const int64_t *targets,
    int32_t *active_count_scratch,
    float *grad_logits,
    int accumulate);

// Philox4x32-10 Dropout (FP32, bit-identical to CPU)
int fa_cuda_dropout_fwd_f32(
    Context *ctx,
    uint64_t n,
    float p,
    uint64_t seed,
    uint64_t step,
    const float *input,
    float *output);

int fa_cuda_dropout_bwd_f32(
    Context *ctx,
    uint64_t n,
    float p,
    uint64_t seed,
    uint64_t step,
    const float *upstream,
    float *output,
    int accumulate);

} // extern "C"
