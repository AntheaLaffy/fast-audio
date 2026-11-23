#include "audio_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace {

__host__ __device__ __forceinline__ uint64_t umin64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

__device__ __forceinline__ float warp_reduce_max_f32(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(0xffffffffu, v, off);
        v = (isnan(v) || isnan(o)) ? NAN : fmaxf(v, o);
    }
    return v;
}

__device__ __forceinline__ double warp_reduce_sum_f64(double v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ int32_t warp_reduce_sum_i32(int32_t v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// Exact Salmon et al. (2011) Philox4x32-10 matching src/dropout.rs bit-for-bit.
__device__ __forceinline__ uint4 philox4x32_10_dev(uint4 ctr, uint2 key) {
    constexpr uint64_t M0 = 0xD2511F53ULL;
    constexpr uint64_t M1 = 0xCD9E8D57ULL;
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        uint64_t p0 = M0 * static_cast<uint64_t>(ctr.x);
        uint64_t p1 = M1 * static_cast<uint64_t>(ctr.z);
        uint32_t hi0 = static_cast<uint32_t>(p0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(p0);
        uint32_t hi1 = static_cast<uint32_t>(p1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(p1);
        ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += W0;
        key.y += W1;
    }
    return ctr;
}

__device__ __forceinline__ float u32_to_unit_f32_dev(uint32_t word) {
    return static_cast<float>(word >> 8) * (1.0f / 16777216.0f);
}

// --- Embedding (FP32) ---

__global__ void embedding_fwd_f32_kernel(
    uint64_t vocab,
    uint64_t dim,
    uint64_t num_idx,
    const float *__restrict__ table,
    const int64_t *__restrict__ indices,
    float *__restrict__ out)
{
    uint64_t total = num_idx * dim;
    bool vec4 = ((dim & 3ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(table) & 15ULL) == 0ULL) &&
                ((reinterpret_cast<uintptr_t>(out) & 15ULL) == 0ULL);
    if (vec4) {
        uint64_t dim4 = dim >> 2;
        uint64_t total4 = num_idx * dim4;
        const float4 *t4 = reinterpret_cast<const float4 *>(table);
        float4 *o4 = reinterpret_cast<float4 *>(out);
        uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
        for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < total4; i += stride) {
            uint64_t pos = i / dim4;
            uint64_t d4 = i - pos * dim4;
            int64_t tok = indices[pos];
            if (tok >= 0 && static_cast<uint64_t>(tok) < vocab) {
                o4[i] = t4[static_cast<uint64_t>(tok) * dim4 + d4];
            } else {
                o4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        return;
    }
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        uint64_t pos = i / dim;
        uint64_t d = i - pos * dim;
        int64_t tok = indices[pos];
        out[i] = (tok >= 0 && static_cast<uint64_t>(tok) < vocab)
            ? table[static_cast<uint64_t>(tok) * dim + d]
            : 0.0f;
    }
}

// Deterministic zero-atomic Embedding backward: each block owns 4 vocab rows,
// streams token index chunks into SMEM once, collects hits in ascending order,
// and performs coalesced sums along embed_dim.
constexpr int EMB_ROWS_PER_BLK = 4;
constexpr int EMB_IDX_TILE = 256;

__global__ void embedding_bwd_f32_det_kernel(
    uint64_t vocab,
    uint64_t dim,
    uint64_t num_idx,
    int64_t padding_idx,
    int has_padding_idx,
    const int64_t *__restrict__ indices,
    const float *__restrict__ upstream,
    float *__restrict__ grad_table,
    int accumulate)
{
    __shared__ int64_t s_idx[EMB_IDX_TILE];
    __shared__ uint32_t s_hits[EMB_ROWS_PER_BLK][EMB_IDX_TILE];
    __shared__ int s_cnt[EMB_ROWS_PER_BLK];

    uint64_t base_row = static_cast<uint64_t>(blockIdx.x) * EMB_ROWS_PER_BLK;
    if (base_row >= vocab) return;
    int n_rows = static_cast<int>(umin64(EMB_ROWS_PER_BLK, vocab - base_row));

    // Zero output rows if Overwrite
    if (!accumulate) {
        for (int r = 0; r < n_rows; ++r) {
            float *dst = grad_table + (base_row + r) * dim;
            for (uint64_t d = threadIdx.x; d < dim; d += blockDim.x) {
                dst[d] = 0.0f;
            }
        }
    }

    for (uint64_t tile = 0; tile < num_idx; tile += EMB_IDX_TILE) {
        int tile_len = static_cast<int>(umin64(EMB_IDX_TILE, num_idx - tile));
        if (threadIdx.x < tile_len) {
            s_idx[threadIdx.x] = indices[tile + threadIdx.x];
        }
        __syncthreads();

        if (threadIdx.x < EMB_ROWS_PER_BLK) {
            int r = threadIdx.x;
            int cnt = 0;
            if (r < n_rows) {
                int64_t target = static_cast<int64_t>(base_row + r);
                if (!has_padding_idx || target != padding_idx) {
                    for (int k = 0; k < tile_len; ++k) {
                        if (s_idx[k] == target) {
                            s_hits[r][cnt++] = static_cast<uint32_t>(tile + k);
                        }
                    }
                }
            }
            s_cnt[r] = cnt;
        }
        __syncthreads();

        for (int r = 0; r < n_rows; ++r) {
            int cnt = s_cnt[r];
            if (cnt == 0) continue;
            float *dst = grad_table + (base_row + r) * dim;
            for (uint64_t d = threadIdx.x; d < dim; d += blockDim.x) {
                float acc = dst[d];
                for (int h = 0; h < cnt; ++h) {
                    acc += upstream[static_cast<uint64_t>(s_hits[r][h]) * dim + d];
                }
                dst[d] = acc;
            }
        }
        __syncthreads();
    }
}

__global__ void embedding_bwd_f32_atomic_kernel(
    uint64_t vocab,
    uint64_t dim,
    uint64_t num_idx,
    int64_t padding_idx,
    int has_padding_idx,
    const int64_t *__restrict__ indices,
    const float *__restrict__ upstream,
    float *__restrict__ grad_table)
{
    uint64_t total = num_idx * dim;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        uint64_t pos = i / dim;
        uint64_t d = i - pos * dim;
        int64_t tok = indices[pos];
        if (tok >= 0 && static_cast<uint64_t>(tok) < vocab &&
            (!has_padding_idx || tok != padding_idx)) {
            atomicAdd(&grad_table[static_cast<uint64_t>(tok) * dim + d], upstream[i]);
        }
    }
}

// --- RoPE (FP32) ---

struct RopePlanDev {
    Context *ctx;
    NativeRopeDims dims;
    float *cos_dev;
    float *sin_dev;
    size_t workspace;
};

__global__ void rope_f32_kernel(
    NativeRopeDims g,
    const float *__restrict__ cos_tbl,
    const float *__restrict__ sin_tbl,
    const float *__restrict__ inp,
    float *__restrict__ out,
    int inverse,
    int accumulate)
{
    uint64_t d0 = g.dims[0], d1 = g.dims[1], d2 = g.dims[2], d3 = g.dims[3];
    uint64_t half = static_cast<uint64_t>(g.rotary_dim >> 1);
    uint64_t pass = d3 - static_cast<uint64_t>(g.rotary_dim);
    uint64_t work_per_row = half + pass;
    uint64_t rows = d0 * d1 * d2;
    uint64_t total = rows * work_per_row;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        uint64_t row = idx / work_per_row;
        uint64_t col = idx - row * work_per_row;
        uint64_t i2 = row % d2;
        uint64_t tmp = row / d2;
        uint64_t i1 = tmp % d1;
        uint64_t i0 = tmp / d1;
        uint64_t t = (g.seq_axis == 2) ? i2 : i1;

        int64_t in_base = g.in_offset + static_cast<int64_t>(i0) * g.in_strides[0]
                        + static_cast<int64_t>(i1) * g.in_strides[1]
                        + static_cast<int64_t>(i2) * g.in_strides[2];
        int64_t out_base = g.out_offset + static_cast<int64_t>(i0) * g.out_strides[0]
                         + static_cast<int64_t>(i1) * g.out_strides[1]
                         + static_cast<int64_t>(i2) * g.out_strides[2];

        if (col < half) {
            int64_t ix0 = in_base + static_cast<int64_t>(col) * g.in_strides[3];
            int64_t ix1 = in_base + static_cast<int64_t>(col + half) * g.in_strides[3];
            int64_t ox0 = out_base + static_cast<int64_t>(col) * g.out_strides[3];
            int64_t ox1 = out_base + static_cast<int64_t>(col + half) * g.out_strides[3];
            float a = inp[ix0];
            float b = inp[ix1];
            float c = cos_tbl[t * half + col];
            float s = sin_tbl[t * half + col];
            float y0 = inverse ? (a * c + b * s) : (a * c - b * s);
            float y1 = inverse ? (-a * s + b * c) : (a * s + b * c);
            if (accumulate) {
                out[ox0] += y0;
                out[ox1] += y1;
            } else {
                out[ox0] = y0;
                out[ox1] = y1;
            }
        } else {
            uint64_t d = static_cast<uint64_t>(g.rotary_dim) + (col - half);
            float v = inp[in_base + static_cast<int64_t>(d) * g.in_strides[3]];
            int64_t ox = out_base + static_cast<int64_t>(d) * g.out_strides[3];
            if (accumulate) {
                out[ox] += v;
            } else {
                out[ox] = v;
            }
        }
    }
}

// --- Fused CrossEntropy (FP32) ---

__global__ void ce_fwd_rows_f32_kernel(
    uint64_t num_rows,
    uint64_t vocab,
    int64_t ignore_index,
    int has_ignore,
    float label_smoothing,
    const float *__restrict__ logits,
    const int64_t *__restrict__ targets,
    float *__restrict__ row_losses,
    int32_t *__restrict__ row_valid)
{
    uint64_t r = blockIdx.x;
    if (r >= num_rows) return;
    int64_t tok = targets[r];
    if ((has_ignore && tok == ignore_index) || tok < 0 || static_cast<uint64_t>(tok) >= vocab) {
        if (threadIdx.x == 0) {
            row_losses[r] = 0.0f;
            row_valid[r] = 0;
        }
        return;
    }

    const float *row = logits + r * vocab;
    __shared__ float s_max[8];
    __shared__ double s_sum[8];
    __shared__ double s_shift[8];

    float local_max = -INFINITY;
    for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        float v = row[j];
        local_max = (isnan(local_max) || isnan(v)) ? NAN : fmaxf(local_max, v);
    }
    local_max = warp_reduce_max_f32(local_max);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    if (lane == 0) s_max[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < (blockDim.x >> 5)) ? s_max[lane] : -INFINITY;
        m = warp_reduce_max_f32(m);
        if (lane == 0) s_max[0] = m;
    }
    __syncthreads();
    float m = s_max[0];

    double local_exp = 0.0;
    double local_shift = 0.0;
    for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        double s = static_cast<double>(row[j]) - static_cast<double>(m);
        local_exp += exp(s);
        local_shift += s;
    }
    local_exp = warp_reduce_sum_f64(local_exp);
    local_shift = warp_reduce_sum_f64(local_shift);
    if (lane == 0) {
        s_sum[warp] = local_exp;
        s_shift[warp] = local_shift;
    }
    __syncthreads();
    if (warp == 0) {
        double se = (lane < (blockDim.x >> 5)) ? s_sum[lane] : 0.0;
        double ss = (lane < (blockDim.x >> 5)) ? s_shift[lane] : 0.0;
        se = warp_reduce_sum_f64(se);
        ss = warp_reduce_sum_f64(ss);
        if (lane == 0) {
            double log_se = log(se);
            double nll = log_se + (static_cast<double>(m) - row[static_cast<uint64_t>(tok)]);
            double eps = static_cast<double>(label_smoothing);
            double loss = nll;
            if (eps > 0.0) {
                double mean_log_p = (ss / static_cast<double>(vocab)) - log_se;
                loss = (1.0 - eps) * nll - eps * mean_log_p;
            }
            row_losses[r] = static_cast<float>(loss);
            row_valid[r] = 1;
        }
    }
}

__global__ void ce_fwd_reduce_f32_kernel(
    uint64_t num_rows,
    const float *__restrict__ row_losses,
    const int32_t *__restrict__ row_valid,
    float *__restrict__ out_scalar)
{
    __shared__ double s_loss[8];
    __shared__ int32_t s_cnt[8];
    double acc = 0.0;
    int32_t cnt = 0;
    for (uint64_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
        acc += static_cast<double>(row_losses[i]);
        cnt += row_valid[i];
    }
    acc = warp_reduce_sum_f64(acc);
    cnt = warp_reduce_sum_i32(cnt);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    if (lane == 0) {
        s_loss[warp] = acc;
        s_cnt[warp] = cnt;
    }
    __syncthreads();
    if (warp == 0) {
        double l = (lane < (blockDim.x >> 5)) ? s_loss[lane] : 0.0;
        int32_t c = (lane < (blockDim.x >> 5)) ? s_cnt[lane] : 0;
        l = warp_reduce_sum_f64(l);
        c = warp_reduce_sum_i32(c);
        if (lane == 0) {
            out_scalar[0] = (c > 0) ? static_cast<float>(l / static_cast<double>(c)) : 0.0f;
        }
    }
}

__global__ void ce_count_valid_kernel(
    uint64_t num_rows,
    uint64_t vocab,
    int64_t ignore_index,
    int has_ignore,
    const int64_t *__restrict__ targets,
    int32_t *__restrict__ active_out)
{
    __shared__ int32_t s_cnt[8];
    int32_t cnt = 0;
    for (uint64_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
        int64_t t = targets[i];
        if ((!has_ignore || t != ignore_index) && t >= 0 && static_cast<uint64_t>(t) < vocab) {
            cnt += 1;
        }
    }
    cnt = warp_reduce_sum_i32(cnt);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    if (lane == 0) s_cnt[warp] = cnt;
    __syncthreads();
    if (warp == 0) {
        int32_t c = (lane < (blockDim.x >> 5)) ? s_cnt[lane] : 0;
        c = warp_reduce_sum_i32(c);
        if (lane == 0) active_out[0] = c;
    }
}

__global__ void ce_bwd_rows_f32_kernel(
    uint64_t num_rows,
    uint64_t vocab,
    int64_t ignore_index,
    int has_ignore,
    float label_smoothing,
    float upstream_scalar,
    const float *__restrict__ logits,
    const int64_t *__restrict__ targets,
    const int32_t *__restrict__ active_ptr,
    float *__restrict__ grad_logits,
    int accumulate)
{
    uint64_t r = blockIdx.x;
    if (r >= num_rows) return;
    int32_t active = active_ptr[0];
    int64_t tok = targets[r];
    float *dst = grad_logits + r * vocab;
    if (active <= 0 || (has_ignore && tok == ignore_index) || tok < 0 ||
        static_cast<uint64_t>(tok) >= vocab) {
        if (!accumulate) {
            for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
                dst[j] = 0.0f;
            }
        }
        return;
    }

    const float *row = logits + r * vocab;
    __shared__ float s_max[8];
    __shared__ double s_sum[8];

    float local_max = -INFINITY;
    for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        float v = row[j];
        local_max = (isnan(local_max) || isnan(v)) ? NAN : fmaxf(local_max, v);
    }
    local_max = warp_reduce_max_f32(local_max);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    if (lane == 0) s_max[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < (blockDim.x >> 5)) ? s_max[lane] : -INFINITY;
        m = warp_reduce_max_f32(m);
        if (lane == 0) s_max[0] = m;
    }
    __syncthreads();
    float m = s_max[0];

    double local_exp = 0.0;
    for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        local_exp += exp(static_cast<double>(row[j] - m));
    }
    local_exp = warp_reduce_sum_f64(local_exp);
    if (lane == 0) s_sum[warp] = local_exp;
    __syncthreads();
    if (warp == 0) {
        double se = (lane < (blockDim.x >> 5)) ? s_sum[lane] : 0.0;
        se = warp_reduce_sum_f64(se);
        if (lane == 0) s_sum[0] = se;
    }
    __syncthreads();

    float inv_z = static_cast<float>(1.0 / s_sum[0]);
    float scale = upstream_scalar / static_cast<float>(active);
    float smooth_off = label_smoothing / static_cast<float>(vocab);
    float target_w = 1.0f - label_smoothing;
    uint64_t t_col = static_cast<uint64_t>(tok);

    for (uint64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        float prob = expf(row[j] - m) * inv_z;
        float g = (prob - smooth_off - (j == t_col ? target_w : 0.0f)) * scale;
        if (accumulate) {
            dst[j] += g;
        } else {
            dst[j] = g;
        }
    }
}

// --- Philox4x32-10 Dropout (FP32) ---

__global__ void dropout_f32_kernel(
    uint64_t n,
    float p,
    float inv_keep,
    uint2 key,
    uint32_t step_lo,
    uint32_t step_hi,
    const float *__restrict__ inp,
    float *__restrict__ out,
    int accumulate)
{
    uint64_t n_blocks = (n + 3ULL) >> 2;
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t blk = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         blk < n_blocks; blk += stride) {
        uint64_t base = blk << 2;
        if (p == 0.0f) {
            for (uint64_t k = 0; k < 4 && base + k < n; ++k) {
                float v = inp[base + k];
                if (accumulate) out[base + k] += v;
                else out[base + k] = v;
            }
            continue;
        }
        uint4 ctr = make_uint4(
            static_cast<uint32_t>(blk),
            static_cast<uint32_t>(blk >> 32),
            step_lo,
            step_hi);
        uint4 r = philox4x32_10_dev(ctr, key);
        uint32_t words[4] = {r.x, r.y, r.z, r.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (base + k < n) {
                float v = (u32_to_unit_f32_dev(words[k]) >= p)
                    ? (inp[base + k] * inv_keep)
                    : 0.0f;
                if (accumulate) out[base + k] += v;
                else out[base + k] = v;
            }
        }
    }
}

} // namespace

extern "C" int fa_cuda_embedding_fwd_f32(
    Context *ctx,
    uint64_t vocab_size,
    uint64_t embed_dim,
    uint64_t num_indices,
    const float *table,
    const int64_t *indices,
    float *out)
{
    if (!ctx || vocab_size == 0 || embed_dim == 0) return rt(cudaErrorInvalidValue);
    if (num_indices == 0) return 0;
    if (!table || !indices || !out) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t work = ((embed_dim & 3ULL) == 0ULL) ? (num_indices * (embed_dim >> 2)) : (num_indices * embed_dim);
    unsigned grid = static_cast<unsigned>(umin64((work + 255ULL) / 256ULL, 65535ULL));
    embedding_fwd_f32_kernel<<<grid, 256, 0, ctx->stream>>>(
        vocab_size, embed_dim, num_indices, table, indices, out);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_embedding_bwd_f32(
    Context *ctx,
    uint64_t vocab_size,
    uint64_t embed_dim,
    uint64_t num_indices,
    int64_t padding_idx,
    int has_padding_idx,
    const int64_t *indices,
    const float *upstream,
    float *grad_table,
    int accumulate)
{
    if (!ctx || vocab_size == 0 || embed_dim == 0 || !grad_table) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    if (num_indices == 0) {
        if (!accumulate) {
            return rt(cudaMemsetAsync(
                grad_table, 0, static_cast<size_t>(vocab_size * embed_dim) * sizeof(float), ctx->stream));
        }
        return 0;
    }
    if (!indices || !upstream) return rt(cudaErrorInvalidValue);

    // Deterministic zero-atomic path for typical audio vocab sizes (V*N <= 134M).
    if (vocab_size <= 32768ULL && vocab_size * num_indices <= 134217728ULL) {
        unsigned grid = static_cast<unsigned>((vocab_size + EMB_ROWS_PER_BLK - 1) / EMB_ROWS_PER_BLK);
        embedding_bwd_f32_det_kernel<<<grid, 256, 0, ctx->stream>>>(
            vocab_size, embed_dim, num_indices, padding_idx, has_padding_idx,
            indices, upstream, grad_table, accumulate);
        return rt(cudaGetLastError());
    }
    if (!accumulate) {
        code = rt(cudaMemsetAsync(
            grad_table, 0, static_cast<size_t>(vocab_size * embed_dim) * sizeof(float), ctx->stream));
        if (code) return code;
    }
    uint64_t total = num_indices * embed_dim;
    unsigned grid = static_cast<unsigned>(umin64((total + 255ULL) / 256ULL, 65535ULL));
    embedding_bwd_f32_atomic_kernel<<<grid, 256, 0, ctx->stream>>>(
        vocab_size, embed_dim, num_indices, padding_idx, has_padding_idx,
        indices, upstream, grad_table);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_rope_create(
    Context *ctx,
    NativeRopeDims dims,
    const float *cos_host,
    const float *sin_host,
    size_t *workspace,
    size_t *shared,
    void **out)
{
    if (!ctx || !workspace || !shared || !out) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t seq_len = (dims.seq_axis == 2) ? dims.dims[2] : dims.dims[1];
    uint64_t half = static_cast<uint64_t>(dims.rotary_dim >> 1);
    size_t tbl_bytes = static_cast<size_t>(seq_len * half) * sizeof(float);

    RopePlanDev *plan = static_cast<RopePlanDev *>(std::calloc(1, sizeof(RopePlanDev)));
    if (!plan) return 1002;
    plan->ctx = ctx;
    plan->dims = dims;
    plan->workspace = tbl_bytes * 2;

    if (tbl_bytes > 0) {
        code = rt(cudaMalloc(&plan->cos_dev, tbl_bytes));
        if (!code) code = rt(cudaMalloc(&plan->sin_dev, tbl_bytes));
        if (!code) code = rt(cudaMemcpyAsync(plan->cos_dev, cos_host, tbl_bytes, cudaMemcpyHostToDevice, ctx->stream));
        if (!code) code = rt(cudaMemcpyAsync(plan->sin_dev, sin_host, tbl_bytes, cudaMemcpyHostToDevice, ctx->stream));
        if (!code) code = rt(cudaStreamSynchronize(ctx->stream));
        if (code) {
            if (plan->cos_dev) cudaFree(plan->cos_dev);
            if (plan->sin_dev) cudaFree(plan->sin_dev);
            std::free(plan);
            return code;
        }
    }
    *workspace = plan->workspace;
    *shared = 0;
    *out = plan;
    return 0;
}

extern "C" int fa_cuda_rope_run(
    void *plan_ptr,
    const float *input,
    float *output,
    int inverse,
    int accumulate)
{
    if (!plan_ptr) return rt(cudaErrorInvalidValue);
    RopePlanDev *p = static_cast<RopePlanDev *>(plan_ptr);
    uint64_t rows = p->dims.dims[0] * p->dims.dims[1] * p->dims.dims[2];
    if (rows == 0 || p->dims.dims[3] == 0) return 0;
    if (!input || !output) return rt(cudaErrorInvalidValue);
    int code = activate(p->ctx);
    if (code) return code;
    uint64_t work_per_row = static_cast<uint64_t>(p->dims.rotary_dim >> 1) +
                            (p->dims.dims[3] - static_cast<uint64_t>(p->dims.rotary_dim));
    uint64_t total = rows * work_per_row;
    unsigned grid = static_cast<unsigned>(umin64((total + 255ULL) / 256ULL, 65535ULL));
    rope_f32_kernel<<<grid, 256, 0, p->ctx->stream>>>(
        p->dims, p->cos_dev, p->sin_dev, input, output, inverse, accumulate);
    return rt(cudaGetLastError());
}

extern "C" void fa_cuda_rope_destroy(void *plan_ptr) {
    if (!plan_ptr) return;
    RopePlanDev *p = static_cast<RopePlanDev *>(plan_ptr);
    cudaSetDevice(p->ctx->device);
    cudaStreamSynchronize(p->ctx->stream);
    if (p->cos_dev) cudaFree(p->cos_dev);
    if (p->sin_dev) cudaFree(p->sin_dev);
    std::free(p);
}

extern "C" int fa_cuda_cross_entropy_fwd_f32(
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
    float *out_scalar)
{
    if (!ctx || vocab_size == 0 || !out_scalar) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    if (num_rows == 0) {
        return rt(cudaMemsetAsync(out_scalar, 0, sizeof(float), ctx->stream));
    }
    if (!logits || !targets || !row_losses_scratch || !row_valid_scratch) {
        return rt(cudaErrorInvalidValue);
    }
    ce_fwd_rows_f32_kernel<<<static_cast<unsigned>(num_rows), 256, 0, ctx->stream>>>(
        num_rows, vocab_size, ignore_index, has_ignore_index, label_smoothing,
        logits, targets, row_losses_scratch, row_valid_scratch);
    code = rt(cudaGetLastError());
    if (code) return code;
    ce_fwd_reduce_f32_kernel<<<1, 256, 0, ctx->stream>>>(
        num_rows, row_losses_scratch, row_valid_scratch, out_scalar);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_cross_entropy_bwd_f32(
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
    int accumulate)
{
    if (!ctx || vocab_size == 0) return rt(cudaErrorInvalidValue);
    if (num_rows == 0) return 0;
    if (!logits || !targets || !active_count_scratch || !grad_logits) {
        return rt(cudaErrorInvalidValue);
    }
    int code = activate(ctx);
    if (code) return code;
    ce_count_valid_kernel<<<1, 256, 0, ctx->stream>>>(
        num_rows, vocab_size, ignore_index, has_ignore_index, targets, active_count_scratch);
    code = rt(cudaGetLastError());
    if (code) return code;
    ce_bwd_rows_f32_kernel<<<static_cast<unsigned>(num_rows), 256, 0, ctx->stream>>>(
        num_rows, vocab_size, ignore_index, has_ignore_index, label_smoothing,
        upstream_scalar, logits, targets, active_count_scratch, grad_logits, accumulate);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_dropout_fwd_f32(
    Context *ctx,
    uint64_t n,
    float p,
    uint64_t seed,
    uint64_t step,
    const float *input,
    float *output)
{
    if (!ctx || !std::isfinite(p) || p < 0.0f || p >= 1.0f) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!input || !output) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t n_blocks = (n + 3ULL) >> 2;
    unsigned grid = static_cast<unsigned>(umin64((n_blocks + 255ULL) / 256ULL, 65535ULL));
    uint2 key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
    dropout_f32_kernel<<<grid, 256, 0, ctx->stream>>>(
        n, p, 1.0f / (1.0f - p), key,
        static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32),
        input, output, 0);
    return rt(cudaGetLastError());
}

extern "C" int fa_cuda_dropout_bwd_f32(
    Context *ctx,
    uint64_t n,
    float p,
    uint64_t seed,
    uint64_t step,
    const float *upstream,
    float *output,
    int accumulate)
{
    if (!ctx || !std::isfinite(p) || p < 0.0f || p >= 1.0f) return rt(cudaErrorInvalidValue);
    if (n == 0) return 0;
    if (!upstream || !output) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    uint64_t n_blocks = (n + 3ULL) >> 2;
    unsigned grid = static_cast<unsigned>(umin64((n_blocks + 255ULL) / 256ULL, 65535ULL));
    uint2 key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
    dropout_f32_kernel<<<grid, 256, 0, ctx->stream>>>(
        n, p, 1.0f / (1.0f - p), key,
        static_cast<uint32_t>(step), static_cast<uint32_t>(step >> 32),
        upstream, output, accumulate);
    return rt(cudaGetLastError());
}
