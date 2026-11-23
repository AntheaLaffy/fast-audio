#pragma once
#include <cstddef>
#include <cstdint>

struct FaBmmDims {
    uint64_t m, n, k;
    uint64_t out_batches, red_batches;
    int64_t origins[3];
    int64_t strides[6]; // [a_row, a_col, b_row, b_col, c_row, c_col]
    int64_t batch_strides[3];
    int32_t leading[3];
    int32_t transposes[2];
    int32_t out_rank;
    int32_t red_rank;
    int32_t route; // 0 = empty, 1 = zero, 2 = collapsed_2d, 3 = cublas_strided_batched, 4 = tiled_cuda
    int32_t swapped;
    int32_t blas_m, blas_n, blas_k;
    int32_t batch_count;
    int32_t reserved;
};

struct FaBmmAxis {
    uint64_t dim;
    int64_t stride_a;
    int64_t stride_b;
    int64_t stride_c;
};

static_assert(sizeof(FaBmmDims) == 192, "FaBmmDims ABI size");
static_assert(offsetof(FaBmmDims, origins) == 40, "FaBmmDims origins offset");
static_assert(offsetof(FaBmmDims, strides) == 64, "FaBmmDims strides offset");
static_assert(offsetof(FaBmmDims, batch_strides) == 112, "FaBmmDims batch_strides offset");
static_assert(offsetof(FaBmmDims, leading) == 136, "FaBmmDims leading offset");
static_assert(sizeof(FaBmmAxis) == 32, "FaBmmAxis ABI size");
