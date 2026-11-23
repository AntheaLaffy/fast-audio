#pragma once
#include <cstddef>
#include <cstdint>

struct FaNormDims {
    uint64_t groups;
    uint64_t inner_len;
    int64_t offsets[3]; // [x, y_or_dy, dx_or_0]
    int32_t outer_rank;
    int32_t inner_rank;
    int32_t kind;             // 0 = LayerNorm, 1 = RmsNorm
    int32_t fully_contiguous; // 1 iff all active views are fully contiguous
    float eps;
    int32_t reserved;
};

struct FaNormAxis {
    uint64_t dim;
    int64_t strides[3];
};

static_assert(sizeof(FaNormDims) == 64, "FaNormDims ABI size");
static_assert(offsetof(FaNormDims, offsets) == 16, "FaNormDims offsets offset");
static_assert(offsetof(FaNormDims, outer_rank) == 40, "FaNormDims outer_rank offset");
static_assert(offsetof(FaNormDims, eps) == 56, "FaNormDims eps offset");
static_assert(sizeof(FaNormAxis) == 32, "FaNormAxis ABI size");
