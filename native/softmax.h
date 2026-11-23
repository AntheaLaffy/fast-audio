#pragma once
#include <cstddef>
#include <cstdint>

struct FaSoftmaxDims {
    uint64_t groups;
    uint64_t axis_len;
    int64_t offsets[3];      // [x, y_or_dy, dx_or_0]
    int64_t axis_strides[3]; // [x, y_or_dy, dx_or_0]
    int32_t group_rank;
    int32_t kind;                  // 0 = Softmax, 1 = LogSoftmax
    int32_t fully_contiguous_last; // 1 iff all active views are contiguous with axis == rank - 1
    int32_t reserved;
};

struct FaSoftmaxAxis {
    uint64_t dim;
    int64_t strides[3];
};

static_assert(sizeof(FaSoftmaxDims) == 80, "FaSoftmaxDims ABI size");
static_assert(offsetof(FaSoftmaxDims, offsets) == 16, "FaSoftmaxDims offsets offset");
static_assert(offsetof(FaSoftmaxDims, axis_strides) == 40, "FaSoftmaxDims axis_strides offset");
static_assert(offsetof(FaSoftmaxDims, group_rank) == 64, "FaSoftmaxDims group_rank offset");
static_assert(sizeof(FaSoftmaxAxis) == 32, "FaSoftmaxAxis ABI size");
