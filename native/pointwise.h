#pragma once
#include <cstddef>
#include <cstdint>
struct PointwiseDims {
    uint64_t len;
    int64_t offsets[3];
    uint64_t inner;
    int64_t strides[3];
    int32_t rank, arity, operation;
};
struct PointwiseAxis { uint64_t dim; int64_t strides[3]; };
static_assert(sizeof(PointwiseDims) == 80 && offsetof(PointwiseDims, rank) == 64 &&
    offsetof(PointwiseDims, operation) == 72, "Rust pointwise geometry ABI mismatch");
static_assert(sizeof(PointwiseAxis) == 32, "Rust pointwise axis ABI mismatch");
