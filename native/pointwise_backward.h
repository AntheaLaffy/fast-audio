#pragma once
#include <cstddef>
#include <cstdint>
struct PointwiseBackwardDims {
    uint64_t groups, count;
    int64_t origins[4];
    int32_t rank, reduction_rank, operation, arity, operand, kind, needs, reserved;
};
struct PointwiseBackwardGroup { uint64_t dim; int64_t reads[3], output; };
struct PointwiseBackwardAxis { uint64_t dim; int64_t reads[3]; };
static_assert(sizeof(PointwiseBackwardDims) == 80 && offsetof(PointwiseBackwardDims,rank) == 48 &&
    offsetof(PointwiseBackwardDims,needs) == 72, "Rust pointwise backward geometry ABI mismatch");
static_assert(sizeof(PointwiseBackwardGroup) == 40 && sizeof(PointwiseBackwardAxis) == 32,
    "Rust pointwise backward axis ABI mismatch");
