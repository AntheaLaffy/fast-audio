#pragma once
#include <cstddef>
#include <cstdint>
struct ReduceDims {
    uint64_t groups, count;
    int64_t source_offset, destination_offset;
    int32_t rank, reduction_rank, kind, operation;
    float divisor;
    int32_t reserved;
};
struct ReduceGroupAxis { uint64_t dim; int64_t source, destination; };
struct ReduceAxis { uint64_t dim; int64_t stride; };
struct ReduceNlBwdDims {
    uint64_t groups, count;
    int64_t source_offset, upstream_offset, destination_offset;
    int32_t rank, reduction_rank, operation, reserved;
};
struct ReduceNlGroupAxis { uint64_t dim; int64_t source, upstream, destination; };
struct ReduceNlRedAxis { uint64_t dim; int64_t source, destination; };
static_assert(sizeof(ReduceDims) == 56 && offsetof(ReduceDims, rank) == 32 &&
    offsetof(ReduceDims, divisor) == 48, "Rust reduction geometry ABI mismatch");
static_assert(sizeof(ReduceGroupAxis) == 24 && sizeof(ReduceAxis) == 16,
    "Rust reduction axis ABI mismatch");
static_assert(sizeof(ReduceNlBwdDims) == 56 && offsetof(ReduceNlBwdDims, rank) == 40,
    "Rust nonlinear reduction backward ABI mismatch");
static_assert(sizeof(ReduceNlGroupAxis) == 32 && sizeof(ReduceNlRedAxis) == 24,
    "Rust nonlinear reduction axis ABI mismatch");
