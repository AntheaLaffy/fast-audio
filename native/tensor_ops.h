#pragma once
#include <cstdint>
struct TensorOpDims {
    uint64_t dims[4], strides[4], len, offset, inner, width;
    int32_t kind, rank;
    float param, eps;
};
static_assert(sizeof(TensorOpDims)==112,"Rust tensor operation ABI mismatch");
struct TensorMap32 {
    uint32_t dims[4], strides[4], reciprocal[4], len, offset, multiply;
};
static_assert(sizeof(TensorMap32)==60,"Rust mapped operation ABI mismatch");
