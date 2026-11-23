#pragma once
#include <cstddef>
#include <cstdint>
struct MatmulDims {
    uint64_t m, n, k;
    int64_t origins[3], strides[6];
    int32_t leading[3], transposes[2], kind, reserved[2];
};
static_assert(sizeof(MatmulDims)==128 && offsetof(MatmulDims,origins)==24 &&
    offsetof(MatmulDims,strides)==48 && offsetof(MatmulDims,leading)==96 &&
    offsetof(MatmulDims,kind)==116,"Rust strided GEMM ABI mismatch");
