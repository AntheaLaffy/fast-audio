#pragma once
#include <cstddef>
#include <cstdint>
struct TensorCopyDims {
    uint64_t len;
    int64_t source_offset, destination_offset;
    int32_t rank, contiguous;
};
static_assert(sizeof(TensorCopyDims) == 32 && offsetof(TensorCopyDims, rank) == 24,
    "Rust tensor copy ABI mismatch");
