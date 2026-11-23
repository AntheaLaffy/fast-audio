#pragma once
#include <cstddef>
#include <cstdint>
struct ViewBackwardDims {
    uint64_t base_len, logical_len, groups, reductions, written_begin, written_end;
    int64_t source_offset, destination_offset;
    int32_t rank, reduction_rank, kind, lanes;
};
static_assert(sizeof(ViewBackwardDims) == 80 && offsetof(ViewBackwardDims, rank) == 64 &&
    offsetof(ViewBackwardDims, kind) == 72, "Rust view backward ABI mismatch");
