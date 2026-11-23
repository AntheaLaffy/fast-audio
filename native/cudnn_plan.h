#pragma once
#include <cstddef>
#include <cstdint>

// Negative statuses are adapter decisions; positive statuses retain vendor domains.
constexpr int FA_CUDNN_NO_CONV_PLAN = -1;
struct CudnnConvInfo {
    size_t workspace_bytes;
    int64_t engine_index;
    uint64_t numerical_notes;
    int32_t deterministic;
    int32_t fused_epilogue;
};
static_assert(sizeof(CudnnConvInfo) == 32, "Rust cuDNN plan metadata ABI mismatch");
static_assert(offsetof(CudnnConvInfo, numerical_notes) == 16 &&
    offsetof(CudnnConvInfo, deterministic) == 24 &&
    offsetof(CudnnConvInfo, fused_epilogue) == 28, "Rust cuDNN metadata offsets mismatch");
