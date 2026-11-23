#pragma once
#include <cstddef>
#include <cstdint>

struct FaPadAxis {
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t pad_before;
    uint64_t pad_after;
    int64_t in_stride;
    int64_t out_stride;
};

struct FaPadDims {
    uint64_t total_in;
    uint64_t total_out;
    int64_t in_offset;
    int64_t out_offset;
    int32_t rank;
    int32_t mode; // 0 = Constant, 1 = Reflect, 2 = Replicate
    float const_val;
    int32_t reserved;
};

struct FaCatAxis {
    uint64_t dim_out;
    int64_t stride_a;
    int64_t stride_b;
    int64_t stride_out;
};

struct FaCatDims {
    uint64_t total;
    uint64_t split;
    int64_t offset_a;
    int64_t offset_b;
    int64_t offset_out;
    int32_t rank;
    int32_t axis;
    int32_t operand; // -1 = forward, 0 = Left bwd, 1 = Right bwd
    int32_t reserved;
};

static_assert(sizeof(FaPadAxis) == 48, "FaPadAxis ABI size");
static_assert(sizeof(FaPadDims) == 48, "FaPadDims ABI size");
static_assert(sizeof(FaCatAxis) == 32, "FaCatAxis ABI size");
static_assert(sizeof(FaCatDims) == 56, "FaCatDims ABI size");
