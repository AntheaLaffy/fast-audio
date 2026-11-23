#pragma once
#include <cstddef>
#include <cstdint>

struct ConvNativeDims {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    int64_t origins[3];
    int64_t strides[12];
    int32_t batch;
    int32_t groups;
    int32_t cin_per_group;
    int32_t cout_per_group;
    int32_t input_hw[2];
    int32_t kernel_hw[2];
    int32_t output_hw[2];
    int32_t stride_hw[2];
    int32_t padding_hw[2];
    int32_t dilation_hw[2];
    int32_t leading[3];
    int32_t transposes[2];
    int32_t swapped;
    int32_t kind;
    int32_t op_mode;
};

static_assert(sizeof(ConvNativeDims) == 240, "ConvNativeDims ABI size mismatch");
static_assert(offsetof(ConvNativeDims, origins) == 24, "ConvNativeDims origins offset mismatch");
static_assert(offsetof(ConvNativeDims, strides) == 48, "ConvNativeDims strides offset mismatch");
static_assert(offsetof(ConvNativeDims, batch) == 144, "ConvNativeDims batch offset mismatch");
static_assert(offsetof(ConvNativeDims, leading) == 208, "ConvNativeDims leading offset mismatch");
static_assert(offsetof(ConvNativeDims, kind) == 232, "ConvNativeDims kind offset mismatch");
