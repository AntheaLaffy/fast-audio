#pragma once
#include <stdint.h>

typedef struct ConvDims {
    int32_t batch, in_channels, out_channels, input_len, kernel;
    int32_t stride, padding, dilation, groups, output_len;
} ConvDims;

#if defined(__cplusplus)
static_assert(sizeof(ConvDims) == 40, "ConvDims ABI must remain ten i32 fields");
#endif
