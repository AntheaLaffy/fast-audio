#pragma once
#include <stdint.h>

typedef struct {
    int32_t batch, in_channels, out_channels;
    int32_t ih, iw, kh, kw, sh, sw, ph, pw, dh, dw, groups, oh, ow;
    int32_t transposed, oph, opw;
} Conv2dDims;
#if defined(__cplusplus)
static_assert(sizeof(Conv2dDims) == 76, "Rust 2D convolution ABI must contain nineteen i32 values");
#endif
