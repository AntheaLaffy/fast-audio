#pragma once
#include <stdint.h>

typedef struct {
    int32_t batch, steps, input, hidden, directions;
} GruDims;

#if defined(__cplusplus)
static_assert(sizeof(GruDims) == 20, "Rust GRU ABI must be five i32 values");
#endif
