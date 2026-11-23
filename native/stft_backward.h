#pragma once
#include <cstddef>
#include <cstdint>

struct FaStftView2 {
    int64_t origin;
    int64_t strides[2];
    int32_t contiguous;
    int32_t reserved;
};

struct FaStftView4 {
    int64_t origin;
    int64_t strides[4];
    int32_t contiguous;
    int32_t direct_r2c;
};

static_assert(sizeof(FaStftView2) == 32, "FaStftView2 ABI size");
static_assert(offsetof(FaStftView2, strides) == 8, "FaStftView2 strides offset");
static_assert(offsetof(FaStftView2, contiguous) == 24, "FaStftView2 contiguous offset");

static_assert(sizeof(FaStftView4) == 48, "FaStftView4 ABI size");
static_assert(offsetof(FaStftView4, strides) == 8, "FaStftView4 strides offset");
static_assert(offsetof(FaStftView4, contiguous) == 40, "FaStftView4 contiguous offset");
static_assert(offsetof(FaStftView4, direct_r2c) == 44, "FaStftView4 direct_r2c offset");
