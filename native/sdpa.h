#pragma once
#include <cstddef>
#include <cstdint>

struct FaSdpaView4 {
    int64_t offset;
    int64_t strides[4];
};

struct FaSdpaDims {
    uint64_t batch;
    uint64_t num_heads;
    uint64_t num_kv_heads;
    uint64_t seq_q;
    uint64_t seq_k;
    uint64_t head_dim;
    float scale;
    int32_t is_causal;
    int32_t operand; // -1 = forward, 0 = Query, 1 = Key, 2 = Value
    int32_t fast_inner; // 1 iff all active views have strides[3] == 1 and head_dim <= 128
    FaSdpaView4 q;
    FaSdpaView4 k;
    FaSdpaView4 v;
    FaSdpaView4 dy_or_o;
    FaSdpaView4 out;
};

static_assert(sizeof(FaSdpaView4) == 40, "FaSdpaView4 ABI size");
static_assert(sizeof(FaSdpaDims) == 264, "FaSdpaDims ABI size");
static_assert(offsetof(FaSdpaDims, scale) == 48, "FaSdpaDims scale offset");
static_assert(offsetof(FaSdpaDims, q) == 64, "FaSdpaDims q offset");
