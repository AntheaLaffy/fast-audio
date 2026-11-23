#include "cuda_backend.h"
#include "conv_backward.h"
#include <algorithm>
#include <climits>

namespace {
__device__ int64_t address(const ConvNativeDims &d, int slot, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const int base = 4 * slot;
    return d.origins[slot]
        + i0 * d.strides[base]
        + i1 * d.strides[base + 1]
        + i2 * d.strides[base + 2]
        + i3 * d.strides[base + 3];
}

__device__ float read_lhs(const ConvNativeDims &d, const float *a, const float *b,
                          int64_t g, uint64_t row, uint64_t r) {
    if (d.op_mode == 0) {
        if (row >= static_cast<uint64_t>(d.cout_per_group) || r >= d.k) return 0.0f;
        const int64_t kw = static_cast<int64_t>(r % static_cast<uint64_t>(d.kernel_hw[1]));
        const uint64_t r1 = r / static_cast<uint64_t>(d.kernel_hw[1]);
        const int64_t kh = static_cast<int64_t>(r1 % static_cast<uint64_t>(d.kernel_hw[0]));
        const int64_t ci = static_cast<int64_t>(r1 / static_cast<uint64_t>(d.kernel_hw[0]));
        const int64_t co = g * d.cout_per_group + static_cast<int64_t>(row);
        return b[address(d, 1, co, ci, kh, kw)];
    }
    if (d.op_mode == 1) {
        if (row >= static_cast<uint64_t>(d.cin_per_group) || r >= d.k) return 0.0f;
        const int64_t kw = static_cast<int64_t>(r % static_cast<uint64_t>(d.kernel_hw[1]));
        const uint64_t r1 = r / static_cast<uint64_t>(d.kernel_hw[1]);
        const int64_t kh = static_cast<int64_t>(r1 % static_cast<uint64_t>(d.kernel_hw[0]));
        const int64_t co_local = static_cast<int64_t>(r1 / static_cast<uint64_t>(d.kernel_hw[0]));
        const int64_t co = g * d.cout_per_group + co_local;
        const int64_t ci = static_cast<int64_t>(row);
        return b[address(d, 1, co, ci, kh, kw)];
    }
    if (row >= static_cast<uint64_t>(d.cout_per_group) || r >= d.k) return 0.0f;
    const int64_t wo = static_cast<int64_t>(r % static_cast<uint64_t>(d.output_hw[1]));
    const uint64_t r1 = r / static_cast<uint64_t>(d.output_hw[1]);
    const int64_t ho = static_cast<int64_t>(r1 % static_cast<uint64_t>(d.output_hw[0]));
    const int64_t n = static_cast<int64_t>(r1 / static_cast<uint64_t>(d.output_hw[0]));
    const int64_t co = g * d.cout_per_group + static_cast<int64_t>(row);
    (void)a;
    return b[address(d, 1, n, co, ho, wo)];
}

__device__ float read_rhs(const ConvNativeDims &d, const float *a, const float *b,
                          int64_t g, uint64_t r, uint64_t col) {
    (void)b;
    if (d.op_mode == 0) {
        if (r >= d.k || col >= d.n) return 0.0f;
        const int64_t kw = static_cast<int64_t>(r % static_cast<uint64_t>(d.kernel_hw[1]));
        const uint64_t r1 = r / static_cast<uint64_t>(d.kernel_hw[1]);
        const int64_t kh = static_cast<int64_t>(r1 % static_cast<uint64_t>(d.kernel_hw[0]));
        const int64_t ci_local = static_cast<int64_t>(r1 / static_cast<uint64_t>(d.kernel_hw[0]));
        const int64_t wo = static_cast<int64_t>(col % static_cast<uint64_t>(d.output_hw[1]));
        const uint64_t c1 = col / static_cast<uint64_t>(d.output_hw[1]);
        const int64_t ho = static_cast<int64_t>(c1 % static_cast<uint64_t>(d.output_hw[0]));
        const int64_t n = static_cast<int64_t>(c1 / static_cast<uint64_t>(d.output_hw[0]));
        const int64_t hi = ho * d.stride_hw[0] + kh * d.dilation_hw[0] - d.padding_hw[0];
        const int64_t wi = wo * d.stride_hw[1] + kw * d.dilation_hw[1] - d.padding_hw[1];
        if (hi < 0 || hi >= d.input_hw[0] || wi < 0 || wi >= d.input_hw[1]) return 0.0f;
        const int64_t ci = g * d.cin_per_group + ci_local;
        return a[address(d, 0, n, ci, hi, wi)];
    }
    if (d.op_mode == 1) {
        if (r >= d.k || col >= d.n) return 0.0f;
        const int64_t kw = static_cast<int64_t>(r % static_cast<uint64_t>(d.kernel_hw[1]));
        const uint64_t r1 = r / static_cast<uint64_t>(d.kernel_hw[1]);
        const int64_t kh = static_cast<int64_t>(r1 % static_cast<uint64_t>(d.kernel_hw[0]));
        const int64_t co_local = static_cast<int64_t>(r1 / static_cast<uint64_t>(d.kernel_hw[0]));
        const int64_t wi = static_cast<int64_t>(col % static_cast<uint64_t>(d.input_hw[1]));
        const uint64_t c1 = col / static_cast<uint64_t>(d.input_hw[1]);
        const int64_t hi = static_cast<int64_t>(c1 % static_cast<uint64_t>(d.input_hw[0]));
        const int64_t n = static_cast<int64_t>(c1 / static_cast<uint64_t>(d.input_hw[0]));
        const int64_t dh_pos = hi + d.padding_hw[0] - kh * d.dilation_hw[0];
        const int64_t dw_pos = wi + d.padding_hw[1] - kw * d.dilation_hw[1];
        if (dh_pos < 0 || dw_pos < 0 || (dh_pos % d.stride_hw[0]) != 0 || (dw_pos % d.stride_hw[1]) != 0)
            return 0.0f;
        const int64_t ho = dh_pos / d.stride_hw[0];
        const int64_t wo = dw_pos / d.stride_hw[1];
        if (ho < 0 || ho >= d.output_hw[0] || wo < 0 || wo >= d.output_hw[1]) return 0.0f;
        const int64_t co = g * d.cout_per_group + co_local;
        return a[address(d, 0, n, co, ho, wo)];
    }
    if (r >= d.k || col >= d.n) return 0.0f;
    const int64_t wo = static_cast<int64_t>(r % static_cast<uint64_t>(d.output_hw[1]));
    const uint64_t r1 = r / static_cast<uint64_t>(d.output_hw[1]);
    const int64_t ho = static_cast<int64_t>(r1 % static_cast<uint64_t>(d.output_hw[0]));
    const int64_t n = static_cast<int64_t>(r1 / static_cast<uint64_t>(d.output_hw[0]));
    const int64_t kw = static_cast<int64_t>(col % static_cast<uint64_t>(d.kernel_hw[1]));
    const uint64_t c1 = col / static_cast<uint64_t>(d.kernel_hw[1]);
    const int64_t kh = static_cast<int64_t>(c1 % static_cast<uint64_t>(d.kernel_hw[0]));
    const int64_t ci_local = static_cast<int64_t>(c1 / static_cast<uint64_t>(d.kernel_hw[0]));
    const int64_t hi = ho * d.stride_hw[0] + kh * d.dilation_hw[0] - d.padding_hw[0];
    const int64_t wi = wo * d.stride_hw[1] + kw * d.dilation_hw[1] - d.padding_hw[1];
    if (hi < 0 || hi >= d.input_hw[0] || wi < 0 || wi >= d.input_hw[1]) return 0.0f;
    const int64_t ci = g * d.cin_per_group + ci_local;
    return a[address(d, 0, n, ci, hi, wi)];
}

__device__ int64_t write_address(const ConvNativeDims &d, int64_t g, uint64_t row, uint64_t col) {
    if (d.op_mode == 0) {
        const int64_t wo = static_cast<int64_t>(col % static_cast<uint64_t>(d.output_hw[1]));
        const uint64_t c1 = col / static_cast<uint64_t>(d.output_hw[1]);
        const int64_t ho = static_cast<int64_t>(c1 % static_cast<uint64_t>(d.output_hw[0]));
        const int64_t n = static_cast<int64_t>(c1 / static_cast<uint64_t>(d.output_hw[0]));
        const int64_t co = g * d.cout_per_group + static_cast<int64_t>(row);
        return address(d, 2, n, co, ho, wo);
    }
    if (d.op_mode == 1) {
        const int64_t wi = static_cast<int64_t>(col % static_cast<uint64_t>(d.input_hw[1]));
        const uint64_t c1 = col / static_cast<uint64_t>(d.input_hw[1]);
        const int64_t hi = static_cast<int64_t>(c1 % static_cast<uint64_t>(d.input_hw[0]));
        const int64_t n = static_cast<int64_t>(c1 / static_cast<uint64_t>(d.input_hw[0]));
        const int64_t ci = g * d.cin_per_group + static_cast<int64_t>(row);
        return address(d, 2, n, ci, hi, wi);
    }
    const int64_t kw = static_cast<int64_t>(col % static_cast<uint64_t>(d.kernel_hw[1]));
    const uint64_t c1 = col / static_cast<uint64_t>(d.kernel_hw[1]);
    const int64_t kh = static_cast<int64_t>(c1 % static_cast<uint64_t>(d.kernel_hw[0]));
    const int64_t ci = static_cast<int64_t>(c1 / static_cast<uint64_t>(d.kernel_hw[0]));
    const int64_t co = g * d.cout_per_group + static_cast<int64_t>(row);
    return address(d, 2, co, ci, kh, kw);
}

__global__ void tiled_conv(ConvNativeDims d, const float *a, const float *b, float *out, int accumulate) {
    __shared__ float aa[16][16], bb[16][16];
    const uint64_t columns = (d.n + 15) / 16;
    const uint64_t group_tiles = ((d.m + 15) / 16) * columns;
    const uint64_t total_tiles = static_cast<uint64_t>(d.groups) * group_tiles;
    for (uint64_t t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        const int64_t g = static_cast<int64_t>(t / group_tiles);
        const uint64_t tile = t % group_tiles;
        const uint64_t row = (tile / columns) * 16 + threadIdx.y;
        const uint64_t col = (tile % columns) * 16 + threadIdx.x;
        float sum = 0.0f;
        for (uint64_t start = 0; start < d.k; start += 16) {
            aa[threadIdx.y][threadIdx.x] = read_lhs(d, a, b, g, row, start + threadIdx.x);
            bb[threadIdx.y][threadIdx.x] = read_rhs(d, a, b, g, start + threadIdx.y, col);
            __syncthreads();
            for (int i = 0; i < 16; ++i)
                sum = __fadd_rn(sum, __fmul_rn(aa[threadIdx.y][i], bb[i][threadIdx.x]));
            __syncthreads();
        }
        if (row < d.m && col < d.n) {
            const int64_t target = write_address(d, g, row, col);
            out[target] = accumulate ? __fadd_rn(out[target], sum) : sum;
        }
    }
}

unsigned grid(uint64_t groups) {
    return static_cast<unsigned>(std::min<uint64_t>(groups, 65535));
}
} // namespace

extern "C" int fa_cuda_conv_shared(Context *ctx, int kind, size_t *bytes) {
    if (!ctx || !bytes || (kind != 3 && kind != 4)) return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    *bytes = 0;
    if (kind == 4) {
        cudaFuncAttributes attr{};
        code = rt(cudaFuncGetAttributes(&attr, tiled_conv));
        if (!code) *bytes = attr.sharedSizeBytes;
    }
    return code;
}

extern "C" int fa_cuda_conv_layout(Context *ctx, ConvNativeDims d, const float *a, const float *b,
                                   float *out, int accumulate) {
    if (!ctx || !a || !b || !out || (accumulate != 0 && accumulate != 1) ||
        (d.kind != 3 && d.kind != 4) || d.op_mode < 0 || d.op_mode > 2 ||
        (d.swapped != 0 && d.swapped != 1) || d.batch <= 0 || d.groups <= 0 ||
        d.cin_per_group <= 0 || d.cout_per_group <= 0 || !d.m || !d.n || !d.k)
        return rt(cudaErrorInvalidValue);
    int code = activate(ctx);
    if (code) return code;
    if (d.kind == 3) {
        const int64_t area = static_cast<int64_t>(d.input_hw[0]) * d.input_hw[1];
        int64_t m_blas = (d.op_mode == 1) ? d.cin_per_group : d.cout_per_group;
        int64_t n_blas = (d.op_mode == 2) ? d.cin_per_group : area;
        int64_t k_blas = (d.op_mode == 0) ? d.cin_per_group : (d.op_mode == 1 ? d.cout_per_group : area);
        if (d.swapped) std::swap(m_blas, n_blas);
        if (m_blas <= 0 || n_blas <= 0 || k_blas <= 0 ||
            m_blas > INT_MAX || n_blas > INT_MAX || k_blas > INT_MAX ||
            d.transposes[0] < 0 || d.transposes[0] > 1 || d.transposes[1] < 0 || d.transposes[1] > 1 ||
            d.leading[0] < (d.transposes[0] ? m_blas : k_blas) ||
            d.leading[1] < (d.transposes[1] ? k_blas : n_blas) ||
            d.leading[2] < n_blas)
            return rt(cudaErrorInvalidValue);
        const float one = 1.0f;
        const auto ta = d.transposes[0] ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto tb = d.transposes[1] ? CUBLAS_OP_T : CUBLAS_OP_N;
        for (int n = 0; n < d.batch; ++n) {
            for (int g = 0; g < d.groups; ++g) {
                int64_t o0 = 0, o1 = 0, o2 = 0;
                if (d.op_mode == 0) {
                    o0 = d.origins[0] + n * d.strides[0] + g * static_cast<int64_t>(d.cin_per_group) * d.strides[1];
                    o1 = d.origins[1] + g * static_cast<int64_t>(d.cout_per_group) * d.strides[4];
                    o2 = d.origins[2] + n * d.strides[8] + g * static_cast<int64_t>(d.cout_per_group) * d.strides[9];
                } else if (d.op_mode == 1) {
                    o0 = d.origins[0] + n * d.strides[0] + g * static_cast<int64_t>(d.cout_per_group) * d.strides[1];
                    o1 = d.origins[1] + g * static_cast<int64_t>(d.cout_per_group) * d.strides[4];
                    o2 = d.origins[2] + n * d.strides[8] + g * static_cast<int64_t>(d.cin_per_group) * d.strides[9];
                } else {
                    o0 = d.origins[0] + n * d.strides[0] + g * static_cast<int64_t>(d.cin_per_group) * d.strides[1];
                    o1 = d.origins[1] + n * d.strides[4] + g * static_cast<int64_t>(d.cout_per_group) * d.strides[5];
                    o2 = d.origins[2] + g * static_cast<int64_t>(d.cout_per_group) * d.strides[8];
                }
                const float *lhs = d.swapped ? (a + o0) : (b + o1);
                const float *rhs = d.swapped ? (b + o1) : (a + o0);
                const float beta = (accumulate || (d.op_mode == 2 && n > 0)) ? 1.0f : 0.0f;
                const auto status = cublasSgemm(
                    ctx->blas, tb, ta,
                    static_cast<int>(n_blas), static_cast<int>(m_blas), static_cast<int>(k_blas),
                    &one, rhs, d.leading[1], lhs, d.leading[0], &beta, out + o2, d.leading[2]);
                if (status != CUBLAS_STATUS_SUCCESS) return 2000 + static_cast<int>(status);
            }
        }
        return 0;
    }
    const uint64_t group_tiles = ((d.m + 15) / 16) * ((d.n + 15) / 16);
    const uint64_t total_tiles = static_cast<uint64_t>(d.groups) * group_tiles;
    tiled_conv<<<grid(total_tiles), dim3(16, 16), 0, ctx->stream>>>(d, a, b, out, accumulate);
    return rt(cudaGetLastError());
}
