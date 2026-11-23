#include "cuda_backend.h"
#include "gru.h"
#include <cudnn.h>
#include <cstdio>
#include <memory>
#include <new>
#include <vector>
#include <algorithm>

static int status(cudnnStatus_t value) { return value == CUDNN_STATUS_SUCCESS ? 0 : 4000+(int)value; }
#define TRY_RNN(expr) do { int code=(expr); if (code) { std::fprintf(stderr,"GRU: %s failed (%d)\n",#expr,code); return code; } } while (0)

struct GruPlan {
    Context *ctx = nullptr;
    cudnnHandle_t handle = nullptr;
    cudnnRNNDescriptor_t rnn = nullptr;
    cudnnDropoutDescriptor_t dropout = nullptr;
    cudnnRNNDataDescriptor_t x = nullptr, y = nullptr;
    cudnnTensorDescriptor_t state = nullptr;
    void *weights = nullptr, *workspace = nullptr;
    float *input_buffer = nullptr, *output_buffer = nullptr;
    GruDims dims{};
    bool sequence_major = false;
    size_t weight_bytes = 0, workspace_bytes = 0;
    std::vector<int> lengths;
    ~GruPlan() {
        if (ctx) { cudaSetDevice(ctx->device); cudaStreamSynchronize(ctx->stream); }
        if (weights) cudaFree(weights);
        if (workspace) cudaFree(workspace);
        if (input_buffer) cudaFree(input_buffer);
        if (output_buffer) cudaFree(output_buffer);
        if (state) cudnnDestroyTensorDescriptor(state);
        if (x) cudnnDestroyRNNDataDescriptor(x);
        if (y) cudnnDestroyRNNDataDescriptor(y);
        if (rnn) cudnnDestroyRNNDescriptor(rnn);
        if (dropout) cudnnDestroyDropoutDescriptor(dropout);
        if (handle) cudnnDestroy(handle);
    }
};

struct Descriptor {
    cudnnTensorDescriptor_t value = nullptr;
    ~Descriptor() { if (value) cudnnDestroyTensorDescriptor(value); }
};

static int copy_parameter(Context *ctx, cudnnTensorDescriptor_t descriptor, void *device,
                          const float *host, size_t count) {
    if (!device) return 4003;
    cudnnDataType_t type;
    int rank, dims[8], strides[8];
    TRY_RNN(status(cudnnGetTensorNdDescriptor(descriptor,8,&type,&rank,dims,strides)));
    if (type != CUDNN_DATA_FLOAT || rank < 1 || rank > 8) return 4003;
    size_t size = 1;
    for (int axis=rank-1; axis>=0; --axis) {
        if (dims[axis] < 1 || (dims[axis] > 1 && (size_t)strides[axis] != size)) return 4009;
        size *= dims[axis];
    }
    if (size != count) return 4003;
    return rt(cudaMemcpyAsync(device,host,count*sizeof(float),cudaMemcpyHostToDevice,ctx->stream));
}

extern "C" int fa_cudnn_gru_create(Context *ctx, GruDims s, const float *wi,
                                     const float *wh, const float *bi, const float *bh,
                                     int automatic, int sequence_major, void **out, size_t *workspace, size_t *parameters,
                                     int *selected_small) {
    try {
        TRY_RNN(activate(ctx));
        auto p = std::make_unique<GruPlan>();
        p->ctx = ctx;
        p->dims = s; p->sequence_major = sequence_major != 0;
        p->lengths.assign(s.batch,s.steps);
        TRY_RNN(status(cudnnCreate(&p->handle)));
        TRY_RNN(status(cudnnSetStream(p->handle,ctx->stream)));
        TRY_RNN(status(cudnnCreateDropoutDescriptor(&p->dropout)));
        // Dropout is disabled in this single-layer inference plan; no RNG state is needed.
        TRY_RNN(status(cudnnSetDropoutDescriptor(p->dropout,p->handle,0.0f,nullptr,0,0)));
        TRY_RNN(status(cudnnCreateRNNDescriptor(&p->rnn)));
        cudaDeviceProp properties;
        TRY_RNN(rt(cudaGetDeviceProperties(&properties,ctx->device)));
        const bool small = automatic && properties.major>=6 && s.hidden<=192;
        const auto algorithm = small ? CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H : CUDNN_RNN_ALGO_STANDARD;
        TRY_RNN(status(cudnnSetRNNDescriptor_v8(p->rnn,algorithm,CUDNN_GRU,
            CUDNN_RNN_DOUBLE_BIAS,s.directions==2 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
            CUDNN_LINEAR_INPUT,CUDNN_DATA_FLOAT,CUDNN_DATA_FLOAT,CUDNN_FMA_MATH,
            s.input,s.hidden,s.hidden,1,p->dropout,CUDNN_RNN_PADDED_IO_ENABLED)));
        TRY_RNN(status(cudnnCreateRNNDataDescriptor(&p->x)));
        TRY_RNN(status(cudnnCreateRNNDataDescriptor(&p->y)));
        float padding = 0.0f;
        const auto layout = sequence_major ? CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED : CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
        TRY_RNN(status(cudnnSetRNNDataDescriptor(p->x,CUDNN_DATA_FLOAT,layout,
            s.steps,s.batch,s.input,p->lengths.data(),&padding)));
        TRY_RNN(status(cudnnSetRNNDataDescriptor(p->y,CUDNN_DATA_FLOAT,layout,
            s.steps,s.batch,s.hidden*s.directions,p->lengths.data(),&padding)));
        TRY_RNN(status(cudnnCreateTensorDescriptor(&p->state)));
        int dims[3] = {s.directions,s.batch,s.hidden};
        int strides[3] = {s.batch*s.hidden,s.hidden,1};
        TRY_RNN(status(cudnnSetTensorNdDescriptor(p->state,CUDNN_DATA_FLOAT,3,dims,strides)));
        TRY_RNN(status(cudnnGetRNNWeightSpaceSize(p->handle,p->rnn,&p->weight_bytes)));
        TRY_RNN(rt(cudaMalloc(&p->weights,p->weight_bytes)));
        TRY_RNN(rt(cudaMemsetAsync(p->weights,0,p->weight_bytes,ctx->stream)));
        Descriptor matrix, bias;
        TRY_RNN(status(cudnnCreateTensorDescriptor(&matrix.value)));
        TRY_RNN(status(cudnnCreateTensorDescriptor(&bias.value)));
        for (int direction=0; direction<s.directions; ++direction) {
            for (int linear=0; linear<6; ++linear) {
                void *matrix_address = nullptr, *bias_address = nullptr;
                TRY_RNN(status(cudnnGetRNNWeightParams(p->handle,p->rnn,direction,
                    p->weight_bytes,p->weights,linear,matrix.value,&matrix_address,bias.value,&bias_address)));
                const int width = linear<3 ? s.input : s.hidden;
                const float *source_weights = linear<3 ? wi : wh;
                const float *source_bias = linear<3 ? bi : bh;
                const size_t gate = (size_t)direction*3+linear%3;
                TRY_RNN(copy_parameter(ctx,matrix.value,matrix_address,source_weights+gate*s.hidden*width,(size_t)s.hidden*width));
                TRY_RNN(copy_parameter(ctx,bias.value,bias_address,source_bias+gate*s.hidden,s.hidden));
            }
        }
        size_t reserve = 0;
        TRY_RNN(status(cudnnGetRNNTempSpaceSizes(p->handle,p->rnn,CUDNN_FWD_MODE_INFERENCE,p->x,
            &p->workspace_bytes,&reserve)));
        if (p->workspace_bytes) TRY_RNN(rt(cudaMalloc(&p->workspace,p->workspace_bytes)));
        size_t layout_bytes = 0;
        if (sequence_major && s.batch>1) {
            const size_t input_bytes = (size_t)s.batch*s.steps*s.input*sizeof(float);
            const size_t output_bytes = (size_t)s.batch*s.steps*s.hidden*s.directions*sizeof(float);
            TRY_RNN(rt(cudaMalloc(&p->input_buffer,input_bytes)));
            TRY_RNN(rt(cudaMalloc(&p->output_buffer,output_bytes)));
            layout_bytes = input_bytes+output_bytes;
        }
        TRY_RNN(rt(cudaStreamSynchronize(ctx->stream)));
        *workspace = p->workspace_bytes+layout_bytes;
        *parameters = p->weight_bytes;
        *selected_small = small;
        *out = p.release();
        return 0;
    } catch (const std::bad_alloc &) { return 1002; }
      catch (...) { return 4199; }
}

template<bool to_sequence_major>
__global__ static void transpose_batch_time(const float *input, float *output,
                                           int batch, int steps, int width, size_t count) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<count; i+=blockDim.x*gridDim.x) {
        const int feature = i%width;
        const size_t row = i/width;
        const int time = row%steps;
        const size_t sequence_index = ((size_t)time*batch+row/steps)*width+feature;
        if constexpr (to_sequence_major) output[sequence_index] = input[i];
        else output[i] = input[sequence_index];
    }
}

extern "C" int fa_cudnn_gru_run(GruPlan *p, const float *x, const float *initial,
                                  float *out, float *final_state) {
    TRY_RNN(activate(p->ctx));
    const auto s = p->dims;
    const bool transpose = p->sequence_major && s.batch>1;
    if (transpose) {
        const size_t count = (size_t)s.batch*s.steps*s.input;
        const auto grid = (unsigned)std::min((count+255)/256,(size_t)65535);
        transpose_batch_time<true><<<grid,256,0,p->ctx->stream>>>(x,p->input_buffer,s.batch,s.steps,s.input,count);
        TRY_RNN(rt(cudaGetLastError()));
    }
    TRY_RNN(status(cudnnRNNForward(p->handle,p->rnn,CUDNN_FWD_MODE_INFERENCE,nullptr,
        p->x,transpose ? p->input_buffer : x,p->y,transpose ? p->output_buffer : out,
        p->state,initial,final_state,nullptr,nullptr,nullptr,
        p->weight_bytes,p->weights,p->workspace_bytes,p->workspace,0,nullptr)));
    if (transpose) {
        const int width = s.hidden*s.directions;
        const size_t count = (size_t)s.batch*s.steps*width;
        const auto grid = (unsigned)std::min((count+255)/256,(size_t)65535);
        transpose_batch_time<false><<<grid,256,0,p->ctx->stream>>>(p->output_buffer,out,s.batch,s.steps,width,count);
        TRY_RNN(rt(cudaGetLastError()));
    }
    return 0;
}

extern "C" void fa_cudnn_gru_destroy(GruPlan *p) { delete p; }
