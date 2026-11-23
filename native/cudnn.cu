#include "cuda_backend.h"
#include "conv2d.h"
#include "cudnn_plan.h"
#include <cudnn.h>
#include <algorithm>
#include <array>
#include <memory>
#include <new>
#include <string>
#include <vector>

static_assert(CUDNN_NUMERICAL_NOTE_TYPE_COUNT <= 64, "Numerical note mask is too small");

extern "C" int fa_cuda_epilogue(Context *, ConvDims, float *, const float *, int);

namespace {
struct Failure { int status; };
void cd(cudnnStatus_t status) { if (status != CUDNN_STATUS_SUCCESS) throw Failure{4000+(int)status}; }
void cu(int status) { if (status) throw Failure{status}; }

struct Descriptor {
    cudnnBackendDescriptor_t ptr = nullptr;
    explicit Descriptor(cudnnBackendDescriptorType_t type) { cd(cudnnBackendCreateDescriptor(type, &ptr)); }
    Descriptor() = default;
    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;
    Descriptor(Descriptor &&other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    Descriptor &operator=(Descriptor &&other) noexcept {
        if (ptr) cudnnBackendDestroyDescriptor(ptr);
        ptr = other.ptr; other.ptr = nullptr; return *this;
    }
    ~Descriptor() { if (ptr) cudnnBackendDestroyDescriptor(ptr); }
};
void set(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name,
         cudnnBackendAttributeType_t type, int64_t count, const void *values) {
    cd(cudnnBackendSetAttribute(d, name, type, count, values));
}
template<class T> void one(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name,
                            cudnnBackendAttributeType_t type, const T &value) { set(d,name,type,1,&value); }
void descriptor(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t name, cudnnBackendDescriptor_t value) {
    one(d,name,CUDNN_TYPE_BACKEND_DESCRIPTOR,value);
}
void finalize(cudnnBackendDescriptor_t d) { cd(cudnnBackendFinalize(d)); }

struct Graph {
    std::vector<Descriptor> nodes;
    cudnnBackendDescriptor_t plan = nullptr;
    size_t workspace = 0;
    int64_t engine_id = -1;
    uint64_t numerical_notes = 0;
    bool fused;
    explicit Graph(bool f) : fused(f) {}
    ~Graph() { while (!nodes.empty()) nodes.pop_back(); }
    cudnnBackendDescriptor_t make(cudnnBackendDescriptorType_t type) {
        nodes.emplace_back(type); return nodes.back().ptr;
    }
    cudnnBackendDescriptor_t tensor(int64_t uid, std::array<int64_t,4> dims, bool virt=false) {
        auto t = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        std::array<int64_t,4> strides{dims[1]*dims[2]*dims[3],dims[2]*dims[3],dims[3],1};
        const auto fp32 = CUDNN_DATA_FLOAT;
        const int64_t alignment = 16;
        one(t,CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,fp32);
        set(t,CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,4,dims.data());
        set(t,CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,4,strides.data());
        one(t,CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,uid);
        one(t,CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,alignment);
        one(t,CUDNN_ATTR_TENSOR_IS_VIRTUAL,CUDNN_TYPE_BOOLEAN,virt);
        finalize(t); return t;
    }
    cudnnBackendDescriptor_t pointwise(cudnnPointwiseMode_t mode, cudnnBackendDescriptor_t x,
        cudnnBackendDescriptor_t y, cudnnBackendDescriptor_t b=nullptr) {
        auto d = make(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        const auto fp32 = CUDNN_DATA_FLOAT;
        one(d,CUDNN_ATTR_POINTWISE_MODE,CUDNN_TYPE_POINTWISE_MODE,mode);
        one(d,CUDNN_ATTR_POINTWISE_MATH_PREC,CUDNN_TYPE_DATA_TYPE,fp32);
        finalize(d);
        auto op = make(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        descriptor(op,CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,d);
        descriptor(op,CUDNN_ATTR_OPERATION_POINTWISE_XDESC,x);
        descriptor(op,CUDNN_ATTR_OPERATION_POINTWISE_YDESC,y);
        if (b) descriptor(op,CUDNN_ATTR_OPERATION_POINTWISE_BDESC,b);
        finalize(op); return op;
    }
};

struct Geometry {
    std::array<int64_t,4> x, w, y;
    std::array<int64_t,2> stride, dilation, padding;
    bool transposed;
};
Geometry geometry(ConvDims s) {
    return {{s.batch,s.in_channels,1,s.input_len},{s.out_channels,s.in_channels/s.groups,1,s.kernel},
        {s.batch,s.out_channels,1,s.output_len},{1,s.stride},{1,s.dilation},{0,s.padding},false};
}
Geometry geometry(Conv2dDims s) {
    return {{s.batch,s.in_channels,s.ih,s.iw},
        s.transposed ? std::array<int64_t,4>{s.in_channels,s.out_channels/s.groups,s.kh,s.kw}
            : std::array<int64_t,4>{s.out_channels,s.in_channels/s.groups,s.kh,s.kw},
        {s.batch,s.out_channels,s.oh,s.ow},{s.sh,s.sw},{s.dh,s.dw},{s.ph,s.pw},s.transposed!=0};
}

std::unique_ptr<Graph> build_graph(cudnnHandle_t handle, const Geometry &s, bool bias, bool relu,
                                   bool fuse, size_t workspace_limit, bool require_deterministic) {
    auto g = std::make_unique<Graph>(fuse);
    const auto x = g->tensor(1,s.x);
    const auto w = g->tensor(2,s.w);
    const auto yd = s.y;
    const auto y = g->tensor(4,yd);
    auto intermediate = (fuse && (bias || relu)) ? g->tensor(10,yd,true) : y;
    auto conv = g->make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    const auto fp32 = CUDNN_DATA_FLOAT;
    const auto mode = CUDNN_CROSS_CORRELATION;
    const int64_t spatial = 2;
    const auto &strides=s.stride; const auto &dilations=s.dilation; const auto &pads=s.padding;
    one(conv,CUDNN_ATTR_CONVOLUTION_COMP_TYPE,CUDNN_TYPE_DATA_TYPE,fp32);
    one(conv,CUDNN_ATTR_CONVOLUTION_CONV_MODE,CUDNN_TYPE_CONVOLUTION_MODE,mode);
    one(conv,CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,CUDNN_TYPE_INT64,spatial);
    set(conv,CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,CUDNN_TYPE_INT64,2,strides.data());
    set(conv,CUDNN_ATTR_CONVOLUTION_DILATIONS,CUDNN_TYPE_INT64,2,dilations.data());
    set(conv,CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,CUDNN_TYPE_INT64,2,pads.data());
    set(conv,CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,CUDNN_TYPE_INT64,2,pads.data());
    finalize(conv);
    auto op = g->make(s.transposed ? CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
        : CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    const float alpha=1.0f, beta=0.0f;
    if (s.transposed) {
        descriptor(op,CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,conv);
        descriptor(op,CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,x);
        descriptor(op,CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,w);
        descriptor(op,CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,intermediate);
        one(op,CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,CUDNN_TYPE_FLOAT,alpha);
        one(op,CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,CUDNN_TYPE_FLOAT,beta);
    } else {
        descriptor(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,conv);
        descriptor(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,x);
        descriptor(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,w);
        descriptor(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,intermediate);
        one(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,CUDNN_TYPE_FLOAT,alpha);
        one(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,CUDNN_TYPE_FLOAT,beta);
    }
    finalize(op);
    std::vector<cudnnBackendDescriptor_t> ops{op};
    if (fuse && bias) {
        auto b = g->tensor(3,{1,s.y[1],1,1});
        auto z = relu ? g->tensor(11,yd,true) : y;
        ops.push_back(g->pointwise(CUDNN_POINTWISE_ADD,intermediate,z,b));
        intermediate = z;
    }
    if (fuse && relu) ops.push_back(g->pointwise(CUDNN_POINTWISE_RELU_FWD,intermediate,y));
    auto graph = g->make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    one(graph,CUDNN_ATTR_OPERATIONGRAPH_HANDLE,CUDNN_TYPE_HANDLE,handle);
    set(graph,CUDNN_ATTR_OPERATIONGRAPH_OPS,CUDNN_TYPE_BACKEND_DESCRIPTOR,ops.size(),ops.data());
    finalize(graph);

    for (const auto heur_mode : {CUDNN_HEUR_MODE_A,CUDNN_HEUR_MODE_FALLBACK}) {
        auto heur = g->make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        descriptor(heur,CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,graph);
        one(heur,CUDNN_ATTR_ENGINEHEUR_MODE,CUDNN_TYPE_HEUR_MODE,heur_mode);
        if (cudnnBackendFinalize(heur) != CUDNN_STATUS_SUCCESS) continue;
        int64_t count=0;
        cd(cudnnBackendGetAttribute(heur,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,0,&count,nullptr));
        count = std::min(count,(int64_t)64);
        std::vector<cudnnBackendDescriptor_t> configs;
        for (int64_t i=0;i<count;++i) configs.push_back(g->make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
        int64_t returned=0;
        cd(cudnnBackendGetAttribute(heur,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,count,&returned,configs.data()));
        returned = std::min(returned,(int64_t)configs.size());
        for (int64_t i=0;i<returned;++i) {
            auto engine = g->make(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            int64_t actual=0;
            cd(cudnnBackendGetAttribute(configs[i],CUDNN_ATTR_ENGINECFG_ENGINE,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&actual,&engine));
            std::array<cudnnBackendNumericalNote_t,CUDNN_NUMERICAL_NOTE_TYPE_COUNT> notes;
            cd(cudnnBackendGetAttribute(engine,CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,CUDNN_TYPE_NUMERICAL_NOTE,notes.size(),&actual,notes.data()));
            if (actual < 0 || actual > (int64_t)notes.size()) throw Failure{4000+(int)CUDNN_STATUS_INTERNAL_ERROR};
            bool strict = true;
            uint64_t mask = 0;
            for (int64_t j=0;j<actual;++j) {
                if (notes[j] < 0 || notes[j] >= CUDNN_NUMERICAL_NOTE_TYPE_COUNT)
                    throw Failure{4000+(int)CUDNN_STATUS_INTERNAL_ERROR};
                mask |= uint64_t{1} << notes[j];
                if (notes[j]==CUDNN_NUMERICAL_NOTE_TENSOR_CORE || notes[j]==CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS
                    || notes[j]==CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION) strict=false;
                if (require_deterministic && notes[j]==CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC) strict=false;
            }
            if (!strict) continue;
            auto plan = g->make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
            one(plan,CUDNN_ATTR_EXECUTION_PLAN_HANDLE,CUDNN_TYPE_HANDLE,handle);
            descriptor(plan,CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,configs[i]);
            if (cudnnBackendFinalize(plan)!=CUDNN_STATUS_SUCCESS) continue;
            int64_t bytes=0;
            cd(cudnnBackendGetAttribute(plan,CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,CUDNN_TYPE_INT64,1,&actual,&bytes));
            if (bytes < 0 || (uint64_t)bytes > workspace_limit) continue;
            cd(cudnnBackendGetAttribute(engine,CUDNN_ATTR_ENGINE_GLOBAL_INDEX,CUDNN_TYPE_INT64,1,&actual,&g->engine_id));
            g->workspace=(size_t)bytes; g->plan=plan; g->numerical_notes=mask;
            return g;
        }
    }
    throw Failure{FA_CUDNN_NO_CONV_PLAN};
}

struct ConvPlan {
    Context *ctx;
    ConvDims spec;
    bool bias, relu;
    cudnnHandle_t handle = nullptr;
    std::unique_ptr<Graph> graph;
    void *workspace = nullptr;
    Descriptor variant;
    std::array<void *,4> bindings{};
    std::string implementation;
    ConvPlan(Context *c, ConvDims s, bool b, bool r) : ctx(c),spec(s),bias(b),relu(r) {}
    ~ConvPlan() {
        cudaSetDevice(ctx->device);
        cudaStreamSynchronize(ctx->stream);
        variant = Descriptor();
        if (workspace) cudaFree(workspace);
        graph.reset();
        if (handle) cudnnDestroy(handle);
    }
};
}

static int create_conv(Context *ctx, ConvDims epilogue, const Geometry &s, int bias, int relu,
                       size_t workspace_limit, int require_deterministic, void **out) noexcept {
    try {
        if (require_deterministic != 0 && require_deterministic != 1)
            return 4000+(int)CUDNN_STATUS_BAD_PARAM;
        cu(activate(ctx));
        auto p = std::make_unique<ConvPlan>(ctx,epilogue,bias!=0,relu!=0);
        cd(cudnnCreate(&p->handle));
        cd(cudnnSetStream(p->handle,ctx->stream));
        try { p->graph = build_graph(p->handle,s,p->bias,p->relu,true,workspace_limit,require_deterministic!=0); }
        catch (const Failure &e) {
            if (e.status != FA_CUDNN_NO_CONV_PLAN && e.status != 4000+(int)CUDNN_STATUS_NOT_SUPPORTED) throw;
            p->graph = build_graph(p->handle,s,p->bias,p->relu,false,workspace_limit,require_deterministic!=0);
        }
        if (p->graph->workspace) cu(rt(cudaMalloc(&p->workspace,p->graph->workspace)));
        p->implementation="cuDNN " + std::to_string(cudnnGetVersion()) + " graph engine=" + std::to_string(p->graph->engine_id)
            + (p->graph->fused ? " fused" : " + CUDA epilogue")
            + ((p->graph->numerical_notes & (uint64_t{1} << CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC))
                ? " nondeterministic" : " deterministic");
        *out = p.release();
        return 0;
    } catch (const Failure &e) {
        return e.status == 4000+(int)CUDNN_STATUS_NOT_SUPPORTED ? FA_CUDNN_NO_CONV_PLAN : e.status;
    }
      catch (const std::bad_alloc &) { return 1002; }
      catch (...) { return 4000+(int)CUDNN_STATUS_INTERNAL_ERROR; }
}

extern "C" int fa_cudnn_conv_run(ConvPlan *p, const float *x, const float *w,
                                   const float *bias, float *out) noexcept {
    try {
        cu(activate(p->ctx));
        const std::array<void *,4> pointers{const_cast<float *>(x),const_cast<float *>(w),out,const_cast<float *>(bias)};
        if (!p->variant.ptr || pointers!=p->bindings) {
            // Finalized variant packs are immutable. Rebind only after older uses
            // complete; stable inference buffers reuse this descriptor without allocation.
            cu(rt(cudaStreamSynchronize(p->ctx->stream)));
            Descriptor variant(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
            const int count = (p->bias && p->graph->fused) ? 4 : 3;
            const int64_t uids[]{1,2,4,3};
            set(variant.ptr,CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,CUDNN_TYPE_VOID_PTR,count,pointers.data());
            set(variant.ptr,CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,CUDNN_TYPE_INT64,count,uids);
            one(variant.ptr,CUDNN_ATTR_VARIANT_PACK_WORKSPACE,CUDNN_TYPE_VOID_PTR,p->workspace);
            finalize(variant.ptr);
            p->variant=std::move(variant); p->bindings=pointers;
        }
        cd(cudnnBackendExecute(p->handle,p->graph->plan,p->variant.ptr));
        if (!p->graph->fused) cu(fa_cuda_epilogue(p->ctx,p->spec,out,bias,p->relu));
        return 0;
    } catch (const Failure &e) { return e.status; }
      catch (const std::bad_alloc &) { return 1002; }
      catch (...) { return 4000+(int)CUDNN_STATUS_INTERNAL_ERROR; }
}
extern "C" void fa_cudnn_conv_info(ConvPlan *p, CudnnConvInfo *info, const char **implementation) noexcept {
    *info={p->graph->workspace,p->graph->engine_id,p->graph->numerical_notes,
        (p->graph->numerical_notes & (uint64_t{1} << CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC)) == 0,
        p->graph->fused && (p->bias || p->relu)};
    *implementation=p->implementation.c_str();
}
extern "C" void fa_cudnn_conv_destroy(ConvPlan *p) noexcept { delete p; }

extern "C" int fa_cudnn_conv_create(Context *ctx, ConvDims s, int bias, int relu,
                                      size_t workspace_limit, int require_deterministic, void **out) noexcept {
    return create_conv(ctx,s,geometry(s),bias,relu,workspace_limit,require_deterministic,out);
}
extern "C" int fa_cudnn_conv2d_create(Context *ctx, Conv2dDims s, int bias, int relu,
                                        size_t workspace_limit, int require_deterministic, void **out) noexcept {
    ConvDims epilogue{s.batch,s.in_channels,s.out_channels,s.ih*s.iw,1,1,0,1,s.groups,s.oh*s.ow};
    return create_conv(ctx,epilogue,geometry(s),bias,relu,workspace_limit,require_deterministic,out);
}

__global__ static void pad_input_2d(const float *x,float *out,int channels,int ih,int iw,int oh,int ow,size_t count) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        const int w=i%ow;const int h=(i/ow)%oh;const size_t image=i/((size_t)oh*ow);
        out[i]=(h<ih && w<iw) ? x[(image*ih+h)*iw+w] : 0.0f;
    }
}
extern "C" int fa_cuda_conv2d_pad(Context *ctx,const float *x,float *out,int channels,
                                   int ih,int iw,int oh,int ow,size_t count) noexcept {
    const int code=activate(ctx);if(code) return code;
    const auto grid=(unsigned)std::min((count+255)/256,(size_t)65535);
    pad_input_2d<<<grid,256,0,ctx->stream>>>(x,out,channels,ih,iw,oh,ow,count);
    return rt(cudaGetLastError());
}
