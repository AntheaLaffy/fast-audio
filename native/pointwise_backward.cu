#include "cuda_backend.h"
#include "pointwise_backward.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <new>

namespace {
struct Backward;
using Launcher = void (*)(const Backward *,const float *,const float *,const float *,float *,int);
struct Backward {
    Context *ctx;
    PointwiseBackwardDims dims;
    PointwiseBackwardGroup *groups = nullptr;
    PointwiseBackwardAxis *reduced = nullptr;
    Launcher launch = nullptr;
    size_t shared = 0;
    Backward(Context *c,PointwiseBackwardDims d):ctx(c),dims(d) {}
    ~Backward() {
        cudaSetDevice(ctx->device);cudaStreamSynchronize(ctx->stream);
        if (groups) cudaFree(groups);
        if (reduced) cudaFree(reduced);
    }
};
__device__ __forceinline__ bool is_pow2(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }
__device__ __forceinline__ unsigned log2_pow2(uint64_t v) { return static_cast<unsigned>(__ffsll(static_cast<long long>(v)) - 1); }

__device__ float neg(float x) { return __uint_as_float(__float_as_uint(x)^0x80000000u); }
__device__ float sigmoid(float x) {
    if (x>=0.0f) return __fdiv_rn(1.0f,__fadd_rn(1.0f,expf(-x)));
    const float e=expf(x);return __fdiv_rn(e,__fadd_rn(1.0f,e));
}
__device__ float dsigmoid(float x) {
    const float q=expf(-fabsf(x)),t=__fadd_rn(1.0f,q);
    return __fdiv_rn(__fdiv_rn(q,t),t);
}
__device__ float dtanh(float x) {
    const float q=expf(-fabsf(x));
    const float u=__fdiv_rn(__fmul_rn(2.0f,q),__fadd_rn(1.0f,__fmul_rn(q,q)));
    return __fmul_rn(u,u);
}
template<int OP> __device__ float unary(float d,float x) {
    if constexpr (OP==0) return neg(d);
    else if constexpr (OP==1) return isnan(x) ? __uint_as_float(0x7fc00000u) : x>0 ? d : x<0 ? neg(d) : 0.0f;
    else if constexpr (OP==2) return __fmul_rn(__fmul_rn(2.0f,x),d);
    else if constexpr (OP==3) return neg(__fdiv_rn(__fdiv_rn(d,x),x));
    else if constexpr (OP==4) return __fdiv_rn(d,__fmul_rn(2.0f,sqrtf(x)));
    else if constexpr (OP==5) return __fmul_rn(expf(x),d);
    else if constexpr (OP==6) return __fdiv_rn(d,x);
    else if constexpr (OP==7) return isnan(x) ? __uint_as_float(0x7fc00000u) : x>0 ? d : 0.0f;
    else if constexpr (OP==8) return __fmul_rn(dsigmoid(x),d);
    else if constexpr (OP==9) return __fmul_rn(dtanh(x),d);
    else if constexpr (OP==10) {
        const float factor=x==INFINITY ? 1.0f : x==-INFINITY ? -0.0f : __fadd_rn(sigmoid(x),__fmul_rn(x,dsigmoid(x)));
        return __fmul_rn(factor,d);
    } else return __fmul_rn(sigmoid(x),d);
}
template<int OP,int SIDE> __device__ float binary(float d,float a,float b) {
    if constexpr (OP==0 || (OP==1 && SIDE==0)) return d;
    else if constexpr (OP==1) return neg(d);
    else if constexpr (OP==2 && SIDE==0) return __fmul_rn(d,b);
    else if constexpr (OP==2) return __fmul_rn(d,a);
    else if constexpr (SIDE==0) return __fdiv_rn(d,b);
    else return neg(__fmul_rn(__fdiv_rn(d,b),__fdiv_rn(a,b)));
}
template<int OP,bool BINARY,int SIDE> __host__ __device__ constexpr bool needs_left() {
    if constexpr (!BINARY) return OP!=0;
    else return (OP==2 || OP==3) && SIDE==1;
}
template<int OP,bool BINARY,int SIDE> __host__ __device__ constexpr bool needs_right() {
    return BINARY && ((OP==2 && SIDE==0) || OP==3);
}
__device__ int64_t group_offsets(PointwiseBackwardDims d,const PointwiseBackwardGroup *axes,uint64_t flat,int64_t *reads) {
    for (int j=0;j<3;++j) reads[j]=d.origins[j];
    int64_t out=d.origins[3];
    for (int a=d.rank-1;a>0;--a) {
        const auto c=static_cast<int64_t>(flat%axes[a].dim);flat/=axes[a].dim;
        for (int j=0;j<3;++j) reads[j]+=c*axes[a].reads[j];out+=c*axes[a].output;
    }
    if (d.rank) {for (int j=0;j<3;++j) reads[j]+=static_cast<int64_t>(flat)*axes[0].reads[j];out+=static_cast<int64_t>(flat)*axes[0].output;}
    return out;
}
template<int OP,bool BINARY,int SIDE>
__device__ __forceinline__ float eval_at(const int64_t *address,
    const float *upstream,const float *left,const float *right) {
    const float dy=upstream[address[0]];
    float a=0.0f,b=0.0f;
    if constexpr (needs_left<OP,BINARY,SIDE>()) a=left[address[1]];
    if constexpr (needs_right<OP,BINARY,SIDE>()) b=right[address[2]];
    if constexpr (BINARY) return binary<OP,SIDE>(dy,a,b);
    else return unary<OP>(dy,a);
}
template<int OP,bool BINARY,int SIDE>
__device__ float contribution(PointwiseBackwardDims d,const PointwiseBackwardAxis *axes,const int64_t *base,uint64_t r,
    const float *upstream,const float *left,const float *right) {
    int64_t address[3]={base[0],base[1],base[2]};
    for (int a=d.reduction_rank-1;a>0;--a) {
        const auto c=static_cast<int64_t>(r%axes[a].dim);r/=axes[a].dim;
        for (int j=0;j<3;++j) address[j]+=c*axes[a].reads[j];
    }
    if (d.reduction_rank) for (int j=0;j<3;++j) address[j]+=static_cast<int64_t>(r)*axes[0].reads[j];
    return eval_at<OP,BINARY,SIDE>(address,upstream,left,right);
}
__device__ __forceinline__ void store(float *destination,int64_t target,float value,int accumulate) {
    destination[target]=accumulate ? __fadd_rn(destination[target],value) : value;
}
template<int OP,bool BINARY,int SIDE>
__global__ void elementwise(PointwiseBackwardDims d,const PointwiseBackwardGroup *axes,
    const float *dy,const float *left,const float *right,float *out,int accumulate) {
    const uint64_t stride=static_cast<uint64_t>(blockDim.x)*gridDim.x;
    const uint64_t start=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (d.rank<=1) {
        const int64_t r0=(d.rank==1)?axes[0].reads[0]:0;
        const int64_t r1=(d.rank==1)?axes[0].reads[1]:0;
        const int64_t r2=(d.rank==1)?axes[0].reads[2]:0;
        const int64_t ro=(d.rank==1)?axes[0].output:0;
        for (uint64_t g=start;g<d.groups;g+=stride) {
            const int64_t gi=static_cast<int64_t>(g);
            const int64_t target=d.origins[3]+gi*ro;
            float value=0.0f;
            if (d.count) {
                int64_t addr[3]={d.origins[0]+gi*r0,d.origins[1]+gi*r1,d.origins[2]+gi*r2};
                value=eval_at<OP,BINARY,SIDE>(addr,dy,left,right);
            }
            store(out,target,value,accumulate);
        }
        return;
    }
    if (d.rank==2 && d.groups<=0xffffffffULL) {
        const uint32_t dim1=static_cast<uint32_t>(axes[1].dim);
        const bool p2=is_pow2(dim1);
        const uint32_t mask=dim1-1;
        const unsigned shift=p2?log2_pow2(dim1):0;
        const auto a0=axes[0], a1=axes[1];
        for (uint64_t g=start;g<d.groups;g+=stride) {
            const uint32_t gi=static_cast<uint32_t>(g);
            const int64_t c1=static_cast<int64_t>(p2?(gi&mask):(gi%dim1));
            const int64_t c0=static_cast<int64_t>(p2?(gi>>shift):(gi/dim1));
            const int64_t target=d.origins[3]+c0*a0.output+c1*a1.output;
            float value=0.0f;
            if (d.count) {
                int64_t addr[3]={
                    d.origins[0]+c0*a0.reads[0]+c1*a1.reads[0],
                    d.origins[1]+c0*a0.reads[1]+c1*a1.reads[1],
                    d.origins[2]+c0*a0.reads[2]+c1*a1.reads[2],
                };
                value=eval_at<OP,BINARY,SIDE>(addr,dy,left,right);
            }
            store(out,target,value,accumulate);
        }
        return;
    }
    for (uint64_t g=start;g<d.groups;g+=stride) {
        int64_t base[3];const auto target=group_offsets(d,axes,g,base);
        const float value=d.count ? eval_at<OP,BINARY,SIDE>(base,dy,left,right) : 0.0f;
        store(out,target,value,accumulate);
    }
}
template<int OP,bool BINARY,int SIDE>
__global__ void small_reduce(PointwiseBackwardDims d,const PointwiseBackwardGroup *groups,const PointwiseBackwardAxis *axes,
    const float *dy,const float *left,const float *right,float *out,int accumulate) {
    const uint64_t stride=static_cast<uint64_t>(blockDim.x)*gridDim.x;
    const uint64_t start=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const bool linear_group=(d.rank<=1);
    const int64_t gr0=(d.rank==1)?groups[0].reads[0]:0;
    const int64_t gr1=(d.rank==1)?groups[0].reads[1]:0;
    const int64_t gr2=(d.rank==1)?groups[0].reads[2]:0;
    const int64_t gro=(d.rank==1)?groups[0].output:0;
    const bool linear_red=(d.reduction_rank==1);
    const int64_t rr0=linear_red?axes[0].reads[0]:0;
    const int64_t rr1=linear_red?axes[0].reads[1]:0;
    const int64_t rr2=linear_red?axes[0].reads[2]:0;
    for (uint64_t g=start;g<d.groups;g+=stride) {
        int64_t base[3];
        int64_t target;
        if (linear_group) {
            const int64_t gi=static_cast<int64_t>(g);
            base[0]=d.origins[0]+gi*gr0;
            base[1]=d.origins[1]+gi*gr1;
            base[2]=d.origins[2]+gi*gr2;
            target=d.origins[3]+gi*gro;
        } else {
            target=group_offsets(d,groups,g,base);
        }
        float t[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        if (linear_red) {
            for (uint64_t r=0;r<d.count;++r) {
                const int64_t ri=static_cast<int64_t>(r);
                int64_t addr[3]={base[0]+ri*rr0,base[1]+ri*rr1,base[2]+ri*rr2};
                t[r]=__fadd_rn(0.0f,eval_at<OP,BINARY,SIDE>(addr,dy,left,right));
            }
        } else {
            for (uint64_t r=0;r<d.count;++r)
                t[r]=__fadd_rn(0.0f,contribution<OP,BINARY,SIDE>(d,axes,base,r,dy,left,right));
        }
        for (int i=0;i<4;++i) t[i]=__fadd_rn(t[i],t[i+4]);
        for (int i=0;i<2;++i) t[i]=__fadd_rn(t[i],t[i+2]);
        store(out,target,__fadd_rn(t[0],t[1]),accumulate);
    }
}
template<int OP,bool BINARY,int SIDE>
__global__ void warp(PointwiseBackwardDims d,const PointwiseBackwardGroup *groups,const PointwiseBackwardAxis *axes,
    const float *dy,const float *left,const float *right,float *out,int accumulate) {
    const unsigned lane=threadIdx.x&31;
    const bool linear_red=(d.reduction_rank==1);
    const int64_t rr0=linear_red?axes[0].reads[0]:0;
    const int64_t rr1=linear_red?axes[0].reads[1]:0;
    const int64_t rr2=linear_red?axes[0].reads[2]:0;
    for (uint64_t g=(static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;g<d.groups;g+=static_cast<uint64_t>(gridDim.x)*(blockDim.x/32)) {
        int64_t base[3];const auto target=group_offsets(d,groups,g,base);float sum=0.0f;
        if (linear_red) {
            for (uint64_t r=lane;r<d.count;r+=32) {
                const int64_t ri=static_cast<int64_t>(r);
                int64_t addr[3]={base[0]+ri*rr0,base[1]+ri*rr1,base[2]+ri*rr2};
                sum=__fadd_rn(sum,eval_at<OP,BINARY,SIDE>(addr,dy,left,right));
            }
        } else {
            for (uint64_t r=lane;r<d.count;r+=32) sum=__fadd_rn(sum,contribution<OP,BINARY,SIDE>(d,axes,base,r,dy,left,right));
        }
        // Every lane in each active group executes each shuffle with the same mask.
        for (unsigned step=16;step;step/=2) sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,step));
        if (!lane) store(out,target,sum,accumulate);
    }
}
template<int OP,bool BINARY,int SIDE>
__global__ void block(PointwiseBackwardDims d,const PointwiseBackwardGroup *groups,const PointwiseBackwardAxis *axes,
    const float *dy,const float *left,const float *right,float *out,int accumulate) {
    __shared__ float partial[256];
    const bool linear_group=(d.rank<=1);
    const int64_t gr0=(d.rank==1)?groups[0].reads[0]:0;
    const int64_t gr1=(d.rank==1)?groups[0].reads[1]:0;
    const int64_t gr2=(d.rank==1)?groups[0].reads[2]:0;
    const int64_t gro=(d.rank==1)?groups[0].output:0;
    const bool linear_red=(d.reduction_rank==1);
    const bool two_red=(d.reduction_rank==2 && d.count<=0xffffffffULL);
    const int64_t a0r0=(d.reduction_rank>=1)?axes[0].reads[0]:0;
    const int64_t a0r1=(d.reduction_rank>=1)?axes[0].reads[1]:0;
    const int64_t a0r2=(d.reduction_rank>=1)?axes[0].reads[2]:0;
    const int64_t a1r0=two_red?axes[1].reads[0]:0;
    const int64_t a1r1=two_red?axes[1].reads[1]:0;
    const int64_t a1r2=two_red?axes[1].reads[2]:0;
    const uint32_t dim1=two_red?static_cast<uint32_t>(axes[1].dim):1;
    const bool p2=two_red && is_pow2(dim1);
    const uint32_t mask=dim1-1;
    const unsigned shift=p2?log2_pow2(dim1):0;
    for (uint64_t g=blockIdx.x;g<d.groups;g+=gridDim.x) {
        int64_t base[3];
        int64_t target;
        if (linear_group) {
            const int64_t gi=static_cast<int64_t>(g);
            base[0]=d.origins[0]+gi*gr0;
            base[1]=d.origins[1]+gi*gr1;
            base[2]=d.origins[2]+gi*gr2;
            target=d.origins[3]+gi*gro;
        } else {
            target=group_offsets(d,groups,g,base);
        }
        float sum=0.0f;
        if (linear_red) {
            for (uint64_t r=threadIdx.x;r<d.count;r+=256) {
                const int64_t ri=static_cast<int64_t>(r);
                int64_t addr[3]={base[0]+ri*a0r0,base[1]+ri*a0r1,base[2]+ri*a0r2};
                sum=__fadd_rn(sum,eval_at<OP,BINARY,SIDE>(addr,dy,left,right));
            }
        } else if (two_red) {
            const uint32_t limit=static_cast<uint32_t>(d.count);
            const uint32_t q256=256u/dim1;
            const uint32_t rem256=256u-q256*dim1;
            uint32_t c0=p2?(threadIdx.x>>shift):(threadIdx.x/dim1);
            uint32_t c1=p2?(threadIdx.x&mask):(threadIdx.x-c0*dim1);
            for (uint32_t r=threadIdx.x;r<limit;r+=256) {
                int64_t addr[3]={
                    base[0]+static_cast<int64_t>(c0)*a0r0+static_cast<int64_t>(c1)*a1r0,
                    base[1]+static_cast<int64_t>(c0)*a0r1+static_cast<int64_t>(c1)*a1r1,
                    base[2]+static_cast<int64_t>(c0)*a0r2+static_cast<int64_t>(c1)*a1r2,
                };
                sum=__fadd_rn(sum,eval_at<OP,BINARY,SIDE>(addr,dy,left,right));
                c0+=q256;
                c1+=rem256;
                if (c1>=dim1) { c1-=dim1; ++c0; }
            }
        } else {
            for (uint64_t r=threadIdx.x;r<d.count;r+=256)
                sum=__fadd_rn(sum,contribution<OP,BINARY,SIDE>(d,axes,base,r,dy,left,right));
        }
        partial[threadIdx.x]=sum;__syncthreads();
        for (unsigned step=128;step;step/=2) {if (threadIdx.x<step) partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+step]);__syncthreads();}
        if (!threadIdx.x) store(out,target,partial[0],accumulate);
        __syncthreads(); // Prevent a later group from replacing partials still in use.
    }
}
template<int OP,bool BINARY,int SIDE>
__global__ void block_split(PointwiseBackwardDims d,const PointwiseBackwardGroup *groups,const PointwiseBackwardAxis *axes,
    const float *dy,const float *left,const float *right,float *scratch) {
    const uint64_t g=blockIdx.x>>3;
    if (g>=d.groups) return;
    const unsigned lane=((blockIdx.x&7u)<<5)+threadIdx.x;
    int64_t base[3];
    if (d.rank<=1) {
        const int64_t gi=static_cast<int64_t>(g);
        base[0]=d.origins[0]+((d.rank==1)?gi*groups[0].reads[0]:0);
        base[1]=d.origins[1]+((d.rank==1)?gi*groups[0].reads[1]:0);
        base[2]=d.origins[2]+((d.rank==1)?gi*groups[0].reads[2]:0);
    } else {
        (void)group_offsets(d,groups,g,base);
    }
    float sum=0.0f;
    if (d.reduction_rank==1) {
        const int64_t s0=axes[0].reads[0], s1=axes[0].reads[1], s2=axes[0].reads[2];
        uint64_t r=lane;
        for (;r+768ULL<d.count;r+=1024ULL) {
            const int64_t r0=static_cast<int64_t>(r);
            const int64_t r1=static_cast<int64_t>(r+256ULL);
            const int64_t r2=static_cast<int64_t>(r+512ULL);
            const int64_t r3=static_cast<int64_t>(r+768ULL);
            int64_t a0[3]={base[0]+r0*s0,base[1]+r0*s1,base[2]+r0*s2};
            int64_t a1[3]={base[0]+r1*s0,base[1]+r1*s1,base[2]+r1*s2};
            int64_t a2[3]={base[0]+r2*s0,base[1]+r2*s1,base[2]+r2*s2};
            int64_t a3[3]={base[0]+r3*s0,base[1]+r3*s1,base[2]+r3*s2};
            const float v0=eval_at<OP,BINARY,SIDE>(a0,dy,left,right);
            const float v1=eval_at<OP,BINARY,SIDE>(a1,dy,left,right);
            const float v2=eval_at<OP,BINARY,SIDE>(a2,dy,left,right);
            const float v3=eval_at<OP,BINARY,SIDE>(a3,dy,left,right);
            sum=__fadd_rn(sum,v0);
            sum=__fadd_rn(sum,v1);
            sum=__fadd_rn(sum,v2);
            sum=__fadd_rn(sum,v3);
        }
        for (;r<d.count;r+=256ULL) {
            const int64_t ri=static_cast<int64_t>(r);
            int64_t addr[3]={base[0]+ri*s0,base[1]+ri*s1,base[2]+ri*s2};
            sum=__fadd_rn(sum,eval_at<OP,BINARY,SIDE>(addr,dy,left,right));
        }
    } else if (d.reduction_rank==2 && d.count<=0xffffffffULL) {
        const uint32_t limit=static_cast<uint32_t>(d.count);
        const uint32_t dim1=static_cast<uint32_t>(axes[1].dim);
        const int64_t a0r0=axes[0].reads[0],a0r1=axes[0].reads[1],a0r2=axes[0].reads[2];
        const int64_t a1r0=axes[1].reads[0],a1r1=axes[1].reads[1],a1r2=axes[1].reads[2];
        const uint32_t q256=256u/dim1;
        const uint32_t rem256=256u-q256*dim1;
        uint32_t c0=lane/dim1;
        uint32_t c1=lane-c0*dim1;
        for (uint32_t r=lane;r<limit;r+=256u) {
            int64_t addr[3]={
                base[0]+static_cast<int64_t>(c0)*a0r0+static_cast<int64_t>(c1)*a1r0,
                base[1]+static_cast<int64_t>(c0)*a0r1+static_cast<int64_t>(c1)*a1r1,
                base[2]+static_cast<int64_t>(c0)*a0r2+static_cast<int64_t>(c1)*a1r2,
            };
            sum=__fadd_rn(sum,eval_at<OP,BINARY,SIDE>(addr,dy,left,right));
            c0+=q256;
            c1+=rem256;
            if (c1>=dim1) { c1-=dim1; ++c0; }
        }
    } else {
        for (uint64_t r=lane;r<d.count;r+=256ULL)
            sum=__fadd_rn(sum,contribution<OP,BINARY,SIDE>(d,axes,base,r,dy,left,right));
    }
    scratch[g*256ULL+lane]=sum;
}
__global__ void block_finalize(PointwiseBackwardDims d,const PointwiseBackwardGroup *groups,
    const float *scratch,float *out,int accumulate) {
    __shared__ float partial[256];
    const uint64_t g=blockIdx.x;
    if (g>=d.groups) return;
    partial[threadIdx.x]=scratch[g*256ULL+threadIdx.x];
    __syncthreads();
    for (unsigned step=128;step;step/=2) {
        if (threadIdx.x<step) partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+step]);
        __syncthreads();
    }
    if (!threadIdx.x) {
        int64_t target;
        if (d.rank<=1) {
            target=d.origins[3]+((d.rank==1)?static_cast<int64_t>(g)*groups[0].output:0);
        } else {
            int64_t unused[3];
            target=group_offsets(d,groups,g,unused);
        }
        store(out,target,partial[0],accumulate);
    }
}
unsigned blocks(uint64_t work,unsigned per_block) {return static_cast<unsigned>(std::min<uint64_t>((work+per_block-1)/per_block,65535));}
template<int OP,bool BINARY,int SIDE>
void launch(const Backward *p,const float *dy,const float *left,const float *right,float *out,int accumulate) {
    const auto d=p->dims;
    if (d.kind==1 || d.kind==2) elementwise<OP,BINARY,SIDE><<<blocks(d.groups,256),256,0,p->ctx->stream>>>(d,p->groups,dy,left,right,out,accumulate);
    else if (d.kind==3 && d.count<=8) small_reduce<OP,BINARY,SIDE><<<blocks(d.groups,256),256,0,p->ctx->stream>>>(d,p->groups,p->reduced,dy,left,right,out,accumulate);
    else if (d.kind==3) warp<OP,BINARY,SIDE><<<blocks(d.groups,8),256,0,p->ctx->stream>>>(d,p->groups,p->reduced,dy,left,right,out,accumulate);
    else if (d.groups<=FA_CUDA_REDUCE_SCRATCH_GROUPS && d.count>=4096 && p->ctx->reduce_scratch) {
        block_split<OP,BINARY,SIDE><<<static_cast<unsigned>(d.groups*8ULL),32,0,p->ctx->stream>>>(d,p->groups,p->reduced,dy,left,right,p->ctx->reduce_scratch);
        block_finalize<<<static_cast<unsigned>(d.groups),256,0,p->ctx->stream>>>(d,p->groups,p->ctx->reduce_scratch,out,accumulate);
    }
    else block<OP,BINARY,SIDE><<<blocks(d.groups,1),256,0,p->ctx->stream>>>(d,p->groups,p->reduced,dy,left,right,out,accumulate);
}
template<int OP,bool BINARY,int SIDE> int configure(Backward *p) {
    const int needs=(p->dims.groups && p->dims.count) ? (needs_left<OP,BINARY,SIDE>()?1:0)|(needs_right<OP,BINARY,SIDE>()?2:0) : 0;
    if (p->dims.needs!=needs) return rt(cudaErrorInvalidValue);
    p->launch=launch<OP,BINARY,SIDE>;
    if (p->dims.kind==4) {cudaFuncAttributes a{};const int code=rt(cudaFuncGetAttributes(&a,block<OP,BINARY,SIDE>));if (code) return code;p->shared=a.sharedSizeBytes;}
    return 0;
}
int select(Backward *p) {
    const auto d=p->dims;
    if (d.arity==1) {
        switch (d.operation) {
#define UNARY(N) case N: return configure<N,false,0>(p);
            UNARY(0) UNARY(1) UNARY(2) UNARY(3) UNARY(4) UNARY(5) UNARY(6) UNARY(7) UNARY(8) UNARY(9) UNARY(10) UNARY(11)
#undef UNARY
        }
    } else if (d.operand==0) {
        switch (d.operation) {
#define BINARY(N) case N: return configure<N,true,0>(p);
            BINARY(0) BINARY(1) BINARY(2) BINARY(3)
#undef BINARY
        }
    } else {
        switch (d.operation) {
#define BINARY(N) case N: return configure<N,true,1>(p);
            BINARY(0) BINARY(1) BINARY(2) BINARY(3)
#undef BINARY
        }
    }
    return rt(cudaErrorInvalidValue);
}
}
#define TRY_GRAD(expr) do {const int code=(expr);if (code) return code;} while (0)
extern "C" int fa_cuda_pointwise_backward_create(Context *ctx,PointwiseBackwardDims d,
    const PointwiseBackwardGroup *groups,const PointwiseBackwardAxis *axes,size_t *workspace,size_t *shared,void **out) {
    if (!ctx || !workspace || !shared || !out || d.rank<0 || d.rank>63 || d.reduction_rank<0 || d.reduction_rank>63 ||
        d.kind<0 || d.kind>4 || (d.arity!=1 && d.arity!=2) || d.operand<0 || d.operand>1 || (d.arity==1 && d.operand) || d.reserved ||
        (d.rank && !groups) || (d.reduction_rank && !axes) || (d.kind==0 && d.groups) || (d.kind==1 && d.count) ||
        (d.kind==2 && d.count!=1) || (d.kind==3 && (d.count<2 || d.count>32)) || (d.kind==4 && d.count<=32)) return rt(cudaErrorInvalidValue);
    TRY_GRAD(activate(ctx));auto p=std::unique_ptr<Backward>(new(std::nothrow) Backward(ctx,d));if (!p) return 1002;
    TRY_GRAD(select(p.get()));
    const size_t group_bytes=d.rank*sizeof(PointwiseBackwardGroup),axis_bytes=d.reduction_rank*sizeof(PointwiseBackwardAxis);
    if (group_bytes) {TRY_GRAD(rt(cudaMalloc(&p->groups,group_bytes)));TRY_GRAD(rt(cudaMemcpyAsync(p->groups,groups,group_bytes,cudaMemcpyHostToDevice,ctx->stream)));}
    if (axis_bytes) {TRY_GRAD(rt(cudaMalloc(&p->reduced,axis_bytes)));TRY_GRAD(rt(cudaMemcpyAsync(p->reduced,axes,axis_bytes,cudaMemcpyHostToDevice,ctx->stream)));}
    if (group_bytes || axis_bytes) TRY_GRAD(rt(cudaStreamSynchronize(ctx->stream)));
    *workspace=group_bytes+axis_bytes;*shared=p->shared;*out=p.release();return 0;
}
extern "C" int fa_cuda_pointwise_backward_run(const Backward *p,const float *dy,const float *left,const float *right,float *out,int accumulate) {
    if (!p || (accumulate!=0 && accumulate!=1) || (p->dims.groups && !out) ||
        (p->dims.groups && p->dims.count && (!dy || ((p->dims.needs&1) && !left) || ((p->dims.needs&2) && !right)))) return rt(cudaErrorInvalidValue);
    if (!p->dims.groups || (!p->dims.count && accumulate)) return 0;
    TRY_GRAD(activate(p->ctx));p->launch(p,dy,left,right,out,accumulate);return rt(cudaGetLastError());
}
extern "C" void fa_cuda_pointwise_backward_destroy(Backward *p) {delete p;}
