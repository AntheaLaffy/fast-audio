#include "cuda_backend.h"
#include "tensor_ops.h"
#include <algorithm>
#include <cmath>

__device__ static uint32_t mapped32(TensorMap32 k, uint32_t i) {
    uint32_t source = k.offset;
    #pragma unroll
    for (int a = 3; a >= 0; --a) {
        const uint32_t d = k.dims[a];
        if (d != 1) {
            uint32_t q = __umulhi(i, k.reciprocal[a]);
            uint32_t r = i - q * d;
            // floor(2^32/d) underestimates the quotient by at most one.
            if (r >= d) { ++q; r -= d; }
            source += r * k.strides[a];
            i = q;
        }
    }
    return source;
}
template<bool Multiply>
__global__ static void mapped_elementwise(TensorMap32 k, const float *x, const float *y, float *out) {
    // The grid stride stays 64-bit so a final step cannot wrap near UINT32_MAX.
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < k.len;
         i += (uint64_t)blockDim.x * gridDim.x) {
        const uint32_t source = mapped32(k, (uint32_t)i);
        out[i] = Multiply ? x[i] * y[source] : x[source];
    }
}
extern "C" int fa_cuda_tensor_map32(Context *ctx, TensorMap32 k, const float *x,
                                    const float *y, float *out) noexcept {
    const int code = activate(ctx); if (code) return code;
    const auto grid = (unsigned)std::min(((uint64_t)k.len + 255) / 256, (uint64_t)65535);
    if (k.multiply) mapped_elementwise<true><<<grid,256,0,ctx->stream>>>(k,x,y,out);
    else mapped_elementwise<false><<<grid,256,0,ctx->stream>>>(k,x,y,out);
    return rt(cudaGetLastError());
}

__device__ static uint64_t mapped(TensorOpDims k,uint64_t i) {
    uint64_t source=k.offset;
    for(int a=k.rank-1;a>=0;--a) {source+=(i%k.dims[a])*k.strides[a];i/=k.dims[a];}
    return source;
}
__global__ static void elementwise(TensorOpDims k,const float *x,const float *y,const float *bias,float *out) {
    for(uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;i<k.len;i+=(uint64_t)blockDim.x*gridDim.x) {
        float v=0.0f;
        switch(k.kind) {
        case 0:v=x[mapped(k,i)];break;
        case 1:v=k.rank==0 ? 1.0f/(1.0f+expf(-x[i])) : (k.rank==1 ? tanhf(x[i]) : (x[i]<0 ? k.param*x[i] : x[i]));break;
        case 2:v=x[i]+y[i];break;
        case 3:v=x[i]*y[mapped(k,i)];break;
        case 4:{
            const auto n=k.inner,a=k.width,b=k.offset,outer=i/((a+b)*n),axis=i/n%(a+b),inner=i%n;
            const bool right=k.rank ? axis%2!=0 : axis>=a;
            const auto channel=k.rank ? axis/2 : (right ? axis-a : axis);
            v=right ? y[(outer*b+channel)*n+inner] : x[(outer*a+channel)*n+inner];break;
        }
        case 5:{
            const auto n=k.inner,axis=i/n%k.dims[0];
            v=(axis>=k.offset && axis-k.offset<k.width) ? x[(i/(k.dims[0]*n)*k.width+axis-k.offset)*n+i%n] : 0.0f;break;
        }
        case 6:{
            const auto c=k.dims[1],t=k.dims[2],f=k.dims[3],width=k.width;
            const auto channel=i/(t*f)%(c*width),batch=i/(t*f*c*width);
            const int64_t frequency=(int64_t)(i%f)+(int64_t)(channel%width)-(int64_t)(width/2);
            v=frequency>=0 && (uint64_t)frequency<f ? x[((batch*c+channel/width)*t+i/f%t)*f+frequency] : 0.0f;break;
        }
        case 7:{
            float sum=0.0f;
            for(uint64_t j=0;j<k.inner;++j) {const float z=x[i*k.inner+j];sum=__fadd_rn(sum,__fmul_rn(z,z));}
            v=sum/(float)k.inner;break;
        }
        case 9:{
            const auto n=k.inner,channel=i/n%3,source=(i/(3*n)*n+i%n)*2;
            const float re=x[source],im=x[source+1];
            v=channel==0 ? sqrtf(__fadd_rn(__fadd_rn(__fmul_rn(re,re),__fmul_rn(im,im)),k.eps)) : (channel==1 ? re : im);break;
        }
        case 10:{
            const auto p=i/2,m=(p/k.inner)*2*k.inner+p%k.inner;
            v=i%2==0 ? __fadd_rn(__fmul_rn(y[2*p],x[m]),-__fmul_rn(y[2*p+1],x[m+k.inner]))
                : __fadd_rn(__fmul_rn(y[2*p+1],x[m]),__fmul_rn(y[2*p],x[m+k.inner]));break;
        }
        case 11:v=x[i]+bias[i%k.inner];break;
        }
        out[i]=v;
    }
}
__global__ static void layer_norm(TensorOpDims k,const float *x,const float *scale,const float *bias,float *out) {
    __shared__ float sums[256];
    const auto tid=threadIdx.x;
    const auto n=k.inner,rows=k.len/n;
    for(uint64_t row=blockIdx.x;row<rows;row+=gridDim.x) {
        float sum=0.0f;
        for(uint64_t j=tid;j<n;j+=256) sum+=x[row*n+j];
        sums[tid]=sum;__syncthreads();
        for(int step=128;step;step/=2) {if(tid<(unsigned)step) sums[tid]+=sums[tid+step];__syncthreads();}
        const float mean=sums[0]/(float)n;
        __syncthreads();
        sum=0.0f;
        for(uint64_t j=tid;j<n;j+=256) {const float d=x[row*n+j]-mean;sum=__fadd_rn(sum,__fmul_rn(d,d));}
        sums[tid]=sum;__syncthreads();
        for(int step=128;step;step/=2) {if(tid<(unsigned)step) sums[tid]+=sums[tid+step];__syncthreads();}
        const float inverse=1.0f/sqrtf(sums[0]/(float)n+k.eps);
        for(uint64_t j=tid;j<n;j+=256) out[row*n+j]=__fadd_rn(__fmul_rn(__fmul_rn(x[row*n+j]-mean,inverse),scale[j]),bias[j]);
        __syncthreads();
    }
}
extern "C" int fa_cuda_tensor_op(Context *ctx,TensorOpDims k,const float *x,const float *y,
                                  const float *scale,const float *bias,float *out) noexcept {
    const int code=activate(ctx);if(code) return code;
    if(k.kind==8) {
        const auto grid=(unsigned)std::min(k.len/k.inner,(uint64_t)65535);
        layer_norm<<<grid,256,0,ctx->stream>>>(k,x,scale,bias,out);
    } else {
        const auto grid=(unsigned)std::min((k.len+255)/256,(uint64_t)65535);
        elementwise<<<grid,256,0,ctx->stream>>>(k,x,y,bias,out);
    }
    return rt(cudaGetLastError());
}
