#include "cuda_backend.h"
#include "matmul.h"
#include <algorithm>
#include <climits>

namespace {
__device__ int64_t address(MatmulDims d,int matrix,uint64_t row,uint64_t col) {
    return d.origins[matrix]+static_cast<int64_t>(row)*d.strides[2*matrix]+static_cast<int64_t>(col)*d.strides[2*matrix+1];
}
__global__ void zero(MatmulDims d,float *out) {
    for (uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<d.m*d.n;i+=static_cast<uint64_t>(blockDim.x)*gridDim.x)
        out[address(d,2,i/d.n,i%d.n)]=0.0f;
}
__global__ void tiled(MatmulDims d,const float *a,const float *b,float *out,int accumulate) {
    __shared__ float aa[16][16],bb[16][16];
    const uint64_t columns=(d.n+15)/16,tiles=((d.m+15)/16)*columns;
    for (uint64_t tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
        const uint64_t row=(tile/columns)*16+threadIdx.y,col=(tile%columns)*16+threadIdx.x;
        float sum=0.0f;
        for (uint64_t start=0;start<d.k;start+=16) {
            const uint64_t p=start+threadIdx.x,q=start+threadIdx.y;
            aa[threadIdx.y][threadIdx.x]=(row<d.m && p<d.k) ? a[address(d,0,row,p)] : 0.0f;
            bb[threadIdx.y][threadIdx.x]=(q<d.k && col<d.n) ? b[address(d,1,q,col)] : 0.0f;
            __syncthreads();
            for (int i=0;i<16;++i) sum=__fadd_rn(sum,__fmul_rn(aa[threadIdx.y][i],bb[i][threadIdx.x]));
            // The last depth barrier also protects shared arrays before a later grid tile.
            __syncthreads();
        }
        if (row<d.m && col<d.n) {
            const int64_t target=address(d,2,row,col);
            out[target]=accumulate ? __fadd_rn(out[target],sum) : sum;
        }
    }
}
unsigned grid(uint64_t groups) {return static_cast<unsigned>(std::min<uint64_t>(groups,65535));}
}
extern "C" int fa_cuda_matmul_shared(Context *ctx,int kind,size_t *bytes) {
    if (!ctx || !bytes || (kind!=0 && kind!=1 && kind!=5 && kind!=6)) return rt(cudaErrorInvalidValue);
    int code=activate(ctx);if (code) return code;
    *bytes=0;
    if (kind==6) {cudaFuncAttributes a{};code=rt(cudaFuncGetAttributes(&a,tiled));if (!code) *bytes=a.sharedSizeBytes;}
    return code;
}
extern "C" int fa_cuda_matmul_layout(Context *ctx,MatmulDims d,const float *a,const float *b,float *out,int accumulate) {
    if (!ctx || (accumulate!=0 && accumulate!=1) || d.reserved[0] || d.reserved[1] ||
        (d.kind!=0 && d.kind!=1 && d.kind!=5 && d.kind!=6) || (d.kind==0 && d.m && d.n) ||
        (d.kind==1 && d.k) || (d.kind>1 && (!d.m || !d.n || !d.k))) return rt(cudaErrorInvalidValue);
    if (!d.m || !d.n || (!d.k && accumulate)) return 0;
    if (!out || (d.k && (!a || !b))) return rt(cudaErrorInvalidValue);
    int code=activate(ctx);if (code) return code;
    if (!d.k) {
        zero<<<grid((d.m*d.n+255)/256),256,0,ctx->stream>>>(d,out);
        return rt(cudaGetLastError());
    }
    if (d.kind==5) {
        if (d.m>INT_MAX || d.n>INT_MAX || d.k>INT_MAX ||
            d.transposes[0]<0 || d.transposes[0]>1 || d.transposes[1]<0 || d.transposes[1]>1 ||
            d.leading[0]<static_cast<int64_t>(d.transposes[0] ? d.m : d.k) ||
            d.leading[1]<static_cast<int64_t>(d.transposes[1] ? d.k : d.n) ||
            d.leading[2]<static_cast<int64_t>(d.n)) return rt(cudaErrorInvalidValue);
        const float one=1.0f,beta=accumulate ? 1.0f : 0.0f;
        const auto ta=d.transposes[0] ? CUBLAS_OP_T : CUBLAS_OP_N,tb=d.transposes[1] ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto status=cublasSgemm(ctx->blas,tb,ta,static_cast<int>(d.n),static_cast<int>(d.m),static_cast<int>(d.k),
            &one,b+d.origins[1],d.leading[1],a+d.origins[0],d.leading[0],&beta,out+d.origins[2],d.leading[2]);
        return status==CUBLAS_STATUS_SUCCESS ? 0 : 2000+static_cast<int>(status);
    }
    const uint64_t tiles=((d.m+15)/16)*((d.n+15)/16);
    tiled<<<grid(tiles),dim3(16,16),0,ctx->stream>>>(d,a,b,out,accumulate);
    return rt(cudaGetLastError());
}
