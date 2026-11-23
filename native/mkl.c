#include <mkl.h>
#include <stddef.h>

void fa_mkl_sgemm(int m, int n, int k, const float *a, const float *b,
                   float *c, int threads) {
    int previous = mkl_set_num_threads_local(threads);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0f, a, k, b, n, 0.0f, c, n);
    mkl_set_num_threads_local(previous);
}

void fa_mkl_sgemm_layout(int m, int n, int k, int ta, int tb,
                        int lda, int ldb, int ldc, const float *a,
                        const float *b, float *c, int accumulate, int threads) {
    int previous = mkl_set_num_threads_local(threads);
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                tb ? CblasTrans : CblasNoTrans, m, n, k, 1.0f,
                a, lda, b, ldb, accumulate ? 1.0f : 0.0f, c, ldc);
    mkl_set_num_threads_local(previous);
}

static int fft_create(void **out, int n, int batch, int threads, int inverse) {
    DFTI_DESCRIPTOR_HANDLE plan = NULL;
    MKL_LONG code = DftiCreateDescriptor(&plan, DFTI_SINGLE, DFTI_REAL, 1, (MKL_LONG)n);
#define CONFIG(key, value) if (!code) code = DftiSetValue(plan, key, value)
    CONFIG(DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    CONFIG(DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    CONFIG(DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)batch);
    CONFIG(DFTI_INPUT_DISTANCE, (MKL_LONG)(inverse ? n / 2 + 1 : n));
    CONFIG(DFTI_OUTPUT_DISTANCE, (MKL_LONG)(inverse ? n : n / 2 + 1));
    CONFIG(DFTI_THREAD_LIMIT, (MKL_LONG)threads);
#undef CONFIG
    if (!code) code = DftiCommitDescriptor(plan);
    if (code) { if (plan) DftiFreeDescriptor(&plan); return (int)code; }
    *out = plan;
    return 0;
}

int fa_mkl_fft_create(void **out, int n, int batch, int threads) {
    return fft_create(out, n, batch, threads, 0);
}
int fa_mkl_ifft_create(void **out, int n, int batch, int threads) {
    return fft_create(out, n, batch, threads, 1);
}

int fa_mkl_fft_run(void *plan, float *input, float *output) {
    return (int)DftiComputeForward((DFTI_DESCRIPTOR_HANDLE)plan, input, output);
}
int fa_mkl_ifft_run(void *plan, float *input, float *output) {
    return (int)DftiComputeBackward((DFTI_DESCRIPTOR_HANDLE)plan, input, output);
}

void fa_mkl_fft_destroy(void *plan) {
    DFTI_DESCRIPTOR_HANDLE handle = (DFTI_DESCRIPTOR_HANDLE)plan;
    DftiFreeDescriptor(&handle);
}

void fa_mkl_vs_tanh(int n, const float *x, float *y) {
    int previous = mkl_set_num_threads_local(1);
    vmsTanh(n, x, y, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);
    mkl_set_num_threads_local(previous);
}
void fa_mkl_vs_exp(int n, const float *x, float *y) {
    int previous = mkl_set_num_threads_local(1);
    vmsExp(n, x, y, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);
    mkl_set_num_threads_local(previous);
}
void fa_mkl_vs_ln(int n, const float *x, float *y) {
    int previous = mkl_set_num_threads_local(1);
    vmsLn(n, x, y, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);
    mkl_set_num_threads_local(previous);
}
void fa_mkl_vs_sqrt(int n, const float *x, float *y) {
    int previous = mkl_set_num_threads_local(1);
    vmsSqrt(n, x, y, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);
    mkl_set_num_threads_local(previous);
}
