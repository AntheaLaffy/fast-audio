#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Minimal NVRTC and CUDA Driver API ABI declarations loaded dynamically via dlopen,
// avoiding hardcoded toolkit-version SONAME link dependencies across CUDA 11/12/13.
using nvrtcProgram = void *;
using nvrtcResult = int;
using CUresult = int;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;

struct DynApi {
    bool initialized = false;
    bool ok = false;

    nvrtcResult (*nvrtcCreateProgram)(
        nvrtcProgram *, const char *, const char *, int, const char *const *, const char *const *) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char *const *) = nullptr;
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t *) = nullptr;
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char *) = nullptr;
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t *) = nullptr;
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char *) = nullptr;
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram *) = nullptr;

    CUresult (*cuModuleLoadData)(CUmodule *, const void *) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
    CUresult (*cuLaunchKernel)(
        CUfunction,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, CUstream, void **, void **) = nullptr;
};

DynApi &dyn_api() {
    static DynApi api;
    static std::once_flag once;
    std::call_once(once, []() {
        api.initialized = true;
        void *nvrtc = nullptr;
        const char *nvrtc_names[] = {
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
            "/usr/local/cuda/lib64/libnvrtc.so",
        };
        for (const char *name : nvrtc_names) {
            nvrtc = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
            if (nvrtc) break;
        }
        void *driver = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!driver) driver = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
        if (!nvrtc || !driver) return;

        auto load = [](void *lib, const char *sym, auto &fn) -> bool {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, sym));
            return fn != nullptr;
        };
        bool good = true;
        good &= load(nvrtc, "nvrtcCreateProgram", api.nvrtcCreateProgram);
        good &= load(nvrtc, "nvrtcCompileProgram", api.nvrtcCompileProgram);
        good &= load(nvrtc, "nvrtcGetPTXSize", api.nvrtcGetPTXSize);
        good &= load(nvrtc, "nvrtcGetPTX", api.nvrtcGetPTX);
        good &= load(nvrtc, "nvrtcGetProgramLogSize", api.nvrtcGetProgramLogSize);
        good &= load(nvrtc, "nvrtcGetProgramLog", api.nvrtcGetProgramLog);
        good &= load(nvrtc, "nvrtcDestroyProgram", api.nvrtcDestroyProgram);
        good &= load(driver, "cuModuleLoadData", api.cuModuleLoadData);
        good &= load(driver, "cuModuleUnload", api.cuModuleUnload);
        good &= load(driver, "cuModuleGetFunction", api.cuModuleGetFunction);
        good &= load(driver, "cuLaunchKernel", api.cuLaunchKernel);
        api.ok = good;
    });
    return api;
}

std::mutex &ptx_cache_mutex() {
    static std::mutex m;
    return m;
}
std::unordered_map<std::string, std::string> &ptx_cache() {
    static std::unordered_map<std::string, std::string> cache;
    return cache;
}

struct FusedModule {
    Context *ctx;
    CUmodule module;
    CUfunction main_fn;
    CUfunction fin_fn;
    float *scratch;
    size_t scratch_bytes;
};

} // namespace

extern "C" int fa_cuda_nvrtc_available() {
    return dyn_api().ok ? 1 : 0;
}

extern "C" int fa_cuda_nvrtc_compile(
    Context *ctx,
    const char *source,
    const char *main_name,
    const char *fin_name,
    size_t scratch_bytes,
    void **out_handle,
    float **out_scratch) {
    if (!ctx || !source || !main_name || !out_handle || !out_scratch) {
        return rt(cudaErrorInvalidValue);
    }
    *out_handle = nullptr;
    *out_scratch = nullptr;
    DynApi &api = dyn_api();
    if (!api.ok) return 4001; // NVRTC/Driver unavailable

    int code = activate(ctx);
    if (code) return code;

    cudaDeviceProp props{};
    code = rt(cudaGetDeviceProperties(&props, ctx->device));
    if (code) return code;

    char arch_opt[64];
    std::snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", props.major, props.minor);
    std::string cache_key = std::string(arch_opt) + "\n" + source;

    std::string ptx;
    {
        std::lock_guard<std::mutex> lock(ptx_cache_mutex());
        auto it = ptx_cache().find(cache_key);
        if (it != ptx_cache().end()) {
            ptx = it->second;
        }
    }

    if (ptx.empty()) {
        nvrtcProgram prog = nullptr;
        nvrtcResult res = api.nvrtcCreateProgram(&prog, source, "fa_fused.cu", 0, nullptr, nullptr);
        if (res != 0) return 4000 + res;

        // Strict FP32 compilation: no --use_fast_math, standard IEEE rounding.
        const char *opts[] = {arch_opt, "--std=c++17"};
        res = api.nvrtcCompileProgram(prog, 2, opts);
        if (res != 0) {
            size_t log_size = 0;
            if (api.nvrtcGetProgramLogSize(prog, &log_size) == 0 && log_size > 1) {
                std::vector<char> log(log_size);
                if (api.nvrtcGetProgramLog(prog, log.data()) == 0) {
                    std::fprintf(stderr, "[fast-audio nvrtc] compile failed:\n%s\n", log.data());
                }
            }
            api.nvrtcDestroyProgram(&prog);
            return 4000 + res;
        }
        size_t ptx_size = 0;
        res = api.nvrtcGetPTXSize(prog, &ptx_size);
        if (res != 0 || ptx_size == 0) {
            api.nvrtcDestroyProgram(&prog);
            return 4000 + (res ? res : 1);
        }
        ptx.resize(ptx_size);
        res = api.nvrtcGetPTX(prog, ptx.data());
        api.nvrtcDestroyProgram(&prog);
        if (res != 0) return 4000 + res;

        std::lock_guard<std::mutex> lock(ptx_cache_mutex());
        if (ptx_cache().size() < 256) {
            ptx_cache().emplace(std::move(cache_key), ptx);
        }
    }

    CUmodule mod = nullptr;
    CUresult dr = api.cuModuleLoadData(&mod, ptx.data());
    if (dr != 0 || !mod) return 5000 + (dr ? dr : 1);

    CUfunction main_fn = nullptr;
    dr = api.cuModuleGetFunction(&main_fn, mod, main_name);
    if (dr != 0 || !main_fn) {
        api.cuModuleUnload(mod);
        return 5000 + (dr ? dr : 2);
    }

    CUfunction fin_fn = nullptr;
    if (fin_name && fin_name[0] != '\0') {
        dr = api.cuModuleGetFunction(&fin_fn, mod, fin_name);
        if (dr != 0 || !fin_fn) {
            api.cuModuleUnload(mod);
            return 5000 + (dr ? dr : 3);
        }
    }

    float *scratch = nullptr;
    if (scratch_bytes > 0) {
        code = rt(cudaMalloc(&scratch, scratch_bytes));
        if (code) {
            api.cuModuleUnload(mod);
            return code;
        }
    }

    FusedModule *fm = static_cast<FusedModule *>(std::calloc(1, sizeof(FusedModule)));
    if (!fm) {
        if (scratch) cudaFree(scratch);
        api.cuModuleUnload(mod);
        return 1002;
    }
    fm->ctx = ctx;
    fm->module = mod;
    fm->main_fn = main_fn;
    fm->fin_fn = fin_fn;
    fm->scratch = scratch;
    fm->scratch_bytes = scratch_bytes;
    *out_handle = fm;
    *out_scratch = scratch;
    return 0;
}

extern "C" int fa_cuda_nvrtc_launch(
    void *handle,
    unsigned int grid_x,
    unsigned int block_x,
    unsigned int smem_bytes,
    void **kernel_params,
    int is_finalize) {
    if (!handle || grid_x == 0 || block_x == 0) return rt(cudaErrorInvalidValue);
    FusedModule *fm = static_cast<FusedModule *>(handle);
    CUfunction fn = is_finalize ? fm->fin_fn : fm->main_fn;
    if (!fn) return rt(cudaErrorInvalidValue);
    int code = activate(fm->ctx);
    if (code) return code;
    DynApi &api = dyn_api();
    CUresult dr = api.cuLaunchKernel(
        fn,
        grid_x, 1, 1,
        block_x, 1, 1,
        smem_bytes,
        reinterpret_cast<CUstream>(fm->ctx->stream),
        kernel_params,
        nullptr);
    return dr == 0 ? 0 : 5000 + dr;
}

extern "C" void fa_cuda_nvrtc_destroy(void *handle) {
    if (!handle) return;
    FusedModule *fm = static_cast<FusedModule *>(handle);
    cudaSetDevice(fm->ctx->device);
    cudaStreamSynchronize(fm->ctx->stream);
    if (fm->scratch) cudaFree(fm->scratch);
    DynApi &api = dyn_api();
    if (api.ok && fm->module) api.cuModuleUnload(fm->module);
    std::free(fm);
}
