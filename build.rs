use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native");
    for name in ["MKLROOT", "CUDA_ROOT", "CUDA_ARCH", "CUDA_HOST_CXX"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if env::var_os("CARGO_FEATURE_MKL").is_some() {
        let root = PathBuf::from(
            env::var("MKLROOT").unwrap_or_else(|_| "/opt/intel/oneapi/mkl/latest".into()),
        );
        let mut build = cc::Build::new();
        build
            .file("native/mkl.c")
            .include(root.join("include"))
            .warnings(true);
        let sys_mkl = PathBuf::from("/usr/include/mkl");
        if sys_mkl.is_dir() {
            build.include(sys_mkl);
        }
        build.compile("fast_audio_mkl");
        for dir in [root.join("lib/intel64"), root.join("lib")] {
            if dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
        println!("cargo:rustc-link-lib=mkl_rt");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "86".into());
        assert!(
            arch.chars().all(|c| c.is_ascii_digit()),
            "CUDA_ARCH must be numeric"
        );
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("shared")
            .file("native/cuda.cu")
            .file("native/tensor_ops.cu")
            .file("native/tensor_copy.cu")
            .file("native/view_backward.cu")
            .file("native/pointwise.cu")
            .file("native/pointwise_backward.cu")
            .file("native/matmul.cu")
            .file("native/bmm.cu")
            .file("native/sdpa.cu")
            .file("native/conv_backward.cu")
            .file("native/stft_backward.cu")
            .file("native/norm.cu")
            .file("native/softmax.cu")
            .file("native/structural.cu")
            .file("native/training.cu")
            .file("native/nvrtc_fusion.cu")
            .file("native/reduce.cu")
            .file("native/audio_ops.cu")
            .file("native/bf16_tc.cu")
            .file("native/sdpa_bf16.cu")
            .flag("-std=c++17")
            .flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
            .flag(format!("-gencode=arch=compute_{arch},code=compute_{arch}"));
        if let Ok(host) = env::var("CUDA_HOST_CXX") {
            build.ccbin(false).flag(format!("-ccbin={host}"));
        }
        if let Ok(root) = env::var("CUDA_ROOT") {
            build.include(PathBuf::from(&root).join("include"));
            println!("cargo:rustc-link-search=native={root}/lib64");
        }
        if env::var_os("CARGO_FEATURE_CUDNN").is_some() {
            build.file("native/cudnn.cu");
            build.file("native/gru.cu");
            println!("cargo:rustc-link-lib=cudnn");
        }
        build.compile("fast_audio_cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cufft");
        println!("cargo:rustc-link-lib=dl");
    }
    if env::var_os("CARGO_FEATURE_ONEDNN").is_some() {
        cc::Build::new()
            .cpp(true)
            .file("native/onednn.cpp")
            .flag("-std=c++17")
            .flag("-fopenmp")
            .compile("fast_audio_onednn");
        println!("cargo:rustc-link-lib=dnnl");
        println!("cargo:rustc-link-lib=gomp");
    }
}
