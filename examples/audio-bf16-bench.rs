use fast_audio::{
    CrossEntropySpec, DropoutSpec, EmbeddingSpec, GradientMode, IndexTensor, Layout, RopeSpec,
    Shape, Tensor, cpu,
};
use std::{env, fs, time::Instant};

#[cfg(feature = "cuda")]
use fast_audio::{
    AdamWConfig, Bf16Tensor, PrecisionPolicy, SdpaSpec,
    cuda::{
        Bf16AudioTransformerConfig, Bf16AudioTransformerTrainer, Bf16CrossEntropyPlan,
        Bf16DropoutPlan, Bf16EmbeddingPlan, Bf16LinearPlan, Bf16NormPlan, Bf16RopePlan,
        Bf16SdpaPlan, Bf16SwiGluPlan, Cuda,
    },
};

fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).expect("read f32 bin");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i64_bin(path: &str) -> Vec<i64> {
    let bytes = fs::read(path).expect("read i64 bin");
    assert_eq!(bytes.len() % 8, 0);
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn write_f32_bin(path: &str, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).expect("write f32 bin");
}

fn bench_cpu_ms<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn run_bench_cpu() {
    let mut results = Vec::new();
    // 1. CPU Embedding fwd+bwd [4, 512] V=4096 D=512
    {
        let idx =
            IndexTensor::from_vec([4, 512], (0..2048).map(|i| (i * 17) % 4096).collect()).unwrap();
        let spec = EmbeddingSpec::new(4096, 512, idx.shape(), Some(0)).unwrap();
        let w = Tensor::from_vec([4096, 512], vec![0.1; 4096 * 512]).unwrap();
        let mut out = Tensor::zeros([4, 512, 512]).unwrap();
        let dy = Tensor::from_vec([4, 512, 512], vec![0.01; 2048 * 512]).unwrap();
        let mut dw = Tensor::zeros([4096, 512]).unwrap();
        let fwd = cpu::EmbeddingPlan::new(spec.clone());
        let bwd = cpu::EmbeddingBackward::new(spec);
        let us = bench_cpu_ms(3, 15, || {
            fwd.run(&w, &idx, &mut out).unwrap();
            bwd.run(&idx, &dy, &mut dw, GradientMode::Overwrite)
                .unwrap();
        });
        results.push(serde_json::json!({"name": "cpu-embedding-v4096-d512", "us": us}));
    }
    // 2. CPU RoPE fwd+bwd [4, 8, 512, 64]
    {
        let dims = [4, 8, 512, 64];
        let shape = Shape::new(dims).unwrap();
        let layout = Layout::contiguous(dims).unwrap();
        let spec = RopeSpec::new(&shape, 2, Some(64), 10_000.0, 0).unwrap();
        let x = Tensor::from_vec(dims, vec![0.25; shape.len()]).unwrap();
        let mut y = Tensor::zeros(dims).unwrap();
        let mut dx = Tensor::zeros(dims).unwrap();
        let fwd = cpu::RopePlan::new(&spec, &layout, &layout).unwrap();
        let bwd = cpu::RopeBackward::new(&spec, &layout, &layout).unwrap();
        let us = bench_cpu_ms(3, 15, || {
            fwd.run(&x.view(&layout).unwrap(), &mut y.view_mut(&layout).unwrap())
                .unwrap();
            bwd.run(
                &y.view(&layout).unwrap(),
                &mut dx.view_mut(&layout).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        results.push(serde_json::json!({"name": "cpu-rope-t512-d64", "us": us}));
    }
    // 3. CPU Fused CrossEntropy fwd+bwd [512, 4096]
    {
        let spec = CrossEntropySpec::new(512, 4096, Some(-100), 0.0).unwrap();
        let logits = Tensor::from_vec([512, 4096], vec![0.1; 512 * 4096]).unwrap();
        let targets =
            IndexTensor::from_vec([512], (0..512).map(|i| (i * 13 % 4096) as i64).collect())
                .unwrap();
        let mut loss = Tensor::zeros([]).unwrap();
        let mut grad = Tensor::zeros([512, 4096]).unwrap();
        let fwd = cpu::CrossEntropyPlan::new(spec);
        let bwd = cpu::CrossEntropyBackward::new(spec);
        let us = bench_cpu_ms(2, 10, || {
            fwd.run(&logits, &targets, &mut loss).unwrap();
            bwd.run(&logits, &targets, 1.0, &mut grad, GradientMode::Overwrite)
                .unwrap();
        });
        results.push(serde_json::json!({"name": "cpu-cross-entropy-512x4096", "us": us}));
    }
    // 4. CPU Philox Dropout fwd+bwd [4, 512, 512]
    {
        let dims = [4, 512, 512];
        let shape = Shape::new(dims).unwrap();
        let layout = Layout::contiguous(dims).unwrap();
        let spec = DropoutSpec::new(&shape, 0.1, 1234).unwrap();
        let x = Tensor::from_vec(dims, vec![0.5; shape.len()]).unwrap();
        let mut y = Tensor::zeros(dims).unwrap();
        let mut dx = Tensor::zeros(dims).unwrap();
        let fwd = cpu::DropoutPlan::new(&spec, &layout, &layout).unwrap();
        let bwd = cpu::DropoutBackward::new(&spec, &layout, &layout).unwrap();
        let us = bench_cpu_ms(3, 15, || {
            fwd.run(
                &x.view(&layout).unwrap(),
                &mut y.view_mut(&layout).unwrap(),
                1,
            )
            .unwrap();
            bwd.run(
                &y.view(&layout).unwrap(),
                &mut dx.view_mut(&layout).unwrap(),
                1,
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        results.push(serde_json::json!({"name": "cpu-dropout-1m", "us": us}));
    }
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}

fn run_verify_cpu(dir: &str) {
    // Embedding
    {
        let table = Tensor::from_vec([32, 64], read_f32_bin(&format!("{dir}/emb_w.bin"))).unwrap();
        let idx =
            IndexTensor::from_vec([4, 16], read_i64_bin(&format!("{dir}/emb_idx.bin"))).unwrap();
        let dy = Tensor::from_vec([4, 16, 64], read_f32_bin(&format!("{dir}/emb_dy.bin"))).unwrap();
        let spec = EmbeddingSpec::new(32, 64, idx.shape(), Some(0)).unwrap();
        let mut out = Tensor::zeros([4, 16, 64]).unwrap();
        let mut dw = Tensor::zeros([32, 64]).unwrap();
        cpu::EmbeddingPlan::new(spec.clone())
            .run(&table, &idx, &mut out)
            .unwrap();
        cpu::EmbeddingBackward::new(spec)
            .run(&idx, &dy, &mut dw, GradientMode::Overwrite)
            .unwrap();
        write_f32_bin(&format!("{dir}/emb_out_fa.bin"), out.data());
        write_f32_bin(&format!("{dir}/emb_dw_fa.bin"), dw.data());
    }
    // RoPE
    {
        let dims = [2, 4, 16, 32];
        let shape = Shape::new(dims).unwrap();
        let layout = Layout::contiguous(dims).unwrap();
        let spec = RopeSpec::new(&shape, 2, Some(32), 10_000.0, 0).unwrap();
        let x = Tensor::from_vec(dims, read_f32_bin(&format!("{dir}/rope_x.bin"))).unwrap();
        let dy = Tensor::from_vec(dims, read_f32_bin(&format!("{dir}/rope_dy.bin"))).unwrap();
        let mut y = Tensor::zeros(dims).unwrap();
        let mut dx = Tensor::zeros(dims).unwrap();
        cpu::RopePlan::new(&spec, &layout, &layout)
            .unwrap()
            .run(&x.view(&layout).unwrap(), &mut y.view_mut(&layout).unwrap())
            .unwrap();
        cpu::RopeBackward::new(&spec, &layout, &layout)
            .unwrap()
            .run(
                &dy.view(&layout).unwrap(),
                &mut dx.view_mut(&layout).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        write_f32_bin(&format!("{dir}/rope_y_fa.bin"), y.data());
        write_f32_bin(&format!("{dir}/rope_dx_fa.bin"), dx.data());
    }
    // CrossEntropy (ls=0.0 and ls=0.1)
    for (tag, ls) in [("ce0", 0.0f32), ("ce1", 0.1f32)] {
        let logits =
            Tensor::from_vec([16, 64], read_f32_bin(&format!("{dir}/ce_logits.bin"))).unwrap();
        let targets =
            IndexTensor::from_vec([16], read_i64_bin(&format!("{dir}/ce_targets.bin"))).unwrap();
        let spec = CrossEntropySpec::new(16, 64, Some(-100), ls).unwrap();
        let mut loss = Tensor::zeros([]).unwrap();
        let mut grad = Tensor::zeros([16, 64]).unwrap();
        cpu::CrossEntropyPlan::new(spec)
            .run(&logits, &targets, &mut loss)
            .unwrap();
        cpu::CrossEntropyBackward::new(spec)
            .run(&logits, &targets, 1.0, &mut grad, GradientMode::Overwrite)
            .unwrap();
        write_f32_bin(&format!("{dir}/{tag}_loss_fa.bin"), loss.data());
        write_f32_bin(&format!("{dir}/{tag}_grad_fa.bin"), grad.data());
    }
}

#[cfg(feature = "cuda")]
fn run_bench_cuda(fixture_dir: Option<&str>) {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();
    let mut results = Vec::new();
    let mut record =
        |name: &str, warmup: usize, iters: usize, f: &mut dyn FnMut() -> fast_audio::Result<()>| {
            let first = cuda.measure(1, &mut *f).unwrap();
            for _ in 0..warmup {
                f().unwrap();
            }
            let timing = cuda.measure(iters, &mut *f).unwrap();
            let mut wall_samples = Vec::with_capacity(100);
            let mut device_samples = Vec::with_capacity(100);
            for _ in 0..100 {
                let sample = cuda.measure(1, &mut *f).unwrap();
                wall_samples.push(sample.wall_us);
                device_samples.push(sample.device_us);
            }
            results.push(serde_json::json!({
                "name": name,
                "device_us": timing.device_us,
                "wall_us": timing.wall_us,
                "first_device_us": first.device_us,
                "first_wall_us": first.wall_us,
                "device_samples_us": device_samples,
                "wall_samples_us": wall_samples,
            }));
        };

    // 1. BF16 Linear fwd+bwd [2048, 1024] x [1024, 1024]
    {
        let (m, k, n) = (2048, 1024, 1024);
        let x = cuda.zeros_bf16([m, k]).unwrap();
        let w = cuda.zeros_bf16([n, k]).unwrap();
        let mut y = cuda.zeros_bf16([m, n]).unwrap();
        let dy = cuda.zeros_bf16([m, n]).unwrap();
        let mut dx = cuda.zeros_bf16([m, k]).unwrap();
        let mut dw = cuda.zeros([n, k]).unwrap();
        let lin = Bf16LinearPlan::new(&cuda, m, k, n).unwrap();
        record("bf16-linear-2048x1024-fwd-bwd", 10, 50, &mut || {
            lin.forward(&x, &w, &mut y)?;
            lin.backward_input(&dy, &w, &mut dx, GradientMode::Overwrite)?;
            lin.backward_weight_f32(&dy, &x, &mut dw, GradientMode::Overwrite)
        });
    }

    // 2. BF16 FlashAttention-2 Tensor Core (D=64 T=512, D=64 T=1024, GQA D=64 T=2048, D=128 T=1024)
    for (name, b, hq, hkv, t, d, causal) in [
        ("bf16-sdpa-causal-t512-d64", 4, 8, 8, 512, 64, true),
        ("bf16-sdpa-causal-t1024-d64", 4, 8, 8, 1024, 64, true),
        ("bf16-sdpa-gqa-causal-t2048-d64", 2, 8, 2, 2048, 64, true),
        ("bf16-sdpa-causal-t1024-d128", 2, 8, 8, 1024, 128, true),
    ] {
        let spec = SdpaSpec::new(b, hq, hkv, t, t, d, None, causal).unwrap();
        let q = cuda.zeros_bf16([b, hq, t, d]).unwrap();
        let k = cuda.zeros_bf16([b, hkv, t, d]).unwrap();
        let v = cuda.zeros_bf16([b, hkv, t, d]).unwrap();
        let mut o = cuda.zeros_bf16([b, hq, t, d]).unwrap();
        let dout = cuda.zeros_bf16([b, hq, t, d]).unwrap();
        let mut dq = cuda.zeros_bf16([b, hq, t, d]).unwrap();
        let mut dk = cuda.zeros_bf16([b, hkv, t, d]).unwrap();
        let mut dv = cuda.zeros_bf16([b, hkv, t, d]).unwrap();
        let mut plan = Bf16SdpaPlan::new(&cuda, spec).unwrap();
        let fwd_name = format!("{name}-fwd");
        record(&fwd_name, 5, 30, &mut || plan.forward(&q, &k, &v, &mut o));
        let fb_name = format!("{name}-fwd-bwd");
        record(&fb_name, 5, 25, &mut || {
            plan.forward(&q, &k, &v, &mut o)?;
            plan.backward(
                &q,
                &k,
                &v,
                &o,
                &dout,
                &mut dq,
                &mut dk,
                &mut dv,
                GradientMode::Overwrite,
            )
        });
    }

    // 3. BF16 Residual+RmsNorm fwd+bwd [2048, 1024]
    {
        let (rows, cols) = (2048, 1024);
        let a = cuda.zeros_bf16([rows, cols]).unwrap();
        let b = cuda.zeros_bf16([rows, cols]).unwrap();
        let w = cuda.zeros_bf16([cols]).unwrap();
        let mut res = cuda.zeros_bf16([rows, cols]).unwrap();
        let mut y = cuda.zeros_bf16([rows, cols]).unwrap();
        let mut inv = cuda.zeros([rows]).unwrap();
        let dy = cuda.zeros_bf16([rows, cols]).unwrap();
        let mut dx = cuda.zeros_bf16([rows, cols]).unwrap();
        let mut dw = cuda.zeros([cols]).unwrap();
        let mut norm = Bf16NormPlan::new(&cuda, rows, cols, 1e-5).unwrap();
        record("bf16-rmsnorm-2048x1024-fwd-bwd", 10, 50, &mut || {
            norm.residual_rmsnorm_forward(&a, &b, Some(&w), &mut res, &mut y, Some(&mut inv))?;
            norm.rmsnorm_backward(
                &res,
                Some(&w),
                Some(&inv),
                &dy,
                &mut dx,
                Some(&mut dw),
                GradientMode::Overwrite,
                GradientMode::Overwrite,
            )
        });
    }

    // 4. BF16 SwiGLU fwd+bwd [2048, 1536]
    {
        let (rows, h) = (2048, 1536);
        let gu = cuda.zeros_bf16([rows, 2 * h]).unwrap();
        let mut out = cuda.zeros_bf16([rows, h]).unwrap();
        let dout = cuda.zeros_bf16([rows, h]).unwrap();
        let mut dgu = cuda.zeros_bf16([rows, 2 * h]).unwrap();
        let sw = Bf16SwiGluPlan::new(&cuda, rows, h).unwrap();
        record("bf16-swiglu-2048x1536-fwd-bwd", 10, 50, &mut || {
            sw.forward(&gu, &mut out)?;
            sw.backward(&gu, &dout, &mut dgu)
        });
    }

    // 5. BF16 RoPE fwd+bwd [4, 8, 1024, 64]
    {
        let dims = [4, 8, 1024, 64];
        let spec = RopeSpec::new(&Shape::new(dims).unwrap(), 2, Some(64), 10_000.0, 0).unwrap();
        let rope = Bf16RopePlan::new(&cuda, spec).unwrap();
        let x = cuda.zeros_bf16(dims).unwrap();
        let mut y = cuda.zeros_bf16(dims).unwrap();
        let mut dx = cuda.zeros_bf16(dims).unwrap();
        record("bf16-rope-t1024-d64-fwd-bwd", 10, 50, &mut || {
            rope.forward(&x, &mut y)?;
            rope.backward(&y, &mut dx, GradientMode::Overwrite)
        });
    }

    // 6. BF16 Single-Pass Fused CrossEntropy fwd+bwd [2048, 4096]
    {
        let spec = CrossEntropySpec::new(2048, 4096, Some(-100), 0.0).unwrap();
        let mut ce = Bf16CrossEntropyPlan::new(&cuda, spec).unwrap();
        let logits = cuda.zeros_bf16([2048, 4096]).unwrap();
        let targets = cuda
            .upload_indices(
                &IndexTensor::from_vec([2048], (0..2048).map(|i| (i * 13 % 4096) as i64).collect())
                    .unwrap(),
            )
            .unwrap();
        let mut loss = cuda.zeros([]).unwrap();
        let mut grad = cuda.zeros_bf16([2048, 4096]).unwrap();
        record("bf16-cross-entropy-2048x4096-fwd-bwd", 10, 50, &mut || {
            ce.forward_backward(&logits, &targets, 1.0, &mut loss, &mut grad)
        });
    }

    // 7. Flagship 6-Layer GQA Causal Audio Transformer (B=4, T=512, D=512, H=8/Hkv=2, FFN=1536, V=4096)
    //    Full Forward + Backward + Global L2 Clip + Fused FP32-Master/BF16-Mirror AdamW in ONE CUDA Graph!
    {
        let cfg = Bf16AudioTransformerConfig {
            batch: 4,
            seq_len: 512,
            vocab_size: 4096,
            d_model: 512,
            num_heads: 8,
            num_kv_heads: 2,
            ffn_hidden: 1536,
            num_layers: 6,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            ignore_index: None,
        };
        let shapes = Bf16AudioTransformerTrainer::parameter_shapes(&cfg).unwrap();
        let weights: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(idx, s)| {
                let data: Vec<f32> = if let Some(dir) = fixture_dir {
                    read_f32_bin(&format!("{dir}/tr_w_{idx}.bin"))
                } else if s.dims().len() == 1 {
                    vec![1.0; s.len()]
                } else {
                    (0..s.len())
                        .map(|i| (((i + idx * 97) as f32 * 0.13).sin()) * 0.02)
                        .collect()
                };
                Tensor::from_vec(s.dims(), data).unwrap()
            })
            .collect();
        let optim_cfg = AdamWConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: Some(1.0),
        };
        let mut trainer =
            Bf16AudioTransformerTrainer::new(&cuda, cfg, &weights, optim_cfg).unwrap();
        let in_idx = IndexTensor::from_vec(
            [cfg.batch, cfg.seq_len],
            (0..cfg.tokens())
                .map(|i| ((i * 19) % cfg.vocab_size) as i64)
                .collect(),
        )
        .unwrap();
        let tgt_idx = IndexTensor::from_vec(
            [cfg.tokens()],
            (0..cfg.tokens())
                .map(|i| (((i + 1) * 19) % cfg.vocab_size) as i64)
                .collect(),
        )
        .unwrap();
        trainer.set_batch(&in_idx, &tgt_idx).unwrap();
        // Warmup & capture single CUDA Graph!
        trainer.enqueue_step().unwrap();
        cuda.synchronize().unwrap();
        let capture_start = Instant::now();
        trainer.capture_step_graph().unwrap();
        let capture_wall_us = capture_start.elapsed().as_secs_f64() * 1e6;
        record("bf16-audio-transformer-6layer-adamw", 5, 30, &mut || {
            trainer.step_captured()
        });
        results.last_mut().unwrap()["capture_wall_us"] = serde_json::json!(capture_wall_us);
    }

    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}

#[cfg(feature = "cuda")]
fn run_bench_cuda_fallback() {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();
    let (queries, keys, dim) = (64, 80, 48);
    let make_input = |heads, length, seed| {
        let data = (0..heads * length * dim)
            .map(|i| ((i * seed % 79) as f32 - 39.0) / 128.0)
            .collect();
        let host = Tensor::from_vec([1, heads, length, dim], data).unwrap();
        cuda.upload_bf16(&Bf16Tensor::from_f32(&host).unwrap())
            .unwrap()
    };
    let q = make_input(4, queries, 19);
    let k = make_input(2, keys, 23);
    let v = make_input(2, keys, 31);
    let mut out = cuda.zeros_bf16([1, 4, queries, dim]).unwrap();
    let spec = SdpaSpec::new(1, 4, 2, queries, keys, dim, None, true).unwrap();
    let mut plan = Bf16SdpaPlan::new(&cuda, spec).unwrap();
    for _ in 0..10 {
        plan.forward(&q, &k, &v, &mut out).unwrap();
    }
    let timing = cuda
        .measure(50, || plan.forward(&q, &k, &v, &mut out))
        .unwrap();
    println!(
        "{}",
        serde_json::json!({"device_us": timing.device_us, "wall_us": timing.wall_us}),
    );
}

#[cfg(feature = "cuda")]
fn run_verify_cuda(dir: &str) {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();

    // 1. BF16 SDPA Tensor Core and general-stride routes.
    for (tag, b, hq, hkv, tq, tk, d, causal) in [
        ("sdpa64", 2, 8, 2, 96, 96, 64, true),
        ("sdpa128", 1, 4, 4, 128, 128, 128, false),
        ("sdpa48", 1, 4, 2, 64, 80, 48, true),
        ("sdpa129", 1, 2, 1, 29, 21, 129, true),
    ] {
        let spec = SdpaSpec::new(b, hq, hkv, tq, tk, d, None, causal).unwrap();
        let q = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([b, hq, tq, d], read_f32_bin(&format!("{dir}/{tag}_q.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let k = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([b, hkv, tk, d], read_f32_bin(&format!("{dir}/{tag}_k.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let v = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([b, hkv, tk, d], read_f32_bin(&format!("{dir}/{tag}_v.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let dout = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([b, hq, tq, d], read_f32_bin(&format!("{dir}/{tag}_do.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let mut o = cuda.zeros_bf16([b, hq, tq, d]).unwrap();
        let mut dq = cuda.zeros_bf16([b, hq, tq, d]).unwrap();
        let mut dk = cuda.zeros_bf16([b, hkv, tk, d]).unwrap();
        let mut dv = cuda.zeros_bf16([b, hkv, tk, d]).unwrap();
        let mut plan = Bf16SdpaPlan::new(&cuda, spec).unwrap();
        plan.forward(&q, &k, &v, &mut o).unwrap();
        plan.backward(
            &q,
            &k,
            &v,
            &o,
            &dout,
            &mut dq,
            &mut dk,
            &mut dv,
            GradientMode::Overwrite,
        )
        .unwrap();
        write_f32_bin(
            &format!("{dir}/{tag}_o_fa.bin"),
            cuda.download_bf16(&o).unwrap().to_f32().unwrap().data(),
        );
        write_f32_bin(
            &format!("{dir}/{tag}_dq_fa.bin"),
            cuda.download_bf16(&dq).unwrap().to_f32().unwrap().data(),
        );
        write_f32_bin(
            &format!("{dir}/{tag}_dk_fa.bin"),
            cuda.download_bf16(&dk).unwrap().to_f32().unwrap().data(),
        );
        write_f32_bin(
            &format!("{dir}/{tag}_dv_fa.bin"),
            cuda.download_bf16(&dv).unwrap().to_f32().unwrap().data(),
        );
    }

    // 2. BF16 Residual+RmsNorm + SwiGLU + CrossEntropy + Embedding + Dropout
    {
        let (rows, cols) = (32, 128);
        let a = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([rows, cols], read_f32_bin(&format!("{dir}/rms_a.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let b = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([rows, cols], read_f32_bin(&format!("{dir}/rms_b.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let w = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([cols], read_f32_bin(&format!("{dir}/rms_w.bin"))).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let dy = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([rows, cols], read_f32_bin(&format!("{dir}/rms_dy.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let mut res = cuda.zeros_bf16([rows, cols]).unwrap();
        let mut y = cuda.zeros_bf16([rows, cols]).unwrap();
        let mut inv = cuda.zeros([rows]).unwrap();
        let mut dx = cuda.zeros_bf16([rows, cols]).unwrap();
        let mut dw = cuda.zeros([cols]).unwrap();
        let mut norm = Bf16NormPlan::new(&cuda, rows, cols, 1e-5).unwrap();
        norm.residual_rmsnorm_forward(&a, &b, Some(&w), &mut res, &mut y, Some(&mut inv))
            .unwrap();
        norm.rmsnorm_backward(
            &res,
            Some(&w),
            Some(&inv),
            &dy,
            &mut dx,
            Some(&mut dw),
            GradientMode::Overwrite,
            GradientMode::Overwrite,
        )
        .unwrap();
        write_f32_bin(
            &format!("{dir}/rms_y_fa.bin"),
            cuda.download_bf16(&y).unwrap().to_f32().unwrap().data(),
        );
        write_f32_bin(
            &format!("{dir}/rms_dx_fa.bin"),
            cuda.download_bf16(&dx).unwrap().to_f32().unwrap().data(),
        );
        write_f32_bin(
            &format!("{dir}/rms_dw_fa.bin"),
            cuda.download(&dw).unwrap().data(),
        );

        // SwiGLU on [32, 128] -> [32, 64]
        let sw = Bf16SwiGluPlan::new(&cuda, rows, cols / 2).unwrap();
        let mut sw_o = cuda.zeros_bf16([rows, cols / 2]).unwrap();
        let mut sw_dgu = cuda.zeros_bf16([rows, cols]).unwrap();
        let sw_do = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([rows, cols / 2], read_f32_bin(&format!("{dir}/sw_do.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        sw.forward(&a, &mut sw_o).unwrap();
        sw.backward(&a, &sw_do, &mut sw_dgu).unwrap();
        write_f32_bin(
            &format!("{dir}/sw_o_fa.bin"),
            cuda.download_bf16(&sw_o).unwrap().to_f32().unwrap().data(),
        );
        write_f32_bin(
            &format!("{dir}/sw_dgu_fa.bin"),
            cuda.download_bf16(&sw_dgu)
                .unwrap()
                .to_f32()
                .unwrap()
                .data(),
        );

        // CrossEntropy BF16 single-pass fwd+bwd
        let ce_spec = CrossEntropySpec::new(rows, cols, Some(-100), 0.1).unwrap();
        let mut ce = Bf16CrossEntropyPlan::new(&cuda, ce_spec).unwrap();
        let ce_t = cuda
            .upload_indices(
                &IndexTensor::from_vec([rows], read_i64_bin(&format!("{dir}/bf16_ce_t.bin")))
                    .unwrap(),
            )
            .unwrap();
        let mut ce_l = cuda.zeros([]).unwrap();
        let mut ce_g = cuda.zeros_bf16([rows, cols]).unwrap();
        ce.forward_backward(&a, &ce_t, 1.0, &mut ce_l, &mut ce_g)
            .unwrap();
        write_f32_bin(
            &format!("{dir}/bf16_ce_l_fa.bin"),
            cuda.download(&ce_l).unwrap().data(),
        );
        write_f32_bin(
            &format!("{dir}/bf16_ce_g_fa.bin"),
            cuda.download_bf16(&ce_g).unwrap().to_f32().unwrap().data(),
        );

        // Embedding BF16 -> FP32 grad
        let emb_idx =
            IndexTensor::from_vec([4, 16], read_i64_bin(&format!("{dir}/emb_idx.bin"))).unwrap();
        let emb_spec = EmbeddingSpec::new(32, 64, emb_idx.shape(), Some(0)).unwrap();
        let emb = Bf16EmbeddingPlan::new(&cuda, emb_spec);
        let emb_w = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([32, 64], read_f32_bin(&format!("{dir}/emb_w.bin"))).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let emb_dy = cuda
            .upload_bf16(
                &Bf16Tensor::from_f32(
                    &Tensor::from_vec([4, 16, 64], read_f32_bin(&format!("{dir}/emb_dy.bin")))
                        .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let d_emb_idx = cuda.upload_indices(&emb_idx).unwrap();
        let mut emb_o = cuda.zeros_bf16([4, 16, 64]).unwrap();
        let mut emb_dw = cuda.zeros([32, 64]).unwrap();
        emb.forward(&emb_w, &d_emb_idx, &mut emb_o).unwrap();
        emb.backward_f32(&d_emb_idx, &emb_dy, &mut emb_dw, GradientMode::Overwrite)
            .unwrap();
        write_f32_bin(
            &format!("{dir}/bf16_emb_o_fa.bin"),
            cuda.download_bf16(&emb_o).unwrap().to_f32().unwrap().data(),
        );
        write_f32_bin(
            &format!("{dir}/bf16_emb_dw_fa.bin"),
            cuda.download(&emb_dw).unwrap().data(),
        );

        // BF16 Dropout matches CPU
        let drop_spec = DropoutSpec::new(&Shape::new([rows, cols]).unwrap(), 0.25, 777).unwrap();
        let bdrop = Bf16DropoutPlan::new(&cuda, drop_spec);
        let mut drop_o = cuda.zeros_bf16([rows, cols]).unwrap();
        bdrop.forward(&a, &mut drop_o, 5).unwrap();
        write_f32_bin(
            &format!("{dir}/bf16_drop_o_fa.bin"),
            cuda.download_bf16(&drop_o)
                .unwrap()
                .to_f32()
                .unwrap()
                .data(),
        );
    }

    // 3. 30-step 6-Layer GQA Causal Audio Transformer convergence match against PyTorch autocast(bfloat16)!
    {
        let cfg = Bf16AudioTransformerConfig {
            batch: 2,
            seq_len: 128,
            vocab_size: 256,
            d_model: 256,
            num_heads: 4,
            num_kv_heads: 2,
            ffn_hidden: 512,
            num_layers: 6,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            ignore_index: None,
        };
        let shapes = Bf16AudioTransformerTrainer::parameter_shapes(&cfg).unwrap();
        let weights: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(idx, s)| {
                Tensor::from_vec(s.dims(), read_f32_bin(&format!("{dir}/tr_w_{idx}.bin"))).unwrap()
            })
            .collect();
        let optim_cfg = AdamWConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: Some(1.0),
        };
        let mut trainer =
            Bf16AudioTransformerTrainer::new(&cuda, cfg, &weights, optim_cfg).unwrap();
        let in_idx = IndexTensor::from_vec(
            [cfg.batch, cfg.seq_len],
            read_i64_bin(&format!("{dir}/tr_in.bin")),
        )
        .unwrap();
        let tgt_idx =
            IndexTensor::from_vec([cfg.tokens()], read_i64_bin(&format!("{dir}/tr_tgt.bin")))
                .unwrap();
        trainer.set_batch(&in_idx, &tgt_idx).unwrap();
        trainer.enqueue_step().unwrap();
        let mut losses = vec![trainer.download_loss().unwrap()];
        trainer.capture_step_graph().unwrap();
        for _ in 1..30 {
            trainer.step_captured().unwrap();
            losses.push(trainer.download_loss().unwrap());
        }
        write_f32_bin(&format!("{dir}/tr_losses_fa.bin"), &losses);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: audio-bf16-bench <bench-cpu|bench-cuda [fixture-dir]|bench-cuda-fallback|verify-cpu <dir>|verify-cuda <dir>>"
        );
        std::process::exit(1);
    }
    match args[1].as_str() {
        "bench-cpu" => run_bench_cpu(),
        "verify-cpu" => run_verify_cpu(&args[2]),
        #[cfg(feature = "cuda")]
        "bench-cuda" => run_bench_cuda(args.get(2).map(String::as_str)),
        #[cfg(feature = "cuda")]
        "bench-cuda-fallback" => run_bench_cuda_fallback(),
        #[cfg(feature = "cuda")]
        "verify-cuda" => run_verify_cuda(&args[2]),
        other => {
            eprintln!("unsupported mode or missing feature: {other}");
            std::process::exit(1);
        }
    }
}
