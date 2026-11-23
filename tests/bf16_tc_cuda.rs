#![cfg(feature = "cuda")]

use fast_audio::{
    AdamWConfig, Bf16Tensor, CrossEntropySpec, DropoutSpec, EmbeddingSpec, GradientMode,
    IndexTensor, Layout, PrecisionPolicy, RopeSpec, SdpaOperand, SdpaSpec, Shape, Tensor, cpu,
    cuda::{
        Bf16AudioTransformerConfig, Bf16AudioTransformerTrainer, Bf16CrossEntropyPlan,
        Bf16DropoutPlan, Bf16EmbeddingPlan, Bf16LinearPlan, Bf16MasterAdamW, Bf16NormPlan,
        Bf16RopePlan, Bf16SdpaPlan, Bf16SdpaStrides, Bf16SoftmaxPlan, Bf16SwiGluPlan, Cuda,
        CudaGradScaler, Tf32MatmulPlan,
    },
};

fn max_and_mean_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max_e = 0.0f32;
    let mut sum_e = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite value {x} vs {y}"
        );
        let e = (x - y).abs();
        if e > max_e {
            max_e = e;
        }
        sum_e += e as f64;
    }
    (max_e, (sum_e / a.len().max(1) as f64) as f32)
}

#[test]
fn precision_policy_isolation_and_bf16_cast_parity() {
    let strict = Cuda::new(0).unwrap();
    assert_eq!(strict.precision_policy(), PrecisionPolicy::StrictFp32);
    assert!(Tf32MatmulPlan::new(&strict, 16, 16, 16).is_err());

    let tf32 = Cuda::with_precision(0, PrecisionPolicy::Tf32).unwrap();
    assert_eq!(tf32.precision_policy(), PrecisionPolicy::Tf32);
    let plan = Tf32MatmulPlan::new(&tf32, 32, 32, 32).unwrap();
    let a = Tensor::from_vec(
        [32, 32],
        (0..1024).map(|i| ((i as f32 * 0.13).sin()) * 0.5).collect(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        [32, 32],
        (0..1024).map(|i| ((i as f32 * 0.17).cos()) * 0.5).collect(),
    )
    .unwrap();
    let da = tf32.upload(&a).unwrap();
    let db = tf32.upload(&b).unwrap();
    let mut dc = tf32.zeros([32, 32]).unwrap();
    plan.run(&da, &db, &mut dc).unwrap();
    let mut dc_strict = strict.zeros([32, 32]).unwrap();
    strict
        .matmul(
            &strict.upload(&a).unwrap(),
            &strict.upload(&b).unwrap(),
            &mut dc_strict,
        )
        .unwrap();
    let (max_e, mean_e) = max_and_mean_err(
        tf32.download(&dc).unwrap().data(),
        strict.download(&dc_strict).unwrap().data(),
    );
    assert!(
        max_e < 2e-2 && mean_e < 3e-3,
        "tf32 err max={max_e} mean={mean_e}"
    );

    // F32 <-> BF16 cast matches CPU IEEE RNE bit-for-bit!
    let mixed = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();
    let vals: Vec<f32> = (0..1027)
        .map(|i| ((i as f32 * 0.37).sin()) * 3.5 + 0.001 * i as f32)
        .collect();
    let host_f32 = Tensor::from_vec([1027], vals).unwrap();
    let host_bf16 = Bf16Tensor::from_f32(&host_f32).unwrap();
    let dev_f32 = mixed.upload(&host_f32).unwrap();
    let mut dev_bf16 = mixed.zeros_bf16([1027]).unwrap();
    mixed.cast_f32_to_bf16(&dev_f32, &mut dev_bf16).unwrap();
    let gpu_bf16 = mixed.download_bf16(&dev_bf16).unwrap();
    assert_eq!(gpu_bf16.bits(), host_bf16.bits());
    let mut dev_back = mixed.zeros([1027]).unwrap();
    mixed.cast_bf16_to_f32(&dev_bf16, &mut dev_back).unwrap();
    assert_eq!(
        mixed.download(&dev_back).unwrap().data(),
        host_bf16.to_f32().unwrap().data()
    );
}

#[test]
fn bf16_linear_and_bmm_tensor_core_parity() {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();
    let (m, k, n) = (64, 128, 96);
    let x = Tensor::from_vec(
        [m, k],
        (0..m * k)
            .map(|i| ((i as f32 * 0.11).sin()) * 0.4)
            .collect(),
    )
    .unwrap();
    let w = Tensor::from_vec(
        [n, k],
        (0..n * k)
            .map(|i| ((i as f32 * 0.19).cos()) * 0.3)
            .collect(),
    )
    .unwrap();
    let dy = Tensor::from_vec(
        [m, n],
        (0..m * n)
            .map(|i| ((i as f32 * 0.23).sin()) * 0.5)
            .collect(),
    )
    .unwrap();

    let dx_bf = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&x).unwrap())
        .unwrap();
    let dw_bf = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&w).unwrap())
        .unwrap();
    let ddy_bf = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&dy).unwrap())
        .unwrap();
    let mut y_bf = cuda.zeros_bf16([m, n]).unwrap();
    let mut dx_out_bf = cuda.zeros_bf16([m, k]).unwrap();
    let mut dw_out_f32 = cuda.zeros([n, k]).unwrap();

    let lin = Bf16LinearPlan::new(&cuda, m, k, n).unwrap();
    lin.forward(&dx_bf, &dw_bf, &mut y_bf).unwrap();
    lin.backward_input(&ddy_bf, &dw_bf, &mut dx_out_bf, GradientMode::Overwrite)
        .unwrap();
    lin.backward_weight_f32(&ddy_bf, &dx_bf, &mut dw_out_f32, GradientMode::Overwrite)
        .unwrap();

    let y_gpu = cuda.download_bf16(&y_bf).unwrap().to_f32().unwrap();
    let dx_gpu = cuda.download_bf16(&dx_out_bf).unwrap().to_f32().unwrap();
    let dw_gpu = cuda.download(&dw_out_f32).unwrap();

    // Exact FP32 host reference
    let mut y_ref = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut s = 0.0f32;
            for j in 0..k {
                s += x.data()[r * k + j] * w.data()[c * k + j];
            }
            y_ref[r * n + c] = s;
        }
    }
    let mut dx_ref = vec![0.0f32; m * k];
    for r in 0..m {
        for j in 0..k {
            let mut s = 0.0f32;
            for c in 0..n {
                s += dy.data()[r * n + c] * w.data()[c * k + j];
            }
            dx_ref[r * k + j] = s;
        }
    }
    let mut dw_ref = vec![0.0f32; n * k];
    for c in 0..n {
        for j in 0..k {
            let mut s = 0.0f32;
            for r in 0..m {
                s += dy.data()[r * n + c] * x.data()[r * k + j];
            }
            dw_ref[c * k + j] = s;
        }
    }
    let (ey, my) = max_and_mean_err(y_gpu.data(), &y_ref);
    let (edx, mdx) = max_and_mean_err(dx_gpu.data(), &dx_ref);
    let (edw, mdw) = max_and_mean_err(dw_gpu.data(), &dw_ref);
    assert!(ey < 3e-2 && my < 5e-3, "y err max={ey} mean={my}");
    assert!(edx < 3e-2 && mdx < 5e-3, "dx err max={edx} mean={mdx}");
    assert!(edw < 3e-2 && mdw < 5e-3, "dw err max={edw} mean={mdw}");
}

#[test]
fn bf16_sdpa_tensor_core_flash_attention2_all_modes() {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();
    // Cover Tensor Core and general widths, both masks and partial sequence tiles.
    let cases = [
        (2, 4, 4, 128, 128, 64, true),
        (2, 4, 4, 128, 128, 64, false),
        (2, 8, 2, 73, 97, 64, true),
        (2, 8, 2, 73, 97, 64, false),
        (1, 4, 2, 128, 128, 128, true),
        (1, 4, 4, 71, 69, 128, false),
        (1, 4, 2, 19, 23, 48, true),
        (1, 2, 2, 17, 13, 80, false),
        (1, 2, 1, 11, 7, 129, true),
    ];

    for (b, hq, hkv, tq, tk, d, causal) in cases {
        let spec = SdpaSpec::new(b, hq, hkv, tq, tk, d, None, causal).unwrap();
        let q = Tensor::from_vec(
            [b, hq, tq, d],
            (0..b * hq * tq * d)
                .map(|i| ((i as f32 * 0.13).sin()) * 0.6)
                .collect(),
        )
        .unwrap();
        let k = Tensor::from_vec(
            [b, hkv, tk, d],
            (0..b * hkv * tk * d)
                .map(|i| ((i as f32 * 0.17).cos()) * 0.6)
                .collect(),
        )
        .unwrap();
        let v = Tensor::from_vec(
            [b, hkv, tk, d],
            (0..b * hkv * tk * d)
                .map(|i| ((i as f32 * 0.23).sin()) * 0.5)
                .collect(),
        )
        .unwrap();
        let dout = Tensor::from_vec(
            [b, hq, tq, d],
            (0..b * hq * tq * d)
                .map(|i| ((i as f32 * 0.29).cos()) * 0.5)
                .collect(),
        )
        .unwrap();

        let lq = Layout::contiguous([b, hq, tq, d]).unwrap();
        let lk = Layout::contiguous([b, hkv, tk, d]).unwrap();
        let mut o_cpu = Tensor::zeros([b, hq, tq, d]).unwrap();
        cpu::SdpaPlan::new(spec, &lq, &lk, &lk, &lq)
            .unwrap()
            .run(
                &q.view(&lq).unwrap(),
                &k.view(&lk).unwrap(),
                &v.view(&lk).unwrap(),
                &mut o_cpu.view_mut(&lq).unwrap(),
            )
            .unwrap();
        let mut dq_cpu = Tensor::zeros([b, hq, tq, d]).unwrap();
        let mut dk_cpu = Tensor::zeros([b, hkv, tk, d]).unwrap();
        let mut dv_cpu = Tensor::zeros([b, hkv, tk, d]).unwrap();
        cpu::SdpaBackward::new(SdpaOperand::Query, spec, &lq, &lk, &lk, &lq, &lq)
            .unwrap()
            .run(
                Some(&q.view(&lq).unwrap()),
                Some(&k.view(&lk).unwrap()),
                Some(&v.view(&lk).unwrap()),
                &dout.view(&lq).unwrap(),
                &mut dq_cpu.view_mut(&lq).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        cpu::SdpaBackward::new(SdpaOperand::Key, spec, &lq, &lk, &lk, &lq, &lk)
            .unwrap()
            .run(
                Some(&q.view(&lq).unwrap()),
                Some(&k.view(&lk).unwrap()),
                Some(&v.view(&lk).unwrap()),
                &dout.view(&lq).unwrap(),
                &mut dk_cpu.view_mut(&lk).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        cpu::SdpaBackward::new(SdpaOperand::Value, spec, &lq, &lk, &lk, &lq, &lk)
            .unwrap()
            .run(
                Some(&q.view(&lq).unwrap()),
                Some(&k.view(&lk).unwrap()),
                Some(&v.view(&lk).unwrap()),
                &dout.view(&lq).unwrap(),
                &mut dv_cpu.view_mut(&lk).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();

        let dq_bf = cuda
            .upload_bf16(&Bf16Tensor::from_f32(&q).unwrap())
            .unwrap();
        let dk_bf = cuda
            .upload_bf16(&Bf16Tensor::from_f32(&k).unwrap())
            .unwrap();
        let dv_bf = cuda
            .upload_bf16(&Bf16Tensor::from_f32(&v).unwrap())
            .unwrap();
        let ddo_bf = cuda
            .upload_bf16(&Bf16Tensor::from_f32(&dout).unwrap())
            .unwrap();
        let mut do_bf = cuda.zeros_bf16([b, hq, tq, d]).unwrap();
        let mut gdq_bf = cuda.zeros_bf16([b, hq, tq, d]).unwrap();
        let mut gdk_bf = cuda.zeros_bf16([b, hkv, tk, d]).unwrap();
        let mut gdv_bf = cuda.zeros_bf16([b, hkv, tk, d]).unwrap();

        let mut plan = Bf16SdpaPlan::new(&cuda, spec).unwrap();
        // Verify strictly O(T) workspace (no T_q * T_k quadratic matrix!)
        if hq == hkv || !matches!(d, 64 | 128) {
            assert_eq!(plan.workspace_bytes(), b * hq * tq * 8);
        }
        plan.forward(&dq_bf, &dk_bf, &dv_bf, &mut do_bf).unwrap();
        plan.backward(
            &dq_bf,
            &dk_bf,
            &dv_bf,
            &do_bf,
            &ddo_bf,
            &mut gdq_bf,
            &mut gdk_bf,
            &mut gdv_bf,
            GradientMode::Overwrite,
        )
        .unwrap();

        let o_gpu = cuda.download_bf16(&do_bf).unwrap().to_f32().unwrap();
        let dq_gpu = cuda.download_bf16(&gdq_bf).unwrap().to_f32().unwrap();
        let dk_gpu = cuda.download_bf16(&gdk_bf).unwrap().to_f32().unwrap();
        let dv_gpu = cuda.download_bf16(&gdv_bf).unwrap().to_f32().unwrap();

        let (eo, mo) = max_and_mean_err(o_gpu.data(), o_cpu.data());
        let (edq, mdq) = max_and_mean_err(dq_gpu.data(), dq_cpu.data());
        let (edk, mdk) = max_and_mean_err(dk_gpu.data(), dk_cpu.data());
        let (edv, mdv) = max_and_mean_err(dv_gpu.data(), dv_cpu.data());
        assert!(
            eo < 3e-2 && mo < 5e-3,
            "SDPA O mismatch ({b},{hq},{hkv},{tq},{tk},{d},{causal}): max={eo} mean={mo}"
        );
        assert!(
            edq < 4e-2 && mdq < 6e-3,
            "SDPA dQ mismatch ({b},{hq},{hkv},{tq},{tk},{d},{causal}): max={edq} mean={mdq}"
        );
        assert!(
            edk < 4e-2 && mdk < 6e-3,
            "SDPA dK mismatch ({b},{hq},{hkv},{tq},{tk},{d},{causal}): max={edk} mean={mdk}"
        );
        assert!(
            edv < 4e-2 && mdv < 6e-3,
            "SDPA dV mismatch ({b},{hq},{hkv},{tq},{tk},{d},{causal}): max={edv} mean={mdv}"
        );
    }

    // Also test zero-copy [B, T, H, D] strided view!
    let (b, h, t, d) = (2, 4, 64, 64);
    let spec = SdpaSpec::new(b, h, h, t, t, d, None, true).unwrap();
    let q_bthd = Tensor::from_vec(
        [b, t, h, d],
        (0..b * t * h * d)
            .map(|i| ((i as f32 * 0.11).sin()) * 0.5)
            .collect(),
    )
    .unwrap();
    let k_bthd = Tensor::from_vec(
        [b, t, h, d],
        (0..b * t * h * d)
            .map(|i| ((i as f32 * 0.19).cos()) * 0.5)
            .collect(),
    )
    .unwrap();
    let v_bthd = Tensor::from_vec(
        [b, t, h, d],
        (0..b * t * h * d)
            .map(|i| ((i as f32 * 0.27).sin()) * 0.5)
            .collect(),
    )
    .unwrap();
    let dq = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&q_bthd).unwrap())
        .unwrap();
    let dk = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&k_bthd).unwrap())
        .unwrap();
    let dv = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&v_bthd).unwrap())
        .unwrap();
    let mut dout = cuda.zeros_bf16([b, t, h, d]).unwrap();
    let s = Bf16SdpaStrides::bthd(t, h, d);
    let mut plan = Bf16SdpaPlan::new(&cuda, spec).unwrap();
    plan.forward_strided(&dq, 0, s, &dk, 0, s, &dv, 0, s, &mut dout, 0, s)
        .unwrap();
    let out_gpu = cuda.download_bf16(&dout).unwrap().to_f32().unwrap();
    assert!(out_gpu.data().iter().all(|v| v.is_finite()));
}

#[test]
fn bf16_sdpa_strided_fallback_checks_views_and_accumulates() {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();
    let (q_dims, kv_dims) = ([1, 2, 3, 7], [1, 1, 5, 7]);
    let reverse_rows = |values: &[f32]| {
        values
            .chunks_exact(7)
            .flat_map(|row| row.iter().rev().copied())
            .collect::<Vec<_>>()
    };
    let inputs = |dims: [usize; 4], seed| {
        let values: Vec<_> = (0..dims.into_iter().product::<usize>())
            .map(|i| ((i * seed % 31) as f32 - 15.0) / 64.0)
            .collect();
        let upload = |values| {
            let host = Tensor::from_vec(dims, values).unwrap();
            cuda.upload_bf16(&Bf16Tensor::from_f32(&host).unwrap())
                .unwrap()
        };
        let reversed = reverse_rows(&values);
        (upload(values), upload(reversed))
    };
    let (q, qr) = inputs(q_dims, 7);
    let (k, kr) = inputs(kv_dims, 9);
    let (v, vr) = inputs(kv_dims, 13);
    let (dout, doutr) = inputs(q_dims, 15);
    let mut o = cuda.zeros_bf16(q_dims).unwrap();
    let mut dq = cuda.zeros_bf16(q_dims).unwrap();
    let mut dk = cuda.zeros_bf16(kv_dims).unwrap();
    let mut dv = cuda.zeros_bf16(kv_dims).unwrap();
    let mut or = cuda.zeros_bf16(q_dims).unwrap();
    let mut dqr = cuda.zeros_bf16(q_dims).unwrap();
    let mut dkr = cuda.zeros_bf16(kv_dims).unwrap();
    let mut dvr = cuda.zeros_bf16(kv_dims).unwrap();
    let q_strides = Bf16SdpaStrides {
        sd: -1,
        ..Bf16SdpaStrides::bhtd(2, 3, 7)
    };
    let kv_strides = Bf16SdpaStrides {
        sd: -1,
        ..Bf16SdpaStrides::bhtd(1, 5, 7)
    };
    let overlap = Bf16SdpaStrides { sh: 0, ..q_strides };
    let spec = SdpaSpec::new(1, 2, 1, 3, 5, 7, None, true).unwrap();
    let mut plan = Bf16SdpaPlan::new(&cuda, spec).unwrap();
    assert!(
        plan.forward_strided(
            &qr, 0, q_strides, &kr, 6, kv_strides, &vr, 6, kv_strides, &mut or, 6, q_strides
        )
        .is_err()
    );
    assert!(
        plan.forward_strided(
            &qr, 6, q_strides, &kr, 6, kv_strides, &vr, 6, kv_strides, &mut or, 6, overlap
        )
        .is_err()
    );
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
    plan.forward_strided(
        &qr, 6, q_strides, &kr, 6, kv_strides, &vr, 6, kv_strides, &mut or, 6, q_strides,
    )
    .unwrap();
    assert!(
        plan.backward_strided(
            &qr,
            6,
            q_strides,
            &kr,
            6,
            kv_strides,
            &vr,
            6,
            kv_strides,
            &or,
            6,
            q_strides,
            &doutr,
            6,
            q_strides,
            &mut dqr,
            6,
            overlap,
            &mut dkr,
            6,
            kv_strides,
            &mut dvr,
            6,
            kv_strides,
            GradientMode::Overwrite
        )
        .is_err()
    );
    for mode in [GradientMode::Overwrite, GradientMode::Accumulate] {
        plan.backward_strided(
            &qr, 6, q_strides, &kr, 6, kv_strides, &vr, 6, kv_strides, &or, 6, q_strides, &doutr,
            6, q_strides, &mut dqr, 6, q_strides, &mut dkr, 6, kv_strides, &mut dvr, 6, kv_strides,
            mode,
        )
        .unwrap();
        for (plain, reversed) in [(&o, &or), (&dq, &dqr), (&dk, &dkr), (&dv, &dvr)] {
            let saved = cuda.download_bf16(plain).unwrap().to_f32().unwrap();
            let got = cuda.download_bf16(reversed).unwrap().to_f32().unwrap();
            let scale = if mode == GradientMode::Accumulate && !std::ptr::eq(plain, &o) {
                2.0
            } else {
                1.0
            };
            let want: Vec<_> = reverse_rows(saved.data())
                .into_iter()
                .map(|v| v * scale)
                .collect();
            let (max, _) = max_and_mean_err(got.data(), &want);
            assert!(max < 1e-3, "strided fallback error: {max}");
        }
    }
}

#[test]
fn bf16_kernels_norm_softmax_swiglu_rope_embedding_ce_dropout_and_grad_scaler() {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();

    // 1. Fused ResidualAdd + RmsNorm + backward (with dWeight)
    let (rows, cols) = (32, 64);
    let a = Tensor::from_vec(
        [rows, cols],
        (0..rows * cols)
            .map(|i| ((i as f32 * 0.13).sin()) * 0.8)
            .collect(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        [rows, cols],
        (0..rows * cols)
            .map(|i| ((i as f32 * 0.21).cos()) * 0.4)
            .collect(),
    )
    .unwrap();
    let w = Tensor::from_vec([cols], (0..cols).map(|i| 0.8 + 0.01 * i as f32).collect()).unwrap();
    let dy = Tensor::from_vec(
        [rows, cols],
        (0..rows * cols)
            .map(|i| ((i as f32 * 0.17).sin()) * 0.5)
            .collect(),
    )
    .unwrap();

    let da = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&a).unwrap())
        .unwrap();
    let db = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&b).unwrap())
        .unwrap();
    let dw = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&w).unwrap())
        .unwrap();
    let ddy = cuda
        .upload_bf16(&Bf16Tensor::from_f32(&dy).unwrap())
        .unwrap();
    let mut dres = cuda.zeros_bf16([rows, cols]).unwrap();
    let mut dy_out = cuda.zeros_bf16([rows, cols]).unwrap();
    let mut inv_rms = cuda.zeros([rows]).unwrap();
    let mut ddx = cuda.zeros_bf16([rows, cols]).unwrap();
    let mut ddw = cuda.zeros([cols]).unwrap();

    let mut norm = Bf16NormPlan::new(&cuda, rows, cols, 1e-5).unwrap();
    norm.residual_rmsnorm_forward(
        &da,
        &db,
        Some(&dw),
        &mut dres,
        &mut dy_out,
        Some(&mut inv_rms),
    )
    .unwrap();
    norm.rmsnorm_backward(
        &dres,
        Some(&dw),
        Some(&inv_rms),
        &ddy,
        &mut ddx,
        Some(&mut ddw),
        GradientMode::Overwrite,
        GradientMode::Overwrite,
    )
    .unwrap();
    assert!(
        cuda.download_bf16(&dy_out)
            .unwrap()
            .to_f32()
            .unwrap()
            .data()
            .iter()
            .all(|v| v.is_finite())
    );
    assert!(
        cuda.download(&ddw)
            .unwrap()
            .data()
            .iter()
            .all(|v| v.is_finite())
    );

    // 2. LayerNorm + Softmax + SwiGLU + RoPE + Embedding + Single-Pass CrossEntropy + Dropout
    let mut ln_out = cuda.zeros_bf16([rows, cols]).unwrap();
    norm.layernorm_forward(&da, &mut ln_out).unwrap();
    norm.layernorm_backward(&da, &ddy, &mut ddx, GradientMode::Overwrite)
        .unwrap();

    let sm = Bf16SoftmaxPlan::new(&cuda, rows, cols).unwrap();
    sm.forward(&da, &mut ln_out).unwrap();
    sm.backward(&ln_out, &ddy, &mut ddx, GradientMode::Overwrite)
        .unwrap();

    let sw = Bf16SwiGluPlan::new(&cuda, rows, cols / 2).unwrap();
    let mut sw_out = cuda.zeros_bf16([rows, cols / 2]).unwrap();
    let mut sw_dgu = cuda.zeros_bf16([rows, cols]).unwrap();
    sw.forward(&da, &mut sw_out).unwrap();
    sw.backward(&da, &sw_out, &mut sw_dgu).unwrap();

    let r_shape = Shape::new([2, 4, 16, 16]).unwrap();
    let r_spec = RopeSpec::new(&r_shape, 2, Some(16), 10_000.0, 0).unwrap();
    let rope = Bf16RopePlan::new(&cuda, r_spec).unwrap();
    let rx = cuda
        .upload_bf16(
            &Bf16Tensor::from_f32(&Tensor::from_vec([2, 4, 16, 16], vec![0.5; 2048]).unwrap())
                .unwrap(),
        )
        .unwrap();
    let mut ry = cuda.zeros_bf16([2, 4, 16, 16]).unwrap();
    let mut rback = cuda.zeros_bf16([2, 4, 16, 16]).unwrap();
    rope.forward(&rx, &mut ry).unwrap();
    rope.backward(&ry, &mut rback, GradientMode::Overwrite)
        .unwrap();
    let (re, _) = max_and_mean_err(
        cuda.download_bf16(&rback).unwrap().to_f32().unwrap().data(),
        &vec![0.5; 2048],
    );
    assert!(re < 2e-2, "bf16 rope orthogonality err={re}");

    // Embedding BF16 -> FP32 master grad
    let idx = IndexTensor::from_vec([2, 8], (0..16).map(|i| (i * 3) % 10).collect()).unwrap();
    let d_idx = cuda.upload_indices(&idx).unwrap();
    let emb_spec = EmbeddingSpec::new(10, 32, idx.shape(), Some(0)).unwrap();
    let emb = Bf16EmbeddingPlan::new(&cuda, emb_spec);
    let emb_w = cuda
        .upload_bf16(
            &Bf16Tensor::from_f32(
                &Tensor::from_vec([10, 32], (0..320).map(|i| i as f32 * 0.01).collect()).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let mut emb_o = cuda.zeros_bf16([2, 8, 32]).unwrap();
    let mut emb_dw = cuda.zeros([10, 32]).unwrap();
    emb.forward(&emb_w, &d_idx, &mut emb_o).unwrap();
    emb.backward_f32(&d_idx, &emb_o, &mut emb_dw, GradientMode::Overwrite)
        .unwrap();
    assert_eq!(&cuda.download(&emb_dw).unwrap().data()[0..32], &[0.0; 32]);

    // Single-pass CrossEntropy fwd_bwd == separate fwd + bwd!
    let ce_spec = CrossEntropySpec::new(rows, cols, Some(-100), 0.1).unwrap();
    let mut ce = Bf16CrossEntropyPlan::new(&cuda, ce_spec).unwrap();
    let ce_t = cuda
        .upload_indices(
            &IndexTensor::from_vec(
                [rows],
                (0..rows)
                    .map(|i| if i == 3 { -100 } else { (i * 5 % cols) as i64 })
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let mut loss1 = cuda.zeros([]).unwrap();
    let mut loss2 = cuda.zeros([]).unwrap();
    let mut grad1 = cuda.zeros_bf16([rows, cols]).unwrap();
    let mut grad2 = cuda.zeros_bf16([rows, cols]).unwrap();
    ce.forward(&da, &ce_t, &mut loss1).unwrap();
    ce.backward(&da, &ce_t, 1.0, &mut grad1, GradientMode::Overwrite)
        .unwrap();
    ce.forward_backward(&da, &ce_t, 1.0, &mut loss2, &mut grad2)
        .unwrap();
    assert_eq!(
        cuda.download(&loss1).unwrap().data(),
        cuda.download(&loss2).unwrap().data()
    );
    assert_eq!(
        cuda.download_bf16(&grad1).unwrap().bits(),
        cuda.download_bf16(&grad2).unwrap().bits()
    );

    // BF16 Dropout matches CPU Philox mask
    let d_spec = DropoutSpec::new(&Shape::new([rows, cols]).unwrap(), 0.25, 999).unwrap();
    let bdrop = Bf16DropoutPlan::new(&cuda, d_spec.clone());
    let mut drop_o = cuda.zeros_bf16([rows, cols]).unwrap();
    bdrop.forward(&da, &mut drop_o, 3).unwrap();
    let mut drop_dx = cuda.zeros_bf16([rows, cols]).unwrap();
    bdrop
        .backward(&ddy, &mut drop_dx, 3, GradientMode::Overwrite)
        .unwrap();
    let lo_cpu = Layout::contiguous([rows, cols]).unwrap();
    let mut y_cpu = Tensor::zeros([rows, cols]).unwrap();
    cpu::DropoutPlan::new(&d_spec, &lo_cpu, &lo_cpu)
        .unwrap()
        .run(
            &a.view(&lo_cpu).unwrap(),
            &mut y_cpu.view_mut(&lo_cpu).unwrap(),
            3,
        )
        .unwrap();
    let drop_gpu = cuda.download_bf16(&drop_o).unwrap().to_f32().unwrap();
    for i in 0..rows * cols {
        assert_eq!(drop_gpu.data()[i] == 0.0, y_cpu.data()[i] == 0.0);
    }

    // 3. Device-side GradScaler + Bf16MasterAdamW zero-host-sync overflow skip!
    let scaler = CudaGradScaler::new(&cuda, 1024.0, 2.0, 0.5, 2).unwrap();
    let mut master = vec![
        cuda.upload(&Tensor::from_vec([16], vec![1.0; 16]).unwrap())
            .unwrap(),
    ];
    let mut mirror = vec![
        cuda.upload_bf16(
            &Bf16Tensor::from_f32(&Tensor::from_vec([16], vec![1.0; 16]).unwrap()).unwrap(),
        )
        .unwrap(),
    ];
    let mut grad_bad = vec![
        cuda.upload(&Tensor::from_vec([16], vec![f32::INFINITY; 16]).unwrap())
            .unwrap(),
    ];
    let mut opt =
        Bf16MasterAdamW::new(&cuda, &[Shape::new([16]).unwrap()], AdamWConfig::default()).unwrap();

    scaler.reset_step().unwrap();
    scaler.unscale_and_check(&mut grad_bad[0]).unwrap();
    opt.step(&mut master, &mut mirror, &grad_bad, Some(&scaler))
        .unwrap();
    scaler.update().unwrap();
    let (sc, found_inf) = scaler.read_state().unwrap();
    assert!(
        found_inf && (sc - 512.0).abs() < 1e-6,
        "expected backoff 512, got {sc} inf={found_inf}"
    );
    // Master weights MUST be unchanged when overflow occurred!
    assert_eq!(cuda.download(&master[0]).unwrap().data(), &[1.0; 16]);

    // Two clean steps -> scale grows back 512 -> 1024 and weights update!
    for _ in 0..2 {
        let mut grad_ok = vec![
            cuda.upload(&Tensor::from_vec([16], vec![512.0; 16]).unwrap())
                .unwrap(),
        ];
        scaler.reset_step().unwrap();
        scaler.unscale_and_check(&mut grad_ok[0]).unwrap();
        opt.step(&mut master, &mut mirror, &grad_ok, Some(&scaler))
            .unwrap();
        scaler.update().unwrap();
    }
    let (sc2, found_inf2) = scaler.read_state().unwrap();
    assert!(
        !found_inf2 && (sc2 - 1024.0).abs() < 1e-6,
        "expected growth 1024, got {sc2}"
    );
    assert!(cuda.download(&master[0]).unwrap().data()[0] < 1.0);
}

#[test]
fn bf16_cross_entropy_preserves_offset_and_rejects_short_buffers() {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();
    let targets = cuda
        .upload_indices(&IndexTensor::from_vec([1], vec![2]).unwrap())
        .unwrap();
    let mut loss = cuda.zeros([]).unwrap();
    for vocab in [7, 16] {
        let host = Tensor::from_vec([1, vocab], vec![1e20; vocab]).unwrap();
        let logits = cuda
            .upload_bf16(&Bf16Tensor::from_f32(&host).unwrap())
            .unwrap();
        let mut grad = cuda.zeros_bf16([1, vocab]).unwrap();
        let mut short_grad = cuda.zeros_bf16([1, vocab - 1]).unwrap();
        for smoothing in [0.0, 0.1] {
            let spec = CrossEntropySpec::new(1, vocab, None, smoothing).unwrap();
            let mut plan = Bf16CrossEntropyPlan::new(&cuda, spec).unwrap();
            assert!(plan.forward(&short_grad, &targets, &mut loss).is_err());
            assert!(
                plan.forward_backward(&logits, &targets, 1.0, &mut loss, &mut short_grad)
                    .is_err()
            );
            plan.forward_backward(&logits, &targets, 1.0, &mut loss, &mut grad)
                .unwrap();
            let actual = cuda.download(&loss).unwrap().data()[0];
            assert!((actual - (vocab as f32).ln()).abs() < 1e-6);
        }
    }
}

#[test]
fn bf16_audio_transformer_cuda_graph_matches_eager_with_dynamic_lr() {
    let cuda = Cuda::with_precision(0, PrecisionPolicy::Bf16Mixed).unwrap();
    let cfg = Bf16AudioTransformerConfig {
        batch: 2,
        seq_len: 64,
        vocab_size: 128,
        d_model: 128,
        num_heads: 4,
        num_kv_heads: 2,
        ffn_hidden: 256,
        num_layers: 2,
        rope_base: 10_000.0,
        rms_eps: 1e-5,
        ignore_index: None,
    };
    let shapes = Bf16AudioTransformerTrainer::parameter_shapes(&cfg).unwrap();
    let mut weights = Vec::with_capacity(shapes.len());
    for (idx, s) in shapes.iter().enumerate() {
        let data: Vec<f32> = if s.dims().len() == 1 {
            vec![1.0; s.len()]
        } else {
            (0..s.len())
                .map(|i| (((i + idx * 131) as f32 * 0.17).sin()) * 0.04)
                .collect()
        };
        weights.push(Tensor::from_vec(s.dims(), data).unwrap());
    }
    let optim_cfg = AdamWConfig {
        lr: 3e-3,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weight_decay: 0.01,
        max_grad_norm: Some(1.0),
    };
    let mut trainer = Bf16AudioTransformerTrainer::new(&cuda, cfg, &weights, optim_cfg).unwrap();
    let mut eager = Bf16AudioTransformerTrainer::new(&cuda, cfg, &weights, optim_cfg).unwrap();

    // Repeating audio-token pattern so loss cleanly drops over 15 steps
    let tokens: Vec<i64> = (0..cfg.tokens())
        .map(|i| ((i * 7 + (i / 16) * 3) % cfg.vocab_size) as i64)
        .collect();
    let targets: Vec<i64> = (0..cfg.tokens())
        .map(|i| (((i + 1) * 7 + (i / 16) * 3) % cfg.vocab_size) as i64)
        .collect();
    let in_idx = IndexTensor::from_vec([cfg.batch, cfg.seq_len], tokens).unwrap();
    let tgt_idx = IndexTensor::from_vec([cfg.tokens()], targets).unwrap();
    trainer.set_batch(&in_idx, &tgt_idx).unwrap();
    eager.set_batch(&in_idx, &tgt_idx).unwrap();

    // Warmup 1 eager step then capture ONE single CUDA Graph for the entire step!
    trainer.enqueue_step().unwrap();
    eager.enqueue_step().unwrap();
    let loss0 = trainer.download_loss().unwrap();
    assert_eq!(loss0, eager.download_loss().unwrap());
    trainer.capture_step_graph().unwrap();
    assert!(trainer.has_captured_graph());

    let mut losses = vec![loss0];
    for step in 1..15 {
        if step == 8 {
            // Dynamic LR change WITHOUT recapturing CUDA Graph!
            trainer.set_lr(1.5e-3).unwrap();
            eager.set_lr(1.5e-3).unwrap();
        }
        trainer.step_captured().unwrap();
        eager.enqueue_step().unwrap();
        let captured_loss = trainer.download_loss().unwrap();
        let eager_loss = eager.download_loss().unwrap();
        assert_eq!(captured_loss, eager_loss, "loss mismatch at step {step}");
        losses.push(captured_loss);
    }
    assert!(trainer.has_captured_graph());
    assert!(
        losses.iter().all(|v| v.is_finite()),
        "non-finite losses: {losses:?}"
    );
    assert!(
        *losses.last().unwrap() < losses[0],
        "Transformer failed to converge: start={} end={} all={losses:?}",
        losses[0],
        losses.last().unwrap()
    );
}
