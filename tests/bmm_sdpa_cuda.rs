#![cfg(feature = "cuda")]

#[allow(dead_code)]
mod support;

use fast_audio::{
    BinaryOp, BinaryOperand, BmmSpec, GradientMode, Layout, ReduceOp, SdpaOperand, SdpaSpec, Shape,
    Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{self, Cpu},
    cuda::{self, Cuda},
};

fn det(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..len)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u32 << 24) as f32;
            (u - 0.5) * 1.5
        })
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label} len mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let err = (x - y).abs();
        assert!(
            err <= tol,
            "{label}[{i}]: cuda={x} cpu={y} err={err} > {tol}"
        );
    }
}

#[test]
fn cuda_bmm_and_sdpa_all_routes_match_cpu() {
    let cuda = Cuda::new(0).unwrap();

    // 1. BMM: collapsed2D, strided-batched, cross-broadcast tiled, transposed
    for &(la_dims, lb_dims) in &[
        (&[2, 4, 8, 12][..], &[12, 6][..]),     // collapsed 2D fwd, dX, dW
        (&[2, 1, 7, 9][..], &[1, 3, 9, 5][..]), // cross-broadcast tiled reduction
        (&[2, 4, 17, 16][..], &[2, 4, 16, 17][..]), // strided batched cuBLAS
    ] {
        let sa = Shape::new(la_dims).unwrap();
        let sb = Shape::new(lb_dims).unwrap();
        let spec = BmmSpec::new(&sa, &sb).unwrap();
        let la = Layout::contiguous(la_dims).unwrap();
        let lb = Layout::contiguous(lb_dims).unwrap();
        let lc = Layout::contiguous(spec.output_shape().dims()).unwrap();

        let a_h = Tensor::from_vec(la_dims, det(sa.len(), 11)).unwrap();
        let b_h = Tensor::from_vec(lb_dims, det(sb.len(), 23)).unwrap();
        let dy_h = Tensor::from_vec(
            spec.output_shape().dims(),
            det(spec.output_shape().len(), 37),
        )
        .unwrap();
        let mut c_cpu = Tensor::zeros(spec.output_shape().dims()).unwrap();
        let mut da_cpu = Tensor::zeros(la_dims).unwrap();
        let mut db_cpu = Tensor::zeros(lb_dims).unwrap();

        cpu::BatchedMatmulPlan::new(Cpu::Reference, &spec, &la, &lb, &lc)
            .unwrap()
            .run(
                &a_h.view(&la).unwrap(),
                &b_h.view(&lb).unwrap(),
                &mut c_cpu.view_mut(&lc).unwrap(),
            )
            .unwrap();
        cpu::BatchedMatmulBackward::new(
            Cpu::Reference,
            BinaryOperand::Left,
            &spec,
            &la,
            &lb,
            &lc,
            &la,
        )
        .unwrap()
        .run(
            None,
            Some(&b_h.view(&lb).unwrap()),
            &dy_h.view(&lc).unwrap(),
            &mut da_cpu.view_mut(&la).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
        cpu::BatchedMatmulBackward::new(
            Cpu::Reference,
            BinaryOperand::Right,
            &spec,
            &la,
            &lb,
            &lc,
            &lb,
        )
        .unwrap()
        .run(
            Some(&a_h.view(&la).unwrap()),
            None,
            &dy_h.view(&lc).unwrap(),
            &mut db_cpu.view_mut(&lb).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

        let a_d = cuda.upload(&a_h).unwrap();
        let b_d = cuda.upload(&b_h).unwrap();
        let dy_d = cuda.upload(&dy_h).unwrap();
        let mut c_d = cuda.zeros(spec.output_shape().dims()).unwrap();
        let mut da_d = cuda.zeros(la_dims).unwrap();
        let mut db_d = cuda.zeros(lb_dims).unwrap();

        let fwd_d = cuda::BatchedMatmulPlan::new(&cuda, &spec, &la, &lb, &lc).unwrap();
        let bwd_a =
            cuda::BatchedMatmulBackward::new(&cuda, BinaryOperand::Left, &spec, &la, &lb, &lc, &la)
                .unwrap();
        let bwd_b = cuda::BatchedMatmulBackward::new(
            &cuda,
            BinaryOperand::Right,
            &spec,
            &la,
            &lb,
            &lc,
            &lb,
        )
        .unwrap();

        support::no_allocations(|| {
            fwd_d
                .run(
                    &a_d.view(&la).unwrap(),
                    &b_d.view(&lb).unwrap(),
                    &mut c_d.view_mut(&lc).unwrap(),
                )
                .unwrap();
            bwd_a
                .run(
                    None,
                    Some(&b_d.view(&lb).unwrap()),
                    &dy_d.view(&lc).unwrap(),
                    &mut da_d.view_mut(&la).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            bwd_b
                .run(
                    Some(&a_d.view(&la).unwrap()),
                    None,
                    &dy_d.view(&lc).unwrap(),
                    &mut db_d.view_mut(&lb).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
        });
        assert_close(
            cuda.download(&c_d).unwrap().data(),
            c_cpu.data(),
            5e-5,
            &format!("CUDA BMM fwd {la_dims:?}"),
        );
        assert_close(
            cuda.download(&da_d).unwrap().data(),
            da_cpu.data(),
            5e-5,
            &format!("CUDA BMM dA {la_dims:?}"),
        );
        assert_close(
            cuda.download(&db_d).unwrap().data(),
            db_cpu.data(),
            5e-5,
            &format!("CUDA BMM dB {la_dims:?}"),
        );
    }

    // 2. SDPA across D <= 32, D <= 64, D <= 128, D > 128 (general), MHA, GQA, causal, non-causal,
    //    custom 64x64 SIMT GEMM fast path (T%64==0, D%64==0), permuted views, and Accumulate mode.
    for &(b, h, hkv, tq, tk, d, causal) in &[
        (2, 4, 4, 35, 35, 32, false),
        (2, 4, 4, 35, 35, 32, true),
        (1, 4, 2, 40, 48, 64, false),  // GQA + D=64 + asymmetric
        (1, 4, 2, 40, 40, 64, true),   // GQA + causal + D=64
        (2, 4, 4, 64, 64, 64, false),  // 64x64 SIMT GEMM MHA non-causal
        (2, 4, 4, 64, 64, 64, true),   // 64x64 SIMT GEMM MHA causal + tile64 skip
        (2, 4, 2, 64, 64, 64, false),  // 64x64 SIMT GEMM GQA non-causal + GQA reduce
        (2, 4, 2, 64, 64, 64, true),   // 64x64 SIMT GEMM GQA causal + GQA reduce
        (1, 2, 2, 33, 33, 96, true),   // D=96 (128 bucket) + causal
        (1, 2, 2, 11, 13, 144, false), // D>128 general fallback
    ] {
        let spec = SdpaSpec::new(b, h, hkv, tq, tk, d, None, causal).unwrap();
        let lq = Layout::contiguous(spec.query_shape().dims()).unwrap();
        let lkv = Layout::contiguous(spec.kv_shape().dims()).unwrap();

        let q_h = Tensor::from_vec(lq.shape().dims(), det(lq.shape().len(), 101)).unwrap();
        let k_h = Tensor::from_vec(lkv.shape().dims(), det(lkv.shape().len(), 202)).unwrap();
        let v_h = Tensor::from_vec(lkv.shape().dims(), det(lkv.shape().len(), 303)).unwrap();
        let dy_h = Tensor::from_vec(lq.shape().dims(), det(lq.shape().len(), 404)).unwrap();

        let mut o_cpu = Tensor::zeros(lq.shape().dims()).unwrap();
        let mut dq_cpu = Tensor::zeros(lq.shape().dims()).unwrap();
        let mut dk_cpu = Tensor::zeros(lkv.shape().dims()).unwrap();
        let mut dv_cpu = Tensor::zeros(lkv.shape().dims()).unwrap();

        cpu::SdpaPlan::new(spec, &lq, &lkv, &lkv, &lq)
            .unwrap()
            .run(
                &q_h.view(&lq).unwrap(),
                &k_h.view(&lkv).unwrap(),
                &v_h.view(&lkv).unwrap(),
                &mut o_cpu.view_mut(&lq).unwrap(),
            )
            .unwrap();
        cpu::SdpaBackward::new(SdpaOperand::Query, spec, &lq, &lkv, &lkv, &lq, &lq)
            .unwrap()
            .run(
                Some(&q_h.view(&lq).unwrap()),
                Some(&k_h.view(&lkv).unwrap()),
                Some(&v_h.view(&lkv).unwrap()),
                &dy_h.view(&lq).unwrap(),
                &mut dq_cpu.view_mut(&lq).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        cpu::SdpaBackward::new(SdpaOperand::Key, spec, &lq, &lkv, &lkv, &lq, &lkv)
            .unwrap()
            .run(
                Some(&q_h.view(&lq).unwrap()),
                Some(&k_h.view(&lkv).unwrap()),
                Some(&v_h.view(&lkv).unwrap()),
                &dy_h.view(&lq).unwrap(),
                &mut dk_cpu.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        cpu::SdpaBackward::new(SdpaOperand::Value, spec, &lq, &lkv, &lkv, &lq, &lkv)
            .unwrap()
            .run(
                Some(&q_h.view(&lq).unwrap()),
                Some(&k_h.view(&lkv).unwrap()),
                None,
                &dy_h.view(&lq).unwrap(),
                &mut dv_cpu.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();

        let q_d = cuda.upload(&q_h).unwrap();
        let k_d = cuda.upload(&k_h).unwrap();
        let v_d = cuda.upload(&v_h).unwrap();
        let dy_d = cuda.upload(&dy_h).unwrap();
        let mut o_d = cuda.zeros(lq.shape().dims()).unwrap();
        let mut dq_d = cuda.zeros(lq.shape().dims()).unwrap();
        let mut dk_d = cuda.zeros(lkv.shape().dims()).unwrap();
        let mut dv_d = cuda.zeros(lkv.shape().dims()).unwrap();

        let fwd_d = cuda::SdpaPlan::new(&cuda, spec, &lq, &lkv, &lkv, &lq).unwrap();
        let bq_d =
            cuda::SdpaBackward::new(&cuda, SdpaOperand::Query, spec, &lq, &lkv, &lkv, &lq, &lq)
                .unwrap();
        let bk_d =
            cuda::SdpaBackward::new(&cuda, SdpaOperand::Key, spec, &lq, &lkv, &lkv, &lq, &lkv)
                .unwrap();
        let bv_d =
            cuda::SdpaBackward::new(&cuda, SdpaOperand::Value, spec, &lq, &lkv, &lkv, &lq, &lkv)
                .unwrap();

        support::no_allocations(|| {
            fwd_d
                .run(
                    &q_d.view(&lq).unwrap(),
                    &k_d.view(&lkv).unwrap(),
                    &v_d.view(&lkv).unwrap(),
                    &mut o_d.view_mut(&lq).unwrap(),
                )
                .unwrap();
            bq_d.run(
                Some(&q_d.view(&lq).unwrap()),
                Some(&k_d.view(&lkv).unwrap()),
                Some(&v_d.view(&lkv).unwrap()),
                &dy_d.view(&lq).unwrap(),
                &mut dq_d.view_mut(&lq).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            bk_d.run(
                Some(&q_d.view(&lq).unwrap()),
                Some(&k_d.view(&lkv).unwrap()),
                Some(&v_d.view(&lkv).unwrap()),
                &dy_d.view(&lq).unwrap(),
                &mut dk_d.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            bv_d.run(
                Some(&q_d.view(&lq).unwrap()),
                Some(&k_d.view(&lkv).unwrap()),
                None,
                &dy_d.view(&lq).unwrap(),
                &mut dv_d.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });

        let tag = format!("CUDA SDPA({b},{h}/{hkv},{tq}x{tk},d={d},c={causal})");
        assert_close(
            cuda.download(&o_d).unwrap().data(),
            o_cpu.data(),
            5e-5,
            &format!("{tag} O"),
        );
        assert_close(
            cuda.download(&dq_d).unwrap().data(),
            dq_cpu.data(),
            1e-4,
            &format!("{tag} dQ"),
        );
        assert_close(
            cuda.download(&dk_d).unwrap().data(),
            dk_cpu.data(),
            1e-4,
            &format!("{tag} dK"),
        );
        assert_close(
            cuda.download(&dv_d).unwrap().data(),
            dv_cpu.data(),
            1e-4,
            &format!("{tag} dV"),
        );

        // Verify Accumulate mode on 64x64 SIMT GEMM fast paths (MHA + GQA, causal + non-causal)
        if tq == 64 && tk == 64 && d == 64 {
            let mut dq_acc = cuda
                .upload(&Tensor::from_vec(lq.shape().dims(), vec![0.25; lq.shape().len()]).unwrap())
                .unwrap();
            let mut dk_acc = cuda
                .upload(
                    &Tensor::from_vec(lkv.shape().dims(), vec![0.25; lkv.shape().len()]).unwrap(),
                )
                .unwrap();
            let mut dv_acc = cuda
                .upload(
                    &Tensor::from_vec(lkv.shape().dims(), vec![0.25; lkv.shape().len()]).unwrap(),
                )
                .unwrap();
            bq_d.run(
                Some(&q_d.view(&lq).unwrap()),
                Some(&k_d.view(&lkv).unwrap()),
                Some(&v_d.view(&lkv).unwrap()),
                &dy_d.view(&lq).unwrap(),
                &mut dq_acc.view_mut(&lq).unwrap(),
                GradientMode::Accumulate,
            )
            .unwrap();
            bk_d.run(
                Some(&q_d.view(&lq).unwrap()),
                Some(&k_d.view(&lkv).unwrap()),
                Some(&v_d.view(&lkv).unwrap()),
                &dy_d.view(&lq).unwrap(),
                &mut dk_acc.view_mut(&lkv).unwrap(),
                GradientMode::Accumulate,
            )
            .unwrap();
            bv_d.run(
                Some(&q_d.view(&lq).unwrap()),
                Some(&k_d.view(&lkv).unwrap()),
                None,
                &dy_d.view(&lq).unwrap(),
                &mut dv_acc.view_mut(&lkv).unwrap(),
                GradientMode::Accumulate,
            )
            .unwrap();
            let exp_dq: Vec<f32> = dq_cpu.data().iter().map(|&v| v + 0.25).collect();
            let exp_dk: Vec<f32> = dk_cpu.data().iter().map(|&v| v + 0.25).collect();
            let exp_dv: Vec<f32> = dv_cpu.data().iter().map(|&v| v + 0.25).collect();
            assert_close(
                cuda.download(&dq_acc).unwrap().data(),
                &exp_dq,
                1e-4,
                &format!("{tag} dQ acc"),
            );
            assert_close(
                cuda.download(&dk_acc).unwrap().data(),
                &exp_dk,
                1e-4,
                &format!("{tag} dK acc"),
            );
            assert_close(
                cuda.download(&dv_acc).unwrap().data(),
                &exp_dv,
                1e-4,
                &format!("{tag} dV acc"),
            );
        }
    }

    // 3. Permuted [B, T, H, D].permute([0, 2, 1, 3]) on 64x64 causal SIMT GEMM path
    {
        let (b, h, t, d) = (2, 4, 64, 64);
        let spec = SdpaSpec::new(b, h, h, t, t, d, None, true).unwrap();
        let base_l = Layout::contiguous([b, t, h, d]).unwrap();
        let perm_l = base_l.permute(&[0, 2, 1, 3]).unwrap();
        let contig_l = Layout::contiguous([b, h, t, d]).unwrap();
        let q_h = Tensor::from_vec([b, t, h, d], det(b * t * h * d, 501)).unwrap();
        let k_h = Tensor::from_vec([b, t, h, d], det(b * t * h * d, 502)).unwrap();
        let v_h = Tensor::from_vec([b, t, h, d], det(b * t * h * d, 503)).unwrap();
        let dy_h = Tensor::from_vec([b, h, t, d], det(b * h * t * d, 504)).unwrap();

        let mut o_cpu = Tensor::zeros([b, h, t, d]).unwrap();
        let mut dq_cpu = Tensor::zeros([b, t, h, d]).unwrap();
        let mut dk_cpu = Tensor::zeros([b, t, h, d]).unwrap();
        let mut dv_cpu = Tensor::zeros([b, t, h, d]).unwrap();
        cpu::SdpaPlan::new(spec, &perm_l, &perm_l, &perm_l, &contig_l)
            .unwrap()
            .run(
                &q_h.view(&perm_l).unwrap(),
                &k_h.view(&perm_l).unwrap(),
                &v_h.view(&perm_l).unwrap(),
                &mut o_cpu.view_mut(&contig_l).unwrap(),
            )
            .unwrap();
        cpu::SdpaBackward::new(
            SdpaOperand::Query,
            spec,
            &perm_l,
            &perm_l,
            &perm_l,
            &contig_l,
            &perm_l,
        )
        .unwrap()
        .run(
            Some(&q_h.view(&perm_l).unwrap()),
            Some(&k_h.view(&perm_l).unwrap()),
            Some(&v_h.view(&perm_l).unwrap()),
            &dy_h.view(&contig_l).unwrap(),
            &mut dq_cpu.view_mut(&perm_l).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
        cpu::SdpaBackward::new(
            SdpaOperand::Key,
            spec,
            &perm_l,
            &perm_l,
            &perm_l,
            &contig_l,
            &perm_l,
        )
        .unwrap()
        .run(
            Some(&q_h.view(&perm_l).unwrap()),
            Some(&k_h.view(&perm_l).unwrap()),
            Some(&v_h.view(&perm_l).unwrap()),
            &dy_h.view(&contig_l).unwrap(),
            &mut dk_cpu.view_mut(&perm_l).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
        cpu::SdpaBackward::new(
            SdpaOperand::Value,
            spec,
            &perm_l,
            &perm_l,
            &perm_l,
            &contig_l,
            &perm_l,
        )
        .unwrap()
        .run(
            Some(&q_h.view(&perm_l).unwrap()),
            Some(&k_h.view(&perm_l).unwrap()),
            None,
            &dy_h.view(&contig_l).unwrap(),
            &mut dv_cpu.view_mut(&perm_l).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

        let q_d = cuda.upload(&q_h).unwrap();
        let k_d = cuda.upload(&k_h).unwrap();
        let v_d = cuda.upload(&v_h).unwrap();
        let dy_d = cuda.upload(&dy_h).unwrap();
        let mut o_d = cuda.zeros([b, h, t, d]).unwrap();
        let mut dq_d = cuda.zeros([b, t, h, d]).unwrap();
        let mut dk_d = cuda.zeros([b, t, h, d]).unwrap();
        let mut dv_d = cuda.zeros([b, t, h, d]).unwrap();
        cuda::SdpaPlan::new(&cuda, spec, &perm_l, &perm_l, &perm_l, &contig_l)
            .unwrap()
            .run(
                &q_d.view(&perm_l).unwrap(),
                &k_d.view(&perm_l).unwrap(),
                &v_d.view(&perm_l).unwrap(),
                &mut o_d.view_mut(&contig_l).unwrap(),
            )
            .unwrap();
        cuda::SdpaBackward::new(
            &cuda,
            SdpaOperand::Query,
            spec,
            &perm_l,
            &perm_l,
            &perm_l,
            &contig_l,
            &perm_l,
        )
        .unwrap()
        .run(
            Some(&q_d.view(&perm_l).unwrap()),
            Some(&k_d.view(&perm_l).unwrap()),
            Some(&v_d.view(&perm_l).unwrap()),
            &dy_d.view(&contig_l).unwrap(),
            &mut dq_d.view_mut(&perm_l).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
        cuda::SdpaBackward::new(
            &cuda,
            SdpaOperand::Key,
            spec,
            &perm_l,
            &perm_l,
            &perm_l,
            &contig_l,
            &perm_l,
        )
        .unwrap()
        .run(
            Some(&q_d.view(&perm_l).unwrap()),
            Some(&k_d.view(&perm_l).unwrap()),
            Some(&v_d.view(&perm_l).unwrap()),
            &dy_d.view(&contig_l).unwrap(),
            &mut dk_d.view_mut(&perm_l).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
        cuda::SdpaBackward::new(
            &cuda,
            SdpaOperand::Value,
            spec,
            &perm_l,
            &perm_l,
            &perm_l,
            &contig_l,
            &perm_l,
        )
        .unwrap()
        .run(
            Some(&q_d.view(&perm_l).unwrap()),
            Some(&k_d.view(&perm_l).unwrap()),
            None,
            &dy_d.view(&contig_l).unwrap(),
            &mut dv_d.view_mut(&perm_l).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
        assert_close(
            cuda.download(&o_d).unwrap().data(),
            o_cpu.data(),
            5e-5,
            "CUDA SDPA permuted 64x64 O",
        );
        assert_close(
            cuda.download(&dq_d).unwrap().data(),
            dq_cpu.data(),
            1e-4,
            "CUDA SDPA permuted 64x64 dQ",
        );
        assert_close(
            cuda.download(&dk_d).unwrap().data(),
            dk_cpu.data(),
            1e-4,
            "CUDA SDPA permuted 64x64 dK",
        );
        assert_close(
            cuda.download(&dv_d).unwrap().data(),
            dv_cpu.data(),
            1e-4,
            "CUDA SDPA permuted 64x64 dV",
        );
    }
}

#[test]
fn cuda_tape_audio_transformer_attention_block_captured_matches_cpu() {
    let cuda = Cuda::new(0).unwrap();
    let x_init = Tensor::from_vec([2, 32, 64], det(2 * 32 * 64, 1)).unwrap();
    let wq_init = Tensor::from_vec([64, 64], det(64 * 64, 2)).unwrap();
    let wk_init = Tensor::from_vec([64, 64], det(64 * 64, 3)).unwrap();
    let wv_init = Tensor::from_vec([64, 64], det(64 * 64, 4)).unwrap();
    let wo_init = Tensor::from_vec([64, 64], det(64 * 64, 5)).unwrap();
    let tgt_init = Tensor::from_vec([2, 32, 64], det(2 * 32 * 64, 6)).unwrap();

    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 32, 64], true).unwrap();
    let wq = b.leaf([64, 64], true).unwrap();
    let wk = b.leaf([64, 64], true).unwrap();
    let wv = b.leaf([64, 64], true).unwrap();
    let wo = b.leaf([64, 64], true).unwrap();
    let tgt = b.leaf([2, 32, 64], false).unwrap();

    let xn = b.rms_norm(x, 1, 1e-5).unwrap();
    let q = b.bmm(xn, wq).unwrap();
    let k = b.bmm(xn, wk).unwrap();
    let v = b.bmm(xn, wv).unwrap();
    let q4 = b.reshape(q, [2, 32, 2, 32]).unwrap();
    let qh = b.permute(q4, &[0, 2, 1, 3]).unwrap();
    let k4 = b.reshape(k, [2, 32, 2, 32]).unwrap();
    let kh = b.permute(k4, &[0, 2, 1, 3]).unwrap();
    let v4 = b.reshape(v, [2, 32, 2, 32]).unwrap();
    let vh = b.permute(v4, &[0, 2, 1, 3]).unwrap();
    let attn = b.sdpa(qh, kh, vh, None, true).unwrap();
    let attn_t = b.permute(attn, &[0, 2, 1, 3]).unwrap();
    let attn_c = b.unary(UnaryOp::Silu, attn_t).unwrap();
    let attn_flat = b.reshape(attn_c, [2, 32, 64]).unwrap();
    let proj = b.bmm(attn_flat, wo).unwrap();
    let res = b.binary(BinaryOp::Add, proj, x).unwrap();
    let diff = b.binary(BinaryOp::Sub, res, tgt).unwrap();
    let sq = b.unary(UnaryOp::Square, diff).unwrap();
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1, 2], false).unwrap();
    let prog = b.finish(&[loss]).unwrap();

    let cx = Shared::new(x_init.clone()).unwrap();
    let cwq = Shared::new(wq_init.clone()).unwrap();
    let cwk = Shared::new(wk_init.clone()).unwrap();
    let cwv = Shared::new(wv_init.clone()).unwrap();
    let cwo = Shared::new(wo_init.clone()).unwrap();
    let ct = Shared::new(tgt_init.clone()).unwrap();
    let mut cpu_tape = cpu::Tape::new(
        Cpu::Reference,
        &prog,
        &[
            Binding::new(x, cx.clone()),
            Binding::new(wq, cwq.clone()),
            Binding::new(wk, cwk.clone()),
            Binding::new(wv, cwv.clone()),
            Binding::new(wo, cwo.clone()),
            Binding::new(tgt, ct.clone()),
        ],
    )
    .unwrap();

    let make_gpu = || {
        let gx = Shared::new(cuda.upload(&x_init).unwrap()).unwrap();
        let gwq = Shared::new(cuda.upload(&wq_init).unwrap()).unwrap();
        let gwk = Shared::new(cuda.upload(&wk_init).unwrap()).unwrap();
        let gwv = Shared::new(cuda.upload(&wv_init).unwrap()).unwrap();
        let gwo = Shared::new(cuda.upload(&wo_init).unwrap()).unwrap();
        let gt = Shared::new(cuda.upload(&tgt_init).unwrap()).unwrap();
        let tape = cuda::Tape::new(
            &cuda,
            &prog,
            &[
                Binding::new(x, gx.clone()),
                Binding::new(wq, gwq.clone()),
                Binding::new(wk, gwk.clone()),
                Binding::new(wv, gwv.clone()),
                Binding::new(wo, gwo.clone()),
                Binding::new(tgt, gt.clone()),
            ],
        )
        .unwrap();
        (tape, gx, gwq, gwk, gwv, gwo)
    };
    let (mut gpu_uncap, ux, uq, uk, uv, uo) = make_gpu();
    let (mut gpu_cap, kx, kq, kk, kv, ko) = make_gpu();

    let params = [x, wq, wk, wv, wo];
    for _ in 0..4 {
        cpu_tape.step_scalar(&params, 0.05).unwrap();
        gpu_uncap.step_scalar(&params, 0.05).unwrap();
        gpu_cap.step_scalar_captured(&params, 0.05).unwrap();
    }
    assert!(gpu_cap.has_fused_plan());
    assert!(gpu_cap.has_captured_graph());
    cuda.synchronize().unwrap();

    for (name, c, u, k) in [
        (
            "x",
            cx.read().unwrap(),
            cuda.download(&ux.read().unwrap()).unwrap(),
            cuda.download(&kx.read().unwrap()).unwrap(),
        ),
        (
            "wq",
            cwq.read().unwrap(),
            cuda.download(&uq.read().unwrap()).unwrap(),
            cuda.download(&kq.read().unwrap()).unwrap(),
        ),
        (
            "wk",
            cwk.read().unwrap(),
            cuda.download(&uk.read().unwrap()).unwrap(),
            cuda.download(&kk.read().unwrap()).unwrap(),
        ),
        (
            "wv",
            cwv.read().unwrap(),
            cuda.download(&uv.read().unwrap()).unwrap(),
            cuda.download(&kv.read().unwrap()).unwrap(),
        ),
        (
            "wo",
            cwo.read().unwrap(),
            cuda.download(&uo.read().unwrap()).unwrap(),
            cuda.download(&ko.read().unwrap()).unwrap(),
        ),
    ] {
        let ubits: Vec<u32> = u.data().iter().map(|v| v.to_bits()).collect();
        let kbits: Vec<u32> = k.data().iter().map(|v| v.to_bits()).collect();
        assert_eq!(ubits, kbits, "CUDA captured vs uncaptured drift on {name}");
        assert_close(k.data(), c.data(), 1e-4, &format!("CUDA vs CPU {name}"));
    }
}
