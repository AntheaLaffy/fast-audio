#![allow(clippy::needless_range_loop)]

#[allow(dead_code)]
mod support;

use fast_audio::{
    BinaryOp, BinaryOperand, BmmSpec, GradientMode, Layout, ReduceOp, SdpaOperand, SdpaSpec, Shape,
    Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{self, Cpu},
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

fn assert_close(actual: &[f32], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} len mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let err = (a as f64 - e).abs();
        assert!(
            err <= tol,
            "{label}[{i}]: actual={a} expected={e} err={err} > {tol}"
        );
    }
}

fn fp64_sdpa_ref(
    spec: SdpaSpec,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dy: &[f32],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let b_n = spec.batch();
    let h_n = spec.num_heads();
    let h_kv_n = spec.num_kv_heads();
    let gqa = h_n / h_kv_n;
    let tq = spec.seq_q();
    let tk = spec.seq_k();
    let d = spec.head_dim();
    let scale = spec.scale() as f64;

    let mut out = vec![0.0f64; b_n * h_n * tq * d];
    let mut dq = vec![0.0f64; b_n * h_n * tq * d];
    let mut dk = vec![0.0f64; b_n * h_kv_n * tk * d];
    let mut dv = vec![0.0f64; b_n * h_kv_n * tk * d];

    for b in 0..b_n {
        for h in 0..h_n {
            let hkv = h / gqa;
            for i in 0..tq {
                let k_lim = if spec.is_causal() {
                    (i + 1).min(tk)
                } else {
                    tk
                };
                let q_off = ((b * h_n + h) * tq + i) * d;
                let mut scores = vec![f64::NEG_INFINITY; k_lim];
                let mut m = f64::NEG_INFINITY;
                for j in 0..k_lim {
                    let k_off = ((b * h_kv_n + hkv) * tk + j) * d;
                    let mut dot = 0.0f64;
                    for dim in 0..d {
                        dot += q[q_off + dim] as f64 * k[k_off + dim] as f64;
                    }
                    let sc = dot * scale;
                    scores[j] = sc;
                    if sc > m {
                        m = sc;
                    }
                }
                let mut probs = vec![0.0f64; k_lim];
                let mut z = 0.0f64;
                for j in 0..k_lim {
                    let e = (scores[j] - m).exp();
                    probs[j] = e;
                    z += e;
                }
                for p in &mut probs {
                    *p /= z;
                }
                for j in 0..k_lim {
                    let v_off = ((b * h_kv_n + hkv) * tk + j) * d;
                    for dim in 0..d {
                        out[q_off + dim] += probs[j] * v[v_off + dim] as f64;
                    }
                }
                // Backward
                let mut d_row = 0.0f64;
                for dim in 0..d {
                    d_row += out[q_off + dim] * dy[q_off + dim] as f64;
                }
                for j in 0..k_lim {
                    let k_off = ((b * h_kv_n + hkv) * tk + j) * d;
                    let v_off = k_off;
                    let mut dp = 0.0f64;
                    for dim in 0..d {
                        dp += dy[q_off + dim] as f64 * v[v_off + dim] as f64;
                        dv[v_off + dim] += probs[j] * dy[q_off + dim] as f64;
                    }
                    let ds = probs[j] * (dp - d_row) * scale;
                    for dim in 0..d {
                        dq[q_off + dim] += ds * k[k_off + dim] as f64;
                        dk[k_off + dim] += ds * q[q_off + dim] as f64;
                    }
                }
            }
        }
    }
    (out, dq, dk, dv)
}

#[test]
fn cpu_bmm_forward_backward_broadcast_collapsed_and_strided() {
    // 1. Batched linear broadcast [2, 3, 5, 7] @ [7, 4] (collapsed 2D route)
    // 2. Cross-broadcast [2, 1, 4, 6] @ [1, 3, 6, 5] -> [2, 3, 4, 5] (broadcast reduction in both dA & dB)
    // 3. Standard MHA [2, 4, 9, 8] @ [2, 4, 8, 9]
    for &(la_dims, lb_dims) in &[
        (&[2, 3, 5, 7][..], &[7, 4][..]),
        (&[2, 1, 4, 6][..], &[1, 3, 6, 5][..]),
        (&[2, 4, 9, 8][..], &[2, 4, 8, 9][..]),
        (&[5, 7][..], &[7, 3][..]),
    ] {
        let sa = Shape::new(la_dims).unwrap();
        let sb = Shape::new(lb_dims).unwrap();
        let spec = BmmSpec::new(&sa, &sb).unwrap();
        let la = Layout::contiguous(la_dims).unwrap();
        let lb = Layout::contiguous(lb_dims).unwrap();
        let lc = Layout::contiguous(spec.output_shape().dims()).unwrap();

        let a = Tensor::from_vec(la_dims, det(sa.len(), 11)).unwrap();
        let b = Tensor::from_vec(lb_dims, det(sb.len(), 23)).unwrap();
        let dy = Tensor::from_vec(
            spec.output_shape().dims(),
            det(spec.output_shape().len(), 37),
        )
        .unwrap();
        let mut c = Tensor::zeros(spec.output_shape().dims()).unwrap();
        let mut da = Tensor::zeros(la_dims).unwrap();
        let mut db = Tensor::zeros(lb_dims).unwrap();

        let fwd = cpu::BatchedMatmulPlan::new(Cpu::Reference, &spec, &la, &lb, &lc).unwrap();
        let bwd_a = cpu::BatchedMatmulBackward::new(
            Cpu::Reference,
            BinaryOperand::Left,
            &spec,
            &la,
            &lb,
            &lc,
            &la,
        )
        .unwrap();
        let bwd_b = cpu::BatchedMatmulBackward::new(
            Cpu::Reference,
            BinaryOperand::Right,
            &spec,
            &la,
            &lb,
            &lc,
            &lb,
        )
        .unwrap();

        support::no_allocations(|| {
            fwd.run(
                &a.view(&la).unwrap(),
                &b.view(&lb).unwrap(),
                &mut c.view_mut(&lc).unwrap(),
            )
            .unwrap();
            bwd_a
                .run(
                    None,
                    Some(&b.view(&lb).unwrap()),
                    &dy.view(&lc).unwrap(),
                    &mut da.view_mut(&la).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            bwd_b
                .run(
                    Some(&a.view(&la).unwrap()),
                    None,
                    &dy.view(&lc).unwrap(),
                    &mut db.view_mut(&lb).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
        });

        // Verify via adjoint inner-product identity: <A @ B, dY> == <A, dA> == <B, dB>
        let dot_out: f64 = c
            .data()
            .iter()
            .zip(dy.data())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum();
        let dot_a: f64 = a
            .data()
            .iter()
            .zip(da.data())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum();
        let dot_b: f64 = b
            .data()
            .iter()
            .zip(db.data())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum();
        assert!(
            (dot_out - dot_a).abs() < 1e-4,
            "bmm {la_dims:?}@{lb_dims:?} dA adjoint: {dot_out} vs {dot_a}"
        );
        assert!(
            (dot_out - dot_b).abs() < 1e-4,
            "bmm {la_dims:?}@{lb_dims:?} dB adjoint: {dot_out} vs {dot_b}"
        );
    }

    // Signed-stride transposed view BMM check
    let raw_a = Tensor::from_vec([2, 6, 4], det(48, 55)).unwrap();
    let raw_b = Tensor::from_vec([2, 5, 6], det(60, 66)).unwrap();
    let la = Layout::contiguous([2, 6, 4])
        .unwrap()
        .permute(&[0, 2, 1])
        .unwrap(); // [2, 4, 6]
    let lb = Layout::contiguous([2, 5, 6])
        .unwrap()
        .permute(&[0, 2, 1])
        .unwrap(); // [2, 6, 5]
    let lc = Layout::contiguous([2, 4, 5]).unwrap();
    let spec = BmmSpec::new(la.shape(), lb.shape()).unwrap();
    let mut c = Tensor::zeros([2, 4, 5]).unwrap();
    cpu::BatchedMatmulPlan::new(Cpu::Reference, &spec, &la, &lb, &lc)
        .unwrap()
        .run(
            &raw_a.view(&la).unwrap(),
            &raw_b.view(&lb).unwrap(),
            &mut c.view_mut(&lc).unwrap(),
        )
        .unwrap();
    // Reference compare element by element
    for batch in 0..2 {
        for i in 0..4 {
            for j in 0..5 {
                let mut sum = 0.0f64;
                for k in 0..6 {
                    let av = raw_a.data()[(batch * 6 + k) * 4 + i] as f64;
                    let bv = raw_b.data()[(batch * 5 + j) * 6 + k] as f64;
                    sum += av * bv;
                }
                assert!((c.data()[(batch * 4 + i) * 5 + j] as f64 - sum).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn cpu_sdpa_forward_backward_mha_gqa_causal_and_permuted_match_fp64() {
    for &(b, h, hkv, tq, tk, d, causal) in &[
        (2, 4, 4, 17, 17, 32, false),
        (2, 4, 4, 17, 17, 32, true),
        (1, 6, 2, 13, 19, 48, false), // GQA + asymmetric T_q != T_k
        (1, 4, 2, 35, 35, 64, true),  // GQA + causal + >BC tile
        (1, 2, 2, 9, 7, 140, false),  // D > 128
    ] {
        let spec = SdpaSpec::new(b, h, hkv, tq, tk, d, None, causal).unwrap();
        let lq = Layout::contiguous(spec.query_shape().dims()).unwrap();
        let lkv = Layout::contiguous(spec.kv_shape().dims()).unwrap();

        let q = Tensor::from_vec(lq.shape().dims(), det(lq.shape().len(), 101)).unwrap();
        let k = Tensor::from_vec(lkv.shape().dims(), det(lkv.shape().len(), 202)).unwrap();
        let v = Tensor::from_vec(lkv.shape().dims(), det(lkv.shape().len(), 303)).unwrap();
        let dy = Tensor::from_vec(lq.shape().dims(), det(lq.shape().len(), 404)).unwrap();

        let mut o = Tensor::zeros(lq.shape().dims()).unwrap();
        let mut dq = Tensor::zeros(lq.shape().dims()).unwrap();
        let mut dk = Tensor::zeros(lkv.shape().dims()).unwrap();
        let mut dv = Tensor::zeros(lkv.shape().dims()).unwrap();

        let fwd = cpu::SdpaPlan::new(spec, &lq, &lkv, &lkv, &lq).unwrap();
        let bq =
            cpu::SdpaBackward::new(SdpaOperand::Query, spec, &lq, &lkv, &lkv, &lq, &lq).unwrap();
        let bk =
            cpu::SdpaBackward::new(SdpaOperand::Key, spec, &lq, &lkv, &lkv, &lq, &lkv).unwrap();
        let bv =
            cpu::SdpaBackward::new(SdpaOperand::Value, spec, &lq, &lkv, &lkv, &lq, &lkv).unwrap();
        assert_eq!(bv.input_dependencies(), [true, true, false]);

        support::no_allocations(|| {
            fwd.run(
                &q.view(&lq).unwrap(),
                &k.view(&lkv).unwrap(),
                &v.view(&lkv).unwrap(),
                &mut o.view_mut(&lq).unwrap(),
            )
            .unwrap();
            bq.run(
                Some(&q.view(&lq).unwrap()),
                Some(&k.view(&lkv).unwrap()),
                Some(&v.view(&lkv).unwrap()),
                &dy.view(&lq).unwrap(),
                &mut dq.view_mut(&lq).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            bk.run(
                Some(&q.view(&lq).unwrap()),
                Some(&k.view(&lkv).unwrap()),
                Some(&v.view(&lkv).unwrap()),
                &dy.view(&lq).unwrap(),
                &mut dk.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            bv.run(
                Some(&q.view(&lq).unwrap()),
                Some(&k.view(&lkv).unwrap()),
                None, // V is NOT required for dV!
                &dy.view(&lq).unwrap(),
                &mut dv.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });

        let (ro, rdq, rdk, rdv) = fp64_sdpa_ref(spec, q.data(), k.data(), v.data(), dy.data());
        let tag = format!("sdpa({b},{h}/{hkv},{tq}x{tk},d={d},c={causal})");
        assert_close(o.data(), &ro, 3e-5, &format!("{tag} O"));
        assert_close(dq.data(), &rdq, 5e-5, &format!("{tag} dQ"));
        assert_close(dk.data(), &rdk, 5e-5, &format!("{tag} dK"));
        assert_close(dv.data(), &rdv, 5e-5, &format!("{tag} dV"));
    }

    // Zero-copy permuted [B, T, H, D].permute([0, 2, 1, 3]) test
    let raw_q = Tensor::from_vec([2, 11, 3, 16], det(2 * 11 * 3 * 16, 701)).unwrap();
    let raw_k = Tensor::from_vec([2, 11, 3, 16], det(2 * 11 * 3 * 16, 702)).unwrap();
    let raw_v = Tensor::from_vec([2, 11, 3, 16], det(2 * 11 * 3 * 16, 703)).unwrap();
    let perm = Layout::contiguous([2, 11, 3, 16])
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap();
    let dense = Layout::contiguous([2, 3, 11, 16]).unwrap();
    let spec = SdpaSpec::new(2, 3, 3, 11, 11, 16, None, true).unwrap();
    let mut o_perm = Tensor::zeros([2, 3, 11, 16]).unwrap();
    cpu::SdpaPlan::new(spec, &perm, &perm, &perm, &dense)
        .unwrap()
        .run(
            &raw_q.view(&perm).unwrap(),
            &raw_k.view(&perm).unwrap(),
            &raw_v.view(&perm).unwrap(),
            &mut o_perm.view_mut(&dense).unwrap(),
        )
        .unwrap();

    let mut q_c = Tensor::zeros([2, 3, 11, 16]).unwrap();
    let mut k_c = Tensor::zeros([2, 3, 11, 16]).unwrap();
    let mut v_c = Tensor::zeros([2, 3, 11, 16]).unwrap();
    let cp = cpu::CopyPlan::new(&perm, &dense).unwrap();
    cp.run(
        &raw_q.view(&perm).unwrap(),
        &mut q_c.view_mut(&dense).unwrap(),
    )
    .unwrap();
    cp.run(
        &raw_k.view(&perm).unwrap(),
        &mut k_c.view_mut(&dense).unwrap(),
    )
    .unwrap();
    cp.run(
        &raw_v.view(&perm).unwrap(),
        &mut v_c.view_mut(&dense).unwrap(),
    )
    .unwrap();
    let (ro, _, _, _) = fp64_sdpa_ref(
        spec,
        q_c.data(),
        k_c.data(),
        v_c.data(),
        &vec![1.0; 2 * 3 * 11 * 16],
    );
    assert_close(o_perm.data(), &ro, 2e-5, "permuted SDPA");
}

#[test]
fn cpu_tape_audio_transformer_attention_block_and_miri() {
    // Full audio transformer attention block:
    // X[B=2, T=8, D=16] -> RmsNorm -> Q/K/V = Bmm(X, Wq/Wk/Wv) -> reshape[2,8,2,8] -> permute[0,2,1,3]
    // -> SDPA(causal) -> permute[0,2,1,3] -> Bmm(O_flat, Wo) -> Add(X) -> MSE -> SGD
    let x_init = Tensor::from_vec([2, 8, 16], det(256, 1)).unwrap();
    let wq_init = Tensor::from_vec([16, 16], det(256, 2)).unwrap();
    let wk_init = Tensor::from_vec([16, 16], det(256, 3)).unwrap();
    let wv_init = Tensor::from_vec([16, 16], det(256, 4)).unwrap();
    let wo_init = Tensor::from_vec([16, 16], det(256, 5)).unwrap();
    let tgt_init = Tensor::from_vec([2, 8, 16], det(256, 6)).unwrap();

    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 8, 16], true).unwrap();
    let wq = b.leaf([16, 16], true).unwrap();
    let wk = b.leaf([16, 16], true).unwrap();
    let wv = b.leaf([16, 16], true).unwrap();
    let wo = b.leaf([16, 16], true).unwrap();
    let tgt = b.leaf([2, 8, 16], false).unwrap();

    let xn = b.rms_norm(x, 1, 1e-5).unwrap();
    let q = b.bmm(xn, wq).unwrap();
    let k = b.bmm(xn, wk).unwrap();
    let v = b.bmm(xn, wv).unwrap();
    let q4 = b.reshape(q, [2, 8, 2, 8]).unwrap();
    let qh = b.permute(q4, &[0, 2, 1, 3]).unwrap();
    let k4 = b.reshape(k, [2, 8, 2, 8]).unwrap();
    let kh = b.permute(k4, &[0, 2, 1, 3]).unwrap();
    let v4 = b.reshape(v, [2, 8, 2, 8]).unwrap();
    let vh = b.permute(v4, &[0, 2, 1, 3]).unwrap();
    let attn = b.sdpa(qh, kh, vh, None, true).unwrap(); // [2, 2, 8, 8]
    // Combine heads via Bmm([2, 8, 16], [16, 16]) after permute -> unary(Identity? No, contiguous!)
    // Use Bmm directly on permuted attn via explicit Bmm([2, 2, 8, 8] @ [2, 8, 8]) or reshape-free:
    // Actually permute [2, 2, 8, 8] -> [2, 8, 2, 8] is non-contiguous, so project per-head or materialize via unary/binary:
    let attn_t = b.permute(attn, &[0, 2, 1, 3]).unwrap(); // [2, 8, 2, 8]
    let attn_c = b.unary(UnaryOp::Silu, attn_t).unwrap(); // contiguous [2, 8, 2, 8]
    let attn_flat = b.reshape(attn_c, [2, 8, 16]).unwrap();
    let proj = b.bmm(attn_flat, wo).unwrap();
    let res = b.binary(BinaryOp::Add, proj, x).unwrap();
    let diff = b.binary(BinaryOp::Sub, res, tgt).unwrap();
    let sq = b.unary(UnaryOp::Square, diff).unwrap();
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1, 2], false).unwrap();
    let prog = b.finish(&[loss]).unwrap();

    let make = || {
        let sx = Shared::new(x_init.clone()).unwrap();
        let swq = Shared::new(wq_init.clone()).unwrap();
        let swk = Shared::new(wk_init.clone()).unwrap();
        let swv = Shared::new(wv_init.clone()).unwrap();
        let swo = Shared::new(wo_init.clone()).unwrap();
        let st = Shared::new(tgt_init.clone()).unwrap();
        let tape = cpu::Tape::new(
            Cpu::Reference,
            &prog,
            &[
                Binding::new(x, sx.clone()),
                Binding::new(wq, swq.clone()),
                Binding::new(wk, swk.clone()),
                Binding::new(wv, swv.clone()),
                Binding::new(wo, swo.clone()),
                Binding::new(tgt, st.clone()),
            ],
        )
        .unwrap();
        (tape, sx, swq, swk, swv, swo)
    };

    let (mut t1, x1, q1, k1, v1, o1) = make();
    let (mut t2, x2, q2, k2, v2, o2) = make();
    let params = [x, wq, wk, wv, wo];
    for _ in 0..3 {
        t1.step_scalar(&params, 0.05).unwrap();
        t2.forward().unwrap();
        t2.backward_scalar(GradientMode::Overwrite).unwrap();
        t2.sgd(&params, 0.05).unwrap();
    }
    assert!(t1.has_fused_plan());
    for (name, a, c) in [
        ("x", x1.read().unwrap(), x2.read().unwrap()),
        ("wq", q1.read().unwrap(), q2.read().unwrap()),
        ("wk", k1.read().unwrap(), k2.read().unwrap()),
        ("wv", v1.read().unwrap(), v2.read().unwrap()),
        ("wo", o1.read().unwrap(), o2.read().unwrap()),
    ] {
        for (i, (&va, &vc)) in a.data().iter().zip(c.data()).enumerate() {
            assert!(
                va.is_finite() && (va - vc).abs() < 1e-5,
                "transformer block {name}[{i}]: {va} vs {vc}"
            );
        }
    }
}
