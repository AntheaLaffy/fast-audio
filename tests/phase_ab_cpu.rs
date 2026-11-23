use fast_audio::{
    Bf16Tensor, CrossEntropySpec, DType, DropoutSpec, EmbeddingSpec, GradientMode, IndexTensor,
    Layout, PrecisionPolicy, ReduceOp, RopeSpec, Shape, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    bf16_to_f32,
    cpu::{
        Cpu, CrossEntropyBackward, CrossEntropyPlan, DropoutBackward, DropoutPlan,
        EmbeddingBackward, EmbeddingPlan, RopeBackward, RopePlan, Tape,
    },
    f16_to_f32, f32_to_bf16, f32_to_f16,
};

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let err = (x - y).abs();
        if err > max_err || err.is_nan() {
            max_err = err;
            max_idx = i;
        }
    }
    assert!(
        max_err <= tol,
        "{label}: max_err={max_err} > tol={tol} at idx={max_idx} (a={}, b={})",
        a[max_idx],
        b[max_idx]
    );
}

#[test]
fn dtype_bf16_f16_rne_edge_cases() {
    assert_eq!(DType::F32.byte_size(), 4);
    assert_eq!(DType::Bf16.byte_size(), 2);
    assert_eq!(DType::F16.byte_size(), 2);
    assert_eq!(DType::I64.byte_size(), 8);
    assert_eq!(PrecisionPolicy::default(), PrecisionPolicy::StrictFp32);

    // BF16 exact round-to-nearest-even ties:
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    // Signed zeros, infinities, NaN
    assert_eq!(f32_to_bf16(0.0), 0x0000);
    assert_eq!(f32_to_bf16(-0.0), 0x8000);
    assert_eq!(bf16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
    assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    let s_nan = f32::from_bits(0x7f80_0001);
    assert!(bf16_to_f32(f32_to_bf16(s_nan)).is_nan());

    // FP16 normal, subnormal, overflow, signed zero, ties-to-even, NaN
    assert_eq!(f32_to_f16(0.0), 0x0000);
    assert_eq!(f32_to_f16(-0.0), 0x8000);
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
    assert_eq!(f16_to_f32(f32_to_f16(-65504.0)), -65504.0);
    assert_eq!(f32_to_f16(65536.0), 0x7c00);
    assert_eq!(f32_to_f16(-65536.0), 0xfc00);
    let min_sub = f32::from_bits(0x3380_0000); // Exactly 2^-24.
    assert_eq!(f32_to_f16(min_sub), 0x0001);
    assert_eq!(f16_to_f32(0x0001), min_sub);
    assert_eq!(f32_to_f16(min_sub * 0.5), 0x0000);
    assert_eq!(f32_to_f16(min_sub * 1.5), 0x0002);
    assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());

    let src = Tensor::from_vec([2, 3], vec![-2.5, -0.125, 0.0, 0.5, 1.0, 3.75]).unwrap();
    let bf = Bf16Tensor::from_f32(&src).unwrap();
    let rt = bf.to_f32().unwrap();
    assert_eq!(rt.data(), src.data());

    let idx = IndexTensor::from_vec([2, 2], vec![0, 3, -100, 7]).unwrap();
    assert_eq!(idx.shape().dims(), &[2, 2]);
    assert_eq!(idx.data(), &[0, 3, -100, 7]);
}

#[test]
fn embedding_cpu_forward_backward_padding_and_validation() {
    let indices = IndexTensor::from_vec([2, 3], vec![0, 1, 3, 3, 4, 1]).unwrap();
    let spec = EmbeddingSpec::new(5, 4, indices.shape(), Some(1)).unwrap();
    let weight =
        Tensor::from_vec([5, 4], (0..20).map(|i| (i as f32) * 0.25 - 1.5).collect()).unwrap();
    let mut out = Tensor::zeros([2, 3, 4]).unwrap();

    let fwd = EmbeddingPlan::new(spec.clone());
    fwd.run(&weight, &indices, &mut out).unwrap();
    for pos in 0..6 {
        let tok = indices.data()[pos] as usize;
        for d in 0..4 {
            assert_eq!(out.data()[pos * 4 + d], weight.data()[tok * 4 + d]);
        }
    }

    let dy =
        Tensor::from_vec([2, 3, 4], (0..24).map(|i| (i as f32 + 1.0) * 0.1).collect()).unwrap();
    let bwd = EmbeddingBackward::new(spec);
    let mut dw = Tensor::from_vec([5, 4], vec![9.0; 20]).unwrap();
    bwd.run(&indices, &dy, &mut dw, GradientMode::Overwrite)
        .unwrap();

    let mut expected = vec![0.0f32; 20];
    for pos in 0..6 {
        let tok = indices.data()[pos] as usize;
        if tok == 1 {
            continue; // padding_idx skipped in backward
        }
        for d in 0..4 {
            expected[tok * 4 + d] += dy.data()[pos * 4 + d];
        }
    }
    assert_close(dw.data(), &expected, 1e-6, "embedding bwd overwrite");

    bwd.run(&indices, &dy, &mut dw, GradientMode::Accumulate)
        .unwrap();
    let expected2: Vec<f32> = expected.iter().map(|v| v * 2.0).collect();
    assert_close(dw.data(), &expected2, 1e-6, "embedding bwd accumulate");

    // Out-of-bounds token index rejected
    let bad = IndexTensor::from_vec([2, 3], vec![0, 5, 1, 2, 3, 4]).unwrap();
    assert!(fwd.run(&weight, &bad, &mut out).is_err());
    let neg = IndexTensor::from_vec([2, 3], vec![0, -1, 1, 2, 3, 4]).unwrap();
    assert!(
        bwd.run(&neg, &dy, &mut dw, GradientMode::Overwrite)
            .is_err()
    );
}

#[test]
fn rope_cpu_layouts_orthogonality_finite_diff_and_tape() {
    for &seq_axis in &[1usize, 2usize] {
        let dims = if seq_axis == 2 {
            [2, 3, 5, 10]
        } else {
            [2, 5, 3, 10]
        };
        let shape = Shape::new(dims).unwrap();
        let spec = RopeSpec::new(&shape, seq_axis, Some(6), 10_000.0, 3).unwrap();
        let layout = Layout::contiguous(dims).unwrap();
        let fwd = RopePlan::new(&spec, &layout, &layout).unwrap();
        let bwd = RopeBackward::new(&spec, &layout, &layout).unwrap();
        assert!(!bwd.input_required());

        let n = shape.len();
        let x = Tensor::from_vec(
            dims,
            (0..n).map(|i| ((i as f32 * 0.37).sin()) * 1.2).collect(),
        )
        .unwrap();
        let mut y = Tensor::zeros(dims).unwrap();
        fwd.run(&x.view(&layout).unwrap(), &mut y.view_mut(&layout).unwrap())
            .unwrap();

        let mut recovered = Tensor::zeros(dims).unwrap();
        bwd.run(
            &y.view(&layout).unwrap(),
            &mut recovered.view_mut(&layout).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
        assert_close(recovered.data(), x.data(), 1e-5, "rope orthogonality");

        let dy = Tensor::from_vec(
            dims,
            (0..n)
                .map(|i| ((i as f32 * 0.23 + 0.4).cos()) * 0.8)
                .collect(),
        )
        .unwrap();
        let mut dx = Tensor::zeros(dims).unwrap();
        bwd.run(
            &dy.view(&layout).unwrap(),
            &mut dx.view_mut(&layout).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

        let eps = 1e-3f32;
        for idx in (0..n).step_by(7) {
            let mut xp = x.data().to_vec();
            let mut xm = x.data().to_vec();
            xp[idx] += eps;
            xm[idx] -= eps;
            let txp = Tensor::from_vec(dims, xp).unwrap();
            let txm = Tensor::from_vec(dims, xm).unwrap();
            let mut yp = Tensor::zeros(dims).unwrap();
            let mut ym = Tensor::zeros(dims).unwrap();
            fwd.run(
                &txp.view(&layout).unwrap(),
                &mut yp.view_mut(&layout).unwrap(),
            )
            .unwrap();
            fwd.run(
                &txm.view(&layout).unwrap(),
                &mut ym.view_mut(&layout).unwrap(),
            )
            .unwrap();
            let lp: f64 = yp
                .data()
                .iter()
                .zip(dy.data())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            let lm: f64 = ym
                .data()
                .iter()
                .zip(dy.data())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
            assert!(
                (dx.data()[idx] - num).abs() < 5e-4,
                "rope fd mismatch seq_axis={seq_axis} idx={idx}: analytic={} num={num}",
                dx.data()[idx]
            );
        }

        // Autograd Tape composition: sum((rope(x))^2) -> grad == 2*x
        let mut b = Builder::new().unwrap();
        let lx = b.leaf(dims, true).unwrap();
        let r = b.rope(lx, seq_axis, Some(6), 10_000.0, 3).unwrap();
        let sq = b.unary(UnaryOp::Square, r).unwrap();
        let loss = b.reduce(ReduceOp::Sum, sq, &[0, 1, 2, 3], false).unwrap();
        let prog = b.finish(&[loss]).unwrap();
        let owner = Shared::new(x.clone()).unwrap();
        let mut tape = Tape::new(Cpu::Reference, &prog, &[Binding::new(lx, owner)]).unwrap();
        tape.forward().unwrap();
        tape.backward_scalar(GradientMode::Overwrite).unwrap();
        let grad = tape.gradient(lx).unwrap().unwrap();
        let expected_grad: Vec<f32> = x.data().iter().map(|&v| 2.0 * v).collect();
        assert_close(grad.data(), &expected_grad, 1e-4, "rope tape norm gradient");
    }
}

#[test]
fn cross_entropy_cpu_forward_backward_ignore_and_smoothing() {
    for &ls in &[0.0f32, 0.1f32] {
        let spec = CrossEntropySpec::new(4, 7, Some(-100), ls).unwrap();
        let fwd = CrossEntropyPlan::new(spec);
        let bwd = CrossEntropyBackward::new(spec);
        let logits = Tensor::from_vec(
            [4, 7],
            (0..28)
                .map(|i| ((i as f32 * 0.41).sin()) * 2.5 - 0.3)
                .collect(),
        )
        .unwrap();
        let targets = IndexTensor::from_vec([4], vec![2, -100, 6, 0]).unwrap();
        let mut loss_t = Tensor::zeros([]).unwrap();
        fwd.run(&logits, &targets, &mut loss_t).unwrap();
        let loss = loss_t.data()[0];

        let mut dlogits = Tensor::zeros([4, 7]).unwrap();
        bwd.run(
            &logits,
            &targets,
            1.0,
            &mut dlogits,
            GradientMode::Overwrite,
        )
        .unwrap();

        let mut ref_sum = 0.0f64;
        let mut valid = 0usize;
        for r in 0..4 {
            let t = targets.data()[r];
            if t == -100 {
                continue;
            }
            valid += 1;
            let row = &logits.data()[r * 7..(r + 1) * 7];
            let m = row
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |a, b| a.max(b as f64));
            let lse = m + row
                .iter()
                .map(|&v| ((v as f64) - m).exp())
                .sum::<f64>()
                .ln();
            let nll = lse - row[t as usize] as f64;
            let smooth = lse - row.iter().map(|&v| v as f64).sum::<f64>() / 7.0;
            ref_sum += (1.0 - ls as f64) * nll + (ls as f64) * smooth;
        }
        let ref_loss = (ref_sum / valid as f64) as f32;
        assert!((loss - ref_loss).abs() < 1e-6, "loss={loss} ref={ref_loss}");
        assert_eq!(&dlogits.data()[7..14], &[0.0; 7]);

        let eps = 1e-3f32;
        let mut lp_t = Tensor::zeros([]).unwrap();
        let mut lm_t = Tensor::zeros([]).unwrap();
        for idx in 0..28 {
            let mut lp_v = logits.data().to_vec();
            let mut lm_v = logits.data().to_vec();
            lp_v[idx] += eps;
            lm_v[idx] -= eps;
            fwd.run(
                &Tensor::from_vec([4, 7], lp_v).unwrap(),
                &targets,
                &mut lp_t,
            )
            .unwrap();
            fwd.run(
                &Tensor::from_vec([4, 7], lm_v).unwrap(),
                &targets,
                &mut lm_t,
            )
            .unwrap();
            let num = (lp_t.data()[0] - lm_t.data()[0]) / (2.0 * eps);
            assert!(
                (dlogits.data()[idx] - num).abs() < 5e-4,
                "ce fd mismatch ls={ls} idx={idx}: analytic={} num={num}",
                dlogits.data()[idx]
            );
        }
    }

    let spec = CrossEntropySpec::new(2, 5, Some(-100), 0.0).unwrap();
    let logits = Tensor::from_vec([2, 5], vec![1.0; 10]).unwrap();
    let targets = IndexTensor::from_vec([2], vec![-100, -100]).unwrap();
    let mut out = Tensor::from_vec([], vec![5.0]).unwrap();
    CrossEntropyPlan::new(spec)
        .run(&logits, &targets, &mut out)
        .unwrap();
    assert_eq!(out.data()[0], 0.0);
    let mut grad = Tensor::from_vec([2, 5], vec![7.0; 10]).unwrap();
    CrossEntropyBackward::new(spec)
        .run(&logits, &targets, 1.0, &mut grad, GradientMode::Overwrite)
        .unwrap();
    assert_eq!(grad.data(), &[0.0; 10]);
}

#[test]
fn cross_entropy_preserves_loss_at_large_common_logit_offsets() {
    for vocab in [7, 131] {
        let logits = Tensor::from_vec([1, vocab], vec![1e20; vocab]).unwrap();
        let targets = IndexTensor::from_vec([1], vec![2]).unwrap();
        let mut loss = Tensor::zeros([]).unwrap();
        let mut grad = Tensor::zeros([1, vocab]).unwrap();
        for smoothing in [0.0, 0.1] {
            let spec = CrossEntropySpec::new(1, vocab, None, smoothing).unwrap();
            CrossEntropyPlan::new(spec)
                .run(&logits, &targets, &mut loss)
                .unwrap();
            CrossEntropyBackward::new(spec)
                .run(&logits, &targets, 1.0, &mut grad, GradientMode::Overwrite)
                .unwrap();
            let expected_loss = (vocab as f64).ln() as f32;
            assert!((loss.data()[0] - expected_loss).abs() < 1e-6);
            let other = (1.0 - smoothing) / vocab as f32;
            for (class, &actual) in grad.data().iter().enumerate() {
                let expected = if class == 2 {
                    other - (1.0 - smoothing)
                } else {
                    other
                };
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn dropout_philox_deterministic_and_unbiased() {
    let shape = Shape::new([64, 64]).unwrap();
    let layout = Layout::contiguous([64, 64]).unwrap();
    let spec = DropoutSpec::new(&shape, 0.25, 12345).unwrap();
    let fwd = DropoutPlan::new(&spec, &layout, &layout).unwrap();
    let bwd = DropoutBackward::new(&spec, &layout, &layout).unwrap();

    let n = shape.len();
    let x = Tensor::from_vec([64, 64], vec![1.0; n]).unwrap();
    let mut y1 = Tensor::zeros([64, 64]).unwrap();
    let mut y2 = Tensor::zeros([64, 64]).unwrap();
    let mut y3 = Tensor::zeros([64, 64]).unwrap();
    fwd.run(
        &x.view(&layout).unwrap(),
        &mut y1.view_mut(&layout).unwrap(),
        7,
    )
    .unwrap();
    fwd.run(
        &x.view(&layout).unwrap(),
        &mut y2.view_mut(&layout).unwrap(),
        7,
    )
    .unwrap();
    fwd.run(
        &x.view(&layout).unwrap(),
        &mut y3.view_mut(&layout).unwrap(),
        8,
    )
    .unwrap();
    assert_eq!(y1.data(), y2.data());
    assert_ne!(y1.data(), y3.data());

    let scale = 1.0f32 / (1.0 - 0.25);
    let kept = y1.data().iter().filter(|&&v| v > 0.0).count();
    let keep_frac = kept as f64 / n as f64;
    assert!((keep_frac - 0.75).abs() < 0.03, "keep_frac={keep_frac}");
    for &v in y1.data() {
        assert!(v == 0.0 || (v - scale).abs() < 1e-6);
    }

    let dy = Tensor::from_vec([64, 64], (0..n).map(|i| (i as f32) * 0.01 + 1.0).collect()).unwrap();
    let mut dx = Tensor::zeros([64, 64]).unwrap();
    bwd.run(
        &dy.view(&layout).unwrap(),
        &mut dx.view_mut(&layout).unwrap(),
        7,
        GradientMode::Overwrite,
    )
    .unwrap();
    for i in 0..n {
        let expected = if y1.data()[i] > 0.0 {
            dy.data()[i] * scale
        } else {
            0.0
        };
        assert!((dx.data()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn phase_ab_miri_dtype_embedding_rope_loss_dropout_and_tape() {
    let src = Tensor::from_vec([2, 2], vec![-1.5, 0.25, 0.0, 2.0]).unwrap();
    let bf = Bf16Tensor::from_f32(&src).unwrap();
    assert_eq!(bf.to_f32().unwrap().data(), src.data());

    let idx = IndexTensor::from_vec([1, 3], vec![0, 2, 3]).unwrap();
    let emb_spec = EmbeddingSpec::new(4, 4, idx.shape(), Some(0)).unwrap();
    let w = Tensor::from_vec([4, 4], (0..16).map(|i| i as f32 * 0.1).collect()).unwrap();
    let mut emb_out = Tensor::zeros([1, 3, 4]).unwrap();
    EmbeddingPlan::new(emb_spec.clone())
        .run(&w, &idx, &mut emb_out)
        .unwrap();
    let mut dw = Tensor::zeros([4, 4]).unwrap();
    EmbeddingBackward::new(emb_spec)
        .run(&idx, &emb_out, &mut dw, GradientMode::Overwrite)
        .unwrap();
    assert_eq!(&dw.data()[0..4], &[0.0; 4]);

    let dims = [1, 2, 3, 4];
    let x = Tensor::from_vec(dims, (0..24).map(|i| i as f32 * 0.05 - 0.5).collect()).unwrap();
    let mut b = Builder::new().unwrap();
    let lx = b.leaf(dims, true).unwrap();
    let r = b.rope(lx, 2, Some(4), 10_000.0, 1).unwrap();
    let sq = b.unary(UnaryOp::Square, r).unwrap();
    let loss = b.reduce(ReduceOp::Sum, sq, &[0, 1, 2, 3], false).unwrap();
    let prog = b.finish(&[loss]).unwrap();
    let owner = Shared::new(x).unwrap();
    let mut tape = Tape::new(Cpu::Reference, &prog, &[Binding::new(lx, owner)]).unwrap();
    tape.forward().unwrap();
    tape.backward_scalar(GradientMode::Overwrite).unwrap();
    assert!(tape.gradient(lx).unwrap().is_some());

    let ce_spec = CrossEntropySpec::new(2, 4, Some(-100), 0.1).unwrap();
    let logits = Tensor::from_vec([2, 4], vec![0.1, -0.2, 0.5, 0.3, -0.4, 0.2, 0.1, 0.8]).unwrap();
    let targets = IndexTensor::from_vec([2], vec![2, -100]).unwrap();
    let mut ce_loss = Tensor::zeros([]).unwrap();
    CrossEntropyPlan::new(ce_spec)
        .run(&logits, &targets, &mut ce_loss)
        .unwrap();
    assert!(ce_loss.data()[0].is_finite() && ce_loss.data()[0] > 0.0);
    let mut dlogits = Tensor::zeros([2, 4]).unwrap();
    CrossEntropyBackward::new(ce_spec)
        .run(
            &logits,
            &targets,
            1.0,
            &mut dlogits,
            GradientMode::Overwrite,
        )
        .unwrap();

    let drop_shape = Shape::new([2, 4]).unwrap();
    let drop_layout = Layout::contiguous([2, 4]).unwrap();
    let drop_spec = DropoutSpec::new(&drop_shape, 0.2, 42).unwrap();
    let mut drop_out = Tensor::zeros([2, 4]).unwrap();
    DropoutPlan::new(&drop_spec, &drop_layout, &drop_layout)
        .unwrap()
        .run(
            &logits.view(&drop_layout).unwrap(),
            &mut drop_out.view_mut(&drop_layout).unwrap(),
            1,
        )
        .unwrap();
    let mut drop_dx = Tensor::zeros([2, 4]).unwrap();
    DropoutBackward::new(&drop_spec, &drop_layout, &drop_layout)
        .unwrap()
        .run(
            &drop_out.view(&drop_layout).unwrap(),
            &mut drop_dx.view_mut(&drop_layout).unwrap(),
            1,
            GradientMode::Overwrite,
        )
        .unwrap();
}
