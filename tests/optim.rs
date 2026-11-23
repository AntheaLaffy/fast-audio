use fast_audio::{
    AdamWConfig, BinaryOp, Checkpoint, GradientMode, ReduceOp, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{AdamW, Cpu, Tape},
};

fn pattern(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 41 + seed * 17 + 3) % 31) as f32 - 15.0) * 0.08)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn cpu_adamw_exact_multistep_weight_decay_and_grad_clip_matches_fp64_oracle() {
    let mut b = Builder::new().unwrap();
    let p1 = b.leaf([2, 5], true).unwrap();
    let p2 = b.leaf([5], true).unwrap();
    let s1 = b.reduce(ReduceOp::Sum, p1, &[0], false).unwrap();
    let sum = b.binary(BinaryOp::Add, s1, p2).unwrap();
    let loss = b.reduce(ReduceOp::Sum, sum, &[0], false).unwrap();
    let program = b.finish(&[loss]).unwrap();

    // Config validation checks
    assert!(
        AdamWConfig {
            lr: -1.0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        AdamWConfig {
            beta1: 1.0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        AdamWConfig {
            max_grad_norm: Some(0.0),
            ..Default::default()
        }
        .validate()
        .is_err()
    );

    let cfg = AdamWConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.05,
        max_grad_norm: Some(0.5),
    };
    let mut opt = AdamW::for_program(&program, &[p1, p2], cfg).unwrap();
    assert_eq!(opt.step_count(), 0);
    assert_eq!(opt.state_bytes(), (10 + 5) * 4 * 2);

    let mut t1 = Tensor::from_vec([2, 5], pattern(10, 1)).unwrap();
    let mut t2 = Tensor::from_vec([5], pattern(5, 2)).unwrap();

    let mut r1: Vec<f64> = t1.data().iter().map(|&v| v as f64).collect();
    let mut r2: Vec<f64> = t2.data().iter().map(|&v| v as f64).collect();
    let mut m1 = [0.0f64; 10];
    let mut v1 = [0.0f64; 10];
    let mut m2 = [0.0f64; 5];
    let mut v2 = [0.0f64; 5];

    for step in 1..=6usize {
        let lr = 0.01 * (1.0 - 0.1 * (step - 1) as f32);
        opt.set_lr(lr).unwrap();
        let g1 = Tensor::from_vec([2, 5], pattern(10, 10 + step)).unwrap();
        let g2 = Tensor::from_vec([5], pattern(5, 20 + step)).unwrap();
        opt.apply_slices(&mut [&mut t1, &mut t2], &[&g1, &g2])
            .unwrap();

        // Independent FP64 oracle
        let total_sq: f64 = g1
            .data()
            .iter()
            .chain(g2.data())
            .map(|&x| (x as f64) * (x as f64))
            .sum();
        let clip = 0.5 / (total_sq.sqrt() + 1e-6);
        let scale = clip.min(1.0);
        let bc1 = 1.0 - (cfg.beta1 as f64).powi(step as i32);
        let bc2 = 1.0 - (cfg.beta2 as f64).powi(step as i32);
        let step_size = lr as f64 / bc1;
        let decay = 1.0 - lr as f64 * cfg.weight_decay as f64;
        for i in 0..10 {
            let g = g1.data()[i] as f64 * scale;
            m1[i] = cfg.beta1 as f64 * m1[i] + (1.0 - cfg.beta1 as f64) * g;
            v1[i] = cfg.beta2 as f64 * v1[i] + (1.0 - cfg.beta2 as f64) * g * g;
            let denom = v1[i].sqrt() / bc2.sqrt() + cfg.eps as f64;
            r1[i] = r1[i] * decay - step_size * (m1[i] / denom);
        }
        for i in 0..5 {
            let g = g2.data()[i] as f64 * scale;
            m2[i] = cfg.beta1 as f64 * m2[i] + (1.0 - cfg.beta1 as f64) * g;
            v2[i] = cfg.beta2 as f64 * v2[i] + (1.0 - cfg.beta2 as f64) * g * g;
            let denom = v2[i].sqrt() / bc2.sqrt() + cfg.eps as f64;
            r2[i] = r2[i] * decay - step_size * (m2[i] / denom);
        }
        assert_eq!(opt.step_count(), step as u64);
    }

    let r1_f32: Vec<f32> = r1.iter().map(|&v| v as f32).collect();
    let r2_f32: Vec<f32> = r2.iter().map(|&v| v as f32).collect();
    assert!(max_abs_diff(t1.data(), &r1_f32) < 1e-5);
    assert!(max_abs_diff(t2.data(), &r2_f32) < 1e-5);
}

#[test]
fn cpu_tape_adamw_checkpoint_bit_exact_and_corruption_rejected() {
    // Audio block with barrier + fusible tail:
    // x [2, 2, 12] -> Conv1d(k=3,p=1) -> RmsNorm -> SiLU -> scale * h + bias -> MSE -> AdamW
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 2, 12], false).unwrap();
    let w = b.leaf([4, 2, 3], true).unwrap();
    let gamma = b.leaf([1, 4, 1], true).unwrap();
    let bias = b.leaf([1, 4, 1], true).unwrap();
    let target = b.leaf([2, 4, 12], false).unwrap();

    let conv = b.conv1d(x, w, 1, 1, 1, 1).unwrap();
    let norm = b.rms_norm(conv, 1, 1e-5).unwrap();
    let act = b.unary(UnaryOp::Silu, norm).unwrap();
    let scaled = b.binary(BinaryOp::Mul, act, gamma).unwrap();
    let shifted = b.binary(BinaryOp::Add, scaled, bias).unwrap();
    let diff = b.binary(BinaryOp::Sub, shifted, target).unwrap();
    let sq = b.unary(UnaryOp::Square, diff).unwrap();
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1, 2], false).unwrap();
    let program = b.finish(&[loss]).unwrap();

    let cfg = AdamWConfig {
        lr: 0.02,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
        max_grad_norm: Some(1.0),
    };

    let make = || {
        let sx = Shared::new(Tensor::from_vec([2, 2, 12], pattern(48, 1)).unwrap()).unwrap();
        let sw = Shared::new(Tensor::from_vec([4, 2, 3], pattern(24, 2)).unwrap()).unwrap();
        let sg = Shared::new(Tensor::from_vec([1, 4, 1], pattern(4, 3)).unwrap()).unwrap();
        let sb = Shared::new(Tensor::from_vec([1, 4, 1], pattern(4, 4)).unwrap()).unwrap();
        let st = Shared::new(Tensor::from_vec([2, 4, 12], pattern(96, 5)).unwrap()).unwrap();
        let tape = Tape::new(
            Cpu::Reference,
            &program,
            &[
                Binding::new(x, sx.clone()),
                Binding::new(w, sw.clone()),
                Binding::new(gamma, sg.clone()),
                Binding::new(bias, sb.clone()),
                Binding::new(target, st.clone()),
            ],
        )
        .unwrap();
        let opt = AdamW::for_program(&program, &[w, gamma, bias], cfg).unwrap();
        (tape, opt, sw, sg, sb)
    };

    // Uninterrupted 10 steps
    let (mut t_full, mut o_full, w_full, g_full, b_full) = make();
    t_full.forward().unwrap();
    let loss0 = t_full.output(loss).unwrap().storage().data()[0];
    for step in 0..10 {
        let lr = 0.02 * (1.0 - 0.05 * step as f32);
        o_full.set_lr(lr).unwrap();
        t_full.step_scalar_adamw(&mut o_full).unwrap();
    }
    assert!(t_full.has_fused_plan());
    t_full.forward().unwrap();
    let loss10 = t_full.output(loss).unwrap().storage().data()[0];
    assert!(
        loss10 < loss0,
        "AdamW did not reduce loss: {loss0} -> {loss10}"
    );

    // 5 steps -> serialize Checkpoint -> deserialize -> restore into fresh Tape+AdamW -> 5 more steps
    let (mut t_a, mut o_a, w_a, g_a, b_a) = make();
    for step in 0..5 {
        let lr = 0.02 * (1.0 - 0.05 * step as f32);
        o_a.set_lr(lr).unwrap();
        t_a.step_scalar_adamw(&mut o_a).unwrap();
    }
    let ckpt = Checkpoint::from_cpu_state(
        &[("w", &w_a), ("gamma", &g_a), ("bias", &b_a)],
        Some((&o_a, &[("w", w), ("gamma", gamma), ("bias", bias)])),
    )
    .unwrap();
    let bytes = ckpt.to_bytes().unwrap();

    // Corruption & truncation checks
    assert!(Checkpoint::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut bad = bytes.clone();
    bad[16] ^= 0x01;
    assert!(Checkpoint::from_bytes(&bad).is_err());

    let restored = Checkpoint::from_bytes(&bytes).unwrap();
    assert_eq!(restored.step_count(), 5);
    assert_eq!(restored.config().unwrap().max_grad_norm, Some(1.0));

    let (mut t_b, mut o_b, w_b, g_b, b_b) = make();
    restored
        .restore_cpu_state(
            &[("w", &w_b), ("gamma", &g_b), ("bias", &b_b)],
            Some((&mut o_b, &[("w", w), ("gamma", gamma), ("bias", bias)])),
        )
        .unwrap();
    assert_eq!(o_b.step_count(), 5);

    for step in 5..10 {
        let lr = 0.02 * (1.0 - 0.05 * step as f32);
        o_b.set_lr(lr).unwrap();
        t_b.step_scalar_adamw(&mut o_b).unwrap();
    }

    // Bit-identical parameters and moments!
    assert_eq!(w_full.read().unwrap().data(), w_b.read().unwrap().data());
    assert_eq!(g_full.read().unwrap().data(), g_b.read().unwrap().data());
    assert_eq!(b_full.read().unwrap().data(), b_b.read().unwrap().data());
    for leaf in [w, gamma, bias] {
        let (mf, vf) = o_full.moments(leaf).unwrap();
        let (mb, vb) = o_b.moments(leaf).unwrap();
        assert_eq!(mf.data(), mb.data());
        assert_eq!(vf.data(), vb.data());
    }

    // Gradient accumulation check: 2 microbatches with apply_adamw
    let (mut t_acc, mut o_acc, _, _, _) = make();
    t_acc.forward().unwrap();
    t_acc.backward_scalar(GradientMode::Overwrite).unwrap();
    t_acc.forward().unwrap();
    t_acc.backward_scalar(GradientMode::Accumulate).unwrap();
    t_acc.apply_adamw(&mut o_acc).unwrap();
    assert_eq!(o_acc.step_count(), 1);
}

#[test]
fn cpu_adamw_and_checkpoint_miri() {
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 4], false).unwrap();
    let w = b.leaf([2, 4], true).unwrap();
    let bias = b.leaf([1, 4], true).unwrap();
    let prod = b.binary(BinaryOp::Mul, x, w).unwrap();
    let out = b.binary(BinaryOp::Add, prod, bias).unwrap();
    let sq = b.unary(UnaryOp::Square, out).unwrap();
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1], false).unwrap();
    let program = b.finish(&[loss]).unwrap();

    let sx = Shared::new(Tensor::from_vec([2, 4], pattern(8, 1)).unwrap()).unwrap();
    let sw = Shared::new(Tensor::from_vec([2, 4], pattern(8, 2)).unwrap()).unwrap();
    let sb = Shared::new(Tensor::from_vec([1, 4], pattern(4, 3)).unwrap()).unwrap();
    let mut tape = Tape::new(
        Cpu::Reference,
        &program,
        &[
            Binding::new(x, sx),
            Binding::new(w, sw.clone()),
            Binding::new(bias, sb.clone()),
        ],
    )
    .unwrap();
    let mut opt = AdamW::for_program(
        &program,
        &[w, bias],
        AdamWConfig {
            lr: 0.05,
            max_grad_norm: Some(0.5),
            ..Default::default()
        },
    )
    .unwrap();
    tape.step_scalar_adamw(&mut opt).unwrap();
    tape.step_scalar_adamw(&mut opt).unwrap();
    assert_eq!(opt.step_count(), 2);

    let ckpt = Checkpoint::from_cpu_state(
        &[("w", &sw), ("b", &sb)],
        Some((&opt, &[("w", w), ("b", bias)])),
    )
    .unwrap();
    let bytes = ckpt.to_bytes().unwrap();
    let back = Checkpoint::from_bytes(&bytes).unwrap();
    back.restore_cpu_state(
        &[("w", &sw), ("b", &sb)],
        Some((&mut opt, &[("w", w), ("b", bias)])),
    )
    .unwrap();
    assert_eq!(opt.step_count(), 2);
}
