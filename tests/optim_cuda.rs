#![cfg(feature = "cuda")]

use fast_audio::{
    AdamWConfig, BinaryOp, Checkpoint, ReduceOp, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu,
    cuda::{self, Cuda},
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
fn cuda_adamw_captured_dynamic_lr_and_cross_device_checkpoint() {
    let cuda = Cuda::new(0).unwrap();

    // Multi-layer audio block: Conv1d -> RmsNorm -> SiLU -> scale+bias -> MSE -> AdamW
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 2, 16], false).unwrap();
    let w = b.leaf([4, 2, 3], true).unwrap();
    let gamma = b.leaf([1, 4, 1], true).unwrap();
    let bias = b.leaf([1, 4, 1], true).unwrap();
    let target = b.leaf([2, 4, 16], false).unwrap();

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
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.02,
        max_grad_norm: Some(0.25),
    };

    let hx = Tensor::from_vec([2, 2, 16], pattern(64, 1)).unwrap();
    let hw = Tensor::from_vec([4, 2, 3], pattern(24, 2)).unwrap();
    let hg = Tensor::from_vec([1, 4, 1], pattern(4, 3)).unwrap();
    let hb = Tensor::from_vec([1, 4, 1], pattern(4, 4)).unwrap();
    let ht = Tensor::from_vec([2, 4, 16], pattern(128, 5)).unwrap();

    // 1. CPU reference (8 steps with dynamic LR every step)
    let cx = Shared::new(hx.clone()).unwrap();
    let cw = Shared::new(hw.clone()).unwrap();
    let cg = Shared::new(hg.clone()).unwrap();
    let cb = Shared::new(hb.clone()).unwrap();
    let ct = Shared::new(ht.clone()).unwrap();
    let mut cpu_tape = cpu::Tape::new(
        cpu::Cpu::Reference,
        &program,
        &[
            Binding::new(x, cx),
            Binding::new(w, cw.clone()),
            Binding::new(gamma, cg.clone()),
            Binding::new(bias, cb.clone()),
            Binding::new(target, ct),
        ],
    )
    .unwrap();
    let mut cpu_opt = cpu::AdamW::for_program(&program, &[w, gamma, bias], cfg).unwrap();

    let make_gpu = || {
        let gx = Shared::new(cuda.upload(&hx).unwrap()).unwrap();
        let gw = Shared::new(cuda.upload(&hw).unwrap()).unwrap();
        let gg = Shared::new(cuda.upload(&hg).unwrap()).unwrap();
        let gb = Shared::new(cuda.upload(&hb).unwrap()).unwrap();
        let gt = Shared::new(cuda.upload(&ht).unwrap()).unwrap();
        let tape = cuda::Tape::new(
            &cuda,
            &program,
            &[
                Binding::new(x, gx),
                Binding::new(w, gw.clone()),
                Binding::new(gamma, gg.clone()),
                Binding::new(bias, gb.clone()),
                Binding::new(target, gt),
            ],
        )
        .unwrap();
        let opt = cuda::AdamW::for_program(&cuda, &program, &[w, gamma, bias], cfg).unwrap();
        (tape, opt, gw, gg, gb)
    };

    let (mut gpu_uncap, mut opt_uncap, wu, gu, bu) = make_gpu();
    let (mut gpu_cap, mut opt_cap, wc, gc, bc) = make_gpu();

    for step in 0..8usize {
        let lr = 0.02 * (1.0 - 0.08 * step as f32);
        cpu_opt.set_lr(lr).unwrap();
        opt_uncap.set_lr(lr).unwrap();
        opt_cap.set_lr(lr).unwrap();

        cpu_tape.step_scalar_adamw(&mut cpu_opt).unwrap();
        gpu_uncap.step_scalar_adamw(&mut opt_uncap).unwrap();
        gpu_cap.step_scalar_adamw_captured(&mut opt_cap).unwrap();
        if step >= 1 {
            assert!(gpu_cap.has_captured_graph());
        }
    }
    cuda.synchronize().unwrap();
    assert_eq!(opt_uncap.step_count(), 8);
    assert_eq!(opt_cap.step_count(), 8);

    // Compare CPU vs GPU uncaptured vs GPU captured
    for (name, hc, du, dc) in [
        (
            "w",
            cw.read().unwrap(),
            wu.read().unwrap(),
            wc.read().unwrap(),
        ),
        (
            "gamma",
            cg.read().unwrap(),
            gu.read().unwrap(),
            gc.read().unwrap(),
        ),
        (
            "bias",
            cb.read().unwrap(),
            bu.read().unwrap(),
            bc.read().unwrap(),
        ),
    ] {
        let hu = cuda.download(&du).unwrap();
        let hcap = cuda.download(&dc).unwrap();
        assert!(
            max_abs_diff(hu.data(), hcap.data()) < 1e-6,
            "CUDA captured vs uncaptured mismatch on {name}"
        );
        assert!(
            max_abs_diff(hc.data(), hu.data()) < 3e-5,
            "CUDA vs CPU mismatch on {name}: diff={}",
            max_abs_diff(hc.data(), hu.data())
        );
    }

    // 2. Cross-device Checkpoint roundtrip: 4 steps on CUDA -> save -> restore on fresh CUDA -> 4 more captured steps
    let (mut g_a, mut o_a, wa, ga, ba) = make_gpu();
    for step in 0..4usize {
        let lr = 0.02 * (1.0 - 0.08 * step as f32);
        o_a.set_lr(lr).unwrap();
        g_a.step_scalar_adamw_captured(&mut o_a).unwrap();
    }
    let ckpt = Checkpoint::from_cuda_state(
        &cuda,
        &[("w", &wa), ("gamma", &ga), ("bias", &ba)],
        Some((&o_a, &[("w", w), ("gamma", gamma), ("bias", bias)])),
    )
    .unwrap();
    let bytes = ckpt.to_bytes().unwrap();
    let decoded = Checkpoint::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.step_count(), 4);

    // Restore onto CPU first to verify cross-device portability
    let tmp_w = Shared::new(Tensor::zeros([4, 2, 3]).unwrap()).unwrap();
    let tmp_g = Shared::new(Tensor::zeros([1, 4, 1]).unwrap()).unwrap();
    let tmp_b = Shared::new(Tensor::zeros([1, 4, 1]).unwrap()).unwrap();
    let mut tmp_cpu_opt = cpu::AdamW::for_program(&program, &[w, gamma, bias], cfg).unwrap();
    decoded
        .restore_cpu_state(
            &[("w", &tmp_w), ("gamma", &tmp_g), ("bias", &tmp_b)],
            Some((
                &mut tmp_cpu_opt,
                &[("w", w), ("gamma", gamma), ("bias", bias)],
            )),
        )
        .unwrap();
    assert_eq!(tmp_cpu_opt.step_count(), 4);
    assert_eq!(
        tmp_w.read().unwrap().data(),
        cuda.download(&wa.read().unwrap()).unwrap().data()
    );

    // Restore onto fresh CUDA tape + optimizer and resume steps 4..8
    let (mut g_b, mut o_b, wb, gb, bb) = make_gpu();
    decoded
        .restore_cuda_state(
            &cuda,
            &[("w", &wb), ("gamma", &gb), ("bias", &bb)],
            Some((&mut o_b, &[("w", w), ("gamma", gamma), ("bias", bias)])),
        )
        .unwrap();
    assert_eq!(o_b.step_count(), 4);
    for step in 4..8usize {
        let lr = 0.02 * (1.0 - 0.08 * step as f32);
        o_b.set_lr(lr).unwrap();
        g_b.step_scalar_adamw_captured(&mut o_b).unwrap();
    }
    cuda.synchronize().unwrap();
    assert_eq!(o_b.step_count(), 8);
    for (a, b) in [(&wc, &wb), (&gc, &gb), (&bc, &bb)] {
        let ha = cuda.download(&a.read().unwrap()).unwrap();
        let hb = cuda.download(&b.read().unwrap()).unwrap();
        assert!(max_abs_diff(ha.data(), hb.data()) < 1e-6);
    }
}
