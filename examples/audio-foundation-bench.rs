//! Comprehensive audio deep learning foundation benchmark & PyTorch verification runner.
//!
//! Supports both CPU (`Cpu::Reference` or `Cpu::Mkl(threads)` when built with `--features mkl`)
//! and CUDA (`--features cuda`), emitting machine-readable JSON for paired head-to-head comparison
//! against PyTorch 2.14.1 (`tools/bench_foundation_vs_pytorch.py` & `tools/check_new_ops.py`).

use fast_audio::{
    AdamWConfig, BinaryOp, BinaryOperand, BmmSpec, CatSpec, Conv2dSpec, ConvTranspose1dSpec,
    GradientMode, Layout, NormKind, NormSpec, PadMode, PadSpec, ReduceOp, ReduceSpec, SdpaOperand,
    SdpaSpec, Shape, Shared, SoftmaxKind, SoftmaxSpec, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{self, Cpu},
};
use serde_json::{Value, json};
use std::time::Instant;

#[cfg(feature = "cuda")]
use fast_audio::cuda::{self, Cuda};

fn pattern(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i.wrapping_mul(37).wrapping_add(seed * 19 + 5)) % 29) as f32 - 14.0) / 32.0)
        .collect()
}

fn training_v1_recipe(
    shape: &[usize],
    multiplier: usize,
    add: usize,
    modulus: usize,
    offset: f32,
    divisor: f32,
) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((((i * multiplier + add) % modulus) as f32) + offset) / divisor)
        .collect();
    Tensor::from_vec(shape, data).unwrap()
}

fn time_cpu<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

#[cfg(feature = "cuda")]
fn time_cuda<F: FnMut()>(cuda: &Cuda, warmup: usize, iters: usize, mut f: F) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let timing = cuda
        .measure(iters, || {
            f();
            Ok(())
        })
        .unwrap();
    (timing.wall_us, timing.device_us)
}

fn run_verify_cpu() -> Value {
    // 1. LayerNorm + RmsNorm
    let norm_shape = Shape::new([2, 4, 16]).unwrap();
    let nl = Layout::contiguous(norm_shape.dims()).unwrap();
    let nx = Tensor::from_vec(norm_shape.dims(), pattern(128, 1)).unwrap();
    let ndy = Tensor::from_vec(norm_shape.dims(), pattern(128, 2)).unwrap();
    let ln_spec = NormSpec::new(NormKind::LayerNorm, &norm_shape, 1, 1e-5).unwrap();
    let rms_spec = NormSpec::new(NormKind::RmsNorm, &norm_shape, 1, 1e-5).unwrap();
    let mut ln_y = Tensor::zeros(norm_shape.dims()).unwrap();
    let mut ln_dx = Tensor::zeros(norm_shape.dims()).unwrap();
    cpu::NormPlan::new(&ln_spec, &nl, &nl)
        .unwrap()
        .run(&nx.view(&nl).unwrap(), &mut ln_y.view_mut(&nl).unwrap())
        .unwrap();
    cpu::NormBackward::new(&ln_spec, &nl, &nl, &nl)
        .unwrap()
        .run(
            &nx.view(&nl).unwrap(),
            &ndy.view(&nl).unwrap(),
            &mut ln_dx.view_mut(&nl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    let mut rms_y = Tensor::zeros(norm_shape.dims()).unwrap();
    let mut rms_dx = Tensor::zeros(norm_shape.dims()).unwrap();
    cpu::NormPlan::new(&rms_spec, &nl, &nl)
        .unwrap()
        .run(&nx.view(&nl).unwrap(), &mut rms_y.view_mut(&nl).unwrap())
        .unwrap();
    cpu::NormBackward::new(&rms_spec, &nl, &nl, &nl)
        .unwrap()
        .run(
            &nx.view(&nl).unwrap(),
            &ndy.view(&nl).unwrap(),
            &mut rms_dx.view_mut(&nl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    // 2. Softmax & LogSoftmax (axis 1 of [2, 5, 7])
    let sm_shape = Shape::new([2, 5, 7]).unwrap();
    let sml = Layout::contiguous(sm_shape.dims()).unwrap();
    let smx = Tensor::from_vec(sm_shape.dims(), pattern(70, 3)).unwrap();
    let smdy = Tensor::from_vec(sm_shape.dims(), pattern(70, 4)).unwrap();
    let sm_spec = SoftmaxSpec::new(SoftmaxKind::Softmax, &sm_shape, 1).unwrap();
    let lsm_spec = SoftmaxSpec::new(SoftmaxKind::LogSoftmax, &sm_shape, 1).unwrap();
    let mut sm_y = Tensor::zeros(sm_shape.dims()).unwrap();
    let mut sm_dx = Tensor::zeros(sm_shape.dims()).unwrap();
    cpu::SoftmaxPlan::new(&sm_spec, &sml, &sml)
        .unwrap()
        .run(&smx.view(&sml).unwrap(), &mut sm_y.view_mut(&sml).unwrap())
        .unwrap();
    cpu::SoftmaxBackward::new(&sm_spec, &sml, &sml, &sml)
        .unwrap()
        .run(
            &smx.view(&sml).unwrap(),
            &smdy.view(&sml).unwrap(),
            &mut sm_dx.view_mut(&sml).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    let mut lsm_y = Tensor::zeros(sm_shape.dims()).unwrap();
    let mut lsm_dx = Tensor::zeros(sm_shape.dims()).unwrap();
    cpu::SoftmaxPlan::new(&lsm_spec, &sml, &sml)
        .unwrap()
        .run(&smx.view(&sml).unwrap(), &mut lsm_y.view_mut(&sml).unwrap())
        .unwrap();
    cpu::SoftmaxBackward::new(&lsm_spec, &sml, &sml, &sml)
        .unwrap()
        .run(
            &smx.view(&sml).unwrap(),
            &smdy.view(&sml).unwrap(),
            &mut lsm_dx.view_mut(&sml).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    // 3. Nonlinear reductions (Max, Min, LogSumExp on [2, 4, 6] along axis 2)
    let r_shape = Shape::new([2, 4, 6]).unwrap();
    let rl = Layout::contiguous(r_shape.dims()).unwrap();
    let rx = Tensor::from_vec(r_shape.dims(), pattern(48, 5)).unwrap();
    let rdy = Tensor::from_vec([2, 4], pattern(8, 6)).unwrap();
    let rol = Layout::contiguous([2, 4]).unwrap();
    let run_red = |op: ReduceOp| -> (Vec<f32>, Vec<f32>) {
        let spec = ReduceSpec::new(op, &r_shape, &[2], false).unwrap();
        let mut y = Tensor::zeros([2, 4]).unwrap();
        let mut dx = Tensor::zeros(r_shape.dims()).unwrap();
        cpu::ReducePlan::new(&spec, &rl, &rol)
            .unwrap()
            .run(&rx.view(&rl).unwrap(), &mut y.view_mut(&rol).unwrap())
            .unwrap();
        cpu::ReduceBackward::with_input(&spec, &rl, &rol, &rl)
            .unwrap()
            .run_with_input(
                Some(&rx.view(&rl).unwrap()),
                &rdy.view(&rol).unwrap(),
                &mut dx.view_mut(&rl).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        (y.data().to_vec(), dx.data().to_vec())
    };
    let (rmax_y, rmax_dx) = run_red(ReduceOp::Max);
    let (rmin_y, rmin_dx) = run_red(ReduceOp::Min);
    let (rlse_y, rlse_dx) = run_red(ReduceOp::LogSumExp);

    // 4. Broadcast Bmm: [2, 3, 5, 7] @ [1, 3, 7, 4] -> [2, 3, 5, 4]
    let ba_s = Shape::new([2, 3, 5, 7]).unwrap();
    let bb_s = Shape::new([1, 3, 7, 4]).unwrap();
    let bmm_spec = BmmSpec::new(&ba_s, &bb_s).unwrap();
    let bal = Layout::contiguous(ba_s.dims()).unwrap();
    let bbl = Layout::contiguous(bb_s.dims()).unwrap();
    let bol = Layout::contiguous(bmm_spec.output_shape().dims()).unwrap();
    let ba = Tensor::from_vec(ba_s.dims(), pattern(ba_s.len(), 7)).unwrap();
    let bb = Tensor::from_vec(bb_s.dims(), pattern(bb_s.len(), 8)).unwrap();
    let bdy = Tensor::from_vec(
        bmm_spec.output_shape().dims(),
        pattern(bmm_spec.output_shape().len(), 9),
    )
    .unwrap();
    let mut by = Tensor::zeros(bmm_spec.output_shape().dims()).unwrap();
    let mut bda = Tensor::zeros(ba_s.dims()).unwrap();
    let mut bdb = Tensor::zeros(bb_s.dims()).unwrap();
    cpu::BatchedMatmulPlan::new(Cpu::Reference, &bmm_spec, &bal, &bbl, &bol)
        .unwrap()
        .run(
            &ba.view(&bal).unwrap(),
            &bb.view(&bbl).unwrap(),
            &mut by.view_mut(&bol).unwrap(),
        )
        .unwrap();
    cpu::BatchedMatmulBackward::new(
        Cpu::Reference,
        BinaryOperand::Left,
        &bmm_spec,
        &bal,
        &bbl,
        &bol,
        &bal,
    )
    .unwrap()
    .run(
        None,
        Some(&bb.view(&bbl).unwrap()),
        &bdy.view(&bol).unwrap(),
        &mut bda.view_mut(&bal).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cpu::BatchedMatmulBackward::new(
        Cpu::Reference,
        BinaryOperand::Right,
        &bmm_spec,
        &bal,
        &bbl,
        &bol,
        &bbl,
    )
    .unwrap()
    .run(
        Some(&ba.view(&bal).unwrap()),
        None,
        &bdy.view(&bol).unwrap(),
        &mut bdb.view_mut(&bbl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    // 5. SDPA (GQA + causal: B=2, H=4, H_kv=2, T=12, D=16)
    let sdpa_spec = SdpaSpec::new(2, 4, 2, 12, 12, 16, None, true).unwrap();
    let ql = Layout::contiguous(sdpa_spec.query_shape().dims()).unwrap();
    let kvl = Layout::contiguous(sdpa_spec.kv_shape().dims()).unwrap();
    let q = Tensor::from_vec(
        sdpa_spec.query_shape().dims(),
        pattern(sdpa_spec.query_shape().len(), 10),
    )
    .unwrap();
    let k = Tensor::from_vec(
        sdpa_spec.kv_shape().dims(),
        pattern(sdpa_spec.kv_shape().len(), 11),
    )
    .unwrap();
    let v = Tensor::from_vec(
        sdpa_spec.kv_shape().dims(),
        pattern(sdpa_spec.kv_shape().len(), 12),
    )
    .unwrap();
    let sdy = Tensor::from_vec(
        sdpa_spec.query_shape().dims(),
        pattern(sdpa_spec.query_shape().len(), 13),
    )
    .unwrap();
    let mut so = Tensor::zeros(sdpa_spec.query_shape().dims()).unwrap();
    let mut sdq = Tensor::zeros(sdpa_spec.query_shape().dims()).unwrap();
    let mut sdk = Tensor::zeros(sdpa_spec.kv_shape().dims()).unwrap();
    let mut sdv = Tensor::zeros(sdpa_spec.kv_shape().dims()).unwrap();
    cpu::SdpaPlan::new(sdpa_spec, &ql, &kvl, &kvl, &ql)
        .unwrap()
        .run(
            &q.view(&ql).unwrap(),
            &k.view(&kvl).unwrap(),
            &v.view(&kvl).unwrap(),
            &mut so.view_mut(&ql).unwrap(),
        )
        .unwrap();
    cpu::SdpaBackward::new(SdpaOperand::Query, sdpa_spec, &ql, &kvl, &kvl, &ql, &ql)
        .unwrap()
        .run(
            Some(&q.view(&ql).unwrap()),
            Some(&k.view(&kvl).unwrap()),
            Some(&v.view(&kvl).unwrap()),
            &sdy.view(&ql).unwrap(),
            &mut sdq.view_mut(&ql).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    cpu::SdpaBackward::new(SdpaOperand::Key, sdpa_spec, &ql, &kvl, &kvl, &ql, &kvl)
        .unwrap()
        .run(
            Some(&q.view(&ql).unwrap()),
            Some(&k.view(&kvl).unwrap()),
            Some(&v.view(&kvl).unwrap()),
            &sdy.view(&ql).unwrap(),
            &mut sdk.view_mut(&kvl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    cpu::SdpaBackward::new(SdpaOperand::Value, sdpa_spec, &ql, &kvl, &kvl, &ql, &kvl)
        .unwrap()
        .run(
            Some(&q.view(&ql).unwrap()),
            Some(&k.view(&kvl).unwrap()),
            None,
            &sdy.view(&ql).unwrap(),
            &mut sdv.view_mut(&kvl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    // 6. ConvTranspose1d & ConvTranspose2d
    let ct1 = ConvTranspose1dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 6,
        input_len: 7,
        kernel: 4,
        stride: 2,
        padding: 1,
        output_padding: 1,
        dilation: 1,
        groups: 2,
    };
    let ct1_is = [2, 4, 7];
    let ct1_ws = ct1.weight_shape();
    let ct1_os = ct1.output_shape().unwrap();
    let ct1_il = Layout::contiguous(ct1_is).unwrap();
    let ct1_wl = Layout::contiguous(ct1_ws).unwrap();
    let ct1_ol = Layout::contiguous(ct1_os).unwrap();
    let ct1_x = Tensor::from_vec(ct1_is, pattern(ct1_is.iter().product(), 14)).unwrap();
    let ct1_w = Tensor::from_vec(ct1_ws, pattern(ct1_ws.iter().product(), 15)).unwrap();
    let ct1_dy = Tensor::from_vec(ct1_os, pattern(ct1_os.iter().product(), 16)).unwrap();
    let mut ct1_y = Tensor::zeros(ct1_os).unwrap();
    let mut ct1_dx = Tensor::zeros(ct1_is).unwrap();
    let mut ct1_dw = Tensor::zeros(ct1_ws).unwrap();
    cpu::ConvTranspose1dPlan::new(Cpu::Reference, ct1, &ct1_il, &ct1_wl, &ct1_ol)
        .unwrap()
        .run(
            &ct1_x.view(&ct1_il).unwrap(),
            &ct1_w.view(&ct1_wl).unwrap(),
            &mut ct1_y.view_mut(&ct1_ol).unwrap(),
        )
        .unwrap();
    cpu::ConvTranspose1dBackward::new(
        Cpu::Reference,
        BinaryOperand::Left,
        ct1,
        &ct1_il,
        &ct1_wl,
        &ct1_ol,
        &ct1_il,
    )
    .unwrap()
    .run(
        None,
        Some(&ct1_w.view(&ct1_wl).unwrap()),
        &ct1_dy.view(&ct1_ol).unwrap(),
        &mut ct1_dx.view_mut(&ct1_il).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cpu::ConvTranspose1dBackward::new(
        Cpu::Reference,
        BinaryOperand::Right,
        ct1,
        &ct1_il,
        &ct1_wl,
        &ct1_ol,
        &ct1_wl,
    )
    .unwrap()
    .run(
        Some(&ct1_x.view(&ct1_il).unwrap()),
        None,
        &ct1_dy.view(&ct1_ol).unwrap(),
        &mut ct1_dw.view_mut(&ct1_wl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    let ct2 = Conv2dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 4,
        input: [4, 5],
        kernel: [3, 2],
        stride: [2, 2],
        padding: [1, 0],
        dilation: [1, 2],
        groups: 2,
        transposed: true,
        output_padding: [1, 1],
    };
    let ct2_is = ct2.input_shape();
    let ct2_ws = ct2.weight_shape();
    let ct2_os = ct2.output_shape().unwrap();
    let ct2_il = Layout::contiguous(ct2_is).unwrap();
    let ct2_wl = Layout::contiguous(ct2_ws).unwrap();
    let ct2_ol = Layout::contiguous(ct2_os).unwrap();
    let ct2_x = Tensor::from_vec(ct2_is, pattern(ct2_is.iter().product(), 17)).unwrap();
    let ct2_w = Tensor::from_vec(ct2_ws, pattern(ct2_ws.iter().product(), 18)).unwrap();
    let ct2_dy = Tensor::from_vec(ct2_os, pattern(ct2_os.iter().product(), 19)).unwrap();
    let mut ct2_y = Tensor::zeros(ct2_os).unwrap();
    let mut ct2_dx = Tensor::zeros(ct2_is).unwrap();
    let mut ct2_dw = Tensor::zeros(ct2_ws).unwrap();
    cpu::ConvTranspose2dPlan::new(Cpu::Reference, ct2, &ct2_il, &ct2_wl, &ct2_ol)
        .unwrap()
        .run(
            &ct2_x.view(&ct2_il).unwrap(),
            &ct2_w.view(&ct2_wl).unwrap(),
            &mut ct2_y.view_mut(&ct2_ol).unwrap(),
        )
        .unwrap();
    cpu::ConvTranspose2dBackward::new(
        Cpu::Reference,
        BinaryOperand::Left,
        ct2,
        &ct2_il,
        &ct2_wl,
        &ct2_ol,
        &ct2_il,
    )
    .unwrap()
    .run(
        None,
        Some(&ct2_w.view(&ct2_wl).unwrap()),
        &ct2_dy.view(&ct2_ol).unwrap(),
        &mut ct2_dx.view_mut(&ct2_il).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cpu::ConvTranspose2dBackward::new(
        Cpu::Reference,
        BinaryOperand::Right,
        ct2,
        &ct2_il,
        &ct2_wl,
        &ct2_ol,
        &ct2_wl,
    )
    .unwrap()
    .run(
        Some(&ct2_x.view(&ct2_il).unwrap()),
        None,
        &ct2_dy.view(&ct2_ol).unwrap(),
        &mut ct2_dw.view_mut(&ct2_wl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    // 7. Pad (Reflect + Replicate) & Cat
    let ps = Shape::new([2, 3, 5]).unwrap();
    let pl = Layout::contiguous(ps.dims()).unwrap();
    let px = Tensor::from_vec(ps.dims(), pattern(ps.len(), 20)).unwrap();
    let pref_spec = PadSpec::new(&ps, &[(0, 0), (1, 2), (2, 1)], PadMode::Reflect).unwrap();
    let prepl_spec = PadSpec::new(&ps, &[(0, 0), (1, 2), (2, 1)], PadMode::Replicate).unwrap();
    let pol = Layout::contiguous(pref_spec.output_shape().dims()).unwrap();
    let pdy = Tensor::from_vec(
        pref_spec.output_shape().dims(),
        pattern(pref_spec.output_shape().len(), 21),
    )
    .unwrap();
    let mut pref_y = Tensor::zeros(pref_spec.output_shape().dims()).unwrap();
    let mut pref_dx = Tensor::zeros(ps.dims()).unwrap();
    cpu::PadPlan::new(&pref_spec, &pl, &pol)
        .unwrap()
        .run(&px.view(&pl).unwrap(), &mut pref_y.view_mut(&pol).unwrap())
        .unwrap();
    cpu::PadBackward::new(&pref_spec, &pol, &pl)
        .unwrap()
        .run(
            &pdy.view(&pol).unwrap(),
            &mut pref_dx.view_mut(&pl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    let mut prepl_y = Tensor::zeros(prepl_spec.output_shape().dims()).unwrap();
    let mut prepl_dx = Tensor::zeros(ps.dims()).unwrap();
    cpu::PadPlan::new(&prepl_spec, &pl, &pol)
        .unwrap()
        .run(&px.view(&pl).unwrap(), &mut prepl_y.view_mut(&pol).unwrap())
        .unwrap();
    cpu::PadBackward::new(&prepl_spec, &pol, &pl)
        .unwrap()
        .run(
            &pdy.view(&pol).unwrap(),
            &mut prepl_dx.view_mut(&pl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    let cs_a = Shape::new([2, 3, 4]).unwrap();
    let cs_b = Shape::new([2, 2, 4]).unwrap();
    let cspec = CatSpec::new(&cs_a, &cs_b, 1).unwrap();
    let cal = Layout::contiguous(cs_a.dims()).unwrap();
    let cbl = Layout::contiguous(cs_b.dims()).unwrap();
    let col = Layout::contiguous(cspec.output_shape().dims()).unwrap();
    let ca = Tensor::from_vec(cs_a.dims(), pattern(cs_a.len(), 22)).unwrap();
    let cb = Tensor::from_vec(cs_b.dims(), pattern(cs_b.len(), 23)).unwrap();
    let cdy = Tensor::from_vec(
        cspec.output_shape().dims(),
        pattern(cspec.output_shape().len(), 24),
    )
    .unwrap();
    let mut cy = Tensor::zeros(cspec.output_shape().dims()).unwrap();
    let mut cda = Tensor::zeros(cs_a.dims()).unwrap();
    let mut cdb = Tensor::zeros(cs_b.dims()).unwrap();
    cpu::CatPlan::new(&cspec, &cal, &cbl, &col)
        .unwrap()
        .run(
            &ca.view(&cal).unwrap(),
            &cb.view(&cbl).unwrap(),
            &mut cy.view_mut(&col).unwrap(),
        )
        .unwrap();
    cpu::CatBackward::new(BinaryOperand::Left, &cspec, &col, &cal)
        .unwrap()
        .run(
            &cdy.view(&col).unwrap(),
            &mut cda.view_mut(&cal).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    cpu::CatBackward::new(BinaryOperand::Right, &cspec, &col, &cbl)
        .unwrap()
        .run(
            &cdy.view(&col).unwrap(),
            &mut cdb.view_mut(&cbl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    // 8. 5-step AdamW + global L2 grad clipping + dynamic LR on Conv1d -> RmsNorm -> SiLU -> MSE
    let mut b = Builder::new().unwrap();
    let ax = b.leaf([2, 2, 12], false).unwrap();
    let aw = b.leaf([4, 2, 3], true).unwrap();
    let ag = b.leaf([1, 4, 1], true).unwrap();
    let ab = b.leaf([1, 4, 1], true).unwrap();
    let at = b.leaf([2, 4, 12], false).unwrap();
    let h = b.conv1d(ax, aw, 1, 1, 1, 1).unwrap();
    let hn = b.rms_norm(h, 1, 1e-5).unwrap();
    let ha = b.unary(UnaryOp::Silu, hn).unwrap();
    let hs = b.binary(BinaryOp::Mul, ha, ag).unwrap();
    let ho = b.binary(BinaryOp::Add, hs, ab).unwrap();
    let hd = b.binary(BinaryOp::Sub, ho, at).unwrap();
    let hq = b.unary(UnaryOp::Square, hd).unwrap();
    let aloss = b.reduce(ReduceOp::Mean, hq, &[0, 1, 2], false).unwrap();
    let aprog = b.finish(&[aloss]).unwrap();

    let sax = Shared::new(Tensor::from_vec([2, 2, 12], pattern(48, 30)).unwrap()).unwrap();
    let saw = Shared::new(Tensor::from_vec([4, 2, 3], pattern(24, 31)).unwrap()).unwrap();
    let sag = Shared::new(Tensor::from_vec([1, 4, 1], pattern(4, 32)).unwrap()).unwrap();
    let sab = Shared::new(Tensor::from_vec([1, 4, 1], pattern(4, 33)).unwrap()).unwrap();
    let sat = Shared::new(Tensor::from_vec([2, 4, 12], pattern(96, 34)).unwrap()).unwrap();
    let mut atape = cpu::Tape::new(
        Cpu::Reference,
        &aprog,
        &[
            Binding::new(ax, sax),
            Binding::new(aw, saw.clone()),
            Binding::new(ag, sag.clone()),
            Binding::new(ab, sab.clone()),
            Binding::new(at, sat),
        ],
    )
    .unwrap();
    let mut aopt = cpu::AdamW::for_program(
        &aprog,
        &[aw, ag, ab],
        AdamWConfig {
            lr: 0.02,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.02,
            max_grad_norm: Some(0.25),
        },
    )
    .unwrap();
    for step in 0..5usize {
        aopt.set_lr(0.02 * (1.0 - 0.1 * step as f32)).unwrap();
        atape.step_scalar_adamw(&mut aopt).unwrap();
    }

    json!({
        "device": "cpu",
        "layernorm": { "y": ln_y.data(), "dx": ln_dx.data() },
        "rmsnorm": { "y": rms_y.data(), "dx": rms_dx.data() },
        "softmax": { "y": sm_y.data(), "dx": sm_dx.data() },
        "log_softmax": { "y": lsm_y.data(), "dx": lsm_dx.data() },
        "reduce_max": { "y": rmax_y, "dx": rmax_dx },
        "reduce_min": { "y": rmin_y, "dx": rmin_dx },
        "reduce_lse": { "y": rlse_y, "dx": rlse_dx },
        "bmm": { "y": by.data(), "da": bda.data(), "db": bdb.data() },
        "sdpa_gqa_causal": { "o": so.data(), "dq": sdq.data(), "dk": sdk.data(), "dv": sdv.data() },
        "conv_transpose1d": { "y": ct1_y.data(), "dx": ct1_dx.data(), "dw": ct1_dw.data() },
        "conv_transpose2d": { "y": ct2_y.data(), "dx": ct2_dx.data(), "dw": ct2_dw.data() },
        "pad_reflect": { "y": pref_y.data(), "dx": pref_dx.data() },
        "pad_replicate": { "y": prepl_y.data(), "dx": prepl_dx.data() },
        "cat": { "y": cy.data(), "da": cda.data(), "db": cdb.data() },
        "adamw_5step": {
            "w": saw.read().unwrap().data().to_vec(),
            "gamma": sag.read().unwrap().data().to_vec(),
            "bias": sab.read().unwrap().data().to_vec(),
        },
    })
}

#[cfg(feature = "cuda")]
fn run_verify_cuda() -> Value {
    let cuda = Cuda::new(0).unwrap();

    // 1. LayerNorm + RmsNorm
    let norm_shape = Shape::new([2, 4, 16]).unwrap();
    let nl = Layout::contiguous(norm_shape.dims()).unwrap();
    let nx = cuda
        .upload(&Tensor::from_vec(norm_shape.dims(), pattern(128, 1)).unwrap())
        .unwrap();
    let ndy = cuda
        .upload(&Tensor::from_vec(norm_shape.dims(), pattern(128, 2)).unwrap())
        .unwrap();
    let ln_spec = NormSpec::new(NormKind::LayerNorm, &norm_shape, 1, 1e-5).unwrap();
    let rms_spec = NormSpec::new(NormKind::RmsNorm, &norm_shape, 1, 1e-5).unwrap();
    let mut ln_y = cuda.zeros(norm_shape.dims()).unwrap();
    let mut ln_dx = cuda.zeros(norm_shape.dims()).unwrap();
    cuda::NormPlan::new(&cuda, &ln_spec, &nl, &nl)
        .unwrap()
        .run(&nx.view(&nl).unwrap(), &mut ln_y.view_mut(&nl).unwrap())
        .unwrap();
    cuda::NormBackward::new(&cuda, &ln_spec, &nl, &nl, &nl)
        .unwrap()
        .run(
            &nx.view(&nl).unwrap(),
            &ndy.view(&nl).unwrap(),
            &mut ln_dx.view_mut(&nl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    let mut rms_y = cuda.zeros(norm_shape.dims()).unwrap();
    let mut rms_dx = cuda.zeros(norm_shape.dims()).unwrap();
    cuda::NormPlan::new(&cuda, &rms_spec, &nl, &nl)
        .unwrap()
        .run(&nx.view(&nl).unwrap(), &mut rms_y.view_mut(&nl).unwrap())
        .unwrap();
    cuda::NormBackward::new(&cuda, &rms_spec, &nl, &nl, &nl)
        .unwrap()
        .run(
            &nx.view(&nl).unwrap(),
            &ndy.view(&nl).unwrap(),
            &mut rms_dx.view_mut(&nl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    // 2. Softmax & LogSoftmax
    let sm_shape = Shape::new([2, 5, 7]).unwrap();
    let sml = Layout::contiguous(sm_shape.dims()).unwrap();
    let smx = cuda
        .upload(&Tensor::from_vec(sm_shape.dims(), pattern(70, 3)).unwrap())
        .unwrap();
    let smdy = cuda
        .upload(&Tensor::from_vec(sm_shape.dims(), pattern(70, 4)).unwrap())
        .unwrap();
    let sm_spec = SoftmaxSpec::new(SoftmaxKind::Softmax, &sm_shape, 1).unwrap();
    let lsm_spec = SoftmaxSpec::new(SoftmaxKind::LogSoftmax, &sm_shape, 1).unwrap();
    let mut sm_y = cuda.zeros(sm_shape.dims()).unwrap();
    let mut sm_dx = cuda.zeros(sm_shape.dims()).unwrap();
    cuda::SoftmaxPlan::new(&cuda, &sm_spec, &sml, &sml)
        .unwrap()
        .run(&smx.view(&sml).unwrap(), &mut sm_y.view_mut(&sml).unwrap())
        .unwrap();
    cuda::SoftmaxBackward::new(&cuda, &sm_spec, &sml, &sml, &sml)
        .unwrap()
        .run(
            &smx.view(&sml).unwrap(),
            &smdy.view(&sml).unwrap(),
            &mut sm_dx.view_mut(&sml).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    let mut lsm_y = cuda.zeros(sm_shape.dims()).unwrap();
    let mut lsm_dx = cuda.zeros(sm_shape.dims()).unwrap();
    cuda::SoftmaxPlan::new(&cuda, &lsm_spec, &sml, &sml)
        .unwrap()
        .run(&smx.view(&sml).unwrap(), &mut lsm_y.view_mut(&sml).unwrap())
        .unwrap();
    cuda::SoftmaxBackward::new(&cuda, &lsm_spec, &sml, &sml, &sml)
        .unwrap()
        .run(
            &smx.view(&sml).unwrap(),
            &smdy.view(&sml).unwrap(),
            &mut lsm_dx.view_mut(&sml).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    // 3. Nonlinear reductions
    let r_shape = Shape::new([2, 4, 6]).unwrap();
    let rl = Layout::contiguous(r_shape.dims()).unwrap();
    let rol = Layout::contiguous([2, 4]).unwrap();
    let rx = cuda
        .upload(&Tensor::from_vec(r_shape.dims(), pattern(48, 5)).unwrap())
        .unwrap();
    let rdy = cuda
        .upload(&Tensor::from_vec([2, 4], pattern(8, 6)).unwrap())
        .unwrap();
    let run_red = |op: ReduceOp| -> (Vec<f32>, Vec<f32>) {
        let spec = ReduceSpec::new(op, &r_shape, &[2], false).unwrap();
        let mut y = cuda.zeros([2, 4]).unwrap();
        let mut dx = cuda.zeros(r_shape.dims()).unwrap();
        cuda::ReducePlan::new(&cuda, &spec, &rl, &rol)
            .unwrap()
            .run(&rx.view(&rl).unwrap(), &mut y.view_mut(&rol).unwrap())
            .unwrap();
        cuda::ReduceBackward::with_input(&cuda, &spec, &rl, &rol, &rl)
            .unwrap()
            .run_with_input(
                Some(&rx.view(&rl).unwrap()),
                &rdy.view(&rol).unwrap(),
                &mut dx.view_mut(&rl).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        (
            cuda.download(&y).unwrap().data().to_vec(),
            cuda.download(&dx).unwrap().data().to_vec(),
        )
    };
    let (rmax_y, rmax_dx) = run_red(ReduceOp::Max);
    let (rmin_y, rmin_dx) = run_red(ReduceOp::Min);
    let (rlse_y, rlse_dx) = run_red(ReduceOp::LogSumExp);

    // 4. Broadcast Bmm
    let ba_s = Shape::new([2, 3, 5, 7]).unwrap();
    let bb_s = Shape::new([1, 3, 7, 4]).unwrap();
    let bmm_spec = BmmSpec::new(&ba_s, &bb_s).unwrap();
    let bal = Layout::contiguous(ba_s.dims()).unwrap();
    let bbl = Layout::contiguous(bb_s.dims()).unwrap();
    let bol = Layout::contiguous(bmm_spec.output_shape().dims()).unwrap();
    let ba = cuda
        .upload(&Tensor::from_vec(ba_s.dims(), pattern(ba_s.len(), 7)).unwrap())
        .unwrap();
    let bb = cuda
        .upload(&Tensor::from_vec(bb_s.dims(), pattern(bb_s.len(), 8)).unwrap())
        .unwrap();
    let bdy = cuda
        .upload(
            &Tensor::from_vec(
                bmm_spec.output_shape().dims(),
                pattern(bmm_spec.output_shape().len(), 9),
            )
            .unwrap(),
        )
        .unwrap();
    let mut by = cuda.zeros(bmm_spec.output_shape().dims()).unwrap();
    let mut bda = cuda.zeros(ba_s.dims()).unwrap();
    let mut bdb = cuda.zeros(bb_s.dims()).unwrap();
    cuda::BatchedMatmulPlan::new(&cuda, &bmm_spec, &bal, &bbl, &bol)
        .unwrap()
        .run(
            &ba.view(&bal).unwrap(),
            &bb.view(&bbl).unwrap(),
            &mut by.view_mut(&bol).unwrap(),
        )
        .unwrap();
    cuda::BatchedMatmulBackward::new(
        &cuda,
        BinaryOperand::Left,
        &bmm_spec,
        &bal,
        &bbl,
        &bol,
        &bal,
    )
    .unwrap()
    .run(
        None,
        Some(&bb.view(&bbl).unwrap()),
        &bdy.view(&bol).unwrap(),
        &mut bda.view_mut(&bal).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cuda::BatchedMatmulBackward::new(
        &cuda,
        BinaryOperand::Right,
        &bmm_spec,
        &bal,
        &bbl,
        &bol,
        &bbl,
    )
    .unwrap()
    .run(
        Some(&ba.view(&bal).unwrap()),
        None,
        &bdy.view(&bol).unwrap(),
        &mut bdb.view_mut(&bbl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    // 5. SDPA GQA causal
    let sdpa_spec = SdpaSpec::new(2, 4, 2, 12, 12, 16, None, true).unwrap();
    let ql = Layout::contiguous(sdpa_spec.query_shape().dims()).unwrap();
    let kvl = Layout::contiguous(sdpa_spec.kv_shape().dims()).unwrap();
    let q = cuda
        .upload(
            &Tensor::from_vec(
                sdpa_spec.query_shape().dims(),
                pattern(sdpa_spec.query_shape().len(), 10),
            )
            .unwrap(),
        )
        .unwrap();
    let k = cuda
        .upload(
            &Tensor::from_vec(
                sdpa_spec.kv_shape().dims(),
                pattern(sdpa_spec.kv_shape().len(), 11),
            )
            .unwrap(),
        )
        .unwrap();
    let v = cuda
        .upload(
            &Tensor::from_vec(
                sdpa_spec.kv_shape().dims(),
                pattern(sdpa_spec.kv_shape().len(), 12),
            )
            .unwrap(),
        )
        .unwrap();
    let sdy = cuda
        .upload(
            &Tensor::from_vec(
                sdpa_spec.query_shape().dims(),
                pattern(sdpa_spec.query_shape().len(), 13),
            )
            .unwrap(),
        )
        .unwrap();
    let mut so = cuda.zeros(sdpa_spec.query_shape().dims()).unwrap();
    let mut sdq = cuda.zeros(sdpa_spec.query_shape().dims()).unwrap();
    let mut sdk = cuda.zeros(sdpa_spec.kv_shape().dims()).unwrap();
    let mut sdv = cuda.zeros(sdpa_spec.kv_shape().dims()).unwrap();
    cuda::SdpaPlan::new(&cuda, sdpa_spec, &ql, &kvl, &kvl, &ql)
        .unwrap()
        .run(
            &q.view(&ql).unwrap(),
            &k.view(&kvl).unwrap(),
            &v.view(&kvl).unwrap(),
            &mut so.view_mut(&ql).unwrap(),
        )
        .unwrap();
    cuda::SdpaBackward::new(
        &cuda,
        SdpaOperand::Query,
        sdpa_spec,
        &ql,
        &kvl,
        &kvl,
        &ql,
        &ql,
    )
    .unwrap()
    .run(
        Some(&q.view(&ql).unwrap()),
        Some(&k.view(&kvl).unwrap()),
        Some(&v.view(&kvl).unwrap()),
        &sdy.view(&ql).unwrap(),
        &mut sdq.view_mut(&ql).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cuda::SdpaBackward::new(
        &cuda,
        SdpaOperand::Key,
        sdpa_spec,
        &ql,
        &kvl,
        &kvl,
        &ql,
        &kvl,
    )
    .unwrap()
    .run(
        Some(&q.view(&ql).unwrap()),
        Some(&k.view(&kvl).unwrap()),
        Some(&v.view(&kvl).unwrap()),
        &sdy.view(&ql).unwrap(),
        &mut sdk.view_mut(&kvl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cuda::SdpaBackward::new(
        &cuda,
        SdpaOperand::Value,
        sdpa_spec,
        &ql,
        &kvl,
        &kvl,
        &ql,
        &kvl,
    )
    .unwrap()
    .run(
        Some(&q.view(&ql).unwrap()),
        Some(&k.view(&kvl).unwrap()),
        None,
        &sdy.view(&ql).unwrap(),
        &mut sdv.view_mut(&kvl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    // 6. ConvTranspose1d & 2d
    let ct1 = ConvTranspose1dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 6,
        input_len: 7,
        kernel: 4,
        stride: 2,
        padding: 1,
        output_padding: 1,
        dilation: 1,
        groups: 2,
    };
    let ct1_is = [2, 4, 7];
    let ct1_ws = ct1.weight_shape();
    let ct1_os = ct1.output_shape().unwrap();
    let ct1_il = Layout::contiguous(ct1_is).unwrap();
    let ct1_wl = Layout::contiguous(ct1_ws).unwrap();
    let ct1_ol = Layout::contiguous(ct1_os).unwrap();
    let ct1_x = cuda
        .upload(&Tensor::from_vec(ct1_is, pattern(ct1_is.iter().product(), 14)).unwrap())
        .unwrap();
    let ct1_w = cuda
        .upload(&Tensor::from_vec(ct1_ws, pattern(ct1_ws.iter().product(), 15)).unwrap())
        .unwrap();
    let ct1_dy = cuda
        .upload(&Tensor::from_vec(ct1_os, pattern(ct1_os.iter().product(), 16)).unwrap())
        .unwrap();
    let mut ct1_y = cuda.zeros(ct1_os).unwrap();
    let mut ct1_dx = cuda.zeros(ct1_is).unwrap();
    let mut ct1_dw = cuda.zeros(ct1_ws).unwrap();
    cuda::ConvTranspose1dPlan::new(&cuda, ct1, &ct1_il, &ct1_wl, &ct1_ol)
        .unwrap()
        .run(
            &ct1_x.view(&ct1_il).unwrap(),
            &ct1_w.view(&ct1_wl).unwrap(),
            &mut ct1_y.view_mut(&ct1_ol).unwrap(),
        )
        .unwrap();
    cuda::ConvTranspose1dBackward::new(
        &cuda,
        BinaryOperand::Left,
        ct1,
        &ct1_il,
        &ct1_wl,
        &ct1_ol,
        &ct1_il,
    )
    .unwrap()
    .run(
        None,
        Some(&ct1_w.view(&ct1_wl).unwrap()),
        &ct1_dy.view(&ct1_ol).unwrap(),
        &mut ct1_dx.view_mut(&ct1_il).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cuda::ConvTranspose1dBackward::new(
        &cuda,
        BinaryOperand::Right,
        ct1,
        &ct1_il,
        &ct1_wl,
        &ct1_ol,
        &ct1_wl,
    )
    .unwrap()
    .run(
        Some(&ct1_x.view(&ct1_il).unwrap()),
        None,
        &ct1_dy.view(&ct1_ol).unwrap(),
        &mut ct1_dw.view_mut(&ct1_wl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    let ct2 = Conv2dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 4,
        input: [4, 5],
        kernel: [3, 2],
        stride: [2, 2],
        padding: [1, 0],
        dilation: [1, 2],
        groups: 2,
        transposed: true,
        output_padding: [1, 1],
    };
    let ct2_is = ct2.input_shape();
    let ct2_ws = ct2.weight_shape();
    let ct2_os = ct2.output_shape().unwrap();
    let ct2_il = Layout::contiguous(ct2_is).unwrap();
    let ct2_wl = Layout::contiguous(ct2_ws).unwrap();
    let ct2_ol = Layout::contiguous(ct2_os).unwrap();
    let ct2_x = cuda
        .upload(&Tensor::from_vec(ct2_is, pattern(ct2_is.iter().product(), 17)).unwrap())
        .unwrap();
    let ct2_w = cuda
        .upload(&Tensor::from_vec(ct2_ws, pattern(ct2_ws.iter().product(), 18)).unwrap())
        .unwrap();
    let ct2_dy = cuda
        .upload(&Tensor::from_vec(ct2_os, pattern(ct2_os.iter().product(), 19)).unwrap())
        .unwrap();
    let mut ct2_y = cuda.zeros(ct2_os).unwrap();
    let mut ct2_dx = cuda.zeros(ct2_is).unwrap();
    let mut ct2_dw = cuda.zeros(ct2_ws).unwrap();
    cuda::ConvTranspose2dPlan::new(&cuda, ct2, &ct2_il, &ct2_wl, &ct2_ol)
        .unwrap()
        .run(
            &ct2_x.view(&ct2_il).unwrap(),
            &ct2_w.view(&ct2_wl).unwrap(),
            &mut ct2_y.view_mut(&ct2_ol).unwrap(),
        )
        .unwrap();
    cuda::ConvTranspose2dBackward::new(
        &cuda,
        BinaryOperand::Left,
        ct2,
        &ct2_il,
        &ct2_wl,
        &ct2_ol,
        &ct2_il,
    )
    .unwrap()
    .run(
        None,
        Some(&ct2_w.view(&ct2_wl).unwrap()),
        &ct2_dy.view(&ct2_ol).unwrap(),
        &mut ct2_dx.view_mut(&ct2_il).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cuda::ConvTranspose2dBackward::new(
        &cuda,
        BinaryOperand::Right,
        ct2,
        &ct2_il,
        &ct2_wl,
        &ct2_ol,
        &ct2_wl,
    )
    .unwrap()
    .run(
        Some(&ct2_x.view(&ct2_il).unwrap()),
        None,
        &ct2_dy.view(&ct2_ol).unwrap(),
        &mut ct2_dw.view_mut(&ct2_wl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    // 7. Pad & Cat
    let ps = Shape::new([2, 3, 5]).unwrap();
    let pl = Layout::contiguous(ps.dims()).unwrap();
    let px = cuda
        .upload(&Tensor::from_vec(ps.dims(), pattern(ps.len(), 20)).unwrap())
        .unwrap();
    let pref_spec = PadSpec::new(&ps, &[(0, 0), (1, 2), (2, 1)], PadMode::Reflect).unwrap();
    let prepl_spec = PadSpec::new(&ps, &[(0, 0), (1, 2), (2, 1)], PadMode::Replicate).unwrap();
    let pol = Layout::contiguous(pref_spec.output_shape().dims()).unwrap();
    let pdy = cuda
        .upload(
            &Tensor::from_vec(
                pref_spec.output_shape().dims(),
                pattern(pref_spec.output_shape().len(), 21),
            )
            .unwrap(),
        )
        .unwrap();
    let mut pref_y = cuda.zeros(pref_spec.output_shape().dims()).unwrap();
    let mut pref_dx = cuda.zeros(ps.dims()).unwrap();
    cuda::PadPlan::new(&cuda, &pref_spec, &pl, &pol)
        .unwrap()
        .run(&px.view(&pl).unwrap(), &mut pref_y.view_mut(&pol).unwrap())
        .unwrap();
    cuda::PadBackward::new(&cuda, &pref_spec, &pol, &pl)
        .unwrap()
        .run(
            &pdy.view(&pol).unwrap(),
            &mut pref_dx.view_mut(&pl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    let mut prepl_y = cuda.zeros(prepl_spec.output_shape().dims()).unwrap();
    let mut prepl_dx = cuda.zeros(ps.dims()).unwrap();
    cuda::PadPlan::new(&cuda, &prepl_spec, &pl, &pol)
        .unwrap()
        .run(&px.view(&pl).unwrap(), &mut prepl_y.view_mut(&pol).unwrap())
        .unwrap();
    cuda::PadBackward::new(&cuda, &prepl_spec, &pol, &pl)
        .unwrap()
        .run(
            &pdy.view(&pol).unwrap(),
            &mut prepl_dx.view_mut(&pl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    let cs_a = Shape::new([2, 3, 4]).unwrap();
    let cs_b = Shape::new([2, 2, 4]).unwrap();
    let cspec = CatSpec::new(&cs_a, &cs_b, 1).unwrap();
    let cal = Layout::contiguous(cs_a.dims()).unwrap();
    let cbl = Layout::contiguous(cs_b.dims()).unwrap();
    let col = Layout::contiguous(cspec.output_shape().dims()).unwrap();
    let ca = cuda
        .upload(&Tensor::from_vec(cs_a.dims(), pattern(cs_a.len(), 22)).unwrap())
        .unwrap();
    let cb = cuda
        .upload(&Tensor::from_vec(cs_b.dims(), pattern(cs_b.len(), 23)).unwrap())
        .unwrap();
    let cdy = cuda
        .upload(
            &Tensor::from_vec(
                cspec.output_shape().dims(),
                pattern(cspec.output_shape().len(), 24),
            )
            .unwrap(),
        )
        .unwrap();
    let mut cy = cuda.zeros(cspec.output_shape().dims()).unwrap();
    let mut cda = cuda.zeros(cs_a.dims()).unwrap();
    let mut cdb = cuda.zeros(cs_b.dims()).unwrap();
    cuda::CatPlan::new(&cuda, &cspec, &cal, &cbl, &col)
        .unwrap()
        .run(
            &ca.view(&cal).unwrap(),
            &cb.view(&cbl).unwrap(),
            &mut cy.view_mut(&col).unwrap(),
        )
        .unwrap();
    cuda::CatBackward::new(&cuda, BinaryOperand::Left, &cspec, &col, &cal)
        .unwrap()
        .run(
            &cdy.view(&col).unwrap(),
            &mut cda.view_mut(&cal).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    cuda::CatBackward::new(&cuda, BinaryOperand::Right, &cspec, &col, &cbl)
        .unwrap()
        .run(
            &cdy.view(&col).unwrap(),
            &mut cdb.view_mut(&cbl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    // 8. 5-step captured CUDA Graph AdamW + grad clip + dynamic LR
    let mut b = Builder::new().unwrap();
    let ax = b.leaf([2, 2, 12], false).unwrap();
    let aw = b.leaf([4, 2, 3], true).unwrap();
    let ag = b.leaf([1, 4, 1], true).unwrap();
    let ab = b.leaf([1, 4, 1], true).unwrap();
    let at = b.leaf([2, 4, 12], false).unwrap();
    let h = b.conv1d(ax, aw, 1, 1, 1, 1).unwrap();
    let hn = b.rms_norm(h, 1, 1e-5).unwrap();
    let ha = b.unary(UnaryOp::Silu, hn).unwrap();
    let hs = b.binary(BinaryOp::Mul, ha, ag).unwrap();
    let ho = b.binary(BinaryOp::Add, hs, ab).unwrap();
    let hd = b.binary(BinaryOp::Sub, ho, at).unwrap();
    let hq = b.unary(UnaryOp::Square, hd).unwrap();
    let aloss = b.reduce(ReduceOp::Mean, hq, &[0, 1, 2], false).unwrap();
    let aprog = b.finish(&[aloss]).unwrap();

    let sax = Shared::new(
        cuda.upload(&Tensor::from_vec([2, 2, 12], pattern(48, 30)).unwrap())
            .unwrap(),
    )
    .unwrap();
    let saw = Shared::new(
        cuda.upload(&Tensor::from_vec([4, 2, 3], pattern(24, 31)).unwrap())
            .unwrap(),
    )
    .unwrap();
    let sag = Shared::new(
        cuda.upload(&Tensor::from_vec([1, 4, 1], pattern(4, 32)).unwrap())
            .unwrap(),
    )
    .unwrap();
    let sab = Shared::new(
        cuda.upload(&Tensor::from_vec([1, 4, 1], pattern(4, 33)).unwrap())
            .unwrap(),
    )
    .unwrap();
    let sat = Shared::new(
        cuda.upload(&Tensor::from_vec([2, 4, 12], pattern(96, 34)).unwrap())
            .unwrap(),
    )
    .unwrap();
    let mut atape = cuda::Tape::new(
        &cuda,
        &aprog,
        &[
            Binding::new(ax, sax),
            Binding::new(aw, saw.clone()),
            Binding::new(ag, sag.clone()),
            Binding::new(ab, sab.clone()),
            Binding::new(at, sat),
        ],
    )
    .unwrap();
    let mut aopt = cuda::AdamW::for_program(
        &cuda,
        &aprog,
        &[aw, ag, ab],
        AdamWConfig {
            lr: 0.02,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.02,
            max_grad_norm: Some(0.25),
        },
    )
    .unwrap();
    for step in 0..5usize {
        aopt.set_lr(0.02 * (1.0 - 0.1 * step as f32)).unwrap();
        atape.step_scalar_adamw_captured(&mut aopt).unwrap();
    }
    cuda.synchronize().unwrap();

    let dl = |t: &cuda::DeviceTensor| cuda.download(t).unwrap().data().to_vec();
    json!({
        "device": "cuda",
        "layernorm": { "y": dl(&ln_y), "dx": dl(&ln_dx) },
        "rmsnorm": { "y": dl(&rms_y), "dx": dl(&rms_dx) },
        "softmax": { "y": dl(&sm_y), "dx": dl(&sm_dx) },
        "log_softmax": { "y": dl(&lsm_y), "dx": dl(&lsm_dx) },
        "reduce_max": { "y": rmax_y, "dx": rmax_dx },
        "reduce_min": { "y": rmin_y, "dx": rmin_dx },
        "reduce_lse": { "y": rlse_y, "dx": rlse_dx },
        "bmm": { "y": dl(&by), "da": dl(&bda), "db": dl(&bdb) },
        "sdpa_gqa_causal": { "o": dl(&so), "dq": dl(&sdq), "dk": dl(&sdk), "dv": dl(&sdv) },
        "conv_transpose1d": { "y": dl(&ct1_y), "dx": dl(&ct1_dx), "dw": dl(&ct1_dw) },
        "conv_transpose2d": { "y": dl(&ct2_y), "dx": dl(&ct2_dx), "dw": dl(&ct2_dw) },
        "pad_reflect": { "y": dl(&pref_y), "dx": dl(&pref_dx) },
        "pad_replicate": { "y": dl(&prepl_y), "dx": dl(&prepl_dx) },
        "cat": { "y": dl(&cy), "da": dl(&cda), "db": dl(&cdb) },
        "adamw_5step": {
            "w": dl(&saw.read().unwrap()),
            "gamma": dl(&sag.read().unwrap()),
            "bias": dl(&sab.read().unwrap()),
        },
    })
}

fn bench_cpu(cpu: Cpu, filter: &str) -> Vec<Value> {
    let mut rows = Vec::new();
    let keep = |name: &str| filter.is_empty() || name.contains(filter);

    // 1. training-v1 spectrum-medium, spectrum-large, wave-multichannel, wave-long
    for &(name, family, dims, reversed, iters) in &[
        (
            "spectrum-medium",
            "spectrum",
            [2usize, 4, 257, 33],
            true,
            64usize,
        ),
        ("spectrum-large", "spectrum", [1, 2, 1025, 257], false, 32),
        ("wave-multichannel", "waveform", [2, 8, 4097, 0], true, 32),
        ("wave-long", "waveform", [1, 2, 48001, 0], false, 32),
    ] {
        if !keep(name) {
            continue;
        }
        let mut b = Builder::new().unwrap();
        let (x_t, tgt_t, g_t, b_t, pred_fn) = if family == "spectrum" {
            let [bs, c, f, t] = dims;
            let xs = [bs, c, f, t, 2];
            let ts = if reversed {
                [bs, t, c, f]
            } else {
                [bs, c, f, t]
            };
            let x_t = training_v1_recipe(&xs, 7, 3, 29, -14.0, 32.0);
            let tgt_t = training_v1_recipe(&ts, 3, 2, 17, 4.0, 64.0);
            let g_t = training_v1_recipe(&[c, f], 1, 0, 7, 16.0, 32.0);
            let b_t = Tensor::from_vec([2], vec![-0.0625, 0.125]).unwrap();
            (x_t, tgt_t, g_t, b_t, true)
        } else {
            let [bs, c, t, _] = dims;
            let xs = [bs, c, t];
            let ts = if reversed { [bs, t, c] } else { [bs, c, t] };
            let x_t = training_v1_recipe(&xs, 7, 3, 29, -14.0, 32.0);
            let tgt_t = training_v1_recipe(&ts, 5, 1, 23, -11.0, 32.0);
            let g_t = training_v1_recipe(&[c], 1, 0, 7, 16.0, 32.0);
            let b_t = training_v1_recipe(&[c], 3, 2, 7, -3.0, 64.0);
            (x_t, tgt_t, g_t, b_t, false)
        };
        let x = b.leaf(x_t.shape().dims(), false).unwrap();
        let tgt = b.leaf(tgt_t.shape().dims(), false).unwrap();
        let gain = b.leaf(g_t.shape().dims(), true).unwrap();
        let bias = b.leaf(b_t.shape().dims(), true).unwrap();
        let pred = if pred_fn {
            let [_, c, f, t] = dims;
            let g5 = b.reshape(gain, [1, c, f, 1, 1]).unwrap();
            let s = b.binary(BinaryOp::Mul, x, g5).unwrap();
            let z = b.binary(BinaryOp::Add, s, bias).unwrap();
            let sq = b.unary(UnaryOp::Square, z).unwrap();
            let raw = b.reduce(ReduceOp::Sum, sq, &[4], false).unwrap();
            if reversed {
                let rev = b.slice(raw, 3, t - 1, t, -1).unwrap();
                b.permute(rev, &[0, 3, 1, 2]).unwrap()
            } else {
                raw
            }
        } else {
            let [_, c, t, _] = dims;
            let g3 = b.reshape(gain, [1, c, 1]).unwrap();
            let b3 = b.reshape(bias, [1, c, 1]).unwrap();
            let s = b.binary(BinaryOp::Mul, x, g3).unwrap();
            let z = b.binary(BinaryOp::Add, s, b3).unwrap();
            let raw = b.unary(UnaryOp::Tanh, z).unwrap();
            if reversed {
                let rev = b.slice(raw, 2, t - 1, t, -1).unwrap();
                b.permute(rev, &[0, 2, 1]).unwrap()
            } else {
                raw
            }
        };
        let diff = b.binary(BinaryOp::Sub, pred, tgt).unwrap();
        let sq = b.unary(UnaryOp::Square, diff).unwrap();
        let axes: Vec<usize> = (0..tgt_t.shape().dims().len()).collect();
        let loss = b.reduce(ReduceOp::Mean, sq, &axes, false).unwrap();
        let prog = b.finish(&[loss]).unwrap();
        let mut tape = cpu::Tape::new(
            cpu,
            &prog,
            &[
                Binding::new(x, Shared::new(x_t).unwrap()),
                Binding::new(tgt, Shared::new(tgt_t).unwrap()),
                Binding::new(gain, Shared::new(g_t).unwrap()),
                Binding::new(bias, Shared::new(b_t).unwrap()),
            ],
        )
        .unwrap();
        let us = time_cpu(6, iters, || {
            tape.step_scalar(&[gain, bias], 0.01).unwrap();
        });
        rows.push(json!({ "name": name, "wall_us": us, "fused": tape.has_fused_plan() }));
    }

    // 2. LayerNorm & RmsNorm fwd+bwd
    for &(name, kind, shape, iters) in &[
        (
            "norm-layernorm-256",
            NormKind::LayerNorm,
            [8usize, 128, 256],
            64usize,
        ),
        ("norm-rmsnorm-512", NormKind::RmsNorm, [4, 128, 512], 64),
    ] {
        if !keep(name) {
            continue;
        }
        let s = Shape::new(shape).unwrap();
        let l = Layout::contiguous(shape).unwrap();
        let spec = NormSpec::new(kind, &s, 1, 1e-5).unwrap();
        let x = Tensor::from_vec(shape, pattern(s.len(), 1)).unwrap();
        let dy = Tensor::from_vec(shape, pattern(s.len(), 2)).unwrap();
        let mut y = Tensor::zeros(shape).unwrap();
        let mut dx = Tensor::zeros(shape).unwrap();
        let fwd = cpu::NormPlan::new(&spec, &l, &l).unwrap();
        let bwd = cpu::NormBackward::new(&spec, &l, &l, &l).unwrap();
        let xv = x.view(&l).unwrap();
        let dyv = dy.view(&l).unwrap();
        let us = time_cpu(6, iters, || {
            fwd.run(&xv, &mut y.view_mut(&l).unwrap()).unwrap();
            bwd.run(
                &xv,
                &dyv,
                &mut dx.view_mut(&l).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        rows.push(json!({ "name": name, "wall_us": us }));
    }

    // 3. Softmax fwd+bwd
    if keep("softmax-256") {
        let shape = [4usize, 8, 128, 128];
        let s = Shape::new(shape).unwrap();
        let l = Layout::contiguous(shape).unwrap();
        let spec = SoftmaxSpec::new(SoftmaxKind::Softmax, &s, 3).unwrap();
        let x = Tensor::from_vec(shape, pattern(s.len(), 3)).unwrap();
        let dy = Tensor::from_vec(shape, pattern(s.len(), 4)).unwrap();
        let mut y = Tensor::zeros(shape).unwrap();
        let mut dx = Tensor::zeros(shape).unwrap();
        let fwd = cpu::SoftmaxPlan::new(&spec, &l, &l).unwrap();
        let bwd = cpu::SoftmaxBackward::new(&spec, &l, &l, &l).unwrap();
        let xv = x.view(&l).unwrap();
        let dyv = dy.view(&l).unwrap();
        let us = time_cpu(4, 32, || {
            fwd.run(&xv, &mut y.view_mut(&l).unwrap()).unwrap();
            bwd.run(
                &xv,
                &dyv,
                &mut dx.view_mut(&l).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        rows.push(json!({ "name": "softmax-256", "wall_us": us }));
    }

    // 4. BMM collapsed linear fwd+bwd ([4, 128, 256] @ [256, 256])
    if keep("bmm-linear") {
        let sa = Shape::new([4, 128, 256]).unwrap();
        let sb = Shape::new([256, 256]).unwrap();
        let spec = BmmSpec::new(&sa, &sb).unwrap();
        let la = Layout::contiguous(sa.dims()).unwrap();
        let lb = Layout::contiguous(sb.dims()).unwrap();
        let lo = Layout::contiguous(spec.output_shape().dims()).unwrap();
        let a = Tensor::from_vec(sa.dims(), pattern(sa.len(), 5)).unwrap();
        let b = Tensor::from_vec(sb.dims(), pattern(sb.len(), 6)).unwrap();
        let dy = Tensor::from_vec(
            spec.output_shape().dims(),
            pattern(spec.output_shape().len(), 7),
        )
        .unwrap();
        let mut y = Tensor::zeros(spec.output_shape().dims()).unwrap();
        let mut da = Tensor::zeros(sa.dims()).unwrap();
        let mut db = Tensor::zeros(sb.dims()).unwrap();
        let fwd = cpu::BatchedMatmulPlan::new(cpu, &spec, &la, &lb, &lo).unwrap();
        let b_a =
            cpu::BatchedMatmulBackward::new(cpu, BinaryOperand::Left, &spec, &la, &lb, &lo, &la)
                .unwrap();
        let b_b =
            cpu::BatchedMatmulBackward::new(cpu, BinaryOperand::Right, &spec, &la, &lb, &lo, &lb)
                .unwrap();
        let av = a.view(&la).unwrap();
        let bv = b.view(&lb).unwrap();
        let dyv = dy.view(&lo).unwrap();
        let us = time_cpu(4, 24, || {
            fwd.run(&av, &bv, &mut y.view_mut(&lo).unwrap()).unwrap();
            b_a.run(
                None,
                Some(&bv),
                &dyv,
                &mut da.view_mut(&la).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            b_b.run(
                Some(&av),
                None,
                &dyv,
                &mut db.view_mut(&lb).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        rows.push(json!({ "name": "bmm-linear", "wall_us": us }));
    }

    // 5. Flash SDPA causal fwd+bwd ([2, 4, 128, 64])
    if keep("sdpa-causal-t128-d64") {
        let spec = SdpaSpec::new(2, 4, 4, 128, 128, 64, None, true).unwrap();
        let lq = Layout::contiguous(spec.query_shape().dims()).unwrap();
        let lkv = Layout::contiguous(spec.kv_shape().dims()).unwrap();
        let q = Tensor::from_vec(
            spec.query_shape().dims(),
            pattern(spec.query_shape().len(), 8),
        )
        .unwrap();
        let k =
            Tensor::from_vec(spec.kv_shape().dims(), pattern(spec.kv_shape().len(), 9)).unwrap();
        let v =
            Tensor::from_vec(spec.kv_shape().dims(), pattern(spec.kv_shape().len(), 10)).unwrap();
        let dy = Tensor::from_vec(
            spec.query_shape().dims(),
            pattern(spec.query_shape().len(), 11),
        )
        .unwrap();
        let mut o = Tensor::zeros(spec.query_shape().dims()).unwrap();
        let mut dq = Tensor::zeros(spec.query_shape().dims()).unwrap();
        let mut dk = Tensor::zeros(spec.kv_shape().dims()).unwrap();
        let mut dv = Tensor::zeros(spec.kv_shape().dims()).unwrap();
        let fwd = cpu::SdpaPlan::new(spec, &lq, &lkv, &lkv, &lq).unwrap();
        let bq =
            cpu::SdpaBackward::new(SdpaOperand::Query, spec, &lq, &lkv, &lkv, &lq, &lq).unwrap();
        let bk =
            cpu::SdpaBackward::new(SdpaOperand::Key, spec, &lq, &lkv, &lkv, &lq, &lkv).unwrap();
        let bv =
            cpu::SdpaBackward::new(SdpaOperand::Value, spec, &lq, &lkv, &lkv, &lq, &lkv).unwrap();
        let qv = q.view(&lq).unwrap();
        let kv = k.view(&lkv).unwrap();
        let vv = v.view(&lkv).unwrap();
        let dyv = dy.view(&lq).unwrap();
        let us = time_cpu(2, 16, || {
            fwd.run(&qv, &kv, &vv, &mut o.view_mut(&lq).unwrap())
                .unwrap();
            bq.run(
                Some(&qv),
                Some(&kv),
                Some(&vv),
                &dyv,
                &mut dq.view_mut(&lq).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            bk.run(
                Some(&qv),
                Some(&kv),
                Some(&vv),
                &dyv,
                &mut dk.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            bv.run(
                Some(&qv),
                Some(&kv),
                None,
                &dyv,
                &mut dv.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        rows.push(json!({ "name": "sdpa-causal-t128-d64", "wall_us": us }));
    }

    // 6. Full Audio Transformer Self-Attention + Residual + FFN Step with AdamW
    if keep("audio-transformer-step-adamw") {
        let (bs, t, h, d) = (2usize, 64, 4, 32);
        let dm = h * d;
        let mut b = Builder::new().unwrap();
        let x = b.leaf([bs, t, dm], false).unwrap();
        let wq = b.leaf([dm, dm], true).unwrap();
        let wk = b.leaf([dm, dm], true).unwrap();
        let wv = b.leaf([dm, dm], true).unwrap();
        let wo = b.leaf([dm, dm], true).unwrap();
        let tgt = b.leaf([bs, t, dm], false).unwrap();
        let xn = b.rms_norm(x, 1, 1e-5).unwrap();
        let q = b.bmm(xn, wq).unwrap();
        let k = b.bmm(xn, wk).unwrap();
        let v = b.bmm(xn, wv).unwrap();
        let q4 = b.reshape(q, [bs, t, h, d]).unwrap();
        let qh = b.permute(q4, &[0, 2, 1, 3]).unwrap();
        let k4 = b.reshape(k, [bs, t, h, d]).unwrap();
        let kh = b.permute(k4, &[0, 2, 1, 3]).unwrap();
        let v4 = b.reshape(v, [bs, t, h, d]).unwrap();
        let vh = b.permute(v4, &[0, 2, 1, 3]).unwrap();
        let attn = b.sdpa(qh, kh, vh, None, true).unwrap();
        let attn_p = b.permute(attn, &[0, 2, 1, 3]).unwrap();
        let attn_c = b.unary(UnaryOp::Silu, attn_p).unwrap();
        let attn_f = b.reshape(attn_c, [bs, t, dm]).unwrap();
        let proj = b.bmm(attn_f, wo).unwrap();
        let res = b.binary(BinaryOp::Add, proj, x).unwrap();
        let diff = b.binary(BinaryOp::Sub, res, tgt).unwrap();
        let sq = b.unary(UnaryOp::Square, diff).unwrap();
        let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1, 2], false).unwrap();
        let prog = b.finish(&[loss]).unwrap();

        let sx =
            Shared::new(Tensor::from_vec([bs, t, dm], pattern(bs * t * dm, 1)).unwrap()).unwrap();
        let swq = Shared::new(Tensor::from_vec([dm, dm], pattern(dm * dm, 2)).unwrap()).unwrap();
        let swk = Shared::new(Tensor::from_vec([dm, dm], pattern(dm * dm, 3)).unwrap()).unwrap();
        let swv = Shared::new(Tensor::from_vec([dm, dm], pattern(dm * dm, 4)).unwrap()).unwrap();
        let swo = Shared::new(Tensor::from_vec([dm, dm], pattern(dm * dm, 5)).unwrap()).unwrap();
        let st =
            Shared::new(Tensor::from_vec([bs, t, dm], pattern(bs * t * dm, 6)).unwrap()).unwrap();
        let mut tape = cpu::Tape::new(
            cpu,
            &prog,
            &[
                Binding::new(x, sx),
                Binding::new(wq, swq),
                Binding::new(wk, swk),
                Binding::new(wv, swv),
                Binding::new(wo, swo),
                Binding::new(tgt, st),
            ],
        )
        .unwrap();
        let mut opt = cpu::AdamW::for_program(
            &prog,
            &[wq, wk, wv, wo],
            AdamWConfig {
                lr: 1e-3,
                max_grad_norm: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap();
        let us = time_cpu(2, 16, || {
            tape.step_scalar_adamw(&mut opt).unwrap();
        });
        rows.push(json!({ "name": "audio-transformer-step-adamw", "wall_us": us, "fused": tape.has_fused_plan() }));
    }

    // 7. Full Spectral-Vocoder U-Net Step with CenteredSTFT + Conv1d + ConvTranspose1d + iSTFT + AdamW
    if keep("spectral-vocoder-step-adamw") {
        let (bs, samples) = (2usize, 4096);
        let mut b = Builder::new().unwrap();
        let wav = b.leaf([bs, samples], false).unwrap();
        let tgt = b.leaf([bs, samples], false).unwrap();
        let w_enc = b.leaf([16, 257, 3], true).unwrap();
        let w_dec = b.leaf([16, 257, 4], true).unwrap();
        let spec = b.centered_stft(wav, 512, 128, None).unwrap(); // [2, 33, 257, 2]
        // Split real/imag via slice on axis 3 -> enhance magnitude/real path cleanly:
        let re_s = b.slice(spec, 3, 0, 1, 1).unwrap();
        let re = b.reshape(re_s, [bs, 33, 257]).unwrap();
        let im_s = b.slice(spec, 3, 1, 1, 1).unwrap();
        let im = b.reshape(im_s, [bs, 33, 257]).unwrap();
        let re_cf = b.permute(re, &[0, 2, 1]).unwrap(); // [2, 257, 33]
        let re_pad = b
            .pad(re_cf, &[(0, 0), (0, 0), (2, 0)], PadMode::Reflect)
            .unwrap();
        let enc = b.conv1d(re_pad, w_enc, 2, 0, 1, 1).unwrap(); // [2, 16, 17]
        let enc_n = b.rms_norm(enc, 1, 1e-5).unwrap();
        let act = b.unary(UnaryOp::Silu, enc_n).unwrap();
        let dec = b.conv_transpose1d(act, w_dec, 2, 1, 0, 1, 1).unwrap(); // [2, 257, 34]
        let dec_crop = b.slice(dec, 2, 0, 33, 1).unwrap(); // [2, 257, 33]
        let dec_p = b.permute(dec_crop, &[0, 2, 1]).unwrap();
        let re_out = b.reshape(dec_p, [bs, 33, 257, 1]).unwrap();
        let im_out = b.reshape(im, [bs, 33, 257, 1]).unwrap();
        let merged = b.cat(re_out, im_out, 3).unwrap();
        let recon = b.istft(merged, samples, 512, 128, None).unwrap();
        let diff = b.binary(BinaryOp::Sub, recon, tgt).unwrap();
        let sq = b.unary(UnaryOp::Square, diff).unwrap();
        let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1], false).unwrap();
        let prog = b.finish(&[loss]).unwrap();

        let swav = Shared::new(Tensor::from_vec([bs, samples], pattern(bs * samples, 1)).unwrap())
            .unwrap();
        let stgt = Shared::new(Tensor::from_vec([bs, samples], pattern(bs * samples, 2)).unwrap())
            .unwrap();
        let swe =
            Shared::new(Tensor::from_vec([16, 257, 3], pattern(16 * 257 * 3, 3)).unwrap()).unwrap();
        let swd =
            Shared::new(Tensor::from_vec([16, 257, 4], pattern(16 * 257 * 4, 4)).unwrap()).unwrap();
        let mut tape = cpu::Tape::new(
            cpu,
            &prog,
            &[
                Binding::new(wav, swav),
                Binding::new(tgt, stgt),
                Binding::new(w_enc, swe),
                Binding::new(w_dec, swd),
            ],
        )
        .unwrap();
        let mut opt = cpu::AdamW::for_program(
            &prog,
            &[w_enc, w_dec],
            AdamWConfig {
                lr: 1e-3,
                max_grad_norm: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap();
        let us = time_cpu(2, 16, || {
            tape.step_scalar_adamw(&mut opt).unwrap();
        });
        rows.push(json!({ "name": "spectral-vocoder-step-adamw", "wall_us": us, "fused": tape.has_fused_plan() }));
    }

    rows
}

#[cfg(feature = "cuda")]
fn bench_cuda(filter: &str) -> Vec<Value> {
    let cuda = Cuda::new(0).unwrap();
    let mut rows = Vec::new();
    let keep = |name: &str| filter.is_empty() || name.contains(filter);

    // 1. LayerNorm & RmsNorm fwd+bwd
    for &(name, kind, shape) in &[
        (
            "norm-layernorm-256",
            NormKind::LayerNorm,
            [16usize, 256, 256],
        ),
        ("norm-rmsnorm-512", NormKind::RmsNorm, [16, 256, 512]),
    ] {
        if !keep(name) {
            continue;
        }
        let s = Shape::new(shape).unwrap();
        let l = Layout::contiguous(shape).unwrap();
        let spec = NormSpec::new(kind, &s, 1, 1e-5).unwrap();
        let x = cuda
            .upload(&Tensor::from_vec(shape, pattern(s.len(), 1)).unwrap())
            .unwrap();
        let dy = cuda
            .upload(&Tensor::from_vec(shape, pattern(s.len(), 2)).unwrap())
            .unwrap();
        let mut y = cuda.zeros(shape).unwrap();
        let mut dx = cuda.zeros(shape).unwrap();
        let fwd = cuda::NormPlan::new(&cuda, &spec, &l, &l).unwrap();
        let bwd = cuda::NormBackward::new(&cuda, &spec, &l, &l, &l).unwrap();
        let xv = x.view(&l).unwrap();
        let dyv = dy.view(&l).unwrap();
        let (wall_us, device_us) = time_cuda(&cuda, 16, 128, || {
            fwd.run(&xv, &mut y.view_mut(&l).unwrap()).unwrap();
            bwd.run(
                &xv,
                &dyv,
                &mut dx.view_mut(&l).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        rows.push(json!({ "name": name, "wall_us": wall_us, "device_us": device_us }));
    }

    // 2. Softmax fwd+bwd
    if keep("softmax-256") {
        let shape = [8usize, 8, 256, 256];
        let s = Shape::new(shape).unwrap();
        let l = Layout::contiguous(shape).unwrap();
        let spec = SoftmaxSpec::new(SoftmaxKind::Softmax, &s, 3).unwrap();
        let x = cuda
            .upload(&Tensor::from_vec(shape, pattern(s.len(), 3)).unwrap())
            .unwrap();
        let dy = cuda
            .upload(&Tensor::from_vec(shape, pattern(s.len(), 4)).unwrap())
            .unwrap();
        let mut y = cuda.zeros(shape).unwrap();
        let mut dx = cuda.zeros(shape).unwrap();
        let fwd = cuda::SoftmaxPlan::new(&cuda, &spec, &l, &l).unwrap();
        let bwd = cuda::SoftmaxBackward::new(&cuda, &spec, &l, &l, &l).unwrap();
        let xv = x.view(&l).unwrap();
        let dyv = dy.view(&l).unwrap();
        let (wall_us, device_us) = time_cuda(&cuda, 16, 128, || {
            fwd.run(&xv, &mut y.view_mut(&l).unwrap()).unwrap();
            bwd.run_from_output(
                &y.view(&l).unwrap(),
                &dyv,
                &mut dx.view_mut(&l).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        rows.push(json!({ "name": "softmax-256", "wall_us": wall_us, "device_us": device_us }));
    }

    // 3. BMM linear fwd+bwd ([8, 256, 256] @ [256, 512])
    if keep("bmm-linear") {
        let sa = Shape::new([8, 256, 256]).unwrap();
        let sb = Shape::new([256, 512]).unwrap();
        let spec = BmmSpec::new(&sa, &sb).unwrap();
        let la = Layout::contiguous(sa.dims()).unwrap();
        let lb = Layout::contiguous(sb.dims()).unwrap();
        let lo = Layout::contiguous(spec.output_shape().dims()).unwrap();
        let a = cuda
            .upload(&Tensor::from_vec(sa.dims(), pattern(sa.len(), 5)).unwrap())
            .unwrap();
        let b = cuda
            .upload(&Tensor::from_vec(sb.dims(), pattern(sb.len(), 6)).unwrap())
            .unwrap();
        let dy = cuda
            .upload(
                &Tensor::from_vec(
                    spec.output_shape().dims(),
                    pattern(spec.output_shape().len(), 7),
                )
                .unwrap(),
            )
            .unwrap();
        let mut y = cuda.zeros(spec.output_shape().dims()).unwrap();
        let mut da = cuda.zeros(sa.dims()).unwrap();
        let mut db = cuda.zeros(sb.dims()).unwrap();
        let fwd = cuda::BatchedMatmulPlan::new(&cuda, &spec, &la, &lb, &lo).unwrap();
        let b_a =
            cuda::BatchedMatmulBackward::new(&cuda, BinaryOperand::Left, &spec, &la, &lb, &lo, &la)
                .unwrap();
        let b_b = cuda::BatchedMatmulBackward::new(
            &cuda,
            BinaryOperand::Right,
            &spec,
            &la,
            &lb,
            &lo,
            &lb,
        )
        .unwrap();
        let av = a.view(&la).unwrap();
        let bv = b.view(&lb).unwrap();
        let dyv = dy.view(&lo).unwrap();
        let (wall_us, device_us) = time_cuda(&cuda, 16, 128, || {
            fwd.run(&av, &bv, &mut y.view_mut(&lo).unwrap()).unwrap();
            b_a.run(
                None,
                Some(&bv),
                &dyv,
                &mut da.view_mut(&la).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            b_b.run(
                Some(&av),
                None,
                &dyv,
                &mut db.view_mut(&lb).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        rows.push(json!({ "name": "bmm-linear", "wall_us": wall_us, "device_us": device_us }));
    }

    // 4. Strict-FP32 Flash SDPA benchmarks (Causal T=256, Non-Causal T=512, GQA Causal T=512)
    for &(name, bs, h, hkv, t, d, causal) in &[
        (
            "sdpa-causal-t256-d64",
            4usize,
            8usize,
            8usize,
            256usize,
            64usize,
            true,
        ),
        ("sdpa-noncausal-t512-d64", 2, 8, 8, 512, 64, false),
        ("sdpa-gqa-causal-t512-d64", 2, 8, 2, 512, 64, true),
    ] {
        if !keep(name) {
            continue;
        }
        let spec = SdpaSpec::new(bs, h, hkv, t, t, d, None, causal).unwrap();
        let lq = Layout::contiguous(spec.query_shape().dims()).unwrap();
        let lkv = Layout::contiguous(spec.kv_shape().dims()).unwrap();
        let q = cuda
            .upload(
                &Tensor::from_vec(
                    spec.query_shape().dims(),
                    pattern(spec.query_shape().len(), 8),
                )
                .unwrap(),
            )
            .unwrap();
        let k = cuda
            .upload(
                &Tensor::from_vec(spec.kv_shape().dims(), pattern(spec.kv_shape().len(), 9))
                    .unwrap(),
            )
            .unwrap();
        let v = cuda
            .upload(
                &Tensor::from_vec(spec.kv_shape().dims(), pattern(spec.kv_shape().len(), 10))
                    .unwrap(),
            )
            .unwrap();
        let dy = cuda
            .upload(
                &Tensor::from_vec(
                    spec.query_shape().dims(),
                    pattern(spec.query_shape().len(), 11),
                )
                .unwrap(),
            )
            .unwrap();
        let mut o = cuda.zeros(spec.query_shape().dims()).unwrap();
        let mut dq = cuda.zeros(spec.query_shape().dims()).unwrap();
        let mut dk = cuda.zeros(spec.kv_shape().dims()).unwrap();
        let mut dv = cuda.zeros(spec.kv_shape().dims()).unwrap();
        let fwd = cuda::SdpaPlan::new(&cuda, spec, &lq, &lkv, &lkv, &lq).unwrap();
        let bq =
            cuda::SdpaBackward::new(&cuda, SdpaOperand::Query, spec, &lq, &lkv, &lkv, &lq, &lq)
                .unwrap();
        let bk = cuda::SdpaBackward::new(&cuda, SdpaOperand::Key, spec, &lq, &lkv, &lkv, &lq, &lkv)
            .unwrap();
        let bv =
            cuda::SdpaBackward::new(&cuda, SdpaOperand::Value, spec, &lq, &lkv, &lkv, &lq, &lkv)
                .unwrap();
        let qv = q.view(&lq).unwrap();
        let kv = k.view(&lkv).unwrap();
        let vv = v.view(&lkv).unwrap();
        let dyv = dy.view(&lq).unwrap();
        let (wall_us, device_us) = time_cuda(&cuda, 8, 64, || {
            fwd.run(&qv, &kv, &vv, &mut o.view_mut(&lq).unwrap())
                .unwrap();
            bq.run(
                Some(&qv),
                Some(&kv),
                Some(&vv),
                &dyv,
                &mut dq.view_mut(&lq).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            bk.run(
                Some(&qv),
                Some(&kv),
                Some(&vv),
                &dyv,
                &mut dk.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
            bv.run(
                Some(&qv),
                Some(&kv),
                None,
                &dyv,
                &mut dv.view_mut(&lkv).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        });
        rows.push(json!({ "name": name, "wall_us": wall_us, "device_us": device_us }));
    }

    // 5. Full Audio Transformer Self-Attention + Residual + AdamW in ONE Single CUDA Graph!
    if keep("audio-transformer-step-adamw") {
        let (bs, t, h, d) = (4usize, 256, 4, 64);
        let dm = h * d;
        let mut b = Builder::new().unwrap();
        let x = b.leaf([bs, t, dm], false).unwrap();
        let wq = b.leaf([dm, dm], true).unwrap();
        let wk = b.leaf([dm, dm], true).unwrap();
        let wv = b.leaf([dm, dm], true).unwrap();
        let wo = b.leaf([dm, dm], true).unwrap();
        let tgt = b.leaf([bs, t, dm], false).unwrap();
        let xn = b.rms_norm(x, 1, 1e-5).unwrap();
        let q = b.bmm(xn, wq).unwrap();
        let k = b.bmm(xn, wk).unwrap();
        let v = b.bmm(xn, wv).unwrap();
        let q4 = b.reshape(q, [bs, t, h, d]).unwrap();
        let qh = b.permute(q4, &[0, 2, 1, 3]).unwrap();
        let k4 = b.reshape(k, [bs, t, h, d]).unwrap();
        let kh = b.permute(k4, &[0, 2, 1, 3]).unwrap();
        let v4 = b.reshape(v, [bs, t, h, d]).unwrap();
        let vh = b.permute(v4, &[0, 2, 1, 3]).unwrap();
        let attn = b.sdpa(qh, kh, vh, None, true).unwrap();
        let attn_p = b.permute(attn, &[0, 2, 1, 3]).unwrap();
        let attn_c = b.unary(UnaryOp::Silu, attn_p).unwrap();
        let attn_f = b.reshape(attn_c, [bs, t, dm]).unwrap();
        let proj = b.bmm(attn_f, wo).unwrap();
        let res = b.binary(BinaryOp::Add, proj, x).unwrap();
        let diff = b.binary(BinaryOp::Sub, res, tgt).unwrap();
        let sq = b.unary(UnaryOp::Square, diff).unwrap();
        let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1, 2], false).unwrap();
        let prog = b.finish(&[loss]).unwrap();

        let sx = Shared::new(
            cuda.upload(&Tensor::from_vec([bs, t, dm], pattern(bs * t * dm, 1)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let swq = Shared::new(
            cuda.upload(&Tensor::from_vec([dm, dm], pattern(dm * dm, 2)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let swk = Shared::new(
            cuda.upload(&Tensor::from_vec([dm, dm], pattern(dm * dm, 3)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let swv = Shared::new(
            cuda.upload(&Tensor::from_vec([dm, dm], pattern(dm * dm, 4)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let swo = Shared::new(
            cuda.upload(&Tensor::from_vec([dm, dm], pattern(dm * dm, 5)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let st = Shared::new(
            cuda.upload(&Tensor::from_vec([bs, t, dm], pattern(bs * t * dm, 6)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let mut tape = cuda::Tape::new(
            &cuda,
            &prog,
            &[
                Binding::new(x, sx),
                Binding::new(wq, swq),
                Binding::new(wk, swk),
                Binding::new(wv, swv),
                Binding::new(wo, swo),
                Binding::new(tgt, st),
            ],
        )
        .unwrap();
        let mut opt = cuda::AdamW::for_program(
            &cuda,
            &prog,
            &[wq, wk, wv, wo],
            AdamWConfig {
                lr: 1e-3,
                max_grad_norm: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap();
        let (wall_us, device_us) = time_cuda(&cuda, 8, 64, || {
            tape.step_scalar_adamw_captured(&mut opt).unwrap();
        });
        rows.push(json!({
            "name": "audio-transformer-step-adamw",
            "wall_us": wall_us,
            "device_us": device_us,
            "captured": tape.has_captured_graph(),
            "fused": tape.has_fused_plan(),
        }));
    }

    // 6. Full Spectral-Vocoder U-Net Step in ONE Single CUDA Graph!
    if keep("spectral-vocoder-step-adamw") {
        let (bs, samples) = (4usize, 16000);
        let mut b = Builder::new().unwrap();
        let wav = b.leaf([bs, samples], false).unwrap();
        let tgt = b.leaf([bs, samples], false).unwrap();
        let w_enc = b.leaf([32, 257, 3], true).unwrap();
        let w_dec = b.leaf([32, 257, 4], true).unwrap();
        let spec = b.centered_stft(wav, 512, 128, None).unwrap(); // [4, 126, 257, 2]
        let f_cnt = b.shape(spec).unwrap().dims()[1];
        let re_s = b.slice(spec, 3, 0, 1, 1).unwrap();
        let re = b.reshape(re_s, [bs, f_cnt, 257]).unwrap();
        let im_s = b.slice(spec, 3, 1, 1, 1).unwrap();
        let im = b.reshape(im_s, [bs, f_cnt, 257]).unwrap();
        let re_cf = b.permute(re, &[0, 2, 1]).unwrap();
        let re_pad = b
            .pad(re_cf, &[(0, 0), (0, 0), (2, 0)], PadMode::Reflect)
            .unwrap();
        let enc = b.conv1d(re_pad, w_enc, 2, 0, 1, 1).unwrap();
        let enc_n = b.rms_norm(enc, 1, 1e-5).unwrap();
        let act = b.unary(UnaryOp::Silu, enc_n).unwrap();
        let dec = b.conv_transpose1d(act, w_dec, 2, 1, 0, 1, 1).unwrap();
        let dec_crop = b.slice(dec, 2, 0, f_cnt, 1).unwrap();
        let dec_p = b.permute(dec_crop, &[0, 2, 1]).unwrap();
        let re_out = b.reshape(dec_p, [bs, f_cnt, 257, 1]).unwrap();
        let im_out = b.reshape(im, [bs, f_cnt, 257, 1]).unwrap();
        let merged = b.cat(re_out, im_out, 3).unwrap();
        let recon = b.istft(merged, samples, 512, 128, None).unwrap();
        let diff = b.binary(BinaryOp::Sub, recon, tgt).unwrap();
        let sq = b.unary(UnaryOp::Square, diff).unwrap();
        let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1], false).unwrap();
        let prog = b.finish(&[loss]).unwrap();

        let swav = Shared::new(
            cuda.upload(&Tensor::from_vec([bs, samples], pattern(bs * samples, 1)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let stgt = Shared::new(
            cuda.upload(&Tensor::from_vec([bs, samples], pattern(bs * samples, 2)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let swe = Shared::new(
            cuda.upload(&Tensor::from_vec([32, 257, 3], pattern(32 * 257 * 3, 3)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let swd = Shared::new(
            cuda.upload(&Tensor::from_vec([32, 257, 4], pattern(32 * 257 * 4, 4)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let mut tape = cuda::Tape::new(
            &cuda,
            &prog,
            &[
                Binding::new(wav, swav),
                Binding::new(tgt, stgt),
                Binding::new(w_enc, swe),
                Binding::new(w_dec, swd),
            ],
        )
        .unwrap();
        let mut opt = cuda::AdamW::for_program(
            &cuda,
            &prog,
            &[w_enc, w_dec],
            AdamWConfig {
                lr: 1e-3,
                max_grad_norm: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap();
        let (wall_us, device_us) = time_cuda(&cuda, 8, 64, || {
            tape.step_scalar_adamw_captured(&mut opt).unwrap();
        });
        rows.push(json!({
            "name": "spectral-vocoder-step-adamw",
            "wall_us": wall_us,
            "device_us": device_us,
            "captured": tape.has_captured_graph(),
            "fused": tape.has_fused_plan(),
        }));
    }

    rows
}

fn main() {
    let mut device = "cpu".to_string();
    let mut mode = "bench".to_string();
    let mut filter = String::new();
    let mut threads = 1usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => device = args.next().unwrap_or_else(|| "cpu".into()),
            "--mode" => mode = args.next().unwrap_or_else(|| "bench".into()),
            "--filter" => filter = args.next().unwrap_or_default(),
            "--threads" => threads = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown flag: {other}"),
        }
    }

    let cpu = {
        #[cfg(feature = "mkl")]
        {
            Cpu::mkl(threads as u16).unwrap()
        }
        #[cfg(not(feature = "mkl"))]
        {
            let _ = threads;
            Cpu::Reference
        }
    };

    match (device.as_str(), mode.as_str()) {
        ("cpu", "verify-json") => println!("{}", run_verify_cpu()),
        ("cpu", "bench") => println!(
            "{}",
            json!({ "device": "cpu", "results": bench_cpu(cpu, &filter) })
        ),
        #[cfg(feature = "cuda")]
        ("cuda", "verify-json") => println!("{}", run_verify_cuda()),
        #[cfg(feature = "cuda")]
        ("cuda", "bench") => println!(
            "{}",
            json!({ "device": "cuda", "results": bench_cuda(&filter) })
        ),
        (d, m) => panic!("unsupported device/mode combination: {d}/{m}"),
    }
}
