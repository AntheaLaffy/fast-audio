#![cfg(feature = "cuda")]

use fast_audio::{
    BinaryOp, BinaryOperand, CatSpec, Conv2dSpec, ConvTranspose1dSpec, GradientMode, Layout,
    PadMode, PadSpec, ReduceOp, Shape, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu,
    cuda::{
        CatBackward, CatPlan, ConvTranspose1dBackward, ConvTranspose1dPlan,
        ConvTranspose2dBackward, ConvTranspose2dPlan, Cuda, PadBackward, PadPlan, Tape,
    },
};

fn pattern(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 19 + 5) % 29) as f32 - 14.0) * 0.07)
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
fn cuda_conv_transpose1d_2d_pad_cat_match_cpu() {
    let cuda = Cuda::new(0).unwrap();

    // 1. ConvTranspose1d (grouped, dilated, output_padding)
    let spec1 = ConvTranspose1dSpec {
        batch: 2,
        in_channels: 6,
        out_channels: 4,
        input_len: 9,
        kernel: 4,
        stride: 2,
        padding: 1,
        output_padding: 1,
        dilation: 1,
        groups: 2,
    };
    let in_s = [spec1.batch, spec1.in_channels, spec1.input_len];
    let w_s = spec1.weight_shape();
    let out_s = spec1.output_shape().unwrap();
    let xl = Layout::contiguous(in_s).unwrap();
    let wl = Layout::contiguous(w_s).unwrap();
    let yl = Layout::contiguous(out_s).unwrap();
    let x = Tensor::from_vec(in_s, pattern(in_s.iter().product(), 1)).unwrap();
    let w = Tensor::from_vec(w_s, pattern(w_s.iter().product(), 2)).unwrap();
    let dy = Tensor::from_vec(out_s, pattern(out_s.iter().product(), 3)).unwrap();

    let mut cpu_fwd =
        cpu::ConvTranspose1dPlan::new(cpu::Cpu::Reference, spec1, &xl, &wl, &yl).unwrap();
    let mut y_cpu = Tensor::zeros(out_s).unwrap();
    cpu_fwd
        .run(
            &x.view(&xl).unwrap(),
            &w.view(&wl).unwrap(),
            &mut y_cpu.view_mut(&yl).unwrap(),
        )
        .unwrap();
    let mut dx_cpu = Tensor::zeros(in_s).unwrap();
    let mut dw_cpu = Tensor::zeros(w_s).unwrap();
    cpu::ConvTranspose1dBackward::new(
        cpu::Cpu::Reference,
        BinaryOperand::Left,
        spec1,
        &xl,
        &wl,
        &yl,
        &xl,
    )
    .unwrap()
    .run(
        None,
        Some(&w.view(&wl).unwrap()),
        &dy.view(&yl).unwrap(),
        &mut dx_cpu.view_mut(&xl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cpu::ConvTranspose1dBackward::new(
        cpu::Cpu::Reference,
        BinaryOperand::Right,
        spec1,
        &xl,
        &wl,
        &yl,
        &wl,
    )
    .unwrap()
    .run(
        Some(&x.view(&xl).unwrap()),
        None,
        &dy.view(&yl).unwrap(),
        &mut dw_cpu.view_mut(&wl).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    let dx_dev_in = cuda.upload(&x).unwrap();
    let dw_dev_in = cuda.upload(&w).unwrap();
    let ddy_dev = cuda.upload(&dy).unwrap();
    let mut dy_dev = cuda.zeros(out_s).unwrap();
    let gpu_fwd = ConvTranspose1dPlan::new(&cuda, spec1, &xl, &wl, &yl).unwrap();
    gpu_fwd
        .run(
            &dx_dev_in.view(&xl).unwrap(),
            &dw_dev_in.view(&wl).unwrap(),
            &mut dy_dev.view_mut(&yl).unwrap(),
        )
        .unwrap();
    assert!(max_abs_diff(cuda.download(&dy_dev).unwrap().data(), y_cpu.data()) < 1e-5);

    let mut ddx_dev = cuda.zeros(in_s).unwrap();
    let mut ddw_dev = cuda.zeros(w_s).unwrap();
    ConvTranspose1dBackward::new(&cuda, BinaryOperand::Left, spec1, &xl, &wl, &yl, &xl)
        .unwrap()
        .run(
            None,
            Some(&dw_dev_in.view(&wl).unwrap()),
            &ddy_dev.view(&yl).unwrap(),
            &mut ddx_dev.view_mut(&xl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    ConvTranspose1dBackward::new(&cuda, BinaryOperand::Right, spec1, &xl, &wl, &yl, &wl)
        .unwrap()
        .run(
            Some(&dx_dev_in.view(&xl).unwrap()),
            None,
            &ddy_dev.view(&yl).unwrap(),
            &mut ddw_dev.view_mut(&wl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    assert!(max_abs_diff(cuda.download(&ddx_dev).unwrap().data(), dx_cpu.data()) < 1e-5);
    assert!(max_abs_diff(cuda.download(&ddw_dev).unwrap().data(), dw_cpu.data()) < 1e-5);

    // 2. ConvTranspose2d
    let spec2 = Conv2dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 6,
        input: [4, 5],
        kernel: [3, 2],
        stride: [2, 2],
        padding: [1, 0],
        dilation: [1, 2],
        groups: 2,
        transposed: true,
        output_padding: [1, 1],
    };
    let in2 = spec2.input_shape();
    let w2 = spec2.weight_shape();
    let out2 = spec2.output_shape().unwrap();
    let xl2 = Layout::contiguous(in2).unwrap();
    let wl2 = Layout::contiguous(w2).unwrap();
    let yl2 = Layout::contiguous(out2).unwrap();
    let x2 = Tensor::from_vec(in2, pattern(in2.iter().product(), 4)).unwrap();
    let wt2 = Tensor::from_vec(w2, pattern(w2.iter().product(), 5)).unwrap();
    let dy2 = Tensor::from_vec(out2, pattern(out2.iter().product(), 6)).unwrap();
    let mut y2_cpu = Tensor::zeros(out2).unwrap();
    cpu::ConvTranspose2dPlan::new(cpu::Cpu::Reference, spec2, &xl2, &wl2, &yl2)
        .unwrap()
        .run(
            &x2.view(&xl2).unwrap(),
            &wt2.view(&wl2).unwrap(),
            &mut y2_cpu.view_mut(&yl2).unwrap(),
        )
        .unwrap();
    let mut dx2_cpu = Tensor::zeros(in2).unwrap();
    let mut dw2_cpu = Tensor::zeros(w2).unwrap();
    cpu::ConvTranspose2dBackward::new(
        cpu::Cpu::Reference,
        BinaryOperand::Left,
        spec2,
        &xl2,
        &wl2,
        &yl2,
        &xl2,
    )
    .unwrap()
    .run(
        None,
        Some(&wt2.view(&wl2).unwrap()),
        &dy2.view(&yl2).unwrap(),
        &mut dx2_cpu.view_mut(&xl2).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();
    cpu::ConvTranspose2dBackward::new(
        cpu::Cpu::Reference,
        BinaryOperand::Right,
        spec2,
        &xl2,
        &wl2,
        &yl2,
        &wl2,
    )
    .unwrap()
    .run(
        Some(&x2.view(&xl2).unwrap()),
        None,
        &dy2.view(&yl2).unwrap(),
        &mut dw2_cpu.view_mut(&wl2).unwrap(),
        GradientMode::Overwrite,
    )
    .unwrap();

    let gx2 = cuda.upload(&x2).unwrap();
    let gw2 = cuda.upload(&wt2).unwrap();
    let gdy2 = cuda.upload(&dy2).unwrap();
    let mut gy2 = cuda.zeros(out2).unwrap();
    ConvTranspose2dPlan::new(&cuda, spec2, &xl2, &wl2, &yl2)
        .unwrap()
        .run(
            &gx2.view(&xl2).unwrap(),
            &gw2.view(&wl2).unwrap(),
            &mut gy2.view_mut(&yl2).unwrap(),
        )
        .unwrap();
    assert!(max_abs_diff(cuda.download(&gy2).unwrap().data(), y2_cpu.data()) < 1e-5);
    let mut gdx2 = cuda.zeros(in2).unwrap();
    let mut gdw2 = cuda.zeros(w2).unwrap();
    ConvTranspose2dBackward::new(&cuda, BinaryOperand::Left, spec2, &xl2, &wl2, &yl2, &xl2)
        .unwrap()
        .run(
            None,
            Some(&gw2.view(&wl2).unwrap()),
            &gdy2.view(&yl2).unwrap(),
            &mut gdx2.view_mut(&xl2).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    ConvTranspose2dBackward::new(&cuda, BinaryOperand::Right, spec2, &xl2, &wl2, &yl2, &wl2)
        .unwrap()
        .run(
            Some(&gx2.view(&xl2).unwrap()),
            None,
            &gdy2.view(&yl2).unwrap(),
            &mut gdw2.view_mut(&wl2).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    assert!(max_abs_diff(cuda.download(&gdx2).unwrap().data(), dx2_cpu.data()) < 1e-5);
    assert!(max_abs_diff(cuda.download(&gdw2).unwrap().data(), dw2_cpu.data()) < 1e-5);

    // 3. Pad (Constant, Reflect, Replicate)
    let pshape = Shape::new([2, 3, 5]).unwrap();
    let pads = [(0, 0), (1, 2), (2, 1)];
    for mode in [PadMode::Constant(0.5), PadMode::Reflect, PadMode::Replicate] {
        let pspec = PadSpec::new(&pshape, &pads, mode).unwrap();
        let pl = Layout::contiguous(pshape.dims()).unwrap();
        let pol = Layout::contiguous(pspec.output_shape().dims()).unwrap();
        let px = Tensor::from_vec(pshape.dims(), pattern(pshape.len(), 11)).unwrap();
        let pdy = Tensor::from_vec(
            pspec.output_shape().dims(),
            pattern(pspec.output_shape().len(), 12),
        )
        .unwrap();
        let mut py_cpu = Tensor::zeros(pspec.output_shape().dims()).unwrap();
        let mut pdx_cpu = Tensor::zeros(pshape.dims()).unwrap();
        cpu::PadPlan::new(&pspec, &pl, &pol)
            .unwrap()
            .run(&px.view(&pl).unwrap(), &mut py_cpu.view_mut(&pol).unwrap())
            .unwrap();
        cpu::PadBackward::new(&pspec, &pol, &pl)
            .unwrap()
            .run(
                &pdy.view(&pol).unwrap(),
                &mut pdx_cpu.view_mut(&pl).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();

        let gpx = cuda.upload(&px).unwrap();
        let gpdy = cuda.upload(&pdy).unwrap();
        let mut gpy = cuda.zeros(pspec.output_shape().dims()).unwrap();
        let mut gpdx = cuda.zeros(pshape.dims()).unwrap();
        PadPlan::new(&cuda, &pspec, &pl, &pol)
            .unwrap()
            .run(&gpx.view(&pl).unwrap(), &mut gpy.view_mut(&pol).unwrap())
            .unwrap();
        PadBackward::new(&cuda, &pspec, &pol, &pl)
            .unwrap()
            .run(
                &gpdy.view(&pol).unwrap(),
                &mut gpdx.view_mut(&pl).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        assert!(max_abs_diff(cuda.download(&gpy).unwrap().data(), py_cpu.data()) < 1e-6);
        assert!(max_abs_diff(cuda.download(&gpdx).unwrap().data(), pdx_cpu.data()) < 1e-5);
    }

    // 4. Cat
    let sa = Shape::new([2, 3, 4]).unwrap();
    let sb = Shape::new([2, 2, 4]).unwrap();
    let cspec = CatSpec::new(&sa, &sb, 1).unwrap();
    let al = Layout::contiguous(sa.dims()).unwrap();
    let bl = Layout::contiguous(sb.dims()).unwrap();
    let cl = Layout::contiguous(cspec.output_shape().dims()).unwrap();
    let ca = Tensor::from_vec(sa.dims(), pattern(sa.len(), 21)).unwrap();
    let cb = Tensor::from_vec(sb.dims(), pattern(sb.len(), 22)).unwrap();
    let cdy = Tensor::from_vec(
        cspec.output_shape().dims(),
        pattern(cspec.output_shape().len(), 23),
    )
    .unwrap();
    let mut cy_cpu = Tensor::zeros(cspec.output_shape().dims()).unwrap();
    let mut cda_cpu = Tensor::zeros(sa.dims()).unwrap();
    let mut cdb_cpu = Tensor::zeros(sb.dims()).unwrap();
    cpu::CatPlan::new(&cspec, &al, &bl, &cl)
        .unwrap()
        .run(
            &ca.view(&al).unwrap(),
            &cb.view(&bl).unwrap(),
            &mut cy_cpu.view_mut(&cl).unwrap(),
        )
        .unwrap();
    cpu::CatBackward::new(BinaryOperand::Left, &cspec, &cl, &al)
        .unwrap()
        .run(
            &cdy.view(&cl).unwrap(),
            &mut cda_cpu.view_mut(&al).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    cpu::CatBackward::new(BinaryOperand::Right, &cspec, &cl, &bl)
        .unwrap()
        .run(
            &cdy.view(&cl).unwrap(),
            &mut cdb_cpu.view_mut(&bl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

    let gca = cuda.upload(&ca).unwrap();
    let gcb = cuda.upload(&cb).unwrap();
    let gcdy = cuda.upload(&cdy).unwrap();
    let mut gcy = cuda.zeros(cspec.output_shape().dims()).unwrap();
    let mut gcda = cuda.zeros(sa.dims()).unwrap();
    let mut gcdb = cuda.zeros(sb.dims()).unwrap();
    CatPlan::new(&cuda, &cspec, &al, &bl, &cl)
        .unwrap()
        .run(
            &gca.view(&al).unwrap(),
            &gcb.view(&bl).unwrap(),
            &mut gcy.view_mut(&cl).unwrap(),
        )
        .unwrap();
    CatBackward::new(&cuda, BinaryOperand::Left, &cspec, &cl, &al)
        .unwrap()
        .run(
            &gcdy.view(&cl).unwrap(),
            &mut gcda.view_mut(&al).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    CatBackward::new(&cuda, BinaryOperand::Right, &cspec, &cl, &bl)
        .unwrap()
        .run(
            &gcdy.view(&cl).unwrap(),
            &mut gcdb.view_mut(&bl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    assert!(max_abs_diff(cuda.download(&gcy).unwrap().data(), cy_cpu.data()) < 1e-6);
    assert!(max_abs_diff(cuda.download(&gcda).unwrap().data(), cda_cpu.data()) < 1e-6);
    assert!(max_abs_diff(cuda.download(&gcdb).unwrap().data(), cdb_cpu.data()) < 1e-6);
}

#[test]
fn cuda_tape_unet_vocoder_captured_matches_uncaptured() {
    let cuda = Cuda::new(0).unwrap();
    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 2, 16], true).unwrap();
    let w_enc = b.leaf([4, 2, 3], true).unwrap();
    let w_dec = b.leaf([4, 2, 4], true).unwrap();
    let w_out = b.leaf([2, 4, 1], true).unwrap();
    let target = b.leaf([2, 2, 16], false).unwrap();

    let x_pad = b
        .pad(x, &[(0, 0), (0, 0), (2, 0)], PadMode::Reflect)
        .unwrap();
    let enc = b.conv1d(x_pad, w_enc, 2, 0, 1, 1).unwrap();
    let h = b.unary(UnaryOp::Silu, enc).unwrap();
    let dec = b.conv_transpose1d(h, w_dec, 2, 1, 0, 1, 1).unwrap();
    let merged = b.cat_all(&[dec, x], 1).unwrap();
    let pred = b.conv1d(merged, w_out, 1, 0, 1, 1).unwrap();
    let diff = b.binary(BinaryOp::Sub, pred, target).unwrap();
    let sq = b.binary(BinaryOp::Mul, diff, diff).unwrap();
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1, 2], false).unwrap();
    let program = b.finish(&[loss]).unwrap();

    let make_bindings = || {
        let tx = Shared::new(
            cuda.upload(&Tensor::from_vec([2, 2, 16], pattern(64, 1)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let twe = Shared::new(
            cuda.upload(&Tensor::from_vec([4, 2, 3], pattern(24, 2)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let twd = Shared::new(
            cuda.upload(&Tensor::from_vec([4, 2, 4], pattern(32, 3)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let two = Shared::new(
            cuda.upload(&Tensor::from_vec([2, 4, 1], pattern(8, 4)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let tt = Shared::new(
            cuda.upload(&Tensor::from_vec([2, 2, 16], pattern(64, 5)).unwrap())
                .unwrap(),
        )
        .unwrap();
        let binds = vec![
            Binding::new(x, tx.clone()),
            Binding::new(w_enc, twe.clone()),
            Binding::new(w_dec, twd.clone()),
            Binding::new(w_out, two.clone()),
            Binding::new(target, tt.clone()),
        ];
        (binds, tx, twe, twd, two)
    };

    let (b1, tx1, twe1, twd1, two1) = make_bindings();
    let (b2, tx2, twe2, twd2, two2) = make_bindings();
    let mut t1 = Tape::new(&cuda, &program, &b1).unwrap();
    let mut t2 = Tape::new(&cuda, &program, &b2).unwrap();

    for _ in 0..3 {
        t1.step_scalar(&[w_enc, w_dec, w_out, x], 0.02).unwrap();
        t2.step_scalar_captured(&[w_enc, w_dec, w_out, x], 0.02)
            .unwrap();
    }
    cuda.synchronize().unwrap();
    for (a, b) in [(&tx1, &tx2), (&twe1, &twe2), (&twd1, &twd2), (&two1, &two2)] {
        let ha = cuda.download(&a.read().unwrap()).unwrap();
        let hb = cuda.download(&b.read().unwrap()).unwrap();
        assert!(max_abs_diff(ha.data(), hb.data()) < 1e-6);
    }
}
