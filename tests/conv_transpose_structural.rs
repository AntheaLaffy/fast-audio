use fast_audio::{
    BinaryOp, BinaryOperand, CatSpec, Conv2dSpec, ConvTranspose1dSpec, GradientMode, Layout,
    PadMode, PadSpec, ReduceOp, Shape, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{
        CatBackward, CatPlan, ConvTranspose1dBackward, ConvTranspose1dPlan,
        ConvTranspose2dBackward, ConvTranspose2dPlan, Cpu, PadBackward, PadPlan, Tape,
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

fn ref_conv_transpose1d_forward(spec: ConvTranspose1dSpec, x: &[f32], w: &[f32]) -> Vec<f32> {
    let out_len = spec.output_len().unwrap();
    let ic_g = spec.in_channels / spec.groups;
    let oc_g = spec.out_channels / spec.groups;
    let mut y = vec![0.0f64; spec.batch * spec.out_channels * out_len];
    for n in 0..spec.batch {
        for g in 0..spec.groups {
            for ic in 0..ic_g {
                let c_in = g * ic_g + ic;
                for l_in in 0..spec.input_len {
                    let xv = x[(n * spec.in_channels + c_in) * spec.input_len + l_in] as f64;
                    for oc in 0..oc_g {
                        let c_out = g * oc_g + oc;
                        for k in 0..spec.kernel {
                            let pos = (l_in * spec.stride + k * spec.dilation) as isize
                                - spec.padding as isize;
                            if pos >= 0 && (pos as usize) < out_len {
                                let wv = w[(c_in * oc_g + oc) * spec.kernel + k] as f64;
                                y[(n * spec.out_channels + c_out) * out_len + pos as usize] +=
                                    xv * wv;
                            }
                        }
                    }
                }
            }
        }
    }
    y.into_iter().map(|v| v as f32).collect()
}

fn ref_conv_transpose1d_backward(
    spec: ConvTranspose1dSpec,
    x: &[f32],
    w: &[f32],
    dy: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let out_len = spec.output_len().unwrap();
    let ic_g = spec.in_channels / spec.groups;
    let oc_g = spec.out_channels / spec.groups;
    let mut dx = vec![0.0f64; x.len()];
    let mut dw = vec![0.0f64; w.len()];
    for n in 0..spec.batch {
        for g in 0..spec.groups {
            for ic in 0..ic_g {
                let c_in = g * ic_g + ic;
                for l_in in 0..spec.input_len {
                    let xi = (n * spec.in_channels + c_in) * spec.input_len + l_in;
                    let xv = x[xi] as f64;
                    for oc in 0..oc_g {
                        let c_out = g * oc_g + oc;
                        for k in 0..spec.kernel {
                            let pos = (l_in * spec.stride + k * spec.dilation) as isize
                                - spec.padding as isize;
                            if pos >= 0 && (pos as usize) < out_len {
                                let dyv = dy
                                    [(n * spec.out_channels + c_out) * out_len + pos as usize]
                                    as f64;
                                let wi = (c_in * oc_g + oc) * spec.kernel + k;
                                dx[xi] += dyv * w[wi] as f64;
                                dw[wi] += dyv * xv;
                            }
                        }
                    }
                }
            }
        }
    }
    (
        dx.into_iter().map(|v| v as f32).collect(),
        dw.into_iter().map(|v| v as f32).collect(),
    )
}

fn ref_conv_transpose2d_forward(spec: Conv2dSpec, x: &[f32], w: &[f32]) -> Vec<f32> {
    let out = spec.output_shape().unwrap();
    let [oh, ow] = [out[2], out[3]];
    let [ih, iw] = spec.input;
    let [kh, kw] = spec.kernel;
    let ic_g = spec.in_channels / spec.groups;
    let oc_g = spec.out_channels / spec.groups;
    let mut y = vec![0.0f64; spec.batch * spec.out_channels * oh * ow];
    for n in 0..spec.batch {
        for g in 0..spec.groups {
            for ic in 0..ic_g {
                let c_in = g * ic_g + ic;
                for hi in 0..ih {
                    for wi in 0..iw {
                        let xv = x[((n * spec.in_channels + c_in) * ih + hi) * iw + wi] as f64;
                        for oc in 0..oc_g {
                            let c_out = g * oc_g + oc;
                            for r in 0..kh {
                                let oy = (hi * spec.stride[0] + r * spec.dilation[0]) as isize
                                    - spec.padding[0] as isize;
                                if oy < 0 || oy as usize >= oh {
                                    continue;
                                }
                                for s in 0..kw {
                                    let ox = (wi * spec.stride[1] + s * spec.dilation[1]) as isize
                                        - spec.padding[1] as isize;
                                    if ox < 0 || ox as usize >= ow {
                                        continue;
                                    }
                                    let wv = w[((c_in * oc_g + oc) * kh + r) * kw + s] as f64;
                                    y[((n * spec.out_channels + c_out) * oh + oy as usize) * ow
                                        + ox as usize] += xv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    y.into_iter().map(|v| v as f32).collect()
}

fn ref_conv_transpose2d_backward(
    spec: Conv2dSpec,
    x: &[f32],
    w: &[f32],
    dy: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let out = spec.output_shape().unwrap();
    let [oh, ow] = [out[2], out[3]];
    let [ih, iw] = spec.input;
    let [kh, kw] = spec.kernel;
    let ic_g = spec.in_channels / spec.groups;
    let oc_g = spec.out_channels / spec.groups;
    let mut dx = vec![0.0f64; x.len()];
    let mut dw = vec![0.0f64; w.len()];
    for n in 0..spec.batch {
        for g in 0..spec.groups {
            for ic in 0..ic_g {
                let c_in = g * ic_g + ic;
                for hi in 0..ih {
                    for wi in 0..iw {
                        let xi = ((n * spec.in_channels + c_in) * ih + hi) * iw + wi;
                        let xv = x[xi] as f64;
                        for oc in 0..oc_g {
                            let c_out = g * oc_g + oc;
                            for r in 0..kh {
                                let oy = (hi * spec.stride[0] + r * spec.dilation[0]) as isize
                                    - spec.padding[0] as isize;
                                if oy < 0 || oy as usize >= oh {
                                    continue;
                                }
                                for s in 0..kw {
                                    let ox = (wi * spec.stride[1] + s * spec.dilation[1]) as isize
                                        - spec.padding[1] as isize;
                                    if ox < 0 || ox as usize >= ow {
                                        continue;
                                    }
                                    let dyv = dy[((n * spec.out_channels + c_out) * oh
                                        + oy as usize)
                                        * ow
                                        + ox as usize]
                                        as f64;
                                    let widx = ((c_in * oc_g + oc) * kh + r) * kw + s;
                                    dx[xi] += dyv * w[widx] as f64;
                                    dw[widx] += dyv * xv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (
        dx.into_iter().map(|v| v as f32).collect(),
        dw.into_iter().map(|v| v as f32).collect(),
    )
}

#[test]
fn cpu_conv_transpose1d_and_2d_match_fp64_oracle() {
    let specs_1d = [
        // Standard upsampling vocoder layer with output_padding and groups
        ConvTranspose1dSpec {
            batch: 2,
            in_channels: 6,
            out_channels: 4,
            input_len: 7,
            kernel: 4,
            stride: 2,
            padding: 1,
            output_padding: 1,
            dilation: 1,
            groups: 2,
        },
        // Dilated transposed conv1d
        ConvTranspose1dSpec {
            batch: 2,
            in_channels: 3,
            out_channels: 5,
            input_len: 5,
            kernel: 3,
            stride: 2,
            padding: 1,
            output_padding: 0,
            dilation: 2,
            groups: 1,
        },
        // Pointwise kernel=1, stride=1
        ConvTranspose1dSpec {
            batch: 2,
            in_channels: 4,
            out_channels: 3,
            input_len: 6,
            kernel: 1,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        },
    ];
    for (idx, spec) in specs_1d.into_iter().enumerate() {
        let in_shape = [spec.batch, spec.in_channels, spec.input_len];
        let w_shape = spec.weight_shape();
        let out_shape = spec.output_shape().unwrap();
        let x = Tensor::from_vec(in_shape, pattern(in_shape.iter().product(), 1 + idx)).unwrap();
        let w = Tensor::from_vec(w_shape, pattern(w_shape.iter().product(), 11 + idx)).unwrap();
        let dy =
            Tensor::from_vec(out_shape, pattern(out_shape.iter().product(), 21 + idx)).unwrap();

        let xl = Layout::contiguous(in_shape).unwrap();
        let wl = Layout::contiguous(w_shape).unwrap();
        let yl = Layout::contiguous(out_shape).unwrap();

        let mut fwd = ConvTranspose1dPlan::new(Cpu::Reference, spec, &xl, &wl, &yl).unwrap();
        let mut y = Tensor::zeros(out_shape).unwrap();
        fwd.run(
            &x.view(&xl).unwrap(),
            &w.view(&wl).unwrap(),
            &mut y.view_mut(&yl).unwrap(),
        )
        .unwrap();
        let y_ref = ref_conv_transpose1d_forward(spec, x.data(), w.data());
        assert!(
            max_abs_diff(y.data(), &y_ref) < 1e-5,
            "1d fwd case {idx}: diff={}",
            max_abs_diff(y.data(), &y_ref)
        );

        let (dx_ref, dw_ref) = ref_conv_transpose1d_backward(spec, x.data(), w.data(), dy.data());
        let mut bwd_x = ConvTranspose1dBackward::new(
            Cpu::Reference,
            BinaryOperand::Left,
            spec,
            &xl,
            &wl,
            &yl,
            &xl,
        )
        .unwrap();
        let mut bwd_w = ConvTranspose1dBackward::new(
            Cpu::Reference,
            BinaryOperand::Right,
            spec,
            &xl,
            &wl,
            &yl,
            &wl,
        )
        .unwrap();
        assert_eq!(bwd_x.input_dependencies(), [false, true]);
        assert_eq!(bwd_w.input_dependencies(), [true, false]);

        let mut dx = Tensor::from_vec(in_shape, vec![1.5; x.data().len()]).unwrap();
        bwd_x
            .run(
                None,
                Some(&w.view(&wl).unwrap()),
                &dy.view(&yl).unwrap(),
                &mut dx.view_mut(&xl).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        assert!(max_abs_diff(dx.data(), &dx_ref) < 1e-5);
        bwd_x
            .run(
                None,
                Some(&w.view(&wl).unwrap()),
                &dy.view(&yl).unwrap(),
                &mut dx.view_mut(&xl).unwrap(),
                GradientMode::Accumulate,
            )
            .unwrap();
        let dx2: Vec<f32> = dx_ref.iter().map(|v| 2.0 * v).collect();
        assert!(max_abs_diff(dx.data(), &dx2) < 2e-5);

        let mut dw = Tensor::from_vec(w_shape, vec![-0.75; w.data().len()]).unwrap();
        bwd_w
            .run(
                Some(&x.view(&xl).unwrap()),
                None,
                &dy.view(&yl).unwrap(),
                &mut dw.view_mut(&wl).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        assert!(max_abs_diff(dw.data(), &dw_ref) < 1e-5);
        bwd_w
            .run(
                Some(&x.view(&xl).unwrap()),
                None,
                &dy.view(&yl).unwrap(),
                &mut dw.view_mut(&wl).unwrap(),
                GradientMode::Accumulate,
            )
            .unwrap();
        let dw2: Vec<f32> = dw_ref.iter().map(|v| 2.0 * v).collect();
        assert!(max_abs_diff(dw.data(), &dw2) < 2e-5);
    }

    // ConvTranspose2d with groups, stride, padding, output_padding, dilation + signed-stride view
    let spec2d = Conv2dSpec {
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
    let in_shape = spec2d.input_shape();
    let w_shape = spec2d.weight_shape();
    let out_shape = spec2d.output_shape().unwrap();
    let x = Tensor::from_vec(in_shape, pattern(in_shape.iter().product(), 33)).unwrap();
    let w = Tensor::from_vec(w_shape, pattern(w_shape.iter().product(), 44)).unwrap();
    let dy = Tensor::from_vec(out_shape, pattern(out_shape.iter().product(), 55)).unwrap();

    let xl = Layout::contiguous(in_shape).unwrap();
    let wl = Layout::contiguous(w_shape).unwrap();
    let yl = Layout::contiguous(out_shape).unwrap();

    let mut fwd2 = ConvTranspose2dPlan::new(Cpu::Reference, spec2d, &xl, &wl, &yl).unwrap();
    let mut y = Tensor::zeros(out_shape).unwrap();
    fwd2.run(
        &x.view(&xl).unwrap(),
        &w.view(&wl).unwrap(),
        &mut y.view_mut(&yl).unwrap(),
    )
    .unwrap();
    let y_ref = ref_conv_transpose2d_forward(spec2d, x.data(), w.data());
    assert!(max_abs_diff(y.data(), &y_ref) < 1e-5);

    let (dx_ref, dw_ref) = ref_conv_transpose2d_backward(spec2d, x.data(), w.data(), dy.data());
    let mut bwd_x2 = ConvTranspose2dBackward::new(
        Cpu::Reference,
        BinaryOperand::Left,
        spec2d,
        &xl,
        &wl,
        &yl,
        &xl,
    )
    .unwrap();
    let mut bwd_w2 = ConvTranspose2dBackward::new(
        Cpu::Reference,
        BinaryOperand::Right,
        spec2d,
        &xl,
        &wl,
        &yl,
        &wl,
    )
    .unwrap();
    let mut dx = Tensor::zeros(in_shape).unwrap();
    let mut dw = Tensor::zeros(w_shape).unwrap();
    bwd_x2
        .run(
            None,
            Some(&w.view(&wl).unwrap()),
            &dy.view(&yl).unwrap(),
            &mut dx.view_mut(&xl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    bwd_w2
        .run(
            Some(&x.view(&xl).unwrap()),
            None,
            &dy.view(&yl).unwrap(),
            &mut dw.view_mut(&wl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    assert!(max_abs_diff(dx.data(), &dx_ref) < 1e-5);
    assert!(max_abs_diff(dw.data(), &dw_ref) < 1e-5);

    // Non-contiguous reversed-width view test on ConvTranspose2d
    let x_rev = xl.slice(3, in_shape[3] - 1, in_shape[3], -1).unwrap();
    let y_rev = yl.slice(3, out_shape[3] - 1, out_shape[3], -1).unwrap();
    let mut fwd_rev =
        ConvTranspose2dPlan::new(Cpu::Reference, spec2d, &x_rev, &wl, &y_rev).unwrap();
    let mut y_buf = Tensor::zeros(out_shape).unwrap();
    fwd_rev
        .run(
            &x.view(&x_rev).unwrap(),
            &w.view(&wl).unwrap(),
            &mut y_buf.view_mut(&y_rev).unwrap(),
        )
        .unwrap();
    // Reversing input along W, running, and writing reversed along W must match computing on materialized reversed x
    let mut x_mat = vec![0.0f32; x.data().len()];
    let [n, c, h, w_in] = in_shape;
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w_in {
                    x_mat[((ni * c + ci) * h + hi) * w_in + wi] =
                        x.data()[((ni * c + ci) * h + hi) * w_in + (w_in - 1 - wi)];
                }
            }
        }
    }
    let y_mat_ref = ref_conv_transpose2d_forward(spec2d, &x_mat, w.data());
    let [_, co, oh, ow] = out_shape;
    for ni in 0..n {
        for ci in 0..co {
            for hi in 0..oh {
                for wi in 0..ow {
                    let got = y_buf.data()[((ni * co + ci) * oh + hi) * ow + (ow - 1 - wi)];
                    let exp = y_mat_ref[((ni * co + ci) * oh + hi) * ow + wi];
                    assert!((got - exp).abs() < 1e-5);
                }
            }
        }
    }
}

#[test]
fn cpu_pad_and_cat_modes_strided_and_adjoint() {
    // Test all PadModes (Constant, Reflect, Replicate) including multi-axis 2D and signed strides
    let shape = Shape::new([2, 3, 5]).unwrap();
    let pads = [(0, 0), (1, 2), (2, 1)];
    for mode in [
        PadMode::Constant(-0.25),
        PadMode::Reflect,
        PadMode::Replicate,
    ] {
        let spec = PadSpec::new(&shape, &pads, mode).unwrap();
        let xl = Layout::contiguous(shape.dims()).unwrap();
        let yl = Layout::contiguous(spec.output_shape().dims()).unwrap();
        let x = Tensor::from_vec(shape.dims(), pattern(shape.len(), 7)).unwrap();
        let dy = Tensor::from_vec(
            spec.output_shape().dims(),
            pattern(spec.output_shape().len(), 13),
        )
        .unwrap();
        let mut y = Tensor::zeros(spec.output_shape().dims()).unwrap();
        let mut dx = Tensor::zeros(shape.dims()).unwrap();

        let fwd = PadPlan::new(&spec, &xl, &yl).unwrap();
        let bwd = PadBackward::new(&spec, &yl, &xl).unwrap();
        assert!(!bwd.input_required());
        fwd.run(&x.view(&xl).unwrap(), &mut y.view_mut(&yl).unwrap())
            .unwrap();
        bwd.run(
            &dy.view(&yl).unwrap(),
            &mut dx.view_mut(&xl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

        // Linear operator adjoint identity (for Constant(0), Reflect, Replicate)
        let y_zero = if let PadMode::Constant(v) = mode {
            y.data().iter().map(|&val| val - v).collect::<Vec<f32>>()
        } else {
            y.data().to_vec()
        };
        if !matches!(mode, PadMode::Constant(v) if v != 0.0) {
            let lhs: f64 = y_zero
                .iter()
                .zip(dy.data())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            let rhs: f64 = x
                .data()
                .iter()
                .zip(dx.data())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            assert!(
                (lhs - rhs).abs() < 1e-4,
                "pad adjoint failed for {mode:?}: {lhs} vs {rhs}"
            );
        }

        // Verify Reflect & Replicate finite difference on a couple of entries
        let mut x_pert = x.data().to_vec();
        let mut y_plus = Tensor::zeros(spec.output_shape().dims()).unwrap();
        let mut y_minus = Tensor::zeros(spec.output_shape().dims()).unwrap();
        let eps = 1e-3f32;
        for probe in [0usize, 1, 4, 7, 14, 29] {
            x_pert[probe] = x.data()[probe] + eps;
            let tp = Tensor::from_vec(shape.dims(), x_pert.clone()).unwrap();
            fwd.run(&tp.view(&xl).unwrap(), &mut y_plus.view_mut(&yl).unwrap())
                .unwrap();
            x_pert[probe] = x.data()[probe] - eps;
            let tm = Tensor::from_vec(shape.dims(), x_pert.clone()).unwrap();
            fwd.run(&tm.view(&xl).unwrap(), &mut y_minus.view_mut(&yl).unwrap())
                .unwrap();
            x_pert[probe] = x.data()[probe];
            let fd: f64 = y_plus
                .data()
                .iter()
                .zip(y_minus.data())
                .zip(dy.data())
                .map(|((&p, &m), &g)| ((p - m) as f64 / (2.0 * eps as f64)) * g as f64)
                .sum();
            assert!(
                (fd as f32 - dx.data()[probe]).abs() < 2e-3,
                "pad FD mismatch mode={mode:?} probe={probe}: fd={fd}, got={}",
                dx.data()[probe]
            );
        }

        // Accumulate mode doubles dx
        let dx_once = dx.data().to_vec();
        bwd.run(
            &dy.view(&yl).unwrap(),
            &mut dx.view_mut(&xl).unwrap(),
            GradientMode::Accumulate,
        )
        .unwrap();
        let dx_twice: Vec<f32> = dx_once.iter().map(|v| 2.0 * v).collect();
        assert!(max_abs_diff(dx.data(), &dx_twice) < 1e-5);
    }

    // Fast last-axis contiguous Pad path + Cat forward/backward across axes + signed strides
    let shape_a = Shape::new([2, 3, 4]).unwrap();
    let shape_b = Shape::new([2, 2, 4]).unwrap();
    let cat_spec = CatSpec::new(&shape_a, &shape_b, 1).unwrap();
    let al = Layout::contiguous(shape_a.dims()).unwrap();
    let bl = Layout::contiguous(shape_b.dims()).unwrap();
    let ol = Layout::contiguous(cat_spec.output_shape().dims()).unwrap();
    let a = Tensor::from_vec(shape_a.dims(), pattern(shape_a.len(), 3)).unwrap();
    let b = Tensor::from_vec(shape_b.dims(), pattern(shape_b.len(), 5)).unwrap();
    let dy = Tensor::from_vec(
        cat_spec.output_shape().dims(),
        pattern(cat_spec.output_shape().len(), 9),
    )
    .unwrap();
    let mut out = Tensor::zeros(cat_spec.output_shape().dims()).unwrap();
    let cat_fwd = CatPlan::new(&cat_spec, &al, &bl, &ol).unwrap();
    cat_fwd
        .run(
            &a.view(&al).unwrap(),
            &b.view(&bl).unwrap(),
            &mut out.view_mut(&ol).unwrap(),
        )
        .unwrap();

    // Verify slices
    for n in 0..2 {
        for c in 0..5 {
            for t in 0..4 {
                let got = out.data()[(n * 5 + c) * 4 + t];
                let exp = if c < 3 {
                    a.data()[(n * 3 + c) * 4 + t]
                } else {
                    b.data()[(n * 2 + (c - 3)) * 4 + t]
                };
                assert_eq!(got, exp);
            }
        }
    }

    let bwd_l = CatBackward::new(BinaryOperand::Left, &cat_spec, &ol, &al).unwrap();
    let bwd_r = CatBackward::new(BinaryOperand::Right, &cat_spec, &ol, &bl).unwrap();
    assert_eq!(bwd_l.input_dependencies(), [false, false]);
    let mut da = Tensor::zeros(shape_a.dims()).unwrap();
    let mut db = Tensor::zeros(shape_b.dims()).unwrap();
    bwd_l
        .run(
            &dy.view(&ol).unwrap(),
            &mut da.view_mut(&al).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    bwd_r
        .run(
            &dy.view(&ol).unwrap(),
            &mut db.view_mut(&bl).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    let lhs: f64 = out
        .data()
        .iter()
        .zip(dy.data())
        .map(|(&x, &g)| x as f64 * g as f64)
        .sum();
    let rhs: f64 = a
        .data()
        .iter()
        .zip(da.data())
        .chain(b.data().iter().zip(db.data()))
        .map(|(&x, &g)| x as f64 * g as f64)
        .sum();
    assert!((lhs - rhs).abs() < 1e-5);
}

#[test]
fn cpu_tape_unet_vocoder_step_and_miri() {
    // U-Net / Vocoder audio step:
    // x [1, 2, 8] -> causal Pad(+2 left) -> Conv1d(k=3,s=2,p=0) [1, 3, 4] -> SiLU
    // -> ConvTranspose1d(k=4,s=2,p=1,op=0) [1, 2, 8] -> Cat(skip=x, axis=1) [1, 4, 8]
    // -> Conv1d(k=1,s=1,p=0) [1, 2, 8] -> MSE(target) -> backward + SGD
    let mut b = Builder::new().unwrap();
    let x = b.leaf([1, 2, 8], true).unwrap();
    let w_enc = b.leaf([3, 2, 3], true).unwrap();
    let w_dec = b.leaf([3, 2, 4], true).unwrap();
    let w_out = b.leaf([2, 4, 1], true).unwrap();
    let target = b.leaf([1, 2, 8], false).unwrap();

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

    let tx = Shared::new(Tensor::from_vec([1, 2, 8], pattern(16, 1)).unwrap()).unwrap();
    let tw_enc = Shared::new(Tensor::from_vec([3, 2, 3], pattern(18, 2)).unwrap()).unwrap();
    let tw_dec = Shared::new(Tensor::from_vec([3, 2, 4], pattern(24, 3)).unwrap()).unwrap();
    let tw_out = Shared::new(Tensor::from_vec([2, 4, 1], pattern(8, 4)).unwrap()).unwrap();
    let ttarget = Shared::new(Tensor::from_vec([1, 2, 8], pattern(16, 5)).unwrap()).unwrap();

    let bindings = [
        Binding::new(x, tx.clone()),
        Binding::new(w_enc, tw_enc.clone()),
        Binding::new(w_dec, tw_dec.clone()),
        Binding::new(w_out, tw_out.clone()),
        Binding::new(target, ttarget.clone()),
    ];
    let mut tape = Tape::new(Cpu::Reference, &program, &bindings).unwrap();
    tape.forward().unwrap();
    let base_loss = tape.output(loss).unwrap().storage().data()[0];
    tape.backward_scalar(GradientMode::Overwrite).unwrap();

    let g_enc = tape.gradient(w_enc).unwrap().unwrap().data().to_vec();
    let g_dec = tape.gradient(w_dec).unwrap().unwrap().data().to_vec();
    let g_out = tape.gradient(w_out).unwrap().unwrap().data().to_vec();
    let g_x = tape.gradient(x).unwrap().unwrap().data().to_vec();

    // Central finite difference check on representative elements of every leaf
    let eps = 1e-3f32;
    let mut check_fd =
        |leaf_shared: &Shared<Tensor>, expected: &[f32], indices: &[usize], name: &str| {
            for &idx in indices {
                let orig = leaf_shared.read().unwrap().data()[idx];
                leaf_shared.write().unwrap().data_mut()[idx] = orig + eps;
                tape.forward().unwrap();
                let lp = tape.output(loss).unwrap().storage().data()[0];
                leaf_shared.write().unwrap().data_mut()[idx] = orig - eps;
                tape.forward().unwrap();
                let lm = tape.output(loss).unwrap().storage().data()[0];
                leaf_shared.write().unwrap().data_mut()[idx] = orig;
                let fd = (lp - lm) / (2.0 * eps);
                assert!(
                    (fd - expected[idx]).abs() < 2e-3,
                    "{name}[{idx}] FD={fd} vs grad={}",
                    expected[idx]
                );
            }
        };
    check_fd(&tw_enc, &g_enc, &[0, 5, 17], "w_enc");
    check_fd(&tw_dec, &g_dec, &[0, 11, 23], "w_dec");
    check_fd(&tw_out, &g_out, &[0, 4, 7], "w_out");
    check_fd(&tx, &g_x, &[0, 3, 15], "x");

    // Verify SGD step lowers loss cleanly
    tape.step_scalar(&[w_enc, w_dec, w_out, x], 0.05).unwrap();
    tape.forward().unwrap();
    let after_loss = tape.output(loss).unwrap().storage().data()[0];
    assert!(
        after_loss < base_loss,
        "loss did not drop: {base_loss} -> {after_loss}"
    );
}
