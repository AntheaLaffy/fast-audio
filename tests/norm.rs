#![allow(clippy::needless_range_loop)]

#[allow(dead_code)]
mod support;

use fast_audio::{
    BinaryOp, GradientMode, Layout, NormKind, NormSpec, ReduceOp, ReduceSpec, Shape, Shared,
    SoftmaxKind, SoftmaxSpec, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{self, Cpu},
};

fn deterministic_data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..len)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u32 << 24) as f32;
            (u - 0.5) * 3.0
        })
        .collect()
}

fn fp64_norm_fwd(kind: NormKind, xs: &[f32], groups: usize, d: usize, eps: f32) -> Vec<f64> {
    let mut out = vec![0.0f64; groups * d];
    for g in 0..groups {
        let row = &xs[g * d..(g + 1) * d];
        match kind {
            NormKind::LayerNorm => {
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
                let var = row
                    .iter()
                    .map(|&v| {
                        let diff = v as f64 - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / d as f64;
                let rstd = 1.0 / (var + eps as f64).sqrt();
                for i in 0..d {
                    out[g * d + i] = (row[i] as f64 - mean) * rstd;
                }
            }
            NormKind::RmsNorm => {
                let ms = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
                let rstd = 1.0 / (ms + eps as f64).sqrt();
                for i in 0..d {
                    out[g * d + i] = (row[i] as f64) * rstd;
                }
            }
        }
    }
    out
}

fn fp64_norm_bwd(
    kind: NormKind,
    xs: &[f32],
    dys: &[f32],
    groups: usize,
    d: usize,
    eps: f32,
) -> Vec<f64> {
    let mut out = vec![0.0f64; groups * d];
    for g in 0..groups {
        let row = &xs[g * d..(g + 1) * d];
        let dy = &dys[g * d..(g + 1) * d];
        match kind {
            NormKind::LayerNorm => {
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
                let var = row
                    .iter()
                    .map(|&v| {
                        let diff = v as f64 - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / d as f64;
                let rstd = 1.0 / (var + eps as f64).sqrt();
                let s1 = dy.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
                let s2 = row
                    .iter()
                    .zip(dy)
                    .map(|(&x, &u)| (u as f64) * ((x as f64 - mean) * rstd))
                    .sum::<f64>()
                    / d as f64;
                for i in 0..d {
                    let xhat = (row[i] as f64 - mean) * rstd;
                    out[g * d + i] = rstd * (dy[i] as f64 - s1 - xhat * s2);
                }
            }
            NormKind::RmsNorm => {
                let ms = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
                let rstd = 1.0 / (ms + eps as f64).sqrt();
                let s2 = row
                    .iter()
                    .zip(dy)
                    .map(|(&x, &u)| (u as f64) * ((x as f64) * rstd))
                    .sum::<f64>()
                    / d as f64;
                for i in 0..d {
                    let xhat = (row[i] as f64) * rstd;
                    out[g * d + i] = rstd * (dy[i] as f64 - xhat * s2);
                }
            }
        }
    }
    out
}

fn fp64_softmax_fwd(kind: SoftmaxKind, xs: &[f32], groups: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; groups * d];
    for g in 0..groups {
        let row = &xs[g * d..(g + 1) * d];
        let m = row
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
        for i in 0..d {
            out[g * d + i] = match kind {
                SoftmaxKind::Softmax => (row[i] as f64 - m).exp() / sum,
                SoftmaxKind::LogSoftmax => (row[i] as f64 - m) - sum.ln(),
            };
        }
    }
    out
}

fn fp64_softmax_bwd(
    kind: SoftmaxKind,
    xs: &[f32],
    dys: &[f32],
    groups: usize,
    d: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; groups * d];
    for g in 0..groups {
        let row = &xs[g * d..(g + 1) * d];
        let dy = &dys[g * d..(g + 1) * d];
        let m = row
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
        let probs: Vec<f64> = row.iter().map(|&v| (v as f64 - m).exp() / sum).collect();
        match kind {
            SoftmaxKind::Softmax => {
                let dot: f64 = probs.iter().zip(dy).map(|(&p, &u)| p * (u as f64)).sum();
                for i in 0..d {
                    out[g * d + i] = probs[i] * (dy[i] as f64 - dot);
                }
            }
            SoftmaxKind::LogSoftmax => {
                let sum_dy: f64 = dy.iter().map(|&u| u as f64).sum();
                for i in 0..d {
                    out[g * d + i] = dy[i] as f64 - probs[i] * sum_dy;
                }
            }
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} len mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let err = (a as f64 - e).abs();
        assert!(
            err <= tol,
            "{label}[{i}] mismatch: actual={a}, expected={e}, err={err} > {tol}"
        );
    }
}

#[test]
fn cpu_norm_forward_backward_contiguous_and_strided_match_fp64_and_finite_diff() {
    let eps = 1e-5f32;
    for kind in [NormKind::LayerNorm, NormKind::RmsNorm] {
        for &(dims, norm_dims) in &[
            (&[4, 1][..], 1),
            (&[3, 7][..], 1),
            (&[5, 32][..], 1),
            (&[3, 128][..], 1),
            (&[2, 256][..], 1),
            (&[2, 3, 19][..], 2),
            (&[2, 1300][..], 1),
        ] {
            let shape = Shape::new(dims).unwrap();
            let layout = Layout::contiguous(dims).unwrap();
            let spec = NormSpec::new(kind, &shape, norm_dims, eps).unwrap();
            let fwd = cpu::NormPlan::new(&spec, &layout, &layout).unwrap();
            let bwd = cpu::NormBackward::new(&spec, &layout, &layout, &layout).unwrap();
            assert_eq!(fwd.workspace_bytes(), 0);
            assert_eq!(bwd.workspace_bytes(), 0);
            assert!(bwd.input_required());

            let x = Tensor::from_vec(dims, deterministic_data(shape.len(), 11)).unwrap();
            let dy = Tensor::from_vec(dims, deterministic_data(shape.len(), 29)).unwrap();
            let mut y = Tensor::zeros(dims).unwrap();
            let mut dx = Tensor::zeros(dims).unwrap();

            support::no_allocations(|| {
                fwd.run(&x.view(&layout).unwrap(), &mut y.view_mut(&layout).unwrap())
                    .unwrap();
                bwd.run(
                    &x.view(&layout).unwrap(),
                    &dy.view(&layout).unwrap(),
                    &mut dx.view_mut(&layout).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            });

            let ref_y = fp64_norm_fwd(kind, x.data(), spec.groups(), spec.inner_len(), eps);
            let ref_dx = fp64_norm_bwd(
                kind,
                x.data(),
                dy.data(),
                spec.groups(),
                spec.inner_len(),
                eps,
            );
            assert_close(y.data(), &ref_y, 2e-5, &format!("{kind:?} fwd {dims:?}"));
            assert_close(dx.data(), &ref_dx, 5e-5, &format!("{kind:?} bwd {dims:?}"));

            // Verify Accumulate mode adds cleanly onto existing values.
            let seed = deterministic_data(shape.len(), 43);
            let mut dx_acc = Tensor::from_vec(dims, seed.clone()).unwrap();
            bwd.run(
                &x.view(&layout).unwrap(),
                &dy.view(&layout).unwrap(),
                &mut dx_acc.view_mut(&layout).unwrap(),
                GradientMode::Accumulate,
            )
            .unwrap();
            let expected_acc: Vec<f64> = seed
                .iter()
                .zip(&ref_dx)
                .map(|(&s, &g)| s as f64 + g)
                .collect();
            assert_close(
                dx_acc.data(),
                &expected_acc,
                5e-5,
                &format!("{kind:?} bwd acc {dims:?}"),
            );
        }

        // Non-contiguous signed-stride (reversed + permuted) test + finite difference oracle.
        let raw_x = Tensor::from_vec([5, 4, 3], deterministic_data(60, 77)).unwrap();
        let raw_dy = Tensor::from_vec([5, 4, 3], deterministic_data(60, 91)).unwrap();
        let base = Layout::contiguous([5, 4, 3]).unwrap();
        let view_l = base
            .slice(0, 4, 3, -1)
            .unwrap()
            .permute(&[2, 0, 1])
            .unwrap(); // [3, 3, 4]
        let out_l = Layout::contiguous([3, 3, 4]).unwrap();
        let spec = NormSpec::new(kind, view_l.shape(), 2, eps).unwrap();
        let fwd = cpu::NormPlan::new(&spec, &view_l, &out_l).unwrap();
        let bwd = cpu::NormBackward::new(&spec, &view_l, &out_l, &out_l).unwrap();

        let mut y = Tensor::zeros([3, 3, 4]).unwrap();
        fwd.run(
            &raw_x.view(&view_l).unwrap(),
            &mut y.view_mut(&out_l).unwrap(),
        )
        .unwrap();
        let mut dy_dense = Tensor::zeros([3, 3, 4]).unwrap();
        cpu::CopyPlan::new(&view_l, &out_l)
            .unwrap()
            .run(
                &raw_dy.view(&view_l).unwrap(),
                &mut dy_dense.view_mut(&out_l).unwrap(),
            )
            .unwrap();
        let mut dx = Tensor::zeros([3, 3, 4]).unwrap();
        bwd.run(
            &raw_x.view(&view_l).unwrap(),
            &dy_dense.view(&out_l).unwrap(),
            &mut dx.view_mut(&out_l).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();

        let mut x_dense = Tensor::zeros([3, 3, 4]).unwrap();
        cpu::CopyPlan::new(&view_l, &out_l)
            .unwrap()
            .run(
                &raw_x.view(&view_l).unwrap(),
                &mut x_dense.view_mut(&out_l).unwrap(),
            )
            .unwrap();
        let ref_y = fp64_norm_fwd(kind, x_dense.data(), 3, 12, eps);
        let ref_dx = fp64_norm_bwd(kind, x_dense.data(), dy_dense.data(), 3, 12, eps);
        assert_close(y.data(), &ref_y, 2e-5, &format!("{kind:?} strided fwd"));
        assert_close(dx.data(), &ref_dx, 5e-5, &format!("{kind:?} strided bwd"));

        // Central finite-difference verification on [2, 5]
        let x0 = deterministic_data(10, 101);
        let dy0 = deterministic_data(10, 202);
        let anal = fp64_norm_bwd(kind, &x0, &dy0, 2, 5, eps);
        let h = 1e-4f64;
        for i in 0..10 {
            let mut xp = x0.iter().map(|&v| v as f64).collect::<Vec<_>>();
            let mut xm = xp.clone();
            xp[i] += h;
            xm[i] -= h;
            let eval = |vals: &[f64]| -> f64 {
                let mut total = 0.0;
                for g in 0..2 {
                    let row = &vals[g * 5..(g + 1) * 5];
                    let dy_r = &dy0[g * 5..(g + 1) * 5];
                    match kind {
                        NormKind::LayerNorm => {
                            let mean = row.iter().sum::<f64>() / 5.0;
                            let var =
                                row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / 5.0;
                            let rstd = 1.0 / (var + eps as f64).sqrt();
                            for j in 0..5 {
                                total += ((row[j] - mean) * rstd) * dy_r[j] as f64;
                            }
                        }
                        NormKind::RmsNorm => {
                            let ms = row.iter().map(|&v| v * v).sum::<f64>() / 5.0;
                            let rstd = 1.0 / (ms + eps as f64).sqrt();
                            for j in 0..5 {
                                total += (row[j] * rstd) * dy_r[j] as f64;
                            }
                        }
                    }
                }
                total
            };
            let fd = (eval(&xp) - eval(&xm)) / (2.0 * h);
            assert!(
                (anal[i] - fd).abs() < 1e-6,
                "{kind:?} fd[{i}]: anal={} fd={fd}",
                anal[i]
            );
        }
    }
}

#[test]
fn cpu_softmax_and_log_softmax_match_fp64_all_axes_and_strided() {
    for kind in [SoftmaxKind::Softmax, SoftmaxKind::LogSoftmax] {
        // Last axis across all length regimes
        for &(dims, axis) in &[
            (&[4, 1][..], 1),
            (&[3, 7][..], 1),
            (&[5, 32][..], 1),
            (&[3, 128][..], 1),
            (&[2, 256][..], 1),
            (&[2, 1300][..], 1),
        ] {
            let shape = Shape::new(dims).unwrap();
            let layout = Layout::contiguous(dims).unwrap();
            let spec = SoftmaxSpec::new(kind, &shape, axis).unwrap();
            let fwd = cpu::SoftmaxPlan::new(&spec, &layout, &layout).unwrap();
            let bwd = cpu::SoftmaxBackward::new(&spec, &layout, &layout, &layout).unwrap();
            assert_eq!(fwd.workspace_bytes(), 0);
            assert_eq!(bwd.workspace_bytes(), 0);
            assert!(bwd.input_required());

            let x = Tensor::from_vec(dims, deterministic_data(shape.len(), 31)).unwrap();
            let dy = Tensor::from_vec(dims, deterministic_data(shape.len(), 53)).unwrap();
            let mut y = Tensor::zeros(dims).unwrap();
            let mut dx = Tensor::zeros(dims).unwrap();
            let mut dx_from_y = Tensor::zeros(dims).unwrap();

            support::no_allocations(|| {
                fwd.run(&x.view(&layout).unwrap(), &mut y.view_mut(&layout).unwrap())
                    .unwrap();
                bwd.run(
                    &x.view(&layout).unwrap(),
                    &dy.view(&layout).unwrap(),
                    &mut dx.view_mut(&layout).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
                bwd.run_from_output(
                    &y.view(&layout).unwrap(),
                    &dy.view(&layout).unwrap(),
                    &mut dx_from_y.view_mut(&layout).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            });

            let ref_y = fp64_softmax_fwd(kind, x.data(), spec.groups(), spec.axis_len());
            let ref_dx =
                fp64_softmax_bwd(kind, x.data(), dy.data(), spec.groups(), spec.axis_len());
            assert_close(y.data(), &ref_y, 2e-5, &format!("{kind:?} fwd {dims:?}"));
            assert_close(dx.data(), &ref_dx, 2e-5, &format!("{kind:?} bwd {dims:?}"));
            assert_close(
                dx_from_y.data(),
                &ref_dx,
                2e-5,
                &format!("{kind:?} bwd run_from_output {dims:?}"),
            );
        }

        // Non-last axis (axis = 1 on [2, 5, 4]) + Accumulate + NaN propagation
        let dims = [2, 5, 4];
        let shape = Shape::new(dims).unwrap();
        let layout = Layout::contiguous(dims).unwrap();
        let spec = SoftmaxSpec::new(kind, &shape, 1).unwrap();
        let fwd = cpu::SoftmaxPlan::new(&spec, &layout, &layout).unwrap();
        let bwd = cpu::SoftmaxBackward::new(&spec, &layout, &layout, &layout).unwrap();
        let x = Tensor::from_vec(dims, deterministic_data(40, 71)).unwrap();
        let dy = Tensor::from_vec(dims, deterministic_data(40, 83)).unwrap();
        let mut y = Tensor::zeros(dims).unwrap();
        let mut dx = Tensor::from_vec(dims, vec![0.25; 40]).unwrap();
        fwd.run(&x.view(&layout).unwrap(), &mut y.view_mut(&layout).unwrap())
            .unwrap();
        bwd.run(
            &x.view(&layout).unwrap(),
            &dy.view(&layout).unwrap(),
            &mut dx.view_mut(&layout).unwrap(),
            GradientMode::Accumulate,
        )
        .unwrap();

        for b in 0..2 {
            for t in 0..4 {
                let row: Vec<f32> = (0..5).map(|c| x.data()[(b * 5 + c) * 4 + t]).collect();
                let u: Vec<f32> = (0..5).map(|c| dy.data()[(b * 5 + c) * 4 + t]).collect();
                let ry = fp64_softmax_fwd(kind, &row, 1, 5);
                let rdx = fp64_softmax_bwd(kind, &row, &u, 1, 5);
                for c in 0..5 {
                    let idx = (b * 5 + c) * 4 + t;
                    assert!((y.data()[idx] as f64 - ry[c]).abs() < 1e-5);
                    assert!((dx.data()[idx] as f64 - (0.25 + rdx[c])).abs() < 1e-5);
                }
            }
        }

        // NaN propagation
        let x_nan = Tensor::from_vec([2, 3], vec![1.0, f32::NAN, 2.0, -1.0, 0.0, 1.0]).unwrap();
        let dy_nan = Tensor::from_vec([2, 3], vec![1.0; 6]).unwrap();
        let l23 = Layout::contiguous([2, 3]).unwrap();
        let s23 = SoftmaxSpec::new(kind, l23.shape(), 1).unwrap();
        let mut y_nan = Tensor::zeros([2, 3]).unwrap();
        let mut dx_nan = Tensor::zeros([2, 3]).unwrap();
        cpu::SoftmaxPlan::new(&s23, &l23, &l23)
            .unwrap()
            .run(
                &x_nan.view(&l23).unwrap(),
                &mut y_nan.view_mut(&l23).unwrap(),
            )
            .unwrap();
        cpu::SoftmaxBackward::new(&s23, &l23, &l23, &l23)
            .unwrap()
            .run(
                &x_nan.view(&l23).unwrap(),
                &dy_nan.view(&l23).unwrap(),
                &mut dx_nan.view_mut(&l23).unwrap(),
                GradientMode::Overwrite,
            )
            .unwrap();
        assert!(y_nan.data()[..3].iter().all(|v| v.is_nan()));
        assert!(y_nan.data()[3..].iter().all(|v| v.is_finite()));
        assert!(dx_nan.data()[..3].iter().all(|v| v.is_nan()));
        assert!(dx_nan.data()[3..].iter().all(|v| v.is_finite()));
    }
}

#[test]
fn cpu_nonlinear_reductions_max_min_logsumexp_forward_backward_and_edges() {
    // Ties, exact FP64 values, Accumulate, and strided views
    let x = Tensor::from_vec([2, 4], vec![-1.0, 2.0, 2.0, 0.5, -3.0, -3.0, 1.0, -3.0]).unwrap();
    let dy = Tensor::from_vec([2], vec![3.0, -6.0]).unwrap();
    let lx = Layout::contiguous([2, 4]).unwrap();
    let ly = Layout::contiguous([2]).unwrap();

    // Max with ties
    let smax = ReduceSpec::new(ReduceOp::Max, lx.shape(), &[1], false).unwrap();
    assert!(cpu::ReduceBackward::new(&smax, &ly, &lx).is_err());
    let fwd = cpu::ReducePlan::new(&smax, &lx, &ly).unwrap();
    let bwd = cpu::ReduceBackward::with_input(&smax, &lx, &ly, &lx).unwrap();
    assert!(bwd.input_required());
    let mut y = Tensor::zeros([2]).unwrap();
    let mut dx = Tensor::zeros([2, 4]).unwrap();
    support::no_allocations(|| {
        fwd.run(&x.view(&lx).unwrap(), &mut y.view_mut(&ly).unwrap())
            .unwrap();
        bwd.run_with_input(
            Some(&x.view(&lx).unwrap()),
            &dy.view(&ly).unwrap(),
            &mut dx.view_mut(&lx).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    });
    assert_eq!(y.data(), &[2.0, 1.0]);
    assert_eq!(dx.data(), &[0.0, 1.5, 1.5, 0.0, 0.0, 0.0, -6.0, 0.0]);

    // Min with 3-way tie
    let smin = ReduceSpec::new(ReduceOp::Min, lx.shape(), &[1], false).unwrap();
    let mut y_min = Tensor::zeros([2]).unwrap();
    let mut dx_min = Tensor::zeros([2, 4]).unwrap();
    cpu::ReducePlan::new(&smin, &lx, &ly)
        .unwrap()
        .run(&x.view(&lx).unwrap(), &mut y_min.view_mut(&ly).unwrap())
        .unwrap();
    cpu::ReduceBackward::with_input(&smin, &lx, &ly, &lx)
        .unwrap()
        .run_with_input(
            Some(&x.view(&lx).unwrap()),
            &dy.view(&ly).unwrap(),
            &mut dx_min.view_mut(&lx).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    assert_eq!(y_min.data(), &[-1.0, -3.0]);
    assert_eq!(dx_min.data(), &[3.0, 0.0, 0.0, 0.0, -2.0, -2.0, 0.0, -2.0]);

    // LogSumExp
    let slse = ReduceSpec::new(ReduceOp::LogSumExp, lx.shape(), &[1], false).unwrap();
    let mut y_lse = Tensor::zeros([2]).unwrap();
    let mut dx_lse = Tensor::zeros([2, 4]).unwrap();
    cpu::ReducePlan::new(&slse, &lx, &ly)
        .unwrap()
        .run(&x.view(&lx).unwrap(), &mut y_lse.view_mut(&ly).unwrap())
        .unwrap();
    cpu::ReduceBackward::with_input(&slse, &lx, &ly, &lx)
        .unwrap()
        .run_with_input(
            Some(&x.view(&lx).unwrap()),
            &dy.view(&ly).unwrap(),
            &mut dx_lse.view_mut(&lx).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    for g in 0..2 {
        let row = &x.data()[g * 4..(g + 1) * 4];
        let m = row
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
        assert!((y_lse.data()[g] as f64 - (m + s.ln())).abs() < 1e-5);
        for i in 0..4 {
            let expected = dy.data()[g] as f64 * (row[i] as f64 - m).exp() / s;
            assert!((dx_lse.data()[g * 4 + i] as f64 - expected).abs() < 1e-5);
        }
    }

    // NaN / +/-inf edge parity with PyTorch
    let edge_x = Tensor::from_vec(
        [3, 2],
        vec![
            f32::NAN,
            1.0,
            f32::INFINITY,
            1.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
    )
    .unwrap();
    let edge_dy = Tensor::from_vec([3], vec![3.0, 3.0, 3.0]).unwrap();
    let l32 = Layout::contiguous([3, 2]).unwrap();
    let l3 = Layout::contiguous([3]).unwrap();
    let s_max = ReduceSpec::new(ReduceOp::Max, l32.shape(), &[1], false).unwrap();
    let mut y_e = Tensor::zeros([3]).unwrap();
    let mut dx_e = Tensor::zeros([3, 2]).unwrap();
    cpu::ReducePlan::new(&s_max, &l32, &l3)
        .unwrap()
        .run(&edge_x.view(&l32).unwrap(), &mut y_e.view_mut(&l3).unwrap())
        .unwrap();
    cpu::ReduceBackward::with_input(&s_max, &l32, &l3, &l32)
        .unwrap()
        .run_with_input(
            Some(&edge_x.view(&l32).unwrap()),
            &edge_dy.view(&l3).unwrap(),
            &mut dx_e.view_mut(&l32).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    assert!(y_e.data()[0].is_nan());
    assert!(dx_e.data()[0].is_nan() && dx_e.data()[1].is_nan());
    assert_eq!(dx_e.data()[2..], [3.0, 0.0, 1.5, 1.5]);

    let s_lse = ReduceSpec::new(ReduceOp::LogSumExp, l32.shape(), &[1], false).unwrap();
    let mut dx_lse_e = Tensor::zeros([3, 2]).unwrap();
    cpu::ReduceBackward::with_input(&s_lse, &l32, &l3, &l32)
        .unwrap()
        .run_with_input(
            Some(&edge_x.view(&l32).unwrap()),
            &edge_dy.view(&l3).unwrap(),
            &mut dx_lse_e.view_mut(&l32).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    assert!(dx_lse_e.data()[0].is_nan() && dx_lse_e.data()[1].is_nan());
    assert!(dx_lse_e.data()[2].is_nan() && dx_lse_e.data()[3] == 0.0);
    assert!(dx_lse_e.data()[4].is_nan() && dx_lse_e.data()[5].is_nan());
}

#[test]
fn cpu_tape_composes_norm_group_norm_softmax_logsumexp_with_fusion_and_sgd() {
    let cpu = Cpu::Reference;
    let x_init = Tensor::from_vec([2, 4, 8], deterministic_data(64, 5)).unwrap();
    let g_init = Tensor::from_vec([1, 4, 8], deterministic_data(32, 13)).unwrap();
    let b_init = Tensor::from_vec([1, 4, 8], deterministic_data(32, 19)).unwrap();
    let target = Tensor::from_vec([2, 4, 8], deterministic_data(64, 23)).unwrap();

    let mut b = Builder::new().unwrap();
    let x = b.leaf([2, 4, 8], true).unwrap();
    let gamma = b.leaf([1, 4, 8], true).unwrap();
    let beta = b.leaf([1, 4, 8], true).unwrap();
    let tgt = b.leaf([2, 4, 8], false).unwrap();

    let gn = b.group_norm(x, 2, 1e-5).unwrap();
    let scaled = b.binary(BinaryOp::Mul, gn, gamma).unwrap();
    let shifted = b.binary(BinaryOp::Add, scaled, beta).unwrap();
    let rn = b.rms_norm(shifted, 1, 1e-5).unwrap();
    let sm = b.softmax(rn, 2).unwrap();
    let lsm = b.log_softmax(shifted, 2).unwrap();
    let combo = b.binary(BinaryOp::Add, sm, lsm).unwrap();
    let lse = b.reduce(ReduceOp::LogSumExp, combo, &[2], true).unwrap();
    let diff = b.binary(BinaryOp::Sub, combo, tgt).unwrap();
    let diff2 = b.binary(BinaryOp::Add, diff, lse).unwrap();
    let sq = b.unary(UnaryOp::Square, diff2).unwrap();
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1, 2], false).unwrap();
    let program = b.finish(&[loss]).unwrap();

    let make_tape = || {
        let sx = Shared::new(x_init.clone()).unwrap();
        let sg = Shared::new(g_init.clone()).unwrap();
        let sb = Shared::new(b_init.clone()).unwrap();
        let st = Shared::new(target.clone()).unwrap();
        let tape = cpu::Tape::new(
            cpu,
            &program,
            &[
                Binding::new(x, sx.clone()),
                Binding::new(gamma, sg.clone()),
                Binding::new(beta, sb.clone()),
                Binding::new(tgt, st.clone()),
            ],
        )
        .unwrap();
        (tape, sx, sg, sb)
    };

    let (mut tape_fused, fx, fg, fb) = make_tape();
    let (mut tape_unfused, ux, ug, ub) = make_tape();

    for _ in 0..4 {
        tape_fused.step_scalar(&[x, gamma, beta], 0.05).unwrap();
        tape_unfused.forward().unwrap();
        tape_unfused
            .backward_scalar(GradientMode::Overwrite)
            .unwrap();
        tape_unfused.sgd(&[x, gamma, beta], 0.05).unwrap();
    }

    for (name, a, c) in [
        ("x", fx.read().unwrap(), ux.read().unwrap()),
        ("gamma", fg.read().unwrap(), ug.read().unwrap()),
        ("beta", fb.read().unwrap(), ub.read().unwrap()),
    ] {
        for (i, (&va, &vc)) in a.data().iter().zip(c.data()).enumerate() {
            assert!(
                (va - vc).abs() < 1e-5,
                "multi-branch step vs unfused {name}[{i}]: {va} vs {vc}"
            );
        }
    }

    // Verify automatic post-Norm / post-Softmax tail fusion (single barrier -> affine + MSE tail).
    let mut fb2 = Builder::new().unwrap();
    let fx2 = fb2.leaf([2, 4, 8], true).unwrap();
    let fg2 = fb2.leaf([1, 4, 8], true).unwrap();
    let ft2 = fb2.leaf([2, 4, 8], false).unwrap();
    let ln2 = fb2.layer_norm(fx2, 1, 1e-5).unwrap();
    let sm2 = fb2.softmax(ln2, 2).unwrap();
    let sc2 = fb2.binary(BinaryOp::Mul, sm2, fg2).unwrap();
    let df2 = fb2.binary(BinaryOp::Sub, sc2, ft2).unwrap();
    let sq2 = fb2.unary(UnaryOp::Square, df2).unwrap();
    let loss2 = fb2.reduce(ReduceOp::Mean, sq2, &[0, 1, 2], false).unwrap();
    let prog2 = fb2.finish(&[loss2]).unwrap();

    let s_fx = Shared::new(x_init.clone()).unwrap();
    let s_fg = Shared::new(g_init.clone()).unwrap();
    let s_ft = Shared::new(target.clone()).unwrap();
    let mut t_fused2 = cpu::Tape::new(
        cpu,
        &prog2,
        &[
            Binding::new(fx2, s_fx.clone()),
            Binding::new(fg2, s_fg.clone()),
            Binding::new(ft2, s_ft.clone()),
        ],
    )
    .unwrap();
    let s_ux = Shared::new(x_init).unwrap();
    let s_ug = Shared::new(g_init).unwrap();
    let s_ut = Shared::new(target).unwrap();
    let mut t_unfused2 = cpu::Tape::new(
        cpu,
        &prog2,
        &[
            Binding::new(fx2, s_ux.clone()),
            Binding::new(fg2, s_ug.clone()),
            Binding::new(ft2, s_ut.clone()),
        ],
    )
    .unwrap();
    for _ in 0..4 {
        t_fused2.step_scalar(&[fx2, fg2], 0.05).unwrap();
        t_unfused2.forward().unwrap();
        t_unfused2.backward_scalar(GradientMode::Overwrite).unwrap();
        t_unfused2.sgd(&[fx2, fg2], 0.05).unwrap();
    }
    assert!(t_fused2.has_fused_plan());
    for (i, (&va, &vc)) in s_fx
        .read()
        .unwrap()
        .data()
        .iter()
        .zip(s_ux.read().unwrap().data())
        .enumerate()
    {
        assert!((va - vc).abs() < 1e-5, "fused tail x[{i}]: {va} vs {vc}");
    }
    for (i, (&va, &vc)) in s_fg
        .read()
        .unwrap()
        .data()
        .iter()
        .zip(s_ug.read().unwrap().data())
        .enumerate()
    {
        assert!(
            (va - vc).abs() < 1e-5,
            "fused tail gamma[{i}]: {va} vs {vc}"
        );
    }
}

#[test]
fn norm_softmax_nl_reduce_miri_strided_and_tape() {
    let raw = Tensor::from_vec([4, 3], deterministic_data(12, 313)).unwrap();
    let base = Layout::contiguous([4, 3]).unwrap();
    let view = base.slice(0, 3, 2, -1).unwrap().permute(&[1, 0]).unwrap(); // [3, 2]
    let out = Layout::contiguous([3, 2]).unwrap();
    let nspec = NormSpec::new(NormKind::LayerNorm, view.shape(), 1, 1e-5).unwrap();
    let mut y = Tensor::zeros([3, 2]).unwrap();
    cpu::NormPlan::new(&nspec, &view, &out)
        .unwrap()
        .run(&raw.view(&view).unwrap(), &mut y.view_mut(&out).unwrap())
        .unwrap();
    let sspec = SoftmaxSpec::new(SoftmaxKind::LogSoftmax, out.shape(), 1).unwrap();
    let mut z = Tensor::zeros([3, 2]).unwrap();
    cpu::SoftmaxPlan::new(&sspec, &out, &out)
        .unwrap()
        .run(&y.view(&out).unwrap(), &mut z.view_mut(&out).unwrap())
        .unwrap();
    let rspec = ReduceSpec::new(ReduceOp::LogSumExp, out.shape(), &[1], false).unwrap();
    let r_out = Layout::contiguous([3]).unwrap();
    let mut r = Tensor::zeros([3]).unwrap();
    cpu::ReducePlan::new(&rspec, &out, &r_out)
        .unwrap()
        .run(&z.view(&out).unwrap(), &mut r.view_mut(&r_out).unwrap())
        .unwrap();
    let mut dx = Tensor::zeros([3, 2]).unwrap();
    let dy = Tensor::from_vec([3], vec![1.0, -0.5, 0.25]).unwrap();
    cpu::ReduceBackward::with_input(&rspec, &out, &r_out, &out)
        .unwrap()
        .run_with_input(
            Some(&z.view(&out).unwrap()),
            &dy.view(&r_out).unwrap(),
            &mut dx.view_mut(&out).unwrap(),
            GradientMode::Overwrite,
        )
        .unwrap();
    assert!(dx.data().iter().all(|v| v.is_finite()));

    let mut b = Builder::new().unwrap();
    let lx = b.leaf([2, 3], true).unwrap();
    let ln = b.layer_norm(lx, 1, 1e-5).unwrap();
    let sm = b.softmax(ln, 1).unwrap();
    let mx = b.reduce(ReduceOp::Max, sm, &[1], false).unwrap();
    let loss = b.reduce(ReduceOp::Sum, mx, &[0], false).unwrap();
    let prog = b.finish(&[loss]).unwrap();
    let sx = Shared::new(Tensor::from_vec([2, 3], vec![0.1, 0.8, -0.3, 0.4, 0.4, -0.2]).unwrap())
        .unwrap();
    let mut tape = cpu::Tape::new(Cpu::Reference, &prog, &[Binding::new(lx, sx.clone())]).unwrap();
    tape.step_scalar(&[lx], 0.1).unwrap();
    assert!(sx.read().unwrap().data().iter().all(|v| v.is_finite()));
}
