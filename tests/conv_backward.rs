use fast_audio::{
    BinaryOp, BinaryOperand as Side, Conv1dSpec, ConvStrategy as Route, GradientMode as G, Layout,
    ReduceOp, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{self, Cpu},
};
#[path = "support/conv.rs"]
mod contract;
#[allow(
    dead_code,
    reason = "Convolution fixtures use their own independent numerical oracle."
)]
mod support;
use contract::*;

fn resources(
    direct: bool,
    rank: usize,
    cpu: Cpu,
    route: Route,
    workspace: usize,
    metadata: usize,
    backward: bool,
) {
    let expected_meta = if backward { 4 } else { 3 } * rank * 16;
    assert_eq!(metadata, expected_meta);
    assert!(workspace <= 57_344);
    if route == Route::PackedMkl {
        assert!(workspace > 0);
    } else {
        assert_eq!(workspace, 0);
    }
    if direct {
        assert_eq!(
            route,
            match cpu {
                Cpu::Reference => Route::Reference,
                #[cfg(feature = "mkl")]
                Cpu::Mkl { .. } => Route::Mkl,
            }
        );
    }
}

fn exercise_1d(cpu: Cpu, c: &Case1d, salt: usize, export: &mut Exporter) {
    let mut forward_plan = cpu::Conv1dPlan::new(cpu, c.spec, &c.x, &c.w, &c.y).unwrap();
    assert_eq!(forward_plan.spec(), c.spec);
    assert_eq!(forward_plan.input_layout(), &c.x);
    assert_eq!(forward_plan.weight_layout(), &c.w);
    assert_eq!(forward_plan.destination_layout(), &c.y);
    resources(
        c.direct,
        3,
        cpu,
        forward_plan.strategy(),
        forward_plan.workspace_bytes(),
        forward_plan.host_metadata_bytes(),
        false,
    );
    let mut backward = SIDES.map(|side| {
        cpu::Conv1dBackward::new(cpu, side, c.spec, &c.x, &c.w, &c.dy, c.gradient(side)).unwrap()
    });
    for (p, side) in backward.iter().zip(SIDES) {
        assert_eq!(p.spec(), c.spec);
        assert_eq!(p.operand(), side);
        assert_eq!(p.input_layout(), &c.x);
        assert_eq!(p.weight_layout(), &c.w);
        assert_eq!(p.upstream_layout(), &c.dy);
        assert_eq!(p.destination_layout(), c.gradient(side));
        assert_eq!(p.input_dependencies(), c.dependencies(side));
        resources(
            c.direct,
            3,
            cpu,
            p.strategy(),
            p.workspace_bytes(),
            p.host_metadata_bytes(),
            true,
        );
    }
    for change in 0..2 {
        let x = buffer(&c.x, salt + change);
        let w = buffer(&c.w, salt + change + 1);
        let dy = buffer(&c.dy, salt + change + 2);
        let original = (bits(&x), bits(&w), bits(&dy));
        let xx = logical(&c.x, &x);
        let ww = logical(&c.w, &w);
        let dd = logical(&c.dy, &dy);
        let initial = seed(&c.y, false);
        let mut y = initial.clone();
        support::no_allocations(|| {
            forward_plan
                .run(
                    &x.view(&c.x).unwrap(),
                    &w.view(&c.w).unwrap(),
                    &mut y.view_mut(&c.y).unwrap(),
                )
                .unwrap()
        });
        verify(
            &c.y,
            &forward_1d(c.spec, &xx, &ww),
            &initial,
            &y,
            G::Overwrite,
        );
        let first = bits(&y);
        forward_plan
            .run(
                &x.view(&c.x).unwrap(),
                &w.view(&c.w).unwrap(),
                &mut y.view_mut(&c.y).unwrap(),
            )
            .unwrap();
        assert_eq!(bits(&y), first);
        let xv = x.view(&c.x).unwrap();
        let wv = w.view(&c.w).unwrap();
        let mut results: [[Option<Tensor>; 2]; 2] = [[None, None], [None, None]];
        for (side_index, (p, side)) in backward.iter_mut().zip(SIDES).enumerate() {
            let grad = c.gradient(side);
            let expected = derivative_1d(c.spec, side, &xx, &ww, &dd);
            let required = p.input_dependencies();
            let inputs = [
                if required[0] || change == 1 {
                    Some(&xv)
                } else {
                    None
                },
                if required[1] || change == 1 {
                    Some(&wv)
                } else {
                    None
                },
            ];
            for (mode_index, mode) in [G::Overwrite, G::Accumulate].into_iter().enumerate() {
                let initial = seed(grad, mode == G::Accumulate);
                let mut actual = initial.clone();
                support::no_allocations(|| {
                    p.run(
                        inputs[0],
                        inputs[1],
                        &dy.view(&c.dy).unwrap(),
                        &mut actual.view_mut(grad).unwrap(),
                        mode,
                    )
                    .unwrap()
                });
                verify(grad, &expected, &initial, &actual, mode);
                let mut repeat = initial.clone();
                p.run(
                    inputs[0],
                    inputs[1],
                    &dy.view(&c.dy).unwrap(),
                    &mut repeat.view_mut(grad).unwrap(),
                    mode,
                )
                .unwrap();
                assert_eq!(bits(&repeat), bits(&actual));
                if export.enabled() {
                    results[mode_index][side_index] = Some(actual.clone());
                }
                if mode == G::Overwrite {
                    identities_1d(c, side, &x, &w, &y, &dy, &actual);
                } else {
                    let before = actual.clone();
                    p.run(
                        inputs[0],
                        inputs[1],
                        &dy.view(&c.dy).unwrap(),
                        &mut actual.view_mut(grad).unwrap(),
                        mode,
                    )
                    .unwrap();
                    verify(grad, &expected, &before, &actual, mode);
                }
            }
        }
        if export.enabled() {
            let backend = match cpu {
                Cpu::Reference => "reference",
                #[cfg(feature = "mkl")]
                Cpu::Mkl { .. } => "mkl",
            };
            export.conv1d(
                backend,
                salt,
                change,
                c,
                [&x, &w, &dy],
                &y,
                [
                    [
                        results[0][0].as_ref().unwrap(),
                        results[0][1].as_ref().unwrap(),
                    ],
                    [
                        results[1][0].as_ref().unwrap(),
                        results[1][1].as_ref().unwrap(),
                    ],
                ],
            );
        }
        assert_eq!((bits(&x), bits(&w), bits(&dy)), original);
    }
}

fn exercise_2d(cpu: Cpu, c: &Case2d, salt: usize, export: &mut Exporter) {
    let mut forward_plan = cpu::Conv2dPlan::new(cpu, c.spec, &c.x, &c.w, &c.y).unwrap();
    assert_eq!(forward_plan.spec(), c.spec);
    assert_eq!(forward_plan.input_layout(), &c.x);
    assert_eq!(forward_plan.weight_layout(), &c.w);
    assert_eq!(forward_plan.destination_layout(), &c.y);
    resources(
        c.direct,
        4,
        cpu,
        forward_plan.strategy(),
        forward_plan.workspace_bytes(),
        forward_plan.host_metadata_bytes(),
        false,
    );
    let mut backward = SIDES.map(|side| {
        cpu::Conv2dBackward::new(cpu, side, c.spec, &c.x, &c.w, &c.dy, c.gradient(side)).unwrap()
    });
    for (p, side) in backward.iter().zip(SIDES) {
        assert_eq!(p.spec(), c.spec);
        assert_eq!(p.operand(), side);
        assert_eq!(p.input_layout(), &c.x);
        assert_eq!(p.weight_layout(), &c.w);
        assert_eq!(p.upstream_layout(), &c.dy);
        assert_eq!(p.destination_layout(), c.gradient(side));
        assert_eq!(p.input_dependencies(), c.dependencies(side));
        resources(
            c.direct,
            4,
            cpu,
            p.strategy(),
            p.workspace_bytes(),
            p.host_metadata_bytes(),
            true,
        );
    }
    for change in 0..2 {
        let x = buffer(&c.x, salt + change);
        let w = buffer(&c.w, salt + change + 1);
        let dy = buffer(&c.dy, salt + change + 2);
        let original = (bits(&x), bits(&w), bits(&dy));
        let xx = logical(&c.x, &x);
        let ww = logical(&c.w, &w);
        let dd = logical(&c.dy, &dy);
        let initial = seed(&c.y, false);
        let mut y = initial.clone();
        support::no_allocations(|| {
            forward_plan
                .run(
                    &x.view(&c.x).unwrap(),
                    &w.view(&c.w).unwrap(),
                    &mut y.view_mut(&c.y).unwrap(),
                )
                .unwrap()
        });
        verify(
            &c.y,
            &forward_2d(c.spec, &xx, &ww),
            &initial,
            &y,
            G::Overwrite,
        );
        let first = bits(&y);
        forward_plan
            .run(
                &x.view(&c.x).unwrap(),
                &w.view(&c.w).unwrap(),
                &mut y.view_mut(&c.y).unwrap(),
            )
            .unwrap();
        assert_eq!(bits(&y), first);
        let xv = x.view(&c.x).unwrap();
        let wv = w.view(&c.w).unwrap();
        let mut results: [[Option<Tensor>; 2]; 2] = [[None, None], [None, None]];
        for (side_index, (p, side)) in backward.iter_mut().zip(SIDES).enumerate() {
            let grad = c.gradient(side);
            let expected = derivative_2d(c.spec, side, &xx, &ww, &dd);
            let required = p.input_dependencies();
            let inputs = [
                if required[0] || change == 1 {
                    Some(&xv)
                } else {
                    None
                },
                if required[1] || change == 1 {
                    Some(&wv)
                } else {
                    None
                },
            ];
            for (mode_index, mode) in [G::Overwrite, G::Accumulate].into_iter().enumerate() {
                let initial = seed(grad, mode == G::Accumulate);
                let mut actual = initial.clone();
                support::no_allocations(|| {
                    p.run(
                        inputs[0],
                        inputs[1],
                        &dy.view(&c.dy).unwrap(),
                        &mut actual.view_mut(grad).unwrap(),
                        mode,
                    )
                    .unwrap()
                });
                verify(grad, &expected, &initial, &actual, mode);
                let mut repeat = initial.clone();
                p.run(
                    inputs[0],
                    inputs[1],
                    &dy.view(&c.dy).unwrap(),
                    &mut repeat.view_mut(grad).unwrap(),
                    mode,
                )
                .unwrap();
                assert_eq!(bits(&repeat), bits(&actual));
                if export.enabled() {
                    results[mode_index][side_index] = Some(actual.clone());
                }
                if mode == G::Overwrite {
                    identities_2d(c, side, &x, &w, &y, &dy, &actual);
                } else {
                    let before = actual.clone();
                    p.run(
                        inputs[0],
                        inputs[1],
                        &dy.view(&c.dy).unwrap(),
                        &mut actual.view_mut(grad).unwrap(),
                        mode,
                    )
                    .unwrap();
                    verify(grad, &expected, &before, &actual, mode);
                }
            }
        }
        if export.enabled() {
            let backend = match cpu {
                Cpu::Reference => "reference",
                #[cfg(feature = "mkl")]
                Cpu::Mkl { .. } => "mkl",
            };
            export.conv2d(
                backend,
                salt,
                change,
                c,
                [&x, &w, &dy],
                &y,
                [
                    [
                        results[0][0].as_ref().unwrap(),
                        results[0][1].as_ref().unwrap(),
                    ],
                    [
                        results[1][0].as_ref().unwrap(),
                        results[1][1].as_ref().unwrap(),
                    ],
                ],
            );
        }
        assert_eq!((bits(&x), bits(&w), bits(&dy)), original);
    }
}

#[test]
fn cpu_prepared_conv_and_vjps_match_independent_oracles() {
    let mut export = Exporter::new(Some("conv-cpu"));
    for cpu in support::backends() {
        for (i, c) in cases_1d().iter().enumerate() {
            exercise_1d(cpu, c, i, &mut export);
        }
        for (i, c) in cases_2d().iter().enumerate() {
            exercise_2d(cpu, c, i, &mut export);
        }
    }
    export.finish();
}

fn shifted(l: &Layout) -> Layout {
    Layout::from_strides(l.shape().dims(), l.strides(), l.offset() + 1).unwrap()
}

#[test]
fn cpu_prepared_conv_rejects_changed_geometry_and_missing_dependencies() {
    for cpu in support::backends() {
        for c in &cases_1d()[..8] {
            let x = buffer(&c.x, 0);
            let w = buffer(&c.w, 1);
            let dy = buffer(&c.dy, 2);
            let initial = seed(&c.y, false);
            let mut out = initial.clone();
            let mut p = cpu::Conv1dPlan::new(cpu, c.spec, &c.x, &c.w, &c.y).unwrap();
            assert!(
                p.run(
                    &x.view(&shifted(&c.x)).unwrap(),
                    &w.view(&c.w).unwrap(),
                    &mut out.view_mut(&c.y).unwrap()
                )
                .is_err()
            );
            assert!(
                p.run(
                    &x.view(&c.x).unwrap(),
                    &w.view(&shifted(&c.w)).unwrap(),
                    &mut out.view_mut(&c.y).unwrap()
                )
                .is_err()
            );
            assert!(
                p.run(
                    &x.view(&c.x).unwrap(),
                    &w.view(&c.w).unwrap(),
                    &mut out.view_mut(&shifted(&c.y)).unwrap()
                )
                .is_err()
            );
            assert_eq!(bits(&out), bits(&initial));
            for side in SIDES {
                let grad = c.gradient(side);
                let initial = seed(grad, true);
                let mut out = initial.clone();
                let mut p =
                    cpu::Conv1dBackward::new(cpu, side, c.spec, &c.x, &c.w, &c.dy, grad).unwrap();
                let xv = x.view(&c.x).unwrap();
                let wv = w.view(&c.w).unwrap();
                for i in 0..2 {
                    if p.input_dependencies()[i] {
                        let mut inputs = [Some(&xv), Some(&wv)];
                        inputs[i] = None;
                        assert!(
                            p.run(
                                inputs[0],
                                inputs[1],
                                &dy.view(&c.dy).unwrap(),
                                &mut out.view_mut(grad).unwrap(),
                                G::Overwrite
                            )
                            .is_err()
                        );
                    }
                }
                assert!(
                    p.run(
                        Some(&x.view(&shifted(&c.x)).unwrap()),
                        Some(&wv),
                        &dy.view(&c.dy).unwrap(),
                        &mut out.view_mut(grad).unwrap(),
                        G::Accumulate
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&xv),
                        Some(&w.view(&shifted(&c.w)).unwrap()),
                        &dy.view(&c.dy).unwrap(),
                        &mut out.view_mut(grad).unwrap(),
                        G::Overwrite
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&xv),
                        Some(&wv),
                        &dy.view(&shifted(&c.dy)).unwrap(),
                        &mut out.view_mut(grad).unwrap(),
                        G::Overwrite
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&xv),
                        Some(&wv),
                        &dy.view(&c.dy).unwrap(),
                        &mut out.view_mut(&shifted(grad)).unwrap(),
                        G::Accumulate
                    )
                    .is_err()
                );
                assert_eq!(bits(&out), bits(&initial));
            }
        }
        let s1 = SPECS_1D[2];
        let x1 = Layout::contiguous(s1.input_shape()).unwrap();
        let w1 = Layout::contiguous(s1.weight_shape()).unwrap();
        let y1 = Layout::contiguous(s1.output_shape().unwrap()).unwrap();
        let overlap1 = Layout::from_strides(s1.output_shape().unwrap(), [0, 1, 1], 0).unwrap();
        assert!(cpu::Conv1dPlan::new(cpu, s1, &x1, &w1, &overlap1).is_err());
        assert!(cpu::Conv1dBackward::new(cpu, Side::Left, s1, &x1, &w1, &y1, &overlap1).is_err());

        let mut transposed = SPECS_2D[2];
        transposed.transposed = true;
        let x2 = Layout::contiguous(SPECS_2D[2].input_shape()).unwrap();
        let w2 = Layout::contiguous(SPECS_2D[2].weight_shape()).unwrap();
        let y2 = Layout::contiguous(SPECS_2D[2].output_shape().unwrap()).unwrap();
        assert!(cpu::Conv2dPlan::new(cpu, transposed, &x2, &w2, &y2).is_err());
        assert!(cpu::Conv2dBackward::new(cpu, Side::Left, transposed, &x2, &w2, &y2, &x2).is_err());
    }
}

#[test]
fn cpu_prepared_conv_budgets_saved_versions_view_composition_and_threads() {
    for cpu in support::backends() {
        let c = &cases_1d()[16];
        let p = cpu::Conv1dPlan::new(cpu, c.spec, &c.x, &c.w, &c.y).unwrap();
        assert!(p.workspace_bytes() <= 57_344);
        if p.strategy() == Route::PackedMkl {
            assert!(
                cpu::Conv1dPlan::with_workspace_limit(
                    cpu,
                    c.spec,
                    &c.x,
                    &c.w,
                    &c.y,
                    p.workspace_bytes() - 1
                )
                .is_err()
            );
            for side in SIDES {
                let grad = c.gradient(side);
                let bp =
                    cpu::Conv1dBackward::new(cpu, side, c.spec, &c.x, &c.w, &c.dy, grad).unwrap();
                assert!(
                    cpu::Conv1dBackward::with_workspace_limit(
                        cpu,
                        side,
                        c.spec,
                        &c.x,
                        &c.w,
                        &c.dy,
                        grad,
                        bp.workspace_bytes() - 1
                    )
                    .is_err()
                );
            }
        }
        let xx = buffer(&c.x, 0);
        let ww = buffer(&c.w, 1);
        let expected = forward_1d(c.spec, &logical(&c.x, &xx), &logical(&c.w, &ww));
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let (c, xx, ww, expected) = (c, &xx, &ww, &expected);
                scope.spawn(move || {
                    let mut p = cpu::Conv1dPlan::new(cpu, c.spec, &c.x, &c.w, &c.y).unwrap();
                    let initial = seed(&c.y, false);
                    let mut out = initial.clone();
                    support::no_allocations(|| {
                        p.run(
                            &xx.view(&c.x).unwrap(),
                            &ww.view(&c.w).unwrap(),
                            &mut out.view_mut(&c.y).unwrap(),
                        )
                        .unwrap()
                    });
                    verify(&c.y, expected, &initial, &out, G::Overwrite);
                });
            }
        });

        let spec = Conv1dSpec {
            batch: 2,
            in_channels: 2,
            out_channels: 2,
            input_len: 3,
            kernel: 2,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        let x_view = Layout::from_strides([2, 2, 3], [0, 2, 1], 1).unwrap();
        let w_view = Layout::from_strides([2, 2, 2], [2, 1, 1], 0).unwrap();
        let y_dense = Layout::contiguous([2, 2, 2]).unwrap();
        let x_dense = Layout::contiguous([2, 2, 3]).unwrap();
        let w_dense = Layout::contiguous([2, 2, 2]).unwrap();
        let base =
            Tensor::from_vec([8], vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.25, -0.35, 0.5]).unwrap();
        let dy = buffer(&y_dense, 3);
        let mut grad = Tensor::zeros([8]).unwrap();
        for side in SIDES {
            let target_dense = if side == Side::Left {
                &x_dense
            } else {
                &w_dense
            };
            let original = if side == Side::Left { &x_view } else { &w_view };
            let mut local = Tensor::zeros(target_dense.shape().dims()).unwrap();
            cpu::Conv1dBackward::new(cpu, side, spec, &x_view, &w_view, &y_dense, target_dense)
                .unwrap()
                .run(
                    Some(&base.view(&x_view).unwrap()),
                    Some(&base.view(&w_view).unwrap()),
                    &dy.view(&y_dense).unwrap(),
                    &mut local.view_mut(target_dense).unwrap(),
                    G::Overwrite,
                )
                .unwrap();
            cpu::ViewBackward::new([8], original, target_dense, 65536)
                .unwrap()
                .run(
                    &local.view(target_dense).unwrap(),
                    &mut grad,
                    if side == Side::Left {
                        G::Overwrite
                    } else {
                        G::Accumulate
                    },
                )
                .unwrap();
        }
        let objective = |b: &[f64]| {
            let xx: Vec<_> = addresses(&x_view).into_iter().map(|i| b[i]).collect();
            let ww: Vec<_> = addresses(&w_view).into_iter().map(|i| b[i]).collect();
            forward_1d(spec, &xx, &ww)
                .iter()
                .zip(dy.data())
                .map(|(y, &d)| y * d as f64)
                .sum::<f64>()
        };
        for i in 0..8 {
            let mut b: Vec<_> = base.data().iter().map(|&v| v as f64).collect();
            b[i] += 1e-4;
            let plus = objective(&b);
            b[i] -= 2e-4;
            close(grad.data()[i], (plus - objective(&b)) / 2e-4);
        }

        let saved_layout = std::sync::Arc::new(x_view.clone());
        let owner = Shared::new(base).unwrap();
        let saved = owner.save(&saved_layout).unwrap();
        let mut bp =
            cpu::Conv1dBackward::new(cpu, Side::Right, spec, &x_view, &w_view, &y_dense, &w_dense)
                .unwrap();
        let mut out = Tensor::zeros([2, 2, 2]).unwrap();
        {
            let read = saved.read().unwrap();
            bp.run(
                Some(&read.view().unwrap()),
                None,
                &dy.view(&y_dense).unwrap(),
                &mut out.view_mut(&w_dense).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        }
        owner.write().unwrap().data_mut()[0] = 77.0;
        let before = bits(&out);
        let result = (|| -> fast_audio::Result<()> {
            let read = saved.read()?;
            bp.run(
                Some(&read.view()?),
                None,
                &dy.view(&y_dense)?,
                &mut out.view_mut(&w_dense)?,
                G::Overwrite,
            )
        })();
        assert!(result.is_err());
        assert_eq!(bits(&out), before);
    }
}

#[test]
fn cpu_prepared_conv_miri_signed_overlap_and_bounds() {
    let c1 = &cases_1d()[19];
    exercise_1d(Cpu::Reference, c1, 0, &mut Exporter::new(None));
    let c2 = &cases_2d()[19];
    exercise_2d(Cpu::Reference, c2, 0, &mut Exporter::new(None));
}

#[test]
fn cpu_tape_conv1d_and_conv2d_sgd_matches_fp64_updates() {
    let mut export = Exporter::new(Some("training-cpu"));
    for cpu in support::backends() {
        let [hx1, hw1, hb1, hx2, hw2, hb2] = training_fixture();
        let mut builder = Builder::new().unwrap();
        let vx1 = builder.leaf([2, 2, 5], false).unwrap();
        let vw1 = builder.leaf([2, 2, 3], true).unwrap();
        let vb1 = builder.leaf([2], true).unwrap();
        let vx2 = builder.leaf([2, 2, 2, 3], false).unwrap();
        let vw2 = builder.leaf([2, 2, 2, 2], true).unwrap();
        let vb2 = builder.leaf([2], true).unwrap();

        assert!(builder.conv1d(vx2, vw1, 1, 1, 1, 1).is_err());
        assert!(builder.conv1d(vx1, vw2, 1, 1, 1, 1).is_err());
        assert!(builder.conv1d(vx1, vw1, 1, 1, 1, 0).is_err());
        assert!(builder.conv2d(vx1, vw2, [1, 1], [0, 0], [1, 1], 1).is_err());
        assert!(builder.conv2d(vx2, vw1, [1, 1], [0, 0], [1, 1], 1).is_err());
        assert!(builder.conv2d(vx2, vw2, [1, 1], [0, 0], [1, 1], 0).is_err());

        let y1 = builder.conv1d(vx1, vw1, 1, 1, 1, 1).unwrap();
        let b1_view = builder.reshape(vb1, [1, 2, 1]).unwrap();
        let z1 = builder.binary(BinaryOp::Add, y1, b1_view).unwrap();
        let sq1 = builder.unary(UnaryOp::Square, z1).unwrap();
        let l1 = builder
            .reduce(ReduceOp::Mean, sq1, &[0, 1, 2], false)
            .unwrap();

        let y2 = builder.conv2d(vx2, vw2, [1, 1], [0, 0], [1, 1], 1).unwrap();
        let b2_view = builder.reshape(vb2, [1, 2, 1, 1]).unwrap();
        let z2 = builder.binary(BinaryOp::Add, y2, b2_view).unwrap();
        let sq2 = builder.unary(UnaryOp::Square, z2).unwrap();
        let l2 = builder
            .reduce(ReduceOp::Mean, sq2, &[0, 1, 2, 3], false)
            .unwrap();

        let loss = builder.binary(BinaryOp::Add, l1, l2).unwrap();
        let program = builder.finish(&[loss]).unwrap();

        let ox1 = Shared::new(hx1.clone()).unwrap();
        let ow1 = Shared::new(hw1.clone()).unwrap();
        let ob1 = Shared::new(hb1.clone()).unwrap();
        let ox2 = Shared::new(hx2.clone()).unwrap();
        let ow2 = Shared::new(hw2.clone()).unwrap();
        let ob2 = Shared::new(hb2.clone()).unwrap();

        let mut tape = cpu::Tape::new(
            cpu,
            &program,
            &[
                Binding::new(vx1, ox1),
                Binding::new(vw1, ow1.clone()),
                Binding::new(vb1, ob1.clone()),
                Binding::new(vx2, ox2),
                Binding::new(vw2, ow2.clone()),
                Binding::new(vb2, ob2.clone()),
            ],
        )
        .unwrap();

        let cx1: Vec<_> = hx1.data().iter().map(|&v| v as f64).collect();
        let mut cw1: Vec<_> = hw1.data().iter().map(|&v| v as f64).collect();
        let mut cb1: Vec<_> = hb1.data().iter().map(|&v| v as f64).collect();
        let cx2: Vec<_> = hx2.data().iter().map(|&v| v as f64).collect();
        let mut cw2: Vec<_> = hw2.data().iter().map(|&v| v as f64).collect();
        let mut cb2: Vec<_> = hb2.data().iter().map(|&v| v as f64).collect();
        let rate = 0.05f32;
        let mut first_loss = 0.0f32;

        for step in 0..8 {
            let before_w1 = ow1.read().unwrap().clone();
            let before_b1 = ob1.read().unwrap().clone();
            let before_w2 = ow2.read().unwrap().clone();
            let before_b2 = ob2.read().unwrap().clone();

            let (expected_loss, gw1, gb1, gw2, gb2) = training_control(
                &cx1,
                &mut cw1,
                &mut cb1,
                &cx2,
                &mut cw2,
                &mut cb2,
                rate as f64,
            );

            support::no_allocations(|| {
                tape.forward().unwrap();
                tape.backward_scalar(G::Overwrite).unwrap();
            });

            let actual_loss = tape.output(loss).unwrap().storage().clone();
            let actual_gw1 = tape.gradient(vw1).unwrap().unwrap().clone();
            let actual_gb1 = tape.gradient(vb1).unwrap().unwrap().clone();
            let actual_gw2 = tape.gradient(vw2).unwrap().unwrap().clone();
            let actual_gb2 = tape.gradient(vb2).unwrap().unwrap().clone();
            support::no_allocations(|| {
                tape.sgd(&[vw1, vb1, vw2, vb2], rate).unwrap();
            });
            let after_w1 = ow1.read().unwrap().clone();
            let after_b1 = ob1.read().unwrap().clone();
            let after_w2 = ow2.read().unwrap().clone();
            let after_b2 = ob2.read().unwrap().clone();

            close(actual_loss.data()[0], expected_loss);
            support::close(actual_gw1.data(), &gw1, 2e-5, 2e-5);
            support::close(actual_gb1.data(), &gb1, 2e-5, 2e-5);
            support::close(actual_gw2.data(), &gw2, 2e-5, 2e-5);
            support::close(actual_gb2.data(), &gb2, 2e-5, 2e-5);
            support::close(after_w1.data(), &cw1, 2e-5, 2e-5);
            support::close(after_b1.data(), &cb1, 2e-5, 2e-5);
            support::close(after_w2.data(), &cw2, 2e-5, 2e-5);
            support::close(after_b2.data(), &cb2, 2e-5, 2e-5);

            if step == 0 {
                first_loss = actual_loss.data()[0];
            }
            if step == 7 {
                assert!(actual_loss.data()[0] < first_loss);
            }
            let backend = match cpu {
                Cpu::Reference => "reference",
                #[cfg(feature = "mkl")]
                Cpu::Mkl { .. } => "mkl",
            };
            export.training(
                backend,
                step,
                [&before_w1, &before_b1, &before_w2, &before_b2],
                &actual_loss,
                [&actual_gw1, &actual_gb1, &actual_gw2, &actual_gb2],
                [&after_w1, &after_b1, &after_w2, &after_b2],
            );
        }
    }
    export.finish();
}
