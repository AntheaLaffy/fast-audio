#![cfg(feature = "cuda")]
use fast_audio::{
    BinaryOp, BinaryOperand as Side, Conv1dSpec, ConvStrategy as Route, GradientMode as G, Layout,
    ReduceOp, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cuda::{Conv1dBackward, Conv1dPlan, Conv2dBackward, Conv2dPlan, Cuda, Tape},
};
#[path = "support/conv.rs"]
mod contract;
#[allow(
    dead_code,
    reason = "Convolution tests use their own independent references."
)]
mod support;
use contract::*;

fn resources(
    direct: bool,
    rank: usize,
    route: Route,
    workspace: usize,
    metadata: usize,
    shared: usize,
    backward: bool,
) {
    let expected_meta = if backward { 4 } else { 3 } * rank * 16;
    assert_eq!(metadata, expected_meta);
    assert_eq!(workspace, 0);
    assert_eq!(shared > 0, route == Route::TiledCuda);
    assert!(shared <= 2048);
    if direct {
        assert_eq!(route, Route::Cublas);
    }
}

#[test]
fn cuda_prepared_conv_and_vjps_match_independent_oracles() {
    let cuda = Cuda::new(0).unwrap();
    let mut export = Exporter::new(Some("conv-cuda"));
    for (i, c) in cases_1d().iter().enumerate() {
        let p = Conv1dPlan::new(&cuda, c.spec, &c.x, &c.w, &c.y).unwrap();
        assert_eq!(p.spec(), c.spec);
        assert_eq!(p.input_layout(), &c.x);
        assert_eq!(p.weight_layout(), &c.w);
        assert_eq!(p.destination_layout(), &c.y);
        resources(
            c.direct,
            3,
            p.strategy(),
            p.workspace_bytes(),
            p.host_metadata_bytes(),
            p.shared_memory_bytes_per_block(),
            false,
        );
        let backward = SIDES.map(|side| {
            Conv1dBackward::new(&cuda, side, c.spec, &c.x, &c.w, &c.dy, c.gradient(side)).unwrap()
        });
        for (bp, side) in backward.iter().zip(SIDES) {
            assert_eq!(bp.spec(), c.spec);
            assert_eq!(bp.operand(), side);
            assert_eq!(bp.input_layout(), &c.x);
            assert_eq!(bp.weight_layout(), &c.w);
            assert_eq!(bp.upstream_layout(), &c.dy);
            assert_eq!(bp.destination_layout(), c.gradient(side));
            assert_eq!(bp.input_dependencies(), c.dependencies(side));
            resources(
                c.direct,
                3,
                bp.strategy(),
                bp.workspace_bytes(),
                bp.host_metadata_bytes(),
                bp.shared_memory_bytes_per_block(),
                true,
            );
        }
        for change in 0..2 {
            let x = buffer(&c.x, i + change);
            let w = buffer(&c.w, i + change + 1);
            let dy = buffer(&c.dy, i + change + 2);
            let xx = logical(&c.x, &x);
            let ww = logical(&c.w, &w);
            let dd = logical(&c.dy, &dy);
            let dx_in = cuda.upload(&x).unwrap();
            let dw_in = cuda.upload(&w).unwrap();
            let incoming = cuda.upload(&dy).unwrap();
            let initial = seed(&c.y, false);
            let mut out = cuda.upload(&initial).unwrap();
            support::no_allocations(|| {
                p.run(
                    &dx_in.view(&c.x).unwrap(),
                    &dw_in.view(&c.w).unwrap(),
                    &mut out.view_mut(&c.y).unwrap(),
                )
                .unwrap()
            });
            let y = cuda.download(&out).unwrap();
            verify(
                &c.y,
                &forward_1d(c.spec, &xx, &ww),
                &initial,
                &y,
                G::Overwrite,
            );
            p.run(
                &dx_in.view(&c.x).unwrap(),
                &dw_in.view(&c.w).unwrap(),
                &mut out.view_mut(&c.y).unwrap(),
            )
            .unwrap();
            assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&y));
            let xv = dx_in.view(&c.x).unwrap();
            let wv = dw_in.view(&c.w).unwrap();
            let mut results: [[Option<Tensor>; 2]; 2] = [[None, None], [None, None]];
            for (side_index, (bp, side)) in backward.iter().zip(SIDES).enumerate() {
                let grad = c.gradient(side);
                let expected = derivative_1d(c.spec, side, &xx, &ww, &dd);
                let deps = bp.input_dependencies();
                let inputs = [
                    if deps[0] || change == 1 {
                        Some(&xv)
                    } else {
                        None
                    },
                    if deps[1] || change == 1 {
                        Some(&wv)
                    } else {
                        None
                    },
                ];
                for (mode_index, mode) in [G::Overwrite, G::Accumulate].into_iter().enumerate() {
                    let initial = seed(grad, mode == G::Accumulate);
                    let mut dg = cuda.upload(&initial).unwrap();
                    support::no_allocations(|| {
                        bp.run(
                            inputs[0],
                            inputs[1],
                            &incoming.view(&c.dy).unwrap(),
                            &mut dg.view_mut(grad).unwrap(),
                            mode,
                        )
                        .unwrap()
                    });
                    let actual = cuda.download(&dg).unwrap();
                    verify(grad, &expected, &initial, &actual, mode);
                    let mut repeat = cuda.upload(&initial).unwrap();
                    bp.run(
                        inputs[0],
                        inputs[1],
                        &incoming.view(&c.dy).unwrap(),
                        &mut repeat.view_mut(grad).unwrap(),
                        mode,
                    )
                    .unwrap();
                    assert_eq!(bits(&cuda.download(&repeat).unwrap()), bits(&actual));
                    if export.enabled() {
                        results[mode_index][side_index] = Some(actual.clone());
                    }
                    if mode == G::Overwrite {
                        identities_1d(c, side, &x, &w, &y, &dy, &actual);
                    } else {
                        bp.run(
                            inputs[0],
                            inputs[1],
                            &incoming.view(&c.dy).unwrap(),
                            &mut dg.view_mut(grad).unwrap(),
                            mode,
                        )
                        .unwrap();
                        verify(grad, &expected, &actual, &cuda.download(&dg).unwrap(), mode);
                    }
                }
            }
            if export.enabled() {
                export.conv1d(
                    "cuda",
                    i,
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
            assert_eq!(bits(&cuda.download(&dx_in).unwrap()), bits(&x));
            assert_eq!(bits(&cuda.download(&dw_in).unwrap()), bits(&w));
            assert_eq!(bits(&cuda.download(&incoming).unwrap()), bits(&dy));
        }
    }

    for (i, c) in cases_2d().iter().enumerate() {
        let p = Conv2dPlan::new(&cuda, c.spec, &c.x, &c.w, &c.y).unwrap();
        assert_eq!(p.spec(), c.spec);
        assert_eq!(p.input_layout(), &c.x);
        assert_eq!(p.weight_layout(), &c.w);
        assert_eq!(p.destination_layout(), &c.y);
        resources(
            c.direct,
            4,
            p.strategy(),
            p.workspace_bytes(),
            p.host_metadata_bytes(),
            p.shared_memory_bytes_per_block(),
            false,
        );
        let backward = SIDES.map(|side| {
            Conv2dBackward::new(&cuda, side, c.spec, &c.x, &c.w, &c.dy, c.gradient(side)).unwrap()
        });
        for (bp, side) in backward.iter().zip(SIDES) {
            assert_eq!(bp.spec(), c.spec);
            assert_eq!(bp.operand(), side);
            assert_eq!(bp.input_layout(), &c.x);
            assert_eq!(bp.weight_layout(), &c.w);
            assert_eq!(bp.upstream_layout(), &c.dy);
            assert_eq!(bp.destination_layout(), c.gradient(side));
            assert_eq!(bp.input_dependencies(), c.dependencies(side));
            resources(
                c.direct,
                4,
                bp.strategy(),
                bp.workspace_bytes(),
                bp.host_metadata_bytes(),
                bp.shared_memory_bytes_per_block(),
                true,
            );
        }
        for change in 0..2 {
            let x = buffer(&c.x, i + change);
            let w = buffer(&c.w, i + change + 1);
            let dy = buffer(&c.dy, i + change + 2);
            let xx = logical(&c.x, &x);
            let ww = logical(&c.w, &w);
            let dd = logical(&c.dy, &dy);
            let dx_in = cuda.upload(&x).unwrap();
            let dw_in = cuda.upload(&w).unwrap();
            let incoming = cuda.upload(&dy).unwrap();
            let initial = seed(&c.y, false);
            let mut out = cuda.upload(&initial).unwrap();
            support::no_allocations(|| {
                p.run(
                    &dx_in.view(&c.x).unwrap(),
                    &dw_in.view(&c.w).unwrap(),
                    &mut out.view_mut(&c.y).unwrap(),
                )
                .unwrap()
            });
            let y = cuda.download(&out).unwrap();
            verify(
                &c.y,
                &forward_2d(c.spec, &xx, &ww),
                &initial,
                &y,
                G::Overwrite,
            );
            p.run(
                &dx_in.view(&c.x).unwrap(),
                &dw_in.view(&c.w).unwrap(),
                &mut out.view_mut(&c.y).unwrap(),
            )
            .unwrap();
            assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&y));
            let xv = dx_in.view(&c.x).unwrap();
            let wv = dw_in.view(&c.w).unwrap();
            let mut results: [[Option<Tensor>; 2]; 2] = [[None, None], [None, None]];
            for (side_index, (bp, side)) in backward.iter().zip(SIDES).enumerate() {
                let grad = c.gradient(side);
                let expected = derivative_2d(c.spec, side, &xx, &ww, &dd);
                let deps = bp.input_dependencies();
                let inputs = [
                    if deps[0] || change == 1 {
                        Some(&xv)
                    } else {
                        None
                    },
                    if deps[1] || change == 1 {
                        Some(&wv)
                    } else {
                        None
                    },
                ];
                for (mode_index, mode) in [G::Overwrite, G::Accumulate].into_iter().enumerate() {
                    let initial = seed(grad, mode == G::Accumulate);
                    let mut dg = cuda.upload(&initial).unwrap();
                    support::no_allocations(|| {
                        bp.run(
                            inputs[0],
                            inputs[1],
                            &incoming.view(&c.dy).unwrap(),
                            &mut dg.view_mut(grad).unwrap(),
                            mode,
                        )
                        .unwrap()
                    });
                    let actual = cuda.download(&dg).unwrap();
                    verify(grad, &expected, &initial, &actual, mode);
                    let mut repeat = cuda.upload(&initial).unwrap();
                    bp.run(
                        inputs[0],
                        inputs[1],
                        &incoming.view(&c.dy).unwrap(),
                        &mut repeat.view_mut(grad).unwrap(),
                        mode,
                    )
                    .unwrap();
                    assert_eq!(bits(&cuda.download(&repeat).unwrap()), bits(&actual));
                    if export.enabled() {
                        results[mode_index][side_index] = Some(actual.clone());
                    }
                    if mode == G::Overwrite {
                        identities_2d(c, side, &x, &w, &y, &dy, &actual);
                    } else {
                        bp.run(
                            inputs[0],
                            inputs[1],
                            &incoming.view(&c.dy).unwrap(),
                            &mut dg.view_mut(grad).unwrap(),
                            mode,
                        )
                        .unwrap();
                        verify(grad, &expected, &actual, &cuda.download(&dg).unwrap(), mode);
                    }
                }
            }
            if export.enabled() {
                export.conv2d(
                    "cuda",
                    i,
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
            assert_eq!(bits(&cuda.download(&dx_in).unwrap()), bits(&x));
            assert_eq!(bits(&cuda.download(&dw_in).unwrap()), bits(&w));
            assert_eq!(bits(&cuda.download(&incoming).unwrap()), bits(&dy));
        }
    }
    export.finish();
}

#[test]
fn cuda_prepared_conv_rejects_geometry_dependencies_and_sessions() {
    let cuda = Cuda::new(0).unwrap();
    let other = Cuda::new(0).unwrap();
    let shifted =
        |l: &Layout| Layout::from_strides(l.shape().dims(), l.strides(), l.offset() + 1).unwrap();
    let c = &cases_1d()[16];
    let x = cuda.upload(&buffer(&c.x, 0)).unwrap();
    let w = cuda.upload(&buffer(&c.w, 1)).unwrap();
    let dy = cuda.upload(&buffer(&c.dy, 2)).unwrap();
    let xv = x.view(&c.x).unwrap();
    let wv = w.view(&c.w).unwrap();
    let initial = seed(&c.y, false);
    let mut out = cuda.upload(&initial).unwrap();
    let p = Conv1dPlan::new(&cuda, c.spec, &c.x, &c.w, &c.y).unwrap();
    assert!(
        p.run(
            &x.view(&shifted(&c.x)).unwrap(),
            &wv,
            &mut out.view_mut(&c.y).unwrap()
        )
        .is_err()
    );
    assert!(
        p.run(
            &xv,
            &w.view(&shifted(&c.w)).unwrap(),
            &mut out.view_mut(&c.y).unwrap()
        )
        .is_err()
    );
    assert!(
        p.run(&xv, &wv, &mut out.view_mut(&shifted(&c.y)).unwrap())
            .is_err()
    );
    assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&initial));

    let fx = other.upload(&buffer(&c.x, 0)).unwrap();
    let fw = other.upload(&buffer(&c.w, 1)).unwrap();
    let fdy = other.upload(&buffer(&c.dy, 2)).unwrap();
    let mut fout = other.upload(&initial).unwrap();
    assert!(
        p.run(
            &fx.view(&c.x).unwrap(),
            &wv,
            &mut out.view_mut(&c.y).unwrap()
        )
        .is_err()
    );
    assert!(
        p.run(
            &xv,
            &fw.view(&c.w).unwrap(),
            &mut out.view_mut(&c.y).unwrap()
        )
        .is_err()
    );
    assert!(p.run(&xv, &wv, &mut fout.view_mut(&c.y).unwrap()).is_err());

    for side in SIDES {
        let grad = c.gradient(side);
        let bp = Conv1dBackward::new(&cuda, side, c.spec, &c.x, &c.w, &c.dy, grad).unwrap();
        let initial = seed(grad, true);
        let mut out = cuda.upload(&initial).unwrap();
        for j in 0..2 {
            if bp.input_dependencies()[j] {
                let mut inputs = [Some(&xv), Some(&wv)];
                inputs[j] = None;
                assert!(
                    bp.run(
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
            bp.run(
                Some(&fx.view(&c.x).unwrap()),
                Some(&wv),
                &dy.view(&c.dy).unwrap(),
                &mut out.view_mut(grad).unwrap(),
                G::Overwrite
            )
            .is_err()
        );
        assert!(
            bp.run(
                Some(&xv),
                Some(&fw.view(&c.w).unwrap()),
                &dy.view(&c.dy).unwrap(),
                &mut out.view_mut(grad).unwrap(),
                G::Overwrite
            )
            .is_err()
        );
        assert!(
            bp.run(
                Some(&xv),
                Some(&wv),
                &fdy.view(&c.dy).unwrap(),
                &mut out.view_mut(grad).unwrap(),
                G::Overwrite
            )
            .is_err()
        );
        assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&initial));
    }
}

#[test]
fn cuda_prepared_conv_saved_versions_view_composition_and_grid_reuse() {
    let cuda = Cuda::new(0).unwrap();
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
    let host = Tensor::from_vec([8], vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.25, -0.35, 0.5]).unwrap();
    let base = cuda.upload(&host).unwrap();
    let dy = buffer(&y_dense, 3);
    let incoming = cuda.upload(&dy).unwrap();
    let mut grad = cuda.zeros([8]).unwrap();
    for side in SIDES {
        let target_dense = if side == Side::Left {
            &x_dense
        } else {
            &w_dense
        };
        let original = if side == Side::Left { &x_view } else { &w_view };
        let mut local = cuda.zeros(target_dense.shape().dims()).unwrap();
        Conv1dBackward::new(&cuda, side, spec, &x_view, &w_view, &y_dense, target_dense)
            .unwrap()
            .run(
                Some(&base.view(&x_view).unwrap()),
                Some(&base.view(&w_view).unwrap()),
                &incoming.view(&y_dense).unwrap(),
                &mut local.view_mut(target_dense).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        fast_audio::cuda::ViewBackward::new(&cuda, [8], original, target_dense, 65536)
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
    let actual = cuda.download(&grad).unwrap();
    for i in 0..8 {
        let mut b: Vec<_> = host.data().iter().map(|&v| v as f64).collect();
        b[i] += 1e-4;
        let plus = objective(&b);
        b[i] -= 2e-4;
        close(actual.data()[i], (plus - objective(&b)) / 2e-4);
    }

    let owner = Shared::new(base).unwrap();
    let saved = owner.save(&std::sync::Arc::new(x_view.clone())).unwrap();
    let mut out = cuda.zeros([2, 2, 2]).unwrap();
    let bp = Conv1dBackward::new(
        &cuda,
        Side::Right,
        spec,
        &x_view,
        &w_view,
        &y_dense,
        &w_dense,
    )
    .unwrap();
    {
        let read = saved.read().unwrap();
        bp.run(
            Some(&read.view().unwrap()),
            None,
            &incoming.view(&y_dense).unwrap(),
            &mut out.view_mut(&w_dense).unwrap(),
            G::Overwrite,
        )
        .unwrap();
    }
    let before = cuda.download(&out).unwrap();
    drop(owner.write().unwrap());
    let result = (|| -> fast_audio::Result<()> {
        let read = saved.read()?;
        bp.run(
            Some(&read.view()?),
            None,
            &incoming.view(&y_dense)?,
            &mut out.view_mut(&w_dense)?,
            G::Overwrite,
        )
    })();
    assert!(result.is_err());
    assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&before));

    let count = 65535 * 16 + 17;
    let grid_spec = Conv1dSpec {
        batch: 1,
        in_channels: 1,
        out_channels: 1,
        input_len: count,
        kernel: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let lx = Layout::from_strides([1, 1, count], [0, 0, 0], 0).unwrap();
    let lw = Layout::from_strides([1, 1, 1], [0, 0, 0], 0).unwrap();
    let ly = Layout::contiguous([1, 1, count]).unwrap();
    let gx = cuda
        .upload(&Tensor::from_vec([1], vec![3.0]).unwrap())
        .unwrap();
    let gw = cuda
        .upload(&Tensor::from_vec([1], vec![2.0]).unwrap())
        .unwrap();
    let mut gout = cuda.zeros([1, 1, count]).unwrap();
    let gp = Conv1dPlan::new(&cuda, grid_spec, &lx, &lw, &ly).unwrap();
    assert_eq!(gp.strategy(), Route::TiledCuda);
    assert_eq!(gp.shared_memory_bytes_per_block(), 2048);
    support::no_allocations(|| {
        gp.run(
            &gx.view(&lx).unwrap(),
            &gw.view(&lw).unwrap(),
            &mut gout.view_mut(&ly).unwrap(),
        )
        .unwrap()
    });
    let host_out = cuda.download(&gout).unwrap();
    assert!(host_out.data().iter().all(|&v| v == 6.0));
}

#[test]
fn cuda_tape_conv1d_and_conv2d_sgd_matches_fp64_updates() {
    let cuda = Cuda::new(0).unwrap();
    let mut export = Exporter::new(Some("training-cuda"));
    let [hx1, hw1, hb1, hx2, hw2, hb2] = training_fixture();
    let mut builder = Builder::new().unwrap();
    let vx1 = builder.leaf([2, 2, 5], false).unwrap();
    let vw1 = builder.leaf([2, 2, 3], true).unwrap();
    let vb1 = builder.leaf([2], true).unwrap();
    let vx2 = builder.leaf([2, 2, 2, 3], false).unwrap();
    let vw2 = builder.leaf([2, 2, 2, 2], true).unwrap();
    let vb2 = builder.leaf([2], true).unwrap();

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

    let ox1 = Shared::new(cuda.upload(&hx1).unwrap()).unwrap();
    let ow1 = Shared::new(cuda.upload(&hw1).unwrap()).unwrap();
    let ob1 = Shared::new(cuda.upload(&hb1).unwrap()).unwrap();
    let ox2 = Shared::new(cuda.upload(&hx2).unwrap()).unwrap();
    let ow2 = Shared::new(cuda.upload(&hw2).unwrap()).unwrap();
    let ob2 = Shared::new(cuda.upload(&hb2).unwrap()).unwrap();

    let mut tape = Tape::new(
        &cuda,
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
    assert_eq!(tape.resources().max_static_shared_bytes_per_block, 2048);

    let cx1: Vec<_> = hx1.data().iter().map(|&v| v as f64).collect();
    let mut cw1: Vec<_> = hw1.data().iter().map(|&v| v as f64).collect();
    let mut cb1: Vec<_> = hb1.data().iter().map(|&v| v as f64).collect();
    let cx2: Vec<_> = hx2.data().iter().map(|&v| v as f64).collect();
    let mut cw2: Vec<_> = hw2.data().iter().map(|&v| v as f64).collect();
    let mut cb2: Vec<_> = hb2.data().iter().map(|&v| v as f64).collect();
    let rate = 0.05f32;
    let mut first_loss = 0.0f32;

    for step in 0..8 {
        let before_w1 = cuda.download(&ow1.read().unwrap()).unwrap();
        let before_b1 = cuda.download(&ob1.read().unwrap()).unwrap();
        let before_w2 = cuda.download(&ow2.read().unwrap()).unwrap();
        let before_b2 = cuda.download(&ob2.read().unwrap()).unwrap();

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

        let actual_loss = cuda.download(tape.output(loss).unwrap().storage()).unwrap();
        let actual_gw1 = cuda.download(tape.gradient(vw1).unwrap().unwrap()).unwrap();
        let actual_gb1 = cuda.download(tape.gradient(vb1).unwrap().unwrap()).unwrap();
        let actual_gw2 = cuda.download(tape.gradient(vw2).unwrap().unwrap()).unwrap();
        let actual_gb2 = cuda.download(tape.gradient(vb2).unwrap().unwrap()).unwrap();
        support::no_allocations(|| {
            tape.sgd(&[vw1, vb1, vw2, vb2], rate).unwrap();
        });
        let after_w1 = cuda.download(&ow1.read().unwrap()).unwrap();
        let after_b1 = cuda.download(&ob1.read().unwrap()).unwrap();
        let after_w2 = cuda.download(&ow2.read().unwrap()).unwrap();
        let after_b2 = cuda.download(&ob2.read().unwrap()).unwrap();

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
        export.training(
            "cuda",
            step,
            [&before_w1, &before_b1, &before_w2, &before_b2],
            &actual_loss,
            [&actual_gw1, &actual_gb1, &actual_gw2, &actual_gb2],
            [&after_w1, &after_b1, &after_w2, &after_b2],
        );
    }
    export.finish();
}
