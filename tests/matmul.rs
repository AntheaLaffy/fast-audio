use fast_audio::{
    BinaryOperand as Side, GradientMode as G, Layout, MatmulStrategy as Route, Shared, Tensor,
    cpu::{self, Cpu},
};
#[path = "support/matmul.rs"]
mod contract;
#[allow(
    dead_code,
    reason = "Matrix fixtures use their own independent numerical oracle."
)]
mod support;
use contract::*;

fn resources(c: &Case, cpu: Cpu, route: Route, workspace: usize, metadata: usize, backward: bool) {
    assert_eq!(metadata, if backward { 128 } else { 96 });
    assert!(workspace <= 57_344);
    if route == Route::PackedMkl {
        assert!(workspace > 0);
    } else {
        assert_eq!(workspace, 0);
    }
    if c.direct {
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
fn exercise(cpu: Cpu, c: &Case, salt: usize, export: &mut Exporter) {
    let mut forward_plan = cpu::MatmulPlan::new(cpu, &c.a, &c.b, &c.y).unwrap();
    assert_eq!(forward_plan.left_layout(), &c.a);
    assert_eq!(forward_plan.right_layout(), &c.b);
    assert_eq!(forward_plan.destination_layout(), &c.y);
    resources(
        c,
        cpu,
        forward_plan.strategy(),
        forward_plan.workspace_bytes(),
        forward_plan.host_metadata_bytes(),
        false,
    );
    let mut backward = SIDES.map(|side| {
        cpu::MatmulBackward::new(cpu, side, &c.a, &c.b, &c.dy, c.gradient(side)).unwrap()
    });
    for (p, side) in backward.iter().zip(SIDES) {
        assert_eq!(p.operand(), side);
        assert_eq!(p.left_layout(), &c.a);
        assert_eq!(p.right_layout(), &c.b);
        assert_eq!(p.upstream_layout(), &c.dy);
        assert_eq!(p.destination_layout(), c.gradient(side));
        assert_eq!(p.input_dependencies(), c.dependencies(side));
        resources(
            c,
            cpu,
            p.strategy(),
            p.workspace_bytes(),
            p.host_metadata_bytes(),
            true,
        );
    }
    for change in 0..2 {
        let a = buffer(&c.a, salt + change);
        let b = buffer(&c.b, salt + change + 1);
        let dy = buffer(&c.dy, salt + change + 2);
        let original = (bits(&a), bits(&b), bits(&dy));
        let aa = logical(&c.a, &a);
        let bb = logical(&c.b, &b);
        let dd = logical(&c.dy, &dy);
        let initial = seed(&c.y, false);
        let mut y = initial.clone();
        support::no_allocations(|| {
            forward_plan
                .run(
                    &a.view(&c.a).unwrap(),
                    &b.view(&c.b).unwrap(),
                    &mut y.view_mut(&c.y).unwrap(),
                )
                .unwrap()
        });
        verify(
            &c.y,
            &forward(c.dims, &aa, &bb),
            &initial,
            &y,
            G::Overwrite,
            c.dims[2] == 0,
        );
        let first = bits(&y);
        forward_plan
            .run(
                &a.view(&c.a).unwrap(),
                &b.view(&c.b).unwrap(),
                &mut y.view_mut(&c.y).unwrap(),
            )
            .unwrap();
        assert_eq!(bits(&y), first);
        let av = a.view(&c.a).unwrap();
        let bv = b.view(&c.b).unwrap();
        let mut results: [[Option<Tensor>; 2]; 2] = [[None, None], [None, None]];
        for (side_index, (p, side)) in backward.iter_mut().zip(SIDES).enumerate() {
            let grad = c.gradient(side);
            let expected = derivative(c.dims, side, &aa, &bb, &dd);
            let required = p.input_dependencies();
            let inputs = [
                if required[0] || change == 1 {
                    Some(&av)
                } else {
                    None
                },
                if required[1] || change == 1 {
                    Some(&bv)
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
                verify(
                    grad,
                    &expected,
                    &initial,
                    &actual,
                    mode,
                    c.dims.contains(&0),
                );
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
                    identities(c, side, &a, &b, &y, &dy, &actual);
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
                    verify(grad, &expected, &before, &actual, mode, false);
                }
            }
        }
        if export.enabled() {
            let backend = match cpu {
                Cpu::Reference => "reference",
                #[cfg(feature = "mkl")]
                Cpu::Mkl { .. } => "mkl",
            };
            export.matrix(Sample {
                index: salt,
                binding: change,
                backend,
                case: c,
                inputs: [&a, &b, &dy],
                output: &y,
                gradients: [
                    [
                        results[0][0].as_ref().unwrap(),
                        results[0][1].as_ref().unwrap(),
                    ],
                    [
                        results[1][0].as_ref().unwrap(),
                        results[1][1].as_ref().unwrap(),
                    ],
                ],
            });
        }
        assert_eq!((bits(&a), bits(&b), bits(&dy)), original);
    }
}
#[test]
fn cpu_prepared_matmul_and_vjps_match_independent_oracles() {
    let mut export = Exporter::new(Some("matmul-cpu"));
    for cpu in support::backends() {
        for (i, c) in cases().iter().enumerate() {
            exercise(cpu, c, i, &mut export);
        }
    }
    export.finish();
}
fn shifted(l: &Layout) -> Layout {
    Layout::from_strides(l.shape().dims(), l.strides(), l.offset() + 1).unwrap()
}
#[test]
fn cpu_prepared_matmul_rejects_changed_geometry_and_missing_dependencies() {
    for cpu in support::backends() {
        for c in cases() {
            let a = buffer(&c.a, 0);
            let b = buffer(&c.b, 1);
            let dy = buffer(&c.dy, 2);
            let initial = seed(&c.y, false);
            let mut out = initial.clone();
            let mut p = cpu::MatmulPlan::new(cpu, &c.a, &c.b, &c.y).unwrap();
            assert!(
                p.run(
                    &a.view(&shifted(&c.a)).unwrap(),
                    &b.view(&c.b).unwrap(),
                    &mut out.view_mut(&c.y).unwrap()
                )
                .is_err()
            );
            assert!(
                p.run(
                    &a.view(&c.a).unwrap(),
                    &b.view(&shifted(&c.b)).unwrap(),
                    &mut out.view_mut(&c.y).unwrap()
                )
                .is_err()
            );
            assert!(
                p.run(
                    &a.view(&c.a).unwrap(),
                    &b.view(&c.b).unwrap(),
                    &mut out.view_mut(&shifted(&c.y)).unwrap()
                )
                .is_err()
            );
            assert_eq!(bits(&out), bits(&initial));
            for side in SIDES {
                let grad = c.gradient(side);
                let initial = seed(grad, true);
                let mut out = initial.clone();
                let mut p = cpu::MatmulBackward::new(cpu, side, &c.a, &c.b, &c.dy, grad).unwrap();
                let av = a.view(&c.a).unwrap();
                let bv = b.view(&c.b).unwrap();
                for i in 0..2 {
                    if p.input_dependencies()[i] {
                        let mut inputs = [Some(&av), Some(&bv)];
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
                        Some(&a.view(&shifted(&c.a)).unwrap()),
                        Some(&bv),
                        &dy.view(&c.dy).unwrap(),
                        &mut out.view_mut(grad).unwrap(),
                        G::Accumulate
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&av),
                        Some(&b.view(&shifted(&c.b)).unwrap()),
                        &dy.view(&c.dy).unwrap(),
                        &mut out.view_mut(grad).unwrap(),
                        G::Overwrite
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&av),
                        Some(&bv),
                        &dy.view(&shifted(&c.dy)).unwrap(),
                        &mut out.view_mut(grad).unwrap(),
                        G::Overwrite
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&av),
                        Some(&bv),
                        &dy.view(&c.dy).unwrap(),
                        &mut out.view_mut(&shifted(grad)).unwrap(),
                        G::Accumulate
                    )
                    .is_err()
                );
                assert_eq!(bits(&out), bits(&initial));
            }
        }
        let a = Layout::contiguous([2, 3]).unwrap();
        let b = Layout::contiguous([3, 4]).unwrap();
        let y = Layout::contiguous([2, 4]).unwrap();
        let flat = Layout::contiguous([6]).unwrap();
        let wrong = Layout::contiguous([4, 2]).unwrap();
        let overlap = Layout::from_strides([2, 4], [0, 1], 0).unwrap();
        assert!(cpu::MatmulPlan::new(cpu, &flat, &b, &y).is_err());
        assert!(cpu::MatmulPlan::new(cpu, &a, &a, &y).is_err());
        assert!(cpu::MatmulPlan::new(cpu, &a, &b, &wrong).is_err());
        assert!(cpu::MatmulPlan::new(cpu, &a, &b, &overlap).is_err());
        for side in SIDES {
            let g = if side == Side::Left { &a } else { &b };
            assert!(cpu::MatmulBackward::new(cpu, side, &a, &b, &wrong, g).is_err());
            assert!(cpu::MatmulBackward::new(cpu, side, &a, &b, &y, &flat).is_err());
            let overlap = Layout::from_strides(g.shape().dims(), [0, 1], 0).unwrap();
            assert!(cpu::MatmulBackward::new(cpu, side, &a, &b, &y, &overlap).is_err());
        }
    }
}
#[test]
fn cpu_prepared_matmul_zero_contractions_preserve_accumulation_bits() {
    for cpu in support::backends() {
        for dims in [[0, 3, 4], [3, 0, 4], [3, 4, 0]] {
            let [m, n, k] = dims;
            let a = Layout::contiguous([m, k]).unwrap();
            let b = Layout::contiguous([k, n]).unwrap();
            let dy = Layout::contiguous([m, n]).unwrap();
            let incoming = Tensor::zeros([m * n]).unwrap();
            for side in SIDES {
                let grad = if side == Side::Left {
                    layout(m, k, 5)
                } else {
                    layout(k, n, 4)
                };
                let mut p = cpu::MatmulBackward::new(cpu, side, &a, &b, &dy, &grad).unwrap();
                assert_eq!(p.input_dependencies(), [false; 2]);
                let mut initial = seed(&grad, true);
                for (i, index) in addresses(&grad).into_iter().enumerate() {
                    initial.data_mut()[index] =
                        f32::from_bits(if i % 2 == 0 { 0x80000000 } else { 0x7fc54321 });
                }
                let mut out = initial.clone();
                support::no_allocations(|| {
                    p.run(
                        None,
                        None,
                        &incoming.view(&dy).unwrap(),
                        &mut out.view_mut(&grad).unwrap(),
                        G::Accumulate,
                    )
                    .unwrap()
                });
                assert_eq!(bits(&out), bits(&initial));
                p.run(
                    None,
                    None,
                    &incoming.view(&dy).unwrap(),
                    &mut out.view_mut(&grad).unwrap(),
                    G::Overwrite,
                )
                .unwrap();
                for index in addresses(&grad) {
                    assert_eq!(out.data()[index].to_bits(), 0);
                }
                holes(&grad, &initial, &out);
            }
            if k == 0 {
                let no_storage = Tensor::zeros([0]).unwrap();
                let result = layout(m, n, 5);
                let initial = seed(&result, false);
                let mut out = initial.clone();
                cpu::MatmulPlan::new(cpu, &a, &b, &result)
                    .unwrap()
                    .run(
                        &no_storage.view(&a).unwrap(),
                        &no_storage.view(&b).unwrap(),
                        &mut out.view_mut(&result).unwrap(),
                    )
                    .unwrap();
                for i in addresses(&result) {
                    assert_eq!(out.data()[i].to_bits(), 0);
                }
                holes(&result, &initial, &out);
            }
        }
    }
}
#[test]
fn cpu_prepared_matmul_budgets_large_geometry_and_independent_threads() {
    for cpu in support::backends() {
        let a = Layout::from_strides([1, 5_000_000_001], [0, 0], 0).unwrap();
        let b = Layout::from_strides([5_000_000_001, 1], [0, 0], 0).unwrap();
        let y = Layout::contiguous([1, 1]).unwrap();
        let p = cpu::MatmulPlan::new(cpu, &a, &b, &y).unwrap();
        assert!(p.workspace_bytes() <= 57_344);
        let large_a = Layout::from_strides([33, 129], [0, 0], 0).unwrap();
        let large_b = Layout::from_strides([129, 65], [0, 0], 0).unwrap();
        let large_y = layout(33, 65, 4);
        let maximum = cpu::MatmulPlan::new(cpu, &large_a, &large_b, &large_y).unwrap();
        if maximum.strategy() == Route::PackedMkl {
            assert_eq!(maximum.workspace_bytes(), 57_344);
        }
        assert!(p.host_metadata_bytes() < 1024 && p.workspace_bytes() <= 57_344);
        if p.strategy() == Route::PackedMkl {
            assert!(
                cpu::MatmulPlan::with_workspace_limit(cpu, &a, &b, &y, p.workspace_bytes() - 1)
                    .is_err()
            );
            assert_eq!(
                cpu::MatmulPlan::with_workspace_limit(cpu, &a, &b, &y, p.workspace_bytes())
                    .unwrap()
                    .workspace_bytes(),
                p.workspace_bytes()
            );
        }
        let a = layout(3, 7, 3);
        let b = layout(7, 5, 5);
        let y = layout(3, 5, 5);
        let p = cpu::MatmulPlan::new(cpu, &a, &b, &y).unwrap();
        assert!(p.workspace_bytes() <= 57_344);
        for side in SIDES {
            let grad = if side == Side::Left { &a } else { &b };
            let p = cpu::MatmulBackward::new(cpu, side, &a, &b, &y, grad).unwrap();
            if p.strategy() == Route::PackedMkl {
                assert!(
                    cpu::MatmulBackward::with_workspace_limit(
                        cpu,
                        side,
                        &a,
                        &b,
                        &y,
                        grad,
                        p.workspace_bytes() - 1
                    )
                    .is_err()
                );
            }
        }
        #[cfg(feature = "mkl")]
        if matches!(cpu, Cpu::Mkl { .. }) {
            assert_eq!(p.strategy(), Route::PackedMkl);
            assert!(cpu::MatmulPlan::with_workspace_limit(cpu, &a, &b, &y, 0).is_err());
        }
        let aa = buffer(&a, 0);
        let bb = buffer(&b, 1);
        let expected = forward([3, 5, 7], &logical(&a, &aa), &logical(&b, &bb));
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let (a, b, y, aa, bb, expected) = (&a, &b, &y, &aa, &bb, &expected);
                scope.spawn(move || {
                    let mut p = cpu::MatmulPlan::new(cpu, a, b, y).unwrap();
                    let initial = seed(y, false);
                    let mut out = initial.clone();
                    support::no_allocations(|| {
                        p.run(
                            &aa.view(a).unwrap(),
                            &bb.view(b).unwrap(),
                            &mut out.view_mut(y).unwrap(),
                        )
                        .unwrap()
                    });
                    verify(y, expected, &initial, &out, G::Overwrite, false);
                });
            }
        });
        let empty = Layout::contiguous([usize::MAX, 0]).unwrap();
        let b = Layout::contiguous([0, 0]).unwrap();
        let t = Tensor::zeros([0]).unwrap();
        let mut out = Tensor::zeros([0]).unwrap();
        let mut p = cpu::MatmulPlan::with_workspace_limit(cpu, &empty, &b, &empty, 0).unwrap();
        assert_eq!(p.strategy(), Route::Empty);
        p.run(
            &t.view(&empty).unwrap(),
            &t.view(&b).unwrap(),
            &mut out.view_mut(&empty).unwrap(),
        )
        .unwrap();
    }
}
#[test]
fn cpu_prepared_matmul_saved_versions_and_aliased_view_composition() {
    for cpu in support::backends() {
        let a = Layout::from_strides([3, 3], [0, 1], 1).unwrap();
        let b = Layout::from_strides([3, 3], [1, 1], 0).unwrap();
        let dense = Layout::contiguous([3, 3]).unwrap();
        let base = Tensor::from_vec([7], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).unwrap();
        let dy = buffer(&dense, 1);
        let mut grad = Tensor::zeros([7]).unwrap();
        for side in SIDES {
            let mut local = Tensor::zeros([3, 3]).unwrap();
            cpu::MatmulBackward::new(cpu, side, &a, &b, &dense, &dense)
                .unwrap()
                .run(
                    Some(&base.view(&a).unwrap()),
                    Some(&base.view(&b).unwrap()),
                    &dy.view(&dense).unwrap(),
                    &mut local.view_mut(&dense).unwrap(),
                    G::Overwrite,
                )
                .unwrap();
            let original = if side == Side::Left { &a } else { &b };
            cpu::ViewBackward::new([7], original, &dense, 65536)
                .unwrap()
                .run(
                    &local.view(&dense).unwrap(),
                    &mut grad,
                    if side == Side::Left {
                        G::Overwrite
                    } else {
                        G::Accumulate
                    },
                )
                .unwrap();
        }
        let objective = |x: &[f64]| {
            let aa: Vec<_> = addresses(&a).into_iter().map(|i| x[i]).collect();
            let bb: Vec<_> = addresses(&b).into_iter().map(|i| x[i]).collect();
            forward([3, 3, 3], &aa, &bb)
                .iter()
                .zip(dy.data())
                .map(|(y, &d)| y * d as f64)
                .sum::<f64>()
        };
        for i in 0..7 {
            let mut x: Vec<_> = base.data().iter().map(|&x| x as f64).collect();
            x[i] += 1e-4;
            let plus = objective(&x);
            x[i] -= 2e-4;
            close(grad.data()[i], (plus - objective(&x)) / 2e-4);
        }
        let saved_layout = std::sync::Arc::new(a.clone());
        let owner = Shared::new(base).unwrap();
        let saved = owner.save(&saved_layout).unwrap();
        let mut p = cpu::MatmulBackward::new(cpu, Side::Right, &a, &b, &dense, &dense).unwrap();
        let mut out = Tensor::zeros([3, 3]).unwrap();
        {
            let read = saved.read().unwrap();
            p.run(
                Some(&read.view().unwrap()),
                None,
                &dy.view(&dense).unwrap(),
                &mut out.view_mut(&dense).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        }
        owner.write().unwrap().data_mut()[0] = 99.;
        let before = bits(&out);
        let result = (|| -> fast_audio::Result<()> {
            let read = saved.read()?;
            p.run(
                Some(&read.view()?),
                None,
                &dy.view(&dense)?,
                &mut out.view_mut(&dense)?,
                G::Overwrite,
            )
        })();
        assert!(result.is_err());
        assert_eq!(bits(&out), before);
    }
}
#[test]
fn cpu_prepared_matmul_miri_signed_overlap_and_bounds() {
    let c = Case {
        dims: [2, 3, 4],
        a: layout(2, 4, 7),
        b: layout(4, 3, 5),
        y: layout(2, 3, 5),
        dy: layout(2, 3, 6),
        da: layout(2, 4, 4),
        db: layout(4, 3, 5),
        direct: false,
    };
    exercise(Cpu::Reference, &c, 0, &mut Exporter::new(None));
}

#[test]
fn cpu_prepared_matmul_manual_training_matches_fp64_updates() {
    use fast_audio::{BinaryOp as B, ReduceOp, ReduceSpec, UnaryOp as U};
    macro_rules! v {
        ($t:ident,$l:ident) => {
            $t.view(&$l).unwrap()
        };
    }
    macro_rules! vm {
        ($t:ident,$l:ident) => {
            $t.view_mut(&$l).unwrap()
        };
    }
    let mut export = Exporter::new(Some("training-cpu"));
    for cpu in support::backends() {
        let [x, target, mut weights, rate] = training_fixture();
        let lx = Layout::contiguous([3, 2]).unwrap();
        let ly = lx.clone();
        let lw = Layout::contiguous([2, 2]).unwrap();
        let scalar = Layout::contiguous([]).unwrap();
        let spec = ReduceSpec::new(ReduceOp::Mean, ly.shape(), &[0, 1], false).unwrap();
        let mut product = cpu::MatmulPlan::new(cpu, &lx, &lw, &ly).unwrap();
        let difference = cpu::BinaryPlan::new(B::Sub, &ly, &ly, &ly).unwrap();
        let square = cpu::UnaryPlan::new(U::Square, &ly, &ly).unwrap();
        let mean = cpu::ReducePlan::new(&spec, &ly, &scalar).unwrap();
        let dmean = cpu::ReduceBackward::new(&spec, &scalar, &ly).unwrap();
        let dsquare = cpu::UnaryBackward::new(U::Square, &ly, &ly, &ly).unwrap();
        let ddifference = cpu::BinaryBackward::new(B::Sub, Side::Left, &ly, &ly, &ly, &ly).unwrap();
        let mut dweights = cpu::MatmulBackward::new(cpu, Side::Right, &lx, &lw, &ly, &lw).unwrap();
        assert_eq!(dweights.input_dependencies(), [true, false]);
        let scale = cpu::BinaryPlan::new(B::Mul, &lw, &scalar, &lw).unwrap();
        let update = cpu::BinaryPlan::new(B::Sub, &lw, &lw, &lw).unwrap();
        let copy = cpu::CopyPlan::new(&lw, &lw).unwrap();
        let mut y = Tensor::zeros([3, 2]).unwrap();
        let mut residual = y.clone();
        let mut squared = y.clone();
        let mut gsquare = y.clone();
        let mut gresidual = y.clone();
        let mut gy = y.clone();
        let mut loss = Tensor::zeros([]).unwrap();
        let one = Tensor::from_vec([], vec![1.]).unwrap();
        let mut gw = Tensor::zeros([2, 2]).unwrap();
        let mut scaled = gw.clone();
        let mut next = gw.clone();
        let mut control: Vec<_> = weights.data().iter().map(|&v| v as f64).collect();
        let original = (bits(&x), bits(&target), bits(&rate));
        let mut first = 0.;
        for step in 0..8 {
            let before = weights.clone();
            for (&a, &b) in before.data().iter().zip(&control) {
                close(a, b);
            }
            let (expected_loss, expected_grad) =
                training_control(&mut control, &x, &target, rate.data()[0] as f64);
            support::no_allocations(|| {
                product
                    .run(&v!(x, lx), &v!(weights, lw), &mut vm!(y, ly))
                    .unwrap();
                difference
                    .run(&v!(y, ly), &v!(target, ly), &mut vm!(residual, ly))
                    .unwrap();
                square
                    .run(&v!(residual, ly), &mut vm!(squared, ly))
                    .unwrap();
                mean.run(&v!(squared, ly), &mut vm!(loss, scalar)).unwrap();
                dmean
                    .run(&v!(one, scalar), &mut vm!(gsquare, ly), G::Overwrite)
                    .unwrap();
                dsquare
                    .run(
                        Some(&v!(residual, ly)),
                        &v!(gsquare, ly),
                        &mut vm!(gresidual, ly),
                        G::Overwrite,
                    )
                    .unwrap();
                ddifference
                    .run(
                        None,
                        None,
                        &v!(gresidual, ly),
                        &mut vm!(gy, ly),
                        G::Overwrite,
                    )
                    .unwrap();
                dweights
                    .run(
                        Some(&v!(x, lx)),
                        None,
                        &v!(gy, ly),
                        &mut vm!(gw, lw),
                        G::Overwrite,
                    )
                    .unwrap();
                scale
                    .run(&v!(gw, lw), &v!(rate, scalar), &mut vm!(scaled, lw))
                    .unwrap();
                update
                    .run(&v!(weights, lw), &v!(scaled, lw), &mut vm!(next, lw))
                    .unwrap();
                copy.run(&v!(next, lw), &mut vm!(weights, lw)).unwrap();
            });
            close(loss.data()[0], expected_loss);
            support::close(gw.data(), &expected_grad, 2e-5, 2e-5);
            support::close(weights.data(), &control, 2e-5, 2e-5);
            if step == 0 {
                first = loss.data()[0];
            }
            let backend = match cpu {
                Cpu::Reference => "reference",
                #[cfg(feature = "mkl")]
                Cpu::Mkl { .. } => "mkl",
            };
            export_training(
                &mut export,
                backend,
                step,
                [&x, &target, &before, &rate],
                &loss,
                &gw,
                &weights,
            );
        }
        assert!(loss.data()[0] < first);
        assert_eq!((bits(&x), bits(&target), bits(&rate)), original);
    }
    export.finish();
}
