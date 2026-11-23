use fast_audio::{
    BinaryOp as B, BinaryOperand as Side, GradientMode as G, Layout, Shared, Tensor, UnaryOp as U,
    cpu,
};
#[path = "support/pointwise_backward.rs"]
mod contract;
#[allow(
    dead_code,
    reason = "Gradient fixtures use their own independent references."
)]
mod support;
use contract::*;

fn exercise(c: &Case, salt: usize) {
    let p = CpuPlan::new(c);
    for change in 0..2 {
        let (left, right, dy) = payloads(c, salt + change);
        let source_bits = (bits(&left), right.as_ref().map(bits), bits(&dy));
        let left_view = left.view(&c.left).unwrap();
        let right_view = right
            .as_ref()
            .zip(c.right.as_ref())
            .map(|(r, l)| r.view(l).unwrap());
        let required = c.dependencies();
        let inputs = [
            if required[0] || change == 1 {
                Some(&left_view)
            } else {
                None
            },
            if required[1] || change == 1 {
                right_view.as_ref()
            } else {
                None
            },
        ];
        let expected = reference(c, &left, right.as_ref(), &dy);
        for mode in [G::Overwrite, G::Accumulate] {
            let initial = seed(&c.gradient);
            let mut dx = initial.clone();
            support::no_allocations(|| {
                p.run(
                    inputs,
                    &dy.view(&c.upstream).unwrap(),
                    &mut dx.view_mut(&c.gradient).unwrap(),
                    mode,
                )
                .unwrap()
            });
            verify(c, &expected, &initial, &dx, mode);
            let mut repeat = initial.clone();
            p.run(
                inputs,
                &dy.view(&c.upstream).unwrap(),
                &mut repeat.view_mut(&c.gradient).unwrap(),
                mode,
            )
            .unwrap();
            assert_eq!(bits(&repeat), bits(&dx));
            if mode == G::Overwrite {
                finite_difference(c, &left, right.as_ref(), &dy, &dx);
            } else {
                let before = dx.clone();
                p.run(
                    inputs,
                    &dy.view(&c.upstream).unwrap(),
                    &mut dx.view_mut(&c.gradient).unwrap(),
                    mode,
                )
                .unwrap();
                verify(c, &expected, &before, &dx, mode);
            }
        }
        assert_eq!(
            (bits(&left), right.as_ref().map(bits), bits(&dy)),
            source_bits
        );
    }
}
#[test]
fn cpu_pointwise_vjps_cover_all_operations_and_broadcast_layouts() {
    for (i, c) in cases().iter().enumerate() {
        exercise(c, i);
    }
}

fn shifted(l: &Layout) -> Layout {
    Layout::from_strides(l.shape().dims(), l.strides(), l.offset() + 1).unwrap()
}
#[test]
fn cpu_pointwise_gradients_reject_missing_and_changed_bindings_before_writes() {
    for (i, c) in cases().iter().enumerate() {
        let p = CpuPlan::new(c);
        let (left, right, dy) = payloads(c, i);
        let lv = left.view(&c.left).unwrap();
        let rv = right
            .as_ref()
            .zip(c.right.as_ref())
            .map(|(r, l)| r.view(l).unwrap());
        let initial = seed(&c.gradient);
        let mut out = initial.clone();
        for a in 0..2 {
            if c.dependencies()[a] {
                let mut bindings = [Some(&lv), rv.as_ref()];
                bindings[a] = None;
                assert!(
                    p.run(
                        bindings,
                        &dy.view(&c.upstream).unwrap(),
                        &mut out.view_mut(&c.gradient).unwrap(),
                        G::Overwrite
                    )
                    .is_err()
                );
                assert_eq!(bits(&out), bits(&initial));
            }
        }
        let wrong = shifted(&c.left);
        assert!(
            p.run(
                [Some(&left.view(&wrong).unwrap()), rv.as_ref()],
                &dy.view(&c.upstream).unwrap(),
                &mut out.view_mut(&c.gradient).unwrap(),
                G::Accumulate
            )
            .is_err()
        );
        assert_eq!(bits(&out), bits(&initial));
        if let Some(r) = c.right.as_ref() {
            let wrong = shifted(r);
            assert!(
                p.run(
                    [
                        Some(&lv),
                        Some(&right.as_ref().unwrap().view(&wrong).unwrap())
                    ],
                    &dy.view(&c.upstream).unwrap(),
                    &mut out.view_mut(&c.gradient).unwrap(),
                    G::Overwrite
                )
                .is_err()
            );
            assert_eq!(bits(&out), bits(&initial));
        }
        let wrong = shifted(&c.upstream);
        assert!(
            p.run(
                [Some(&lv), rv.as_ref()],
                &dy.view(&wrong).unwrap(),
                &mut out.view_mut(&c.gradient).unwrap(),
                G::Overwrite
            )
            .is_err()
        );
        let wrong = shifted(&c.gradient);
        assert!(
            p.run(
                [Some(&lv), rv.as_ref()],
                &dy.view(&c.upstream).unwrap(),
                &mut out.view_mut(&wrong).unwrap(),
                G::Overwrite
            )
            .is_err()
        );
        assert_eq!(bits(&out), bits(&initial));
    }
    let a = Layout::contiguous([2, 3]).unwrap();
    let b = Layout::contiguous([3, 2]).unwrap();
    let overlap = Layout::from_strides([2, 3], [0, 1], 0).unwrap();
    for op in UNARY {
        assert!(cpu::UnaryBackward::new(op, &a, &b, &a).is_err());
        assert!(cpu::UnaryBackward::new(op, &a, &a, &b).is_err());
        assert!(cpu::UnaryBackward::new(op, &a, &a, &overlap).is_err());
    }
    let incompatible = Layout::contiguous([4]).unwrap();
    for op in BINARY {
        for side in SIDES {
            assert!(cpu::BinaryBackward::new(op, side, &a, &incompatible, &a, &a).is_err());
            assert!(cpu::BinaryBackward::new(op, side, &a, &a, &b, &a).is_err());
            assert!(cpu::BinaryBackward::new(op, side, &a, &a, &a, &overlap).is_err());
            assert!(cpu::BinaryBackward::new(op, side, &a, &a, &a, &b).is_err());
        }
    }
}

#[test]
fn cpu_pointwise_gradient_special_values_and_ordered_broadcast_trees() {
    special_binary(|op, side, a, b, d| {
        let l = Layout::contiguous([d.len()]).unwrap();
        let a = Tensor::from_vec([d.len()], a.to_vec()).unwrap();
        let b = Tensor::from_vec([d.len()], b.to_vec()).unwrap();
        let dy = Tensor::from_vec([d.len()], d.to_vec()).unwrap();
        let mut dx = Tensor::zeros([d.len()]).unwrap();
        cpu::BinaryBackward::new(op, side, &l, &l, &l, &l)
            .unwrap()
            .run(
                Some(&a.view(&l).unwrap()),
                Some(&b.view(&l).unwrap()),
                &dy.view(&l).unwrap(),
                &mut dx.view_mut(&l).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        dx.data().to_vec()
    });
    special_unary(|op, x, d| {
        let layout = Layout::contiguous([x.len()]).unwrap();
        let x = Tensor::from_vec([x.len()], x.to_vec()).unwrap();
        let dy = Tensor::from_vec([d.len()], d.to_vec()).unwrap();
        let mut dx = Tensor::zeros([d.len()]).unwrap();
        let p = cpu::UnaryBackward::new(op, &layout, &layout, &layout).unwrap();
        let input = x.view(&layout).unwrap();
        p.run(
            if p.input_required() {
                Some(&input)
            } else {
                None
            },
            &dy.view(&layout).unwrap(),
            &mut dx.view_mut(&layout).unwrap(),
            G::Overwrite,
        )
        .unwrap();
        dx.data().to_vec()
    });
    let l = Layout::contiguous([4]).unwrap();
    let values = [
        -0.,
        f32::from_bits(0x7fc54321),
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let dy = Tensor::from_vec([4], values.to_vec()).unwrap();
    for op in [B::Add, B::Sub] {
        for side in SIDES {
            let mut dx = Tensor::zeros([4]).unwrap();
            cpu::BinaryBackward::new(op, side, &l, &l, &l, &l)
                .unwrap()
                .run(
                    None,
                    None,
                    &dy.view(&l).unwrap(),
                    &mut dx.view_mut(&l).unwrap(),
                    G::Overwrite,
                )
                .unwrap();
            for (&got, value) in dx.data().iter().zip(values) {
                assert_eq!(
                    got.to_bits(),
                    value.to_bits()
                        ^ if op == B::Sub && side == Side::Right {
                            0x80000000
                        } else {
                            0
                        }
                );
            }
        }
    }
    for (count, points, expected) in cancellation() {
        let scalar = Layout::contiguous([]).unwrap();
        let common = Layout::contiguous([count]).unwrap();
        let ones = Tensor::from_vec([count], vec![1.; count]).unwrap();
        let ones = ones.view(&common).unwrap();
        let mut dy = Tensor::zeros([count]).unwrap();
        for (i, v) in points {
            dy.data_mut()[i] = v;
        }
        for op in [B::Add, B::Mul] {
            for side in SIDES {
                let (left, right) = if side == Side::Left {
                    (&scalar, &common)
                } else {
                    (&common, &scalar)
                };
                let p = cpu::BinaryBackward::new(op, side, left, right, &common, &scalar).unwrap();
                let mut out = Tensor::zeros([]).unwrap();
                for _ in 0..3 {
                    p.run(
                        if op == B::Mul && side == Side::Right {
                            Some(&ones)
                        } else {
                            None
                        },
                        if op == B::Mul && side == Side::Left {
                            Some(&ones)
                        } else {
                            None
                        },
                        &dy.view(&common).unwrap(),
                        &mut out.view_mut(&scalar).unwrap(),
                        G::Overwrite,
                    )
                    .unwrap();
                    assert_eq!(out.data(), &[expected]);
                }
            }
        }
    }
    // The other operand prevents coalescing even though upstream is contiguous.
    let common = Layout::contiguous([2, 17]).unwrap();
    let operand = Layout::from_strides([2, 17], [1, 2], 0).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let mut x = Tensor::from_vec([34], vec![1.; 34]).unwrap();
    x.data_mut()[0] = 1e20;
    x.data_mut()[30] = 1.;
    x.data_mut()[31] = -1e20;
    let dy = Tensor::from_vec([34], vec![1.; 34]).unwrap();
    let mut out = Tensor::zeros([]).unwrap();
    cpu::BinaryBackward::new(B::Mul, Side::Right, &operand, &scalar, &common, &scalar)
        .unwrap()
        .run(
            Some(&x.view(&operand).unwrap()),
            None,
            &dy.view(&common).unwrap(),
            &mut out.view_mut(&scalar).unwrap(),
            G::Overwrite,
        )
        .unwrap();
    // Canonical row-major contributions: lane 0 cancels positions 0/32;
    // the remaining 32 ones occupy the other lanes.
    assert_eq!(out.data(), &[32.]);
}

#[test]
fn cpu_pointwise_gradient_empty_accumulation_preserves_all_bits() {
    for c in cases()
        .into_iter()
        .filter(|c| c.upstream.shape().is_empty() && !c.gradient.shape().is_empty())
    {
        let p = CpuPlan::new(&c);
        let dy = buffer(&c.upstream, 0, false);
        let mut initial = seed(&c.gradient);
        for (i, a) in addresses(&c.gradient).into_iter().enumerate() {
            initial.data_mut()[a] =
                f32::from_bits(if i % 2 == 0 { 0x80000000 } else { 0x7fc54321 });
        }
        let mut out = initial.clone();
        support::no_allocations(|| {
            p.run(
                [None; 2],
                &dy.view(&c.upstream).unwrap(),
                &mut out.view_mut(&c.gradient).unwrap(),
                G::Accumulate,
            )
            .unwrap()
        });
        assert_eq!(bits(&initial), bits(&out));
        p.run(
            [None; 2],
            &dy.view(&c.upstream).unwrap(),
            &mut out.view_mut(&c.gradient).unwrap(),
            G::Overwrite,
        )
        .unwrap();
        for a in addresses(&c.gradient) {
            assert_eq!(out.data()[a].to_bits(), 0);
        }
        holes(&c.gradient, &initial, &out);
    }
    let empty = Layout::contiguous([usize::MAX, usize::MAX, 0]).unwrap();
    let no_storage = Tensor::zeros([0]).unwrap();
    let mut dx = Tensor::zeros([0]).unwrap();
    for op in UNARY {
        let p = cpu::UnaryBackward::new(op, &empty, &empty, &empty).unwrap();
        assert!(!p.input_required());
        for mode in [G::Overwrite, G::Accumulate] {
            p.run(
                None,
                &no_storage.view(&empty).unwrap(),
                &mut dx.view_mut(&empty).unwrap(),
                mode,
            )
            .unwrap();
        }
    }
}

#[test]
fn cpu_pointwise_gradient_resources_threads_and_saved_versions() {
    let scalar = Layout::contiguous([]).unwrap();
    let huge = Layout::from_strides([1_000_000_001], [0], 0).unwrap();
    let p = cpu::BinaryBackward::new(B::Mul, Side::Left, &scalar, &huge, &huge, &scalar).unwrap();
    assert_eq!(p.contraction().reduction_len(), Some(1_000_000_001));
    assert_eq!(p.workspace_bytes(), 0);
    assert!(p.host_metadata_bytes() < 1024);
    let c = Case {
        op: Op::Binary(B::Mul, Side::Left),
        left: padded(&[1, 3], 0),
        right: Some(padded(&[2, 3], 1)),
        upstream: padded(&[2, 3], 2),
        gradient: padded(&[1, 3], 3),
    };
    let p = CpuPlan::new(&c);
    let unary = cpu::UnaryBackward::new(
        U::Square,
        c.right.as_ref().unwrap(),
        &c.upstream,
        c.right.as_ref().unwrap(),
    )
    .unwrap();
    let (left, right, dy) = payloads(&c, 0);
    let expected = reference(&c, &left, right.as_ref(), &dy);
    std::thread::scope(|scope| {
        for _ in 0..3 {
            let (c, p, unary, right, dy, expected) =
                (&c, &p, &unary, right.as_ref().unwrap(), &dy, &expected);
            scope.spawn(move || {
                let initial = seed(&c.gradient);
                let mut dx = initial.clone();
                support::no_allocations(|| {
                    p.run(
                        [None, Some(&right.view(c.right.as_ref().unwrap()).unwrap())],
                        &dy.view(&c.upstream).unwrap(),
                        &mut dx.view_mut(&c.gradient).unwrap(),
                        G::Overwrite,
                    )
                    .unwrap()
                });
                verify(c, expected, &initial, &dx, G::Overwrite);
                let layout = c.right.as_ref().unwrap();
                let initial = seed(layout);
                let mut dx = initial.clone();
                support::no_allocations(|| {
                    unary
                        .run(
                            Some(&right.view(layout).unwrap()),
                            &dy.view(&c.upstream).unwrap(),
                            &mut dx.view_mut(layout).unwrap(),
                            G::Overwrite,
                        )
                        .unwrap()
                });
                for (a, d) in addresses(layout).into_iter().zip(addresses(&c.upstream)) {
                    close(
                        dx.data()[a],
                        2. * right.data()[a] as f64 * dy.data()[d] as f64,
                    );
                }
                holes(layout, &initial, &dx);
            });
        }
    });
    let layout = std::sync::Arc::new(padded(&[2, 3], 1));
    let owner = Shared::new(buffer(&layout, 1, true)).unwrap();
    let saved = owner.save(&layout).unwrap();
    let dy = buffer(&layout, 2, false);
    let mut dx = seed(&layout);
    let p = cpu::UnaryBackward::new(U::Square, &layout, &layout, &layout).unwrap();
    {
        let read = saved.read().unwrap();
        support::no_allocations(|| {
            p.run(
                Some(&read.view().unwrap()),
                &dy.view(&layout).unwrap(),
                &mut dx.view_mut(&layout).unwrap(),
                G::Overwrite,
            )
            .unwrap()
        });
        for a in addresses(&layout) {
            close(
                dx.data()[a],
                2. * read.storage().data()[a] as f64 * dy.data()[a] as f64,
            );
        }
    }
    owner.write().unwrap().data_mut()[0] = 99.;
    let before = bits(&dx);
    let result = (|| -> fast_audio::Result<()> {
        let read = saved.read()?;
        p.run(
            Some(&read.view()?),
            &dy.view(&layout)?,
            &mut dx.view_mut(&layout)?,
            G::Overwrite,
        )
    })();
    assert!(result.is_err());
    assert_eq!(bits(&dx), before);
}

#[test]
fn cpu_pointwise_gradients_compose_with_aliased_forward_views() {
    let base = Tensor::from_vec([6], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    let dense = Layout::contiguous([3, 4]).unwrap();
    let dy = buffer(&dense, 4, false);
    for source in [
        Layout::from_strides([3, 4], [1, 1], 0).unwrap(),
        Layout::from_strides([3, 4], [0, 1], 1).unwrap(),
    ] {
        let mut local = Tensor::zeros([3, 4]).unwrap();
        cpu::UnaryBackward::new(U::Square, &source, &dense, &dense)
            .unwrap()
            .run(
                Some(&base.view(&source).unwrap()),
                &dy.view(&dense).unwrap(),
                &mut local.view_mut(&dense).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        let mut dx = Tensor::zeros([6]).unwrap();
        cpu::ViewBackward::new([6], &source, &dense, 4096)
            .unwrap()
            .run(&local.view(&dense).unwrap(), &mut dx, G::Overwrite)
            .unwrap();
        let mut expected = [0.; 6];
        for row in 0..3 {
            for col in 0..4 {
                let a = address(&source, &[row, col]);
                expected[a] += 2. * base.data()[a] as f64 * dy.data()[row * 4 + col] as f64;
            }
        }
        support::close(dx.data(), &expected, 2e-5, 2e-5);
    }
    let left = Layout::from_strides([3, 1], [1, 0], 1).unwrap();
    let right = Layout::from_strides([1, 4], [0, 1], 0).unwrap();
    let mut dx = Tensor::zeros([6]).unwrap();
    for side in SIDES {
        let original = if side == Side::Left { &left } else { &right };
        let grad = Layout::contiguous(original.shape().dims()).unwrap();
        let mut local = Tensor::zeros(grad.shape().dims()).unwrap();
        cpu::BinaryBackward::new(B::Mul, side, &left, &right, &dense, &grad)
            .unwrap()
            .run(
                Some(&base.view(&left).unwrap()),
                Some(&base.view(&right).unwrap()),
                &dy.view(&dense).unwrap(),
                &mut local.view_mut(&grad).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        cpu::ViewBackward::new([6], original, &grad, 4096)
            .unwrap()
            .run(
                &local.view(&grad).unwrap(),
                &mut dx,
                if side == Side::Left {
                    G::Overwrite
                } else {
                    G::Accumulate
                },
            )
            .unwrap();
    }
    let objective = |x: &[f64]| {
        (0..3)
            .flat_map(|row| (0..4).map(move |col| (row, col)))
            .map(|(row, col)| x[row + 1] * x[col] * dy.data()[row * 4 + col] as f64)
            .sum::<f64>()
    };
    for i in 0..6 {
        let mut x: Vec<_> = base.data().iter().map(|&x| x as f64).collect();
        x[i] += 1e-4;
        let plus = objective(&x);
        x[i] -= 2e-4;
        close(dx.data()[i], (plus - objective(&x)) / 2e-4);
    }
}

#[test]
fn cpu_pointwise_gradient_miri_signed_alias_bounds() {
    let rows = Case {
        op: Op::Unary(U::Square),
        left: Layout::from_strides([2, 17], [-23, 1], 23).unwrap(),
        right: None,
        upstream: Layout::from_strides([2, 17], [0, -1], 16).unwrap(),
        gradient: padded(&[2, 17], 3),
    };
    exercise(&rows, 7);
    for op in [
        Op::Unary(U::Neg),
        Op::Unary(U::Square),
        Op::Binary(B::Mul, Side::Left),
        Op::Binary(B::Div, Side::Right),
    ] {
        let left = Layout::from_strides([2, 1], [-1, 0], 2).unwrap();
        let right = if matches!(op, Op::Unary(_)) {
            None
        } else {
            Some(Layout::from_strides([1, 3], [0, 1], 1).unwrap())
        };
        let common = if right.is_some() {
            vec![2, 3]
        } else {
            vec![2, 1]
        };
        let target = if matches!(op, Op::Binary(_, Side::Right)) {
            right.as_ref().unwrap()
        } else {
            &left
        };
        let c = Case {
            op,
            upstream: Layout::from_strides(&common, [0, -1], common[1] - 1).unwrap(),
            gradient: padded(target.shape().dims(), 2),
            left,
            right,
        };
        exercise(&c, 0);
    }
    for op in [Op::Unary(U::Square), Op::Binary(B::Div, Side::Right)] {
        let c = Case {
            op,
            left: Layout::from_strides([2, 3], [-1, 3], 1).unwrap(),
            right: if matches!(op, Op::Unary(_)) {
                None
            } else {
                Some(Layout::from_strides([2, 3], [0, 1], 0).unwrap())
            },
            upstream: Layout::from_strides([2, 3], [0, -1], 2).unwrap(),
            gradient: Layout::from_strides([2, 3], [1, -3], 6).unwrap(),
        };
        exercise(&c, 2);
    }
}
