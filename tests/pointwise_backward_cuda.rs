#![cfg(feature = "cuda")]
use fast_audio::{
    BinaryOp as B, BinaryOperand as Side, GradientMode as G, Layout, ReduceStrategy, Shared,
    Tensor, UnaryOp as U,
    cuda::{self, Cuda, DeviceView, DeviceViewMut},
};
#[path = "support/pointwise_backward.rs"]
mod contract;
#[allow(
    dead_code,
    reason = "Gradient fixtures use their own independent references."
)]
mod support;
use contract::*;

enum Plan {
    Unary(cuda::UnaryBackward),
    Binary(cuda::BinaryBackward),
}
impl Plan {
    fn new(cuda: &Cuda, c: &Case) -> Self {
        let (result, shared, bytes, host, strategy) = match c.op {
            Op::Unary(op) => {
                let p =
                    cuda::UnaryBackward::new(cuda, op, &c.left, &c.upstream, &c.gradient).unwrap();
                assert_eq!(p.operation(), op);
                assert_eq!(p.input_layout(), &c.left);
                assert_eq!(p.upstream_layout(), &c.upstream);
                assert_eq!(p.destination_layout(), &c.gradient);
                assert_eq!([p.input_required(), false], c.dependencies());
                let resources = (
                    p.shared_memory_bytes_per_block(),
                    p.workspace_bytes(),
                    p.host_metadata_bytes(),
                    p.contraction().strategy(),
                );
                (
                    Self::Unary(p),
                    resources.0,
                    resources.1,
                    resources.2,
                    resources.3,
                )
            }
            Op::Binary(op, side) => {
                let p = cuda::BinaryBackward::new(
                    cuda,
                    op,
                    side,
                    &c.left,
                    c.right.as_ref().unwrap(),
                    &c.upstream,
                    &c.gradient,
                )
                .unwrap();
                assert_eq!(p.operation(), op);
                assert_eq!(p.operand(), side);
                assert_eq!(p.left_layout(), &c.left);
                assert_eq!(p.right_layout(), c.right.as_ref().unwrap());
                assert_eq!(p.upstream_layout(), &c.upstream);
                assert_eq!(p.destination_layout(), &c.gradient);
                assert_eq!(p.input_dependencies(), c.dependencies());
                let resources = (
                    p.shared_memory_bytes_per_block(),
                    p.workspace_bytes(),
                    p.host_metadata_bytes(),
                    p.contraction().strategy(),
                );
                (
                    Self::Binary(p),
                    resources.0,
                    resources.1,
                    resources.2,
                    resources.3,
                )
            }
        };
        assert_eq!(shared > 0, strategy == ReduceStrategy::Tree256);
        assert!(shared <= 1024 && bytes < 4096 && host < 200_000);
        result
    }
    fn run(
        &self,
        inputs: [Option<&DeviceView<'_>>; 2],
        dy: &DeviceView<'_>,
        dx: &mut DeviceViewMut<'_>,
        mode: G,
    ) -> fast_audio::Result<()> {
        match self {
            Self::Unary(p) => p.run(inputs[0], dy, dx, mode),
            Self::Binary(p) => p.run(inputs[0], inputs[1], dy, dx, mode),
        }
    }
}

#[test]
fn cuda_pointwise_vjps_cover_all_operations_and_broadcast_layouts() {
    let cuda = Cuda::new(0).unwrap();
    for (i, c) in cases().iter().enumerate() {
        let p = Plan::new(&cuda, c);
        let control = CpuPlan::new(c);
        for change in 0..2 {
            let (left, right, dy) = payloads(c, i + change);
            let a = cuda.upload(&left).unwrap();
            let b = right.as_ref().map(|t| cuda.upload(t).unwrap());
            let incoming = cuda.upload(&dy).unwrap();
            let av = a.view(&c.left).unwrap();
            let bv = b
                .as_ref()
                .zip(c.right.as_ref())
                .map(|(b, l)| b.view(l).unwrap());
            let deps = c.dependencies();
            let inputs = [
                if deps[0] || change == 1 {
                    Some(&av)
                } else {
                    None
                },
                if deps[1] || change == 1 {
                    bv.as_ref()
                } else {
                    None
                },
            ];
            let expected = reference(c, &left, right.as_ref(), &dy);
            for mode in [G::Overwrite, G::Accumulate] {
                let initial = seed(&c.gradient);
                let mut dx = cuda.upload(&initial).unwrap();
                support::no_allocations(|| {
                    p.run(
                        inputs,
                        &incoming.view(&c.upstream).unwrap(),
                        &mut dx.view_mut(&c.gradient).unwrap(),
                        mode,
                    )
                    .unwrap()
                });
                let actual = cuda.download(&dx).unwrap();
                verify(c, &expected, &initial, &actual, mode);
                let mut repeat = cuda.upload(&initial).unwrap();
                p.run(
                    inputs,
                    &incoming.view(&c.upstream).unwrap(),
                    &mut repeat.view_mut(&c.gradient).unwrap(),
                    mode,
                )
                .unwrap();
                assert_eq!(bits(&cuda.download(&repeat).unwrap()), bits(&actual));
                let mut cpu_dx = initial.clone();
                control
                    .run(
                        [
                            Some(&left.view(&c.left).unwrap()),
                            right
                                .as_ref()
                                .zip(c.right.as_ref())
                                .map(|(t, l)| t.view(l).unwrap())
                                .as_ref(),
                        ],
                        &dy.view(&c.upstream).unwrap(),
                        &mut cpu_dx.view_mut(&c.gradient).unwrap(),
                        mode,
                    )
                    .unwrap();
                if matches!(
                    c.op,
                    Op::Binary(_, _)
                        | Op::Unary(U::Neg | U::Abs | U::Square | U::Reciprocal | U::Log | U::Relu)
                ) {
                    assert_eq!(bits(&cpu_dx), bits(&actual));
                } else {
                    for a in addresses(&c.gradient) {
                        close(actual.data()[a], cpu_dx.data()[a] as f64);
                    }
                }
                if mode == G::Overwrite {
                    finite_difference(c, &left, right.as_ref(), &dy, &actual);
                } else {
                    p.run(
                        inputs,
                        &incoming.view(&c.upstream).unwrap(),
                        &mut dx.view_mut(&c.gradient).unwrap(),
                        mode,
                    )
                    .unwrap();
                    verify(c, &expected, &actual, &cuda.download(&dx).unwrap(), mode);
                }
            }
            assert_eq!(bits(&cuda.download(&a).unwrap()), bits(&left));
            if let (Some(b), Some(right)) = (&b, &right) {
                assert_eq!(bits(&cuda.download(b).unwrap()), bits(right));
            }
            assert_eq!(bits(&cuda.download(&incoming).unwrap()), bits(&dy));
        }
    }
}

#[test]
fn cuda_pointwise_gradient_special_values_and_ordered_broadcast_trees() {
    let cuda = Cuda::new(0).unwrap();
    special_unary(|op, x, d| {
        let l = Layout::contiguous([x.len()]).unwrap();
        let a = cuda
            .upload(&Tensor::from_vec([x.len()], x.to_vec()).unwrap())
            .unwrap();
        let incoming = cuda
            .upload(&Tensor::from_vec([d.len()], d.to_vec()).unwrap())
            .unwrap();
        let mut dx = cuda.zeros([d.len()]).unwrap();
        let p = cuda::UnaryBackward::new(&cuda, op, &l, &l, &l).unwrap();
        let view = a.view(&l).unwrap();
        p.run(
            if p.input_required() {
                Some(&view)
            } else {
                None
            },
            &incoming.view(&l).unwrap(),
            &mut dx.view_mut(&l).unwrap(),
            G::Overwrite,
        )
        .unwrap();
        cuda.download(&dx).unwrap().data().to_vec()
    });
    special_binary(|op, side, a, b, d| {
        let l = Layout::contiguous([d.len()]).unwrap();
        let a = cuda
            .upload(&Tensor::from_vec([d.len()], a.to_vec()).unwrap())
            .unwrap();
        let b = cuda
            .upload(&Tensor::from_vec([d.len()], b.to_vec()).unwrap())
            .unwrap();
        let incoming = cuda
            .upload(&Tensor::from_vec([d.len()], d.to_vec()).unwrap())
            .unwrap();
        let mut dx = cuda.zeros([d.len()]).unwrap();
        cuda::BinaryBackward::new(&cuda, op, side, &l, &l, &l, &l)
            .unwrap()
            .run(
                Some(&a.view(&l).unwrap()),
                Some(&b.view(&l).unwrap()),
                &incoming.view(&l).unwrap(),
                &mut dx.view_mut(&l).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        cuda.download(&dx).unwrap().data().to_vec()
    });
    let l = Layout::contiguous([4]).unwrap();
    let values = [
        -0.,
        f32::from_bits(0x7fc54321),
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let incoming = cuda
        .upload(&Tensor::from_vec([4], values.to_vec()).unwrap())
        .unwrap();
    for op in [B::Add, B::Sub] {
        for side in SIDES {
            let mut dx = cuda.zeros([4]).unwrap();
            cuda::BinaryBackward::new(&cuda, op, side, &l, &l, &l, &l)
                .unwrap()
                .run(
                    None,
                    None,
                    &incoming.view(&l).unwrap(),
                    &mut dx.view_mut(&l).unwrap(),
                    G::Overwrite,
                )
                .unwrap();
            for (&got, value) in cuda.download(&dx).unwrap().data().iter().zip(values) {
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
        let ones = cuda
            .upload(&Tensor::from_vec([count], vec![1.; count]).unwrap())
            .unwrap();
        let view = ones.view(&common).unwrap();
        let mut dy = Tensor::zeros([count]).unwrap();
        for (i, value) in points {
            dy.data_mut()[i] = value;
        }
        let incoming = cuda.upload(&dy).unwrap();
        for op in [B::Add, B::Mul] {
            for side in SIDES {
                let (left, right) = if side == Side::Left {
                    (&scalar, &common)
                } else {
                    (&common, &scalar)
                };
                let p = cuda::BinaryBackward::new(&cuda, op, side, left, right, &common, &scalar)
                    .unwrap();
                let mut dx = cuda.zeros([]).unwrap();
                for _ in 0..3 {
                    p.run(
                        if op == B::Mul && side == Side::Right {
                            Some(&view)
                        } else {
                            None
                        },
                        if op == B::Mul && side == Side::Left {
                            Some(&view)
                        } else {
                            None
                        },
                        &incoming.view(&common).unwrap(),
                        &mut dx.view_mut(&scalar).unwrap(),
                        G::Overwrite,
                    )
                    .unwrap();
                    assert_eq!(cuda.download(&dx).unwrap().data(), &[expected]);
                }
            }
        }
    }
    let common = Layout::contiguous([2, 17]).unwrap();
    let operand = Layout::from_strides([2, 17], [1, 2], 0).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let mut x = Tensor::from_vec([34], vec![1.; 34]).unwrap();
    x.data_mut()[0] = 1e20;
    x.data_mut()[30] = 1.;
    x.data_mut()[31] = -1e20;
    let a = cuda.upload(&x).unwrap();
    let incoming = cuda
        .upload(&Tensor::from_vec([34], vec![1.; 34]).unwrap())
        .unwrap();
    let mut dx = cuda.zeros([]).unwrap();
    cuda::BinaryBackward::new(
        &cuda,
        B::Mul,
        Side::Right,
        &operand,
        &scalar,
        &common,
        &scalar,
    )
    .unwrap()
    .run(
        Some(&a.view(&operand).unwrap()),
        None,
        &incoming.view(&common).unwrap(),
        &mut dx.view_mut(&scalar).unwrap(),
        G::Overwrite,
    )
    .unwrap();
    assert_eq!(cuda.download(&dx).unwrap().data(), &[32.]);
}

#[test]
fn cuda_pointwise_gradient_empty_accumulation_preserves_all_bits() {
    let cuda = Cuda::new(0).unwrap();
    for c in cases()
        .into_iter()
        .filter(|c| c.upstream.shape().is_empty() && !c.gradient.shape().is_empty())
    {
        let p = Plan::new(&cuda, &c);
        let incoming = cuda.upload(&buffer(&c.upstream, 0, false)).unwrap();
        let mut initial = seed(&c.gradient);
        for (i, a) in addresses(&c.gradient).into_iter().enumerate() {
            initial.data_mut()[a] =
                f32::from_bits(if i % 2 == 0 { 0x80000000 } else { 0x7fc54321 });
        }
        let mut dx = cuda.upload(&initial).unwrap();
        support::no_allocations(|| {
            p.run(
                [None; 2],
                &incoming.view(&c.upstream).unwrap(),
                &mut dx.view_mut(&c.gradient).unwrap(),
                G::Accumulate,
            )
            .unwrap()
        });
        assert_eq!(bits(&cuda.download(&dx).unwrap()), bits(&initial));
        p.run(
            [None; 2],
            &incoming.view(&c.upstream).unwrap(),
            &mut dx.view_mut(&c.gradient).unwrap(),
            G::Overwrite,
        )
        .unwrap();
        let actual = cuda.download(&dx).unwrap();
        for a in addresses(&c.gradient) {
            assert_eq!(actual.data()[a].to_bits(), 0);
        }
        holes(&c.gradient, &initial, &actual);
    }
    let empty = Layout::contiguous([usize::MAX, usize::MAX, 0]).unwrap();
    let no_storage = cuda.zeros([0]).unwrap();
    let mut dx = cuda.zeros([0]).unwrap();
    for op in UNARY {
        let p = cuda::UnaryBackward::new(&cuda, op, &empty, &empty, &empty).unwrap();
        assert!(!p.input_required());
        assert_eq!(p.workspace_bytes(), 0);
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
fn cuda_pointwise_gradients_validate_dependencies_geometry_and_sessions() {
    let cuda = Cuda::new(0).unwrap();
    for (i, c) in cases().iter().enumerate() {
        let p = Plan::new(&cuda, c);
        let (left, right, dy) = payloads(c, i);
        let a = cuda.upload(&left).unwrap();
        let b = right.as_ref().map(|t| cuda.upload(t).unwrap());
        let incoming = cuda.upload(&dy).unwrap();
        let av = a.view(&c.left).unwrap();
        let bv = b
            .as_ref()
            .zip(c.right.as_ref())
            .map(|(b, l)| b.view(l).unwrap());
        let initial = seed(&c.gradient);
        let mut dx = cuda.upload(&initial).unwrap();
        for j in 0..2 {
            if c.dependencies()[j] {
                let mut inputs = [Some(&av), bv.as_ref()];
                inputs[j] = None;
                assert!(
                    p.run(
                        inputs,
                        &incoming.view(&c.upstream).unwrap(),
                        &mut dx.view_mut(&c.gradient).unwrap(),
                        G::Overwrite
                    )
                    .is_err()
                );
            }
        }
        let shifted = |l: &Layout| {
            Layout::from_strides(l.shape().dims(), l.strides(), l.offset() + 1).unwrap()
        };
        assert!(
            p.run(
                [Some(&a.view(&shifted(&c.left)).unwrap()), bv.as_ref()],
                &incoming.view(&c.upstream).unwrap(),
                &mut dx.view_mut(&c.gradient).unwrap(),
                G::Overwrite
            )
            .is_err()
        );
        if let (Some(b), Some(l)) = (&b, &c.right) {
            assert!(
                p.run(
                    [Some(&av), Some(&b.view(&shifted(l)).unwrap())],
                    &incoming.view(&c.upstream).unwrap(),
                    &mut dx.view_mut(&c.gradient).unwrap(),
                    G::Accumulate
                )
                .is_err()
            );
        }
        assert!(
            p.run(
                [Some(&av), bv.as_ref()],
                &incoming.view(&shifted(&c.upstream)).unwrap(),
                &mut dx.view_mut(&c.gradient).unwrap(),
                G::Overwrite
            )
            .is_err()
        );
        assert!(
            p.run(
                [Some(&av), bv.as_ref()],
                &incoming.view(&c.upstream).unwrap(),
                &mut dx.view_mut(&shifted(&c.gradient)).unwrap(),
                G::Overwrite
            )
            .is_err()
        );
        assert_eq!(bits(&cuda.download(&dx).unwrap()), bits(&initial));
    }
    let other = Cuda::new(0).unwrap();
    for n in [0, 4] {
        let l = Layout::contiguous([n]).unwrap();
        let home = cuda
            .upload(&Tensor::from_vec([n], vec![1.; n]).unwrap())
            .unwrap();
        let foreign = other
            .upload(&Tensor::from_vec([n], vec![1.; n]).unwrap())
            .unwrap();
        let mut dx = cuda.zeros([n]).unwrap();
        let mut foreign_dx = other.zeros([n]).unwrap();
        for op in UNARY {
            let p = cuda::UnaryBackward::new(&cuda, op, &l, &l, &l).unwrap();
            assert!(
                p.run(
                    Some(&foreign.view(&l).unwrap()),
                    &home.view(&l).unwrap(),
                    &mut dx.view_mut(&l).unwrap(),
                    G::Overwrite
                )
                .is_err()
            );
            assert!(
                p.run(
                    Some(&home.view(&l).unwrap()),
                    &foreign.view(&l).unwrap(),
                    &mut dx.view_mut(&l).unwrap(),
                    G::Accumulate
                )
                .is_err()
            );
            assert!(
                p.run(
                    Some(&home.view(&l).unwrap()),
                    &home.view(&l).unwrap(),
                    &mut foreign_dx.view_mut(&l).unwrap(),
                    G::Overwrite
                )
                .is_err()
            );
        }
        for op in BINARY {
            for side in SIDES {
                let p = cuda::BinaryBackward::new(&cuda, op, side, &l, &l, &l, &l).unwrap();
                assert!(
                    p.run(
                        Some(&foreign.view(&l).unwrap()),
                        Some(&home.view(&l).unwrap()),
                        &home.view(&l).unwrap(),
                        &mut dx.view_mut(&l).unwrap(),
                        G::Overwrite
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&home.view(&l).unwrap()),
                        Some(&foreign.view(&l).unwrap()),
                        &home.view(&l).unwrap(),
                        &mut dx.view_mut(&l).unwrap(),
                        G::Accumulate
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&home.view(&l).unwrap()),
                        Some(&home.view(&l).unwrap()),
                        &foreign.view(&l).unwrap(),
                        &mut dx.view_mut(&l).unwrap(),
                        G::Overwrite
                    )
                    .is_err()
                );
                assert!(
                    p.run(
                        Some(&home.view(&l).unwrap()),
                        Some(&home.view(&l).unwrap()),
                        &home.view(&l).unwrap(),
                        &mut foreign_dx.view_mut(&l).unwrap(),
                        G::Accumulate
                    )
                    .is_err()
                );
            }
        }
        assert!(
            cuda.download(&dx)
                .unwrap()
                .data()
                .iter()
                .all(|&x| x.to_bits() == 0)
        );
        assert!(
            other
                .download(&foreign_dx)
                .unwrap()
                .data()
                .iter()
                .all(|&x| x.to_bits() == 0)
        );
    }
    let a = Layout::contiguous([2, 3]).unwrap();
    let b = Layout::contiguous([3, 2]).unwrap();
    let overlap = Layout::from_strides([2, 3], [0, 1], 0).unwrap();
    for op in UNARY {
        assert!(cuda::UnaryBackward::new(&cuda, op, &a, &b, &a).is_err());
        assert!(cuda::UnaryBackward::new(&cuda, op, &a, &a, &b).is_err());
        assert!(cuda::UnaryBackward::new(&cuda, op, &a, &a, &overlap).is_err());
    }
    let incompatible = Layout::contiguous([4]).unwrap();
    for op in BINARY {
        for side in SIDES {
            assert!(cuda::BinaryBackward::new(&cuda, op, side, &a, &incompatible, &a, &a).is_err());
            assert!(cuda::BinaryBackward::new(&cuda, op, side, &a, &a, &b, &a).is_err());
            assert!(cuda::BinaryBackward::new(&cuda, op, side, &a, &a, &a, &b).is_err());
            assert!(cuda::BinaryBackward::new(&cuda, op, side, &a, &a, &a, &overlap).is_err());
        }
    }
}

#[test]
fn cuda_pointwise_gradient_resources_saved_versions_and_alias_composition() {
    let cuda = Cuda::new(0).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let huge = Layout::from_strides([1_000_000_001], [0], 0).unwrap();
    let p = cuda::BinaryBackward::new(&cuda, B::Mul, Side::Left, &scalar, &huge, &huge, &scalar)
        .unwrap();
    assert_eq!(p.contraction().reduction_len(), Some(1_000_000_001));
    assert!(p.host_metadata_bytes() < 1024 && p.workspace_bytes() <= 72);
    assert_eq!(p.shared_memory_bytes_per_block(), 1024);
    assert_eq!(
        cuda::UnaryBackward::new(&cuda, U::Neg, &scalar, &scalar, &scalar)
            .unwrap()
            .workspace_bytes(),
        0
    );
    let layout = std::sync::Arc::new(padded(&[2, 3], 1));
    let host = buffer(&layout, 1, true);
    let owner = Shared::new(cuda.upload(&host).unwrap()).unwrap();
    let saved = owner.save(&layout).unwrap();
    let dy = buffer(&layout, 2, false);
    let incoming = cuda.upload(&dy).unwrap();
    let mut dx = cuda.upload(&seed(&layout)).unwrap();
    let p = cuda::UnaryBackward::new(&cuda, U::Square, &layout, &layout, &layout).unwrap();
    {
        let read = saved.read().unwrap();
        support::no_allocations(|| {
            p.run(
                Some(&read.view().unwrap()),
                &incoming.view(&layout).unwrap(),
                &mut dx.view_mut(&layout).unwrap(),
                G::Overwrite,
            )
            .unwrap()
        });
    }
    let actual = cuda.download(&dx).unwrap();
    for a in addresses(&layout) {
        close(
            actual.data()[a],
            2. * host.data()[a] as f64 * dy.data()[a] as f64,
        );
    }
    drop(owner.write().unwrap());
    let result = (|| -> fast_audio::Result<()> {
        let read = saved.read()?;
        p.run(
            Some(&read.view()?),
            &incoming.view(&layout)?,
            &mut dx.view_mut(&layout)?,
            G::Overwrite,
        )
    })();
    assert!(result.is_err());
    assert_eq!(bits(&cuda.download(&dx).unwrap()), bits(&actual));
    let host = Tensor::from_vec([6], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    let base = cuda.upload(&host).unwrap();
    let dense = Layout::contiguous([3, 4]).unwrap();
    let dy = buffer(&dense, 4, false);
    let incoming = cuda.upload(&dy).unwrap();
    for source in [
        Layout::from_strides([3, 4], [1, 1], 0).unwrap(),
        Layout::from_strides([3, 4], [0, 1], 1).unwrap(),
    ] {
        let mut local = cuda.zeros([3, 4]).unwrap();
        cuda::UnaryBackward::new(&cuda, U::Square, &source, &dense, &dense)
            .unwrap()
            .run(
                Some(&base.view(&source).unwrap()),
                &incoming.view(&dense).unwrap(),
                &mut local.view_mut(&dense).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        let mut dx = cuda.zeros([6]).unwrap();
        cuda::ViewBackward::new(&cuda, [6], &source, &dense, 4096)
            .unwrap()
            .run(&local.view(&dense).unwrap(), &mut dx, G::Overwrite)
            .unwrap();
        let mut expected = [0.; 6];
        for row in 0..3 {
            for col in 0..4 {
                let a = address(&source, &[row, col]);
                expected[a] += 2. * host.data()[a] as f64 * dy.data()[row * 4 + col] as f64;
            }
        }
        support::close(cuda.download(&dx).unwrap().data(), &expected, 2e-5, 2e-5);
    }
    let left = Layout::from_strides([3, 1], [1, 0], 1).unwrap();
    let right = Layout::from_strides([1, 4], [0, 1], 0).unwrap();
    let mut dx = cuda.zeros([6]).unwrap();
    for side in SIDES {
        let original = if side == Side::Left { &left } else { &right };
        let grad = Layout::contiguous(original.shape().dims()).unwrap();
        let mut local = cuda.zeros(grad.shape().dims()).unwrap();
        cuda::BinaryBackward::new(&cuda, B::Mul, side, &left, &right, &dense, &grad)
            .unwrap()
            .run(
                Some(&base.view(&left).unwrap()),
                Some(&base.view(&right).unwrap()),
                &incoming.view(&dense).unwrap(),
                &mut local.view_mut(&grad).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        cuda::ViewBackward::new(&cuda, [6], original, &grad, 4096)
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
    let actual = cuda.download(&dx).unwrap();
    for i in 0..6 {
        let mut x: Vec<_> = host.data().iter().map(|&x| x as f64).collect();
        x[i] += 1e-4;
        let plus = objective(&x);
        x[i] -= 2e-4;
        close(actual.data()[i], (plus - objective(&x)) / 2e-4);
    }
}

#[test]
fn cuda_pointwise_gradient_element_warp_and_block_grids_reuse_all_tails() {
    let cuda = Cuda::new(0).unwrap();
    for (groups, count) in [(65535 * 8 + 9, 2), (65535 + 2, 33)] {
        let input = Layout::contiguous([groups, 1]).unwrap();
        let common = Layout::contiguous([groups, count]).unwrap();
        let repeated = Layout::from_strides([groups, count], [0, 0], 0).unwrap();
        let host = Tensor::from_vec(
            [groups * count],
            (0..groups * count)
                .map(|i| 1. + ((i / count) % 7) as f32 / 8.)
                .collect(),
        )
        .unwrap();
        let dy = cuda.upload(&host).unwrap();
        let right = cuda
            .upload(&Tensor::from_vec([], vec![2.]).unwrap())
            .unwrap();
        let mut dx = cuda.zeros([groups]).unwrap();
        let p = cuda::BinaryBackward::new(
            &cuda,
            B::Mul,
            Side::Left,
            &input,
            &repeated,
            &common,
            &input,
        )
        .unwrap();
        for mode in [G::Overwrite, G::Accumulate] {
            support::no_allocations(|| {
                p.run(
                    None,
                    Some(&right.view(&repeated).unwrap()),
                    &dy.view(&common).unwrap(),
                    &mut dx.view_mut(&input).unwrap(),
                    mode,
                )
                .unwrap()
            });
            let actual = cuda.download(&dx).unwrap();
            for (g, &value) in actual.data().iter().enumerate() {
                assert_eq!(
                    value,
                    2. * count as f32
                        * (1. + (g % 7) as f32 / 8.)
                        * if mode == G::Accumulate { 2. } else { 1. }
                );
            }
        }
    }
    let n = 65535 * 256 + 17;
    let l = Layout::contiguous([n]).unwrap();
    let incoming = cuda
        .upload(&Tensor::from_vec([n], vec![1.25; n]).unwrap())
        .unwrap();
    let mut dx = cuda.zeros([n]).unwrap();
    let p = cuda::UnaryBackward::new(&cuda, U::Neg, &l, &l, &l).unwrap();
    support::no_allocations(|| {
        p.run(
            None,
            &incoming.view(&l).unwrap(),
            &mut dx.view_mut(&l).unwrap(),
            G::Overwrite,
        )
        .unwrap()
    });
    assert!(
        cuda.download(&dx)
            .unwrap()
            .data()
            .iter()
            .all(|&x| x == -1.25)
    );
}
