use fast_audio::{
    GradientMode as G, Layout, ReduceOp as Op, ReduceSpec, ReduceStrategy as Strategy, Shape,
    Tensor, cpu,
};
#[path = "support/reduce.rs"]
mod contract;
#[allow(
    dead_code,
    reason = "Reduction fixtures use independent Cartesian references."
)]
mod support;
use contract::*;

fn exercise(c: &Case, seed: usize) {
    for op in [Op::Sum, Op::Mean] {
        let spec = ReduceSpec::new(op, c.source.shape(), &c.axes, c.keep).unwrap();
        assert_eq!(
            spec.output_shape().dims(),
            output_dims(c.source.shape().dims(), &c.axes, c.keep)
        );
        assert_eq!(spec.reduction_len(), count(c));
        let out = padded(spec.output_shape().dims(), seed);
        let upstream = padded(spec.output_shape().dims(), seed + 1);
        let gradient = padded(spec.input_shape().dims(), seed + 2);
        let plan = cpu::ReducePlan::new(&spec, &c.source, &out).unwrap();
        let backward = cpu::ReduceBackward::new(&spec, &upstream, &gradient).unwrap();
        assert_eq!(plan.workspace_bytes(), 0);
        assert_eq!(backward.workspace_bytes(), 0);
        assert!(plan.host_metadata_bytes() < 200_000 && backward.host_metadata_bytes() < 200_000);
        for change in 0..2 {
            let source = buffer(&c.source, seed + change, false);
            let original = bits(&source);
            let initial = buffer(&out, 0, true);
            let mut y = initial.clone();
            support::no_allocations(|| {
                plan.run(
                    &source.view(&c.source).unwrap(),
                    &mut y.view_mut(&out).unwrap(),
                )
                .unwrap()
            });
            verify_forward(c, op, &source, &out, &initial, &y);
            let first = bits(&y);
            plan.run(
                &source.view(&c.source).unwrap(),
                &mut y.view_mut(&out).unwrap(),
            )
            .unwrap();
            assert_eq!(bits(&y), first);
            assert_eq!(bits(&source), original);
            let dy = buffer(&upstream, seed + change + 4, false);
            let saved = bits(&dy);
            for mode in [G::Overwrite, G::Accumulate] {
                let before = gradient_seed(&gradient, 0);
                let mut dx = before.clone();
                support::no_allocations(|| {
                    backward
                        .run(
                            &dy.view(&upstream).unwrap(),
                            &mut dx.view_mut(&gradient).unwrap(),
                            mode,
                        )
                        .unwrap()
                });
                verify_backward(c, op, (&upstream, &dy), (&gradient, &before, &dx), mode);
                if mode == G::Overwrite {
                    adjoint_and_difference(
                        c,
                        op,
                        &source,
                        (&out, &y),
                        (&upstream, &dy),
                        (&gradient, &dx),
                    );
                }
                let once = bits(&dx);
                let mut again = before.clone();
                backward
                    .run(
                        &dy.view(&upstream).unwrap(),
                        &mut again.view_mut(&gradient).unwrap(),
                        mode,
                    )
                    .unwrap();
                assert_eq!(bits(&again), once);
                if mode == G::Accumulate {
                    let before_second = dx.clone();
                    backward
                        .run(
                            &dy.view(&upstream).unwrap(),
                            &mut dx.view_mut(&gradient).unwrap(),
                            mode,
                        )
                        .unwrap();
                    verify_backward(
                        c,
                        op,
                        (&upstream, &dy),
                        (&gradient, &before_second, &dx),
                        mode,
                    );
                }
            }
            assert_eq!(bits(&dy), saved);
        }
    }
}

#[test]
fn cpu_reductions_and_local_adjoints_match_independent_oracles() {
    let cases = cases();
    assert_eq!(cases.len(), 80);
    for (i, c) in cases.iter().enumerate() {
        exercise(c, i);
    }
}

#[test]
fn cpu_reduction_signed_stride_bounds_under_miri() {
    let c = Case {
        source: Layout::from_strides([3, 2, 5], [-2, 19, -4], 20).unwrap(),
        axes: vec![2, 0],
        keep: false,
    };
    exercise(&c, 3);
    let empty = Case {
        source: Layout::from_strides([2, 0, 3], [isize::MIN, isize::MAX, isize::MIN], 3).unwrap(),
        axes: vec![1],
        keep: false,
    };
    exercise(&empty, 1);
}

#[test]
fn cpu_reductions_reject_wrong_specs_and_bindings_before_writes() {
    let shape = Shape::new([2, 3]).unwrap();
    assert!(ReduceSpec::new(Op::Sum, &shape, &[0, 0], false).is_err());
    assert!(ReduceSpec::new(Op::Sum, &shape, &[2], false).is_err());
    assert!(ReduceSpec::new(Op::Mean, &Shape::new([]).unwrap(), &[0], false).is_err());
    let spec = ReduceSpec::new(Op::Mean, &shape, &[0, 1], true).unwrap();
    let reversed = ReduceSpec::new(Op::Mean, &shape, &[1, 0], true).unwrap();
    assert_eq!(spec, reversed);
    assert!(
        ReduceSpec::new(
            Op::Sum,
            &Shape::new([usize::MAX, usize::MAX, 0]).unwrap(),
            &[2],
            false
        )
        .is_err()
    );
    let a = Layout::contiguous([2, 3]).unwrap();
    let b = Layout::contiguous([1, 1]).unwrap();
    let wrong = Layout::contiguous([1]).unwrap();
    assert!(cpu::ReducePlan::new(&spec, &a, &wrong).is_err());
    assert!(cpu::ReduceBackward::new(&spec, &wrong, &a).is_err());
    let unproved = Layout::from_strides([2, 3], [3, 2], 0).unwrap();
    assert!(cpu::ReduceBackward::new(&spec, &b, &unproved).is_err());
    let none = ReduceSpec::new(Op::Sum, &shape, &[], false).unwrap();
    assert!(cpu::ReducePlan::new(&none, &a, &unproved).is_err());
    let plan = cpu::ReducePlan::new(&spec, &a, &b).unwrap();
    let backward = cpu::ReduceBackward::new(&spec, &b, &a).unwrap();
    let source = Tensor::from_vec([30], vec![1.; 30]).unwrap();
    let mut out = Tensor::from_vec([30], vec![9.; 30]).unwrap();
    let changed = Layout::from_strides([2, 3], [4, 1], 1).unwrap();
    let changed_b = Layout::from_strides([1, 1], [0, 0], 1).unwrap();
    assert!(
        plan.run(
            &source.view(&changed).unwrap(),
            &mut out.view_mut(&b).unwrap()
        )
        .is_err()
    );
    assert!(
        plan.run(
            &source.view(&a).unwrap(),
            &mut out.view_mut(&changed_b).unwrap()
        )
        .is_err()
    );
    assert!(
        backward
            .run(
                &source.view(&changed_b).unwrap(),
                &mut out.view_mut(&a).unwrap(),
                G::Overwrite
            )
            .is_err()
    );
    assert!(
        backward
            .run(
                &source.view(&b).unwrap(),
                &mut out.view_mut(&changed).unwrap(),
                G::Accumulate
            )
            .is_err()
    );
    assert_eq!(out.data(), &[9.; 30]);
    let empty = Layout::contiguous([0, 3]).unwrap();
    let spec = ReduceSpec::new(Op::Sum, empty.shape(), &[1], false).unwrap();
    let dest = Layout::contiguous([0]).unwrap();
    let plan = cpu::ReducePlan::new(&spec, &empty, &dest).unwrap();
    assert!(
        plan.run(
            &source.view(&Layout::contiguous([3, 0]).unwrap()).unwrap(),
            &mut out.view_mut(&dest).unwrap()
        )
        .is_err()
    );
    assert_eq!(out.data(), &[9.; 30]);
}

#[test]
fn cpu_reduction_tree_order_and_single_contribution_bits() {
    let empty = Layout::contiguous([2, 0, 3]).unwrap();
    let result = Layout::contiguous([2, 3]).unwrap();
    let no_storage = Tensor::zeros([0]).unwrap();
    for op in [Op::Sum, Op::Mean] {
        let spec = ReduceSpec::new(op, empty.shape(), &[1], false).unwrap();
        let mut out = Tensor::zeros([6]).unwrap();
        cpu::ReducePlan::new(&spec, &empty, &result)
            .unwrap()
            .run(
                &no_storage.view(&empty).unwrap(),
                &mut out.view_mut(&result).unwrap(),
            )
            .unwrap();
        assert!(
            out.data()
                .iter()
                .all(|x| x.to_bits() == if op == Op::Sum { 0 } else { 0x7fc00000 })
        );
    }
    let a = Layout::contiguous([2, 17]).unwrap();
    let b = Layout::contiguous([]).unwrap();
    let mut source = Tensor::zeros([34]).unwrap();
    source.data_mut()[0] = 1e20;
    source.data_mut()[32] = -1e20;
    source.data_mut()[16] = 1.;
    let spec = ReduceSpec::new(Op::Sum, a.shape(), &[1, 0], false).unwrap();
    let mut output = Tensor::zeros([]).unwrap();
    cpu::ReducePlan::new(&spec, &a, &b)
        .unwrap()
        .run(&source.view(&a).unwrap(), &mut output.view_mut(&b).unwrap())
        .unwrap();
    assert_eq!(output.data(), &[1.]);
    for (count, points, expected) in [
        (32, vec![(0, 1e20), (16, -1e20), (1, 3.)], 3.),
        (33, vec![(0, 1e20), (32, -1e20), (16, 1.)], 1.),
        (257, vec![(0, 1e20), (256, -1e20), (128, 1.), (1, 3.)], 4.),
    ] {
        let a = Layout::contiguous([count]).unwrap();
        let b = Layout::contiguous([]).unwrap();
        let spec = ReduceSpec::new(Op::Sum, a.shape(), &[0], false).unwrap();
        assert_eq!(spec.reduction_lanes(), if count <= 32 { 32 } else { 256 });
        let plan = cpu::ReducePlan::new(&spec, &a, &b).unwrap();
        let mut x = Tensor::zeros([count]).unwrap();
        for (i, v) in points {
            x.data_mut()[i] = v;
        }
        let mut y = Tensor::zeros([]).unwrap();
        plan.run(&x.view(&a).unwrap(), &mut y.view_mut(&b).unwrap())
            .unwrap();
        assert_eq!(y.data()[0], expected);
    }
    let x = Tensor::from_vec(
        [4],
        vec![
            -0.,
            f32::from_bits(0x7fc54321),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
    )
    .unwrap();
    let a = Layout::contiguous([4, 1]).unwrap();
    let b = Layout::contiguous([4]).unwrap();
    for op in [Op::Sum, Op::Mean] {
        let spec = ReduceSpec::new(op, a.shape(), &[1], false).unwrap();
        assert_eq!(spec.strategy(), Strategy::Copy);
        let mut y = Tensor::zeros([4]).unwrap();
        cpu::ReducePlan::new(&spec, &a, &b)
            .unwrap()
            .run(&x.view(&a).unwrap(), &mut y.view_mut(&b).unwrap())
            .unwrap();
        assert_eq!(bits(&x), bits(&y));
        let mut dx = Tensor::zeros([4]).unwrap();
        cpu::ReduceBackward::new(&spec, &b, &a)
            .unwrap()
            .run(
                &x.view(&b).unwrap(),
                &mut dx.view_mut(&a).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        assert_eq!(bits(&x), bits(&dx));
    }
    for values in [
        vec![f32::INFINITY, 1.],
        vec![f32::NEG_INFINITY, 1.],
        vec![f32::INFINITY, f32::NEG_INFINITY],
        vec![f32::NAN, 1.],
        vec![-0., -0.],
    ] {
        let a = Layout::contiguous([2]).unwrap();
        let b = Layout::contiguous([]).unwrap();
        let x = Tensor::from_vec([2], values.clone()).unwrap();
        for op in [Op::Sum, Op::Mean] {
            let spec = ReduceSpec::new(op, a.shape(), &[0], false).unwrap();
            let mut y = Tensor::zeros([]).unwrap();
            cpu::ReducePlan::new(&spec, &a, &b)
                .unwrap()
                .run(&x.view(&a).unwrap(), &mut y.view_mut(&b).unwrap())
                .unwrap();
            let expected =
                (values[0] as f64 + values[1] as f64) / if op == Op::Mean { 2. } else { 1. };
            if expected.is_infinite() {
                assert_eq!(y.data()[0] as f64, expected);
            } else {
                check_value(y.data()[0], expected);
            }
            if expected == 0. {
                assert_eq!(y.data()[0].to_bits(), 0);
            }
        }
    }
}

#[test]
fn cpu_reduction_geometry_resources_sharing_and_rounded_count() {
    let huge = Layout::from_strides([1_000_000_001], [0], 0).unwrap();
    let out = Layout::contiguous([]).unwrap();
    let spec = ReduceSpec::new(Op::Sum, huge.shape(), &[0], false).unwrap();
    assert_eq!(spec.strategy(), Strategy::Tree256);
    assert!(
        cpu::ReducePlan::new(&spec, &huge, &out)
            .unwrap()
            .host_metadata_bytes()
            < 1024
    );
    let shape = Shape::new([usize::MAX, usize::MAX, 0]).unwrap();
    let spec = ReduceSpec::new(Op::Mean, &shape, &[1, 0], false).unwrap();
    assert_eq!(spec.reduction_len(), None);
    assert_eq!(spec.strategy(), Strategy::Empty);
    let a = Layout::contiguous([2, 3]).unwrap();
    let b = Layout::contiguous([2]).unwrap();
    let x = Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let spec = ReduceSpec::new(Op::Sum, a.shape(), &[1], false).unwrap();
    let plan = cpu::ReducePlan::new(&spec, &a, &b).unwrap();
    let shared_backward = cpu::ReduceBackward::new(&spec, &b, &a).unwrap();
    std::thread::scope(|scope| {
        let plan = &plan;
        let shared_backward = &shared_backward;
        let x = &x;
        let a = &a;
        let b = &b;
        for _ in 0..3 {
            scope.spawn(move || {
                let mut y = Tensor::zeros([2]).unwrap();
                plan.run(&x.view(a).unwrap(), &mut y.view_mut(b).unwrap())
                    .unwrap();
                assert_eq!(y.data(), &[6., 15.]);
                let dy = Tensor::from_vec([2], vec![1., 2.]).unwrap();
                let mut dx = Tensor::zeros([2, 3]).unwrap();
                shared_backward
                    .run(
                        &dy.view(b).unwrap(),
                        &mut dx.view_mut(a).unwrap(),
                        G::Overwrite,
                    )
                    .unwrap();
                assert_eq!(dx.data(), &[1., 1., 1., 2., 2., 2.]);
            });
        }
    });
    let count = 16_777_217;
    let dense = Layout::contiguous([count]).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let spec = ReduceSpec::new(Op::Mean, dense.shape(), &[0], false).unwrap();
    let backward = cpu::ReduceBackward::new(&spec, &scalar, &dense).unwrap();
    let dy = Tensor::from_vec([], vec![1.]).unwrap();
    let mut dx = Tensor::zeros([count]).unwrap();
    support::no_allocations(|| {
        backward
            .run(
                &dy.view(&scalar).unwrap(),
                &mut dx.view_mut(&dense).unwrap(),
                G::Overwrite,
            )
            .unwrap()
    });
    assert!(dx.data().iter().all(|v| v.to_bits() == 0x33800000));
    assert_ne!((1.0f64 / count as f64) as f32, dx.data()[0]);
}

#[test]
fn cpu_local_reduction_gradient_composes_with_overlapping_view_adjoint() {
    let base = Tensor::from_vec([6], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    let reduced = Layout::contiguous([3]).unwrap();
    let dense = Layout::contiguous([3, 4]).unwrap();
    for source in [
        Layout::from_strides([3, 4], [1, 1], 0).unwrap(),
        Layout::from_strides([3, 4], [0, 1], 1).unwrap(),
    ] {
        for op in [Op::Sum, Op::Mean] {
            let spec = ReduceSpec::new(op, source.shape(), &[1], false).unwrap();
            let mut y = Tensor::zeros([3]).unwrap();
            cpu::ReducePlan::new(&spec, &source, &reduced)
                .unwrap()
                .run(
                    &base.view(&source).unwrap(),
                    &mut y.view_mut(&reduced).unwrap(),
                )
                .unwrap();
            let dy = Tensor::from_vec([3], vec![0.1, 0.3, -0.2]).unwrap();
            let mut local = Tensor::zeros([3, 4]).unwrap();
            cpu::ReduceBackward::new(&spec, &reduced, &dense)
                .unwrap()
                .run(
                    &dy.view(&reduced).unwrap(),
                    &mut local.view_mut(&dense).unwrap(),
                    G::Overwrite,
                )
                .unwrap();
            let mut dx = Tensor::zeros([6]).unwrap();
            cpu::ViewBackward::new([6], &source, &dense, 4096)
                .unwrap()
                .run(&local.view(&dense).unwrap(), &mut dx, G::Overwrite)
                .unwrap();
            let mut expected = vec![0.0f64; 6];
            for row in 0..3 {
                for column in 0..4 {
                    expected[address(&source, &[row, column])] +=
                        dy.data()[row] as f64 / if op == Op::Mean { 4. } else { 1. };
                }
            }
            support::close(dx.data(), &expected, 2e-5, 2e-5);
            let lhs: f64 = y
                .data()
                .iter()
                .zip(dy.data())
                .map(|(&y, &d)| y as f64 * d as f64)
                .sum();
            let rhs: f64 = base
                .data()
                .iter()
                .zip(dx.data())
                .map(|(&x, &d)| x as f64 * d as f64)
                .sum();
            assert!((lhs - rhs).abs() < 2e-5);
        }
    }
}

#[test]
fn cpu_reduction_small_and_chunked_fast_paths_are_bit_identical_to_canonical_trees() {
    fn canonical_tree<const N: usize>(values: &[f32]) -> f32 {
        let mut lanes = [0.0f32; N];
        for (lane, sum) in lanes.iter_mut().enumerate() {
            for &v in values.iter().skip(lane).step_by(N) {
                *sum += v;
            }
        }
        let mut step = N / 2;
        while step > 0 {
            for lane in 0..step {
                lanes[lane] += lanes[lane + step];
            }
            step /= 2;
        }
        lanes[0]
    }
    let special = [
        -0.0f32,
        0.0,
        1.0,
        -1.0,
        1e20,
        -1e20,
        0.125,
        -0.375,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    for count in [
        2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 255, 256, 257, 513,
    ] {
        for groups in [1usize, 3, 19] {
            let mut data = Vec::with_capacity(groups * count);
            for g in 0..groups {
                for r in 0..count {
                    let v = if g == 0 && r < special.len() {
                        special[r]
                    } else if g == 1 && count == 2 {
                        -0.0
                    } else {
                        ((((g * 17 + r * 31) % 97) as f32) - 48.0) / 37.0
                    };
                    data.push(v);
                }
            }
            let x = Tensor::from_vec([groups, count], data.clone()).unwrap();
            let src = Layout::contiguous([groups, count]).unwrap();
            let dst = Layout::contiguous([groups]).unwrap();
            for op in [Op::Sum, Op::Mean] {
                let spec = ReduceSpec::new(op, src.shape(), &[1], false).unwrap();
                let plan = cpu::ReducePlan::new(&spec, &src, &dst).unwrap();
                let mut y = Tensor::zeros([groups]).unwrap();
                plan.run(&x.view(&src).unwrap(), &mut y.view_mut(&dst).unwrap())
                    .unwrap();
                for g in 0..groups {
                    let row = &data[g * count..(g + 1) * count];
                    let sum = if count <= 32 {
                        canonical_tree::<32>(row)
                    } else {
                        canonical_tree::<256>(row)
                    };
                    let want = if op == Op::Mean {
                        sum / (count as f32)
                    } else {
                        sum
                    };
                    assert_eq!(
                        y.data()[g].to_bits(),
                        want.to_bits(),
                        "bit mismatch for op={op:?} count={count} groups={groups} g={g}"
                    );
                }
            }
        }
    }
}
