#![cfg(feature = "cuda")]
use fast_audio::{
    GradientMode as G, Layout, ReduceOp as Op, ReduceSpec, Tensor, cpu,
    cuda::{Cuda, ReduceBackward, ReducePlan},
};
#[path = "support/reduce.rs"]
mod contract;
#[allow(
    dead_code,
    reason = "Reduction tests use independent Cartesian and gradient references."
)]
mod support;
use contract::*;

#[test]
fn cuda_reductions_and_local_adjoints_cover_strided_geometry() {
    let cuda = Cuda::new(0).unwrap();
    let cases = cases();
    assert_eq!(cases.len(), 80);
    for (index, c) in cases.iter().enumerate() {
        for op in [Op::Sum, Op::Mean] {
            let spec = ReduceSpec::new(op, c.source.shape(), &c.axes, c.keep).unwrap();
            let out = padded(spec.output_shape().dims(), index);
            let upstream = padded(spec.output_shape().dims(), index + 1);
            let gradient = padded(spec.input_shape().dims(), index + 2);
            let forward = ReducePlan::new(&cuda, &spec, &c.source, &out).unwrap();
            let backward = ReduceBackward::new(&cuda, &spec, &upstream, &gradient).unwrap();
            assert_eq!(
                forward.shared_memory_bytes_per_block() > 0,
                spec.reduction_lanes() == 256
            );
            assert_eq!(backward.shared_memory_bytes_per_block(), 0);
            assert!(forward.workspace_bytes() < 4096 && backward.workspace_bytes() < 4096);
            assert!(
                forward.host_metadata_bytes() < 200_000 && backward.host_metadata_bytes() < 200_000
            );
            for change in 0..2 {
                let source = buffer(&c.source, index + change, false);
                let input = cuda.upload(&source).unwrap();
                let initial = buffer(&out, 0, true);
                let mut y = cuda.upload(&initial).unwrap();
                support::no_allocations(|| {
                    forward
                        .run(
                            &input.view(&c.source).unwrap(),
                            &mut y.view_mut(&out).unwrap(),
                        )
                        .unwrap()
                });
                let actual = cuda.download(&y).unwrap();
                verify_forward(c, op, &source, &out, &initial, &actual);
                let mut control = initial.clone();
                cpu::ReducePlan::new(&spec, &c.source, &out)
                    .unwrap()
                    .run(
                        &source.view(&c.source).unwrap(),
                        &mut control.view_mut(&out).unwrap(),
                    )
                    .unwrap();
                assert_eq!(bits(&actual), bits(&control));
                forward
                    .run(
                        &input.view(&c.source).unwrap(),
                        &mut y.view_mut(&out).unwrap(),
                    )
                    .unwrap();
                assert_eq!(bits(&cuda.download(&y).unwrap()), bits(&actual));
                assert_eq!(bits(&cuda.download(&input).unwrap()), bits(&source));
                let dy = buffer(&upstream, index + change + 4, false);
                let incoming = cuda.upload(&dy).unwrap();
                for mode in [G::Overwrite, G::Accumulate] {
                    let initial = gradient_seed(&gradient, 0);
                    let mut dx = cuda.upload(&initial).unwrap();
                    support::no_allocations(|| {
                        backward
                            .run(
                                &incoming.view(&upstream).unwrap(),
                                &mut dx.view_mut(&gradient).unwrap(),
                                mode,
                            )
                            .unwrap()
                    });
                    let actual_dx = cuda.download(&dx).unwrap();
                    verify_backward(
                        c,
                        op,
                        (&upstream, &dy),
                        (&gradient, &initial, &actual_dx),
                        mode,
                    );
                    let mut control = initial.clone();
                    cpu::ReduceBackward::new(&spec, &upstream, &gradient)
                        .unwrap()
                        .run(
                            &dy.view(&upstream).unwrap(),
                            &mut control.view_mut(&gradient).unwrap(),
                            mode,
                        )
                        .unwrap();
                    assert_eq!(bits(&control), bits(&actual_dx));
                    let mut again = cuda.upload(&initial).unwrap();
                    backward
                        .run(
                            &incoming.view(&upstream).unwrap(),
                            &mut again.view_mut(&gradient).unwrap(),
                            mode,
                        )
                        .unwrap();
                    assert_eq!(bits(&cuda.download(&again).unwrap()), bits(&actual_dx));
                    if mode == G::Overwrite {
                        adjoint_and_difference(
                            c,
                            op,
                            &source,
                            (&out, &actual),
                            (&upstream, &dy),
                            (&gradient, &actual_dx),
                        );
                    }
                    if mode == G::Accumulate {
                        backward
                            .run(
                                &incoming.view(&upstream).unwrap(),
                                &mut dx.view_mut(&gradient).unwrap(),
                                mode,
                            )
                            .unwrap();
                        verify_backward(
                            c,
                            op,
                            (&upstream, &dy),
                            (&gradient, &actual_dx, &cuda.download(&dx).unwrap()),
                            mode,
                        );
                    }
                }
                assert_eq!(bits(&cuda.download(&incoming).unwrap()), bits(&dy));
            }
        }
    }
}

#[test]
fn cuda_reduction_tree_bits_special_values_and_rounded_count() {
    let cuda = Cuda::new(0).unwrap();
    let empty = Layout::contiguous([2, 0, 3]).unwrap();
    let result = Layout::contiguous([2, 3]).unwrap();
    let no_storage = cuda.zeros([0]).unwrap();
    for op in [Op::Sum, Op::Mean] {
        let spec = ReduceSpec::new(op, empty.shape(), &[1], false).unwrap();
        let mut out = cuda.zeros([6]).unwrap();
        ReducePlan::new(&cuda, &spec, &empty, &result)
            .unwrap()
            .run(
                &no_storage.view(&empty).unwrap(),
                &mut out.view_mut(&result).unwrap(),
            )
            .unwrap();
        assert!(
            cuda.download(&out)
                .unwrap()
                .data()
                .iter()
                .all(|x| x.to_bits() == if op == Op::Sum { 0 } else { 0x7fc00000 })
        );
    }
    for (count, points, expected) in [
        (32, vec![(0, 1e20), (16, -1e20), (1, 3.)], 3.),
        (33, vec![(0, 1e20), (32, -1e20), (16, 1.)], 1.),
        (257, vec![(0, 1e20), (256, -1e20), (128, 1.), (1, 3.)], 4.),
    ] {
        let a = Layout::contiguous([count]).unwrap();
        let b = Layout::contiguous([]).unwrap();
        let spec = ReduceSpec::new(Op::Sum, a.shape(), &[0], false).unwrap();
        let plan = ReducePlan::new(&cuda, &spec, &a, &b).unwrap();
        let mut host = Tensor::zeros([count]).unwrap();
        for (i, v) in points {
            host.data_mut()[i] = v;
        }
        let x = cuda.upload(&host).unwrap();
        let mut y = cuda.zeros([]).unwrap();
        for _ in 0..3 {
            plan.run(&x.view(&a).unwrap(), &mut y.view_mut(&b).unwrap())
                .unwrap();
            assert_eq!(cuda.download(&y).unwrap().data()[0], expected);
        }
    }
    let a = Layout::contiguous([2, 17]).unwrap();
    let b = Layout::contiguous([]).unwrap();
    let mut host = Tensor::zeros([34]).unwrap();
    host.data_mut()[0] = 1e20;
    host.data_mut()[32] = -1e20;
    host.data_mut()[16] = 1.;
    let x = cuda.upload(&host).unwrap();
    let mut y = cuda.zeros([]).unwrap();
    let spec = ReduceSpec::new(Op::Sum, a.shape(), &[1, 0], false).unwrap();
    ReducePlan::new(&cuda, &spec, &a, &b)
        .unwrap()
        .run(&x.view(&a).unwrap(), &mut y.view_mut(&b).unwrap())
        .unwrap();
    assert_eq!(cuda.download(&y).unwrap().data(), &[1.]);
    let host = Tensor::from_vec(
        [4],
        vec![
            -0.,
            f32::from_bits(0x7fc54321),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
    )
    .unwrap();
    let x = cuda.upload(&host).unwrap();
    let a = Layout::contiguous([4, 1]).unwrap();
    let b = Layout::contiguous([4]).unwrap();
    for op in [Op::Sum, Op::Mean] {
        let spec = ReduceSpec::new(op, a.shape(), &[1], false).unwrap();
        let mut y = cuda.zeros([4]).unwrap();
        ReducePlan::new(&cuda, &spec, &a, &b)
            .unwrap()
            .run(&x.view(&a).unwrap(), &mut y.view_mut(&b).unwrap())
            .unwrap();
        assert_eq!(bits(&cuda.download(&y).unwrap()), bits(&host));
        let mut dx = cuda.zeros([4]).unwrap();
        ReduceBackward::new(&cuda, &spec, &b, &a)
            .unwrap()
            .run(
                &x.view(&b).unwrap(),
                &mut dx.view_mut(&a).unwrap(),
                G::Overwrite,
            )
            .unwrap();
        assert_eq!(bits(&cuda.download(&dx).unwrap()), bits(&host));
    }
    for values in [
        vec![f32::INFINITY, 1.],
        vec![f32::NEG_INFINITY, 1.],
        vec![f32::INFINITY, f32::NEG_INFINITY],
        vec![f32::NAN, 1.],
        vec![-0., -0.],
    ] {
        let host = Tensor::from_vec([2], values.clone()).unwrap();
        let x = cuda.upload(&host).unwrap();
        let a = Layout::contiguous([2]).unwrap();
        let b = Layout::contiguous([]).unwrap();
        for op in [Op::Sum, Op::Mean] {
            let spec = ReduceSpec::new(op, a.shape(), &[0], false).unwrap();
            let mut y = cuda.zeros([]).unwrap();
            ReducePlan::new(&cuda, &spec, &a, &b)
                .unwrap()
                .run(&x.view(&a).unwrap(), &mut y.view_mut(&b).unwrap())
                .unwrap();
            let result = cuda.download(&y).unwrap();
            let actual = result.data()[0];
            let expected =
                (values[0] as f64 + values[1] as f64) / if op == Op::Mean { 2. } else { 1. };
            if expected.is_infinite() {
                assert_eq!(actual as f64, expected);
            } else {
                check_value(actual, expected);
            }
            if expected == 0. {
                assert_eq!(actual.to_bits(), 0);
            }
        }
    }
    let count = 16_777_217;
    let dense = Layout::contiguous([count]).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let spec = ReduceSpec::new(Op::Mean, dense.shape(), &[0], false).unwrap();
    let backward = ReduceBackward::new(&cuda, &spec, &scalar, &dense).unwrap();
    let dy = cuda
        .upload(&Tensor::from_vec([], vec![1.]).unwrap())
        .unwrap();
    let mut dx = cuda.zeros([count]).unwrap();
    support::no_allocations(|| {
        backward
            .run(
                &dy.view(&scalar).unwrap(),
                &mut dx.view_mut(&dense).unwrap(),
                G::Overwrite,
            )
            .unwrap()
    });
    let actual = cuda.download(&dx).unwrap();
    assert!(actual.data().iter().all(|v| v.to_bits() == 0x33800000));
    assert_ne!((1.0f64 / count as f64) as f32, actual.data()[0]);
}

#[test]
fn cuda_reductions_reject_foreign_sessions_and_changed_empty_geometry() {
    let cuda = Cuda::new(0).unwrap();
    let foreign = Cuda::new(0).unwrap();
    let a = Layout::contiguous([2, 3]).unwrap();
    let b = Layout::contiguous([2]).unwrap();
    let spec = ReduceSpec::new(Op::Sum, a.shape(), &[1], false).unwrap();
    let forward = ReducePlan::new(&cuda, &spec, &a, &b).unwrap();
    let backward = ReduceBackward::new(&cuda, &spec, &b, &a).unwrap();
    let seed = Tensor::from_vec([30], vec![9.; 30]).unwrap();
    let local = cuda.upload(&seed).unwrap();
    let other = foreign.upload(&seed).unwrap();
    let mut out = cuda.upload(&seed).unwrap();
    let mut foreign_out = foreign.upload(&seed).unwrap();
    let changed = Layout::from_strides([2, 3], [4, 1], 1).unwrap();
    let changed_b = Layout::from_strides([2], [2], 1).unwrap();
    assert!(
        forward
            .run(&other.view(&a).unwrap(), &mut out.view_mut(&b).unwrap())
            .is_err()
    );
    assert!(
        forward
            .run(
                &local.view(&a).unwrap(),
                &mut foreign_out.view_mut(&b).unwrap()
            )
            .is_err()
    );
    assert!(
        forward
            .run(
                &local.view(&changed).unwrap(),
                &mut out.view_mut(&b).unwrap()
            )
            .is_err()
    );
    assert!(
        forward
            .run(
                &local.view(&a).unwrap(),
                &mut out.view_mut(&changed_b).unwrap()
            )
            .is_err()
    );
    assert!(
        backward
            .run(
                &other.view(&b).unwrap(),
                &mut out.view_mut(&a).unwrap(),
                G::Overwrite
            )
            .is_err()
    );
    assert!(
        backward
            .run(
                &local.view(&b).unwrap(),
                &mut foreign_out.view_mut(&a).unwrap(),
                G::Accumulate
            )
            .is_err()
    );
    assert!(
        backward
            .run(
                &local.view(&changed_b).unwrap(),
                &mut out.view_mut(&a).unwrap(),
                G::Overwrite
            )
            .is_err()
    );
    assert!(
        backward
            .run(
                &local.view(&b).unwrap(),
                &mut out.view_mut(&changed).unwrap(),
                G::Accumulate
            )
            .is_err()
    );
    assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&seed));
    assert_eq!(bits(&foreign.download(&foreign_out).unwrap()), bits(&seed));
    let unproved = Layout::from_strides([2, 3], [3, 2], 0).unwrap();
    assert!(ReduceBackward::new(&cuda, &spec, &b, &unproved).is_err());
    let empty = Layout::contiguous([0, 3]).unwrap();
    let result = Layout::contiguous([0]).unwrap();
    let spec = ReduceSpec::new(Op::Mean, empty.shape(), &[1], false).unwrap();
    let forward = ReducePlan::new(&cuda, &spec, &empty, &result).unwrap();
    let backward = ReduceBackward::new(&cuda, &spec, &result, &empty).unwrap();
    assert!(
        forward
            .run(
                &other.view(&empty).unwrap(),
                &mut out.view_mut(&result).unwrap()
            )
            .is_err()
    );
    assert!(
        backward
            .run(
                &other.view(&result).unwrap(),
                &mut out.view_mut(&empty).unwrap(),
                G::Overwrite
            )
            .is_err()
    );
    assert!(
        forward
            .run(
                &local.view(&Layout::contiguous([3, 0]).unwrap()).unwrap(),
                &mut out.view_mut(&result).unwrap()
            )
            .is_err()
    );
    forward
        .run(
            &local.view(&empty).unwrap(),
            &mut out.view_mut(&result).unwrap(),
        )
        .unwrap();
    backward
        .run(
            &local.view(&result).unwrap(),
            &mut out.view_mut(&empty).unwrap(),
            G::Accumulate,
        )
        .unwrap();
    assert_eq!(bits(&cuda.download(&out).unwrap()), bits(&seed));
}

#[test]
fn cuda_reduction_grid_stride_resources_and_view_gradient_composition() {
    let cuda = Cuda::new(0).unwrap();
    let input = cuda
        .upload(&Tensor::from_vec([1], vec![1.]).unwrap())
        .unwrap();
    for (groups, count) in [(524_289, 2), (65_537, 33)] {
        let source = Layout::from_strides([groups, count], [0, 0], 0).unwrap();
        let out = Layout::contiguous([groups]).unwrap();
        for op in [Op::Sum, Op::Mean] {
            let spec = ReduceSpec::new(op, source.shape(), &[1], false).unwrap();
            let forward = ReducePlan::new(&cuda, &spec, &source, &out).unwrap();
            let mut y = cuda.zeros([groups]).unwrap();
            forward
                .run(
                    &input.view(&source).unwrap(),
                    &mut y.view_mut(&out).unwrap(),
                )
                .unwrap();
            let result = cuda.download(&y).unwrap();
            assert!(
                result
                    .data()
                    .iter()
                    .all(|&x| x == if op == Op::Sum { count as f32 } else { 1. })
            );
            assert_eq!(forward.workspace_bytes(), 40);
            assert_eq!(forward.shared_memory_bytes_per_block() > 0, count > 32);
        }
    }
    let scalar = Layout::contiguous([]).unwrap();
    let spec = ReduceSpec::new(Op::Sum, scalar.shape(), &[], false).unwrap();
    assert_eq!(
        ReducePlan::new(&cuda, &spec, &scalar, &scalar)
            .unwrap()
            .workspace_bytes(),
        0
    );
    assert_eq!(
        ReduceBackward::new(&cuda, &spec, &scalar, &scalar)
            .unwrap()
            .workspace_bytes(),
        0
    );
    let huge = Layout::from_strides([1_000_000_001], [0], 0).unwrap();
    let spec = ReduceSpec::new(Op::Sum, huge.shape(), &[0], false).unwrap();
    let prepared = ReducePlan::new(&cuda, &spec, &huge, &scalar).unwrap();
    assert!(prepared.host_metadata_bytes() < 1024);
    assert_eq!(prepared.workspace_bytes(), 16);
    let base = Tensor::from_vec([6], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    let reduced = Layout::contiguous([3]).unwrap();
    let dense = Layout::contiguous([3, 4]).unwrap();
    let x = cuda.upload(&base).unwrap();
    for source in [
        Layout::from_strides([3, 4], [1, 1], 0).unwrap(),
        Layout::from_strides([3, 4], [0, 1], 1).unwrap(),
    ] {
        for op in [Op::Sum, Op::Mean] {
            let spec = ReduceSpec::new(op, source.shape(), &[1], false).unwrap();
            let mut y = cuda.zeros([3]).unwrap();
            ReducePlan::new(&cuda, &spec, &source, &reduced)
                .unwrap()
                .run(
                    &x.view(&source).unwrap(),
                    &mut y.view_mut(&reduced).unwrap(),
                )
                .unwrap();
            let dy_host = Tensor::from_vec([3], vec![0.1, 0.3, -0.2]).unwrap();
            let dy = cuda.upload(&dy_host).unwrap();
            let mut local = cuda.zeros([3, 4]).unwrap();
            ReduceBackward::new(&cuda, &spec, &reduced, &dense)
                .unwrap()
                .run(
                    &dy.view(&reduced).unwrap(),
                    &mut local.view_mut(&dense).unwrap(),
                    G::Overwrite,
                )
                .unwrap();
            let mut dx = cuda.zeros([6]).unwrap();
            fast_audio::cuda::ViewBackward::new(&cuda, [6], &source, &dense, 4096)
                .unwrap()
                .run(&local.view(&dense).unwrap(), &mut dx, G::Overwrite)
                .unwrap();
            let (actual_y, actual_dx) = (cuda.download(&y).unwrap(), cuda.download(&dx).unwrap());
            let mut expected = vec![0.0f64; 6];
            for row in 0..3 {
                for column in 0..4 {
                    expected[address(&source, &[row, column])] +=
                        dy_host.data()[row] as f64 / if op == Op::Mean { 4. } else { 1. };
                }
            }
            support::close(actual_dx.data(), &expected, 2e-5, 2e-5);
            let lhs: f64 = actual_y
                .data()
                .iter()
                .zip(dy_host.data())
                .map(|(&y, &d)| y as f64 * d as f64)
                .sum();
            let rhs: f64 = base
                .data()
                .iter()
                .zip(actual_dx.data())
                .map(|(&x, &d)| x as f64 * d as f64)
                .sum();
            assert!((lhs - rhs).abs() < 2e-5);
        }
    }
}

#[test]
fn cuda_reduction_small_halfwarp_and_block_fast_paths_match_cpu_bits() {
    let cuda = Cuda::new(0).unwrap();
    for count in [
        2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 255, 256, 257, 513,
    ] {
        for groups in [1usize, 3, 19, 33] {
            let mut data = Vec::with_capacity(groups * count);
            for g in 0..groups {
                for r in 0..count {
                    let v = if g == 1 && count == 2 {
                        -0.0
                    } else {
                        ((((g * 17 + r * 31) % 97) as f32) - 48.0) / 37.0
                    };
                    data.push(v);
                }
            }
            let host = Tensor::from_vec([groups, count], data).unwrap();
            let dev = cuda.upload(&host).unwrap();
            let src = Layout::contiguous([groups, count]).unwrap();
            let dst = Layout::contiguous([groups]).unwrap();
            for op in [Op::Sum, Op::Mean] {
                let spec = ReduceSpec::new(op, src.shape(), &[1], false).unwrap();
                let gpu_plan = ReducePlan::new(&cuda, &spec, &src, &dst).unwrap();
                let cpu_plan = cpu::ReducePlan::new(&spec, &src, &dst).unwrap();
                let mut dev_y = cuda.zeros([groups]).unwrap();
                let mut cpu_y = Tensor::zeros([groups]).unwrap();
                gpu_plan
                    .run(&dev.view(&src).unwrap(), &mut dev_y.view_mut(&dst).unwrap())
                    .unwrap();
                cpu_plan
                    .run(
                        &host.view(&src).unwrap(),
                        &mut cpu_y.view_mut(&dst).unwrap(),
                    )
                    .unwrap();
                assert_eq!(
                    bits(&cuda.download(&dev_y).unwrap()),
                    bits(&cpu_y),
                    "GPU/CPU bit mismatch for op={op:?} count={count} groups={groups}"
                );
            }
        }
    }
}
