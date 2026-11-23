use fast_audio::{
    Error, GradientMode, Layout, Shape, Tensor, ViewBackwardStrategy as Strategy, cpu,
};
#[allow(
    dead_code,
    reason = "Adjoint tests use independent coordinate and linear-objective references."
)]
mod support;
use support::no_allocations;

const LIMIT: usize = 1 << 20;
struct Case {
    base: Vec<usize>,
    view: Layout,
    upstream: Layout,
    strategy: Strategy,
}
fn case(base: impl AsRef<[usize]>, view: Layout, upstream: Layout, strategy: Strategy) -> Case {
    Case {
        base: base.as_ref().to_vec(),
        view,
        upstream,
        strategy,
    }
}
fn cases() -> Vec<Case> {
    let high = Layout::contiguous([2, 3, 2, 2, 3])
        .unwrap()
        .permute(&[4, 2, 0, 3, 1])
        .unwrap();
    let mut out = vec![
        case(
            [2, 3],
            Layout::contiguous([2, 3]).unwrap(),
            Layout::contiguous([2, 3]).unwrap(),
            Strategy::Contiguous,
        ),
        case(
            [8],
            Layout::from_strides([2, 3], [3, 1], 1).unwrap(),
            Layout::contiguous([2, 3]).unwrap(),
            Strategy::Contiguous,
        ),
        case(
            [2, 3],
            Layout::contiguous([2, 3])
                .unwrap()
                .permute(&[1, 0])
                .unwrap(),
            Layout::contiguous([3, 2]).unwrap(),
            Strategy::InjectiveStrided,
        ),
        case(
            [8],
            Layout::from_strides([2, 3], [-3, -1], 6).unwrap(),
            Layout::from_strides([2, 3], [7, -2], 4).unwrap(),
            Strategy::InjectiveStrided,
        ),
        case(
            [16],
            Layout::from_strides([2, 3], [7, 2], 1).unwrap(),
            Layout::contiguous([2, 3]).unwrap(),
            Strategy::InjectiveStrided,
        ),
        case(
            [8],
            Layout::from_strides([3, 2], [0, 1], 2).unwrap(),
            Layout::contiguous([3, 2]).unwrap(),
            Strategy::Broadcast,
        ),
        case(
            [8],
            Layout::from_strides([2, 3, 2], [2, 0, 1], 1).unwrap(),
            Layout::from_strides([2, 3, 2], [8, -2, 1], 4).unwrap(),
            Strategy::Broadcast,
        ),
        case(
            [6],
            Layout::from_strides([257], [0], 3).unwrap(),
            Layout::from_strides([257], [-2], 512).unwrap(),
            Strategy::Broadcast,
        ),
        case(
            [4],
            Layout::from_strides([2, 1025], [1, 0], 1).unwrap(),
            Layout::contiguous([2, 1025]).unwrap(),
            Strategy::Broadcast,
        ),
        case(
            [8],
            Layout::from_strides([2, 2], [1, 1], 1).unwrap(),
            Layout::contiguous([2, 2]).unwrap(),
            Strategy::Grouped,
        ),
        case(
            [8],
            Layout::from_strides([2, 3], [3, 2], 0).unwrap(),
            Layout::contiguous([2, 3]).unwrap(),
            Strategy::Indexed,
        ),
        case(
            [10],
            Layout::from_strides([2, 3, 2], [3, 1, 2], 1).unwrap(),
            Layout::contiguous([2, 3, 2]).unwrap(),
            Strategy::Grouped,
        ),
        case(
            [131],
            Layout::from_strides([65, 65], [1, 1], 1).unwrap(),
            Layout::contiguous([65, 65]).unwrap(),
            Strategy::Grouped,
        ),
        case(
            [2, 3, 2, 2, 3],
            high.clone(),
            Layout::contiguous(high.shape().dims()).unwrap(),
            Strategy::InjectiveStrided,
        ),
        case(
            [8],
            Layout::from_strides([2, 3, 4], [3, 2, 0], 0).unwrap(),
            Layout::contiguous([2, 3, 4]).unwrap(),
            Strategy::Grouped,
        ),
        case(
            [516],
            Layout::from_strides([257], [2], 1).unwrap(),
            Layout::from_strides([257], [0], 2).unwrap(),
            Strategy::InjectiveStrided,
        ),
        case(
            [2, 3],
            Layout::contiguous([0, usize::MAX, usize::MAX]).unwrap(),
            Layout::contiguous([0, usize::MAX, usize::MAX]).unwrap(),
            Strategy::Empty,
        ),
        case(
            [0, usize::MAX],
            Layout::contiguous([0]).unwrap(),
            Layout::contiguous([0]).unwrap(),
            Strategy::Empty,
        ),
        case(
            [],
            Layout::contiguous([0]).unwrap(),
            Layout::contiguous([0]).unwrap(),
            Strategy::Empty,
        ),
        case(
            [4],
            Layout::from_strides([], [], 2).unwrap(),
            Layout::from_strides([], [], 3).unwrap(),
            Strategy::Contiguous,
        ),
        case(
            [],
            Layout::contiguous([]).unwrap(),
            Layout::contiguous([]).unwrap(),
            Strategy::Contiguous,
        ),
    ];
    let mut dims = vec![1; 24];
    dims[7] = 2;
    dims[20] = 3;
    let mut strides = vec![isize::MIN; 24];
    strides[7] = 3;
    strides[20] = 1;
    let view = Layout::from_strides(&dims, strides, 0).unwrap();
    out.push(case(
        [2, 3],
        view,
        Layout::contiguous(dims).unwrap(),
        Strategy::Contiguous,
    ));
    out
}
fn source(case: &Case) -> Tensor {
    let len = case
        .upstream
        .span()
        .map_or(case.upstream.offset(), |r| r.end + 3);
    let mut data = vec![f32::from_bits(0x7fc12345); len];
    for (_, q) in addresses(case) {
        data[q] = ((q * 17 + 9) % 31) as f32 / 32. - 0.5;
    }
    Tensor::from_vec([len], data).unwrap()
}
// Independent Cartesian carry enumeration; no implementation address/offset iterator.
fn addresses(case: &Case) -> Vec<(usize, usize)> {
    let dims = case.view.shape().dims();
    let mut coordinates = vec![0; dims.len()];
    let mut out = Vec::new();
    for _ in 0..case.view.shape().len() {
        let address = |layout: &Layout| -> usize {
            (layout.offset() as i128
                + coordinates
                    .iter()
                    .zip(layout.strides())
                    .map(|(&i, &s)| i as i128 * s as i128)
                    .sum::<i128>()) as usize
        };
        out.push((address(&case.view), address(&case.upstream)));
        for axis in (0..dims.len()).rev() {
            coordinates[axis] += 1;
            if coordinates[axis] < dims[axis] {
                break;
            }
            coordinates[axis] = 0;
        }
    }
    out
}
fn oracle(case: &Case, dy: &[f32], old: &[f32], mode: GradientMode) -> Vec<f64> {
    let mut out = match mode {
        GradientMode::Overwrite => vec![0.; old.len()],
        GradientMode::Accumulate => old.iter().map(|&x| x as f64).collect(),
    };
    for (target, source) in addresses(case) {
        out[target] += dy[source] as f64;
    }
    out
}
fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a as f64 - b).abs() <= 2e-5 + 2e-5 * b.abs(),
            "index {i}: {a} vs {b}"
        );
    }
}
fn initial(case: &Case) -> Tensor {
    let len = Shape::new(&case.base).unwrap().len();
    Tensor::from_vec(
        &case.base,
        (0..len).map(|i| (i % 7) as f32 / 16. - 0.25).collect(),
    )
    .unwrap()
}

#[test]
fn cpu_view_adjoints_match_independent_coordinates_adjoint_identities_and_finite_differences() {
    for case in cases() {
        let plan = cpu::ViewBackward::new(&case.base, &case.view, &case.upstream, LIMIT).unwrap();
        assert_eq!(plan.strategy(), case.strategy);
        assert_eq!(plan.base_shape(), case.base);
        assert_eq!(plan.view_layout(), &case.view);
        assert_eq!(plan.workspace_bytes(), 0);
        let dy = source(&case);
        let before: Vec<_> = dy.data().iter().map(|v| v.to_bits()).collect();
        let mut dx = initial(&case);
        let old = dx.data().to_vec();
        let expected = oracle(&case, dy.data(), &old, GradientMode::Overwrite);
        dx.data_mut().fill(f32::NAN);
        no_allocations(|| {
            plan.run(
                &dy.view(plan.upstream_layout()).unwrap(),
                &mut dx,
                GradientMode::Overwrite,
            )
            .unwrap()
        });
        close(dx.data(), &expected);
        let gradient = dx.data().to_vec();
        dx.data_mut().copy_from_slice(&old);
        plan.run(
            &dy.view(plan.upstream_layout()).unwrap(),
            &mut dx,
            GradientMode::Accumulate,
        )
        .unwrap();
        close(
            dx.data(),
            &oracle(&case, dy.data(), &old, GradientMode::Accumulate),
        );
        assert_eq!(
            dy.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before
        );

        let x = initial(&case);
        let dense = Layout::contiguous(case.view.shape().dims()).unwrap();
        let forward = cpu::CopyPlan::new(&case.view, &dense).unwrap();
        let mut y = Tensor::zeros(case.view.shape().dims()).unwrap();
        forward
            .run(
                &x.view(forward.source_layout()).unwrap(),
                &mut y.view_mut(forward.destination_layout()).unwrap(),
            )
            .unwrap();
        let pairs = addresses(&case);
        let lhs: f64 = y
            .data()
            .iter()
            .zip(&pairs)
            .map(|(&y, &(_, q))| y as f64 * dy.data()[q] as f64)
            .sum();
        let rhs: f64 = x
            .data()
            .iter()
            .zip(&gradient)
            .map(|(&x, &g)| x as f64 * g as f64)
            .sum();
        assert!((lhs - rhs).abs() <= 2e-5 + 2e-5 * lhs.abs());
        let mut point: Vec<_> = x.data().iter().map(|&x| x as f64).collect();
        let objective = |values: &[f64]| -> f64 {
            pairs
                .iter()
                .map(|&(p, q)| values[p] * dy.data()[q] as f64)
                .sum()
        };
        for i in 0..point.len() {
            let original = point[i];
            point[i] = original + 1e-4;
            let plus = objective(&point);
            point[i] = original - 1e-4;
            let minus = objective(&point);
            point[i] = original;
            let finite_difference = (plus - minus) / 2e-4;
            assert!(
                (gradient[i] as f64 - finite_difference).abs()
                    <= 2e-5 + 2e-5 * finite_difference.abs()
            );
        }
    }
}

#[test]
fn gradient_order_is_fixed_and_unselected_accumulation_bits_are_preserved() {
    for count in [3, 65, 257] {
        let p = Layout::from_strides([count], [0], 1).unwrap();
        let q = Layout::contiguous([count]).unwrap();
        let plan = cpu::ViewBackward::new([3], &p, &q, 0).unwrap();
        let mut dy = Tensor::zeros([count]).unwrap();
        let positions = match count {
            3 => [0, 1, 2],
            65 => [0, 32, 64],
            _ => [0, 128, 256],
        };
        for (index, value) in positions.into_iter().zip([16_777_216., -16_777_216., 1.]) {
            dy.data_mut()[index] = value;
        }
        let mut out = Tensor::from_vec([3], vec![f32::from_bits(0x7fc12345), 9., -0.]).unwrap();
        plan.run(
            &dy.view(plan.upstream_layout()).unwrap(),
            &mut out,
            GradientMode::Accumulate,
        )
        .unwrap();
        assert_eq!(out.data()[0].to_bits(), 0x7fc12345);
        assert_eq!(out.data()[1], 9.); // The declared tree gives zero, unlike a serial fold.
        assert_eq!(out.data()[2].to_bits(), (-0.0f32).to_bits());
        plan.run(
            &dy.view(plan.upstream_layout()).unwrap(),
            &mut out,
            GradientMode::Overwrite,
        )
        .unwrap();
        assert!(out.data().iter().all(|v| v.to_bits() == 0));
        assert_eq!(plan.reduction_lanes(), if count <= 32 { 32 } else { 256 });
    }
    let p = Layout::from_strides([], [], 1).unwrap();
    let q = Layout::contiguous([]).unwrap();
    let plan = cpu::ViewBackward::new([3], &p, &q, 0).unwrap();
    let dy = Tensor::from_vec([], vec![-0.]).unwrap();
    let mut out = Tensor::zeros([3]).unwrap();
    plan.run(&dy.view(&q).unwrap(), &mut out, GradientMode::Overwrite)
        .unwrap();
    assert_eq!(out.data()[1].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn index_limits_bound_general_preparation_and_broadcasts_do_not_enumerate_logical_elements() {
    let p = Layout::from_strides([2, 2], [1, 1], 0).unwrap();
    let q = Layout::contiguous([2, 2]).unwrap();
    assert!(matches!(
        cpu::ViewBackward::new([3], &p, &q, 0),
        Err(Error::Unsupported(_))
    ));
    let plan = cpu::ViewBackward::new([3], &p, &q, LIMIT).unwrap();
    assert_eq!(plan.index_bytes(), (3 + 4 + 4) * 8);
    assert_eq!(plan.preparation_index_bytes(), 4 * 16 + plan.index_bytes());
    assert!(cpu::ViewBackward::new([3], &p, &q, plan.preparation_index_bytes() - 1).is_err());
    assert!(cpu::ViewBackward::new([3], &p, &q, plan.preparation_index_bytes()).is_ok());
    assert!(plan.host_metadata_bytes() >= plan.index_bytes());
    let p = Layout::from_strides([1_000_000_000], [0], 0).unwrap();
    let q = Layout::from_strides([1_000_000_000], [0], 0).unwrap();
    let huge = cpu::ViewBackward::new([1], &p, &q, 0).unwrap();
    assert_eq!(huge.strategy(), Strategy::Broadcast);
    assert_eq!(huge.index_bytes(), 0);
    assert_eq!(huge.preparation_index_bytes(), 0);
    assert!(huge.host_metadata_bytes() < 2048);
    let p = Layout::from_strides([1_000_000_000, 2], [1, 1], 0).unwrap();
    let q = Layout::contiguous([1_000_000_000, 2]).unwrap();
    assert!(matches!(
        cpu::ViewBackward::new([1_000_000_001], &p, &q, 128),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn view_adjoints_reject_changed_geometry_and_base_shapes_before_writing() {
    let p = Layout::from_strides([2, 2], [1, 1], 0).unwrap();
    let q = Layout::contiguous([2, 2]).unwrap();
    assert!(cpu::ViewBackward::new([2], &p, &q, LIMIT).is_err());
    assert!(cpu::ViewBackward::new([3], &p, &Layout::contiguous([4]).unwrap(), LIMIT).is_err());
    let plan = cpu::ViewBackward::new([1, 3], &p, &q, LIMIT).unwrap();
    let dy = Tensor::from_vec([8], vec![2.; 8]).unwrap();
    let changed = Layout::from_strides([2, 2], [2, 1], 1).unwrap();
    let mut out = Tensor::from_vec([1, 3], vec![9.; 3]).unwrap();
    assert!(
        plan.run(
            &dy.view(&changed).unwrap(),
            &mut out,
            GradientMode::Overwrite
        )
        .is_err()
    );
    assert_eq!(out.data(), &[9.; 3]);
    let mut wrong = Tensor::from_vec([3], vec![9.; 3]).unwrap();
    assert!(
        plan.run(&dy.view(&q).unwrap(), &mut wrong, GradientMode::Overwrite)
            .is_err()
    );
    assert_eq!(wrong.data(), &[9.; 3]);
}

#[test]
fn plans_rebind_and_share_without_retaining_forward_payload_or_execution_scratch() {
    let p = Layout::from_strides([2, 3], [0, 1], 1).unwrap();
    let q = Layout::contiguous([2, 3]).unwrap();
    let plan = cpu::ViewBackward::new([5], &p, &q, 0).unwrap();
    let forward_payload = Tensor::zeros([5]).unwrap();
    drop(forward_payload);
    let first = Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let second = Tensor::from_vec([2, 3], vec![-1., -2., -3., -4., -5., -6.]).unwrap();
    let mut a = Tensor::zeros([5]).unwrap();
    let mut b = Tensor::zeros([5]).unwrap();
    std::thread::scope(|scope| {
        let one = scope.spawn(|| {
            plan.run(
                &first.view(plan.upstream_layout()).unwrap(),
                &mut a,
                GradientMode::Overwrite,
            )
            .unwrap()
        });
        let two = scope.spawn(|| {
            plan.run(
                &second.view(plan.upstream_layout()).unwrap(),
                &mut b,
                GradientMode::Overwrite,
            )
            .unwrap()
        });
        one.join().unwrap();
        two.join().unwrap();
    });
    assert_eq!(a.data(), &[0., 5., 7., 9., 0.]);
    assert_eq!(b.data(), &[0., -5., -7., -9., 0.]);
    no_allocations(|| {
        for _ in 0..8 {
            plan.run(&first.view(&q).unwrap(), &mut b, GradientMode::Accumulate)
                .unwrap();
        }
    });
    assert_eq!(b.data(), &[0., 35., 49., 63., 0.]);
}

#[test]
fn cpu_strided_add_keeps_exclusive_bounds_under_miri() {
    let p = Layout::from_strides([2, 3], [-4, -1], 7).unwrap();
    let q = Layout::from_strides([2, 3], [1, 0], 0).unwrap();
    let plan = cpu::ViewBackward::new([8], &p, &q, 0).unwrap();
    let source = Tensor::from_vec([2], vec![1., 2.]).unwrap();
    let mut out = Tensor::from_vec([8], vec![9.; 8]).unwrap();
    plan.run(&source.view(&q).unwrap(), &mut out, GradientMode::Overwrite)
        .unwrap();
    assert_eq!(out.data(), &[0., 2., 2., 2., 0., 1., 1., 1.]);
    out.data_mut()[0] = -0.;
    out.data_mut()[4] = f32::from_bits(0x7fc12345);
    plan.run(
        &source.view(&q).unwrap(),
        &mut out,
        GradientMode::Accumulate,
    )
    .unwrap();
    assert_eq!(&out.data()[1..4], &[4.; 3]);
    assert_eq!(&out.data()[5..8], &[2.; 3]);
    assert_eq!(out.data()[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(out.data()[4].to_bits(), 0x7fc12345);
    assert_eq!(source.data(), &[1., 2.]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_view_adjoints_match_oracles_repeat_and_rebind_without_allocations() {
    use fast_audio::cuda::{Cuda, ViewBackward};
    let cuda = Cuda::new(0).unwrap();
    for case in cases() {
        let plan = ViewBackward::new(&cuda, &case.base, &case.view, &case.upstream, LIMIT).unwrap();
        let control =
            cpu::ViewBackward::new(&case.base, &case.view, &case.upstream, LIMIT).unwrap();
        assert_eq!(plan.strategy(), case.strategy);
        assert_eq!(plan.base_shape(), case.base);
        assert_eq!(plan.view_layout(), &case.view);
        assert_eq!(plan.index_bytes(), control.index_bytes());
        assert_eq!(
            plan.preparation_index_bytes(),
            control
                .preparation_index_bytes()
                .max(2 * plan.index_bytes())
        );
        assert!(plan.workspace_bytes() >= plan.index_bytes());
        let dy = source(&case);
        let input = cuda.upload(&dy).unwrap();
        let mut out = cuda.upload(&initial(&case)).unwrap();
        let mut expected = initial(&case);
        control
            .run(
                &dy.view(control.upstream_layout()).unwrap(),
                &mut expected,
                GradientMode::Overwrite,
            )
            .unwrap();
        plan.run(
            &input.view(plan.upstream_layout()).unwrap(),
            &mut out,
            GradientMode::Overwrite,
        )
        .unwrap();
        let first = cuda.download(&out).unwrap();
        close(
            first.data(),
            &oracle(
                &case,
                dy.data(),
                initial(&case).data(),
                GradientMode::Overwrite,
            ),
        );
        let bits: Vec<_> = first.data().iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            bits,
            expected
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        no_allocations(|| {
            for _ in 0..4 {
                plan.run(
                    &input.view(plan.upstream_layout()).unwrap(),
                    &mut out,
                    GradientMode::Overwrite,
                )
                .unwrap();
            }
        });
        assert_eq!(
            cuda.download(&out)
                .unwrap()
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        assert_eq!(
            cuda.download(&input)
                .unwrap()
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            dy.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let mut later = dy.clone();
        for value in later.data_mut() {
            if value.is_finite() {
                *value = -*value;
            }
        }
        let input = cuda.upload(&later).unwrap();
        let old = initial(&case);
        let mut out = cuda.upload(&old).unwrap();
        plan.run(
            &input.view(plan.upstream_layout()).unwrap(),
            &mut out,
            GradientMode::Accumulate,
        )
        .unwrap();
        close(
            cuda.download(&out).unwrap().data(),
            &oracle(&case, later.data(), old.data(), GradientMode::Accumulate),
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_view_adjoints_check_budget_geometry_and_sessions_before_writing() {
    use fast_audio::cuda::{Cuda, ViewBackward};
    let cuda = Cuda::new(0).unwrap();
    let foreign = Cuda::new(0).unwrap();
    let p = Layout::from_strides([2, 2], [1, 1], 0).unwrap();
    let q = Layout::contiguous([2, 2]).unwrap();
    let plan = ViewBackward::new(&cuda, [1, 3], &p, &q, LIMIT).unwrap();
    assert_eq!(plan.index_bytes(), 88);
    assert_eq!(plan.workspace_bytes(), 88);
    assert_eq!(plan.preparation_index_bytes(), 176);
    assert_eq!(plan.shared_memory_bytes_per_block(), 0);
    assert!(matches!(
        ViewBackward::new(&cuda, [1, 3], &p, &q, 175),
        Err(Error::Unsupported(_))
    ));
    assert!(ViewBackward::new(&cuda, [1, 3], &p, &q, 176).is_ok());
    let host = Tensor::from_vec([8], vec![2.; 8]).unwrap();
    let input = cuda.upload(&host).unwrap();
    let foreign_input = foreign.upload(&host).unwrap();
    let seed = Tensor::from_vec([1, 3], vec![9.; 3]).unwrap();
    let mut out = cuda.upload(&seed).unwrap();
    let changed = Layout::from_strides([2, 2], [2, 1], 1).unwrap();
    assert!(
        plan.run(
            &input.view(&changed).unwrap(),
            &mut out,
            GradientMode::Overwrite
        )
        .is_err()
    );
    assert!(
        plan.run(
            &foreign_input.view(&q).unwrap(),
            &mut out,
            GradientMode::Overwrite
        )
        .is_err()
    );
    assert_eq!(cuda.download(&out).unwrap().data(), &[9.; 3]);
    let mut wrong = cuda
        .upload(&Tensor::from_vec([3], vec![9.; 3]).unwrap())
        .unwrap();
    assert!(
        plan.run(
            &input.view(&q).unwrap(),
            &mut wrong,
            GradientMode::Overwrite
        )
        .is_err()
    );
    assert_eq!(cuda.download(&wrong).unwrap().data(), &[9.; 3]);
    let mut foreign_out = foreign.upload(&seed).unwrap();
    assert!(
        plan.run(
            &input.view(&q).unwrap(),
            &mut foreign_out,
            GradientMode::Overwrite
        )
        .is_err()
    );
    assert_eq!(foreign.download(&foreign_out).unwrap().data(), &[9.; 3]);
    let huge = Layout::from_strides([1_000_000_000], [0], 0).unwrap();
    let analytical = ViewBackward::new(&cuda, [1], &huge, &huge, 0).unwrap();
    assert_eq!(analytical.strategy(), Strategy::Broadcast);
    assert_eq!(analytical.index_bytes(), 0);
    assert_eq!(analytical.preparation_index_bytes(), 0);
    assert_eq!(analytical.workspace_bytes(), 16);
    let empty = Layout::contiguous([0]).unwrap();
    let plan = ViewBackward::new(&cuda, [3], &empty, &empty, 0).unwrap();
    let input = cuda.zeros([0]).unwrap();
    let wrong_input = foreign.zeros([0]).unwrap();
    let sentinel = Tensor::from_vec([3], vec![f32::from_bits(0x7fc12345), -0., 1.]).unwrap();
    let mut out = cuda.upload(&sentinel).unwrap();
    assert!(
        plan.run(
            &wrong_input.view(&empty).unwrap(),
            &mut out,
            GradientMode::Accumulate
        )
        .is_err()
    );
    plan.run(
        &input.view(&empty).unwrap(),
        &mut out,
        GradientMode::Accumulate,
    )
    .unwrap();
    assert_eq!(
        cuda.download(&out)
            .unwrap()
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        sentinel
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    plan.run(
        &input.view(&empty).unwrap(),
        &mut out,
        GradientMode::Overwrite,
    )
    .unwrap();
    assert!(
        cuda.download(&out)
            .unwrap()
            .data()
            .iter()
            .all(|v| v.to_bits() == 0)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_reduction_order_matches_fixed_lane_trees_and_repeated_outputs() {
    use fast_audio::cuda::{Cuda, ViewBackward};
    let cuda = Cuda::new(0).unwrap();
    for count in [3, 65, 257] {
        let p = Layout::from_strides([count], [0], 1).unwrap();
        let q = Layout::contiguous([count]).unwrap();
        let plan = ViewBackward::new(&cuda, [3], &p, &q, 0).unwrap();
        let mut host = Tensor::zeros([count]).unwrap();
        let positions = match count {
            3 => [0, 1, 2],
            65 => [0, 32, 64],
            _ => [0, 128, 256],
        };
        for (i, value) in positions.into_iter().zip([16_777_216., -16_777_216., 1.]) {
            host.data_mut()[i] = value;
        }
        let input = cuda.upload(&host).unwrap();
        let mut out = cuda
            .upload(&Tensor::from_vec([3], vec![f32::from_bits(0x7fc12345), 9., -0.]).unwrap())
            .unwrap();
        plan.run(&input.view(&q).unwrap(), &mut out, GradientMode::Accumulate)
            .unwrap();
        let accumulated = cuda.download(&out).unwrap();
        assert_eq!(accumulated.data()[0].to_bits(), 0x7fc12345);
        assert_eq!(accumulated.data()[1], 9.);
        assert_eq!(accumulated.data()[2].to_bits(), (-0.0f32).to_bits());
        assert_eq!(plan.shared_memory_bytes_per_block() > 0, count > 32);
        for _ in 0..4 {
            plan.run(&input.view(&q).unwrap(), &mut out, GradientMode::Overwrite)
                .unwrap();
            assert!(
                cuda.download(&out)
                    .unwrap()
                    .data()
                    .iter()
                    .all(|v| v.to_bits() == 0)
            );
        }
    }
    let p = Layout::from_strides([3, 3], [1, 1], 0).unwrap();
    let q = Layout::contiguous([3, 3]).unwrap();
    let plan = ViewBackward::new(&cuda, [5], &p, &q, LIMIT).unwrap();
    let mut host = Tensor::zeros([3, 3]).unwrap();
    for (i, value) in [2, 4, 6].into_iter().zip([16_777_216., -16_777_216., 1.]) {
        host.data_mut()[i] = value;
    }
    let input = cuda.upload(&host).unwrap();
    let mut out = cuda.zeros([5]).unwrap();
    plan.run(&input.view(&q).unwrap(), &mut out, GradientMode::Overwrite)
        .unwrap();
    assert!(
        cuda.download(&out)
            .unwrap()
            .data()
            .iter()
            .all(|v| v.to_bits() == 0)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_grid_stride_reductions_reuse_block_and_warp_state() {
    use fast_audio::cuda::{Cuda, ViewBackward};
    let cuda = Cuda::new(0).unwrap();
    let input = cuda
        .upload(&Tensor::from_vec([1], vec![1.]).unwrap())
        .unwrap();
    for (groups, repetitions) in [(65537, 33), (524281, 2)] {
        let p = Layout::from_strides([groups, repetitions], [1, 0], 0).unwrap();
        let q = Layout::from_strides([groups, repetitions], [0, 0], 0).unwrap();
        let plan = ViewBackward::new(&cuda, [groups], &p, &q, 0).unwrap();
        assert_eq!(plan.strategy(), Strategy::Broadcast);
        assert_eq!(plan.index_bytes(), 0);
        let mut out = cuda.zeros([groups]).unwrap();
        no_allocations(|| {
            plan.run(&input.view(&q).unwrap(), &mut out, GradientMode::Overwrite)
                .unwrap()
        });
        assert!(
            cuda.download(&out)
                .unwrap()
                .data()
                .iter()
                .all(|&v| v == repetitions as f32)
        );
        plan.run(&input.view(&q).unwrap(), &mut out, GradientMode::Accumulate)
            .unwrap();
        assert!(
            cuda.download(&out)
                .unwrap()
                .data()
                .iter()
                .all(|&v| v == 2. * repetitions as f32)
        );
    }
}
