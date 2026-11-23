use fast_audio::{Layout, Tensor, TensorViewMut, cpu};
#[allow(
    dead_code,
    reason = "Copy tests compare raw bits or exact empty sums rather than tolerance helpers."
)]
mod support;
use support::{backends, no_allocations};

fn bits(len: usize) -> Tensor {
    let patterns = [
        0x80000000, 0, 0x7fc12345, 0x7f812345, 0x7f800000, 0xff800000, 0x3f800001, 0xbf246789,
    ];
    Tensor::from_vec(
        [len],
        (0..len)
            .map(|i| f32::from_bits(patterns[i % patterns.len()]))
            .collect(),
    )
    .unwrap()
}
fn layouts() -> Vec<(Layout, Layout)> {
    vec![
        (
            Layout::from_strides([2, 3], [3, 1], 2).unwrap(),
            Layout::from_strides([2, 3], [3, 1], 1).unwrap(),
        ),
        (
            Layout::from_strides([3, 2], [1, 3], 0).unwrap(),
            Layout::contiguous([3, 2]).unwrap(),
        ),
        (
            Layout::from_strides([2, 3], [-5, -1], 9).unwrap(),
            Layout::from_strides([2, 3], [4, 1], 2).unwrap(),
        ),
        (
            Layout::from_strides([2, 3], [7, 2], 1).unwrap(),
            Layout::from_strides([2, 3], [-1, -3], 9).unwrap(),
        ),
        (
            Layout::from_strides([2, 3], [0, 1], 1).unwrap(),
            Layout::from_strides([2, 3], [5, 1], 0).unwrap(),
        ),
        (
            Layout::from_strides([2, 3], [0, 0], 2).unwrap(),
            Layout::contiguous([2, 3]).unwrap(),
        ),
        (
            Layout::from_strides([2, 3], [-3, -1], 5).unwrap(),
            Layout::from_strides([2, 3], [-3, -1], 7).unwrap(),
        ),
        (
            Layout::from_strides([2, 1, 3, 1, 2, 2], [30, 99, 8, -2, 3, -1], 1).unwrap(),
            Layout::from_strides([2, 1, 3, 1, 2, 2], [24, -8, 8, 13, 3, 1], 0).unwrap(),
        ),
        (
            Layout::from_strides([], [], 3).unwrap(),
            Layout::from_strides([], [], 1).unwrap(),
        ),
        (
            Layout::contiguous([2, 3, 2, 2, 3])
                .unwrap()
                .permute(&[4, 2, 0, 3, 1])
                .unwrap(),
            Layout::contiguous([3, 2, 2, 2, 3]).unwrap(),
        ),
        (
            Layout::contiguous([0, usize::MAX, usize::MAX]).unwrap(),
            Layout::contiguous([0, usize::MAX, usize::MAX]).unwrap(),
        ),
        (
            Layout::contiguous([257]).unwrap(),
            Layout::from_strides([257], [-1], 256).unwrap(),
        ),
    ]
}

// Enumerate nested coordinates and compute independent source/destination byte slots.
fn reference(source: &Layout, destination: &Layout, input: &[f32], initial: &[f32]) -> Vec<u32> {
    let mut coordinates = vec![vec![]];
    if source.shape().is_empty() {
        return initial.iter().map(|x| x.to_bits()).collect();
    }
    for &dimension in source.shape().dims() {
        coordinates = coordinates
            .into_iter()
            .flat_map(|prefix| {
                (0..dimension).map(move |i| {
                    let mut next = prefix.clone();
                    next.push(i);
                    next
                })
            })
            .collect();
    }
    let mut expected: Vec<_> = initial.iter().map(|x| x.to_bits()).collect();
    for c in coordinates {
        let x = source.offset() as i128
            + c.iter()
                .zip(source.strides())
                .map(|(&i, &s)| i as i128 * s as i128)
                .sum::<i128>();
        let y = destination.offset() as i128
            + c.iter()
                .zip(destination.strides())
                .map(|(&i, &s)| i as i128 * s as i128)
                .sum::<i128>();
        expected[y as usize] = input[x as usize].to_bits();
    }
    expected
}

#[test]
fn cpu_copy_preserves_bits_destination_gaps_rebinding_and_allocations() {
    for (source, destination) in layouts() {
        let size = source
            .span()
            .map_or(source.offset(), |r| r.end)
            .max(destination.span().map_or(destination.offset(), |r| r.end))
            + 3;
        let plan = cpu::CopyPlan::new(&source, &destination).unwrap();
        assert_eq!(plan.workspace_bytes(), 0);
        for reverse in [false, true] {
            let mut input = bits(size);
            if reverse {
                input.data_mut().reverse();
            }
            let mut output = Tensor::from_vec([size], vec![-99.0; size]).unwrap();
            let expected = reference(&source, &destination, input.data(), output.data());
            no_allocations(|| {
                for _ in 0..3 {
                    plan.run(
                        &input.view(plan.source_layout()).unwrap(),
                        &mut output.view_mut(plan.destination_layout()).unwrap(),
                    )
                    .unwrap();
                }
            });
            assert_eq!(
                output
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn cpu_copy_miri_negative_and_broadcast_rows() {
    for (source, destination) in layouts().into_iter().take(9) {
        let input = bits(128);
        let mut output = bits(128);
        let expected = reference(&source, &destination, input.data(), output.data());
        let plan = cpu::CopyPlan::new(&source, &destination).unwrap();
        plan.run(
            &input.view(&source).unwrap(),
            &mut output.view_mut(&destination).unwrap(),
        )
        .unwrap();
        assert_eq!(
            output
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn copies_reject_equal_size_wrong_geometry_and_mutable_overlap() {
    let layout = Layout::contiguous([2, 3]).unwrap();
    let transposed = Layout::from_strides([2, 3], [1, 2], 0).unwrap();
    let wrong_shape = Layout::contiguous([3, 2]).unwrap();
    assert!(cpu::CopyPlan::new(&layout, &wrong_shape).is_err());
    let aliased = Layout::from_strides([2, 3], [0, 1], 0).unwrap();
    assert!(cpu::CopyPlan::new(&layout, &aliased).is_err());
    let plan = cpu::CopyPlan::new(&layout, &layout).unwrap();
    let input = bits(6);
    let mut output = bits(6);
    assert!(
        plan.run(
            &input.view(&transposed).unwrap(),
            &mut output.view_mut(&layout).unwrap()
        )
        .is_err()
    );
    assert!(
        plan.run(
            &input.view(&layout).unwrap(),
            &mut output.view_mut(&transposed).unwrap()
        )
        .is_err()
    );
    let source = input.view(&layout).unwrap();
    let mut a = [0.0; 6];
    let mut b = [0.0; 6];
    std::thread::scope(|scope| {
        let plan = &plan;
        scope.spawn(|| {
            plan.run(
                &source,
                &mut TensorViewMut::from_slice(&mut a, &layout).unwrap(),
            )
            .unwrap()
        });
        scope.spawn(|| {
            plan.run(
                &source,
                &mut TensorViewMut::from_slice(&mut b, &layout).unwrap(),
            )
            .unwrap()
        });
    });
    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
}

#[test]
fn dense_empty_matmul_overwrites_zero_contractions_and_graphs_keep_explicit_limits() {
    for cpu in backends() {
        for (m, n, k) in [(2, 3, 0), (0, 3, 7), (3, 0, 2), (0, 0, usize::MAX)] {
            let a = Tensor::zeros([m, k]).unwrap();
            let b = Tensor::zeros([k, n]).unwrap();
            let mut c = Tensor::from_vec([m, n], vec![f32::NAN; m * n]).unwrap();
            no_allocations(|| cpu.matmul(&a, &b, &mut c).unwrap());
            assert!(c.data().iter().all(|x| x.to_bits() == 0));
        }
    }
    assert!(fast_audio::graph::Builder::new([]).is_err());
    assert!(fast_audio::graph::Builder::new([0, 3]).is_err());
    let mut builder = fast_audio::graph::Builder::new([2, 3]).unwrap();
    assert!(builder.constant(Tensor::zeros([]).unwrap()).is_err());
    assert!(builder.constant(Tensor::zeros([0]).unwrap()).is_err());
    let root = builder.input();
    assert!(
        builder
            .layer_norm(
                root,
                &Tensor::zeros([]).unwrap(),
                &Tensor::zeros([]).unwrap(),
                1e-5
            )
            .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_layout_copy_matches_independent_bits_and_rejects_foreign_sessions() {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0).unwrap();
    let foreign = cuda::Cuda::new(0).unwrap();
    for (source, destination) in layouts() {
        let size = source
            .span()
            .map_or(source.offset(), |r| r.end)
            .max(destination.span().map_or(destination.offset(), |r| r.end))
            + 3;
        let plan = cuda::CopyPlan::new(&gpu, &source, &destination).unwrap();
        if source.is_contiguous() && destination.is_contiguous() {
            assert_eq!(plan.workspace_bytes(), 0);
        }
        for reverse in [false, true] {
            let mut input = bits(size);
            if reverse {
                input.data_mut().reverse();
            }
            let initial = Tensor::from_vec([size], vec![-99.0; size]).unwrap();
            let expected = reference(&source, &destination, input.data(), initial.data());
            let input = gpu.upload(&input).unwrap();
            let mut output = gpu.upload(&initial).unwrap();
            no_allocations(|| {
                for _ in 0..3 {
                    plan.run(
                        &input.view(plan.source_layout()).unwrap(),
                        &mut output.view_mut(plan.destination_layout()).unwrap(),
                    )
                    .unwrap();
                }
            });
            assert_eq!(
                gpu.download(&output)
                    .unwrap()
                    .data()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
            let bad = foreign.zeros([size]).unwrap();
            assert!(
                plan.run(
                    &bad.view(&source).unwrap(),
                    &mut output.view_mut(&destination).unwrap()
                )
                .is_err()
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_scalar_and_empty_storage_do_not_enter_invalid_vendor_calls() {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0).unwrap();
    let foreign = cuda::Cuda::new(0).unwrap();
    for dims in [vec![], vec![0], vec![2, 0, 3], vec![usize::MAX, 0]] {
        let host = Tensor::zeros(&dims).unwrap();
        let mut device = gpu.upload(&host).unwrap();
        gpu.copy_from_host(&host, &mut device).unwrap();
        let recovered = gpu.download(&device).unwrap();
        assert_eq!(recovered.shape(), host.shape());
        assert_eq!(recovered.data(), host.data());
        assert!(foreign.download(&device).is_err());
        let layout = Layout::contiguous(&dims).unwrap();
        let plan = cuda::CopyPlan::new(&gpu, &layout, &layout).unwrap();
        let mut output = gpu.zeros(&dims).unwrap();
        no_allocations(|| {
            plan.run(
                &device.view(&layout).unwrap(),
                &mut output.view_mut(&layout).unwrap(),
            )
            .unwrap()
        });
        assert_eq!(gpu.download(&output).unwrap().data(), host.data());
    }
    for (m, n, k) in [(2, 3, 0), (0, 3, 7), (3, 0, 2), (0, 0, usize::MAX)] {
        let a = gpu.zeros([m, k]).unwrap();
        let b = gpu.zeros([k, n]).unwrap();
        let mut c = gpu
            .upload(&Tensor::from_vec([m, n], vec![f32::NAN; m * n]).unwrap())
            .unwrap();
        no_allocations(|| gpu.matmul(&a, &b, &mut c).unwrap());
        assert!(
            gpu.download(&c)
                .unwrap()
                .data()
                .iter()
                .all(|x| x.to_bits() == 0)
        );
    }
}
