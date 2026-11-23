use fast_audio::{Layout, Shape, Tensor, TensorView, TensorViewMut};
use std::collections::HashSet;
#[allow(
    dead_code,
    reason = "Layout tests use only the shared allocation guard."
)]
mod support;
use support::no_allocations;

// Cartesian coordinates are independent of the implementation's flat-index traversal.
fn coordinates(dims: &[usize]) -> Vec<Vec<usize>> {
    if dims.contains(&0) {
        return vec![];
    }
    let mut result = vec![vec![]];
    for &dim in dims {
        result = result
            .into_iter()
            .flat_map(|prefix| {
                (0..dim).map(move |i| {
                    let mut next = prefix.clone();
                    next.push(i);
                    next
                })
            })
            .collect();
    }
    result
}
fn expected_indices(dims: &[usize], strides: &[isize], offset: usize) -> Vec<usize> {
    coordinates(dims)
        .iter()
        .map(|c| {
            usize::try_from(
                offset as i128
                    + c.iter()
                        .zip(strides)
                        .map(|(&i, &s)| i as i128 * s as i128)
                        .sum::<i128>(),
            )
            .unwrap()
        })
        .collect()
}
fn stride_vectors(rank: usize) -> Vec<Vec<isize>> {
    let mut out = vec![vec![]];
    for _ in 0..rank {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                (-4..=4).map(move |s| {
                    let mut next = prefix.clone();
                    next.push(s);
                    next
                })
            })
            .collect();
    }
    out
}

#[test]
fn scalars_empty_axes_and_nonempty_byte_limits_are_distinct() {
    assert_eq!(Shape::new([]).unwrap().len(), 1);
    assert!(!Shape::new([]).unwrap().is_empty());
    for dims in [vec![0], vec![2, 0, 3], vec![usize::MAX, 0, usize::MAX]] {
        assert_eq!(Shape::new(&dims).unwrap().len(), 0);
        assert!(Tensor::zeros(&dims).unwrap().data().is_empty());
        let layout = Layout::contiguous(&dims).unwrap();
        assert_eq!(layout.span(), None);
        assert_eq!(layout.offsets().count(), 0);
        assert!(layout.is_contiguous() && layout.is_non_overlapping());
    }
    assert!(Shape::new([usize::MAX]).is_err());
    assert!(Shape::new([isize::MAX as usize / 4 + 1]).is_err());
    assert!(Tensor::from_vec([], vec![]).is_err());
    assert!(Tensor::from_vec([0], vec![0.0]).is_err());
    let huge_empty = Layout::from_strides([usize::MAX, 0], [isize::MAX, isize::MIN], 0).unwrap();
    assert_eq!(huge_empty.index(&[usize::MAX - 1, 0]), None);
}

#[test]
fn address_spans_and_overlap_proofs_match_coordinate_sets() {
    for dims in [
        vec![1, 1],
        vec![2, 3],
        vec![3, 2],
        vec![1, 2, 3],
        vec![2, 3, 2],
    ] {
        for strides in stride_vectors(dims.len()) {
            let offset = dims
                .iter()
                .zip(&strides)
                .map(
                    |(&d, &s)| {
                        if s < 0 { (d - 1) * s.unsigned_abs() } else { 0 }
                    },
                )
                .sum();
            let layout = Layout::from_strides(&dims, &strides, offset).unwrap();
            let expected = expected_indices(&dims, &strides, offset);
            assert_eq!(layout.offsets().collect::<Vec<_>>(), expected);
            assert_eq!(
                layout.offsets().rev().collect::<Vec<_>>(),
                expected.iter().rev().copied().collect::<Vec<_>>()
            );
            let span = *expected.iter().min().unwrap()..*expected.iter().max().unwrap() + 1;
            assert_eq!(layout.span(), Some(span.clone()));
            for (flat, coordinate) in coordinates(&dims).iter().enumerate() {
                assert_eq!(layout.index(coordinate), Some(expected[flat]));
                assert_eq!(layout.linear_index(flat), Some(expected[flat]));
            }
            assert_eq!(layout.linear_index(expected.len()), None);
            let actual_unique =
                expected.iter().copied().collect::<HashSet<_>>().len() == expected.len();
            if layout.is_non_overlapping() {
                assert!(actual_unique, "false proof: {dims:?} {strides:?}");
            }
            let mut storage = vec![0.0; span.end];
            assert_eq!(
                TensorViewMut::from_slice(&mut storage, &layout).is_ok(),
                layout.is_non_overlapping()
            );
            if layout.is_contiguous() {
                assert_eq!(
                    expected,
                    (offset..offset + expected.len()).collect::<Vec<_>>()
                );
            }
        }
    }
    // Injective but not covered by the deliberately conservative span-separation proof.
    let exotic = Layout::from_strides([2, 3], [3, 2], 0).unwrap();
    assert_eq!(
        expected_indices(&[2, 3], &[3, 2], 0)
            .into_iter()
            .collect::<HashSet<_>>()
            .len(),
        6
    );
    assert!(!exotic.is_non_overlapping());
}

#[test]
fn reshape_acceptance_matches_independent_affine_feasibility() {
    for dims in [vec![2, 3], vec![2, 2, 2]] {
        let count: usize = dims.iter().product();
        let mut targets = vec![
            vec![count],
            vec![1, count],
            vec![count, 1],
            vec![2, count / 2],
            vec![count / 2, 1, 2],
        ];
        if count.is_multiple_of(3) {
            targets.push(vec![3, count / 3]);
        }
        for strides in stride_vectors(dims.len()) {
            let origin = dims
                .iter()
                .zip(&strides)
                .map(|(&d, &s)| if s < 0 { (d - 1) * s.unsigned_abs() } else { 0 })
                .sum();
            let layout = Layout::from_strides(&dims, &strides, origin).unwrap();
            let expected = expected_indices(&dims, &strides, origin);
            for target in &targets {
                // Each affine stride is fixed by a coordinate basis vector. Check
                // every address, independently of stride-contiguous chunk inference.
                let candidate: Vec<_> = (0..target.len())
                    .map(|axis| {
                        if target[axis] == 1 {
                            0
                        } else {
                            let basis: usize = target[axis + 1..].iter().product();
                            expected[basis] as i128 - expected[0] as i128
                        }
                    })
                    .collect();
                let affine = coordinates(target)
                    .iter()
                    .zip(&expected)
                    .all(|(c, &expected)| {
                        origin as i128
                            + c.iter()
                                .zip(&candidate)
                                .map(|(&i, &s)| i as i128 * s)
                                .sum::<i128>()
                            == expected as i128
                    });
                let reshaped = layout.reshape(target);
                assert_eq!(
                    reshaped.is_ok(),
                    affine,
                    "{dims:?} {strides:?} -> {target:?}"
                );
                if let Ok(reshaped) = reshaped {
                    assert_eq!(reshaped.offsets().collect::<Vec<_>>(), expected);
                }
            }
        }
    }
}

#[test]
fn metadata_transforms_preserve_exact_logical_order_without_copying() {
    let source = Tensor::from_vec([2, 3, 4], (0..24).map(|i| i as f32).collect()).unwrap();
    let dense = Layout::contiguous([2, 3, 4]).unwrap();
    let permuted = dense
        .permute(&[1, 0, 2])
        .unwrap()
        .slice(2, 3, 2, -2)
        .unwrap();
    let expected: Vec<_> = (0..3)
        .flat_map(|c| (0..2).flat_map(move |b| [3, 1].map(move |t| ((b * 3 + c) * 4 + t) as f32)))
        .collect();
    assert_eq!(
        source
            .view(&permuted)
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        source.view(&permuted).unwrap().to_tensor().unwrap().data(),
        expected
    );
    let stepped = dense.slice(2, 0, 2, 2).unwrap().reshape([3, 4]).unwrap();
    assert_eq!(stepped.strides(), &[8, 2]);
    assert_eq!(
        stepped.offsets().collect::<Vec<_>>(),
        (0..24).step_by(2).collect::<Vec<_>>()
    );
    let reversed = Layout::from_strides([2, 3], [-3, -1], 5)
        .unwrap()
        .reshape([3, 1, 2])
        .unwrap();
    assert_eq!(
        reversed.offsets().collect::<Vec<_>>(),
        vec![5, 4, 3, 2, 1, 0]
    );
    let transpose = Layout::contiguous([3, 2])
        .unwrap()
        .permute(&[1, 0])
        .unwrap();
    assert!(transpose.reshape([6]).is_err());
    assert_eq!(
        transpose
            .reshape([2, 1, 3])
            .unwrap()
            .offsets()
            .collect::<Vec<_>>(),
        vec![0, 2, 4, 1, 3, 5]
    );
    let broadcast = Layout::from_strides([], [], 2)
        .unwrap()
        .broadcast_to([2, 3, 4])
        .unwrap()
        .reshape([6, 4])
        .unwrap();
    assert_eq!(broadcast.offsets().collect::<Vec<_>>(), vec![2; 24]);
    assert!(source.view(&broadcast).unwrap().as_slice().is_none());
    assert!(Layout::contiguous([0]).unwrap().broadcast_to([1]).is_err());
    let singleton = Layout::from_strides(
        [1, 2, 1, 3, 1, 4, 1],
        [isize::MIN, 12, isize::MAX, 4, 0, 1, isize::MIN],
        0,
    )
    .unwrap();
    assert!(singleton.is_contiguous());
    assert_eq!(singleton.reshape([24]).unwrap().strides(), &[1]);
}

#[test]
fn bounds_and_shape_failures_do_not_use_uninitialized_capacity() {
    assert!(Layout::from_strides([2], [-1], 0).is_err());
    assert!(Layout::from_strides([2], [isize::MIN], 0).is_err());
    assert!(Layout::from_strides([2, 2], [isize::MAX, 1], 0).is_err());
    assert!(Layout::from_strides([2], [], 0).is_err());
    assert!(Layout::from_strides([0], [1], usize::MAX).is_err());
    let dense = Layout::contiguous([2, 3]).unwrap();
    assert!(dense.permute(&[0, 0]).is_err());
    assert!(dense.permute(&[1]).is_err());
    for (start, count, step) in [(3, 1, 1), (1, 3, 1), (0, 2, -1), (0, 1, 0)] {
        assert!(dense.slice(1, start, count, step).is_err());
    }
    assert!(dense.reshape([5]).is_err());
    assert!(dense.broadcast_to([4, 3]).is_err());
    assert_eq!(dense.index(&[0]), None);
    assert_eq!(dense.index(&[2, 0]), None);
    let capacity_only = Vec::<f32>::with_capacity(32);
    assert!(TensorView::from_slice(&capacity_only, &Layout::contiguous([]).unwrap()).is_err());
    let end = Layout::from_strides([0], [isize::MIN], 1).unwrap();
    assert!(
        TensorView::from_slice(&[2.0], &end)
            .unwrap()
            .as_slice()
            .unwrap()
            .is_empty()
    );
    assert!(TensorView::from_slice(&[], &end).is_err());
    let alias = Layout::from_strides([2, 2], [1, 1], 0).unwrap();
    assert!(TensorViewMut::from_slice(&mut [0.0; 3], &alias).is_err());
    let broadcast = Layout::contiguous([1]).unwrap().broadcast_to([3]).unwrap();
    assert!(TensorViewMut::from_slice(&mut [0.0], &broadcast).is_err());
}

#[test]
fn borrowed_mutation_iterators_retain_disjoint_references() {
    let layout = Layout::from_strides([2, 3], [-5, 1], 5).unwrap();
    let mut storage = [99.0; 8];
    {
        let mut view = TensorViewMut::from_slice(&mut storage, &layout).unwrap();
        let mut iterator = view.iter_mut();
        let front = iterator.next().unwrap();
        let back = iterator.next_back().unwrap();
        let middle: Vec<_> = iterator.collect();
        *front = 1.0;
        *back = 6.0;
        for (i, element) in middle.into_iter().enumerate() {
            *element = (i + 2) as f32;
        }
        assert_eq!(
            view.as_view().iter().copied().collect::<Vec<_>>(),
            vec![1., 2., 3., 4., 5., 6.]
        );
        assert_eq!(*view.get_mut(&[0, 1]).unwrap(), 2.0);
    }
    assert_eq!(storage, [4., 5., 6., 99., 99., 1., 2., 3.]);
    let empty = Layout::contiguous([0]).unwrap();
    assert_eq!(
        TensorViewMut::from_slice(&mut [], &empty)
            .unwrap()
            .iter_mut()
            .count(),
        0
    );
    let scalar = Layout::contiguous([]).unwrap();
    let mut data = [3.0];
    *TensorViewMut::from_slice(&mut data, &scalar)
        .unwrap()
        .iter_mut()
        .next()
        .unwrap() = 8.0;
    assert_eq!(data, [8.0]);
}

#[test]
fn prepared_binding_and_iteration_do_not_allocate() {
    let layout = Layout::from_strides([2, 3], [-5, 1], 5).unwrap();
    let mut tensor = Tensor::zeros([8]).unwrap();
    no_allocations(|| {
        for _ in 0..32 {
            let mut view = tensor.view_mut(&layout).unwrap();
            for value in view.iter_mut() {
                *value += 1.0;
            }
            assert_eq!(view.as_view().iter().count(), 6);
        }
    });
    assert_eq!(tensor.data(), &[32., 32., 32., 0., 0., 32., 32., 32.]);
}
