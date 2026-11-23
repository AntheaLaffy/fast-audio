use fast_audio::{GradientMode, Layout, ReduceOp, Shape, Tensor};
use std::collections::BTreeSet;

pub const SENTINEL: u32 = 0x7fc12345;
#[derive(Clone)]
pub struct Case {
    pub source: Layout,
    pub axes: Vec<usize>,
    pub keep: bool,
}
pub fn cases() -> Vec<Case> {
    let mut out = vec![
        (Layout::from_strides([], [], 2).unwrap(), vec![], false),
        (
            Layout::from_strides([2, 3], [-7, 2], 7).unwrap(),
            vec![],
            true,
        ),
        (Layout::contiguous([3, 5, 7]).unwrap(), vec![1], false),
        (Layout::contiguous([3, 5, 7]).unwrap(), vec![2, 0], true),
        (
            Layout::from_strides([3, 2, 5], [-2, 19, -4], 20).unwrap(),
            vec![2, 0],
            false,
        ),
        (
            Layout::from_strides([4, 3, 33], [0, 1, 0], 2).unwrap(),
            vec![2, 0],
            true,
        ),
        (
            Layout::from_strides([3, 4, 7], [1, 1, 2], 1).unwrap(),
            vec![0, 2],
            false,
        ),
        (
            Layout::contiguous([2, 3, 2, 2, 3])
                .unwrap()
                .permute(&[4, 2, 0, 3, 1])
                .unwrap(),
            vec![1, 3],
            true,
        ),
        (
            Layout::from_strides([2, 17, 37], [1100, -50, 2], 805).unwrap(),
            vec![1],
            false,
        ),
        (
            Layout::from_strides([257, 3], [-1, 260], 256).unwrap(),
            vec![1, 0],
            false,
        ),
        (Layout::contiguous([11, 32]).unwrap(), vec![1], true),
        (Layout::contiguous([9, 33]).unwrap(), vec![1], false),
        (Layout::contiguous([13, 257]).unwrap(), vec![1], false),
        (Layout::contiguous([33, 19]).unwrap(), vec![0], true),
        (
            Layout::from_strides([2, 0, 3], [isize::MIN, isize::MAX, isize::MIN], 3).unwrap(),
            vec![2],
            false,
        ),
        (
            Layout::from_strides([2, 0, 3], [isize::MIN, isize::MAX, isize::MIN], 3).unwrap(),
            vec![1],
            false,
        ),
        (
            Layout::from_strides([2, 0, 3], [isize::MAX, isize::MIN, isize::MIN], 3).unwrap(),
            vec![0, 1, 2],
            true,
        ),
        (
            Layout::contiguous([usize::MAX, usize::MAX, 0]).unwrap(),
            vec![0, 1],
            false,
        ),
        (
            Layout::contiguous([usize::MAX, 0]).unwrap(),
            vec![1, 0],
            false,
        ),
        (
            Layout::from_strides(vec![1; 128], vec![isize::MIN; 128], 2).unwrap(),
            vec![],
            true,
        ),
    ]
    .into_iter()
    .map(|(source, axes, keep)| Case { source, axes, keep })
    .collect::<Vec<_>>();
    // Generated permutations, sign/padding and nonadjacent axis sets complement fixtures.
    for seed in 0..20 {
        let dims = [2 + seed % 3, 3, 2 + seed % 4];
        let mut source = padded(&dims, seed);
        if seed % 2 == 0 {
            source = source.permute(&[2, 0, 1]).unwrap();
        }
        out.push(Case {
            source,
            axes: if seed % 3 == 0 {
                vec![2, 0]
            } else {
                vec![seed % 3]
            },
            keep: seed % 2 == 0,
        });
    }
    let other_policy: Vec<_> = out
        .iter()
        .cloned()
        .map(|mut c| {
            c.keep = !c.keep;
            c
        })
        .collect();
    out.extend(other_policy);
    out
}
pub fn output_dims(input: &[usize], axes: &[usize], keep: bool) -> Vec<usize> {
    input
        .iter()
        .enumerate()
        .filter_map(|(a, &d)| {
            if axes.contains(&a) {
                keep.then_some(1)
            } else {
                Some(d)
            }
        })
        .collect()
}
pub fn padded(dims: &[usize], seed: usize) -> Layout {
    if dims.contains(&0) {
        return Layout::from_strides(dims, vec![isize::MIN; dims.len()], 3).unwrap();
    }
    let mut strides = vec![isize::MIN; dims.len()];
    let (mut step, mut origin) = (2, 3);
    for a in (0..dims.len()).rev() {
        if dims[a] == 1 {
            continue;
        }
        strides[a] = if (a + seed) % 2 == 0 {
            step as isize
        } else {
            origin += (dims[a] - 1) * step;
            -(step as isize)
        };
        step = step * dims[a] + 3;
    }
    Layout::from_strides(dims, strides, origin).unwrap()
}
pub fn coordinates(dims: &[usize], mut flat: usize) -> Vec<usize> {
    let mut coords = vec![0; dims.len()];
    for a in (0..dims.len()).rev() {
        coords[a] = flat % dims[a];
        flat /= dims[a];
    }
    coords
}
pub fn address(layout: &Layout, coords: &[usize]) -> usize {
    (layout.offset() as i128
        + coords
            .iter()
            .zip(layout.strides())
            .map(|(&i, &s)| i as i128 * s as i128)
            .sum::<i128>()) as usize
}
pub fn addresses(layout: &Layout) -> Vec<usize> {
    (0..layout.shape().len())
        .map(|i| address(layout, &coordinates(layout.shape().dims(), i)))
        .collect()
}
pub fn buffer(layout: &Layout, salt: usize, holes: bool) -> Tensor {
    let len = layout.span().map_or(layout.offset(), |s| s.end) + 2;
    let values = (0..len)
        .map(|i| {
            if holes {
                f32::from_bits(SENTINEL)
            } else {
                (((i * 31 + salt * 17) % 101) as f32 - 50.) / 53.
            }
        })
        .collect();
    Tensor::from_vec([len], values).unwrap()
}
pub fn bits(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|x| x.to_bits()).collect()
}
fn result_flat(c: &Case, coordinates: &[usize]) -> usize {
    let dims = output_dims(c.source.shape().dims(), &c.axes, c.keep);
    let reduced: Vec<_> = coordinates
        .iter()
        .enumerate()
        .filter_map(|(a, &n)| {
            if c.axes.contains(&a) {
                c.keep.then_some(0)
            } else {
                Some(n)
            }
        })
        .collect();
    dims.iter()
        .zip(reduced)
        .fold(0, |flat, (&d, i)| flat * d + i)
}
pub fn count(c: &Case) -> Option<usize> {
    if c.axes.iter().any(|&a| c.source.shape().dims()[a] == 0) {
        Some(0)
    } else {
        c.axes
            .iter()
            .try_fold(1usize, |n, &a| n.checked_mul(c.source.shape().dims()[a]))
    }
}
pub fn logical_values(c: &Case, source: &Tensor) -> Vec<f64> {
    addresses(&c.source)
        .into_iter()
        .map(|a| source.data()[a] as f64)
        .collect()
}
pub fn reference(c: &Case, op: ReduceOp, logical: &[f64]) -> Vec<f64> {
    let dims = output_dims(c.source.shape().dims(), &c.axes, c.keep);
    let mut sums = vec![0.; Shape::new(dims).unwrap().len()];
    for (i, &value) in logical.iter().enumerate() {
        let coords = coordinates(c.source.shape().dims(), i);
        sums[result_flat(c, &coords)] += value;
    }
    if op == ReduceOp::Mean {
        // Mathematical FP64 oracle uses the exact integer extent; rounded-count
        // semantics have a separate real large-extent boundary test.
        for value in &mut sums {
            *value /= count(c).unwrap() as f64;
        }
    }
    sums
}
pub fn backward_reference(c: &Case, op: ReduceOp, upstream: &Layout, data: &Tensor) -> Vec<f64> {
    let values: Vec<_> = addresses(upstream)
        .into_iter()
        .map(|a| data.data()[a] as f64)
        .collect();
    (0..c.source.shape().len())
        .map(|i| {
            let coord = coordinates(c.source.shape().dims(), i);
            let value = values[result_flat(c, &coord)];
            if op == ReduceOp::Mean {
                value / count(c).unwrap() as f64
            } else {
                value
            }
        })
        .collect()
}
pub fn check_value(actual: f32, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert!(
            (actual as f64 - expected).abs() <= 2e-5 + 2e-5 * expected.abs(),
            "{actual} != {expected}"
        );
    }
}
pub fn verify_forward(
    c: &Case,
    op: ReduceOp,
    source: &Tensor,
    dest: &Layout,
    seed: &Tensor,
    actual: &Tensor,
) {
    let expected = reference(c, op, &logical_values(c, source));
    for (&a, &value) in addresses(dest).iter().zip(&expected) {
        check_value(actual.data()[a], value);
        if count(c) == Some(0) && op == ReduceOp::Mean {
            assert_eq!(actual.data()[a].to_bits(), 0x7fc00000);
        }
    }
    check_holes(dest, seed, actual);
}
pub fn check_holes(layout: &Layout, seed: &Tensor, actual: &Tensor) {
    let selected: BTreeSet<_> = addresses(layout).into_iter().collect();
    for (i, (&before, &after)) in seed.data().iter().zip(actual.data()).enumerate() {
        if !selected.contains(&i) {
            assert_eq!(before.to_bits(), after.to_bits(), "hole {i}");
        }
    }
}
pub fn gradient_seed(layout: &Layout, salt: usize) -> Tensor {
    let mut seed = buffer(layout, salt, true);
    for (i, a) in addresses(layout).into_iter().enumerate() {
        seed.data_mut()[a] = (i % 7) as f32 / 13. + 0.2;
    }
    seed
}
pub fn verify_backward(
    c: &Case,
    op: ReduceOp,
    upstream: (&Layout, &Tensor),
    destination: (&Layout, &Tensor, &Tensor),
    mode: GradientMode,
) {
    let (upstream, dy) = upstream;
    let (dest, seed, actual) = destination;
    let expected = backward_reference(c, op, upstream, dy);
    for (&a, &value) in addresses(dest).iter().zip(&expected) {
        check_value(
            actual.data()[a],
            if mode == GradientMode::Accumulate {
                seed.data()[a] as f64 + value
            } else {
                value
            },
        );
    }
    check_holes(dest, seed, actual);
}
pub fn adjoint_and_difference(
    c: &Case,
    op: ReduceOp,
    source: &Tensor,
    output: (&Layout, &Tensor),
    upstream: (&Layout, &Tensor),
    gradient: (&Layout, &Tensor),
) {
    let (out, y) = output;
    let (upstream, dy) = upstream;
    let (gradient, dx) = gradient;
    if c.source.shape().is_empty() {
        return;
    }
    let input = logical_values(c, source);
    let upstream_values: Vec<_> = addresses(upstream)
        .into_iter()
        .map(|a| dy.data()[a] as f64)
        .collect();
    let gradients: Vec<_> = addresses(gradient)
        .into_iter()
        .map(|a| dx.data()[a] as f64)
        .collect();
    let lhs: f64 = addresses(out)
        .iter()
        .zip(&upstream_values)
        .map(|(&a, &d)| y.data()[a] as f64 * d)
        .sum();
    let rhs: f64 = input.iter().zip(&gradients).map(|(&x, &d)| x * d).sum();
    assert!(
        (lhs - rhs).abs() <= 2e-5 + 2e-5 * lhs.abs(),
        "adjoint {lhs} != {rhs}"
    );
    for i in [0, input.len() / 2, input.len() - 1] {
        let mut perturbed = input.clone();
        let h = 1e-4;
        perturbed[i] += h;
        let plus: f64 = reference(c, op, &perturbed)
            .iter()
            .zip(&upstream_values)
            .map(|(y, d)| y * d)
            .sum();
        perturbed[i] -= 2. * h;
        let minus: f64 = reference(c, op, &perturbed)
            .iter()
            .zip(&upstream_values)
            .map(|(y, d)| y * d)
            .sum();
        assert!(
            (gradients[i] - (plus - minus) / (2. * h)).abs() <= 2e-5 + 2e-5 * gradients[i].abs()
        );
    }
}
