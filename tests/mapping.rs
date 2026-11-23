use fast_audio::{Tensor, cpu::Graph, graph::Builder};
#[allow(
    dead_code,
    reason = "Exact mapping tests do not use the shared approximate comparator."
)]
mod support;
use support::{backends, data, no_allocations};

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn visit(at: usize, axes: &mut [usize], out: &mut Vec<Vec<usize>>) {
        if at == axes.len() {
            out.push(axes.to_vec());
        } else {
            for i in at..axes.len() {
                axes.swap(at, i);
                visit(at + 1, axes, out);
                axes.swap(at, i);
            }
        }
    }
    let mut out = Vec::new();
    visit(0, &mut (0..n).collect::<Vec<_>>(), &mut out);
    out
}

// Enumerate input coordinates, retain the slice, then sort by output coordinates.
// This oracle uses neither coalesced dimensions nor the candidate's mapped indices.
fn sliced_bits(x: &Tensor, ranges: &[(usize, usize)], axes: &[usize]) -> Vec<u32> {
    let mut kept = Vec::new();
    for (i, value) in x.data().iter().enumerate() {
        let mut rest = i;
        let mut coordinate = vec![0; ranges.len()];
        for a in (0..ranges.len()).rev() {
            coordinate[a] = rest % x.shape().dims()[a];
            rest /= x.shape().dims()[a];
        }
        if coordinate
            .iter()
            .zip(ranges)
            .all(|(&v, &(start, len))| v >= start && v - start < len)
        {
            let key = axes
                .iter()
                .map(|&a| coordinate[a] - ranges[a].0)
                .collect::<Vec<_>>();
            kept.push((key, value.to_bits()));
        }
    }
    kept.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    kept.into_iter().map(|(_, v)| v).collect()
}

fn layouts(mut check: impl FnMut(Tensor, &[(usize, usize)], &[usize])) {
    type SliceCase<'a> = (&'a [usize], &'a [(usize, usize)]);
    let cases: &[SliceCase<'_>] = &[
        (&[1, 1, 1, 1], &[(0, 1), (0, 1), (0, 1), (0, 1)]),
        (&[257], &[(1, 255)]),
        (&[5, 7], &[(1, 3), (2, 5)]),
        (&[2, 3, 7], &[(0, 2), (1, 2), (1, 5)]),
        (&[3, 4, 5, 11], &[(1, 2), (1, 3), (2, 2), (2, 7)]),
        (&[1, 3, 1, 257], &[(0, 1), (0, 3), (0, 1), (1, 255)]),
    ];
    for &(shape, ranges) in cases {
        for axes in permutations(shape.len()) {
            let mut x = data(shape);
            let special = [0x80000000, 0, 0x7fc12345, 0x7f800000, 0xff800000];
            for (i, v) in x.data_mut().iter_mut().enumerate().step_by(7) {
                *v = f32::from_bits(special[i % special.len()]);
            }
            check(x, ranges, &axes);
        }
    }
}

#[test]
fn sliced_layouts_preserve_bits_for_all_permutations_singletons_tails_and_rebinding() {
    for cpu in backends() {
        layouts(|mut x, ranges, axes| {
            let mut b = Builder::new(x.shape().dims()).unwrap();
            let output = b
                .layout(
                    b.input(),
                    ranges,
                    axes,
                    [ranges.iter().map(|r| r.1).product::<usize>()],
                )
                .unwrap();
            let mut g = Graph::new(cpu, b.finish(output).unwrap()).unwrap();
            let mut out = Tensor::zeros(g.output_shape()).unwrap();
            for _ in 0..2 {
                let expected = sliced_bits(&x, ranges, axes);
                g.run(&x, &mut out).unwrap();
                no_allocations(|| g.run(&x, &mut out).unwrap());
                assert_eq!(
                    out.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected
                );
                x.data_mut().reverse();
            }
        });
    }
}

fn broadcast_expected(x: &Tensor, rhs: &Tensor) -> Vec<u32> {
    x.data()
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let mut coordinate = vec![0; x.shape().dims().len()];
            let mut remaining = i;
            for a in (0..coordinate.len()).rev() {
                coordinate[a] = remaining % x.shape().dims()[a];
                remaining /= x.shape().dims()[a];
            }
            let mut at = 0;
            for (a, &dim) in rhs.shape().dims().iter().enumerate() {
                at = at * dim
                    + if dim == 1 {
                        0
                    } else {
                        coordinate[a + coordinate.len() - rhs.shape().dims().len()]
                    };
            }
            (v * rhs.data()[at]).to_bits()
        })
        .collect()
}

const BROADCASTS: &[&[usize]] = &[
    &[1],
    &[7],
    &[5, 1],
    &[3, 1, 7],
    &[2, 1, 5, 1],
    &[2, 3, 5, 7],
];

#[test]
fn broadcast_mapping_matches_coordinate_oracle_and_changes_with_input() {
    for cpu in backends() {
        for shape in BROADCASTS {
            let mut x = data([2, 3, 5, 7]);
            let rhs = data(shape);
            let mut b = Builder::new(x.shape().dims()).unwrap();
            let r = b.constant(rhs.clone()).unwrap();
            let output = b.multiply_broadcast(b.input(), r).unwrap();
            let mut g = Graph::new(cpu, b.finish(output).unwrap()).unwrap();
            let mut out = Tensor::zeros(g.output_shape()).unwrap();
            for _ in 0..2 {
                let expected = broadcast_expected(&x, &rhs);
                g.run(&x, &mut out).unwrap();
                no_allocations(|| g.run(&x, &mut out).unwrap());
                assert_eq!(
                    out.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected
                );
                x.data_mut().reverse();
            }
        }
    }
}

#[cfg(feature = "cudnn")]
#[test]
fn cuda_mapped_layouts_and_broadcasts_preserve_bits_with_rebinding() {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0).unwrap();
    layouts(|mut x, ranges, axes| {
        let mut b = Builder::new(x.shape().dims()).unwrap();
        let output = b
            .layout(
                b.input(),
                ranges,
                axes,
                [ranges.iter().map(|r| r.1).product::<usize>()],
            )
            .unwrap();
        let mut g = cuda::Graph::new(&gpu, b.finish(output).unwrap()).unwrap();
        for _ in 0..2 {
            let expected = sliced_bits(&x, ranges, axes);
            let input = gpu.upload(&x).unwrap();
            let mut out = gpu.zeros(g.output_shape()).unwrap();
            g.run(&input, &mut out).unwrap();
            no_allocations(|| g.run(&input, &mut out).unwrap());
            assert_eq!(
                gpu.download(&out)
                    .unwrap()
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
            x.data_mut().reverse();
        }
    });
    for shape in BROADCASTS {
        let mut x = data([2, 3, 5, 7]);
        let rhs = data(shape);
        let mut b = Builder::new(x.shape().dims()).unwrap();
        let r = b.constant(rhs.clone()).unwrap();
        let output = b.multiply_broadcast(b.input(), r).unwrap();
        let mut g = cuda::Graph::new(&gpu, b.finish(output).unwrap()).unwrap();
        for _ in 0..2 {
            let expected = broadcast_expected(&x, &rhs);
            let input = gpu.upload(&x).unwrap();
            let mut out = gpu.zeros(g.output_shape()).unwrap();
            g.run(&input, &mut out).unwrap();
            no_allocations(|| g.run(&input, &mut out).unwrap());
            assert_eq!(
                gpu.download(&out)
                    .unwrap()
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
            x.data_mut().reverse();
        }
    }
}
