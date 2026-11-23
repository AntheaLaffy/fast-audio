use fast_audio::{BinaryOp as B, Layout, PointwiseStrategy as S, Tensor, UnaryOp as U, cpu};
#[allow(
    dead_code,
    reason = "Pointwise tests supply independent coordinate and FP64 references."
)]
mod support;
use support::no_allocations;

const UNARY: [U; 12] = [
    U::Neg,
    U::Abs,
    U::Square,
    U::Reciprocal,
    U::Sqrt,
    U::Exp,
    U::Log,
    U::Relu,
    U::Sigmoid,
    U::Tanh,
    U::Silu,
    U::Softplus,
];
const BINARY: [B; 4] = [B::Add, B::Sub, B::Mul, B::Div];
const SENTINEL: f32 = f32::from_bits(0x7fc12345);

#[derive(Clone)]
struct Case {
    a: Layout,
    b: Layout,
    out: Layout,
}
fn coordinates(dims: &[usize]) -> Vec<Vec<usize>> {
    if dims.contains(&0) {
        return vec![];
    }
    let mut current = vec![0; dims.len()];
    let mut result = vec![];
    loop {
        result.push(current.clone());
        let mut axis = dims.len();
        loop {
            if axis == 0 {
                return result;
            }
            axis -= 1;
            current[axis] += 1;
            if current[axis] < dims[axis] {
                break;
            }
            current[axis] = 0;
        }
    }
}
fn address(layout: &Layout, common: &[usize]) -> usize {
    let prefix = common.len() - layout.shape().dims().len();
    let mut offset = layout.offset() as isize;
    for (axis, (&dim, &stride)) in layout
        .shape()
        .dims()
        .iter()
        .zip(layout.strides())
        .enumerate()
    {
        let coordinate = if dim == 1 { 0 } else { common[prefix + axis] };
        offset += coordinate as isize * stride;
    }
    offset as usize
}
fn extent(layout: &Layout) -> usize {
    layout.span().map_or(layout.offset(), |s| s.end) + 3
}
fn input(layout: &Layout, salt: usize, positive: bool) -> Tensor {
    let mut values = vec![SENTINEL; extent(layout)];
    for coordinate in coordinates(layout.shape().dims()) {
        let i = address(layout, &coordinate);
        values[i] = if positive {
            0.125 + ((i * 17 + salt * 13) % 97) as f32 / 16.
        } else {
            ((i * 17 + salt * 13) % 127) as f32 / 16. - 3.9375
        };
    }
    Tensor::from_vec([values.len()], values).unwrap()
}
fn output(layout: &Layout) -> Tensor {
    Tensor::from_vec([extent(layout)], vec![SENTINEL; extent(layout)]).unwrap()
}
fn bits(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|x| x.to_bits()).collect()
}
fn unary_reference(op: U, x: f64) -> f64 {
    match op {
        U::Neg => -x,
        U::Abs => x.abs(),
        U::Square => x * x,
        U::Reciprocal => x.recip(),
        U::Sqrt => x.sqrt(),
        U::Exp => x.exp(),
        U::Log => x.ln(),
        U::Relu => {
            if x < 0. {
                0.
            } else {
                x
            }
        }
        U::Sigmoid => 1. / (1. + (-x).exp()),
        U::Tanh => x.tanh(),
        U::Silu => {
            if x == f64::NEG_INFINITY {
                -0.
            } else {
                x / (1. + (-x).exp())
            }
        }
        U::Softplus => {
            if x > 0. {
                x + (-x).exp().ln_1p()
            } else {
                x.exp().ln_1p()
            }
        }
    }
}
fn binary_reference(op: B, a: f64, b: f64) -> f64 {
    match op {
        B::Add => a + b,
        B::Sub => a - b,
        B::Mul => a * b,
        B::Div => a / b,
    }
}
fn check(actual: f32, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{actual} should be NaN");
    } else if expected.is_infinite() {
        assert_eq!(actual as f64, expected);
    } else {
        assert!(
            (actual as f64 - expected).abs() <= 2e-6 + 3e-6 * expected.abs(),
            "{actual} != {expected}"
        );
        if expected == 0. {
            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        }
    }
}
fn check_unary(case: &Case, op: U, source: &Tensor, actual: &Tensor) {
    let layout = case.a.broadcast_to(case.out.shape().dims()).unwrap();
    let mut selected = vec![false; actual.data().len()];
    for coordinate in coordinates(case.out.shape().dims()) {
        let index = address(&case.out, &coordinate);
        selected[index] = true;
        check(
            actual.data()[index],
            unary_reference(op, source.data()[address(&layout, &coordinate)] as f64),
        );
    }
    for (i, value) in actual.data().iter().enumerate() {
        if !selected[i] {
            assert_eq!(value.to_bits(), SENTINEL.to_bits());
        }
    }
}
fn check_binary(case: &Case, op: B, a: &Tensor, b: &Tensor, actual: &Tensor) {
    let mut selected = vec![false; actual.data().len()];
    for coordinate in coordinates(case.out.shape().dims()) {
        let index = address(&case.out, &coordinate);
        selected[index] = true;
        check(
            actual.data()[index],
            binary_reference(
                op,
                a.data()[address(&case.a, &coordinate)] as f64,
                b.data()[address(&case.b, &coordinate)] as f64,
            ),
        );
    }
    for (i, value) in actual.data().iter().enumerate() {
        if !selected[i] {
            assert_eq!(value.to_bits(), SENTINEL.to_bits());
        }
    }
}
fn generated_layout(dims: &[usize], seed: usize, read: bool) -> Layout {
    let mut axes: Vec<_> = (0..dims.len()).collect();
    if !axes.is_empty() {
        let n = axes.len();
        axes.rotate_left(seed % n);
    }
    if seed & 1 == 0 {
        axes.reverse();
    }
    let mut strides = vec![0; dims.len()];
    let mut step = 1 + seed % 2;
    let mut offset = 2;
    for axis in axes {
        let sign = if (seed >> (axis % 8)) & 1 == 0 { 1 } else { -1 };
        let stride = if read && (seed + axis) % 5 == 0 {
            0
        } else {
            step as isize * sign
        };
        strides[axis] = stride;
        if stride < 0 {
            offset += (dims[axis] - 1) * stride.unsigned_abs();
        }
        step = step * dims[axis] + 1;
    }
    Layout::from_strides(dims, strides, offset).unwrap()
}
fn cases(generated: usize) -> Vec<Case> {
    let dense = |d: &[usize]| Layout::contiguous(d).unwrap();
    let mut out = vec![
        Case {
            a: dense(&[2, 3]),
            b: dense(&[2, 3]),
            out: dense(&[2, 3]),
        },
        Case {
            a: dense(&[2, 1]),
            b: dense(&[3]),
            out: dense(&[2, 3]),
        },
        Case {
            a: dense(&[3]),
            b: dense(&[2, 1]),
            out: dense(&[2, 3]),
        },
        Case {
            a: dense(&[]),
            b: dense(&[257]),
            out: dense(&[257]),
        },
        Case {
            a: dense(&[257]),
            b: dense(&[]),
            out: dense(&[257]),
        },
        Case {
            a: Layout::from_strides([2, 3], [1, 1], 1).unwrap(),
            b: dense(&[]),
            out: Layout::from_strides([2, 3], [9, -2], 4).unwrap(),
        },
        Case {
            a: Layout::from_strides([2, 3], [7, -2], 5).unwrap(),
            b: Layout::from_strides([1, 3], [0, 3], 1).unwrap(),
            out: Layout::from_strides([2, 3], [9, -2], 4).unwrap(),
        },
        Case {
            a: Layout::from_strides([3, 2], [1, 3], 2).unwrap(),
            b: dense(&[]),
            out: Layout::from_strides([3, 2], [1, 3], 2).unwrap(),
        },
        Case {
            a: Layout::from_strides([2, 3], [-3, -1], 5).unwrap(),
            b: dense(&[]),
            out: Layout::from_strides([2, 3], [-3, -1], 6).unwrap(),
        },
        Case {
            a: dense(&[1, 2, 1, 3]),
            b: dense(&[2, 1]),
            out: dense(&[1, 2, 2, 3]),
        },
        Case {
            a: dense(&[0, 3]),
            b: dense(&[1, 3]),
            out: dense(&[0, 3]),
        },
        Case {
            a: dense(&[]),
            b: dense(&[]),
            out: Layout::from_strides([], [], 2).unwrap(),
        },
        Case {
            a: dense(&[2, 3, 2, 2, 3]).permute(&[4, 2, 0, 3, 1]).unwrap(),
            b: dense(&[2, 1]),
            out: generated_layout(&[3, 2, 2, 2, 3], 7, false),
        },
        Case {
            a: dense(&[1; 24]),
            b: dense(&[]),
            out: dense(&[1; 24]),
        },
        Case {
            a: dense(&[0, usize::MAX, usize::MAX]),
            b: dense(&[]),
            out: dense(&[0, usize::MAX, usize::MAX]),
        },
    ];
    for seed in 0..generated {
        let rank = 1 + seed % 5;
        let dims: Vec<_> = (0..rank).map(|axis| 2 + (seed + axis) % 3).collect();
        let a_dims: Vec<_> = dims
            .iter()
            .enumerate()
            .map(|(axis, &d)| if (seed + axis) % 3 == 0 { 1 } else { d })
            .collect();
        let prefix = seed % rank;
        let b_dims: Vec<_> = dims[prefix..]
            .iter()
            .enumerate()
            .map(|(axis, &d)| {
                if a_dims[prefix + axis] == 1 || (seed + axis) % 3 != 0 {
                    d
                } else {
                    1
                }
            })
            .collect();
        // Leading dropped axes must still be represented by the other input.
        let mut a_dims = a_dims;
        a_dims[..prefix].copy_from_slice(&dims[..prefix]);
        out.push(Case {
            a: generated_layout(&a_dims, seed + 3, true),
            b: generated_layout(&b_dims, seed + 11, true),
            out: generated_layout(&dims, seed + 17, false),
        });
    }
    out
}

#[test]
fn cpu_pointwise_matches_independent_coordinates_and_fp64_for_all_operations() {
    for case in cases(64) {
        for op in UNARY {
            let layout = case.a.broadcast_to(case.out.shape().dims()).unwrap();
            let plan = cpu::UnaryPlan::new(op, &layout, &case.out).unwrap();
            assert_eq!(plan.operation(), op);
            assert_eq!(plan.source_layout(), &layout);
            assert_eq!(plan.destination_layout(), &case.out);
            assert_eq!(plan.workspace_bytes(), 0);
            for salt in [1, 5] {
                let source = input(
                    &layout,
                    salt,
                    matches!(op, U::Sqrt | U::Log | U::Reciprocal),
                );
                let before = bits(&source);
                let mut actual = output(&case.out);
                no_allocations(|| {
                    plan.run(
                        &source.view(plan.source_layout()).unwrap(),
                        &mut actual.view_mut(plan.destination_layout()).unwrap(),
                    )
                    .unwrap()
                });
                check_unary(&case, op, &source, &actual);
                let first = bits(&actual);
                plan.run(
                    &source.view(&layout).unwrap(),
                    &mut actual.view_mut(&case.out).unwrap(),
                )
                .unwrap();
                assert_eq!(bits(&actual), first);
                assert_eq!(bits(&source), before);
            }
        }
        for op in BINARY {
            let plan = cpu::BinaryPlan::new(op, &case.a, &case.b, &case.out).unwrap();
            assert_eq!(plan.operation(), op);
            assert_eq!(plan.left_layout(), &case.a);
            assert_eq!(plan.right_layout(), &case.b);
            assert_eq!(plan.destination_layout(), &case.out);
            assert_eq!(plan.workspace_bytes(), 0);
            for salt in [2, 9] {
                let a = input(&case.a, salt, false);
                let b = input(&case.b, salt + 2, true);
                let before = (bits(&a), bits(&b));
                let mut actual = output(&case.out);
                no_allocations(|| {
                    plan.run(
                        &a.view(plan.left_layout()).unwrap(),
                        &b.view(plan.right_layout()).unwrap(),
                        &mut actual.view_mut(plan.destination_layout()).unwrap(),
                    )
                    .unwrap()
                });
                check_binary(&case, op, &a, &b, &actual);
                let first = bits(&actual);
                plan.run(
                    &a.view(&case.a).unwrap(),
                    &b.view(&case.b).unwrap(),
                    &mut actual.view_mut(&case.out).unwrap(),
                )
                .unwrap();
                assert_eq!(bits(&actual), first);
                assert_eq!((bits(&a), bits(&b)), before);
            }
        }
    }
}

fn special_values() -> Vec<f32> {
    vec![
        f32::NEG_INFINITY,
        -1000.,
        -90.,
        -80.,
        -1.,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.,
        0.,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1.,
        80.,
        90.,
        1000.,
        f32::INFINITY,
        f32::from_bits(0x7fc23456),
        f32::from_bits(0xffc45678),
    ]
}
fn check_special_unary(op: U, values: &[f32], actual: &[f32]) {
    for (&x, &y) in values.iter().zip(actual) {
        if op == U::Neg {
            assert_eq!(y.to_bits(), x.to_bits() ^ 0x80000000);
        } else if op == U::Abs {
            assert_eq!(y.to_bits(), x.to_bits() & 0x7fffffff);
        } else if op == U::Relu {
            assert_eq!(y.to_bits(), if x < 0. { 0 } else { x.to_bits() });
        } else {
            let reference = unary_reference(op, x as f64);
            let rounded = reference as f32;
            // Overflow/underflow are FP32 category gates, not a finite FP64 error bound.
            if rounded.is_infinite() || rounded == 0. {
                check(y, rounded as f64);
            } else {
                check(y, reference);
            }
        }
        if matches!(op, U::Sigmoid | U::Softplus) && x == -90. {
            assert!(y > 0.);
        }
    }
}
#[test]
fn cpu_pointwise_stable_extremes_and_exceptional_bits_match_the_contract() {
    let values = special_values();
    let layout = Layout::contiguous([values.len()]).unwrap();
    let source = Tensor::from_vec([values.len()], values.clone()).unwrap();
    for op in UNARY {
        let plan = cpu::UnaryPlan::new(op, &layout, &layout).unwrap();
        let mut actual = Tensor::zeros([values.len()]).unwrap();
        plan.run(
            &source.view(&layout).unwrap(),
            &mut actual.view_mut(&layout).unwrap(),
        )
        .unwrap();
        check_special_unary(op, &values, actual.data());
    }
    for op in BINARY {
        for &rhs in &[-0., 0., -2., 2., f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let scalar = Layout::contiguous([]).unwrap();
            let other = Tensor::from_vec([], vec![rhs]).unwrap();
            let plan = cpu::BinaryPlan::new(op, &layout, &scalar, &layout).unwrap();
            let mut actual = Tensor::zeros([values.len()]).unwrap();
            plan.run(
                &source.view(&layout).unwrap(),
                &other.view(&scalar).unwrap(),
                &mut actual.view_mut(&layout).unwrap(),
            )
            .unwrap();
            for (&a, &y) in values.iter().zip(actual.data()) {
                check(
                    y,
                    (binary_reference(op, a as f64, rhs as f64) as f32) as f64,
                );
            }
        }
    }
}

#[test]
fn cpu_pointwise_rejects_invalid_preparation_and_changed_geometry_before_writing() {
    let a = Layout::contiguous([2, 1]).unwrap();
    let b = Layout::contiguous([3]).unwrap();
    let out = Layout::contiguous([2, 3]).unwrap();
    let wrong = Layout::contiguous([3, 2]).unwrap();
    assert!(cpu::UnaryPlan::new(U::Neg, &out, &wrong).is_err());
    assert!(cpu::BinaryPlan::new(B::Add, &a, &b, &wrong).is_err());
    assert!(cpu::BinaryPlan::new(B::Add, &out, &wrong, &out).is_err());
    let unproved = Layout::from_strides([2, 3], [3, 2], 0).unwrap();
    assert!(!unproved.is_non_overlapping());
    assert!(cpu::UnaryPlan::new(U::Neg, &out, &unproved).is_err());
    assert!(cpu::BinaryPlan::new(B::Add, &a, &b, &unproved).is_err());
    let plan = cpu::BinaryPlan::new(B::Add, &a, &b, &out).unwrap();
    let source = Tensor::from_vec([20], vec![2.; 20]).unwrap();
    let mut actual = Tensor::from_vec([20], vec![9.; 20]).unwrap();
    let changed = Layout::from_strides([2, 1], [2, 0], 1).unwrap();
    assert!(
        plan.run(
            &source.view(&changed).unwrap(),
            &source.view(&b).unwrap(),
            &mut actual.view_mut(&out).unwrap()
        )
        .is_err()
    );
    assert!(
        plan.run(
            &source.view(&a).unwrap(),
            &source.view(&b).unwrap(),
            &mut actual.view_mut(&wrong).unwrap()
        )
        .is_err()
    );
    let changed_b = Layout::from_strides([3], [2], 0).unwrap();
    assert!(
        plan.run(
            &source.view(&a).unwrap(),
            &source.view(&changed_b).unwrap(),
            &mut actual.view_mut(&out).unwrap()
        )
        .is_err()
    );
    let unary = cpu::UnaryPlan::new(U::Neg, &out, &out).unwrap();
    assert!(
        unary
            .run(
                &source.view(&wrong).unwrap(),
                &mut actual.view_mut(&out).unwrap()
            )
            .is_err()
    );
    assert!(
        unary
            .run(
                &source.view(&out).unwrap(),
                &mut actual.view_mut(&wrong).unwrap()
            )
            .is_err()
    );
    assert_eq!(actual.data(), &[9.; 20]);
    let empty = Layout::contiguous([0, 3]).unwrap();
    let plan = cpu::BinaryPlan::new(B::Div, &empty, &b, &empty).unwrap();
    assert!(
        plan.run(
            &source.view(&empty).unwrap(),
            &source.view(&changed_b).unwrap(),
            &mut actual.view_mut(&empty).unwrap()
        )
        .is_err()
    );
    assert_eq!(actual.data(), &[9.; 20]);
    assert!(
        cpu::BinaryPlan::new(B::Add, &empty, &Layout::contiguous([2, 1]).unwrap(), &empty).is_err()
    );
}

#[test]
fn pointwise_coalescing_and_metadata_scale_with_rank_instead_of_element_count() {
    let layout = Layout::contiguous([2, 3, 5]).unwrap();
    let plan = cpu::UnaryPlan::new(U::Neg, &layout, &layout).unwrap();
    assert_eq!(
        (plan.strategy(), plan.coalesced_rank(), plan.inner_len()),
        (S::Contiguous, 1, 30)
    );
    let permuted = layout.permute(&[2, 0, 1]).unwrap();
    let plan = cpu::UnaryPlan::new(U::Neg, &permuted, &permuted).unwrap();
    assert_eq!(
        (plan.strategy(), plan.coalesced_rank(), plan.inner_len()),
        (S::Contiguous, 1, 30)
    );
    let reversed = Layout::from_strides([2, 3], [-3, -1], 5).unwrap();
    let plan = cpu::UnaryPlan::new(U::Neg, &reversed, &reversed).unwrap();
    assert_eq!(
        (plan.strategy(), plan.coalesced_rank(), plan.inner_len()),
        (S::Contiguous, 1, 6)
    );
    let scalar = Layout::contiguous([]).unwrap();
    let plan = cpu::BinaryPlan::new(B::Add, &scalar, &layout, &layout).unwrap();
    assert_eq!(
        (plan.strategy(), plan.coalesced_rank(), plan.inner_len()),
        (S::ContiguousRows, 1, 30)
    );
    let strided = Layout::from_strides([3], [2], 1).unwrap();
    assert_eq!(
        cpu::UnaryPlan::new(U::Neg, &strided, &strided)
            .unwrap()
            .strategy(),
        S::Strided
    );
    let empty = Layout::contiguous([0, usize::MAX]).unwrap();
    let plan = cpu::UnaryPlan::new(U::Neg, &empty, &empty).unwrap();
    assert_eq!(
        (plan.strategy(), plan.coalesced_rank(), plan.inner_len()),
        (S::Empty, 0, 0)
    );
    let large = Layout::contiguous([u32::MAX as usize + 17]).unwrap();
    let plan = cpu::BinaryPlan::new(B::Mul, &scalar, &large, &large).unwrap();
    assert_eq!(plan.coalesced_rank(), 1);
    assert!(plan.host_metadata_bytes() < 1024);
    let mut high = vec![1; 24];
    high[7] = 257;
    let high = Layout::contiguous(high).unwrap();
    assert_eq!(
        cpu::UnaryPlan::new(U::Square, &high, &high)
            .unwrap()
            .coalesced_rank(),
        1
    );
}

#[test]
fn cpu_pointwise_plans_can_be_shared_with_independent_outputs() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<cpu::UnaryPlan>();
    send_sync::<cpu::BinaryPlan>();
    let a = Layout::contiguous([7, 1]).unwrap();
    let b = Layout::contiguous([31]).unwrap();
    let out = Layout::contiguous([7, 31]).unwrap();
    let plan = std::sync::Arc::new(cpu::BinaryPlan::new(B::Sub, &a, &b, &out).unwrap());
    std::thread::scope(|scope| {
        for salt in [2, 5, 8] {
            let plan = plan.clone();
            let case = Case {
                a: a.clone(),
                b: b.clone(),
                out: out.clone(),
            };
            scope.spawn(move || {
                let a = input(&case.a, salt, false);
                let b = input(&case.b, salt + 1, false);
                let mut out = output(&case.out);
                plan.run(
                    &a.view(&case.a).unwrap(),
                    &b.view(&case.b).unwrap(),
                    &mut out.view_mut(&case.out).unwrap(),
                )
                .unwrap();
                check_binary(&case, B::Sub, &a, &b, &out);
            });
        }
    });
}

#[test]
fn strided_pointwise_bounds_and_read_aliases_under_miri() {
    let case = cases(0).remove(6);
    let data = Tensor::from_vec([30], (0..30).map(|x| x as f32 * 0.125 + 1.).collect()).unwrap();
    for op in BINARY {
        let plan = cpu::BinaryPlan::new(op, &case.a, &case.b, &case.out).unwrap();
        let mut out = output(&case.out);
        no_allocations(|| {
            plan.run(
                &data.view(&case.a).unwrap(),
                &data.view(&case.b).unwrap(),
                &mut out.view_mut(&case.out).unwrap(),
            )
            .unwrap()
        });
        check_binary(&case, op, &data, &data, &out);
    }
    for op in [U::Neg, U::Sigmoid] {
        let plan = cpu::UnaryPlan::new(op, &case.a, &case.out).unwrap();
        let mut out = output(&case.out);
        plan.run(
            &data.view(&case.a).unwrap(),
            &mut out.view_mut(&case.out).unwrap(),
        )
        .unwrap();
        check_unary(&case, op, &data, &out);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_pointwise_matches_independent_oracles_and_reuses_bindings() {
    use fast_audio::cuda::{BinaryPlan, Cuda, UnaryPlan};
    let cuda = Cuda::new(0).unwrap();
    for case in cases(12) {
        for op in UNARY {
            let layout = case.a.broadcast_to(case.out.shape().dims()).unwrap();
            let plan = UnaryPlan::new(&cuda, op, &layout, &case.out).unwrap();
            assert_eq!(plan.operation(), op);
            let control = cpu::UnaryPlan::new(op, &layout, &case.out).unwrap();
            assert_eq!(
                (plan.strategy(), plan.coalesced_rank(), plan.inner_len()),
                (
                    control.strategy(),
                    control.coalesced_rank(),
                    control.inner_len()
                )
            );
            assert_eq!(plan.host_metadata_bytes(), control.host_metadata_bytes());
            assert_eq!(
                plan.workspace_bytes(),
                plan.coalesced_rank().saturating_sub(1) * 32
            );
            for salt in [1, 5] {
                let source = input(
                    &layout,
                    salt,
                    matches!(op, U::Sqrt | U::Log | U::Reciprocal),
                );
                let input = cuda.upload(&source).unwrap();
                let mut out = cuda.upload(&output(&case.out)).unwrap();
                no_allocations(|| {
                    plan.run(
                        &input.view(plan.source_layout()).unwrap(),
                        &mut out.view_mut(plan.destination_layout()).unwrap(),
                    )
                    .unwrap()
                });
                let actual = cuda.download(&out).unwrap();
                check_unary(&case, op, &source, &actual);
                let first = bits(&actual);
                plan.run(
                    &input.view(&layout).unwrap(),
                    &mut out.view_mut(&case.out).unwrap(),
                )
                .unwrap();
                assert_eq!(bits(&cuda.download(&out).unwrap()), first);
                assert_eq!(bits(&cuda.download(&input).unwrap()), bits(&source));
            }
        }
        for op in BINARY {
            let plan = BinaryPlan::new(&cuda, op, &case.a, &case.b, &case.out).unwrap();
            assert_eq!(plan.operation(), op);
            assert_eq!(plan.left_layout(), &case.a);
            assert_eq!(plan.right_layout(), &case.b);
            for salt in [2, 9] {
                let a = input(&case.a, salt, false);
                let b = input(&case.b, salt + 2, true);
                let da = cuda.upload(&a).unwrap();
                let db = cuda.upload(&b).unwrap();
                let mut out = cuda.upload(&output(&case.out)).unwrap();
                no_allocations(|| {
                    plan.run(
                        &da.view(plan.left_layout()).unwrap(),
                        &db.view(plan.right_layout()).unwrap(),
                        &mut out.view_mut(plan.destination_layout()).unwrap(),
                    )
                    .unwrap()
                });
                let actual = cuda.download(&out).unwrap();
                check_binary(&case, op, &a, &b, &actual);
                let first = bits(&actual);
                plan.run(
                    &da.view(&case.a).unwrap(),
                    &db.view(&case.b).unwrap(),
                    &mut out.view_mut(&case.out).unwrap(),
                )
                .unwrap();
                assert_eq!(bits(&cuda.download(&out).unwrap()), first);
                assert_eq!(bits(&cuda.download(&da).unwrap()), bits(&a));
                assert_eq!(bits(&cuda.download(&db).unwrap()), bits(&b));
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_pointwise_stable_extremes_and_exceptional_bits_match_the_contract() {
    use fast_audio::cuda::{BinaryPlan, Cuda, UnaryPlan};
    let cuda = Cuda::new(0).unwrap();
    let values = special_values();
    let layout = Layout::contiguous([values.len()]).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let source = cuda
        .upload(&Tensor::from_vec([values.len()], values.clone()).unwrap())
        .unwrap();
    let mut out = cuda.zeros([values.len()]).unwrap();
    for op in UNARY {
        let plan = UnaryPlan::new(&cuda, op, &layout, &layout).unwrap();
        plan.run(
            &source.view(&layout).unwrap(),
            &mut out.view_mut(&layout).unwrap(),
        )
        .unwrap();
        check_special_unary(op, &values, cuda.download(&out).unwrap().data());
    }
    for op in BINARY {
        for rhs in [-0., 0., -2., 2., f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let other = cuda
                .upload(&Tensor::from_vec([], vec![rhs]).unwrap())
                .unwrap();
            let plan = BinaryPlan::new(&cuda, op, &layout, &scalar, &layout).unwrap();
            plan.run(
                &source.view(&layout).unwrap(),
                &other.view(&scalar).unwrap(),
                &mut out.view_mut(&layout).unwrap(),
            )
            .unwrap();
            for (&a, &y) in values.iter().zip(cuda.download(&out).unwrap().data()) {
                check(
                    y,
                    (binary_reference(op, a as f64, rhs as f64) as f32) as f64,
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_pointwise_checks_geometry_sessions_and_resources_before_writing() {
    use fast_audio::cuda::{BinaryPlan, Cuda, UnaryPlan};
    let cuda = Cuda::new(0).unwrap();
    let foreign = Cuda::new(0).unwrap();
    let a = Layout::contiguous([2, 1]).unwrap();
    let b = Layout::contiguous([3]).unwrap();
    let out = Layout::contiguous([2, 3]).unwrap();
    let source = Tensor::from_vec([12], vec![2.; 12]).unwrap();
    let seed = Tensor::from_vec([12], vec![9.; 12]).unwrap();
    let input = cuda.upload(&source).unwrap();
    let bad_input = foreign.upload(&source).unwrap();
    let mut dest = cuda.upload(&seed).unwrap();
    let mut bad_dest = foreign.upload(&seed).unwrap();
    let plan = BinaryPlan::new(&cuda, B::Add, &a, &b, &out).unwrap();
    assert!(
        plan.run(
            &bad_input.view(&a).unwrap(),
            &input.view(&b).unwrap(),
            &mut dest.view_mut(&out).unwrap()
        )
        .is_err()
    );
    assert!(
        plan.run(
            &input.view(&a).unwrap(),
            &bad_input.view(&b).unwrap(),
            &mut dest.view_mut(&out).unwrap()
        )
        .is_err()
    );
    assert!(
        plan.run(
            &input.view(&a).unwrap(),
            &input.view(&b).unwrap(),
            &mut bad_dest.view_mut(&out).unwrap()
        )
        .is_err()
    );
    let wrong = Layout::contiguous([3, 2]).unwrap();
    assert!(
        plan.run(
            &input.view(&a).unwrap(),
            &input.view(&b).unwrap(),
            &mut dest.view_mut(&wrong).unwrap()
        )
        .is_err()
    );
    let changed_a = Layout::from_strides([2, 1], [2, 0], 1).unwrap();
    let changed_b = Layout::from_strides([3], [2], 0).unwrap();
    assert!(
        plan.run(
            &input.view(&changed_a).unwrap(),
            &input.view(&b).unwrap(),
            &mut dest.view_mut(&out).unwrap()
        )
        .is_err()
    );
    assert!(
        plan.run(
            &input.view(&a).unwrap(),
            &input.view(&changed_b).unwrap(),
            &mut dest.view_mut(&out).unwrap()
        )
        .is_err()
    );
    let unary = UnaryPlan::new(&cuda, U::Neg, &out, &out).unwrap();
    assert!(
        unary
            .run(
                &bad_input.view(&out).unwrap(),
                &mut dest.view_mut(&out).unwrap()
            )
            .is_err()
    );
    assert!(
        unary
            .run(
                &input.view(&out).unwrap(),
                &mut bad_dest.view_mut(&out).unwrap()
            )
            .is_err()
    );
    assert!(
        unary
            .run(
                &input.view(&wrong).unwrap(),
                &mut dest.view_mut(&out).unwrap()
            )
            .is_err()
    );
    assert!(
        unary
            .run(
                &input.view(&out).unwrap(),
                &mut dest.view_mut(&wrong).unwrap()
            )
            .is_err()
    );
    assert_eq!(cuda.download(&dest).unwrap().data(), seed.data());
    assert_eq!(foreign.download(&bad_dest).unwrap().data(), seed.data());
    let empty = Layout::contiguous([0, 3]).unwrap();
    let scalar = Layout::contiguous([]).unwrap();
    let plan = BinaryPlan::new(&cuda, B::Add, &scalar, &empty, &empty).unwrap();
    assert_eq!(plan.workspace_bytes(), 0);
    assert!(
        plan.run(
            &bad_input.view(&scalar).unwrap(),
            &input.view(&empty).unwrap(),
            &mut dest.view_mut(&empty).unwrap()
        )
        .is_err()
    );
    assert!(
        plan.run(
            &input.view(&scalar).unwrap(),
            &bad_input.view(&empty).unwrap(),
            &mut dest.view_mut(&empty).unwrap()
        )
        .is_err()
    );
    assert!(
        plan.run(
            &input.view(&scalar).unwrap(),
            &input.view(&empty).unwrap(),
            &mut bad_dest.view_mut(&empty).unwrap()
        )
        .is_err()
    );
    plan.run(
        &input.view(&scalar).unwrap(),
        &input.view(&empty).unwrap(),
        &mut dest.view_mut(&empty).unwrap(),
    )
    .unwrap();
    assert_eq!(cuda.download(&dest).unwrap().data(), seed.data());
    let large = Layout::contiguous([u32::MAX as usize + 17]).unwrap();
    let plan = BinaryPlan::new(&cuda, B::Mul, &scalar, &large, &large).unwrap();
    assert_eq!(plan.workspace_bytes(), 0);
    assert!(plan.host_metadata_bytes() < 1024);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_pointwise_grid_stride_covers_the_launch_cap() {
    use fast_audio::cuda::{Cuda, UnaryPlan};
    let cuda = Cuda::new(0).unwrap();
    let count = 65535 * 256 + 17;
    let layout = Layout::contiguous([count]).unwrap();
    let input = cuda
        .upload(&Tensor::from_vec([count], vec![1.; count]).unwrap())
        .unwrap();
    let mut out = cuda.zeros([count]).unwrap();
    let plan = UnaryPlan::new(&cuda, U::Neg, &layout, &layout).unwrap();
    plan.run(
        &input.view(&layout).unwrap(),
        &mut out.view_mut(&layout).unwrap(),
    )
    .unwrap();
    assert!(
        cuda.download(&out)
            .unwrap()
            .data()
            .iter()
            .all(|x| x.to_bits() == (-1_f32).to_bits())
    );
}
