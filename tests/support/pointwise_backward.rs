use fast_audio::{
    BinaryOp as B, BinaryOperand as Side, GradientMode as G, Layout, Tensor, TensorView,
    TensorViewMut, UnaryOp as U, cpu,
};
use std::collections::BTreeSet;

pub const UNARY: [U; 12] = [
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
pub const BINARY: [B; 4] = [B::Add, B::Sub, B::Mul, B::Div];
pub const SIDES: [Side; 2] = [Side::Left, Side::Right];
pub const SENTINEL: u32 = 0x7fc12345;
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Unary(U),
    Binary(B, Side),
}
#[derive(Clone)]
pub struct Case {
    pub op: Op,
    pub left: Layout,
    pub right: Option<Layout>,
    pub upstream: Layout,
    pub gradient: Layout,
}
impl Case {
    pub fn dependencies(&self) -> [bool; 2] {
        if self.upstream.shape().is_empty() || self.gradient.shape().is_empty() {
            return [false; 2];
        }
        match self.op {
            Op::Unary(U::Neg) | Op::Binary(B::Add | B::Sub, _) => [false; 2],
            Op::Unary(_) => [true, false],
            Op::Binary(B::Mul, Side::Left) | Op::Binary(B::Div, Side::Left) => [false, true],
            Op::Binary(B::Mul, Side::Right) => [true, false],
            Op::Binary(B::Div, Side::Right) => [true, true],
        }
    }
}
pub fn padded(dims: &[usize], seed: usize) -> Layout {
    if dims.contains(&0) {
        return Layout::from_strides(dims, vec![isize::MIN; dims.len()], 3).unwrap();
    }
    let mut strides = vec![isize::MIN; dims.len()];
    let (mut step, mut origin) = (2usize, 3usize);
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
pub fn coordinates(dims: &[usize], mut i: usize) -> Vec<usize> {
    let mut c = vec![0; dims.len()];
    for a in (0..dims.len()).rev() {
        c[a] = i % dims[a];
        i /= dims[a];
    }
    c
}
pub fn address(layout: &Layout, c: &[usize]) -> usize {
    (layout.offset() as i128
        + c.iter()
            .zip(layout.strides())
            .map(|(&i, &s)| i as i128 * s as i128)
            .sum::<i128>()) as usize
}
pub fn addresses(layout: &Layout) -> Vec<usize> {
    (0..layout.shape().len())
        .map(|i| address(layout, &coordinates(layout.shape().dims(), i)))
        .collect()
}
pub fn bits(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|x| x.to_bits()).collect()
}
pub fn buffer(layout: &Layout, salt: usize, positive: bool) -> Tensor {
    let len = layout.span().map_or(layout.offset(), |s| s.end) + 2;
    Tensor::from_vec(
        [len],
        (0..len)
            .map(|i| {
                let x = 0.25 + ((i * 31 + salt * 17) % 101) as f32 / 67.;
                if positive || (i + salt) % 2 == 0 {
                    x
                } else {
                    -x
                }
            })
            .collect(),
    )
    .unwrap()
}
pub fn seed(layout: &Layout) -> Tensor {
    let mut t = buffer(layout, 0, true);
    t.data_mut().fill(f32::from_bits(SENTINEL));
    for (i, a) in addresses(layout).into_iter().enumerate() {
        t.data_mut()[a] = 0.2 + (i % 7) as f32 / 13.;
    }
    t
}
pub fn payloads(c: &Case, salt: usize) -> (Tensor, Option<Tensor>, Tensor) {
    let positive = matches!(c.op, Op::Unary(U::Reciprocal | U::Sqrt | U::Log));
    (
        buffer(&c.left, salt, positive),
        c.right.as_ref().map(|r| buffer(r, salt + 1, true)),
        buffer(&c.upstream, salt + 2, false),
    )
}
pub fn cases() -> Vec<Case> {
    let mut unary = vec![
        Layout::from_strides([], [], 2).unwrap(),
        Layout::contiguous([17]).unwrap(),
        Layout::from_strides([2, 3], [-7, 2], 7).unwrap(),
        Layout::contiguous([2, 3, 5])
            .unwrap()
            .permute(&[2, 0, 1])
            .unwrap(),
        Layout::from_strides([3, 2, 5], [-2, 19, -4], 20).unwrap(),
        Layout::from_strides([3, 4, 7], [1, 1, 2], 1).unwrap(),
        Layout::from_strides([4, 3, 33], [0, 1, 0], 2).unwrap(),
        Layout::contiguous([2, 3, 2, 2, 3])
            .unwrap()
            .permute(&[4, 2, 0, 3, 1])
            .unwrap(),
        Layout::from_strides([257], [-2], 512).unwrap(),
        Layout::from_strides([2, 0, 3], [isize::MIN, isize::MAX, isize::MIN], 3).unwrap(),
        Layout::contiguous([usize::MAX, usize::MAX, 0]).unwrap(),
        Layout::from_strides(vec![1; 128], vec![isize::MIN; 128], 2).unwrap(),
    ];
    for i in 0usize..12 {
        let p = padded(&[2 + i % 3, 3, 2 + i % 4], i);
        unary.push(if i % 2 == 0 {
            p.permute(&[2, 0, 1]).unwrap()
        } else {
            p
        });
    }
    let mut binary = vec![
        (
            Layout::from_strides([], [], 2).unwrap(),
            Layout::contiguous([]).unwrap(),
            vec![],
        ),
        (padded(&[2, 3], 0), padded(&[2, 3], 1), vec![2, 3]),
        (padded(&[3], 0), padded(&[2, 3], 1), vec![2, 3]),
        (padded(&[2, 1], 0), padded(&[1, 3], 1), vec![2, 3]),
        (padded(&[2, 1, 3], 0), padded(&[1, 4, 1], 1), vec![2, 4, 3]),
        (padded(&[], 0), padded(&[2, 3, 5], 1), vec![2, 3, 5]),
        (padded(&[1, 3, 1], 0), padded(&[2, 1, 5], 1), vec![2, 3, 5]),
        (
            Layout::from_strides([3, 4], [1, 1], 0).unwrap(),
            padded(&[1, 4], 1),
            vec![3, 4],
        ),
        (padded(&[1, 3], 0), padded(&[0, 3], 1), vec![0, 3]),
        (padded(&[0, 3], 0), padded(&[1, 3], 1), vec![0, 3]),
        (padded(&[], 0), padded(&[0, 3], 1), vec![0, 3]),
        (
            Layout::contiguous([usize::MAX, usize::MAX, 0]).unwrap(),
            padded(&[], 0),
            vec![usize::MAX, usize::MAX, 0],
        ),
        (
            padded(&[], 0),
            Layout::from_strides(vec![1; 128], vec![isize::MIN; 128], 2).unwrap(),
            vec![1; 128],
        ),
        (
            Layout::from_strides([2, 17, 37], [1100, -50, 2], 805).unwrap(),
            padded(&[1, 17, 1], 1),
            vec![2, 17, 37],
        ),
    ];
    for i in 0usize..12 {
        let dims = [2 + i % 3, 3, 2 + i % 4];
        let left = if i % 2 == 0 {
            vec![dims[0], 1, dims[2]]
        } else {
            vec![dims[2]]
        };
        let right = if i % 3 == 0 {
            vec![1, dims[1], 1]
        } else {
            dims.to_vec()
        };
        let mut common = dims.to_vec();
        if left.len() == 1 && right[0] == 1 {
            common[0] = 1;
        }
        binary.push((padded(&left, i), padded(&right, i + 1), common));
    }
    assert_eq!(unary.len(), 24);
    assert_eq!(binary.len(), 26);
    let mut all = Vec::new();
    for (i, left) in unary.into_iter().enumerate() {
        for op in UNARY {
            all.push(Case {
                op: Op::Unary(op),
                upstream: padded(left.shape().dims(), i + 1),
                gradient: padded(left.shape().dims(), i + 2),
                left: left.clone(),
                right: None,
            });
        }
    }
    for (i, (left, right, common)) in binary.into_iter().enumerate() {
        for op in BINARY {
            for side in SIDES {
                let target = if side == Side::Left { &left } else { &right };
                all.push(Case {
                    op: Op::Binary(op, side),
                    gradient: padded(target.shape().dims(), i + 2),
                    upstream: padded(&common, i + 1),
                    left: left.clone(),
                    right: Some(right.clone()),
                });
            }
        }
    }
    assert_eq!(all.len(), 496);
    // Keep upstream aliasing and output-axis permutation coupled: neither
    // property should be inferred from testing only overlapping forward inputs.
    for op in UNARY {
        all.push(Case {
            op: Op::Unary(op),
            left: Layout::from_strides([2, 3, 5], [1, 14, -2], 8).unwrap(),
            right: None,
            upstream: Layout::from_strides([2, 3, 5], [0, -1, 1], 2).unwrap(),
            gradient: Layout::from_strides([2, 3, 5], [1, -13, 2], 26).unwrap(),
        });
    }
    for count in [3usize, 33] {
        for op in BINARY {
            for side in SIDES {
                let gradient = if side == Side::Left {
                    Layout::from_strides([3, 1, 5], [1, isize::MIN, -5], 20).unwrap()
                } else {
                    Layout::from_strides(
                        [1, count, 1],
                        [isize::MIN, -2, isize::MIN],
                        2 * (count - 1) + 1,
                    )
                    .unwrap()
                };
                all.push(Case {
                    op: Op::Binary(op, side),
                    left: Layout::from_strides([3, 1, 5], [1, 0, -4], 16).unwrap(),
                    right: Some(
                        Layout::from_strides([1, count, 1], [0, -1, 0], count - 1).unwrap(),
                    ),
                    upstream: Layout::from_strides([3, count, 5], [0, 1, -1], 4).unwrap(),
                    gradient,
                });
            }
        }
    }
    assert_eq!(all.len(), 524);
    all
}
pub enum CpuPlan {
    Unary(cpu::UnaryBackward),
    Binary(cpu::BinaryBackward),
}
impl CpuPlan {
    pub fn new(c: &Case) -> Self {
        match c.op {
            Op::Unary(op) => {
                let p = cpu::UnaryBackward::new(op, &c.left, &c.upstream, &c.gradient).unwrap();
                assert_eq!(p.operation(), op);
                assert_eq!(p.input_layout(), &c.left);
                assert_eq!(p.upstream_layout(), &c.upstream);
                assert_eq!(p.destination_layout(), &c.gradient);
                assert_eq!([p.input_required(), false], c.dependencies());
                assert_eq!(p.workspace_bytes(), 0);
                assert!(p.host_metadata_bytes() < 200_000);
                Self::Unary(p)
            }
            Op::Binary(op, side) => {
                let p = cpu::BinaryBackward::new(
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
                assert_eq!(p.workspace_bytes(), 0);
                assert!(p.host_metadata_bytes() < 200_000);
                Self::Binary(p)
            }
        }
    }
    pub fn run(
        &self,
        inputs: [Option<&TensorView<'_>>; 2],
        dy: &TensorView<'_>,
        dx: &mut TensorViewMut<'_>,
        mode: G,
    ) -> fast_audio::Result<()> {
        match self {
            Self::Unary(p) => p.run(inputs[0], dy, dx, mode),
            Self::Binary(p) => p.run(inputs[0], inputs[1], dy, dx, mode),
        }
    }
}
pub fn logical(layout: &Layout, t: &Tensor) -> Vec<f64> {
    addresses(layout)
        .into_iter()
        .map(|a| t.data()[a] as f64)
        .collect()
}
fn input_index(dims: &[usize], c: &[usize]) -> usize {
    dims.iter()
        .zip(&c[c.len() - dims.len()..])
        .fold(0, |i, (&d, &v)| i * d + if d == 1 { 0 } else { v })
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn forward(op: Op, x: f64, y: f64) -> f64 {
    match op {
        Op::Unary(u) => match u {
            U::Neg => -x,
            U::Abs => x.abs(),
            U::Square => x * x,
            U::Reciprocal => 1. / x,
            U::Sqrt => x.sqrt(),
            U::Exp => x.exp(),
            U::Log => x.ln(),
            U::Relu => x.max(0.),
            U::Sigmoid => sigmoid(x),
            U::Tanh => x.tanh(),
            U::Silu => x * sigmoid(x),
            U::Softplus => x.exp().ln_1p(),
        },
        Op::Binary(b, _) => match b {
            B::Add => x + y,
            B::Sub => x - y,
            B::Mul => x * y,
            B::Div => x / y,
        },
    }
}
fn derivative(op: Op, x: f64, y: f64) -> f64 {
    match op {
        Op::Unary(u) => match u {
            U::Neg => -1.,
            U::Abs => x.signum(),
            U::Square => 2. * x,
            U::Reciprocal => -1. / (x * x),
            U::Sqrt => 0.5 / x.sqrt(),
            U::Exp => x.exp(),
            U::Log => 1. / x,
            U::Relu => {
                if x > 0. {
                    1.
                } else {
                    0.
                }
            }
            U::Sigmoid => sigmoid(x) * (1. - sigmoid(x)),
            U::Tanh => 1. - x.tanh().powi(2),
            U::Silu => sigmoid(x) + x * sigmoid(x) * (1. - sigmoid(x)),
            U::Softplus => sigmoid(x),
        },
        Op::Binary(b, side) => match (b, side) {
            (B::Add, _) | (B::Sub, Side::Left) => 1.,
            (B::Sub, _) => -1.,
            (B::Mul, Side::Left) => y,
            (B::Mul, _) => x,
            (B::Div, Side::Left) => 1. / y,
            (B::Div, _) => -x / (y * y),
        },
    }
}
pub fn reference(c: &Case, x: &Tensor, y: Option<&Tensor>, dy: &Tensor) -> Vec<f64> {
    let x = logical(&c.left, x);
    let y = c.right.as_ref().zip(y).map(|(r, y)| logical(r, y));
    let d = logical(&c.upstream, dy);
    let mut out = vec![0.; c.gradient.shape().len()];
    for (i, &d) in d.iter().enumerate() {
        let coord = coordinates(c.upstream.shape().dims(), i);
        let a = input_index(c.left.shape().dims(), &coord);
        let b = c
            .right
            .as_ref()
            .map_or(0, |r| input_index(r.shape().dims(), &coord));
        let target = match c.op {
            Op::Binary(_, Side::Right) => b,
            _ => a,
        };
        out[target] += d * derivative(c.op, x[a], y.as_ref().map_or(0., |y| y[b]));
    }
    out
}
pub fn close(actual: f32, expected: f64) {
    assert!(
        (actual as f64 - expected).abs() <= 2e-5 + 2e-5 * expected.abs(),
        "{actual} != {expected}"
    );
}
pub fn holes(layout: &Layout, seed: &Tensor, out: &Tensor) {
    let selected: BTreeSet<_> = addresses(layout).into_iter().collect();
    for (i, (&a, &b)) in seed.data().iter().zip(out.data()).enumerate() {
        if !selected.contains(&i) {
            assert_eq!(a.to_bits(), b.to_bits(), "hole {i}");
        }
    }
}
pub fn verify(c: &Case, expected: &[f64], before: &Tensor, actual: &Tensor, mode: G) {
    assert_eq!(expected.len(), c.gradient.shape().len());
    for (&a, &v) in addresses(&c.gradient).iter().zip(expected) {
        close(
            actual.data()[a],
            v + if mode == G::Accumulate {
                before.data()[a] as f64
            } else {
                0.
            },
        );
        if c.upstream.shape().is_empty() && mode == G::Overwrite {
            assert_eq!(actual.data()[a].to_bits(), 0);
        }
    }
    holes(&c.gradient, before, actual);
}
pub fn finite_difference(c: &Case, x: &Tensor, y: Option<&Tensor>, dy: &Tensor, actual: &Tensor) {
    if c.gradient.shape().is_empty() {
        return;
    }
    let x = logical(&c.left, x);
    let y = c.right.as_ref().zip(y).map(|(r, y)| logical(r, y));
    let d = logical(&c.upstream, dy);
    let objective = |selected: &[f64]| {
        d.iter()
            .enumerate()
            .map(|(i, &d)| {
                let coord = coordinates(c.upstream.shape().dims(), i);
                let a = input_index(c.left.shape().dims(), &coord);
                let b = c
                    .right
                    .as_ref()
                    .map_or(0, |r| input_index(r.shape().dims(), &coord));
                let (xx, yy) = match c.op {
                    Op::Binary(_, Side::Right) => (x[a], selected[b]),
                    _ => (selected[a], y.as_ref().map_or(0., |y| y[b])),
                };
                d * forward(c.op, xx, yy)
            })
            .sum::<f64>()
    };
    let original = match c.op {
        Op::Binary(_, Side::Right) => y.as_ref().unwrap(),
        _ => &x,
    };
    let indices = addresses(&c.gradient);
    for i in [0, original.len() / 2, original.len() - 1] {
        let mut value = original.clone();
        value[i] += 1e-4;
        let plus = objective(&value);
        value[i] -= 2e-4;
        let numerical = (plus - objective(&value)) / 2e-4;
        close(actual.data()[indices[i]], numerical);
    }
}

pub fn special_unary(mut run: impl FnMut(U, &[f32], &[f32]) -> Vec<f32>) {
    let nan = f32::from_bits(0x7fc54321);
    let d = [-0., nan, f32::INFINITY, f32::NEG_INFINITY];
    assert_eq!(
        run(U::Neg, &[1.; 4], &d)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        d.iter()
            .map(|x| x.to_bits() ^ 0x80000000)
            .collect::<Vec<_>>()
    );
    let x = [-2., 2., 0., -0., nan, f32::NEG_INFINITY, f32::INFINITY];
    let d = [3., -4., nan, f32::INFINITY, 2., 5., -6.];
    for (op, expected) in [
        (U::Abs, [-3., -4., 0., 0., f32::NAN, -5., -6.]),
        (U::Relu, [0., -4., 0., 0., f32::NAN, 0., -6.]),
    ] {
        for (&got, want) in run(op, &x, &d).iter().zip(expected) {
            assert_eq!(
                got.to_bits(),
                if want.is_nan() {
                    0x7fc00000
                } else {
                    want.to_bits()
                }
            );
        }
    }
    for (op, x, d, expected) in [
        (
            U::Reciprocal,
            vec![0., -0., f32::INFINITY, f32::NEG_INFINITY, -2.],
            vec![1., 1., 1., 1., 8.],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, -0., -0., -2.],
        ),
        (
            U::Sqrt,
            vec![0., -0., -1., f32::INFINITY],
            vec![1.; 4],
            vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.],
        ),
        (
            U::Log,
            vec![-2., 0., -0., f32::INFINITY],
            vec![4., 1., 1., 1.],
            vec![-2., f32::INFINITY, f32::NEG_INFINITY, 0.],
        ),
        (
            U::Exp,
            vec![f32::INFINITY, f32::NEG_INFINITY],
            vec![0., f32::INFINITY],
            vec![f32::NAN; 2],
        ),
        (
            U::Silu,
            vec![
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            vec![2., 2., -2., f32::INFINITY],
            vec![2., -0., 0., f32::NAN],
        ),
        (
            U::Softplus,
            vec![f32::INFINITY, f32::NEG_INFINITY],
            vec![2., 2.],
            vec![2., 0.],
        ),
    ] {
        for (&got, want) in run(op, &x, &d).iter().zip(expected) {
            if want.is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(got.to_bits(), want.to_bits());
            }
        }
    }
    for (op, points) in [
        (U::Sigmoid, vec![-90., -80., 20., 80., 90.]),
        (U::Tanh, vec![-50., -20., 20., 50.]),
    ] {
        let actual = run(op, &points, &vec![1.; points.len()]);
        for (&got, &x) in actual.iter().zip(&points) {
            let q = (-(x as f64).abs()).exp();
            let expected = if op == U::Sigmoid {
                q / (1. + q).powi(2)
            } else {
                4. * q * q / (1. + q * q).powi(2)
            } as f32;
            assert!(got > 0., "lost {op:?} derivative at {x}");
            assert!(
                got.to_bits().abs_diff(expected.to_bits()) <= 16,
                "tail {op:?}({x}): {got} != {expected}"
            );
        }
    }
}
pub type Cancellation = (usize, Vec<(usize, f32)>, f32);
pub fn cancellation() -> Vec<Cancellation> {
    vec![
        (32, vec![(0, 1e20), (16, -1e20), (1, 3.)], 3.),
        (33, vec![(0, 1e20), (32, -1e20), (16, 1.)], 1.),
        (257, vec![(0, 1e20), (256, -1e20), (128, 1.), (1, 3.)], 4.),
    ]
}
pub fn special_binary(mut run: impl FnMut(B, Side, &[f32], &[f32], &[f32]) -> Vec<f32>) {
    for (op, side, a, b, d, expected) in [
        (
            B::Mul,
            Side::Left,
            vec![1.; 4],
            vec![0., -0., f32::INFINITY, f32::NEG_INFINITY],
            vec![f32::INFINITY, 2., 0., -2.],
            vec![f32::NAN, -0., f32::NAN, f32::INFINITY],
        ),
        (
            B::Mul,
            Side::Right,
            vec![0., -0., f32::INFINITY, f32::NEG_INFINITY],
            vec![1.; 4],
            vec![f32::INFINITY, 2., 0., -2.],
            vec![f32::NAN, -0., f32::NAN, f32::INFINITY],
        ),
        (
            B::Div,
            Side::Left,
            vec![1.; 4],
            vec![0., -0., f32::INFINITY, -2.],
            vec![1., 1., 1., 4.],
            vec![f32::INFINITY, f32::NEG_INFINITY, 0., -2.],
        ),
        (
            B::Div,
            Side::Right,
            vec![1.; 4],
            vec![0., -0., f32::INFINITY, -2.],
            vec![1., 1., 1., 4.],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, -0., -1.],
        ),
        (
            B::Div,
            Side::Right,
            vec![1e20, 1e-20],
            vec![1e20, 1e-20],
            vec![1e20, 1e-20],
            vec![-1.; 2],
        ),
    ] {
        for (&got, want) in run(op, side, &a, &b, &d).iter().zip(expected) {
            if want.is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(got.to_bits(), want.to_bits());
            }
        }
    }
}
