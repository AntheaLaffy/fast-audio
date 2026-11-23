use fast_audio::{
    BinaryOperand as Side, Conv1dSpec, Conv2dSpec, GradientMode as G, Layout, Tensor,
};
use std::collections::BTreeSet;

pub const SIDES: [Side; 2] = [Side::Left, Side::Right];
pub const SENTINEL: u32 = 0x7fc12345;

#[derive(Clone)]
pub struct Case1d {
    pub spec: Conv1dSpec,
    pub x: Layout,
    pub w: Layout,
    pub y: Layout,
    pub dy: Layout,
    pub dx: Layout,
    pub dw: Layout,
    pub direct: bool,
}
impl Case1d {
    pub fn gradient(&self, side: Side) -> &Layout {
        if side == Side::Left {
            &self.dx
        } else {
            &self.dw
        }
    }
    pub fn dependencies(&self, side: Side) -> [bool; 2] {
        [side == Side::Right, side == Side::Left]
    }
}

#[derive(Clone)]
pub struct Case2d {
    pub spec: Conv2dSpec,
    pub x: Layout,
    pub w: Layout,
    pub y: Layout,
    pub dy: Layout,
    pub dx: Layout,
    pub dw: Layout,
    pub direct: bool,
}
impl Case2d {
    pub fn gradient(&self, side: Side) -> &Layout {
        if side == Side::Left {
            &self.dx
        } else {
            &self.dw
        }
    }
    pub fn dependencies(&self, side: Side) -> [bool; 2] {
        [side == Side::Right, side == Side::Left]
    }
}

pub fn layout3(dims: [usize; 3], style: usize) -> Layout {
    let [d0, d1, d2] = dims;
    let s1 = (d2 + 1) as isize;
    let s0 = (d1 + 1) as isize * s1;
    let (mut steps, mut origin) = match style % 8 {
        0 => ([(d1 * d2) as isize, d2 as isize, 1], 2),
        1 => ([s0, s1, 1], 3),
        2 => ([d2 as isize, (d0 * d2 + 2) as isize, 1], 1),
        3 => (
            [-s0, s1, 1],
            2 + (d0.saturating_sub(1)) * (s0.unsigned_abs()),
        ),
        4 => ([s0 * 2, s1 * 2, -2], 3 + (d2.saturating_sub(1)) * 2),
        5 => (
            [
                -2,
                (d0 * 2 + 3) as isize,
                ((d0 * 2 + 3) * (d1 + 1)) as isize,
            ],
            1 + (d0.saturating_sub(1)) * 2,
        ),
        6 => ([0, d2 as isize, 1], 2),
        _ => ([1, 1, 1], 1),
    };
    for axis in 0..3 {
        if dims[axis] == 1 {
            steps[axis] = if style.is_multiple_of(2) {
                0
            } else {
                steps[axis]
            };
        }
    }
    if origin == 0 {
        origin = 1;
    }
    Layout::from_strides(dims, steps, origin).unwrap()
}

pub fn layout4(dims: [usize; 4], style: usize) -> Layout {
    let [d0, d1, d2, d3] = dims;
    let s2 = (d3 + 1) as isize;
    let s1 = (d2 + 1) as isize * s2;
    let s0 = (d1 + 1) as isize * s1;
    let (mut steps, mut origin) = match style % 8 {
        0 => (
            [(d1 * d2 * d3) as isize, (d2 * d3) as isize, d3 as isize, 1],
            2,
        ),
        1 => ([s0, s1, s2, 1], 3),
        2 => (
            [
                (d2 * d3) as isize,
                (d0 * d2 * d3 + 2) as isize,
                d3 as isize,
                1,
            ],
            1,
        ),
        3 => (
            [-s0, s1, s2, 1],
            2 + (d0.saturating_sub(1)) * (s0.unsigned_abs()),
        ),
        4 => ([s0 * 2, s1 * 2, s2 * 2, -2], 3 + (d3.saturating_sub(1)) * 2),
        5 => (
            [
                -2,
                (d0 * 2 + 1) as isize,
                ((d0 * 2 + 1) * (d1 + 1)) as isize,
                ((d0 * 2 + 1) * (d1 + 1) * (d2 + 1)) as isize,
            ],
            1 + (d0.saturating_sub(1)) * 2,
        ),
        6 => ([0, (d2 * d3) as isize, d3 as isize, 1], 2),
        _ => ([1, 1, 1, 1], 1),
    };
    for axis in 0..4 {
        if dims[axis] == 1 {
            steps[axis] = if style.is_multiple_of(2) {
                0
            } else {
                steps[axis]
            };
        }
    }
    if origin == 0 {
        origin = 1;
    }
    Layout::from_strides(dims, steps, origin).unwrap()
}

pub const SPECS_1D: [Conv1dSpec; 7] = [
    Conv1dSpec {
        batch: 1,
        in_channels: 1,
        out_channels: 1,
        input_len: 1,
        kernel: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    },
    Conv1dSpec {
        batch: 2,
        in_channels: 3,
        out_channels: 4,
        input_len: 5,
        kernel: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    },
    Conv1dSpec {
        batch: 2,
        in_channels: 3,
        out_channels: 4,
        input_len: 7,
        kernel: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    },
    Conv1dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 6,
        input_len: 9,
        kernel: 3,
        stride: 2,
        padding: 1,
        dilation: 2,
        groups: 2,
    },
    Conv1dSpec {
        batch: 3,
        in_channels: 3,
        out_channels: 3,
        input_len: 11,
        kernel: 5,
        stride: 2,
        padding: 2,
        dilation: 1,
        groups: 3,
    },
    Conv1dSpec {
        batch: 1,
        in_channels: 2,
        out_channels: 17,
        input_len: 16,
        kernel: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    },
    Conv1dSpec {
        batch: 1,
        in_channels: 5,
        out_channels: 3,
        input_len: 33,
        kernel: 7,
        stride: 1,
        padding: 3,
        dilation: 1,
        groups: 1,
    },
];

pub const SPECS_2D: [Conv2dSpec; 6] = [
    Conv2dSpec {
        batch: 1,
        in_channels: 1,
        out_channels: 1,
        input: [1, 1],
        kernel: [1, 1],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
        transposed: false,
        output_padding: [0, 0],
    },
    Conv2dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 6,
        input: [3, 4],
        kernel: [1, 1],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 2,
        transposed: false,
        output_padding: [0, 0],
    },
    Conv2dSpec {
        batch: 2,
        in_channels: 3,
        out_channels: 4,
        input: [4, 5],
        kernel: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 1,
        transposed: false,
        output_padding: [0, 0],
    },
    Conv2dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 6,
        input: [5, 6],
        kernel: [2, 3],
        stride: [2, 1],
        padding: [1, 0],
        dilation: [1, 2],
        groups: 2,
        transposed: false,
        output_padding: [0, 0],
    },
    Conv2dSpec {
        batch: 1,
        in_channels: 3,
        out_channels: 3,
        input: [4, 4],
        kernel: [3, 3],
        stride: [2, 2],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 3,
        transposed: false,
        output_padding: [0, 0],
    },
    Conv2dSpec {
        batch: 1,
        in_channels: 2,
        out_channels: 17,
        input: [4, 4],
        kernel: [3, 2],
        stride: [1, 1],
        padding: [1, 0],
        dilation: [1, 1],
        groups: 1,
        transposed: false,
        output_padding: [0, 0],
    },
];

pub fn cases_1d() -> Vec<Case1d> {
    let mut out = Vec::new();
    for spec in SPECS_1D {
        let x_shape = spec.input_shape();
        let w_shape = spec.weight_shape();
        let y_shape = spec.output_shape().unwrap();
        for s in 0..8 {
            out.push(Case1d {
                spec,
                x: layout3(x_shape, s),
                w: layout3(w_shape, s * 3 + 1),
                y: layout3(y_shape, (s * 5 + 1) % 6),
                dy: layout3(y_shape, s * 5 + 2),
                dx: layout3(x_shape, (s * 3 + 2) % 6),
                dw: layout3(w_shape, (s * 7 + 3) % 6),
                direct: false,
            });
        }
    }
    for spec in [SPECS_1D[0], SPECS_1D[1]] {
        let x_shape = spec.input_shape();
        let w_shape = spec.weight_shape();
        let y_shape = spec.output_shape().unwrap();
        for flags in 0..4 {
            out.push(Case1d {
                spec,
                x: layout3(x_shape, if flags & 1 == 0 { 0 } else { 2 }),
                w: layout3(w_shape, 0),
                y: layout3(y_shape, if flags & 2 == 0 { 0 } else { 2 }),
                dy: layout3(y_shape, if flags & 1 == 0 { 0 } else { 2 }),
                dx: layout3(x_shape, if flags & 2 == 0 { 0 } else { 2 }),
                dw: layout3(w_shape, 0),
                direct: true,
            });
        }
    }
    assert_eq!(out.len(), 64);
    out
}

pub fn cases_2d() -> Vec<Case2d> {
    let mut out = Vec::new();
    for spec in SPECS_2D {
        let x_shape = spec.input_shape();
        let w_shape = spec.weight_shape();
        let y_shape = spec.output_shape().unwrap();
        for s in 0..8 {
            out.push(Case2d {
                spec,
                x: layout4(x_shape, s),
                w: layout4(w_shape, s * 3 + 1),
                y: layout4(y_shape, (s * 5 + 1) % 6),
                dy: layout4(y_shape, s * 5 + 2),
                dx: layout4(x_shape, (s * 3 + 2) % 6),
                dw: layout4(w_shape, (s * 7 + 3) % 6),
                direct: false,
            });
        }
    }
    for spec in [SPECS_2D[0], SPECS_2D[1]] {
        let x_shape = spec.input_shape();
        let w_shape = spec.weight_shape();
        let y_shape = spec.output_shape().unwrap();
        for flags in 0..4 {
            out.push(Case2d {
                spec,
                x: layout4(x_shape, if flags & 1 == 0 { 0 } else { 2 }),
                w: layout4(w_shape, 0),
                y: layout4(y_shape, if flags & 2 == 0 { 0 } else { 2 }),
                dy: layout4(y_shape, if flags & 1 == 0 { 0 } else { 2 }),
                dx: layout4(x_shape, if flags & 2 == 0 { 0 } else { 2 }),
                dw: layout4(w_shape, 0),
                direct: true,
            });
        }
    }
    assert_eq!(out.len(), 56);
    out
}

pub fn addresses(l: &Layout) -> Vec<usize> {
    let dims = l.shape().dims();
    let strides = l.strides();
    let total = l.shape().len();
    let mut out = Vec::with_capacity(total);
    for mut linear in 0..total {
        let mut offset = l.offset() as i128;
        for axis in (0..dims.len()).rev() {
            let coord = linear % dims[axis];
            linear /= dims[axis];
            offset += coord as i128 * strides[axis] as i128;
        }
        out.push(offset as usize);
    }
    out
}

pub fn buffer(l: &Layout, salt: usize) -> Tensor {
    let len = l.span().map_or(l.offset(), |s| s.end) + 2;
    Tensor::from_vec(
        [len],
        (0..len)
            .map(|i| (((i * 13 + salt * 17) % 97) as f32 - 48.) / 97.)
            .collect(),
    )
    .unwrap()
}

pub fn seed(l: &Layout, finite: bool) -> Tensor {
    let mut t = buffer(l, 0);
    t.data_mut().fill(f32::from_bits(SENTINEL));
    if finite {
        for (i, a) in addresses(l).into_iter().enumerate() {
            t.data_mut()[a] = 0.2 + (i % 7) as f32 / 13.;
        }
    }
    t
}

pub fn bits(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|x| x.to_bits()).collect()
}

pub fn logical(l: &Layout, t: &Tensor) -> Vec<f64> {
    addresses(l)
        .into_iter()
        .map(|a| t.data()[a] as f64)
        .collect()
}

pub fn forward_1d(spec: Conv1dSpec, x: &[f64], w: &[f64]) -> Vec<f64> {
    let out_len = spec.output_len().unwrap();
    let cin_g = spec.in_channels / spec.groups;
    let cout_g = spec.out_channels / spec.groups;
    let mut out = vec![0.0; spec.batch * spec.out_channels * out_len];
    for n in 0..spec.batch {
        for g in 0..spec.groups {
            for co_l in 0..cout_g {
                let co = g * cout_g + co_l;
                for t in 0..out_len {
                    let mut sum = 0.0;
                    for ci_l in 0..cin_g {
                        let ci = g * cin_g + ci_l;
                        for k in 0..spec.kernel {
                            let pos = (t * spec.stride + k * spec.dilation) as isize
                                - spec.padding as isize;
                            if pos >= 0 && (pos as usize) < spec.input_len {
                                let xv =
                                    x[(n * spec.in_channels + ci) * spec.input_len + pos as usize];
                                let wv = w[(co * cin_g + ci_l) * spec.kernel + k];
                                sum += xv * wv;
                            }
                        }
                    }
                    out[(n * spec.out_channels + co) * out_len + t] = sum;
                }
            }
        }
    }
    out
}

pub fn derivative_1d(spec: Conv1dSpec, side: Side, x: &[f64], w: &[f64], dy: &[f64]) -> Vec<f64> {
    let out_len = spec.output_len().unwrap();
    let cin_g = spec.in_channels / spec.groups;
    let cout_g = spec.out_channels / spec.groups;
    if side == Side::Left {
        let mut dx = vec![0.0; spec.batch * spec.in_channels * spec.input_len];
        for n in 0..spec.batch {
            for g in 0..spec.groups {
                for co_l in 0..cout_g {
                    let co = g * cout_g + co_l;
                    for t in 0..out_len {
                        let grad_y = dy[(n * spec.out_channels + co) * out_len + t];
                        for ci_l in 0..cin_g {
                            let ci = g * cin_g + ci_l;
                            for k in 0..spec.kernel {
                                let pos = (t * spec.stride + k * spec.dilation) as isize
                                    - spec.padding as isize;
                                if pos >= 0 && (pos as usize) < spec.input_len {
                                    let wv = w[(co * cin_g + ci_l) * spec.kernel + k];
                                    dx[(n * spec.in_channels + ci) * spec.input_len
                                        + pos as usize] += grad_y * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
        dx
    } else {
        let mut dw = vec![0.0; spec.out_channels * cin_g * spec.kernel];
        for n in 0..spec.batch {
            for g in 0..spec.groups {
                for co_l in 0..cout_g {
                    let co = g * cout_g + co_l;
                    for t in 0..out_len {
                        let grad_y = dy[(n * spec.out_channels + co) * out_len + t];
                        for ci_l in 0..cin_g {
                            let ci = g * cin_g + ci_l;
                            for k in 0..spec.kernel {
                                let pos = (t * spec.stride + k * spec.dilation) as isize
                                    - spec.padding as isize;
                                if pos >= 0 && (pos as usize) < spec.input_len {
                                    let xv = x[(n * spec.in_channels + ci) * spec.input_len
                                        + pos as usize];
                                    dw[(co * cin_g + ci_l) * spec.kernel + k] += grad_y * xv;
                                }
                            }
                        }
                    }
                }
            }
        }
        dw
    }
}

pub fn forward_2d(spec: Conv2dSpec, x: &[f64], w: &[f64]) -> Vec<f64> {
    let [_, _, oh, ow] = spec.output_shape().unwrap();
    let [ih, iw] = spec.input;
    let [kh, kw] = spec.kernel;
    let cin_g = spec.in_channels / spec.groups;
    let cout_g = spec.out_channels / spec.groups;
    let mut out = vec![0.0; spec.batch * spec.out_channels * oh * ow];
    for n in 0..spec.batch {
        for g in 0..spec.groups {
            for co_l in 0..cout_g {
                let co = g * cout_g + co_l;
                for ho in 0..oh {
                    for wo in 0..ow {
                        let mut sum = 0.0;
                        for ci_l in 0..cin_g {
                            let ci = g * cin_g + ci_l;
                            for r in 0..kh {
                                let hi = (ho * spec.stride[0] + r * spec.dilation[0]) as isize
                                    - spec.padding[0] as isize;
                                if hi < 0 || (hi as usize) >= ih {
                                    continue;
                                }
                                for s in 0..kw {
                                    let wi = (wo * spec.stride[1] + s * spec.dilation[1]) as isize
                                        - spec.padding[1] as isize;
                                    if wi >= 0 && (wi as usize) < iw {
                                        let xv = x[((n * spec.in_channels + ci) * ih
                                            + hi as usize)
                                            * iw
                                            + wi as usize];
                                        let wv = w[((co * cin_g + ci_l) * kh + r) * kw + s];
                                        sum += xv * wv;
                                    }
                                }
                            }
                        }
                        out[((n * spec.out_channels + co) * oh + ho) * ow + wo] = sum;
                    }
                }
            }
        }
    }
    out
}

pub fn derivative_2d(spec: Conv2dSpec, side: Side, x: &[f64], w: &[f64], dy: &[f64]) -> Vec<f64> {
    let [_, _, oh, ow] = spec.output_shape().unwrap();
    let [ih, iw] = spec.input;
    let [kh, kw] = spec.kernel;
    let cin_g = spec.in_channels / spec.groups;
    let cout_g = spec.out_channels / spec.groups;
    if side == Side::Left {
        let mut dx = vec![0.0; spec.batch * spec.in_channels * ih * iw];
        for n in 0..spec.batch {
            for g in 0..spec.groups {
                for co_l in 0..cout_g {
                    let co = g * cout_g + co_l;
                    for ho in 0..oh {
                        for wo in 0..ow {
                            let grad_y = dy[((n * spec.out_channels + co) * oh + ho) * ow + wo];
                            for ci_l in 0..cin_g {
                                let ci = g * cin_g + ci_l;
                                for r in 0..kh {
                                    let hi = (ho * spec.stride[0] + r * spec.dilation[0]) as isize
                                        - spec.padding[0] as isize;
                                    if hi < 0 || (hi as usize) >= ih {
                                        continue;
                                    }
                                    for s in 0..kw {
                                        let wi = (wo * spec.stride[1] + s * spec.dilation[1])
                                            as isize
                                            - spec.padding[1] as isize;
                                        if wi >= 0 && (wi as usize) < iw {
                                            let wv = w[((co * cin_g + ci_l) * kh + r) * kw + s];
                                            dx[((n * spec.in_channels + ci) * ih + hi as usize)
                                                * iw
                                                + wi as usize] += grad_y * wv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        dx
    } else {
        let mut dw = vec![0.0; spec.out_channels * cin_g * kh * kw];
        for n in 0..spec.batch {
            for g in 0..spec.groups {
                for co_l in 0..cout_g {
                    let co = g * cout_g + co_l;
                    for ho in 0..oh {
                        for wo in 0..ow {
                            let grad_y = dy[((n * spec.out_channels + co) * oh + ho) * ow + wo];
                            for ci_l in 0..cin_g {
                                let ci = g * cin_g + ci_l;
                                for r in 0..kh {
                                    let hi = (ho * spec.stride[0] + r * spec.dilation[0]) as isize
                                        - spec.padding[0] as isize;
                                    if hi < 0 || (hi as usize) >= ih {
                                        continue;
                                    }
                                    for s in 0..kw {
                                        let wi = (wo * spec.stride[1] + s * spec.dilation[1])
                                            as isize
                                            - spec.padding[1] as isize;
                                        if wi >= 0 && (wi as usize) < iw {
                                            let xv = x[((n * spec.in_channels + ci) * ih
                                                + hi as usize)
                                                * iw
                                                + wi as usize];
                                            dw[((co * cin_g + ci_l) * kh + r) * kw + s] +=
                                                grad_y * xv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        dw
    }
}

pub fn close(actual: f32, want: f64) {
    assert!(
        (actual as f64 - want).abs() <= 2e-5 + 2e-5 * want.abs(),
        "{actual} != {want}"
    );
}

pub fn holes(l: &Layout, initial: &Tensor, actual: &Tensor) {
    let used: BTreeSet<_> = addresses(l).into_iter().collect();
    for (i, (&a, &b)) in initial.data().iter().zip(actual.data()).enumerate() {
        if !used.contains(&i) {
            assert_eq!(a.to_bits(), b.to_bits(), "hole {i}");
        }
    }
}

pub fn verify(l: &Layout, expected: &[f64], initial: &Tensor, actual: &Tensor, mode: G) {
    assert_eq!(expected.len(), l.shape().len());
    for (&index, &want) in addresses(l).iter().zip(expected) {
        close(
            actual.data()[index],
            want + if mode == G::Accumulate {
                initial.data()[index] as f64
            } else {
                0.0
            },
        );
    }
    holes(l, initial, actual);
}

pub fn identities_1d(
    c: &Case1d,
    side: Side,
    x: &Tensor,
    w: &Tensor,
    y: &Tensor,
    dy: &Tensor,
    gradient: &Tensor,
) {
    let xx = logical(&c.x, x);
    let ww = logical(&c.w, w);
    let dd = logical(&c.dy, dy);
    let gg = logical(c.gradient(side), gradient);
    let lhs: f64 = logical(&c.y, y).iter().zip(&dd).map(|(y, d)| y * d).sum();
    let input = if side == Side::Left { &xx } else { &ww };
    let rhs: f64 = input.iter().zip(&gg).map(|(v, g)| v * g).sum();
    assert!(
        (lhs - rhs).abs() <= 2e-5 + 2e-5 * lhs.abs(),
        "1D adjoint {lhs} != {rhs}"
    );
    for i in [0, input.len() / 2, input.len() - 1] {
        let mut p = input.clone();
        p[i] += 1e-4;
        let objective = |p: &[f64]| {
            let value = if side == Side::Left {
                forward_1d(c.spec, p, &ww)
            } else {
                forward_1d(c.spec, &xx, p)
            };
            value.iter().zip(&dd).map(|(y, d)| y * d).sum::<f64>()
        };
        let plus = objective(&p);
        p[i] -= 2e-4;
        close(gg[i] as f32, (plus - objective(&p)) / 2e-4);
    }
}

pub fn identities_2d(
    c: &Case2d,
    side: Side,
    x: &Tensor,
    w: &Tensor,
    y: &Tensor,
    dy: &Tensor,
    gradient: &Tensor,
) {
    let xx = logical(&c.x, x);
    let ww = logical(&c.w, w);
    let dd = logical(&c.dy, dy);
    let gg = logical(c.gradient(side), gradient);
    let lhs: f64 = logical(&c.y, y).iter().zip(&dd).map(|(y, d)| y * d).sum();
    let input = if side == Side::Left { &xx } else { &ww };
    let rhs: f64 = input.iter().zip(&gg).map(|(v, g)| v * g).sum();
    assert!(
        (lhs - rhs).abs() <= 2e-5 + 2e-5 * lhs.abs(),
        "2D adjoint {lhs} != {rhs}"
    );
    for i in [0, input.len() / 2, input.len() - 1] {
        let mut p = input.clone();
        p[i] += 1e-4;
        let objective = |p: &[f64]| {
            let value = if side == Side::Left {
                forward_2d(c.spec, p, &ww)
            } else {
                forward_2d(c.spec, &xx, p)
            };
            value.iter().zip(&dd).map(|(y, d)| y * d).sum::<f64>()
        };
        let plus = objective(&p);
        p[i] -= 2e-4;
        close(gg[i] as f32, (plus - objective(&p)) / 2e-4);
    }
}

pub const TRAIN_SPEC_1D: Conv1dSpec = Conv1dSpec {
    batch: 2,
    in_channels: 2,
    out_channels: 2,
    input_len: 5,
    kernel: 3,
    stride: 1,
    padding: 1,
    dilation: 1,
    groups: 1,
};

pub const TRAIN_SPEC_2D: Conv2dSpec = Conv2dSpec {
    batch: 2,
    in_channels: 2,
    out_channels: 2,
    input: [2, 3],
    kernel: [2, 2],
    stride: [1, 1],
    padding: [0, 0],
    dilation: [1, 1],
    groups: 1,
    transposed: false,
    output_padding: [0, 0],
};

pub fn training_fixture() -> [Tensor; 6] {
    let x1 = Tensor::from_vec(
        [2, 2, 5],
        (0..20)
            .map(|i| (((i * 7 + 3) % 19) as f32 - 9.0) / 16.0)
            .collect(),
    )
    .unwrap();
    let w1 = Tensor::from_vec(
        [2, 2, 3],
        (0..12)
            .map(|i| (((i * 5 + 1) % 13) as f32 - 6.0) / 16.0)
            .collect(),
    )
    .unwrap();
    let b1 = Tensor::from_vec([2], vec![0.125, -0.1875]).unwrap();
    let x2 = Tensor::from_vec(
        [2, 2, 2, 3],
        (0..24)
            .map(|i| (((i * 11 + 2) % 23) as f32 - 11.0) / 32.0)
            .collect(),
    )
    .unwrap();
    let w2 = Tensor::from_vec(
        [2, 2, 2, 2],
        (0..16)
            .map(|i| (((i * 3 + 7) % 17) as f32 - 8.0) / 16.0)
            .collect(),
    )
    .unwrap();
    let b2 = Tensor::from_vec([2], vec![-0.0625, 0.125]).unwrap();
    [x1, w1, b1, x2, w2, b2]
}

pub fn training_control(
    x1: &[f64],
    w1: &mut [f64],
    b1: &mut [f64],
    x2: &[f64],
    w2: &mut [f64],
    b2: &mut [f64],
    rate: f64,
) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let y1 = forward_1d(TRAIN_SPEC_1D, x1, w1);
    let mut z1 = y1.clone();
    for n in 0..2 {
        for c in 0..2 {
            for t in 0..5 {
                z1[(n * 2 + c) * 5 + t] += b1[c];
            }
        }
    }
    let y2 = forward_2d(TRAIN_SPEC_2D, x2, w2);
    let mut z2 = y2.clone();
    for n in 0..2 {
        for c in 0..2 {
            for hw in 0..2 {
                z2[(n * 2 + c) * 2 + hw] += b2[c];
            }
        }
    }
    let loss1 = z1.iter().map(|v| v * v).sum::<f64>() / (z1.len() as f64);
    let loss2 = z2.iter().map(|v| v * v).sum::<f64>() / (z2.len() as f64);
    let loss = loss1 + loss2;
    let dz1: Vec<_> = z1.iter().map(|v| 2.0 * v / (z1.len() as f64)).collect();
    let dz2: Vec<_> = z2.iter().map(|v| 2.0 * v / (z2.len() as f64)).collect();
    let gw1 = derivative_1d(TRAIN_SPEC_1D, Side::Right, x1, w1, &dz1);
    let gw2 = derivative_2d(TRAIN_SPEC_2D, Side::Right, x2, w2, &dz2);
    let mut gb1 = vec![0.0; 2];
    for n in 0..2 {
        for c in 0..2 {
            for t in 0..5 {
                gb1[c] += dz1[(n * 2 + c) * 5 + t];
            }
        }
    }
    let mut gb2 = vec![0.0; 2];
    for n in 0..2 {
        for c in 0..2 {
            for hw in 0..2 {
                gb2[c] += dz2[(n * 2 + c) * 2 + hw];
            }
        }
    }
    for (p, &g) in w1.iter_mut().zip(&gw1) {
        *p -= rate * g;
    }
    for (p, &g) in b1.iter_mut().zip(&gb1) {
        *p -= rate * g;
    }
    for (p, &g) in w2.iter_mut().zip(&gw2) {
        *p -= rate * g;
    }
    for (p, &g) in b2.iter_mut().zip(&gb2) {
        *p -= rate * g;
    }
    (loss, gw1, gb1, gw2, gb2)
}

pub struct Exporter {
    writer: Option<std::io::BufWriter<std::fs::File>>,
}

fn logical_bits(layout: &Layout, tensor: &Tensor) -> Vec<u32> {
    addresses(layout)
        .into_iter()
        .map(|a| tensor.data()[a].to_bits())
        .collect()
}

impl Exporter {
    pub fn new(stem: Option<&str>) -> Self {
        let directory = stem.and_then(|s| {
            std::env::var_os("FA_CONV_EXPORT_DIR").map(|p| (std::path::PathBuf::from(p), s))
        });
        let writer = directory.map(|(dir, stem)| {
            std::fs::create_dir_all(&dir).unwrap();
            std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(dir.join(format!("{stem}.jsonl")))
                    .unwrap(),
            )
        });
        Self { writer }
    }
    pub fn enabled(&self) -> bool {
        self.writer.is_some()
    }
    pub fn record(&mut self, value: serde_json::Value) {
        use std::io::Write;
        if let Some(writer) = self.writer.as_mut() {
            serde_json::to_writer(&mut *writer, &value).unwrap();
            writer.write_all(b"\n").unwrap();
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &mut self,
        backend: &str,
        index: usize,
        binding: usize,
        c: &Case1d,
        inputs: [&Tensor; 3],
        output: &Tensor,
        gradients: [[&Tensor; 2]; 2],
    ) {
        if !self.enabled() {
            return;
        }
        let s = c.spec;
        self.record(serde_json::json!({
            "schema_version": 1,
            "kind": "conv1d",
            "backend": backend,
            "case_index": index,
            "binding": binding,
            "dims": [s.batch, s.in_channels, s.out_channels, s.input_len, s.kernel, s.stride, s.padding, s.dilation, s.groups],
            "x": logical_bits(&c.x, inputs[0]),
            "w": logical_bits(&c.w, inputs[1]),
            "upstream": logical_bits(&c.dy, inputs[2]),
            "output": logical_bits(&c.y, output),
            "left_gradient": logical_bits(&c.dx, gradients[0][0]),
            "right_gradient": logical_bits(&c.dw, gradients[0][1]),
            "left_accumulate": logical_bits(&c.dx, gradients[1][0]),
            "right_accumulate": logical_bits(&c.dw, gradients[1][1]),
            "left_seed": logical_bits(&c.dx, &seed(&c.dx, true)),
            "right_seed": logical_bits(&c.dw, &seed(&c.dw, true)),
        }));
    }
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &mut self,
        backend: &str,
        index: usize,
        binding: usize,
        c: &Case2d,
        inputs: [&Tensor; 3],
        output: &Tensor,
        gradients: [[&Tensor; 2]; 2],
    ) {
        if !self.enabled() {
            return;
        }
        let s = c.spec;
        self.record(serde_json::json!({
            "schema_version": 1,
            "kind": "conv2d",
            "backend": backend,
            "case_index": index,
            "binding": binding,
            "dims": [
                s.batch, s.in_channels, s.out_channels,
                s.input[0], s.input[1], s.kernel[0], s.kernel[1],
                s.stride[0], s.stride[1], s.padding[0], s.padding[1],
                s.dilation[0], s.dilation[1], s.groups
            ],
            "x": logical_bits(&c.x, inputs[0]),
            "w": logical_bits(&c.w, inputs[1]),
            "upstream": logical_bits(&c.dy, inputs[2]),
            "output": logical_bits(&c.y, output),
            "left_gradient": logical_bits(&c.dx, gradients[0][0]),
            "right_gradient": logical_bits(&c.dw, gradients[0][1]),
            "left_accumulate": logical_bits(&c.dx, gradients[1][0]),
            "right_accumulate": logical_bits(&c.dw, gradients[1][1]),
            "left_seed": logical_bits(&c.dx, &seed(&c.dx, true)),
            "right_seed": logical_bits(&c.dw, &seed(&c.dw, true)),
        }));
    }
    #[allow(clippy::too_many_arguments)]
    pub fn training(
        &mut self,
        backend: &str,
        step: usize,
        before: [&Tensor; 4],
        loss: &Tensor,
        grads: [&Tensor; 4],
        after: [&Tensor; 4],
    ) {
        if !self.enabled() {
            return;
        }
        self.record(serde_json::json!({
            "schema_version": 1,
            "kind": "sgd",
            "backend": backend,
            "step": step,
            "steps": 8,
            "w1": bits(before[0]),
            "b1": bits(before[1]),
            "w2": bits(before[2]),
            "b2": bits(before[3]),
            "loss": bits(loss),
            "gw1": bits(grads[0]),
            "gb1": bits(grads[1]),
            "gw2": bits(grads[2]),
            "gb2": bits(grads[3]),
            "updated_w1": bits(after[0]),
            "updated_b1": bits(after[1]),
            "updated_w2": bits(after[2]),
            "updated_b2": bits(after[3]),
        }));
    }
    pub fn finish(mut self) {
        use std::io::Write;
        if let Some(writer) = self.writer.as_mut() {
            writer.flush().unwrap();
        }
    }
}
