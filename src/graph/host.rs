use super::mapping::Mapping;
use super::*;
use crate::cpu::{self, Cpu};

struct Host(Cpu);
enum Plan {
    Constant,
    Kernel(Kernel, Option<(Tensor, Tensor)>),
    Mapped(Mapping),
    Linear {
        rows: usize,
        input: usize,
        output: usize,
        weights: Tensor,
        bias: Option<Tensor>,
    },
    Conv(Box<cpu::Conv2d>),
    Gru(Box<cpu::Gru>),
}
impl Backend for Host {
    type Buffer = Tensor;
    type Plan = Plan;
    fn zeros(&self, shape: &Shape) -> Result<Tensor> {
        Tensor::zeros(shape.dims())
    }
    fn upload(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
    fn validate(&self, input: &Tensor, shape: &Shape) -> Result<()> {
        input.shape().expect(shape.dims())
    }
    fn prepare(&self, op: &Op) -> Result<Plan> {
        Ok(match op {
            Op::Constant(_) => Plan::Constant,
            Op::Kernel(k, _) if matches!(k.kind, 0 | 3) => Plan::Mapped(Mapping::new(*k)),
            Op::Kernel(k, p) => Plan::Kernel(*k, p.clone()),
            Op::Linear {
                rows,
                input,
                output,
                weights,
                bias,
            } => Plan::Linear {
                rows: *rows,
                input: *input,
                output: *output,
                weights: weights.clone(),
                bias: bias.clone(),
            },
            Op::Conv(s, w, b, a) => {
                Plan::Conv(Box::new(cpu::Conv2d::new(self.0, *s, w, b.as_ref(), *a)?))
            }
            Op::Gru(s, w) => Plan::Gru(Box::new(cpu::Gru::new(self.0, *s, w)?)),
        })
    }
    fn execute(
        &self,
        plan: &mut Plan,
        input: [Option<&Tensor>; 2],
        output: &mut [Tensor],
    ) -> Result<()> {
        let x = input[0].unwrap();
        match plan {
            Plan::Constant => {}
            Plan::Kernel(k, p) => kernel(*k, x, input[1], p.as_ref(), &mut output[0]),
            Plan::Mapped(p) => p.run(x.data(), input[1].map(Tensor::data), output[0].data_mut()),
            Plan::Linear {
                rows,
                input,
                output: width,
                weights,
                bias,
            } => {
                self.0.gemm(
                    *rows,
                    *width,
                    *input,
                    x.data(),
                    weights.data(),
                    output[0].data_mut(),
                );
                if let Some(b) = bias {
                    for row in output[0].data_mut().chunks_exact_mut(*width) {
                        for (v, &offset) in row.iter_mut().zip(b.data()) {
                            *v += offset;
                        }
                    }
                }
            }
            Plan::Conv(plan) => plan.run(x, &mut output[0])?,
            Plan::Gru(plan) => {
                let (y, h) = output.split_at_mut(1);
                plan.run(x, input[1].unwrap(), &mut y[0], &mut h[0])?;
            }
        }
        Ok(())
    }
    fn bytes(plan: &Plan) -> (usize, usize) {
        match plan {
            Plan::Constant => (0, 0),
            Plan::Mapped(_) => (0, 0),
            Plan::Kernel(_, p) => (
                0,
                p.as_ref()
                    .map_or(0, |(a, b)| (a.shape().len() + b.shape().len()) * 4),
            ),
            Plan::Linear { weights, bias, .. } => (
                0,
                (weights.shape().len() + bias.as_ref().map_or(0, |b| b.shape().len())) * 4,
            ),
            Plan::Conv(p) => (p.workspace_bytes(), p.parameter_bytes()),
            Plan::Gru(p) => (p.workspace_bytes(), p.parameter_bytes()),
        }
    }
}
fn kernel(
    k: Kernel,
    x: &Tensor,
    y: Option<&Tensor>,
    parameters: Option<&(Tensor, Tensor)>,
    out: &mut Tensor,
) {
    let x = x.data();
    let y = y.map(Tensor::data);
    let n = k.inner as usize;
    let dest = out.data_mut();
    match k.kind {
        4 => {
            let a = k.width as usize * n;
            let b = k.offset as usize * n;
            let rows = x
                .chunks_exact(a)
                .zip(y.unwrap().chunks_exact(b))
                .zip(dest.chunks_exact_mut(a + b));
            for ((left, right), dst) in rows {
                if k.rank == 0 {
                    dst[..a].copy_from_slice(left);
                    dst[a..].copy_from_slice(right);
                } else {
                    for ((left, right), pair) in left
                        .chunks_exact(n)
                        .zip(right.chunks_exact(n))
                        .zip(dst.chunks_exact_mut(2 * n))
                    {
                        pair[..n].copy_from_slice(left);
                        pair[n..].copy_from_slice(right);
                    }
                }
            }
            return;
        }
        5 => {
            let before = k.offset as usize * n;
            let old = k.width as usize * n;
            let new = k.dims[0] as usize * n;
            for (src, dst) in x.chunks_exact(old).zip(dest.chunks_exact_mut(new)) {
                dst[..before].fill(0.0);
                dst[before..before + old].copy_from_slice(src);
                dst[before + old..].fill(0.0);
            }
            return;
        }
        6 => {
            let frequency = k.dims[3] as usize;
            let plane_len = k.dims[2] as usize * frequency;
            let width = k.width as usize;
            let radius = width / 2;
            for (channel, dst) in x
                .chunks_exact(plane_len)
                .zip(dest.chunks_exact_mut(width * plane_len))
            {
                for (tap, plane) in dst.chunks_exact_mut(plane_len).enumerate() {
                    let shift = tap.abs_diff(radius).min(frequency);
                    let (from, to) = if tap < radius { (0, shift) } else { (shift, 0) };
                    let count = frequency - shift;
                    for (src, out) in channel
                        .chunks_exact(frequency)
                        .zip(plane.chunks_exact_mut(frequency))
                    {
                        out[..to].fill(0.0);
                        out[to..to + count].copy_from_slice(&src[from..from + count]);
                        out[to + count..].fill(0.0);
                    }
                }
            }
            return;
        }
        _ => {}
    }
    if k.kind == 8 {
        let (scale, bias) = parameters.unwrap();
        for (row, dest) in x.chunks_exact(n).zip(dest.chunks_exact_mut(n)) {
            let mean = row.iter().sum::<f32>() / n as f32;
            let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inverse = 1.0 / (variance + k.eps).sqrt();
            for j in 0..n {
                dest[j] = (row[j] - mean) * inverse * scale.data()[j] + bias.data()[j];
            }
        }
        return;
    }
    for (i, out) in dest.iter_mut().enumerate() {
        *out = match k.kind {
            1 => match k.rank {
                0 => 1.0 / (1.0 + (-x[i]).exp()),
                1 => x[i].tanh(),
                _ => {
                    if x[i] < 0.0 {
                        k.param * x[i]
                    } else {
                        x[i]
                    }
                }
            },
            2 => x[i] + y.unwrap()[i],
            7 => {
                let row = &x[i * n..][..n];
                row.iter().map(|v| v * v).sum::<f32>() / n as f32
            }
            9 => {
                let channel = i / n % 3;
                let source = (i / (3 * n) * n + i % n) * 2;
                let re = x[source];
                let im = x[source + 1];
                match channel {
                    0 => (re * re + im * im + k.eps).sqrt(),
                    1 => re,
                    _ => im,
                }
            }
            10 => {
                let p = i / 2;
                let m = (p / n) * 2 * n + p % n;
                let s = y.unwrap();
                if i % 2 == 0 {
                    s[2 * p] * x[m] - s[2 * p + 1] * x[m + n]
                } else {
                    s[2 * p + 1] * x[m] + s[2 * p] * x[m + n]
                }
            }
            _ => unreachable!("validated kernel kind"),
        }
    }
}

/// Prepared fixed-shape CPU inference graph, with caller-owned input/output.
pub struct Graph(Executor<Host>);
impl Graph {
    pub fn new(cpu: Cpu, program: Program) -> Result<Self> {
        Ok(Self(Executor::new(Host(cpu), program)?))
    }
    pub fn input_shape(&self) -> &[usize] {
        self.0.input_shape()
    }
    pub fn output_shape(&self) -> &[usize] {
        self.0.output_shape()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.0.workspace_bytes()
    }
    pub fn parameter_bytes(&self) -> usize {
        self.0.parameter_bytes()
    }
    pub fn intermediate(&self, value: Value) -> Option<&Tensor> {
        self.0.intermediate(value)
    }
    pub fn run(&mut self, input: &Tensor, output: &mut Tensor) -> Result<()> {
        self.0.run(input, output)
    }
}
