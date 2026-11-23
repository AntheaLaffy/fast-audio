use super::*;
use crate::graph::mapping::{Map32, Mapping};
use crate::graph::{Backend, Executor, Kernel, Op, Program, Value};

unsafe extern "C" {
    fn fa_cuda_tensor_map32(
        ctx: *mut c_void,
        k: Map32,
        x: *const f32,
        y: *const f32,
        out: *mut f32,
    ) -> i32;
    fn fa_cuda_tensor_op(
        ctx: *mut c_void,
        k: Kernel,
        x: *const f32,
        y: *const f32,
        scale: *const f32,
        bias: *const f32,
        out: *mut f32,
    ) -> i32;
}
struct Gpu(Cuda);
enum Plan {
    Constant,
    Kernel(Kernel, Option<(DeviceTensor, DeviceTensor)>),
    Mapped(Map32),
    Linear {
        rows: i32,
        input: i32,
        output: i32,
        weights: DeviceTensor,
        bias: Option<DeviceTensor>,
    },
    Conv(Box<Conv2d>),
    Gru(Box<Gru>),
}
impl Backend for Gpu {
    type Buffer = DeviceTensor;
    type Plan = Plan;
    fn zeros(&self, shape: &Shape) -> Result<DeviceTensor> {
        self.0.zeros(shape.dims())
    }
    fn upload(&self, t: &Tensor) -> Result<DeviceTensor> {
        self.0.upload(t)
    }
    fn validate(&self, t: &DeviceTensor, shape: &Shape) -> Result<()> {
        self.0.owns(t)?;
        t.shape.expect(shape.dims())
    }
    fn prepare(&self, op: &Op) -> Result<Plan> {
        Ok(match op {
            Op::Constant(_) => Plan::Constant,
            Op::Kernel(k, _) if matches!(k.kind, 0 | 3) => match Mapping::new(*k).cuda32() {
                Some(map) => Plan::Mapped(map),
                None => Plan::Kernel(*k, None),
            },
            Op::Kernel(k, p) => Plan::Kernel(
                *k,
                p.as_ref()
                    .map(|(a, b)| -> Result<_> { Ok((self.0.upload(a)?, self.0.upload(b)?)) })
                    .transpose()?,
            ),
            Op::Linear {
                rows,
                input,
                output,
                weights,
                bias,
            } => Plan::Linear {
                rows: int(*rows)?,
                input: int(*input)?,
                output: int(*output)?,
                weights: self.0.upload(weights)?,
                bias: bias.as_ref().map(|b| self.0.upload(b)).transpose()?,
            },
            Op::Conv(s, w, b, a) => Plan::Conv(Box::new(Conv2d::new(
                &self.0,
                *s,
                w,
                b.as_ref(),
                *a,
                64 << 20,
            )?)),
            Op::Gru(s, w) => Plan::Gru(Box::new(Gru::new(&self.0, *s, w)?)),
        })
    }
    fn execute(
        &self,
        plan: &mut Plan,
        input: [Option<&DeviceTensor>; 2],
        output: &mut [DeviceTensor],
    ) -> Result<()> {
        let x = input[0].unwrap();
        match plan {
            Plan::Constant => Ok(()),
            Plan::Mapped(k) => {
                // SAFETY: preparation proves all mapped source/output indices fit u32;
                // the builder validates their extents and the retained stream owns storage.
                check(unsafe {
                    fa_cuda_tensor_map32(
                        self.0.session.ptr(),
                        *k,
                        x.read(),
                        input[1].map_or(std::ptr::null(), DeviceTensor::read),
                        output[0].write(),
                    )
                })
            }
            Plan::Kernel(k, p) => {
                let (scale, bias) = p
                    .as_ref()
                    .map_or((std::ptr::null(), std::ptr::null()), |(a, b)| {
                        (a.read(), b.read())
                    });
                // SAFETY: the builder checked every index and parameter extent. All storage
                // belongs to this retained session; the stream orders all graph edges.
                check(unsafe {
                    fa_cuda_tensor_op(
                        self.0.session.ptr(),
                        *k,
                        x.read(),
                        input[1].map_or(std::ptr::null(), DeviceTensor::read),
                        scale,
                        bias,
                        output[0].write(),
                    )
                })
            }
            Plan::Linear {
                rows,
                input,
                output: width,
                weights,
                bias,
            } => {
                // SAFETY: checked dense row counts and packed [in,out] weights; output is
                // exclusive and every pointer remains live for this session's queued work.
                check(unsafe {
                    fa_cuda_matmul(
                        self.0.session.ptr(),
                        *rows,
                        *width,
                        *input,
                        x.read(),
                        weights.read(),
                        output[0].write(),
                    )
                })?;
                if let Some(b) = bias {
                    let k = Kernel {
                        kind: 11,
                        len: output[0].shape.len() as u64,
                        inner: *width as u64,
                        ..Kernel::default()
                    };
                    // SAFETY: each thread updates one already-written GEMM output element;
                    // the bias has exactly one element per column and the stream orders GEMM.
                    check(unsafe {
                        fa_cuda_tensor_op(
                            self.0.session.ptr(),
                            k,
                            output[0].read(),
                            std::ptr::null(),
                            std::ptr::null(),
                            b.read(),
                            output[0].write(),
                        )
                    })?;
                }
                Ok(())
            }
            Plan::Conv(p) => p.run(x, &mut output[0]),
            Plan::Gru(p) => {
                let (y, h) = output.split_at_mut(1);
                p.run(x, input[1].unwrap(), &mut y[0], &mut h[0])
            }
        }
    }
    fn bytes(plan: &Plan) -> (usize, usize) {
        match plan {
            Plan::Constant => (0, 0),
            Plan::Mapped(_) => (0, 0),
            Plan::Kernel(_, p) => (
                0,
                p.as_ref()
                    .map_or(0, |(a, b)| (a.shape.len() + b.shape.len()) * 4),
            ),
            Plan::Linear { weights, bias, .. } => (
                0,
                (weights.shape.len() + bias.as_ref().map_or(0, |b| b.shape.len())) * 4,
            ),
            Plan::Conv(p) => (p.workspace_bytes(), p.parameter_bytes()),
            Plan::Gru(p) => (p.workspace_bytes(), p.parameter_bytes()),
        }
    }
}
/// Fixed-shape, device-resident inference. Preparation uploads parameters exactly once.
pub struct Graph(Executor<Gpu>);
impl Graph {
    pub fn new(cuda: &Cuda, program: Program) -> Result<Self> {
        Ok(Self(Executor::new(Gpu(cuda.clone()), program)?))
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
    /// Selected cuDNN convolutions by zero-based graph step. Pointwise cuBLAS is omitted.
    pub fn cudnn_convolution_plans(&self) -> impl Iterator<Item = (usize, &CudnnConvolutionInfo)> {
        self.0
            .plans()
            .enumerate()
            .filter_map(|(step, plan)| match plan {
                Plan::Conv(conv) => conv.cudnn_plan_info().map(|info| (step, info)),
                _ => None,
            })
    }
    pub fn intermediate(&self, value: Value) -> Option<&DeviceTensor> {
        self.0.intermediate(value)
    }
    pub fn run(&mut self, input: &DeviceTensor, output: &mut DeviceTensor) -> Result<()> {
        self.0.run(input, output)
    }
}
