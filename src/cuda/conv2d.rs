use super::*;
use crate::{Conv2dSpec, conv2d::Conv2dDims};

unsafe extern "C" {
    fn fa_cudnn_conv2d_create(
        ctx: *mut c_void,
        dims: Conv2dDims,
        bias: i32,
        relu: i32,
        workspace_limit: usize,
        require_deterministic: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_conv2d_pad(
        ctx: *mut c_void,
        x: *const f32,
        out: *mut f32,
        channels: i32,
        ih: i32,
        iw: i32,
        oh: i32,
        ow: i32,
        count: usize,
    ) -> i32;
}
enum Engine {
    Pointwise(ConvDims),
    Graph(CudnnPlan),
}

/// Constant-parameter NCHW inference plan; graph or direct pointwise cuBLAS execution.
pub struct Conv2d {
    cuda: Cuda,
    spec: Conv2dSpec,
    output: [usize; 4],
    weights: DeviceTensor,
    bias: Option<DeviceTensor>,
    engine: Engine,
    activation: Activation,
    padded: Option<DeviceTensor>,
}
impl Conv2d {
    pub fn new(
        cuda: &Cuda,
        spec: Conv2dSpec,
        weights: &Tensor,
        bias: Option<&Tensor>,
        activation: Activation,
        workspace_limit: usize,
    ) -> Result<Self> {
        spec.check_weights(weights, bias)?;
        let output = spec.output_shape()?;
        let mut dims = spec.native_dims()?;
        let packed = if spec.pointwise() {
            Tensor::from_vec(
                [spec.out_channels, spec.in_channels / spec.groups, 1, 1],
                spec.pointwise_weights(weights),
            )?
        } else {
            weights.clone()
        };
        let weights = cuda.upload(&packed)?;
        let bias = bias.map(|b| cuda.upload(b)).transpose()?;
        let mut padded = None;
        if spec.transposed
            && spec
                .output_padding
                .iter()
                .zip(spec.stride)
                .any(|(&p, s)| p >= s)
        {
            // Backward-data infers its source extent from the requested destination.
            // Larger dilation-valid output padding needs explicit high-edge source zeros.
            let h = spec.input[0]
                .checked_add(spec.output_padding[0] / spec.stride[0])
                .ok_or_else(|| invalid("extended transpose input overflows"))?;
            let w = spec.input[1]
                .checked_add(spec.output_padding[1] / spec.stride[1])
                .ok_or_else(|| invalid("extended transpose input overflows"))?;
            dims.ih = int(h)?;
            dims.iw = int(w)?;
            int(h
                .checked_mul(w)
                .ok_or_else(|| invalid("extended input area overflows"))?)?;
            padded = Some(cuda.zeros([spec.batch, spec.in_channels, h, w])?);
        }
        let engine = if spec.pointwise() {
            Engine::Pointwise(ConvDims {
                batch: dims.batch,
                in_channels: dims.in_channels,
                out_channels: dims.out_channels,
                input_len: dims.ih * dims.iw,
                kernel: 1,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: dims.groups,
                output_len: dims.oh * dims.ow,
            })
        } else {
            let padding_bytes = padded.as_ref().map_or(0, |p| p.shape.len() * 4);
            let engine_limit = workspace_limit
                .checked_sub(padding_bytes)
                .ok_or_else(|| invalid("workspace limit is smaller than extended input storage"))?;
            let mut ptr = std::ptr::null_mut();
            // SAFETY: exact NCHW/weight shapes, groups and all integer bounds are checked.
            // Native retains no host buffers and catches exceptions at its ABI boundary.
            let status = unsafe {
                fa_cudnn_conv2d_create(
                    cuda.session.ptr(),
                    dims,
                    i32::from(bias.is_some()),
                    i32::from(activation == Activation::Relu),
                    engine_limit,
                    cuda.convolution_policy.native(),
                    &mut ptr,
                )
            };
            CudnnPlan::check_preparation(status, cuda.convolution_policy, engine_limit)?;
            // SAFETY: successful native creation transfers its sole owner to this wrapper.
            Engine::Graph(unsafe { CudnnPlan::from_raw(cuda, ptr)? })
        };
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            output,
            weights,
            bias,
            engine,
            activation,
            padded,
        })
    }
    pub fn workspace_bytes(&self) -> usize {
        let graph = match &self.engine {
            Engine::Pointwise(_) => 0,
            Engine::Graph(g) => g.info.workspace_bytes,
        };
        graph + self.padded.as_ref().map_or(0, |x| x.shape.len() * 4)
    }
    pub fn parameter_bytes(&self) -> usize {
        (self.weights.shape.len() + self.bias.as_ref().map_or(0, |b| b.shape.len())) * 4
    }
    /// Actual cuDNN metadata, or None for the direct pointwise cuBLAS route.
    pub fn cudnn_plan_info(&self) -> Option<&CudnnConvolutionInfo> {
        match &self.engine {
            Engine::Pointwise(_) => None,
            Engine::Graph(plan) => Some(&plan.info),
        }
    }
    pub fn implementation(&self) -> &str {
        match &self.engine {
            Engine::Pointwise(_) => "cuBLAS pointwise FP32",
            Engine::Graph(g) => &g.implementation,
        }
    }
    pub fn run(&mut self, x: &DeviceTensor, out: &mut DeviceTensor) -> Result<()> {
        self.cuda.owns(x)?;
        self.cuda.owns(out)?;
        x.shape.expect(&self.spec.input_shape())?;
        out.shape.expect(&self.output)?;
        let input = if let Some(padded) = &mut self.padded {
            let dims = padded.shape.dims();
            let (h, w) = (dims[2], dims[3]);
            let count = padded.shape.len();
            // SAFETY: the checked extended tensor owns complete output storage. The kernel
            // copies the original extent and fills all remaining high-edge positions with zero.
            check(unsafe {
                fa_cuda_conv2d_pad(
                    self.cuda.session.ptr(),
                    x.read(),
                    padded.write(),
                    self.spec.in_channels as i32,
                    self.spec.input[0] as i32,
                    self.spec.input[1] as i32,
                    h as i32,
                    w as i32,
                    count,
                )
            })?;
            padded.read()
        } else {
            x.read()
        };
        match &mut self.engine {
            Engine::Pointwise(dims) => {
                // SAFETY: a 1x1 convolution is the same contiguous grouped GEMM as NCL,
                // with flattened spatial axes. Prepared IOHW weights were transposed to OIHW.
                check(unsafe {
                    fa_cuda_conv(
                        self.cuda.session.ptr(),
                        *dims,
                        input,
                        self.weights.read(),
                        self.bias
                            .as_ref()
                            .map_or(std::ptr::null(), DeviceTensor::read),
                        std::ptr::null_mut(),
                        out.write(),
                        i32::from(self.activation == Activation::Relu),
                    )
                })
            }
            Engine::Graph(g) => {
                // SAFETY: all inputs and output share the retained stream. Native caches
                // stable bindings and synchronizes before replacing a finalized variant pack.
                check(unsafe {
                    fa_cudnn_conv_run(
                        g.ptr.as_ptr(),
                        input,
                        self.weights.read(),
                        self.bias
                            .as_ref()
                            .map_or(std::ptr::null(), DeviceTensor::read),
                        out.write(),
                    )
                })
            }
        }
    }
}
