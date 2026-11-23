//! Single-device, single-stream CUDA execution. Handles are confined to one host thread.
//! Calls enqueue work; `synchronize` and downloads report asynchronous execution errors.
mod adjoint;
pub use adjoint::ViewBackward;
mod audio_ops;
pub use audio_ops::{
    CrossEntropyBackward, CrossEntropyPlan, DropoutBackward, DropoutPlan, EmbeddingBackward,
    EmbeddingPlan, RopeBackward, RopePlan,
};
mod mixed;
pub use mixed::{
    Bf16AudioTransformerConfig, Bf16AudioTransformerTrainer, Bf16CrossEntropyPlan, Bf16DropoutPlan,
    Bf16EmbeddingPlan, Bf16LinearPlan, Bf16MasterAdamW, Bf16NormPlan, Bf16RopePlan, Bf16SdpaPlan,
    Bf16SdpaStrides, Bf16SoftmaxPlan, Bf16SwiGluPlan, CudaGradScaler, DeviceBf16Tensor,
    DeviceIndexTensor, Tf32MatmulPlan,
};
mod conv;
pub use conv::{
    Conv1dBackward, Conv1dPlan, Conv2dBackward, Conv2dPlan, ConvTranspose1dBackward,
    ConvTranspose1dPlan, ConvTranspose2dBackward, ConvTranspose2dPlan,
};
mod matmul;
pub use matmul::{MatmulBackward, MatmulPlan};
mod bmm;
pub use bmm::{BatchedMatmulBackward, BatchedMatmulPlan};
mod sdpa;
pub use sdpa::{SdpaBackward, SdpaPlan};
mod norm;
pub use crate::optim::CudaAdamW as AdamW;
pub use norm::{NormBackward, NormPlan};
mod softmax;
pub use softmax::{SoftmaxBackward, SoftmaxPlan};
mod structural;
pub use structural::{CatBackward, CatPlan, PadBackward, PadPlan};
mod tape;
pub use tape::Tape;
mod pointwise;
pub use pointwise::{BinaryPlan, UnaryPlan};
mod pointwise_backward;
pub use pointwise_backward::{BinaryBackward, UnaryBackward};
mod reduce;
pub use reduce::{ReduceBackward, ReducePlan};
mod stft;
pub use stft::{
    CenteredStftBackward, CenteredStftPlan, IstftBackward, IstftPlan, StftBackward, StftPlan,
};
mod views;
pub use views::{CopyPlan, DeviceView, DeviceViewMut};
#[cfg(feature = "cudnn")]
mod conv2d;
#[cfg(feature = "cudnn")]
mod cudnn;
#[cfg(feature = "cudnn")]
mod graph;
use crate::{
    Activation, CenteredStftSpec, Conv1dSpec, Error, PrecisionPolicy, Result, Shape, StftSpec,
    Tensor,
    error::{int, invalid, native},
    spec::ConvDims,
    tensor::matmul_dims,
};
#[cfg(feature = "cudnn")]
pub use conv2d::Conv2d;
#[cfg(feature = "cudnn")]
use cudnn::CudnnPlan;
#[cfg(feature = "cudnn")]
pub use cudnn::{ConvolutionPolicy, CudnnConvolutionInfo};
#[cfg(feature = "cudnn")]
pub use graph::Graph;
use std::{
    ffi::{CStr, c_char, c_void},
    ptr::NonNull,
    rc::Rc,
    time::Instant,
};

unsafe extern "C" {
    fn fa_cuda_create(device: i32, out: *mut *mut c_void) -> i32;
    fn fa_cuda_create_with_precision(device: i32, policy: i32, out: *mut *mut c_void) -> i32;
    fn fa_cuda_destroy(ctx: *mut c_void);
    fn fa_cuda_name(ctx: *mut c_void, name: *mut c_char) -> i32;
    fn fa_cuda_sync(ctx: *mut c_void) -> i32;
    fn fa_cuda_alloc(ctx: *mut c_void, bytes: usize, out: *mut *mut c_void) -> i32;
    fn fa_cuda_free(ctx: *mut c_void, ptr: *mut c_void);
    fn fa_cuda_copy(
        ctx: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        to_device: i32,
    ) -> i32;
    fn fa_cuda_zero(ctx: *mut c_void, dst: *mut c_void, bytes: usize) -> i32;
    fn fa_cuda_matmul(
        ctx: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
    ) -> i32;
    fn fa_cuda_conv(
        ctx: *mut c_void,
        dims: ConvDims,
        x: *const f32,
        w: *const f32,
        bias: *const f32,
        scratch: *mut f32,
        out: *mut f32,
        relu: i32,
    ) -> i32;
    fn fa_cuda_stft_create(
        ctx: *mut c_void,
        input_len: i32,
        n: i32,
        hop: i32,
        padding: i32,
        frames: i32,
        transforms: i32,
        window: *const f32,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_stft_run(plan: *mut c_void, x: *const f32, out: *mut f32) -> i32;
    fn fa_cuda_stft_destroy(plan: *mut c_void);
    fn fa_cuda_istft_create(
        ctx: *mut c_void,
        output_len: i32,
        n: i32,
        hop: i32,
        padding: i32,
        frames: i32,
        transforms: i32,
        window: *const f32,
        normalization: *const f32,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_istft_run(plan: *mut c_void, spectrum: *const f32, out: *mut f32) -> i32;
    fn fa_cuda_istft_destroy(plan: *mut c_void);
    fn fa_cuda_event_create(ctx: *mut c_void, out: *mut *mut c_void) -> i32;
    fn fa_cuda_event_record(ctx: *mut c_void, event: *mut c_void) -> i32;
    fn fa_cuda_event_elapsed(
        ctx: *mut c_void,
        start: *mut c_void,
        end: *mut c_void,
        ms: *mut f32,
    ) -> i32;
    fn fa_cuda_event_destroy(ctx: *mut c_void, event: *mut c_void);
}
#[cfg(feature = "cudnn")]
unsafe extern "C" {
    fn fa_cudnn_gru_create(
        ctx: *mut c_void,
        dims: GruDims,
        wi: *const f32,
        wh: *const f32,
        bi: *const f32,
        bh: *const f32,
        automatic: i32,
        sequence_major: i32,
        out: *mut *mut c_void,
        workspace: *mut usize,
        parameters: *mut usize,
        selected_small: *mut i32,
    ) -> i32;
    fn fa_cudnn_gru_run(
        plan: *mut c_void,
        x: *const f32,
        initial: *const f32,
        out: *mut f32,
        final_state: *mut f32,
    ) -> i32;
    fn fa_cudnn_gru_destroy(plan: *mut c_void);
    fn fa_cudnn_conv_create(
        ctx: *mut c_void,
        dims: ConvDims,
        bias: i32,
        relu: i32,
        workspace_limit: usize,
        require_deterministic: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cudnn_conv_run(
        plan: *mut c_void,
        x: *const f32,
        w: *const f32,
        bias: *const f32,
        out: *mut f32,
    ) -> i32;
}

#[cfg(feature = "cudnn")]
#[repr(C)]
#[derive(Clone, Copy)]
struct GruDims {
    batch: i32,
    steps: i32,
    input: i32,
    hidden: i32,
    directions: i32,
}

/// cuDNN inference algorithm policy; Standard retains an explicit control.
#[cfg(feature = "cudnn")]
#[derive(Clone, Copy, Debug)]
pub enum GruAlgorithm {
    Automatic,
    Standard,
}
#[cfg(feature = "cudnn")]
#[derive(Clone, Copy, Debug)]
/// Internal cuDNN layout. Public input and output tensors remain batch-first.
pub enum GruLayout {
    BatchMajor,
    SequenceMajor,
}
#[cfg(feature = "cudnn")]
#[derive(Clone, Copy, Debug)]
/// Fixed preparation policy; execution performs no algorithm search.
pub struct GruOptions {
    pub algorithm: GruAlgorithm,
    pub layout: GruLayout,
}
#[cfg(feature = "cudnn")]
impl Default for GruOptions {
    fn default() -> Self {
        Self {
            algorithm: GruAlgorithm::Automatic,
            layout: GruLayout::SequenceMajor,
        }
    }
}

/// Prepared, snapshot-parameter GRU. The automatic policy uses persistent small-hidden
/// inference for hidden_size <= 192 on SM 6+ and otherwise uses the standard algorithm.
#[cfg(feature = "cudnn")]
pub struct Gru {
    cuda: Cuda,
    spec: crate::GruSpec,
    ptr: NonNull<c_void>,
    workspace: usize,
    parameters: usize,
    small_hidden: bool,
    sequence_major: bool,
}
#[cfg(feature = "cudnn")]
impl Gru {
    pub fn new(cuda: &Cuda, spec: crate::GruSpec, weights: &crate::GruWeights) -> Result<Self> {
        Self::with_options(cuda, spec, weights, GruOptions::default())
    }
    pub fn with_options(
        cuda: &Cuda,
        spec: crate::GruSpec,
        weights: &crate::GruWeights,
        options: GruOptions,
    ) -> Result<Self> {
        spec.check_weights(weights)?;
        if spec.steps > u16::MAX as usize {
            return Err(invalid("cuDNN GRU sequence length exceeds 65535"));
        }
        let dims = GruDims {
            batch: spec.batch as i32,
            steps: spec.steps as i32,
            input: spec.input_size as i32,
            hidden: spec.hidden_size as i32,
            directions: spec.directions() as i32,
        };
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        let mut parameters = 0;
        let mut small_hidden = 0;
        // SAFETY: exact direction/gate shapes and i32 limits are checked above. Native
        // copies borrowed parameters synchronously into owned vendor weight storage.
        check(unsafe {
            fa_cudnn_gru_create(
                cuda.session.ptr(),
                dims,
                weights.input.data().as_ptr(),
                weights.hidden.data().as_ptr(),
                weights.input_bias.data().as_ptr(),
                weights.hidden_bias.data().as_ptr(),
                i32::from(matches!(options.algorithm, GruAlgorithm::Automatic)),
                i32::from(matches!(options.layout, GruLayout::SequenceMajor)),
                &mut ptr,
                &mut workspace,
                &mut parameters,
                &mut small_hidden,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            ptr: nonnull(ptr)?,
            workspace,
            parameters,
            small_hidden: small_hidden != 0,
            sequence_major: matches!(options.layout, GruLayout::SequenceMajor),
        })
    }
    pub fn spec(&self) -> crate::GruSpec {
        self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
    }
    pub fn parameter_bytes(&self) -> usize {
        self.parameters
    }
    pub fn implementation(&self) -> &'static str {
        match (self.small_hidden, self.sequence_major) {
            (true, true) => "cuDNN GRU persist-small-h FP32 seq-major",
            (true, false) => "cuDNN GRU persist-small-h FP32 batch-major",
            (false, true) => "cuDNN GRU standard FP32 seq-major",
            (false, false) => "cuDNN GRU standard FP32 batch-major",
        }
    }
    pub fn run(
        &mut self,
        input: &DeviceTensor,
        initial: &DeviceTensor,
        output: &mut DeviceTensor,
        final_state: &mut DeviceTensor,
    ) -> Result<()> {
        for tensor in [input, initial, &*output, &*final_state] {
            self.cuda.owns(tensor)?;
        }
        input.shape.expect(&self.spec.input_shape())?;
        initial.shape.expect(&self.spec.state_shape())?;
        output.shape.expect(&[
            self.spec.batch,
            self.spec.steps,
            self.spec.directions() * self.spec.hidden_size,
        ])?;
        final_state.shape.expect(&self.spec.state_shape())?;
        // SAFETY: all buffers match the prepared geometry and session; output borrows
        // are exclusive, and the single stream orders all pending input and plan uses.
        check(unsafe {
            fa_cudnn_gru_run(
                self.ptr.as_ptr(),
                input.read(),
                initial.read(),
                output.write(),
                final_state.write(),
            )
        })
    }
}
#[cfg(feature = "cudnn")]
impl Drop for Gru {
    fn drop(&mut self) {
        // SAFETY: one plan owner retains the CUDA session. Native waits for pending uses.
        unsafe { fa_cudnn_gru_destroy(self.ptr.as_ptr()) }
    }
}

pub(crate) fn check(status: i32) -> Result<()> {
    native("CUDA", status.into())
}
pub(crate) fn nonnull(ptr: *mut c_void) -> Result<NonNull<c_void>> {
    NonNull::new(ptr).ok_or_else(|| Error::Allocation("native backend returned null".into()))
}

pub(crate) struct Session(NonNull<c_void>);
impl Session {
    pub(crate) fn ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: Rc keeps this session alive until all buffers, plans, and events are destroyed.
        unsafe { fa_cuda_destroy(self.ptr()) }
    }
}

#[derive(Clone)]
pub struct Cuda {
    pub(crate) session: Rc<Session>,
    precision_policy: PrecisionPolicy,
    #[cfg(feature = "cudnn")]
    convolution_policy: ConvolutionPolicy,
}

/// Owned FP32 device allocation, initialized to zeros and tied to its original stream.
pub struct DeviceTensor {
    session: Rc<Session>,
    shape: Shape,
    ptr: Option<NonNull<c_void>>,
}
impl DeviceTensor {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub(crate) fn read(&self) -> *const f32 {
        self.ptr.map_or(std::ptr::null(), |ptr| ptr.as_ptr().cast())
    }
    pub(crate) fn write(&mut self) -> *mut f32 {
        self.ptr
            .map_or(std::ptr::null_mut(), |ptr| ptr.as_ptr().cast())
    }
}
impl Drop for DeviceTensor {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr {
            // SAFETY: allocation has one owner. cudaFree waits for outstanding device uses.
            unsafe { fa_cuda_free(self.session.ptr(), ptr.as_ptr()) }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub wall_us: f64,
    pub device_us: f64,
}

impl Cuda {
    pub fn new(device: i32) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: native validates the device index and writes one owned context handle.
        check(unsafe { fa_cuda_create(device, &mut ptr) })?;
        Ok(Self {
            session: Rc::new(Session(nonnull(ptr)?)),
            precision_policy: PrecisionPolicy::StrictFp32,
            #[cfg(feature = "cudnn")]
            convolution_policy: ConvolutionPolicy::default(),
        })
    }
    /// Create a CUDA session with an explicit precision policy (`StrictFp32`, `Tf32`, or `Bf16Mixed`).
    /// The strict-FP32 cuBLAS handle (`CUBLAS_PEDANTIC_MATH`) remains uncontaminated in all modes.
    pub fn with_precision(device: i32, policy: PrecisionPolicy) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let code = match policy {
            PrecisionPolicy::StrictFp32 => 0,
            PrecisionPolicy::Tf32 => 1,
            PrecisionPolicy::Bf16Mixed => 2,
        };
        // SAFETY: native validates device and policy code and writes one owned context handle.
        check(unsafe { fa_cuda_create_with_precision(device, code, &mut ptr) })?;
        Ok(Self {
            session: Rc::new(Session(nonnull(ptr)?)),
            precision_policy: policy,
            #[cfg(feature = "cudnn")]
            convolution_policy: ConvolutionPolicy::default(),
        })
    }
    pub fn precision_policy(&self) -> PrecisionPolicy {
        self.precision_policy
    }
    /// Create a session with immutable cuDNN convolution selection policy.
    /// GRU, cuBLAS and FFT policies are unaffected; graph convolutions inherit this.
    #[cfg(feature = "cudnn")]
    pub fn with_convolution_policy(device: i32, policy: ConvolutionPolicy) -> Result<Self> {
        let mut cuda = Self::new(device)?;
        cuda.convolution_policy = policy;
        Ok(cuda)
    }
    #[cfg(feature = "cudnn")]
    pub fn convolution_policy(&self) -> ConvolutionPolicy {
        self.convolution_policy
    }
    pub fn device_name(&self) -> Result<String> {
        let mut buffer = [0 as c_char; 256];
        // SAFETY: buffer has the exact 256 bytes required and native guarantees a trailing NUL.
        check(unsafe { fa_cuda_name(self.session.ptr(), buffer.as_mut_ptr()) })?;
        // SAFETY: native has NUL-terminated the buffer and no pointer escapes.
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: self retains a live context on this host thread.
        check(unsafe { fa_cuda_sync(self.session.ptr()) })
    }
    pub(crate) fn owns(&self, tensor: &DeviceTensor) -> Result<()> {
        if Rc::ptr_eq(&self.session, &tensor.session) {
            Ok(())
        } else {
            Err(invalid("device tensor belongs to another CUDA session"))
        }
    }
    pub fn zeros(&self, dims: impl AsRef<[usize]>) -> Result<DeviceTensor> {
        let shape = Shape::new(dims)?;
        let mut ptr = std::ptr::null_mut();
        let ptr = if shape.is_empty() {
            None
        } else {
            // SAFETY: positive byte count was checked by Shape; the writable handle is valid.
            check(unsafe {
                fa_cuda_alloc(self.session.ptr(), shape.len() * size_of::<f32>(), &mut ptr)
            })?;
            Some(nonnull(ptr)?)
        };
        Ok(DeviceTensor {
            session: self.session.clone(),
            shape,
            ptr,
        })
    }
    pub fn upload(&self, tensor: &Tensor) -> Result<DeviceTensor> {
        let mut out = self.zeros(tensor.shape().dims())?;
        self.copy_from_host(tensor, &mut out)?;
        Ok(out)
    }
    /// Synchronous host transfer; borrowed host memory is never retained by CUDA.
    pub fn copy_from_host(&self, input: &Tensor, out: &mut DeviceTensor) -> Result<()> {
        self.owns(out)?;
        out.shape.expect(input.shape().dims())?;
        if out.shape.is_empty() {
            return Ok(());
        }
        // SAFETY: exact equal shapes give equal byte sizes; native synchronizes before returning.
        check(unsafe {
            fa_cuda_copy(
                self.session.ptr(),
                out.write().cast(),
                input.data().as_ptr().cast(),
                out.shape.len() * size_of::<f32>(),
                1,
            )
        })
    }
    pub fn download(&self, tensor: &DeviceTensor) -> Result<Tensor> {
        self.owns(tensor)?;
        let mut out = Tensor::zeros(tensor.shape.dims())?;
        if tensor.shape.is_empty() {
            return Ok(out);
        }
        // SAFETY: initialized host storage has the exact device byte size; copy synchronizes.
        check(unsafe {
            fa_cuda_copy(
                self.session.ptr(),
                out.data_mut().as_mut_ptr().cast(),
                tensor.read().cast(),
                tensor.shape.len() * size_of::<f32>(),
                0,
            )
        })?;
        Ok(out)
    }
    pub fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor, out: &mut DeviceTensor) -> Result<()> {
        self.owns(a)?;
        self.owns(b)?;
        self.owns(out)?;
        let (m, n, k) = matmul_dims(&a.shape, &b.shape, &out.shape)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            // SAFETY: output is an exclusive positive-size allocation on this stream.
            return check(unsafe {
                fa_cuda_zero(
                    self.session.ptr(),
                    out.write().cast(),
                    out.shape.len() * size_of::<f32>(),
                )
            });
        }
        let (m, n, k) = (int(m)?, int(n)?, int(k)?);
        // SAFETY: dimensions, bounds, session identity, and exclusive output access were checked.
        check(unsafe {
            fa_cuda_matmul(self.session.ptr(), m, n, k, a.read(), b.read(), out.write())
        })
    }
    /// Measures warmed, device-resident work. Both times are per iteration, in microseconds.
    pub fn measure(
        &self,
        iterations: usize,
        mut operation: impl FnMut() -> Result<()>,
    ) -> Result<Timing> {
        if iterations == 0 {
            return Err(invalid("timing iterations must be positive"));
        }
        self.synchronize()?;
        let start = Event::new(self)?;
        let end = Event::new(self)?;
        let wall = Instant::now();
        start.record()?;
        for _ in 0..iterations {
            operation()?;
        }
        end.record()?;
        let mut ms = 0.0;
        // SAFETY: both recorded events belong to this session; native synchronizes the end event.
        check(unsafe {
            fa_cuda_event_elapsed(
                self.session.ptr(),
                start.ptr.as_ptr(),
                end.ptr.as_ptr(),
                &mut ms,
            )
        })?;
        Ok(Timing {
            wall_us: wall.elapsed().as_secs_f64() * 1e6 / iterations as f64,
            device_us: f64::from(ms) * 1000.0 / iterations as f64,
        })
    }
}

struct Event {
    session: Rc<Session>,
    ptr: NonNull<c_void>,
}
impl Event {
    fn new(cuda: &Cuda) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: live session and one writable event handle.
        check(unsafe { fa_cuda_event_create(cuda.session.ptr(), &mut ptr) })?;
        Ok(Self {
            session: cuda.session.clone(),
            ptr: nonnull(ptr)?,
        })
    }
    fn record(&self) -> Result<()> {
        // SAFETY: event and stream belong to the same live session.
        check(unsafe { fa_cuda_event_record(self.session.ptr(), self.ptr.as_ptr()) })
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: event has one owner; CUDA permits destroying a pending event.
        unsafe { fa_cuda_event_destroy(self.session.ptr(), self.ptr.as_ptr()) }
    }
}

pub struct Conv1d {
    cuda: Cuda,
    spec: Conv1dSpec,
    dims: ConvDims,
    scratch: Option<DeviceTensor>,
    activation: Activation,
    #[cfg(feature = "cudnn")]
    cudnn: Option<CudnnPlan>,
    #[cfg(feature = "cudnn")]
    fixed_bias: Option<bool>,
}
impl Conv1d {
    pub fn new(cuda: &Cuda, spec: Conv1dSpec, activation: Activation) -> Result<Self> {
        let dims = spec.native_dims()?;
        let n = dims.output_len as usize;
        let scratch = if spec.pointwise() {
            None
        } else {
            Some(cuda.zeros([spec.batch, spec.in_channels, spec.kernel, n])?)
        };
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            dims,
            scratch,
            activation,
            #[cfg(feature = "cudnn")]
            cudnn: None,
            #[cfg(feature = "cudnn")]
            fixed_bias: None,
        })
    }
    /// Prepare a strict-FP32 cuDNN graph with a bounded reusable workspace.
    /// Bias presence is fixed; values and device buffers can change between calls.
    /// A changed buffer address rebuilds bindings and synchronizes older plan uses.
    #[cfg(feature = "cudnn")]
    pub fn with_cudnn(
        cuda: &Cuda,
        spec: Conv1dSpec,
        bias: bool,
        activation: Activation,
        workspace_limit: usize,
    ) -> Result<Self> {
        let dims = spec.native_dims()?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: validated dimensions and a live retained context; creation owns its
        // descriptors/workspace, checks all statuses, and catches C++ exceptions.
        let status = unsafe {
            fa_cudnn_conv_create(
                cuda.session.ptr(),
                dims,
                i32::from(bias),
                i32::from(activation == Activation::Relu),
                workspace_limit,
                cuda.convolution_policy.native(),
                &mut ptr,
            )
        };
        CudnnPlan::check_preparation(status, cuda.convolution_policy, workspace_limit)?;
        // SAFETY: successful native creation transfers its sole owner to this wrapper.
        let cudnn = Some(unsafe { CudnnPlan::from_raw(cuda, ptr)? });
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            dims,
            scratch: None,
            activation,
            cudnn,
            fixed_bias: Some(bias),
        })
    }
    /// Select the direct pointwise route or a prepared cuDNN temporal/grouped graph.
    #[cfg(feature = "cudnn")]
    pub fn planned(
        cuda: &Cuda,
        spec: Conv1dSpec,
        bias: bool,
        activation: Activation,
        workspace_limit: usize,
    ) -> Result<Self> {
        spec.output_len()?;
        if spec.pointwise() {
            let mut plan = Self::new(cuda, spec, activation)?;
            plan.fixed_bias = Some(bias);
            Ok(plan)
        } else {
            Self::with_cudnn(cuda, spec, bias, activation, workspace_limit)
        }
    }
    pub fn implementation(&self) -> &str {
        #[cfg(feature = "cudnn")]
        if let Some(plan) = &self.cudnn {
            return &plan.implementation;
        }
        if self.spec.pointwise() {
            "cuBLAS pointwise GEMM + CUDA epilogue"
        } else {
            "CUDA im2col + cuBLAS GEMM + CUDA epilogue"
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        #[cfg(feature = "cudnn")]
        if let Some(plan) = &self.cudnn {
            return plan.info.workspace_bytes;
        }
        self.scratch.as_ref().map_or(0, |t| t.shape.len() * 4)
    }
    /// None for the direct cuBLAS path.
    #[cfg(feature = "cudnn")]
    pub fn cudnn_plan_info(&self) -> Option<&CudnnConvolutionInfo> {
        self.cudnn.as_ref().map(|plan| &plan.info)
    }
    pub fn run(
        &mut self,
        x: &DeviceTensor,
        w: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        out: &mut DeviceTensor,
    ) -> Result<()> {
        self.cuda.owns(x)?;
        self.cuda.owns(w)?;
        self.cuda.owns(out)?;
        if let Some(bias) = bias {
            self.cuda.owns(bias)?;
        }
        self.spec.check(
            &x.shape,
            &w.shape,
            bias.map(DeviceTensor::shape),
            &out.shape,
            self.dims.output_len as usize,
        )?;
        #[cfg(feature = "cudnn")]
        if let Some(expected) = self.fixed_bias {
            if bias.is_some() != expected {
                return Err(invalid(
                    "bias presence differs from the prepared convolution",
                ));
            }
        }
        #[cfg(feature = "cudnn")]
        if let Some(plan) = &mut self.cudnn {
            // SAFETY: exact shapes, session identity and exclusive output are validated;
            // native enqueues on the same retained stream and owns workspace/bindings.
            return check(unsafe {
                fa_cudnn_conv_run(
                    plan.ptr.as_ptr(),
                    x.read(),
                    w.read(),
                    bias.map_or(std::ptr::null(), DeviceTensor::read),
                    out.write(),
                )
            });
        }
        let scratch = self
            .scratch
            .as_mut()
            .map_or(std::ptr::null_mut(), DeviceTensor::write);
        // SAFETY: all shapes and sessions match the validated plan. Scratch and output are
        // exclusively borrowed; all subsequent uses are ordered on the same stream.
        check(unsafe {
            fa_cuda_conv(
                self.cuda.session.ptr(),
                self.dims,
                x.read(),
                w.read(),
                bias.map_or(std::ptr::null(), DeviceTensor::read),
                scratch,
                out.write(),
                i32::from(self.activation == Activation::Relu),
            )
        })
    }
}

pub struct Stft {
    cuda: Cuda,
    spec: StftSpec,
    frames: usize,
    ptr: NonNull<c_void>,
}
impl Stft {
    pub fn new(cuda: &Cuda, spec: StftSpec) -> Result<Self> {
        let frames = spec.frames()?;
        Self::with_geometry(cuda, spec, frames, 0, &spec.window())
    }
    pub fn with_window(cuda: &Cuda, spec: StftSpec, window: &Tensor) -> Result<Self> {
        let frames = spec.frames()?;
        Self::with_geometry(cuda, spec, frames, 0, &spec.checked_window(window)?)
    }
    fn with_geometry(
        cuda: &Cuda,
        spec: StftSpec,
        frames: usize,
        padding: usize,
        window: &[f32],
    ) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: sizes fit i32, window length is n, and native copies the window synchronously.
        check(unsafe {
            fa_cuda_stft_create(
                cuda.session.ptr(),
                spec.input_len as i32,
                spec.fft_size as i32,
                spec.hop as i32,
                padding as i32,
                frames as i32,
                (spec.batch * frames) as i32,
                window.as_ptr(),
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            frames,
            ptr: nonnull(ptr)?,
        })
    }
    pub fn run(&mut self, x: &DeviceTensor, out: &mut DeviceTensor) -> Result<()> {
        self.cuda.owns(x)?;
        self.cuda.owns(out)?;
        x.shape.expect(&self.spec.input_shape())?;
        out.shape
            .expect(&[self.spec.batch, self.frames, self.spec.bins(), 2])?;
        // SAFETY: plan owns workspace; all storage has matching sizes and one stream orders uses.
        check(unsafe { fa_cuda_stft_run(self.ptr.as_ptr(), x.read(), out.write()) })
    }
}

/// Reflect-centered forward transform; framing and reflection stay on the device.
pub struct CenteredStft {
    inner: Stft,
    spec: CenteredStftSpec,
}
impl CenteredStft {
    pub fn new(cuda: &Cuda, spec: CenteredStftSpec) -> Result<Self> {
        Ok(Self {
            inner: Stft::with_geometry(
                cuda,
                spec.spec(),
                spec.frames(),
                spec.padding(),
                &spec.spec().window(),
            )?,
            spec,
        })
    }
    pub fn with_window(cuda: &Cuda, spec: CenteredStftSpec, window: &Tensor) -> Result<Self> {
        Ok(Self {
            inner: Stft::with_geometry(
                cuda,
                spec.spec(),
                spec.frames(),
                spec.padding(),
                &spec.spec().checked_window(window)?,
            )?,
            spec,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn run(&mut self, x: &DeviceTensor, out: &mut DeviceTensor) -> Result<()> {
        self.inner.run(x, out)
    }
}

/// Input-preserving centered inverse with deterministic overlap-add on the session stream.
pub struct Istft {
    cuda: Cuda,
    spec: CenteredStftSpec,
    ptr: NonNull<c_void>,
    workspace: usize,
}
impl Istft {
    pub fn new(cuda: &Cuda, spec: CenteredStftSpec) -> Result<Self> {
        Self::with_coefficients(cuda, spec, &spec.spec().window())
    }
    pub fn with_window(cuda: &Cuda, spec: CenteredStftSpec, window: &Tensor) -> Result<Self> {
        Self::with_coefficients(cuda, spec, &spec.spec().checked_window(window)?)
    }
    fn with_coefficients(cuda: &Cuda, spec: CenteredStftSpec, window: &[f32]) -> Result<Self> {
        let s = spec.spec();
        let normalization = spec.normalization(window)?;
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: validated geometry bounds all buffers and i32 fields. Native copies
        // both host slices synchronously, returns one owned plan and its device workspace size.
        check(unsafe {
            fa_cuda_istft_create(
                cuda.session.ptr(),
                s.input_len as i32,
                s.fft_size as i32,
                s.hop as i32,
                spec.padding() as i32,
                spec.frames() as i32,
                (s.batch * spec.frames()) as i32,
                window.as_ptr(),
                normalization.as_ptr(),
                &mut workspace,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            ptr: nonnull(ptr)?,
            workspace,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    /// Device scratch, window, normalization and cuFFT work area; excludes caller buffers.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
    }
    pub fn run(&mut self, spectrum: &DeviceTensor, out: &mut DeviceTensor) -> Result<()> {
        self.cuda.owns(spectrum)?;
        self.cuda.owns(out)?;
        spectrum.shape.expect(&self.spec.output_shape())?;
        out.shape.expect(&self.spec.input_shape())?;
        // SAFETY: validated sizes and matching session. Native copies the read-only spectrum
        // before destructive C2R execution and orders scratch/output writes on this stream.
        check(unsafe { fa_cuda_istft_run(self.ptr.as_ptr(), spectrum.read(), out.write()) })
    }
}
impl Drop for Istft {
    fn drop(&mut self) {
        // SAFETY: one owner retains a live session; native waits before releasing plan buffers.
        unsafe { fa_cuda_istft_destroy(self.ptr.as_ptr()) }
    }
}
impl Drop for Stft {
    fn drop(&mut self) {
        // SAFETY: one owner, live session, native waits before freeing pending FFT workspace.
        unsafe { fa_cuda_stft_destroy(self.ptr.as_ptr()) }
    }
}
