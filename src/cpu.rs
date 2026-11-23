//! CPU execution with caller-owned outputs and plan-owned scratch storage.
pub use crate::adjoint::ViewBackward;
pub use crate::autograd::host::Tape;
pub use crate::bmm::{BatchedMatmulBackward, BatchedMatmulPlan};
pub use crate::conv::{
    Conv1dBackward, Conv1dPlan, Conv2dBackward, Conv2dPlan, ConvTranspose1dBackward,
    ConvTranspose1dPlan, ConvTranspose2dBackward, ConvTranspose2dPlan,
};
pub use crate::conv2d::Conv2d;
pub use crate::copy::CopyPlan;
pub use crate::dropout::{DropoutBackward, DropoutPlan};
pub use crate::embedding::{EmbeddingBackward, EmbeddingPlan};
pub use crate::graph::host::Graph;
pub use crate::gru::Gru;
pub use crate::loss::{CrossEntropyBackward, CrossEntropyPlan};
pub use crate::matmul::{MatmulBackward, MatmulPlan};
pub use crate::norm::{NormBackward, NormPlan};
pub use crate::optim::CpuAdamW as AdamW;
pub use crate::pointwise::{BinaryPlan, UnaryPlan};
pub use crate::pointwise_backward::{BinaryBackward, UnaryBackward};
pub use crate::reduce::{ReduceBackward, ReducePlan};
pub use crate::rope::{RopeBackward, RopePlan};
pub use crate::sdpa::{SdpaBackward, SdpaPlan};
pub use crate::softmax::{SoftmaxBackward, SoftmaxPlan};
pub use crate::stft::{
    CenteredStftBackward, CenteredStftPlan, IstftBackward, IstftPlan, StftBackward, StftPlan,
};
pub use crate::structural::{CatBackward, CatPlan, PadBackward, PadPlan};
use crate::{
    Activation, CenteredStftSpec, Conv1dSpec, Result, StftSpec, Tensor, tensor::matmul_dims,
};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::sync::Arc;

#[cfg(feature = "mkl")]
pub(crate) mod mkl {
    use crate::{Result, error::native};
    use std::{ffi::c_void, ptr::NonNull};
    unsafe extern "C" {
        pub(super) fn fa_mkl_sgemm(
            m: i32,
            n: i32,
            k: i32,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            threads: i32,
        );
        fn fa_mkl_fft_create(out: *mut *mut c_void, n: i32, batch: i32, threads: i32) -> i32;
        fn fa_mkl_fft_run(plan: *mut c_void, input: *mut f32, output: *mut f32) -> i32;
        fn fa_mkl_ifft_create(out: *mut *mut c_void, n: i32, batch: i32, threads: i32) -> i32;
        fn fa_mkl_ifft_run(plan: *mut c_void, input: *mut f32, output: *mut f32) -> i32;
        fn fa_mkl_fft_destroy(plan: *mut c_void);
    }
    pub(crate) struct FftPlan {
        ptr: NonNull<c_void>,
        inverse: bool,
    }
    // SAFETY: a committed oneMKL DFTI descriptor is uniquely owned, has no host-thread
    // affinity, and is only executed or destroyed through exclusive `&mut self` access.
    unsafe impl Send for FftPlan {}
    impl FftPlan {
        pub(crate) fn new(n: i32, batch: i32, threads: i32) -> Result<Self> {
            Self::create(n, batch, threads, false)
        }
        pub(crate) fn inverse(n: i32, batch: i32, threads: i32) -> Result<Self> {
            Self::create(n, batch, threads, true)
        }
        fn create(n: i32, batch: i32, threads: i32, inverse: bool) -> Result<Self> {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: all dimensions were validated; out points to a writable handle.
            let status = unsafe {
                if inverse {
                    fa_mkl_ifft_create(&mut ptr, n, batch, threads)
                } else {
                    fa_mkl_fft_create(&mut ptr, n, batch, threads)
                }
            };
            native("oneMKL DFTI", status.into())?;
            Ok(Self {
                ptr: NonNull::new(ptr).expect("successful DFTI creation returned null"),
                inverse,
            })
        }
        pub(crate) fn run(&mut self, input: &mut [f32], output: &mut [f32]) -> Result<()> {
            // SAFETY: the enclosing spectral plan validates exact input/output sizes and owns the descriptor.
            native(
                "oneMKL DFTI",
                unsafe {
                    if self.inverse {
                        fa_mkl_ifft_run(self.ptr.as_ptr(), input.as_mut_ptr(), output.as_mut_ptr())
                    } else {
                        fa_mkl_fft_run(self.ptr.as_ptr(), input.as_mut_ptr(), output.as_mut_ptr())
                    }
                }
                .into(),
            )
        }
    }
    impl Drop for FftPlan {
        fn drop(&mut self) {
            // SAFETY: this handle was successfully created and has one owner.
            unsafe { fa_mkl_fft_destroy(self.ptr.as_ptr()) }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Cpu {
    /// Scalar GEMM and RustFFT, intended as a portable correctness reference.
    Reference,
    #[cfg(feature = "mkl")]
    Mkl { threads: std::num::NonZeroU16 },
}

impl Cpu {
    #[cfg(feature = "mkl")]
    pub fn mkl(threads: u16) -> Result<Self> {
        Ok(Self::Mkl {
            threads: std::num::NonZeroU16::new(threads)
                .ok_or_else(|| crate::error::invalid("thread count must be positive"))?,
        })
    }
    /// Computes C = A B; A [M,K], B [K,N], C [M,N]. C is overwritten.
    pub fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let (m, n, k) = matmul_dims(a.shape(), b.shape(), out.shape())?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            out.data_mut().fill(0.0);
            return Ok(());
        }
        crate::error::int(m)?;
        crate::error::int(n)?;
        crate::error::int(k)?;
        self.gemm(m, n, k, a.data(), b.data(), out.data_mut());
        Ok(())
    }
    // Only called with validated dimensions, disjoint outputs, and full contiguous slices.
    pub(crate) fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        match self {
            Self::Reference => {
                c.fill(0.0);
                for i in 0..m {
                    for p in 0..k {
                        let alpha = a[i * k + p];
                        for j in 0..n {
                            c[i * n + j] += alpha * b[p * n + j];
                        }
                    }
                }
            }
            #[cfg(feature = "mkl")]
            Self::Mkl { threads } => {
                // SAFETY: dimensions fit i32 and slice bounds were checked by the caller.
                // Shared inputs cannot alias the exclusively borrowed output.
                unsafe {
                    mkl::fa_mkl_sgemm(
                        m as i32,
                        n as i32,
                        k as i32,
                        a.as_ptr(),
                        b.as_ptr(),
                        c.as_mut_ptr(),
                        threads.get().into(),
                    )
                }
            }
        }
    }
}

/// Prepared grouped Conv1d. Scratch is allocated once and reused across calls.
pub struct Conv1d {
    cpu: Cpu,
    spec: Conv1dSpec,
    output_len: usize,
    scratch: Vec<f32>,
    activation: Activation,
}

impl Conv1d {
    pub fn new(cpu: Cpu, spec: Conv1dSpec, activation: Activation) -> Result<Self> {
        let output_len = spec.output_len()?;
        let scratch = if spec.pointwise() {
            Vec::new()
        } else {
            Tensor::zeros([spec.reduction_len(), output_len])?.into_vec()
        };
        Ok(Self {
            cpu,
            spec,
            output_len,
            scratch,
            activation,
        })
    }
    pub fn spec(&self) -> Conv1dSpec {
        self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        self.scratch.len() * size_of::<f32>()
    }
    pub fn run(
        &mut self,
        x: &Tensor,
        weights: &Tensor,
        bias: Option<&Tensor>,
        out: &mut Tensor,
    ) -> Result<()> {
        self.spec.check(
            x.shape(),
            weights.shape(),
            bias.map(Tensor::shape),
            out.shape(),
            self.output_len,
        )?;
        let s = self.spec;
        let k = s.reduction_len();
        let cout = s.out_channels / s.groups;
        let cin = s.in_channels / s.groups;
        let n = self.output_len;
        for batch in 0..s.batch {
            for group in 0..s.groups {
                let input_offset = (batch * s.in_channels + group * cin) * s.input_len;
                let columns = if s.pointwise() {
                    &x.data()[input_offset..input_offset + cin * s.input_len]
                } else {
                    for channel in 0..cin {
                        for kernel in 0..s.kernel {
                            let row = &mut self.scratch[(channel * s.kernel + kernel) * n..][..n];
                            for (t, value) in row.iter_mut().enumerate() {
                                let pos = t * s.stride + kernel * s.dilation;
                                *value = if pos >= s.padding && pos - s.padding < s.input_len {
                                    x.data()[input_offset + channel * s.input_len + pos - s.padding]
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                    &self.scratch
                };
                let weight_offset = group * cout * k;
                let output_offset = (batch * s.out_channels + group * cout) * n;
                self.cpu.gemm(
                    cout,
                    n,
                    k,
                    &weights.data()[weight_offset..weight_offset + cout * k],
                    columns,
                    &mut out.data_mut()[output_offset..output_offset + cout * n],
                );
            }
        }
        if bias.is_some() || self.activation == Activation::Relu {
            for batch in out.data_mut().chunks_exact_mut(s.out_channels * n) {
                for (channel, row) in batch.chunks_exact_mut(n).enumerate() {
                    let b = bias.map_or(0.0, |v| v.data()[channel]);
                    for value in row {
                        *value += b;
                        if self.activation == Activation::Relu && *value < 0.0 {
                            *value = 0.0;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

enum PreparedConvImpl {
    Dense {
        plan: Conv1d,
        weights: Tensor,
        bias: Option<Tensor>,
    },
    #[cfg(all(feature = "onednn", feature = "mkl"))]
    OneDnn(crate::onednn::Conv1d),
}

/// Constant-parameter inference. oneDNN packs temporal/grouped weights when enabled;
/// pointwise Intel convolution retains its direct SGEMM route to avoid layout reorders.
pub struct PreparedConv1d {
    implementation: PreparedConvImpl,
}
impl PreparedConv1d {
    pub fn new(
        cpu: Cpu,
        spec: Conv1dSpec,
        weights: &Tensor,
        bias: Option<&Tensor>,
        activation: Activation,
    ) -> Result<Self> {
        spec.output_len()?;
        weights.shape().expect(&spec.weight_shape())?;
        if let Some(bias) = bias {
            bias.shape().expect(&[spec.out_channels])?;
        }
        #[cfg(feature = "onednn")]
        #[cfg(feature = "mkl")]
        if let Cpu::Mkl { threads } = cpu {
            if !spec.pointwise() {
                return Ok(Self {
                    implementation: PreparedConvImpl::OneDnn(crate::onednn::Conv1d::new(
                        spec,
                        weights,
                        bias,
                        activation,
                        threads.get(),
                    )?),
                });
            }
        }
        Ok(Self {
            implementation: PreparedConvImpl::Dense {
                plan: Conv1d::new(cpu, spec, activation)?,
                weights: weights.clone(),
                bias: bias.cloned(),
            },
        })
    }
    pub fn workspace_bytes(&self) -> usize {
        match &self.implementation {
            PreparedConvImpl::Dense { plan, .. } => plan.workspace_bytes(),
            #[cfg(all(feature = "onednn", feature = "mkl"))]
            PreparedConvImpl::OneDnn(plan) => plan.workspace_bytes(),
        }
    }
    pub fn parameter_bytes(&self) -> usize {
        match &self.implementation {
            PreparedConvImpl::Dense { weights, bias, .. } => {
                (weights.data().len() + bias.as_ref().map_or(0, |b| b.data().len())) * 4
            }
            #[cfg(all(feature = "onednn", feature = "mkl"))]
            PreparedConvImpl::OneDnn(plan) => plan.parameter_bytes(),
        }
    }
    pub fn implementation(&self) -> &str {
        match &self.implementation {
            PreparedConvImpl::Dense { plan, .. } => match plan.cpu {
                Cpu::Reference => "reference GEMM + bias/ReLU",
                #[cfg(feature = "mkl")]
                Cpu::Mkl { .. } => "oneMKL GEMM + bias/ReLU",
            },
            #[cfg(all(feature = "onednn", feature = "mkl"))]
            PreparedConvImpl::OneDnn(plan) => plan.implementation(),
        }
    }
    pub fn run(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        match &mut self.implementation {
            PreparedConvImpl::Dense {
                plan,
                weights,
                bias,
            } => plan.run(x, weights, bias.as_ref(), out),
            #[cfg(all(feature = "onednn", feature = "mkl"))]
            PreparedConvImpl::OneDnn(plan) => plan.run(x, out),
        }
    }
}

enum FftImpl {
    Portable {
        fft: Arc<dyn Fft<f32>>,
        buffer: Vec<Complex32>,
        scratch: Vec<Complex32>,
    },
    #[cfg(feature = "mkl")]
    Mkl(mkl::FftPlan),
}

/// Prepared window, batched framing storage, and FFT descriptor.
pub struct Stft {
    spec: StftSpec,
    count: usize,
    padding: usize,
    window: Vec<f32>,
    frames: Tensor,
    fft: FftImpl,
}

impl Stft {
    pub fn new(cpu: Cpu, spec: StftSpec) -> Result<Self> {
        let count = spec.frames()?;
        Self::with_geometry(cpu, spec, count, 0, spec.window())
    }
    /// Snapshot an explicit length-N window, including square-root Hann model frontends.
    pub fn with_window(cpu: Cpu, spec: StftSpec, window: &Tensor) -> Result<Self> {
        let count = spec.frames()?;
        Self::with_geometry(cpu, spec, count, 0, spec.checked_window(window)?)
    }
    fn with_geometry(
        cpu: Cpu,
        spec: StftSpec,
        count: usize,
        padding: usize,
        window: Vec<f32>,
    ) -> Result<Self> {
        let transforms = spec.batch * count;
        let frames = Tensor::zeros([transforms, spec.fft_size])?;
        let fft = match cpu {
            Cpu::Reference => {
                let fft = FftPlanner::new().plan_fft_forward(spec.fft_size);
                let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
                FftImpl::Portable {
                    fft,
                    scratch,
                    buffer: vec![Complex32::default(); spec.fft_size],
                }
            }
            #[cfg(feature = "mkl")]
            Cpu::Mkl { threads } => FftImpl::Mkl(mkl::FftPlan::new(
                spec.fft_size as i32,
                transforms as i32,
                threads.get().into(),
            )?),
        };
        Ok(Self {
            spec,
            count,
            padding,
            window,
            frames,
            fft,
        })
    }
    pub fn spec(&self) -> StftSpec {
        self.spec
    }
    pub fn run(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        let s = self.spec;
        x.shape().expect(&s.input_shape())?;
        out.shape().expect(&[s.batch, self.count, s.bins(), 2])?;
        let count = self.count;
        for (index, frame) in self
            .frames
            .data_mut()
            .chunks_exact_mut(s.fft_size)
            .enumerate()
        {
            let batch = (index / count) * s.input_len;
            let start = (index % count) * s.hop;
            if self.padding == 0 {
                for ((y, &x), &w) in frame
                    .iter_mut()
                    .zip(&x.data()[batch + start..batch + start + s.fft_size])
                    .zip(&self.window)
                {
                    *y = x * w;
                }
            } else {
                for (t, (y, &w)) in frame.iter_mut().zip(&self.window).enumerate() {
                    let position = start + t;
                    let source = if position < self.padding {
                        self.padding - position
                    } else {
                        let p = position - self.padding;
                        if p < s.input_len {
                            p
                        } else {
                            (s.input_len - 1) - (p - (s.input_len - 1))
                        }
                    };
                    *y = x.data()[batch + source] * w;
                }
            }
        }
        match &mut self.fft {
            FftImpl::Portable {
                fft,
                buffer,
                scratch,
            } => {
                for (input, output) in self
                    .frames
                    .data()
                    .chunks_exact(s.fft_size)
                    .zip(out.data_mut().chunks_exact_mut(s.bins() * 2))
                {
                    for (value, &real) in buffer.iter_mut().zip(input) {
                        *value = Complex32::new(real, 0.0);
                    }
                    fft.process_with_scratch(buffer, scratch);
                    for (pair, z) in output.chunks_exact_mut(2).zip(buffer.iter()) {
                        pair[0] = z.re;
                        pair[1] = z.im;
                    }
                }
            }
            #[cfg(feature = "mkl")]
            FftImpl::Mkl(plan) => plan.run(self.frames.data_mut(), out.data_mut())?,
        }
        Ok(())
    }
}

/// Reflect-centered forward transform with no materialized padded waveform.
pub struct CenteredStft {
    inner: Stft,
    spec: CenteredStftSpec,
}
impl CenteredStft {
    pub fn new(cpu: Cpu, spec: CenteredStftSpec) -> Result<Self> {
        Ok(Self {
            inner: Stft::with_geometry(
                cpu,
                spec.spec(),
                spec.frames(),
                spec.padding(),
                spec.spec().window(),
            )?,
            spec,
        })
    }
    pub fn with_window(cpu: Cpu, spec: CenteredStftSpec, window: &Tensor) -> Result<Self> {
        Ok(Self {
            inner: Stft::with_geometry(
                cpu,
                spec.spec(),
                spec.frames(),
                spec.padding(),
                spec.spec().checked_window(window)?,
            )?,
            spec,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn run(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        self.inner.run(x, out)
    }
}

/// Centered inverse with one-sided Hermitian input and a fixed output length.
/// Synthesis defaults to periodic Hann; explicit windows also determine the NOLA envelope.
/// The input is preserved. DC and even-size Nyquist imaginary components are ignored.
pub struct Istft {
    spec: CenteredStftSpec,
    window: Vec<f32>,
    normalization: Vec<f32>,
    frames: Tensor,
    spectrum: Option<Tensor>,
    fft: FftImpl,
}
impl Istft {
    pub fn new(cpu: Cpu, spec: CenteredStftSpec) -> Result<Self> {
        Self::with_coefficients(cpu, spec, spec.spec().window())
    }
    pub fn with_window(cpu: Cpu, spec: CenteredStftSpec, window: &Tensor) -> Result<Self> {
        Self::with_coefficients(cpu, spec, spec.spec().checked_window(window)?)
    }
    fn with_coefficients(cpu: Cpu, spec: CenteredStftSpec, window: Vec<f32>) -> Result<Self> {
        let s = spec.spec();
        let transforms = s.batch * spec.frames();
        let normalization = spec.normalization(&window)?;
        let (fft, spectrum) = match cpu {
            Cpu::Reference => {
                let fft = FftPlanner::new().plan_fft_inverse(s.fft_size);
                let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
                (
                    FftImpl::Portable {
                        fft,
                        scratch,
                        buffer: vec![Complex32::default(); s.fft_size],
                    },
                    None,
                )
            }
            #[cfg(feature = "mkl")]
            Cpu::Mkl { threads } => (
                FftImpl::Mkl(mkl::FftPlan::inverse(
                    s.fft_size as i32,
                    transforms as i32,
                    threads.get().into(),
                )?),
                Some(Tensor::zeros(spec.output_shape())?),
            ),
        };
        Ok(Self {
            spec,
            window,
            normalization,
            frames: Tensor::zeros([transforms, s.fft_size])?,
            spectrum,
            fft,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    /// Explicit execution buffer bytes; FFT implementation internals are excluded.
    pub fn workspace_bytes(&self) -> usize {
        let fft = match &self.fft {
            FftImpl::Portable {
                buffer, scratch, ..
            } => (buffer.len() + scratch.len()) * size_of::<Complex32>(),
            #[cfg(feature = "mkl")]
            FftImpl::Mkl(_) => 0,
        };
        (self.window.len()
            + self.normalization.len()
            + self.frames.data().len()
            + self.spectrum.as_ref().map_or(0, |x| x.data().len()))
            * size_of::<f32>()
            + fft
    }
    pub fn run(&mut self, spectrum: &Tensor, out: &mut Tensor) -> Result<()> {
        spectrum.shape().expect(&self.spec.output_shape())?;
        out.shape().expect(&self.spec.input_shape())?;
        let s = self.spec.spec();
        match &mut self.fft {
            FftImpl::Portable {
                fft,
                buffer,
                scratch,
            } => {
                for (input, frame) in spectrum
                    .data()
                    .chunks_exact(s.bins() * 2)
                    .zip(self.frames.data_mut().chunks_exact_mut(s.fft_size))
                {
                    for (bin, pair) in input.chunks_exact(2).enumerate() {
                        buffer[bin] = Complex32::new(
                            pair[0],
                            if bin == 0 || (s.fft_size % 2 == 0 && bin == s.fft_size / 2) {
                                0.0
                            } else {
                                pair[1]
                            },
                        );
                    }
                    for bin in s.bins()..s.fft_size {
                        buffer[bin] = buffer[s.fft_size - bin].conj();
                    }
                    fft.process_with_scratch(buffer, scratch);
                    for (y, z) in frame.iter_mut().zip(buffer.iter()) {
                        *y = z.re;
                    }
                }
            }
            #[cfg(feature = "mkl")]
            FftImpl::Mkl(plan) => {
                let input = self
                    .spectrum
                    .as_mut()
                    .expect("MKL inverse owns spectrum scratch");
                input.data_mut().copy_from_slice(spectrum.data());
                for frame in input.data_mut().chunks_exact_mut(s.bins() * 2) {
                    frame[1] = 0.0;
                    if s.fft_size % 2 == 0 {
                        frame[2 * (s.bins() - 1) + 1] = 0.0;
                    }
                }
                plan.run(input.data_mut(), self.frames.data_mut())?;
            }
        }
        out.data_mut().fill(0.0);
        let padding = self.spec.padding();
        for (index, frame) in self.frames.data().chunks_exact(s.fft_size).enumerate() {
            let batch = (index / self.spec.frames()) * s.input_len;
            let start = (index % self.spec.frames()) * s.hop;
            for (t, (&value, &window)) in frame.iter().zip(&self.window).enumerate() {
                let position = start + t;
                if position >= padding && position - padding < s.input_len {
                    out.data_mut()[batch + position - padding] += value * window;
                }
            }
        }
        for batch in out.data_mut().chunks_exact_mut(s.input_len) {
            for (value, &scale) in batch.iter_mut().zip(&self.normalization) {
                *value *= scale;
            }
        }
        Ok(())
    }
}
