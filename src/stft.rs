use crate::{
    CenteredStftSpec, Error, GradientMode, Layout, Result, StftSpec, Tensor, TensorView,
    TensorViewMut, cpu::Cpu, error::invalid,
};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(crate) struct View2 {
    pub origin: isize,
    pub strides: [isize; 2],
    pub contiguous: bool,
}

impl View2 {
    pub(crate) fn from_layout(
        layout: &Layout,
        expected: [usize; 2],
        writable: bool,
    ) -> Result<Self> {
        layout.shape().expect(&expected)?;
        if writable && !layout.is_non_overlapping() {
            return Err(Error::Unsupported(
                "mutable spectral output requires a proved non-overlapping layout".into(),
            ));
        }
        let &[s0, s1] = layout.strides() else {
            return Err(invalid("waveform layout must be rank two"));
        };
        Ok(Self {
            origin: layout.offset() as isize,
            strides: [s0, s1],
            contiguous: layout.is_contiguous(),
        })
    }

    #[inline]
    fn index(self, b: usize, s: usize) -> usize {
        (self.origin + b as isize * self.strides[0] + s as isize * self.strides[1]) as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct View4 {
    pub origin: isize,
    pub strides: [isize; 4],
    #[cfg_attr(not(any(feature = "mkl", feature = "cuda")), allow(dead_code))]
    pub contiguous: bool,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub direct_r2c: bool,
}

impl View4 {
    pub(crate) fn from_layout(
        layout: &Layout,
        expected: [usize; 4],
        writable: bool,
    ) -> Result<Self> {
        layout.shape().expect(&expected)?;
        if writable && !layout.is_non_overlapping() {
            return Err(Error::Unsupported(
                "mutable spectral output requires a proved non-overlapping layout".into(),
            ));
        }
        let &[s0, s1, s2, s3] = layout.strides() else {
            return Err(invalid("spectrum layout must be rank four"));
        };
        let contiguous = layout.is_contiguous();
        let origin = layout.offset() as isize;
        Ok(Self {
            origin,
            strides: [s0, s1, s2, s3],
            contiguous,
            // Direct cuFFT R2C output requires contiguous memory AND 8-byte alignment (even f32 offset).
            direct_r2c: contiguous && (origin % 2 == 0),
        })
    }

    #[inline]
    fn index(self, b: usize, f: usize, k: usize, c: usize) -> usize {
        (self.origin
            + b as isize * self.strides[0]
            + f as isize * self.strides[1]
            + k as isize * self.strides[2]
            + c as isize * self.strides[3]) as usize
    }
}

pub(crate) fn resolve_window(spec: StftSpec, window: Option<&[f32]>) -> Result<Vec<f32>> {
    match window {
        Some(w) => spec.checked_window_slice(w),
        None => Ok(spec.window()),
    }
}

enum CpuFft {
    Portable {
        fft: Arc<dyn Fft<f32>>,
        buffer: Vec<Complex32>,
        scratch: Vec<Complex32>,
    },
    #[cfg(feature = "mkl")]
    Mkl(crate::cpu::mkl::FftPlan),
}

impl CpuFft {
    fn forward(
        cpu: Cpu,
        fft_size: usize,
        #[cfg_attr(not(feature = "mkl"), allow(unused_variables))] transforms: usize,
    ) -> Result<Self> {
        Ok(match cpu {
            Cpu::Reference => {
                let fft = FftPlanner::new().plan_fft_forward(fft_size);
                let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
                Self::Portable {
                    fft,
                    buffer: vec![Complex32::default(); fft_size],
                    scratch,
                }
            }
            #[cfg(feature = "mkl")]
            Cpu::Mkl { threads } => Self::Mkl(crate::cpu::mkl::FftPlan::new(
                fft_size as i32,
                transforms as i32,
                threads.get().into(),
            )?),
        })
    }

    fn inverse(
        cpu: Cpu,
        fft_size: usize,
        #[cfg_attr(not(feature = "mkl"), allow(unused_variables))] transforms: usize,
    ) -> Result<Self> {
        Ok(match cpu {
            Cpu::Reference => {
                let fft = FftPlanner::new().plan_fft_inverse(fft_size);
                let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
                Self::Portable {
                    fft,
                    buffer: vec![Complex32::default(); fft_size],
                    scratch,
                }
            }
            #[cfg(feature = "mkl")]
            Cpu::Mkl { threads } => Self::Mkl(crate::cpu::mkl::FftPlan::inverse(
                fft_size as i32,
                transforms as i32,
                threads.get().into(),
            )?),
        })
    }

    fn workspace_bytes(&self) -> usize {
        match self {
            Self::Portable {
                buffer, scratch, ..
            } => (buffer.len() + scratch.len()) * size_of::<Complex32>(),
            #[cfg(feature = "mkl")]
            Self::Mkl(_) => 0,
        }
    }
}

#[inline]
fn gather_framed_pos(
    framed: &[f32],
    window: &[f32],
    b: usize,
    pos: usize,
    frames: usize,
    n: usize,
    hop: usize,
) -> f32 {
    let first = if pos >= n { (pos - n) / hop + 1 } else { 0 };
    let last = (pos / hop).min(frames - 1);
    let mut acc = 0.0f32;
    let batch_base = b * frames;
    for f in first..=last {
        let t = pos - f * hop;
        acc += framed[(batch_base + f) * n + t] * window[t];
    }
    acc
}

// ============================================================================
// Internal shared CPU STFT forward / backward cores
// ============================================================================

struct StftForwardCore {
    spec: StftSpec,
    frames_count: usize,
    padding: usize,
    input_layout: Layout,
    output_layout: Layout,
    x_view: View2,
    y_view: View4,
    window: Vec<f32>,
    framed: Vec<f32>,
    spectrum: Option<Vec<f32>>,
    fft: CpuFft,
}

impl StftForwardCore {
    fn new(
        cpu: Cpu,
        spec: StftSpec,
        frames_count: usize,
        padding: usize,
        window: Vec<f32>,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let x_view = View2::from_layout(input, spec.input_shape(), false)?;
        let y_view = View4::from_layout(output, [spec.batch, frames_count, spec.bins(), 2], true)?;
        let transforms = spec.batch * frames_count;
        let fft = CpuFft::forward(cpu, spec.fft_size, transforms)?;
        #[cfg(feature = "mkl")]
        let need_spectrum = matches!(fft, CpuFft::Mkl(_)) && !y_view.contiguous;
        #[cfg(not(feature = "mkl"))]
        let need_spectrum = false;
        let spectrum = need_spectrum.then(|| vec![0.0f32; transforms * spec.bins() * 2]);
        Ok(Self {
            spec,
            frames_count,
            padding,
            input_layout: input.clone(),
            output_layout: output.clone(),
            x_view,
            y_view,
            window,
            framed: vec![0.0f32; transforms * spec.fft_size],
            spectrum,
            fft,
        })
    }

    fn workspace_bytes(&self) -> usize {
        (self.window.len() + self.framed.len() + self.spectrum.as_ref().map_or(0, Vec::len))
            * size_of::<f32>()
            + self.fft.workspace_bytes()
    }

    fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }

    fn run(&mut self, input: &TensorView<'_>, output: &mut TensorViewMut<'_>) -> Result<()> {
        if input.layout() != &self.input_layout {
            return Err(invalid("STFT input layout differs from prepared plan"));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid("STFT output layout differs from prepared plan"));
        }
        let s = self.spec;
        let n = s.fft_size;
        let hop = s.hop;
        let input_len = s.input_len;
        let frames_count = self.frames_count;
        let padding = self.padding;
        let x_data = input.data;
        let xv = self.x_view;

        for (index, frame) in self.framed.chunks_exact_mut(n).enumerate() {
            let b = index / frames_count;
            let f = index % frames_count;
            let start = f * hop;
            if padding == 0 {
                if xv.contiguous {
                    let base = xv.origin as usize + b * input_len + start;
                    for ((dst, &src), &w) in frame
                        .iter_mut()
                        .zip(&x_data[base..base + n])
                        .zip(&self.window)
                    {
                        *dst = src * w;
                    }
                } else {
                    for (t, (dst, &w)) in frame.iter_mut().zip(&self.window).enumerate() {
                        *dst = x_data[xv.index(b, start + t)] * w;
                    }
                }
            } else {
                for (t, (dst, &w)) in frame.iter_mut().zip(&self.window).enumerate() {
                    let pos = start + t;
                    let src_idx = if pos < padding {
                        padding - pos
                    } else {
                        let q = pos - padding;
                        if q < input_len {
                            q
                        } else {
                            (input_len - 1) - (q - (input_len - 1))
                        }
                    };
                    let val = if xv.contiguous {
                        x_data[xv.origin as usize + b * input_len + src_idx]
                    } else {
                        x_data[xv.index(b, src_idx)]
                    };
                    *dst = val * w;
                }
            }
        }

        let bins = s.bins();
        let yv = self.y_view;
        match &mut self.fft {
            CpuFft::Portable {
                fft,
                buffer,
                scratch,
            } => {
                if let Some(out_slice) = output.as_slice_mut() {
                    for (frame, out_frame) in self
                        .framed
                        .chunks_exact(n)
                        .zip(out_slice.chunks_exact_mut(bins * 2))
                    {
                        for (c, &real) in buffer.iter_mut().zip(frame) {
                            *c = Complex32::new(real, 0.0);
                        }
                        fft.process_with_scratch(buffer, scratch);
                        for (pair, z) in out_frame.chunks_exact_mut(2).zip(buffer.iter()) {
                            pair[0] = z.re;
                            pair[1] = z.im;
                        }
                    }
                } else {
                    let out_data = &mut *output.data;
                    for (index, frame) in self.framed.chunks_exact(n).enumerate() {
                        let b = index / frames_count;
                        let f = index % frames_count;
                        for (c, &real) in buffer.iter_mut().zip(frame) {
                            *c = Complex32::new(real, 0.0);
                        }
                        fft.process_with_scratch(buffer, scratch);
                        for (k, z) in buffer.iter().take(bins).enumerate() {
                            out_data[yv.index(b, f, k, 0)] = z.re;
                            out_data[yv.index(b, f, k, 1)] = z.im;
                        }
                    }
                }
            }
            #[cfg(feature = "mkl")]
            CpuFft::Mkl(plan) => {
                if let Some(out_slice) = output.as_slice_mut() {
                    plan.run(&mut self.framed, out_slice)?;
                } else {
                    let spec_buf = self
                        .spectrum
                        .as_mut()
                        .expect("MKL non-contiguous STFT owns spectrum scratch");
                    plan.run(&mut self.framed, spec_buf)?;
                    let out_data = &mut *output.data;
                    for b in 0..s.batch {
                        for f in 0..frames_count {
                            let base = (b * frames_count + f) * bins * 2;
                            for k in 0..bins {
                                out_data[yv.index(b, f, k, 0)] = spec_buf[base + 2 * k];
                                out_data[yv.index(b, f, k, 1)] = spec_buf[base + 2 * k + 1];
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

struct StftBackwardCore {
    spec: StftSpec,
    frames_count: usize,
    padding: usize,
    upstream_layout: Layout,
    output_layout: Layout,
    dy_view: View4,
    dx_view: View2,
    window: Vec<f32>,
    framed: Vec<f32>,
    #[cfg_attr(not(feature = "mkl"), allow(dead_code))]
    spectrum: Option<Vec<f32>>,
    fft: CpuFft,
}

impl StftBackwardCore {
    fn new(
        cpu: Cpu,
        spec: StftSpec,
        frames_count: usize,
        padding: usize,
        window: Vec<f32>,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let dy_view =
            View4::from_layout(upstream, [spec.batch, frames_count, spec.bins(), 2], false)?;
        let dx_view = View2::from_layout(output, spec.input_shape(), true)?;
        let transforms = spec.batch * frames_count;
        let fft = CpuFft::inverse(cpu, spec.fft_size, transforms)?;
        #[cfg(feature = "mkl")]
        let spectrum =
            matches!(fft, CpuFft::Mkl(_)).then(|| vec![0.0f32; transforms * spec.bins() * 2]);
        #[cfg(not(feature = "mkl"))]
        let spectrum = None;
        Ok(Self {
            spec,
            frames_count,
            padding,
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            dy_view,
            dx_view,
            window,
            framed: vec![0.0f32; transforms * spec.fft_size],
            spectrum,
            fft,
        })
    }

    fn workspace_bytes(&self) -> usize {
        (self.window.len() + self.framed.len() + self.spectrum.as_ref().map_or(0, Vec::len))
            * size_of::<f32>()
            + self.fft.workspace_bytes()
    }

    fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }

    fn run(
        &mut self,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        if upstream.layout() != &self.upstream_layout {
            return Err(invalid(
                "STFT backward upstream layout differs from prepared plan",
            ));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "STFT backward output layout differs from prepared plan",
            ));
        }
        let s = self.spec;
        let n = s.fft_size;
        let bins = s.bins();
        let even_n = n.is_multiple_of(2);
        let frames_count = self.frames_count;
        let dy_data = upstream.data;
        let dyv = self.dy_view;

        match &mut self.fft {
            CpuFft::Portable {
                fft,
                buffer,
                scratch,
            } => {
                if let Some(dy_slice) = upstream.as_slice() {
                    for (in_frame, out_frame) in dy_slice
                        .chunks_exact(bins * 2)
                        .zip(self.framed.chunks_exact_mut(n))
                    {
                        for (k, pair) in in_frame.chunks_exact(2).enumerate() {
                            if k == 0 || (even_n && k == n / 2) {
                                buffer[k] = Complex32::new(pair[0], 0.0);
                            } else {
                                buffer[k] = Complex32::new(0.5 * pair[0], 0.5 * pair[1]);
                            }
                        }
                        for k in bins..n {
                            buffer[k] = buffer[n - k].conj();
                        }
                        fft.process_with_scratch(buffer, scratch);
                        for (dst, z) in out_frame.iter_mut().zip(buffer.iter()) {
                            *dst = z.re;
                        }
                    }
                } else {
                    for (index, out_frame) in self.framed.chunks_exact_mut(n).enumerate() {
                        let b = index / frames_count;
                        let f = index % frames_count;
                        for k in 0..bins {
                            let re = dy_data[dyv.index(b, f, k, 0)];
                            let im = dy_data[dyv.index(b, f, k, 1)];
                            if k == 0 || (even_n && k == n / 2) {
                                buffer[k] = Complex32::new(re, 0.0);
                            } else {
                                buffer[k] = Complex32::new(0.5 * re, 0.5 * im);
                            }
                        }
                        for k in bins..n {
                            buffer[k] = buffer[n - k].conj();
                        }
                        fft.process_with_scratch(buffer, scratch);
                        for (dst, z) in out_frame.iter_mut().zip(buffer.iter()) {
                            *dst = z.re;
                        }
                    }
                }
            }
            #[cfg(feature = "mkl")]
            CpuFft::Mkl(plan) => {
                let spec_buf = self
                    .spectrum
                    .as_mut()
                    .expect("MKL STFT backward owns spectrum scratch");
                if let Some(dy_slice) = upstream.as_slice() {
                    for (in_frame, dst_frame) in dy_slice
                        .chunks_exact(bins * 2)
                        .zip(spec_buf.chunks_exact_mut(bins * 2))
                    {
                        for k in 0..bins {
                            let re = in_frame[2 * k];
                            let im = in_frame[2 * k + 1];
                            if k == 0 || (even_n && k == n / 2) {
                                dst_frame[2 * k] = re;
                                dst_frame[2 * k + 1] = 0.0;
                            } else {
                                dst_frame[2 * k] = 0.5 * re;
                                dst_frame[2 * k + 1] = 0.5 * im;
                            }
                        }
                    }
                } else {
                    for b in 0..s.batch {
                        for f in 0..frames_count {
                            let base = (b * frames_count + f) * bins * 2;
                            for k in 0..bins {
                                let re = dy_data[dyv.index(b, f, k, 0)];
                                let im = dy_data[dyv.index(b, f, k, 1)];
                                if k == 0 || (even_n && k == n / 2) {
                                    spec_buf[base + 2 * k] = re;
                                    spec_buf[base + 2 * k + 1] = 0.0;
                                } else {
                                    spec_buf[base + 2 * k] = 0.5 * re;
                                    spec_buf[base + 2 * k + 1] = 0.5 * im;
                                }
                            }
                        }
                    }
                }
                plan.run(spec_buf, &mut self.framed)?;
            }
        }

        // Deterministic per-sample gather matching CUDA's stft_bwd_overlap_gather order.
        let input_len = s.input_len;
        let hop = s.hop;
        let padding = self.padding;
        let dxv = self.dx_view;
        let accumulate = mode == GradientMode::Accumulate;
        let out_data = &mut *output.data;
        for b in 0..s.batch {
            for sample in 0..input_len {
                let mut acc = gather_framed_pos(
                    &self.framed,
                    &self.window,
                    b,
                    sample + padding,
                    frames_count,
                    n,
                    hop,
                );
                if padding > 0 {
                    if sample >= 1 && sample <= padding {
                        acc += gather_framed_pos(
                            &self.framed,
                            &self.window,
                            b,
                            padding - sample,
                            frames_count,
                            n,
                            hop,
                        );
                    }
                    if sample >= input_len - 1 - padding && sample + 2 <= input_len {
                        acc += gather_framed_pos(
                            &self.framed,
                            &self.window,
                            b,
                            padding + 2 * (input_len - 1) - sample,
                            frames_count,
                            n,
                            hop,
                        );
                    }
                }
                let dst = if dxv.contiguous {
                    dxv.origin as usize + b * input_len + sample
                } else {
                    dxv.index(b, sample)
                };
                if accumulate {
                    out_data[dst] += acc;
                } else {
                    out_data[dst] = acc;
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Public CPU Plans: StftPlan, StftBackward, CenteredStftPlan, CenteredStftBackward
// ============================================================================

/// Prepared strided uncentered STFT forward plan on CPU.
pub struct StftPlan {
    core: StftForwardCore,
}

impl StftPlan {
    pub fn new(cpu: Cpu, spec: StftSpec, input: &Layout, output: &Layout) -> Result<Self> {
        Self::with_window_slice(cpu, spec, None, input, output)
    }
    pub fn with_window(
        cpu: Cpu,
        spec: StftSpec,
        window: &Tensor,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.fft_size])?;
        Self::with_window_slice(cpu, spec, Some(window.data()), input, output)
    }
    pub(crate) fn with_window_slice(
        cpu: Cpu,
        spec: StftSpec,
        window: Option<&[f32]>,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let frames = spec.frames()?;
        let coeffs = resolve_window(spec, window)?;
        Ok(Self {
            core: StftForwardCore::new(cpu, spec, frames, 0, coeffs, input, output)?,
        })
    }
    pub fn spec(&self) -> StftSpec {
        self.core.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.core.input_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.core.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace_bytes()
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.core.host_metadata_bytes()
    }
    pub fn run(&mut self, input: &TensorView<'_>, output: &mut TensorViewMut<'_>) -> Result<()> {
        self.core.run(input, output)
    }
}

/// Prepared strided uncentered STFT local VJP on CPU (`input_required() == false`).
pub struct StftBackward {
    core: StftBackwardCore,
}

impl StftBackward {
    pub fn new(cpu: Cpu, spec: StftSpec, upstream: &Layout, output: &Layout) -> Result<Self> {
        Self::with_window_slice(cpu, spec, None, upstream, output)
    }
    pub fn with_window(
        cpu: Cpu,
        spec: StftSpec,
        window: &Tensor,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.fft_size])?;
        Self::with_window_slice(cpu, spec, Some(window.data()), upstream, output)
    }
    pub(crate) fn with_window_slice(
        cpu: Cpu,
        spec: StftSpec,
        window: Option<&[f32]>,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let frames = spec.frames()?;
        let coeffs = resolve_window(spec, window)?;
        Ok(Self {
            core: StftBackwardCore::new(cpu, spec, frames, 0, coeffs, upstream, output)?,
        })
    }
    pub fn spec(&self) -> StftSpec {
        self.core.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.core.upstream_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.core.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace_bytes()
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.core.host_metadata_bytes()
    }
    pub fn run(
        &mut self,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.run(upstream, output, mode)
    }
}

/// Prepared strided reflect-centered STFT forward plan on CPU.
pub struct CenteredStftPlan {
    spec: CenteredStftSpec,
    core: StftForwardCore,
}

impl CenteredStftPlan {
    pub fn new(cpu: Cpu, spec: CenteredStftSpec, input: &Layout, output: &Layout) -> Result<Self> {
        Self::with_window_slice(cpu, spec, None, input, output)
    }
    pub fn with_window(
        cpu: Cpu,
        spec: CenteredStftSpec,
        window: &Tensor,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.spec().fft_size])?;
        Self::with_window_slice(cpu, spec, Some(window.data()), input, output)
    }
    pub(crate) fn with_window_slice(
        cpu: Cpu,
        spec: CenteredStftSpec,
        window: Option<&[f32]>,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let coeffs = resolve_window(spec.spec(), window)?;
        Ok(Self {
            spec,
            core: StftForwardCore::new(
                cpu,
                spec.spec(),
                spec.frames(),
                spec.padding(),
                coeffs,
                input,
                output,
            )?,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.core.input_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.core.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace_bytes()
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.core.host_metadata_bytes() + size_of::<CenteredStftSpec>()
    }
    pub fn run(&mut self, input: &TensorView<'_>, output: &mut TensorViewMut<'_>) -> Result<()> {
        self.core.run(input, output)
    }
}

/// Prepared strided reflect-centered STFT local VJP on CPU (`input_required() == false`).
pub struct CenteredStftBackward {
    spec: CenteredStftSpec,
    core: StftBackwardCore,
}

impl CenteredStftBackward {
    pub fn new(
        cpu: Cpu,
        spec: CenteredStftSpec,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        Self::with_window_slice(cpu, spec, None, upstream, output)
    }
    pub fn with_window(
        cpu: Cpu,
        spec: CenteredStftSpec,
        window: &Tensor,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.spec().fft_size])?;
        Self::with_window_slice(cpu, spec, Some(window.data()), upstream, output)
    }
    pub(crate) fn with_window_slice(
        cpu: Cpu,
        spec: CenteredStftSpec,
        window: Option<&[f32]>,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let coeffs = resolve_window(spec.spec(), window)?;
        Ok(Self {
            spec,
            core: StftBackwardCore::new(
                cpu,
                spec.spec(),
                spec.frames(),
                spec.padding(),
                coeffs,
                upstream,
                output,
            )?,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.core.upstream_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.core.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace_bytes()
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.core.host_metadata_bytes() + size_of::<CenteredStftSpec>()
    }
    pub fn run(
        &mut self,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.run(upstream, output, mode)
    }
}

// ============================================================================
// Public CPU Plans: IstftPlan and IstftBackward
// ============================================================================

/// Prepared strided centered inverse STFT forward plan on CPU.
pub struct IstftPlan {
    spec: CenteredStftSpec,
    input_layout: Layout,
    output_layout: Layout,
    s_view: View4,
    y_view: View2,
    window: Vec<f32>,
    normalization: Vec<f32>,
    framed: Vec<f32>,
    #[cfg_attr(not(feature = "mkl"), allow(dead_code))]
    spectrum: Option<Vec<f32>>,
    fft: CpuFft,
}

impl IstftPlan {
    pub fn new(cpu: Cpu, spec: CenteredStftSpec, input: &Layout, output: &Layout) -> Result<Self> {
        Self::with_window_slice(cpu, spec, None, input, output)
    }
    pub fn with_window(
        cpu: Cpu,
        spec: CenteredStftSpec,
        window: &Tensor,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.spec().fft_size])?;
        Self::with_window_slice(cpu, spec, Some(window.data()), input, output)
    }
    pub(crate) fn with_window_slice(
        cpu: Cpu,
        spec: CenteredStftSpec,
        window: Option<&[f32]>,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let s = spec.spec();
        let s_view = View4::from_layout(input, spec.output_shape(), false)?;
        let y_view = View2::from_layout(output, spec.input_shape(), true)?;
        let coeffs = resolve_window(s, window)?;
        let normalization = spec.normalization(&coeffs)?;
        let transforms = s.batch * spec.frames();
        let fft = CpuFft::inverse(cpu, s.fft_size, transforms)?;
        #[cfg(feature = "mkl")]
        let spectrum =
            matches!(fft, CpuFft::Mkl(_)).then(|| vec![0.0f32; transforms * s.bins() * 2]);
        #[cfg(not(feature = "mkl"))]
        let spectrum = None;
        Ok(Self {
            spec,
            input_layout: input.clone(),
            output_layout: output.clone(),
            s_view,
            y_view,
            window: coeffs,
            normalization,
            framed: vec![0.0f32; transforms * s.fft_size],
            spectrum,
            fft,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.input_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        (self.window.len()
            + self.normalization.len()
            + self.framed.len()
            + self.spectrum.as_ref().map_or(0, Vec::len))
            * size_of::<f32>()
            + self.fft.workspace_bytes()
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn run(&mut self, input: &TensorView<'_>, output: &mut TensorViewMut<'_>) -> Result<()> {
        if input.layout() != &self.input_layout {
            return Err(invalid("iSTFT input layout differs from prepared plan"));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid("iSTFT output layout differs from prepared plan"));
        }
        let s = self.spec.spec();
        let n = s.fft_size;
        let bins = s.bins();
        let even_n = n.is_multiple_of(2);
        let frames_count = self.spec.frames();
        let in_data = input.data;
        let sv = self.s_view;

        match &mut self.fft {
            CpuFft::Portable {
                fft,
                buffer,
                scratch,
            } => {
                if let Some(in_slice) = input.as_slice() {
                    for (in_frame, out_frame) in in_slice
                        .chunks_exact(bins * 2)
                        .zip(self.framed.chunks_exact_mut(n))
                    {
                        for (k, pair) in in_frame.chunks_exact(2).enumerate() {
                            buffer[k] = Complex32::new(
                                pair[0],
                                if k == 0 || (even_n && k == n / 2) {
                                    0.0
                                } else {
                                    pair[1]
                                },
                            );
                        }
                        for k in bins..n {
                            buffer[k] = buffer[n - k].conj();
                        }
                        fft.process_with_scratch(buffer, scratch);
                        for (dst, z) in out_frame.iter_mut().zip(buffer.iter()) {
                            *dst = z.re;
                        }
                    }
                } else {
                    for (index, out_frame) in self.framed.chunks_exact_mut(n).enumerate() {
                        let b = index / frames_count;
                        let f = index % frames_count;
                        for k in 0..bins {
                            let re = in_data[sv.index(b, f, k, 0)];
                            let im = in_data[sv.index(b, f, k, 1)];
                            buffer[k] = Complex32::new(
                                re,
                                if k == 0 || (even_n && k == n / 2) {
                                    0.0
                                } else {
                                    im
                                },
                            );
                        }
                        for k in bins..n {
                            buffer[k] = buffer[n - k].conj();
                        }
                        fft.process_with_scratch(buffer, scratch);
                        for (dst, z) in out_frame.iter_mut().zip(buffer.iter()) {
                            *dst = z.re;
                        }
                    }
                }
            }
            #[cfg(feature = "mkl")]
            CpuFft::Mkl(plan) => {
                let spec_buf = self
                    .spectrum
                    .as_mut()
                    .expect("MKL iSTFT owns spectrum scratch");
                if let Some(in_slice) = input.as_slice() {
                    spec_buf.copy_from_slice(in_slice);
                    for frame in spec_buf.chunks_exact_mut(bins * 2) {
                        frame[1] = 0.0;
                        if even_n {
                            frame[2 * (bins - 1) + 1] = 0.0;
                        }
                    }
                } else {
                    for b in 0..s.batch {
                        for f in 0..frames_count {
                            let base = (b * frames_count + f) * bins * 2;
                            for k in 0..bins {
                                spec_buf[base + 2 * k] = in_data[sv.index(b, f, k, 0)];
                                spec_buf[base + 2 * k + 1] = if k == 0 || (even_n && k == n / 2) {
                                    0.0
                                } else {
                                    in_data[sv.index(b, f, k, 1)]
                                };
                            }
                        }
                    }
                }
                plan.run(spec_buf, &mut self.framed)?;
            }
        }

        let padding = self.spec.padding();
        let output_len = s.input_len;
        let hop = s.hop;
        let yv = self.y_view;
        let out_data = &mut *output.data;
        for b in 0..s.batch {
            for sample in 0..output_len {
                let val = gather_framed_pos(
                    &self.framed,
                    &self.window,
                    b,
                    sample + padding,
                    frames_count,
                    n,
                    hop,
                ) * self.normalization[sample];
                let dst = if yv.contiguous {
                    yv.origin as usize + b * output_len + sample
                } else {
                    yv.index(b, sample)
                };
                out_data[dst] = val;
            }
        }
        Ok(())
    }
}

/// Prepared strided centered inverse STFT local VJP on CPU (`input_required() == false`).
pub struct IstftBackward {
    spec: CenteredStftSpec,
    upstream_layout: Layout,
    output_layout: Layout,
    dy_view: View2,
    ds_view: View4,
    window: Vec<f32>,
    normalization: Vec<f32>,
    framed: Vec<f32>,
    #[cfg_attr(not(feature = "mkl"), allow(dead_code))]
    spectrum: Option<Vec<f32>>,
    fft: CpuFft,
}

impl IstftBackward {
    pub fn new(
        cpu: Cpu,
        spec: CenteredStftSpec,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        Self::with_window_slice(cpu, spec, None, upstream, output)
    }
    pub fn with_window(
        cpu: Cpu,
        spec: CenteredStftSpec,
        window: &Tensor,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.spec().fft_size])?;
        Self::with_window_slice(cpu, spec, Some(window.data()), upstream, output)
    }
    pub(crate) fn with_window_slice(
        cpu: Cpu,
        spec: CenteredStftSpec,
        window: Option<&[f32]>,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let s = spec.spec();
        let dy_view = View2::from_layout(upstream, spec.input_shape(), false)?;
        let ds_view = View4::from_layout(output, spec.output_shape(), true)?;
        let coeffs = resolve_window(s, window)?;
        let normalization = spec.normalization(&coeffs)?;
        let transforms = s.batch * spec.frames();
        let fft = CpuFft::forward(cpu, s.fft_size, transforms)?;
        #[cfg(feature = "mkl")]
        let spectrum =
            matches!(fft, CpuFft::Mkl(_)).then(|| vec![0.0f32; transforms * s.bins() * 2]);
        #[cfg(not(feature = "mkl"))]
        let spectrum = None;
        Ok(Self {
            spec,
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            dy_view,
            ds_view,
            window: coeffs,
            normalization,
            framed: vec![0.0f32; transforms * s.fft_size],
            spectrum,
            fft,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.upstream_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        (self.window.len()
            + self.normalization.len()
            + self.framed.len()
            + self.spectrum.as_ref().map_or(0, Vec::len))
            * size_of::<f32>()
            + self.fft.workspace_bytes()
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn run(
        &mut self,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        if upstream.layout() != &self.upstream_layout {
            return Err(invalid(
                "iSTFT backward upstream layout differs from prepared plan",
            ));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "iSTFT backward output layout differs from prepared plan",
            ));
        }
        let s = self.spec.spec();
        let n = s.fft_size;
        let bins = s.bins();
        let even_n = n.is_multiple_of(2);
        let frames_count = self.spec.frames();
        let padding = self.spec.padding();
        let output_len = s.input_len;
        let hop = s.hop;
        let dy_data = upstream.data;
        let dyv = self.dy_view;

        for (index, frame) in self.framed.chunks_exact_mut(n).enumerate() {
            let b = index / frames_count;
            let f = index % frames_count;
            let start = f * hop;
            for (t, (dst, &w)) in frame.iter_mut().zip(&self.window).enumerate() {
                let pos = start + t;
                if pos >= padding && pos - padding < output_len {
                    let sample = pos - padding;
                    let g = if dyv.contiguous {
                        dy_data[dyv.origin as usize + b * output_len + sample]
                    } else {
                        dy_data[dyv.index(b, sample)]
                    };
                    *dst = g * self.normalization[sample] * w;
                } else {
                    *dst = 0.0;
                }
            }
        }

        let dsv = self.ds_view;
        let accumulate = mode == GradientMode::Accumulate;
        match &mut self.fft {
            CpuFft::Portable {
                fft,
                buffer,
                scratch,
            } => {
                if let Some(out_slice) = output.as_slice_mut() {
                    for (frame, out_frame) in self
                        .framed
                        .chunks_exact(n)
                        .zip(out_slice.chunks_exact_mut(bins * 2))
                    {
                        for (c, &real) in buffer.iter_mut().zip(frame) {
                            *c = Complex32::new(real, 0.0);
                        }
                        fft.process_with_scratch(buffer, scratch);
                        for (k, (pair, z)) in
                            out_frame.chunks_exact_mut(2).zip(buffer.iter()).enumerate()
                        {
                            let endpoint = k == 0 || (even_n && k == n / 2);
                            let re = if endpoint { z.re } else { 2.0 * z.re };
                            let im = if endpoint { 0.0 } else { 2.0 * z.im };
                            if accumulate {
                                pair[0] += re;
                                pair[1] += im;
                            } else {
                                pair[0] = re;
                                pair[1] = im;
                            }
                        }
                    }
                } else {
                    let out_data = &mut *output.data;
                    for (index, frame) in self.framed.chunks_exact(n).enumerate() {
                        let b = index / frames_count;
                        let f = index % frames_count;
                        for (c, &real) in buffer.iter_mut().zip(frame) {
                            *c = Complex32::new(real, 0.0);
                        }
                        fft.process_with_scratch(buffer, scratch);
                        for (k, z) in buffer.iter().take(bins).enumerate() {
                            let endpoint = k == 0 || (even_n && k == n / 2);
                            let re = if endpoint { z.re } else { 2.0 * z.re };
                            let im = if endpoint { 0.0 } else { 2.0 * z.im };
                            let idx_re = dsv.index(b, f, k, 0);
                            let idx_im = dsv.index(b, f, k, 1);
                            if accumulate {
                                out_data[idx_re] += re;
                                out_data[idx_im] += im;
                            } else {
                                out_data[idx_re] = re;
                                out_data[idx_im] = im;
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "mkl")]
            CpuFft::Mkl(plan) => {
                let spec_buf = self
                    .spectrum
                    .as_mut()
                    .expect("MKL iSTFT backward owns spectrum scratch");
                plan.run(&mut self.framed, spec_buf)?;
                if let Some(out_slice) = output.as_slice_mut() {
                    for (src_frame, out_frame) in spec_buf
                        .chunks_exact(bins * 2)
                        .zip(out_slice.chunks_exact_mut(bins * 2))
                    {
                        for k in 0..bins {
                            let endpoint = k == 0 || (even_n && k == n / 2);
                            let re = if endpoint {
                                src_frame[2 * k]
                            } else {
                                2.0 * src_frame[2 * k]
                            };
                            let im = if endpoint {
                                0.0
                            } else {
                                2.0 * src_frame[2 * k + 1]
                            };
                            if accumulate {
                                out_frame[2 * k] += re;
                                out_frame[2 * k + 1] += im;
                            } else {
                                out_frame[2 * k] = re;
                                out_frame[2 * k + 1] = im;
                            }
                        }
                    }
                } else {
                    let out_data = &mut *output.data;
                    for b in 0..s.batch {
                        for f in 0..frames_count {
                            let base = (b * frames_count + f) * bins * 2;
                            for k in 0..bins {
                                let endpoint = k == 0 || (even_n && k == n / 2);
                                let re = if endpoint {
                                    spec_buf[base + 2 * k]
                                } else {
                                    2.0 * spec_buf[base + 2 * k]
                                };
                                let im = if endpoint {
                                    0.0
                                } else {
                                    2.0 * spec_buf[base + 2 * k + 1]
                                };
                                let idx_re = dsv.index(b, f, k, 0);
                                let idx_im = dsv.index(b, f, k, 1);
                                if accumulate {
                                    out_data[idx_re] += re;
                                    out_data[idx_im] += im;
                                } else {
                                    out_data[idx_re] = re;
                                    out_data[idx_im] = im;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
