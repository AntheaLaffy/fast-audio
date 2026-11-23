use crate::{
    Result, Shape,
    error::{int, invalid},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Relu,
}

#[cfg(any(feature = "cuda", feature = "onednn"))]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ConvDims {
    pub batch: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub input_len: i32,
    pub kernel: i32,
    pub stride: i32,
    pub padding: i32,
    pub dilation: i32,
    pub groups: i32,
    pub output_len: i32,
}

#[cfg(any(feature = "cuda", feature = "onednn"))]
impl Conv1dSpec {
    pub(crate) fn native_dims(self) -> Result<ConvDims> {
        let output_len = self.output_len()?;
        Ok(ConvDims {
            batch: int(self.batch)?,
            in_channels: int(self.in_channels)?,
            out_channels: int(self.out_channels)?,
            input_len: int(self.input_len)?,
            kernel: int(self.kernel)?,
            stride: int(self.stride)?,
            padding: int(self.padding)?,
            dilation: int(self.dilation)?,
            groups: int(self.groups)?,
            output_len: int(output_len)?,
        })
    }
}

/// NCL cross-correlation with OIK weights. Padding is symmetric and uses zeros.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dSpec {
    pub batch: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub input_len: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv1dSpec {
    pub fn output_len(&self) -> Result<usize> {
        for x in [
            self.batch,
            self.in_channels,
            self.out_channels,
            self.input_len,
            self.kernel,
            self.stride,
            self.dilation,
            self.groups,
        ] {
            if x == 0 {
                return Err(invalid("convolution dimensions must be positive"));
            }
            int(x)?;
        }
        int(self.padding)?;
        if self.in_channels % self.groups != 0 || self.out_channels % self.groups != 0 {
            return Err(invalid("groups must divide input and output channels"));
        }
        let receptive = (self.kernel - 1)
            .checked_mul(self.dilation)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| invalid("convolution receptive field overflows"))?;
        let padded = self
            .padding
            .checked_mul(2)
            .and_then(|x| x.checked_add(self.input_len))
            .ok_or_else(|| invalid("padded length overflows"))?;
        int(padded)?;
        if padded < receptive {
            return Err(invalid("kernel is larger than padded input"));
        }
        let output_len = (padded - receptive) / self.stride + 1;
        int(output_len)?;
        Shape::new(self.input_shape())?;
        Shape::new(self.weight_shape())?;
        Shape::new([self.batch, self.out_channels, output_len])?;
        int(self.reduction_len())?;
        Ok(output_len)
    }
    pub fn input_shape(&self) -> [usize; 3] {
        [self.batch, self.in_channels, self.input_len]
    }
    pub fn weight_shape(&self) -> [usize; 3] {
        [
            self.out_channels,
            self.in_channels.checked_div(self.groups).unwrap_or(0),
            self.kernel,
        ]
    }
    pub fn output_shape(&self) -> Result<[usize; 3]> {
        Ok([self.batch, self.out_channels, self.output_len()?])
    }
    pub(crate) fn reduction_len(&self) -> usize {
        (self.in_channels / self.groups) * self.kernel
    }
    pub(crate) fn pointwise(&self) -> bool {
        self.kernel == 1 && self.stride == 1 && self.padding == 0
    }
    pub(crate) fn check(
        &self,
        x: &Shape,
        w: &Shape,
        bias: Option<&Shape>,
        y: &Shape,
        output_len: usize,
    ) -> Result<()> {
        x.expect(&self.input_shape())?;
        w.expect(&self.weight_shape())?;
        if let Some(bias) = bias {
            bias.expect(&[self.out_channels])?;
        }
        y.expect(&[self.batch, self.out_channels, output_len])
    }
}

/// NCL transposed 1D convolution with IOK weights matching PyTorch `conv_transpose1d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose1dSpec {
    pub batch: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub input_len: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl ConvTranspose1dSpec {
    pub fn input_shape(&self) -> [usize; 3] {
        [self.batch, self.in_channels, self.input_len]
    }
    pub fn weight_shape(&self) -> [usize; 3] {
        [
            self.in_channels,
            self.out_channels.checked_div(self.groups).unwrap_or(0),
            self.kernel,
        ]
    }
    pub fn output_len(&self) -> Result<usize> {
        for x in [
            self.batch,
            self.in_channels,
            self.out_channels,
            self.input_len,
            self.kernel,
            self.stride,
            self.dilation,
            self.groups,
        ] {
            if x == 0 {
                return Err(invalid(
                    "transposed 1D convolution dimensions must be positive",
                ));
            }
            int(x)?;
        }
        int(self.padding)?;
        int(self.output_padding)?;
        if self.in_channels % self.groups != 0 || self.out_channels % self.groups != 0 {
            return Err(invalid("groups must divide both channel counts"));
        }
        if self.output_padding >= self.stride && self.output_padding >= self.dilation {
            return Err(invalid(
                "output_padding must be smaller than stride or dilation",
            ));
        }
        let effective = (self.kernel - 1)
            .checked_mul(self.dilation)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| invalid("transposed 1D convolution kernel overflows"))?;
        let pads = self
            .padding
            .checked_mul(2)
            .ok_or_else(|| invalid("transposed 1D convolution padding overflows"))?;
        let out = (self.input_len - 1)
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(effective))
            .and_then(|v| v.checked_add(self.output_padding))
            .and_then(|v| v.checked_sub(pads))
            .filter(|&v| v > 0)
            .ok_or_else(|| invalid("invalid transposed 1D convolution output length"))?;
        int(out)?;
        Shape::new(self.input_shape())?;
        Shape::new(self.weight_shape())?;
        Shape::new([self.batch, self.out_channels, out])?;
        Ok(out)
    }
    pub fn output_shape(&self) -> Result<[usize; 3]> {
        Ok([self.batch, self.out_channels, self.output_len()?])
    }
}

/// Unnormalized, one-sided STFT geometry with no centering.
/// Plans default to periodic Hann and can snapshot an explicit window.
/// Input [batch, samples]; output [batch, frames, fft_size / 2 + 1, real/imag].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StftSpec {
    pub batch: usize,
    pub input_len: usize,
    pub fft_size: usize,
    pub hop: usize,
}

impl StftSpec {
    pub fn frames(&self) -> Result<usize> {
        if self.batch == 0 || self.hop == 0 || self.fft_size < 2 || self.input_len < self.fft_size {
            return Err(invalid(
                "STFT needs batch/hop > 0 and input_len >= fft_size >= 2",
            ));
        }
        let frames = 1 + (self.input_len - self.fft_size) / self.hop;
        self.check_frames(frames)?;
        Ok(frames)
    }
    fn check_frames(&self, frames: usize) -> Result<()> {
        let transforms = self
            .batch
            .checked_mul(frames)
            .ok_or_else(|| invalid("STFT batch overflows"))?;
        int(self.fft_size)?;
        int(self.input_len)?;
        int(self.hop)?;
        int(transforms)?;
        Shape::new([self.batch, self.input_len])?;
        Shape::new([transforms, self.fft_size])?;
        Shape::new([transforms, self.bins(), 2])?;
        Ok(())
    }
    pub fn bins(&self) -> usize {
        self.fft_size / 2 + 1
    }
    pub fn input_shape(&self) -> [usize; 2] {
        [self.batch, self.input_len]
    }
    pub fn output_shape(&self) -> Result<[usize; 4]> {
        Ok([self.batch, self.frames()?, self.bins(), 2])
    }
    /// Reflect-pad both ends by fft_size/2. The original non-centered contract is unchanged.
    pub fn centered(self) -> Result<CenteredStftSpec> {
        let padding = self.fft_size / 2;
        if self.batch == 0 || self.hop == 0 || self.fft_size < 2 || self.input_len <= padding {
            return Err(invalid(
                "centered STFT needs batch/hop > 0, fft_size >= 2 and input_len > fft_size/2",
            ));
        }
        let padded = padding
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.input_len))
            .ok_or_else(|| invalid("centered STFT padded length overflows"))?;
        int(padded)?;
        let frames = 1 + (padded - self.fft_size) / self.hop;
        self.check_frames(frames)?;
        Ok(CenteredStftSpec { base: self, frames })
    }
    pub(crate) fn window(&self) -> Vec<f32> {
        (0..self.fft_size)
            .map(|i| {
                (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / self.fft_size as f64).cos()) as f32
            })
            .collect()
    }
    pub(crate) fn checked_window(&self, window: &crate::Tensor) -> Result<Vec<f32>> {
        window.shape().expect(&[self.fft_size])?;
        self.checked_window_slice(window.data())
    }
    pub(crate) fn checked_window_slice(&self, window: &[f32]) -> Result<Vec<f32>> {
        if window.len() != self.fft_size {
            return Err(invalid("FFT window length differs from fft_size"));
        }
        if window.iter().any(|v| !v.is_finite()) {
            return Err(invalid("FFT window coefficients must be finite"));
        }
        Ok(window.to_vec())
    }
}

/// Validated reflect-centered geometry for a fixed waveform length.
/// Forward input is [batch, samples], spectrum is [batch, frames, bins, real/imag].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CenteredStftSpec {
    base: StftSpec,
    frames: usize,
}

impl CenteredStftSpec {
    pub fn spec(&self) -> StftSpec {
        self.base
    }
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn bins(&self) -> usize {
        self.base.bins()
    }
    pub fn input_shape(&self) -> [usize; 2] {
        self.base.input_shape()
    }
    pub fn output_shape(&self) -> [usize; 4] {
        [self.base.batch, self.frames, self.bins(), 2]
    }
    pub(crate) fn padding(&self) -> usize {
        self.base.fft_size / 2
    }
    pub(crate) fn normalization(&self, window: &[f32]) -> Result<Vec<f32>> {
        let s = self.base;
        let padding = self.padding();
        let mut envelope = vec![0.0f64; s.input_len];
        for frame in 0..self.frames {
            for (t, &w) in window.iter().enumerate() {
                let position = frame * s.hop + t;
                if position >= padding && position - padding < s.input_len {
                    envelope[position - padding] += (w as f64) * (w as f64);
                }
            }
        }
        if envelope.iter().any(|&v| v < 1e-11) {
            return Err(invalid(
                "inverse STFT has an uncovered or near-zero window envelope (NOLA)",
            ));
        }
        // All FFT backends return an unscaled inverse; fold 1/N into overlap normalization.
        Ok(envelope
            .into_iter()
            .map(|v| (1.0 / (s.fft_size as f64 * v)) as f32)
            .collect())
    }
}
