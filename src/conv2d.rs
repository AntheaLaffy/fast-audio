use crate::{
    Activation, Result, Shape, Tensor,
    cpu::Cpu,
    error::{int, invalid},
};

/// NCHW cross-correlation or transposed convolution. Public weights use OIHW
/// for forward and IOHW for transposed, matching PyTorch group semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dSpec {
    pub batch: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub input: [usize; 2],
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
    pub transposed: bool,
    pub output_padding: [usize; 2],
}
impl Conv2dSpec {
    pub fn input_shape(&self) -> [usize; 4] {
        [self.batch, self.in_channels, self.input[0], self.input[1]]
    }
    pub fn weight_shape(&self) -> [usize; 4] {
        let (rows, columns) = if self.transposed {
            (self.in_channels, self.out_channels)
        } else {
            (self.out_channels, self.in_channels)
        };
        [
            rows,
            columns.checked_div(self.groups).unwrap_or(0),
            self.kernel[0],
            self.kernel[1],
        ]
    }
    pub fn output_shape(&self) -> Result<[usize; 4]> {
        for value in [
            self.batch,
            self.in_channels,
            self.out_channels,
            self.groups,
            self.input[0],
            self.input[1],
            self.kernel[0],
            self.kernel[1],
            self.stride[0],
            self.stride[1],
            self.dilation[0],
            self.dilation[1],
        ] {
            if value == 0 {
                return Err(invalid("2D convolution dimensions must be positive"));
            }
            int(value)?;
        }
        if self.in_channels % self.groups != 0 || self.out_channels % self.groups != 0 {
            return Err(invalid("groups must divide both channel counts"));
        }
        let mut output = [self.batch, self.out_channels, 0, 0];
        for axis in 0..2 {
            int(self.padding[axis])?;
            int(self.output_padding[axis])?;
            let effective = (self.kernel[axis] - 1)
                .checked_mul(self.dilation[axis])
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| invalid("effective 2D kernel overflows"))?;
            let pads = self.padding[axis]
                .checked_mul(2)
                .ok_or_else(|| invalid("2D padding overflows"))?;
            output[axis + 2] = if self.transposed {
                if self.output_padding[axis] >= self.stride[axis]
                    && self.output_padding[axis] >= self.dilation[axis]
                {
                    return Err(invalid(
                        "output padding must be smaller than stride or dilation",
                    ));
                }
                (self.input[axis] - 1)
                    .checked_mul(self.stride[axis])
                    .and_then(|v| v.checked_add(effective))
                    .and_then(|v| v.checked_add(self.output_padding[axis]))
                    .and_then(|v| v.checked_sub(pads))
                    .filter(|&v| v > 0)
                    .ok_or_else(|| invalid("invalid transposed 2D output extent"))?
            } else {
                if self.output_padding[axis] != 0 {
                    return Err(invalid(
                        "output padding applies only to transposed convolution",
                    ));
                }
                let padded = self.input[axis]
                    .checked_add(pads)
                    .ok_or_else(|| invalid("padded 2D input overflows"))?;
                let available = padded
                    .checked_sub(effective)
                    .ok_or_else(|| invalid("2D kernel exceeds padded input"))?;
                available / self.stride[axis] + 1
            };
            int(output[axis + 2])?;
        }
        Shape::new(self.input_shape())?;
        Shape::new(self.weight_shape())?;
        Shape::new(output)?;
        int(self.input[0] * self.input[1])?;
        int(output[2] * output[3])?;
        let reduction = (self.in_channels / self.groups)
            .checked_mul(self.kernel[0])
            .and_then(|v| v.checked_mul(self.kernel[1]))
            .ok_or_else(|| invalid("2D reduction overflows"))?;
        int(reduction)?;
        let transposed_rows = (self.out_channels / self.groups)
            .checked_mul(self.kernel[0])
            .and_then(|v| v.checked_mul(self.kernel[1]))
            .ok_or_else(|| invalid("2D columns overflow"))?;
        int(transposed_rows)?;
        Ok(output)
    }
    pub(crate) fn pointwise(&self) -> bool {
        self.kernel == [1, 1]
            && self.stride == [1, 1]
            && self.padding == [0, 0]
            && self.output_padding == [0, 0]
    }
    pub(crate) fn check_weights(&self, weights: &Tensor, bias: Option<&Tensor>) -> Result<()> {
        self.output_shape()?;
        weights.shape().expect(&self.weight_shape())?;
        if let Some(b) = bias {
            b.shape().expect(&[self.out_channels])?;
        }
        Ok(())
    }
    pub(crate) fn pointwise_weights(&self, weights: &Tensor) -> Vec<f32> {
        if !self.transposed {
            return weights.data().to_vec();
        }
        let mut out = vec![0.0; weights.data().len()];
        let cin = self.in_channels / self.groups;
        let cout = self.out_channels / self.groups;
        for group in 0..self.groups {
            for oc in 0..cout {
                for ic in 0..cin {
                    out[(group * cout + oc) * cin + ic] =
                        weights.data()[(group * cin + ic) * cout + oc];
                }
            }
        }
        out
    }
}

#[cfg(any(feature = "onednn", feature = "cudnn"))]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Conv2dDims {
    pub batch: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub ih: i32,
    pub iw: i32,
    pub kh: i32,
    pub kw: i32,
    pub sh: i32,
    pub sw: i32,
    pub ph: i32,
    pub pw: i32,
    pub dh: i32,
    pub dw: i32,
    pub groups: i32,
    pub oh: i32,
    pub ow: i32,
    pub transposed: i32,
    pub oph: i32,
    pub opw: i32,
}
#[cfg(any(feature = "onednn", feature = "cudnn"))]
impl Conv2dSpec {
    pub(crate) fn native_dims(&self) -> Result<Conv2dDims> {
        let o = self.output_shape()?;
        Ok(Conv2dDims {
            batch: self.batch as i32,
            in_channels: self.in_channels as i32,
            out_channels: self.out_channels as i32,
            ih: self.input[0] as i32,
            iw: self.input[1] as i32,
            kh: self.kernel[0] as i32,
            kw: self.kernel[1] as i32,
            sh: self.stride[0] as i32,
            sw: self.stride[1] as i32,
            ph: self.padding[0] as i32,
            pw: self.padding[1] as i32,
            dh: self.dilation[0] as i32,
            dw: self.dilation[1] as i32,
            groups: self.groups as i32,
            oh: o[2] as i32,
            ow: o[3] as i32,
            transposed: i32::from(self.transposed),
            oph: self.output_padding[0] as i32,
            opw: self.output_padding[1] as i32,
        })
    }
}

enum Implementation {
    Dense,
    #[cfg(all(feature = "mkl", feature = "onednn"))]
    OneDnn(Box<crate::onednn::Conv2d>),
}
/// Immutable prepared convolution. Output and all explicit scratch are reused.
pub struct Conv2d {
    cpu: Cpu,
    spec: Conv2dSpec,
    output: [usize; 4],
    weights: Vec<f32>,
    bias: Option<Vec<f32>>,
    scratch: Vec<f32>,
    activation: Activation,
    implementation: Implementation,
}
impl Conv2d {
    pub fn new(
        cpu: Cpu,
        spec: Conv2dSpec,
        weights: &Tensor,
        bias: Option<&Tensor>,
        activation: Activation,
    ) -> Result<Self> {
        spec.check_weights(weights, bias)?;
        let output = spec.output_shape()?;
        #[cfg(all(feature = "mkl", feature = "onednn"))]
        if let Cpu::Mkl { threads } = cpu {
            if !spec.pointwise() {
                let plan =
                    crate::onednn::Conv2d::new(spec, weights, bias, activation, threads.get())?;
                return Ok(Self {
                    cpu,
                    spec,
                    output,
                    weights: Vec::new(),
                    bias: None,
                    scratch: Vec::new(),
                    activation,
                    implementation: Implementation::OneDnn(Box::new(plan)),
                });
            }
        }
        let mut packed = if spec.pointwise() {
            spec.pointwise_weights(weights)
        } else {
            weights.data().to_vec()
        };
        let ci = spec.in_channels / spec.groups;
        let co = spec.out_channels / spec.groups;
        let k = spec.kernel[0] * spec.kernel[1];
        if spec.transposed && !spec.pointwise() {
            // GEMM produces [output-channel,kernel-position,input-position] columns.
            packed = vec![0.0; weights.data().len()];
            for group in 0..spec.groups {
                for ic in 0..ci {
                    for row in 0..co * k {
                        packed[(group * co * k + row) * ci + ic] =
                            weights.data()[(group * ci + ic) * co * k + row];
                    }
                }
            }
        }
        let scratch_len = if spec.pointwise() {
            0
        } else if spec.transposed {
            Shape::new([co, k, spec.input[0], spec.input[1]])?.len()
        } else {
            Shape::new([ci, k, output[2], output[3]])?.len()
        };
        Ok(Self {
            cpu,
            spec,
            output,
            weights: packed,
            bias: bias.map(|b| b.data().to_vec()),
            scratch: vec![0.0; scratch_len],
            activation,
            implementation: Implementation::Dense,
        })
    }
    pub fn workspace_bytes(&self) -> usize {
        match &self.implementation {
            Implementation::Dense => self.scratch.len() * 4,
            #[cfg(all(feature = "mkl", feature = "onednn"))]
            Implementation::OneDnn(p) => p.workspace_bytes(),
        }
    }
    pub fn parameter_bytes(&self) -> usize {
        match &self.implementation {
            Implementation::Dense => {
                (self.weights.len() + self.bias.as_ref().map_or(0, Vec::len)) * 4
            }
            #[cfg(all(feature = "mkl", feature = "onednn"))]
            Implementation::OneDnn(p) => p.parameter_bytes(),
        }
    }
    pub fn implementation(&self) -> &str {
        match &self.implementation {
            Implementation::Dense => "im2col/col2im GEMM",
            #[cfg(all(feature = "mkl", feature = "onednn"))]
            Implementation::OneDnn(p) => p.implementation(),
        }
    }
    pub fn run(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        x.shape().expect(&self.spec.input_shape())?;
        out.shape().expect(&self.output)?;
        #[cfg(all(feature = "mkl", feature = "onednn"))]
        if let Implementation::OneDnn(p) = &mut self.implementation {
            return p.run(x, out);
        }
        let s = self.spec;
        let ci = s.in_channels / s.groups;
        let co = s.out_channels / s.groups;
        let input_area = s.input[0] * s.input[1];
        let output_area = self.output[2] * self.output[3];
        let kernel = s.kernel[0] * s.kernel[1];
        out.data_mut().fill(0.0);
        for batch in 0..s.batch {
            for group in 0..s.groups {
                let input = &x.data()[(batch * s.in_channels + group * ci) * input_area..]
                    [..ci * input_area];
                let output = &mut out.data_mut()
                    [(batch * s.out_channels + group * co) * output_area..][..co * output_area];
                if s.pointwise() {
                    self.cpu.gemm(
                        co,
                        output_area,
                        ci,
                        &self.weights[group * co * ci..][..co * ci],
                        input,
                        output,
                    );
                } else if !s.transposed {
                    for channel in 0..ci {
                        for kh in 0..s.kernel[0] {
                            for kw in 0..s.kernel[1] {
                                let row = (channel * kernel + kh * s.kernel[1] + kw) * output_area;
                                for oh in 0..self.output[2] {
                                    for ow in 0..self.output[3] {
                                        let ih = oh * s.stride[0] + kh * s.dilation[0];
                                        let iw = ow * s.stride[1] + kw * s.dilation[1];
                                        self.scratch[row + oh * self.output[3] + ow] = if ih
                                            >= s.padding[0]
                                            && ih - s.padding[0] < s.input[0]
                                            && iw >= s.padding[1]
                                            && iw - s.padding[1] < s.input[1]
                                        {
                                            input[(channel * s.input[0] + ih - s.padding[0])
                                                * s.input[1]
                                                + iw
                                                - s.padding[1]]
                                        } else {
                                            0.0
                                        };
                                    }
                                }
                            }
                        }
                    }
                    self.cpu.gemm(
                        co,
                        output_area,
                        ci * kernel,
                        &self.weights[group * co * ci * kernel..][..co * ci * kernel],
                        &self.scratch,
                        output,
                    );
                } else {
                    self.cpu.gemm(
                        co * kernel,
                        input_area,
                        ci,
                        &self.weights[group * co * kernel * ci..][..co * kernel * ci],
                        input,
                        &mut self.scratch,
                    );
                    for oc in 0..co {
                        for kh in 0..s.kernel[0] {
                            for kw in 0..s.kernel[1] {
                                let row = (oc * kernel + kh * s.kernel[1] + kw) * input_area;
                                for ih in 0..s.input[0] {
                                    for iw in 0..s.input[1] {
                                        let oh = ih * s.stride[0] + kh * s.dilation[0];
                                        let ow = iw * s.stride[1] + kw * s.dilation[1];
                                        if oh >= s.padding[0]
                                            && oh - s.padding[0] < self.output[2]
                                            && ow >= s.padding[1]
                                            && ow - s.padding[1] < self.output[3]
                                        {
                                            output[(oc * self.output[2] + oh - s.padding[0])
                                                * self.output[3]
                                                + ow
                                                - s.padding[1]] +=
                                                self.scratch[row + ih * s.input[1] + iw];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (i, value) in out.data_mut().iter_mut().enumerate() {
            if let Some(bias) = &self.bias {
                *value += bias[(i / output_area) % s.out_channels];
            }
            if self.activation == Activation::Relu && *value < 0.0 {
                *value = 0.0;
            }
        }
        Ok(())
    }
}
