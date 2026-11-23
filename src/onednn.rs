//! Prepared Intel inference convolution with owned, packed snapshots of weights and bias.
use crate::{
    Activation, Conv1dSpec, Result, Tensor,
    error::{int, invalid, native},
    spec::ConvDims,
};
use std::{
    ffi::{CStr, c_char, c_void},
    ptr::NonNull,
};

unsafe extern "C" {
    fn fa_onednn_conv2d_create(
        dims: crate::conv2d::Conv2dDims,
        weights: *const f32,
        bias: *const f32,
        relu: i32,
        threads: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_onednn_conv_create(
        dims: ConvDims,
        weights: *const f32,
        bias: *const f32,
        relu: i32,
        threads: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_onednn_conv_run(plan: *mut c_void, input: *const f32, output: *mut f32) -> i32;
    fn fa_onednn_conv_info(
        plan: *mut c_void,
        workspace: *mut usize,
        weights: *mut usize,
        implementation: *mut *const c_char,
    );
    fn fa_onednn_conv_destroy(plan: *mut c_void);
}

/// Immutable NCHW forward/transposed 2D inference convolution with vendor packing.
pub struct Conv2d {
    ptr: NonNull<c_void>,
    spec: crate::Conv2dSpec,
    output: [usize; 4],
    workspace: usize,
    parameters: usize,
    implementation: String,
    padded: Option<Tensor>,
}
impl Conv2d {
    pub fn new(
        spec: crate::Conv2dSpec,
        weights: &Tensor,
        bias: Option<&Tensor>,
        activation: Activation,
        threads: u16,
    ) -> Result<Self> {
        spec.check_weights(weights, bias)?;
        if threads == 0 {
            return Err(invalid("thread count must be positive"));
        }
        let mut dims = spec.native_dims()?;
        let mut padded = None;
        if spec.transposed
            && spec
                .output_padding
                .iter()
                .zip(spec.stride)
                .any(|(&p, s)| p >= s)
        {
            // Vendor deconvolution requires output padding below stride. Extend the
            // source with zeros so larger dilation-valid output padding has the same sum.
            let h = spec.input[0]
                .checked_add(spec.output_padding[0] / spec.stride[0])
                .ok_or_else(|| invalid("extended transpose input overflows"))?;
            let w = spec.input[1]
                .checked_add(spec.output_padding[1] / spec.stride[1])
                .ok_or_else(|| invalid("extended transpose input overflows"))?;
            dims.ih = int(h)?;
            dims.iw = int(w)?;
            dims.oph = (spec.output_padding[0] % spec.stride[0]) as i32;
            dims.opw = (spec.output_padding[1] % spec.stride[1]) as i32;
            padded = Some(Tensor::zeros([spec.batch, spec.in_channels, h, w])?);
        }
        let mut ptr = std::ptr::null_mut();
        native(
            "oneDNN",
            // SAFETY: exact parameter shapes, nonzero threads and geometry are checked;
            // native packing is synchronous and no C++ exception crosses this boundary.
            unsafe {
                fa_onednn_conv2d_create(
                    dims,
                    weights.data().as_ptr(),
                    bias.map_or(std::ptr::null(), |b| b.data().as_ptr()),
                    i32::from(activation == Activation::Relu),
                    threads.into(),
                    &mut ptr,
                )
            }
            .into(),
        )?;
        let ptr = NonNull::new(ptr).ok_or_else(|| invalid("oneDNN returned a null plan"))?;
        let (mut workspace, mut parameters, mut name) = (0, 0, std::ptr::null());
        // SAFETY: one successful, live plan and writable metadata pointers; name stays plan-owned.
        unsafe { fa_onednn_conv_info(ptr.as_ptr(), &mut workspace, &mut parameters, &mut name) };
        // SAFETY: native returns a valid NUL-terminated implementation name.
        let implementation = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            ptr,
            spec,
            output: spec.output_shape()?,
            workspace,
            parameters,
            implementation,
            padded,
        })
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace + self.padded.as_ref().map_or(0, |p| p.shape().len() * 4)
    }
    pub fn parameter_bytes(&self) -> usize {
        self.parameters
    }
    pub fn implementation(&self) -> &str {
        &self.implementation
    }
    pub fn run(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        x.shape().expect(&self.spec.input_shape())?;
        out.shape().expect(&self.output)?;
        let input = if let Some(padded) = &mut self.padded {
            let width = padded.shape().dims()[3];
            let height = padded.shape().dims()[2];
            padded.data_mut().fill(0.0);
            for image in 0..self.spec.batch * self.spec.in_channels {
                for row in 0..self.spec.input[0] {
                    padded.data_mut()[(image * height + row) * width..][..self.spec.input[1]]
                        .copy_from_slice(
                            &x.data()[(image * self.spec.input[0] + row) * self.spec.input[1]..]
                                [..self.spec.input[1]],
                        );
                }
            }
            padded.data().as_ptr()
        } else {
            x.data().as_ptr()
        };
        // SAFETY: exact NCHW buffers and exclusive output; native completes CPU execution before return.
        native(
            "oneDNN",
            unsafe { fa_onednn_conv_run(self.ptr.as_ptr(), input, out.data_mut().as_mut_ptr()) }
                .into(),
        )
    }
}
impl Drop for Conv2d {
    fn drop(&mut self) {
        // SAFETY: this plan has one owner and all CPU operations completed before return.
        unsafe { fa_onednn_conv_destroy(self.ptr.as_ptr()) }
    }
}

/// Immutable parameter snapshot. Updates to source weights require constructing a new plan.
/// Input/output remain NCL; any vendor layout reorders are included in `run`.
pub struct Conv1d {
    ptr: NonNull<c_void>,
    spec: Conv1dSpec,
    output_len: usize,
    workspace_bytes: usize,
    parameter_bytes: usize,
    implementation: String,
}
impl Conv1d {
    pub fn new(
        spec: Conv1dSpec,
        weights: &Tensor,
        bias: Option<&Tensor>,
        activation: Activation,
        threads: u16,
    ) -> Result<Self> {
        let dims = spec.native_dims()?;
        if threads == 0 {
            return Err(invalid("thread count must be positive"));
        }
        weights.shape().expect(&spec.weight_shape())?;
        if let Some(bias) = bias {
            bias.shape().expect(&[spec.out_channels])?;
        }
        let mut ptr = std::ptr::null_mut();
        // SAFETY: validated geometry and exact parameter shapes; native copies/reorders
        // parameters synchronously and catches every exception at the C ABI boundary.
        let status = unsafe {
            fa_onednn_conv_create(
                dims,
                weights.data().as_ptr(),
                bias.map_or(std::ptr::null(), |b| b.data().as_ptr()),
                i32::from(activation == Activation::Relu),
                int(threads.into())?,
                &mut ptr,
            )
        };
        native("oneDNN", status.into())?;
        let ptr = NonNull::new(ptr).ok_or_else(|| invalid("oneDNN returned a null plan"))?;
        let (mut workspace_bytes, mut parameter_bytes, mut implementation) =
            (0, 0, std::ptr::null());
        // SAFETY: live, exclusively owned plan and valid output pointers. The returned
        // implementation string is retained by the native primitive descriptor.
        unsafe {
            fa_onednn_conv_info(
                ptr.as_ptr(),
                &mut workspace_bytes,
                &mut parameter_bytes,
                &mut implementation,
            )
        };
        // SAFETY: oneDNN's implementation query returns a NUL-terminated string.
        let implementation = unsafe { CStr::from_ptr(implementation) }
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            ptr,
            spec,
            output_len: dims.output_len as usize,
            workspace_bytes,
            parameter_bytes,
            implementation,
        })
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    pub fn parameter_bytes(&self) -> usize {
        self.parameter_bytes
    }
    pub fn implementation(&self) -> &str {
        &self.implementation
    }
    pub fn run(&mut self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        x.shape().expect(&self.spec.input_shape())?;
        out.shape()
            .expect(&[self.spec.batch, self.spec.out_channels, self.output_len])?;
        // SAFETY: exact buffer shapes match the prepared plan; output is exclusive and
        // the native CPU stream completes before borrowed buffers can expire.
        let status = unsafe {
            fa_onednn_conv_run(
                self.ptr.as_ptr(),
                x.data().as_ptr(),
                out.data_mut().as_mut_ptr(),
            )
        };
        native("oneDNN", status.into())
    }
}
impl Drop for Conv1d {
    fn drop(&mut self) {
        // SAFETY: one successful native allocation has exactly one Rust owner.
        unsafe { fa_onednn_conv_destroy(self.ptr.as_ptr()) }
    }
}
