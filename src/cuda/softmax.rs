use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    GradientMode, Layout, Result, SoftmaxSpec,
    error::{int, invalid},
    softmax::SoftmaxStridedView,
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeSoftmaxDims {
    groups: u64,
    axis_len: u64,
    offsets: [i64; 3],
    axis_strides: [i64; 3],
    group_rank: i32,
    kind: i32,
    fully_contiguous_last: i32,
    reserved: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeSoftmaxAxis {
    dim: u64,
    strides: [i64; 3],
}

const _: () = assert!(
    size_of::<NativeSoftmaxDims>() == 80
        && std::mem::offset_of!(NativeSoftmaxDims, offsets) == 16
        && std::mem::offset_of!(NativeSoftmaxDims, axis_strides) == 40
        && std::mem::offset_of!(NativeSoftmaxDims, group_rank) == 64
        && size_of::<NativeSoftmaxAxis>() == 32
);

unsafe extern "C" {
    fn fa_cuda_softmax_create(
        ctx: *mut c_void,
        dims: NativeSoftmaxDims,
        group_axes: *const NativeSoftmaxAxis,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_softmax_forward_run(plan: *mut c_void, x: *const f32, y: *mut f32) -> i32;
    fn fa_cuda_softmax_backward_run(
        plan: *mut c_void,
        x: *const f32,
        dy: *const f32,
        dx: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_softmax_backward_from_output_run(
        plan: *mut c_void,
        y: *const f32,
        dy: *const f32,
        dx: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_softmax_destroy(plan: *mut c_void);
}

struct Core {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    workspace: usize,
    shared: usize,
}

impl Core {
    fn new(cuda: &Cuda, dims: NativeSoftmaxDims, group_axes: &[NativeSoftmaxAxis]) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (mut workspace, mut shared) = (0, 0);
        // SAFETY: asserted ABI and validated immutable geometry; native syncs borrowed metadata upload.
        check(unsafe {
            fa_cuda_softmax_create(
                cuda.session.ptr(),
                dims,
                group_axes.as_ptr(),
                &mut workspace,
                &mut shared,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            ptr: nonnull(ptr)?,
            workspace,
            shared,
        })
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // SAFETY: sole native plan owner retains the CUDA session; native syncs before destroying.
        unsafe { fa_cuda_softmax_destroy(self.ptr.as_ptr()) }
    }
}

/// Prepared strided Softmax / LogSoftmax forward plan on CUDA.
pub struct SoftmaxPlan {
    spec: SoftmaxSpec,
    input_layout: Layout,
    output_layout: Layout,
    core: Core,
}

impl SoftmaxPlan {
    pub fn new(cuda: &Cuda, spec: &SoftmaxSpec, input: &Layout, output: &Layout) -> Result<Self> {
        let xv = SoftmaxStridedView::new(spec, input, false)?;
        let yv = SoftmaxStridedView::new(spec, output, true)?;
        let fully_contiguous_last = xv.fully_contiguous_last && yv.fully_contiguous_last;
        let group_axes: Vec<NativeSoftmaxAxis> = xv
            .group_dims
            .iter()
            .enumerate()
            .map(|(a, &dim)| NativeSoftmaxAxis {
                dim: dim as u64,
                strides: [xv.group_strides[a] as i64, yv.group_strides[a] as i64, 0],
            })
            .collect();
        let dims = NativeSoftmaxDims {
            groups: spec.groups() as u64,
            axis_len: spec.axis_len() as u64,
            offsets: [xv.offset as i64, yv.offset as i64, 0],
            axis_strides: [xv.axis_stride as i64, yv.axis_stride as i64, 0],
            group_rank: int(group_axes.len())?,
            kind: spec.kind() as i32,
            fully_contiguous_last: i32::from(fully_contiguous_last),
            reserved: 0,
        };
        let core = Core::new(cuda, dims, &group_axes)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
            core,
        })
    }

    pub fn spec(&self) -> &SoftmaxSpec {
        &self.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.input_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }
    pub fn run(&self, input: &DeviceView<'_>, output: &mut DeviceViewMut<'_>) -> Result<()> {
        self.core.cuda.owns(input.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if input.layout() != &self.input_layout || output.layout() != &self.output_layout {
            return Err(invalid("CUDA softmax layouts differ from prepared plan"));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output; enqueues on stream.
        check(unsafe {
            fa_cuda_softmax_forward_run(
                self.core.ptr.as_ptr(),
                input.tensor.read(),
                output.tensor.write(),
            )
        })
    }
}

/// Prepared strided Softmax / LogSoftmax local VJP on CUDA (`input_required() == true`).
pub struct SoftmaxBackward {
    spec: SoftmaxSpec,
    input_layout: Layout,
    upstream_layout: Layout,
    output_layout: Layout,
    core: Core,
}

impl SoftmaxBackward {
    pub fn new(
        cuda: &Cuda,
        spec: &SoftmaxSpec,
        input: &Layout,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let xv = SoftmaxStridedView::new(spec, input, false)?;
        let dyv = SoftmaxStridedView::new(spec, upstream, false)?;
        let dxv = SoftmaxStridedView::new(spec, output, true)?;
        let fully_contiguous_last =
            xv.fully_contiguous_last && dyv.fully_contiguous_last && dxv.fully_contiguous_last;
        let group_axes: Vec<NativeSoftmaxAxis> = xv
            .group_dims
            .iter()
            .enumerate()
            .map(|(a, &dim)| NativeSoftmaxAxis {
                dim: dim as u64,
                strides: [
                    xv.group_strides[a] as i64,
                    dyv.group_strides[a] as i64,
                    dxv.group_strides[a] as i64,
                ],
            })
            .collect();
        let dims = NativeSoftmaxDims {
            groups: spec.groups() as u64,
            axis_len: spec.axis_len() as u64,
            offsets: [xv.offset as i64, dyv.offset as i64, dxv.offset as i64],
            axis_strides: [
                xv.axis_stride as i64,
                dyv.axis_stride as i64,
                dxv.axis_stride as i64,
            ],
            group_rank: int(group_axes.len())?,
            kind: spec.kind() as i32,
            fully_contiguous_last: i32::from(fully_contiguous_last),
            reserved: 0,
        };
        let core = Core::new(cuda, dims, &group_axes)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            core,
        })
    }

    pub fn spec(&self) -> &SoftmaxSpec {
        &self.spec
    }
    pub fn input_required(&self) -> bool {
        true
    }
    pub fn input_layout(&self) -> &Layout {
        &self.input_layout
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.upstream_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }
    pub fn run(
        &self,
        input: &DeviceView<'_>,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.cuda.owns(input.tensor)?;
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if input.layout() != &self.input_layout
            || upstream.layout() != &self.upstream_layout
            || output.layout() != &self.output_layout
        {
            return Err(invalid(
                "CUDA softmax backward layouts differ from prepared plan",
            ));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output; enqueues on stream.
        check(unsafe {
            fa_cuda_softmax_backward_run(
                self.core.ptr.as_ptr(),
                input.tensor.read(),
                upstream.tensor.read(),
                output.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }

    /// Computes the Softmax / LogSoftmax backward pass directly from the forward output `y`
    /// without recomputing `exp(x - m)` for `Softmax` (or `max`/`log_sum_exp` for `LogSoftmax`).
    ///
    /// `output` must match the prepared `input_layout`.
    pub fn run_from_output(
        &self,
        output: &DeviceView<'_>,
        upstream: &DeviceView<'_>,
        grad_input: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.cuda.owns(output.tensor)?;
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(grad_input.tensor)?;
        if output.layout() != &self.input_layout
            || upstream.layout() != &self.upstream_layout
            || grad_input.layout() != &self.output_layout
        {
            return Err(invalid(
                "CUDA softmax backward layouts differ from prepared plan",
            ));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output; enqueues on stream.
        check(unsafe {
            fa_cuda_softmax_backward_from_output_run(
                self.core.ptr.as_ptr(),
                output.tensor.read(),
                upstream.tensor.read(),
                grad_input.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}
