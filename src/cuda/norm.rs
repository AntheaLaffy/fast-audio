use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    GradientMode, Layout, NormSpec, Result,
    error::{int, invalid},
    norm::NormStridedView,
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeNormDims {
    groups: u64,
    inner_len: u64,
    offsets: [i64; 3],
    outer_rank: i32,
    inner_rank: i32,
    kind: i32,
    fully_contiguous: i32,
    eps: f32,
    reserved: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeNormAxis {
    dim: u64,
    strides: [i64; 3],
}

const _: () = assert!(
    size_of::<NativeNormDims>() == 64
        && std::mem::offset_of!(NativeNormDims, offsets) == 16
        && std::mem::offset_of!(NativeNormDims, outer_rank) == 40
        && std::mem::offset_of!(NativeNormDims, eps) == 56
        && size_of::<NativeNormAxis>() == 32
);

unsafe extern "C" {
    fn fa_cuda_norm_create(
        ctx: *mut c_void,
        dims: NativeNormDims,
        outer_axes: *const NativeNormAxis,
        inner_axes: *const NativeNormAxis,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_norm_forward_run(plan: *mut c_void, x: *const f32, y: *mut f32) -> i32;
    fn fa_cuda_norm_backward_run(
        plan: *mut c_void,
        x: *const f32,
        dy: *const f32,
        dx: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_norm_destroy(plan: *mut c_void);
}

struct Core {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    workspace: usize,
    shared: usize,
}

impl Core {
    fn new(
        cuda: &Cuda,
        dims: NativeNormDims,
        outer_axes: &[NativeNormAxis],
        inner_axes: &[NativeNormAxis],
    ) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (mut workspace, mut shared) = (0, 0);
        // SAFETY: validated ABI sizes and checked immutable geometry; native synchronizes
        // borrowed O(rank) metadata copies before returning.
        check(unsafe {
            fa_cuda_norm_create(
                cuda.session.ptr(),
                dims,
                outer_axes.as_ptr(),
                inner_axes.as_ptr(),
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
        unsafe { fa_cuda_norm_destroy(self.ptr.as_ptr()) }
    }
}

/// Prepared strided LayerNorm / RmsNorm forward plan on CUDA.
pub struct NormPlan {
    spec: NormSpec,
    input_layout: Layout,
    output_layout: Layout,
    core: Core,
}

impl NormPlan {
    pub fn new(cuda: &Cuda, spec: &NormSpec, input: &Layout, output: &Layout) -> Result<Self> {
        let xv = NormStridedView::new(spec, input, false)?;
        let yv = NormStridedView::new(spec, output, true)?;
        let fully_contiguous = xv.fully_contiguous && yv.fully_contiguous;
        let outer_axes: Vec<NativeNormAxis> = xv
            .outer_dims
            .iter()
            .enumerate()
            .map(|(a, &dim)| NativeNormAxis {
                dim: dim as u64,
                strides: [xv.outer_strides[a] as i64, yv.outer_strides[a] as i64, 0],
            })
            .collect();
        let inner_axes: Vec<NativeNormAxis> = xv
            .inner_dims
            .iter()
            .enumerate()
            .map(|(a, &dim)| NativeNormAxis {
                dim: dim as u64,
                strides: [xv.inner_strides[a] as i64, yv.inner_strides[a] as i64, 0],
            })
            .collect();
        let dims = NativeNormDims {
            groups: spec.groups() as u64,
            inner_len: spec.inner_len() as u64,
            offsets: [xv.offset as i64, yv.offset as i64, 0],
            outer_rank: int(outer_axes.len())?,
            inner_rank: int(inner_axes.len())?,
            kind: spec.kind() as i32,
            fully_contiguous: i32::from(fully_contiguous),
            eps: spec.eps(),
            reserved: 0,
        };
        let core = Core::new(cuda, dims, &outer_axes, &inner_axes)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
            core,
        })
    }

    pub fn spec(&self) -> &NormSpec {
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
            return Err(invalid(
                "CUDA normalization layouts differ from prepared plan",
            ));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output; enqueues on stream.
        check(unsafe {
            fa_cuda_norm_forward_run(
                self.core.ptr.as_ptr(),
                input.tensor.read(),
                output.tensor.write(),
            )
        })
    }
}

/// Prepared strided LayerNorm / RmsNorm local VJP on CUDA (`input_required() == true`).
pub struct NormBackward {
    spec: NormSpec,
    input_layout: Layout,
    upstream_layout: Layout,
    output_layout: Layout,
    core: Core,
}

impl NormBackward {
    pub fn new(
        cuda: &Cuda,
        spec: &NormSpec,
        input: &Layout,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let xv = NormStridedView::new(spec, input, false)?;
        let dyv = NormStridedView::new(spec, upstream, false)?;
        let dxv = NormStridedView::new(spec, output, true)?;
        let fully_contiguous = xv.fully_contiguous && dyv.fully_contiguous && dxv.fully_contiguous;
        let outer_axes: Vec<NativeNormAxis> = xv
            .outer_dims
            .iter()
            .enumerate()
            .map(|(a, &dim)| NativeNormAxis {
                dim: dim as u64,
                strides: [
                    xv.outer_strides[a] as i64,
                    dyv.outer_strides[a] as i64,
                    dxv.outer_strides[a] as i64,
                ],
            })
            .collect();
        let inner_axes: Vec<NativeNormAxis> = xv
            .inner_dims
            .iter()
            .enumerate()
            .map(|(a, &dim)| NativeNormAxis {
                dim: dim as u64,
                strides: [
                    xv.inner_strides[a] as i64,
                    dyv.inner_strides[a] as i64,
                    dxv.inner_strides[a] as i64,
                ],
            })
            .collect();
        let dims = NativeNormDims {
            groups: spec.groups() as u64,
            inner_len: spec.inner_len() as u64,
            offsets: [xv.offset as i64, dyv.offset as i64, dxv.offset as i64],
            outer_rank: int(outer_axes.len())?,
            inner_rank: int(inner_axes.len())?,
            kind: spec.kind() as i32,
            fully_contiguous: i32::from(fully_contiguous),
            eps: spec.eps(),
            reserved: 0,
        };
        let core = Core::new(cuda, dims, &outer_axes, &inner_axes)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            core,
        })
    }

    pub fn spec(&self) -> &NormSpec {
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
                "CUDA normalization backward layouts differ from prepared plan",
            ));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output; enqueues on stream.
        check(unsafe {
            fa_cuda_norm_backward_run(
                self.core.ptr.as_ptr(),
                input.tensor.read(),
                upstream.tensor.read(),
                output.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}
