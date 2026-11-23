use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    BinaryOperand, CatSpec, Error, GradientMode, Layout, PadSpec, Result,
    error::{int, invalid},
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
#[derive(Clone, Copy)]
struct NativePadAxis {
    in_dim: u64,
    out_dim: u64,
    pad_before: u64,
    pad_after: u64,
    in_stride: i64,
    out_stride: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativePadDims {
    total_in: u64,
    total_out: u64,
    in_offset: i64,
    out_offset: i64,
    rank: i32,
    mode: i32,
    const_val: f32,
    reserved: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeCatAxis {
    dim_out: u64,
    stride_a: i64,
    stride_b: i64,
    stride_out: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeCatDims {
    total: u64,
    split: u64,
    offset_a: i64,
    offset_b: i64,
    offset_out: i64,
    rank: i32,
    axis: i32,
    operand: i32,
    reserved: i32,
}

const _: () = assert!(
    size_of::<NativePadAxis>() == 48
        && size_of::<NativePadDims>() == 48
        && size_of::<NativeCatAxis>() == 32
        && size_of::<NativeCatDims>() == 56
);

unsafe extern "C" {
    fn fa_cuda_pad_create(
        ctx: *mut c_void,
        dims: NativePadDims,
        axes: *const NativePadAxis,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_pad_forward_run(plan: *mut c_void, x: *const f32, y: *mut f32) -> i32;
    fn fa_cuda_pad_backward_run(
        plan: *mut c_void,
        dy: *const f32,
        dx: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_pad_destroy(plan: *mut c_void);

    fn fa_cuda_cat_create(
        ctx: *mut c_void,
        dims: NativeCatDims,
        axes: *const NativeCatAxis,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_cat_forward_run(
        plan: *mut c_void,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
    ) -> i32;
    fn fa_cuda_cat_backward_run(
        plan: *mut c_void,
        dy: *const f32,
        dx: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_cat_destroy(plan: *mut c_void);
}

struct PadCore {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    workspace: usize,
}
impl Drop for PadCore {
    fn drop(&mut self) {
        // SAFETY: sole native plan owner retains CUDA session; native syncs before destroying.
        unsafe { fa_cuda_pad_destroy(self.ptr.as_ptr()) }
    }
}

fn build_pad_core(
    cuda: &Cuda,
    spec: &PadSpec,
    in_layout: &Layout,
    out_layout: &Layout,
) -> Result<PadCore> {
    let rank = spec.input_shape().dims().len();
    if rank > 16 {
        return Err(Error::Unsupported("CUDA pad supports rank <= 16".into()));
    }
    let axes: Vec<NativePadAxis> = (0..rank)
        .map(|a| NativePadAxis {
            in_dim: spec.input_shape().dims()[a] as u64,
            out_dim: spec.output_shape().dims()[a] as u64,
            pad_before: spec.pads()[a].0 as u64,
            pad_after: spec.pads()[a].1 as u64,
            in_stride: in_layout.strides()[a] as i64,
            out_stride: out_layout.strides()[a] as i64,
        })
        .collect();
    let dims = NativePadDims {
        total_in: spec.input_shape().len() as u64,
        total_out: spec.output_shape().len() as u64,
        in_offset: in_layout.offset() as i64,
        out_offset: out_layout.offset() as i64,
        rank: int(rank)?,
        mode: spec.mode().code(),
        const_val: spec.mode().constant_value(),
        reserved: 0,
    };
    let mut ptr = std::ptr::null_mut();
    let mut workspace = 0;
    // SAFETY: asserted ABI and validated geometry; native syncs metadata upload.
    check(unsafe {
        fa_cuda_pad_create(
            cuda.session.ptr(),
            dims,
            axes.as_ptr(),
            &mut workspace,
            &mut ptr,
        )
    })?;
    Ok(PadCore {
        cuda: cuda.clone(),
        ptr: nonnull(ptr)?,
        workspace,
    })
}

/// Prepared arbitrary-rank strided `Pad` forward plan on CUDA.
pub struct PadPlan {
    spec: PadSpec,
    input_layout: Layout,
    output_layout: Layout,
    core: PadCore,
}

impl PadPlan {
    pub fn new(cuda: &Cuda, spec: &PadSpec, input: &Layout, output: &Layout) -> Result<Self> {
        if input.shape() != spec.input_shape() || output.shape() != spec.output_shape() {
            return Err(invalid("CUDA pad layouts differ from spec"));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "CUDA pad output layout must be proved non-overlapping".into(),
            ));
        }
        let core = build_pad_core(cuda, spec, input, output)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
            core,
        })
    }
    pub fn spec(&self) -> &PadSpec {
        &self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(&self, input: &DeviceView<'_>, output: &mut DeviceViewMut<'_>) -> Result<()> {
        self.core.cuda.owns(input.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if input.layout() != &self.input_layout || output.layout() != &self.output_layout {
            return Err(invalid("CUDA pad bindings differ from prepared plan"));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output.
        check(unsafe {
            fa_cuda_pad_forward_run(
                self.core.ptr.as_ptr(),
                input.tensor.read(),
                output.tensor.write(),
            )
        })
    }
}

/// Prepared arbitrary-rank strided `Pad` local VJP (`input_required() == false`) on CUDA.
pub struct PadBackward {
    spec: PadSpec,
    upstream_layout: Layout,
    output_layout: Layout,
    core: PadCore,
}

impl PadBackward {
    pub fn new(cuda: &Cuda, spec: &PadSpec, upstream: &Layout, output: &Layout) -> Result<Self> {
        if upstream.shape() != spec.output_shape() || output.shape() != spec.input_shape() {
            return Err(invalid("CUDA pad backward layouts differ from spec"));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "CUDA pad backward output must be proved non-overlapping".into(),
            ));
        }
        let core = build_pad_core(cuda, spec, output, upstream)?;
        Ok(Self {
            spec: spec.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            core,
        })
    }
    pub fn spec(&self) -> &PadSpec {
        &self.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(
        &self,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if upstream.layout() != &self.upstream_layout || output.layout() != &self.output_layout {
            return Err(invalid(
                "CUDA pad backward bindings differ from prepared plan",
            ));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output.
        check(unsafe {
            fa_cuda_pad_backward_run(
                self.core.ptr.as_ptr(),
                upstream.tensor.read(),
                output.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

struct CatCore {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    workspace: usize,
}
impl Drop for CatCore {
    fn drop(&mut self) {
        // SAFETY: sole native plan owner retains CUDA session; native syncs before destroying.
        unsafe { fa_cuda_cat_destroy(self.ptr.as_ptr()) }
    }
}

/// Prepared arbitrary-rank binary `Cat` forward plan on CUDA.
pub struct CatPlan {
    spec: CatSpec,
    left_layout: Layout,
    right_layout: Layout,
    output_layout: Layout,
    core: CatCore,
}

impl CatPlan {
    pub fn new(
        cuda: &Cuda,
        spec: &CatSpec,
        left: &Layout,
        right: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        if left.shape() != spec.left_shape()
            || right.shape() != spec.right_shape()
            || output.shape() != spec.output_shape()
        {
            return Err(invalid("CUDA cat layouts differ from spec"));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "CUDA cat output layout must be proved non-overlapping".into(),
            ));
        }
        let rank = spec.output_shape().dims().len();
        let axes: Vec<NativeCatAxis> = (0..rank)
            .map(|a| NativeCatAxis {
                dim_out: spec.output_shape().dims()[a] as u64,
                stride_a: left.strides()[a] as i64,
                stride_b: right.strides()[a] as i64,
                stride_out: output.strides()[a] as i64,
            })
            .collect();
        let dims = NativeCatDims {
            total: spec.output_shape().len() as u64,
            split: spec.left_shape().dims()[spec.axis()] as u64,
            offset_a: left.offset() as i64,
            offset_b: right.offset() as i64,
            offset_out: output.offset() as i64,
            rank: int(rank)?,
            axis: int(spec.axis())?,
            operand: -1,
            reserved: 0,
        };
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: asserted ABI and validated geometry.
        check(unsafe {
            fa_cuda_cat_create(
                cuda.session.ptr(),
                dims,
                axes.as_ptr(),
                &mut workspace,
                &mut ptr,
            )
        })?;
        Ok(Self {
            spec: spec.clone(),
            left_layout: left.clone(),
            right_layout: right.clone(),
            output_layout: output.clone(),
            core: CatCore {
                cuda: cuda.clone(),
                ptr: nonnull(ptr)?,
                workspace,
            },
        })
    }
    pub fn spec(&self) -> &CatSpec {
        &self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(
        &self,
        left: &DeviceView<'_>,
        right: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.cuda.owns(left.tensor)?;
        self.core.cuda.owns(right.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if left.layout() != &self.left_layout
            || right.layout() != &self.right_layout
            || output.layout() != &self.output_layout
        {
            return Err(invalid("CUDA cat bindings differ from prepared plan"));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output.
        check(unsafe {
            fa_cuda_cat_forward_run(
                self.core.ptr.as_ptr(),
                left.tensor.read(),
                right.tensor.read(),
                output.tensor.write(),
            )
        })
    }
}

/// Prepared arbitrary-rank binary `Cat` local VJP (`input_dependencies() == [false, false]`) on CUDA.
pub struct CatBackward {
    spec: CatSpec,
    operand: BinaryOperand,
    upstream_layout: Layout,
    output_layout: Layout,
    core: CatCore,
}

impl CatBackward {
    pub fn new(
        cuda: &Cuda,
        operand: BinaryOperand,
        spec: &CatSpec,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let target_shape = match operand {
            BinaryOperand::Left => spec.left_shape(),
            BinaryOperand::Right => spec.right_shape(),
        };
        if upstream.shape() != spec.output_shape() || output.shape() != target_shape {
            return Err(invalid("CUDA cat backward layouts differ from spec"));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "CUDA cat backward output must be proved non-overlapping".into(),
            ));
        }
        let rank = target_shape.dims().len();
        let axes: Vec<NativeCatAxis> = (0..rank)
            .map(|a| NativeCatAxis {
                dim_out: target_shape.dims()[a] as u64,
                stride_a: output.strides()[a] as i64,
                stride_b: 0,
                stride_out: upstream.strides()[a] as i64,
            })
            .collect();
        let dims = NativeCatDims {
            total: target_shape.len() as u64,
            split: spec.left_shape().dims()[spec.axis()] as u64,
            offset_a: output.offset() as i64,
            offset_b: 0,
            offset_out: upstream.offset() as i64,
            rank: int(rank)?,
            axis: int(spec.axis())?,
            operand: if operand == BinaryOperand::Left { 0 } else { 1 },
            reserved: 0,
        };
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: asserted ABI and validated geometry.
        check(unsafe {
            fa_cuda_cat_create(
                cuda.session.ptr(),
                dims,
                axes.as_ptr(),
                &mut workspace,
                &mut ptr,
            )
        })?;
        Ok(Self {
            spec: spec.clone(),
            operand,
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            core: CatCore {
                cuda: cuda.clone(),
                ptr: nonnull(ptr)?,
                workspace,
            },
        })
    }
    pub fn spec(&self) -> &CatSpec {
        &self.spec
    }
    pub fn operand(&self) -> BinaryOperand {
        self.operand
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        [false, false]
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(
        &self,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if upstream.layout() != &self.upstream_layout || output.layout() != &self.output_layout {
            return Err(invalid(
                "CUDA cat backward bindings differ from prepared plan",
            ));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output.
        check(unsafe {
            fa_cuda_cat_backward_run(
                self.core.ptr.as_ptr(),
                upstream.tensor.read(),
                output.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}
