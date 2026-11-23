use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    BinaryOp, BinaryOperand, GradientMode, Layout, ReduceSpec, Result, UnaryOp,
    error::int,
    pointwise_backward::{Formula, Geometry},
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
struct NativeDims {
    groups: u64,
    count: u64,
    origins: [i64; 4],
    rank: i32,
    reduction_rank: i32,
    operation: i32,
    arity: i32,
    operand: i32,
    kind: i32,
    needs: i32,
    reserved: i32,
}
#[repr(C)]
struct NativeGroup {
    dim: u64,
    reads: [i64; 3],
    output: i64,
}
#[repr(C)]
struct NativeAxis {
    dim: u64,
    reads: [i64; 3],
}
const _: () = assert!(
    size_of::<NativeDims>() == 80
        && std::mem::offset_of!(NativeDims, rank) == 48
        && std::mem::offset_of!(NativeDims, needs) == 72
);
const _: () = assert!(size_of::<NativeGroup>() == 40 && size_of::<NativeAxis>() == 32);
unsafe extern "C" {
    fn fa_cuda_pointwise_backward_create(
        ctx: *mut c_void,
        dims: NativeDims,
        groups: *const NativeGroup,
        axes: *const NativeAxis,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_pointwise_backward_run(
        plan: *mut c_void,
        upstream: *const f32,
        left: *const f32,
        right: *const f32,
        out: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_pointwise_backward_destroy(plan: *mut c_void);
}
struct Core {
    cuda: Cuda,
    geometry: Geometry,
    ptr: NonNull<c_void>,
    workspace: usize,
    shared: usize,
}
impl Core {
    fn new(cuda: &Cuda, geometry: Geometry) -> Result<Self> {
        let (operation, arity, operand) = geometry.formula.code();
        let dims = NativeDims {
            groups: geometry.destination.shape().len() as u64,
            count: geometry.contraction.reduction_len().unwrap_or(0) as u64,
            origins: geometry.origins.map(|x| x as i64),
            rank: int(geometry.groups.len())?,
            reduction_rank: int(geometry.reduced.len())?,
            operation,
            arity,
            operand,
            kind: geometry.contraction.strategy() as i32,
            needs: i32::from(geometry.dependencies[0]) | (i32::from(geometry.dependencies[1]) * 2),
            reserved: 0,
        };
        let groups: Vec<_> = geometry
            .groups
            .iter()
            .map(|a| NativeGroup {
                dim: a.dim as u64,
                reads: a.reads.map(|s| s as i64),
                output: a.output as i64,
            })
            .collect();
        let axes: Vec<_> = geometry
            .reduced
            .iter()
            .map(|a| NativeAxis {
                dim: a.dim as u64,
                reads: a.reads.map(|s| s as i64),
            })
            .collect();
        let (mut workspace, mut shared) = (0, 0);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: asserted ABI, validated layouts and jointly coalesced address maps.
        // Native synchronizes borrowed descriptor uploads before return or failure cleanup.
        check(unsafe {
            fa_cuda_pointwise_backward_create(
                cuda.session.ptr(),
                dims,
                groups.as_ptr(),
                axes.as_ptr(),
                &mut workspace,
                &mut shared,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            geometry,
            ptr: nonnull(ptr)?,
            workspace,
            shared,
        })
    }
    fn run(
        &self,
        inputs: [Option<&DeviceView<'_>>; 2],
        upstream: &DeviceView<'_>,
        destination: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        for source in inputs.into_iter().flatten() {
            self.cuda.owns(source.tensor)?;
        }
        self.cuda.owns(upstream.tensor)?;
        self.cuda.owns(destination.tensor)?;
        self.geometry.validate(
            inputs.map(|s| s.map(|v| v.layout())),
            upstream.layout(),
            destination.layout(),
        )?;
        if self.geometry.destination.shape().is_empty()
            || (self.geometry.contraction.reduction_len() == Some(0)
                && mode == GradientMode::Accumulate)
        {
            return Ok(());
        }
        // SAFETY: required payloads and every supplied session/layout checked first.
        // Compile-time-selected kernels read only needed operands and write injective targets.
        check(unsafe {
            fa_cuda_pointwise_backward_run(
                self.ptr.as_ptr(),
                upstream.tensor.read(),
                inputs[0].map_or(std::ptr::null(), |s| s.tensor.read()),
                inputs[1].map_or(std::ptr::null(), |s| s.tensor.read()),
                destination.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}
impl Drop for Core {
    fn drop(&mut self) {
        // SAFETY: unique native owner retains its session through synchronized destruction.
        unsafe {
            fa_cuda_pointwise_backward_destroy(self.ptr.as_ptr());
        }
    }
}

/// Prepared same-stream unary local VJP with explicit optional input retention.
pub struct UnaryBackward {
    core: Core,
}
impl UnaryBackward {
    pub fn new(
        cuda: &Cuda,
        operation: UnaryOp,
        input: &Layout,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        Ok(Self {
            core: Core::new(
                cuda,
                Geometry::new(
                    Formula::Unary(operation),
                    input,
                    None,
                    upstream,
                    destination,
                )?,
            )?,
        })
    }
    pub fn operation(&self) -> UnaryOp {
        match self.core.geometry.formula {
            Formula::Unary(op) => op,
            _ => unreachable!(),
        }
    }
    pub fn input_required(&self) -> bool {
        self.core.geometry.dependencies[0]
    }
    pub fn input_layout(&self) -> &Layout {
        self.core.geometry.inputs[0].as_ref().unwrap()
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.core.geometry.upstream
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.core.geometry.destination
    }
    pub fn contraction(&self) -> &ReduceSpec {
        &self.core.geometry.contraction
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.core.geometry.metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }
    pub fn run(
        &self,
        input: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        destination: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.run([input, None], upstream, destination, mode)
    }
}

/// Prepared one-operand VJP with fused fixed-order broadcast contraction.
pub struct BinaryBackward {
    core: Core,
}
impl BinaryBackward {
    pub fn new(
        cuda: &Cuda,
        operation: BinaryOp,
        operand: BinaryOperand,
        left: &Layout,
        right: &Layout,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        Ok(Self {
            core: Core::new(
                cuda,
                Geometry::new(
                    Formula::Binary(operation, operand),
                    left,
                    Some(right),
                    upstream,
                    destination,
                )?,
            )?,
        })
    }
    pub fn operation(&self) -> BinaryOp {
        match self.core.geometry.formula {
            Formula::Binary(op, _) => op,
            _ => unreachable!(),
        }
    }
    pub fn operand(&self) -> BinaryOperand {
        match self.core.geometry.formula {
            Formula::Binary(_, side) => side,
            _ => unreachable!(),
        }
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        self.core.geometry.dependencies
    }
    pub fn left_layout(&self) -> &Layout {
        self.core.geometry.inputs[0].as_ref().unwrap()
    }
    pub fn right_layout(&self) -> &Layout {
        self.core.geometry.inputs[1].as_ref().unwrap()
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.core.geometry.upstream
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.core.geometry.destination
    }
    pub fn contraction(&self) -> &ReduceSpec {
        &self.core.geometry.contraction
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.core.geometry.metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }
    pub fn run(
        &self,
        left: Option<&DeviceView<'_>>,
        right: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        destination: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.core.run([left, right], upstream, destination, mode)
    }
}
