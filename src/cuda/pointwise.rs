use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    BinaryOp, Layout, PointwiseStrategy, Result, UnaryOp, error::int, pointwise::PointwiseSpec,
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
struct NativeGeometry {
    len: u64,
    offsets: [i64; 3],
    inner: u64,
    strides: [i64; 3],
    rank: i32,
    arity: i32,
    operation: i32,
}
#[repr(C)]
struct NativeAxis {
    dim: u64,
    strides: [i64; 3],
}
const _: () = assert!(size_of::<NativeGeometry>() == 80);
const _: () = assert!(std::mem::offset_of!(NativeGeometry, rank) == 64);
const _: () = assert!(std::mem::offset_of!(NativeGeometry, operation) == 72);
const _: () = assert!(size_of::<NativeAxis>() == 32);
unsafe extern "C" {
    fn fa_cuda_pointwise_create(
        ctx: *mut c_void,
        geometry: NativeGeometry,
        axes: *const NativeAxis,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_pointwise_run(
        plan: *mut c_void,
        left: *const f32,
        right: *const f32,
        destination: *mut f32,
    ) -> i32;
    fn fa_cuda_pointwise_destroy(plan: *mut c_void);
}
struct Core<const N: usize> {
    cuda: Cuda,
    spec: PointwiseSpec<N>,
    ptr: NonNull<c_void>,
    workspace: usize,
}
impl<const N: usize> Core<N> {
    fn new(cuda: &Cuda, spec: PointwiseSpec<N>, operation: i32) -> Result<Self> {
        let rank = spec.axes.len().saturating_sub(1);
        let pack = |input: [isize; N], output: isize| {
            [
                input[0] as i64,
                if N == 2 { input[1] as i64 } else { 0 },
                output as i64,
            ]
        };
        let (input, output) = spec.inner_strides();
        let geometry = NativeGeometry {
            len: spec.len as u64,
            offsets: pack(spec.offsets, spec.output_offset),
            inner: spec.inner as u64,
            strides: pack(input, output),
            rank: int(rank)?,
            arity: int(N)?,
            operation,
        };
        let axes: Vec<_> = spec
            .axes
            .iter()
            .take(rank)
            .map(|a| NativeAxis {
                dim: a.dim as u64,
                strides: pack(a.inputs, a.output),
            })
            .collect();
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: asserted ABI and checked layouts define the complete address sets.
        // Native copies/synchronizes this borrowed O(rank) array before it expires,
        // including on failure. The resulting owner retains the session.
        check(unsafe {
            fa_cuda_pointwise_create(
                cuda.session.ptr(),
                geometry,
                axes.as_ptr(),
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
    fn run(
        &self,
        sources: [&DeviceView<'_>; N],
        destination: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        for source in sources {
            self.cuda.owns(source.tensor)?;
        }
        self.cuda.owns(destination.tensor)?;
        self.spec
            .validate(sources.map(|s| s.layout()), destination.layout())?;
        if self.spec.len == 0 {
            return Ok(());
        }
        // SAFETY: exact bound layouts, initialized spans and session identity were
        // checked. Inputs may share read storage; exclusive output cannot alias them.
        // One writer per output and original single-stream ordering are preserved.
        check(unsafe {
            fa_cuda_pointwise_run(
                self.ptr.as_ptr(),
                sources[0].tensor.read(),
                if N == 2 {
                    sources[1].tensor.read()
                } else {
                    std::ptr::null()
                },
                destination.tensor.write(),
            )
        })
    }
}
impl<const N: usize> Drop for Core<N> {
    fn drop(&mut self) {
        // SAFETY: exactly one owner; the session lives through synchronized destruction.
        unsafe {
            fa_cuda_pointwise_destroy(self.ptr.as_ptr());
        }
    }
}

/// Prepared same-stream unary math with checked borrowed storage and explicit outputs.
pub struct UnaryPlan {
    core: Core<1>,
    operation: UnaryOp,
}
impl UnaryPlan {
    pub fn new(
        cuda: &Cuda,
        operation: UnaryOp,
        source: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        Ok(Self {
            core: Core::new(
                cuda,
                PointwiseSpec::new([source], destination)?,
                operation as i32,
            )?,
            operation,
        })
    }
    pub fn operation(&self) -> UnaryOp {
        self.operation
    }
    pub fn source_layout(&self) -> &Layout {
        &self.core.spec.inputs[0]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.core.spec.output
    }
    pub fn strategy(&self) -> PointwiseStrategy {
        self.core.spec.strategy()
    }
    pub fn coalesced_rank(&self) -> usize {
        self.core.spec.axes.len()
    }
    pub fn inner_len(&self) -> usize {
        self.core.spec.inner
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.core.spec.host_metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn run(&self, source: &DeviceView<'_>, destination: &mut DeviceViewMut<'_>) -> Result<()> {
        self.core.run([source], destination)
    }
}
/// Prepared same-stream binary math with immutable original broadcast geometry.
pub struct BinaryPlan {
    core: Core<2>,
    operation: BinaryOp,
}
impl BinaryPlan {
    pub fn new(
        cuda: &Cuda,
        operation: BinaryOp,
        left: &Layout,
        right: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        Ok(Self {
            core: Core::new(
                cuda,
                PointwiseSpec::new([left, right], destination)?,
                operation as i32,
            )?,
            operation,
        })
    }
    pub fn operation(&self) -> BinaryOp {
        self.operation
    }
    pub fn left_layout(&self) -> &Layout {
        &self.core.spec.inputs[0]
    }
    pub fn right_layout(&self) -> &Layout {
        &self.core.spec.inputs[1]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.core.spec.output
    }
    pub fn strategy(&self) -> PointwiseStrategy {
        self.core.spec.strategy()
    }
    pub fn coalesced_rank(&self) -> usize {
        self.core.spec.axes.len()
    }
    pub fn inner_len(&self) -> usize {
        self.core.spec.inner
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.core.spec.host_metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn run(
        &self,
        left: &DeviceView<'_>,
        right: &DeviceView<'_>,
        destination: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.run([left, right], destination)
    }
}
