use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    GradientMode, Layout, ReduceSpec, Result,
    error::{int, invalid},
    reduce::{BackwardGeometry, GroupAxis, NonLinearBackwardGeometry, ReduceGeometry},
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
struct NativeDims {
    groups: u64,
    count: u64,
    source_offset: i64,
    destination_offset: i64,
    rank: i32,
    reduction_rank: i32,
    kind: i32,
    operation: i32,
    divisor: f32,
    reserved: i32,
}
#[repr(C)]
struct NativeGroup {
    dim: u64,
    source: i64,
    destination: i64,
}
#[repr(C)]
struct NativeAxis {
    dim: u64,
    stride: i64,
}
#[repr(C)]
struct NativeNlBwdDims {
    groups: u64,
    count: u64,
    source_offset: i64,
    upstream_offset: i64,
    destination_offset: i64,
    rank: i32,
    reduction_rank: i32,
    operation: i32,
    reserved: i32,
}
#[repr(C)]
struct NativeNlGroup {
    dim: u64,
    source: i64,
    upstream: i64,
    destination: i64,
}
#[repr(C)]
struct NativeNlRed {
    dim: u64,
    source: i64,
    destination: i64,
}
const _: () = assert!(size_of::<NativeDims>() == 56);
const _: () = assert!(std::mem::offset_of!(NativeDims, rank) == 32);
const _: () = assert!(std::mem::offset_of!(NativeDims, divisor) == 48);
const _: () = assert!(size_of::<NativeGroup>() == 24 && size_of::<NativeAxis>() == 16);
const _: () = assert!(
    size_of::<NativeNlBwdDims>() == 56
        && std::mem::offset_of!(NativeNlBwdDims, rank) == 40
        && size_of::<NativeNlGroup>() == 32
        && size_of::<NativeNlRed>() == 24
);
unsafe extern "C" {
    fn fa_cuda_reduce_create(
        ctx: *mut c_void,
        dims: NativeDims,
        groups: *const NativeGroup,
        axes: *const NativeAxis,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_reduce_run(plan: *mut c_void, source: *const f32, destination: *mut f32) -> i32;
    fn fa_cuda_reduce_backward_run(
        plan: *mut c_void,
        source: *const f32,
        destination: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_reduce_destroy(plan: *mut c_void);
    fn fa_cuda_reduce_nl_bwd_create(
        ctx: *mut c_void,
        dims: NativeNlBwdDims,
        groups: *const NativeNlGroup,
        axes: *const NativeNlRed,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_reduce_nl_bwd_run(
        plan: *mut c_void,
        source: *const f32,
        upstream: *const f32,
        destination: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_reduce_nl_bwd_destroy(plan: *mut c_void);
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
        dims: NativeDims,
        groups: &[NativeGroup],
        axes: &[NativeAxis],
    ) -> Result<Self> {
        let (mut workspace, mut shared) = (0, 0);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: asserted ABI and checked immutable geometry; native synchronizes
        // borrowed O(rank) uploads before they expire, including during failure cleanup.
        check(unsafe {
            fa_cuda_reduce_create(
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
            ptr: nonnull(ptr)?,
            workspace,
            shared,
        })
    }
}
impl Drop for Core {
    fn drop(&mut self) {
        // SAFETY: unique native owner; retained session survives synchronized deletion.
        unsafe {
            fa_cuda_reduce_destroy(self.ptr.as_ptr());
        }
    }
}
fn group(a: &GroupAxis) -> NativeGroup {
    NativeGroup {
        dim: a.dim as u64,
        source: a.source as i64,
        destination: a.destination as i64,
    }
}

/// Prepared same-stream ordered Sum/Mean; no execution-time allocation or upload.
pub struct ReducePlan {
    geometry: ReduceGeometry,
    core: Core,
}
impl ReducePlan {
    pub fn new(
        cuda: &Cuda,
        spec: &ReduceSpec,
        source: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let geometry = ReduceGeometry::new(spec, source, destination)?;
        let dims = NativeDims {
            groups: spec.output_shape().len() as u64,
            count: spec.reduction_len().unwrap_or(0) as u64,
            source_offset: source.offset() as i64,
            destination_offset: destination.offset() as i64,
            rank: int(geometry.groups.len())?,
            reduction_rank: int(geometry.reduced.len())?,
            kind: spec.strategy() as i32,
            operation: spec.operation() as i32,
            divisor: spec.divisor(),
            reserved: 0,
        };
        let groups: Vec<_> = geometry.groups.iter().map(group).collect();
        let axes: Vec<_> = geometry
            .reduced
            .iter()
            .map(|a| NativeAxis {
                dim: a.dim as u64,
                stride: a.stride as i64,
            })
            .collect();
        let core = Core::new(cuda, dims, &groups, &axes)?;
        Ok(Self { geometry, core })
    }
    pub fn spec(&self) -> &ReduceSpec {
        &self.geometry.spec
    }
    pub fn source_layout(&self) -> &Layout {
        &self.geometry.source
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.destination
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.geometry.metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }
    pub fn run(&self, source: &DeviceView<'_>, destination: &mut DeviceViewMut<'_>) -> Result<()> {
        self.core.cuda.owns(source.tensor)?;
        self.core.cuda.owns(destination.tensor)?;
        self.geometry
            .validate(source.layout(), destination.layout())?;
        if self.geometry.spec.output_shape().is_empty() {
            return Ok(());
        }
        // SAFETY: exact initialized layouts and session checked before writing. Ordered
        // groups cover injective output coordinates; sources may overlap read-only.
        check(unsafe {
            fa_cuda_reduce_run(
                self.core.ptr.as_ptr(),
                source.tensor.read(),
                destination.tensor.write(),
            )
        })
    }
}

struct NlBwdCore {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    workspace: usize,
    shared: usize,
}
impl Drop for NlBwdCore {
    fn drop(&mut self) {
        // SAFETY: sole native owner retains the CUDA session; native syncs before destroying.
        unsafe { fa_cuda_reduce_nl_bwd_destroy(self.ptr.as_ptr()) }
    }
}

enum ReduceBackwardInner {
    Linear {
        geometry: BackwardGeometry,
        core: Core,
    },
    NonLinear {
        geometry: NonLinearBackwardGeometry,
        core: NlBwdCore,
    },
}

/// Local reduction derivative (`Sum`/`Mean` payload-free; `Max`/`Min`/`LogSumExp` require input).
pub struct ReduceBackward {
    inner: ReduceBackwardInner,
}
impl ReduceBackward {
    pub fn new(
        cuda: &Cuda,
        spec: &ReduceSpec,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        if spec.operation().requires_input() {
            return Err(invalid(
                "CUDA Max, Min, and LogSumExp backward require source layout via with_input",
            ));
        }
        Self::build_linear(cuda, spec, upstream, destination)
    }
    pub fn with_input(
        cuda: &Cuda,
        spec: &ReduceSpec,
        source: &Layout,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        if !spec.operation().requires_input() {
            if source.shape() != spec.input_shape() {
                return Err(invalid("CUDA reduction source layout differs from spec"));
            }
            return Self::build_linear(cuda, spec, upstream, destination);
        }
        let geometry = NonLinearBackwardGeometry::new(spec, source, upstream, destination)?;
        let dims = NativeNlBwdDims {
            groups: spec.output_shape().len() as u64,
            count: spec.reduction_len().unwrap_or(0) as u64,
            source_offset: source.offset() as i64,
            upstream_offset: upstream.offset() as i64,
            destination_offset: destination.offset() as i64,
            rank: int(geometry.groups.len())?,
            reduction_rank: int(geometry.reduced.len())?,
            operation: spec.operation() as i32,
            reserved: 0,
        };
        let groups: Vec<_> = geometry
            .groups
            .iter()
            .map(|a| NativeNlGroup {
                dim: a.dim as u64,
                source: a.source as i64,
                upstream: a.upstream as i64,
                destination: a.destination as i64,
            })
            .collect();
        let axes: Vec<_> = geometry
            .reduced
            .iter()
            .map(|a| NativeNlRed {
                dim: a.dim as u64,
                source: a.source as i64,
                destination: a.destination as i64,
            })
            .collect();
        let (mut workspace, mut shared) = (0, 0);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: asserted ABI and validated geometry; native syncs borrowed metadata uploads.
        check(unsafe {
            fa_cuda_reduce_nl_bwd_create(
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
            inner: ReduceBackwardInner::NonLinear {
                geometry,
                core: NlBwdCore {
                    cuda: cuda.clone(),
                    ptr: nonnull(ptr)?,
                    workspace,
                    shared,
                },
            },
        })
    }
    fn build_linear(
        cuda: &Cuda,
        spec: &ReduceSpec,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let geometry = BackwardGeometry::new(spec, upstream, destination)?;
        let map = &geometry.map;
        let dims = NativeDims {
            groups: map.len as u64,
            count: spec.reduction_len().unwrap_or(0) as u64,
            source_offset: upstream.offset() as i64,
            destination_offset: destination.offset() as i64,
            rank: int(map.dims.len())?,
            reduction_rank: 0,
            kind: 5,
            operation: spec.operation() as i32,
            divisor: spec.divisor(),
            reserved: 0,
        };
        let groups: Vec<_> = map
            .dims
            .iter()
            .enumerate()
            .map(|(a, &dim)| NativeGroup {
                dim: dim as u64,
                source: map.source_strides[a] as i64,
                destination: map.destination_strides[a] as i64,
            })
            .collect();
        let core = Core::new(cuda, dims, &groups, &[])?;
        Ok(Self {
            inner: ReduceBackwardInner::Linear { geometry, core },
        })
    }
    pub fn input_required(&self) -> bool {
        self.spec().operation().requires_input()
    }
    pub fn spec(&self) -> &ReduceSpec {
        match &self.inner {
            ReduceBackwardInner::Linear { geometry, .. } => &geometry.spec,
            ReduceBackwardInner::NonLinear { geometry, .. } => &geometry.spec,
        }
    }
    pub fn upstream_layout(&self) -> &Layout {
        match &self.inner {
            ReduceBackwardInner::Linear { geometry, .. } => &geometry.upstream,
            ReduceBackwardInner::NonLinear { geometry, .. } => &geometry.upstream,
        }
    }
    pub fn destination_layout(&self) -> &Layout {
        match &self.inner {
            ReduceBackwardInner::Linear { geometry, .. } => &geometry.map.destination,
            ReduceBackwardInner::NonLinear { geometry, .. } => &geometry.destination,
        }
    }
    pub fn host_metadata_bytes(&self) -> usize {
        match &self.inner {
            ReduceBackwardInner::Linear { geometry, .. } => geometry.metadata_bytes(),
            ReduceBackwardInner::NonLinear { geometry, .. } => geometry.metadata_bytes(),
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        match &self.inner {
            ReduceBackwardInner::Linear { core, .. } => core.workspace,
            ReduceBackwardInner::NonLinear { core, .. } => core.workspace,
        }
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        match &self.inner {
            ReduceBackwardInner::Linear { core, .. } => core.shared,
            ReduceBackwardInner::NonLinear { core, .. } => core.shared,
        }
    }
    pub fn run(
        &self,
        upstream: &DeviceView<'_>,
        destination: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        let ReduceBackwardInner::Linear { geometry, core } = &self.inner else {
            return Err(invalid(
                "CUDA Max, Min, and LogSumExp backward require source input via run_with_input",
            ));
        };
        core.cuda.owns(upstream.tensor)?;
        core.cuda.owns(destination.tensor)?;
        geometry.validate(upstream.layout(), destination.layout())?;
        if geometry.map.len == 0 {
            return Ok(());
        }
        // SAFETY: checked broadcast map and initialized source; the exclusive injective
        // destination receives one copy/division and at most one addition per coordinate.
        check(unsafe {
            fa_cuda_reduce_backward_run(
                core.ptr.as_ptr(),
                upstream.tensor.read(),
                destination.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
    pub fn run_with_input(
        &self,
        source: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        destination: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        match &self.inner {
            ReduceBackwardInner::Linear { .. } => self.run(upstream, destination, mode),
            ReduceBackwardInner::NonLinear { geometry, core } => {
                let src = source.ok_or_else(|| {
                    invalid("CUDA Max, Min, and LogSumExp backward require saved forward source")
                })?;
                core.cuda.owns(src.tensor)?;
                core.cuda.owns(upstream.tensor)?;
                core.cuda.owns(destination.tensor)?;
                geometry.validate(src.layout(), upstream.layout(), destination.layout())?;
                if geometry.spec.output_shape().is_empty()
                    || geometry.spec.reduction_len().unwrap_or(0) == 0
                {
                    return Ok(());
                }
                // SAFETY: validated session, layouts, and exclusive injective output; enqueues on stream.
                check(unsafe {
                    fa_cuda_reduce_nl_bwd_run(
                        core.ptr.as_ptr(),
                        src.tensor.read(),
                        upstream.tensor.read(),
                        destination.tensor.write(),
                        i32::from(mode == GradientMode::Accumulate),
                    )
                })
            }
        }
    }
}
