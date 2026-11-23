use super::{Cuda, DeviceTensor, DeviceView, check, nonnull};
use crate::{
    GradientMode, Layout, Result, Shape, ViewBackwardStrategy,
    adjoint::{BackwardSpec, Route, limit_index},
    error::{int, invalid},
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
struct NativeBackward {
    base_len: u64,
    logical_len: u64,
    groups: u64,
    reductions: u64,
    written_begin: u64,
    written_end: u64,
    source_offset: i64,
    destination_offset: i64,
    rank: i32,
    reduction_rank: i32,
    kind: i32,
    lanes: i32,
}
const _: () = assert!(size_of::<NativeBackward>() == 80);
const _: () = assert!(std::mem::offset_of!(NativeBackward, rank) == 64);
const _: () = assert!(std::mem::offset_of!(NativeBackward, kind) == 72);
unsafe extern "C" {
    fn fa_cuda_view_backward_create(
        ctx: *mut c_void,
        geometry: NativeBackward,
        a: *const u64,
        b: *const u64,
        c: *const u64,
        d: *const u64,
        e: *const u64,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_view_backward_run(
        plan: *mut c_void,
        source: *const f32,
        destination: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_view_backward_destroy(plan: *mut c_void);
}

/// Prepared device view adjoint with immutable, same-stream geometry/index storage.
pub struct ViewBackward {
    cuda: Cuda,
    base: Shape,
    view: Layout,
    upstream: Layout,
    ptr: NonNull<c_void>,
    strategy: ViewBackwardStrategy,
    lanes: usize,
    index_bytes: usize,
    preparation_index_bytes: usize,
    workspace: usize,
    shared: usize,
}
impl ViewBackward {
    pub fn new(
        cuda: &Cuda,
        base_shape: impl AsRef<[usize]>,
        view: &Layout,
        upstream: &Layout,
        index_limit_bytes: usize,
    ) -> Result<Self> {
        let spec = BackwardSpec::new(base_shape, view, upstream, index_limit_bytes)?;
        let upload_index = spec
            .index_bytes
            .checked_mul(2)
            .ok_or_else(|| invalid("view backward index upload bytes overflow"))?;
        let preparation_index_bytes = spec.preparation_index_bytes.max(upload_index);
        limit_index(preparation_index_bytes, index_limit_bytes)?;
        let strategy = spec.strategy();
        let lanes = spec.reduction_lanes();
        let (begin, end) = spec.written.unwrap_or((0, 0));
        let mut geometry = NativeBackward {
            base_len: spec.base.len() as u64,
            logical_len: spec.map.len as u64,
            groups: spec.groups() as u64,
            reductions: 0,
            written_begin: begin as u64,
            written_end: end as u64,
            source_offset: spec.map.source.offset() as i64,
            destination_offset: spec.map.destination.offset() as i64,
            rank: 0,
            reduction_rank: 0,
            kind: strategy as i32,
            lanes: int(lanes)?,
        };
        // Temporary descriptor conversions are O(rank), independent of index-table size.
        let mut descriptors = Vec::<Vec<u64>>::new();
        let mut arrays: [&[u64]; 5] = [&[]; 5];
        match &spec.route {
            Route::Injective if !spec.map.contiguous => {
                geometry.rank = int(spec.map.dims.len())?;
                descriptors.push(spec.map.dims.iter().map(|&d| d as u64).collect());
                descriptors.push(spec.map.source_strides.iter().map(|&s| s as u64).collect());
                descriptors.push(
                    spec.map
                        .destination_strides
                        .iter()
                        .map(|&s| s as u64)
                        .collect(),
                );
            }
            Route::Broadcast {
                unique, reduction, ..
            } => {
                geometry.rank = int(unique.dims.len())?;
                geometry.reduction_rank = int(reduction.axes.len())?;
                geometry.reductions = reduction.len as u64;
                descriptors.push(unique.dims.iter().map(|&d| d as u64).collect());
                descriptors.push(unique.source_strides.iter().map(|&s| s as u64).collect());
                descriptors.push(
                    unique
                        .destination_strides
                        .iter()
                        .map(|&s| s as u64)
                        .collect(),
                );
                descriptors.push(reduction.axes.iter().map(|&(d, _)| d as u64).collect());
                descriptors.push(reduction.axes.iter().map(|&(_, s)| s as u64).collect());
            }
            Route::Indexed { targets, sources } => {
                arrays[0] = targets;
                arrays[1] = sources;
            }
            Route::Grouped {
                targets,
                offsets,
                sources,
                ..
            } => {
                arrays[0] = targets;
                arrays[1] = offsets;
                arrays[2] = sources;
            }
            _ => {}
        }
        for (slot, values) in arrays.iter_mut().zip(&descriptors) {
            *slot = values;
        }
        let mut ptr = std::ptr::null_mut();
        let (mut workspace, mut shared) = (0, 0);
        // SAFETY: geometry derives from checked spans/unique target counts; CSR groups
        // are sorted and bounded. The asserted header matches native. Native copies
        // and synchronizes every borrowed metadata array before returning, even on error.
        check(unsafe {
            fa_cuda_view_backward_create(
                cuda.session.ptr(),
                geometry,
                arrays[0].as_ptr(),
                arrays[1].as_ptr(),
                arrays[2].as_ptr(),
                arrays[3].as_ptr(),
                arrays[4].as_ptr(),
                &mut workspace,
                &mut shared,
                &mut ptr,
            )
        })?;
        let index_bytes = spec.index_bytes;
        let BackwardSpec { base, map, .. } = spec;
        Ok(Self {
            cuda: cuda.clone(),
            base,
            view: map.destination,
            upstream: map.source,
            ptr: nonnull(ptr)?,
            strategy,
            lanes,
            index_bytes,
            preparation_index_bytes,
            workspace,
            shared,
        })
    }
    pub fn base_shape(&self) -> &[usize] {
        self.base.dims()
    }
    pub fn view_layout(&self) -> &Layout {
        &self.view
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.upstream
    }
    pub fn strategy(&self) -> ViewBackwardStrategy {
        self.strategy
    }
    pub fn reduction_lanes(&self) -> usize {
        self.lanes
    }
    pub fn index_bytes(&self) -> usize {
        self.index_bytes
    }
    pub fn preparation_index_bytes(&self) -> usize {
        self.preparation_index_bytes
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of_val(self.base.dims())
            + (self.view.shape().dims().len() + self.upstream.shape().dims().len())
                * (size_of::<usize>() + size_of::<isize>())
    }
    /// Persistent device coordinate/index metadata; execution uses no global scratch.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.shared
    }
    pub fn run(
        &self,
        upstream: &DeviceView<'_>,
        out: &mut DeviceTensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns(upstream.tensor)?;
        self.cuda.owns(out)?;
        out.shape().expect(self.base.dims())?;
        if !(std::ptr::eq(upstream.layout(), &self.upstream) || upstream.layout() == &self.upstream)
        {
            return Err(invalid(
                "view backward upstream geometry differs from the prepared layout",
            ));
        }
        if self.base.is_empty()
            || (self.strategy == ViewBackwardStrategy::Empty && mode == GradientMode::Accumulate)
        {
            return Ok(());
        }
        // SAFETY: bound upstream covers its prepared span, the dense output is exclusive
        // and exactly shaped, and both buffers share the retained session. Each selected
        // target has one writer, including grouped reductions; the stream orders uses.
        check(unsafe {
            fa_cuda_view_backward_run(
                self.ptr.as_ptr(),
                upstream.tensor.read(),
                out.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}
impl Drop for ViewBackward {
    fn drop(&mut self) {
        // SAFETY: one native plan owner retains its context; native waits before freeing metadata.
        unsafe { fa_cuda_view_backward_destroy(self.ptr.as_ptr()) }
    }
}
