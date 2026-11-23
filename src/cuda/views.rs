use super::{Cuda, DeviceTensor, check, nonnull};
use crate::{Error, Layout, Result, copy::MappingSpec, error::int};
use std::{ffi::c_void, ptr::NonNull};

/// Borrowed device storage; the prepared layout never exposes a host reference.
#[derive(Clone, Copy)]
pub struct DeviceView<'a> {
    pub(super) tensor: &'a DeviceTensor,
    layout: &'a Layout,
}
impl DeviceView<'_> {
    pub fn layout(&self) -> &Layout {
        self.layout
    }
}
/// Exclusive device view with an injective layout and the original session lifetime.
pub struct DeviceViewMut<'a> {
    pub(super) tensor: &'a mut DeviceTensor,
    layout: &'a Layout,
}
impl DeviceViewMut<'_> {
    pub fn layout(&self) -> &Layout {
        self.layout
    }
    pub fn as_view(&self) -> DeviceView<'_> {
        DeviceView {
            tensor: self.tensor,
            layout: self.layout,
        }
    }
}
impl DeviceTensor {
    pub fn view<'a>(&'a self, layout: &'a Layout) -> Result<DeviceView<'a>> {
        layout.validate_storage(self.shape.len())?;
        Ok(DeviceView {
            tensor: self,
            layout,
        })
    }
    pub fn view_mut<'a>(&'a mut self, layout: &'a Layout) -> Result<DeviceViewMut<'a>> {
        layout.validate_storage(self.shape.len())?;
        if !layout.is_non_overlapping() {
            return Err(Error::Unsupported(
                "mutable device views require a proved non-overlapping layout".into(),
            ));
        }
        Ok(DeviceViewMut {
            tensor: self,
            layout,
        })
    }
}
#[repr(C)]
struct NativeCopy {
    len: u64,
    source_offset: i64,
    destination_offset: i64,
    rank: i32,
    contiguous: i32,
}
const _: () = assert!(std::mem::size_of::<NativeCopy>() == 32);
const _: () = assert!(std::mem::offset_of!(NativeCopy, rank) == 24);
unsafe extern "C" {
    fn fa_cuda_tensor_copy_create(
        ctx: *mut c_void,
        geometry: NativeCopy,
        dims: *const u64,
        sx: *const i64,
        sy: *const i64,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_tensor_copy_run(plan: *mut c_void, source: *const f32, destination: *mut f32)
    -> i32;
    fn fa_cuda_tensor_copy_destroy(plan: *mut c_void);
}

/// Prepared same-stream layout copy; only metadata is uploaded during preparation.
pub struct CopyPlan {
    cuda: Cuda,
    spec: MappingSpec,
    ptr: NonNull<c_void>,
    workspace: usize,
}
impl CopyPlan {
    pub fn new(cuda: &Cuda, source: &Layout, destination: &Layout) -> Result<Self> {
        let spec = MappingSpec::for_copy(source, destination)?;
        let geometry = NativeCopy {
            len: spec.len as u64,
            source_offset: source.offset() as i64,
            destination_offset: destination.offset() as i64,
            rank: int(spec.dims.len())?,
            contiguous: i32::from(spec.contiguous),
        };
        let dims: Vec<_> = spec.dims.iter().map(|&x| x as u64).collect();
        let sx: Vec<_> = spec.source_strides.iter().map(|&x| x as i64).collect();
        let sy: Vec<_> = spec.destination_strides.iter().map(|&x| x as i64).collect();
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: both complete spans were checked and destination injectivity was proved.
        // Geometry has the asserted C layout; native copies and synchronizes host metadata
        // before these vectors can expire, and retains the session through this owner.
        check(unsafe {
            fa_cuda_tensor_copy_create(
                cuda.session.ptr(),
                geometry,
                dims.as_ptr(),
                sx.as_ptr(),
                sy.as_ptr(),
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
    pub fn source_layout(&self) -> &Layout {
        &self.spec.source
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.spec.destination
    }
    /// Device descriptor arrays; no payload staging or per-call scratch is allocated.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
    }
    pub fn run(&self, source: &DeviceView<'_>, destination: &mut DeviceViewMut<'_>) -> Result<()> {
        self.cuda.owns(source.tensor)?;
        self.cuda.owns(destination.tensor)?;
        self.spec.validate(source.layout, destination.layout)?;
        if self.spec.len == 0 {
            return Ok(());
        }
        // SAFETY: bound views cover the prepared spans and share the retained session.
        // The exclusive output allocation cannot alias the shared input. One stream
        // orders data/descriptor uses; native destroys metadata only after completion.
        check(unsafe {
            fa_cuda_tensor_copy_run(
                self.ptr.as_ptr(),
                source.tensor.read(),
                destination.tensor.write(),
            )
        })
    }
}
impl Drop for CopyPlan {
    fn drop(&mut self) {
        // SAFETY: one native owner retains its context; destruction waits for pending uses.
        unsafe { fa_cuda_tensor_copy_destroy(self.ptr.as_ptr()) }
    }
}
