use super::{Cuda, check, nonnull};
use crate::{Error, Result};
use std::{
    ffi::{CStr, c_char, c_void},
    ptr::NonNull,
};

/// cuDNN convolution engine selection, independent of the strict-FP32 policy.
/// This does not promise bit identity across different engines, devices or SDKs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConvolutionPolicy {
    /// Preserve heuristic selection, including engines marked nondeterministic.
    #[default]
    AllowNonDeterministic,
    /// Reject nondeterministic engines in both fused and unfused searches.
    RequireDeterministic,
}
impl ConvolutionPolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::AllowNonDeterministic => "allow_nondeterministic",
            Self::RequireDeterministic => "require_deterministic",
        }
    }
    pub(super) fn native(self) -> i32 {
        i32::from(self == Self::RequireDeterministic)
    }
}

/// Actual selected cuDNN convolution engine properties, queried during preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudnnConvolutionInfo {
    pub workspace_bytes: usize,
    pub engine_index: i64,
    /// Bit positions use the installed cuDNN `cudnnBackendNumericalNote_t` values.
    pub numerical_notes: u64,
    /// The selected engine has no `NONDETERMINISTIC` numerical note.
    pub deterministic: bool,
    pub fused_epilogue: bool,
}
#[repr(C)]
#[derive(Default)]
struct NativeInfo {
    workspace_bytes: usize,
    engine_index: i64,
    numerical_notes: u64,
    deterministic: i32,
    fused_epilogue: i32,
}
const _: () = assert!(std::mem::size_of::<NativeInfo>() == 32);
const _: () = assert!(std::mem::offset_of!(NativeInfo, numerical_notes) == 16);
const _: () = assert!(std::mem::offset_of!(NativeInfo, deterministic) == 24);
const _: () = assert!(std::mem::offset_of!(NativeInfo, fused_epilogue) == 28);
unsafe extern "C" {
    fn fa_cudnn_conv_info(plan: *mut c_void, info: *mut NativeInfo, name: *mut *const c_char);
    fn fa_cudnn_conv_destroy(plan: *mut c_void);
}
pub(super) struct CudnnPlan {
    _cuda: Cuda,
    pub ptr: NonNull<c_void>,
    pub info: CudnnConvolutionInfo,
    pub implementation: String,
}
impl CudnnPlan {
    pub fn check_preparation(status: i32, policy: ConvolutionPolicy, limit: usize) -> Result<()> {
        if status == -1 {
            // FA_CUDNN_NO_CONV_PLAN in native/cudnn_plan.h.
            return Err(Error::Unsupported(format!(
                "no strict-FP32 cuDNN convolution plan satisfies {} with workspace <= {limit} bytes",
                policy.name()
            )));
        }
        check(status)
    }
    /// # Safety
    /// `ptr` must be an unowned successful cuDNN convolution creation result.
    pub unsafe fn from_raw(cuda: &Cuda, ptr: *mut c_void) -> Result<Self> {
        let ptr = nonnull(ptr)?;
        let mut native = NativeInfo::default();
        let mut name = std::ptr::null();
        // SAFETY: caller transfers a live native plan; metadata outputs are initialized.
        unsafe { fa_cudnn_conv_info(ptr.as_ptr(), &mut native, &mut name) };
        // SAFETY: the live native plan owns a NUL-terminated implementation name.
        let implementation = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            _cuda: cuda.clone(),
            ptr,
            implementation,
            info: CudnnConvolutionInfo {
                workspace_bytes: native.workspace_bytes,
                engine_index: native.engine_index,
                numerical_notes: native.numerical_notes,
                deterministic: native.deterministic != 0,
                fused_epilogue: native.fused_epilogue != 0,
            },
        })
    }
}
impl Drop for CudnnPlan {
    fn drop(&mut self) {
        // SAFETY: one owner retains the session; native waits before releasing resources.
        unsafe { fa_cudnn_conv_destroy(self.ptr.as_ptr()) }
    }
}
