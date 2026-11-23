use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    CenteredStftSpec, GradientMode, Layout, Result, StftSpec, Tensor,
    error::invalid,
    stft::{View2, View4, resolve_window},
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeView2 {
    origin: i64,
    strides: [i64; 2],
    contiguous: i32,
    reserved: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeView4 {
    origin: i64,
    strides: [i64; 4],
    contiguous: i32,
    direct_r2c: i32,
}

const _: () = assert!(
    size_of::<NativeView2>() == 32
        && std::mem::offset_of!(NativeView2, strides) == 8
        && std::mem::offset_of!(NativeView2, contiguous) == 24
        && size_of::<NativeView4>() == 48
        && std::mem::offset_of!(NativeView4, strides) == 8
        && std::mem::offset_of!(NativeView4, contiguous) == 40
        && std::mem::offset_of!(NativeView4, direct_r2c) == 44
);

impl From<View2> for NativeView2 {
    fn from(v: View2) -> Self {
        Self {
            origin: v.origin as i64,
            strides: [v.strides[0] as i64, v.strides[1] as i64],
            contiguous: i32::from(v.contiguous),
            reserved: 0,
        }
    }
}

impl From<View4> for NativeView4 {
    fn from(v: View4) -> Self {
        Self {
            origin: v.origin as i64,
            strides: [
                v.strides[0] as i64,
                v.strides[1] as i64,
                v.strides[2] as i64,
                v.strides[3] as i64,
            ],
            contiguous: i32::from(v.contiguous),
            direct_r2c: i32::from(v.direct_r2c),
        }
    }
}

unsafe extern "C" {
    fn fa_cuda_stft_plan_create(
        ctx: *mut c_void,
        batch: i32,
        input_len: i32,
        n: i32,
        hop: i32,
        padding: i32,
        frames: i32,
        x_view: NativeView2,
        y_view: NativeView4,
        window: *const f32,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_stft_plan_run(plan: *mut c_void, x: *const f32, out: *mut f32) -> i32;
    fn fa_cuda_stft_plan_destroy(plan: *mut c_void);

    fn fa_cuda_stft_bwd_create(
        ctx: *mut c_void,
        batch: i32,
        input_len: i32,
        n: i32,
        hop: i32,
        padding: i32,
        frames: i32,
        dy_view: NativeView4,
        dx_view: NativeView2,
        window: *const f32,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_stft_bwd_run(
        plan: *mut c_void,
        dy: *const f32,
        dx: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_stft_bwd_destroy(plan: *mut c_void);

    fn fa_cuda_istft_plan_create(
        ctx: *mut c_void,
        batch: i32,
        output_len: i32,
        n: i32,
        hop: i32,
        padding: i32,
        frames: i32,
        s_view: NativeView4,
        y_view: NativeView2,
        window: *const f32,
        normalization: *const f32,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_istft_plan_run(plan: *mut c_void, input: *const f32, out: *mut f32) -> i32;
    fn fa_cuda_istft_plan_destroy(plan: *mut c_void);

    fn fa_cuda_istft_bwd_create(
        ctx: *mut c_void,
        batch: i32,
        output_len: i32,
        n: i32,
        hop: i32,
        padding: i32,
        frames: i32,
        dy_view: NativeView2,
        ds_view: NativeView4,
        window: *const f32,
        normalization: *const f32,
        workspace: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_istft_bwd_run(
        plan: *mut c_void,
        dy: *const f32,
        ds: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_istft_bwd_destroy(plan: *mut c_void);
}

struct ForwardHandle {
    cuda: Cuda,
    spec: StftSpec,
    input_layout: Layout,
    output_layout: Layout,
    ptr: NonNull<c_void>,
    workspace: usize,
}

impl ForwardHandle {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cuda: &Cuda,
        spec: StftSpec,
        frames: usize,
        padding: usize,
        window: &[f32],
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let x_view = View2::from_layout(input, spec.input_shape(), false)?;
        let y_view = View4::from_layout(output, [spec.batch, frames, spec.bins(), 2], true)?;
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: validated geometry bounds all i32 sizes and layouts; native copies the
        // length-n window synchronously and returns one owned plan handle + workspace size.
        check(unsafe {
            fa_cuda_stft_plan_create(
                cuda.session.ptr(),
                spec.batch as i32,
                spec.input_len as i32,
                spec.fft_size as i32,
                spec.hop as i32,
                padding as i32,
                frames as i32,
                x_view.into(),
                y_view.into(),
                window.as_ptr(),
                &mut workspace,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            input_layout: input.clone(),
            output_layout: output.clone(),
            ptr: nonnull(ptr)?,
            workspace,
        })
    }

    fn run(&mut self, input: &DeviceView<'_>, output: &mut DeviceViewMut<'_>) -> Result<()> {
        self.cuda.owns(input.tensor)?;
        self.cuda.owns(output.tensor)?;
        if input.layout() != &self.input_layout {
            return Err(invalid("CUDA STFT input layout differs from prepared plan"));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "CUDA STFT output layout differs from prepared plan",
            ));
        }
        // SAFETY: matching session, exact validated layouts, exclusive injective destination;
        // enqueues strictly on the session stream with zero hot allocations.
        check(unsafe {
            fa_cuda_stft_plan_run(
                self.ptr.as_ptr(),
                input.tensor.read(),
                output.tensor.write(),
            )
        })
    }
}

impl Drop for ForwardHandle {
    fn drop(&mut self) {
        // SAFETY: sole plan owner retains the CUDA session; native syncs before destroying.
        unsafe { fa_cuda_stft_plan_destroy(self.ptr.as_ptr()) }
    }
}

struct BackwardHandle {
    cuda: Cuda,
    spec: StftSpec,
    upstream_layout: Layout,
    output_layout: Layout,
    ptr: NonNull<c_void>,
    workspace: usize,
}

impl BackwardHandle {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cuda: &Cuda,
        spec: StftSpec,
        frames: usize,
        padding: usize,
        window: &[f32],
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let dy_view = View4::from_layout(upstream, [spec.batch, frames, spec.bins(), 2], false)?;
        let dx_view = View2::from_layout(output, spec.input_shape(), true)?;
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: validated geometry bounds all i32 sizes and layouts; native copies the
        // length-n window synchronously and returns one owned plan handle + workspace size.
        check(unsafe {
            fa_cuda_stft_bwd_create(
                cuda.session.ptr(),
                spec.batch as i32,
                spec.input_len as i32,
                spec.fft_size as i32,
                spec.hop as i32,
                padding as i32,
                frames as i32,
                dy_view.into(),
                dx_view.into(),
                window.as_ptr(),
                &mut workspace,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            ptr: nonnull(ptr)?,
            workspace,
        })
    }

    fn run(
        &mut self,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns(upstream.tensor)?;
        self.cuda.owns(output.tensor)?;
        if upstream.layout() != &self.upstream_layout {
            return Err(invalid(
                "CUDA STFT backward upstream layout differs from prepared plan",
            ));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "CUDA STFT backward output layout differs from prepared plan",
            ));
        }
        // SAFETY: matching session, exact validated layouts, exclusive injective destination;
        // deterministic zero-atomic kernels enqueue strictly on the session stream.
        check(unsafe {
            fa_cuda_stft_bwd_run(
                self.ptr.as_ptr(),
                upstream.tensor.read(),
                output.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

impl Drop for BackwardHandle {
    fn drop(&mut self) {
        // SAFETY: sole plan owner retains the CUDA session; native syncs before destroying.
        unsafe { fa_cuda_stft_bwd_destroy(self.ptr.as_ptr()) }
    }
}

/// Prepared strided uncentered STFT forward plan on CUDA.
pub struct StftPlan {
    handle: ForwardHandle,
}

impl StftPlan {
    pub fn new(cuda: &Cuda, spec: StftSpec, input: &Layout, output: &Layout) -> Result<Self> {
        Self::with_window_slice(cuda, spec, None, input, output)
    }
    pub fn with_window(
        cuda: &Cuda,
        spec: StftSpec,
        window: &Tensor,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.fft_size])?;
        Self::with_window_slice(cuda, spec, Some(window.data()), input, output)
    }
    pub(crate) fn with_window_slice(
        cuda: &Cuda,
        spec: StftSpec,
        window: Option<&[f32]>,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let frames = spec.frames()?;
        let coeffs = resolve_window(spec, window)?;
        Ok(Self {
            handle: ForwardHandle::new(cuda, spec, frames, 0, &coeffs, input, output)?,
        })
    }
    pub fn spec(&self) -> StftSpec {
        self.handle.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.handle.input_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.handle.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.handle.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(&mut self, input: &DeviceView<'_>, output: &mut DeviceViewMut<'_>) -> Result<()> {
        self.handle.run(input, output)
    }
}

/// Prepared strided uncentered STFT local VJP on CUDA (`input_required() == false`).
pub struct StftBackward {
    handle: BackwardHandle,
}

impl StftBackward {
    pub fn new(cuda: &Cuda, spec: StftSpec, upstream: &Layout, output: &Layout) -> Result<Self> {
        Self::with_window_slice(cuda, spec, None, upstream, output)
    }
    pub fn with_window(
        cuda: &Cuda,
        spec: StftSpec,
        window: &Tensor,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.fft_size])?;
        Self::with_window_slice(cuda, spec, Some(window.data()), upstream, output)
    }
    pub(crate) fn with_window_slice(
        cuda: &Cuda,
        spec: StftSpec,
        window: Option<&[f32]>,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let frames = spec.frames()?;
        let coeffs = resolve_window(spec, window)?;
        Ok(Self {
            handle: BackwardHandle::new(cuda, spec, frames, 0, &coeffs, upstream, output)?,
        })
    }
    pub fn spec(&self) -> StftSpec {
        self.handle.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.handle.upstream_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.handle.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.handle.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(
        &mut self,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.handle.run(upstream, output, mode)
    }
}

/// Prepared strided reflect-centered STFT forward plan on CUDA.
pub struct CenteredStftPlan {
    spec: CenteredStftSpec,
    handle: ForwardHandle,
}

impl CenteredStftPlan {
    pub fn new(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        Self::with_window_slice(cuda, spec, None, input, output)
    }
    pub fn with_window(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        window: &Tensor,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.spec().fft_size])?;
        Self::with_window_slice(cuda, spec, Some(window.data()), input, output)
    }
    pub(crate) fn with_window_slice(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        window: Option<&[f32]>,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let coeffs = resolve_window(spec.spec(), window)?;
        Ok(Self {
            spec,
            handle: ForwardHandle::new(
                cuda,
                spec.spec(),
                spec.frames(),
                spec.padding(),
                &coeffs,
                input,
                output,
            )?,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.handle.input_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.handle.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.handle.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(&mut self, input: &DeviceView<'_>, output: &mut DeviceViewMut<'_>) -> Result<()> {
        self.handle.run(input, output)
    }
}

/// Prepared strided reflect-centered STFT local VJP on CUDA (`input_required() == false`).
pub struct CenteredStftBackward {
    spec: CenteredStftSpec,
    handle: BackwardHandle,
}

impl CenteredStftBackward {
    pub fn new(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        Self::with_window_slice(cuda, spec, None, upstream, output)
    }
    pub fn with_window(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        window: &Tensor,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.spec().fft_size])?;
        Self::with_window_slice(cuda, spec, Some(window.data()), upstream, output)
    }
    pub(crate) fn with_window_slice(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        window: Option<&[f32]>,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let coeffs = resolve_window(spec.spec(), window)?;
        Ok(Self {
            spec,
            handle: BackwardHandle::new(
                cuda,
                spec.spec(),
                spec.frames(),
                spec.padding(),
                &coeffs,
                upstream,
                output,
            )?,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.handle.upstream_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.handle.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.handle.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(
        &mut self,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.handle.run(upstream, output, mode)
    }
}

/// Prepared strided centered inverse STFT forward plan on CUDA.
pub struct IstftPlan {
    cuda: Cuda,
    spec: CenteredStftSpec,
    input_layout: Layout,
    output_layout: Layout,
    ptr: NonNull<c_void>,
    workspace: usize,
}

impl IstftPlan {
    pub fn new(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        Self::with_window_slice(cuda, spec, None, input, output)
    }
    pub fn with_window(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        window: &Tensor,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.spec().fft_size])?;
        Self::with_window_slice(cuda, spec, Some(window.data()), input, output)
    }
    pub(crate) fn with_window_slice(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        window: Option<&[f32]>,
        input: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let s = spec.spec();
        let s_view = View4::from_layout(input, spec.output_shape(), false)?;
        let y_view = View2::from_layout(output, spec.input_shape(), true)?;
        let coeffs = resolve_window(s, window)?;
        let normalization = spec.normalization(&coeffs)?;
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: validated geometry bounds all i32 sizes and layouts; native copies both host
        // slices synchronously and returns one owned plan handle + workspace byte count.
        check(unsafe {
            fa_cuda_istft_plan_create(
                cuda.session.ptr(),
                s.batch as i32,
                s.input_len as i32,
                s.fft_size as i32,
                s.hop as i32,
                spec.padding() as i32,
                spec.frames() as i32,
                s_view.into(),
                y_view.into(),
                coeffs.as_ptr(),
                normalization.as_ptr(),
                &mut workspace,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            input_layout: input.clone(),
            output_layout: output.clone(),
            ptr: nonnull(ptr)?,
            workspace,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.input_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(&mut self, input: &DeviceView<'_>, output: &mut DeviceViewMut<'_>) -> Result<()> {
        self.cuda.owns(input.tensor)?;
        self.cuda.owns(output.tensor)?;
        if input.layout() != &self.input_layout {
            return Err(invalid(
                "CUDA iSTFT input layout differs from prepared plan",
            ));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "CUDA iSTFT output layout differs from prepared plan",
            ));
        }
        // SAFETY: matching session, validated layouts, exclusive injective destination;
        // enqueues strictly on the session stream with zero hot allocations.
        check(unsafe {
            fa_cuda_istft_plan_run(
                self.ptr.as_ptr(),
                input.tensor.read(),
                output.tensor.write(),
            )
        })
    }
}

impl Drop for IstftPlan {
    fn drop(&mut self) {
        // SAFETY: sole plan owner retains the CUDA session; native syncs before destroying.
        unsafe { fa_cuda_istft_plan_destroy(self.ptr.as_ptr()) }
    }
}

/// Prepared strided centered inverse STFT local VJP on CUDA (`input_required() == false`).
pub struct IstftBackward {
    cuda: Cuda,
    spec: CenteredStftSpec,
    upstream_layout: Layout,
    output_layout: Layout,
    ptr: NonNull<c_void>,
    workspace: usize,
}

impl IstftBackward {
    pub fn new(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        Self::with_window_slice(cuda, spec, None, upstream, output)
    }
    pub fn with_window(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        window: &Tensor,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        window.shape().expect(&[spec.spec().fft_size])?;
        Self::with_window_slice(cuda, spec, Some(window.data()), upstream, output)
    }
    pub(crate) fn with_window_slice(
        cuda: &Cuda,
        spec: CenteredStftSpec,
        window: Option<&[f32]>,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let s = spec.spec();
        let dy_view = View2::from_layout(upstream, spec.input_shape(), false)?;
        let ds_view = View4::from_layout(output, spec.output_shape(), true)?;
        let coeffs = resolve_window(s, window)?;
        let normalization = spec.normalization(&coeffs)?;
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        // SAFETY: validated geometry bounds all i32 sizes and layouts; native copies both host
        // slices synchronously and returns one owned plan handle + workspace byte count.
        check(unsafe {
            fa_cuda_istft_bwd_create(
                cuda.session.ptr(),
                s.batch as i32,
                s.input_len as i32,
                s.fft_size as i32,
                s.hop as i32,
                spec.padding() as i32,
                spec.frames() as i32,
                dy_view.into(),
                ds_view.into(),
                coeffs.as_ptr(),
                normalization.as_ptr(),
                &mut workspace,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            ptr: nonnull(ptr)?,
            workspace,
        })
    }
    pub fn spec(&self) -> CenteredStftSpec {
        self.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.upstream_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        0
    }
    pub fn run(
        &mut self,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns(upstream.tensor)?;
        self.cuda.owns(output.tensor)?;
        if upstream.layout() != &self.upstream_layout {
            return Err(invalid(
                "CUDA iSTFT backward upstream layout differs from prepared plan",
            ));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "CUDA iSTFT backward output layout differs from prepared plan",
            ));
        }
        // SAFETY: matching session, validated layouts, exclusive injective destination;
        // deterministic zero-atomic kernels enqueue strictly on the session stream.
        check(unsafe {
            fa_cuda_istft_bwd_run(
                self.ptr.as_ptr(),
                upstream.tensor.read(),
                output.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

impl Drop for IstftBackward {
    fn drop(&mut self) {
        // SAFETY: sole plan owner retains the CUDA session; native syncs before destroying.
        unsafe { fa_cuda_istft_bwd_destroy(self.ptr.as_ptr()) }
    }
}
