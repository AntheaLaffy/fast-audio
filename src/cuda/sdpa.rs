use super::{Cuda, DeviceView, DeviceViewMut, check, nonnull};
use crate::{GradientMode, Layout, Result, SdpaOperand, SdpaSpec, error::invalid, sdpa::SdpaView4};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeSdpaView4 {
    offset: i64,
    strides: [i64; 4],
}

impl From<SdpaView4> for NativeSdpaView4 {
    fn from(v: SdpaView4) -> Self {
        Self {
            offset: v.offset as i64,
            strides: [
                v.strides[0] as i64,
                v.strides[1] as i64,
                v.strides[2] as i64,
                v.strides[3] as i64,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeSdpaDims {
    batch: u64,
    num_heads: u64,
    num_kv_heads: u64,
    seq_q: u64,
    seq_k: u64,
    head_dim: u64,
    scale: f32,
    is_causal: i32,
    operand: i32,
    fast_inner: i32,
    q: NativeSdpaView4,
    k: NativeSdpaView4,
    v: NativeSdpaView4,
    dy_or_o: NativeSdpaView4,
    out: NativeSdpaView4,
}

const _: () = assert!(
    size_of::<NativeSdpaView4>() == 40
        && size_of::<NativeSdpaDims>() == 264
        && std::mem::offset_of!(NativeSdpaDims, scale) == 48
        && std::mem::offset_of!(NativeSdpaDims, q) == 64
);

unsafe extern "C" {
    fn fa_cuda_sdpa_create(
        ctx: *mut c_void,
        dims: NativeSdpaDims,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_sdpa_forward_run(
        plan: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
    ) -> i32;
    fn fa_cuda_sdpa_backward_run(
        plan: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        dy: *const f32,
        out: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_sdpa_destroy(plan: *mut c_void);
}

struct Core {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    workspace: usize,
    shared: usize,
}

impl Core {
    fn new(cuda: &Cuda, dims: NativeSdpaDims) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (mut workspace, mut shared) = (0, 0);
        // SAFETY: asserted ABI sizes and validated geometry; native allocates optional O(B*H*T_q) workspace.
        check(unsafe {
            fa_cuda_sdpa_create(
                cuda.session.ptr(),
                dims,
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
        // SAFETY: sole native plan owner retains CUDA session; native syncs before destroying.
        unsafe { fa_cuda_sdpa_destroy(self.ptr.as_ptr()) }
    }
}

/// Prepared O(T)-memory FlashAttention-2 forward plan on CUDA.
pub struct SdpaPlan {
    spec: SdpaSpec,
    q_layout: Layout,
    k_layout: Layout,
    v_layout: Layout,
    o_layout: Layout,
    core: Core,
}

impl SdpaPlan {
    pub fn new(
        cuda: &Cuda,
        spec: SdpaSpec,
        query: &Layout,
        key: &Layout,
        value: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let qs = spec.query_shape();
        let kvs = spec.kv_shape();
        let qv = SdpaView4::new(query, &qs, false)?;
        let kv = SdpaView4::new(key, &kvs, false)?;
        let vv = SdpaView4::new(value, &kvs, false)?;
        let ov = SdpaView4::new(output, &qs, true)?;
        let d = spec.head_dim();
        let fast_inner = d <= 128
            && qv.inner_contiguous(d)
            && kv.inner_contiguous(d)
            && vv.inner_contiguous(d)
            && ov.inner_contiguous(d);
        let dims = NativeSdpaDims {
            batch: spec.batch() as u64,
            num_heads: spec.num_heads() as u64,
            num_kv_heads: spec.num_kv_heads() as u64,
            seq_q: spec.seq_q() as u64,
            seq_k: spec.seq_k() as u64,
            head_dim: d as u64,
            scale: spec.scale(),
            is_causal: i32::from(spec.is_causal()),
            operand: -1,
            fast_inner: i32::from(fast_inner),
            q: qv.into(),
            k: kv.into(),
            v: vv.into(),
            dy_or_o: ov.into(),
            out: ov.into(),
        };
        let core = Core::new(cuda, dims)?;
        Ok(Self {
            spec,
            q_layout: query.clone(),
            k_layout: key.clone(),
            v_layout: value.clone(),
            o_layout: output.clone(),
            core,
        })
    }

    pub fn spec(&self) -> SdpaSpec {
        self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }

    pub fn run(
        &self,
        query: &DeviceView<'_>,
        key: &DeviceView<'_>,
        value: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
    ) -> Result<()> {
        self.core.cuda.owns(query.tensor)?;
        self.core.cuda.owns(key.tensor)?;
        self.core.cuda.owns(value.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if query.layout() != &self.q_layout
            || key.layout() != &self.k_layout
            || value.layout() != &self.v_layout
            || output.layout() != &self.o_layout
        {
            return Err(invalid("CUDA sdpa layouts differ from prepared plan"));
        }
        // SAFETY: exact session, validated layouts, exclusive injective output; enqueues on stream.
        check(unsafe {
            fa_cuda_sdpa_forward_run(
                self.core.ptr.as_ptr(),
                query.tensor.read(),
                key.tensor.read(),
                value.tensor.read(),
                output.tensor.write(),
            )
        })
    }
}

/// Prepared O(T)-memory FlashAttention-2 local VJP (`dQ`, `dK`, or `dV`) on CUDA.
pub struct SdpaBackward {
    spec: SdpaSpec,
    operand: SdpaOperand,
    q_layout: Layout,
    k_layout: Layout,
    v_layout: Layout,
    dy_layout: Layout,
    out_layout: Layout,
    core: Core,
}

impl SdpaBackward {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cuda: &Cuda,
        operand: SdpaOperand,
        spec: SdpaSpec,
        query: &Layout,
        key: &Layout,
        value: &Layout,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let qs = spec.query_shape();
        let kvs = spec.kv_shape();
        let qv = SdpaView4::new(query, &qs, false)?;
        let kv = SdpaView4::new(key, &kvs, false)?;
        let vv = SdpaView4::new(value, &kvs, false)?;
        let dyv = SdpaView4::new(upstream, &qs, false)?;
        let target_shape = match operand {
            SdpaOperand::Query => &qs,
            SdpaOperand::Key | SdpaOperand::Value => &kvs,
        };
        let outv = SdpaView4::new(output, target_shape, true)?;
        let d = spec.head_dim();
        let fast_inner = d <= 128
            && qv.inner_contiguous(d)
            && kv.inner_contiguous(d)
            && vv.inner_contiguous(d)
            && dyv.inner_contiguous(d)
            && outv.inner_contiguous(d);
        let dims = NativeSdpaDims {
            batch: spec.batch() as u64,
            num_heads: spec.num_heads() as u64,
            num_kv_heads: spec.num_kv_heads() as u64,
            seq_q: spec.seq_q() as u64,
            seq_k: spec.seq_k() as u64,
            head_dim: d as u64,
            scale: spec.scale(),
            is_causal: i32::from(spec.is_causal()),
            operand: operand as i32,
            fast_inner: i32::from(fast_inner),
            q: qv.into(),
            k: kv.into(),
            v: vv.into(),
            dy_or_o: dyv.into(),
            out: outv.into(),
        };
        let core = Core::new(cuda, dims)?;
        Ok(Self {
            spec,
            operand,
            q_layout: query.clone(),
            k_layout: key.clone(),
            v_layout: value.clone(),
            dy_layout: upstream.clone(),
            out_layout: output.clone(),
            core,
        })
    }

    pub fn spec(&self) -> SdpaSpec {
        self.spec
    }
    pub fn operand(&self) -> SdpaOperand {
        self.operand
    }
    pub fn input_dependencies(&self) -> [bool; 3] {
        match self.operand {
            SdpaOperand::Query | SdpaOperand::Key => [true, true, true],
            SdpaOperand::Value => [true, true, false],
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }

    pub fn run(
        &self,
        query: Option<&DeviceView<'_>>,
        key: Option<&DeviceView<'_>>,
        value: Option<&DeviceView<'_>>,
        upstream: &DeviceView<'_>,
        output: &mut DeviceViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        let q = query.ok_or_else(|| invalid("CUDA sdpa backward requires saved query"))?;
        let k = key.ok_or_else(|| invalid("CUDA sdpa backward requires saved key"))?;
        self.core.cuda.owns(q.tensor)?;
        self.core.cuda.owns(k.tensor)?;
        self.core.cuda.owns(upstream.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if q.layout() != &self.q_layout
            || k.layout() != &self.k_layout
            || upstream.layout() != &self.dy_layout
            || output.layout() != &self.out_layout
        {
            return Err(invalid(
                "CUDA sdpa backward layouts differ from prepared plan",
            ));
        }
        let v_ptr = if self.operand != SdpaOperand::Value {
            let v =
                value.ok_or_else(|| invalid("CUDA sdpa dQ/dK backward requires saved value"))?;
            self.core.cuda.owns(v.tensor)?;
            if v.layout() != &self.v_layout {
                return Err(invalid(
                    "CUDA sdpa backward value layout differs from prepared plan",
                ));
            }
            v.tensor.read()
        } else {
            std::ptr::null()
        };
        // SAFETY: exact session, validated layouts, exclusive injective output; enqueues on stream.
        check(unsafe {
            fa_cuda_sdpa_backward_run(
                self.core.ptr.as_ptr(),
                q.tensor.read(),
                k.tensor.read(),
                v_ptr,
                upstream.tensor.read(),
                output.tensor.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}
