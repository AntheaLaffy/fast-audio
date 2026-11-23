//! Explicit mixed-precision (`BF16` / `TF32` / `I64`) CUDA storage, NVIDIA Tensor Core kernels,
//! `O(T)`-memory `mma.m16n8k16` BF16 FlashAttention-2, fused FP32-master/BF16-mirror `AdamW`,
//! device-side zero-host-sync `GradScaler`, and single-CUDA-Graph `Bf16AudioTransformerTrainer`.

use super::{Cuda, DeviceTensor, Session, check, nonnull};
use crate::{
    AdamWConfig, Bf16Tensor, CrossEntropySpec, DType, DropoutSpec, EmbeddingSpec, GradientMode,
    IndexTensor, PrecisionPolicy, Result, RopeSpec, SdpaSpec, Shape, Tensor,
    error::{int, invalid},
};
use std::{ffi::c_void, ptr::NonNull, rc::Rc};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Bf16SdpaStrides {
    pub sb: i64,
    pub sh: i64,
    pub st: i64,
    pub sd: i64,
}

impl Bf16SdpaStrides {
    /// Contiguous `[B, H, T, D]` strides.
    pub fn bhtd(h: usize, t: usize, d: usize) -> Self {
        Self {
            sb: (h * t * d) as i64,
            sh: (t * d) as i64,
            st: d as i64,
            sd: 1,
        }
    }
    /// Contiguous `[B, T, H, D]` strides (zero-copy attention view!).
    pub fn bthd(t: usize, h: usize, d: usize) -> Self {
        Self {
            sb: (t * h * d) as i64,
            sh: d as i64,
            st: (h * d) as i64,
            sd: 1,
        }
    }

    fn check_view(self, dims: [usize; 4], offset: usize, len: usize, writable: bool) -> Result<()> {
        let origin = i64::try_from(offset).map_err(|_| invalid("BF16 SDPA offset overflows"))?;
        let (mut low, mut high) = (origin, origin);
        let strides = [self.sb, self.sh, self.st, self.sd];
        for (dim, stride) in dims.into_iter().zip(strides) {
            let extent = stride
                .checked_mul((dim - 1) as i64)
                .ok_or_else(|| invalid("BF16 SDPA stride extent overflows"))?;
            let bound = if extent < 0 { &mut low } else { &mut high };
            *bound = bound
                .checked_add(extent)
                .ok_or_else(|| invalid("BF16 SDPA address overflows"))?;
        }
        if low < 0 || high as u64 >= len as u64 {
            return Err(invalid("BF16 SDPA view exceeds its device allocation"));
        }
        if writable {
            // In stride order, each nontrivial axis must separate the spans of earlier axes.
            let mut axes = std::array::from_fn::<_, 4, _>(|i| (strides[i].unsigned_abs(), dims[i]));
            axes.sort_unstable();
            let mut span = 1;
            for (stride, dim) in axes {
                if dim > 1 {
                    if stride < span {
                        return Err(invalid("BF16 SDPA output view may overlap"));
                    }
                    span += stride * (dim - 1) as u64;
                }
            }
        }
        Ok(())
    }
}

unsafe extern "C" {
    fn fa_cuda_alloc(ctx: *mut c_void, bytes: usize, out: *mut *mut c_void) -> i32;
    fn fa_cuda_free(ctx: *mut c_void, ptr: *mut c_void);
    fn fa_cuda_copy(
        ctx: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        to_device: i32,
    ) -> i32;
    fn fa_cuda_zero(ctx: *mut c_void, dst: *mut c_void, bytes: usize) -> i32;
    fn fa_cuda_cast_f32_to_bf16(ctx: *mut c_void, n: u64, src: *const f32, dst: *mut u16) -> i32;
    fn fa_cuda_cast_bf16_to_f32(ctx: *mut c_void, n: u64, src: *const u16, dst: *mut f32) -> i32;
    fn fa_cuda_bf16_gemm(
        ctx: *mut c_void,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const u16,
        lda: i32,
        b: *const u16,
        ldb: i32,
        c: *mut c_void,
        ldc: i32,
        c_is_f32: i32,
        alpha: f32,
        beta: f32,
    ) -> i32;
    fn fa_cuda_bf16_bmm_strided(
        ctx: *mut c_void,
        trans_a: i32,
        trans_b: i32,
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const u16,
        lda: i32,
        stride_a: i64,
        b: *const u16,
        ldb: i32,
        stride_b: i64,
        c: *mut c_void,
        ldc: i32,
        stride_c: i64,
        c_is_f32: i32,
        alpha: f32,
        beta: f32,
    ) -> i32;
    fn fa_cuda_tf32_gemm(
        ctx: *mut c_void,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        c: *mut f32,
        ldc: i32,
        alpha: f32,
        beta: f32,
    ) -> i32;
    fn fa_cuda_bf16_rmsnorm_fwd(
        ctx: *mut c_void,
        rows: u64,
        cols: u64,
        eps: f32,
        x: *const u16,
        weight_opt: *const u16,
        y: *mut u16,
        inv_rms_opt: *mut f32,
    ) -> i32;
    fn fa_cuda_bf16_residual_rmsnorm_fwd(
        ctx: *mut c_void,
        rows: u64,
        cols: u64,
        eps: f32,
        a: *const u16,
        b: *const u16,
        weight_opt: *const u16,
        residual_out: *mut u16,
        y: *mut u16,
        inv_rms_opt: *mut f32,
    ) -> i32;
    fn fa_cuda_bf16_rmsnorm_bwd(
        ctx: *mut c_void,
        rows: u64,
        cols: u64,
        eps: f32,
        x: *const u16,
        weight_opt: *const u16,
        inv_rms_opt: *const f32,
        dy: *const u16,
        dx: *mut u16,
        dweight_opt: *mut f32,
        dweight_partials_scratch: *mut f32,
        accumulate_dx: i32,
        accumulate_dw: i32,
    ) -> i32;
    fn fa_cuda_bf16_layernorm_fwd(
        ctx: *mut c_void,
        rows: u64,
        cols: u64,
        eps: f32,
        x: *const u16,
        y: *mut u16,
    ) -> i32;
    fn fa_cuda_bf16_layernorm_bwd(
        ctx: *mut c_void,
        rows: u64,
        cols: u64,
        eps: f32,
        x: *const u16,
        dy: *const u16,
        dx: *mut u16,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_bf16_softmax_fwd(
        ctx: *mut c_void,
        rows: u64,
        cols: u64,
        x: *const u16,
        y: *mut u16,
    ) -> i32;
    fn fa_cuda_bf16_softmax_bwd(
        ctx: *mut c_void,
        rows: u64,
        cols: u64,
        y: *const u16,
        dy: *const u16,
        dx: *mut u16,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_bf16_swiglu_fwd(
        ctx: *mut c_void,
        rows: u64,
        hidden: u64,
        gate_up: *const u16,
        out: *mut u16,
    ) -> i32;
    fn fa_cuda_bf16_swiglu_bwd(
        ctx: *mut c_void,
        rows: u64,
        hidden: u64,
        gate_up: *const u16,
        dout: *const u16,
        dgate_up: *mut u16,
    ) -> i32;
    fn fa_cuda_bf16_add(
        ctx: *mut c_void,
        n: u64,
        a: *const u16,
        b: *const u16,
        out: *mut u16,
    ) -> i32;
    fn fa_cuda_bf16_rope_run(
        ctx: *mut c_void,
        batch: u64,
        heads: u64,
        seq_len: u64,
        head_dim: u64,
        rotary_dim: u64,
        in_sb: i64,
        in_sh: i64,
        in_st: i64,
        out_sb: i64,
        out_sh: i64,
        out_st: i64,
        cos_dev: *const f32,
        sin_dev: *const f32,
        inp: *const u16,
        out: *mut u16,
        inverse: i32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_bf16_embedding_fwd(
        ctx: *mut c_void,
        vocab_size: u64,
        embed_dim: u64,
        num_indices: u64,
        table_bf16: *const u16,
        indices: *const i64,
        out_bf16: *mut u16,
    ) -> i32;
    fn fa_cuda_bf16_embedding_bwd_f32(
        ctx: *mut c_void,
        vocab_size: u64,
        embed_dim: u64,
        num_indices: u64,
        padding_idx: i64,
        has_padding_idx: i32,
        indices: *const i64,
        upstream_bf16: *const u16,
        grad_table_f32: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_bf16_cross_entropy_fwd(
        ctx: *mut c_void,
        num_rows: u64,
        vocab_size: u64,
        ignore_index: i64,
        has_ignore_index: i32,
        label_smoothing: f32,
        logits_bf16: *const u16,
        targets: *const i64,
        row_losses_scratch: *mut f32,
        row_valid_scratch: *mut i32,
        out_scalar_f32: *mut f32,
    ) -> i32;
    fn fa_cuda_bf16_cross_entropy_bwd(
        ctx: *mut c_void,
        num_rows: u64,
        vocab_size: u64,
        ignore_index: i64,
        has_ignore_index: i32,
        label_smoothing: f32,
        upstream_scalar: f32,
        logits_bf16: *const u16,
        targets: *const i64,
        active_count_scratch: *mut i32,
        grad_logits_bf16: *mut u16,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_bf16_cross_entropy_fwd_bwd(
        ctx: *mut c_void,
        num_rows: u64,
        vocab_size: u64,
        ignore_index: i64,
        has_ignore_index: i32,
        label_smoothing: f32,
        upstream_scalar: f32,
        logits_bf16: *const u16,
        targets: *const i64,
        active_count_scratch: *mut i32,
        row_losses_scratch: *mut f32,
        row_valid_scratch: *mut i32,
        out_scalar_f32: *mut f32,
        grad_logits_bf16: *mut u16,
    ) -> i32;
    fn fa_cuda_bf16_dropout_fwd(
        ctx: *mut c_void,
        n: u64,
        p: f32,
        seed: u64,
        step: u64,
        input: *const u16,
        output: *mut u16,
    ) -> i32;
    fn fa_cuda_bf16_dropout_bwd(
        ctx: *mut c_void,
        n: u64,
        p: f32,
        seed: u64,
        step: u64,
        upstream: *const u16,
        output: *mut u16,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_sdpa_bf16_prepare(ctx: *mut c_void) -> i32;
    fn fa_cuda_sdpa_bf16_fwd(
        ctx: *mut c_void,
        batch: u64,
        h_q: u64,
        h_kv: u64,
        t_q: u64,
        t_k: u64,
        d: u64,
        scale: f32,
        is_causal: i32,
        q: *const u16,
        sq: Bf16SdpaStrides,
        k: *const u16,
        sk: Bf16SdpaStrides,
        v: *const u16,
        sv: Bf16SdpaStrides,
        out: *mut u16,
        so: Bf16SdpaStrides,
        lse_f32_opt: *mut f32,
    ) -> i32;
    fn fa_cuda_sdpa_bf16_bwd(
        ctx: *mut c_void,
        batch: u64,
        h_q: u64,
        h_kv: u64,
        t_q: u64,
        t_k: u64,
        d: u64,
        scale: f32,
        is_causal: i32,
        q: *const u16,
        sq: Bf16SdpaStrides,
        k: *const u16,
        sk: Bf16SdpaStrides,
        v: *const u16,
        sv: Bf16SdpaStrides,
        o: *const u16,
        so: Bf16SdpaStrides,
        do_in: *const u16,
        sdo: Bf16SdpaStrides,
        lse_f32: *const f32,
        d_row_f32: *mut f32,
        gqa_dk_f32: *mut f32,
        gqa_dv_f32: *mut f32,
        dq: *mut u16,
        sdq: Bf16SdpaStrides,
        dk: *mut u16,
        sdk: Bf16SdpaStrides,
        dv: *mut u16,
        sdv: Bf16SdpaStrides,
        accumulate_dq: i32,
        accumulate_dk: i32,
        accumulate_dv: i32,
    ) -> i32;
    fn fa_cuda_adamw_state_create(
        ctx: *mut c_void,
        total_partials: u64,
        initial_lr: f32,
        out_ctrl: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_adamw_state_destroy(ctx: *mut c_void, ctrl: *mut c_void);
    fn fa_cuda_adamw_set_lr(ctx: *mut c_void, ctrl: *mut c_void, lr: f32) -> i32;
    fn fa_cuda_adamw_bump_step(ctx: *mut c_void, ctrl: *mut c_void) -> i32;
    fn fa_cuda_adamw_grad_norm_partial(
        ctx: *mut c_void,
        n: u64,
        grad: *const f32,
        ctrl: *mut c_void,
        partials_offset: u64,
        grid: u32,
    ) -> i32;
    fn fa_cuda_adamw_grad_norm_finalize(
        ctx: *mut c_void,
        ctrl: *mut c_void,
        total_partials: u64,
        max_grad_norm: f32,
    ) -> i32;
    fn fa_cuda_bf16_adamw_param(
        ctx: *mut c_void,
        n: u64,
        param_f32: *mut f32,
        param_bf16: *mut u16,
        grad_f32: *const f32,
        m_f32: *mut f32,
        v_f32: *mut f32,
        ctrl: *mut c_void,
        found_inf_opt: *const u32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        use_grad_scale: i32,
    ) -> i32;
    fn fa_cuda_grad_scaler_create(
        ctx: *mut c_void,
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: i32,
        out_scaler: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_grad_scaler_destroy(ctx: *mut c_void, scaler: *mut c_void);
    fn fa_cuda_grad_scaler_reset(ctx: *mut c_void, scaler: *mut c_void) -> i32;
    fn fa_cuda_grad_scaler_unscale_check(
        ctx: *mut c_void,
        scaler: *mut c_void,
        n: u64,
        grad_f32: *mut f32,
    ) -> i32;
    fn fa_cuda_grad_scaler_update(ctx: *mut c_void, scaler: *mut c_void) -> i32;
    fn fa_cuda_grad_scaler_read_state(
        ctx: *mut c_void,
        scaler: *mut c_void,
        out_scale: *mut f32,
        out_found_inf: *mut u32,
    ) -> i32;
    fn fa_cuda_grad_scaler_found_inf_ptr(scaler: *mut c_void) -> *const u32;
    fn fa_cuda_capture_begin(ctx: *mut c_void) -> i32;
    fn fa_cuda_capture_end(ctx: *mut c_void, exec_out: *mut *mut c_void) -> i32;
    fn fa_cuda_capture_abort(ctx: *mut c_void);
    fn fa_cuda_graph_launch(ctx: *mut c_void, exec: *mut c_void) -> i32;
    fn fa_cuda_graph_destroy(ctx: *mut c_void, exec: *mut c_void);
}

// ============================================================================
// DeviceBf16Tensor & DeviceIndexTensor
// ============================================================================

/// Owned contiguous `bfloat16` (`u16` bit-pattern) CUDA device allocation.
pub struct DeviceBf16Tensor {
    session: Rc<Session>,
    shape: Shape,
    ptr: Option<NonNull<c_void>>,
}

impl DeviceBf16Tensor {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        DType::Bf16
    }
    pub fn byte_len(&self) -> usize {
        self.shape.len() * size_of::<u16>()
    }
    pub(crate) fn read(&self) -> *const u16 {
        self.ptr.map_or(std::ptr::null(), |p| p.as_ptr().cast())
    }
    pub(crate) fn write(&mut self) -> *mut u16 {
        self.ptr.map_or(std::ptr::null_mut(), |p| p.as_ptr().cast())
    }
}

impl Drop for DeviceBf16Tensor {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr {
            // SAFETY: sole owner; retains live CUDA session; cudaFree waits for stream uses.
            unsafe { fa_cuda_free(self.session.ptr(), ptr.as_ptr()) }
        }
    }
}

/// Owned contiguous `i64` CUDA device index allocation with host-tracked min/max bounds
/// for O(1) zero-sync validation inside CUDA Graphs.
pub struct DeviceIndexTensor {
    session: Rc<Session>,
    shape: Shape,
    ptr: Option<NonNull<c_void>>,
    bounds: Option<(i64, i64)>,
    raw_host_checked: Vec<i64>,
}

impl DeviceIndexTensor {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        DType::I64
    }
    pub(crate) fn read(&self) -> *const i64 {
        self.ptr.map_or(std::ptr::null(), |p| p.as_ptr().cast())
    }
    pub(crate) fn write(&mut self) -> *mut i64 {
        self.ptr.map_or(std::ptr::null_mut(), |p| p.as_ptr().cast())
    }
    pub(crate) fn validate_range(
        &self,
        vocab_size: usize,
        ignore_index: Option<i64>,
    ) -> Result<()> {
        let Some((min_v, max_v)) = self.bounds else {
            return Ok(());
        };
        let v = vocab_size as i64;
        if min_v >= 0 && max_v < v {
            return Ok(());
        }
        if let Some(ignore) = ignore_index {
            for &tok in &self.raw_host_checked {
                if tok != ignore && (tok < 0 || tok >= v) {
                    return Err(invalid(format!("CUDA index {tok} out of range [0, {v})")));
                }
            }
            return Ok(());
        }
        Err(invalid(format!(
            "CUDA index range [{min_v}, {max_v}] out of [0, {v})"
        )))
    }
}

impl Drop for DeviceIndexTensor {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr {
            // SAFETY: sole owner; retains live CUDA session.
            unsafe { fa_cuda_free(self.session.ptr(), ptr.as_ptr()) }
        }
    }
}

fn compute_bounds(data: &[i64]) -> Option<(i64, i64)> {
    let mut it = data.iter().copied();
    let first = it.next()?;
    let (mut lo, mut hi) = (first, first);
    for v in it {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    Some((lo, hi))
}

impl Cuda {
    pub(crate) fn owns_bf16(&self, tensor: &DeviceBf16Tensor) -> Result<()> {
        if Rc::ptr_eq(&self.session, &tensor.session) {
            Ok(())
        } else {
            Err(invalid(
                "BF16 device tensor belongs to another CUDA session",
            ))
        }
    }
    pub(crate) fn owns_index(&self, tensor: &DeviceIndexTensor) -> Result<()> {
        if Rc::ptr_eq(&self.session, &tensor.session) {
            Ok(())
        } else {
            Err(invalid(
                "index device tensor belongs to another CUDA session",
            ))
        }
    }
    pub fn zeros_bf16(&self, dims: impl AsRef<[usize]>) -> Result<DeviceBf16Tensor> {
        let shape = Shape::new(dims)?;
        let mut ptr = std::ptr::null_mut();
        let ptr = if shape.is_empty() {
            None
        } else {
            // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
            check(unsafe {
                fa_cuda_alloc(self.session.ptr(), shape.len() * size_of::<u16>(), &mut ptr)
            })?;
            Some(nonnull(ptr)?)
        };
        Ok(DeviceBf16Tensor {
            session: self.session.clone(),
            shape,
            ptr,
        })
    }
    pub fn upload_bf16(&self, tensor: &Bf16Tensor) -> Result<DeviceBf16Tensor> {
        let mut out = self.zeros_bf16(tensor.shape().dims())?;
        self.copy_bf16_from_host(tensor, &mut out)?;
        Ok(out)
    }
    pub fn copy_bf16_from_host(&self, src: &Bf16Tensor, dst: &mut DeviceBf16Tensor) -> Result<()> {
        self.owns_bf16(dst)?;
        dst.shape.expect(src.shape().dims())?;
        if dst.shape.is_empty() {
            return Ok(());
        }
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_copy(
                self.session.ptr(),
                dst.write().cast(),
                src.bits().as_ptr().cast(),
                dst.shape.len() * size_of::<u16>(),
                1,
            )
        })
    }
    pub fn download_bf16(&self, tensor: &DeviceBf16Tensor) -> Result<Bf16Tensor> {
        self.owns_bf16(tensor)?;
        let mut out = Bf16Tensor::zeros(tensor.shape.dims())?;
        if tensor.shape.is_empty() {
            return Ok(out);
        }
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_copy(
                self.session.ptr(),
                out.bits_mut().as_mut_ptr().cast(),
                tensor.read().cast(),
                tensor.shape.len() * size_of::<u16>(),
                0,
            )
        })?;
        Ok(out)
    }
    pub fn cast_f32_to_bf16(&self, src: &DeviceTensor, dst: &mut DeviceBf16Tensor) -> Result<()> {
        self.owns(src)?;
        self.owns_bf16(dst)?;
        dst.shape.expect(src.shape().dims())?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_cast_f32_to_bf16(
                self.session.ptr(),
                src.shape().len() as u64,
                src.read(),
                dst.write(),
            )
        })
    }
    pub fn cast_bf16_to_f32(&self, src: &DeviceBf16Tensor, dst: &mut DeviceTensor) -> Result<()> {
        self.owns_bf16(src)?;
        self.owns(dst)?;
        dst.shape().expect(src.shape().dims())?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_cast_bf16_to_f32(
                self.session.ptr(),
                src.shape().len() as u64,
                src.read(),
                dst.write(),
            )
        })
    }
    pub fn upload_indices(&self, indices: &IndexTensor) -> Result<DeviceIndexTensor> {
        let shape = indices.shape().clone();
        let mut ptr = std::ptr::null_mut();
        let ptr = if shape.is_empty() {
            None
        } else {
            // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
            check(unsafe {
                fa_cuda_alloc(self.session.ptr(), shape.len() * size_of::<i64>(), &mut ptr)
            })?;
            Some(nonnull(ptr)?)
        };
        let mut dev = DeviceIndexTensor {
            session: self.session.clone(),
            shape,
            ptr,
            bounds: None,
            raw_host_checked: Vec::new(),
        };
        self.copy_indices_from_host(indices, &mut dev)?;
        Ok(dev)
    }
    pub fn copy_indices_from_host(
        &self,
        src: &IndexTensor,
        dst: &mut DeviceIndexTensor,
    ) -> Result<()> {
        self.owns_index(dst)?;
        dst.shape.expect(src.shape().dims())?;
        dst.bounds = compute_bounds(src.data());
        // Only retain host copy when negative indices exist (e.g. ignore_index=-100);
        // typical non-negative batches are validated in O(1) via (min, max).
        if let Some((lo, _)) = dst.bounds {
            if lo < 0 {
                dst.raw_host_checked.clear();
                dst.raw_host_checked.extend_from_slice(src.data());
            } else {
                dst.raw_host_checked.clear();
            }
        }
        if dst.shape.is_empty() {
            return Ok(());
        }
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_copy(
                self.session.ptr(),
                dst.write().cast(),
                src.data().as_ptr().cast(),
                dst.shape.len() * size_of::<i64>(),
                1,
            )
        })
    }
    pub fn download_indices(&self, tensor: &DeviceIndexTensor) -> Result<IndexTensor> {
        self.owns_index(tensor)?;
        let mut out = vec![0i64; tensor.shape.len()];
        if !out.is_empty() {
            // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
            check(unsafe {
                fa_cuda_copy(
                    self.session.ptr(),
                    out.as_mut_ptr().cast(),
                    tensor.read().cast(),
                    out.len() * size_of::<i64>(),
                    0,
                )
            })?;
        }
        IndexTensor::from_vec(tensor.shape.dims(), out)
    }
    pub fn zero_bf16(&self, tensor: &mut DeviceBf16Tensor) -> Result<()> {
        self.owns_bf16(tensor)?;
        if tensor.shape.is_empty() {
            return Ok(());
        }
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe { fa_cuda_zero(self.session.ptr(), tensor.write().cast(), tensor.byte_len()) })
    }
    pub fn bf16_add(
        &self,
        a: &DeviceBf16Tensor,
        b: &DeviceBf16Tensor,
        out: &mut DeviceBf16Tensor,
    ) -> Result<()> {
        self.owns_bf16(a)?;
        self.owns_bf16(b)?;
        self.owns_bf16(out)?;
        b.shape().expect(a.shape().dims())?;
        out.shape().expect(a.shape().dims())?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_add(
                self.session.ptr(),
                a.shape().len() as u64,
                a.read(),
                b.read(),
                out.write(),
            )
        })
    }
}

// ============================================================================
// Explicit TF32 & BF16 Tensor Core GEMM / Linear / BMM Plans
// ============================================================================

/// Explicit `TF32` Tensor Core matmul plan (`PrecisionPolicy::Tf32` required on `Cuda`).
pub struct Tf32MatmulPlan {
    cuda: Cuda,
    m: i32,
    n: i32,
    k: i32,
}

impl Tf32MatmulPlan {
    pub fn new(cuda: &Cuda, m: usize, n: usize, k: usize) -> Result<Self> {
        if cuda.precision_policy() == PrecisionPolicy::StrictFp32 {
            return Err(invalid(
                "Tf32MatmulPlan requires PrecisionPolicy::Tf32 or Bf16Mixed (strict-FP32 session rejects TF32)",
            ));
        }
        Ok(Self {
            cuda: cuda.clone(),
            m: int(m)?,
            n: int(n)?,
            k: int(k)?,
        })
    }
    pub fn run(&self, a: &DeviceTensor, b: &DeviceTensor, out: &mut DeviceTensor) -> Result<()> {
        self.cuda.owns(a)?;
        self.cuda.owns(b)?;
        self.cuda.owns(out)?;
        a.shape().expect(&[self.m as usize, self.k as usize])?;
        b.shape().expect(&[self.k as usize, self.n as usize])?;
        out.shape().expect(&[self.m as usize, self.n as usize])?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_tf32_gemm(
                self.cuda.session.ptr(),
                0,
                0,
                self.m,
                self.n,
                self.k,
                a.read(),
                self.k,
                b.read(),
                self.n,
                out.write(),
                self.n,
                1.0,
                0.0,
            )
        })
    }
}

/// BF16 Tensor Core linear projection (`Y = X @ W^T`, `dX = dY @ W`, `dW_f32 = dY^T @ X`)
/// using `cublasGemmEx` (`CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`).
pub struct Bf16LinearPlan {
    cuda: Cuda,
    rows: i32,
    in_features: i32,
    out_features: i32,
}

impl Bf16LinearPlan {
    pub fn new(cuda: &Cuda, rows: usize, in_features: usize, out_features: usize) -> Result<Self> {
        Ok(Self {
            cuda: cuda.clone(),
            rows: int(rows)?,
            in_features: int(in_features)?,
            out_features: int(out_features)?,
        })
    }
    /// Forward: `out [rows, out_features] = x [rows, in_features] @ weight [out_features, in_features]^T`.
    pub fn forward(
        &self,
        x: &DeviceBf16Tensor,
        weight: &DeviceBf16Tensor,
        out: &mut DeviceBf16Tensor,
    ) -> Result<()> {
        self.cuda.owns_bf16(x)?;
        self.cuda.owns_bf16(weight)?;
        self.cuda.owns_bf16(out)?;
        if x.shape().len() != (self.rows as usize) * (self.in_features as usize)
            || weight.shape().len() != (self.out_features as usize) * (self.in_features as usize)
            || out.shape().len() != (self.rows as usize) * (self.out_features as usize)
        {
            return Err(invalid("Bf16LinearPlan forward tensor size mismatch"));
        }
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_gemm(
                self.cuda.session.ptr(),
                0,
                1,
                self.rows,
                self.out_features,
                self.in_features,
                x.read(),
                self.in_features,
                weight.read(),
                self.in_features,
                out.write().cast(),
                self.out_features,
                0,
                1.0,
                0.0,
            )
        })
    }
    /// Backward input: `dx [rows, in_features] = dy [rows, out_features] @ weight [out_features, in_features]`.
    pub fn backward_input(
        &self,
        dy: &DeviceBf16Tensor,
        weight: &DeviceBf16Tensor,
        dx: &mut DeviceBf16Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_bf16(dy)?;
        self.cuda.owns_bf16(weight)?;
        self.cuda.owns_bf16(dx)?;
        let beta = if mode == GradientMode::Accumulate {
            1.0
        } else {
            0.0
        };
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_gemm(
                self.cuda.session.ptr(),
                0,
                0,
                self.rows,
                self.in_features,
                self.out_features,
                dy.read(),
                self.out_features,
                weight.read(),
                self.in_features,
                dx.write().cast(),
                self.in_features,
                0,
                1.0,
                beta,
            )
        })
    }
    /// Backward weight directly into FP32 master gradient:
    /// `dw_f32 [out_features, in_features] = dy^T @ x`.
    pub fn backward_weight_f32(
        &self,
        dy: &DeviceBf16Tensor,
        x: &DeviceBf16Tensor,
        dw_f32: &mut DeviceTensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_bf16(dy)?;
        self.cuda.owns_bf16(x)?;
        self.cuda.owns(dw_f32)?;
        let beta = if mode == GradientMode::Accumulate {
            1.0
        } else {
            0.0
        };
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_gemm(
                self.cuda.session.ptr(),
                1,
                0,
                self.out_features,
                self.in_features,
                self.rows,
                dy.read(),
                self.out_features,
                x.read(),
                self.in_features,
                dw_f32.write().cast(),
                self.in_features,
                1,
                1.0,
                beta,
            )
        })
    }
    /// Batched BF16 Tensor Core matmul: `C [B, M, N] = A [B, M, K] @ B [B, K, N]`.
    #[allow(clippy::too_many_arguments)]
    pub fn bmm(
        cuda: &Cuda,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        a: &DeviceBf16Tensor,
        b: &DeviceBf16Tensor,
        out: &mut DeviceBf16Tensor,
    ) -> Result<()> {
        cuda.owns_bf16(a)?;
        cuda.owns_bf16(b)?;
        cuda.owns_bf16(out)?;
        a.shape().expect(&[batch, m, k])?;
        b.shape().expect(&[batch, k, n])?;
        out.shape().expect(&[batch, m, n])?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_bmm_strided(
                cuda.session.ptr(),
                0,
                0,
                int(batch)?,
                int(m)?,
                int(n)?,
                int(k)?,
                a.read(),
                int(k)?,
                (m * k) as i64,
                b.read(),
                int(n)?,
                (k * n) as i64,
                out.write().cast(),
                int(n)?,
                (m * n) as i64,
                0,
                1.0,
                0.0,
            )
        })
    }
}

// ============================================================================
// True O(T)-Memory BF16 Tensor Core FlashAttention-2 Plan
// ============================================================================

/// Prepared `O(T)`-memory BF16 Tensor Core FlashAttention-2 forward + zero-atomic backward plan.
pub struct Bf16SdpaPlan {
    cuda: Cuda,
    spec: SdpaSpec,
    lse: DeviceTensor,
    d_row: DeviceTensor,
    gqa_dk: Option<DeviceTensor>,
    gqa_dv: Option<DeviceTensor>,
}

impl Bf16SdpaPlan {
    pub fn new(cuda: &Cuda, spec: SdpaSpec) -> Result<Self> {
        if spec.head_dim() == 128 && spec.batch() != 0 && spec.num_heads() != 0 && spec.seq_q() != 0
        {
            // SAFETY: the retained CUDA session identifies the current device.
            check(unsafe { fa_cuda_sdpa_bf16_prepare(cuda.session.ptr()) })?;
        }
        let bh = if spec.seq_q() == 0 {
            0
        } else {
            spec.batch() * spec.num_heads()
        };
        let lse = cuda.zeros([bh, spec.seq_q()])?;
        let d_row = cuda.zeros([bh, spec.seq_q()])?;
        let (gqa_dk, gqa_dv) = if bh != 0
            && spec.num_heads() > spec.num_kv_heads()
            && matches!(spec.head_dim(), 64 | 128)
        {
            (
                Some(cuda.zeros([bh, spec.seq_k(), spec.head_dim()])?),
                Some(cuda.zeros([bh, spec.seq_k(), spec.head_dim()])?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            lse,
            d_row,
            gqa_dk,
            gqa_dv,
        })
    }
    pub fn spec(&self) -> SdpaSpec {
        self.spec
    }
    fn query_dims(&self) -> [usize; 4] {
        [
            self.spec.batch(),
            self.spec.num_heads(),
            self.spec.seq_q(),
            self.spec.head_dim(),
        ]
    }
    fn kv_dims(&self) -> [usize; 4] {
        [
            self.spec.batch(),
            self.spec.num_kv_heads(),
            self.spec.seq_k(),
            self.spec.head_dim(),
        ]
    }
    /// Workspace bytes (`O(T)` — zero `T_q * T_k` allocation!).
    pub fn workspace_bytes(&self) -> usize {
        let mut total = (self.lse.shape().len() + self.d_row.shape().len()) * size_of::<f32>();
        if let Some(ref g) = self.gqa_dk {
            total += g.shape().len() * size_of::<f32>() * 2;
        }
        total
    }
    /// Forward on contiguous `[B, H, T, D]` tensors.
    pub fn forward(
        &mut self,
        q: &DeviceBf16Tensor,
        k: &DeviceBf16Tensor,
        v: &DeviceBf16Tensor,
        out: &mut DeviceBf16Tensor,
    ) -> Result<()> {
        let query = self.query_dims();
        let kv = self.kv_dims();
        q.shape().expect(&query)?;
        k.shape().expect(&kv)?;
        v.shape().expect(&kv)?;
        out.shape().expect(&query)?;
        if query.contains(&0) {
            for tensor in [q, k, v, out] {
                self.cuda.owns_bf16(tensor)?;
            }
            return Ok(());
        }
        let sq = Bf16SdpaStrides::bhtd(
            self.spec.num_heads(),
            self.spec.seq_q(),
            self.spec.head_dim(),
        );
        let sk = Bf16SdpaStrides::bhtd(
            self.spec.num_kv_heads(),
            self.spec.seq_k(),
            self.spec.head_dim(),
        );
        self.forward_strided(q, 0, sq, k, 0, sk, v, 0, sk, out, 0, sq)
    }
    /// Backward on contiguous `[B, H, T, D]` tensors (reuses saved `O(T)` LSE from last `forward`).
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &mut self,
        q: &DeviceBf16Tensor,
        k: &DeviceBf16Tensor,
        v: &DeviceBf16Tensor,
        o: &DeviceBf16Tensor,
        dout: &DeviceBf16Tensor,
        dq: &mut DeviceBf16Tensor,
        dk: &mut DeviceBf16Tensor,
        dv: &mut DeviceBf16Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        if self.query_dims().contains(&0) {
            for tensor in [q, k, v, o, dout, dq, dk, dv] {
                self.cuda.owns_bf16(tensor)?;
            }
            return Ok(());
        }
        let sq = Bf16SdpaStrides::bhtd(
            self.spec.num_heads(),
            self.spec.seq_q(),
            self.spec.head_dim(),
        );
        let sk = Bf16SdpaStrides::bhtd(
            self.spec.num_kv_heads(),
            self.spec.seq_k(),
            self.spec.head_dim(),
        );
        self.backward_strided(
            q, 0, sq, k, 0, sk, v, 0, sk, o, 0, sq, dout, 0, sq, dq, 0, sq, dk, 0, sk, dv, 0, sk,
            mode,
        )
    }
    /// Zero-copy strided forward supporting `[B, T, H, D]` and fused `[B, T, H_q+2*H_kv, D]` QKV views!
    #[allow(clippy::too_many_arguments)]
    pub fn forward_strided(
        &mut self,
        q: &DeviceBf16Tensor,
        q_off: usize,
        sq: Bf16SdpaStrides,
        k: &DeviceBf16Tensor,
        k_off: usize,
        sk: Bf16SdpaStrides,
        v: &DeviceBf16Tensor,
        v_off: usize,
        sv: Bf16SdpaStrides,
        out: &mut DeviceBf16Tensor,
        o_off: usize,
        so: Bf16SdpaStrides,
    ) -> Result<()> {
        self.cuda.owns_bf16(q)?;
        self.cuda.owns_bf16(k)?;
        self.cuda.owns_bf16(v)?;
        self.cuda.owns_bf16(out)?;
        let query = self.query_dims();
        if query.contains(&0) {
            return Ok(());
        }
        let kv = self.kv_dims();
        sq.check_view(query, q_off, q.shape().len(), false)?;
        sk.check_view(kv, k_off, k.shape().len(), false)?;
        sv.check_view(kv, v_off, v.shape().len(), false)?;
        so.check_view(query, o_off, out.shape().len(), true)?;
        // SAFETY: the device session and all view spans are checked; output views are injective.
        check(unsafe {
            fa_cuda_sdpa_bf16_fwd(
                self.cuda.session.ptr(),
                self.spec.batch() as u64,
                self.spec.num_heads() as u64,
                self.spec.num_kv_heads() as u64,
                self.spec.seq_q() as u64,
                self.spec.seq_k() as u64,
                self.spec.head_dim() as u64,
                self.spec.scale(),
                i32::from(self.spec.is_causal()),
                q.read().add(q_off),
                sq,
                k.read().add(k_off),
                sk,
                v.read().add(v_off),
                sv,
                out.write().add(o_off),
                so,
                self.lse.write(),
            )
        })
    }
    /// Zero-copy strided backward supporting `[B, T, H, D]` and fused QKV buffers!
    #[allow(clippy::too_many_arguments)]
    pub fn backward_strided(
        &mut self,
        q: &DeviceBf16Tensor,
        q_off: usize,
        sq: Bf16SdpaStrides,
        k: &DeviceBf16Tensor,
        k_off: usize,
        sk: Bf16SdpaStrides,
        v: &DeviceBf16Tensor,
        v_off: usize,
        sv: Bf16SdpaStrides,
        o: &DeviceBf16Tensor,
        o_off: usize,
        so: Bf16SdpaStrides,
        dout: &DeviceBf16Tensor,
        do_off: usize,
        sdo: Bf16SdpaStrides,
        dq: &mut DeviceBf16Tensor,
        dq_off: usize,
        sdq: Bf16SdpaStrides,
        dk: &mut DeviceBf16Tensor,
        dk_off: usize,
        sdk: Bf16SdpaStrides,
        dv: &mut DeviceBf16Tensor,
        dv_off: usize,
        sdv: Bf16SdpaStrides,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_bf16(q)?;
        self.cuda.owns_bf16(k)?;
        self.cuda.owns_bf16(v)?;
        self.cuda.owns_bf16(o)?;
        self.cuda.owns_bf16(dout)?;
        self.cuda.owns_bf16(dq)?;
        self.cuda.owns_bf16(dk)?;
        self.cuda.owns_bf16(dv)?;
        let query = self.query_dims();
        if query.contains(&0) {
            return Ok(());
        }
        let kv = self.kv_dims();
        sq.check_view(query, q_off, q.shape().len(), false)?;
        sk.check_view(kv, k_off, k.shape().len(), false)?;
        sv.check_view(kv, v_off, v.shape().len(), false)?;
        so.check_view(query, o_off, o.shape().len(), false)?;
        sdo.check_view(query, do_off, dout.shape().len(), false)?;
        sdq.check_view(query, dq_off, dq.shape().len(), true)?;
        sdk.check_view(kv, dk_off, dk.shape().len(), true)?;
        sdv.check_view(kv, dv_off, dv.shape().len(), true)?;
        let acc = i32::from(mode == GradientMode::Accumulate);
        let gqa_dk_ptr = self
            .gqa_dk
            .as_mut()
            .map_or(std::ptr::null_mut(), DeviceTensor::write);
        let gqa_dv_ptr = self
            .gqa_dv
            .as_mut()
            .map_or(std::ptr::null_mut(), DeviceTensor::write);
        // SAFETY: validated session ownership, non-aliasing slices within buffers, and GQA scratch.
        check(unsafe {
            fa_cuda_sdpa_bf16_bwd(
                self.cuda.session.ptr(),
                self.spec.batch() as u64,
                self.spec.num_heads() as u64,
                self.spec.num_kv_heads() as u64,
                self.spec.seq_q() as u64,
                self.spec.seq_k() as u64,
                self.spec.head_dim() as u64,
                self.spec.scale(),
                i32::from(self.spec.is_causal()),
                q.read().add(q_off),
                sq,
                k.read().add(k_off),
                sk,
                v.read().add(v_off),
                sv,
                o.read().add(o_off),
                so,
                dout.read().add(do_off),
                sdo,
                self.lse.read(),
                self.d_row.write(),
                gqa_dk_ptr,
                gqa_dv_ptr,
                dq.write().add(dq_off),
                sdq,
                dk.write().add(dk_off),
                sdk,
                dv.write().add(dv_off),
                sdv,
                acc,
                acc,
                acc,
            )
        })
    }
    /// Internal zero-copy backward writing `dQ`, `dK`, `dV` into disjoint slices of ONE `dQKV` buffer!
    #[allow(clippy::too_many_arguments)]
    fn backward_fused_qkv(
        &mut self,
        qkv: &DeviceBf16Tensor,
        o: &DeviceBf16Tensor,
        dout: &DeviceBf16Tensor,
        dqkv: &mut DeviceBf16Tensor,
    ) -> Result<()> {
        let hq = self.spec.num_heads();
        let hkv = self.spec.num_kv_heads();
        let h_tot = hq + 2 * hkv;
        let t = self.spec.seq_q();
        let d = self.spec.head_dim();
        let s_qkv = Bf16SdpaStrides::bthd(t, h_tot, d);
        let s_o = Bf16SdpaStrides::bthd(t, hq, d);
        let q_ptr = qkv.read();
        // SAFETY: offset lies strictly inside the allocated BF16 device tensor.
        let k_ptr = unsafe { q_ptr.add(hq * d) };
        // SAFETY: offset lies strictly inside the allocated BF16 device tensor.
        let v_ptr = unsafe { q_ptr.add((hq + hkv) * d) };
        let dq_ptr = dqkv.write();
        // SAFETY: offset lies strictly inside the allocated BF16 device tensor.
        let dk_ptr = unsafe { dq_ptr.add(hq * d) };
        // SAFETY: offset lies strictly inside the allocated BF16 device tensor.
        let dv_ptr = unsafe { dq_ptr.add((hq + hkv) * d) };
        let gqa_dk_ptr = self
            .gqa_dk
            .as_mut()
            .map_or(std::ptr::null_mut(), DeviceTensor::write);
        let gqa_dv_ptr = self
            .gqa_dv
            .as_mut()
            .map_or(std::ptr::null_mut(), DeviceTensor::write);
        // SAFETY: Q, K, V and dQ, dK, dV occupy strictly disjoint head ranges [0..hq), [hq..hq+hkv), [hq+hkv..h_tot).
        check(unsafe {
            fa_cuda_sdpa_bf16_bwd(
                self.cuda.session.ptr(),
                self.spec.batch() as u64,
                hq as u64,
                hkv as u64,
                t as u64,
                t as u64,
                d as u64,
                self.spec.scale(),
                i32::from(self.spec.is_causal()),
                q_ptr,
                s_qkv,
                k_ptr,
                s_qkv,
                v_ptr,
                s_qkv,
                o.read(),
                s_o,
                dout.read(),
                s_o,
                self.lse.read(),
                self.d_row.write(),
                gqa_dk_ptr,
                gqa_dv_ptr,
                dq_ptr,
                s_qkv,
                dk_ptr,
                s_qkv,
                dv_ptr,
                s_qkv,
                0,
                0,
                0,
            )
        })
    }
}

// ============================================================================
// Vectorized BF16 Norm, Softmax, SwiGLU, RoPE, Embedding, CrossEntropy, Dropout
// ============================================================================

/// Vectorized BF16 `RmsNorm` (optional learnable BF16 weight + fused ResidualAdd) and `LayerNorm`.
pub struct Bf16NormPlan {
    cuda: Cuda,
    rows: usize,
    cols: usize,
    eps: f32,
    dw_partials: DeviceTensor,
}

impl Bf16NormPlan {
    pub fn new(cuda: &Cuda, rows: usize, cols: usize, eps: f32) -> Result<Self> {
        if cols == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(invalid("Bf16NormPlan requires cols > 0 and finite eps > 0"));
        }
        let chunks = rows.clamp(1, 64);
        let dw_partials = cuda.zeros([chunks, cols])?;
        Ok(Self {
            cuda: cuda.clone(),
            rows,
            cols,
            eps,
            dw_partials,
        })
    }
    pub fn rmsnorm_forward(
        &self,
        x: &DeviceBf16Tensor,
        weight: Option<&DeviceBf16Tensor>,
        y: &mut DeviceBf16Tensor,
        inv_rms: Option<&mut DeviceTensor>,
    ) -> Result<()> {
        self.cuda.owns_bf16(x)?;
        self.cuda.owns_bf16(y)?;
        if let Some(w) = weight {
            self.cuda.owns_bf16(w)?;
            w.shape().expect(&[self.cols])?;
        }
        let inv_ptr = match inv_rms {
            Some(r) => {
                self.cuda.owns(r)?;
                r.shape().expect(&[self.rows])?;
                r.write()
            }
            None => std::ptr::null_mut(),
        };
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_rmsnorm_fwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.cols as u64,
                self.eps,
                x.read(),
                weight.map_or(std::ptr::null(), DeviceBf16Tensor::read),
                y.write(),
                inv_ptr,
            )
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn residual_rmsnorm_forward(
        &self,
        a: &DeviceBf16Tensor,
        b: &DeviceBf16Tensor,
        weight: Option<&DeviceBf16Tensor>,
        residual_out: &mut DeviceBf16Tensor,
        y: &mut DeviceBf16Tensor,
        inv_rms: Option<&mut DeviceTensor>,
    ) -> Result<()> {
        self.cuda.owns_bf16(a)?;
        self.cuda.owns_bf16(b)?;
        self.cuda.owns_bf16(residual_out)?;
        self.cuda.owns_bf16(y)?;
        let inv_ptr = match inv_rms {
            Some(r) => {
                self.cuda.owns(r)?;
                r.write()
            }
            None => std::ptr::null_mut(),
        };
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_residual_rmsnorm_fwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.cols as u64,
                self.eps,
                a.read(),
                b.read(),
                weight.map_or(std::ptr::null(), DeviceBf16Tensor::read),
                residual_out.write(),
                y.write(),
                inv_ptr,
            )
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_backward(
        &mut self,
        x: &DeviceBf16Tensor,
        weight: Option<&DeviceBf16Tensor>,
        inv_rms: Option<&DeviceTensor>,
        dy: &DeviceBf16Tensor,
        dx: &mut DeviceBf16Tensor,
        dweight_f32: Option<&mut DeviceTensor>,
        mode_dx: GradientMode,
        mode_dw: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_bf16(x)?;
        self.cuda.owns_bf16(dy)?;
        self.cuda.owns_bf16(dx)?;
        let dw_ptr = match dweight_f32 {
            Some(dw) => {
                self.cuda.owns(dw)?;
                dw.write()
            }
            None => std::ptr::null_mut(),
        };
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_rmsnorm_bwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.cols as u64,
                self.eps,
                x.read(),
                weight.map_or(std::ptr::null(), DeviceBf16Tensor::read),
                inv_rms.map_or(std::ptr::null(), DeviceTensor::read),
                dy.read(),
                dx.write(),
                dw_ptr,
                self.dw_partials.write(),
                i32::from(mode_dx == GradientMode::Accumulate),
                i32::from(mode_dw == GradientMode::Accumulate),
            )
        })
    }
    pub fn layernorm_forward(&self, x: &DeviceBf16Tensor, y: &mut DeviceBf16Tensor) -> Result<()> {
        self.cuda.owns_bf16(x)?;
        self.cuda.owns_bf16(y)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_layernorm_fwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.cols as u64,
                self.eps,
                x.read(),
                y.write(),
            )
        })
    }
    pub fn layernorm_backward(
        &self,
        x: &DeviceBf16Tensor,
        dy: &DeviceBf16Tensor,
        dx: &mut DeviceBf16Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_bf16(x)?;
        self.cuda.owns_bf16(dy)?;
        self.cuda.owns_bf16(dx)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_layernorm_bwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.cols as u64,
                self.eps,
                x.read(),
                dy.read(),
                dx.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

pub struct Bf16SoftmaxPlan {
    cuda: Cuda,
    rows: usize,
    cols: usize,
}

impl Bf16SoftmaxPlan {
    pub fn new(cuda: &Cuda, rows: usize, cols: usize) -> Result<Self> {
        if cols == 0 {
            return Err(invalid("Bf16SoftmaxPlan requires cols > 0"));
        }
        Ok(Self {
            cuda: cuda.clone(),
            rows,
            cols,
        })
    }
    pub fn forward(&self, x: &DeviceBf16Tensor, y: &mut DeviceBf16Tensor) -> Result<()> {
        self.cuda.owns_bf16(x)?;
        self.cuda.owns_bf16(y)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_softmax_fwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.cols as u64,
                x.read(),
                y.write(),
            )
        })
    }
    pub fn backward(
        &self,
        y: &DeviceBf16Tensor,
        dy: &DeviceBf16Tensor,
        dx: &mut DeviceBf16Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_bf16(y)?;
        self.cuda.owns_bf16(dy)?;
        self.cuda.owns_bf16(dx)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_softmax_bwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.cols as u64,
                y.read(),
                dy.read(),
                dx.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

pub struct Bf16SwiGluPlan {
    cuda: Cuda,
    rows: usize,
    hidden: usize,
}

impl Bf16SwiGluPlan {
    pub fn new(cuda: &Cuda, rows: usize, hidden: usize) -> Result<Self> {
        if hidden == 0 {
            return Err(invalid("Bf16SwiGluPlan requires hidden > 0"));
        }
        Ok(Self {
            cuda: cuda.clone(),
            rows,
            hidden,
        })
    }
    pub fn forward(&self, gate_up: &DeviceBf16Tensor, out: &mut DeviceBf16Tensor) -> Result<()> {
        self.cuda.owns_bf16(gate_up)?;
        self.cuda.owns_bf16(out)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_swiglu_fwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.hidden as u64,
                gate_up.read(),
                out.write(),
            )
        })
    }
    pub fn backward(
        &self,
        gate_up: &DeviceBf16Tensor,
        dout: &DeviceBf16Tensor,
        dgate_up: &mut DeviceBf16Tensor,
    ) -> Result<()> {
        self.cuda.owns_bf16(gate_up)?;
        self.cuda.owns_bf16(dout)?;
        self.cuda.owns_bf16(dgate_up)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_swiglu_bwd(
                self.cuda.session.ptr(),
                self.rows as u64,
                self.hidden as u64,
                gate_up.read(),
                dout.read(),
                dgate_up.write(),
            )
        })
    }
}

/// Prepared BF16 `RoPE` plan with device-cached FP32 `(cos, sin)` tables (in-place and strided safe).
pub struct Bf16RopePlan {
    cuda: Cuda,
    spec: RopeSpec,
    cos_dev: DeviceTensor,
    sin_dev: DeviceTensor,
}

impl Bf16RopePlan {
    pub fn new(cuda: &Cuda, spec: RopeSpec) -> Result<Self> {
        let (cos, sin) = spec.build_tables();
        let half = spec.rotary_dim() / 2;
        let cos_dev = cuda.upload(&Tensor::from_vec([spec.seq_len(), half], cos)?)?;
        let sin_dev = cuda.upload(&Tensor::from_vec([spec.seq_len(), half], sin)?)?;
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            cos_dev,
            sin_dev,
        })
    }
    pub fn spec(&self) -> &RopeSpec {
        &self.spec
    }
    pub fn forward(&self, x: &DeviceBf16Tensor, out: &mut DeviceBf16Tensor) -> Result<()> {
        self.run_contiguous(x, out, false, false)
    }
    pub fn backward(
        &self,
        dy: &DeviceBf16Tensor,
        dx: &mut DeviceBf16Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.run_contiguous(dy, dx, true, mode == GradientMode::Accumulate)
    }
    fn run_contiguous(
        &self,
        inp: &DeviceBf16Tensor,
        out: &mut DeviceBf16Tensor,
        inverse: bool,
        accumulate: bool,
    ) -> Result<()> {
        self.cuda.owns_bf16(inp)?;
        self.cuda.owns_bf16(out)?;
        inp.shape().expect(self.spec.shape().dims())?;
        out.shape().expect(self.spec.shape().dims())?;
        let b = self.spec.batch();
        let h = self.spec.num_heads();
        let t = self.spec.seq_len();
        let d = self.spec.head_dim();
        let s = if self.spec.seq_axis() == 2 {
            Bf16SdpaStrides::bhtd(h, t, d)
        } else {
            Bf16SdpaStrides::bthd(t, h, d)
        };
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_rope_run(
                self.cuda.session.ptr(),
                b as u64,
                h as u64,
                t as u64,
                d as u64,
                self.spec.rotary_dim() as u64,
                s.sb,
                s.sh,
                s.st,
                s.sb,
                s.sh,
                s.st,
                self.cos_dev.read(),
                self.sin_dev.read(),
                inp.read(),
                out.write(),
                i32::from(inverse),
                i32::from(accumulate),
            )
        })
    }
    /// In-place RoPE on a head slice `[head_start .. head_start + heads]` of a `[B, T, H_tot, D]` QKV tensor.
    pub fn run_inplace_bthd_slice(
        &self,
        qkv: &mut DeviceBf16Tensor,
        head_start: usize,
        heads: usize,
        h_tot: usize,
        inverse: bool,
    ) -> Result<()> {
        self.cuda.owns_bf16(qkv)?;
        let b = self.spec.batch();
        let t = self.spec.seq_len();
        let d = self.spec.head_dim();
        let s = Bf16SdpaStrides::bthd(t, h_tot, d);
        // SAFETY: offset lies strictly inside the allocated BF16 device tensor.
        let ptr = unsafe { qkv.write().add(head_start * d) };
        // SAFETY: in-place rotation reads `(col, col+half)` into registers before writing back;
        // every thread owns a unique `(b, h, t, col)` pair.
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_rope_run(
                self.cuda.session.ptr(),
                b as u64,
                heads as u64,
                t as u64,
                d as u64,
                self.spec.rotary_dim() as u64,
                s.sb,
                s.sh,
                s.st,
                s.sb,
                s.sh,
                s.st,
                self.cos_dev.read(),
                self.sin_dev.read(),
                ptr,
                ptr,
                i32::from(inverse),
                0,
            )
        })
    }
}

pub struct Bf16EmbeddingPlan {
    cuda: Cuda,
    spec: EmbeddingSpec,
}

impl Bf16EmbeddingPlan {
    pub fn new(cuda: &Cuda, spec: EmbeddingSpec) -> Self {
        Self {
            cuda: cuda.clone(),
            spec,
        }
    }
    pub fn forward(
        &self,
        table_bf16: &DeviceBf16Tensor,
        indices: &DeviceIndexTensor,
        out_bf16: &mut DeviceBf16Tensor,
    ) -> Result<()> {
        self.cuda.owns_bf16(table_bf16)?;
        self.cuda.owns_index(indices)?;
        self.cuda.owns_bf16(out_bf16)?;
        indices.validate_range(self.spec.vocab_size(), None)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_embedding_fwd(
                self.cuda.session.ptr(),
                self.spec.vocab_size() as u64,
                self.spec.embed_dim() as u64,
                self.spec.num_indices() as u64,
                table_bf16.read(),
                indices.read(),
                out_bf16.write(),
            )
        })
    }
    pub fn backward_f32(
        &self,
        indices: &DeviceIndexTensor,
        upstream_bf16: &DeviceBf16Tensor,
        grad_table_f32: &mut DeviceTensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_index(indices)?;
        self.cuda.owns_bf16(upstream_bf16)?;
        self.cuda.owns(grad_table_f32)?;
        indices.validate_range(self.spec.vocab_size(), None)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_embedding_bwd_f32(
                self.cuda.session.ptr(),
                self.spec.vocab_size() as u64,
                self.spec.embed_dim() as u64,
                self.spec.num_indices() as u64,
                self.spec.padding_idx().unwrap_or(0),
                i32::from(self.spec.padding_idx().is_some()),
                indices.read(),
                upstream_bf16.read(),
                grad_table_f32.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

/// Prepared BF16 fused `CrossEntropy` plan with single-pass `forward_backward` (zero `[N, V]` intermediate).
pub struct Bf16CrossEntropyPlan {
    cuda: Cuda,
    spec: CrossEntropySpec,
    active_count: DeviceTensor,
    row_losses: DeviceTensor,
    row_valid: DeviceTensor,
}

impl Bf16CrossEntropyPlan {
    pub fn new(cuda: &Cuda, spec: CrossEntropySpec) -> Result<Self> {
        let n = spec.num_rows().max(1);
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            active_count: cuda.zeros([1])?,
            row_losses: cuda.zeros([n])?,
            row_valid: cuda.zeros([n])?,
        })
    }
    pub fn forward(
        &mut self,
        logits: &DeviceBf16Tensor,
        targets: &DeviceIndexTensor,
        out_loss_f32: &mut DeviceTensor,
    ) -> Result<()> {
        self.cuda.owns_bf16(logits)?;
        self.cuda.owns_index(targets)?;
        self.cuda.owns(out_loss_f32)?;
        logits
            .shape()
            .expect(&[self.spec.num_rows(), self.spec.vocab_size()])?;
        targets.shape().expect(&[self.spec.num_rows()])?;
        out_loss_f32.shape().expect(&[])?;
        targets.validate_range(self.spec.vocab_size(), self.spec.ignore_index())?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_cross_entropy_fwd(
                self.cuda.session.ptr(),
                self.spec.num_rows() as u64,
                self.spec.vocab_size() as u64,
                self.spec.ignore_index().unwrap_or(-100),
                i32::from(self.spec.ignore_index().is_some()),
                self.spec.label_smoothing(),
                logits.read(),
                targets.read(),
                self.row_losses.write(),
                self.row_valid.write().cast(),
                out_loss_f32.write(),
            )
        })
    }
    pub fn backward(
        &mut self,
        logits: &DeviceBf16Tensor,
        targets: &DeviceIndexTensor,
        upstream_scalar: f32,
        grad_logits: &mut DeviceBf16Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_bf16(logits)?;
        self.cuda.owns_index(targets)?;
        self.cuda.owns_bf16(grad_logits)?;
        let logits_dims = [self.spec.num_rows(), self.spec.vocab_size()];
        logits.shape().expect(&logits_dims)?;
        targets.shape().expect(&[self.spec.num_rows()])?;
        grad_logits.shape().expect(&logits_dims)?;
        targets.validate_range(self.spec.vocab_size(), self.spec.ignore_index())?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_cross_entropy_bwd(
                self.cuda.session.ptr(),
                self.spec.num_rows() as u64,
                self.spec.vocab_size() as u64,
                self.spec.ignore_index().unwrap_or(-100),
                i32::from(self.spec.ignore_index().is_some()),
                self.spec.label_smoothing(),
                upstream_scalar,
                logits.read(),
                targets.read(),
                self.active_count.write().cast(),
                grad_logits.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
    /// Single-pass fused forward + backward: reads BF16 logits ONCE and writes both scalar FP32 loss and BF16 dLogits.
    pub fn forward_backward(
        &mut self,
        logits: &DeviceBf16Tensor,
        targets: &DeviceIndexTensor,
        upstream_scalar: f32,
        out_loss_f32: &mut DeviceTensor,
        grad_logits: &mut DeviceBf16Tensor,
    ) -> Result<()> {
        self.cuda.owns_bf16(logits)?;
        self.cuda.owns_index(targets)?;
        self.cuda.owns(out_loss_f32)?;
        self.cuda.owns_bf16(grad_logits)?;
        let logits_dims = [self.spec.num_rows(), self.spec.vocab_size()];
        logits.shape().expect(&logits_dims)?;
        targets.shape().expect(&[self.spec.num_rows()])?;
        out_loss_f32.shape().expect(&[])?;
        grad_logits.shape().expect(&logits_dims)?;
        targets.validate_range(self.spec.vocab_size(), self.spec.ignore_index())?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_cross_entropy_fwd_bwd(
                self.cuda.session.ptr(),
                self.spec.num_rows() as u64,
                self.spec.vocab_size() as u64,
                self.spec.ignore_index().unwrap_or(-100),
                i32::from(self.spec.ignore_index().is_some()),
                self.spec.label_smoothing(),
                upstream_scalar,
                logits.read(),
                targets.read(),
                self.active_count.write().cast(),
                self.row_losses.write(),
                self.row_valid.write().cast(),
                out_loss_f32.write(),
                grad_logits.write(),
            )
        })
    }
}

pub struct Bf16DropoutPlan {
    cuda: Cuda,
    spec: DropoutSpec,
}

impl Bf16DropoutPlan {
    pub fn new(cuda: &Cuda, spec: DropoutSpec) -> Self {
        Self {
            cuda: cuda.clone(),
            spec,
        }
    }
    pub fn forward(&self, x: &DeviceBf16Tensor, y: &mut DeviceBf16Tensor, step: u64) -> Result<()> {
        self.cuda.owns_bf16(x)?;
        self.cuda.owns_bf16(y)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_dropout_fwd(
                self.cuda.session.ptr(),
                self.spec.shape().len() as u64,
                self.spec.p(),
                self.spec.seed(),
                step,
                x.read(),
                y.write(),
            )
        })
    }
    pub fn backward(
        &self,
        dy: &DeviceBf16Tensor,
        dx: &mut DeviceBf16Tensor,
        step: u64,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_bf16(dy)?;
        self.cuda.owns_bf16(dx)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_bf16_dropout_bwd(
                self.cuda.session.ptr(),
                self.spec.shape().len() as u64,
                self.spec.p(),
                self.spec.seed(),
                step,
                dy.read(),
                dx.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

// ============================================================================
// Device-Side Zero-Host-Sync GradScaler (100% CUDA Graph Capturable)
// ============================================================================

/// Device-resident dynamic loss scaler for FP16/BF16 mixed precision (zero host sync on step!).
pub struct CudaGradScaler {
    cuda: Cuda,
    ptr: NonNull<c_void>,
}

impl CudaGradScaler {
    pub fn new(
        cuda: &Cuda,
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: i32,
    ) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_grad_scaler_create(
                cuda.session.ptr(),
                init_scale,
                growth_factor,
                backoff_factor,
                growth_interval,
                &mut ptr,
            )
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            ptr: nonnull(ptr)?,
        })
    }
    pub fn reset_step(&self) -> Result<()> {
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe { fa_cuda_grad_scaler_reset(self.cuda.session.ptr(), self.ptr.as_ptr()) })
    }
    pub fn unscale_and_check(&self, grad_f32: &mut DeviceTensor) -> Result<()> {
        self.cuda.owns(grad_f32)?;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_grad_scaler_unscale_check(
                self.cuda.session.ptr(),
                self.ptr.as_ptr(),
                grad_f32.shape().len() as u64,
                grad_f32.write(),
            )
        })
    }
    pub fn update(&self) -> Result<()> {
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe { fa_cuda_grad_scaler_update(self.cuda.session.ptr(), self.ptr.as_ptr()) })
    }
    pub fn read_state(&self) -> Result<(f32, bool)> {
        let (mut scale, mut found_inf) = (0.0f32, 0u32);
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_grad_scaler_read_state(
                self.cuda.session.ptr(),
                self.ptr.as_ptr(),
                &mut scale,
                &mut found_inf,
            )
        })?;
        Ok((scale, found_inf != 0))
    }
    pub(crate) fn found_inf_ptr(&self) -> *const u32 {
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        unsafe { fa_cuda_grad_scaler_found_inf_ptr(self.ptr.as_ptr()) }
    }
}

impl Drop for CudaGradScaler {
    fn drop(&mut self) {
        // SAFETY: sole owner retains live CUDA session; native syncs before release.
        unsafe { fa_cuda_grad_scaler_destroy(self.cuda.session.ptr(), self.ptr.as_ptr()) }
    }
}

// ============================================================================
// Fused FP32-Master-Weight + BF16-Mirror AdamW
// ============================================================================

/// Preallocated fused FP32-master-weight + BF16-mirror `AdamW` optimizer with CUDA-Graph-safe
/// dynamic LR, deterministic FP64 global L2 gradient clipping, and optional `CudaGradScaler` skip.
pub struct Bf16MasterAdamW {
    cuda: Cuda,
    config: AdamWConfig,
    ctrl: NonNull<c_void>,
    total_partials: u64,
    param_grids: Vec<(u64, u32)>,
    m: Vec<DeviceTensor>,
    v: Vec<DeviceTensor>,
    step_count: u64,
}

impl Bf16MasterAdamW {
    pub fn new(cuda: &Cuda, shapes: &[Shape], config: AdamWConfig) -> Result<Self> {
        config.validate()?;
        if shapes.is_empty() {
            return Err(invalid("Bf16MasterAdamW requires at least one parameter"));
        }
        let mut param_grids = Vec::with_capacity(shapes.len());
        let mut total_partials = 0u64;
        let mut m = Vec::with_capacity(shapes.len());
        let mut v = Vec::with_capacity(shapes.len());
        for s in shapes {
            let n = s.len() as u64;
            let grid = ((n.div_ceil(256)).clamp(1, 64)) as u32;
            param_grids.push((total_partials, grid));
            total_partials += grid as u64;
            m.push(cuda.zeros(s.dims())?);
            v.push(cuda.zeros(s.dims())?);
        }
        let mut ctrl = std::ptr::null_mut();
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe {
            fa_cuda_adamw_state_create(cuda.session.ptr(), total_partials, config.lr, &mut ctrl)
        })?;
        Ok(Self {
            cuda: cuda.clone(),
            config,
            ctrl: nonnull(ctrl)?,
            total_partials,
            param_grids,
            m,
            v,
            step_count: 0,
        })
    }
    pub fn config(&self) -> AdamWConfig {
        self.config
    }
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
    pub fn set_lr(&mut self, lr: f32) -> Result<()> {
        if !lr.is_finite() || lr < 0.0 {
            return Err(invalid("AdamW lr must be finite and nonnegative"));
        }
        self.config.lr = lr;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe { fa_cuda_adamw_set_lr(self.cuda.session.ptr(), self.ctrl.as_ptr(), lr) })
    }
    /// Enqueues step bump + optional global L2 clip + fused FP32-master/BF16-mirror AdamW update.
    /// 100% CUDA Graph capturable (zero host sync!).
    pub fn step(
        &mut self,
        master_f32: &mut [DeviceTensor],
        mirror_bf16: &mut [DeviceBf16Tensor],
        grads_f32: &[DeviceTensor],
        scaler: Option<&CudaGradScaler>,
    ) -> Result<()> {
        if master_f32.len() != self.m.len()
            || mirror_bf16.len() != self.m.len()
            || grads_f32.len() != self.m.len()
        {
            return Err(invalid("Bf16MasterAdamW parameter list length mismatch"));
        }
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe { fa_cuda_adamw_bump_step(self.cuda.session.ptr(), self.ctrl.as_ptr()) })?;
        let use_clip = if let Some(max_norm) = self.config.max_grad_norm {
            for (idx, g) in grads_f32.iter().enumerate() {
                let (off, grid) = self.param_grids[idx];
                // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
                check(unsafe {
                    fa_cuda_adamw_grad_norm_partial(
                        self.cuda.session.ptr(),
                        g.shape().len() as u64,
                        g.read(),
                        self.ctrl.as_ptr(),
                        off,
                        grid,
                    )
                })?;
            }
            // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
            check(unsafe {
                fa_cuda_adamw_grad_norm_finalize(
                    self.cuda.session.ptr(),
                    self.ctrl.as_ptr(),
                    self.total_partials,
                    max_norm,
                )
            })?;
            1
        } else {
            0
        };
        let found_inf_ptr = scaler.map_or(std::ptr::null(), CudaGradScaler::found_inf_ptr);
        for i in 0..self.m.len() {
            let n = master_f32[i].shape().len() as u64;
            // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
            check(unsafe {
                fa_cuda_bf16_adamw_param(
                    self.cuda.session.ptr(),
                    n,
                    master_f32[i].write(),
                    mirror_bf16[i].write(),
                    grads_f32[i].read(),
                    self.m[i].write(),
                    self.v[i].write(),
                    self.ctrl.as_ptr(),
                    found_inf_ptr,
                    self.config.beta1,
                    self.config.beta2,
                    self.config.eps,
                    self.config.weight_decay,
                    use_clip,
                )
            })?;
        }
        self.step_count += 1;
        Ok(())
    }
}

impl Drop for Bf16MasterAdamW {
    fn drop(&mut self) {
        // SAFETY: sole owner retains live CUDA session; native syncs before release.
        unsafe { fa_cuda_adamw_state_destroy(self.cuda.session.ptr(), self.ctrl.as_ptr()) }
    }
}

// ============================================================================
// Single-CUDA-Graph BF16 Tensor Core Causal Audio Transformer Trainer
// ============================================================================

/// Architecture configuration for `Bf16AudioTransformerTrainer`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16AudioTransformerConfig {
    pub batch: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub ffn_hidden: usize,
    pub num_layers: usize,
    pub rope_base: f32,
    pub rms_eps: f32,
    pub ignore_index: Option<i64>,
}

impl Bf16AudioTransformerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0
            || self.seq_len == 0
            || self.vocab_size == 0
            || self.d_model == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.ffn_hidden == 0
            || self.num_layers == 0
        {
            return Err(invalid("Transformer config dimensions must be positive"));
        }
        if !self.d_model.is_multiple_of(self.num_heads)
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
        {
            return Err(invalid(
                "Transformer d_model % num_heads == 0 and num_heads % num_kv_heads == 0 required",
            ));
        }
        let head_dim = self.d_model / self.num_heads;
        if head_dim == 0 || (head_dim & 1) != 0 {
            return Err(invalid("Transformer head_dim must be positive and even"));
        }
        if !self.rope_base.is_finite() || self.rope_base <= 0.0 {
            return Err(invalid("Transformer rope_base must be finite and positive"));
        }
        if !self.rms_eps.is_finite() || self.rms_eps <= 0.0 {
            return Err(invalid("Transformer rms_eps must be finite and positive"));
        }
        Ok(())
    }
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
    pub fn qkv_heads(&self) -> usize {
        self.num_heads + 2 * self.num_kv_heads
    }
    pub fn qkv_dim(&self) -> usize {
        self.qkv_heads() * self.head_dim()
    }
    pub fn tokens(&self) -> usize {
        self.batch * self.seq_len
    }
}

struct CapturedGraphExec {
    cuda: Cuda,
    exec: NonNull<c_void>,
}

impl Drop for CapturedGraphExec {
    fn drop(&mut self) {
        // SAFETY: sole owner retains live CUDA session; native syncs before release.
        unsafe { fa_cuda_graph_destroy(self.cuda.session.ptr(), self.exec.as_ptr()) }
    }
}

struct CaptureAbortGuard(*mut c_void, bool);
impl Drop for CaptureAbortGuard {
    fn drop(&mut self) {
        if self.1 {
            // SAFETY: sole owner retains live CUDA session; native syncs before release.
            unsafe { fa_cuda_capture_abort(self.0) }
        }
    }
}

/// Complete N-layer Causal Audio Transformer trainer in BF16 Tensor Core mixed precision with
/// FP32 master weights, zero-copy QKV FlashAttention-2, fused Residual+RmsNorm, fused SwiGLU,
/// single-pass CrossEntropy, and single-CUDA-Graph dynamic-LR training step!
pub struct Bf16AudioTransformerTrainer {
    cuda: Cuda,
    cfg: Bf16AudioTransformerConfig,
    // Flat parameter lists (indices: 0=tok_emb, 1..1+6*L=layers, 1+6*L=final_rms, 2+6*L=lm_head)
    master_f32: Vec<DeviceTensor>,
    mirror_bf16: Vec<DeviceBf16Tensor>,
    grads_f32: Vec<DeviceTensor>,
    optim: Bf16MasterAdamW,
    // Fixed device I/O buffers for CUDA Graph replay
    input_ids: DeviceIndexTensor,
    target_ids: DeviceIndexTensor,
    loss_scalar: DeviceTensor,
    // Saved per-layer forward activations (O(L * B * T * D) — zero T^2 allocation!)
    x_in: Vec<DeviceBf16Tensor>,
    x_norm1: Vec<DeviceBf16Tensor>,
    inv_rms1: Vec<DeviceTensor>,
    qkv: Vec<DeviceBf16Tensor>,
    attn_out: Vec<DeviceBf16Tensor>,
    res1: Vec<DeviceBf16Tensor>,
    x_norm2: Vec<DeviceBf16Tensor>,
    inv_rms2: Vec<DeviceTensor>,
    gate_up: Vec<DeviceBf16Tensor>,
    swiglu: Vec<DeviceBf16Tensor>,
    x_final: DeviceBf16Tensor,
    h_final: DeviceBf16Tensor,
    inv_rms_final: DeviceTensor,
    logits: DeviceBf16Tensor,
    // Reused backward scratch (shared across layers — O(1) layers!)
    dlogits: DeviceBf16Tensor,
    dres: DeviceBf16Tensor,
    dnorm: DeviceBf16Tensor,
    dswiglu: DeviceBf16Tensor,
    dgate_up: DeviceBf16Tensor,
    dattn_out: DeviceBf16Tensor,
    dqkv: DeviceBf16Tensor,
    proj_tmp: DeviceBf16Tensor,
    // Prepared plans
    emb_plan: Bf16EmbeddingPlan,
    norm_plan: Bf16NormPlan,
    rope_plan: Bf16RopePlan,
    sdpa_plans: Vec<Bf16SdpaPlan>,
    qkv_lin: Bf16LinearPlan,
    out_lin: Bf16LinearPlan,
    gate_up_lin: Bf16LinearPlan,
    down_lin: Bf16LinearPlan,
    lm_head_lin: Bf16LinearPlan,
    swiglu_plan: Bf16SwiGluPlan,
    ce_plan: Bf16CrossEntropyPlan,
    // Single-CUDA-Graph handle
    captured: Option<CapturedGraphExec>,
}

impl Bf16AudioTransformerTrainer {
    /// Parameter slot order:
    /// `[tok_emb, (rms1, w_qkv, w_o, rms2, w_gate_up, w_down) * num_layers, final_rms, lm_head]`.
    pub fn parameter_shapes(cfg: &Bf16AudioTransformerConfig) -> Result<Vec<Shape>> {
        cfg.validate()?;
        let mut shapes = Vec::with_capacity(3 + cfg.num_layers * 6);
        shapes.push(Shape::new([cfg.vocab_size, cfg.d_model])?);
        for _ in 0..cfg.num_layers {
            shapes.push(Shape::new([cfg.d_model])?);
            shapes.push(Shape::new([cfg.qkv_dim(), cfg.d_model])?);
            shapes.push(Shape::new([cfg.d_model, cfg.d_model])?);
            shapes.push(Shape::new([cfg.d_model])?);
            shapes.push(Shape::new([2 * cfg.ffn_hidden, cfg.d_model])?);
            shapes.push(Shape::new([cfg.d_model, cfg.ffn_hidden])?);
        }
        shapes.push(Shape::new([cfg.d_model])?);
        shapes.push(Shape::new([cfg.vocab_size, cfg.d_model])?);
        Ok(shapes)
    }

    pub fn new(
        cuda: &Cuda,
        cfg: Bf16AudioTransformerConfig,
        initial_weights_f32: &[Tensor],
        optim_cfg: AdamWConfig,
    ) -> Result<Self> {
        cfg.validate()?;
        let shapes = Self::parameter_shapes(&cfg)?;
        if initial_weights_f32.len() != shapes.len() {
            return Err(invalid(format!(
                "expected {} weight tensors, got {}",
                shapes.len(),
                initial_weights_f32.len()
            )));
        }
        let mut master_f32 = Vec::with_capacity(shapes.len());
        let mut mirror_bf16 = Vec::with_capacity(shapes.len());
        let mut grads_f32 = Vec::with_capacity(shapes.len());
        for (w, s) in initial_weights_f32.iter().zip(&shapes) {
            w.shape().expect(s.dims())?;
            let dev_f32 = cuda.upload(w)?;
            let mut dev_bf16 = cuda.zeros_bf16(s.dims())?;
            cuda.cast_f32_to_bf16(&dev_f32, &mut dev_bf16)?;
            master_f32.push(dev_f32);
            mirror_bf16.push(dev_bf16);
            grads_f32.push(cuda.zeros(s.dims())?);
        }
        let optim = Bf16MasterAdamW::new(cuda, &shapes, optim_cfg)?;

        let bt = cfg.tokens();
        let d = cfg.d_model;
        let qkv_d = cfg.qkv_dim();
        let ffn = cfg.ffn_hidden;

        let input_ids = cuda.upload_indices(&IndexTensor::from_vec(
            [cfg.batch, cfg.seq_len],
            vec![0; bt],
        )?)?;
        let target_ids = cuda.upload_indices(&IndexTensor::from_vec([bt], vec![0; bt])?)?;
        let loss_scalar = cuda.zeros([])?;

        let mut x_in = Vec::with_capacity(cfg.num_layers);
        let mut x_norm1 = Vec::with_capacity(cfg.num_layers);
        let mut inv_rms1 = Vec::with_capacity(cfg.num_layers);
        let mut qkv = Vec::with_capacity(cfg.num_layers);
        let mut attn_out = Vec::with_capacity(cfg.num_layers);
        let mut res1 = Vec::with_capacity(cfg.num_layers);
        let mut x_norm2 = Vec::with_capacity(cfg.num_layers);
        let mut inv_rms2 = Vec::with_capacity(cfg.num_layers);
        let mut gate_up = Vec::with_capacity(cfg.num_layers);
        let mut swiglu = Vec::with_capacity(cfg.num_layers);
        let mut sdpa_plans = Vec::with_capacity(cfg.num_layers);

        let sdpa_spec = SdpaSpec::new(
            cfg.batch,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.seq_len,
            cfg.seq_len,
            cfg.head_dim(),
            None,
            true,
        )?;

        for _ in 0..cfg.num_layers {
            x_in.push(cuda.zeros_bf16([bt, d])?);
            x_norm1.push(cuda.zeros_bf16([bt, d])?);
            inv_rms1.push(cuda.zeros([bt])?);
            qkv.push(cuda.zeros_bf16([bt, qkv_d])?);
            attn_out.push(cuda.zeros_bf16([bt, d])?);
            res1.push(cuda.zeros_bf16([bt, d])?);
            x_norm2.push(cuda.zeros_bf16([bt, d])?);
            inv_rms2.push(cuda.zeros([bt])?);
            gate_up.push(cuda.zeros_bf16([bt, 2 * ffn])?);
            swiglu.push(cuda.zeros_bf16([bt, ffn])?);
            sdpa_plans.push(Bf16SdpaPlan::new(cuda, sdpa_spec)?);
        }

        let x_final = cuda.zeros_bf16([bt, d])?;
        let h_final = cuda.zeros_bf16([bt, d])?;
        let inv_rms_final = cuda.zeros([bt])?;
        let logits = cuda.zeros_bf16([bt, cfg.vocab_size])?;

        let dlogits = cuda.zeros_bf16([bt, cfg.vocab_size])?;
        let dres = cuda.zeros_bf16([bt, d])?;
        let dnorm = cuda.zeros_bf16([bt, d])?;
        let dswiglu = cuda.zeros_bf16([bt, ffn])?;
        let dgate_up = cuda.zeros_bf16([bt, 2 * ffn])?;
        let dattn_out = cuda.zeros_bf16([bt, d])?;
        let dqkv = cuda.zeros_bf16([bt, qkv_d])?;
        let proj_tmp = cuda.zeros_bf16([bt, d])?;

        let emb_spec = EmbeddingSpec::new(
            cfg.vocab_size,
            d,
            &Shape::new([cfg.batch, cfg.seq_len])?,
            None,
        )?;
        let emb_plan = Bf16EmbeddingPlan::new(cuda, emb_spec);
        let norm_plan = Bf16NormPlan::new(cuda, bt, d, cfg.rms_eps)?;
        let rope_shape = Shape::new([cfg.batch, cfg.seq_len, cfg.num_heads, cfg.head_dim()])?;
        let rope_spec = RopeSpec::new(&rope_shape, 1, Some(cfg.head_dim()), cfg.rope_base, 0)?;
        let rope_plan = Bf16RopePlan::new(cuda, rope_spec)?;
        let qkv_lin = Bf16LinearPlan::new(cuda, bt, d, qkv_d)?;
        let out_lin = Bf16LinearPlan::new(cuda, bt, d, d)?;
        let gate_up_lin = Bf16LinearPlan::new(cuda, bt, d, 2 * ffn)?;
        let down_lin = Bf16LinearPlan::new(cuda, bt, ffn, d)?;
        let lm_head_lin = Bf16LinearPlan::new(cuda, bt, d, cfg.vocab_size)?;
        let swiglu_plan = Bf16SwiGluPlan::new(cuda, bt, ffn)?;
        let ce_spec = CrossEntropySpec::new(bt, cfg.vocab_size, cfg.ignore_index, 0.0)?;
        let ce_plan = Bf16CrossEntropyPlan::new(cuda, ce_spec)?;

        Ok(Self {
            cuda: cuda.clone(),
            cfg,
            master_f32,
            mirror_bf16,
            grads_f32,
            optim,
            input_ids,
            target_ids,
            loss_scalar,
            x_in,
            x_norm1,
            inv_rms1,
            qkv,
            attn_out,
            res1,
            x_norm2,
            inv_rms2,
            gate_up,
            swiglu,
            x_final,
            h_final,
            inv_rms_final,
            logits,
            dlogits,
            dres,
            dnorm,
            dswiglu,
            dgate_up,
            dattn_out,
            dqkv,
            proj_tmp,
            emb_plan,
            norm_plan,
            rope_plan,
            sdpa_plans,
            qkv_lin,
            out_lin,
            gate_up_lin,
            down_lin,
            lm_head_lin,
            swiglu_plan,
            ce_plan,
            captured: None,
        })
    }

    pub fn config(&self) -> Bf16AudioTransformerConfig {
        self.cfg
    }
    pub fn has_captured_graph(&self) -> bool {
        self.captured.is_some()
    }
    pub fn set_lr(&mut self, lr: f32) -> Result<()> {
        self.optim.set_lr(lr)
    }
    pub fn set_batch(&mut self, input_ids: &IndexTensor, target_ids: &IndexTensor) -> Result<()> {
        self.cuda
            .copy_indices_from_host(input_ids, &mut self.input_ids)?;
        self.cuda
            .copy_indices_from_host(target_ids, &mut self.target_ids)?;
        Ok(())
    }
    pub fn download_loss(&self) -> Result<f32> {
        Ok(self.cuda.download(&self.loss_scalar)?.data()[0])
    }
    pub fn download_master_weights(&self) -> Result<Vec<Tensor>> {
        self.master_f32
            .iter()
            .map(|w| self.cuda.download(w))
            .collect()
    }

    /// Enqueues the entire Forward + Backward + Global Clip + Master-Weight AdamW on `ctx->stream`.
    pub fn enqueue_step(&mut self) -> Result<()> {
        let l_count = self.cfg.num_layers;
        let hq = self.cfg.num_heads;
        let hkv = self.cfg.num_kv_heads;
        let h_tot = self.cfg.qkv_heads();
        let t = self.cfg.seq_len;
        let hd = self.cfg.head_dim();
        let s_qkv = Bf16SdpaStrides::bthd(t, h_tot, hd);
        let s_o = Bf16SdpaStrides::bthd(t, hq, hd);
        let final_rms_idx = 1 + l_count * 6;
        let lm_head_idx = 2 + l_count * 6;

        // --- FORWARD ---
        self.emb_plan
            .forward(&self.mirror_bf16[0], &self.input_ids, &mut self.x_in[0])?;
        self.norm_plan.rmsnorm_forward(
            &self.x_in[0],
            Some(&self.mirror_bf16[1]),
            &mut self.x_norm1[0],
            Some(&mut self.inv_rms1[0]),
        )?;

        for l in 0..l_count {
            let p = 1 + l * 6;
            // Fused QKV projection -> in-place Q/K RoPE -> zero-copy strided FlashAttention-2
            self.qkv_lin
                .forward(&self.x_norm1[l], &self.mirror_bf16[p + 1], &mut self.qkv[l])?;
            self.rope_plan
                .run_inplace_bthd_slice(&mut self.qkv[l], 0, hq, h_tot, false)?;
            self.rope_plan
                .run_inplace_bthd_slice(&mut self.qkv[l], hq, hkv, h_tot, false)?;
            self.sdpa_plans[l].forward_strided(
                &self.qkv[l],
                0,
                s_qkv,
                &self.qkv[l],
                hq * hd,
                s_qkv,
                &self.qkv[l],
                (hq + hkv) * hd,
                s_qkv,
                &mut self.attn_out[l],
                0,
                s_o,
            )?;
            // Output projection + fused ResidualAdd + RmsNorm2
            self.out_lin.forward(
                &self.attn_out[l],
                &self.mirror_bf16[p + 2],
                &mut self.proj_tmp,
            )?;
            self.norm_plan.residual_rmsnorm_forward(
                &self.x_in[l],
                &self.proj_tmp,
                Some(&self.mirror_bf16[p + 3]),
                &mut self.res1[l],
                &mut self.x_norm2[l],
                Some(&mut self.inv_rms2[l]),
            )?;
            // Fused GateUp -> SwiGLU -> DownProj + fused ResidualAdd + next RmsNorm
            self.gate_up_lin.forward(
                &self.x_norm2[l],
                &self.mirror_bf16[p + 4],
                &mut self.gate_up[l],
            )?;
            self.swiglu_plan
                .forward(&self.gate_up[l], &mut self.swiglu[l])?;
            self.down_lin.forward(
                &self.swiglu[l],
                &self.mirror_bf16[p + 5],
                &mut self.proj_tmp,
            )?;
            if l + 1 < l_count {
                let next_p = 1 + (l + 1) * 6;
                self.norm_plan.residual_rmsnorm_forward(
                    &self.res1[l],
                    &self.proj_tmp,
                    Some(&self.mirror_bf16[next_p]),
                    &mut self.x_in[l + 1],
                    &mut self.x_norm1[l + 1],
                    Some(&mut self.inv_rms1[l + 1]),
                )?;
            } else {
                self.norm_plan.residual_rmsnorm_forward(
                    &self.res1[l],
                    &self.proj_tmp,
                    Some(&self.mirror_bf16[final_rms_idx]),
                    &mut self.x_final,
                    &mut self.h_final,
                    Some(&mut self.inv_rms_final),
                )?;
            }
        }

        // LM Head + Single-Pass Fused CrossEntropy (forward + backward in 1 pass!)
        self.lm_head_lin.forward(
            &self.h_final,
            &self.mirror_bf16[lm_head_idx],
            &mut self.logits,
        )?;
        self.ce_plan.forward_backward(
            &self.logits,
            &self.target_ids,
            1.0,
            &mut self.loss_scalar,
            &mut self.dlogits,
        )?;

        // --- BACKWARD ---
        self.lm_head_lin.backward_input(
            &self.dlogits,
            &self.mirror_bf16[lm_head_idx],
            &mut self.dnorm,
            GradientMode::Overwrite,
        )?;
        self.lm_head_lin.backward_weight_f32(
            &self.dlogits,
            &self.h_final,
            &mut self.grads_f32[lm_head_idx],
            GradientMode::Overwrite,
        )?;
        self.norm_plan.rmsnorm_backward(
            &self.x_final,
            Some(&self.mirror_bf16[final_rms_idx]),
            Some(&self.inv_rms_final),
            &self.dnorm,
            &mut self.dres,
            Some(&mut self.grads_f32[final_rms_idx]),
            GradientMode::Overwrite,
            GradientMode::Overwrite,
        )?;

        for l in (0..l_count).rev() {
            let p = 1 + l * 6;
            // DownProj bwd -> SwiGLU bwd -> GateUp bwd -> RmsNorm2 bwd (accumulates into dres!)
            self.down_lin.backward_input(
                &self.dres,
                &self.mirror_bf16[p + 5],
                &mut self.dswiglu,
                GradientMode::Overwrite,
            )?;
            self.down_lin.backward_weight_f32(
                &self.dres,
                &self.swiglu[l],
                &mut self.grads_f32[p + 5],
                GradientMode::Overwrite,
            )?;
            self.swiglu_plan
                .backward(&self.gate_up[l], &self.dswiglu, &mut self.dgate_up)?;
            self.gate_up_lin.backward_input(
                &self.dgate_up,
                &self.mirror_bf16[p + 4],
                &mut self.dnorm,
                GradientMode::Overwrite,
            )?;
            self.gate_up_lin.backward_weight_f32(
                &self.dgate_up,
                &self.x_norm2[l],
                &mut self.grads_f32[p + 4],
                GradientMode::Overwrite,
            )?;
            self.norm_plan.rmsnorm_backward(
                &self.res1[l],
                Some(&self.mirror_bf16[p + 3]),
                Some(&self.inv_rms2[l]),
                &self.dnorm,
                &mut self.dres,
                Some(&mut self.grads_f32[p + 3]),
                GradientMode::Accumulate,
                GradientMode::Overwrite,
            )?;

            // OutProj bwd -> zero-copy FlashAttention-2 bwd -> in-place RoPE bwd -> QKV bwd -> RmsNorm1 bwd (accumulates into dres!)
            self.out_lin.backward_input(
                &self.dres,
                &self.mirror_bf16[p + 2],
                &mut self.dattn_out,
                GradientMode::Overwrite,
            )?;
            self.out_lin.backward_weight_f32(
                &self.dres,
                &self.attn_out[l],
                &mut self.grads_f32[p + 2],
                GradientMode::Overwrite,
            )?;
            self.sdpa_plans[l].backward_fused_qkv(
                &self.qkv[l],
                &self.attn_out[l],
                &self.dattn_out,
                &mut self.dqkv,
            )?;
            self.rope_plan
                .run_inplace_bthd_slice(&mut self.dqkv, 0, hq, h_tot, true)?;
            self.rope_plan
                .run_inplace_bthd_slice(&mut self.dqkv, hq, hkv, h_tot, true)?;
            self.qkv_lin.backward_input(
                &self.dqkv,
                &self.mirror_bf16[p + 1],
                &mut self.dnorm,
                GradientMode::Overwrite,
            )?;
            self.qkv_lin.backward_weight_f32(
                &self.dqkv,
                &self.x_norm1[l],
                &mut self.grads_f32[p + 1],
                GradientMode::Overwrite,
            )?;
            self.norm_plan.rmsnorm_backward(
                &self.x_in[l],
                Some(&self.mirror_bf16[p]),
                Some(&self.inv_rms1[l]),
                &self.dnorm,
                &mut self.dres,
                Some(&mut self.grads_f32[p]),
                GradientMode::Accumulate,
                GradientMode::Overwrite,
            )?;
        }

        self.emb_plan.backward_f32(
            &self.input_ids,
            &self.dres,
            &mut self.grads_f32[0],
            GradientMode::Overwrite,
        )?;

        // --- OPTIMIZER (Global L2 Clip + Fused FP32-Master / BF16-Mirror AdamW) ---
        self.optim.step(
            &mut self.master_f32,
            &mut self.mirror_bf16,
            &self.grads_f32,
            None,
        )
    }

    /// Captures the full Forward + Backward + Master-Weight AdamW step into ONE CUDA Graph
    /// (zero host launch overhead on replay; dynamic LR via `set_lr` without recapture).
    pub fn capture_step_graph(&mut self) -> Result<()> {
        self.captured = None;
        self.cuda.synchronize()?;
        let ptr = self.cuda.session.ptr();
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe { fa_cuda_capture_begin(ptr) })?;
        let mut guard = CaptureAbortGuard(ptr, true);
        self.enqueue_step()?;
        let mut exec = std::ptr::null_mut();
        guard.1 = false;
        // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
        check(unsafe { fa_cuda_capture_end(ptr, &mut exec) })?;
        self.captured = Some(CapturedGraphExec {
            cuda: self.cuda.clone(),
            exec: nonnull(exec)?,
        });
        Ok(())
    }

    /// Executes one full training step using the captured single CUDA Graph (replays if captured,
    /// otherwise runs eager on stream).
    pub fn step_captured(&mut self) -> Result<()> {
        if let Some(ref g) = self.captured {
            // SAFETY: session ownership, shapes, and exclusive output buffers are validated.
            check(unsafe { fa_cuda_graph_launch(self.cuda.session.ptr(), g.exec.as_ptr()) })
        } else {
            self.enqueue_step()
        }
    }
}
