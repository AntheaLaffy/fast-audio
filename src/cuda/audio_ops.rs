use super::{Cuda, DeviceIndexTensor, DeviceTensor, DeviceView, DeviceViewMut, check, nonnull};
use crate::{
    CrossEntropySpec, DropoutSpec, EmbeddingSpec, GradientMode, Layout, Result, RopeSpec,
    error::invalid, rope::RopeView4,
};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeRopeDims {
    dims: [u64; 4],
    in_strides: [i64; 4],
    out_strides: [i64; 4],
    in_offset: i64,
    out_offset: i64,
    seq_axis: i32,
    rotary_dim: i32,
    contiguous_same: i32,
    reserved: i32,
}

const _: () = assert!(size_of::<NativeRopeDims>() == 128);

unsafe extern "C" {
    fn fa_cuda_embedding_fwd_f32(
        ctx: *mut c_void,
        vocab_size: u64,
        embed_dim: u64,
        num_indices: u64,
        table: *const f32,
        indices: *const i64,
        out: *mut f32,
    ) -> i32;
    fn fa_cuda_embedding_bwd_f32(
        ctx: *mut c_void,
        vocab_size: u64,
        embed_dim: u64,
        num_indices: u64,
        padding_idx: i64,
        has_padding_idx: i32,
        indices: *const i64,
        upstream: *const f32,
        grad_table: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_rope_create(
        ctx: *mut c_void,
        dims: NativeRopeDims,
        cos_host: *const f32,
        sin_host: *const f32,
        workspace: *mut usize,
        shared: *mut usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_rope_run(
        plan: *mut c_void,
        input: *const f32,
        output: *mut f32,
        inverse: i32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_rope_destroy(plan: *mut c_void);
    fn fa_cuda_cross_entropy_fwd_f32(
        ctx: *mut c_void,
        num_rows: u64,
        vocab_size: u64,
        ignore_index: i64,
        has_ignore_index: i32,
        label_smoothing: f32,
        logits: *const f32,
        targets: *const i64,
        row_losses_scratch: *mut f32,
        row_valid_scratch: *mut i32,
        out_scalar: *mut f32,
    ) -> i32;
    fn fa_cuda_cross_entropy_bwd_f32(
        ctx: *mut c_void,
        num_rows: u64,
        vocab_size: u64,
        ignore_index: i64,
        has_ignore_index: i32,
        label_smoothing: f32,
        upstream_scalar: f32,
        logits: *const f32,
        targets: *const i64,
        active_count_scratch: *mut i32,
        grad_logits: *mut f32,
        accumulate: i32,
    ) -> i32;
    fn fa_cuda_dropout_fwd_f32(
        ctx: *mut c_void,
        n: u64,
        p: f32,
        seed: u64,
        step: u64,
        input: *const f32,
        output: *mut f32,
    ) -> i32;
    fn fa_cuda_dropout_bwd_f32(
        ctx: *mut c_void,
        n: u64,
        p: f32,
        seed: u64,
        step: u64,
        upstream: *const f32,
        output: *mut f32,
        accumulate: i32,
    ) -> i32;
}

/// Prepared CUDA FP32 token `Embedding` forward plan.
pub struct EmbeddingPlan {
    cuda: Cuda,
    spec: EmbeddingSpec,
}

impl EmbeddingPlan {
    pub fn new(cuda: &Cuda, spec: EmbeddingSpec) -> Self {
        Self {
            cuda: cuda.clone(),
            spec,
        }
    }
    pub fn spec(&self) -> &EmbeddingSpec {
        &self.spec
    }
    pub fn run(
        &self,
        table: &DeviceTensor,
        indices: &DeviceIndexTensor,
        output: &mut DeviceTensor,
    ) -> Result<()> {
        self.cuda.owns(table)?;
        self.cuda.owns_index(indices)?;
        self.cuda.owns(output)?;
        table
            .shape()
            .expect(&[self.spec.vocab_size(), self.spec.embed_dim()])?;
        indices.shape().expect(self.spec.index_shape().dims())?;
        output.shape().expect(self.spec.output_shape().dims())?;
        indices.validate_range(self.spec.vocab_size(), None)?;
        // SAFETY: validated session ownership, shapes, and host-tracked index bounds.
        check(unsafe {
            fa_cuda_embedding_fwd_f32(
                self.cuda.session.ptr(),
                self.spec.vocab_size() as u64,
                self.spec.embed_dim() as u64,
                self.spec.num_indices() as u64,
                table.read(),
                indices.read(),
                output.write(),
            )
        })
    }
}

/// Prepared CUDA FP32 token `Embedding` deterministic backward plan.
pub struct EmbeddingBackward {
    cuda: Cuda,
    spec: EmbeddingSpec,
}

impl EmbeddingBackward {
    pub fn new(cuda: &Cuda, spec: EmbeddingSpec) -> Self {
        Self {
            cuda: cuda.clone(),
            spec,
        }
    }
    pub fn spec(&self) -> &EmbeddingSpec {
        &self.spec
    }
    pub fn run(
        &self,
        indices: &DeviceIndexTensor,
        upstream: &DeviceTensor,
        grad_table: &mut DeviceTensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns_index(indices)?;
        self.cuda.owns(upstream)?;
        self.cuda.owns(grad_table)?;
        indices.shape().expect(self.spec.index_shape().dims())?;
        upstream.shape().expect(self.spec.output_shape().dims())?;
        grad_table
            .shape()
            .expect(&[self.spec.vocab_size(), self.spec.embed_dim()])?;
        indices.validate_range(self.spec.vocab_size(), None)?;
        // SAFETY: validated shapes, index bounds, and exclusive grad_table write buffer.
        check(unsafe {
            fa_cuda_embedding_bwd_f32(
                self.cuda.session.ptr(),
                self.spec.vocab_size() as u64,
                self.spec.embed_dim() as u64,
                self.spec.num_indices() as u64,
                self.spec.padding_idx().unwrap_or(0),
                i32::from(self.spec.padding_idx().is_some()),
                indices.read(),
                upstream.read(),
                grad_table.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

struct RopeCore {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    workspace: usize,
    shared: usize,
}

impl RopeCore {
    fn new(cuda: &Cuda, spec: &RopeSpec, inp: &Layout, out: &Layout) -> Result<Self> {
        let iv = RopeView4::new(inp, spec.shape(), false)?;
        let ov = RopeView4::new(out, spec.shape(), true)?;
        let d = spec.shape().dims();
        let dims = NativeRopeDims {
            dims: [d[0] as u64, d[1] as u64, d[2] as u64, d[3] as u64],
            in_strides: [
                iv.strides[0] as i64,
                iv.strides[1] as i64,
                iv.strides[2] as i64,
                iv.strides[3] as i64,
            ],
            out_strides: [
                ov.strides[0] as i64,
                ov.strides[1] as i64,
                ov.strides[2] as i64,
                ov.strides[3] as i64,
            ],
            in_offset: iv.offset as i64,
            out_offset: ov.offset as i64,
            seq_axis: spec.seq_axis() as i32,
            rotary_dim: spec.rotary_dim() as i32,
            contiguous_same: i32::from(inp.is_contiguous() && out.is_contiguous()),
            reserved: 0,
        };
        let (cos, sin) = spec.build_tables();
        let mut ptr = std::ptr::null_mut();
        let (mut workspace, mut shared) = (0, 0);
        // SAFETY: validated geometry; native copies cos/sin tables synchronously before return.
        check(unsafe {
            fa_cuda_rope_create(
                cuda.session.ptr(),
                dims,
                cos.as_ptr(),
                sin.as_ptr(),
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

impl Drop for RopeCore {
    fn drop(&mut self) {
        // SAFETY: sole plan owner retains live CUDA session; native syncs before free.
        unsafe { fa_cuda_rope_destroy(self.ptr.as_ptr()) }
    }
}

/// Prepared strided 4D `RoPE` forward plan on CUDA (integrated with `cuda::Tape`).
pub struct RopePlan {
    spec: RopeSpec,
    input_layout: Layout,
    output_layout: Layout,
    core: RopeCore,
}

impl RopePlan {
    pub fn new(cuda: &Cuda, spec: &RopeSpec, input: &Layout, output: &Layout) -> Result<Self> {
        let core = RopeCore::new(cuda, spec, input, output)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
            core,
        })
    }
    pub fn spec(&self) -> &RopeSpec {
        &self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        self.core.workspace
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }
    pub fn shared_memory_bytes_per_block(&self) -> usize {
        self.core.shared
    }
    pub fn run(&self, input: &DeviceView<'_>, output: &mut DeviceViewMut<'_>) -> Result<()> {
        self.core.cuda.owns(input.tensor)?;
        self.core.cuda.owns(output.tensor)?;
        if input.layout() != &self.input_layout || output.layout() != &self.output_layout {
            return Err(invalid("CUDA RoPE layouts differ from prepared plan"));
        }
        // SAFETY: validated session, shapes, layouts, and exclusive injective output.
        check(unsafe {
            fa_cuda_rope_run(
                self.core.ptr.as_ptr(),
                input.tensor.read(),
                output.tensor.write(),
                0,
                0,
            )
        })
    }
}

/// Prepared strided 4D `RoPE` local VJP on CUDA (`input_required() == false`).
pub struct RopeBackward {
    spec: RopeSpec,
    upstream_layout: Layout,
    output_layout: Layout,
    core: RopeCore,
}

impl RopeBackward {
    pub fn new(cuda: &Cuda, spec: &RopeSpec, upstream: &Layout, output: &Layout) -> Result<Self> {
        let core = RopeCore::new(cuda, spec, upstream, output)?;
        Ok(Self {
            spec: spec.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            core,
        })
    }
    pub fn spec(&self) -> &RopeSpec {
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
        self.core.shared
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
                "CUDA RoPE backward layouts differ from prepared plan",
            ));
        }
        // SAFETY: validated session, layouts, and exclusive output.
        check(unsafe {
            fa_cuda_rope_run(
                self.core.ptr.as_ptr(),
                upstream.tensor.read(),
                output.tensor.write(),
                1,
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}

/// Prepared CUDA FP32 fused `CrossEntropy` forward plan (O(N) scratch, zero `[N, V]` intermediate).
pub struct CrossEntropyPlan {
    cuda: Cuda,
    spec: CrossEntropySpec,
    row_losses: DeviceTensor,
    row_valid: DeviceTensor,
}

impl CrossEntropyPlan {
    pub fn new(cuda: &Cuda, spec: CrossEntropySpec) -> Result<Self> {
        let n = spec.num_rows().max(1);
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            row_losses: cuda.zeros([n])?,
            row_valid: cuda.zeros([n])?,
        })
    }
    pub fn spec(&self) -> CrossEntropySpec {
        self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        self.spec.num_rows().max(1) * 8
    }
    pub fn run(
        &mut self,
        logits: &DeviceTensor,
        targets: &DeviceIndexTensor,
        output_scalar: &mut DeviceTensor,
    ) -> Result<()> {
        self.cuda.owns(logits)?;
        self.cuda.owns_index(targets)?;
        self.cuda.owns(output_scalar)?;
        logits
            .shape()
            .expect(&[self.spec.num_rows(), self.spec.vocab_size()])?;
        targets.shape().expect(&[self.spec.num_rows()])?;
        output_scalar.shape().expect(&[])?;
        targets.validate_range(self.spec.vocab_size(), self.spec.ignore_index())?;
        // SAFETY: validated shapes, scratch capacities (`row_valid` holds i32s with same 4B stride as f32),
        // and session ownership.
        check(unsafe {
            fa_cuda_cross_entropy_fwd_f32(
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
                output_scalar.write(),
            )
        })
    }
}

/// Prepared CUDA FP32 fused `CrossEntropy` backward plan (O(1) scratch, zero `[N, V]` intermediate).
pub struct CrossEntropyBackward {
    cuda: Cuda,
    spec: CrossEntropySpec,
    active_count: DeviceTensor,
}

impl CrossEntropyBackward {
    pub fn new(cuda: &Cuda, spec: CrossEntropySpec) -> Result<Self> {
        Ok(Self {
            cuda: cuda.clone(),
            spec,
            active_count: cuda.zeros([1])?,
        })
    }
    pub fn spec(&self) -> CrossEntropySpec {
        self.spec
    }
    pub fn run(
        &mut self,
        logits: &DeviceTensor,
        targets: &DeviceIndexTensor,
        upstream_scalar: f32,
        grad_logits: &mut DeviceTensor,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns(logits)?;
        self.cuda.owns_index(targets)?;
        self.cuda.owns(grad_logits)?;
        logits
            .shape()
            .expect(&[self.spec.num_rows(), self.spec.vocab_size()])?;
        targets.shape().expect(&[self.spec.num_rows()])?;
        grad_logits
            .shape()
            .expect(&[self.spec.num_rows(), self.spec.vocab_size()])?;
        targets.validate_range(self.spec.vocab_size(), self.spec.ignore_index())?;
        // SAFETY: validated shapes, target bounds, and exclusive output buffer.
        check(unsafe {
            fa_cuda_cross_entropy_bwd_f32(
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
}

/// Prepared deterministic CUDA FP32 `Philox4x32-10` Dropout forward plan (bit-identical to CPU).
pub struct DropoutPlan {
    cuda: Cuda,
    spec: DropoutSpec,
}

impl DropoutPlan {
    pub fn new(cuda: &Cuda, spec: DropoutSpec) -> Self {
        Self {
            cuda: cuda.clone(),
            spec,
        }
    }
    pub fn spec(&self) -> &DropoutSpec {
        &self.spec
    }
    pub fn run(&self, input: &DeviceTensor, output: &mut DeviceTensor, step: u64) -> Result<()> {
        self.cuda.owns(input)?;
        self.cuda.owns(output)?;
        input.shape().expect(self.spec.shape().dims())?;
        output.shape().expect(self.spec.shape().dims())?;
        // SAFETY: validated shapes, session, and exclusive output.
        check(unsafe {
            fa_cuda_dropout_fwd_f32(
                self.cuda.session.ptr(),
                self.spec.shape().len() as u64,
                self.spec.p(),
                self.spec.seed(),
                step,
                input.read(),
                output.write(),
            )
        })
    }
}

/// Prepared deterministic CUDA FP32 `Philox4x32-10` Dropout local VJP (zero saved mask).
pub struct DropoutBackward {
    cuda: Cuda,
    spec: DropoutSpec,
}

impl DropoutBackward {
    pub fn new(cuda: &Cuda, spec: DropoutSpec) -> Self {
        Self {
            cuda: cuda.clone(),
            spec,
        }
    }
    pub fn spec(&self) -> &DropoutSpec {
        &self.spec
    }
    pub fn run(
        &self,
        upstream: &DeviceTensor,
        output: &mut DeviceTensor,
        step: u64,
        mode: GradientMode,
    ) -> Result<()> {
        self.cuda.owns(upstream)?;
        self.cuda.owns(output)?;
        upstream.shape().expect(self.spec.shape().dims())?;
        output.shape().expect(self.spec.shape().dims())?;
        // SAFETY: validated shapes, session, and exclusive output.
        check(unsafe {
            fa_cuda_dropout_bwd_f32(
                self.cuda.session.ptr(),
                self.spec.shape().len() as u64,
                self.spec.p(),
                self.spec.seed(),
                step,
                upstream.read(),
                output.write(),
                i32::from(mode == GradientMode::Accumulate),
            )
        })
    }
}
