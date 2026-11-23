use crate::{
    Result, Shape, Shared, Tensor,
    autograd::{Program, Value},
    error::invalid,
};
use std::{collections::BTreeMap, fs, path::Path};

#[cfg(feature = "cuda")]
use crate::cuda::{Cuda, DeviceTensor};
#[cfg(feature = "cuda")]
use std::{ffi::c_void, ptr::NonNull};

/// Hyperparameters for strict-FP32 decoupled `AdamW` with optional global L2 gradient clipping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: Option<f32>,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: None,
        }
    }
}

impl AdamWConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err(invalid("AdamW lr must be finite and nonnegative"));
        }
        if !self.beta1.is_finite() || !(0.0..1.0).contains(&self.beta1) {
            return Err(invalid("AdamW beta1 must lie in [0, 1)"));
        }
        if !self.beta2.is_finite() || !(0.0..1.0).contains(&self.beta2) {
            return Err(invalid("AdamW beta2 must lie in [0, 1)"));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(invalid("AdamW eps must be finite and positive"));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(invalid("AdamW weight_decay must be finite and nonnegative"));
        }
        if let Some(norm) = self.max_grad_norm {
            if !norm.is_finite() || norm <= 0.0 {
                return Err(invalid("AdamW max_grad_norm must be finite and positive"));
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn non_lr_fingerprint(&self) -> (u32, u32, u32, u32, Option<u32>) {
        (
            self.beta1.to_bits(),
            self.beta2.to_bits(),
            self.eps.to_bits(),
            self.weight_decay.to_bits(),
            self.max_grad_norm.map(f32::to_bits),
        )
    }
}

fn validate_program_leaves(program: &Program, leaves: &[Value]) -> Result<Vec<Shape>> {
    if leaves.is_empty() {
        return Err(invalid("AdamW requires at least one parameter leaf"));
    }
    let mut shapes = Vec::with_capacity(leaves.len());
    for (idx, &leaf) in leaves.iter().enumerate() {
        if leaves[..idx].contains(&leaf) {
            return Err(invalid("AdamW received duplicate parameter leaf"));
        }
        if !program.leaves().contains(&leaf) {
            return Err(invalid(
                "AdamW leaf is not a declared leaf of this training program",
            ));
        }
        if !program.requires_grad(leaf)? {
            return Err(invalid("AdamW parameter leaf must have requires_grad=true"));
        }
        shapes.push(program.shape(leaf)?.clone());
    }
    Ok(shapes)
}

#[inline]
fn sum_sq_f64(data: &[f32]) -> f64 {
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    let mut s4 = 0.0f64;
    let mut s5 = 0.0f64;
    let mut s6 = 0.0f64;
    let mut s7 = 0.0f64;
    let chunks = data.chunks_exact(8);
    let rem = chunks.remainder();
    for c in chunks {
        let a0 = c[0] as f64;
        let a1 = c[1] as f64;
        let a2 = c[2] as f64;
        let a3 = c[3] as f64;
        let a4 = c[4] as f64;
        let a5 = c[5] as f64;
        let a6 = c[6] as f64;
        let a7 = c[7] as f64;
        s0 += a0 * a0;
        s1 += a1 * a1;
        s2 += a2 * a2;
        s3 += a3 * a3;
        s4 += a4 * a4;
        s5 += a5 * a5;
        s6 += a6 * a6;
        s7 += a7 * a7;
    }
    let mut total = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for &x in rem {
        let v = x as f64;
        total += v * v;
    }
    total
}

/// Native strict-FP32 CPU decoupled `AdamW` optimizer with preallocated moments and optional
/// deterministic global L2 gradient clipping.
pub struct CpuAdamW {
    config: AdamWConfig,
    step: u64,
    leaves: Vec<Value>,
    shapes: Vec<Shape>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl CpuAdamW {
    pub fn for_program(program: &Program, leaves: &[Value], config: AdamWConfig) -> Result<Self> {
        config.validate()?;
        let shapes = validate_program_leaves(program, leaves)?;
        let mut m = Vec::with_capacity(shapes.len());
        let mut v = Vec::with_capacity(shapes.len());
        for shape in &shapes {
            m.push(Tensor::zeros(shape.dims())?);
            v.push(Tensor::zeros(shape.dims())?);
        }
        Ok(Self {
            config,
            step: 0,
            leaves: leaves.to_vec(),
            shapes,
            m,
            v,
        })
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    pub fn set_lr(&mut self, lr: f32) -> Result<()> {
        if !lr.is_finite() || lr < 0.0 {
            return Err(invalid("AdamW lr must be finite and nonnegative"));
        }
        self.config.lr = lr;
        Ok(())
    }

    pub fn set_config(&mut self, config: AdamWConfig) -> Result<()> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn leaves(&self) -> &[Value] {
        &self.leaves
    }

    pub fn state_bytes(&self) -> usize {
        self.shapes
            .iter()
            .map(|s| s.len() * size_of::<f32>() * 2)
            .sum()
    }

    pub fn moments(&self, leaf: Value) -> Result<(&Tensor, &Tensor)> {
        let idx = self
            .leaves
            .iter()
            .position(|&l| l == leaf)
            .ok_or_else(|| invalid("leaf is not managed by this CPU AdamW optimizer"))?;
        Ok((&self.m[idx], &self.v[idx]))
    }

    pub(crate) fn load_state_parts(
        &mut self,
        step: u64,
        config: AdamWConfig,
        moments: &[(&Value, &Tensor, &Tensor)],
    ) -> Result<()> {
        config.validate()?;
        if moments.len() != self.leaves.len() {
            return Err(invalid("checkpoint moment count differs from optimizer"));
        }
        for &(leaf, m_src, v_src) in moments {
            let idx = self
                .leaves
                .iter()
                .position(|l| l == leaf)
                .ok_or_else(|| invalid("checkpoint references leaf absent from optimizer"))?;
            if m_src.shape() != &self.shapes[idx] || v_src.shape() != &self.shapes[idx] {
                return Err(invalid("checkpoint moment shape differs from optimizer"));
            }
        }
        for &(leaf, m_src, v_src) in moments {
            let idx = self.leaves.iter().position(|l| l == leaf).unwrap();
            self.m[idx].data_mut().copy_from_slice(m_src.data());
            self.v[idx].data_mut().copy_from_slice(v_src.data());
        }
        self.step = step;
        self.config = config;
        Ok(())
    }

    /// Apply one AdamW update directly to `(params, grads)` slices aligned with `self.leaves()`.
    pub fn apply_slices(&mut self, params: &mut [&mut Tensor], grads: &[&Tensor]) -> Result<()> {
        if params.len() != self.leaves.len() || grads.len() != self.leaves.len() {
            return Err(invalid("AdamW parameter/gradient count mismatch"));
        }
        for (idx, ((p, g), expected)) in params.iter().zip(grads).zip(&self.shapes).enumerate() {
            if p.shape() != expected || g.shape() != expected {
                return Err(invalid(format!(
                    "AdamW tensor shape mismatch at parameter {idx}"
                )));
            }
        }
        let scale = if let Some(max_norm) = self.config.max_grad_norm {
            let mut total_sq = 0.0f64;
            for g in grads {
                total_sq += sum_sq_f64(g.data());
            }
            let norm = total_sq.sqrt();
            let clip = (max_norm as f64) / (norm + 1e-6);
            if clip < 1.0 { clip as f32 } else { 1.0f32 }
        } else {
            1.0f32
        };

        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| invalid("AdamW step counter overflow"))?;
        let t = next_step as f64;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        let bc1 = 1.0 - (b1 as f64).powf(t);
        let bc2 = 1.0 - (b2 as f64).powf(t);
        let step_size = (lr as f64 / bc1) as f32;
        let inv_sqrt_bc2 = (1.0 / bc2.sqrt()) as f32;
        let decay_factor = (1.0 - (lr as f64) * (wd as f64)) as f32;
        let one_minus_b1 = 1.0f32 - b1;
        let one_minus_b2 = 1.0f32 - b2;

        for idx in 0..self.leaves.len() {
            let p_data = params[idx].data_mut();
            let g_data = grads[idx].data();
            let m_data = self.m[idx].data_mut();
            let v_data = self.v[idx].data_mut();
            for i in 0..p_data.len() {
                let g = g_data[i] * scale;
                let p_decayed = p_data[i] * decay_factor;
                let mi = b1 * m_data[i] + one_minus_b1 * g;
                let vi = b2 * v_data[i] + one_minus_b2 * (g * g);
                let denom = vi.sqrt() * inv_sqrt_bc2 + eps;
                m_data[i] = mi;
                v_data[i] = vi;
                p_data[i] = p_decayed - step_size * (mi / denom);
            }
        }
        self.step = next_step;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn fa_cuda_adamw_state_create(
        ctx: *mut c_void,
        total_partials: u64,
        initial_lr: f32,
        out_ctrl: *mut *mut c_void,
    ) -> i32;
    fn fa_cuda_adamw_state_destroy(ctx: *mut c_void, ctrl: *mut c_void);
    fn fa_cuda_adamw_set_lr(ctx: *mut c_void, ctrl: *mut c_void, lr: f32) -> i32;
    fn fa_cuda_adamw_set_step(ctx: *mut c_void, ctrl: *mut c_void, step: u64) -> i32;
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
    fn fa_cuda_adamw_param(
        ctx: *mut c_void,
        n: u64,
        param: *mut f32,
        grad: *const f32,
        m: *mut f32,
        v: *mut f32,
        ctrl: *mut c_void,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        use_grad_scale: i32,
    ) -> i32;
}

#[cfg(feature = "cuda")]
struct CudaAdamWCtrl {
    cuda: Cuda,
    ptr: NonNull<c_void>,
    total_partials: u64,
}

#[cfg(feature = "cuda")]
impl Drop for CudaAdamWCtrl {
    fn drop(&mut self) {
        // SAFETY: sole device control-block owner retains live CUDA session; native syncs before free.
        unsafe { fa_cuda_adamw_state_destroy(self.cuda.session.ptr(), self.ptr.as_ptr()) }
    }
}

/// Native strict-FP32 CUDA decoupled `AdamW` optimizer supporting zero-sync deterministic global
/// L2 gradient clipping and single-CUDA-Graph dynamic-LR training steps.
#[cfg(feature = "cuda")]
pub struct CudaAdamW {
    cuda: Cuda,
    config: AdamWConfig,
    host_step: u64,
    leaves: Vec<Value>,
    shapes: Vec<Shape>,
    m: Vec<DeviceTensor>,
    v: Vec<DeviceTensor>,
    norm_grids: Vec<u32>,
    partials_offsets: Vec<u64>,
    ctrl: CudaAdamWCtrl,
}

#[cfg(feature = "cuda")]
impl CudaAdamW {
    pub fn for_program(
        cuda: &Cuda,
        program: &Program,
        leaves: &[Value],
        config: AdamWConfig,
    ) -> Result<Self> {
        config.validate()?;
        let shapes = validate_program_leaves(program, leaves)?;
        let mut m = Vec::with_capacity(shapes.len());
        let mut v = Vec::with_capacity(shapes.len());
        let mut norm_grids = Vec::with_capacity(shapes.len());
        let mut partials_offsets = Vec::with_capacity(shapes.len());
        let mut total_partials = 0u64;
        for shape in &shapes {
            m.push(cuda.zeros(shape.dims())?);
            v.push(cuda.zeros(shape.dims())?);
            let n = shape.len() as u64;
            let grid = if n == 0 {
                1u32
            } else {
                n.div_ceil(256).min(256) as u32
            };
            partials_offsets.push(total_partials);
            norm_grids.push(grid);
            total_partials += grid as u64;
        }
        let mut raw_ctrl = std::ptr::null_mut();
        // SAFETY: live CUDA session and validated initial_lr.
        crate::cuda::check(unsafe {
            fa_cuda_adamw_state_create(cuda.session.ptr(), total_partials, config.lr, &mut raw_ctrl)
        })?;
        let ptr = crate::cuda::nonnull(raw_ctrl)?;
        Ok(Self {
            cuda: cuda.clone(),
            config,
            host_step: 0,
            leaves: leaves.to_vec(),
            shapes,
            m,
            v,
            norm_grids,
            partials_offsets,
            ctrl: CudaAdamWCtrl {
                cuda: cuda.clone(),
                ptr,
                total_partials,
            },
        })
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    /// Update the learning rate on device asynchronously (4-byte scalar write). Captured CUDA
    /// Graphs continue replaying with zero re-capture under dynamic LR schedules.
    pub fn set_lr(&mut self, lr: f32) -> Result<()> {
        if !lr.is_finite() || lr < 0.0 {
            return Err(invalid("AdamW lr must be finite and nonnegative"));
        }
        self.config.lr = lr;
        // SAFETY: live session and owned device control block; scalar passed by value.
        crate::cuda::check(unsafe {
            fa_cuda_adamw_set_lr(self.cuda.session.ptr(), self.ctrl.ptr.as_ptr(), lr)
        })
    }

    pub fn set_config(&mut self, config: AdamWConfig) -> Result<()> {
        config.validate()?;
        let lr_changed = config.lr.to_bits() != self.config.lr.to_bits();
        self.config = config;
        if lr_changed {
            // SAFETY: live session and owned device control block.
            crate::cuda::check(unsafe {
                fa_cuda_adamw_set_lr(self.cuda.session.ptr(), self.ctrl.ptr.as_ptr(), config.lr)
            })?;
        }
        Ok(())
    }

    pub fn step_count(&self) -> u64 {
        self.host_step
    }

    pub fn leaves(&self) -> &[Value] {
        &self.leaves
    }

    pub fn state_bytes(&self) -> usize {
        let moments: usize = self
            .shapes
            .iter()
            .map(|s| s.len() * size_of::<f32>() * 2)
            .sum();
        let ctrl_bytes = 16 + (self.ctrl.total_partials.max(1) as usize) * size_of::<f64>();
        moments + ctrl_bytes
    }

    pub fn download_moments(&self, leaf: Value) -> Result<(Tensor, Tensor)> {
        let idx = self
            .leaves
            .iter()
            .position(|&l| l == leaf)
            .ok_or_else(|| invalid("leaf is not managed by this CUDA AdamW optimizer"))?;
        Ok((
            self.cuda.download(&self.m[idx])?,
            self.cuda.download(&self.v[idx])?,
        ))
    }

    pub(crate) fn ctrl_ptr(&self) -> *mut c_void {
        self.ctrl.ptr.as_ptr()
    }

    pub(crate) fn moment_ptrs(&self) -> Vec<(*const f32, *const f32)> {
        self.m
            .iter()
            .zip(&self.v)
            .map(|(m, v)| (m.read(), v.read()))
            .collect()
    }

    pub(crate) fn matches_moment_ptrs(&self, expected: &[(*const f32, *const f32)]) -> bool {
        self.m.len() == expected.len()
            && self
                .m
                .iter()
                .zip(&self.v)
                .zip(expected)
                .all(|((m, v), &(em, ev))| m.read() == em && v.read() == ev)
    }

    pub(crate) fn same_session(&self, other: &Cuda) -> bool {
        std::rc::Rc::ptr_eq(&self.cuda.session, &other.session)
    }

    pub(crate) fn bump_host_step(&mut self) -> Result<()> {
        self.host_step = self
            .host_step
            .checked_add(1)
            .ok_or_else(|| invalid("AdamW step counter overflow"))?;
        Ok(())
    }

    pub(crate) fn enqueue_apply_slices(
        &mut self,
        params: &mut [&mut DeviceTensor],
        grads: &[&DeviceTensor],
        bump_host: bool,
    ) -> Result<()> {
        if params.len() != self.leaves.len() || grads.len() != self.leaves.len() {
            return Err(invalid("CUDA AdamW parameter/gradient count mismatch"));
        }
        for (idx, ((p, g), expected)) in params.iter().zip(grads).zip(&self.shapes).enumerate() {
            self.cuda.owns(p)?;
            self.cuda.owns(g)?;
            if p.shape() != expected || g.shape() != expected {
                return Err(invalid(format!(
                    "CUDA AdamW tensor shape mismatch at parameter {idx}"
                )));
            }
        }
        let ctx = self.cuda.session.ptr();
        let ctrl = self.ctrl.ptr.as_ptr();

        // 1. Optional deterministic two-stage FP64 global L2 gradient norm clipping
        let use_grad_scale = if let Some(max_norm) = self.config.max_grad_norm {
            for (idx, g) in grads.iter().enumerate() {
                // SAFETY: validated session, non-null owned gradient & preallocated partials slice.
                crate::cuda::check(unsafe {
                    fa_cuda_adamw_grad_norm_partial(
                        ctx,
                        g.shape().len() as u64,
                        g.read(),
                        ctrl,
                        self.partials_offsets[idx],
                        self.norm_grids[idx],
                    )
                })?;
            }
            // SAFETY: all block partials were enqueued on the same FIFO stream above.
            crate::cuda::check(unsafe {
                fa_cuda_adamw_grad_norm_finalize(ctx, ctrl, self.ctrl.total_partials, max_norm)
            })?;
            1
        } else {
            0
        };

        // 2. Increment on-device step counter before parameter updates read `*d_step`
        // SAFETY: live session and owned device control block.
        crate::cuda::check(unsafe { fa_cuda_adamw_bump_step(ctx, ctrl) })?;

        // 3. Single-pass decoupled AdamW kernel per parameter
        for idx in 0..self.leaves.len() {
            let n = self.shapes[idx].len() as u64;
            // SAFETY: exclusive mutable param/m/v and distinct immutable grad on the same session.
            crate::cuda::check(unsafe {
                fa_cuda_adamw_param(
                    ctx,
                    n,
                    params[idx].write(),
                    grads[idx].read(),
                    self.m[idx].write(),
                    self.v[idx].write(),
                    ctrl,
                    self.config.beta1,
                    self.config.beta2,
                    self.config.eps,
                    self.config.weight_decay,
                    use_grad_scale,
                )
            })?;
        }

        if bump_host {
            self.bump_host_step()?;
        }
        Ok(())
    }

    pub(crate) fn load_state_parts(
        &mut self,
        step: u64,
        config: AdamWConfig,
        moments: &[(&Value, &Tensor, &Tensor)],
    ) -> Result<()> {
        config.validate()?;
        if moments.len() != self.leaves.len() {
            return Err(invalid(
                "checkpoint moment count differs from CUDA optimizer",
            ));
        }
        for &(leaf, m_src, v_src) in moments {
            let idx =
                self.leaves.iter().position(|l| l == leaf).ok_or_else(|| {
                    invalid("checkpoint references leaf absent from CUDA optimizer")
                })?;
            if m_src.shape() != &self.shapes[idx] || v_src.shape() != &self.shapes[idx] {
                return Err(invalid(
                    "checkpoint moment shape differs from CUDA optimizer",
                ));
            }
        }
        for &(leaf, m_src, v_src) in moments {
            let idx = self.leaves.iter().position(|l| l == leaf).unwrap();
            self.cuda.copy_from_host(m_src, &mut self.m[idx])?;
            self.cuda.copy_from_host(v_src, &mut self.v[idx])?;
        }
        let ctx = self.cuda.session.ptr();
        let ctrl = self.ctrl.ptr.as_ptr();
        // SAFETY: live session and owned device control block; scalars passed by value.
        crate::cuda::check(unsafe { fa_cuda_adamw_set_step(ctx, ctrl, step) })?;
        // SAFETY: live session and owned device control block; finite positive lr checked above.
        crate::cuda::check(unsafe { fa_cuda_adamw_set_lr(ctx, ctrl, config.lr) })?;
        self.host_step = step;
        self.config = config;
        Ok(())
    }
}

const CKPT_MAGIC: &[u8; 8] = b"FACKPT01";
const CKPT_VERSION: u32 = 1;

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Deterministic, self-checksummed, cross-device (CPU <-> CUDA) training checkpoint.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    step_count: u64,
    config: Option<AdamWConfig>,
    params: BTreeMap<String, Tensor>,
    moments: BTreeMap<String, (Tensor, Tensor)>,
}

impl Checkpoint {
    pub fn new(step_count: u64, config: Option<AdamWConfig>) -> Result<Self> {
        if let Some(cfg) = config {
            cfg.validate()?;
        }
        Ok(Self {
            step_count,
            config,
            params: BTreeMap::new(),
            moments: BTreeMap::new(),
        })
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn config(&self) -> Option<AdamWConfig> {
        self.config
    }

    pub fn param(&self, name: &str) -> Option<&Tensor> {
        self.params.get(name)
    }

    pub fn moments(&self, name: &str) -> Option<(&Tensor, &Tensor)> {
        self.moments.get(name).map(|(m, v)| (m, v))
    }

    pub fn add_param(&mut self, name: impl Into<String>, tensor: &Tensor) -> Result<()> {
        let key = name.into();
        if key.is_empty() || self.params.contains_key(&key) {
            return Err(invalid(
                "checkpoint parameter name must be non-empty and unique",
            ));
        }
        self.params.insert(key, tensor.clone());
        Ok(())
    }

    pub fn add_moments(&mut self, name: impl Into<String>, m: &Tensor, v: &Tensor) -> Result<()> {
        let key = name.into();
        if key.is_empty() || self.moments.contains_key(&key) {
            return Err(invalid(
                "checkpoint moment name must be non-empty and unique",
            ));
        }
        if m.shape() != v.shape() {
            return Err(invalid("checkpoint m and v shapes differ"));
        }
        self.moments.insert(key, (m.clone(), v.clone()));
        Ok(())
    }

    /// Snapshot CPU shared parameter tensors and optional CPU `AdamW` optimizer state.
    pub fn from_cpu_state(
        params: &[(&str, &Shared<Tensor>)],
        optimizer: Option<(&CpuAdamW, &[(&str, Value)])>,
    ) -> Result<Self> {
        let (step, cfg) = match optimizer {
            Some((opt, _)) => (opt.step_count(), Some(opt.config())),
            None => (0, None),
        };
        let mut ckpt = Self::new(step, cfg)?;
        for &(name, shared) in params {
            let guard = shared.read()?;
            ckpt.add_param(name, &guard)?;
        }
        if let Some((opt, mapping)) = optimizer {
            if mapping.len() != opt.leaves().len() {
                return Err(invalid(
                    "checkpoint optimizer mapping count differs from leaves",
                ));
            }
            for &(name, leaf) in mapping {
                let (m, v) = opt.moments(leaf)?;
                ckpt.add_moments(name, m, v)?;
            }
        }
        Ok(ckpt)
    }

    /// Restore CPU shared parameter tensors and optional CPU `AdamW` optimizer state atomically.
    pub fn restore_cpu_state(
        &self,
        params: &[(&str, &Shared<Tensor>)],
        optimizer: Option<(&mut CpuAdamW, &[(&str, Value)])>,
    ) -> Result<()> {
        // Validate every parameter before mutating anything.
        for &(name, shared) in params {
            let src = self
                .params
                .get(name)
                .ok_or_else(|| invalid(format!("checkpoint missing parameter '{name}'")))?;
            let read = shared.read()?;
            if read.shape() != src.shape() {
                return Err(invalid(format!(
                    "checkpoint parameter '{name}' shape differs"
                )));
            }
            drop(read);
            shared.check_writable()?;
        }
        if let Some((opt, mapping)) = optimizer {
            let cfg = self
                .config
                .ok_or_else(|| invalid("checkpoint does not contain AdamW configuration"))?;
            let mut entries = Vec::with_capacity(mapping.len());
            for (name, leaf) in mapping {
                let (m, v) = self
                    .moments
                    .get(*name)
                    .ok_or_else(|| invalid(format!("checkpoint missing moments '{name}'")))?;
                entries.push((leaf, m, v));
            }
            opt.load_state_parts(self.step_count, cfg, &entries)?;
        }
        for &(name, shared) in params {
            let src = &self.params[name];
            let mut dst = shared.write()?;
            dst.data_mut().copy_from_slice(src.data());
        }
        Ok(())
    }

    /// Snapshot CUDA shared parameter tensors and optional CUDA `AdamW` optimizer state to host.
    #[cfg(feature = "cuda")]
    pub fn from_cuda_state(
        cuda: &Cuda,
        params: &[(&str, &Shared<DeviceTensor>)],
        optimizer: Option<(&CudaAdamW, &[(&str, Value)])>,
    ) -> Result<Self> {
        let (step, cfg) = match optimizer {
            Some((opt, _)) => (opt.step_count(), Some(opt.config())),
            None => (0, None),
        };
        let mut ckpt = Self::new(step, cfg)?;
        for &(name, shared) in params {
            let guard = shared.read()?;
            let host = cuda.download(&guard)?;
            ckpt.add_param(name, &host)?;
        }
        if let Some((opt, mapping)) = optimizer {
            if mapping.len() != opt.leaves().len() {
                return Err(invalid(
                    "checkpoint optimizer mapping count differs from leaves",
                ));
            }
            for &(name, leaf) in mapping {
                let (m, v) = opt.download_moments(leaf)?;
                ckpt.add_moments(name, &m, &v)?;
            }
        }
        Ok(ckpt)
    }

    /// Restore CUDA shared parameter tensors and optional CUDA `AdamW` optimizer state from host.
    #[cfg(feature = "cuda")]
    pub fn restore_cuda_state(
        &self,
        cuda: &Cuda,
        params: &[(&str, &Shared<DeviceTensor>)],
        optimizer: Option<(&mut CudaAdamW, &[(&str, Value)])>,
    ) -> Result<()> {
        for &(name, shared) in params {
            let src = self
                .params
                .get(name)
                .ok_or_else(|| invalid(format!("checkpoint missing parameter '{name}'")))?;
            let read = shared.read()?;
            cuda.owns(&read)?;
            if read.shape() != src.shape() {
                return Err(invalid(format!(
                    "checkpoint parameter '{name}' shape differs"
                )));
            }
            drop(read);
            shared.check_writable()?;
        }
        if let Some((opt, mapping)) = optimizer {
            let cfg = self
                .config
                .ok_or_else(|| invalid("checkpoint does not contain AdamW configuration"))?;
            let mut entries = Vec::with_capacity(mapping.len());
            for (name, leaf) in mapping {
                let (m, v) = self
                    .moments
                    .get(*name)
                    .ok_or_else(|| invalid(format!("checkpoint missing moments '{name}'")))?;
                entries.push((leaf, m, v));
            }
            opt.load_state_parts(self.step_count, cfg, &entries)?;
        }
        for &(name, shared) in params {
            let src = &self.params[name];
            let mut dst = shared.write()?;
            cuda.copy_from_host(src, &mut dst)?;
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.extend_from_slice(CKPT_MAGIC);
        buf.extend_from_slice(&CKPT_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags
        buf.extend_from_slice(&self.step_count.to_le_bytes());
        match self.config {
            Some(cfg) => {
                buf.push(1u8);
                buf.extend_from_slice(&cfg.lr.to_le_bytes());
                buf.extend_from_slice(&cfg.beta1.to_le_bytes());
                buf.extend_from_slice(&cfg.beta2.to_le_bytes());
                buf.extend_from_slice(&cfg.eps.to_le_bytes());
                buf.extend_from_slice(&cfg.weight_decay.to_le_bytes());
                match cfg.max_grad_norm {
                    Some(n) => {
                        buf.push(1u8);
                        buf.extend_from_slice(&n.to_le_bytes());
                    }
                    None => {
                        buf.push(0u8);
                        buf.extend_from_slice(&0.0f32.to_le_bytes());
                    }
                }
            }
            None => {
                buf.push(0u8);
                buf.extend_from_slice(&[0u8; 25]);
            }
        }
        let n_params = u32::try_from(self.params.len())
            .map_err(|_| invalid("checkpoint parameter count overflows u32"))?;
        let n_moments = u32::try_from(self.moments.len())
            .map_err(|_| invalid("checkpoint moment count overflows u32"))?;
        buf.extend_from_slice(&n_params.to_le_bytes());
        buf.extend_from_slice(&n_moments.to_le_bytes());

        let write_tensor = |out: &mut Vec<u8>, name: &str, t: &Tensor| -> Result<()> {
            let name_bytes = name.as_bytes();
            let name_len = u32::try_from(name_bytes.len())
                .map_err(|_| invalid("checkpoint tensor name too long"))?;
            let rank = u32::try_from(t.shape().dims().len())
                .map_err(|_| invalid("checkpoint tensor rank overflows u32"))?;
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&rank.to_le_bytes());
            for &d in t.shape().dims() {
                out.extend_from_slice(&(d as u64).to_le_bytes());
            }
            for &v in t.data() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(())
        };

        for (name, t) in &self.params {
            write_tensor(&mut buf, name, t)?;
        }
        for (name, (m, v)) in &self.moments {
            write_tensor(&mut buf, name, m)?;
            write_tensor(&mut buf, name, v)?;
        }
        let checksum = fnv1a_64(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 + 4 + 4 + 8 + 26 + 4 + 4 + 8 {
            return Err(invalid("checkpoint archive is truncated"));
        }
        let payload_len = bytes.len() - 8;
        let payload = &bytes[..payload_len];
        let expected_crc = u64::from_le_bytes(bytes[payload_len..].try_into().unwrap());
        if fnv1a_64(payload) != expected_crc {
            return Err(invalid("checkpoint checksum mismatch (corrupted data)"));
        }
        let mut cursor = 0usize;
        let read_exact = |cur: &mut usize, n: usize| -> Result<&[u8]> {
            if *cur + n > payload.len() {
                return Err(invalid("checkpoint payload truncated"));
            }
            let slice = &payload[*cur..*cur + n];
            *cur += n;
            Ok(slice)
        };
        if read_exact(&mut cursor, 8)? != CKPT_MAGIC {
            return Err(invalid("checkpoint magic mismatch"));
        }
        let version = u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap());
        if version != CKPT_VERSION {
            return Err(invalid(format!("unsupported checkpoint version {version}")));
        }
        let flags = u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap());
        if flags != 0 {
            return Err(invalid("unsupported checkpoint flags"));
        }
        let step_count = u64::from_le_bytes(read_exact(&mut cursor, 8)?.try_into().unwrap());
        let has_cfg = read_exact(&mut cursor, 1)?[0];
        let config = match has_cfg {
            0 => {
                let _ = read_exact(&mut cursor, 25)?;
                None
            }
            1 => {
                let lr = f32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap());
                let beta1 = f32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap());
                let beta2 = f32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap());
                let eps = f32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap());
                let weight_decay =
                    f32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap());
                let has_norm = read_exact(&mut cursor, 1)?[0];
                let norm_val = f32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap());
                let max_grad_norm = match has_norm {
                    0 => None,
                    1 => Some(norm_val),
                    _ => return Err(invalid("invalid max_grad_norm tag in checkpoint")),
                };
                let cfg = AdamWConfig {
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    max_grad_norm,
                };
                cfg.validate()?;
                Some(cfg)
            }
            _ => return Err(invalid("invalid optimizer config tag in checkpoint")),
        };
        let n_params = u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap()) as usize;
        let n_moments =
            u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().unwrap()) as usize;

        let read_tensor = |cur: &mut usize| -> Result<(String, Tensor)> {
            let name_len = u32::from_le_bytes(read_exact(cur, 4)?.try_into().unwrap()) as usize;
            let name = std::str::from_utf8(read_exact(cur, name_len)?)
                .map_err(|_| invalid("checkpoint tensor name is not valid UTF-8"))?
                .to_string();
            let rank = u32::from_le_bytes(read_exact(cur, 4)?.try_into().unwrap()) as usize;
            if rank > 64 {
                return Err(invalid("checkpoint tensor rank exceeds limit"));
            }
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                let d = u64::from_le_bytes(read_exact(cur, 8)?.try_into().unwrap());
                let d_usize =
                    usize::try_from(d).map_err(|_| invalid("checkpoint dim overflows usize"))?;
                dims.push(d_usize);
            }
            let shape = Shape::new(dims)?;
            let byte_len = shape
                .len()
                .checked_mul(4)
                .ok_or_else(|| invalid("checkpoint tensor byte length overflows"))?;
            let raw = read_exact(cur, byte_len)?;
            let mut data = Vec::with_capacity(shape.len());
            for chunk in raw.chunks_exact(4) {
                data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok((name, Tensor::from_vec(shape.dims(), data)?))
        };

        let mut ckpt = Self::new(step_count, config)?;
        for _ in 0..n_params {
            let (name, t) = read_tensor(&mut cursor)?;
            ckpt.add_param(name, &t)?;
        }
        for _ in 0..n_moments {
            let (name_m, m) = read_tensor(&mut cursor)?;
            let (name_v, v) = read_tensor(&mut cursor)?;
            if name_m != name_v {
                return Err(invalid("checkpoint m/v moment names do not match"));
            }
            ckpt.add_moments(name_m, &m, &v)?;
        }
        if cursor != payload.len() {
            return Err(invalid("checkpoint contains unexpected trailing bytes"));
        }
        Ok(ckpt)
    }

    pub fn save_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.to_bytes()?;
        fs::write(path, bytes).map_err(|e| invalid(format!("failed to write checkpoint: {e}")))
    }

    pub fn load_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes =
            fs::read(path).map_err(|e| invalid(format!("failed to read checkpoint: {e}")))?;
        Self::from_bytes(&bytes)
    }
}
