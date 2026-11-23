use crate::{
    Error, GradientMode, Layout, Result, Shape, TensorView, TensorViewMut, error::invalid,
};

/// Normalization family over trailing axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum NormKind {
    LayerNorm = 0,
    RmsNorm = 1,
}

/// Validated shape, trailing axis count, and numerical stability epsilon for LayerNorm/RmsNorm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormSpec {
    kind: NormKind,
    shape: Shape,
    normalized_dims: usize,
    eps_bits: u32,
    groups: usize,
    inner_len: usize,
}

impl NormSpec {
    pub fn new(kind: NormKind, shape: &Shape, normalized_dims: usize, eps: f32) -> Result<Self> {
        let rank = shape.dims().len();
        if rank == 0 || normalized_dims == 0 || normalized_dims > rank {
            return Err(invalid(
                "normalization requires rank >= 1 and 1 <= normalized_dims <= rank",
            ));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(invalid("normalization eps must be finite and positive"));
        }
        let split = rank - normalized_dims;
        let groups: usize = shape.dims()[..split].iter().product();
        let inner_len: usize = shape.dims()[split..].iter().product();
        if inner_len == 0 && groups > 0 {
            return Err(invalid("normalization inner extent must be positive"));
        }
        Ok(Self {
            kind,
            shape: shape.clone(),
            normalized_dims,
            eps_bits: eps.to_bits(),
            groups,
            inner_len,
        })
    }

    pub fn kind(&self) -> NormKind {
        self.kind
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn normalized_dims(&self) -> usize {
        self.normalized_dims
    }
    pub fn eps(&self) -> f32 {
        f32::from_bits(self.eps_bits)
    }
    pub fn groups(&self) -> usize {
        self.groups
    }
    pub fn inner_len(&self) -> usize {
        self.inner_len
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        std::mem::size_of_val(self.shape.dims()) + size_of::<Self>()
    }
}

#[derive(Clone)]
pub(crate) struct NormStridedView {
    pub offset: isize,
    pub inner_len: usize,
    pub outer_dims: Vec<usize>,
    pub outer_strides: Vec<isize>,
    pub inner_dims: Vec<usize>,
    pub inner_strides: Vec<isize>,
    pub inner_contiguous: bool,
    pub fully_contiguous: bool,
}

impl NormStridedView {
    pub(crate) fn new(spec: &NormSpec, layout: &Layout, writable: bool) -> Result<Self> {
        if layout.shape() != &spec.shape {
            return Err(invalid("normalization layout shape differs from spec"));
        }
        if writable && !layout.is_non_overlapping() {
            return Err(Error::Unsupported(
                "normalization output layout must be proved non-overlapping".into(),
            ));
        }
        let dims = layout.shape().dims();
        let strides = layout.strides();
        let split = dims.len() - spec.normalized_dims;

        let mut inner_contiguous = true;
        let mut expected = 1isize;
        for a in (split..dims.len()).rev() {
            if dims[a] > 1 && strides[a] != expected {
                inner_contiguous = false;
            }
            expected *= dims[a] as isize;
        }
        let fully_contiguous = layout.is_contiguous();

        let outer_dims = dims[..split].to_vec();
        let outer_strides = strides[..split].to_vec();
        let inner_dims = dims[split..].to_vec();
        let inner_strides = strides[split..].to_vec();

        Ok(Self {
            offset: layout.offset() as isize,
            inner_len: spec.inner_len,
            outer_dims,
            outer_strides,
            inner_dims,
            inner_strides,
            inner_contiguous,
            fully_contiguous,
        })
    }

    #[inline]
    fn group_base(&self, mut g: usize) -> isize {
        if self.fully_contiguous {
            return self.offset + (g * self.inner_len) as isize;
        }
        let mut off = self.offset;
        for a in (0..self.outer_dims.len()).rev() {
            let d = self.outer_dims[a];
            let c = g % d;
            g /= d;
            off += c as isize * self.outer_strides[a];
        }
        off
    }

    #[inline]
    fn inner_offset(&self, mut idx: usize) -> isize {
        if self.inner_contiguous {
            return idx as isize;
        }
        let mut off = 0isize;
        for a in (0..self.inner_dims.len()).rev() {
            let d = self.inner_dims[a];
            let c = idx % d;
            idx /= d;
            off += c as isize * self.inner_strides[a];
        }
        off
    }
}

/// Prepared strided LayerNorm / RmsNorm forward plan on CPU (100% safe Rust, zero allocations).
pub struct NormPlan {
    pub(crate) spec: NormSpec,
    pub(crate) input_layout: Layout,
    pub(crate) output_layout: Layout,
    pub(crate) x_view: NormStridedView,
    pub(crate) y_view: NormStridedView,
}

impl NormPlan {
    pub fn new(spec: &NormSpec, input: &Layout, output: &Layout) -> Result<Self> {
        let x_view = NormStridedView::new(spec, input, false)?;
        let y_view = NormStridedView::new(spec, output, true)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
            x_view,
            y_view,
        })
    }

    pub fn spec(&self) -> &NormSpec {
        &self.spec
    }
    pub fn input_layout(&self) -> &Layout {
        &self.input_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
            + (self.x_view.outer_dims.len() + self.x_view.inner_dims.len())
                * 2
                * (size_of::<usize>() + size_of::<isize>())
    }

    pub fn run(&self, input: &TensorView<'_>, output: &mut TensorViewMut<'_>) -> Result<()> {
        if input.layout() != &self.input_layout {
            return Err(invalid(
                "normalization input layout differs from prepared plan",
            ));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "normalization output layout differs from prepared plan",
            ));
        }
        let groups = self.spec.groups;
        let d = self.spec.inner_len;
        if groups == 0 || d == 0 {
            return Ok(());
        }
        let inv_d = 1.0f32 / d as f32;
        let eps = self.spec.eps();
        let x_data = input.data;
        let y_data = &mut *output.data;

        if self.x_view.inner_contiguous && self.y_view.inner_contiguous {
            for g in 0..groups {
                let xb = self.x_view.group_base(g) as usize;
                let yb = self.y_view.group_base(g) as usize;
                let xs = &x_data[xb..xb + d];
                let ys = &mut y_data[yb..yb + d];
                norm_fwd_row(self.spec.kind, xs, ys, inv_d, eps);
            }
            return Ok(());
        }

        // General signed-stride path.
        for g in 0..groups {
            let xb = self.x_view.group_base(g);
            let yb = self.y_view.group_base(g);
            match self.spec.kind {
                NormKind::LayerNorm => {
                    let mut sum = 0.0f32;
                    for i in 0..d {
                        sum += x_data[(xb + self.x_view.inner_offset(i)) as usize];
                    }
                    let mean = sum * inv_d;
                    let mut var_sum = 0.0f32;
                    for i in 0..d {
                        let diff = x_data[(xb + self.x_view.inner_offset(i)) as usize] - mean;
                        var_sum += diff * diff;
                    }
                    let rstd = 1.0f32 / (var_sum * inv_d + eps).sqrt();
                    for i in 0..d {
                        let v = x_data[(xb + self.x_view.inner_offset(i)) as usize];
                        y_data[(yb + self.y_view.inner_offset(i)) as usize] = (v - mean) * rstd;
                    }
                }
                NormKind::RmsNorm => {
                    let mut sq_sum = 0.0f32;
                    for i in 0..d {
                        let v = x_data[(xb + self.x_view.inner_offset(i)) as usize];
                        sq_sum += v * v;
                    }
                    let rstd = 1.0f32 / (sq_sum * inv_d + eps).sqrt();
                    for i in 0..d {
                        let v = x_data[(xb + self.x_view.inner_offset(i)) as usize];
                        y_data[(yb + self.y_view.inner_offset(i)) as usize] = v * rstd;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Prepared strided LayerNorm / RmsNorm local VJP on CPU (`input_required() == true`).
pub struct NormBackward {
    pub(crate) spec: NormSpec,
    pub(crate) input_layout: Layout,
    pub(crate) upstream_layout: Layout,
    pub(crate) output_layout: Layout,
    pub(crate) x_view: NormStridedView,
    pub(crate) dy_view: NormStridedView,
    pub(crate) dx_view: NormStridedView,
}

impl NormBackward {
    pub fn new(
        spec: &NormSpec,
        input: &Layout,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let x_view = NormStridedView::new(spec, input, false)?;
        let dy_view = NormStridedView::new(spec, upstream, false)?;
        let dx_view = NormStridedView::new(spec, output, true)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            x_view,
            dy_view,
            dx_view,
        })
    }

    pub fn spec(&self) -> &NormSpec {
        &self.spec
    }
    pub fn input_required(&self) -> bool {
        true
    }
    pub fn input_layout(&self) -> &Layout {
        &self.input_layout
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.upstream_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
            + (self.x_view.outer_dims.len() + self.x_view.inner_dims.len())
                * 3
                * (size_of::<usize>() + size_of::<isize>())
    }

    pub fn run(
        &self,
        input: &TensorView<'_>,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        if input.layout() != &self.input_layout {
            return Err(invalid(
                "normalization backward input layout differs from prepared plan",
            ));
        }
        if upstream.layout() != &self.upstream_layout {
            return Err(invalid(
                "normalization backward upstream layout differs from prepared plan",
            ));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "normalization backward output layout differs from prepared plan",
            ));
        }
        let groups = self.spec.groups;
        let d = self.spec.inner_len;
        if groups == 0 || d == 0 {
            return Ok(());
        }
        let inv_d = 1.0f32 / d as f32;
        let eps = self.spec.eps();
        let accumulate = mode == GradientMode::Accumulate;
        let x_data = input.data;
        let dy_data = upstream.data;
        let dx_data = &mut *output.data;

        if self.x_view.inner_contiguous
            && self.dy_view.inner_contiguous
            && self.dx_view.inner_contiguous
        {
            for g in 0..groups {
                let xb = self.x_view.group_base(g) as usize;
                let dyb = self.dy_view.group_base(g) as usize;
                let dxb = self.dx_view.group_base(g) as usize;
                let xs = &x_data[xb..xb + d];
                let dys = &dy_data[dyb..dyb + d];
                let dxs = &mut dx_data[dxb..dxb + d];
                norm_bwd_row(self.spec.kind, xs, dys, dxs, inv_d, eps, accumulate);
            }
            return Ok(());
        }

        // General signed-stride path.
        for g in 0..groups {
            let xb = self.x_view.group_base(g);
            let dyb = self.dy_view.group_base(g);
            let dxb = self.dx_view.group_base(g);
            match self.spec.kind {
                NormKind::LayerNorm => {
                    let mut sum = 0.0f32;
                    for i in 0..d {
                        sum += x_data[(xb + self.x_view.inner_offset(i)) as usize];
                    }
                    let mean = sum * inv_d;
                    let mut var_sum = 0.0f32;
                    for i in 0..d {
                        let diff = x_data[(xb + self.x_view.inner_offset(i)) as usize] - mean;
                        var_sum += diff * diff;
                    }
                    let rstd = 1.0f32 / (var_sum * inv_d + eps).sqrt();
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    for i in 0..d {
                        let x = x_data[(xb + self.x_view.inner_offset(i)) as usize];
                        let dy = dy_data[(dyb + self.dy_view.inner_offset(i)) as usize];
                        s1 += dy;
                        s2 += dy * ((x - mean) * rstd);
                    }
                    s1 *= inv_d;
                    s2 *= inv_d;
                    for i in 0..d {
                        let x = x_data[(xb + self.x_view.inner_offset(i)) as usize];
                        let dy = dy_data[(dyb + self.dy_view.inner_offset(i)) as usize];
                        let grad = rstd * (dy - s1 - ((x - mean) * rstd) * s2);
                        let dst = (dxb + self.dx_view.inner_offset(i)) as usize;
                        if accumulate {
                            dx_data[dst] += grad;
                        } else {
                            dx_data[dst] = grad;
                        }
                    }
                }
                NormKind::RmsNorm => {
                    let mut sq_sum = 0.0f32;
                    for i in 0..d {
                        let v = x_data[(xb + self.x_view.inner_offset(i)) as usize];
                        sq_sum += v * v;
                    }
                    let rstd = 1.0f32 / (sq_sum * inv_d + eps).sqrt();
                    let mut s2 = 0.0f32;
                    for i in 0..d {
                        let x = x_data[(xb + self.x_view.inner_offset(i)) as usize];
                        let dy = dy_data[(dyb + self.dy_view.inner_offset(i)) as usize];
                        s2 += dy * (x * rstd);
                    }
                    s2 *= inv_d;
                    for i in 0..d {
                        let x = x_data[(xb + self.x_view.inner_offset(i)) as usize];
                        let dy = dy_data[(dyb + self.dy_view.inner_offset(i)) as usize];
                        let grad = rstd * (dy - (x * rstd) * s2);
                        let dst = (dxb + self.dx_view.inner_offset(i)) as usize;
                        if accumulate {
                            dx_data[dst] += grad;
                        } else {
                            dx_data[dst] = grad;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[inline(always)]
fn norm_fwd_row_impl(kind: NormKind, xs: &[f32], ys: &mut [f32], inv_d: f32, eps: f32) {
    match kind {
        NormKind::LayerNorm => {
            let mut sum8 = [0.0f32; 8];
            let mut ch = xs.chunks_exact(8);
            for c in &mut ch {
                for j in 0..8 {
                    sum8[j] += c[j];
                }
            }
            for &v in ch.remainder() {
                sum8[0] += v;
            }
            let mean = (((sum8[0] + sum8[1]) + (sum8[2] + sum8[3]))
                + ((sum8[4] + sum8[5]) + (sum8[6] + sum8[7])))
                * inv_d;
            let mut var8 = [0.0f32; 8];
            let mut ch = xs.chunks_exact(8);
            for c in &mut ch {
                for j in 0..8 {
                    let d = c[j] - mean;
                    var8[j] += d * d;
                }
            }
            for &v in ch.remainder() {
                let d = v - mean;
                var8[0] += d * d;
            }
            let var = (((var8[0] + var8[1]) + (var8[2] + var8[3]))
                + ((var8[4] + var8[5]) + (var8[6] + var8[7])))
                * inv_d;
            let rstd = 1.0f32 / (var + eps).sqrt();
            for (dst, &v) in ys.iter_mut().zip(xs) {
                *dst = (v - mean) * rstd;
            }
        }
        NormKind::RmsNorm => {
            let mut sq8 = [0.0f32; 8];
            let mut ch = xs.chunks_exact(8);
            for c in &mut ch {
                for j in 0..8 {
                    sq8[j] += c[j] * c[j];
                }
            }
            for &v in ch.remainder() {
                sq8[0] += v * v;
            }
            let ms = (((sq8[0] + sq8[1]) + (sq8[2] + sq8[3]))
                + ((sq8[4] + sq8[5]) + (sq8[6] + sq8[7])))
                * inv_d;
            let rstd = 1.0f32 / (ms + eps).sqrt();
            for (dst, &v) in ys.iter_mut().zip(xs) {
                *dst = v * rstd;
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn norm_fwd_row_avx2(kind: NormKind, xs: &[f32], ys: &mut [f32], inv_d: f32, eps: f32) {
    norm_fwd_row_impl(kind, xs, ys, inv_d, eps);
}

#[inline]
fn norm_fwd_row(kind: NormKind, xs: &[f32], ys: &mut [f32], inv_d: f32, eps: f32) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        unsafe { norm_fwd_row_avx2(kind, xs, ys, inv_d, eps) };
        return;
    }
    norm_fwd_row_impl(kind, xs, ys, inv_d, eps);
}

#[inline(always)]
fn norm_bwd_row_impl(
    kind: NormKind,
    xs: &[f32],
    dys: &[f32],
    dxs: &mut [f32],
    inv_d: f32,
    eps: f32,
    accumulate: bool,
) {
    match kind {
        NormKind::LayerNorm => {
            let mut sum8 = [0.0f32; 8];
            let mut ch = xs.chunks_exact(8);
            for c in &mut ch {
                for j in 0..8 {
                    sum8[j] += c[j];
                }
            }
            for &v in ch.remainder() {
                sum8[0] += v;
            }
            let mean = (((sum8[0] + sum8[1]) + (sum8[2] + sum8[3]))
                + ((sum8[4] + sum8[5]) + (sum8[6] + sum8[7])))
                * inv_d;
            let mut var8 = [0.0f32; 8];
            let mut s1_8 = [0.0f32; 8];
            let mut s2_8 = [0.0f32; 8];
            let mut cx = xs.chunks_exact(8);
            let mut cd = dys.chunks_exact(8);
            for (xc, dc) in (&mut cx).zip(&mut cd) {
                for j in 0..8 {
                    let diff = xc[j] - mean;
                    var8[j] += diff * diff;
                    s1_8[j] += dc[j];
                    s2_8[j] += dc[j] * diff;
                }
            }
            for (&xv, &dv) in cx.remainder().iter().zip(cd.remainder()) {
                let diff = xv - mean;
                var8[0] += diff * diff;
                s1_8[0] += dv;
                s2_8[0] += dv * diff;
            }
            let var = (((var8[0] + var8[1]) + (var8[2] + var8[3]))
                + ((var8[4] + var8[5]) + (var8[6] + var8[7])))
                * inv_d;
            let rstd = 1.0f32 / (var + eps).sqrt();
            let s1 = (((s1_8[0] + s1_8[1]) + (s1_8[2] + s1_8[3]))
                + ((s1_8[4] + s1_8[5]) + (s1_8[6] + s1_8[7])))
                * inv_d;
            let s2_rstd = (((s2_8[0] + s2_8[1]) + (s2_8[2] + s2_8[3]))
                + ((s2_8[4] + s2_8[5]) + (s2_8[6] + s2_8[7])))
                * (inv_d * rstd * rstd);
            if accumulate {
                for ((dst, &x), &dy) in dxs.iter_mut().zip(xs).zip(dys) {
                    *dst += rstd * (dy - s1 - (x - mean) * s2_rstd);
                }
            } else {
                for ((dst, &x), &dy) in dxs.iter_mut().zip(xs).zip(dys) {
                    *dst = rstd * (dy - s1 - (x - mean) * s2_rstd);
                }
            }
        }
        NormKind::RmsNorm => {
            let mut sq8 = [0.0f32; 8];
            let mut dot8 = [0.0f32; 8];
            let mut cx = xs.chunks_exact(8);
            let mut cd = dys.chunks_exact(8);
            for (xc, dc) in (&mut cx).zip(&mut cd) {
                for j in 0..8 {
                    sq8[j] += xc[j] * xc[j];
                    dot8[j] += dc[j] * xc[j];
                }
            }
            for (&xv, &dv) in cx.remainder().iter().zip(cd.remainder()) {
                sq8[0] += xv * xv;
                dot8[0] += dv * xv;
            }
            let ms = (((sq8[0] + sq8[1]) + (sq8[2] + sq8[3]))
                + ((sq8[4] + sq8[5]) + (sq8[6] + sq8[7])))
                * inv_d;
            let rstd = 1.0f32 / (ms + eps).sqrt();
            let coeff = (((dot8[0] + dot8[1]) + (dot8[2] + dot8[3]))
                + ((dot8[4] + dot8[5]) + (dot8[6] + dot8[7])))
                * (inv_d * rstd * rstd);
            if accumulate {
                for ((dst, &x), &dy) in dxs.iter_mut().zip(xs).zip(dys) {
                    *dst += rstd * (dy - x * coeff);
                }
            } else {
                for ((dst, &x), &dy) in dxs.iter_mut().zip(xs).zip(dys) {
                    *dst = rstd * (dy - x * coeff);
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn norm_bwd_row_avx2(
    kind: NormKind,
    xs: &[f32],
    dys: &[f32],
    dxs: &mut [f32],
    inv_d: f32,
    eps: f32,
    accumulate: bool,
) {
    norm_bwd_row_impl(kind, xs, dys, dxs, inv_d, eps, accumulate);
}

#[inline]
fn norm_bwd_row(
    kind: NormKind,
    xs: &[f32],
    dys: &[f32],
    dxs: &mut [f32],
    inv_d: f32,
    eps: f32,
    accumulate: bool,
) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        unsafe { norm_bwd_row_avx2(kind, xs, dys, dxs, inv_d, eps, accumulate) };
        return;
    }
    norm_bwd_row_impl(kind, xs, dys, dxs, inv_d, eps, accumulate);
}
