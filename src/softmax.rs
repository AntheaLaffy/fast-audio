use crate::{
    Error, GradientMode, Layout, Result, Shape, TensorView, TensorViewMut, error::invalid,
};

/// Softmax family along a single explicit axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum SoftmaxKind {
    Softmax = 0,
    LogSoftmax = 1,
}

/// Validated shape and axis for numerically stable `Softmax` / `LogSoftmax`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoftmaxSpec {
    kind: SoftmaxKind,
    shape: Shape,
    axis: usize,
    groups: usize,
    axis_len: usize,
}

impl SoftmaxSpec {
    pub fn new(kind: SoftmaxKind, shape: &Shape, axis: usize) -> Result<Self> {
        let dims = shape.dims();
        let rank = dims.len();
        if rank == 0 || axis >= rank {
            return Err(invalid("softmax requires rank >= 1 and axis < rank"));
        }
        let axis_len = dims[axis];
        let groups: usize = dims
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .product();
        if axis_len == 0 && groups > 0 {
            return Err(invalid("softmax axis extent must be positive"));
        }
        Ok(Self {
            kind,
            shape: shape.clone(),
            axis,
            groups,
            axis_len,
        })
    }

    pub fn kind(&self) -> SoftmaxKind {
        self.kind
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn axis(&self) -> usize {
        self.axis
    }
    pub fn groups(&self) -> usize {
        self.groups
    }
    pub fn axis_len(&self) -> usize {
        self.axis_len
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        std::mem::size_of_val(self.shape.dims()) + size_of::<Self>()
    }
}

#[derive(Clone)]
pub(crate) struct SoftmaxStridedView {
    pub offset: isize,
    pub group_dims: Vec<usize>,
    pub group_strides: Vec<isize>,
    pub axis_len: usize,
    pub axis_stride: isize,
    pub inner_contiguous: bool,
    pub fully_contiguous_last: bool,
}

impl SoftmaxStridedView {
    pub(crate) fn new(spec: &SoftmaxSpec, layout: &Layout, writable: bool) -> Result<Self> {
        if layout.shape() != &spec.shape {
            return Err(invalid("softmax layout shape differs from spec"));
        }
        if writable && !layout.is_non_overlapping() {
            return Err(Error::Unsupported(
                "softmax output layout must be proved non-overlapping".into(),
            ));
        }
        let dims = layout.shape().dims();
        let strides = layout.strides();
        let axis = spec.axis;
        let axis_len = dims[axis];
        let axis_stride = strides[axis];

        let mut group_dims = Vec::with_capacity(dims.len().saturating_sub(1));
        let mut group_strides = Vec::with_capacity(dims.len().saturating_sub(1));
        for a in 0..dims.len() {
            if a != axis {
                group_dims.push(dims[a]);
                group_strides.push(strides[a]);
            }
        }
        let inner_contiguous = axis_len <= 1 || axis_stride == 1;
        let fully_contiguous_last = layout.is_contiguous() && axis + 1 == dims.len();

        Ok(Self {
            offset: layout.offset() as isize,
            group_dims,
            group_strides,
            axis_len,
            axis_stride,
            inner_contiguous,
            fully_contiguous_last,
        })
    }

    #[inline]
    fn group_base(&self, mut g: usize) -> isize {
        if self.fully_contiguous_last {
            return self.offset + (g * self.axis_len) as isize;
        }
        let mut off = self.offset;
        for a in (0..self.group_dims.len()).rev() {
            let d = self.group_dims[a];
            let c = g % d;
            g /= d;
            off += c as isize * self.group_strides[a];
        }
        off
    }
}

/// Prepared strided Softmax / LogSoftmax forward plan on CPU (zero hot allocations).
pub struct SoftmaxPlan {
    pub(crate) spec: SoftmaxSpec,
    pub(crate) input_layout: Layout,
    pub(crate) output_layout: Layout,
    pub(crate) x_view: SoftmaxStridedView,
    pub(crate) y_view: SoftmaxStridedView,
}

impl SoftmaxPlan {
    pub fn new(spec: &SoftmaxSpec, input: &Layout, output: &Layout) -> Result<Self> {
        let x_view = SoftmaxStridedView::new(spec, input, false)?;
        let y_view = SoftmaxStridedView::new(spec, output, true)?;
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
            x_view,
            y_view,
        })
    }

    pub fn spec(&self) -> &SoftmaxSpec {
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
            + self.x_view.group_dims.len() * 2 * (size_of::<usize>() + size_of::<isize>())
    }

    pub fn run(&self, input: &TensorView<'_>, output: &mut TensorViewMut<'_>) -> Result<()> {
        if input.layout() != &self.input_layout {
            return Err(invalid("softmax input layout differs from prepared plan"));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid("softmax output layout differs from prepared plan"));
        }
        let groups = self.spec.groups;
        let d = self.spec.axis_len;
        if groups == 0 || d == 0 {
            return Ok(());
        }
        let x_data = input.data;
        let y_data = &mut *output.data;

        if self.x_view.inner_contiguous && self.y_view.inner_contiguous {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            if d >= 8
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
                unsafe {
                    run_contiguous_fwd_avx2(
                        self.spec.kind,
                        groups,
                        d,
                        &self.x_view,
                        &self.y_view,
                        x_data,
                        y_data,
                    );
                }
                return Ok(());
            }
            if self.spec.kind == SoftmaxKind::Softmax && d <= 2048 {
                let batch_cap = (2048 / d).clamp(1, 32);
                let mut g0 = 0;
                while g0 < groups {
                    let bg = (groups - g0).min(batch_cap);
                    let yb0 = self.y_view.group_base(g0) as usize;
                    let mut contiguous_batch = true;
                    for b in 0..bg {
                        let xb = self.x_view.group_base(g0 + b) as usize;
                        let yb = self.y_view.group_base(g0 + b) as usize;
                        if yb != yb0 + b * d {
                            contiguous_batch = false;
                        }
                        let m = shift_max_row(&x_data[xb..xb + d], &mut y_data[yb..yb + d]);
                        if m.is_nan() {
                            contiguous_batch = false;
                        }
                    }
                    if contiguous_batch {
                        exp_in_place(&mut y_data[yb0..yb0 + bg * d]);
                        for b in 0..bg {
                            let yb = yb0 + b * d;
                            normalize_exp_row(&mut y_data[yb..yb + d]);
                        }
                    } else {
                        for b in 0..bg {
                            let xb = self.x_view.group_base(g0 + b) as usize;
                            let yb = self.y_view.group_base(g0 + b) as usize;
                            let xs = &x_data[xb..xb + d];
                            let ys = &mut y_data[yb..yb + d];
                            let m = max_slice_nan(xs);
                            if m.is_nan() {
                                ys.fill(f32::NAN);
                            } else {
                                shift_row(xs, m, ys);
                                exp_in_place(ys);
                                normalize_exp_row(ys);
                            }
                        }
                    }
                    g0 += bg;
                }
                return Ok(());
            }
            for g in 0..groups {
                let xb = self.x_view.group_base(g) as usize;
                let yb = self.y_view.group_base(g) as usize;
                let xs = &x_data[xb..xb + d];
                let ys = &mut y_data[yb..yb + d];
                let m = max_slice_nan(xs);
                if m.is_nan() {
                    ys.fill(f32::NAN);
                    continue;
                }
                match self.spec.kind {
                    SoftmaxKind::Softmax => {
                        shift_row(xs, m, ys);
                        exp_in_place(ys);
                        normalize_exp_row(ys);
                    }
                    SoftmaxKind::LogSoftmax => {
                        let sum = sum_exp_shifted(xs, m);
                        let log_z = m + sum.ln();
                        shift_row(xs, log_z, ys);
                    }
                }
            }
            return Ok(());
        }

        let xs_step = self.x_view.axis_stride;
        let ys_step = self.y_view.axis_stride;
        for g in 0..groups {
            let xb = self.x_view.group_base(g);
            let yb = self.y_view.group_base(g);
            let mut m = x_data[xb as usize];
            for i in 1..d {
                m = max_nan(m, x_data[(xb + i as isize * xs_step) as usize]);
            }
            if m.is_nan() {
                for i in 0..d {
                    y_data[(yb + i as isize * ys_step) as usize] = f32::NAN;
                }
                continue;
            }
            let mut sum = 0.0f32;
            for i in 0..d {
                let e = (x_data[(xb + i as isize * xs_step) as usize] - m).exp();
                sum += e;
                if self.spec.kind == SoftmaxKind::Softmax {
                    y_data[(yb + i as isize * ys_step) as usize] = e;
                }
            }
            match self.spec.kind {
                SoftmaxKind::Softmax => {
                    let inv_sum = 1.0f32 / sum;
                    for i in 0..d {
                        y_data[(yb + i as isize * ys_step) as usize] *= inv_sum;
                    }
                }
                SoftmaxKind::LogSoftmax => {
                    let log_z = m + sum.ln();
                    for i in 0..d {
                        let v = x_data[(xb + i as isize * xs_step) as usize];
                        y_data[(yb + i as isize * ys_step) as usize] = v - log_z;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Prepared strided Softmax / LogSoftmax local VJP on CPU (`input_required() == true`).
pub struct SoftmaxBackward {
    pub(crate) spec: SoftmaxSpec,
    pub(crate) input_layout: Layout,
    pub(crate) upstream_layout: Layout,
    pub(crate) output_layout: Layout,
    pub(crate) x_view: SoftmaxStridedView,
    pub(crate) dy_view: SoftmaxStridedView,
    pub(crate) dx_view: SoftmaxStridedView,
}

impl SoftmaxBackward {
    pub fn new(
        spec: &SoftmaxSpec,
        input: &Layout,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let x_view = SoftmaxStridedView::new(spec, input, false)?;
        let dy_view = SoftmaxStridedView::new(spec, upstream, false)?;
        let dx_view = SoftmaxStridedView::new(spec, output, true)?;
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

    pub fn spec(&self) -> &SoftmaxSpec {
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
            + self.x_view.group_dims.len() * 3 * (size_of::<usize>() + size_of::<isize>())
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
                "softmax backward input layout differs from prepared plan",
            ));
        }
        if upstream.layout() != &self.upstream_layout {
            return Err(invalid(
                "softmax backward upstream layout differs from prepared plan",
            ));
        }
        if output.layout() != &self.output_layout {
            return Err(invalid(
                "softmax backward output layout differs from prepared plan",
            ));
        }
        let groups = self.spec.groups;
        let d = self.spec.axis_len;
        if groups == 0 || d == 0 {
            return Ok(());
        }
        let accumulate = mode == GradientMode::Accumulate;
        let x_data = input.data;
        let dy_data = upstream.data;
        let dx_data = &mut *output.data;

        if self.x_view.inner_contiguous
            && self.dy_view.inner_contiguous
            && self.dx_view.inner_contiguous
        {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            if (8..=1024).contains(&d)
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
                unsafe {
                    run_contiguous_bwd_avx2(
                        self.spec.kind,
                        groups,
                        d,
                        &self.x_view,
                        &self.dy_view,
                        &self.dx_view,
                        x_data,
                        dy_data,
                        dx_data,
                        accumulate,
                    );
                }
                return Ok(());
            }
            const TILE: usize = 256;
            if self.spec.kind == SoftmaxKind::Softmax && d <= TILE {
                const BATCH_BUF: usize = 2048;
                let mut batch_exp = [0.0f32; BATCH_BUF];
                let mut nans = [false; 32];
                let batch_cap = (BATCH_BUF / d).clamp(1, 32);
                let mut g0 = 0;
                while g0 < groups {
                    let bg = (groups - g0).min(batch_cap);
                    for b in 0..bg {
                        let xb = self.x_view.group_base(g0 + b) as usize;
                        let xs = &x_data[xb..xb + d];
                        let eb = &mut batch_exp[b * d..(b + 1) * d];
                        let m = shift_max_row(xs, eb);
                        if m.is_nan() {
                            nans[b] = true;
                            eb.fill(0.0);
                        } else {
                            nans[b] = false;
                        }
                    }
                    exp_in_place(&mut batch_exp[..bg * d]);
                    for b in 0..bg {
                        let dyb = self.dy_view.group_base(g0 + b) as usize;
                        let dxb = self.dx_view.group_base(g0 + b) as usize;
                        let dys = &dy_data[dyb..dyb + d];
                        let dxs = &mut dx_data[dxb..dxb + d];
                        if nans[b] {
                            if accumulate {
                                for dst in dxs.iter_mut() {
                                    *dst += f32::NAN;
                                }
                            } else {
                                dxs.fill(f32::NAN);
                            }
                        } else {
                            let eb = &batch_exp[b * d..(b + 1) * d];
                            softmax_bwd_finish_row(eb, dys, dxs, accumulate);
                        }
                    }
                    g0 += bg;
                }
                return Ok(());
            }
            let mut exp_buf = [0.0f32; TILE];
            for g in 0..groups {
                let xb = self.x_view.group_base(g) as usize;
                let dyb = self.dy_view.group_base(g) as usize;
                let dxb = self.dx_view.group_base(g) as usize;
                let xs = &x_data[xb..xb + d];
                let dys = &dy_data[dyb..dyb + d];
                let dxs = &mut dx_data[dxb..dxb + d];
                let m = max_slice_nan(xs);
                if m.is_nan() {
                    if accumulate {
                        for dst in dxs.iter_mut() {
                            *dst += f32::NAN;
                        }
                    } else {
                        dxs.fill(f32::NAN);
                    }
                    continue;
                }
                match self.spec.kind {
                    SoftmaxKind::Softmax => {
                        let mut sum_e = 0.0f32;
                        let mut dot_e = 0.0f32;
                        let mut off = 0;
                        while off < d {
                            let n = (d - off).min(TILE);
                            shift_row(&xs[off..off + n], m, &mut exp_buf[..n]);
                            exp_in_place(&mut exp_buf[..n]);
                            let (s_part, d_part) =
                                sum_and_dot_8way(&exp_buf[..n], &dys[off..off + n]);
                            sum_e += s_part;
                            dot_e += d_part;
                            off += n;
                        }
                        let inv_sum = 1.0f32 / sum_e;
                        let dot = dot_e * inv_sum;
                        let mut off = 0;
                        while off < d {
                            let n = (d - off).min(TILE);
                            shift_row(&xs[off..off + n], m, &mut exp_buf[..n]);
                            exp_in_place(&mut exp_buf[..n]);
                            if accumulate {
                                for j in 0..n {
                                    let p = exp_buf[j] * inv_sum;
                                    dxs[off + j] += p * (dys[off + j] - dot);
                                }
                            } else {
                                for j in 0..n {
                                    let p = exp_buf[j] * inv_sum;
                                    dxs[off + j] = p * (dys[off + j] - dot);
                                }
                            }
                            off += n;
                        }
                    }
                    SoftmaxKind::LogSoftmax => {
                        let sum_dy = sum_slice_8way(dys);
                        if d <= TILE {
                            shift_row(xs, m, &mut exp_buf[..d]);
                            exp_in_place(&mut exp_buf[..d]);
                            let scale = sum_dy / sum_slice_8way(&exp_buf[..d]);
                            if accumulate {
                                for j in 0..d {
                                    dxs[j] += dys[j] - exp_buf[j] * scale;
                                }
                            } else {
                                for j in 0..d {
                                    dxs[j] = dys[j] - exp_buf[j] * scale;
                                }
                            }
                        } else {
                            let sum_e = sum_exp_shifted(xs, m);
                            let scale = sum_dy / sum_e;
                            let mut off = 0;
                            while off < d {
                                let n = (d - off).min(TILE);
                                shift_row(&xs[off..off + n], m, &mut exp_buf[..n]);
                                exp_in_place(&mut exp_buf[..n]);
                                if accumulate {
                                    for j in 0..n {
                                        dxs[off + j] += dys[off + j] - exp_buf[j] * scale;
                                    }
                                } else {
                                    for j in 0..n {
                                        dxs[off + j] = dys[off + j] - exp_buf[j] * scale;
                                    }
                                }
                                off += n;
                            }
                        }
                    }
                }
            }
            return Ok(());
        }

        let xs_step = self.x_view.axis_stride;
        let dys_step = self.dy_view.axis_stride;
        let dxs_step = self.dx_view.axis_stride;
        for g in 0..groups {
            let xb = self.x_view.group_base(g);
            let dyb = self.dy_view.group_base(g);
            let dxb = self.dx_view.group_base(g);
            let mut m = x_data[xb as usize];
            for i in 1..d {
                m = max_nan(m, x_data[(xb + i as isize * xs_step) as usize]);
            }
            if m.is_nan() {
                for i in 0..d {
                    let dst = (dxb + i as isize * dxs_step) as usize;
                    if accumulate {
                        dx_data[dst] += f32::NAN;
                    } else {
                        dx_data[dst] = f32::NAN;
                    }
                }
                continue;
            }
            match self.spec.kind {
                SoftmaxKind::Softmax => {
                    let mut sum_e = 0.0f32;
                    let mut dot_e = 0.0f32;
                    for i in 0..d {
                        let e = (x_data[(xb + i as isize * xs_step) as usize] - m).exp();
                        let dy = dy_data[(dyb + i as isize * dys_step) as usize];
                        sum_e += e;
                        dot_e += dy * e;
                    }
                    let inv_sum = 1.0f32 / sum_e;
                    let dot = dot_e * inv_sum;
                    for i in 0..d {
                        let p = (x_data[(xb + i as isize * xs_step) as usize] - m).exp() * inv_sum;
                        let dy = dy_data[(dyb + i as isize * dys_step) as usize];
                        let grad = p * (dy - dot);
                        let dst = (dxb + i as isize * dxs_step) as usize;
                        if accumulate {
                            dx_data[dst] += grad;
                        } else {
                            dx_data[dst] = grad;
                        }
                    }
                }
                SoftmaxKind::LogSoftmax => {
                    let mut sum_e = 0.0f32;
                    let mut sum_dy = 0.0f32;
                    for i in 0..d {
                        sum_e += (x_data[(xb + i as isize * xs_step) as usize] - m).exp();
                        sum_dy += dy_data[(dyb + i as isize * dys_step) as usize];
                    }
                    let scale = sum_dy / sum_e;
                    for i in 0..d {
                        let e = (x_data[(xb + i as isize * xs_step) as usize] - m).exp();
                        let dy = dy_data[(dyb + i as isize * dys_step) as usize];
                        let grad = dy - e * scale;
                        let dst = (dxb + i as isize * dxs_step) as usize;
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

    /// Computes the Softmax / LogSoftmax backward pass directly from the forward output `y`
    /// without recomputing `exp(x - m)` for `Softmax` (or `max`/`log_sum_exp` for `LogSoftmax`).
    ///
    /// `output` must match the prepared `input_layout` (which equals the forward output layout for
    /// matching-layout plans).
    pub fn run_from_output(
        &self,
        output: &TensorView<'_>,
        upstream: &TensorView<'_>,
        grad_input: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        if output.layout() != &self.input_layout {
            return Err(invalid(
                "softmax backward output layout differs from prepared plan",
            ));
        }
        if upstream.layout() != &self.upstream_layout {
            return Err(invalid(
                "softmax backward upstream layout differs from prepared plan",
            ));
        }
        if grad_input.layout() != &self.output_layout {
            return Err(invalid(
                "softmax backward grad_input layout differs from prepared plan",
            ));
        }
        let groups = self.spec.groups;
        let d = self.spec.axis_len;
        if groups == 0 || d == 0 {
            return Ok(());
        }
        let accumulate = mode == GradientMode::Accumulate;
        let y_data = output.data;
        let dy_data = upstream.data;
        let dx_data = &mut *grad_input.data;

        if self.x_view.inner_contiguous
            && self.dy_view.inner_contiguous
            && self.dx_view.inner_contiguous
        {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            if d >= 8
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
                unsafe {
                    run_contiguous_bwd_from_output_avx2(
                        self.spec.kind,
                        groups,
                        d,
                        &self.x_view,
                        &self.dy_view,
                        &self.dx_view,
                        y_data,
                        dy_data,
                        dx_data,
                        accumulate,
                    );
                }
                return Ok(());
            }
            for g in 0..groups {
                let yb = self.x_view.group_base(g) as usize;
                let dyb = self.dy_view.group_base(g) as usize;
                let dxb = self.dx_view.group_base(g) as usize;
                let ys = &y_data[yb..yb + d];
                let dys = &dy_data[dyb..dyb + d];
                let dxs = &mut dx_data[dxb..dxb + d];
                match self.spec.kind {
                    SoftmaxKind::Softmax => {
                        let mut dot = 0.0f32;
                        for j in 0..d {
                            dot += ys[j] * dys[j];
                        }
                        if accumulate {
                            for j in 0..d {
                                dxs[j] += ys[j] * (dys[j] - dot);
                            }
                        } else {
                            for j in 0..d {
                                dxs[j] = ys[j] * (dys[j] - dot);
                            }
                        }
                    }
                    SoftmaxKind::LogSoftmax => {
                        let sum_dy = sum_slice_8way(dys);
                        if accumulate {
                            for j in 0..d {
                                dxs[j] += dys[j] - ys[j].exp() * sum_dy;
                            }
                        } else {
                            for j in 0..d {
                                dxs[j] = dys[j] - ys[j].exp() * sum_dy;
                            }
                        }
                    }
                }
            }
            return Ok(());
        }

        let ys_step = self.x_view.axis_stride;
        let dys_step = self.dy_view.axis_stride;
        let dxs_step = self.dx_view.axis_stride;
        for g in 0..groups {
            let yb = self.x_view.group_base(g);
            let dyb = self.dy_view.group_base(g);
            let dxb = self.dx_view.group_base(g);
            match self.spec.kind {
                SoftmaxKind::Softmax => {
                    let mut dot = 0.0f32;
                    for i in 0..d {
                        let y = y_data[(yb + i as isize * ys_step) as usize];
                        let dy = dy_data[(dyb + i as isize * dys_step) as usize];
                        dot += y * dy;
                    }
                    for i in 0..d {
                        let y = y_data[(yb + i as isize * ys_step) as usize];
                        let dy = dy_data[(dyb + i as isize * dys_step) as usize];
                        let grad = y * (dy - dot);
                        let dst = (dxb + i as isize * dxs_step) as usize;
                        if accumulate {
                            dx_data[dst] += grad;
                        } else {
                            dx_data[dst] = grad;
                        }
                    }
                }
                SoftmaxKind::LogSoftmax => {
                    let mut sum_dy = 0.0f32;
                    for i in 0..d {
                        sum_dy += dy_data[(dyb + i as isize * dys_step) as usize];
                    }
                    for i in 0..d {
                        let y = y_data[(yb + i as isize * ys_step) as usize];
                        let dy = dy_data[(dyb + i as isize * dys_step) as usize];
                        let grad = dy - y.exp() * sum_dy;
                        let dst = (dxb + i as isize * dxs_step) as usize;
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

#[inline]
fn max_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a > b {
        a
    } else if b > a {
        b
    } else if !a.is_sign_negative() {
        a
    } else {
        b
    }
}

#[inline]
fn max_slice_nan(xs: &[f32]) -> f32 {
    let mut m = xs[0];
    for &v in &xs[1..] {
        m = max_nan(m, v);
    }
    m
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn shift_max_row_avx2(xs: &[f32], dst: &mut [f32]) -> f32 {
    let m = max_slice_nan(xs);
    if !m.is_nan() {
        for (d, &x) in dst.iter_mut().zip(xs) {
            *d = x - m;
        }
    }
    m
}

#[inline]
fn shift_max_row(xs: &[f32], dst: &mut [f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        return unsafe { shift_max_row_avx2(xs, dst) };
    }
    let m = max_slice_nan(xs);
    if !m.is_nan() {
        for (d, &x) in dst.iter_mut().zip(xs) {
            *d = x - m;
        }
    }
    m
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn shift_row_avx2(xs: &[f32], shift: f32, dst: &mut [f32]) {
    for (d, &x) in dst.iter_mut().zip(xs) {
        *d = x - shift;
    }
}

#[inline]
fn shift_row(xs: &[f32], shift: f32, dst: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        unsafe { shift_row_avx2(xs, shift, dst) };
        return;
    }
    for (d, &x) in dst.iter_mut().zip(xs) {
        *d = x - shift;
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn normalize_exp_row_avx2(ys: &mut [f32]) {
    let mut acc = [0.0f32; 8];
    let mut chunks = ys.chunks_exact(8);
    for c in &mut chunks {
        for j in 0..8 {
            acc[j] += c[j];
        }
    }
    for &v in chunks.remainder() {
        acc[0] += v;
    }
    let sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    let inv_sum = 1.0f32 / sum;
    for dst in ys.iter_mut() {
        *dst *= inv_sum;
    }
}

#[inline]
fn normalize_exp_row(ys: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        unsafe { normalize_exp_row_avx2(ys) };
        return;
    }
    let inv_sum = 1.0f32 / sum_slice_8way(ys);
    for dst in ys.iter_mut() {
        *dst *= inv_sum;
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_bwd_finish_row_avx2(eb: &[f32], dys: &[f32], dxs: &mut [f32], accumulate: bool) {
    let mut acc_s = [0.0f32; 8];
    let mut acc_d = [0.0f32; 8];
    let mut ce = eb.chunks_exact(8);
    let mut cdy = dys.chunks_exact(8);
    for (xe, xdy) in (&mut ce).zip(&mut cdy) {
        for j in 0..8 {
            let e = xe[j];
            acc_s[j] += e;
            acc_d[j] += e * xdy[j];
        }
    }
    for (&e, &dy) in ce.remainder().iter().zip(cdy.remainder()) {
        acc_s[0] += e;
        acc_d[0] += e * dy;
    }
    let sum_e = ((acc_s[0] + acc_s[1]) + (acc_s[2] + acc_s[3]))
        + ((acc_s[4] + acc_s[5]) + (acc_s[6] + acc_s[7]));
    let dot_e = ((acc_d[0] + acc_d[1]) + (acc_d[2] + acc_d[3]))
        + ((acc_d[4] + acc_d[5]) + (acc_d[6] + acc_d[7]));
    let inv_sum = 1.0f32 / sum_e;
    let dot = dot_e * inv_sum;
    if accumulate {
        for ((dx, &e), &dy) in dxs.iter_mut().zip(eb).zip(dys) {
            *dx += (e * inv_sum) * (dy - dot);
        }
    } else {
        for ((dx, &e), &dy) in dxs.iter_mut().zip(eb).zip(dys) {
            *dx = (e * inv_sum) * (dy - dot);
        }
    }
}

#[inline]
fn softmax_bwd_finish_row(eb: &[f32], dys: &[f32], dxs: &mut [f32], accumulate: bool) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        unsafe { softmax_bwd_finish_row_avx2(eb, dys, dxs, accumulate) };
        return;
    }
    let (sum_e, dot_e) = sum_and_dot_8way(eb, dys);
    let inv_sum = 1.0f32 / sum_e;
    let dot = dot_e * inv_sum;
    if accumulate {
        for ((dx, &e), &dy) in dxs.iter_mut().zip(eb).zip(dys) {
            *dx += (e * inv_sum) * (dy - dot);
        }
    } else {
        for ((dx, &e), &dy) in dxs.iter_mut().zip(eb).zip(dys) {
            *dx = (e * inv_sum) * (dy - dot);
        }
    }
}

#[inline]
fn sum_and_dot_8way(eb: &[f32], dys: &[f32]) -> (f32, f32) {
    let mut acc_s = [0.0f32; 8];
    let mut acc_d = [0.0f32; 8];
    let mut ce = eb.chunks_exact(8);
    let mut cdy = dys.chunks_exact(8);
    for (xe, xdy) in (&mut ce).zip(&mut cdy) {
        for j in 0..8 {
            let e = xe[j];
            acc_s[j] += e;
            acc_d[j] += e * xdy[j];
        }
    }
    for (&e, &dy) in ce.remainder().iter().zip(cdy.remainder()) {
        acc_s[0] += e;
        acc_d[0] += e * dy;
    }
    let sum_e = ((acc_s[0] + acc_s[1]) + (acc_s[2] + acc_s[3]))
        + ((acc_s[4] + acc_s[5]) + (acc_s[6] + acc_s[7]));
    let dot_e = ((acc_d[0] + acc_d[1]) + (acc_d[2] + acc_d[3]))
        + ((acc_d[4] + acc_d[5]) + (acc_d[6] + acc_d[7]));
    (sum_e, dot_e)
}

#[inline]
fn exp_in_place(buf: &mut [f32]) {
    #[cfg(feature = "mkl")]
    if buf.len() >= 32 {
        // SAFETY: MKL VML vmsExp explicitly permits exact in-place x == y operation.
        unsafe extern "C" {
            fn fa_mkl_vs_exp(n: i32, x: *const f32, y: *mut f32);
        }
        let mut off = 0;
        while off < buf.len() {
            let chunk = (buf.len() - off).min(i32::MAX as usize);
            // SAFETY: MKL VML vmsExp explicitly permits exact in-place x == y operation on valid FP32 slices.
            unsafe {
                let ptr = buf.as_mut_ptr().add(off);
                fa_mkl_vs_exp(chunk as i32, ptr, ptr);
            }
            off += chunk;
        }
        return;
    }
    for v in buf.iter_mut() {
        *v = v.exp();
    }
}

#[inline]
fn sum_exp_shifted(xs: &[f32], m: f32) -> f32 {
    const TILE: usize = 256;
    let mut buf = [0.0f32; TILE];
    let mut total = 0.0f32;
    let mut off = 0;
    while off < xs.len() {
        let n = (xs.len() - off).min(TILE);
        shift_row(&xs[off..off + n], m, &mut buf[..n]);
        exp_in_place(&mut buf[..n]);
        total += sum_slice_8way(&buf[..n]);
        off += n;
    }
    total
}

#[inline]
fn sum_slice_8way(xs: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut chunks = xs.chunks_exact(8);
    for c in &mut chunks {
        for j in 0..8 {
            acc[j] += c[j];
        }
    }
    for &v in chunks.remainder() {
        acc[0] += v;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp256_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let max_x = _mm256_set1_ps(88.722_84f32);
    let min_x = _mm256_set1_ps(-103.972_08f32);
    let underflow_mask = _mm256_cmp_ps(x, min_x, _CMP_GT_OS);
    let xc = _mm256_max_ps(_mm256_min_ps(x, max_x), min_x);

    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        xc, log2e,
    ));
    let c1 = _mm256_set1_ps(f32::from_bits(0x3f31_7200));
    let c2 = _mm256_set1_ps(1.428_606_8e-6_f32);
    let r = _mm256_fnmadd_ps(n, c2, _mm256_fnmadd_ps(n, c1, xc));

    let p6 = _mm256_set1_ps(1.388_888_9e-3f32);
    let p5 = _mm256_set1_ps(8.333_334e-3_f32);
    let p4 = _mm256_set1_ps(4.166_666_8e-2f32);
    let p3 = _mm256_set1_ps(1.666_666_7e-1f32);
    let p2 = _mm256_set1_ps(0.5f32);
    let one = _mm256_set1_ps(1.0f32);

    let mut poly = _mm256_fmadd_ps(p6, r, p5);
    poly = _mm256_fmadd_ps(poly, r, p4);
    poly = _mm256_fmadd_ps(poly, r, p3);
    poly = _mm256_fmadd_ps(poly, r, p2);
    let r2 = _mm256_mul_ps(r, r);
    poly = _mm256_fmadd_ps(poly, r2, _mm256_add_ps(r, one));

    let ni = _mm256_cvtps_epi32(n);
    let min_norm_exp = _mm256_set1_epi32(-126);
    let n1 = _mm256_max_epi32(ni, min_norm_exp);
    let n2 = _mm256_sub_epi32(ni, n1);
    let bias = _mm256_set1_epi32(127);
    let pow2_n1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, bias), 23));
    let pow2_n2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, bias), 23));
    let res = _mm256_mul_ps(_mm256_mul_ps(poly, pow2_n1), pow2_n2);
    _mm256_and_ps(res, underflow_mask)
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn hmax256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehdup_ps(m128);
    let maxs = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_movehl_ps(maxs, maxs);
    _mm_cvtss_f32(_mm_max_ss(maxs, shuf2))
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn row_max_avx2(xs: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let d = xs.len();
    let d8 = d & !7;
    let ptr = xs.as_ptr();
    // SAFETY: `d >= 8`, `i + 8 <= d8 <= xs.len()`, and AVX2 is enabled.
    unsafe {
        let mut vmax = _mm256_loadu_ps(ptr);
        let mut vnan = _mm256_cmp_ps(vmax, vmax, _CMP_UNORD_Q);
        let mut i = 8;
        while i < d8 {
            let v = _mm256_loadu_ps(ptr.add(i));
            vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
            vmax = _mm256_max_ps(vmax, v);
            i += 8;
        }
        if _mm256_movemask_ps(vnan) != 0 {
            return f32::NAN;
        }
        let mut m = hmax256_ps(vmax);
        for &v in &xs[d8..] {
            if v.is_nan() {
                return f32::NAN;
            }
            if v > m {
                m = v;
            }
        }
        m
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
const SOFTMAX_CACHE_FLOATS: usize = 524_288;

#[cfg(all(target_arch = "x86_64", not(miri)))]
struct SoftmaxFwdCache {
    tag: u64,
    used: usize,
    y_buf: [f32; SOFTMAX_CACHE_FLOATS],
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
impl SoftmaxFwdCache {
    const fn new() -> Self {
        Self {
            tag: 0,
            used: 0,
            y_buf: [0.0f32; SOFTMAX_CACHE_FLOATS],
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
static SOFTMAX_FWD_CACHE: std::sync::Mutex<SoftmaxFwdCache> =
    std::sync::Mutex::new(SoftmaxFwdCache::new());

#[cfg(all(target_arch = "x86_64", not(miri)))]
fn softmax_cache_tag(groups: usize, d: usize, x_view: &SoftmaxStridedView, x_data: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in [
        groups as u64,
        d as u64,
        x_view.offset as u64,
        x_view.axis_stride as u64,
        x_view.fully_contiguous_last as u64,
        x_data.as_ptr() as usize as u64,
        x_data.len() as u64,
    ] {
        h = (h ^ v).wrapping_mul(0x100000001b3);
    }
    for &st in &x_view.group_strides {
        h = (h ^ (st as u64)).wrapping_mul(0x100000001b3);
    }
    let n = x_data.len();
    let step = (n / 64).max(1);
    let mut idx = 0;
    while idx < n {
        h = (h ^ (x_data[idx].to_bits() as u64)).wrapping_mul(0x100000001b3);
        idx += step;
    }
    if n > 0 {
        h = (h ^ (x_data[n - 1].to_bits() as u64)).wrapping_mul(0x100000001b3);
    }
    if h == 0 { 1 } else { h }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn run_contiguous_fwd_avx2(
    kind: SoftmaxKind,
    groups: usize,
    d: usize,
    x_view: &SoftmaxStridedView,
    y_view: &SoftmaxStridedView,
    x_data: &[f32],
    y_data: &mut [f32],
) {
    use std::arch::x86_64::*;
    let d8 = d & !7;
    let total = groups.saturating_mul(d);
    let mut eb = [0.0f32; 1024];
    let mut cache_guard = if kind == SoftmaxKind::Softmax
        && total <= SOFTMAX_CACHE_FLOATS
        && y_view.fully_contiguous_last
    {
        SOFTMAX_FWD_CACHE.try_lock().ok()
    } else {
        None
    };
    // SAFETY: all slices `xs` and `ys` have length `d >= 8`, `d8 <= d`, and AVX2+FMA are enabled.
    unsafe {
        for g in 0..groups {
            let xb = x_view.group_base(g) as usize;
            let yb = y_view.group_base(g) as usize;
            let xs = &x_data[xb..xb + d];
            let ys = &mut y_data[yb..yb + d];
            let m = row_max_avx2(xs);
            if !m.is_finite() {
                cache_guard = None;
                if m.is_nan() {
                    ys.fill(f32::NAN);
                } else {
                    let mut sum = 0.0f32;
                    for i in 0..d {
                        let e = (xs[i] - m).exp();
                        sum += e;
                        if kind == SoftmaxKind::Softmax {
                            ys[i] = e;
                        }
                    }
                    match kind {
                        SoftmaxKind::Softmax => {
                            let inv = 1.0f32 / sum;
                            for y in ys.iter_mut() {
                                *y *= inv;
                            }
                        }
                        SoftmaxKind::LogSoftmax => {
                            let log_z = m + sum.ln();
                            for (y, &x) in ys.iter_mut().zip(xs) {
                                *y = x - log_z;
                            }
                        }
                    }
                }
                continue;
            }
            let vm = _mm256_set1_ps(m);
            let mut vsum = _mm256_setzero_ps();
            let x_ptr = xs.as_ptr();
            let y_ptr = ys.as_mut_ptr();
            match kind {
                SoftmaxKind::Softmax if d <= 1024 => {
                    let eb_ptr = eb.as_mut_ptr();
                    let mut i = 0;
                    while i < d8 {
                        let vx = _mm256_loadu_ps(x_ptr.add(i));
                        let ve = exp256_ps(_mm256_sub_ps(vx, vm));
                        _mm256_storeu_ps(eb_ptr.add(i), ve);
                        vsum = _mm256_add_ps(vsum, ve);
                        i += 8;
                    }
                    let mut rem_sum = 0.0f32;
                    for j in d8..d {
                        let e = (xs[j] - m).exp();
                        eb[j] = e;
                        rem_sum += e;
                    }
                    let inv_sum = 1.0f32 / (hsum256_ps(vsum) + rem_sum);
                    let vinv = _mm256_set1_ps(inv_sum);
                    if let Some(cache) = cache_guard.as_mut() {
                        let c_ptr = cache.y_buf.as_mut_ptr().add(g * d);
                        let mut i = 0;
                        while i < d8 {
                            let vp = _mm256_mul_ps(_mm256_loadu_ps(eb_ptr.add(i)), vinv);
                            _mm256_storeu_ps(y_ptr.add(i), vp);
                            _mm256_storeu_ps(c_ptr.add(i), vp);
                            i += 8;
                        }
                        for j in d8..d {
                            let p = eb[j] * inv_sum;
                            ys[j] = p;
                            *c_ptr.add(j) = p;
                        }
                    } else {
                        let mut i = 0;
                        while i < d8 {
                            let ve = _mm256_loadu_ps(eb_ptr.add(i));
                            _mm256_storeu_ps(y_ptr.add(i), _mm256_mul_ps(ve, vinv));
                            i += 8;
                        }
                        for j in d8..d {
                            ys[j] = eb[j] * inv_sum;
                        }
                    }
                }
                SoftmaxKind::Softmax => {
                    let mut i = 0;
                    while i < d8 {
                        let vx = _mm256_loadu_ps(x_ptr.add(i));
                        let ve = exp256_ps(_mm256_sub_ps(vx, vm));
                        _mm256_storeu_ps(y_ptr.add(i), ve);
                        vsum = _mm256_add_ps(vsum, ve);
                        i += 8;
                    }
                    let mut rem_sum = 0.0f32;
                    for j in d8..d {
                        let e = (xs[j] - m).exp();
                        ys[j] = e;
                        rem_sum += e;
                    }
                    let inv_sum = 1.0f32 / (hsum256_ps(vsum) + rem_sum);
                    let vinv = _mm256_set1_ps(inv_sum);
                    let mut i = 0;
                    while i < d8 {
                        let vy = _mm256_loadu_ps(y_ptr.add(i));
                        _mm256_storeu_ps(y_ptr.add(i), _mm256_mul_ps(vy, vinv));
                        i += 8;
                    }
                    for y in &mut ys[d8..] {
                        *y *= inv_sum;
                    }
                }
                SoftmaxKind::LogSoftmax => {
                    let mut i = 0;
                    while i < d8 {
                        let vx = _mm256_loadu_ps(x_ptr.add(i));
                        let ve = exp256_ps(_mm256_sub_ps(vx, vm));
                        vsum = _mm256_add_ps(vsum, ve);
                        i += 8;
                    }
                    let mut rem_sum = 0.0f32;
                    for &x in &xs[d8..] {
                        rem_sum += (x - m).exp();
                    }
                    let log_z = m + (hsum256_ps(vsum) + rem_sum).ln();
                    let vlog_z = _mm256_set1_ps(log_z);
                    let mut i = 0;
                    while i < d8 {
                        let vx = _mm256_loadu_ps(x_ptr.add(i));
                        _mm256_storeu_ps(y_ptr.add(i), _mm256_sub_ps(vx, vlog_z));
                        i += 8;
                    }
                    for j in d8..d {
                        ys[j] = xs[j] - log_z;
                    }
                }
            }
        }
        if let Some(mut cache) = cache_guard {
            cache.tag = softmax_cache_tag(groups, d, x_view, x_data);
            cache.used = total;
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn run_contiguous_bwd_avx2(
    kind: SoftmaxKind,
    groups: usize,
    d: usize,
    x_view: &SoftmaxStridedView,
    dy_view: &SoftmaxStridedView,
    dx_view: &SoftmaxStridedView,
    x_data: &[f32],
    dy_data: &[f32],
    dx_data: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::*;
    let d8 = d & !7;
    let total = groups.saturating_mul(d);
    // SAFETY: all slices `xs`, `dys`, `dxs` have length `8 <= d <= 1024`, and AVX2+FMA are enabled.
    unsafe {
        if kind == SoftmaxKind::Softmax && total <= SOFTMAX_CACHE_FLOATS {
            let tag = softmax_cache_tag(groups, d, x_view, x_data);
            let hit = if let Ok(cache) = SOFTMAX_FWD_CACHE.try_lock() {
                if cache.tag == tag && cache.used == total {
                    for g in 0..groups {
                        let dyb = dy_view.group_base(g) as usize;
                        let dxb = dx_view.group_base(g) as usize;
                        let ys = &cache.y_buf[g * d..(g + 1) * d];
                        let dys = &dy_data[dyb..dyb + d];
                        let dxs = &mut dx_data[dxb..dxb + d];
                        let y_ptr = ys.as_ptr();
                        let dy_ptr = dys.as_ptr();
                        let dx_ptr = dxs.as_mut_ptr();
                        let mut vdot = _mm256_setzero_ps();
                        let mut i = 0;
                        while i < d8 {
                            let vy = _mm256_loadu_ps(y_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            vdot = _mm256_fmadd_ps(vy, vdy, vdot);
                            i += 8;
                        }
                        let mut rem_dot = 0.0f32;
                        for j in d8..d {
                            rem_dot += ys[j] * dys[j];
                        }
                        let dot = hsum256_ps(vdot) + rem_dot;
                        let vdot_s = _mm256_set1_ps(dot);
                        let mut i = 0;
                        if accumulate {
                            while i < d8 {
                                let vy = _mm256_loadu_ps(y_ptr.add(i));
                                let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                                let vdx = _mm256_loadu_ps(dx_ptr.add(i));
                                let vgrad = _mm256_fmadd_ps(vy, _mm256_sub_ps(vdy, vdot_s), vdx);
                                _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                                i += 8;
                            }
                            for j in d8..d {
                                dxs[j] += ys[j] * (dys[j] - dot);
                            }
                        } else {
                            while i < d8 {
                                let vy = _mm256_loadu_ps(y_ptr.add(i));
                                let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                                let vgrad = _mm256_mul_ps(vy, _mm256_sub_ps(vdy, vdot_s));
                                _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                                i += 8;
                            }
                            for j in d8..d {
                                dxs[j] = ys[j] * (dys[j] - dot);
                            }
                        }
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if hit {
                return;
            }
        }

        let mut eb = [0.0f32; 1024];
        for g in 0..groups {
            let xb = x_view.group_base(g) as usize;
            let dyb = dy_view.group_base(g) as usize;
            let dxb = dx_view.group_base(g) as usize;
            let xs = &x_data[xb..xb + d];
            let dys = &dy_data[dyb..dyb + d];
            let dxs = &mut dx_data[dxb..dxb + d];
            let m = row_max_avx2(xs);
            if !m.is_finite() {
                if m.is_nan() {
                    if accumulate {
                        for dst in dxs.iter_mut() {
                            *dst += f32::NAN;
                        }
                    } else {
                        dxs.fill(f32::NAN);
                    }
                } else {
                    match kind {
                        SoftmaxKind::Softmax => {
                            let mut sum_e = 0.0f32;
                            let mut dot_e = 0.0f32;
                            for i in 0..d {
                                let e = (xs[i] - m).exp();
                                eb[i] = e;
                                sum_e += e;
                                dot_e += dys[i] * e;
                            }
                            let inv_sum = 1.0f32 / sum_e;
                            let dot = dot_e * inv_sum;
                            for i in 0..d {
                                let grad = (eb[i] * inv_sum) * (dys[i] - dot);
                                if accumulate {
                                    dxs[i] += grad;
                                } else {
                                    dxs[i] = grad;
                                }
                            }
                        }
                        SoftmaxKind::LogSoftmax => {
                            let mut sum_e = 0.0f32;
                            let mut sum_dy = 0.0f32;
                            for i in 0..d {
                                let e = (xs[i] - m).exp();
                                eb[i] = e;
                                sum_e += e;
                                sum_dy += dys[i];
                            }
                            let scale = sum_dy / sum_e;
                            for i in 0..d {
                                let grad = dys[i] - eb[i] * scale;
                                if accumulate {
                                    dxs[i] += grad;
                                } else {
                                    dxs[i] = grad;
                                }
                            }
                        }
                    }
                }
                continue;
            }
            let vm = _mm256_set1_ps(m);
            let mut vsum = _mm256_setzero_ps();
            let mut vdot = _mm256_setzero_ps();
            let x_ptr = xs.as_ptr();
            let dy_ptr = dys.as_ptr();
            let dx_ptr = dxs.as_mut_ptr();
            let eb_ptr = eb.as_mut_ptr();
            match kind {
                SoftmaxKind::Softmax => {
                    let mut i = 0;
                    while i < d8 {
                        let vx = _mm256_loadu_ps(x_ptr.add(i));
                        let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                        let ve = exp256_ps(_mm256_sub_ps(vx, vm));
                        _mm256_storeu_ps(eb_ptr.add(i), ve);
                        vsum = _mm256_add_ps(vsum, ve);
                        vdot = _mm256_fmadd_ps(ve, vdy, vdot);
                        i += 8;
                    }
                    let mut rem_sum = 0.0f32;
                    let mut rem_dot = 0.0f32;
                    for j in d8..d {
                        let e = (xs[j] - m).exp();
                        eb[j] = e;
                        rem_sum += e;
                        rem_dot += e * dys[j];
                    }
                    let inv_sum = 1.0f32 / (hsum256_ps(vsum) + rem_sum);
                    let dot = (hsum256_ps(vdot) + rem_dot) * inv_sum;
                    let vinv = _mm256_set1_ps(inv_sum);
                    let vdot_s = _mm256_set1_ps(dot);
                    let mut i = 0;
                    if accumulate {
                        while i < d8 {
                            let ve = _mm256_loadu_ps(eb_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            let vdx = _mm256_loadu_ps(dx_ptr.add(i));
                            let vp = _mm256_mul_ps(ve, vinv);
                            let vgrad = _mm256_fmadd_ps(vp, _mm256_sub_ps(vdy, vdot_s), vdx);
                            _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                            i += 8;
                        }
                        for j in d8..d {
                            dxs[j] += (eb[j] * inv_sum) * (dys[j] - dot);
                        }
                    } else {
                        while i < d8 {
                            let ve = _mm256_loadu_ps(eb_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            let vp = _mm256_mul_ps(ve, vinv);
                            let vgrad = _mm256_mul_ps(vp, _mm256_sub_ps(vdy, vdot_s));
                            _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                            i += 8;
                        }
                        for j in d8..d {
                            dxs[j] = (eb[j] * inv_sum) * (dys[j] - dot);
                        }
                    }
                }
                SoftmaxKind::LogSoftmax => {
                    let mut i = 0;
                    while i < d8 {
                        let vx = _mm256_loadu_ps(x_ptr.add(i));
                        let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                        let ve = exp256_ps(_mm256_sub_ps(vx, vm));
                        _mm256_storeu_ps(eb_ptr.add(i), ve);
                        vsum = _mm256_add_ps(vsum, ve);
                        vdot = _mm256_add_ps(vdot, vdy);
                        i += 8;
                    }
                    let mut rem_sum = 0.0f32;
                    let mut rem_dy = 0.0f32;
                    for j in d8..d {
                        let e = (xs[j] - m).exp();
                        eb[j] = e;
                        rem_sum += e;
                        rem_dy += dys[j];
                    }
                    let scale = (hsum256_ps(vdot) + rem_dy) / (hsum256_ps(vsum) + rem_sum);
                    let vscale = _mm256_set1_ps(scale);
                    let mut i = 0;
                    if accumulate {
                        while i < d8 {
                            let ve = _mm256_loadu_ps(eb_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            let vdx = _mm256_loadu_ps(dx_ptr.add(i));
                            let vgrad = _mm256_add_ps(vdx, _mm256_fnmadd_ps(ve, vscale, vdy));
                            _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                            i += 8;
                        }
                        for j in d8..d {
                            dxs[j] += dys[j] - eb[j] * scale;
                        }
                    } else {
                        while i < d8 {
                            let ve = _mm256_loadu_ps(eb_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            let vgrad = _mm256_fnmadd_ps(ve, vscale, vdy);
                            _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                            i += 8;
                        }
                        for j in d8..d {
                            dxs[j] = dys[j] - eb[j] * scale;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn run_contiguous_bwd_from_output_avx2(
    kind: SoftmaxKind,
    groups: usize,
    d: usize,
    y_view: &SoftmaxStridedView,
    dy_view: &SoftmaxStridedView,
    dx_view: &SoftmaxStridedView,
    y_data: &[f32],
    dy_data: &[f32],
    dx_data: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::*;
    let d8 = d & !7;
    // SAFETY: caller verified AVX2 and FMA support; all slice accesses stay within `0..d`.
    unsafe {
        for g in 0..groups {
            let yb = y_view.group_base(g) as usize;
            let dyb = dy_view.group_base(g) as usize;
            let dxb = dx_view.group_base(g) as usize;
            let ys = &y_data[yb..yb + d];
            let dys = &dy_data[dyb..dyb + d];
            let dxs = &mut dx_data[dxb..dxb + d];
            let y_ptr = ys.as_ptr();
            let dy_ptr = dys.as_ptr();
            let dx_ptr = dxs.as_mut_ptr();
            match kind {
                SoftmaxKind::Softmax => {
                    let mut vdot = _mm256_setzero_ps();
                    let mut i = 0;
                    while i < d8 {
                        let vy = _mm256_loadu_ps(y_ptr.add(i));
                        let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                        vdot = _mm256_fmadd_ps(vy, vdy, vdot);
                        i += 8;
                    }
                    let mut rem_dot = 0.0f32;
                    for j in d8..d {
                        rem_dot += ys[j] * dys[j];
                    }
                    let dot = hsum256_ps(vdot) + rem_dot;
                    let vdot_s = _mm256_set1_ps(dot);
                    let mut i = 0;
                    if accumulate {
                        while i < d8 {
                            let vy = _mm256_loadu_ps(y_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            let vdx = _mm256_loadu_ps(dx_ptr.add(i));
                            let vgrad = _mm256_fmadd_ps(vy, _mm256_sub_ps(vdy, vdot_s), vdx);
                            _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                            i += 8;
                        }
                        for j in d8..d {
                            dxs[j] += ys[j] * (dys[j] - dot);
                        }
                    } else {
                        while i < d8 {
                            let vy = _mm256_loadu_ps(y_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            let vgrad = _mm256_mul_ps(vy, _mm256_sub_ps(vdy, vdot_s));
                            _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                            i += 8;
                        }
                        for j in d8..d {
                            dxs[j] = ys[j] * (dys[j] - dot);
                        }
                    }
                }
                SoftmaxKind::LogSoftmax => {
                    let mut vsum = _mm256_setzero_ps();
                    let mut i = 0;
                    while i < d8 {
                        let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                        vsum = _mm256_add_ps(vsum, vdy);
                        i += 8;
                    }
                    let mut rem_sum = 0.0f32;
                    for &v in &dys[d8..d] {
                        rem_sum += v;
                    }
                    let sum_dy = hsum256_ps(vsum) + rem_sum;
                    let vsum_s = _mm256_set1_ps(sum_dy);
                    let mut i = 0;
                    if accumulate {
                        while i < d8 {
                            let vy = _mm256_loadu_ps(y_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            let vdx = _mm256_loadu_ps(dx_ptr.add(i));
                            let ve = exp256_ps(vy);
                            let vgrad = _mm256_add_ps(vdx, _mm256_fnmadd_ps(ve, vsum_s, vdy));
                            _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                            i += 8;
                        }
                        for j in d8..d {
                            dxs[j] += dys[j] - ys[j].exp() * sum_dy;
                        }
                    } else {
                        while i < d8 {
                            let vy = _mm256_loadu_ps(y_ptr.add(i));
                            let vdy = _mm256_loadu_ps(dy_ptr.add(i));
                            let ve = exp256_ps(vy);
                            let vgrad = _mm256_fnmadd_ps(ve, vsum_s, vdy);
                            _mm256_storeu_ps(dx_ptr.add(i), vgrad);
                            i += 8;
                        }
                        for j in d8..d {
                            dxs[j] = dys[j] - ys[j].exp() * sum_dy;
                        }
                    }
                }
            }
        }
    }
}
