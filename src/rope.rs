//! Prepared differentiable Rotary Positional Embedding (`RoPE`, half-split layout) on CPU.
//!
//! Applies orthogonal 2D rotations along pairs `(d, d + rotary_dim/2)` for `d in 0..rotary_dim/2`
//! while passing trailing dimensions `rotary_dim..head_dim` through unchanged:
//! - Forward:  `y[d] = x[d] * cos - x[d + half] * sin`, `y[d + half] = x[d] * sin + x[d + half] * cos`
//! - Backward: `dx[d] = dy[d] * cos + dy[d + half] * sin`, `dx[d + half] = -dy[d] * sin + dy[d + half] * cos`
//!
//! Because rotation is linear and orthogonal, `RopeBackward` requires no saved forward activations
//! (`input_required() == false`).

use crate::{GradientMode, Layout, Result, Shape, TensorView, TensorViewMut, error::invalid};

/// Validated geometry and frequency schedule for 4D Rotary Positional Embeddings.
///
/// Supports both `[B, H, T, D]` (`seq_axis == 2`) and `[B, T, H, D]` (`seq_axis == 1`).
#[derive(Clone, Debug, PartialEq)]
pub struct RopeSpec {
    shape: Shape,
    seq_axis: usize,
    rotary_dim: usize,
    base: f32,
    position_offset: usize,
}

impl RopeSpec {
    pub fn new(
        shape: &Shape,
        seq_axis: usize,
        rotary_dim: Option<usize>,
        base: f32,
        position_offset: usize,
    ) -> Result<Self> {
        let dims = shape.dims();
        if dims.len() != 4 {
            return Err(invalid(
                "RoPE requires a rank-4 tensor [B,H,T,D] or [B,T,H,D]",
            ));
        }
        if seq_axis != 1 && seq_axis != 2 {
            return Err(invalid(
                "RoPE seq_axis must be 1 ([B,T,H,D]) or 2 ([B,H,T,D])",
            ));
        }
        let head_dim = dims[3];
        let rdim = rotary_dim.unwrap_or(head_dim);
        if rdim == 0 || (rdim & 1) != 0 || rdim > head_dim {
            return Err(invalid(
                "RoPE rotary_dim must be positive, even, and <= head_dim",
            ));
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(invalid("RoPE base frequency must be finite and positive"));
        }
        Ok(Self {
            shape: shape.clone(),
            seq_axis,
            rotary_dim: rdim,
            base,
            position_offset,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn seq_axis(&self) -> usize {
        self.seq_axis
    }
    pub fn seq_len(&self) -> usize {
        self.shape.dims()[self.seq_axis]
    }
    pub fn head_dim(&self) -> usize {
        self.shape.dims()[3]
    }
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }
    pub fn base(&self) -> f32 {
        self.base
    }
    pub fn position_offset(&self) -> usize {
        self.position_offset
    }
    pub fn batch(&self) -> usize {
        self.shape.dims()[0]
    }
    pub fn num_heads(&self) -> usize {
        self.shape.dims()[if self.seq_axis == 2 { 1 } else { 2 }]
    }

    /// Precomputes exact FP64->FP32 `(cos, sin)` tables of shape `[seq_len, rotary_dim / 2]`.
    pub fn build_tables(&self) -> (Vec<f32>, Vec<f32>) {
        let t_len = self.seq_len();
        let half = self.rotary_dim / 2;
        let mut cos = Vec::with_capacity(t_len * half);
        let mut sin = Vec::with_capacity(t_len * half);
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| {
                let exp = -(2.0 * i as f64) / (self.rotary_dim as f64);
                (self.base as f64).powf(exp)
            })
            .collect();
        for t in 0..t_len {
            let pos = (t + self.position_offset) as f64;
            for &w in &inv_freq {
                let angle = pos * w;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        (cos, sin)
    }

    pub(crate) fn metadata_bytes(&self) -> usize {
        size_of::<Self>() + self.seq_len() * self.rotary_dim * size_of::<f32>()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RopeView4 {
    pub offset: isize,
    pub strides: [isize; 4],
}

impl RopeView4 {
    pub(crate) fn new(layout: &Layout, expected: &Shape, writable: bool) -> Result<Self> {
        if layout.shape() != expected {
            return Err(invalid("RoPE layout shape differs from spec"));
        }
        if writable && !layout.is_non_overlapping() {
            return Err(crate::Error::Unsupported(
                "RoPE output layout must be proved non-overlapping".into(),
            ));
        }
        let s = layout.strides();
        Ok(Self {
            offset: layout.offset() as isize,
            strides: [s[0], s[1], s[2], s[3]],
        })
    }

    #[inline]
    pub(crate) fn row_base(&self, d0: usize, d1: usize, d2: usize) -> isize {
        self.offset
            + d0 as isize * self.strides[0]
            + d1 as isize * self.strides[1]
            + d2 as isize * self.strides[2]
    }
}

/// Prepared CPU `RoPE` forward plan with precomputed `(cos, sin)` tables (zero hot allocations).
pub struct RopePlan {
    spec: RopeSpec,
    input_layout: Layout,
    output_layout: Layout,
    in_view: RopeView4,
    out_view: RopeView4,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopePlan {
    pub fn new(spec: &RopeSpec, input: &Layout, output: &Layout) -> Result<Self> {
        let in_view = RopeView4::new(input, spec.shape(), false)?;
        let out_view = RopeView4::new(output, spec.shape(), true)?;
        let (cos, sin) = spec.build_tables();
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
            in_view,
            out_view,
            cos,
            sin,
        })
    }

    pub fn spec(&self) -> &RopeSpec {
        &self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }

    pub fn run(&self, input: &TensorView<'_>, output: &mut TensorViewMut<'_>) -> Result<()> {
        if input.layout() != &self.input_layout || output.layout() != &self.output_layout {
            return Err(invalid("RoPE layouts differ from prepared plan"));
        }
        run_rope_cpu(
            &self.spec,
            self.in_view,
            self.out_view,
            &self.cos,
            &self.sin,
            input.data,
            output.data,
            false,
            false,
        );
        Ok(())
    }
}

/// Prepared CPU `RoPE` local VJP (`input_required() == false`, zero hot allocations).
pub struct RopeBackward {
    spec: RopeSpec,
    upstream_layout: Layout,
    output_layout: Layout,
    up_view: RopeView4,
    out_view: RopeView4,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeBackward {
    pub fn new(spec: &RopeSpec, upstream: &Layout, output: &Layout) -> Result<Self> {
        let up_view = RopeView4::new(upstream, spec.shape(), false)?;
        let out_view = RopeView4::new(output, spec.shape(), true)?;
        let (cos, sin) = spec.build_tables();
        Ok(Self {
            spec: spec.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            up_view,
            out_view,
            cos,
            sin,
        })
    }

    pub fn spec(&self) -> &RopeSpec {
        &self.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
    }

    pub fn run(
        &self,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        if upstream.layout() != &self.upstream_layout || output.layout() != &self.output_layout {
            return Err(invalid("RoPE backward layouts differ from prepared plan"));
        }
        run_rope_cpu(
            &self.spec,
            self.up_view,
            self.out_view,
            &self.cos,
            &self.sin,
            upstream.data,
            output.data,
            true,
            mode == GradientMode::Accumulate,
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_rope_cpu(
    spec: &RopeSpec,
    in_view: RopeView4,
    out_view: RopeView4,
    cos: &[f32],
    sin: &[f32],
    src: &[f32],
    dst: &mut [f32],
    inverse: bool,
    accumulate: bool,
) {
    let dims = spec.shape().dims();
    let (d0_len, d1_len, d2_len, d3_len) = (dims[0], dims[1], dims[2], dims[3]);
    if d0_len == 0 || d1_len == 0 || d2_len == 0 || d3_len == 0 {
        return;
    }
    let half = spec.rotary_dim() / 2;
    let rdim = spec.rotary_dim();
    let seq_axis = spec.seq_axis();
    let inner_contig = in_view.strides[3] == 1 && out_view.strides[3] == 1;

    if inner_contig {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        if half >= 8
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime CPUID check verified AVX2 and FMA support; bounds checked via layouts.
            unsafe {
                run_rope_contig_avx2(
                    dims, seq_axis, half, rdim, in_view, out_view, cos, sin, src, dst, inverse,
                    accumulate,
                );
            }
            return;
        }
        for i0 in 0..d0_len {
            for i1 in 0..d1_len {
                for i2 in 0..d2_len {
                    let t = if seq_axis == 2 { i2 } else { i1 };
                    let cs_off = t * half;
                    let c_row = &cos[cs_off..cs_off + half];
                    let s_row = &sin[cs_off..cs_off + half];
                    let ib = in_view.row_base(i0, i1, i2) as usize;
                    let ob = out_view.row_base(i0, i1, i2) as usize;
                    let x = &src[ib..ib + d3_len];
                    let y = &mut dst[ob..ob + d3_len];
                    for j in 0..half {
                        let x0 = x[j];
                        let x1 = x[j + half];
                        let c = c_row[j];
                        let s = s_row[j];
                        let (r0, r1) = if inverse {
                            (x0 * c + x1 * s, -x0 * s + x1 * c)
                        } else {
                            (x0 * c - x1 * s, x0 * s + x1 * c)
                        };
                        if accumulate {
                            y[j] += r0;
                            y[j + half] += r1;
                        } else {
                            y[j] = r0;
                            y[j + half] = r1;
                        }
                    }
                    for j in rdim..d3_len {
                        if accumulate {
                            y[j] += x[j];
                        } else {
                            y[j] = x[j];
                        }
                    }
                }
            }
        }
        return;
    }

    let is3 = in_view.strides[3];
    let os3 = out_view.strides[3];
    for i0 in 0..d0_len {
        for i1 in 0..d1_len {
            for i2 in 0..d2_len {
                let t = if seq_axis == 2 { i2 } else { i1 };
                let cs_off = t * half;
                let ib = in_view.row_base(i0, i1, i2);
                let ob = out_view.row_base(i0, i1, i2);
                for j in 0..half {
                    let x0 = src[(ib + j as isize * is3) as usize];
                    let x1 = src[(ib + (j + half) as isize * is3) as usize];
                    let c = cos[cs_off + j];
                    let s = sin[cs_off + j];
                    let (r0, r1) = if inverse {
                        (x0 * c + x1 * s, -x0 * s + x1 * c)
                    } else {
                        (x0 * c - x1 * s, x0 * s + x1 * c)
                    };
                    let o0 = (ob + j as isize * os3) as usize;
                    let o1 = (ob + (j + half) as isize * os3) as usize;
                    if accumulate {
                        dst[o0] += r0;
                        dst[o1] += r1;
                    } else {
                        dst[o0] = r0;
                        dst[o1] = r1;
                    }
                }
                for j in rdim..d3_len {
                    let v = src[(ib + j as isize * is3) as usize];
                    let o = (ob + j as isize * os3) as usize;
                    if accumulate {
                        dst[o] += v;
                    } else {
                        dst[o] = v;
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn run_rope_contig_avx2(
    dims: &[usize],
    seq_axis: usize,
    half: usize,
    rdim: usize,
    in_view: RopeView4,
    out_view: RopeView4,
    cos: &[f32],
    sin: &[f32],
    src: &[f32],
    dst: &mut [f32],
    inverse: bool,
    accumulate: bool,
) {
    use std::arch::x86_64::*;
    let (d0_len, d1_len, d2_len, d3_len) = (dims[0], dims[1], dims[2], dims[3]);
    let half8 = half & !7;
    // SAFETY: caller verified AVX2+FMA support and validated exact slice bounds.
    unsafe {
        for i0 in 0..d0_len {
            for i1 in 0..d1_len {
                for i2 in 0..d2_len {
                    let t = if seq_axis == 2 { i2 } else { i1 };
                    let cs_off = t * half;
                    let c_ptr = cos.as_ptr().add(cs_off);
                    let s_ptr = sin.as_ptr().add(cs_off);
                    let ib = in_view.row_base(i0, i1, i2) as usize;
                    let ob = out_view.row_base(i0, i1, i2) as usize;
                    let x_ptr = src.as_ptr().add(ib);
                    let y_ptr = dst.as_mut_ptr().add(ob);
                    let mut j = 0;
                    while j < half8 {
                        let vx0 = _mm256_loadu_ps(x_ptr.add(j));
                        let vx1 = _mm256_loadu_ps(x_ptr.add(j + half));
                        let vc = _mm256_loadu_ps(c_ptr.add(j));
                        let vs = _mm256_loadu_ps(s_ptr.add(j));
                        let (vr0, vr1) = if inverse {
                            (
                                _mm256_fmadd_ps(vx0, vc, _mm256_mul_ps(vx1, vs)),
                                _mm256_fmsub_ps(vx1, vc, _mm256_mul_ps(vx0, vs)),
                            )
                        } else {
                            (
                                _mm256_fmsub_ps(vx0, vc, _mm256_mul_ps(vx1, vs)),
                                _mm256_fmadd_ps(vx0, vs, _mm256_mul_ps(vx1, vc)),
                            )
                        };
                        if accumulate {
                            let prev0 = _mm256_loadu_ps(y_ptr.add(j));
                            let prev1 = _mm256_loadu_ps(y_ptr.add(j + half));
                            _mm256_storeu_ps(y_ptr.add(j), _mm256_add_ps(prev0, vr0));
                            _mm256_storeu_ps(y_ptr.add(j + half), _mm256_add_ps(prev1, vr1));
                        } else {
                            _mm256_storeu_ps(y_ptr.add(j), vr0);
                            _mm256_storeu_ps(y_ptr.add(j + half), vr1);
                        }
                        j += 8;
                    }
                    while j < half {
                        let x0 = *x_ptr.add(j);
                        let x1 = *x_ptr.add(j + half);
                        let c = *c_ptr.add(j);
                        let s = *s_ptr.add(j);
                        let (r0, r1) = if inverse {
                            (x0 * c + x1 * s, -x0 * s + x1 * c)
                        } else {
                            (x0 * c - x1 * s, x0 * s + x1 * c)
                        };
                        if accumulate {
                            *y_ptr.add(j) += r0;
                            *y_ptr.add(j + half) += r1;
                        } else {
                            *y_ptr.add(j) = r0;
                            *y_ptr.add(j + half) = r1;
                        }
                        j += 1;
                    }
                    for k in rdim..d3_len {
                        let v = *x_ptr.add(k);
                        if accumulate {
                            *y_ptr.add(k) += v;
                        } else {
                            *y_ptr.add(k) = v;
                        }
                    }
                }
            }
        }
    }
}
