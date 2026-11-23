//! Prepared fused numerically stable `CrossEntropy` loss and local VJP on CPU.
//!
//! Computes mean cross-entropy over `logits [N, V]` and `targets [N]` without materializing an
//! `[N, V]` softmax probability matrix:
//! - Forward:  `L_i = -log_softmax(logits_i)[target_i]` (with optional `ignore_index` and `label_smoothing`)
//! - Backward: `dLogits[i, j] = upstream * (p[i, j] - target_dist[i, j]) / active_count`

use crate::{GradientMode, IndexTensor, Result, Tensor, error::invalid};

fn use_simd(vocab: usize) -> bool {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if vocab >= 32 {
        return is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    }
    let _ = vocab;
    false
}

fn row_max(row: &[f32], simd: bool) -> f32 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if simd {
        // SAFETY: use_simd checked the instruction set and a width of at least 32.
        return unsafe { crate::softmax::row_max_avx2(row) };
    }
    let _ = simd;
    row.iter().fold(
        f32::NEG_INFINITY,
        |m, &x| {
            if x.is_nan() || x > m { x } else { m }
        },
    )
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_sum_avx2<const STORE: bool>(row: &[f32], max: f32, exps: &mut [f32]) -> f64 {
    use std::arch::x86_64::*;
    debug_assert!(!STORE || row.len() == exps.len());
    let width = row.len() & !7;
    let vmax = _mm256_set1_ps(max);
    let mut sum_lo = _mm256_setzero_pd();
    let mut sum_hi = _mm256_setzero_pd();
    // SAFETY: every full-vector load and optional store is within the row.
    unsafe {
        for col in (0..width).step_by(8) {
            let x = _mm256_loadu_ps(row.as_ptr().add(col));
            let e = crate::softmax::exp256_ps(_mm256_sub_ps(x, vmax));
            if STORE {
                _mm256_storeu_ps(exps.as_mut_ptr().add(col), e);
            }
            sum_lo = _mm256_add_pd(sum_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(e)));
            sum_hi = _mm256_add_pd(sum_hi, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(e)));
        }
        let mut lanes = [0.0; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), _mm256_add_pd(sum_lo, sum_hi));
        let mut sum: f64 = lanes.iter().sum();
        for col in width..row.len() {
            let e = (row[col] - max).exp();
            if STORE {
                exps[col] = e;
            }
            sum += f64::from(e);
        }
        sum
    }
}

fn forward_exp_sum(row: &[f32], max: f32, simd: bool) -> f64 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if simd && max.is_finite() {
        // SAFETY: use_simd checked AVX2 and FMA. The forward route performs no stores.
        return unsafe { exp_sum_avx2::<false>(row, max, &mut []) };
    }
    let _ = simd;
    row.iter()
        .map(|&x| (f64::from(x) - f64::from(max)).exp())
        .sum()
}

fn overwrite_exp_sum(row: &[f32], max: f32, exps: &mut [f32], simd: bool) -> f64 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if simd && max.is_finite() {
        // SAFETY: use_simd checked AVX2 and FMA. Source and destination have identical shapes.
        return unsafe { exp_sum_avx2::<true>(row, max, exps) };
    }
    let _ = simd;
    row.iter()
        .zip(exps)
        .map(|(&val, out)| {
            *out = (val - max).exp();
            f64::from(*out)
        })
        .sum()
}

/// Validated specification for fused `CrossEntropy` over `[num_rows, vocab_size]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossEntropySpec {
    num_rows: usize,
    vocab_size: usize,
    ignore_index: Option<i64>,
    label_smoothing: f32,
}

impl CrossEntropySpec {
    pub fn new(
        num_rows: usize,
        vocab_size: usize,
        ignore_index: Option<i64>,
        label_smoothing: f32,
    ) -> Result<Self> {
        if vocab_size == 0 {
            return Err(invalid("cross_entropy vocab_size must be positive"));
        }
        if !label_smoothing.is_finite() || !(0.0..1.0).contains(&label_smoothing) {
            return Err(invalid("cross_entropy label_smoothing must lie in [0, 1)"));
        }
        Ok(Self {
            num_rows,
            vocab_size,
            ignore_index,
            label_smoothing,
        })
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn ignore_index(&self) -> Option<i64> {
        self.ignore_index
    }
    pub fn label_smoothing(&self) -> f32 {
        self.label_smoothing
    }
}

/// Prepared CPU fused `CrossEntropy` forward plan (zero hot allocations).
pub struct CrossEntropyPlan {
    spec: CrossEntropySpec,
}

impl CrossEntropyPlan {
    pub fn new(spec: CrossEntropySpec) -> Self {
        Self { spec }
    }

    pub fn spec(&self) -> CrossEntropySpec {
        self.spec
    }

    /// Computes scalar mean cross-entropy loss (`output` has shape `[]`).
    pub fn run(&self, logits: &Tensor, targets: &IndexTensor, output: &mut Tensor) -> Result<()> {
        logits
            .shape()
            .expect(&[self.spec.num_rows, self.spec.vocab_size])?;
        targets.shape().expect(&[self.spec.num_rows])?;
        output.shape().expect(&[])?;

        let n = self.spec.num_rows;
        let v = self.spec.vocab_size;
        let ignore = self.spec.ignore_index;
        let eps = self.spec.label_smoothing as f64;
        let simd = use_simd(v);
        let t_data = targets.data();
        let x_data = logits.data();

        for &tok in t_data {
            if Some(tok) != ignore && (tok < 0 || tok >= v as i64) {
                return Err(invalid(format!(
                    "cross_entropy target {tok} out of range [0, {v})"
                )));
            }
        }

        let mut total_loss = 0.0f64;
        let mut active = 0usize;
        for (i, &tok) in t_data.iter().enumerate().take(n) {
            if Some(tok) == ignore {
                continue;
            }
            active += 1;
            let row = &x_data[i * v..(i + 1) * v];
            let m = row_max(row, simd);
            let log_sum_exp = forward_exp_sum(row, m, simd).ln();
            let nll = log_sum_exp + (f64::from(m) - f64::from(row[tok as usize]));
            let smooth_loss = if eps > 0.0 {
                let sum_shifted: f64 = row.iter().map(|&x| f64::from(x) - f64::from(m)).sum();
                let mean_log_p = (sum_shifted / v as f64) - log_sum_exp;
                (1.0 - eps) * nll - eps * mean_log_p
            } else {
                nll
            };
            total_loss += smooth_loss;
        }
        output.data_mut()[0] = if active > 0 {
            (total_loss / active as f64) as f32
        } else {
            0.0
        };
        Ok(())
    }
}

/// Prepared CPU fused `CrossEntropy` local VJP (`dLogits` without materializing `[N, V]` probs).
pub struct CrossEntropyBackward {
    spec: CrossEntropySpec,
}

impl CrossEntropyBackward {
    pub fn new(spec: CrossEntropySpec) -> Self {
        Self { spec }
    }

    pub fn spec(&self) -> CrossEntropySpec {
        self.spec
    }

    pub fn run(
        &self,
        logits: &Tensor,
        targets: &IndexTensor,
        upstream: f32,
        grad_logits: &mut Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        logits
            .shape()
            .expect(&[self.spec.num_rows, self.spec.vocab_size])?;
        targets.shape().expect(&[self.spec.num_rows])?;
        grad_logits
            .shape()
            .expect(&[self.spec.num_rows, self.spec.vocab_size])?;

        let n = self.spec.num_rows;
        let v = self.spec.vocab_size;
        let ignore = self.spec.ignore_index;
        let eps = self.spec.label_smoothing;
        let t_data = targets.data();
        let x_data = logits.data();
        let accumulate = mode == GradientMode::Accumulate;
        let simd = use_simd(v) && !accumulate;

        let mut active = 0usize;
        for &tok in t_data {
            if Some(tok) == ignore {
                continue;
            }
            if tok < 0 || tok >= v as i64 {
                return Err(invalid(format!(
                    "cross_entropy target {tok} out of range [0, {v})"
                )));
            }
            active += 1;
        }

        let gx = grad_logits.data_mut();
        if active == 0 {
            if !accumulate {
                gx.fill(0.0);
            }
            return Ok(());
        }

        let scale = upstream / active as f32;
        let smooth_off = eps / v as f32;
        let target_weight = 1.0f32 - eps;

        for (i, &tok) in t_data.iter().enumerate().take(n) {
            let row = &x_data[i * v..(i + 1) * v];
            let dst = &mut gx[i * v..(i + 1) * v];
            if Some(tok) == ignore {
                if !accumulate {
                    dst.fill(0.0);
                }
                continue;
            }
            let m = row_max(row, simd);
            let t_idx = tok as usize;
            if accumulate {
                let sum_exp: f64 = row.iter().map(|&val| f64::from((val - m).exp())).sum();
                let inv_sum = (1.0 / sum_exp) as f32;
                for j in 0..v {
                    let p = (row[j] - m).exp() * inv_sum;
                    let tgt = smooth_off + if j == t_idx { target_weight } else { 0.0 };
                    dst[j] += (p - tgt) * scale;
                }
            } else {
                // Reuse the overwrite destination for exponentials until the row sum is known.
                let sum_exp = overwrite_exp_sum(row, m, dst, simd);
                let inv_sum = (1.0 / sum_exp) as f32;
                for (j, out) in dst.iter_mut().enumerate() {
                    let p = *out * inv_sum;
                    let tgt = smooth_off + if j == t_idx { target_weight } else { 0.0 };
                    *out = (p - tgt) * scale;
                }
            }
        }
        Ok(())
    }
}
