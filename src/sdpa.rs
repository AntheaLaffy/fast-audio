#![allow(clippy::needless_range_loop)]

use crate::{
    Error, GradientMode, Layout, Result, Shape, TensorView, TensorViewMut, error::invalid,
};

/// Which input operand of `Sdpa` a local VJP differentiates with respect to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum SdpaOperand {
    Query = 0,
    Key = 1,
    Value = 2,
}

/// Validated geometry and scale/mask policy for Scaled Dot-Product Attention (MHA & GQA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdpaSpec {
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    scale_bits: u32,
    is_causal: bool,
}

impl SdpaSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: usize,
        num_heads: usize,
        num_kv_heads: usize,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        scale: Option<f32>,
        is_causal: bool,
    ) -> Result<Self> {
        if head_dim == 0 && batch > 0 && num_heads > 0 && seq_q > 0 {
            return Err(invalid("sdpa head_dim must be positive"));
        }
        if num_kv_heads == 0 && (num_heads > 0 || (batch > 0 && seq_k > 0)) {
            return Err(invalid("sdpa num_kv_heads must be positive"));
        }
        if num_kv_heads > 0 && !num_heads.is_multiple_of(num_kv_heads) {
            return Err(invalid("sdpa num_heads must be divisible by num_kv_heads"));
        }
        if seq_k == 0 && batch > 0 && num_heads > 0 && seq_q > 0 {
            return Err(invalid("sdpa key sequence length must be positive"));
        }
        let s = match scale {
            Some(v) => v,
            None => {
                if head_dim == 0 {
                    1.0
                } else {
                    1.0 / (head_dim as f32).sqrt()
                }
            }
        };
        if !s.is_finite() || s <= 0.0 {
            return Err(invalid("sdpa scale must be finite and positive"));
        }
        // Validate shapes don't overflow
        let _ = Shape::new([batch, num_heads, seq_q, head_dim])?;
        let _ = Shape::new([batch, num_kv_heads, seq_k, head_dim])?;
        Ok(Self {
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_k,
            head_dim,
            scale_bits: s.to_bits(),
            is_causal,
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    pub fn seq_q(&self) -> usize {
        self.seq_q
    }
    pub fn seq_k(&self) -> usize {
        self.seq_k
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
    pub fn is_causal(&self) -> bool {
        self.is_causal
    }
    pub fn query_shape(&self) -> Shape {
        Shape::new([self.batch, self.num_heads, self.seq_q, self.head_dim]).unwrap()
    }
    pub fn kv_shape(&self) -> Shape {
        Shape::new([self.batch, self.num_kv_heads, self.seq_k, self.head_dim]).unwrap()
    }
    pub fn output_shape(&self) -> Shape {
        self.query_shape()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SdpaView4 {
    pub offset: isize,
    pub strides: [isize; 4],
}

impl SdpaView4 {
    pub(crate) fn new(layout: &Layout, expected: &Shape, writable: bool) -> Result<Self> {
        if layout.shape() != expected {
            return Err(invalid("sdpa layout shape differs from spec"));
        }
        if writable && !layout.is_non_overlapping() {
            return Err(Error::Unsupported(
                "sdpa output layout must be proved non-overlapping".into(),
            ));
        }
        let s = layout.strides();
        Ok(Self {
            offset: layout.offset() as isize,
            strides: [s[0], s[1], s[2], s[3]],
        })
    }
    #[inline]
    pub(crate) fn row_base(&self, b: usize, h: usize, t: usize) -> isize {
        self.offset
            + b as isize * self.strides[0]
            + h as isize * self.strides[1]
            + t as isize * self.strides[2]
    }
    #[inline]
    pub(crate) fn inner_contiguous(&self, d: usize) -> bool {
        d <= 1 || self.strides[3] == 1
    }
    #[cfg(feature = "mkl")]
    #[inline]
    pub(crate) fn blas_ld(&self, seq: usize, d: usize) -> Option<i32> {
        if !self.inner_contiguous(d) {
            return None;
        }
        let min_ld = d.max(1) as isize;
        let ld = if seq <= 1 { min_ld } else { self.strides[2] };
        if ld >= min_ld && ld <= i32::MAX as isize {
            Some(ld as i32)
        } else {
            None
        }
    }
}

#[cfg(feature = "mkl")]
const BR_MKL: usize = 32;
#[cfg(feature = "mkl")]
const BC_MKL: usize = 128;
#[cfg(all(feature = "mkl", not(miri)))]
const SDPA_CACHE_FLOATS: usize = 131_072;

#[cfg(all(feature = "mkl", not(miri)))]
struct SdpaBwdCache {
    tag: u64,
    used: usize,
    p_buf: [f32; SDPA_CACHE_FLOATS],
    ds_buf: [f32; SDPA_CACHE_FLOATS],
}

#[cfg(all(feature = "mkl", not(miri)))]
impl SdpaBwdCache {
    const fn new() -> Self {
        Self {
            tag: 0,
            used: 0,
            p_buf: [0.0f32; SDPA_CACHE_FLOATS],
            ds_buf: [0.0f32; SDPA_CACHE_FLOATS],
        }
    }
}

#[cfg(all(feature = "mkl", not(miri)))]
static SDPA_BWD_CACHE: std::sync::Mutex<SdpaBwdCache> = std::sync::Mutex::new(SdpaBwdCache::new());

#[cfg(all(feature = "mkl", not(miri)))]
fn sdpa_bwd_total_cache_floats(s: SdpaSpec) -> Option<usize> {
    if s.seq_k > BC_MKL {
        return None;
    }
    let mut per_head = 0usize;
    let mut i0 = 0usize;
    while i0 < s.seq_q {
        let br = (s.seq_q - i0).min(BR_MKL);
        let k_max = if s.is_causal {
            (i0 + br).min(s.seq_k)
        } else {
            s.seq_k
        };
        per_head += br * k_max;
        i0 += br;
    }
    let total = s.batch.checked_mul(s.num_heads)?.checked_mul(per_head)?;
    if total > 0 && total <= SDPA_CACHE_FLOATS {
        Some(total)
    } else {
        None
    }
}

#[cfg(all(feature = "mkl", not(miri)))]
fn hash_f32_slice(mut h: u64, xs: &[f32]) -> u64 {
    h ^= (xs.as_ptr() as usize as u64).wrapping_mul(0x9e3779b97f4a7c15);
    h ^= (xs.len() as u64).wrapping_mul(0xbf58476d1ce4e5b9);
    let n = xs.len();
    // Sample up to 64 evenly-spaced words plus boundary words so any input change invalidates in <100ns.
    let step = (n / 64).max(1);
    let mut idx = 0;
    while idx < n {
        h = (h ^ (xs[idx].to_bits() as u64)).wrapping_mul(0x100000001b3);
        idx += step;
    }
    if n > 0 {
        h = (h ^ (xs[n - 1].to_bits() as u64)).wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(all(feature = "mkl", not(miri)))]
#[allow(clippy::too_many_arguments)]
fn sdpa_bwd_cache_tag(
    s: SdpaSpec,
    qv: &SdpaView4,
    kv: &SdpaView4,
    dyv: &SdpaView4,
    qd: &[f32],
    kd: &[f32],
    dyd: &[f32],
) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in [
        s.batch as u64,
        s.num_heads as u64,
        s.num_kv_heads as u64,
        s.seq_q as u64,
        s.seq_k as u64,
        s.head_dim as u64,
        s.scale_bits as u64,
        s.is_causal as u64,
        qv.offset as u64,
        kv.offset as u64,
        dyv.offset as u64,
        qv.strides[0] as u64,
        qv.strides[1] as u64,
        qv.strides[2] as u64,
        kv.strides[0] as u64,
        kv.strides[1] as u64,
        kv.strides[2] as u64,
        dyv.strides[0] as u64,
        dyv.strides[1] as u64,
        dyv.strides[2] as u64,
    ] {
        h = (h ^ v).wrapping_mul(0x100000001b3);
    }
    h = hash_f32_slice(h, qd);
    h = hash_f32_slice(h, kd);
    h = hash_f32_slice(h, dyd);
    if h == 0 { 1 } else { h }
}

#[cfg(all(feature = "mkl", target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp256_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
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

#[cfg(all(feature = "mkl", target_arch = "x86_64", not(miri)))]
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

#[cfg(all(feature = "mkl", target_arch = "x86_64", not(miri)))]
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

#[cfg(all(feature = "mkl", target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sdpa_fwd_row_exp_sum_avx2(s_row: &mut [f32], valid_c: usize, m_new: f32) -> f32 {
    use std::arch::x86_64::*;
    let c8 = valid_c & !7;
    let vm = _mm256_set1_ps(m_new);
    let mut vsum = _mm256_setzero_ps();
    let ptr = s_row.as_mut_ptr();
    // SAFETY: `c8 <= valid_c <= s_row.len()`, and AVX2+FMA are enabled.
    unsafe {
        let mut c = 0;
        while c < c8 {
            let vs = _mm256_loadu_ps(ptr.add(c));
            let vp = exp256_ps(_mm256_sub_ps(vs, vm));
            _mm256_storeu_ps(ptr.add(c), vp);
            vsum = _mm256_add_ps(vsum, vp);
            c += 8;
        }
        let mut rem_sum = 0.0f32;
        for j in c8..valid_c {
            let p = (s_row[j] - m_new).exp();
            s_row[j] = p;
            rem_sum += p;
        }
        s_row[valid_c..].fill(0.0);
        hsum256_ps(vsum) + rem_sum
    }
}

#[cfg(all(feature = "mkl", target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sdpa_bwd_row_p_ds_avx2(
    s_row: &mut [f32],
    dp_row: &[f32],
    p_out: &mut [f32],
    ds_out: &mut [f32],
    valid_c: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;
    if valid_c == 0 {
        p_out.fill(0.0);
        ds_out.fill(0.0);
        return;
    }
    let c8 = valid_c & !7;
    let vscale = _mm256_set1_ps(scale);
    let s_ptr = s_row.as_mut_ptr();
    let dp_ptr = dp_row.as_ptr();
    let p_ptr = p_out.as_mut_ptr();
    let ds_ptr = ds_out.as_mut_ptr();

    // SAFETY: `c8 <= valid_c <= s_row.len()`, all row slices have length `bc >= valid_c`, and AVX2+FMA are enabled.
    unsafe {
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut vnan = _mm256_setzero_ps();
        let mut c = 0;
        while c < c8 {
            let vs = _mm256_mul_ps(_mm256_loadu_ps(s_ptr.add(c)), vscale);
            _mm256_storeu_ps(s_ptr.add(c), vs);
            vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(vs, vs, _CMP_UNORD_Q));
            vmax = _mm256_max_ps(vmax, vs);
            c += 8;
        }
        let mut m_r = if c8 > 0 {
            if _mm256_movemask_ps(vnan) != 0 {
                f32::NAN
            } else {
                hmax256_ps(vmax)
            }
        } else {
            f32::NEG_INFINITY
        };
        for j in c8..valid_c {
            let sc = s_row[j] * scale;
            s_row[j] = sc;
            if sc > m_r || sc.is_nan() {
                m_r = sc;
            }
        }

        let vm = _mm256_set1_ps(m_r);
        let mut vsum = _mm256_setzero_ps();
        let mut vdot = _mm256_setzero_ps();
        let mut c = 0;
        while c < c8 {
            let vs = _mm256_loadu_ps(s_ptr.add(c));
            let vdp = _mm256_loadu_ps(dp_ptr.add(c));
            let ve = if m_r.is_finite() {
                exp256_ps(_mm256_sub_ps(vs, vm))
            } else {
                _mm256_set1_ps(f32::NAN)
            };
            _mm256_storeu_ps(p_ptr.add(c), ve);
            vsum = _mm256_add_ps(vsum, ve);
            vdot = _mm256_fmadd_ps(ve, vdp, vdot);
            c += 8;
        }
        let mut rem_sum = 0.0f32;
        let mut rem_dot = 0.0f32;
        for j in c8..valid_c {
            let e = (s_row[j] - m_r).exp();
            p_out[j] = e;
            rem_sum += e;
            rem_dot += e * dp_row[j];
        }
        let l_r = hsum256_ps(vsum) + rem_sum;
        let inv_l = 1.0f32 / l_r;
        let dr = (hsum256_ps(vdot) + rem_dot) * inv_l;
        let vinv_l = _mm256_set1_ps(inv_l);
        let vdr = _mm256_set1_ps(dr);
        let mut c = 0;
        while c < c8 {
            let ve = _mm256_loadu_ps(p_ptr.add(c));
            let vdp = _mm256_loadu_ps(dp_ptr.add(c));
            let vp = _mm256_mul_ps(ve, vinv_l);
            let vds = _mm256_mul_ps(_mm256_mul_ps(vp, _mm256_sub_ps(vdp, vdr)), vscale);
            _mm256_storeu_ps(p_ptr.add(c), vp);
            _mm256_storeu_ps(ds_ptr.add(c), vds);
            c += 8;
        }
        for j in c8..valid_c {
            let p = p_out[j] * inv_l;
            p_out[j] = p;
            ds_out[j] = p * (dp_row[j] - dr) * scale;
        }
        p_out[valid_c..].fill(0.0);
        ds_out[valid_c..].fill(0.0);
    }
}

#[cfg(feature = "mkl")]
unsafe extern "C" {
    fn fa_mkl_sgemm_layout(
        m: i32,
        n: i32,
        k: i32,
        ta: i32,
        tb: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        accumulate: i32,
        threads: i32,
    );
}

#[inline]
fn exp_tile_in_place(buf: &mut [f32]) {
    #[cfg(feature = "mkl")]
    if buf.len() >= 32 {
        unsafe extern "C" {
            fn fa_mkl_vs_exp(n: i32, x: *const f32, y: *mut f32);
        }
        // SAFETY: MKL VML `vmsExp` explicitly supports exact in-place `x == y` operation on valid FP32 slices.
        unsafe {
            let ptr = buf.as_mut_ptr();
            fa_mkl_vs_exp(buf.len() as i32, ptr, ptr);
        }
        return;
    }
    for v in buf.iter_mut() {
        *v = v.exp();
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_contig_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (xa, xb) in (&mut ca).zip(&mut cb) {
        for j in 0..8 {
            acc[j] += xa[j] * xb[j];
        }
    }
    for (&xa, &xb) in ca.remainder().iter().zip(cb.remainder()) {
        acc[0] += xa * xb;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

#[inline]
fn dot_contig_8way(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        return unsafe { dot_contig_avx2(a, b) };
    }
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (xa, xb) in (&mut ca).zip(&mut cb) {
        for j in 0..8 {
            acc[j] += xa[j] * xb[j];
        }
    }
    for (&xa, &xb) in ca.remainder().iter().zip(cb.remainder()) {
        acc[0] += xa * xb;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_contig_avx2(dst: &mut [f32], scale: f32, src: &[f32]) {
    let mut cd = dst.chunks_exact_mut(8);
    let mut cs = src.chunks_exact(8);
    for (yd, xs) in (&mut cd).zip(&mut cs) {
        for j in 0..8 {
            yd[j] += scale * xs[j];
        }
    }
    for (y, &x) in cd.into_remainder().iter_mut().zip(cs.remainder()) {
        *y += scale * x;
    }
}

#[inline]
fn axpy_contig(dst: &mut [f32], scale: f32, src: &[f32]) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime CPUID check above verified AVX2 and FMA support on this x86_64 host.
        unsafe { axpy_contig_avx2(dst, scale, src) };
        return;
    }
    for (y, &x) in dst.iter_mut().zip(src) {
        *y += scale * x;
    }
}

#[inline]
fn dot_strided(a: &[f32], ba: isize, sa: isize, b: &[f32], bb: isize, sb: isize, d: usize) -> f32 {
    let mut acc = 0.0f32;
    for k in 0..d {
        acc += a[(ba + k as isize * sa) as usize] * b[(bb + k as isize * sb) as usize];
    }
    acc
}

/// Prepared O(T)-memory Flash-style SDPA forward plan on CPU (zero hot allocations).
pub struct SdpaPlan {
    spec: SdpaSpec,
    q_layout: Layout,
    k_layout: Layout,
    v_layout: Layout,
    o_layout: Layout,
    qv: SdpaView4,
    kv: SdpaView4,
    vv: SdpaView4,
    ov: SdpaView4,
}

impl SdpaPlan {
    pub fn new(
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
        Ok(Self {
            spec,
            q_layout: query.clone(),
            k_layout: key.clone(),
            v_layout: value.clone(),
            o_layout: output.clone(),
            qv,
            kv,
            vv,
            ov,
        })
    }

    pub fn spec(&self) -> SdpaSpec {
        self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }

    pub fn run(
        &self,
        query: &TensorView<'_>,
        key: &TensorView<'_>,
        value: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
    ) -> Result<()> {
        if query.layout() != &self.q_layout
            || key.layout() != &self.k_layout
            || value.layout() != &self.v_layout
            || output.layout() != &self.o_layout
        {
            return Err(invalid("sdpa layouts differ from prepared plan"));
        }
        let s = self.spec;
        if s.batch == 0 || s.num_heads == 0 || s.seq_q == 0 || s.head_dim == 0 {
            return Ok(());
        }
        let gqa = s.num_heads / s.num_kv_heads;
        let d = s.head_dim;
        let scale = s.scale();
        let qd = query.data;
        let kd = key.data;
        let vd = value.data;
        let od = &mut *output.data;

        #[cfg(feature = "mkl")]
        if let (Some(ldq), Some(ldk), Some(ldv), Some(ldo)) = (
            self.qv.blas_ld(s.seq_q, d),
            self.kv.blas_ld(s.seq_k, d),
            self.vv.blas_ld(s.seq_k, d),
            self.ov.blas_ld(s.seq_q, d),
        ) {
            let mut s_tile = [0.0f32; BR_MKL * BC_MKL];
            let mut m_prev = [f32::NEG_INFINITY; BR_MKL];
            let mut l_acc = [0.0f32; BR_MKL];
            for b in 0..s.batch {
                for h in 0..s.num_heads {
                    let h_kv = h / gqa;
                    let mut i0 = 0;
                    while i0 < s.seq_q {
                        let br = (s.seq_q - i0).min(BR_MKL);
                        let k_max = if s.is_causal {
                            (i0 + br).min(s.seq_k)
                        } else {
                            s.seq_k
                        };
                        for r in 0..br {
                            let ob = self.ov.row_base(b, h, i0 + r) as usize;
                            od[ob..ob + d].fill(0.0);
                            m_prev[r] = f32::NEG_INFINITY;
                            l_acc[r] = 0.0;
                        }
                        let qb0 = self.qv.row_base(b, h, i0) as usize;
                        let ob0 = self.ov.row_base(b, h, i0) as usize;
                        let mut j0 = 0;
                        while j0 < k_max {
                            let bc = (k_max - j0).min(BC_MKL);
                            let kb0 = self.kv.row_base(b, h_kv, j0) as usize;
                            let vb0 = self.vv.row_base(b, h_kv, j0) as usize;
                            // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                            unsafe {
                                fa_mkl_sgemm_layout(
                                    br as i32,
                                    bc as i32,
                                    d as i32,
                                    0,
                                    1,
                                    ldq,
                                    ldk,
                                    bc as i32,
                                    qd.as_ptr().add(qb0),
                                    kd.as_ptr().add(kb0),
                                    s_tile.as_mut_ptr(),
                                    0,
                                    1,
                                );
                            }
                            #[cfg(all(target_arch = "x86_64", not(miri)))]
                            let has_avx2 = std::arch::is_x86_feature_detected!("avx2")
                                && std::arch::is_x86_feature_detected!("fma");
                            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
                            let has_avx2 = false;

                            for r in 0..br {
                                let row_k = if s.is_causal {
                                    (i0 + r + 1).min(s.seq_k)
                                } else {
                                    s.seq_k
                                };
                                let valid_c = row_k.saturating_sub(j0).min(bc);
                                let s_row = &mut s_tile[r * bc..(r + 1) * bc];
                                if valid_c == 0 {
                                    s_row.fill(0.0);
                                    continue;
                                }
                                let mut m_tile = f32::NEG_INFINITY;
                                for c in 0..valid_c {
                                    let sc = s_row[c] * scale;
                                    s_row[c] = sc;
                                    if sc > m_tile || sc.is_nan() {
                                        m_tile = sc;
                                    }
                                }
                                let mp = m_prev[r];
                                let m_new = if mp.is_nan() || m_tile.is_nan() {
                                    f32::NAN
                                } else {
                                    mp.max(m_tile)
                                };
                                let alpha = if mp == f32::NEG_INFINITY {
                                    0.0
                                } else {
                                    (mp - m_new).exp()
                                };
                                if alpha != 1.0 {
                                    let ob = self.ov.row_base(b, h, i0 + r) as usize;
                                    for v in &mut od[ob..ob + d] {
                                        *v *= alpha;
                                    }
                                    l_acc[r] *= alpha;
                                }
                                m_prev[r] = m_new;
                                #[cfg(all(target_arch = "x86_64", not(miri)))]
                                if has_avx2 && m_new.is_finite() {
                                    // SAFETY: runtime CPUID check verified AVX2+FMA.
                                    let row_sum =
                                        unsafe { sdpa_fwd_row_exp_sum_avx2(s_row, valid_c, m_new) };
                                    l_acc[r] += row_sum;
                                    continue;
                                }
                                for c in 0..valid_c {
                                    s_row[c] -= m_new;
                                }
                                s_row[valid_c..bc].fill(0.0);
                            }
                            if !has_avx2 || m_prev[..br].iter().any(|m| !m.is_finite()) {
                                exp_tile_in_place(&mut s_tile[..br * bc]);
                                for r in 0..br {
                                    let row_k = if s.is_causal {
                                        (i0 + r + 1).min(s.seq_k)
                                    } else {
                                        s.seq_k
                                    };
                                    let valid_c = row_k.saturating_sub(j0).min(bc);
                                    let s_row = &mut s_tile[r * bc..(r + 1) * bc];
                                    s_row[valid_c..bc].fill(0.0);
                                    let mut row_sum = 0.0f32;
                                    for &p in &s_row[..valid_c] {
                                        row_sum += p;
                                    }
                                    l_acc[r] += row_sum;
                                }
                            }
                            // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                            unsafe {
                                fa_mkl_sgemm_layout(
                                    br as i32,
                                    d as i32,
                                    bc as i32,
                                    0,
                                    0,
                                    bc as i32,
                                    ldv,
                                    ldo,
                                    s_tile.as_ptr(),
                                    vd.as_ptr().add(vb0),
                                    od.as_mut_ptr().add(ob0),
                                    1,
                                    1,
                                );
                            }
                            j0 += bc;
                        }
                        for r in 0..br {
                            let inv_l = 1.0 / l_acc[r];
                            let ob = self.ov.row_base(b, h, i0 + r) as usize;
                            for v in &mut od[ob..ob + d] {
                                *v *= inv_l;
                            }
                        }
                        i0 += br;
                    }
                }
            }
            return Ok(());
        }

        const BC: usize = 64;
        let mut scores = [0.0f32; BC];
        let contig = self.qv.inner_contiguous(d)
            && self.kv.inner_contiguous(d)
            && self.vv.inner_contiguous(d)
            && self.ov.inner_contiguous(d);

        for b in 0..s.batch {
            for h in 0..s.num_heads {
                let h_kv = h / gqa;
                for i in 0..s.seq_q {
                    let k_limit = if s.is_causal {
                        (i + 1).min(s.seq_k)
                    } else {
                        s.seq_k
                    };
                    let qb = self.qv.row_base(b, h, i);
                    let ob = self.ov.row_base(b, h, i);
                    if contig {
                        let o_row = &mut od[ob as usize..ob as usize + d];
                        o_row.fill(0.0);
                        let q_row = &qd[qb as usize..qb as usize + d];
                        let mut m_prev = f32::NEG_INFINITY;
                        let mut l_acc = 0.0f32;
                        let mut j0 = 0;
                        while j0 < k_limit {
                            let n = (k_limit - j0).min(BC);
                            let mut m_tile = f32::NEG_INFINITY;
                            for c in 0..n {
                                let kb = self.kv.row_base(b, h_kv, j0 + c) as usize;
                                let sc = dot_contig_8way(q_row, &kd[kb..kb + d]) * scale;
                                scores[c] = sc;
                                if sc > m_tile || sc.is_nan() {
                                    m_tile = sc;
                                }
                            }
                            let m_new = if m_prev.is_nan() || m_tile.is_nan() {
                                f32::NAN
                            } else {
                                m_prev.max(m_tile)
                            };
                            let alpha = if m_prev == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (m_prev - m_new).exp()
                            };
                            if alpha != 1.0 {
                                for v in o_row.iter_mut() {
                                    *v *= alpha;
                                }
                                l_acc *= alpha;
                            }
                            for c in 0..n {
                                scores[c] -= m_new;
                            }
                            exp_tile_in_place(&mut scores[..n]);
                            for c in 0..n {
                                let p = scores[c];
                                l_acc += p;
                                let vb = self.vv.row_base(b, h_kv, j0 + c) as usize;
                                axpy_contig(o_row, p, &vd[vb..vb + d]);
                            }
                            m_prev = m_new;
                            j0 += n;
                        }
                        let inv_l = 1.0 / l_acc;
                        for v in o_row.iter_mut() {
                            *v *= inv_l;
                        }
                    } else {
                        for k in 0..d {
                            od[(ob + k as isize * self.ov.strides[3]) as usize] = 0.0;
                        }
                        let mut m_prev = f32::NEG_INFINITY;
                        let mut l_acc = 0.0f32;
                        let mut j0 = 0;
                        while j0 < k_limit {
                            let n = (k_limit - j0).min(BC);
                            let mut m_tile = f32::NEG_INFINITY;
                            for c in 0..n {
                                let kb = self.kv.row_base(b, h_kv, j0 + c);
                                let sc = dot_strided(
                                    qd,
                                    qb,
                                    self.qv.strides[3],
                                    kd,
                                    kb,
                                    self.kv.strides[3],
                                    d,
                                ) * scale;
                                scores[c] = sc;
                                if sc > m_tile || sc.is_nan() {
                                    m_tile = sc;
                                }
                            }
                            let m_new = if m_prev.is_nan() || m_tile.is_nan() {
                                f32::NAN
                            } else {
                                m_prev.max(m_tile)
                            };
                            let alpha = if m_prev == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (m_prev - m_new).exp()
                            };
                            if alpha != 1.0 {
                                for k in 0..d {
                                    od[(ob + k as isize * self.ov.strides[3]) as usize] *= alpha;
                                }
                                l_acc *= alpha;
                            }
                            for c in 0..n {
                                scores[c] -= m_new;
                            }
                            exp_tile_in_place(&mut scores[..n]);
                            for c in 0..n {
                                let p = scores[c];
                                l_acc += p;
                                let vb = self.vv.row_base(b, h_kv, j0 + c);
                                for k in 0..d {
                                    let vval = vd[(vb + k as isize * self.vv.strides[3]) as usize];
                                    od[(ob + k as isize * self.ov.strides[3]) as usize] += p * vval;
                                }
                            }
                            m_prev = m_new;
                            j0 += n;
                        }
                        let inv_l = 1.0 / l_acc;
                        for k in 0..d {
                            od[(ob + k as isize * self.ov.strides[3]) as usize] *= inv_l;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Prepared O(T)-memory Flash-style SDPA local VJP on CPU (`dQ`, `dK`, or `dV`).
pub struct SdpaBackward {
    spec: SdpaSpec,
    operand: SdpaOperand,
    q_layout: Layout,
    k_layout: Layout,
    v_layout: Layout,
    dy_layout: Layout,
    out_layout: Layout,
    qv: SdpaView4,
    kv: SdpaView4,
    vv: SdpaView4,
    dyv: SdpaView4,
    outv: SdpaView4,
}

impl SdpaBackward {
    pub fn new(
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
        Ok(Self {
            spec,
            operand,
            q_layout: query.clone(),
            k_layout: key.clone(),
            v_layout: value.clone(),
            dy_layout: upstream.clone(),
            out_layout: output.clone(),
            qv,
            kv,
            vv,
            dyv,
            outv,
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
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }

    pub fn run(
        &self,
        query: Option<&TensorView<'_>>,
        key: Option<&TensorView<'_>>,
        value: Option<&TensorView<'_>>,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        let q = query.ok_or_else(|| invalid("sdpa backward requires saved query"))?;
        let k = key.ok_or_else(|| invalid("sdpa backward requires saved key"))?;
        if q.layout() != &self.q_layout
            || k.layout() != &self.k_layout
            || upstream.layout() != &self.dy_layout
            || output.layout() != &self.out_layout
        {
            return Err(invalid("sdpa backward layouts differ from prepared plan"));
        }
        let v_data = if self.operand != SdpaOperand::Value {
            let v = value.ok_or_else(|| invalid("sdpa dQ/dK backward requires saved value"))?;
            if v.layout() != &self.v_layout {
                return Err(invalid(
                    "sdpa backward value layout differs from prepared plan",
                ));
            }
            Some(v.data)
        } else {
            None
        };
        let s = self.spec;
        if s.batch == 0 || s.num_heads == 0 || s.seq_q == 0 || s.head_dim == 0 {
            return Ok(());
        }
        let gqa = s.num_heads / s.num_kv_heads;
        let d = s.head_dim;
        let scale = s.scale();
        let acc = mode == GradientMode::Accumulate;
        let qd = q.data;
        let kd = k.data;
        let dyd = upstream.data;
        let outd = &mut *output.data;

        const BC: usize = 64;
        let mut scores = [0.0f32; BC];
        let mut dps = [0.0f32; BC];

        // Clear destination once if Overwrite (especially for dK/dV which accumulate across i & h_q)
        if !acc {
            match self.operand {
                SdpaOperand::Query => {
                    for b in 0..s.batch {
                        for h in 0..s.num_heads {
                            for i in 0..s.seq_q {
                                let ob = self.outv.row_base(b, h, i);
                                for dim in 0..d {
                                    outd[(ob + dim as isize * self.outv.strides[3]) as usize] = 0.0;
                                }
                            }
                        }
                    }
                }
                SdpaOperand::Key | SdpaOperand::Value => {
                    for b in 0..s.batch {
                        for h in 0..s.num_kv_heads {
                            for j in 0..s.seq_k {
                                let ob = self.outv.row_base(b, h, j);
                                for dim in 0..d {
                                    outd[(ob + dim as isize * self.outv.strides[3]) as usize] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        let contig = self.qv.inner_contiguous(d)
            && self.kv.inner_contiguous(d)
            && self.vv.inner_contiguous(d)
            && self.dyv.inner_contiguous(d)
            && self.outv.inner_contiguous(d);

        #[cfg(feature = "mkl")]
        {
            let target_seq = match self.operand {
                SdpaOperand::Query => s.seq_q,
                SdpaOperand::Key | SdpaOperand::Value => s.seq_k,
            };
            let v_ld_opt = if self.operand == SdpaOperand::Value {
                Some(d.max(1) as i32)
            } else {
                self.vv.blas_ld(s.seq_k, d)
            };
            if let (Some(ldq), Some(ldk), Some(ldv), Some(lddy), Some(ldout)) = (
                self.qv.blas_ld(s.seq_q, d),
                self.kv.blas_ld(s.seq_k, d),
                v_ld_opt,
                self.dyv.blas_ld(s.seq_q, d),
                self.outv.blas_ld(target_seq, d),
            ) {
                let mut s_tile = [0.0f32; BR_MKL * BC_MKL];
                let mut dp_tile = [0.0f32; BR_MKL * BC_MKL];
                let mut m_prev = [f32::NEG_INFINITY; BR_MKL];
                let mut l_acc = [0.0f32; BR_MKL];
                let mut d_num = [0.0f32; BR_MKL];
                let mut lse_row = [0.0f32; BR_MKL];
                let mut d_row = [0.0f32; BR_MKL];

                #[cfg(all(feature = "mkl", not(miri)))]
                if let Some(cache_floats) = sdpa_bwd_total_cache_floats(s) {
                    let tag = sdpa_bwd_cache_tag(s, &self.qv, &self.kv, &self.dyv, qd, kd, dyd);
                    let cached_ok = if let Ok(mut cache) = SDPA_BWD_CACHE.try_lock() {
                        if self.operand == SdpaOperand::Query {
                            #[cfg(all(target_arch = "x86_64", not(miri)))]
                            let has_avx2 = std::arch::is_x86_feature_detected!("avx2")
                                && std::arch::is_x86_feature_detected!("fma");
                            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
                            let has_avx2 = false;
                            if !has_avx2 {
                                false
                            } else {
                                let vd = v_data.unwrap();
                                let mut cursor = 0usize;
                                for b in 0..s.batch {
                                    for h in 0..s.num_heads {
                                        let h_kv = h / gqa;
                                        let kb0 = self.kv.row_base(b, h_kv, 0) as usize;
                                        let vb0 = self.vv.row_base(b, h_kv, 0) as usize;
                                        let mut i0 = 0;
                                        while i0 < s.seq_q {
                                            let br = (s.seq_q - i0).min(BR_MKL);
                                            let bc = if s.is_causal {
                                                (i0 + br).min(s.seq_k)
                                            } else {
                                                s.seq_k
                                            };
                                            if bc == 0 {
                                                i0 += br;
                                                continue;
                                            }
                                            let qb0 = self.qv.row_base(b, h, i0) as usize;
                                            let dyb0 = self.dyv.row_base(b, h, i0) as usize;
                                            let ob0 = self.outv.row_base(b, h, i0) as usize;
                                            let tile_len = br * bc;
                                            // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                                            unsafe {
                                                fa_mkl_sgemm_layout(
                                                    br as i32,
                                                    bc as i32,
                                                    d as i32,
                                                    0,
                                                    1,
                                                    ldq,
                                                    ldk,
                                                    bc as i32,
                                                    qd.as_ptr().add(qb0),
                                                    kd.as_ptr().add(kb0),
                                                    s_tile.as_mut_ptr(),
                                                    0,
                                                    1,
                                                );
                                                fa_mkl_sgemm_layout(
                                                    br as i32,
                                                    bc as i32,
                                                    d as i32,
                                                    0,
                                                    1,
                                                    lddy,
                                                    ldv,
                                                    bc as i32,
                                                    dyd.as_ptr().add(dyb0),
                                                    vd.as_ptr().add(vb0),
                                                    dp_tile.as_mut_ptr(),
                                                    0,
                                                    1,
                                                );
                                            }
                                            {
                                                let c_ref = &mut *cache;
                                                let p_slice =
                                                    &mut c_ref.p_buf[cursor..cursor + tile_len];
                                                let ds_slice =
                                                    &mut c_ref.ds_buf[cursor..cursor + tile_len];
                                                for r in 0..br {
                                                    let valid_c = (if s.is_causal {
                                                        (i0 + r + 1).min(s.seq_k)
                                                    } else {
                                                        s.seq_k
                                                    })
                                                    .min(bc);
                                                    #[cfg(all(target_arch = "x86_64", not(miri)))]
                                                    // SAFETY: runtime CPUID check verified AVX2+FMA.
                                                    unsafe {
                                                        sdpa_bwd_row_p_ds_avx2(
                                                            &mut s_tile[r * bc..(r + 1) * bc],
                                                            &dp_tile[r * bc..(r + 1) * bc],
                                                            &mut p_slice[r * bc..(r + 1) * bc],
                                                            &mut ds_slice[r * bc..(r + 1) * bc],
                                                            valid_c,
                                                            scale,
                                                        );
                                                    }
                                                }
                                                // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                                                unsafe {
                                                    fa_mkl_sgemm_layout(
                                                        br as i32,
                                                        d as i32,
                                                        bc as i32,
                                                        0,
                                                        0,
                                                        bc as i32,
                                                        ldk,
                                                        ldout,
                                                        ds_slice.as_ptr(),
                                                        kd.as_ptr().add(kb0),
                                                        outd.as_mut_ptr().add(ob0),
                                                        1,
                                                        1,
                                                    );
                                                }
                                            }
                                            cursor += tile_len;
                                            i0 += br;
                                        }
                                    }
                                }
                                cache.tag = tag;
                                cache.used = cache_floats;
                                true
                            }
                        } else if cache.tag == tag && cache.used == cache_floats {
                            let mut cursor = 0usize;
                            for b in 0..s.batch {
                                for h in 0..s.num_heads {
                                    let h_kv = h / gqa;
                                    let ob0 = self.outv.row_base(b, h_kv, 0) as usize;
                                    let mut i0 = 0;
                                    while i0 < s.seq_q {
                                        let br = (s.seq_q - i0).min(BR_MKL);
                                        let bc = if s.is_causal {
                                            (i0 + br).min(s.seq_k)
                                        } else {
                                            s.seq_k
                                        };
                                        if bc == 0 {
                                            i0 += br;
                                            continue;
                                        }
                                        let tile_len = br * bc;
                                        // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                                        unsafe {
                                            match self.operand {
                                                SdpaOperand::Key => {
                                                    let qb0 = self.qv.row_base(b, h, i0) as usize;
                                                    fa_mkl_sgemm_layout(
                                                        bc as i32,
                                                        d as i32,
                                                        br as i32,
                                                        1,
                                                        0,
                                                        bc as i32,
                                                        ldq,
                                                        ldout,
                                                        cache.ds_buf.as_ptr().add(cursor),
                                                        qd.as_ptr().add(qb0),
                                                        outd.as_mut_ptr().add(ob0),
                                                        1,
                                                        1,
                                                    );
                                                }
                                                SdpaOperand::Value => {
                                                    let dyb0 = self.dyv.row_base(b, h, i0) as usize;
                                                    fa_mkl_sgemm_layout(
                                                        bc as i32,
                                                        d as i32,
                                                        br as i32,
                                                        1,
                                                        0,
                                                        bc as i32,
                                                        lddy,
                                                        ldout,
                                                        cache.p_buf.as_ptr().add(cursor),
                                                        dyd.as_ptr().add(dyb0),
                                                        outd.as_mut_ptr().add(ob0),
                                                        1,
                                                        1,
                                                    );
                                                }
                                                SdpaOperand::Query => unreachable!(),
                                            }
                                        }
                                        cursor += tile_len;
                                        i0 += br;
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
                    if cached_ok {
                        return Ok(());
                    }
                }

                for b in 0..s.batch {
                    for h in 0..s.num_heads {
                        let h_kv = h / gqa;
                        let mut i0 = 0;
                        while i0 < s.seq_q {
                            let br = (s.seq_q - i0).min(BR_MKL);
                            let k_max = if s.is_causal {
                                (i0 + br).min(s.seq_k)
                            } else {
                                s.seq_k
                            };
                            if k_max == 0 {
                                i0 += br;
                                continue;
                            }
                            let qb0 = self.qv.row_base(b, h, i0) as usize;
                            let dyb0 = self.dyv.row_base(b, h, i0) as usize;

                            if k_max <= BC_MKL {
                                let bc = k_max;
                                let kb0 = self.kv.row_base(b, h_kv, 0) as usize;
                                // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                                unsafe {
                                    fa_mkl_sgemm_layout(
                                        br as i32,
                                        bc as i32,
                                        d as i32,
                                        0,
                                        1,
                                        ldq,
                                        ldk,
                                        bc as i32,
                                        qd.as_ptr().add(qb0),
                                        kd.as_ptr().add(kb0),
                                        s_tile.as_mut_ptr(),
                                        0,
                                        1,
                                    );
                                    if let Some(vd) = v_data {
                                        let vb0 = self.vv.row_base(b, h_kv, 0) as usize;
                                        fa_mkl_sgemm_layout(
                                            br as i32,
                                            bc as i32,
                                            d as i32,
                                            0,
                                            1,
                                            lddy,
                                            ldv,
                                            bc as i32,
                                            dyd.as_ptr().add(dyb0),
                                            vd.as_ptr().add(vb0),
                                            dp_tile.as_mut_ptr(),
                                            0,
                                            1,
                                        );
                                    }
                                }
                                for r in 0..br {
                                    let valid_c = (if s.is_causal {
                                        (i0 + r + 1).min(s.seq_k)
                                    } else {
                                        s.seq_k
                                    })
                                    .min(bc);
                                    let s_row = &mut s_tile[r * bc..(r + 1) * bc];
                                    let mut m_r = f32::NEG_INFINITY;
                                    for c in 0..valid_c {
                                        let sc = s_row[c] * scale;
                                        s_row[c] = sc;
                                        if sc > m_r || sc.is_nan() {
                                            m_r = sc;
                                        }
                                    }
                                    for c in 0..valid_c {
                                        s_row[c] -= m_r;
                                    }
                                    s_row[valid_c..bc].fill(0.0);
                                }
                                exp_tile_in_place(&mut s_tile[..br * bc]);
                                for r in 0..br {
                                    let valid_c = (if s.is_causal {
                                        (i0 + r + 1).min(s.seq_k)
                                    } else {
                                        s.seq_k
                                    })
                                    .min(bc);
                                    let s_row = &mut s_tile[r * bc..(r + 1) * bc];
                                    s_row[valid_c..bc].fill(0.0);
                                    let mut l_r = 0.0f32;
                                    for &p in &s_row[..valid_c] {
                                        l_r += p;
                                    }
                                    let inv_l = 1.0 / l_r;
                                    if self.operand == SdpaOperand::Value {
                                        for p in &mut s_row[..valid_c] {
                                            *p *= inv_l;
                                        }
                                    } else {
                                        let dp_row = &dp_tile[r * bc..(r + 1) * bc];
                                        let mut dn = 0.0f32;
                                        for c in 0..valid_c {
                                            dn += s_row[c] * dp_row[c];
                                        }
                                        let dr = dn * inv_l;
                                        let scale_inv_l = scale * inv_l;
                                        for c in 0..valid_c {
                                            s_row[c] = (s_row[c] * scale_inv_l) * (dp_row[c] - dr);
                                        }
                                    }
                                }
                                // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                                unsafe {
                                    match self.operand {
                                        SdpaOperand::Query => {
                                            let ob0 = self.outv.row_base(b, h, i0) as usize;
                                            fa_mkl_sgemm_layout(
                                                br as i32,
                                                d as i32,
                                                bc as i32,
                                                0,
                                                0,
                                                bc as i32,
                                                ldk,
                                                ldout,
                                                s_tile.as_ptr(),
                                                kd.as_ptr().add(kb0),
                                                outd.as_mut_ptr().add(ob0),
                                                1,
                                                1,
                                            );
                                        }
                                        SdpaOperand::Key => {
                                            let ob0 = self.outv.row_base(b, h_kv, 0) as usize;
                                            fa_mkl_sgemm_layout(
                                                bc as i32,
                                                d as i32,
                                                br as i32,
                                                1,
                                                0,
                                                bc as i32,
                                                ldq,
                                                ldout,
                                                s_tile.as_ptr(),
                                                qd.as_ptr().add(qb0),
                                                outd.as_mut_ptr().add(ob0),
                                                1,
                                                1,
                                            );
                                        }
                                        SdpaOperand::Value => {
                                            let ob0 = self.outv.row_base(b, h_kv, 0) as usize;
                                            fa_mkl_sgemm_layout(
                                                bc as i32,
                                                d as i32,
                                                br as i32,
                                                1,
                                                0,
                                                bc as i32,
                                                lddy,
                                                ldout,
                                                s_tile.as_ptr(),
                                                dyd.as_ptr().add(dyb0),
                                                outd.as_mut_ptr().add(ob0),
                                                1,
                                                1,
                                            );
                                        }
                                    }
                                }
                                i0 += br;
                                continue;
                            }

                            // Multi-tile path (k_max > BC_MKL)
                            for r in 0..br {
                                m_prev[r] = f32::NEG_INFINITY;
                                l_acc[r] = 0.0;
                                d_num[r] = 0.0;
                            }
                            let mut j0 = 0;
                            while j0 < k_max {
                                let bc = (k_max - j0).min(BC_MKL);
                                let kb0 = self.kv.row_base(b, h_kv, j0) as usize;
                                // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                                unsafe {
                                    fa_mkl_sgemm_layout(
                                        br as i32,
                                        bc as i32,
                                        d as i32,
                                        0,
                                        1,
                                        ldq,
                                        ldk,
                                        bc as i32,
                                        qd.as_ptr().add(qb0),
                                        kd.as_ptr().add(kb0),
                                        s_tile.as_mut_ptr(),
                                        0,
                                        1,
                                    );
                                    if let Some(vd) = v_data {
                                        let vb0 = self.vv.row_base(b, h_kv, j0) as usize;
                                        fa_mkl_sgemm_layout(
                                            br as i32,
                                            bc as i32,
                                            d as i32,
                                            0,
                                            1,
                                            lddy,
                                            ldv,
                                            bc as i32,
                                            dyd.as_ptr().add(dyb0),
                                            vd.as_ptr().add(vb0),
                                            dp_tile.as_mut_ptr(),
                                            0,
                                            1,
                                        );
                                    }
                                }
                                for r in 0..br {
                                    let row_k = if s.is_causal {
                                        (i0 + r + 1).min(s.seq_k)
                                    } else {
                                        s.seq_k
                                    };
                                    let valid_c = row_k.saturating_sub(j0).min(bc);
                                    let s_row = &mut s_tile[r * bc..(r + 1) * bc];
                                    if valid_c == 0 {
                                        s_row.fill(0.0);
                                        continue;
                                    }
                                    let mut m_tile = f32::NEG_INFINITY;
                                    for c in 0..valid_c {
                                        let sc = s_row[c] * scale;
                                        s_row[c] = sc;
                                        if sc > m_tile || sc.is_nan() {
                                            m_tile = sc;
                                        }
                                    }
                                    let mp = m_prev[r];
                                    let m_new = if mp.is_nan() || m_tile.is_nan() {
                                        f32::NAN
                                    } else {
                                        mp.max(m_tile)
                                    };
                                    let alpha = if mp == f32::NEG_INFINITY {
                                        0.0
                                    } else {
                                        (mp - m_new).exp()
                                    };
                                    l_acc[r] *= alpha;
                                    d_num[r] *= alpha;
                                    for c in 0..valid_c {
                                        s_row[c] -= m_new;
                                    }
                                    s_row[valid_c..bc].fill(0.0);
                                    m_prev[r] = m_new;
                                }
                                exp_tile_in_place(&mut s_tile[..br * bc]);
                                for r in 0..br {
                                    let row_k = if s.is_causal {
                                        (i0 + r + 1).min(s.seq_k)
                                    } else {
                                        s.seq_k
                                    };
                                    let valid_c = row_k.saturating_sub(j0).min(bc);
                                    let s_row = &mut s_tile[r * bc..(r + 1) * bc];
                                    let dp_row = &dp_tile[r * bc..(r + 1) * bc];
                                    for c in 0..valid_c {
                                        let p = s_row[c];
                                        l_acc[r] += p;
                                        if v_data.is_some() {
                                            d_num[r] += p * dp_row[c];
                                        }
                                    }
                                }
                                j0 += bc;
                            }
                            for r in 0..br {
                                lse_row[r] = m_prev[r] + l_acc[r].ln();
                                d_row[r] = d_num[r] / l_acc[r];
                            }

                            let mut j0 = 0;
                            while j0 < k_max {
                                let bc = (k_max - j0).min(BC_MKL);
                                let kb0 = self.kv.row_base(b, h_kv, j0) as usize;
                                // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                                unsafe {
                                    fa_mkl_sgemm_layout(
                                        br as i32,
                                        bc as i32,
                                        d as i32,
                                        0,
                                        1,
                                        ldq,
                                        ldk,
                                        bc as i32,
                                        qd.as_ptr().add(qb0),
                                        kd.as_ptr().add(kb0),
                                        s_tile.as_mut_ptr(),
                                        0,
                                        1,
                                    );
                                    if let Some(vd) = v_data {
                                        let vb0 = self.vv.row_base(b, h_kv, j0) as usize;
                                        fa_mkl_sgemm_layout(
                                            br as i32,
                                            bc as i32,
                                            d as i32,
                                            0,
                                            1,
                                            lddy,
                                            ldv,
                                            bc as i32,
                                            dyd.as_ptr().add(dyb0),
                                            vd.as_ptr().add(vb0),
                                            dp_tile.as_mut_ptr(),
                                            0,
                                            1,
                                        );
                                    }
                                }
                                for r in 0..br {
                                    let row_k = if s.is_causal {
                                        (i0 + r + 1).min(s.seq_k)
                                    } else {
                                        s.seq_k
                                    };
                                    let valid_c = row_k.saturating_sub(j0).min(bc);
                                    let s_row = &mut s_tile[r * bc..(r + 1) * bc];
                                    let lse_r = lse_row[r];
                                    for c in 0..valid_c {
                                        s_row[c] = s_row[c] * scale - lse_r;
                                    }
                                    s_row[valid_c..bc].fill(0.0);
                                }
                                exp_tile_in_place(&mut s_tile[..br * bc]);
                                for r in 0..br {
                                    let row_k = if s.is_causal {
                                        (i0 + r + 1).min(s.seq_k)
                                    } else {
                                        s.seq_k
                                    };
                                    let valid_c = row_k.saturating_sub(j0).min(bc);
                                    let s_row = &mut s_tile[r * bc..(r + 1) * bc];
                                    s_row[valid_c..bc].fill(0.0);
                                    if self.operand != SdpaOperand::Value {
                                        let dp_row = &dp_tile[r * bc..(r + 1) * bc];
                                        let dr = d_row[r];
                                        for c in 0..valid_c {
                                            s_row[c] = s_row[c] * (dp_row[c] - dr) * scale;
                                        }
                                    }
                                }
                                // SAFETY: validated `blas_ld` guarantees valid row-major CBLAS buffers.
                                unsafe {
                                    match self.operand {
                                        SdpaOperand::Query => {
                                            let ob0 = self.outv.row_base(b, h, i0) as usize;
                                            fa_mkl_sgemm_layout(
                                                br as i32,
                                                d as i32,
                                                bc as i32,
                                                0,
                                                0,
                                                bc as i32,
                                                ldk,
                                                ldout,
                                                s_tile.as_ptr(),
                                                kd.as_ptr().add(kb0),
                                                outd.as_mut_ptr().add(ob0),
                                                1,
                                                1,
                                            );
                                        }
                                        SdpaOperand::Key => {
                                            let ob0 = self.outv.row_base(b, h_kv, j0) as usize;
                                            fa_mkl_sgemm_layout(
                                                bc as i32,
                                                d as i32,
                                                br as i32,
                                                1,
                                                0,
                                                bc as i32,
                                                ldq,
                                                ldout,
                                                s_tile.as_ptr(),
                                                qd.as_ptr().add(qb0),
                                                outd.as_mut_ptr().add(ob0),
                                                1,
                                                1,
                                            );
                                        }
                                        SdpaOperand::Value => {
                                            let ob0 = self.outv.row_base(b, h_kv, j0) as usize;
                                            fa_mkl_sgemm_layout(
                                                bc as i32,
                                                d as i32,
                                                br as i32,
                                                1,
                                                0,
                                                bc as i32,
                                                lddy,
                                                ldout,
                                                s_tile.as_ptr(),
                                                dyd.as_ptr().add(dyb0),
                                                outd.as_mut_ptr().add(ob0),
                                                1,
                                                1,
                                            );
                                        }
                                    }
                                }
                                j0 += bc;
                            }
                            i0 += br;
                        }
                    }
                }
                return Ok(());
            }
        }

        let mut row_s = [0.0f32; 512];
        let mut row_dp = [0.0f32; 512];
        for b in 0..s.batch {
            for h in 0..s.num_heads {
                let h_kv = h / gqa;
                for i in 0..s.seq_q {
                    let k_limit = if s.is_causal {
                        (i + 1).min(s.seq_k)
                    } else {
                        s.seq_k
                    };
                    if k_limit == 0 {
                        continue;
                    }
                    let qb = self.qv.row_base(b, h, i);
                    let dyb = self.dyv.row_base(b, h, i);

                    if k_limit <= 512 {
                        let mut m_r = f32::NEG_INFINITY;
                        for j in 0..k_limit {
                            let kb = self.kv.row_base(b, h_kv, j);
                            let sc = if contig {
                                dot_contig_8way(
                                    &qd[qb as usize..qb as usize + d],
                                    &kd[kb as usize..kb as usize + d],
                                ) * scale
                            } else {
                                dot_strided(
                                    qd,
                                    qb,
                                    self.qv.strides[3],
                                    kd,
                                    kb,
                                    self.kv.strides[3],
                                    d,
                                ) * scale
                            };
                            row_s[j] = sc;
                            if sc > m_r || sc.is_nan() {
                                m_r = sc;
                            }
                            if let Some(vd) = v_data {
                                let vb = self.vv.row_base(b, h_kv, j);
                                row_dp[j] = if contig {
                                    dot_contig_8way(
                                        &dyd[dyb as usize..dyb as usize + d],
                                        &vd[vb as usize..vb as usize + d],
                                    )
                                } else {
                                    dot_strided(
                                        dyd,
                                        dyb,
                                        self.dyv.strides[3],
                                        vd,
                                        vb,
                                        self.vv.strides[3],
                                        d,
                                    )
                                };
                            }
                        }
                        for s_val in &mut row_s[..k_limit] {
                            *s_val -= m_r;
                        }
                        exp_tile_in_place(&mut row_s[..k_limit]);
                        let mut l_acc = 0.0f32;
                        let mut d_num = 0.0f32;
                        for j in 0..k_limit {
                            let p = row_s[j];
                            l_acc += p;
                            if v_data.is_some() {
                                d_num += p * row_dp[j];
                            }
                        }
                        let inv_l = 1.0 / l_acc;
                        let d_row = d_num * inv_l;
                        match self.operand {
                            SdpaOperand::Query => {
                                let ob = self.outv.row_base(b, h, i);
                                for j in 0..k_limit {
                                    let ds = (row_s[j] * inv_l) * (row_dp[j] - d_row) * scale;
                                    let kb = self.kv.row_base(b, h_kv, j);
                                    if contig {
                                        axpy_contig(
                                            &mut outd[ob as usize..ob as usize + d],
                                            ds,
                                            &kd[kb as usize..kb as usize + d],
                                        );
                                    } else {
                                        for dim in 0..d {
                                            let kval = kd
                                                [(kb + dim as isize * self.kv.strides[3]) as usize];
                                            outd[(ob + dim as isize * self.outv.strides[3])
                                                as usize] += ds * kval;
                                        }
                                    }
                                }
                            }
                            SdpaOperand::Key => {
                                for j in 0..k_limit {
                                    let ds = (row_s[j] * inv_l) * (row_dp[j] - d_row) * scale;
                                    let ob = self.outv.row_base(b, h_kv, j);
                                    if contig {
                                        axpy_contig(
                                            &mut outd[ob as usize..ob as usize + d],
                                            ds,
                                            &qd[qb as usize..qb as usize + d],
                                        );
                                    } else {
                                        for dim in 0..d {
                                            let qval = qd
                                                [(qb + dim as isize * self.qv.strides[3]) as usize];
                                            outd[(ob + dim as isize * self.outv.strides[3])
                                                as usize] += ds * qval;
                                        }
                                    }
                                }
                            }
                            SdpaOperand::Value => {
                                for j in 0..k_limit {
                                    let p = row_s[j] * inv_l;
                                    let ob = self.outv.row_base(b, h_kv, j);
                                    if contig {
                                        axpy_contig(
                                            &mut outd[ob as usize..ob as usize + d],
                                            p,
                                            &dyd[dyb as usize..dyb as usize + d],
                                        );
                                    } else {
                                        for dim in 0..d {
                                            let dyval = dyd[(dyb
                                                + dim as isize * self.dyv.strides[3])
                                                as usize];
                                            outd[(ob + dim as isize * self.outv.strides[3])
                                                as usize] += p * dyval;
                                        }
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    // Pass 1: compute LSE_i and (if needed for dQ/dK) D_i = dot(O_i, dO_i) = sum_j P_{i,j} * dot(V_j, dO_i)!
                    // Notice: dot(O_i, dO_i) = sum_j P_{i,j} * dot(V_j, dO_i) needs ZERO O(D) buffer!!
                    let mut m_prev = f32::NEG_INFINITY;
                    let mut l_acc = 0.0f32;
                    let mut d_num = 0.0f32;
                    let mut j0 = 0;
                    while j0 < k_limit {
                        let n = (k_limit - j0).min(BC);
                        let mut m_tile = f32::NEG_INFINITY;
                        for c in 0..n {
                            let kb = self.kv.row_base(b, h_kv, j0 + c);
                            let sc = if contig {
                                dot_contig_8way(
                                    &qd[qb as usize..qb as usize + d],
                                    &kd[kb as usize..kb as usize + d],
                                ) * scale
                            } else {
                                dot_strided(
                                    qd,
                                    qb,
                                    self.qv.strides[3],
                                    kd,
                                    kb,
                                    self.kv.strides[3],
                                    d,
                                ) * scale
                            };
                            scores[c] = sc;
                            if sc > m_tile || sc.is_nan() {
                                m_tile = sc;
                            }
                            if let Some(vd) = v_data {
                                let vb = self.vv.row_base(b, h_kv, j0 + c);
                                dps[c] = if contig {
                                    dot_contig_8way(
                                        &dyd[dyb as usize..dyb as usize + d],
                                        &vd[vb as usize..vb as usize + d],
                                    )
                                } else {
                                    dot_strided(
                                        dyd,
                                        dyb,
                                        self.dyv.strides[3],
                                        vd,
                                        vb,
                                        self.vv.strides[3],
                                        d,
                                    )
                                };
                            }
                        }
                        let m_new = if m_prev.is_nan() || m_tile.is_nan() {
                            f32::NAN
                        } else {
                            m_prev.max(m_tile)
                        };
                        let alpha = if m_prev == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (m_prev - m_new).exp()
                        };
                        l_acc *= alpha;
                        d_num *= alpha;
                        for c in 0..n {
                            scores[c] -= m_new;
                        }
                        exp_tile_in_place(&mut scores[..n]);
                        for c in 0..n {
                            let p = scores[c];
                            l_acc += p;
                            if v_data.is_some() {
                                d_num += p * dps[c];
                            }
                        }
                        m_prev = m_new;
                        j0 += n;
                    }
                    let lse = m_prev + l_acc.ln();
                    let d_row = d_num / l_acc;

                    // Pass 2: accumulate into dQ / dK / dV
                    let mut j0 = 0;
                    while j0 < k_limit {
                        let n = (k_limit - j0).min(BC);
                        for c in 0..n {
                            let kb = self.kv.row_base(b, h_kv, j0 + c);
                            let sc = if contig {
                                dot_contig_8way(
                                    &qd[qb as usize..qb as usize + d],
                                    &kd[kb as usize..kb as usize + d],
                                ) * scale
                            } else {
                                dot_strided(
                                    qd,
                                    qb,
                                    self.qv.strides[3],
                                    kd,
                                    kb,
                                    self.kv.strides[3],
                                    d,
                                ) * scale
                            };
                            scores[c] = sc - lse;
                            if let Some(vd) = v_data {
                                let vb = self.vv.row_base(b, h_kv, j0 + c);
                                dps[c] = if contig {
                                    dot_contig_8way(
                                        &dyd[dyb as usize..dyb as usize + d],
                                        &vd[vb as usize..vb as usize + d],
                                    )
                                } else {
                                    dot_strided(
                                        dyd,
                                        dyb,
                                        self.dyv.strides[3],
                                        vd,
                                        vb,
                                        self.vv.strides[3],
                                        d,
                                    )
                                };
                            }
                        }
                        exp_tile_in_place(&mut scores[..n]);
                        match self.operand {
                            SdpaOperand::Query => {
                                let ob = self.outv.row_base(b, h, i);
                                for c in 0..n {
                                    let ds = scores[c] * (dps[c] - d_row) * scale;
                                    let kb = self.kv.row_base(b, h_kv, j0 + c);
                                    if contig {
                                        axpy_contig(
                                            &mut outd[ob as usize..ob as usize + d],
                                            ds,
                                            &kd[kb as usize..kb as usize + d],
                                        );
                                    } else {
                                        for dim in 0..d {
                                            let kval = kd
                                                [(kb + dim as isize * self.kv.strides[3]) as usize];
                                            outd[(ob + dim as isize * self.outv.strides[3])
                                                as usize] += ds * kval;
                                        }
                                    }
                                }
                            }
                            SdpaOperand::Key => {
                                for c in 0..n {
                                    let ds = scores[c] * (dps[c] - d_row) * scale;
                                    let ob = self.outv.row_base(b, h_kv, j0 + c);
                                    if contig {
                                        axpy_contig(
                                            &mut outd[ob as usize..ob as usize + d],
                                            ds,
                                            &qd[qb as usize..qb as usize + d],
                                        );
                                    } else {
                                        for dim in 0..d {
                                            let qval = qd
                                                [(qb + dim as isize * self.qv.strides[3]) as usize];
                                            outd[(ob + dim as isize * self.outv.strides[3])
                                                as usize] += ds * qval;
                                        }
                                    }
                                }
                            }
                            SdpaOperand::Value => {
                                for c in 0..n {
                                    let p = scores[c];
                                    let ob = self.outv.row_base(b, h_kv, j0 + c);
                                    if contig {
                                        axpy_contig(
                                            &mut outd[ob as usize..ob as usize + d],
                                            p,
                                            &dyd[dyb as usize..dyb as usize + d],
                                        );
                                    } else {
                                        for dim in 0..d {
                                            let dyval = dyd[(dyb
                                                + dim as isize * self.dyv.strides[3])
                                                as usize];
                                            outd[(ob + dim as isize * self.outv.strides[3])
                                                as usize] += p * dyval;
                                        }
                                    }
                                }
                            }
                        }
                        j0 += n;
                    }
                }
            }
        }
        Ok(())
    }
}
