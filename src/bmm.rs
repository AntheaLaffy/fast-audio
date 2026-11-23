use crate::{
    BinaryOperand, Error, GradientMode, Layout, Result, Shape, TensorView, TensorViewMut, cpu::Cpu,
    error::invalid,
};

/// Validated geometry for broadcast batched matrix multiplication `[..., M, K] @ [..., K, N]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BmmSpec {
    left: Shape,
    right: Shape,
    output: Shape,
    batch_dims: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
}

impl BmmSpec {
    pub fn new(left: &Shape, right: &Shape) -> Result<Self> {
        let ld = left.dims();
        let rd = right.dims();
        if ld.len() < 2 || rd.len() < 2 {
            return Err(invalid("batched matmul requires rank >= 2 inputs"));
        }
        let m = ld[ld.len() - 2];
        let k = ld[ld.len() - 1];
        let rk = rd[rd.len() - 2];
        let n = rd[rd.len() - 1];
        if k != rk {
            return Err(invalid(
                "batched matmul inner contraction dimensions differ",
            ));
        }
        let lb = &ld[..ld.len() - 2];
        let rb = &rd[..rd.len() - 2];
        let brank = lb.len().max(rb.len());
        let mut batch_dims = Vec::with_capacity(brank);
        for i in 0..brank {
            let da = if i + lb.len() >= brank {
                lb[i + lb.len() - brank]
            } else {
                1
            };
            let db = if i + rb.len() >= brank {
                rb[i + rb.len() - brank]
            } else {
                1
            };
            if da == db {
                batch_dims.push(da);
            } else if da == 1 {
                batch_dims.push(db);
            } else if db == 1 {
                batch_dims.push(da);
            } else {
                return Err(invalid(
                    "batched matmul batch dimensions are not broadcastable",
                ));
            }
        }
        let mut out_dims = batch_dims.clone();
        out_dims.push(m);
        out_dims.push(n);
        let output = Shape::new(out_dims)?;
        Ok(Self {
            left: left.clone(),
            right: right.clone(),
            output,
            batch_dims,
            m,
            k,
            n,
        })
    }

    pub fn left_shape(&self) -> &Shape {
        &self.left
    }
    pub fn right_shape(&self) -> &Shape {
        &self.right
    }
    pub fn output_shape(&self) -> &Shape {
        &self.output
    }
    pub fn batch_dims(&self) -> &[usize] {
        &self.batch_dims
    }
    pub fn m(&self) -> usize {
        self.m
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        size_of::<Self>()
            + (self.left.dims().len()
                + self.right.dims().len()
                + self.output.dims().len()
                + self.batch_dims.len())
                * size_of::<usize>()
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(not(any(feature = "mkl", feature = "cuda")), allow(dead_code))]
pub(crate) struct Mat2d {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
}

#[cfg_attr(not(any(feature = "mkl", feature = "cuda")), allow(dead_code))]
impl Mat2d {
    pub fn row_ld(self) -> Option<i32> {
        if self.cols > 1 && self.col_stride != 1 {
            return None;
        }
        let ld = if self.rows <= 1 {
            self.cols.max(1) as isize
        } else {
            self.row_stride
        };
        if ld <= 0 || (ld as usize) < self.cols.max(1) {
            return None;
        }
        i32::try_from(ld).ok()
    }
    pub fn transposed(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }
    #[cfg(any(feature = "mkl", feature = "cuda"))]
    pub fn blas(self) -> Option<(i32, i32)> {
        self.row_ld()
            .map(|ld| (0, ld))
            .or_else(|| self.transposed().row_ld().map(|ld| (1, ld)))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BatchAxis {
    pub dim: usize,
    pub stride_a: isize,
    pub stride_b: isize,
    pub stride_c: isize,
}

#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct BlasRoute {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub ta: i32,
    pub tb: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub swapped: bool,
    pub collapsed_2d: bool,
    pub strided_batched: bool,
    pub batch_count: i32,
    pub batch_stride_a: i64,
    pub batch_stride_b: i64,
    pub batch_stride_c: i64,
}

/// Unified batched matrix-multiply contraction descriptor:
/// For each output batch `g in 0..out_batches`:
///   C[g] (+)= sum_{r in 0..red_batches} A[g, r] @ B[g, r]
/// (with `red_batches == 1` for forward and non-broadcast backward).
pub(crate) struct BmmKernelDesc {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub out_batches: usize,
    pub red_batches: usize,
    pub origins: [isize; 3],
    pub mat_a: Mat2d,
    pub mat_b: Mat2d,
    pub mat_c: Mat2d,
    pub out_axes: Vec<BatchAxis>,
    pub red_axes: Vec<BatchAxis>,
    #[cfg_attr(not(any(feature = "mkl", feature = "cuda")), allow(dead_code))]
    pub blas: Option<BlasRoute>,
}

fn aligned_batch_strides(layout: &Layout, batch_rank: usize) -> Vec<isize> {
    let dims = layout.shape().dims();
    let strides = layout.strides();
    let b_len = dims.len() - 2;
    let mut out = vec![0isize; batch_rank];
    let pad = batch_rank - b_len;
    for i in 0..b_len {
        out[pad + i] = if dims[i] == 1 { 0 } else { strides[i] };
    }
    out
}

#[cfg(any(feature = "mkl", feature = "cuda"))]
fn flatten_single_stride(axes: &[BatchAxis], which: usize) -> Option<i64> {
    if axes.is_empty() {
        return Some(0);
    }
    let get = |a: &BatchAxis| -> i64 {
        match which {
            0 => a.stride_a as i64,
            1 => a.stride_b as i64,
            _ => a.stride_c as i64,
        }
    };
    // All-zero broadcast stride flattens cleanly to stride 0.
    if axes.iter().all(|a| a.dim <= 1 || get(a) == 0) {
        return Some(0);
    }
    let mut expected: Option<i64> = None;
    let mut base: Option<i64> = None;
    for a in axes.iter().rev() {
        if a.dim <= 1 {
            continue;
        }
        let s = get(a);
        if let Some(exp) = expected {
            if s != exp {
                return None;
            }
            expected = exp.checked_mul(a.dim as i64);
        } else {
            base = Some(s);
            expected = s.checked_mul(a.dim as i64);
        }
        expected?;
    }
    base
}

impl BmmKernelDesc {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        m: usize,
        n: usize,
        k: usize,
        origins: [isize; 3],
        mat_a: Mat2d,
        mat_b: Mat2d,
        mat_c: Mat2d,
        out_axes: Vec<BatchAxis>,
        red_axes: Vec<BatchAxis>,
    ) -> Self {
        let out_batches: usize = out_axes.iter().map(|a| a.dim).product();
        let red_batches: usize = red_axes.iter().map(|a| a.dim).product();

        #[cfg(any(feature = "mkl", feature = "cuda"))]
        let blas = Self::detect_blas(
            m,
            n,
            k,
            out_batches,
            red_batches,
            mat_a,
            mat_b,
            mat_c,
            &out_axes,
            &red_axes,
        );
        #[cfg(not(any(feature = "mkl", feature = "cuda")))]
        let blas = None;

        Self {
            m,
            n,
            k,
            out_batches,
            red_batches,
            origins,
            mat_a,
            mat_b,
            mat_c,
            out_axes,
            red_axes,
            blas,
        }
    }

    #[cfg(any(feature = "mkl", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    fn detect_blas(
        m: usize,
        n: usize,
        k: usize,
        out_batches: usize,
        red_batches: usize,
        mat_a: Mat2d,
        mat_b: Mat2d,
        mat_c: Mat2d,
        out_axes: &[BatchAxis],
        red_axes: &[BatchAxis],
    ) -> Option<BlasRoute> {
        if m == 0 || n == 0 || k == 0 || out_batches == 0 || red_batches == 0 {
            return None;
        }
        // Check Collapsed 2D case 1 (batched linear forward & dX):
        // red_batches == 1, B is broadcast across all out_axes (stride_b == 0),
        // A and C are row-major with flat contiguous batch stride == m * row_stride.
        if red_batches == 1 && out_batches > 1 {
            if let (Some(sa), Some(sb), Some(sc)) = (
                flatten_single_stride(out_axes, 0),
                flatten_single_stride(out_axes, 1),
                flatten_single_stride(out_axes, 2),
            ) {
                if sb == 0 {
                    if let (Some(lda), Some((tb, ldb)), Some(ldc)) =
                        (mat_a.row_ld(), mat_b.blas(), mat_c.row_ld())
                    {
                        if sa == (m as i64) * (lda as i64) && sc == (m as i64) * (ldc as i64) {
                            if let (Ok(cm), Ok(cn), Ok(ck)) = (
                                i32::try_from(out_batches.checked_mul(m)?),
                                i32::try_from(n),
                                i32::try_from(k),
                            ) {
                                return Some(BlasRoute {
                                    m: cm,
                                    n: cn,
                                    k: ck,
                                    ta: 0,
                                    tb,
                                    lda,
                                    ldb,
                                    ldc,
                                    swapped: false,
                                    collapsed_2d: true,
                                    strided_batched: false,
                                    batch_count: 1,
                                    batch_stride_a: 0,
                                    batch_stride_b: 0,
                                    batch_stride_c: 0,
                                });
                            }
                        }
                    }
                }
            }
        }
        // Check Collapsed 2D case 2 (batched linear dW: C[M, N] = sum_r A_r[M, K] @ B_r[K, N]):
        // out_batches == 1, red_batches > 1, A is transposed row-major [K, M]^T (col_stride == M*...)
        // i.e. A has row_stride == 1, col_stride == lda_t (A^T is [K, M] contiguous across r!)
        // and B is row-major [K, N] contiguous across r!
        if out_batches == 1 && red_batches > 1 {
            if let (Some(sa), Some(sb)) = (
                flatten_single_stride(red_axes, 0),
                flatten_single_stride(red_axes, 1),
            ) {
                if let (Some(lda_t), Some(ldb), Some(ldc)) =
                    (mat_a.transposed().row_ld(), mat_b.row_ld(), mat_c.row_ld())
                {
                    if sa == (k as i64) * (lda_t as i64) && sb == (k as i64) * (ldb as i64) {
                        if let (Ok(cm), Ok(cn), Ok(ck)) = (
                            i32::try_from(m),
                            i32::try_from(n),
                            i32::try_from(red_batches.checked_mul(k)?),
                        ) {
                            return Some(BlasRoute {
                                m: cm,
                                n: cn,
                                k: ck,
                                ta: 1,
                                tb: 0,
                                lda: lda_t,
                                ldb,
                                ldc,
                                swapped: false,
                                collapsed_2d: true,
                                strided_batched: false,
                                batch_count: 1,
                                batch_stride_a: 0,
                                batch_stride_b: 0,
                                batch_stride_c: 0,
                            });
                        }
                    }
                }
            }
        }

        // Standard per-slice or strided-batched BLAS check.
        let (swapped, ma, mb, mc) = if mat_c.row_ld().is_some() {
            (false, mat_a, mat_b, mat_c)
        } else if mat_c.transposed().row_ld().is_some() {
            (
                true,
                mat_b.transposed(),
                mat_a.transposed(),
                mat_c.transposed(),
            )
        } else {
            return None;
        };
        let (ta, lda) = ma.blas()?;
        let (tb, ldb) = mb.blas()?;
        let ldc = mc.row_ld()?;
        let (bm, bn, bk) = if swapped {
            (
                i32::try_from(n).ok()?,
                i32::try_from(m).ok()?,
                i32::try_from(k).ok()?,
            )
        } else {
            (
                i32::try_from(m).ok()?,
                i32::try_from(n).ok()?,
                i32::try_from(k).ok()?,
            )
        };

        let mut strided_batched = false;
        let mut batch_count = 1i32;
        let (mut bsa, mut bsb, mut bsc) = (0i64, 0i64, 0i64);
        if red_batches == 1 {
            if let (Ok(bc), Some(sa), Some(sb), Some(sc)) = (
                i32::try_from(out_batches),
                flatten_single_stride(out_axes, 0),
                flatten_single_stride(out_axes, 1),
                flatten_single_stride(out_axes, 2),
            ) {
                if bc == 1 || sc > 0 {
                    strided_batched = true;
                    batch_count = bc;
                    if swapped {
                        bsa = sb;
                        bsb = sa;
                        bsc = sc;
                    } else {
                        bsa = sa;
                        bsb = sb;
                        bsc = sc;
                    }
                }
            }
        }

        Some(BlasRoute {
            m: bm,
            n: bn,
            k: bk,
            ta,
            tb,
            lda,
            ldb,
            ldc,
            swapped,
            collapsed_2d: false,
            strided_batched,
            batch_count,
            batch_stride_a: bsa,
            batch_stride_b: bsb,
            batch_stride_c: bsc,
        })
    }

    pub fn batch_offsets(axes: &[BatchAxis], mut idx: usize) -> (isize, isize, isize) {
        let (mut oa, mut ob, mut oc) = (0isize, 0isize, 0isize);
        for a in axes.iter().rev() {
            let c = (idx % a.dim) as isize;
            idx /= a.dim;
            oa += c * a.stride_a;
            ob += c * a.stride_b;
            oc += c * a.stride_c;
        }
        (oa, ob, oc)
    }

    pub fn metadata_bytes(&self) -> usize {
        size_of::<Self>() + (self.out_axes.len() + self.red_axes.len()) * size_of::<BatchAxis>()
    }
}

pub(crate) fn build_forward_desc(
    spec: &BmmSpec,
    left: &Layout,
    right: &Layout,
    output: &Layout,
) -> Result<BmmKernelDesc> {
    if left.shape() != &spec.left || right.shape() != &spec.right || output.shape() != &spec.output
    {
        return Err(invalid("batched matmul layouts differ from spec"));
    }
    if !output.is_non_overlapping() {
        return Err(Error::Unsupported(
            "batched matmul output layout must be proved non-overlapping".into(),
        ));
    }
    let brank = spec.batch_dims.len();
    let sa = aligned_batch_strides(left, brank);
    let sb = aligned_batch_strides(right, brank);
    let sc = aligned_batch_strides(output, brank);
    let mut out_axes = Vec::with_capacity(brank);
    for i in 0..brank {
        if spec.batch_dims[i] != 1 {
            out_axes.push(BatchAxis {
                dim: spec.batch_dims[i],
                stride_a: sa[i],
                stride_b: sb[i],
                stride_c: sc[i],
            });
        }
    }
    let ls = left.strides();
    let rs = right.strides();
    let os = output.strides();
    let mat_a = Mat2d {
        rows: spec.m,
        cols: spec.k,
        row_stride: ls[ls.len() - 2],
        col_stride: ls[ls.len() - 1],
    };
    let mat_b = Mat2d {
        rows: spec.k,
        cols: spec.n,
        row_stride: rs[rs.len() - 2],
        col_stride: rs[rs.len() - 1],
    };
    let mat_c = Mat2d {
        rows: spec.m,
        cols: spec.n,
        row_stride: os[os.len() - 2],
        col_stride: os[os.len() - 1],
    };
    Ok(BmmKernelDesc::build(
        spec.m,
        spec.n,
        spec.k,
        [
            left.offset() as isize,
            right.offset() as isize,
            output.offset() as isize,
        ],
        mat_a,
        mat_b,
        mat_c,
        out_axes,
        Vec::new(),
    ))
}

pub(crate) fn build_backward_desc(
    spec: &BmmSpec,
    operand: BinaryOperand,
    left: &Layout,
    right: &Layout,
    upstream: &Layout,
    output: &Layout,
) -> Result<BmmKernelDesc> {
    if left.shape() != &spec.left
        || right.shape() != &spec.right
        || upstream.shape() != &spec.output
    {
        return Err(invalid("batched matmul backward layouts differ from spec"));
    }
    let target_shape = match operand {
        BinaryOperand::Left => &spec.left,
        BinaryOperand::Right => &spec.right,
    };
    if output.shape() != target_shape {
        return Err(invalid(
            "batched matmul backward destination shape differs from operand shape",
        ));
    }
    if !output.is_non_overlapping() {
        return Err(Error::Unsupported(
            "batched matmul backward output must be proved non-overlapping".into(),
        ));
    }
    let brank = spec.batch_dims.len();
    let su = aligned_batch_strides(upstream, brank);
    let st = aligned_batch_strides(output, brank);
    let other_layout = match operand {
        BinaryOperand::Left => right,
        BinaryOperand::Right => left,
    };
    let so = aligned_batch_strides(other_layout, brank);
    let target_b = &target_shape.dims()[..target_shape.dims().len() - 2];
    let pad = brank - target_b.len();

    // Operand Left: dA[m, k] = sum_red dY[m, n] @ B^T[n, k] -> A_eff = dY, B_eff = B^T
    // Operand Right: dB[k, n] = sum_red A^T[k, m] @ dY[m, n] -> A_eff = A^T, B_eff = dY
    let mut out_axes = Vec::new();
    let mut red_axes = Vec::new();
    for i in 0..brank {
        let dim = spec.batch_dims[i];
        let target_dim = if i >= pad { target_b[i - pad] } else { 1 };
        let (sa_eff, sb_eff) = match operand {
            BinaryOperand::Left => (su[i], so[i]),
            BinaryOperand::Right => (so[i], su[i]),
        };
        if dim == 1 {
            continue;
        }
        if target_dim == dim {
            out_axes.push(BatchAxis {
                dim,
                stride_a: sa_eff,
                stride_b: sb_eff,
                stride_c: st[i],
            });
        } else {
            // Target dimension was 1 (broadcasted): reduce over this batch axis!
            red_axes.push(BatchAxis {
                dim,
                stride_a: sa_eff,
                stride_b: sb_eff,
                stride_c: 0,
            });
        }
    }

    let us = upstream.strides();
    let ts = output.strides();
    let os = other_layout.strides();
    match operand {
        BinaryOperand::Left => {
            let mat_a = Mat2d {
                rows: spec.m,
                cols: spec.n,
                row_stride: us[us.len() - 2],
                col_stride: us[us.len() - 1],
            };
            let mat_b = Mat2d {
                rows: spec.n,
                cols: spec.k,
                row_stride: os[os.len() - 1],
                col_stride: os[os.len() - 2],
            };
            let mat_c = Mat2d {
                rows: spec.m,
                cols: spec.k,
                row_stride: ts[ts.len() - 2],
                col_stride: ts[ts.len() - 1],
            };
            Ok(BmmKernelDesc::build(
                spec.m,
                spec.k,
                spec.n,
                [
                    upstream.offset() as isize,
                    right.offset() as isize,
                    output.offset() as isize,
                ],
                mat_a,
                mat_b,
                mat_c,
                out_axes,
                red_axes,
            ))
        }
        BinaryOperand::Right => {
            let mat_a = Mat2d {
                rows: spec.k,
                cols: spec.m,
                row_stride: os[os.len() - 1],
                col_stride: os[os.len() - 2],
            };
            let mat_b = Mat2d {
                rows: spec.m,
                cols: spec.n,
                row_stride: us[us.len() - 2],
                col_stride: us[us.len() - 1],
            };
            let mat_c = Mat2d {
                rows: spec.k,
                cols: spec.n,
                row_stride: ts[ts.len() - 2],
                col_stride: ts[ts.len() - 1],
            };
            Ok(BmmKernelDesc::build(
                spec.k,
                spec.n,
                spec.m,
                [
                    left.offset() as isize,
                    upstream.offset() as isize,
                    output.offset() as isize,
                ],
                mat_a,
                mat_b,
                mat_c,
                out_axes,
                red_axes,
            ))
        }
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

fn run_cpu_desc(
    cpu: Cpu,
    desc: &BmmKernelDesc,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    accumulate: bool,
) {
    if desc.out_batches == 0 || desc.m == 0 || desc.n == 0 {
        return;
    }
    if desc.k == 0 || desc.red_batches == 0 {
        if !accumulate {
            for g in 0..desc.out_batches {
                let (_, _, oc) = BmmKernelDesc::batch_offsets(&desc.out_axes, g);
                let base_c = desc.origins[2] + oc;
                for i in 0..desc.m {
                    for j in 0..desc.n {
                        let off = (base_c
                            + i as isize * desc.mat_c.row_stride
                            + j as isize * desc.mat_c.col_stride)
                            as usize;
                        c[off] = 0.0;
                    }
                }
            }
        }
        return;
    }

    #[cfg(feature = "mkl")]
    if let (Cpu::Mkl { threads }, Some(blas)) = (cpu, desc.blas) {
        let threads = i32::from(threads.get());
        if blas.collapsed_2d {
            // SAFETY: collapsed_2d verified contiguous strides and valid bounds for the full range.
            unsafe {
                fa_mkl_sgemm_layout(
                    blas.m,
                    blas.n,
                    blas.k,
                    blas.ta,
                    blas.tb,
                    blas.lda,
                    blas.ldb,
                    blas.ldc,
                    a.as_ptr().offset(desc.origins[0]),
                    b.as_ptr().offset(desc.origins[1]),
                    c.as_mut_ptr().offset(desc.origins[2]),
                    i32::from(accumulate),
                    threads,
                );
            }
            return;
        }
        for g in 0..desc.out_batches {
            let (oa_g, ob_g, oc_g) = BmmKernelDesc::batch_offsets(&desc.out_axes, g);
            for r in 0..desc.red_batches {
                let (oa_r, ob_r, _) = BmmKernelDesc::batch_offsets(&desc.red_axes, r);
                let acc = accumulate || r > 0;
                let pa = desc.origins[0] + oa_g + oa_r;
                let pb = desc.origins[1] + ob_g + ob_r;
                let pc = desc.origins[2] + oc_g;
                // SAFETY: all matrix layouts and bounds were validated at preparation/binding time.
                unsafe {
                    if blas.swapped {
                        fa_mkl_sgemm_layout(
                            blas.m,
                            blas.n,
                            blas.k,
                            blas.ta,
                            blas.tb,
                            blas.lda,
                            blas.ldb,
                            blas.ldc,
                            b.as_ptr().offset(pb),
                            a.as_ptr().offset(pa),
                            c.as_mut_ptr().offset(pc),
                            i32::from(acc),
                            threads,
                        );
                    } else {
                        fa_mkl_sgemm_layout(
                            blas.m,
                            blas.n,
                            blas.k,
                            blas.ta,
                            blas.tb,
                            blas.lda,
                            blas.ldb,
                            blas.ldc,
                            a.as_ptr().offset(pa),
                            b.as_ptr().offset(pb),
                            c.as_mut_ptr().offset(pc),
                            i32::from(acc),
                            threads,
                        );
                    }
                }
            }
        }
        return;
    }

    let _ = cpu;
    let fast_row_major = desc.mat_a.col_stride == 1
        && desc.mat_b.col_stride == 1
        && desc.mat_c.col_stride == 1
        && desc.mat_a.row_stride >= 0
        && desc.mat_b.row_stride >= 0
        && desc.mat_c.row_stride >= 0;

    for g in 0..desc.out_batches {
        let (oa_g, ob_g, oc_g) = BmmKernelDesc::batch_offsets(&desc.out_axes, g);
        let base_c = desc.origins[2] + oc_g;
        if !accumulate {
            for i in 0..desc.m {
                let row_c = base_c + i as isize * desc.mat_c.row_stride;
                if fast_row_major {
                    let cs = &mut c[row_c as usize..row_c as usize + desc.n];
                    cs.fill(0.0);
                } else {
                    for j in 0..desc.n {
                        c[(row_c + j as isize * desc.mat_c.col_stride) as usize] = 0.0;
                    }
                }
            }
        }
        for r in 0..desc.red_batches {
            let (oa_r, ob_r, _) = BmmKernelDesc::batch_offsets(&desc.red_axes, r);
            let base_a = desc.origins[0] + oa_g + oa_r;
            let base_b = desc.origins[1] + ob_g + ob_r;
            if fast_row_major {
                // ikj cache-friendly loop (auto-vectorizes inner j to AVX2 FMA).
                for i in 0..desc.m {
                    let row_a = (base_a + i as isize * desc.mat_a.row_stride) as usize;
                    let row_c = (base_c + i as isize * desc.mat_c.row_stride) as usize;
                    let cs = &mut c[row_c..row_c + desc.n];
                    for p in 0..desc.k {
                        let av = a[row_a + p];
                        let row_b = (base_b + p as isize * desc.mat_b.row_stride) as usize;
                        let bs = &b[row_b..row_b + desc.n];
                        for (dst, &bv) in cs.iter_mut().zip(bs) {
                            *dst += av * bv;
                        }
                    }
                }
            } else {
                for i in 0..desc.m {
                    let row_a = base_a + i as isize * desc.mat_a.row_stride;
                    let row_c = base_c + i as isize * desc.mat_c.row_stride;
                    for j in 0..desc.n {
                        let col_b = base_b + j as isize * desc.mat_b.col_stride;
                        let mut sum = 0.0f32;
                        for p in 0..desc.k {
                            let av = a[(row_a + p as isize * desc.mat_a.col_stride) as usize];
                            let bv = b[(col_b + p as isize * desc.mat_b.row_stride) as usize];
                            sum += av * bv;
                        }
                        c[(row_c + j as isize * desc.mat_c.col_stride) as usize] += sum;
                    }
                }
            }
        }
    }
}

/// Prepared arbitrary-rank broadcast batched matrix multiplication on CPU.
pub struct BatchedMatmulPlan {
    cpu: Cpu,
    spec: BmmSpec,
    left_layout: Layout,
    right_layout: Layout,
    output_layout: Layout,
    desc: BmmKernelDesc,
}

impl BatchedMatmulPlan {
    pub fn new(
        cpu: Cpu,
        spec: &BmmSpec,
        left: &Layout,
        right: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let desc = build_forward_desc(spec, left, right, output)?;
        Ok(Self {
            cpu,
            spec: spec.clone(),
            left_layout: left.clone(),
            right_layout: right.clone(),
            output_layout: output.clone(),
            desc,
        })
    }

    pub fn spec(&self) -> &BmmSpec {
        &self.spec
    }
    pub fn left_layout(&self) -> &Layout {
        &self.left_layout
    }
    pub fn right_layout(&self) -> &Layout {
        &self.right_layout
    }
    pub fn output_layout(&self) -> &Layout {
        &self.output_layout
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes() + self.desc.metadata_bytes()
    }

    pub fn run(
        &self,
        left: &TensorView<'_>,
        right: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
    ) -> Result<()> {
        if left.layout() != &self.left_layout
            || right.layout() != &self.right_layout
            || output.layout() != &self.output_layout
        {
            return Err(invalid("batched matmul layouts differ from prepared plan"));
        }
        run_cpu_desc(
            self.cpu,
            &self.desc,
            left.data,
            right.data,
            output.data,
            false,
        );
        Ok(())
    }
}

/// Prepared arbitrary-rank broadcast batched matmul local VJP on CPU.
pub struct BatchedMatmulBackward {
    cpu: Cpu,
    spec: BmmSpec,
    operand: BinaryOperand,
    left_layout: Layout,
    right_layout: Layout,
    upstream_layout: Layout,
    output_layout: Layout,
    desc: BmmKernelDesc,
}

impl BatchedMatmulBackward {
    pub fn new(
        cpu: Cpu,
        operand: BinaryOperand,
        spec: &BmmSpec,
        left: &Layout,
        right: &Layout,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let desc = build_backward_desc(spec, operand, left, right, upstream, output)?;
        Ok(Self {
            cpu,
            spec: spec.clone(),
            operand,
            left_layout: left.clone(),
            right_layout: right.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
            desc,
        })
    }

    pub fn spec(&self) -> &BmmSpec {
        &self.spec
    }
    pub fn operand(&self) -> BinaryOperand {
        self.operand
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        match self.operand {
            BinaryOperand::Left => [false, true],
            BinaryOperand::Right => [true, false],
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes() + self.desc.metadata_bytes()
    }

    pub fn run(
        &self,
        left: Option<&TensorView<'_>>,
        right: Option<&TensorView<'_>>,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        if upstream.layout() != &self.upstream_layout || output.layout() != &self.output_layout {
            return Err(invalid(
                "batched matmul backward layouts differ from prepared plan",
            ));
        }
        let acc = mode == GradientMode::Accumulate;
        match self.operand {
            BinaryOperand::Left => {
                let r = right.ok_or_else(|| {
                    invalid("batched matmul Left backward requires saved right input")
                })?;
                if r.layout() != &self.right_layout {
                    return Err(invalid(
                        "batched matmul backward right layout differs from prepared plan",
                    ));
                }
                run_cpu_desc(
                    self.cpu,
                    &self.desc,
                    upstream.data,
                    r.data,
                    output.data,
                    acc,
                );
            }
            BinaryOperand::Right => {
                let l = left.ok_or_else(|| {
                    invalid("batched matmul Right backward requires saved left input")
                })?;
                if l.layout() != &self.left_layout {
                    return Err(invalid(
                        "batched matmul backward left layout differs from prepared plan",
                    ));
                }
                run_cpu_desc(
                    self.cpu,
                    &self.desc,
                    l.data,
                    upstream.data,
                    output.data,
                    acc,
                );
            }
        }
        Ok(())
    }
}
