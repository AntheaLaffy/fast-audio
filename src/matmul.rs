use crate::{
    BinaryOperand, Error, GradientMode, Layout, Result, TensorView, TensorViewMut, cpu::Cpu,
    error::invalid,
};

/// Prepared dense algebra route; explicit workspace excludes vendor internals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum MatmulStrategy {
    Empty = 0,
    Zero = 1,
    Reference = 2,
    Mkl = 3,
    PackedMkl = 4,
    Cublas = 5,
    StridedCuda = 6,
}
/// Default bound on explicit CPU packing payloads.
pub const MATMUL_WORKSPACE_LIMIT: usize = 64 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub origin: isize,
    pub steps: [isize; 2],
}
impl Matrix {
    fn new(l: &Layout) -> Result<Self> {
        let &[rows, cols] = l.shape().dims() else {
            return Err(invalid("prepared matmul requires rank-two layouts"));
        };
        Ok(Self {
            rows,
            cols,
            origin: l.offset() as isize,
            steps: [l.strides()[0], l.strides()[1]],
        })
    }
    pub fn transposed(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
            steps: [self.steps[1], self.steps[0]],
            ..self
        }
    }
    fn offset(self, i: usize, j: usize) -> usize {
        (self.origin + i as isize * self.steps[0] + j as isize * self.steps[1]) as usize
    }
    fn row_ld(self) -> Option<i32> {
        if self.cols > 1 && self.steps[1] != 1 {
            return None;
        }
        let ld = if self.rows <= 1 {
            self.cols.max(1) as isize
        } else {
            self.steps[0]
        };
        if ld <= 0 || (ld as usize) < self.cols.max(1) {
            return None;
        }
        i32::try_from(ld).ok()
    }
    #[cfg(any(feature = "mkl", feature = "cuda"))]
    fn blas(self) -> Option<(i32, i32)> {
        self.row_ld()
            .map(|ld| (0, ld))
            .or_else(|| self.transposed().row_ld().map(|ld| (1, ld)))
    }
    fn read(self, data: &[f32], i: usize, j: usize) -> f32 {
        // SAFETY: binding checks preserve the original initialized layout. Transposition
        // only exchanges axes; every coordinate here remains in that checked affine map.
        unsafe { *data.get_unchecked(self.offset(i, j)) }
    }
}
#[cfg(any(feature = "mkl", feature = "cuda"))]
#[derive(Clone, Copy)]
pub(crate) struct Blas {
    pub leading: [i32; 3],
    pub transposes: [i32; 2],
}
pub(crate) struct Geometry {
    pub matrices: [Matrix; 3],
    pub swapped: bool,
    #[cfg(any(feature = "mkl", feature = "cuda"))]
    pub blas: Option<Blas>,
}
impl Geometry {
    pub fn new(mut matrices: [Matrix; 3]) -> Result<Self> {
        let [a, b, c] = matrices;
        if a.cols != b.rows || c.rows != a.rows || c.cols != b.cols {
            return Err(invalid("matmul logical dimensions differ"));
        }
        let swapped = c.rows > 0
            && c.cols > 0
            && c.row_ld().is_none()
            && (c.transposed().row_ld().is_some()
                || (c.rows > 1
                    && c.cols > 1
                    && c.steps[0].unsigned_abs() < c.steps[1].unsigned_abs()));
        if swapped {
            matrices = [b.transposed(), a.transposed(), c.transposed()];
        }
        #[cfg(any(feature = "mkl", feature = "cuda"))]
        let blas = (|| {
            let [a, b, c] = matrices;
            if [a.rows, b.cols, a.cols]
                .iter()
                .any(|&n| n == 0 || n > i32::MAX as usize)
            {
                return None;
            }
            let (ta, lda) = a.blas()?;
            let (tb, ldb) = b.blas()?;
            let ldc = c.row_ld()?;
            Some(Blas {
                leading: [lda, ldb, ldc],
                transposes: [ta, tb],
            })
        })();
        Ok(Self {
            matrices,
            swapped,
            #[cfg(any(feature = "mkl", feature = "cuda"))]
            blas,
        })
    }
    pub fn dimensions(&self) -> [usize; 3] {
        let [a, b, _] = self.matrices;
        [a.rows, b.cols, a.cols]
    }
    pub fn empty(&self) -> bool {
        let [m, n, _] = self.dimensions();
        m == 0 || n == 0
    }
    pub fn zero(&self) -> bool {
        self.dimensions()[2] == 0
    }
    pub fn orient<'a>(&self, data: [&'a [f32]; 2]) -> [&'a [f32]; 2] {
        if self.swapped {
            [data[1], data[0]]
        } else {
            data
        }
    }
}
fn layouts_equal(a: &Layout, b: &Layout) -> bool {
    std::ptr::eq(a, b) || a == b
}
fn destination(layout: &Layout) -> Result<()> {
    if !layout.is_non_overlapping() {
        return Err(Error::Unsupported(
            "matmul destination must be proved non-overlapping".into(),
        ));
    }
    Ok(())
}
pub(crate) struct ForwardGeometry {
    pub layouts: [Layout; 3],
    pub core: Geometry,
}
impl ForwardGeometry {
    pub fn new(a: &Layout, b: &Layout, out: &Layout) -> Result<Self> {
        destination(out)?;
        let core = Geometry::new([Matrix::new(a)?, Matrix::new(b)?, Matrix::new(out)?])?;
        Ok(Self {
            layouts: [a.clone(), b.clone(), out.clone()],
            core,
        })
    }
    pub fn validate(&self, a: &Layout, b: &Layout, out: &Layout) -> Result<()> {
        if [a, b, out]
            .into_iter()
            .zip(&self.layouts)
            .all(|(a, b)| layouts_equal(a, b))
        {
            Ok(())
        } else {
            Err(invalid("matmul bindings differ from prepared layouts"))
        }
    }
    pub fn metadata_bytes(&self) -> usize {
        3 * 2 * (size_of::<usize>() + size_of::<isize>())
    }
}
pub(crate) struct BackwardGeometry {
    pub operand: BinaryOperand,
    pub layouts: [Layout; 4],
    pub dependencies: [bool; 2],
    pub core: Geometry,
}
impl BackwardGeometry {
    pub fn new(
        operand: BinaryOperand,
        a: &Layout,
        b: &Layout,
        dy: &Layout,
        gradient: &Layout,
    ) -> Result<Self> {
        let aa = Matrix::new(a)?;
        let bb = Matrix::new(b)?;
        let dd = Matrix::new(dy)?;
        let gg = Matrix::new(gradient)?;
        if aa.cols != bb.rows || dd.rows != aa.rows || dd.cols != bb.cols {
            return Err(invalid(
                "matmul backward upstream/contraction shape mismatch",
            ));
        }
        let target = if operand == BinaryOperand::Left { a } else { b };
        if gradient.shape() != target.shape() {
            return Err(invalid(
                "matmul gradient must match the original operand shape",
            ));
        }
        destination(gradient)?;
        let core = Geometry::new(if operand == BinaryOperand::Left {
            [dd, bb.transposed(), gg]
        } else {
            [aa.transposed(), dd, gg]
        })?;
        let active = !core.empty() && !core.zero();
        let dependencies = [
            active && operand == BinaryOperand::Right,
            active && operand == BinaryOperand::Left,
        ];
        Ok(Self {
            operand,
            layouts: [a.clone(), b.clone(), dy.clone(), gradient.clone()],
            dependencies,
            core,
        })
    }
    pub fn validate(&self, inputs: [Option<&Layout>; 2], dy: &Layout, out: &Layout) -> Result<()> {
        for (i, source) in inputs.into_iter().enumerate() {
            if let Some(source) = source {
                if !layouts_equal(source, &self.layouts[i]) {
                    return Err(invalid("matmul backward supplied input geometry changed"));
                }
            } else if self.dependencies[i] {
                return Err(invalid("matmul backward required payload is missing"));
            }
        }
        if !layouts_equal(dy, &self.layouts[2]) || !layouts_equal(out, &self.layouts[3]) {
            return Err(invalid("matmul backward upstream/output geometry changed"));
        }
        Ok(())
    }
    pub fn metadata_bytes(&self) -> usize {
        4 * 2 * (size_of::<usize>() + size_of::<isize>())
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
#[cfg(feature = "mkl")]
pub(crate) fn gemm(
    cpu: Cpu,
    dims: [usize; 3],
    blas: Blas,
    data: [&[f32]; 2],
    out: &mut [f32],
    accumulate: bool,
) {
    let Cpu::Mkl { threads } = cpu else {
        unreachable!()
    };
    // SAFETY: preparation checked LP64 dimensions/leading strides and exact initialized
    // layouts; packed callers supply full tile capacities with the same leading strides.
    // Output access is exclusive, inputs stay borrowed, and the adapter restores threads.
    unsafe {
        fa_mkl_sgemm_layout(
            dims[0] as i32,
            dims[1] as i32,
            dims[2] as i32,
            blas.transposes[0],
            blas.transposes[1],
            blas.leading[0],
            blas.leading[1],
            blas.leading[2],
            data[0].as_ptr(),
            data[1].as_ptr(),
            out.as_mut_ptr(),
            i32::from(accumulate),
            threads.get().into(),
        )
    }
}

struct Workspace {
    #[cfg(feature = "mkl")]
    tiles: [usize; 3],
    data: Vec<f32>,
}
impl Workspace {
    fn new(strategy: MatmulStrategy, g: &Geometry, limit: usize) -> Result<Self> {
        let [m, n, k] = g.dimensions();
        let tiles = if strategy == MatmulStrategy::PackedMkl {
            [m.min(32), n.min(64), k.min(128)]
        } else {
            [0; 3]
        };
        let [tm, tn, tk] = tiles;
        let len = tm * tk + tk * tn + tm * tn;
        if len * size_of::<f32>() > limit {
            return Err(Error::Unsupported(format!(
                "matmul explicit workspace requires {} bytes, limit is {limit}",
                len * size_of::<f32>()
            )));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|e| Error::Allocation(e.to_string()))?;
        data.resize(len, 0.);
        Ok(Self {
            #[cfg(feature = "mkl")]
            tiles,
            data,
        })
    }
    fn bytes(&self) -> usize {
        self.data.capacity() * size_of::<f32>()
    }
    #[cfg(feature = "mkl")]
    fn packed(
        &mut self,
        cpu: Cpu,
        g: &Geometry,
        data: [&[f32]; 2],
        out: &mut [f32],
        mode: GradientMode,
    ) {
        let [m, n, k] = g.dimensions();
        let [tm, tn, tk] = self.tiles;
        let [a, b, c] = g.matrices;
        let (pa, rest) = self.data.split_at_mut(tm * tk);
        let (pb, pc) = rest.split_at_mut(tk * tn);
        let blas = Blas {
            leading: [tk as i32, tn as i32, tn as i32],
            transposes: [0; 2],
        };
        for row in (0..m).step_by(tm) {
            let rows = (m - row).min(tm);
            for col in (0..n).step_by(tn) {
                let cols = (n - col).min(tn);
                for start in (0..k).step_by(tk) {
                    let depth = (k - start).min(tk);
                    for i in 0..rows {
                        for p in 0..depth {
                            pa[i * tk + p] = a.read(data[0], row + i, start + p);
                        }
                    }
                    for p in 0..depth {
                        for j in 0..cols {
                            pb[p * tn + j] = b.read(data[1], start + p, col + j);
                        }
                    }
                    gemm(cpu, [rows, cols, depth], blas, [pa, pb], pc, start != 0);
                }
                for i in 0..rows {
                    for j in 0..cols {
                        // SAFETY: tiled coordinates partition the original injective output map.
                        let target = unsafe { out.get_unchecked_mut(c.offset(row + i, col + j)) };
                        *target = if mode == GradientMode::Overwrite {
                            pc[i * tn + j]
                        } else {
                            *target + pc[i * tn + j]
                        };
                    }
                }
            }
        }
    }
}
fn strategy(cpu: Cpu, g: &Geometry) -> MatmulStrategy {
    if g.empty() {
        return MatmulStrategy::Empty;
    }
    if g.zero() {
        return MatmulStrategy::Zero;
    }
    match cpu {
        Cpu::Reference => MatmulStrategy::Reference,
        #[cfg(feature = "mkl")]
        Cpu::Mkl { .. } => {
            if g.blas.is_some() {
                MatmulStrategy::Mkl
            } else {
                MatmulStrategy::PackedMkl
            }
        }
    }
}
fn execute(
    cpu: Cpu,
    g: &Geometry,
    strategy: MatmulStrategy,
    work: &mut Workspace,
    data: [&[f32]; 2],
    out: &mut [f32],
    mode: GradientMode,
) {
    if g.empty() || (g.zero() && mode == GradientMode::Accumulate) {
        return;
    }
    let data = g.orient(data);
    let [a, b, c] = g.matrices;
    let [m, n, k] = g.dimensions();
    #[cfg(feature = "mkl")]
    if strategy == MatmulStrategy::Mkl {
        let data = [&data[0][a.origin as usize..], &data[1][b.origin as usize..]];
        gemm(
            cpu,
            [m, n, k],
            g.blas.unwrap(),
            data,
            &mut out[c.origin as usize..],
            mode == GradientMode::Accumulate,
        );
        return;
    }
    #[cfg(feature = "mkl")]
    if strategy == MatmulStrategy::PackedMkl {
        work.packed(cpu, g, data, out, mode);
        return;
    }
    #[cfg(not(feature = "mkl"))]
    let _ = (cpu, strategy, work);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.;
            for p in 0..k {
                sum += a.read(data[0], i, p) * b.read(data[1], p, j);
            }
            // SAFETY: exact binding and injectivity checks establish initialized unique cells.
            let target = unsafe { out.get_unchecked_mut(c.offset(i, j)) };
            *target = if mode == GradientMode::Overwrite {
                sum
            } else {
                *target + sum
            };
        }
    }
}

/// Prepared rank-two FP32 multiplication with bounded layout fallback scratch.
pub struct MatmulPlan {
    pub(crate) geometry: ForwardGeometry,
    cpu: Cpu,
    strategy: MatmulStrategy,
    work: Workspace,
}
impl MatmulPlan {
    pub fn new(cpu: Cpu, left: &Layout, right: &Layout, destination: &Layout) -> Result<Self> {
        Self::with_workspace_limit(cpu, left, right, destination, MATMUL_WORKSPACE_LIMIT)
    }
    pub fn with_workspace_limit(
        cpu: Cpu,
        left: &Layout,
        right: &Layout,
        destination: &Layout,
        limit: usize,
    ) -> Result<Self> {
        let geometry = ForwardGeometry::new(left, right, destination)?;
        let strategy = strategy(cpu, &geometry.core);
        let work = Workspace::new(strategy, &geometry.core, limit)?;
        Ok(Self {
            geometry,
            cpu,
            strategy,
            work,
        })
    }
    pub fn left_layout(&self) -> &Layout {
        &self.geometry.layouts[0]
    }
    pub fn right_layout(&self) -> &Layout {
        &self.geometry.layouts[1]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.layouts[2]
    }
    pub fn strategy(&self) -> MatmulStrategy {
        self.strategy
    }
    pub fn workspace_bytes(&self) -> usize {
        self.work.bytes()
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.geometry.metadata_bytes()
    }
    pub fn run(
        &mut self,
        left: &TensorView<'_>,
        right: &TensorView<'_>,
        out: &mut TensorViewMut<'_>,
    ) -> Result<()> {
        self.geometry
            .validate(left.layout, right.layout, out.layout)?;
        execute(
            self.cpu,
            &self.geometry.core,
            self.strategy,
            &mut self.work,
            [left.data, right.data],
            out.data,
            GradientMode::Overwrite,
        );
        Ok(())
    }
}
/// One original matrix operand's local VJP; forward payloads remain caller-owned.
pub struct MatmulBackward {
    geometry: BackwardGeometry,
    cpu: Cpu,
    strategy: MatmulStrategy,
    work: Workspace,
}
impl MatmulBackward {
    pub fn new(
        cpu: Cpu,
        operand: BinaryOperand,
        left: &Layout,
        right: &Layout,
        upstream: &Layout,
        gradient: &Layout,
    ) -> Result<Self> {
        Self::with_workspace_limit(
            cpu,
            operand,
            left,
            right,
            upstream,
            gradient,
            MATMUL_WORKSPACE_LIMIT,
        )
    }
    pub fn with_workspace_limit(
        cpu: Cpu,
        operand: BinaryOperand,
        left: &Layout,
        right: &Layout,
        upstream: &Layout,
        gradient: &Layout,
        limit: usize,
    ) -> Result<Self> {
        let geometry = BackwardGeometry::new(operand, left, right, upstream, gradient)?;
        let strategy = strategy(cpu, &geometry.core);
        let work = Workspace::new(strategy, &geometry.core, limit)?;
        Ok(Self {
            geometry,
            cpu,
            strategy,
            work,
        })
    }
    pub fn operand(&self) -> BinaryOperand {
        self.geometry.operand
    }
    pub fn left_layout(&self) -> &Layout {
        &self.geometry.layouts[0]
    }
    pub fn right_layout(&self) -> &Layout {
        &self.geometry.layouts[1]
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.geometry.layouts[2]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.layouts[3]
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        self.geometry.dependencies
    }
    pub fn strategy(&self) -> MatmulStrategy {
        self.strategy
    }
    pub fn workspace_bytes(&self) -> usize {
        self.work.bytes()
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.geometry.metadata_bytes()
    }
    pub fn run(
        &mut self,
        left: Option<&TensorView<'_>>,
        right: Option<&TensorView<'_>>,
        upstream: &TensorView<'_>,
        out: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.geometry.validate(
            [left.map(|v| v.layout), right.map(|v| v.layout)],
            upstream.layout,
            out.layout,
        )?;
        let data = if self.operand() == BinaryOperand::Left {
            [upstream.data, right.map_or(&[][..], |v| v.data)]
        } else {
            [left.map_or(&[][..], |v| v.data), upstream.data]
        };
        execute(
            self.cpu,
            &self.geometry.core,
            self.strategy,
            &mut self.work,
            data,
            out.data,
            mode,
        );
        Ok(())
    }
}
