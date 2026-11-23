use crate::{Error, Layout, Result, TensorView, TensorViewMut, error::invalid};

/// FP32 unary equations; stable/exceptional rules are specified in docs/pointwise-v1.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum UnaryOp {
    Neg,
    Abs,
    Square,
    Reciprocal,
    Sqrt,
    Exp,
    Log,
    Relu,
    Sigmoid,
    Tanh,
    Silu,
    Softplus,
}
/// Ordered FP32 binary arithmetic with an explicitly prepared common broadcast shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}
/// Immutable traversal classification, not a measured optimal dispatch guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointwiseStrategy {
    Empty,
    Contiguous,
    ContiguousRows,
    Strided,
}

pub(crate) struct Axis<const N: usize> {
    pub dim: usize,
    pub inputs: [isize; N],
    pub output: isize,
}
pub(crate) struct PointwiseSpec<const N: usize> {
    pub inputs: [Layout; N],
    pub output: Layout,
    pub axes: Vec<Axis<N>>,
    pub offsets: [isize; N],
    pub output_offset: isize,
    pub len: usize,
    pub inner: usize,
    rows: usize,
}
pub(crate) fn broadcast_dims(inputs: &[&Layout]) -> Result<Vec<usize>> {
    let rank = inputs
        .iter()
        .map(|s| s.shape().dims().len())
        .max()
        .unwrap_or(0);
    let mut common = vec![1; rank];
    for source in inputs {
        for (target, &dim) in common
            .iter_mut()
            .rev()
            .zip(source.shape().dims().iter().rev())
        {
            if *target == 1 {
                *target = dim;
            } else if dim != 1 && dim != *target {
                return Err(invalid("pointwise input shapes cannot broadcast"));
            }
        }
    }
    Ok(common)
}
impl<const N: usize> PointwiseSpec<N> {
    pub fn new(inputs: [&Layout; N], output: &Layout) -> Result<Self> {
        let common = broadcast_dims(&inputs)?;
        if common != output.shape().dims() {
            return Err(invalid(
                "pointwise destination must have the exact common input shape",
            ));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "pointwise destination must be proved non-overlapping".into(),
            ));
        }
        let expanded: Vec<_> = inputs
            .iter()
            .map(|s| s.broadcast_to(&common))
            .collect::<Result<_>>()?;
        let mut offsets = inputs.map(|s| s.offset() as isize);
        let mut output_offset = output.offset() as isize;
        let len = output.shape().len();
        let mut axes = Vec::new();
        if len > 0 {
            for (axis, &dim) in common.iter().enumerate() {
                if dim == 1 {
                    continue;
                }
                let mut strides = std::array::from_fn(|i| expanded[i].strides()[axis]);
                let mut stride = output.strides()[axis];
                if stride < 0 {
                    // These are valid corners of checked layouts; singleton/empty strides
                    // never enter this path, so negation cannot involve isize::MIN.
                    for (offset, step) in offsets.iter_mut().zip(&mut strides) {
                        *offset += (dim - 1) as isize * *step;
                        *step = -*step;
                    }
                    output_offset += (dim - 1) as isize * stride;
                    stride = -stride;
                }
                axes.push(Axis {
                    dim,
                    inputs: strides,
                    output: stride,
                });
            }
            axes.sort_unstable_by_key(|a| std::cmp::Reverse(a.output));
        }
        let mut coalesced: Vec<Axis<N>> = Vec::with_capacity(axes.len());
        for outer in axes.into_iter().rev() {
            if let Some(inner) = coalesced.last_mut() {
                if inner.output.checked_mul(inner.dim as isize) == Some(outer.output)
                    && inner
                        .inputs
                        .iter()
                        .zip(outer.inputs)
                        .all(|(&s, t)| s.checked_mul(inner.dim as isize) == Some(t))
                {
                    inner.dim *= outer.dim;
                    continue;
                }
            }
            coalesced.push(outer);
        }
        coalesced.reverse();
        let inner = if len == 0 {
            0
        } else {
            coalesced.last().map_or(1, |axis| axis.dim)
        };
        Ok(Self {
            inputs: inputs.map(Clone::clone),
            output: output.clone(),
            axes: coalesced,
            offsets,
            output_offset,
            len,
            inner,
            rows: if len == 0 { 0 } else { len / inner },
        })
    }
    pub fn validate(&self, inputs: [&Layout; N], output: &Layout) -> Result<()> {
        if inputs
            .into_iter()
            .zip(&self.inputs)
            .any(|(a, b)| !(std::ptr::eq(a, b) || a == b))
            || !(std::ptr::eq(output, &self.output) || output == &self.output)
        {
            return Err(invalid(
                "pointwise view geometry differs from the prepared layouts",
            ));
        }
        Ok(())
    }
    pub fn inner_strides(&self) -> ([isize; N], isize) {
        self.axes
            .last()
            .map_or(([1; N], 1), |axis| (axis.inputs, axis.output))
    }
    pub fn strategy(&self) -> PointwiseStrategy {
        let (inputs, output) = self.inner_strides();
        if self.len == 0 {
            PointwiseStrategy::Empty
        } else if output == 1 && inputs.iter().all(|&s| s == 1) && self.axes.len() <= 1 {
            PointwiseStrategy::Contiguous
        } else if output == 1 && inputs.iter().all(|&s| s == 0 || s == 1) {
            PointwiseStrategy::ContiguousRows
        } else {
            PointwiseStrategy::Strided
        }
    }
    pub fn host_metadata_bytes(&self) -> usize {
        (self
            .inputs
            .iter()
            .map(|s| s.shape().dims().len())
            .sum::<usize>()
            + self.output.shape().dims().len())
            * (size_of::<usize>() + size_of::<isize>())
            + self.axes.capacity() * size_of::<Axis<N>>()
    }
    fn row_offsets(&self, mut row: usize) -> ([isize; N], isize) {
        let (mut inputs, mut output) = (self.offsets, self.output_offset);
        for axis in self.axes[..self.axes.len().saturating_sub(1)].iter().rev() {
            let coordinate = (row % axis.dim) as isize;
            row /= axis.dim;
            for (offset, step) in inputs.iter_mut().zip(axis.inputs) {
                *offset += coordinate * step;
            }
            output += coordinate * axis.output;
        }
        (inputs, output)
    }
}

#[cfg(feature = "mkl")]
pub(crate) mod vml {
    unsafe extern "C" {
        fn fa_mkl_vs_tanh(n: i32, x: *const f32, y: *mut f32);
        fn fa_mkl_vs_exp(n: i32, x: *const f32, y: *mut f32);
        fn fa_mkl_vs_ln(n: i32, x: *const f32, y: *mut f32);
        fn fa_mkl_vs_sqrt(n: i32, x: *const f32, y: *mut f32);
    }
    const MAX_CHUNK: usize = i32::MAX as usize;
    fn run(f: unsafe extern "C" fn(i32, *const f32, *mut f32), x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), y.len());
        let mut offset = 0;
        while offset < x.len() {
            let chunk = (x.len() - offset).min(MAX_CHUNK);
            // SAFETY: x and y are distinct initialized slices with identical length >= offset + chunk.
            unsafe {
                f(
                    chunk as i32,
                    x.as_ptr().add(offset),
                    y.as_mut_ptr().add(offset),
                )
            }
            offset += chunk;
        }
    }
    pub(crate) fn tanh(x: &[f32], y: &mut [f32]) {
        run(fa_mkl_vs_tanh, x, y);
    }
    pub(crate) fn exp(x: &[f32], y: &mut [f32]) {
        run(fa_mkl_vs_exp, x, y);
    }
    pub(crate) fn ln(x: &[f32], y: &mut [f32]) {
        run(fa_mkl_vs_ln, x, y);
    }
    pub(crate) fn sqrt(x: &[f32], y: &mut [f32]) {
        run(fa_mkl_vs_sqrt, x, y);
    }
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let e = x.exp();
        e / (1. + e)
    }
}
fn relu(x: f32) -> f32 {
    if x < 0. { 0. } else { x }
}
fn silu(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        -0.
    } else {
        x * sigmoid(x)
    }
}
fn softplus(x: f32) -> f32 {
    x.max(0.) + (-x.abs()).exp().ln_1p()
}

/// Prepared host unary math with allocation-free execution and explicit output views.
pub struct UnaryPlan {
    spec: PointwiseSpec<1>,
    operation: UnaryOp,
}
impl UnaryPlan {
    pub fn new(operation: UnaryOp, source: &Layout, destination: &Layout) -> Result<Self> {
        Ok(Self {
            spec: PointwiseSpec::new([source], destination)?,
            operation,
        })
    }
    pub fn operation(&self) -> UnaryOp {
        self.operation
    }
    pub fn source_layout(&self) -> &Layout {
        &self.spec.inputs[0]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.spec.output
    }
    pub fn strategy(&self) -> PointwiseStrategy {
        self.spec.strategy()
    }
    pub fn coalesced_rank(&self) -> usize {
        self.spec.axes.len()
    }
    pub fn inner_len(&self) -> usize {
        self.spec.inner
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.host_metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn run(&self, source: &TensorView<'_>, destination: &mut TensorViewMut<'_>) -> Result<()> {
        self.spec.validate([source.layout], destination.layout)?;
        match self.operation {
            UnaryOp::Neg => self.spec.unary(source.data, destination.data, |x| {
                f32::from_bits(x.to_bits() ^ 0x80000000)
            }),
            UnaryOp::Abs => self.spec.unary(source.data, destination.data, |x| {
                f32::from_bits(x.to_bits() & 0x7fffffff)
            }),
            UnaryOp::Square => self.spec.unary(source.data, destination.data, |x| x * x),
            UnaryOp::Reciprocal => self.spec.unary(source.data, destination.data, f32::recip),
            UnaryOp::Sqrt => {
                #[cfg(feature = "mkl")]
                {
                    self.spec
                        .unary_vml(source.data, destination.data, vml::sqrt, f32::sqrt);
                }
                #[cfg(not(feature = "mkl"))]
                self.spec.unary(source.data, destination.data, f32::sqrt);
            }
            UnaryOp::Exp => {
                #[cfg(feature = "mkl")]
                {
                    self.spec
                        .unary_vml(source.data, destination.data, vml::exp, f32::exp);
                }
                #[cfg(not(feature = "mkl"))]
                self.spec.unary(source.data, destination.data, f32::exp);
            }
            UnaryOp::Log => {
                #[cfg(feature = "mkl")]
                {
                    self.spec
                        .unary_vml(source.data, destination.data, vml::ln, f32::ln);
                }
                #[cfg(not(feature = "mkl"))]
                self.spec.unary(source.data, destination.data, f32::ln);
            }
            UnaryOp::Relu => self.spec.unary(source.data, destination.data, relu),
            UnaryOp::Sigmoid => {
                #[cfg(feature = "mkl")]
                {
                    self.spec
                        .unary_vml(source.data, destination.data, sigmoid_vml, sigmoid);
                }
                #[cfg(not(feature = "mkl"))]
                self.spec.unary(source.data, destination.data, sigmoid);
            }
            UnaryOp::Tanh => {
                #[cfg(feature = "mkl")]
                {
                    self.spec
                        .unary_vml(source.data, destination.data, vml::tanh, f32::tanh);
                }
                #[cfg(not(feature = "mkl"))]
                self.spec.unary(source.data, destination.data, f32::tanh);
            }
            UnaryOp::Silu => {
                #[cfg(feature = "mkl")]
                {
                    self.spec
                        .unary_vml(source.data, destination.data, silu_vml, silu);
                }
                #[cfg(not(feature = "mkl"))]
                self.spec.unary(source.data, destination.data, silu);
            }
            UnaryOp::Softplus => {
                #[cfg(feature = "mkl")]
                {
                    self.spec
                        .unary_vml(source.data, destination.data, softplus_vml, softplus);
                }
                #[cfg(not(feature = "mkl"))]
                self.spec.unary(source.data, destination.data, softplus);
            }
        }
        Ok(())
    }
}
#[cfg(feature = "mkl")]
fn sigmoid_vml(x: &[f32], out: &mut [f32]) {
    let mut neg_abs = [0.0f32; 256];
    let mut e = [0.0f32; 256];
    for (xs, ys) in x.chunks(256).zip(out.chunks_mut(256)) {
        let n = xs.len();
        for (dst, &v) in neg_abs[..n].iter_mut().zip(xs) {
            *dst = -v.abs();
        }
        vml::exp(&neg_abs[..n], &mut e[..n]);
        for ((y, &v), &ev) in ys.iter_mut().zip(xs).zip(&e[..n]) {
            *y = if v >= 0.0 {
                1.0 / (1.0 + ev)
            } else {
                ev / (1.0 + ev)
            };
        }
    }
}
#[cfg(feature = "mkl")]
fn silu_vml(x: &[f32], out: &mut [f32]) {
    let mut neg_abs = [0.0f32; 256];
    let mut e = [0.0f32; 256];
    for (xs, ys) in x.chunks(256).zip(out.chunks_mut(256)) {
        let n = xs.len();
        for (dst, &v) in neg_abs[..n].iter_mut().zip(xs) {
            *dst = -v.abs();
        }
        vml::exp(&neg_abs[..n], &mut e[..n]);
        for ((y, &v), &ev) in ys.iter_mut().zip(xs).zip(&e[..n]) {
            *y = if v == f32::NEG_INFINITY {
                -0.0
            } else if v >= 0.0 {
                v * (1.0 / (1.0 + ev))
            } else {
                v * (ev / (1.0 + ev))
            };
        }
    }
}
#[cfg(feature = "mkl")]
fn softplus_vml(x: &[f32], out: &mut [f32]) {
    let mut neg_abs = [0.0f32; 256];
    let mut e = [0.0f32; 256];
    for (xs, ys) in x.chunks(256).zip(out.chunks_mut(256)) {
        let n = xs.len();
        for (dst, &v) in neg_abs[..n].iter_mut().zip(xs) {
            *dst = -v.abs();
        }
        vml::exp(&neg_abs[..n], &mut e[..n]);
        for ((y, &v), &ev) in ys.iter_mut().zip(xs).zip(&e[..n]) {
            *y = v.max(0.0) + ev.ln_1p();
        }
    }
}
impl PointwiseSpec<1> {
    #[cfg(feature = "mkl")]
    fn unary_vml(
        &self,
        source: &[f32],
        destination: &mut [f32],
        vec_op: fn(&[f32], &mut [f32]),
        scalar_op: impl Fn(f32) -> f32,
    ) {
        let ([sx], sy) = self.inner_strides();
        if sy == 1 && sx == 1 && self.inner >= 64 {
            for row in 0..self.rows {
                let ([x], y) = self.row_offsets(row);
                vec_op(
                    &source[x as usize..][..self.inner],
                    &mut destination[y as usize..][..self.inner],
                );
            }
            return;
        }
        self.unary(source, destination, scalar_op);
    }
    fn unary(&self, source: &[f32], destination: &mut [f32], operation: impl Fn(f32) -> f32) {
        let ([sx], sy) = self.inner_strides();
        for row in 0..self.rows {
            let ([x], y) = self.row_offsets(row);
            if sy == 1 && sx == 1 {
                for (out, &value) in destination[y as usize..][..self.inner]
                    .iter_mut()
                    .zip(&source[x as usize..][..self.inner])
                {
                    *out = operation(value);
                }
            } else if sy == 1 && sx == 0 {
                destination[y as usize..][..self.inner].fill(operation(source[x as usize]));
            } else {
                for column in 0..self.inner {
                    // SAFETY: bound initialized views validate every original coordinate.
                    // Axis reversal/permutation/coalescing preserve that coordinate set;
                    // the exclusive output has an injective map and cannot alias source.
                    unsafe {
                        let value = source.as_ptr().offset(x + column as isize * sx).read();
                        destination
                            .as_mut_ptr()
                            .offset(y + column as isize * sy)
                            .write(operation(value));
                    }
                }
            }
        }
    }
}

/// Prepared ordered binary math. Broadcasting is fixed before binding payloads.
///
/// ```
/// use fast_audio::{BinaryOp, Layout, Tensor, cpu};
/// let a = Tensor::from_vec([2,1], vec![1., 2.])?;
/// let b = Tensor::from_vec([3], vec![10., 20., 30.])?;
/// let left = Layout::contiguous([2,1])?;
/// let right = Layout::contiguous([3])?;
/// let output = Layout::contiguous([2,3])?;
/// let plan = cpu::BinaryPlan::new(BinaryOp::Add, &left, &right, &output)?;
/// let mut out = Tensor::zeros([2,3])?;
/// plan.run(&a.view(plan.left_layout())?, &b.view(plan.right_layout())?,
///          &mut out.view_mut(plan.destination_layout())?)?;
/// assert_eq!(out.data(), &[11.,21.,31.,12.,22.,32.]);
/// # Ok::<(), fast_audio::Error>(())
/// ```
pub struct BinaryPlan {
    spec: PointwiseSpec<2>,
    operation: BinaryOp,
}
impl BinaryPlan {
    pub fn new(
        operation: BinaryOp,
        left: &Layout,
        right: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        Ok(Self {
            spec: PointwiseSpec::new([left, right], destination)?,
            operation,
        })
    }
    pub fn operation(&self) -> BinaryOp {
        self.operation
    }
    pub fn left_layout(&self) -> &Layout {
        &self.spec.inputs[0]
    }
    pub fn right_layout(&self) -> &Layout {
        &self.spec.inputs[1]
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.spec.output
    }
    pub fn strategy(&self) -> PointwiseStrategy {
        self.spec.strategy()
    }
    pub fn coalesced_rank(&self) -> usize {
        self.spec.axes.len()
    }
    pub fn inner_len(&self) -> usize {
        self.spec.inner
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.host_metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn run(
        &self,
        left: &TensorView<'_>,
        right: &TensorView<'_>,
        destination: &mut TensorViewMut<'_>,
    ) -> Result<()> {
        self.spec
            .validate([left.layout, right.layout], destination.layout)?;
        match self.operation {
            BinaryOp::Add => self
                .spec
                .binary(left.data, right.data, destination.data, |a, b| a + b),
            BinaryOp::Sub => self
                .spec
                .binary(left.data, right.data, destination.data, |a, b| a - b),
            BinaryOp::Mul => self
                .spec
                .binary(left.data, right.data, destination.data, |a, b| a * b),
            BinaryOp::Div => self
                .spec
                .binary(left.data, right.data, destination.data, |a, b| a / b),
        }
        Ok(())
    }
}
fn dense_binary<const A: bool, const B: bool>(
    left: &[f32],
    right: &[f32],
    out: &mut [f32],
    operation: &impl Fn(f32, f32) -> f32,
) {
    match (A, B) {
        (false, false) => {
            for ((value, &a), &b) in out.iter_mut().zip(left).zip(right) {
                *value = operation(a, b);
            }
        }
        (true, false) => {
            let a = left[0];
            for (value, &b) in out.iter_mut().zip(right) {
                *value = operation(a, b);
            }
        }
        (false, true) => {
            let b = right[0];
            for (value, &a) in out.iter_mut().zip(left) {
                *value = operation(a, b);
            }
        }
        (true, true) => {
            out.fill(operation(left[0], right[0]));
        }
    }
}
impl PointwiseSpec<2> {
    fn binary(
        &self,
        left: &[f32],
        right: &[f32],
        destination: &mut [f32],
        operation: impl Fn(f32, f32) -> f32,
    ) {
        let ([sa, sb], so) = self.inner_strides();
        for row in 0..self.rows {
            let ([a, b], out) = self.row_offsets(row);
            if so == 1 && (sa == 0 || sa == 1) && (sb == 0 || sb == 1) {
                let a = &left[a as usize..][..if sa == 0 { 1 } else { self.inner }];
                let b = &right[b as usize..][..if sb == 0 { 1 } else { self.inner }];
                let out = &mut destination[out as usize..][..self.inner];
                match (sa == 0, sb == 0) {
                    (false, false) => dense_binary::<false, false>(a, b, out, &operation),
                    (true, false) => dense_binary::<true, false>(a, b, out, &operation),
                    (false, true) => dense_binary::<false, true>(a, b, out, &operation),
                    (true, true) => dense_binary::<true, true>(a, b, out, &operation),
                }
            } else {
                for column in 0..self.inner {
                    // SAFETY: same checked coordinate-set/injective-output proof as unary.
                    // Read inputs may alias each other; neither can alias the exclusive out.
                    unsafe {
                        let x = left.as_ptr().offset(a + column as isize * sa).read();
                        let y = right.as_ptr().offset(b + column as isize * sb).read();
                        destination
                            .as_mut_ptr()
                            .offset(out + column as isize * so)
                            .write(operation(x, y));
                    }
                }
            }
        }
    }
}
