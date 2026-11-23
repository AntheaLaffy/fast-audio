use crate::{
    Error, GradientMode, Layout, Result, Shape, TensorView, TensorViewMut, copy::MappingSpec,
    error::invalid,
};

/// Ordered FP32 reductions; empty/rounding rules are specified in docs/reduction-v1.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    Sum = 0,
    Mean = 1,
    Max = 2,
    Min = 3,
    LogSumExp = 4,
}

impl ReduceOp {
    /// Whether the local backward derivative requires the saved forward input tensor.
    pub const fn requires_input(self) -> bool {
        matches!(self, Self::Max | Self::Min | Self::LogSumExp)
    }
}

/// Deterministic initial routes, not measured optimal dispatch choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceStrategy {
    Empty,
    EmptyReduction,
    Copy,
    Tree32,
    Tree256,
}

/// Shape/axis policy shared by forward and payload-free local backward plans.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReduceSpec {
    operation: ReduceOp,
    input: Shape,
    output: Shape,
    axes: Box<[usize]>,
    keep_dims: bool,
    count: Option<usize>,
}
impl ReduceSpec {
    pub fn new(
        operation: ReduceOp,
        input: &Shape,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<Self> {
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        if axes.iter().any(|&a| a >= input.dims().len())
            || axes.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(invalid("reduction axes must be distinct and in range"));
        }
        let dims: Vec<_> = input
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(a, &d)| {
                if axes.binary_search(&a).is_ok() {
                    keep_dims.then_some(1)
                } else {
                    Some(d)
                }
            })
            .collect();
        let output = Shape::new(dims)?;
        let count = if axes.iter().any(|&a| input.dims()[a] == 0) {
            Some(0)
        } else {
            axes.iter()
                .try_fold(1usize, |n, &a| n.checked_mul(input.dims()[a]))
        };
        if count.is_none() && !output.is_empty() {
            return Err(invalid("nonempty reduction extent overflows"));
        }
        Ok(Self {
            operation,
            input: input.clone(),
            output,
            axes: axes.into_boxed_slice(),
            keep_dims,
            count,
        })
    }
    pub fn operation(&self) -> ReduceOp {
        self.operation
    }
    pub fn input_shape(&self) -> &Shape {
        &self.input
    }
    pub fn output_shape(&self) -> &Shape {
        &self.output
    }
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
    pub fn keep_dims(&self) -> bool {
        self.keep_dims
    }
    /// None is possible only for an empty output with an unrepresentable extent.
    pub fn reduction_len(&self) -> Option<usize> {
        self.count
    }
    pub fn strategy(&self) -> ReduceStrategy {
        if self.output.is_empty() {
            ReduceStrategy::Empty
        } else {
            match self.count.unwrap() {
                0 => ReduceStrategy::EmptyReduction,
                1 => ReduceStrategy::Copy,
                2..=32 => ReduceStrategy::Tree32,
                _ => ReduceStrategy::Tree256,
            }
        }
    }
    pub fn reduction_lanes(&self) -> usize {
        match self.strategy() {
            ReduceStrategy::Empty | ReduceStrategy::EmptyReduction => 0,
            ReduceStrategy::Copy => 1,
            ReduceStrategy::Tree32 => 32,
            ReduceStrategy::Tree256 => 256,
        }
    }
    pub(crate) fn divisor(&self) -> f32 {
        if self.operation == ReduceOp::Mean {
            self.count.unwrap_or(1).max(1) as f32
        } else {
            1.
        }
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        (self.input.dims().len() + self.output.dims().len() + self.axes.len()) * size_of::<usize>()
    }
}

pub(crate) struct GroupAxis {
    pub dim: usize,
    pub source: isize,
    pub destination: isize,
}
pub(crate) struct ReduceAxis {
    pub dim: usize,
    pub stride: isize,
}
pub(crate) struct ReduceGeometry {
    pub spec: ReduceSpec,
    pub source: Layout,
    pub destination: Layout,
    pub groups: Vec<GroupAxis>,
    pub reduced: Vec<ReduceAxis>,
}
impl ReduceGeometry {
    pub fn new(spec: &ReduceSpec, source: &Layout, destination: &Layout) -> Result<Self> {
        if source.shape() != spec.input_shape() || destination.shape() != spec.output_shape() {
            return Err(invalid(
                "reduction layouts must match the exact spec shapes",
            ));
        }
        if !destination.is_non_overlapping() {
            return Err(Error::Unsupported(
                "reduction destination must be proved non-overlapping".into(),
            ));
        }
        let mut groups: Vec<GroupAxis> = Vec::new();
        let mut reduced: Vec<ReduceAxis> = Vec::new();
        if !spec.output.is_empty() {
            let mut output_axis = destination.strides().len();
            for (axis, &dim) in spec.input.dims().iter().enumerate().rev() {
                if spec.axes.binary_search(&axis).is_ok() {
                    if spec.keep_dims {
                        output_axis -= 1;
                    }
                    if spec.count.unwrap() > 1 && dim > 1 {
                        let stride = source.strides()[axis];
                        if let Some(inner) = reduced.last_mut() {
                            if inner.stride.checked_mul(inner.dim as isize) == Some(stride) {
                                inner.dim *= dim;
                                continue;
                            }
                        }
                        reduced.push(ReduceAxis { dim, stride });
                    }
                } else {
                    output_axis -= 1;
                    if dim <= 1 {
                        continue;
                    }
                    // Empty contractions never inspect or calculate unused source addresses.
                    let x = if spec.count == Some(0) {
                        0
                    } else {
                        source.strides()[axis]
                    };
                    let y = destination.strides()[output_axis];
                    if let Some(inner) = groups.last_mut() {
                        if inner.source.checked_mul(inner.dim as isize) == Some(x)
                            && inner.destination.checked_mul(inner.dim as isize) == Some(y)
                        {
                            inner.dim *= dim;
                            continue;
                        }
                    }
                    groups.push(GroupAxis {
                        dim,
                        source: x,
                        destination: y,
                    });
                }
            }
            groups.reverse();
            reduced.reverse();
        }
        Ok(Self {
            spec: spec.clone(),
            source: source.clone(),
            destination: destination.clone(),
            groups,
            reduced,
        })
    }
    pub fn validate(&self, source: &Layout, destination: &Layout) -> Result<()> {
        if !(std::ptr::eq(source, &self.source) || source == &self.source)
            || !(std::ptr::eq(destination, &self.destination) || destination == &self.destination)
        {
            return Err(invalid(
                "reduction bindings differ from the prepared layouts",
            ));
        }
        Ok(())
    }
    fn group_offsets(&self, mut flat: usize) -> (isize, usize) {
        let (mut x, mut y) = (
            self.source.offset() as isize,
            self.destination.offset() as isize,
        );
        for axis in (1..self.groups.len()).rev() {
            let a = &self.groups[axis];
            let c = (flat % a.dim) as isize;
            flat /= a.dim;
            x += c * a.source;
            y += c * a.destination;
        }
        if let Some(a) = self.groups.first() {
            x += flat as isize * a.source;
            y += flat as isize * a.destination;
        }
        (x, y as usize)
    }
    fn reduction_offset(&self, mut flat: usize) -> isize {
        let mut x = 0;
        for axis in (1..self.reduced.len()).rev() {
            let a = &self.reduced[axis];
            x += (flat % a.dim) as isize * a.stride;
            flat /= a.dim;
        }
        if let Some(a) = self.reduced.first() {
            x += flat as isize * a.stride;
        }
        x
    }
    pub fn metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
            + (self.source.shape().dims().len() + self.destination.shape().dims().len())
                * (size_of::<usize>() + size_of::<isize>())
            + self.groups.capacity() * size_of::<GroupAxis>()
            + self.reduced.capacity() * size_of::<ReduceAxis>()
    }
}

#[inline]
pub(crate) fn reduce_lanes<const N: usize>(mut lanes: [f32; N]) -> f32 {
    let mut step = N / 2;
    while step > 0 {
        for lane in 0..step {
            lanes[lane] += lanes[lane + step];
        }
        step /= 2;
    }
    lanes[0]
}

pub(crate) fn tree<const N: usize>(count: usize, value: impl Fn(usize) -> f32) -> f32 {
    let mut lanes = [0.0f32; N];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        for r in (lane..count).step_by(N) {
            *sum += value(r);
        }
    }
    reduce_lanes(lanes)
}

/// Prepared ordered Sum/Mean with borrowed storage and caller-owned strided outputs.
///
/// ```
/// use fast_audio::{ReduceOp, ReduceSpec, Layout, Tensor, cpu};
/// let x = Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.])?;
/// let spec = ReduceSpec::new(ReduceOp::Mean, x.shape(), &[1], false)?;
/// let source = Layout::contiguous([2, 3])?;
/// let destination = Layout::contiguous([2])?;
/// let plan = cpu::ReducePlan::new(&spec, &source, &destination)?;
/// let mut y = Tensor::zeros([2])?;
/// plan.run(&x.view(&source)?, &mut y.view_mut(&destination)?)?;
/// assert_eq!(y.data(), &[2., 5.]);
/// # Ok::<(), fast_audio::Error>(())
/// ```
pub struct ReducePlan {
    geometry: ReduceGeometry,
    kernel: ForwardKernel,
}
type ForwardKernel = fn(&ReduceGeometry, &[f32], &mut [f32]);

impl ReducePlan {
    pub fn new(spec: &ReduceSpec, source: &Layout, destination: &Layout) -> Result<Self> {
        let geometry = ReduceGeometry::new(spec, source, destination)?;
        let kernel = Self::select_kernel(&geometry);
        Ok(Self { geometry, kernel })
    }
    fn select_kernel(g: &ReduceGeometry) -> ForwardKernel {
        let op = g.spec.operation;
        let count = g.spec.count.unwrap_or(0);
        let linear_group = g.groups.len() <= 1;
        let linear_red = g.reduced.len() == 1;
        match g.spec.strategy() {
            ReduceStrategy::Empty => Self::empty,
            ReduceStrategy::EmptyReduction => match op {
                ReduceOp::Sum => Self::fill_empty_value::<0>,
                ReduceOp::Mean => Self::fill_empty_value::<1>,
                ReduceOp::Max | ReduceOp::LogSumExp => Self::fill_empty_value::<2>,
                ReduceOp::Min => Self::fill_empty_value::<3>,
            },
            ReduceStrategy::Copy => {
                if linear_group {
                    Self::copy_linear
                } else {
                    Self::copy_general
                }
            }
            ReduceStrategy::Tree32 | ReduceStrategy::Tree256 if op.requires_input() => match op {
                ReduceOp::Max => Self::reduce_max,
                ReduceOp::Min => Self::reduce_min,
                ReduceOp::LogSumExp => Self::reduce_logsumexp,
                _ => unreachable!(),
            },
            ReduceStrategy::Tree32 => {
                let mean = op == ReduceOp::Mean;
                if count == 2 {
                    if linear_group && linear_red {
                        let contiguous_pairs = g
                            .groups
                            .first()
                            .is_none_or(|a| a.source == 2 && a.destination == 1)
                            && g.reduced[0].stride == 1;
                        match (mean, contiguous_pairs) {
                            (false, true) => Self::pair_contiguous::<false>,
                            (true, true) => Self::pair_contiguous::<true>,
                            (false, false) => Self::pair_linear::<false>,
                            (true, false) => Self::pair_linear::<true>,
                        }
                    } else if mean {
                        Self::pair_general::<true>
                    } else {
                        Self::pair_general::<false>
                    }
                } else {
                    match (mean, count) {
                        (false, 3..=4) => Self::small_tree::<false, 4>,
                        (true, 3..=4) => Self::small_tree::<true, 4>,
                        (false, 5..=8) => Self::small_tree::<false, 8>,
                        (true, 5..=8) => Self::small_tree::<true, 8>,
                        (false, 9..=16) => Self::small_tree::<false, 16>,
                        (true, 9..=16) => Self::small_tree::<true, 16>,
                        (false, _) => Self::small_tree::<false, 32>,
                        (true, _) => Self::small_tree::<true, 32>,
                    }
                }
            }
            ReduceStrategy::Tree256 => {
                let mean = op == ReduceOp::Mean;
                if linear_red && g.reduced[0].stride == 1 {
                    if mean {
                        Self::tree256_contiguous::<true>
                    } else {
                        Self::tree256_contiguous::<false>
                    }
                } else if linear_red {
                    if mean {
                        Self::tree256_linear::<true>
                    } else {
                        Self::tree256_linear::<false>
                    }
                } else if mean {
                    Self::tree256_general::<true>
                } else {
                    Self::tree256_general::<false>
                }
            }
        }
    }
    pub fn spec(&self) -> &ReduceSpec {
        &self.geometry.spec
    }
    pub fn source_layout(&self) -> &Layout {
        &self.geometry.source
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.destination
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.geometry.metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn run(&self, source: &TensorView<'_>, destination: &mut TensorViewMut<'_>) -> Result<()> {
        let g = &self.geometry;
        g.validate(source.layout, destination.layout)?;
        (self.kernel)(g, source.data, destination.data);
        Ok(())
    }

    #[inline(never)]
    fn empty(_: &ReduceGeometry, _: &[f32], _: &mut [f32]) {}

    #[inline(never)]
    fn fill_empty_value<const CODE: u8>(g: &ReduceGeometry, _: &[f32], destination: &mut [f32]) {
        let value = match CODE {
            0 => 0.0f32,
            1 => f32::from_bits(0x7fc00000),
            2 => f32::NEG_INFINITY,
            _ => f32::INFINITY,
        };
        let total = g.spec.output.len();
        if g.groups.len() <= 1 {
            let dst_origin = g.destination.offset() as isize;
            let dst_step = g.groups.first().map_or(0, |a| a.destination);
            for i in 0..total {
                // SAFETY: exact injective output layout keeps every group coordinate in bounds.
                unsafe {
                    *destination.get_unchecked_mut((dst_origin + i as isize * dst_step) as usize) =
                        value;
                }
            }
        } else {
            for group in 0..total {
                let (_, y) = g.group_offsets(group);
                // SAFETY: exact injective output layout keeps every group coordinate in bounds.
                unsafe {
                    *destination.get_unchecked_mut(y) = value;
                }
            }
        }
    }

    #[inline(never)]
    fn reduce_max(g: &ReduceGeometry, source: &[f32], destination: &mut [f32]) {
        let count = g.spec.count.unwrap_or(0);
        let linear_red = g.reduced.len() == 1;
        let r_step = g.reduced.first().map_or(0, |a| a.stride);
        if linear_red && r_step == 1 {
            for group in 0..g.spec.output.len() {
                let (x, y) = g.group_offsets(group);
                let xs = &source[x as usize..x as usize + count];
                let mut acc = xs[0];
                for &v in &xs[1..] {
                    acc = max_nan(acc, v);
                }
                destination[y] = acc;
            }
            return;
        }
        for group in 0..g.spec.output.len() {
            let (x, y) = g.group_offsets(group);
            let mut acc = source[x as usize];
            for r in 1..count {
                let off = if linear_red {
                    x + r as isize * r_step
                } else {
                    x + g.reduction_offset(r)
                };
                acc = max_nan(acc, source[off as usize]);
            }
            destination[y] = acc;
        }
    }

    #[inline(never)]
    fn reduce_min(g: &ReduceGeometry, source: &[f32], destination: &mut [f32]) {
        let count = g.spec.count.unwrap_or(0);
        let linear_red = g.reduced.len() == 1;
        let r_step = g.reduced.first().map_or(0, |a| a.stride);
        if linear_red && r_step == 1 {
            for group in 0..g.spec.output.len() {
                let (x, y) = g.group_offsets(group);
                let xs = &source[x as usize..x as usize + count];
                let mut acc = xs[0];
                for &v in &xs[1..] {
                    acc = min_nan(acc, v);
                }
                destination[y] = acc;
            }
            return;
        }
        for group in 0..g.spec.output.len() {
            let (x, y) = g.group_offsets(group);
            let mut acc = source[x as usize];
            for r in 1..count {
                let off = if linear_red {
                    x + r as isize * r_step
                } else {
                    x + g.reduction_offset(r)
                };
                acc = min_nan(acc, source[off as usize]);
            }
            destination[y] = acc;
        }
    }

    #[inline(never)]
    fn reduce_logsumexp(g: &ReduceGeometry, source: &[f32], destination: &mut [f32]) {
        let count = g.spec.count.unwrap_or(0);
        let linear_red = g.reduced.len() == 1;
        let r_step = g.reduced.first().map_or(0, |a| a.stride);
        if linear_red && r_step == 1 {
            const TILE: usize = 256;
            let mut buf = [0.0f32; TILE];
            for group in 0..g.spec.output.len() {
                let (x, y) = g.group_offsets(group);
                let xs = &source[x as usize..x as usize + count];
                let mut m = xs[0];
                for &v in &xs[1..] {
                    m = max_nan(m, v);
                }
                if m.is_nan() || m.is_infinite() {
                    destination[y] = m;
                    continue;
                }
                let mut sum = 0.0f32;
                let mut off = 0;
                while off < count {
                    let n = (count - off).min(TILE);
                    for j in 0..n {
                        buf[j] = xs[off + j] - m;
                    }
                    exp_tile_in_place(&mut buf[..n]);
                    sum += sum_tile_8way(&buf[..n]);
                    off += n;
                }
                destination[y] = m + sum.ln();
            }
            return;
        }
        for group in 0..g.spec.output.len() {
            let (x, y) = g.group_offsets(group);
            let mut m = source[x as usize];
            for r in 1..count {
                let off = if linear_red {
                    x + r as isize * r_step
                } else {
                    x + g.reduction_offset(r)
                };
                m = max_nan(m, source[off as usize]);
            }
            if m.is_nan() || m.is_infinite() {
                destination[y] = m;
                continue;
            }
            let mut sum = 0.0f32;
            for r in 0..count {
                let off = if linear_red {
                    x + r as isize * r_step
                } else {
                    x + g.reduction_offset(r)
                };
                sum += (source[off as usize] - m).exp();
            }
            destination[y] = m + sum.ln();
        }
    }

    #[inline(never)]
    fn copy_linear(g: &ReduceGeometry, source: &[f32], destination: &mut [f32]) {
        let total = g.spec.output.len();
        let src_origin = g.source.offset() as isize;
        let dst_origin = g.destination.offset() as isize;
        let (src_step, dst_step) = g
            .groups
            .first()
            .map_or((0, 0), |a| (a.source, a.destination));
        for i in 0..total {
            // SAFETY: exact validated source/destination layouts keep every coordinate in bounds.
            unsafe {
                let v = *source.get_unchecked((src_origin + i as isize * src_step) as usize);
                *destination.get_unchecked_mut((dst_origin + i as isize * dst_step) as usize) = v;
            }
        }
    }

    #[inline(never)]
    fn copy_general(g: &ReduceGeometry, source: &[f32], destination: &mut [f32]) {
        for group in 0..g.spec.output.len() {
            let (x, y) = g.group_offsets(group);
            // SAFETY: exact validated source/destination layouts keep every coordinate in bounds.
            unsafe {
                *destination.get_unchecked_mut(y) = *source.get_unchecked(x as usize);
            }
        }
    }

    #[inline(never)]
    fn pair_contiguous<const MEAN: bool>(
        g: &ReduceGeometry,
        source: &[f32],
        destination: &mut [f32],
    ) {
        let total = g.spec.output.len();
        let src = &source[g.source.offset()..g.source.offset() + total * 2];
        let dst = &mut destination[g.destination.offset()..g.destination.offset() + total];
        for (out, pair) in dst.iter_mut().zip(src.chunks_exact(2)) {
            let sum = (0.0f32 + pair[0]) + (0.0f32 + pair[1]);
            *out = if MEAN { sum / 2.0f32 } else { sum };
        }
    }

    #[inline(never)]
    fn pair_linear<const MEAN: bool>(g: &ReduceGeometry, source: &[f32], destination: &mut [f32]) {
        let total = g.spec.output.len();
        let src_origin = g.source.offset() as isize;
        let dst_origin = g.destination.offset() as isize;
        let (src_step, dst_step) = g
            .groups
            .first()
            .map_or((0, 0), |a| (a.source, a.destination));
        let r_step = g.reduced[0].stride;
        for i in 0..total {
            let x = src_origin + i as isize * src_step;
            let y = (dst_origin + i as isize * dst_step) as usize;
            // SAFETY: exact validated layouts guarantee both pair reads and injective write are in bounds.
            unsafe {
                let v0 = *source.get_unchecked(x as usize);
                let v1 = *source.get_unchecked((x + r_step) as usize);
                let sum = (0.0f32 + v0) + (0.0f32 + v1);
                *destination.get_unchecked_mut(y) = if MEAN { sum / 2.0f32 } else { sum };
            }
        }
    }

    #[inline(never)]
    fn pair_general<const MEAN: bool>(g: &ReduceGeometry, source: &[f32], destination: &mut [f32]) {
        let r1 = g.reduction_offset(1);
        for group in 0..g.spec.output.len() {
            let (x, y) = g.group_offsets(group);
            // SAFETY: exact validated layouts guarantee both pair reads and injective write are in bounds.
            unsafe {
                let v0 = *source.get_unchecked(x as usize);
                let v1 = *source.get_unchecked((x + r1) as usize);
                let sum = (0.0f32 + v0) + (0.0f32 + v1);
                *destination.get_unchecked_mut(y) = if MEAN { sum / 2.0f32 } else { sum };
            }
        }
    }

    #[inline(never)]
    fn small_tree<const MEAN: bool, const N: usize>(
        g: &ReduceGeometry,
        source: &[f32],
        destination: &mut [f32],
    ) {
        let total = g.spec.output.len();
        let count = g.spec.count.unwrap_or(0);
        let divisor = g.spec.divisor();
        let linear_group = g.groups.len() <= 1;
        let linear_red = g.reduced.len() == 1;
        let src_origin = g.source.offset() as isize;
        let dst_origin = g.destination.offset() as isize;
        let (src_step, dst_step) = g
            .groups
            .first()
            .map_or((0, 0), |a| (a.source, a.destination));
        let r_step = g.reduced.first().map_or(0, |a| a.stride);
        for group in 0..total {
            let (x, y) = if linear_group {
                (
                    src_origin + group as isize * src_step,
                    (dst_origin + group as isize * dst_step) as usize,
                )
            } else {
                g.group_offsets(group)
            };
            let mut lanes = [0.0f32; N];
            if linear_red {
                for (r, lane) in lanes[..count].iter_mut().enumerate() {
                    // SAFETY: exact validated layouts keep every (group, r) coordinate in bounds.
                    *lane += unsafe { *source.get_unchecked((x + r as isize * r_step) as usize) };
                }
            } else {
                for (r, lane) in lanes[..count].iter_mut().enumerate() {
                    // SAFETY: exact validated layouts keep every (group, r) coordinate in bounds.
                    *lane += unsafe { *source.get_unchecked((x + g.reduction_offset(r)) as usize) };
                }
            }
            let sum = reduce_lanes(lanes);
            // SAFETY: injective output map guarantees unique valid write per group.
            unsafe {
                *destination.get_unchecked_mut(y) = if MEAN { sum / divisor } else { sum };
            }
        }
    }

    #[inline(never)]
    fn tree256_contiguous<const MEAN: bool>(
        g: &ReduceGeometry,
        source: &[f32],
        destination: &mut [f32],
    ) {
        let total = g.spec.output.len();
        let count = g.spec.count.unwrap_or(0);
        let divisor = g.spec.divisor();
        let linear_group = g.groups.len() <= 1;
        let src_origin = g.source.offset() as isize;
        let dst_origin = g.destination.offset() as isize;
        let (src_step, dst_step) = g
            .groups
            .first()
            .map_or((0, 0), |a| (a.source, a.destination));
        for group in 0..total {
            let (x, y) = if linear_group {
                (
                    src_origin + group as isize * src_step,
                    (dst_origin + group as isize * dst_step) as usize,
                )
            } else {
                g.group_offsets(group)
            };
            let slice = &source[x as usize..x as usize + count];
            let mut lanes = [0.0f32; 256];
            let mut chunks = slice.chunks_exact(256);
            for chunk in &mut chunks {
                for (sum, &v) in lanes.iter_mut().zip(chunk) {
                    *sum += v;
                }
            }
            for (sum, &v) in lanes.iter_mut().zip(chunks.remainder()) {
                *sum += v;
            }
            let sum = reduce_lanes(lanes);
            // SAFETY: injective output map guarantees unique valid write per group.
            unsafe {
                *destination.get_unchecked_mut(y) = if MEAN { sum / divisor } else { sum };
            }
        }
    }

    #[inline(never)]
    fn tree256_linear<const MEAN: bool>(
        g: &ReduceGeometry,
        source: &[f32],
        destination: &mut [f32],
    ) {
        let total = g.spec.output.len();
        let count = g.spec.count.unwrap_or(0);
        let divisor = g.spec.divisor();
        let r_step = g.reduced[0].stride;
        let linear_group = g.groups.len() <= 1;
        let src_origin = g.source.offset() as isize;
        let dst_origin = g.destination.offset() as isize;
        let (src_step, dst_step) = g
            .groups
            .first()
            .map_or((0, 0), |a| (a.source, a.destination));
        for group in 0..total {
            let (x, y) = if linear_group {
                (
                    src_origin + group as isize * src_step,
                    (dst_origin + group as isize * dst_step) as usize,
                )
            } else {
                g.group_offsets(group)
            };
            let mut lanes = [0.0f32; 256];
            let full = count / 256;
            let rem = count % 256;
            let mut base = 0usize;
            for _ in 0..full {
                let chunk_x = x + base as isize * r_step;
                for (lane, sum) in lanes.iter_mut().enumerate() {
                    // SAFETY: validated layouts guarantee all reduced coordinates lie in bounds.
                    *sum += unsafe {
                        *source.get_unchecked((chunk_x + lane as isize * r_step) as usize)
                    };
                }
                base += 256;
            }
            let tail_x = x + base as isize * r_step;
            for (lane, sum) in lanes[..rem].iter_mut().enumerate() {
                // SAFETY: validated layouts guarantee all reduced coordinates lie in bounds.
                *sum +=
                    unsafe { *source.get_unchecked((tail_x + lane as isize * r_step) as usize) };
            }
            let sum = reduce_lanes(lanes);
            // SAFETY: injective output map guarantees unique valid write per group.
            unsafe {
                *destination.get_unchecked_mut(y) = if MEAN { sum / divisor } else { sum };
            }
        }
    }

    #[inline(never)]
    fn tree256_general<const MEAN: bool>(
        g: &ReduceGeometry,
        source: &[f32],
        destination: &mut [f32],
    ) {
        let count = g.spec.count.unwrap_or(0);
        let divisor = g.spec.divisor();
        for group in 0..g.spec.output.len() {
            let (x, y) = g.group_offsets(group);
            let sum = tree::<256>(count, |r| {
                // SAFETY: exact bound layouts validate initialized spans.
                unsafe { *source.get_unchecked((x + g.reduction_offset(r)) as usize) }
            });
            // SAFETY: injective output map guarantees unique valid write per group.
            unsafe {
                *destination.get_unchecked_mut(y) = if MEAN { sum / divisor } else { sum };
            }
        }
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
fn min_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a < b {
        a
    } else if b < a {
        b
    } else if a.is_sign_negative() {
        a
    } else {
        b
    }
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

#[inline]
fn sum_tile_8way(xs: &[f32]) -> f32 {
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

pub(crate) struct NonLinearGroupAxis {
    pub dim: usize,
    pub source: isize,
    pub upstream: isize,
    pub destination: isize,
}

pub(crate) struct NonLinearReduceAxis {
    pub dim: usize,
    pub source: isize,
    pub destination: isize,
}

pub(crate) struct NonLinearBackwardGeometry {
    pub spec: ReduceSpec,
    pub source: Layout,
    pub upstream: Layout,
    pub destination: Layout,
    pub groups: Vec<NonLinearGroupAxis>,
    pub reduced: Vec<NonLinearReduceAxis>,
}

impl NonLinearBackwardGeometry {
    pub fn new(
        spec: &ReduceSpec,
        source: &Layout,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        if source.shape() != spec.input_shape()
            || upstream.shape() != spec.output_shape()
            || destination.shape() != spec.input_shape()
        {
            return Err(invalid(
                "nonlinear reduction backward layouts must match the exact spec shapes",
            ));
        }
        if !destination.is_non_overlapping() {
            return Err(Error::Unsupported(
                "reduction backward destination must be proved non-overlapping".into(),
            ));
        }
        let mut groups = Vec::new();
        let mut reduced = Vec::new();
        if !spec.output.is_empty() && spec.count.unwrap_or(0) > 0 {
            let mut up_axis = upstream.strides().len();
            for (axis, &dim) in spec.input.dims().iter().enumerate().rev() {
                if spec.axes.binary_search(&axis).is_ok() {
                    if spec.keep_dims {
                        up_axis -= 1;
                    }
                    if dim > 1 {
                        reduced.push(NonLinearReduceAxis {
                            dim,
                            source: source.strides()[axis],
                            destination: destination.strides()[axis],
                        });
                    }
                } else {
                    up_axis -= 1;
                    if dim > 1 {
                        groups.push(NonLinearGroupAxis {
                            dim,
                            source: source.strides()[axis],
                            upstream: upstream.strides()[up_axis],
                            destination: destination.strides()[axis],
                        });
                    }
                }
            }
            groups.reverse();
            reduced.reverse();
        }
        Ok(Self {
            spec: spec.clone(),
            source: source.clone(),
            upstream: upstream.clone(),
            destination: destination.clone(),
            groups,
            reduced,
        })
    }

    pub fn validate(&self, source: &Layout, upstream: &Layout, destination: &Layout) -> Result<()> {
        if source != &self.source || upstream != &self.upstream || destination != &self.destination
        {
            return Err(invalid(
                "reduction backward bindings differ from the prepared layouts",
            ));
        }
        Ok(())
    }

    fn group_offsets(&self, mut g: usize) -> (isize, usize, isize) {
        let (mut sx, mut su, mut sd) = (
            self.source.offset() as isize,
            self.upstream.offset() as isize,
            self.destination.offset() as isize,
        );
        for a in (0..self.groups.len()).rev() {
            let ax = &self.groups[a];
            let c = (g % ax.dim) as isize;
            g /= ax.dim;
            sx += c * ax.source;
            su += c * ax.upstream;
            sd += c * ax.destination;
        }
        (sx, su as usize, sd)
    }

    fn red_offsets(&self, mut r: usize) -> (isize, isize) {
        let (mut rx, mut rd) = (0isize, 0isize);
        for a in (0..self.reduced.len()).rev() {
            let ax = &self.reduced[a];
            let c = (r % ax.dim) as isize;
            r /= ax.dim;
            rx += c * ax.source;
            rd += c * ax.destination;
        }
        (rx, rd)
    }

    pub fn metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
            + (self.source.shape().dims().len()
                + self.upstream.shape().dims().len()
                + self.destination.shape().dims().len())
                * (size_of::<usize>() + size_of::<isize>())
            + self.groups.capacity() * size_of::<NonLinearGroupAxis>()
            + self.reduced.capacity() * size_of::<NonLinearReduceAxis>()
    }
}

pub(crate) struct BackwardGeometry {
    pub spec: ReduceSpec,
    pub upstream: Layout,
    pub map: MappingSpec,
}
impl BackwardGeometry {
    pub fn new(spec: &ReduceSpec, upstream: &Layout, destination: &Layout) -> Result<Self> {
        if upstream.shape() != spec.output_shape() || destination.shape() != spec.input_shape() {
            return Err(invalid(
                "reduction backward layouts must match the exact spec shapes",
            ));
        }
        let mut cursor = 0;
        let strides: Vec<_> = (0..spec.input.dims().len())
            .map(|axis| {
                if spec.axes.binary_search(&axis).is_ok() {
                    if spec.keep_dims {
                        cursor += 1;
                    }
                    0
                } else {
                    let stride = upstream.strides()[cursor];
                    cursor += 1;
                    stride
                }
            })
            .collect();
        let expanded = Layout::from_strides(spec.input.dims(), strides, upstream.offset())?;
        Ok(Self {
            spec: spec.clone(),
            upstream: upstream.clone(),
            map: MappingSpec::for_copy(&expanded, destination)?,
        })
    }
    pub fn validate(&self, upstream: &Layout, destination: &Layout) -> Result<()> {
        if !(std::ptr::eq(upstream, &self.upstream) || upstream == &self.upstream)
            || !(std::ptr::eq(destination, &self.map.destination)
                || destination == &self.map.destination)
        {
            return Err(invalid(
                "reduction backward bindings differ from the prepared layouts",
            ));
        }
        Ok(())
    }
    pub fn metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
            + self.upstream.shape().dims().len() * (size_of::<usize>() + size_of::<isize>())
            + self.map.metadata_bytes()
    }
}

/// Local reduction derivative (`Sum`/`Mean` payload-free; `Max`/`Min`/`LogSumExp` require input).
pub struct ReduceBackward {
    geometry: BackwardGeometry,
    nonlinear: Option<NonLinearBackwardGeometry>,
    kernel: BackwardKernel,
}
type BackwardKernel = fn(&BackwardGeometry, &[f32], &mut [f32], GradientMode);

impl ReduceBackward {
    pub fn new(spec: &ReduceSpec, upstream: &Layout, destination: &Layout) -> Result<Self> {
        if spec.operation.requires_input() {
            return Err(invalid(
                "Max, Min, and LogSumExp backward require source input layout via ReduceBackward::with_input",
            ));
        }
        Self::build_linear(spec, upstream, destination)
    }

    pub fn with_input(
        spec: &ReduceSpec,
        source: &Layout,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        if !spec.operation.requires_input() {
            if source.shape() != spec.input_shape() {
                return Err(invalid("reduction source layout differs from spec"));
            }
            return Self::build_linear(spec, upstream, destination);
        }
        let geometry = BackwardGeometry::new(spec, upstream, destination)?;
        let nonlinear = NonLinearBackwardGeometry::new(spec, source, upstream, destination)?;
        Ok(Self {
            geometry,
            nonlinear: Some(nonlinear),
            kernel: Self::empty,
        })
    }

    fn build_linear(spec: &ReduceSpec, upstream: &Layout, destination: &Layout) -> Result<Self> {
        let geometry = BackwardGeometry::new(spec, upstream, destination)?;
        let divide = geometry.spec.operation == ReduceOp::Mean && geometry.spec.count != Some(1);
        let kernel: BackwardKernel = if geometry.map.len == 0 {
            Self::empty
        } else {
            match (divide, geometry.map.len >= 16) {
                (false, false) => Self::scalar::<false>,
                (true, false) => Self::scalar::<true>,
                (false, true) => Self::rows::<false>,
                (true, true) => Self::rows::<true>,
            }
        };
        Ok(Self {
            geometry,
            nonlinear: None,
            kernel,
        })
    }

    pub fn input_required(&self) -> bool {
        self.geometry.spec.operation.requires_input()
    }
    pub fn spec(&self) -> &ReduceSpec {
        &self.geometry.spec
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.geometry.upstream
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.map.destination
    }
    pub fn host_metadata_bytes(&self) -> usize {
        if let Some(nl) = &self.nonlinear {
            nl.metadata_bytes()
        } else {
            self.geometry.metadata_bytes()
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn run(
        &self,
        upstream: &TensorView<'_>,
        destination: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        if self.input_required() {
            return Err(invalid(
                "Max, Min, and LogSumExp backward require source input via run_with_input",
            ));
        }
        let g = &self.geometry;
        g.validate(upstream.layout, destination.layout)?;
        (self.kernel)(g, upstream.data, destination.data, mode);
        Ok(())
    }

    pub fn run_with_input(
        &self,
        source: Option<&TensorView<'_>>,
        upstream: &TensorView<'_>,
        destination: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        let Some(nl) = &self.nonlinear else {
            return self.run(upstream, destination, mode);
        };
        let src = source.ok_or_else(|| {
            invalid("Max, Min, and LogSumExp backward require saved forward source input")
        })?;
        nl.validate(src.layout, upstream.layout, destination.layout)?;
        let groups = nl.spec.output.len();
        let count = nl.spec.count.unwrap_or(0);
        if groups == 0 || count == 0 {
            return Ok(());
        }
        let accumulate = mode == GradientMode::Accumulate;
        let x_data = src.data;
        let dy_data = upstream.data;
        let dx_data = &mut *destination.data;
        let contig_red =
            nl.reduced.len() == 1 && nl.reduced[0].source == 1 && nl.reduced[0].destination == 1;
        match nl.spec.operation {
            ReduceOp::Max | ReduceOp::Min => {
                let is_max = nl.spec.operation == ReduceOp::Max;
                for g in 0..groups {
                    let (sx, su, sd) = nl.group_offsets(g);
                    let u = dy_data[su];
                    if contig_red {
                        let xs = &x_data[sx as usize..sx as usize + count];
                        let dxs = &mut dx_data[sd as usize..sd as usize + count];
                        let mut extremum = xs[0];
                        for &v in &xs[1..] {
                            extremum = if is_max {
                                max_nan(extremum, v)
                            } else {
                                min_nan(extremum, v)
                            };
                        }
                        if extremum.is_nan() {
                            if accumulate {
                                for dst in dxs.iter_mut() {
                                    *dst += f32::NAN;
                                }
                            } else {
                                dxs.fill(f32::NAN);
                            }
                            continue;
                        }
                        let mut ties = 0u64;
                        for &v in xs {
                            if v == extremum {
                                ties += 1;
                            }
                        }
                        let scale = u / ties as f32;
                        if accumulate {
                            for (dst, &v) in dxs.iter_mut().zip(xs) {
                                if v == extremum {
                                    *dst += scale;
                                }
                            }
                        } else {
                            for (dst, &v) in dxs.iter_mut().zip(xs) {
                                *dst = if v == extremum { scale } else { 0.0f32 };
                            }
                        }
                        continue;
                    }
                    let mut extremum = x_data[(sx + nl.red_offsets(0).0) as usize];
                    for r in 1..count {
                        let v = x_data[(sx + nl.red_offsets(r).0) as usize];
                        extremum = if is_max {
                            max_nan(extremum, v)
                        } else {
                            min_nan(extremum, v)
                        };
                    }
                    if extremum.is_nan() {
                        for r in 0..count {
                            let (_, rd) = nl.red_offsets(r);
                            let dst = (sd + rd) as usize;
                            if accumulate {
                                dx_data[dst] += f32::NAN;
                            } else {
                                dx_data[dst] = f32::NAN;
                            }
                        }
                        continue;
                    }
                    let mut ties = 0u64;
                    for r in 0..count {
                        let v = x_data[(sx + nl.red_offsets(r).0) as usize];
                        if v == extremum {
                            ties += 1;
                        }
                    }
                    let scale = u / ties as f32;
                    for r in 0..count {
                        let (rx, rd) = nl.red_offsets(r);
                        let v = x_data[(sx + rx) as usize];
                        let grad = if v == extremum { scale } else { 0.0f32 };
                        let dst = (sd + rd) as usize;
                        if accumulate {
                            dx_data[dst] += grad;
                        } else {
                            dx_data[dst] = grad;
                        }
                    }
                }
            }
            ReduceOp::LogSumExp => {
                const TILE: usize = 256;
                let mut buf = [0.0f32; TILE];
                for g in 0..groups {
                    let (sx, su, sd) = nl.group_offsets(g);
                    let u = dy_data[su];
                    if contig_red {
                        let xs = &x_data[sx as usize..sx as usize + count];
                        let dxs = &mut dx_data[sd as usize..sd as usize + count];
                        let mut m = xs[0];
                        for &v in &xs[1..] {
                            m = max_nan(m, v);
                        }
                        if m.is_nan() || m.is_infinite() {
                            for (dst, &v) in dxs.iter_mut().zip(xs) {
                                let grad =
                                    if m.is_infinite() && m.is_sign_positive() && v.is_finite() {
                                        0.0f32
                                    } else {
                                        f32::NAN
                                    };
                                if accumulate {
                                    *dst += grad;
                                } else {
                                    *dst = grad;
                                }
                            }
                            continue;
                        }
                        if count <= TILE {
                            for j in 0..count {
                                buf[j] = xs[j] - m;
                            }
                            exp_tile_in_place(&mut buf[..count]);
                            let scale = u / sum_tile_8way(&buf[..count]);
                            if accumulate {
                                for j in 0..count {
                                    dxs[j] += buf[j] * scale;
                                }
                            } else {
                                for j in 0..count {
                                    dxs[j] = buf[j] * scale;
                                }
                            }
                        } else {
                            let mut sum = 0.0f32;
                            let mut off = 0;
                            while off < count {
                                let n = (count - off).min(TILE);
                                for j in 0..n {
                                    buf[j] = xs[off + j] - m;
                                }
                                exp_tile_in_place(&mut buf[..n]);
                                sum += sum_tile_8way(&buf[..n]);
                                off += n;
                            }
                            let scale = u / sum;
                            let mut off = 0;
                            while off < count {
                                let n = (count - off).min(TILE);
                                for j in 0..n {
                                    buf[j] = xs[off + j] - m;
                                }
                                exp_tile_in_place(&mut buf[..n]);
                                if accumulate {
                                    for j in 0..n {
                                        dxs[off + j] += buf[j] * scale;
                                    }
                                } else {
                                    for j in 0..n {
                                        dxs[off + j] = buf[j] * scale;
                                    }
                                }
                                off += n;
                            }
                        }
                        continue;
                    }
                    let mut m = x_data[(sx + nl.red_offsets(0).0) as usize];
                    for r in 1..count {
                        let v = x_data[(sx + nl.red_offsets(r).0) as usize];
                        m = max_nan(m, v);
                    }
                    if m.is_nan() || m.is_infinite() {
                        for r in 0..count {
                            let (rx, rd) = nl.red_offsets(r);
                            let v = x_data[(sx + rx) as usize];
                            let grad = if m.is_infinite() && m.is_sign_positive() && v.is_finite() {
                                0.0f32
                            } else {
                                f32::NAN
                            };
                            let dst = (sd + rd) as usize;
                            if accumulate {
                                dx_data[dst] += grad;
                            } else {
                                dx_data[dst] = grad;
                            }
                        }
                        continue;
                    }
                    let mut sum = 0.0f32;
                    for r in 0..count {
                        let v = x_data[(sx + nl.red_offsets(r).0) as usize];
                        sum += (v - m).exp();
                    }
                    let scale = u / sum;
                    for r in 0..count {
                        let (rx, rd) = nl.red_offsets(r);
                        let v = x_data[(sx + rx) as usize];
                        let grad = (v - m).exp() * scale;
                        let dst = (sd + rd) as usize;
                        if accumulate {
                            dx_data[dst] += grad;
                        } else {
                            dx_data[dst] = grad;
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    #[inline(never)]
    fn empty(_: &BackwardGeometry, _: &[f32], _: &mut [f32], _: GradientMode) {}

    #[inline(never)]
    fn scalar<const DIVIDE: bool>(
        g: &BackwardGeometry,
        upstream: &[f32],
        destination: &mut [f32],
        mode: GradientMode,
    ) {
        let divisor = if DIVIDE { g.spec.divisor() } else { 1. };
        for i in 0..g.map.len {
            let (x, y) = g.map.element_offsets(i);
            // SAFETY: expansion only inserts zero-stride axes into the checked
            // upstream map. The exclusive destination is injective and in bounds.
            unsafe {
                let value = *upstream.get_unchecked(x);
                let value = if DIVIDE { value / divisor } else { value };
                let out = destination.get_unchecked_mut(y);
                *out = match mode {
                    GradientMode::Overwrite => value,
                    GradientMode::Accumulate => *out + value,
                };
            }
        }
    }

    #[inline(never)]
    fn rows<const DIVIDE: bool>(
        g: &BackwardGeometry,
        upstream: &[f32],
        destination: &mut [f32],
        mode: GradientMode,
    ) {
        let divisor = if DIVIDE { g.spec.divisor() } else { 1. };
        let sx = g.map.source_strides.last().copied().unwrap_or(0);
        let sy = g.map.destination_strides.last().copied().unwrap_or(0);
        for row in 0..g.map.rows {
            let (x, y) = g.map.row_offsets(row);
            for column in 0..g.map.inner {
                // SAFETY: run validates the expanded upstream and injective target.
                // Coalesced signed rows preserve the same checked logical coordinates.
                unsafe {
                    let value = *upstream.get_unchecked((x + column as isize * sx) as usize);
                    let value = if DIVIDE { value / divisor } else { value };
                    let out = destination.get_unchecked_mut((y + column as isize * sy) as usize);
                    *out = match mode {
                        GradientMode::Overwrite => value,
                        GradientMode::Accumulate => *out + value,
                    };
                }
            }
        }
    }
}
