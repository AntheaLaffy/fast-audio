use crate::{
    BinaryOp, Error, GradientMode, Layout, ReduceOp, ReduceSpec, ReduceStrategy, Result,
    TensorView, TensorViewMut, UnaryOp,
    error::invalid,
    pointwise::{broadcast_dims, sigmoid},
    reduce::tree,
};

/// Select the logical binary operand whose gradient is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum BinaryOperand {
    Left,
    Right,
}

#[derive(Clone, Copy)]
pub(crate) enum Formula {
    Unary(UnaryOp),
    Binary(BinaryOp, BinaryOperand),
}
impl Formula {
    pub fn code(self) -> (i32, i32, i32) {
        match self {
            Self::Unary(op) => (op as i32, 1, 0),
            Self::Binary(op, side) => (op as i32, 2, side as i32),
        }
    }
}
pub(crate) const fn dependencies(op: i32, arity: i32, side: i32) -> [bool; 2] {
    if arity == 1 {
        [op != 0, false]
    } else {
        match op {
            0 | 1 => [false, false],
            2 => [side == 1, side == 0],
            _ => [side == 1, true],
        }
    }
}
pub(crate) struct GroupAxis {
    pub dim: usize,
    pub reads: [isize; 3],
    pub output: isize,
}
pub(crate) struct ReducedAxis {
    pub dim: usize,
    pub reads: [isize; 3],
}
pub(crate) struct Geometry {
    pub formula: Formula,
    pub inputs: [Option<Layout>; 2],
    pub upstream: Layout,
    pub destination: Layout,
    pub contraction: ReduceSpec,
    pub dependencies: [bool; 2],
    pub origins: [isize; 4],
    pub groups: Vec<GroupAxis>,
    pub reduced: Vec<ReducedAxis>,
}
type CpuKernel = fn(&Geometry, [&[f32]; 3], &mut [f32], GradientMode);

impl Geometry {
    pub fn new(
        formula: Formula,
        left: &Layout,
        right: Option<&Layout>,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let (op, arity, side) = formula.code();
        let common = if let Some(right) = right {
            broadcast_dims(&[left, right])?
        } else {
            left.shape().dims().to_vec()
        };
        if upstream.shape().dims() != common {
            return Err(invalid(
                "pointwise backward upstream must have the exact common shape",
            ));
        }
        let target = if side == 1 { right.unwrap() } else { left };
        if destination.shape() != target.shape() {
            return Err(invalid(
                "pointwise gradient must match the exact original operand shape",
            ));
        }
        if !destination.is_non_overlapping() {
            return Err(Error::Unsupported(
                "pointwise gradient destination must be proved non-overlapping".into(),
            ));
        }
        let prefix = common.len() - target.shape().dims().len();
        let axes: Vec<_> = (0..common.len())
            .filter(|&a| a < prefix || (target.shape().dims()[a - prefix] == 1 && common[a] != 1))
            .collect();
        let contraction = ReduceSpec::new(ReduceOp::Sum, upstream.shape(), &axes, true)?;
        let expanded_destination = destination.broadcast_to(contraction.output_shape().dims())?;
        let left_expanded = left.broadcast_to(&common)?;
        let right_expanded = right.map(|r| r.broadcast_to(&common)).transpose()?;
        let active = !upstream.shape().is_empty() && !destination.shape().is_empty();
        let dependencies = dependencies(op, arity, side).map(|need| need && active);
        let origins = [
            upstream.offset() as isize,
            if dependencies[0] {
                left.offset() as isize
            } else {
                0
            },
            if dependencies[1] {
                right.unwrap().offset() as isize
            } else {
                0
            },
            destination.offset() as isize,
        ];
        let mut groups: Vec<GroupAxis> = Vec::new();
        let mut reduced: Vec<ReducedAxis> = Vec::new();
        if !destination.shape().is_empty() {
            for axis in (0..common.len()).rev() {
                let dim = common[axis];
                if dim <= 1 {
                    continue;
                }
                let reads = if active {
                    [
                        upstream.strides()[axis],
                        if dependencies[0] {
                            left_expanded.strides()[axis]
                        } else {
                            0
                        },
                        if dependencies[1] {
                            right_expanded.as_ref().unwrap().strides()[axis]
                        } else {
                            0
                        },
                    ]
                } else {
                    [0; 3]
                };
                if axes.binary_search(&axis).is_ok() {
                    if contraction.reduction_len() == Some(0) {
                        continue;
                    }
                    if let Some(inner) = reduced.last_mut() {
                        if inner
                            .reads
                            .iter()
                            .zip(reads)
                            .all(|(&s, t)| s.checked_mul(inner.dim as isize) == Some(t))
                        {
                            inner.dim *= dim;
                            continue;
                        }
                    }
                    reduced.push(ReducedAxis { dim, reads });
                } else {
                    let output = expanded_destination.strides()[axis];
                    if let Some(inner) = groups.last_mut() {
                        if inner.output.checked_mul(inner.dim as isize) == Some(output)
                            && inner
                                .reads
                                .iter()
                                .zip(reads)
                                .all(|(&s, t)| s.checked_mul(inner.dim as isize) == Some(t))
                        {
                            inner.dim *= dim;
                            continue;
                        }
                    }
                    groups.push(GroupAxis { dim, reads, output });
                }
            }
            groups.reverse();
            reduced.reverse();
        }
        Ok(Self {
            formula,
            inputs: [Some(left.clone()), right.cloned()],
            upstream: upstream.clone(),
            destination: destination.clone(),
            contraction,
            dependencies,
            origins,
            groups,
            reduced,
        })
    }
    pub fn validate(
        &self,
        inputs: [Option<&Layout>; 2],
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<()> {
        for (i, source) in inputs.into_iter().enumerate() {
            if let Some(source) = source {
                if !self.inputs[i]
                    .as_ref()
                    .is_some_and(|expected| std::ptr::eq(source, expected) || source == expected)
                {
                    return Err(invalid(
                        "pointwise backward supplied input geometry changed",
                    ));
                }
            } else if self.dependencies[i] {
                return Err(invalid(
                    "pointwise backward required input payload is missing",
                ));
            }
        }
        if !(std::ptr::eq(upstream, &self.upstream) || upstream == &self.upstream)
            || !(std::ptr::eq(destination, &self.destination) || destination == &self.destination)
        {
            return Err(invalid(
                "pointwise backward upstream/output bindings changed",
            ));
        }
        Ok(())
    }
    pub fn metadata_bytes(&self) -> usize {
        self.contraction.metadata_bytes()
            + self
                .inputs
                .iter()
                .flatten()
                .chain([&self.upstream, &self.destination])
                .map(|s| s.shape().dims().len() * (size_of::<usize>() + size_of::<isize>()))
                .sum::<usize>()
            + self.groups.capacity() * size_of::<GroupAxis>()
            + self.reduced.capacity() * size_of::<ReducedAxis>()
    }
    fn group_offsets(&self, mut flat: usize) -> ([isize; 3], usize) {
        let mut reads = [self.origins[0], self.origins[1], self.origins[2]];
        let mut out = self.origins[3];
        for a in (1..self.groups.len()).rev() {
            let axis = &self.groups[a];
            let c = (flat % axis.dim) as isize;
            flat /= axis.dim;
            for (offset, step) in reads.iter_mut().zip(axis.reads) {
                *offset += c * step;
            }
            out += c * axis.output;
        }
        if let Some(axis) = self.groups.first() {
            for (offset, step) in reads.iter_mut().zip(axis.reads) {
                *offset += flat as isize * step;
            }
            out += flat as isize * axis.output;
        }
        (reads, out as usize)
    }
    fn reduce_offsets(&self, mut flat: usize) -> [isize; 3] {
        let mut reads = [0; 3];
        for a in (1..self.reduced.len()).rev() {
            let axis = &self.reduced[a];
            let c = (flat % axis.dim) as isize;
            flat /= axis.dim;
            for (offset, step) in reads.iter_mut().zip(axis.reads) {
                *offset += c * step;
            }
        }
        if let Some(axis) = self.reduced.first() {
            for (offset, step) in reads.iter_mut().zip(axis.reads) {
                *offset += flat as isize * step;
            }
        }
        reads
    }
    fn execute(
        &self,
        kernel: CpuKernel,
        inputs: [Option<&TensorView<'_>>; 2],
        upstream: &TensorView<'_>,
        destination: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.validate(
            inputs.map(|v| v.map(|s| s.layout())),
            upstream.layout,
            destination.layout,
        )?;
        let data = [
            upstream.data,
            inputs[0].map_or(&[][..], |s| s.data),
            inputs[1].map_or(&[][..], |s| s.data),
        ];
        kernel(self, data, destination.data, mode);
        Ok(())
    }

    fn cpu_kernel(&self) -> CpuKernel {
        macro_rules! unary {
            ($op:ident) => {
                self.select_kernel::<{ UnaryOp::$op as i32 }, 1, 0>()
            };
        }
        macro_rules! binary {
            ($op:ident, $side:expr) => {
                self.select_kernel::<{ BinaryOp::$op as i32 }, 2, $side>()
            };
        }
        match self.formula {
            Formula::Unary(op) => match op {
                UnaryOp::Neg => unary!(Neg),
                UnaryOp::Abs => unary!(Abs),
                UnaryOp::Square => unary!(Square),
                UnaryOp::Reciprocal => unary!(Reciprocal),
                UnaryOp::Sqrt => unary!(Sqrt),
                UnaryOp::Exp => unary!(Exp),
                UnaryOp::Log => unary!(Log),
                UnaryOp::Relu => unary!(Relu),
                UnaryOp::Sigmoid => unary!(Sigmoid),
                UnaryOp::Tanh => unary!(Tanh),
                UnaryOp::Silu => unary!(Silu),
                UnaryOp::Softplus => unary!(Softplus),
            },
            Formula::Binary(op, side) => match (op, side) {
                (BinaryOp::Add, BinaryOperand::Left) => binary!(Add, 0),
                (BinaryOp::Add, BinaryOperand::Right) => binary!(Add, 1),
                (BinaryOp::Sub, BinaryOperand::Left) => binary!(Sub, 0),
                (BinaryOp::Sub, BinaryOperand::Right) => binary!(Sub, 1),
                (BinaryOp::Mul, BinaryOperand::Left) => binary!(Mul, 0),
                (BinaryOp::Mul, BinaryOperand::Right) => binary!(Mul, 1),
                (BinaryOp::Div, BinaryOperand::Left) => binary!(Div, 0),
                (BinaryOp::Div, BinaryOperand::Right) => binary!(Div, 1),
            },
        }
    }

    fn select_kernel<const OP: i32, const ARITY: i32, const SIDE: i32>(&self) -> CpuKernel {
        match self.contraction.strategy() {
            ReduceStrategy::Empty | ReduceStrategy::EmptyReduction => Self::empty,
            ReduceStrategy::Copy if self.destination.shape().len() < 16 => {
                Self::copy_scalar::<OP, ARITY, SIDE>
            }
            ReduceStrategy::Copy => Self::copy_rows::<OP, ARITY, SIDE>,
            ReduceStrategy::Tree32 => Self::contract::<OP, ARITY, SIDE, 32>,
            ReduceStrategy::Tree256 => Self::contract::<OP, ARITY, SIDE, 256>,
        }
    }

    // Selected bodies stay separate so Copy calls never reserve a contraction frame.
    #[inline(never)]
    fn empty(g: &Self, _: [&[f32]; 3], destination: &mut [f32], mode: GradientMode) {
        if mode == GradientMode::Accumulate {
            return;
        }
        for group in 0..g.destination.shape().len() {
            put(destination, g.group_offsets(group).1, 0., mode);
        }
    }

    #[inline(never)]
    fn copy_scalar<const OP: i32, const ARITY: i32, const SIDE: i32>(
        g: &Self,
        data: [&[f32]; 3],
        destination: &mut [f32],
        mode: GradientMode,
    ) {
        for group in 0..g.destination.shape().len() {
            let (offsets, target) = g.group_offsets(group);
            put(
                destination,
                target,
                local::<OP, ARITY, SIDE>(&data, offsets),
                mode,
            );
        }
    }

    #[inline(never)]
    fn copy_rows<const OP: i32, const ARITY: i32, const SIDE: i32>(
        g: &Self,
        data: [&[f32]; 3],
        destination: &mut [f32],
        mode: GradientMode,
    ) {
        let inner = g.groups.last().map_or(1, |a| a.dim);
        let strides = g.groups.last().map_or([0; 3], |a| a.reads);
        let output_stride = g.groups.last().map_or(0, |a| a.output);
        let rows = g.destination.shape().len() / inner;
        #[cfg(feature = "mkl")]
        if ARITY == 1
            && matches!(OP, 5 | 8 | 9 | 10 | 11)
            && inner >= 64
            && strides[0] == 1
            && strides[1] == 1
            && output_stride == 1
        {
            let mut buf = [0.0f32; 256];
            let mut q = [0.0f32; 256];
            for row in 0..rows {
                let (base, target) = g.group_offsets(row * inner);
                let dy_row = &data[0][base[0] as usize..base[0] as usize + inner];
                let x_row = &data[1][base[1] as usize..base[1] as usize + inner];
                let out_row = &mut destination[target..target + inner];
                for ((out_c, dy_c), x_c) in out_row
                    .chunks_mut(256)
                    .zip(dy_row.chunks(256))
                    .zip(x_row.chunks(256))
                {
                    let n = out_c.len();
                    if OP == 5 {
                        crate::pointwise::vml::exp(x_c, &mut q[..n]);
                    } else {
                        for (dst, &x) in buf[..n].iter_mut().zip(x_c) {
                            *dst = -x.abs();
                        }
                        crate::pointwise::vml::exp(&buf[..n], &mut q[..n]);
                    }
                    for (((o, &d), &x), &qv) in out_c.iter_mut().zip(dy_c).zip(x_c).zip(&q[..n]) {
                        let contrib = match OP {
                            5 => qv * d,
                            8 => {
                                let t = 1.0 + qv;
                                ((qv / t) / t) * d
                            }
                            9 => {
                                let u = (2.0 * qv) / (1.0 + qv * qv);
                                (u * u) * d
                            }
                            10 => {
                                let factor = if x == f32::INFINITY {
                                    1.0
                                } else if x == f32::NEG_INFINITY {
                                    -0.0
                                } else {
                                    let sig = if x >= 0.0 {
                                        1.0 / (1.0 + qv)
                                    } else {
                                        qv / (1.0 + qv)
                                    };
                                    let t = 1.0 + qv;
                                    sig + x * ((qv / t) / t)
                                };
                                factor * d
                            }
                            _ => {
                                let sig = if x >= 0.0 {
                                    1.0 / (1.0 + qv)
                                } else {
                                    qv / (1.0 + qv)
                                };
                                sig * d
                            }
                        };
                        *o = match mode {
                            GradientMode::Overwrite => contrib,
                            GradientMode::Accumulate => *o + contrib,
                        };
                    }
                }
            }
            return;
        }
        let req = dependencies(OP, ARITY, SIDE);
        let dense_ok = output_stride == 1
            && strides[0] == 1
            && (!req[0] || strides[1] == 0 || strides[1] == 1)
            && (!req[1] || strides[2] == 0 || strides[2] == 1);
        if dense_ok {
            for row in 0..rows {
                let (base, target) = g.group_offsets(row * inner);
                let dy_row = &data[0][base[0] as usize..base[0] as usize + inner];
                let out_row = &mut destination[target..target + inner];
                let a_row = if req[0] {
                    &data[1][base[1] as usize
                        ..base[1] as usize + if strides[1] == 0 { 1 } else { inner }]
                } else {
                    &[][..]
                };
                let b_row = if req[1] {
                    &data[2][base[2] as usize
                        ..base[2] as usize + if strides[2] == 0 { 1 } else { inner }]
                } else {
                    &[][..]
                };
                let a_bcast = req[0] && strides[1] == 0;
                let b_bcast = req[1] && strides[2] == 0;
                let a0 = if a_bcast { a_row[0] } else { 0.0 };
                let b0 = if b_bcast { b_row[0] } else { 0.0 };
                match mode {
                    GradientMode::Overwrite => {
                        for (col, (o, &d)) in out_row.iter_mut().zip(dy_row).enumerate() {
                            let a = if !req[0] {
                                0.0
                            } else if a_bcast {
                                a0
                            } else {
                                a_row[col]
                            };
                            let b = if !req[1] {
                                0.0
                            } else if b_bcast {
                                b0
                            } else {
                                b_row[col]
                            };
                            *o = if ARITY == 1 {
                                unary::<OP>(d, a)
                            } else {
                                binary::<OP, SIDE>(d, a, b)
                            };
                        }
                    }
                    GradientMode::Accumulate => {
                        for (col, (o, &d)) in out_row.iter_mut().zip(dy_row).enumerate() {
                            let a = if !req[0] {
                                0.0
                            } else if a_bcast {
                                a0
                            } else {
                                a_row[col]
                            };
                            let b = if !req[1] {
                                0.0
                            } else if b_bcast {
                                b0
                            } else {
                                b_row[col]
                            };
                            *o += if ARITY == 1 {
                                unary::<OP>(d, a)
                            } else {
                                binary::<OP, SIDE>(d, a, b)
                            };
                        }
                    }
                }
            }
            return;
        }
        match mode {
            GradientMode::Overwrite => {
                for row in 0..rows {
                    let (base, target) = g.group_offsets(row * inner);
                    for column in 0..inner {
                        let offsets = [
                            base[0] + column as isize * strides[0],
                            base[1] + column as isize * strides[1],
                            base[2] + column as isize * strides[2],
                        ];
                        let dst = (target as isize + column as isize * output_stride) as usize;
                        // SAFETY: injective output map guarantees unique valid write per coordinate.
                        unsafe {
                            *destination.get_unchecked_mut(dst) =
                                local::<OP, ARITY, SIDE>(&data, offsets);
                        }
                    }
                }
            }
            GradientMode::Accumulate => {
                for row in 0..rows {
                    let (base, target) = g.group_offsets(row * inner);
                    for column in 0..inner {
                        let offsets = [
                            base[0] + column as isize * strides[0],
                            base[1] + column as isize * strides[1],
                            base[2] + column as isize * strides[2],
                        ];
                        let dst = (target as isize + column as isize * output_stride) as usize;
                        // SAFETY: injective output map guarantees unique valid write per coordinate.
                        unsafe {
                            let out = destination.get_unchecked_mut(dst);
                            *out += local::<OP, ARITY, SIDE>(&data, offsets);
                        }
                    }
                }
            }
        }
    }

    #[inline(never)]
    fn contract<const OP: i32, const ARITY: i32, const SIDE: i32, const LANES: usize>(
        g: &Self,
        data: [&[f32]; 3],
        destination: &mut [f32],
        mode: GradientMode,
    ) {
        let count = g.contraction.reduction_len().unwrap_or(0);
        let total = g.destination.shape().len();
        if g.reduced.len() == 1 {
            let rs = g.reduced[0].reads;
            let full = count / LANES;
            let rem = count % LANES;
            for group in 0..total {
                let (base, target) = g.group_offsets(group);
                let contribution = if LANES == 32 && count == 2 {
                    let v0 = local::<OP, ARITY, SIDE>(&data, base);
                    let v1 = local::<OP, ARITY, SIDE>(
                        &data,
                        [base[0] + rs[0], base[1] + rs[1], base[2] + rs[2]],
                    );
                    (0.0f32 + v0) + (0.0f32 + v1)
                } else {
                    let mut lanes = [0.0f32; LANES];
                    let mut r_base = 0usize;
                    for _ in 0..full {
                        let cb = [
                            base[0] + r_base as isize * rs[0],
                            base[1] + r_base as isize * rs[1],
                            base[2] + r_base as isize * rs[2],
                        ];
                        for (lane, sum) in lanes.iter_mut().enumerate() {
                            let off = [
                                cb[0] + lane as isize * rs[0],
                                cb[1] + lane as isize * rs[1],
                                cb[2] + lane as isize * rs[2],
                            ];
                            *sum += local::<OP, ARITY, SIDE>(&data, off);
                        }
                        r_base += LANES;
                    }
                    let tb = [
                        base[0] + r_base as isize * rs[0],
                        base[1] + r_base as isize * rs[1],
                        base[2] + r_base as isize * rs[2],
                    ];
                    for (lane, sum) in lanes[..rem].iter_mut().enumerate() {
                        let off = [
                            tb[0] + lane as isize * rs[0],
                            tb[1] + lane as isize * rs[1],
                            tb[2] + lane as isize * rs[2],
                        ];
                        *sum += local::<OP, ARITY, SIDE>(&data, off);
                    }
                    crate::reduce::reduce_lanes(lanes)
                };
                put(destination, target, contribution, mode);
            }
        } else if g.reduced.len() == 2 {
            let r0 = &g.reduced[0];
            let r1 = &g.reduced[1];
            for group in 0..total {
                let (base, target) = g.group_offsets(group);
                let mut lanes = [0.0f32; LANES];
                let mut lane = 0usize;
                for outer in 0..r0.dim {
                    let ob = [
                        base[0] + outer as isize * r0.reads[0],
                        base[1] + outer as isize * r0.reads[1],
                        base[2] + outer as isize * r0.reads[2],
                    ];
                    let mut inner = 0usize;
                    if lane != 0 {
                        let take = (LANES - lane).min(r1.dim);
                        for k in 0..take {
                            let off = [
                                ob[0] + k as isize * r1.reads[0],
                                ob[1] + k as isize * r1.reads[1],
                                ob[2] + k as isize * r1.reads[2],
                            ];
                            lanes[lane + k] += local::<OP, ARITY, SIDE>(&data, off);
                        }
                        inner = take;
                        lane = (lane + take) % LANES;
                    }
                    while inner + LANES <= r1.dim {
                        let cb = [
                            ob[0] + inner as isize * r1.reads[0],
                            ob[1] + inner as isize * r1.reads[1],
                            ob[2] + inner as isize * r1.reads[2],
                        ];
                        for (k, sum) in lanes.iter_mut().enumerate() {
                            let off = [
                                cb[0] + k as isize * r1.reads[0],
                                cb[1] + k as isize * r1.reads[1],
                                cb[2] + k as isize * r1.reads[2],
                            ];
                            *sum += local::<OP, ARITY, SIDE>(&data, off);
                        }
                        inner += LANES;
                    }
                    let tail = r1.dim - inner;
                    if tail > 0 {
                        let cb = [
                            ob[0] + inner as isize * r1.reads[0],
                            ob[1] + inner as isize * r1.reads[1],
                            ob[2] + inner as isize * r1.reads[2],
                        ];
                        for (k, sum) in lanes[..tail].iter_mut().enumerate() {
                            let off = [
                                cb[0] + k as isize * r1.reads[0],
                                cb[1] + k as isize * r1.reads[1],
                                cb[2] + k as isize * r1.reads[2],
                            ];
                            *sum += local::<OP, ARITY, SIDE>(&data, off);
                        }
                        lane = tail;
                    }
                }
                put(
                    destination,
                    target,
                    crate::reduce::reduce_lanes(lanes),
                    mode,
                );
            }
        } else {
            for group in 0..total {
                let (base, target) = g.group_offsets(group);
                let contribution = tree::<LANES>(count, |r| {
                    let offsets = g.reduce_offsets(r);
                    local::<OP, ARITY, SIDE>(&data, std::array::from_fn(|j| base[j] + offsets[j]))
                });
                put(destination, target, contribution, mode);
            }
        }
    }
}

#[inline]
fn local<const OP: i32, const ARITY: i32, const SIDE: i32>(
    data: &[&[f32]; 3],
    offsets: [isize; 3],
) -> f32 {
    let required = dependencies(OP, ARITY, SIDE);
    let mut values = [0.; 3];
    for j in 0..3 {
        if j == 0 || required[j - 1] {
            // SAFETY: execute validates all bound layouts and dependencies before
            // invoking a selected kernel; scalar, row and tree maps preserve them.
            values[j] = unsafe { *data[j].get_unchecked(offsets[j] as usize) };
        }
    }
    if ARITY == 1 {
        unary::<OP>(values[0], values[1])
    } else {
        binary::<OP, SIDE>(values[0], values[1], values[2])
    }
}

#[inline]
fn put(destination: &mut [f32], target: usize, contribution: f32, mode: GradientMode) {
    // SAFETY: selected kernels derive target from the validated injective output
    // map, so no output hole or logical coordinate outside that map is visited.
    let out = unsafe { destination.get_unchecked_mut(target) };
    *out = match mode {
        GradientMode::Overwrite => contribution,
        GradientMode::Accumulate => *out + contribution,
    };
}
fn neg(x: f32) -> f32 {
    f32::from_bits(x.to_bits() ^ 0x80000000)
}
fn sigmoid_derivative(x: f32) -> f32 {
    let q = (-x.abs()).exp();
    let t = 1. + q;
    (q / t) / t
}
fn tanh_derivative(x: f32) -> f32 {
    let q = (-x.abs()).exp();
    let u = (2. * q) / (1. + q * q);
    u * u
}
fn unary<const OP: i32>(d: f32, x: f32) -> f32 {
    match OP {
        0 => neg(d),
        1 => {
            if x.is_nan() {
                f32::from_bits(0x7fc00000)
            } else if x > 0. {
                d
            } else if x < 0. {
                neg(d)
            } else {
                0.
            }
        }
        2 => (2. * x) * d,
        3 => neg((d / x) / x),
        4 => d / (2. * x.sqrt()),
        5 => x.exp() * d,
        6 => d / x,
        7 => {
            if x.is_nan() {
                f32::from_bits(0x7fc00000)
            } else if x > 0. {
                d
            } else {
                0.
            }
        }
        8 => sigmoid_derivative(x) * d,
        9 => tanh_derivative(x) * d,
        10 => {
            let factor = if x == f32::INFINITY {
                1.
            } else if x == f32::NEG_INFINITY {
                -0.
            } else {
                sigmoid(x) + x * sigmoid_derivative(x)
            };
            factor * d
        }
        11 => sigmoid(x) * d,
        _ => unreachable!(),
    }
}
fn binary<const OP: i32, const SIDE: i32>(d: f32, a: f32, b: f32) -> f32 {
    match (OP, SIDE) {
        (0, _) | (1, 0) => d,
        (1, _) => neg(d),
        (2, 0) => d * b,
        (2, _) => d * a,
        (3, 0) => d / b,
        (3, _) => neg((d / b) * (a / b)),
        _ => unreachable!(),
    }
}

/// Explicit unary local VJP. Required input values remain borrowed from the caller.
pub struct UnaryBackward {
    geometry: Geometry,
    kernel: CpuKernel,
}
impl UnaryBackward {
    pub fn new(
        operation: UnaryOp,
        input: &Layout,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let geometry = Geometry::new(
            Formula::Unary(operation),
            input,
            None,
            upstream,
            destination,
        )?;
        let kernel = geometry.cpu_kernel();
        Ok(Self { geometry, kernel })
    }
    pub fn operation(&self) -> UnaryOp {
        match self.geometry.formula {
            Formula::Unary(op) => op,
            _ => unreachable!(),
        }
    }
    pub fn input_required(&self) -> bool {
        self.geometry.dependencies[0]
    }
    pub fn input_layout(&self) -> &Layout {
        self.geometry.inputs[0].as_ref().unwrap()
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.geometry.upstream
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.destination
    }
    pub fn contraction(&self) -> &ReduceSpec {
        &self.geometry.contraction
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.geometry.metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn run(
        &self,
        input: Option<&TensorView<'_>>,
        upstream: &TensorView<'_>,
        destination: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.geometry
            .execute(self.kernel, [input, None], upstream, destination, mode)
    }
}

/// One operand's local VJP with fused deterministic broadcast contraction.
pub struct BinaryBackward {
    geometry: Geometry,
    kernel: CpuKernel,
}
impl BinaryBackward {
    pub fn new(
        operation: BinaryOp,
        operand: BinaryOperand,
        left: &Layout,
        right: &Layout,
        upstream: &Layout,
        destination: &Layout,
    ) -> Result<Self> {
        let geometry = Geometry::new(
            Formula::Binary(operation, operand),
            left,
            Some(right),
            upstream,
            destination,
        )?;
        let kernel = geometry.cpu_kernel();
        Ok(Self { geometry, kernel })
    }
    pub fn operation(&self) -> BinaryOp {
        match self.geometry.formula {
            Formula::Binary(op, _) => op,
            _ => unreachable!(),
        }
    }
    pub fn operand(&self) -> BinaryOperand {
        match self.geometry.formula {
            Formula::Binary(_, side) => side,
            _ => unreachable!(),
        }
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        self.geometry.dependencies
    }
    pub fn left_layout(&self) -> &Layout {
        self.geometry.inputs[0].as_ref().unwrap()
    }
    pub fn right_layout(&self) -> &Layout {
        self.geometry.inputs[1].as_ref().unwrap()
    }
    pub fn upstream_layout(&self) -> &Layout {
        &self.geometry.upstream
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.geometry.destination
    }
    pub fn contraction(&self) -> &ReduceSpec {
        &self.geometry.contraction
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.geometry.metadata_bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn run(
        &self,
        left: Option<&TensorView<'_>>,
        right: Option<&TensorView<'_>>,
        upstream: &TensorView<'_>,
        destination: &mut TensorViewMut<'_>,
        mode: GradientMode,
    ) -> Result<()> {
        self.geometry
            .execute(self.kernel, [left, right], upstream, destination, mode)
    }
}
