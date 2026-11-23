use crate::{
    BinaryOperand, Error, GradientMode, Layout, Result, Shape, TensorView, TensorViewMut,
    error::invalid,
};

/// Boundary extension policy for `Pad`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PadMode {
    Constant(f32),
    Reflect,
    Replicate,
}

impl PadMode {
    #[cfg(feature = "cuda")]
    pub(crate) fn code(self) -> i32 {
        match self {
            Self::Constant(_) => 0,
            Self::Reflect => 1,
            Self::Replicate => 2,
        }
    }
    pub(crate) fn constant_value(self) -> f32 {
        match self {
            Self::Constant(v) => v,
            _ => 0.0,
        }
    }
}

/// Validated arbitrary-rank per-axis padding geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct PadSpec {
    input: Shape,
    output: Shape,
    pads: Vec<(usize, usize)>,
    mode: PadMode,
}

impl PadSpec {
    pub fn new(input: &Shape, pads: &[(usize, usize)], mode: PadMode) -> Result<Self> {
        let dims = input.dims();
        if dims.is_empty() || dims.len() > 64 || pads.len() != dims.len() {
            return Err(invalid(
                "pad requires 1 <= rank <= 64 and one (before, after) pair per axis",
            ));
        }
        let mut out_dims = Vec::with_capacity(dims.len());
        for (&d, &(pb, pa)) in dims.iter().zip(pads) {
            if pb > 0 || pa > 0 {
                match mode {
                    PadMode::Constant(_) => {}
                    PadMode::Reflect => {
                        if d <= 1 || pb >= d || pa >= d {
                            return Err(invalid(
                                "reflect padding requires pad < dim and dim > 1 on padded axes",
                            ));
                        }
                    }
                    PadMode::Replicate => {
                        if d == 0 {
                            return Err(invalid(
                                "replicate padding requires positive dim on padded axes",
                            ));
                        }
                    }
                }
            }
            let od = d
                .checked_add(pb)
                .and_then(|v| v.checked_add(pa))
                .ok_or_else(|| invalid("padded dimension overflows"))?;
            out_dims.push(od);
        }
        let output = Shape::new(out_dims)?;
        Ok(Self {
            input: input.clone(),
            output,
            pads: pads.to_vec(),
            mode,
        })
    }

    pub fn input_shape(&self) -> &Shape {
        &self.input
    }
    pub fn output_shape(&self) -> &Shape {
        &self.output
    }
    pub fn pads(&self) -> &[(usize, usize)] {
        &self.pads
    }
    pub fn mode(&self) -> PadMode {
        self.mode
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        size_of::<Self>()
            + (self.input.dims().len() + self.output.dims().len()) * size_of::<usize>()
            + self.pads.len() * size_of::<(usize, usize)>()
    }
}

/// Prepared arbitrary-rank strided `Pad` forward plan on CPU (zero hot allocations).
pub struct PadPlan {
    spec: PadSpec,
    input_layout: Layout,
    output_layout: Layout,
}

impl PadPlan {
    pub fn new(spec: &PadSpec, input: &Layout, output: &Layout) -> Result<Self> {
        if input.shape() != &spec.input || output.shape() != &spec.output {
            return Err(invalid("pad layouts differ from spec"));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "pad output layout must be proved non-overlapping".into(),
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
        })
    }

    pub fn spec(&self) -> &PadSpec {
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
            return Err(invalid("pad bindings differ from prepared plan"));
        }
        let total = self.spec.output.len();
        if total == 0 {
            return Ok(());
        }
        let rank = self.spec.input.dims().len();
        let in_dims = self.spec.input.dims();
        let out_dims = self.spec.output.dims();
        let in_strides = self.input_layout.strides();
        let out_strides = self.output_layout.strides();
        let in_off = self.input_layout.offset() as isize;
        let out_off = self.output_layout.offset() as isize;
        let pads = &self.spec.pads;
        let mode = self.spec.mode;

        let src = input.data;
        let dst = &mut *output.data;

        // Fast contiguous last-axis constant/reflect/replicate path when only last axis is padded.
        if self.input_layout.is_contiguous()
            && self.output_layout.is_contiguous()
            && pads[..rank - 1].iter().all(|&(b, a)| b == 0 && a == 0)
        {
            let inner_in = in_dims[rank - 1];
            let inner_out = out_dims[rank - 1];
            let (pb, pa) = pads[rank - 1];
            let outer: usize = in_dims[..rank - 1].iter().product();
            for g in 0..outer {
                let sb = in_off as usize + g * inner_in;
                let db = out_off as usize + g * inner_out;
                let row_in = &src[sb..sb + inner_in];
                let row_out = &mut dst[db..db + inner_out];
                row_out[pb..pb + inner_in].copy_from_slice(row_in);
                match mode {
                    PadMode::Constant(val) => {
                        row_out[..pb].fill(val);
                        row_out[pb + inner_in..].fill(val);
                    }
                    PadMode::Replicate => {
                        row_out[..pb].fill(row_in[0]);
                        row_out[pb + inner_in..].fill(row_in[inner_in - 1]);
                    }
                    PadMode::Reflect => {
                        for j in 0..pb {
                            row_out[j] = row_in[pb - j];
                        }
                        for j in 0..pa {
                            row_out[pb + inner_in + j] = row_in[inner_in - 2 - j];
                        }
                    }
                }
            }
            return Ok(());
        }

        let d_last_out = out_dims[rank - 1];
        let d_last_in = in_dims[rank - 1];
        let (pb_last, pa_last) = pads[rank - 1];
        let in_s_last = in_strides[rank - 1];
        let out_s_last = out_strides[rank - 1];
        let outer = total / d_last_out;

        for g in 0..outer {
            let mut rem = g;
            let mut io = in_off;
            let mut oo = out_off;
            let mut outside = false;
            for a in (0..rank - 1).rev() {
                let d_out = out_dims[a];
                let y = rem % d_out;
                rem /= d_out;
                oo += y as isize * out_strides[a];
                let (pb, _) = pads[a];
                let d_in = in_dims[a];
                let x = if y >= pb && y < pb + d_in {
                    y - pb
                } else {
                    match mode {
                        PadMode::Constant(_) => {
                            outside = true;
                            0
                        }
                        PadMode::Replicate => {
                            if y < pb {
                                0
                            } else {
                                d_in - 1
                            }
                        }
                        PadMode::Reflect => {
                            if y < pb {
                                pb - y
                            } else {
                                2 * (d_in - 1) - (y - pb)
                            }
                        }
                    }
                };
                io += x as isize * in_strides[a];
            }
            if outside {
                let cval = mode.constant_value();
                for y in 0..d_last_out {
                    dst[(oo + y as isize * out_s_last) as usize] = cval;
                }
                continue;
            }
            match mode {
                PadMode::Constant(cval) => {
                    for y in 0..pb_last {
                        dst[(oo + y as isize * out_s_last) as usize] = cval;
                    }
                    for x in 0..d_last_in {
                        dst[(oo + (pb_last + x) as isize * out_s_last) as usize] =
                            src[(io + x as isize * in_s_last) as usize];
                    }
                    for j in 0..pa_last {
                        dst[(oo + (pb_last + d_last_in + j) as isize * out_s_last) as usize] = cval;
                    }
                }
                PadMode::Replicate => {
                    let first_val = src[io as usize];
                    for y in 0..pb_last {
                        dst[(oo + y as isize * out_s_last) as usize] = first_val;
                    }
                    for x in 0..d_last_in {
                        dst[(oo + (pb_last + x) as isize * out_s_last) as usize] =
                            src[(io + x as isize * in_s_last) as usize];
                    }
                    let last_val = src[(io + (d_last_in - 1) as isize * in_s_last) as usize];
                    for j in 0..pa_last {
                        dst[(oo + (pb_last + d_last_in + j) as isize * out_s_last) as usize] =
                            last_val;
                    }
                }
                PadMode::Reflect => {
                    for y in 0..pb_last {
                        dst[(oo + y as isize * out_s_last) as usize] =
                            src[(io + (pb_last - y) as isize * in_s_last) as usize];
                    }
                    for x in 0..d_last_in {
                        dst[(oo + (pb_last + x) as isize * out_s_last) as usize] =
                            src[(io + x as isize * in_s_last) as usize];
                    }
                    for j in 0..pa_last {
                        let x_ref = d_last_in - 2 - j;
                        dst[(oo + (pb_last + d_last_in + j) as isize * out_s_last) as usize] =
                            src[(io + x_ref as isize * in_s_last) as usize];
                    }
                }
            }
        }
        Ok(())
    }
}

/// Prepared arbitrary-rank strided `Pad` local VJP (`input_required() == false`, zero atomics).
pub struct PadBackward {
    spec: PadSpec,
    upstream_layout: Layout,
    output_layout: Layout,
}

impl PadBackward {
    pub fn new(spec: &PadSpec, upstream: &Layout, output: &Layout) -> Result<Self> {
        if upstream.shape() != &spec.output || output.shape() != &spec.input {
            return Err(invalid("pad backward layouts differ from spec"));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "pad backward output layout must be proved non-overlapping".into(),
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
        })
    }

    pub fn spec(&self) -> &PadSpec {
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
            return Err(invalid("pad backward bindings differ from prepared plan"));
        }
        let total = self.spec.input.len();
        if total == 0 {
            return Ok(());
        }
        let rank = self.spec.input.dims().len();
        let in_dims = self.spec.input.dims();
        let out_dims = self.spec.output.dims();
        let up_strides = self.upstream_layout.strides();
        let dx_strides = self.output_layout.strides();
        let up_off = self.upstream_layout.offset() as isize;
        let dx_off = self.output_layout.offset() as isize;
        let pads = &self.spec.pads;
        let acc = mode == GradientMode::Accumulate;
        let dy = upstream.data;
        let dx = &mut *output.data;

        match self.spec.mode {
            PadMode::Constant(_) => {
                for idx in 0..total {
                    let mut rem = idx;
                    let mut uo = up_off;
                    let mut xo = dx_off;
                    for a in (0..rank).rev() {
                        let x = rem % in_dims[a];
                        rem /= in_dims[a];
                        xo += x as isize * dx_strides[a];
                        uo += (x + pads[a].0) as isize * up_strides[a];
                    }
                    let v = dy[uo as usize];
                    if acc {
                        dx[xo as usize] += v;
                    } else {
                        dx[xo as usize] = v;
                    }
                }
            }
            PadMode::Reflect => {
                // Recursive gather over at most 3 contributing coordinates per axis (zero atomics).
                let mut coords = [0usize; 64];
                for idx in 0..total {
                    let mut rem = idx;
                    let mut xo = dx_off;
                    for a in (0..rank).rev() {
                        let x = rem % in_dims[a];
                        rem /= in_dims[a];
                        coords[a] = x;
                        xo += x as isize * dx_strides[a];
                    }
                    let sum = gather_reflect(
                        dy,
                        up_off,
                        0,
                        rank,
                        &coords[..rank],
                        in_dims,
                        pads,
                        up_strides,
                    );
                    if acc {
                        dx[xo as usize] += sum;
                    } else {
                        dx[xo as usize] = sum;
                    }
                }
            }
            PadMode::Replicate => {
                let mut coords = [0usize; 64];
                for idx in 0..total {
                    let mut rem = idx;
                    let mut xo = dx_off;
                    for a in (0..rank).rev() {
                        let x = rem % in_dims[a];
                        rem /= in_dims[a];
                        coords[a] = x;
                        xo += x as isize * dx_strides[a];
                    }
                    let sum = gather_replicate(
                        dy,
                        up_off,
                        0,
                        rank,
                        &coords[..rank],
                        in_dims,
                        out_dims,
                        pads,
                        up_strides,
                    );
                    if acc {
                        dx[xo as usize] += sum;
                    } else {
                        dx[xo as usize] = sum;
                    }
                }
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_reflect(
    dy: &[f32],
    offset: isize,
    axis: usize,
    rank: usize,
    coords: &[usize],
    in_dims: &[usize],
    pads: &[(usize, usize)],
    up_strides: &[isize],
) -> f32 {
    if axis == rank {
        return dy[offset as usize];
    }
    let x = coords[axis];
    let d = in_dims[axis];
    let (pb, pa) = pads[axis];
    let st = up_strides[axis];
    let mut sum = gather_reflect(
        dy,
        offset + (x + pb) as isize * st,
        axis + 1,
        rank,
        coords,
        in_dims,
        pads,
        up_strides,
    );
    if x >= 1 && x <= pb {
        sum += gather_reflect(
            dy,
            offset + (pb - x) as isize * st,
            axis + 1,
            rank,
            coords,
            in_dims,
            pads,
            up_strides,
        );
    }
    if x + 1 < d && (d - 1 - x) <= pa {
        let y_r = pb + 2 * (d - 1) - x;
        sum += gather_reflect(
            dy,
            offset + y_r as isize * st,
            axis + 1,
            rank,
            coords,
            in_dims,
            pads,
            up_strides,
        );
    }
    sum
}

#[allow(clippy::too_many_arguments)]
fn gather_replicate(
    dy: &[f32],
    offset: isize,
    axis: usize,
    rank: usize,
    coords: &[usize],
    in_dims: &[usize],
    out_dims: &[usize],
    pads: &[(usize, usize)],
    up_strides: &[isize],
) -> f32 {
    if axis == rank {
        return dy[offset as usize];
    }
    let x = coords[axis];
    let d = in_dims[axis];
    let od = out_dims[axis];
    let (pb, _) = pads[axis];
    let st = up_strides[axis];
    let lo = if x == 0 { 0 } else { x + pb };
    let hi = if x + 1 == d { od } else { x + pb + 1 };
    let mut sum = 0.0f32;
    for y in lo..hi {
        sum += gather_replicate(
            dy,
            offset + y as isize * st,
            axis + 1,
            rank,
            coords,
            in_dims,
            out_dims,
            pads,
            up_strides,
        );
    }
    sum
}

/// Validated binary concatenation geometry along `axis`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatSpec {
    left: Shape,
    right: Shape,
    output: Shape,
    axis: usize,
}

impl CatSpec {
    pub fn new(left: &Shape, right: &Shape, axis: usize) -> Result<Self> {
        let ld = left.dims();
        let rd = right.dims();
        if ld.is_empty() || ld.len() != rd.len() || axis >= ld.len() {
            return Err(invalid("cat requires matching ranks >= 1 and axis < rank"));
        }
        let mut out_dims = ld.to_vec();
        for a in 0..ld.len() {
            if a == axis {
                out_dims[a] = ld[a]
                    .checked_add(rd[a])
                    .ok_or_else(|| invalid("cat output axis overflows"))?;
            } else if ld[a] != rd[a] {
                return Err(invalid("cat non-concatenated dimensions must match"));
            }
        }
        let output = Shape::new(out_dims)?;
        Ok(Self {
            left: left.clone(),
            right: right.clone(),
            output,
            axis,
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
    pub fn axis(&self) -> usize {
        self.axis
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        size_of::<Self>()
            + (self.left.dims().len() + self.right.dims().len() + self.output.dims().len())
                * size_of::<usize>()
    }
}

/// Prepared arbitrary-rank binary `Cat` forward plan on CPU (zero hot allocations).
pub struct CatPlan {
    spec: CatSpec,
    left_layout: Layout,
    right_layout: Layout,
    output_layout: Layout,
}

impl CatPlan {
    pub fn new(spec: &CatSpec, left: &Layout, right: &Layout, output: &Layout) -> Result<Self> {
        if left.shape() != &spec.left
            || right.shape() != &spec.right
            || output.shape() != &spec.output
        {
            return Err(invalid("cat layouts differ from spec"));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "cat output layout must be proved non-overlapping".into(),
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            left_layout: left.clone(),
            right_layout: right.clone(),
            output_layout: output.clone(),
        })
    }

    pub fn spec(&self) -> &CatSpec {
        &self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn host_metadata_bytes(&self) -> usize {
        self.spec.metadata_bytes()
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
            return Err(invalid("cat bindings differ from prepared plan"));
        }
        let total = self.spec.output.len();
        if total == 0 {
            return Ok(());
        }
        let rank = self.spec.output.dims().len();
        let axis = self.spec.axis;
        let split = self.spec.left.dims()[axis];

        // Fast contiguous slice copy path
        if self.left_layout.is_contiguous()
            && self.right_layout.is_contiguous()
            && self.output_layout.is_contiguous()
        {
            let outer: usize = self.spec.output.dims()[..axis].iter().product();
            let inner: usize = self.spec.output.dims()[axis + 1..].iter().product();
            let l_chunk = split * inner;
            let r_chunk = self.spec.right.dims()[axis] * inner;
            let o_chunk = l_chunk + r_chunk;
            let l0 = self.left_layout.offset();
            let r0 = self.right_layout.offset();
            let o0 = self.output_layout.offset();
            for g in 0..outer {
                let dst = &mut output.data[o0 + g * o_chunk..o0 + (g + 1) * o_chunk];
                dst[..l_chunk]
                    .copy_from_slice(&left.data[l0 + g * l_chunk..l0 + (g + 1) * l_chunk]);
                dst[l_chunk..]
                    .copy_from_slice(&right.data[r0 + g * r_chunk..r0 + (g + 1) * r_chunk]);
            }
            return Ok(());
        }

        let out_dims = self.spec.output.dims();
        let ls = self.left_layout.strides();
        let rs = self.right_layout.strides();
        let os = self.output_layout.strides();
        let lo0 = self.left_layout.offset() as isize;
        let ro0 = self.right_layout.offset() as isize;
        let oo0 = self.output_layout.offset() as isize;
        let d_last = out_dims[rank - 1];
        let outer = total / d_last;
        let ls_last = ls[rank - 1];
        let rs_last = rs[rank - 1];
        let os_last = os[rank - 1];

        for g in 0..outer {
            let mut rem = g;
            let mut lo = lo0;
            let mut ro = ro0;
            let mut oo = oo0;
            let mut is_right = false;
            for a in (0..rank - 1).rev() {
                let c = rem % out_dims[a];
                rem /= out_dims[a];
                oo += c as isize * os[a];
                if a == axis {
                    if c < split {
                        lo += c as isize * ls[a];
                    } else {
                        is_right = true;
                        ro += (c - split) as isize * rs[a];
                    }
                } else {
                    lo += c as isize * ls[a];
                    ro += c as isize * rs[a];
                }
            }
            if axis == rank - 1 {
                for c in 0..split {
                    output.data[(oo + c as isize * os_last) as usize] =
                        left.data[(lo + c as isize * ls_last) as usize];
                }
                for c in split..d_last {
                    output.data[(oo + c as isize * os_last) as usize] =
                        right.data[(ro + (c - split) as isize * rs_last) as usize];
                }
            } else if is_right {
                for c in 0..d_last {
                    output.data[(oo + c as isize * os_last) as usize] =
                        right.data[(ro + c as isize * rs_last) as usize];
                }
            } else {
                for c in 0..d_last {
                    output.data[(oo + c as isize * os_last) as usize] =
                        left.data[(lo + c as isize * ls_last) as usize];
                }
            }
        }
        Ok(())
    }
}

/// Prepared arbitrary-rank binary `Cat` local VJP (`input_dependencies() == [false, false]`).
pub struct CatBackward {
    spec: CatSpec,
    operand: BinaryOperand,
    upstream_layout: Layout,
    output_layout: Layout,
}

impl CatBackward {
    pub fn new(
        operand: BinaryOperand,
        spec: &CatSpec,
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self> {
        let target_shape = match operand {
            BinaryOperand::Left => &spec.left,
            BinaryOperand::Right => &spec.right,
        };
        if upstream.shape() != &spec.output || output.shape() != target_shape {
            return Err(invalid("cat backward layouts differ from spec"));
        }
        if !output.is_non_overlapping() {
            return Err(Error::Unsupported(
                "cat backward output layout must be proved non-overlapping".into(),
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            operand,
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
        })
    }

    pub fn spec(&self) -> &CatSpec {
        &self.spec
    }
    pub fn operand(&self) -> BinaryOperand {
        self.operand
    }
    pub fn input_dependencies(&self) -> [bool; 2] {
        [false, false]
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
            return Err(invalid("cat backward bindings differ from prepared plan"));
        }
        let target_shape = match self.operand {
            BinaryOperand::Left => &self.spec.left,
            BinaryOperand::Right => &self.spec.right,
        };
        let total = target_shape.len();
        if total == 0 {
            return Ok(());
        }
        let rank = target_shape.dims().len();
        let axis = self.spec.axis;
        let axis_shift = match self.operand {
            BinaryOperand::Left => 0,
            BinaryOperand::Right => self.spec.left.dims()[axis],
        };
        let acc = mode == GradientMode::Accumulate;

        if self.upstream_layout.is_contiguous() && self.output_layout.is_contiguous() {
            let outer: usize = target_shape.dims()[..axis].iter().product();
            let inner: usize = target_shape.dims()[axis + 1..].iter().product();
            let t_chunk = target_shape.dims()[axis] * inner;
            let u_chunk = self.spec.output.dims()[axis] * inner;
            let shift_elems = axis_shift * inner;
            let u0 = self.upstream_layout.offset();
            let d0 = self.output_layout.offset();
            for g in 0..outer {
                let src = &upstream.data
                    [u0 + g * u_chunk + shift_elems..u0 + g * u_chunk + shift_elems + t_chunk];
                let dst = &mut output.data[d0 + g * t_chunk..d0 + (g + 1) * t_chunk];
                if acc {
                    for (y, &x) in dst.iter_mut().zip(src) {
                        *y += x;
                    }
                } else {
                    dst.copy_from_slice(src);
                }
            }
            return Ok(());
        }

        let dims = target_shape.dims();
        let us = self.upstream_layout.strides();
        let ds = self.output_layout.strides();
        let u0 = self.upstream_layout.offset() as isize;
        let d0 = self.output_layout.offset() as isize;
        let d_last = dims[rank - 1];
        let outer = total / d_last;
        let us_last = us[rank - 1];
        let ds_last = ds[rank - 1];

        for g in 0..outer {
            let mut rem = g;
            let mut uo = u0;
            let mut doff = d0;
            for a in (0..rank - 1).rev() {
                let c = rem % dims[a];
                rem /= dims[a];
                doff += c as isize * ds[a];
                let uc = if a == axis { c + axis_shift } else { c };
                uo += uc as isize * us[a];
            }
            if axis == rank - 1 {
                uo += axis_shift as isize * us_last;
            }
            if acc {
                for c in 0..d_last {
                    output.data[(doff + c as isize * ds_last) as usize] +=
                        upstream.data[(uo + c as isize * us_last) as usize];
                }
            } else {
                for c in 0..d_last {
                    output.data[(doff + c as isize * ds_last) as usize] =
                        upstream.data[(uo + c as isize * us_last) as usize];
                }
            }
        }
        Ok(())
    }
}
