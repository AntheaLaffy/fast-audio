//! Fixed-shape inference graphs. Construction validates edges and allocates execution storage.
use crate::{
    Activation, Conv2dSpec, GruSpec, GruWeights, Result, Shape, Tensor,
    error::{int, invalid},
};

pub(crate) mod host;
pub(crate) mod mapping;

/// A value belongs to one builder; cross-builder values are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Value {
    index: usize,
    owner: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum Unary {
    Sigmoid,
    Tanh,
    PRelu(f32),
}

/// Private native ABI shared with tensor_ops.h. Dimensions and indices are FP32 element counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Kernel {
    pub dims: [u64; 4],
    pub strides: [u64; 4],
    pub len: u64,
    pub offset: u64,
    pub inner: u64,
    pub width: u64,
    pub kind: i32,
    pub rank: i32,
    pub param: f32,
    pub eps: f32,
}
const _: [(); 112] = [(); size_of::<Kernel>()];
pub(crate) enum Op {
    Constant(Tensor),
    Kernel(Kernel, Option<(Tensor, Tensor)>),
    Linear {
        rows: usize,
        input: usize,
        output: usize,
        weights: Tensor,
        bias: Option<Tensor>,
    },
    Conv(Conv2dSpec, Tensor, Option<Tensor>, Activation),
    Gru(GruSpec, GruWeights),
}
pub(crate) struct Node {
    pub inputs: [Option<usize>; 2],
    pub first: usize,
    pub outputs: usize,
    pub op: Op,
}

/// An immutable description consumed when preparing a CPU or CUDA graph.
pub struct Program {
    pub(crate) shapes: Vec<Shape>,
    pub(crate) nodes: Vec<Node>,
    owner: u64,
}

pub struct Builder {
    owner: u64,
    shapes: Vec<Shape>,
    nodes: Vec<Node>,
}
impl Builder {
    pub fn new(input: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(input)?;
        if shape.is_empty() || shape.dims().is_empty() {
            return Err(crate::Error::Unsupported(
                "prepared graph requires positive non-scalar values".into(),
            ));
        }
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        #[allow(deprecated)]
        let owner = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| invalid("graph identity space exhausted"))?;
        Ok(Self {
            owner,
            shapes: vec![shape],
            nodes: Vec::new(),
        })
    }
    pub fn input(&self) -> Value {
        self.value(0)
    }
    fn value(&self, index: usize) -> Value {
        Value {
            index,
            owner: self.owner,
        }
    }
    pub fn shape(&self, value: Value) -> Result<&[usize]> {
        if value.owner != self.owner {
            return Err(invalid("value belongs to another graph builder"));
        }
        self.shapes
            .get(value.index)
            .map(Shape::dims)
            .ok_or_else(|| invalid("unknown graph value"))
    }
    fn push(&mut self, inputs: &[Value], outputs: &[Shape], op: Op) -> Result<Value> {
        if outputs.iter().any(|s| s.is_empty() || s.dims().is_empty()) {
            return Err(crate::Error::Unsupported(
                "prepared graph requires positive non-scalar values".into(),
            ));
        }
        let mut edges = [None; 2];
        for (slot, &value) in edges.iter_mut().zip(inputs) {
            self.shape(value)?;
            *slot = Some(value.index);
        }
        let first = self.shapes.len();
        self.shapes.extend_from_slice(outputs);
        self.nodes.push(Node {
            inputs: edges,
            first,
            outputs: outputs.len(),
            op,
        });
        Ok(self.value(first))
    }
    fn kernel(
        &mut self,
        inputs: &[Value],
        shape: impl AsRef<[usize]>,
        mut k: Kernel,
        parameters: Option<(Tensor, Tensor)>,
    ) -> Result<Value> {
        let shape = Shape::new(shape)?;
        k.len = shape.len() as u64;
        self.push(inputs, &[shape], Op::Kernel(k, parameters))
    }
    pub fn constant(&mut self, tensor: Tensor) -> Result<Value> {
        self.push(&[], &[tensor.shape().clone()], Op::Constant(tensor))
    }
    /// Slice, then permute, then reshape in one explicit materialization (rank <= 4).
    pub fn layout(
        &mut self,
        input: Value,
        ranges: &[(usize, usize)],
        axes: &[usize],
        output: impl AsRef<[usize]>,
    ) -> Result<Value> {
        let shape = self.shape(input)?;
        let rank = shape.len();
        if rank > 4 || ranges.len() != rank || axes.len() != rank {
            return Err(invalid("layout rank mismatch"));
        }
        let mut strides = [0; 4];
        let mut stride = 1usize;
        let mut offset = 0usize;
        for a in (0..rank).rev() {
            let (start, len) = ranges[a];
            if len == 0 || start.checked_add(len).is_none_or(|end| end > shape[a]) {
                return Err(invalid("layout slice is outside its input"));
            }
            strides[a] = stride as u64;
            offset += start * stride;
            stride *= shape[a];
        }
        let mut k = Kernel {
            kind: 0,
            rank: rank as i32,
            offset: offset as u64,
            ..Kernel::default()
        };
        let mut seen = [false; 4];
        let mut len = 1usize;
        for (i, &axis) in axes.iter().enumerate() {
            if axis >= rank || seen[axis] {
                return Err(invalid("layout axes must be a permutation"));
            }
            seen[axis] = true;
            k.dims[i] = ranges[axis].1 as u64;
            k.strides[i] = strides[axis];
            len *= ranges[axis].1;
        }
        if Shape::new(output.as_ref())?.len() != len {
            return Err(invalid("layout reshape changes element count"));
        }
        self.kernel(&[input], output, k, None)
    }
    pub fn unary(&mut self, input: Value, operation: Unary) -> Result<Value> {
        let shape = self.shape(input)?.to_vec();
        let (rank, param) = match operation {
            Unary::Sigmoid => (0, 0.0),
            Unary::Tanh => (1, 0.0),
            Unary::PRelu(alpha) if alpha.is_finite() => (2, alpha),
            _ => return Err(invalid("PReLU coefficient must be finite")),
        };
        self.kernel(
            &[input],
            shape,
            Kernel {
                kind: 1,
                rank,
                param,
                ..Kernel::default()
            },
            None,
        )
    }
    pub fn add(&mut self, a: Value, b: Value) -> Result<Value> {
        let shape = self.shape(a)?.to_vec();
        if self.shape(b)? != shape {
            return Err(invalid("graph addition needs equal shapes"));
        }
        self.kernel(
            &[a, b],
            shape,
            Kernel {
                kind: 2,
                ..Kernel::default()
            },
            None,
        )
    }
    /// Explicit right-aligned broadcasting of the second argument only.
    pub fn multiply_broadcast(&mut self, a: Value, b: Value) -> Result<Value> {
        let shape = self.shape(a)?.to_vec();
        let rhs = self.shape(b)?;
        if shape.len() > 4 || rhs.len() > shape.len() {
            return Err(invalid("broadcast ranks are unsupported"));
        }
        let mut k = Kernel {
            kind: 3,
            rank: shape.len() as i32,
            ..Kernel::default()
        };
        let mut stride = 1usize;
        for axis in (0..shape.len()).rev() {
            k.dims[axis] = shape[axis] as u64;
            if axis + rhs.len() >= shape.len() {
                let dim = rhs[axis + rhs.len() - shape.len()];
                if dim != 1 && dim != shape[axis] {
                    return Err(invalid("broadcast axis mismatch"));
                }
                k.strides[axis] = if dim == 1 { 0 } else { stride as u64 };
                stride *= dim;
            }
        }
        self.kernel(&[a, b], shape, k, None)
    }
    pub fn concat(&mut self, a: Value, b: Value, axis: usize, interleave: bool) -> Result<Value> {
        let mut shape = self.shape(a)?.to_vec();
        let rhs = self.shape(b)?;
        if axis >= shape.len()
            || rhs.len() != shape.len()
            || shape
                .iter()
                .zip(rhs)
                .enumerate()
                .any(|(i, (x, y))| i != axis && x != y)
            || (interleave && shape[axis] != rhs[axis])
        {
            return Err(invalid("concatenation shapes differ"));
        }
        let k = Kernel {
            kind: 4,
            rank: i32::from(interleave),
            width: shape[axis] as u64,
            offset: rhs[axis] as u64,
            inner: shape[axis + 1..].iter().product::<usize>() as u64,
            ..Kernel::default()
        };
        shape[axis] = shape[axis]
            .checked_add(rhs[axis])
            .ok_or_else(|| invalid("concat axis overflows"))?;
        self.kernel(&[a, b], shape, k, None)
    }
    pub fn pad_axis(
        &mut self,
        input: Value,
        axis: usize,
        before: usize,
        after: usize,
    ) -> Result<Value> {
        let mut shape = self.shape(input)?.to_vec();
        if axis >= shape.len() {
            return Err(invalid("padding axis is outside rank"));
        }
        let k = Kernel {
            kind: 5,
            width: shape[axis] as u64,
            offset: before as u64,
            inner: shape[axis + 1..].iter().product::<usize>() as u64,
            ..Kernel::default()
        };
        shape[axis] = shape[axis]
            .checked_add(before)
            .and_then(|n| n.checked_add(after))
            .ok_or_else(|| invalid("padding overflows"))?;
        let mut k = k;
        k.dims[0] = shape[axis] as u64;
        self.kernel(&[input], shape, k, None)
    }
    /// Zero-padded frequency neighborhoods in [channel, kernel-position] order.
    pub fn unfold_frequency(&mut self, input: Value, kernel: usize) -> Result<Value> {
        let &[b, c, t, f] = self.shape(input)? else {
            return Err(invalid("frequency unfold needs NCHW"));
        };
        if kernel == 0 || kernel % 2 == 0 {
            return Err(invalid("frequency kernel must be positive and odd"));
        }
        let outc = c
            .checked_mul(kernel)
            .ok_or_else(|| invalid("unfold channels overflow"))?;
        self.kernel(
            &[input],
            [b, outc, t, f],
            Kernel {
                kind: 6,
                dims: [b as u64, c as u64, t as u64, f as u64],
                width: kernel as u64,
                ..Kernel::default()
            },
            None,
        )
    }
    pub fn mean_square_last(&mut self, input: Value) -> Result<Value> {
        let shape = self.shape(input)?;
        if shape.len() < 2 {
            return Err(invalid("reduction needs rank >= 2"));
        }
        let width = *shape.last().unwrap();
        let out = shape[..shape.len() - 1].to_vec();
        self.kernel(
            &[input],
            out,
            Kernel {
                kind: 7,
                inner: width as u64,
                ..Kernel::default()
            },
            None,
        )
    }
    pub fn layer_norm(
        &mut self,
        input: Value,
        scale: &Tensor,
        bias: &Tensor,
        epsilon: f32,
    ) -> Result<Value> {
        let shape = self.shape(input)?.to_vec();
        let suffix = scale.shape().dims();
        if suffix.is_empty()
            || suffix.len() > shape.len()
            || &shape[shape.len() - suffix.len()..] != suffix
            || bias.shape() != scale.shape()
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(invalid("invalid layer normalization parameters"));
        }
        self.kernel(
            &[input],
            shape,
            Kernel {
                kind: 8,
                inner: scale.shape().len() as u64,
                eps: epsilon,
                ..Kernel::default()
            },
            Some((scale.clone(), bias.clone())),
        )
    }
    pub fn complex_features(&mut self, spectrum: Value, epsilon: f32) -> Result<Value> {
        let &[b, t, f, 2] = self.shape(spectrum)? else {
            return Err(invalid("complex features need [B,T,F,2]"));
        };
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(invalid("magnitude epsilon must be finite and nonnegative"));
        }
        self.kernel(
            &[spectrum],
            [b, 3, t, f],
            Kernel {
                kind: 9,
                inner: (t * f) as u64,
                eps: epsilon,
                ..Kernel::default()
            },
            None,
        )
    }
    pub fn complex_mask(&mut self, mask: Value, spectrum: Value) -> Result<Value> {
        let shape = self.shape(spectrum)?.to_vec();
        let &[b, t, f, 2] = shape.as_slice() else {
            return Err(invalid("complex mask spectrum needs [B,T,F,2]"));
        };
        if self.shape(mask)? != [b, 2, t, f] {
            return Err(invalid("complex mask shape differs"));
        }
        self.kernel(
            &[mask, spectrum],
            &shape,
            Kernel {
                kind: 10,
                inner: (t * f) as u64,
                ..Kernel::default()
            },
            None,
        )
    }
    /// Last-axis projection; `output` may explicitly reshape the projected contiguous rows.
    pub fn linear(
        &mut self,
        input: Value,
        weights: &Tensor,
        bias: Option<&Tensor>,
        output: impl AsRef<[usize]>,
    ) -> Result<Value> {
        let shape = self.shape(input)?;
        let &[cout, cin] = weights.shape().dims() else {
            return Err(invalid("linear weights need [out,in]"));
        };
        if shape.last() != Some(&cin) {
            return Err(invalid("linear contraction axis differs"));
        }
        if let Some(b) = bias {
            b.shape().expect(&[cout])?;
        }
        let rows = self.shapes[input.index].len() / cin;
        for n in [rows, cin, cout] {
            int(n)?;
        }
        let output = Shape::new(output)?;
        if rows.checked_mul(cout) != Some(output.len()) {
            return Err(invalid("linear output element count differs"));
        }
        let mut packed = vec![0.0; weights.shape().len()];
        for o in 0..cout {
            for i in 0..cin {
                packed[i * cout + o] = weights.data()[o * cin + i];
            }
        }
        self.push(
            &[input],
            &[output],
            Op::Linear {
                rows,
                input: cin,
                output: cout,
                weights: Tensor::from_vec([cin, cout], packed)?,
                bias: bias.cloned(),
            },
        )
    }
    pub fn conv2d(
        &mut self,
        input: Value,
        spec: Conv2dSpec,
        weights: &Tensor,
        bias: Option<&Tensor>,
        activation: Activation,
    ) -> Result<Value> {
        if self.shape(input)? != spec.input_shape() {
            return Err(invalid("graph convolution input differs"));
        }
        spec.check_weights(weights, bias)?;
        self.push(
            &[input],
            &[Shape::new(spec.output_shape()?)?],
            Op::Conv(spec, weights.clone(), bias.cloned(), activation),
        )
    }
    pub fn gru(
        &mut self,
        input: Value,
        initial: Value,
        spec: GruSpec,
        weights: GruWeights,
    ) -> Result<(Value, Value)> {
        if self.shape(input)? != spec.input_shape() || self.shape(initial)? != spec.state_shape() {
            return Err(invalid("graph GRU input/state shapes differ"));
        }
        spec.check_weights(&weights)?;
        let y = self.push(
            &[input, initial],
            &[
                Shape::new(spec.output_shape()?)?,
                Shape::new(spec.state_shape())?,
            ],
            Op::Gru(spec, weights),
        )?;
        Ok((y, self.value(y.index + 1)))
    }
    pub fn finish(self, output: Value) -> Result<Program> {
        self.shape(output)?;
        let last = self
            .nodes
            .last()
            .ok_or_else(|| invalid("graph has no operations"))?;
        if output.index != self.shapes.len() - 1
            || last.outputs != 1
            || matches!(last.op, Op::Constant(_))
        {
            return Err(invalid(
                "graph output must be its final single-output operation",
            ));
        }
        Ok(Program {
            shapes: self.shapes,
            nodes: self.nodes,
            owner: self.owner,
        })
    }
}

pub(crate) trait Backend {
    type Buffer;
    type Plan;
    fn zeros(&self, shape: &Shape) -> Result<Self::Buffer>;
    fn upload(&self, input: &Tensor) -> Result<Self::Buffer>;
    fn validate(&self, input: &Self::Buffer, shape: &Shape) -> Result<()>;
    fn prepare(&self, op: &Op) -> Result<Self::Plan>;
    fn execute(
        &self,
        plan: &mut Self::Plan,
        input: [Option<&Self::Buffer>; 2],
        output: &mut [Self::Buffer],
    ) -> Result<()>;
    fn bytes(plan: &Self::Plan) -> (usize, usize);
}
pub(crate) struct Step<P> {
    pub plan: P,
    inputs: [Option<usize>; 2],
    first: usize,
    outputs: usize,
    constant: bool,
}
pub(crate) struct Executor<B: Backend> {
    backend: B,
    input: Shape,
    output: Shape,
    buffers: Vec<B::Buffer>,
    steps: Vec<Step<B::Plan>>,
    workspace: usize,
    parameters: usize,
    owner: u64,
}
impl<B: Backend> Executor<B> {
    #[allow(clippy::needless_range_loop)]
    pub fn new(backend: B, program: Program) -> Result<Self> {
        let last = program.shapes.len() - 1;
        let mut buffers = Vec::with_capacity(last - 1);
        let mut steps = Vec::with_capacity(program.nodes.len());
        let (mut workspace, mut parameters) = (0, 0);
        for node in program.nodes {
            let constant = matches!(node.op, Op::Constant(_));
            for id in node.first..node.first + node.outputs {
                if id == last {
                    continue;
                }
                let bytes = program.shapes[id].len() * 4;
                let buffer = if let Op::Constant(t) = &node.op {
                    parameters += bytes;
                    backend.upload(t)?
                } else {
                    workspace += bytes;
                    backend.zeros(&program.shapes[id])?
                };
                buffers.push(buffer);
            }
            let plan = backend.prepare(&node.op)?;
            let (work, params) = B::bytes(&plan);
            workspace += work;
            parameters += params;
            steps.push(Step {
                plan,
                inputs: node.inputs,
                first: node.first,
                outputs: node.outputs,
                constant,
            });
        }
        Ok(Self {
            backend,
            input: program.shapes[0].clone(),
            output: program.shapes[last].clone(),
            buffers,
            steps,
            workspace,
            parameters,
            owner: program.owner,
        })
    }
    pub fn input_shape(&self) -> &[usize] {
        self.input.dims()
    }
    pub fn output_shape(&self) -> &[usize] {
        self.output.dims()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
    }
    pub fn parameter_bytes(&self) -> usize {
        self.parameters
    }
    #[cfg(feature = "cudnn")]
    pub(crate) fn plans(&self) -> impl Iterator<Item = &B::Plan> {
        self.steps.iter().map(|step| &step.plan)
    }
    pub fn run(&mut self, input: &B::Buffer, output: &mut B::Buffer) -> Result<()> {
        self.backend.validate(input, &self.input)?;
        self.backend.validate(output, &self.output)?;
        let external_output = self.buffers.len() + 1;
        for step in &mut self.steps {
            if step.constant {
                continue;
            }
            let (previous, destination) = self.buffers.split_at_mut(step.first - 1);
            let arguments = step
                .inputs
                .map(|id| id.map(|i| if i == 0 { input } else { &previous[i - 1] }));
            if step.first == external_output {
                self.backend
                    .execute(&mut step.plan, arguments, std::slice::from_mut(output))?;
            } else {
                self.backend.execute(
                    &mut step.plan,
                    arguments,
                    &mut destination[..step.outputs],
                )?;
            }
        }
        Ok(())
    }
    pub fn intermediate(&self, value: Value) -> Option<&B::Buffer> {
        if value.owner != self.owner {
            return None;
        }
        value.index.checked_sub(1).and_then(|i| self.buffers.get(i))
    }
}
