//! Explicit fixed-shape reverse-mode programs; inference APIs do not maintain a tape.
//!
//! ```
//! use fast_audio::{GradientMode, ReduceOp, Shared, Tensor, UnaryOp, autograd::{Binding,Builder}, cpu::{Cpu,Tape}};
//! let mut builder=Builder::new()?;
//! let parameter=builder.leaf([2],true)?;
//! let square=builder.unary(UnaryOp::Square,parameter)?;
//! let loss=builder.reduce(ReduceOp::Mean,square,&[0],false)?;
//! let program=builder.finish(&[loss])?;
//! let owner=Shared::new(Tensor::from_vec([2],vec![1.,2.])?)?;
//! let mut tape=Tape::new(Cpu::Reference,&program,&[Binding::new(parameter,owner.clone())])?;
//! tape.forward()?;
//! tape.backward_scalar(GradientMode::Overwrite)?;
//! assert_eq!(tape.gradient(parameter)?.unwrap().data(),&[1.,2.]);
//! tape.sgd(&[parameter],0.1)?;
//! assert_eq!(owner.read()?.data(),&[0.9,1.8]);
//! # Ok::<(),fast_audio::Error>(())
//! ```
pub(crate) mod backend;
pub(crate) mod engine;
pub(crate) mod fusion;
pub(crate) mod host;
mod prepare;
#[cfg(test)]
mod tests;
mod update;

use crate::{
    BinaryOp, BmmSpec, CatSpec, CenteredStftSpec, Conv1dSpec, Conv2dSpec, ConvTranspose1dSpec,
    Layout, NormKind, NormSpec, PadMode, PadSpec, ReadGuard, ReduceOp, ReduceSpec, Result,
    RopeSpec, SdpaSpec, Shape, Shared, SoftmaxKind, SoftmaxSpec, StftSpec, Tensor, Trackable,
    UnaryOp, error::invalid,
};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// Builder-branded identity. Equal shapes never make values interchangeable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Value {
    owner: u64,
    pub(crate) index: usize,
}

#[derive(Clone)]
pub(crate) enum Operation {
    Unary(UnaryOp),
    Binary(BinaryOp),
    Reduce(ReduceSpec),
    Matmul,
    Conv1d(Conv1dSpec),
    Conv2d(Conv2dSpec),
    ConvTranspose1d(ConvTranspose1dSpec),
    ConvTranspose2d(Conv2dSpec),
    Stft {
        spec: StftSpec,
        window: Option<Arc<[f32]>>,
    },
    CenteredStft {
        spec: CenteredStftSpec,
        window: Option<Arc<[f32]>>,
    },
    Istft {
        spec: CenteredStftSpec,
        window: Option<Arc<[f32]>>,
    },
    Norm(NormSpec),
    Softmax(SoftmaxSpec),
    Bmm(BmmSpec),
    Sdpa(SdpaSpec),
    Rope(RopeSpec),
    Pad(PadSpec),
    Cat(CatSpec),
}
pub(crate) enum Kind {
    Leaf,
    Alias(usize),
    Compute {
        operation: Operation,
        inputs: [Option<usize>; 3],
    },
}
pub(crate) struct Definition {
    pub layout: Arc<Layout>,
    pub root: usize,
    pub requires_grad: bool,
    pub kind: Kind,
}
pub(crate) struct DefinitionGraph {
    owner: u64,
    pub values: Box<[Definition]>,
    pub roots: Box<[Value]>,
    pub leaves: Box<[Value]>,
    pub forward: Box<[usize]>,
    pub gradient_roots: Box<[bool]>,
}

/// Immutable topology shared by independent CPU and CUDA execution instances.
#[derive(Clone)]
pub struct Program(pub(crate) Arc<DefinitionGraph>);
impl Program {
    pub(crate) fn index(&self, value: Value) -> Result<usize> {
        if value.owner != self.0.owner || value.index >= self.0.values.len() {
            return Err(invalid("value does not belong to this training program"));
        }
        Ok(value.index)
    }
    pub fn roots(&self) -> &[Value] {
        &self.0.roots
    }
    pub fn leaves(&self) -> &[Value] {
        &self.0.leaves
    }
    pub fn layout(&self, value: Value) -> Result<&Layout> {
        Ok(&self.0.values[self.index(value)?].layout)
    }
    pub fn shape(&self, value: Value) -> Result<&Shape> {
        Ok(self.layout(value)?.shape())
    }
    pub fn requires_grad(&self, value: Value) -> Result<bool> {
        Ok(self.0.values[self.index(value)?].requires_grad)
    }
}

/// Append-only checked DAG. Failed operations do not append partial nodes.
pub struct Builder {
    owner: u64,
    values: Vec<Definition>,
    leaves: Vec<Value>,
}
impl Builder {
    pub fn new() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let mut current = NEXT.load(Ordering::Relaxed);
        let owner = loop {
            let next = current
                .checked_add(1)
                .ok_or_else(|| invalid("training builder identity exhausted"))?;
            match NEXT.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break current,
                Err(actual) => current = actual,
            }
        };
        Ok(Self {
            owner,
            values: Vec::new(),
            leaves: Vec::new(),
        })
    }
    fn value(&self, index: usize) -> Value {
        Value {
            owner: self.owner,
            index,
        }
    }
    fn index(&self, value: Value) -> Result<usize> {
        if value.owner != self.owner || value.index >= self.values.len() {
            return Err(invalid("value does not belong to this training builder"));
        }
        Ok(value.index)
    }
    pub fn layout(&self, value: Value) -> Result<&Layout> {
        Ok(&self.values[self.index(value)?].layout)
    }
    pub fn shape(&self, value: Value) -> Result<&Shape> {
        Ok(self.layout(value)?.shape())
    }
    pub fn leaf(&mut self, shape: impl AsRef<[usize]>, requires_grad: bool) -> Result<Value> {
        let layout = Arc::new(Layout::contiguous(shape)?);
        let index = self.values.len();
        let value = self.value(index);
        self.values.push(Definition {
            layout,
            root: index,
            requires_grad,
            kind: Kind::Leaf,
        });
        self.leaves.push(value);
        Ok(value)
    }
    fn compute(
        &mut self,
        operation: Operation,
        inputs: [Option<usize>; 2],
        shape: &[usize],
    ) -> Result<Value> {
        self.compute3(operation, [inputs[0], inputs[1], None], shape)
    }
    fn compute3(
        &mut self,
        operation: Operation,
        inputs: [Option<usize>; 3],
        shape: &[usize],
    ) -> Result<Value> {
        let layout = Arc::new(Layout::contiguous(shape)?);
        let requires_grad = inputs
            .into_iter()
            .flatten()
            .any(|i| self.values[i].requires_grad);
        let index = self.values.len();
        self.values.push(Definition {
            layout,
            root: index,
            requires_grad,
            kind: Kind::Compute { operation, inputs },
        });
        Ok(self.value(index))
    }
    pub fn unary(&mut self, operation: UnaryOp, input: Value) -> Result<Value> {
        let i = self.index(input)?;
        let shape = self.values[i].layout.shape().dims().to_vec();
        self.compute(Operation::Unary(operation), [Some(i), None], &shape)
    }
    pub fn binary(&mut self, operation: BinaryOp, left: Value, right: Value) -> Result<Value> {
        let a = self.index(left)?;
        let b = self.index(right)?;
        let shape =
            crate::pointwise::broadcast_dims(&[&self.values[a].layout, &self.values[b].layout])?;
        self.compute(Operation::Binary(operation), [Some(a), Some(b)], &shape)
    }
    pub fn reduce(
        &mut self,
        operation: ReduceOp,
        input: Value,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<Value> {
        let i = self.index(input)?;
        let spec = ReduceSpec::new(operation, self.values[i].layout.shape(), axes, keep_dims)?;
        let shape = spec.output_shape().dims().to_vec();
        self.compute(Operation::Reduce(spec), [Some(i), None], &shape)
    }
    pub fn matmul(&mut self, left: Value, right: Value) -> Result<Value> {
        let a = self.index(left)?;
        let b = self.index(right)?;
        let &[m, k] = self.values[a].layout.shape().dims() else {
            return Err(invalid("training matmul left must be rank two"));
        };
        let &[bk, n] = self.values[b].layout.shape().dims() else {
            return Err(invalid("training matmul right must be rank two"));
        };
        if k != bk {
            return Err(invalid("training matmul contraction differs"));
        }
        self.compute(Operation::Matmul, [Some(a), Some(b)], &[m, n])
    }
    pub fn conv1d(
        &mut self,
        input: Value,
        weight: Value,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Value> {
        let a = self.index(input)?;
        let b = self.index(weight)?;
        let &[batch, in_channels, input_len] = self.values[a].layout.shape().dims() else {
            return Err(invalid("training conv1d input must be rank three"));
        };
        let &[out_channels, in_per_group, kernel] = self.values[b].layout.shape().dims() else {
            return Err(invalid("training conv1d weight must be rank three"));
        };
        if groups == 0
            || in_channels % groups != 0
            || out_channels % groups != 0
            || in_per_group != in_channels / groups
        {
            return Err(invalid("training conv1d channel/group mismatch"));
        }
        let spec = Conv1dSpec {
            batch,
            in_channels,
            out_channels,
            input_len,
            kernel,
            stride,
            padding,
            dilation,
            groups,
        };
        let shape = spec.output_shape()?;
        self.compute(Operation::Conv1d(spec), [Some(a), Some(b)], &shape)
    }
    pub fn conv2d(
        &mut self,
        input: Value,
        weight: Value,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Value> {
        let a = self.index(input)?;
        let b = self.index(weight)?;
        let &[batch, in_channels, ih, iw] = self.values[a].layout.shape().dims() else {
            return Err(invalid("training conv2d input must be rank four"));
        };
        let &[out_channels, in_per_group, kh, kw] = self.values[b].layout.shape().dims() else {
            return Err(invalid("training conv2d weight must be rank four"));
        };
        if groups == 0
            || in_channels % groups != 0
            || out_channels % groups != 0
            || in_per_group != in_channels / groups
        {
            return Err(invalid("training conv2d channel/group mismatch"));
        }
        let spec = Conv2dSpec {
            batch,
            in_channels,
            out_channels,
            input: [ih, iw],
            kernel: [kh, kw],
            stride,
            padding,
            dilation,
            groups,
            transposed: false,
            output_padding: [0, 0],
        };
        let shape = spec.output_shape()?;
        self.compute(Operation::Conv2d(spec), [Some(a), Some(b)], &shape)
    }
    /// Unnormalized one-sided STFT with optional snapshotted length-`fft_size` window.
    pub fn stft(
        &mut self,
        input: Value,
        fft_size: usize,
        hop: usize,
        window: Option<&Tensor>,
    ) -> Result<Value> {
        let a = self.index(input)?;
        let &[batch, input_len] = self.values[a].layout.shape().dims() else {
            return Err(invalid(
                "training STFT input must be rank two [batch, samples]",
            ));
        };
        let spec = StftSpec {
            batch,
            input_len,
            fft_size,
            hop,
        };
        let shape = spec.output_shape()?;
        let window = match window {
            Some(w) => Some(Arc::<[f32]>::from(spec.checked_window(w)?)),
            None => None,
        };
        self.compute(Operation::Stft { spec, window }, [Some(a), None], &shape)
    }
    /// Reflect-centered one-sided STFT with optional snapshotted length-`fft_size` window.
    pub fn centered_stft(
        &mut self,
        input: Value,
        fft_size: usize,
        hop: usize,
        window: Option<&Tensor>,
    ) -> Result<Value> {
        let a = self.index(input)?;
        let &[batch, input_len] = self.values[a].layout.shape().dims() else {
            return Err(invalid(
                "training centered STFT input must be rank two [batch, samples]",
            ));
        };
        let spec = StftSpec {
            batch,
            input_len,
            fft_size,
            hop,
        }
        .centered()?;
        let window = match window {
            Some(w) => Some(Arc::<[f32]>::from(spec.spec().checked_window(w)?)),
            None => None,
        };
        let shape = spec.output_shape();
        self.compute(
            Operation::CenteredStft { spec, window },
            [Some(a), None],
            &shape,
        )
    }
    /// Reflect-centered inverse STFT with NOLA validation and optional snapshotted window.
    pub fn istft(
        &mut self,
        spectrum: Value,
        output_len: usize,
        fft_size: usize,
        hop: usize,
        window: Option<&Tensor>,
    ) -> Result<Value> {
        let a = self.index(spectrum)?;
        let &[batch, frames, bins, pair] = self.values[a].layout.shape().dims() else {
            return Err(invalid(
                "training iSTFT spectrum must be rank four [batch, frames, bins, 2]",
            ));
        };
        let spec = StftSpec {
            batch,
            input_len: output_len,
            fft_size,
            hop,
        }
        .centered()?;
        if frames != spec.frames() || bins != spec.bins() || pair != 2 {
            return Err(invalid("training iSTFT spectrum shape differs from spec"));
        }
        let coeffs = match window {
            Some(w) => spec.spec().checked_window(w)?,
            None => spec.spec().window(),
        };
        spec.normalization(&coeffs)?;
        let window = window.map(|_| Arc::<[f32]>::from(coeffs));
        let shape = spec.input_shape();
        self.compute(Operation::Istft { spec, window }, [Some(a), None], &shape)
    }
    /// Parameter-free LayerNorm over the last `normalized_dims` axes.
    pub fn layer_norm(&mut self, input: Value, normalized_dims: usize, eps: f32) -> Result<Value> {
        let a = self.index(input)?;
        let spec = NormSpec::new(
            NormKind::LayerNorm,
            self.values[a].layout.shape(),
            normalized_dims,
            eps,
        )?;
        let shape = spec.shape().dims().to_vec();
        self.compute(Operation::Norm(spec), [Some(a), None], &shape)
    }
    /// Parameter-free RmsNorm over the last `normalized_dims` axes.
    pub fn rms_norm(&mut self, input: Value, normalized_dims: usize, eps: f32) -> Result<Value> {
        let a = self.index(input)?;
        let spec = NormSpec::new(
            NormKind::RmsNorm,
            self.values[a].layout.shape(),
            normalized_dims,
            eps,
        )?;
        let shape = spec.shape().dims().to_vec();
        self.compute(Operation::Norm(spec), [Some(a), None], &shape)
    }
    /// Parameter-free GroupNorm over `[N, C, ...]` split into `num_groups` channel groups.
    pub fn group_norm(&mut self, input: Value, num_groups: usize, eps: f32) -> Result<Value> {
        let a = self.index(input)?;
        let orig_dims = self.values[a].layout.shape().dims().to_vec();
        if orig_dims.len() < 2 || num_groups == 0 || !orig_dims[1].is_multiple_of(num_groups) {
            return Err(invalid(
                "group_norm requires rank >= 2, num_groups > 0, and C % num_groups == 0",
            ));
        }
        let mut grouped = Vec::with_capacity(orig_dims.len() + 1);
        grouped.push(orig_dims[0]);
        grouped.push(num_groups);
        grouped.push(orig_dims[1] / num_groups);
        grouped.extend_from_slice(&orig_dims[2..]);
        let reshaped = self.reshape(input, &grouped)?;
        let normed = self.layer_norm(reshaped, orig_dims.len() - 1, eps)?;
        self.reshape(normed, &orig_dims)
    }
    /// Numerically stable Softmax along `axis`.
    pub fn softmax(&mut self, input: Value, axis: usize) -> Result<Value> {
        let a = self.index(input)?;
        let spec = SoftmaxSpec::new(SoftmaxKind::Softmax, self.values[a].layout.shape(), axis)?;
        let shape = spec.shape().dims().to_vec();
        self.compute(Operation::Softmax(spec), [Some(a), None], &shape)
    }
    /// Numerically stable LogSoftmax along `axis`.
    pub fn log_softmax(&mut self, input: Value, axis: usize) -> Result<Value> {
        let a = self.index(input)?;
        let spec = SoftmaxSpec::new(SoftmaxKind::LogSoftmax, self.values[a].layout.shape(), axis)?;
        let shape = spec.shape().dims().to_vec();
        self.compute(Operation::Softmax(spec), [Some(a), None], &shape)
    }
    /// Arbitrary-rank (`rank >= 2`) broadcast batched matrix multiplication (`left @ right`).
    pub fn bmm(&mut self, left: Value, right: Value) -> Result<Value> {
        let a = self.index(left)?;
        let b = self.index(right)?;
        let spec = BmmSpec::new(self.values[a].layout.shape(), self.values[b].layout.shape())?;
        let shape = spec.output_shape().dims().to_vec();
        self.compute(Operation::Bmm(spec), [Some(a), Some(b)], &shape)
    }
    /// Fused Flash-style Scaled Dot-Product Attention (`Q: [B,H,Tq,D]`, `K,V: [B,H_kv,Tk,D]`).
    pub fn sdpa(
        &mut self,
        query: Value,
        key: Value,
        value: Value,
        scale: Option<f32>,
        is_causal: bool,
    ) -> Result<Value> {
        let iq = self.index(query)?;
        let ik = self.index(key)?;
        let iv = self.index(value)?;
        let &[bq, hq, tq, dq] = self.values[iq].layout.shape().dims() else {
            return Err(invalid(
                "training sdpa query must be rank four [B, H, Tq, D]",
            ));
        };
        let &[bk, hk, tk, dk] = self.values[ik].layout.shape().dims() else {
            return Err(invalid(
                "training sdpa key must be rank four [B, H_kv, Tk, D]",
            ));
        };
        let &[bv, hv, tv, dv] = self.values[iv].layout.shape().dims() else {
            return Err(invalid(
                "training sdpa value must be rank four [B, H_kv, Tk, D]",
            ));
        };
        if bq != bk || bq != bv || hk != hv || tk != tv || dq != dk || dq != dv {
            return Err(invalid("training sdpa Q/K/V shapes are incompatible"));
        }
        let spec = SdpaSpec::new(bq, hq, hk, tq, tk, dq, scale, is_causal)?;
        let shape = spec.output_shape().dims().to_vec();
        self.compute3(
            Operation::Sdpa(spec),
            [Some(iq), Some(ik), Some(iv)],
            &shape,
        )
    }
    /// Differentiable 4D Rotary Positional Embedding (`RoPE`, half-split layout).
    pub fn rope(
        &mut self,
        input: Value,
        seq_axis: usize,
        rotary_dim: Option<usize>,
        base: f32,
        position_offset: usize,
    ) -> Result<Value> {
        let a = self.index(input)?;
        let spec = RopeSpec::new(
            self.values[a].layout.shape(),
            seq_axis,
            rotary_dim,
            base,
            position_offset,
        )?;
        let shape = spec.shape().dims().to_vec();
        self.compute(Operation::Rope(spec), [Some(a), None], &shape)
    }
    /// Differentiable 1D transposed convolution (`X: [N,C_in,L]`, `W: [C_in,C_out/g,K]`).
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &mut self,
        input: Value,
        weight: Value,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Value> {
        let a = self.index(input)?;
        let b = self.index(weight)?;
        let &[batch, in_channels, input_len] = self.values[a].layout.shape().dims() else {
            return Err(invalid(
                "training conv_transpose1d input must be rank three",
            ));
        };
        let &[w_in, out_per_group, kernel] = self.values[b].layout.shape().dims() else {
            return Err(invalid(
                "training conv_transpose1d weight must be rank three",
            ));
        };
        if groups == 0 || w_in != in_channels || in_channels % groups != 0 {
            return Err(invalid("training conv_transpose1d channel/group mismatch"));
        }
        let out_channels = out_per_group
            .checked_mul(groups)
            .ok_or_else(|| invalid("conv_transpose1d out_channels overflow"))?;
        let spec = ConvTranspose1dSpec {
            batch,
            in_channels,
            out_channels,
            input_len,
            kernel,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        };
        let shape = spec.output_shape()?;
        self.compute(Operation::ConvTranspose1d(spec), [Some(a), Some(b)], &shape)
    }
    /// Differentiable 2D transposed convolution (`X: [N,C_in,H,W]`, `W: [C_in,C_out/g,Kh,Kw]`).
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(
        &mut self,
        input: Value,
        weight: Value,
        stride: [usize; 2],
        padding: [usize; 2],
        output_padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Value> {
        let a = self.index(input)?;
        let b = self.index(weight)?;
        let &[batch, in_channels, ih, iw] = self.values[a].layout.shape().dims() else {
            return Err(invalid("training conv_transpose2d input must be rank four"));
        };
        let &[w_in, out_per_group, kh, kw] = self.values[b].layout.shape().dims() else {
            return Err(invalid(
                "training conv_transpose2d weight must be rank four",
            ));
        };
        if groups == 0 || w_in != in_channels || in_channels % groups != 0 {
            return Err(invalid("training conv_transpose2d channel/group mismatch"));
        }
        let out_channels = out_per_group
            .checked_mul(groups)
            .ok_or_else(|| invalid("conv_transpose2d out_channels overflow"))?;
        let spec = Conv2dSpec {
            batch,
            in_channels,
            out_channels,
            input: [ih, iw],
            kernel: [kh, kw],
            stride,
            padding,
            dilation,
            groups,
            transposed: true,
            output_padding,
        };
        let shape = spec.output_shape()?;
        self.compute(Operation::ConvTranspose2d(spec), [Some(a), Some(b)], &shape)
    }
    /// Differentiable per-axis `Pad` (`Constant`, `Reflect`, or `Replicate`).
    pub fn pad(&mut self, input: Value, pads: &[(usize, usize)], mode: PadMode) -> Result<Value> {
        let a = self.index(input)?;
        let spec = PadSpec::new(self.values[a].layout.shape(), pads, mode)?;
        let shape = spec.output_shape().dims().to_vec();
        self.compute(Operation::Pad(spec), [Some(a), None], &shape)
    }
    /// Differentiable binary concatenation along `axis`.
    pub fn cat(&mut self, left: Value, right: Value, axis: usize) -> Result<Value> {
        let a = self.index(left)?;
        let b = self.index(right)?;
        let spec = CatSpec::new(
            self.values[a].layout.shape(),
            self.values[b].layout.shape(),
            axis,
        )?;
        let shape = spec.output_shape().dims().to_vec();
        self.compute(Operation::Cat(spec), [Some(a), Some(b)], &shape)
    }
    /// Convenience multi-tensor concatenation chaining `cat` along `axis`.
    pub fn cat_all(&mut self, inputs: &[Value], axis: usize) -> Result<Value> {
        if inputs.is_empty() {
            return Err(invalid("cat_all requires at least one input"));
        }
        let mut acc = inputs[0];
        for &next in &inputs[1..] {
            acc = self.cat(acc, next, axis)?;
        }
        Ok(acc)
    }
    fn alias(&mut self, input: Value, layout: Layout, detach: bool) -> Result<Value> {
        let i = self.index(input)?;
        let source = &self.values[i];
        layout.validate_storage(self.values[source.root].layout.shape().len())?;
        let definition = Definition {
            layout: Arc::new(layout),
            root: source.root,
            requires_grad: source.requires_grad && !detach,
            kind: Kind::Alias(i),
        };
        let index = self.values.len();
        self.values.push(definition);
        Ok(self.value(index))
    }
    /// Explicit absolute storage-root layout; does not materialize the view.
    pub fn view(&mut self, input: Value, layout: &Layout) -> Result<Value> {
        self.alias(input, layout.clone(), false)
    }
    pub fn permute(&mut self, input: Value, axes: &[usize]) -> Result<Value> {
        self.alias(input, self.layout(input)?.permute(axes)?, false)
    }
    pub fn slice(
        &mut self,
        input: Value,
        axis: usize,
        start: usize,
        count: usize,
        step: isize,
    ) -> Result<Value> {
        self.alias(
            input,
            self.layout(input)?.slice(axis, start, count, step)?,
            false,
        )
    }
    pub fn broadcast_to(&mut self, input: Value, shape: impl AsRef<[usize]>) -> Result<Value> {
        self.alias(input, self.layout(input)?.broadcast_to(shape)?, false)
    }
    pub fn reshape(&mut self, input: Value, shape: impl AsRef<[usize]>) -> Result<Value> {
        self.alias(input, self.layout(input)?.reshape(shape)?, false)
    }
    pub fn detach(&mut self, input: Value) -> Result<Value> {
        self.alias(input, self.layout(input)?.clone(), true)
    }
    pub fn finish(self, roots: &[Value]) -> Result<Program> {
        if roots.is_empty() {
            return Err(invalid("training program needs an output root"));
        }
        for &v in roots {
            self.index(v)?;
        }
        let mut reachable = vec![false; self.values.len()];
        let mut gradients = reachable.clone();
        for &v in roots {
            reachable[v.index] = true;
            if self.values[v.index].requires_grad {
                gradients[self.values[v.index].root] = true;
            }
        }
        for i in (0..self.values.len()).rev() {
            let value = &self.values[i];
            if reachable[i] {
                match &value.kind {
                    Kind::Alias(source) => reachable[*source] = true,
                    Kind::Compute { inputs, .. } => {
                        for source in inputs.iter().flatten() {
                            reachable[*source] = true;
                        }
                    }
                    Kind::Leaf => (),
                }
            }
            if gradients[i] {
                if let Kind::Compute { inputs, .. } = &value.kind {
                    for &source in inputs.iter().flatten() {
                        if self.values[source].requires_grad {
                            gradients[self.values[source].root] = true;
                        }
                    }
                }
            }
        }
        let forward = self
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                (reachable[i] && matches!(v.kind, Kind::Compute { .. })).then_some(i)
            })
            .collect();
        Ok(Program(Arc::new(DefinitionGraph {
            owner: self.owner,
            values: self.values.into_boxed_slice(),
            roots: roots.into(),
            leaves: self.leaves.into_boxed_slice(),
            forward,
            gradient_roots: gradients.into_boxed_slice(),
        })))
    }
}

/// Exact leaf binding. Shared ownership does not copy its payload.
pub struct Binding<T: Trackable> {
    pub(crate) value: Value,
    pub(crate) owner: Shared<T>,
}
impl<T: Trackable> Binding<T> {
    pub fn new(value: Value, owner: Shared<T>) -> Self {
        Self { value, owner }
    }
}
/// Borrowed root seed with geometry fixed by preparation.
pub struct Seed<'a, T: Trackable> {
    pub(crate) value: Value,
    pub(crate) tensor: &'a T,
    pub(crate) layout: &'a Layout,
}
impl<'a, T: Trackable> Seed<'a, T> {
    pub fn new(value: Value, tensor: &'a T, layout: &'a Layout) -> Self {
        Self {
            value,
            tensor,
            layout,
        }
    }
}
pub struct PrepareOptions<'a> {
    pub seed_layouts: Option<&'a [Layout]>,
    pub view_index_limit_bytes: usize,
}
impl Default for PrepareOptions<'_> {
    fn default() -> Self {
        Self {
            seed_layouts: None,
            view_index_limit_bytes: 16 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientStatus {
    Unused,
    Uncomputed,
    Valid,
    Invalid,
}
/// Explicit payload/plan accounting; fixed control headers and vendor internals excluded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    pub activation_bytes: usize,
    pub gradient_bytes: usize,
    pub edge_scratch_bytes: usize,
    pub scalar_seed_bytes: usize,
    pub operator_workspace_bytes: usize,
    pub view_index_bytes: usize,
    pub operator_host_metadata_bytes: usize,
    pub max_static_shared_bytes_per_block: usize,
}
/// Output access keeps the tape and original shared owner immutably borrowed.
///
/// ```compile_fail
/// use fast_audio::{Shared,Tensor,autograd::{Binding,Builder},cpu::{Cpu,Tape}};
/// let mut builder=Builder::new().unwrap();
/// let root=builder.leaf([1],false).unwrap();
/// let program=builder.finish(&[root]).unwrap();
/// let owner=Shared::new(Tensor::zeros([1]).unwrap()).unwrap();
/// let mut tape=Tape::new(Cpu::Reference,&program,&[Binding::new(root,owner)]).unwrap();
/// tape.forward().unwrap();
/// let reader=tape.output(root).unwrap();
/// tape.forward().unwrap();
/// assert_eq!(reader.storage().data(),&[0.]);
/// ```
pub struct ReadValue<'a, T: Trackable> {
    pub(crate) storage: ReadGuard<'a, T>,
    pub(crate) layout: &'a Layout,
}
impl<T: Trackable> ReadValue<'_, T> {
    pub fn storage(&self) -> &T {
        &self.storage
    }
    pub fn layout(&self) -> &Layout {
        self.layout
    }
}
impl ReadValue<'_, Tensor> {
    pub fn view(&self) -> Result<crate::TensorView<'_>> {
        self.storage.view(self.layout)
    }
}
#[cfg(feature = "cuda")]
impl ReadValue<'_, crate::cuda::DeviceTensor> {
    pub fn view(&self) -> Result<crate::cuda::DeviceView<'_>> {
        self.storage.view(self.layout)
    }
}

macro_rules! tape_methods {
    ($tensor:ty) => {
        pub fn forward(&mut self) -> crate::Result<()> {
            self.core.forward()
        }
        pub fn backward(
            &mut self,
            seeds: &[crate::autograd::Seed<'_, $tensor>],
            mode: crate::GradientMode,
        ) -> crate::Result<()> {
            self.core.backward(seeds, mode)
        }
        pub fn backward_scalar(&mut self, mode: crate::GradientMode) -> crate::Result<()> {
            self.core.backward_scalar(mode)
        }
        pub fn output(
            &self,
            root: crate::autograd::Value,
        ) -> crate::Result<crate::autograd::ReadValue<'_, $tensor>> {
            self.core.output(root)
        }
        pub fn gradient_status(
            &self,
            leaf: crate::autograd::Value,
        ) -> crate::Result<crate::autograd::GradientStatus> {
            self.core.gradient_status(leaf)
        }
        pub fn gradient(&self, leaf: crate::autograd::Value) -> crate::Result<Option<&$tensor>> {
            self.core.gradient(leaf)
        }
        pub fn zero_grad(&mut self) -> crate::Result<()> {
            self.core.zero_grad()
        }
        pub fn sgd(&mut self, leaves: &[crate::autograd::Value], rate: f32) -> crate::Result<()> {
            self.core.sgd(leaves, rate)
        }
        pub fn synchronize(&mut self) -> crate::Result<()> {
            self.core.synchronize()
        }
        pub fn resources(&self) -> crate::autograd::Resources {
            self.core.resources
        }
    };
}
pub(crate) use tape_methods;
