//! Prepared differentiable token `Embedding` lookup and zero-atomic deterministic backward on CPU.

use crate::{GradientMode, IndexTensor, Result, Shape, Tensor, error::invalid};

/// Validated geometry for token `Embedding`: `table [vocab_size, embed_dim]` indexed by
/// `indices [..index_shape]` -> `output [..index_shape, embed_dim]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingSpec {
    vocab_size: usize,
    embed_dim: usize,
    index_shape: Shape,
    output_shape: Shape,
    padding_idx: Option<i64>,
}

impl EmbeddingSpec {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        index_shape: &Shape,
        padding_idx: Option<i64>,
    ) -> Result<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(invalid(
                "embedding vocab_size and embed_dim must be positive",
            ));
        }
        if let Some(pad) = padding_idx {
            if pad < 0 || (pad as u64) >= vocab_size as u64 {
                return Err(invalid("embedding padding_idx out of [0, vocab_size)"));
            }
        }
        let mut out_dims = index_shape.dims().to_vec();
        out_dims.push(embed_dim);
        let output_shape = Shape::new(out_dims)?;
        Ok(Self {
            vocab_size,
            embed_dim,
            index_shape: index_shape.clone(),
            output_shape,
            padding_idx,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
    pub fn index_shape(&self) -> &Shape {
        &self.index_shape
    }
    pub fn output_shape(&self) -> &Shape {
        &self.output_shape
    }
    pub fn table_shape(&self) -> Result<Shape> {
        Shape::new([self.vocab_size, self.embed_dim])
    }
    pub fn padding_idx(&self) -> Option<i64> {
        self.padding_idx
    }
    pub fn num_indices(&self) -> usize {
        self.index_shape.len()
    }
}

/// Prepared CPU `Embedding` forward plan (zero hot allocations).
pub struct EmbeddingPlan {
    spec: EmbeddingSpec,
}

impl EmbeddingPlan {
    pub fn new(spec: EmbeddingSpec) -> Self {
        Self { spec }
    }

    pub fn spec(&self) -> &EmbeddingSpec {
        &self.spec
    }

    pub fn run(&self, table: &Tensor, indices: &IndexTensor, output: &mut Tensor) -> Result<()> {
        table
            .shape()
            .expect(&[self.spec.vocab_size, self.spec.embed_dim])?;
        indices.shape().expect(self.spec.index_shape.dims())?;
        output.shape().expect(self.spec.output_shape.dims())?;

        let v = self.spec.vocab_size as i64;
        let d = self.spec.embed_dim;
        let w = table.data();
        let idx_data = indices.data();
        let out = output.data_mut();

        // Validate every token index before writing output.
        for &tok in idx_data {
            if tok < 0 || tok >= v {
                return Err(invalid(format!(
                    "embedding index {tok} out of range [0, {v})"
                )));
            }
        }
        for (pos, &tok) in idx_data.iter().enumerate() {
            let dst = &mut out[pos * d..(pos + 1) * d];
            let row = tok as usize;
            dst.copy_from_slice(&w[row * d..(row + 1) * d]);
        }
        Ok(())
    }
}

/// Prepared CPU `Embedding` local VJP (`dTable` scatter-add in fixed token order, zero allocations).
pub struct EmbeddingBackward {
    spec: EmbeddingSpec,
}

impl EmbeddingBackward {
    pub fn new(spec: EmbeddingSpec) -> Self {
        Self { spec }
    }

    pub fn spec(&self) -> &EmbeddingSpec {
        &self.spec
    }

    pub fn run(
        &self,
        indices: &IndexTensor,
        upstream: &Tensor,
        grad_table: &mut Tensor,
        mode: GradientMode,
    ) -> Result<()> {
        indices.shape().expect(self.spec.index_shape.dims())?;
        upstream.shape().expect(self.spec.output_shape.dims())?;
        grad_table
            .shape()
            .expect(&[self.spec.vocab_size, self.spec.embed_dim])?;

        let v = self.spec.vocab_size as i64;
        let d = self.spec.embed_dim;
        let pad = self.spec.padding_idx;
        let idx_data = indices.data();
        for &tok in idx_data {
            if tok < 0 || tok >= v {
                return Err(invalid(format!(
                    "embedding index {tok} out of range [0, {v})"
                )));
            }
        }
        let gw = grad_table.data_mut();
        if mode == GradientMode::Overwrite {
            gw.fill(0.0);
        }
        let dy = upstream.data();
        for (pos, &tok) in idx_data.iter().enumerate() {
            if Some(tok) == pad {
                continue;
            }
            let row = tok as usize;
            let dst = &mut gw[row * d..(row + 1) * d];
            let src = &dy[pos * d..(pos + 1) * d];
            for j in 0..d {
                dst[j] += src[j];
            }
        }
        Ok(())
    }
}
