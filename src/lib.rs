//! Audio tensor and execution primitives with explicit storage, reusable plans, and vendor kernels.
//!
//! `mkl` enables Intel oneMKL. `cuda` enables the NVIDIA CUDA backend. Without
//! either feature the CPU implementation is a correctness reference.
mod adjoint;
pub mod autograd;
mod bmm;
mod conv;
mod conv2d;
mod copy;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
mod dropout;
mod dtype;
mod embedding;
mod error;
pub mod graph;
mod gru;
mod layout;
mod loss;
mod matmul;
pub mod models;
mod norm;
#[cfg(feature = "onednn")]
pub mod onednn;
mod optim;
mod pointwise;
mod pointwise_backward;
mod reduce;
mod rope;
mod sdpa;
mod softmax;
mod spec;
mod stft;
mod storage;
mod structural;
mod tensor;
mod view;

pub use adjoint::{GradientMode, ViewBackwardStrategy};
pub use bmm::BmmSpec;
pub use conv::{CONV_WORKSPACE_LIMIT, ConvStrategy};
pub use conv2d::Conv2dSpec;
pub use dropout::{DropoutSpec, philox4x32_10, u32_to_unit_f32};
pub use dtype::{
    Bf16Tensor, DType, IndexTensor, PrecisionPolicy, bf16_to_f32, f16_to_f32, f32_to_bf16,
    f32_to_f16,
};
pub use embedding::EmbeddingSpec;
pub use error::{Error, Result};
pub use gru::{GruSpec, GruWeights};
pub use layout::{Layout, LayoutOffsets};
pub use loss::CrossEntropySpec;
pub use matmul::{MATMUL_WORKSPACE_LIMIT, MatmulStrategy};
pub use norm::{NormKind, NormSpec};
pub use optim::{AdamWConfig, Checkpoint};
pub use pointwise::{BinaryOp, PointwiseStrategy, UnaryOp};
pub use pointwise_backward::BinaryOperand;
pub use reduce::{ReduceOp, ReduceSpec, ReduceStrategy};
pub use rope::RopeSpec;
pub use sdpa::{SdpaOperand, SdpaSpec};
pub use softmax::{SoftmaxKind, SoftmaxSpec};
pub use spec::{Activation, CenteredStftSpec, Conv1dSpec, ConvTranspose1dSpec, StftSpec};
pub use storage::{
    ReadGuard, SavedRead, SavedValue, Shared, StorageId, StorageVersion, Trackable, Tracked,
    WriteGuard,
};
pub use structural::{CatSpec, PadMode, PadSpec};
pub use tensor::{Shape, Tensor};
pub use view::{Elements, ElementsMut, TensorView, TensorViewMut};
