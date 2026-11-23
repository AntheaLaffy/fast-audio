# Native Fused AdamW, Global Gradient Clipping, and Cross-Device Checkpointing (`optim-v1`)

## Scope

`fast-audio` provides first-class strict-FP32 optimization and portable checkpointing directly integrated with `cpu::Tape` and `cuda::Tape`:

- **`AdamWConfig`**: decoupled weight decay AdamW hyperparameters (`lr`, `beta1`, `beta2`, `eps`, `weight_decay`, optional global L2 `max_grad_norm`).
- **`cpu::AdamW`** and **`cuda::AdamW`**: preallocated first/second moment (`m`, `v`) state bound to a `Program` leaf set, with zero hot-loop heap allocations.
- **Deterministic Global L2 Gradient Clipping**:
  - CPU computes the global squared norm across all selected parameter gradients in FP64 (`scale = min(max_grad_norm / (norm + 1e-6), 1.0)`).
  - CUDA computes deterministic two-stage FP64 block-partial reductions into a preallocated device buffer followed by a single-block finalize kernel, avoiding non-deterministic `atomicAdd` and device-to-host synchronization.
- **Single-CUDA-Graph AdamW with Dynamic Learning Rate**:
  - `cuda::Tape::step_adamw_captured` and `cuda::Tape::step_scalar_adamw_captured` capture the entire training step — forward (with NVRTC fusion when eligible), reverse backward, deterministic global L2 gradient clipping, device step increment (`d_step += 1`), and fused single-pass AdamW updates for all parameters — into a single CUDA Graph.
  - Learning rate updates (`cuda::AdamW::set_lr(new_lr)`) update a 4-byte device scalar asynchronously via a 1-thread kernel so cosine/linear learning-rate schedules replay the existing CUDA Graph with zero re-captures.
- **Portable Cross-Device Binary Checkpointing (`Checkpoint`)**:
  - Deterministic little-endian binary format (`FACKPT01`, schema v1) storing parameter tensors, optimizer step count, `AdamWConfig`, and per-parameter `(m, v)` moment tensors with an FNV-1a 64-bit integrity checksum.
  - Transfers seamlessly between CPU and CUDA without external serialization dependencies, rejecting truncated, corrupted, mismatched-shape, or duplicate-key archives explicitly.

## Update Equations

For step $t \ge 1$, gradient scale $s \in (0, 1]$, and decoupled weight decay $\lambda \ge 0$:

$$
\tilde{g}_i = s \cdot g_i, \qquad
m_i \leftarrow \beta_1 m_i + (1 - \beta_1)\tilde{g}_i, \qquad
v_i \leftarrow \beta_2 v_i + (1 - \beta_2)\tilde{g}_i^2
$$

$$
p_i \leftarrow p_i (1 - \eta \lambda) - \frac{\eta}{1 - \beta_1^t} \cdot \frac{m_i}{\sqrt{v_i} / \sqrt{1 - \beta_2^t} + \epsilon}
$$

Bias corrections $1 - \beta_1^t$ and $1 - \beta_2^t$ and the global L2 gradient norm are accumulated in FP64 to prevent precision loss at large step counts and wide parameter sets; parameter and moment storage remain strictly FP32.
