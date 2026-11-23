---
mode: maintenance
generated_at: 2026-10-20
---

## Summary

Prepared native reverse-mode execution, shared owner/version checks and explicit parameter updates.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Change value identity, shapes, metadata aliases or reachability | Training topology | `../autograd.rs` | `prepare.rs`, `../../docs/autodiff-v1.md`, public CPU/CUDA tape tests |
| Change forward epochs, selective saves or reverse accumulation | Training execution | `engine.rs` | `prepare.rs`, `../storage.rs`, local VJP/view-adjoint contracts and failure-state tests |
| Change canonical gradients, scratch, seeds or resource budgets | Preparation | `prepare.rs`, `backend.rs` | `engine.rs`, CPU/CUDA plan resource queries, tied/overlap/empty roots |
| Change SGD selection, versions or failure behavior | Parameter updates | `update.rs` | `../storage.rs`, `host.rs`, `../cuda/tape.rs`, `../../native/training.cu` |
| Change CPU tape primitive dispatch or fused CPU step execution | Intel / Reference | `host.rs`, `fusion.rs` | Shared engine, public operator plans and `../../tests/autograd.rs` |
| Change whole-step DAG fusion analysis or CUDA NVRTC codegen | Fusion / Compilation | `fusion.rs` | `engine.rs`, `host.rs`, `../cuda/tape.rs`, `../../native/nvrtc_fusion.cu` |

## Files

| File | Domain | Function |
| --- | --- | --- |
| `backend.rs` | Execution contract | Private arity-3 typed primitive adapter, borrowed read/write arguments and explicit plan resource accounting |
| `prepare.rs` | Training preparation | Exact leaf binding, reachable storage/plans, exact activation liveness analysis and dead-buffer storage reuse (`free_after < pos`), tied owner gradient deduplication, identity/zero-contribution projection elimination, bounded view indices and fixed root seeds |
| `engine.rs` | Reverse mode | Prepared arity-3 activation/gradient state, guarded outputs, per-node saved versions, ordered roots/edges, fresh internal accumulation, leaf validity, shared step/barrier preflight/split-borrow helpers, generic optimizer-update transaction (`apply_optimizer_update`), CUDA-Graph capture validation helpers and explicit completion |
| `fusion.rs` | Whole-step fusion | General tape DAG analyzer (`WholeRegion` and `BarrierWithTail`, residual-skip pre-barrier leaf protection, bijective alias pull-back, `SmallReduce` inlining, parameter-reduction classification), zero-allocation CPU fused executor with runtime AVX2+FMA & 2D L1 cache-blocked micro-transpose fast kernels (`ComplexAffinePowerMse`, `AffineUnaryMse`) + MKL VML, and strict-FP32 CUDA NVRTC kernel generator across four launch topologies |
| `update.rs` | Training updates | SGD rate/leaf/epoch validation, deduplicated parameter selection, nonmutating preflight and checked writers with partial-failure invalidation |
| `host.rs` | Intel / Reference | Public CPU tape wrapper (`step`, `step_scalar`, `apply_adamw`, `step_adamw`, `step_scalar_adamw`, `has_fused_plan`), L1-tiled fused step execution with unfused fallback, all forward/local-VJP/view plans (including `Norm`, `Softmax`, `Bmm`, `Sdpa`, `Conv1d`/`2d`, `ConvTranspose1d`/`2d`, `Stft`/`CenteredStft`/`Istft`, `Pad`, `Cat`), explicit zeroing and ordered FP32 SGD/AdamW without hot buffers |
| `tests.rs` | Failure-state correctness | Real CPU execution with alias writes between forward nodes or after partial backward/SGD work; snapshot rejection, checked writers, invalid gradients, unwind-safe seeds and overwrite/zero recovery |
