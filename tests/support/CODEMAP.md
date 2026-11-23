---
mode: maintenance
generated_at: 2026-10-19
---

## Summary

Shared integration-test helpers; each test executable installs its own thread-local allocation guard.

## Files

| File | Domain | Function |
| --- | --- | --- |
| `mod.rs` | Correctness | Counting allocator, warm execution guard, enabled CPU list, deterministic tensors and FP64 tolerance checks |
| `reduce.rs` | Reduction oracles | Shared independent Cartesian Sum/Mean/local-gradient references, signed/padded/generated/empty fixtures, backing checks and adjoint/finite differences; included directly by the two reduction test executables |
| `pointwise_backward.rs` | Local-gradient oracles | Independent Cartesian broadcast VJPs, FP64 analytic forward/derivative and finite-difference objectives, 524 operation/geometry combinations including aliased upstream/permuted gradients, special/tail/tree fixtures, backing checks and public CPU plan adapter; included by the two pointwise backward test executables |
| `matmul.rs` | Dense algebra oracles | Independent affine FP64 forward/both-VJP/adjoint/finite-difference controls, 389 strided/transpose/tile/empty cases, backing checks, eight-step analytic training control and optional raw FP32 exports for a separate Torch process |
| `conv.rs` | Spatial gradient oracles | Independent coordinate FP64 1D/2D forward/dX/dW/adjoint/finite-difference oracles, 64 1D + 56 2D strided/grouped/dilated/padded/pointwise fixtures, backing/hole checks, 8-step Conv1d+Conv2d analytic training control and optional raw FP32 JSONL exports (`FA_CONV_EXPORT_DIR`) |
| `autograd.rs` | Training composition oracles | Shared nonlinear transposed/repeated-operand and asymmetric broadcast graphs, changed bindings, independent FP64 objectives/leaf derivatives and all-coordinate central differences for every current unary/binary operation |
| `autograd_audio.rs` | Procedural training oracles | Multichannel waveform and five-dimensional spectral-pair graphs with reversed/permuted predictions, independent physical-coordinate FP64 derivatives and all-parameter central differences |
| `autograd_export.rs` | Training reference transport | Exclusive optional raw FP32 tensor/shape records, backend labels and complete composition exports outside hot allocation guards |
