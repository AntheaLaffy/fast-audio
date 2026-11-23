---
mode: maintenance
generated_at: 2026-10-17
---

## Summary

Prepared CPU execution of the shared fixed-shape inference graph.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Change graph CPU math or parameter preparation | Intel | `host.rs` | `../graph.rs`, public primitive plans, independent FP64 tests |

## Files

| File | Domain | Deps | Function |
| --- | --- | --- | --- |
| `host.rs` | Graph / Intel | ← shared executor/descriptors, CPU convolution/GRU/GEMM | Public CPU graph wrapper, immutable parameters, row-copy concatenation, padding and frequency unfold, tensor math and workspace accounting |
| `mapping.rs` | Graph addressing | ← checked layout/broadcast descriptors; → host and CUDA graph preparation | Coalesced mapped rows, bounded 60-byte CUDA indexing descriptors, exact reciprocal arithmetic and large-index fallback checks |
