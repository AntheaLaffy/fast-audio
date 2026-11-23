---
mode: maintenance
generated_at: 2026-10-17
---

## Summary

Byte-identical official GTCRN snapshots; adapt execution outside these immutable files.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Run or inspect the pinned model | Reference | `upstream.json`, `gtcrn.py` | `../../tools/gtcrn_reference.py`, `../../docs/gtcrn.md` |

## Files

| File | Domain | Function |
| --- | --- | --- |
| `gtcrn.py` | Reference | Frozen official ERB, convolution, attention/dual-path GRU and complex-mask model; imports PyTorch, NumPy and einops |
| `upstream_infer.py` | Contract | Frozen original inference wrapper documenting sqrt-Hann and 16-kHz framing; not the runnable project entry point |
| `upstream.json` | Reproduction | Immutable revision, per-file/checkpoint hashes, window/sample-rate and fixture provenance |
| `LICENSE` | Attribution | Unmodified upstream MIT license and copyright notice |
| `requirements-reference.txt` | Environment | Extra CPU-reference dependency pins; install the platform-specific PyTorch stack separately |
