---
mode: maintenance
generated_at: 2026-10-17
---

## Summary

Architecture adapters lowering named FP32 parameters to the shared native graph.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Change GTCRN architecture or BatchNorm folding | Model | `gtcrn.rs` | `../../models/gtcrn/gtcrn.py`, full reference traces and `../../tools/check_gtcrn.py` |

## Files

| File | Domain | Deps | Function |
| --- | --- | --- | --- |
| `gtcrn.rs` | Model | ← public graph, Conv2d/GRU contracts; pinned architecture | ERB/SFE, grouped temporal attention, dual-path recurrence, folded inference BatchNorm, decoder skips and complex masking with 15 named trace points |
