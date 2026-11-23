---
mode: maintenance
generated_at: 2026-10-16
---

## Summary

Prepared-operator benchmarks emit complete outputs and raw latency samples; pointwise and training workers keep buffers resident across counterbalanced requests.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Change fixtures, CLI, timing or output schema | Measurement | `fa-bench.rs` | `../../tools/compare_torch.py`, `../../benchmarks/workloads.json` |
| Change persistent strided pointwise timing or profiler boundaries | Tensor measurement | `fa-pointwise.rs` | `../../tools/pointwise_benchmark.py`, `../../benchmarks/protocols/pointwise-characterization-v2.json`, `../../docs/pointwise-characterization-v2.md` |
| Change complete training timing, reset or snapshot boundaries | Training measurement | `fa-training.rs` | `../../tools/training_benchmark.py`, immutable training protocol, independent analytic/eager trajectories and whole-step/capture/resource guards |

## Files

| File | Domain | Deps | Function |
| --- | --- | --- | --- |
| `fa-bench.rs` | Measurement | ← library API, `../../benchmarks/workloads.json`; → `../../tools/compare_torch.py` | Deterministic inputs, explicit/shape-selected vendor backends, warmed timing, implementation/workspace metadata and full outputs |
| `fa-pointwise.rs` | Tensor measurement | ← prepared unary/binary APIs, fixed protocol; → persistent collector | Resident CPU/CUDA buffers, exact fixture/backing dumps, black-boxed checked calls, per-request windows, source/feature/affinity metadata and explicit diagnostic CUDA profiler range |
| `fa-training.rs` | Training measurement | ← native CPU/CUDA tapes and fixed protocol; → persistent training collector | Twelve dense/waveform/spectral graphs, resident forward/backward/SGD (plus explicit `--captured` CUDA Graph mode), exact resets, detached diagnostic roots, post-update snapshots, hot allocation accounting and separate profiler loops |
