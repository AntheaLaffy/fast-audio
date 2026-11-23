---
mode: maintenance
generated_at: 2026-10-20
---

## Summary

Authoritative operator workload definitions and reproducibility artifacts.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Add a shape stratum or change workload semantics | Measurement | `workloads.json` | `../src/bin/fa-bench.rs`, `../tools/compare_torch.py`, measurement contract |
| Reproduce a CPU baseline environment | Environment | `requirements-cpu.txt` | Recorded `comparison.json` environment and `../README.md` |
| Inspect recorded evidence | Reproduction | `results/` | `../docs/progress.md`, `../tools/verify_archive.py` |
| Characterize pointwise operators and layouts | Tensor measurement | `protocols/pointwise-characterization-v2.json` | `../tools/pointwise_benchmark.py`, `../src/bin/fa-pointwise.rs`, declared numerical/mode boundaries; v1 remains a failed preserved declaration |
| Characterize complete native training steps | Training measurement | `protocols/training-characterization-v1.json` | `../docs/training-characterization-v1.md`, resident workers, independent full-trajectory controls and fixed evidence/mode/reset boundaries |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
| --- | --- | --- | --- |
| `results/` | Reproduction | Measurement harness | Boundary only; internals not indexed. Raw reports, sources/executables, compressed or chunked exact profiler streams with restore receipts, archived analysis scripts and sealed/regenerable FP32 payloads |
| [protocols/](protocols/CODEMAP.md) | Measurement | Collector and assessor | Predeclared measurement policies and artifact identities |

## Files

| File | Domain | Function |
| --- | --- | --- |
| `workloads.json` | Measurement | Versioned, shared seven-case matrix/Conv1d/STFT workload suite |
| `requirements-cpu.txt` | Environment | Observed exact CPU comparison Python package versions and index |
| `champions.json` | Reproduction | Current accepted artifact per hardware stratum, assessment and source/binary identity |
