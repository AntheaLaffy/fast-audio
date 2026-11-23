---
mode: maintenance
generated_at: 2026-10-17
---

## Summary

Runnable public-API examples with complete outputs for independent validation.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Reconstruct a waveform or emit PyTorch comparison fixtures | Audio | `audio-roundtrip.rs` | `../tools/check_audio.py`, `../docs/architecture.md` |
| Replay real-model GRU layers with native plans | Sequence | `gru-reference.rs` | `../tools/check_gru.py`, pinned reference captures |
| Replay exact GTCRN spectra and all spatial layers | Model | `gtcrn-stages.rs` | `../tools/check_gtcrn_stages.py`, fresh convolution captures |
| Execute complete native GTCRN on waveforms | Model | `gtcrn-infer.rs` | `../src/models/`, `../tools/check_gtcrn.py`, exported parameters and full traces |
| Benchmark differentiable STFT/iSTFT training | Audio / Training | `spectral-train.rs` | `../docs/stft-backward-v1.md`, CPU MKL/L1-fusion and CUDA cuFFT/NVRTC/CUDA-Graph spectral steps |
| Verify and benchmark foundation operators and training steps against PyTorch | Foundation / Measurement | `audio-foundation-bench.rs` | `../tools/check_new_ops.py`, `../tools/bench_foundation_vs_pytorch.py`, CPU and CUDA strict-FP32 controls |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
| --- | --- | --- | --- |
| [support/](support/CODEMAP.md) | Fixtures | Public host tensor API; serde | Shared FP32 tensor loading and output writing for model replay examples |

## Files

| File | Domain | Deps | Function |
| --- | --- | --- | --- |
| `audio-bf16-bench.rs` | Training / Measurement | ← public CPU and CUDA prepared operators and BF16 trainer | Numerical export and latency worker with CUDA serial wall/event samples, matched initial Transformer weights and a separate general-shape attention probe |
| `audio-roundtrip.rs` | Audio | ← CPU/CUDA public plans; → `../tools/check_audio.py` | Six fixed small/long stereo cases with full input, spectrum, roundtrip and modified-spectrum inverse outputs |
| `gru-reference.rs` | Sequence | ← captured GTCRN tensors, `support/` and public GRU plans; → `../tools/check_gru.py` | Snapshot parameters, replay all 56 layer/shape cases, record outputs/states, workspace and exploratory warm samples |
| `gtcrn-stages.rs` | Model | ← captured GTCRN tensors, `support/`, explicit-window and spatial plans; → stage checker | Replay four exact-window audio cases and 88 isolated convolution instances; record full outputs, preparation/resources and exploratory layer samples |
| `gtcrn-infer.rs` | Model fixture | ← named parameter bundle, graph/model builder, spectral plans, `support/`; → complete checker and paired collector | Existing complete regression pipelines, case/trace controls, explicit cuDNN convolution-policy selection and actual engine/resource metadata |
| `spectral-train.rs` | Audio / Training | ← `autograd::Builder::{centered_stft, istft}`, `cpu::Tape`, `cuda::Tape` | End-to-end differentiable `CenteredStft -> mask+bias -> Istft -> MSE + SGD` benchmark and snapshot exporter for CPU, CUDA and CUDA Graph capture/replay |
| `audio-foundation-bench.rs` | Foundation / Training / Measurement | ← public CPU/CUDA operator plans, `autograd::Builder`, `cpu::{AdamW, Tape}`, `cuda::{AdamW, Tape}` | Unified CPU/CUDA strict-FP32 verification exporter (`--mode verify-json` for `../tools/check_new_ops.py`) and benchmark suite (`--mode bench` for `../tools/bench_foundation_vs_pytorch.py`) covering `training-v1` spectrum/waveform, `LayerNorm`/`RmsNorm`, `Softmax`, `Bmm`, Flash `Sdpa`, Audio Transformer + AdamW and Spectral-Vocoder U-Net + AdamW |
