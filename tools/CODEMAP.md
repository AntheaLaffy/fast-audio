---
mode: maintenance
generated_at: 2026-10-20
---

## Summary

Full-output PyTorch comparison and checksum-verified source archive restoration.

## Task Guide

| Task | Domain | Target | Also Check |
| --- | --- | --- | --- |
| Change baseline generation or statistics | Measurement | `compare_torch.py` | `../src/bin/fa-bench.rs`, `../docs/benchmarking.md` |
| Recover a measured source snapshot | Reproduction | `verify_archive.py` | The result report and recorded environment |
| Check centered audio against PyTorch | Correctness | `check_audio.py` | `../examples/audio-roundtrip.rs`, `../docs/architecture.md` |
| Capture a complete GTCRN reference or diagnose its operators | Model | `gtcrn_reference.py` | `../models/gtcrn/`, `../docs/gtcrn.md` |
| Verify GRUs on actual model activations and weights | Sequence | `check_gru.py` | `../examples/gru-reference.rs`, reference hashes and layer inventory |
| Verify exact windows and actual 2D model convolutions | Model | `check_gtcrn_stages.py` | `../examples/gtcrn-stages.rs`, hash-verified convolution captures |
| Verify complete native speech enhancement | Model | `check_gtcrn.py` | `../examples/gtcrn-infer.rs`, named weight bundle and all 15 model trace points |
| Measure whole-model latency, process footprint or CUDA capture baseline | Model measurement | `gtcrn_benchmark.py`, `gtcrn_torch_worker.py` | Declared model protocol, frozen executable and full numerical references |
| Characterize general pointwise layouts and compiled/captured baselines | Tensor measurement | `pointwise_benchmark.py`, `pointwise_protocol.py`, `pointwise_torch_worker.py` | `../src/bin/fa-pointwise.rs`, frozen pointwise protocol, `assess_pointwise.py`, full-output/resource boundaries |
| Check prepared matrices, both VJPs and repeated primitive SGD | Dense algebra correctness | `check_matmul.py` | Native test FP32 exports, complete geometry/binding/profile/backend scope, pinned Torch and `../docs/matmul-v1.md` |
| Check prepared 1D/2D convolutions, both VJPs and Tape SGD | Spatial gradient correctness | `check_conv.py` | Native test FP32 exports (`FA_CONV_EXPORT_DIR`), complete 1D/2D geometry/binding/backend scope, pinned PyTorch strict-FP32 and `../docs/conv-backward-v1.md` |
| Check differentiable STFT/CenteredSTFT/iSTFT autograd | Audio gradient correctness | `check_stft_backward.py` | Strict-FP32 PyTorch `torch.stft`/`torch.istft` autograd and adjoint identity controls on CPU/CUDA |
| Check and benchmark foundation operators, Flash SDPA, Audio Transformer, Vocoder and AdamW against PyTorch | Foundation correctness / Measurement | `check_new_ops.py`, `bench_foundation_vs_pytorch.py` | `../examples/audio-foundation-bench.rs`, 35-output strict-FP32 PyTorch 2.14.1 cross-check and 11-workload CPU/CUDA paired benchmark suite |
| Check native tapes and repeated procedural training | Training correctness | `check_autograd.py` | Full composition/branch/dense/audio FP32 exports, fixture/schema/profile/backend scope, independent evolving Torch trajectories and `../docs/autodiff-v1.md` |
| Characterize native complete training against eager/compiled/captured Torch | Training measurement | `training_protocol.py`, `training_benchmark.py`, `training_torch_worker.py`, `build_training.py`, `assess_training.py` | Fixed training protocol, `../src/bin/fa-training.rs`, independent evolving controls, reset/repeat/resource/transport gates and exact artifact recovery |

## Files

`check_release.py` checks candidate version metadata and selects a merged stable tag for the portable API compatibility CI job.

| File | Domain | Deps | Function |
| --- | --- | --- | --- |
| `check_phase_ab.py` | Prototype correctness | ← `../examples/audio-bf16-bench.rs` and PyTorch | Independent CPU FP32 and CUDA BF16 operator and short training trajectory checks |
| `bench_bf16_vs_pytorch.py` | Prototype measurement | ← `../examples/audio-bf16-bench.rs` and CUDA PyTorch | BF16 comparisons with shared Transformer weights and tokens, synchronized wall-time ratios and raw latency samples |
| `compare_torch.py` | Measurement | ← PyTorch, workload JSON, benchmark binary | Matched timing/fixtures, numerical gates, statistics and source archive including runnable examples and protocol fixtures |
| `verify_archive.py` | Reproduction | Python standard library | Validate archive/member hashes, restore regular files and recover hashed companion protocols at canonical fixture paths |
| `assess_candidate.py` | Measurement | Python standard library | Recompute paired ratios and assess three frozen seven-case confirmation executions with primary/regression intervals |
| `benchmark_protocol.py` | Measurement | Python standard library | Validate declared timing profiles, artifact identity, actual counts and execution order |
| `check_audio.py` | Correctness | ← PyTorch, `compare_torch.py`, audio example | Full spectra, fixed-length reconstruction and modified-spectrum checks with archived source; no latency measurement |
| `gtcrn_reference.py` | Model | ← pinned upstream module/checkpoint, PyTorch | Hash-checked pipeline, named FP32 parameter export and optional spatial/complete model captures, separate timing/profiling runs |
| `check_gru.py` | Sequence | ← captured tensors, native replay example, PyTorch | Verify every input/parameter hash and all 56 output/state cases; retain exploratory native/Torch samples and source archive |
| `check_gtcrn_stages.py` | Model | ← captured tensors, native stage example, shared comparison/blob helpers | Hash-check inputs/parameters, validate four spectra/inverse waveforms and all 88 convolutions, archive source and retain native-only diagnostics |
| `check_gtcrn.py` | Model fixture / Runtime | ← complete reference/parameters, native inference example, shared helpers | Hash-verify/freeze artifacts, validate all outputs/traces and actual selected convolution-policy metadata, retain recoverable evidence |
| `gtcrn_benchmark.py` | Model measurement | ← fixed protocols, full reference, frozen native artifacts and Torch worker | Counterbalanced native/Torch characterization or declared native-control screens, exact-output/storage gates, artifact seals/source recovery, RSS and separate process-group NVML |
| `gtcrn_torch_worker.py` | Model measurement | ← pinned reference model and input blobs, PyTorch | Eager and explicitly named complete CUDA graph workers, prechecked real-FFT/fold inverse variant, preparation outside timing and canonical full outputs |
| `pointwise_protocol.py` | Tensor measurement | ← fixed v2 protocol and fixture generator, optional PyTorch | Immutable case/window/writeback validation, retired-v1 rejection, balanced mode order, independent FP64 equations, complete backing/input/hole gates and descriptive statistics |
| `pointwise_torch_worker.py` | Tensor measurement | ← protocol/helpers, PyTorch/Inductor | Eager out-parameters and full-graph functional math/writeback, persistent captured modes, explicit preparation, enforced/reported compiler precision, writeback/codegen metadata, timed commands and full backing dumps |
| `pointwise_benchmark.py` | Tensor measurement | ← native/Torch workers, protocol/gates and existing environment helpers | Frozen build/source receipts, timed JSON transport, counterbalanced windows, exact repeats, terminal worker cleanup, separate profiler commands and sealed logs |
| `assess_pointwise.py` | Tensor evidence | ← protocol/summary helpers, archive verifier | Reject partial/validation results, verify complete declared scope/counts/order, recorded numerical/repeat gates, exact executable/source/log/profile recovery and recomputed summaries without promotion |
| `build_pointwise.py` | Tensor measurement build | ← protocol, source inventory and Cargo | Locked/offline hardware-specific worker build with exact flags/full producer ID, atomic process/status receipt and sealed source/log/executable identity |
| `check_matmul.py` | Dense algebra correctness | ← native raw-bit exports and pinned Torch/autograd | Strict complete 389-case/two-binding debug/release checks, both matrix gradients and seeded accumulation, independent evolving eight-step SGD control, fixed precision/tolerances and raw export identity reports |
| `check_conv.py` | Spatial gradient correctness | ← native raw-bit exports and pinned PyTorch/autograd | Strict complete 64 1D + 56 2D / two-binding checks for forward, `dX`, `dW` and seeded accumulation, plus independent evolving 8-step Conv1d+Conv2d Tape SGD control and SHA-256 export reports |
| `check_stft_backward.py` | Audio gradient correctness | ← PyTorch strict-FP32 `torch.stft`/`torch.istft` autograd | Independent CPU/CUDA strict-FP32 spectral forward/backward and inner-product adjoint verification across even/odd FFT sizes and custom windows |
| `check_new_ops.py` | Foundation operator correctness | ← `../examples/audio-foundation-bench.rs` and PyTorch strict-FP32 | Independent CPU/CUDA strict-FP32 PyTorch 2.14.1 cross-check of 35 forward/backward/trajectory outputs across `LayerNorm`, `RmsNorm`, `Softmax`, `LogSoftmax`, `Max`/`Min`/`LogSumExp`, `Bmm`, `Sdpa` (GQA causal), `ConvTranspose1d`/`2d`, `Pad`, `Cat` and 5-step `AdamW` + grad clip |
| `bench_foundation_vs_pytorch.py` | Foundation performance comparison | ← `../examples/audio-foundation-bench.rs`, `training_protocol.py`, `training_torch_worker.py` and PyTorch | Paired strict-FP32 CPU and CUDA benchmark harness comparing `fast-audio` (Fused / MKL / CUDA Graph) against PyTorch 2.14.1 Eager, `torch.compile(inductor)` and CUDA Graph across `training-v1` spectrum/waveform, Norm, Softmax, BMM, Flash SDPA, Audio Transformer + AdamW and Spectral-Vocoder U-Net + AdamW |
| `check_autograd.py` | Training correctness | ← native raw-bit exports and pinned Torch/autograd | Strict complete sixteen-operation/two-binding compositions, tied/detached/overlapping roots, dense and multichannel waveform/spectral-shaped eight-step SGD, full predictions/gradients/parameters, independent trajectories and fixed precision/tolerances |
| `training_protocol.py` | Training measurement controls | ← fixed training declaration and Torch | Dyadic twelve-case fixtures, physical-coordinate analytic FP64 and independent eager trajectories, layout transforms, full snapshot/source gates and fixed counterbalanced order |
| `training_torch_worker.py` | Training measurement execution | ← native-matched fixtures, Torch/autograd/func/Inductor, shared compiler policy | Resident eager/full-graph and complete CUDA-graph training, in-place SGD, independent state resets, strict request budgets and post-update diagnostic payloads |
| `training_benchmark.py` | Training measurement collection | ← fixed protocol, shared timed transport/logging and artifact helpers | Exact committed builds, full independent controls, raw transport, complete-step/reset/repeat windows, sealed payloads and separate profiles; matching Nsight importer discovery preserves captures by importing copies |
| `build_training.py` | Training measurement build | ← source manifest, Git archive, Cargo and fixed build policy | Reject uncommitted producer drift, build locked/offline hardware features and freeze exact source/log/ELF/compressed receipts |
| `assess_training.py` | Training evidence | ← fixed protocol, sealed artifacts and SQLite | Reject partial/wrong-kind contexts; verify sources/executable/transport/payloads/windows; explicitly bind recovered diagnostics to an immutable export failure and audit stream/update/launch events without promotion |
