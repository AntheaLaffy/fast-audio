# GTCRN reference and native execution work

The first real-model reference is the official GTCRN speech enhancer, pinned to
[`502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73).
Its MIT notice and byte-identical source snapshot are under `models/gtcrn`.
`upstream.json` locks the source and DNS3 checkpoint hashes. The checkpoint stays
in ignored `target/models/gtcrn`; download verification precedes `weights_only=True`
loading. No upstream test WAV is redistributed or used as a licensed quality corpus.

The reference runs a complete waveform/STFT/model/inverse-STFT pipeline at 16 kHz,
FFT size 512, hop 256 and the upstream square-root periodic Hann window. The wrapper
uses the modern complex PyTorch API and supplies the original inverse waveform length.
Inputs are project-generated deterministic tones/noise, with four strata: one-channel
batches of one with 4,096, 16,000 and 64,000 samples, plus batch two at 16,000 samples.
These fixtures establish numerical and execution contracts, not speech quality.

`tools/gtcrn_reference.py` records full spectra/waveforms, 104 leaf-module shapes and
14 actual GRU inputs/parameters/outputs/final states for each stratum. Uninstrumented
warm model/pipeline timing is separate from the shape-enabled profiler invocation.
CPU profiling attributes roughly 73–81% of its recorded CPU time to GRU calls.
Do not sum nested GPU attribution and kernel event times as independent costs.

## GRU implementation and initial diagnostic

The public GRU contract is one layer, batch-first FP32, reset-after candidate gating,
explicit initial/final hidden state, one or two directions, and immutable prepared
parameters. CPU input projections are batched GEMMs; recurrent projections and gate
updates reuse scratch. CUDA uses a prepared cuDNN inference descriptor with strict FMA
math, owned packed weights and workspace. Both preserve caller inputs and perform no
Rust allocation in warmed execution. Backward/training, variable-length packed batches
and stacked layers are not implemented by this primitive.

All 56 captured layer/shape cases pass the original full-output numerical gate on
MKL and the initial standard cuDNN route. The standard GPU route is nevertheless slow:
for the 64,000-sample temporal attention GRU it takes about 1.7 ms against roughly
0.10 ms for PyTorch. This is a diagnostic observation, not an accepted implementation
or an end-to-end performance result.

The first controlled follow-up changes only the cuDNN algorithm: persistent small-hidden
inference for FP32 GRUs with hidden size at most 192 on SM 6+, retaining standard mode
outside that domain and as an explicit control. Batch-major layout, precision, frozen
reference tensors, all 56 cases, and 20 warm-ups / 7 samples / 50 timed iterations stay
fixed. The mechanism is fewer recurrent launches and less intermediate memory traffic;
failure would be unsupported shapes, numerical failure, or no reduction in long-sequence
latency. At most one additional sequence-major layout screen is reserved if this change
is insufficient. This is initial operator exploration, not a framework promotion campaign.
Any accepted performance claim still needs paired fresh confirmation and full-model tests.

The choice follows the vendor's [small-hidden algorithm contract](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/developer/legacy-api.html#rnn-functions)
and the [PyTorch 2.14 selection logic](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/cudnn/RNN.cpp).
Both native algorithms retain `CUDNN_FMA_MATH`; tensor-core conversion is not enabled.

Both reserved screens are complete. All 56 output/final-state cases and fresh Torch
replays pass on standard, persistent batch-major and persistent sequence-major paths.
For the 64,000-sample temporal attention GRU, median native wall time changes from
1,702 us (standard batch-major) to 106 us (persistent batch-major) and 98 us (persistent
sequence-major); the corresponding Torch probes are roughly 95–98 us. For that stratum's
first intra-GRU, the sequence-major path is about 51 us against Torch's 126 us, including
native transpose kernels. These are separate, sequential diagnostic executions with
uncontrolled drift and different installed vendor stacks; they do not establish confidence
bounds, a complete-model gain or a general advantage over PyTorch. No further screen was
added after observing these results.

The initial default uses automatic small-hidden selection and internal sequence-major
layout. For batch one, public batch-major storage already has the same memory order.
Larger batches use owned transpose scratch inside the measured operation. `GruOptions`
retains standard mode and batch-major layout as explicit controls. The current whole-suite
champions remain unchanged; fresh paired confirmation and full-model execution are still
required before performance promotion.

Evidence: [MKL layers](../benchmarks/results/2026-10-17-gtcrn-gru-mkl/validation.json),
[standard GPU](../benchmarks/results/2026-10-17-gtcrn-gru-cudnn/validation.json),
[persistent batch-major](../benchmarks/results/2026-10-17-gtcrn-gru-cudnn-small-h/validation.json),
and [persistent sequence-major](../benchmarks/results/2026-10-17-gtcrn-gru-cudnn-seq-major/validation.json).
The original full reference is preserved on [CPU](../benchmarks/results/2026-10-17-gtcrn-cpu-reference/reference.json)
and [CUDA](../benchmarks/results/2026-10-17-gtcrn-cuda-reference/reference.json).

The CPU reference uses Torch 2.14.1+cpu, NumPy 2.5.4 and einops 0.7.0; the GPU reference
uses Torch 2.14.0+cu130, NumPy 2.2.6 and einops 0.8.2. Environment companions record these
observations without rewriting the original reference reports/hashes. CPU source was
archived immediately after collection with every member checked against its recorded
manifest; subsequent runs archive before measuring. All archives include the pinned model
source and license. Checkpoints and generated FP32 tensors are regenerated from hashes
and kept out of Git.

## Exact window and convolution stages

The spectral plans now snapshot explicit coefficients. GTCRN's exact captured FP32
square-root periodic Hann window drives native centered STFT and inverse synthesis;
the inverse NOLA envelope is prepared from those coefficients. Default constructors
continue to use periodic Hann.

Prepared NCHW 2D convolution covers all 11 forward and 11 transposed modules, including
groups, dilation and output padding. Constant weights/bias are snapshotted once. Intel
uses oneDNN forward/deconvolution and direct MKL pointwise GEMM; CUDA uses strict-FP32
cuDNN forward/backward-data graphs and direct cuBLAS pointwise GEMM. Vendor layout
reorders, transpose output-padding extensions and fallback epilogues are execution costs.

`--capture-conv` saves actual convolution inputs, parameters and outputs in fresh
reference directories. `examples/gtcrn-stages.rs` replays all 88 layer/shape instances
and four audio cases. `tools/check_gtcrn_stages.py` verifies reference blob hashes and
every output element using the original `2e-4 + 2e-4*abs(expected)` gate. Both hosts pass.
Maximum convolution absolute error is `3.82e-5` on Intel and `2.39e-6` on RTX 3060.
Native spectra match the captured spectra exactly; enhanced inverse waveforms differ
by at most `1.50e-8` on Intel and `2.99e-8` on GPU. Each inverse consumes the captured
PyTorch enhanced spectrum, so this establishes stage correctness rather than a native
end-to-end model result.

Evidence: [Intel stages](../benchmarks/results/2026-10-17-gtcrn-stages-mkl/validation.json),
[NVIDIA stages](../benchmarks/results/2026-10-17-gtcrn-stages-cudnn/validation.json),
[CPU convolution captures](../benchmarks/results/2026-10-17-gtcrn-conv-cpu-reference/reference.json)
and [CUDA convolution captures](../benchmarks/results/2026-10-17-gtcrn-conv-cuda-reference/reference.json).
Earlier reference and GRU artifacts remain unchanged. The new captures use one warm-up
and one timing sample solely to collect additional fixtures. Native layer samples use
20 warm-ups and 7 by 50 iterations, without matched Torch timing. These diagnostics
support no new speedup or promotion claim; existing champion identities remain active.
Stage directories also retain native environment observations taken after collection,
linked to the validation report digest; these do not rewrite the measured manifests.

## Complete native offline model

The complete spectrum model is lowered by `models::gtcrn::build` to the shared prepared
graph. It includes ERB analysis/synthesis, frequency neighborhoods, grouped temporal
attention, two dual-path GRNNs, layer normalization, activations, channel shuffles,
decoder skips and complex masking. Inference BatchNorm is folded into the prepared
forward/grouped-transpose weights and biases. The Rust adapter follows the pinned MIT
architecture and retains the original license/attribution under `models/gtcrn`.

`examples/gtcrn-infer.rs` runs original waveform -> exact-window centered STFT -> native
model -> native inverse STFT on both targets. Input/output spectra use `[B,T,257,2]`;
the checker explicitly transposes the captured PyTorch `[B,257,T,2]` layout. Every
recurrent invocation starts at zero, preserving the upstream offline contract.
All intermediate activations remain on the selected backend. The Rust runtime uses
named exported FP32 tensors and native libraries, with no PyTorch execution dependency.

`--capture-model` adds a named parameter bundle and 15 complete block outputs per input
to a fresh reference run. The full checker validates the pinned identity and every
parameter/activation hash, freezes source/binary identities, and compares all enhanced
spectra, final waveforms and 60 block outputs under the unchanged numerical gate.
These checks connect native stages rather than replaying captured activations as inputs.

| Maximum discrepancy across all four inputs | Intel | RTX 3060 |
| --- | ---: | ---: |
| Enhanced spectrum absolute error | `2.01e-5` | `4.47e-4` |
| Final waveform absolute error | `1.57e-7` | `2.38e-6` |
| Largest error / original elementwise tolerance | `0.056` | `0.753` |

Every output passes `2e-4 + 2e-4*abs(expected)`, including values whose absolute error
exceeds the constant term. No tolerance was relaxed for integration or BatchNorm folding.

Evidence: [complete Intel validation](../benchmarks/results/2026-10-17-gtcrn-native-mkl/validation.json),
[complete NVIDIA validation](../benchmarks/results/2026-10-17-gtcrn-native-cudnn/validation.json),
[CPU weights/traces](../benchmarks/results/2026-10-17-gtcrn-model-cpu-reference/reference.json)
and [CUDA weights/traces](../benchmarks/results/2026-10-17-gtcrn-model-cuda-reference/reference.json).
Earlier isolated-stage diagnostics remain available as controls and numerical evidence.
Native timing uses two warm-ups and three samples of five calls, without matched
PyTorch timing; reference capture uses one sample for additional fixtures. This establishes
no accepted speedup. A fresh paired model comparison, appropriate baselines and defined
quality/memory gates are the next performance work.

The executor currently retains all intermediates for tracing. Reported graph workspace
includes those tensors and prepared operator scratch; inverse spectral storage is a
separate partial count. Caller tensors, forward FFT internals and complete peak memory
are outside these fields. Buffer liveness, fusion and launch overhead should be measured
before selecting the next optimization.

After subsequent regression builds, the NVIDIA executable digest differs from the
collection digest while all 73 archived source hashes still match. The native environment
companion records that observation. Byte-identical CUDA rebuilds remain unverified;
future performance campaigns must retain immutable copies of both measured binaries.

## Remaining model acceptance

Licensed speech/noise evaluation, held-out quality checks, memory/transfer
accounting, compiled/vendor baselines, streaming state and full CPU/GPU model latency
remain required before model or framework acceptance.
