# GTCRN complete-model characterization v1

This context characterizes the first complete native model before selecting another
optimization. It preserves the operator champions and the complete-model numerical
references. It does not promote a framework implementation or establish speech quality.
The machine-readable policy is
[`gtcrn-characterization-v1.json`](../benchmarks/protocols/gtcrn-characterization-v1.json).
Collection is complete; see the [results and profile analysis](gtcrn-characterization-results-v1.md).

## Workload and comparisons

Use the pinned GTCRN revision `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`, its verified
DNS3 parameters, exact FP32 square-root Hann and existing procedural inputs. The four
equally weighted cases are batch one at 4,096, 16,000 and 64,000 samples and batch two
at 16,000 samples, all at 16 kHz. The primary measurement is completed warmed waveform
pipeline wall latency: STFT, network and inverse STFT. Input/weights stay resident;
downloads, output serialization and setup are outside that timer.

Intel compares `mkl,onednn` against eager PyTorch. NVIDIA compares `cudnn` against
eager PyTorch and explicitly named complete-pipeline CUDA graph variants when their
single compatibility probes succeed. Capture includes the original model's
BatchNorm and tensor operations; the native graph folds inference BatchNorm. Record
each PyTorch mode as a separate context and remeasure native execution in each one.
General optimizing-compiler and matched vendor/C++ baselines remain acceptance work.

The unmodified `cuda-graph` probe failed inside `torch.istft` with a forbidden stream
synchronization; preserve its failure and original protocol companion. The pinned
[PyTorch inverse implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/SpectralOps.cpp#L1074-L1113)
computes an overlap envelope and reads a scalar NOLA check. A separately declared
`cuda-graph-prepared-inverse` variant checks and caches that same fixed envelope during
preparation, then uses PyTorch real inverse FFT, window multiplication, overlap fold and
normalization inside the captured pipeline. It retains FP32 and is checked against the
original inverse, including modified spectra, all four lengths/batches and NOLA failure.
This is a prepared Torch inverse baseline; its name and interpretation stay distinct
from capture of the unmodified API. Reserve exactly one compatibility probe for this
additional named variant before collecting any of its characterization data.

Each pair uses fresh native and Python worker processes, one CPU thread and affinity
`{0}`. Both prepare and warm up independently. Record CPU governor/frequency, load and
GPU clock/temperature/power snapshots before and after each pair, outside timing;
these snapshots do not establish continuously locked conditions. CUDA event timing is secondary and can
include idle gaps while the host dispatches work. The system CUDA/cuDNN and PyTorch's
bundled stack differ, so results compare the installed stacks and execution APIs.

## Fixed sampling and gates

Collect exactly seven blocks. In even blocks native runs before Torch; reverse that
order in odd blocks. Rotate the listed case order by block index. Every worker performs
20 warm-ups and one timed window. Intel uses 50/20/5/10 iterations for the four cases;
NVIDIA uses 100 for each. These counts apply to both sides of a pair. Initialization,
warm-up, and process startup remain outside the primary timer.

Every spectrum, enhanced spectrum and final waveform must be finite and pass the
original `2e-4 + 2e-4*abs(expected)` gate against the reference and the paired output.
Verify reference/parameter hashes, compiled features, exact counts and executable
identity. A failure invalidates that context; preserve partial output and the reason.
Do not discard or replace slow or failing blocks.

Report each case and the equally weighted geometric mean of paired Torch/native
ratios. Resample complete block log-ratios with the fixed seeded bootstrap for an
exploratory interval. Seven blocks in one execution do not establish independent
confirmation. The budget allows one characterization execution per host/mode, one
single-case compatibility probe per mode and no optimization candidates. A future
candidate campaign must be declared separately, retain the current native control,
and satisfy useful-gain, regression, quality and memory requirements.

## Process memory and profiling

GNU time records command maximum RSS, CPU seconds and whole-command wall duration.
RSS is the fresh command's process footprint, including framework import, preparation
and output I/O. It is separate from planned tensor/scratch storage and inference-only
allocator measurements. Native `inverse_workspace_bytes` counts inverse storage only;
historical reports used `spectral_workspace_bytes` for that same partial count.

For NVIDIA, replay a separate fresh worker with 20 warm-ups and 200 iterations. Sample
NVML compute-process memory for that worker's process group, requesting 20-ms polls and
recording the actual timestamps/intervals. Preserve each observation and sampler errors.
Its high-water value is a sampled lower bound, includes context/vendor overhead and can
miss short allocation peaks. Torch's own allocator peaks are additional implementation-
specific data. Memory replay timings are excluded from latency summaries.

Profile only the four-second native case, separately from characterization. Per host,
reserve one ordinary 20-warm-up/100-iteration control and one instrumented run with the
same counts. Intel uses user-space cycle sampling at 199 Hz and DWARF call stacks;
NVIDIA uses a CUDA-only Nsight Systems trace. Save exact commands, versions and raw
artifacts. Profiled timing diagnoses overhead and attribution, and never replaces the
uninstrumented primary samples. Do not sum overlapping or nested timers.

## Artifact identity

Before collection, copy the executable to a SHA-256-named directory under ignored
`target/benchmark-artifacts/gtcrn`, mark it read-only and verify it before/after use.
Cargo rebuilds operate on their normal output and cannot replace this copy. Record
the original and frozen identity, source manifest/archive, protocol hash, model and
fixture identities, run order, resource observations, library linkage and raw outputs.
Source/executable changes during collection fail the run. Recovered source establishes
functional reproduction; byte-identical CUDA rebuilds remain unverified.
