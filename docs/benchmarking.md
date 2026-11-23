# Benchmark contract

## Current external baseline

PyTorch is the sole external framework baseline for current training and inference
performance work. Pair each performance change with the preceding fast-audio revision
and the strongest compatible PyTorch mode in the same hardware session. Defer external
framework and standalone C++ inference-engine comparisons as described in the
[execution plan](plan-phases-a-d.md#8-deferred-post-pytorch-scope).

- **Compatible PyTorch modes.** Record the build, precision settings, and execution mode.
  Evaluate eager and `torch.compile` on CPU with matched thread counts. On GPU, include
  CUDA Graph replay where capture works. Check numerical quality before selecting the
  fastest valid mode for the declared primary metric. Record results per mode and note
  unsupported cases or capture failures.
- **Training.** Use identical inputs, initialization, precision policy, and equivalent
  optimizer state. Time the full forward, backward, and update step. Report throughput,
  peak memory, and wall-clock time to matched validation quality when measuring convergence.
- **Inference.** Use `eval()` and `torch.inference_mode()`. Match weights, precision,
  batch and input lengths, chunking or KV-cache state, and the measured boundary. For
  waveform-to-waveform results, include the same DSP stages and transfers on both sides.
  Report startup and first-call costs separately from warmed median, `p95`, `p99`, domain
  throughput, and memory.

Use the [Step Measurement Receipt](plan-phases-a-d.md#7-mandatory-measure-every-step-protocol)
to record correctness, before/after results, and the hardware and resource measurements.

## Existing benchmark protocols

The [complete training-step protocol](training-characterization-v1.md) separately
measures prepared native forward/backward/SGD against eager, full-graph compiled and
complete CUDA-graph Torch paths. Its twelve procedural workloads, stricter numerical
gates, state resets, output epochs and evidence scopes form a separate characterization
context. Operator or inference results do not imply training-step performance.

## Initial workload and objective

The authoritative operator cases are in `benchmarks/workloads.json`. They cover a small
streaming GEMM, a larger projection, temporal convolution, pointwise convolution,
grouped/strided/dilated convolution, stereo STFT, and short streaming STFT. Shapes are
representative probes, not a distribution measured from a production model. The first
hardware strata are an Intel Core i5-13420H and an NVIDIA RTX 3060 12 GB.

The primary metric is warmed wall latency per operation in microseconds. CPU timing
is synchronous. GPU wall time includes enqueueing and a final completion wait; CUDA
events are recorded as a separate device-stream metric. Event intervals can include
GPU idle gaps while the host dispatches small operations; they are not isolated kernel
durations. Across cases, use the geometric
mean of paired latency ratios, with `torch / fast-audio > 1` favoring fast-audio.
Keep each case visible: one aggregate must not conceal a convolution regression.

Correctness is a hard constraint. Every output element must be finite and satisfy
`abs(actual-expected) <= 2e-4 + 2e-4*abs(expected)` against strict-FP32 PyTorch. Small Rust
tests independently use FP64 convolution/GEMM and direct DFT oracles with tighter bounds.
Changing precision or tolerances creates a new contract and cannot count as a speedup
under this one. Shape errors and memory ownership are correctness gates too.

## Reproduction

Build release artifacts using `--locked --features mkl` or `--locked --features cuda`.
Keep CPU threads equal, and set `CUDA_ARCH` for the actual GPU. Start from an idle machine
when possible; record contention, power/clock policies, thermal state, and affinity when
controlling them. The harness captures tool/library versions, CPU model, GPU driver and
snapshot clocks/temperature, relevant environment variables, exact inputs/parameters,
binary SHA-256, per-file SHA-256, and a source archive. It does not lock clocks or reserve
the machine. Do not run competing benchmark jobs on the same hardware during confirmation.

`fa-bench --list` lists cases. The standalone binary emits JSON and can write its entire
FP32 result as little-endian bytes using `--output`. The comparison harness regenerates
identical hashed inputs, runs both implementations sequentially, checks all output values,
and saves raw block results in `comparison.json`. A new output directory is mandatory.
Source archives contain only explicit source/build/test/harness paths, excluding access
credentials and virtual environments. Preserve their checksums with the report. Restore
the source archive to an empty directory, check its file hashes against the manifest,
and rebuild using the recorded toolchain/features before interpreting historical results.

The initial exploration uses 7 alternating-order blocks, 20 warm-up iterations per
implementation per block, and 50 timed iterations. Rust executes in a fresh child process
for each block; PyTorch remains in the harness process. Both have a warm-up phase. This
asymmetry, shared hardware, and uncontrolled clock drift limit statistical independence.
The seeded bootstrap resamples block log-ratios; its interval is exploratory, not a
deployment guarantee or a formal release test. An inner iteration is not an independent
sample. Initialization is recorded separately; startup and transfer latency need separate
end-to-end experiments.

## Fairness boundaries

- Inputs and weights remain resident. Reusable Rust outputs and scratch are allocated
  before measurement. PyTorch GEMM also uses `out=`; eager Conv1d/STFT allocate outputs.
  PyTorch's ReLU is in place. This initial comparison covers prepared execution against
  PyTorch eager. Use matched-buffer PyTorch cases, `torch.compile`, and graph capture where
  supported to investigate dispatch overhead or compare broader execution modes.
- TF32 is disabled on both sides. Autotuning is allowed during PyTorch warm-up. Rust's
  present GEMM uses vendor SGEMM without a separate algorithm search. Reduced-precision
  tensor-core experiments must also match error and model-quality constraints.
- STFT layout canonicalization and downloads are outside timing. PyTorch exposes a
  frequency-major view; fast-audio writes frame-major real/imaginary storage. Both perform
  the same periodic-Hann, non-centered, unnormalized transform.
- Installed system libraries and PyTorch-bundled libraries can have different versions.
  The initial CUDA build uses Toolkit 12.4 while the available PyTorch uses CUDA 13.0.
  Record this as a stack comparison, not an isolated language or wrapper comparison.
- These are operator results. They do not establish real-time deadline guarantees,
  model throughput, training performance, audio quality, memory-pool behavior, or performance
  on unmeasured machines and shapes.

## Promotion of later optimizations

The first verified implementation is a reproducible anchor, not a claim of success over
PyTorch. For a subsequent same-context candidate, a useful primary gain is at least 5%
against the retained fast-audio anchor, with no case regressing more than 5%. Validate
all cases first. Freeze the chosen implementation and configuration, then use fresh,
counterbalanced confirmation runs: 3 fresh harness executions, each with 7 blocks,
20 warm-ups, and 50 iterations. Treat executions as the uncertainty unit; if a conservative
95% interval cannot resolve the primary gain or any regression guard, the outcome is
inconclusive and the anchor stays. Reserve confirmation for the chosen candidate rather
than repeatedly testing many candidates until one passes. This initial campaign allows
at most three screened candidates and one confirmation campaign before reviewing evidence.

A change to hardware, major vendor libraries, precision, shapes, or the timing protocol
requires remeasuring both anchor and candidate in the new context. Preserve old reports.
For CPU changes, include the original Intel stratum; for CUDA changes, include the original
NVIDIA stratum. Confirm model-level latency and output quality before promoting a framework
release. One microbenchmark win does not meet the project objective.

## Paired vendor-plan campaign

`compare_torch.py` accepts `--anchor-binary`, `--anchor-backend` and `--anchor-report`
together. It verifies the retained binary/workload hashes, alternates anchor/candidate
order, and runs an independent PyTorch output check after each pair. Both Rust outputs
must pass the original numerical gate. Reports preserve the original anchor report,
anchor source archive and candidate source archive. `verify_archive.py --anchor` verifies
the retained anchor source; the default verifies the candidate source.

The first vendor-plan campaign uses the original `201026f` binaries, 1 CPU thread and
CPU affinity `{0}` on each host. Intel CPU 0 is a performance core. Both implementations
are remeasured under that affinity; historical absolute latencies are not used to claim
gains. GPU clocks remain driver-managed and are recorded, not locked. OneDNN packing is
done during preparation. The cuDNN pointer pack is created on first use and reused in
timed execution. `setup_us` measures construction, not all vendor first-use initialization.

Three screened choices exhaust this campaign's screening budget: all-oneDNN convolution
(rejected for pointwise regression), Intel dispatch preserving pointwise MKL, and NVIDIA
dispatch preserving pointwise cuBLAS. The final two implementations are frozen before
three fresh confirmation executions on each host, each covering 7 cases, 7 blocks,
20 warm-ups and 50 timed iterations. No confirmation run is discarded or extended.

`assess_candidate.py` recomputes ratios from raw paired samples and treats each fresh
execution as one uncertainty unit. It uses a two-sided Student t 95% interval on three
execution log-ratios (df=2), assuming approximately normal execution log-ratios. A lower
primary ratio bound of at least `1/0.95` establishes a 5% latency reduction; every per-case
lower bound must be at least `1/1.05` to establish no more than 5% latency regression.
All gates must pass for one preselected candidate in one campaign. This intersection
of gates avoids promoting a candidate with an unresolved guard; it is not a guarantee
over arbitrary workloads or repeated future selection campaigns.

```sh
python3 tools/assess_candidate.py \
  benchmarks/results/2026-10-16-intel-confirm-1/comparison.json \
  benchmarks/results/2026-10-16-intel-confirm-2/comparison.json \
  benchmarks/results/2026-10-16-intel-confirm-3/comparison.json \
  --output /tmp/intel-assessment.json
```

The NVIDIA candidate passed; the Intel result is inconclusive because the short-STFT
interval narrowly misses its guard. Keep the Intel anchor and review measurement
precision before any new campaign. A longer per-case timing window must be specified
in advance and applied to both implementations. Do not simply add runs until the gate
passes. Rebuilt or relocated artifacts may have new binary hashes: collect a fresh
baseline report for such an artifact and preserve the old record rather than editing
historical hashes. Recovery checks establish rebuildable source, not bit-identical binaries.

The predeclared [Intel long-window protocol](intel-long-window-v1.md) supplies fixed
per-case counts through `--protocol benchmarks/protocols/intel-long-window-v1.json`
and execution IDs 1–3. It retains the same binaries, seven cases and acceptance gates.
The collector checks artifact identity and applies the counts to both Rust binaries
and PyTorch; the assessor checks the saved profile hash, actual counts and paired order.
This is a new measurement context, with its own three executions and preserved evidence.
Collection is complete: short STFT passes, but projection GEMM remains unresolved.
The [assessment](../benchmarks/results/2026-10-16-intel-long-v1-assessment.json) retains
the original MKL anchor; see progress for the intervals and remaining control gap.

## Complete-model characterization

The [GTCRN v1 context](gtcrn-characterization-v1.md) adds four complete waveform cases,
fresh processes for both frameworks, frozen executable copies and strict output gates.
It compares eager inference and, on NVIDIA where supported, complete CUDA graph replay.
GNU time command RSS and separate process-group NVML observations describe different
resource scopes; planned graph storage and Torch allocator peaks remain separate fields.
This is initial characterization with a fixed budget and no promotion. Its observations
guide the next declared optimization; the operator champion records remain unchanged.
The [completed v1 characterization](gtcrn-characterization-results-v1.md) retains
all three supported contexts, separate profiles and observed memory lower bounds.

## Explicit convolution repeatability policy

The [correctness-only policy suite](cudnn-repeatability-v1.md) validates the opt-in
deterministic convolution selection on the current NVIDIA stack. Its native CLI
timer fields are retained only as raw emissions and provide no performance result.
Future campaigns must declare and record the policy for every compared artifact,
alongside precision, vendor versions and the usual execution boundary. A frozen
artifact without that policy cannot stand in for a matched deterministic control.
Changing the requested algorithm policy requires a new declared context with both
alternatives measured there; it does not establish a speedup or erase older failures.

## General pointwise characterization

The [pointwise v2 context](pointwise-characterization-v2.md) declares a separate 56-case
suite with tighter preexisting pointwise error gates, resident reusable strided outputs,
fresh persistent worker pairs and balanced eager/compiled/captured timing windows.
It records all modes and full backing checks rather than applying the older seven-case
eager-only assumptions. Raw arrays remain on collecting hosts under the existing ignored
artifact policy; source/exact binary/log receipts are retained. Profiling is separate,
there are no implementation candidates or promotion, and uncaptured native execution
does not imply native replay support. Existing workload gates and champions stay intact.
The original [v1 readiness failures](pointwise-characterization-v1.md#validation-outcome-direct-compiled-out-parameter-path-unavailable)
remain preserved. V2 compiles functional math and caller-view writeback together;
it retains every case/layout/window and the original numerical gates.
The [completed v2 results](pointwise-characterization-results-v2.md) verify both full
contexts and worker transcripts, preserve all per-case losses and recover the five
original GPU streams after an importer-path failure. Profiling diagnostics remain
separate from warmed timing; no optimization is accepted by this characterization.

## Complete-training characterization

The [training protocol](training-characterization-v1.md) measures twelve procedural
dense/waveform/spectral objectives through complete native forward/backward/SGD steps.
Independent FP64/eager trajectories, actual-update ULP checks, exact resets/repeats,
sealed payloads and raw transport gates apply to every declared mode and window.
The [completed baseline](training-characterization-results-v1.md) retains all Intel
and NVIDIA results, including losing cases and the original GPU profiler-export
failure. Explicit copied-input recovery adds only missing diagnostics; assessment
keeps the failed report immutable and rechecks the full timing scope. Profile streams
are losslessly recoverable from compressed/chunked archives. No optimization is
promoted, and sampled/resource-plan observations do not establish memory peaks.
