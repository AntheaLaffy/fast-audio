# Pointwise characterization v1

## Purpose and fixed workload

Establish a recoverable performance baseline for the prepared FP32 pointwise surface,
not an optimization promotion. The [protocol](../benchmarks/protocols/pointwise-characterization-v1.json)
fixes all 56 cases, layouts, input policy, numerical gates, timing windows, mode order
and budget before collection. The existing seven-case/model protocols and champions
remain separate. There are zero implementation candidates and no confirmation or
promotion in this context.

All twelve unary and four binary operations run at `[2,257]` and `[8,64,4096]`.
Add/Mul also cover channel broadcast, outer-product broadcast, transposed reads and
writes, positive gapped strides, overlapping reads and scalar inputs in both operand
orders. Sigmoid/Softplus cover the four noncontiguous/read-overlap regimes. These
procedural fixtures probe call overhead, working-set size and traversal costs; they
are not a sampled application distribution. Negative strides remain functionally
tested but have no direct PyTorch strided-view comparator in this protocol.

Inputs use the existing fixed wrapping integer hash, salts 11/29 and exact FP32
power-of-two scaling. General unary/left values are `4*x`; Log/Sqrt/Reciprocal inputs
are `0.125+abs(x)`, and binary right values are `0.5+abs(x)`. Backing buffers cover the
complete positive-stride span plus two tail elements. Output backing starts with quiet
NaN bits `0x7fc12345`, including holes/tails. Every geometry is explicit; no hidden
contiguity copy, dtype change or materialized broadcast is allowed in timed work.

## Compared execution modes

Intel uses the common Rust pointwise route, PyTorch eager out-parameter bindings, and Inductor
compiled execution. NVIDIA adds eager and compiled single-operation CUDA graph replay.
Every implementation consumes resident FP32 inputs and overwrites the same explicitly
strided reusable output. Use public `torch.*(..., out=...)` bindings where exposed,
and `aten.relu/silu/softplus.out` for the remaining operations, with the actual endpoint
recorded. `softplus.out` uses beta 1 and threshold 20; fixture magnitudes
are below the threshold. Single-operation capture is never unrolled into a graph of
many repetitions. Native generic pointwise graph replay is not implemented, so its
absence stays visible rather than implying matched replay support.

Compilation uses `backend="inductor"`, `fullgraph=True`, `dynamic=False`, and
`mode="max-autotune-no-cudagraphs"`. The explicit graph modes capture an already
prepared eager/compiled call after five side-stream warm-ups, retaining all input,
output and graph owners. This follows the version-pinned
[compile mode](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html) and
[CUDA graph lifetime/stream rules](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).
Required mode preparation failures terminate that hardware context; no eager fallback,
case removal or relaxed error gate is admitted.

Use release/locked builds with `RUSTFLAGS="-C target-cpu=native"`, and CUDA architecture
86 / g++-13 on the current NVIDIA host. Record binary/toolchain/linkage and flags.
These are hardware-specific artifacts, not portable-ISA claims. CPU processes use
affinity `{0}`, one intra/inter-op thread, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
and one Inductor compiler thread. There is a new dedicated compiler cache per context;
it may be reused by later ordered cases, so first-call times are not uniformly cold
compiler-cache measurements. GPU clocks remain driver-managed. Record versions,
affinity, load/governor/frequency and available GPU telemetry before/after each case.

## Experimental unit and timing boundary

Start one fresh native and one fresh PyTorch worker per case. The Torch worker retains
the required eager/compiled/graph variants with shared resident payload buffers. Prepare
workers sequentially; keep them idle while the other implementation runs. No timing
jobs compete on the same host. This reduces process-startup/import asymmetry, while
retaining shared-host/cache/clock limits. Both hosts may collect independently.

Run six blocks on Intel and ten on NVIDIA. In each half, cyclically rotate the listed
mode order by `(block + case_index) % mode_count`; reverse the order in the second
half. Thus every mode occupies every position once in each direction. Preserve exact
order. A worker performs the regime's declared warm-ups before each window, followed
by the exact fixed iteration count. No time-based calibration, trimming, replacement
windows or extension after inspecting results is allowed.

Primary latency is host wall microseconds per completed operation: synchronous CPU,
or the GPU enqueue loop plus final completion. Native timing includes safe view binding
and geometry/session checks; its per-call optimizer barrier prevents eliminating
repeated pure CPU writes. PyTorch timing includes its selected callable/graph wrapper.
GPU event stream intervals are secondary and can contain host-driven idle gaps; they
are not isolated kernel times. Event construction is outside the wall window; event
record/completion calls are inside. Downloads, file output, checking, preparation,
compilation and graph capture are outside warmed windows and reported separately.

Record worker-ready wall time, payload/context preparation, plan/callable preparation,
first completed call, raw window latency, coordinate workspace and available allocator
observations. Scopes differ and must not be added as exclusive costs or treated as
matched process/device peaks. Report nominal logical operand bytes per second only
as an explanatory metric; broadcasts, cache reuse and nonlinear compute prevent
equating it to measured DRAM bandwidth.

## Hard gates and preserved evidence

Before any timing and after every window, dump/check the complete input/output backing
buffers outside timing. Inputs must retain all generated bits; output holes/tails must
retain the sentinel. Every logical result must be finite and satisfy independent CPU
FP64 equations with `atol=2e-6, rtol=3e-6`, matching the pointwise contract. Require exact
same-mode repeated output hashes for fixed inputs. Cross-library nonlinear outputs
need not have identical bits. Check actual case/mode/count/feature/affinity metadata.
A failed correctness/repeatability/artifact gate stops the context and preserves all
completed records and the failing raw output; it cannot become a partial-suite speedup.

Freeze a hash-named executable before collection; retain its compressed exact bytes,
the protocol, all source hashes and archive, worker logs, commands, readiness records,
input/output hashes, numerical metrics and every raw latency window/order. Check source
and frozen-binary hashes throughout. Raw FP32 files follow the existing ignored result
policy and remain on the collecting host with hashes; copied receipts alone do not
recompute omitted arrays. A recovery check verifies archive bytes and the exact binary;
isolated rebuilding and cross-host full-output rechecking are separate evidence.

The budget is one context per host, at most two hours per context, 600 seconds for
worker preparation and 120 seconds per command. Transport/observation timeouts do not
restart a live worker. Preserve terminal failures and unavailable tools. Validation-only
harness checks before the frozen collection may exercise readiness/commands but do
not supply performance results or consume a replacement measurement context.

## Diagnosis and interpretation

After timing completes, profile the five preselected native cases with the fixed counts
in the protocol, once each. Try CPU perf sampling and NVIDIA Nsight Systems CUDA traces
when available. CPU profiles include complete process setup and report that scope;
CUDA profiler start/stop bounds the warmed native loop. Save exact commands, tool
versions, errors, raw profiles and summaries. Profiles diagnose and are never mixed
into latency acceptance. Missing tools/permissions remain explicit; do not change
global profiling/security settings or infer unavailable counter data.

Publish all case/mode medians, paired descriptive log-ratios and the equal-case
geometric mean, grouped by size/layout/operator where useful. Inner iterations and
blocks within the same workers are not independent confirmations, so there is no
release confidence interval or generalization claim. A fastest observed Torch mode is
descriptive selection, not a deployable cross-case oracle or proof of broad superiority.
Use observed gaps and profiles to choose a later intervention with its own frozen
native control, useful-gain/regression gates and fresh confirmation. This baseline
does not close training, real-time, application-quality or framework-wide performance.

## Commands

Use the recorded virtualenv entry point without resolving its symlink. Build on each
actual host; add `--vendor-dir /path/to/vendor` for the offline NVIDIA environment.
The producer argument identifies the committed implementation, while per-file hashes
enforce the actual bytes. Use new build/collection directories for every invocation.

```sh
.venv/bin/python tools/build_pointwise.py --platform intel --producer COMMIT \
  --output-dir benchmarks/results/local-pointwise-build
.venv/bin/python tools/pointwise_benchmark.py --platform intel \
  --binary target/release/fa-pointwise \
  --build-receipt benchmarks/results/local-pointwise-build/build-receipt.json \
  --output-dir benchmarks/results/local-pointwise-characterization
.venv/bin/python tools/assess_pointwise.py \
  benchmarks/results/local-pointwise-characterization/comparison.json \
  --output benchmarks/results/local-pointwise-assessment.json
```

`--validate-only` permits explicit `--case` subsets and one transport/checking block,
with no characterization summary or performance acceptance. Its status is rejected by
the assessor. It is a harness validation path, not a smaller accepted workload suite.

## Validation outcome: direct compiled out-parameter path unavailable

Producer `a93acc4` implements declaration `a9d9d35`. Portable checks pass 68 Rust tests,
thirteen doctests, 26 Python tests and all-target Clippy/format checks. Hardware-specific
release builds complete with the declared flags. Both 24-case readiness/transport
validation attempts terminate at their twentieth selected case, `gapped-add`: installed
Torch 2.14.1+cpu and 2.14.0+cu130 reject a non-contiguous `out=` argument under Dynamo
full-graph compilation. The native readiness check passes; the required compiled
Torch worker exits before emitting readiness. Both controllers and their workers have
been observed terminal; this is a frontend capability failure, not a live job to restart.

Nineteen prior validation cases complete on each host, with 57 Intel and 95 NVIDIA
validation-only windows. Their recorded numerical/input/hole/repeat checks pass, but
they are not admitted as a partial performance characterization. No formal context,
profile or comparison summary was collected. No case, mode, threshold or stopping rule
was weakened, and no eager fallback or replacement performance run was used.

The [failure audit](../benchmarks/results/2026-10-19-pointwise-v1-failure-audit.json)
and [audit runner](../benchmarks/results/2026-10-19-pointwise-v1-failure-audit.py) verify
both exact compressed executables, 114-file archives against the committed producer,
build receipts and all worker log hashes. Original failed receipts remain at
[Intel](../benchmarks/results/2026-10-18-pointwise-intel-harness/comparison.json) and
[NVIDIA](../benchmarks/results/2026-10-18-pointwise-nvidia-harness/comparison.json).
The Intel build's literal requested `HEAD` label is preserved; its separate
[resolution receipt](../benchmarks/results/2026-10-18-pointwise-intel-build/producer-resolution.json)
proves the same committed source bytes. Raw arrays remain on the collecting hosts,
and isolated rebuilding is not claimed.

The direct-out compiled frontend cannot meet this declaration on the tested stack.
A follow-up must declare its compiled writeback route before collection while retaining
all 56 cases, layouts and original gates. V1 remains a preserved failed declaration.
