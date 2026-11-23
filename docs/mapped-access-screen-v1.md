# Prepared mapped-access mechanism screen v1

The complete-model CPU profile places most user-cycle weight in generic tensor
execution, and generic GPU tensor kernels account for half of summed kernel time.
This experiment tests address calculation for layout and broadcast multiplication
only. It is a development screen, with one implementation and no promotion budget.
The authoritative counts/controls are in
[`mapped-access-screen-v1.json`](../benchmarks/protocols/mapped-access-screen-v1.json).

CPU prepares coalesced mapping dimensions and traverses contiguous/strided rows with
no per-element division. CUDA prepares a small descriptor with checked 32-bit address
bounds and reciprocal divisors. For unsigned `n < 2^32` and divisor `d > 1`,
`high32(n * floor(2^32/d))` underestimates `floor(n/d)` by at most one; one remainder
correction recovers exact integer indexing. Unit divisors are handled directly.
Mappings exceeding the bounds retain the original 64-bit kernel. The existing
112-byte generic tensor descriptor remains unchanged; the new descriptor is private.

Only copies and the indexing of existing FP32 multiplies change. Floating-point
equations, layer-normalization/reduction order, convolution/GRU policies, ownership,
FFT geometry and buffer lifetimes remain fixed. Tests must cover slices, singleton
axes, permutations, broadcasts, changed inputs, tail sizes, reciprocal extremes and
64-bit fallback selection without allocating multi-gigabyte tensors.

Before timing, complete native reference checks must pass all four final pipelines
and all 60 captured model traces. Screen workers check full outputs against the same
original tolerance, then require bit-identical final outputs against the retained
native control and unchanged planned graph/parameter storage. Freeze and seal the
candidate's executable/source identities before the first measured worker. Archive
the exact control report/source and preserve both binaries by hash.

Use the existing collector with an explicit native control:

```sh
taskset -c 0 .venv/bin/python tools/gtcrn_benchmark.py \
  --protocol benchmarks/protocols/mapped-access-screen-v1.json --stratum intel \
  --binary target/release/examples/gtcrn-infer \
  --control-binary target/benchmark-artifacts/gtcrn/8acf9ce2bb09627845b15e1450d3c5dba93354996f9f857c273351cd19ab9686/gtcrn-infer \
  --reference benchmarks/results/2026-10-17-gtcrn-model-cpu-reference \
  --output-dir benchmarks/results/local-mapped-access-screen
```

NVIDIA uses its declared control, `--stratum nvidia`, CUDA reference and the NAS Torch
Python environment. Reports name the implementations `candidate` and `control`; no
Torch latency is collected in this mechanism screen. The original characterization
entry point and report labels remain available.

The screen uses seven counterbalanced fresh-process blocks and rotates four cases.
Intel windows are longer than characterization to reduce short-window sensitivity;
both alternatives use the same new counts. Old absolute measurements are not paired
with new samples. Threads/affinity stay at one/core 0, and observations include CPU
frequency/governor/load and GPU clocks/temperature/power. Conditions remain observed,
not locked. Report every block and an exploratory paired-block bootstrap, without
treating iterations or blocks as independent campaign confirmations.

GNU time records whole-command RSS separately from planned storage. Separate CUDA
memory replay uses process-group NVML sampling and preserves actual poll intervals;
its high-water values are lower bounds, not exact peaks. No profiler or NVML sampler
runs inside outcome latency collection.

Stop after the one declared screen per host, including failed or inconclusive output.
Do not tune another implementation from this screen or append runs until an interval
passes. This fixed GTCRN probe does not establish music/synthesis generality, licensed
audio quality, exact peak memory or a framework release. The roadmap's music separator
and synthesis decoder need their own pinned workloads and acceptance contexts.

## Collected results

The foundation-first [roadmap](roadmap.md) now defers the application milestones
mentioned in the original declaration until after year one. This scope change does
not alter the protocol, its stopping rules or the outcomes below.

The frozen implementation is `e176704`; the protocol was committed earlier as
`6cb3ec3`. Full reference validation passes four pipelines and 60 model traces on
each host, using the same 81 source hashes. Portable/Intel/NVIDIA tests pass 24/26/34
cases, and both hosts pass 16 Python tests. Relevant portable, Intel, CUDA and cuDNN
Clippy builds pass. No floating-point tolerance or vendor algorithm policy changed.

### Intel screen

All 28 pairs pass the original reference gate, bit-identical final outputs and
unchanged planned graph/parameter/inverse storage. Completed wall medians are:

| Input | Frozen control, ms | Candidate, ms | Paired control/candidate geometric mean |
| --- | ---: | ---: | ---: |
| B1 / 4,096 | 8.657 | 6.618 | 1.313 |
| B1 / 16,000 | 31.442 | 24.094 | 1.279 |
| B1 / 64,000 | 125.938 | 98.291 | 1.268 |
| B2 / 16,000 | 63.697 | 47.759 | 1.285 |

The equally weighted paired aggregate is 1.286, approximately 22.2% lower latency
than the retained native control. Its exploratory block interval is [1.261,1.311].
Ratios use paired samples, not the ratio of medians. Operating conditions remain
observed rather than locked. This is a single screen with no independent confirmation;
it does not establish a new comparison against Torch or music/synthesis generality.

### NVIDIA failed exact-output gate

The first pair passes. The second pair, B1/16,000, passes the original numerical and
planned-storage gates but differs in two enhanced-spectrum components by one ULP,
at most `1.863e-9`. Its final waveform is bit-identical. The context is retained as
**failed** and stops after those two pairs. No GPU summary or memory replay was admitted,
and no extra timing run or candidate replaced it.

A separately declared correctness-only probe runs each frozen binary twice, at
B1/16,000 with zero warm-ups and one timed call plus the CLI's initial call. All
four commands retain their mandatory CLI timer fields, which are excluded from
analysis. The unchanged old binary itself produces differing outputs between those
two fresh commands; the candidate does too. The earliest visible differing model
trace is `decoder.de_convs.3`, a transposed convolution/PReLU block. For the control
repeat, its waveform maximum difference is `2.911e-11`; for the candidate repeat,
`2.329e-10`. All probe final outputs pass the original reference gate.

This demonstrates lack of bit repeatability in that diagnostic regime. It does not
identify a particular vendor kernel or establish repeatability under every warm-up
policy. The current cuDNN policy filters reduced-precision numerical notes, without
filtering nondeterministic plans. A later GPU campaign must establish an explicit
repeatability policy before assuming end-to-end bit identity. The failed screen's
gate remains unchanged; the diagnostic does not convert it into a successful result.

### Evidence and recovery

- [Intel raw screen](../benchmarks/results/2026-10-17-mapped-access-screen-intel/comparison.json).
- [Preserved NVIDIA failure](../benchmarks/results/2026-10-17-mapped-access-screen-nvidia/comparison.json).
- [Correctness-only repeatability probe](../benchmarks/results/2026-10-17-mapped-access-repeatability-nvidia/probe.json).
- [CPU complete validation](../benchmarks/results/2026-10-17-mapped-access-mkl-validation/validation.json)
  and [GPU complete validation](../benchmarks/results/2026-10-17-mapped-access-cudnn-validation/validation.json).

All candidate/control source archives verify; both compressed binary pairs recover
their exact measured SHA-256. Restored candidate snapshots rebuild offline and pass
26 Intel / 34 NVIDIA release tests plus 16 Python tests each. This is functional
recovery, not bit-identical rebuilding. The u32 fallback-selection tests avoid huge
allocations; actual above-u32 GPU execution was not exercised on the RTX 3060.
Existing operator champions and unresolved Intel guards remain unchanged. Fresh
confirmation, applicable quality evidence, broader workloads and exact GPU peak
memory remain open.
