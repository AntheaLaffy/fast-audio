# Prepared CPU backward dispatch: qualified native training gain

## Decision and retained control

The [predeclared experiment](cpu-prepared-backward-v1.md) is accepted for its fixed
native CPU suite. Across twelve fresh contexts, the four primary audio cases have
paired geometric-mean speedup **1.5648**, with simultaneous one-sided lower bound
**1.5341**, above the required 1.10. All twelve per-case mean slowdown guards pass.
The tightest is `dense-tiny`: estimate **1.0556**, lower bound **0.9969**, above
`1/1.05 = 0.952381`. Its bound includes parity, so a tiny-dense speedup is not established.

Keep the prepared implementation as the default CPU backward path. The
[full evidence audit](../benchmarks/results/2026-10-20-cpu-prepared-assessment.json)
binds exact artifacts, functional/header/bit/resource gates, all twelve contexts
and terminal processes. The
[independent decision](../benchmarks/results/2026-10-20-cpu-prepared-confirmation-assessment.json)
recomputes the complete acceptance rule without a timing screen or measured revision.

| Role | Producer | Exact executable SHA-256 |
| --- | --- | --- |
| Fixed original anchor | `3b6635754637ca81e897e6431ae88695b8c51496` | `121f8931a976d349e315d6b4f74be7f4330473fba25f43338ab6c4c4b5cbcef5` |
| Qualified native CPU control | `04ce7d65562c2049b708ff1b38d7cdbb0bc4395d` | `976483c1b7905784b6068d1f0ce04b8f85fbff2f4a57f9b803922fd8aa14ddec` |

Declaration `9d5056f` precedes implementation. Protocol SHA-256 is
`034696009a1c4ac99c8a8ea2bfee06c09dcc38ea94e49963b0916af219891389`.
Future CPU training experiments should retain the qualified control and original
anchor, including cumulative guards against the latter. Existing operator/model
champions and both negative [row](cpu-backward-traversal-results-v1.md)/
[isolation](cpu-backward-isolation-results-v1.md) outcomes remain unchanged.

This compares native binaries on the i5-13420H, CPU 0, one MKL thread, strict FP32
and Rust 1.98.1 with the original release flags. No fresh Torch timing, C++/AOTInductor
deployment or NVIDIA latency qualification is supplied. It does not update historical
Torch ratios or establish framework-wide performance leadership.

## Complete fixed confirmation

Each case has six balanced paired windows in each of twelve fresh contexts. Original
worker windows, eight validation updates, resets, repeats, numerical tolerances and
actual-update checks remain unchanged. Startup/execution order is counterbalanced;
case order rotates/reverses. One candidate receives zero timing screens and no source,
sample-count or threshold changes after admission.

Each case/context effect is the mean paired log wall-latency ratio. The primary
effect equally averages the four declared large waveform/spectral cases. Compute
df=11 Student-t lower bounds across the twelve contexts with `alpha=0.05/13` for
one primary and twelve case bounds. Inner updates and resident blocks are not extra
independent samples. The model assumes approximately normal independent context
effects on this pinned host; error control excludes previous/unlimited future trials,
unseen inputs and deployment populations.

Wall latencies are microseconds. Pooled medians describe 72 windows per mode;
acceptance uses paired context means and their bounds, not median ratios.

| Case | Control median | Candidate median | Paired speedup | Lower bound | Guard |
| --- | ---: | ---: | ---: | ---: | --- |
| dense-tiny | 3.005 | 2.848 | 1.0556 | 0.9969 | pass |
| dense-tail | 23.764 | 13.582 | 1.7584 | 1.6422 | pass |
| dense-projection | 2255.775 | 1236.120 | 1.7550 | 1.6995 | pass |
| dense-large | 13005.166 | 9655.497 | 1.3455 | 1.3037 | pass |
| wave-tiny | 11.212 | 9.453 | 1.2447 | 1.1641 | pass |
| wave-block | 38.543 | 23.389 | 1.6554 | 1.5710 | pass |
| wave-multichannel | 4963.975 | 3037.491 | 1.6453 | 1.5798 | pass |
| wave-long | 6094.623 | 3272.446 | 1.8706 | 1.7582 | pass |
| spectrum-tiny | 16.450 | 13.127 | 1.2389 | 1.1689 | pass |
| spectrum-block | 200.212 | 141.662 | 1.3929 | 1.2639 | pass |
| spectrum-medium | 10294.582 | 7126.375 | 1.4071 | 1.3437 | pass |
| spectrum-large | 58731.334 | 44026.648 | 1.3845 | 1.3372 | pass |

![Every case's paired estimate and simultaneous lower bound](../benchmarks/results/2026-10-20-cpu-prepared-all-cases.svg)

The [CSV](../benchmarks/results/2026-10-20-cpu-prepared-all-cases.csv) and
[summary](../benchmarks/results/2026-10-20-cpu-prepared-summary.json) retain every case.
Segments connect lower bounds to estimates, not symmetric intervals. The guards
constrain mean effects over this balanced design, not every individual context,
startup order, request or tail observation.

[Context/order diagnostics](../benchmarks/results/2026-10-20-cpu-prepared-context-diagnostics.json)
retain absolute medians and all effects. `dense-tiny` has descriptive odd/even-context
geometric means 1.1010 and 1.0121; order-independent gains are not established.
As an explicitly post-hoc sensitivity, grouping adjacent context pairs keeps the
primary lower bound at 1.5371 and every case bound above 1.0040. That grouping does
not replace the frozen df=11 decision or prove independent, normal noise. Absolute
latencies differ from historical contexts; both alternatives are remeasured here.

## Implementation, resources and numerical evidence

CPU unary/binary plans preselect operation/operand and Empty, Copy, Tree32 or
Tree256 kernels at construction. Copy selects scalar mapping below sixteen elements
and coalesced signed rows above it. Reduction adjoints preselect empty, scalar/row
and Sum/Mean scaling. Runtime validates exact layouts and dependencies before calling
an immutable function pointer. Local formulas, ordered trees, saved values, ownership,
public API and CUDA kernels are unchanged.

Fixed headers are explicitly outside existing dynamic resource queries. Separate
probes account for their real logical growth:

| CPU plan | Original bytes | Qualified bytes | Growth |
| --- | ---: | ---: | ---: |
| UnaryBackward | 504 | 512 | 8 |
| BinaryBackward | 504 | 512 | 8 |
| ReduceBackward | 432 | 440 | 8 |

The [baseline](../benchmarks/results/2026-10-20-cpu-prepared-abi-baseline.json) checks
all 71 compiled sources against the original control before implementation. The
[candidate probe](../benchmarks/results/2026-10-20-cpu-prepared-abi-candidate.json)
checks the frozen 167-file producer and one-pointer budget. Exact probe executables,
compiler comments and lossless recovery hashes are preserved. Fixed headers are not
allocator usable-size, code-page or process-peak measurements.

Reported dynamic resources stay identical, warmed Rust allocations are zero and
no payload buffer or per-run metadata upload is added. Portable/MKL debug/release
pass 103 Rust tests and 16 doctests each; CUDA/cuDNN pass 152 and 17. All-target Clippy,
formatting, two strict-provenance signed-layout Miri checks, 50 Python tests and eight
separate decision/order/archive/phase guards pass. A supplemental 80-binding probe
checks 2720 long special-value gradient elements, signed/aliased reads, output holes
and unchanged sources against the independently verified scalar policy.

Preflight plus confirmation audit **2496 complete validation snapshots** and
**1728 measured windows**. Predictions, losses, gradients, updated parameters and
unchanged sources satisfy evolving FP64/eager and actual-update controls, agree
bitwise between alternatives and reset/repeat exactly. All **330 recorded processes**
are terminal. These checks do not qualify all layouts' latency, global concurrency,
vendor allocations or process/device memory peaks.

[Read-only code-generation evidence](../benchmarks/results/2026-10-20-cpu-prepared-codegen-observation.json)
records 73 selected-kernel symbols. Unary/binary public `run` methods are not emitted
as standalone symbols; no standalone frame size is inferred for them. Source separates
Copy from contraction frames, but code size/symbol count does not give an isolated
causal speedup. No additional profiler run replaces the complete-step result.

## Recovery and remaining work

The driver uses about 1557.7 seconds; a conservative sixty-second static diagnostic
charge brings the campaign to 1617.7 of 2400 seconds. All twelve contexts and every
sample remain. An early phase-guard test insertion causes a NameError; its log and
explicitly post-failure source copy are preserved. It is corrected before admission
and acceptance timing, with no production-code revision.

The [admission](../benchmarks/results/2026-10-20-cpu-prepared-admission.json) freezes
source, binary, collector, assessor and prerequisites before timing. Immutable
collector copies, raw transports, content-addressed payloads, source/build archives,
compressed executables and actual command exits remain available. No isolated rebuild
is claimed. Recompute the full decision with a fresh output:

```bash
.venv/bin/python benchmarks/results/2026-10-20-cpu-prepared-assess.py --contexts benchmarks/results/2026-10-20-cpu-prepared-candidate-confirmation{1..12} --output target/cpu-prepared-reassessment.json
```

The [default-source continuity check](../benchmarks/results/2026-10-20-cpu-prepared-qualified-default-source-assessment.json)
verifies all 71 compiled library/native/build/worker files and the embedded training
protocol against the qualified producer after documentation/navigation maintenance.

This closes one native CPU optimization family. CPU forward reductions/nonlinear
math, NVIDIA reduction/broadcast/replay, required convolution/normalization/spectral
derivatives, dtype/ownership breadth and allocator/stream/liveness contracts remain
foundation work. Year one continues without new application delivery.
