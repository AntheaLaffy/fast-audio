# Prepared CPU backward dispatch experiment

## Mechanism and boundaries

The [row family](cpu-backward-traversal-results-v1.md) leaves its tiny guard
unresolved. The [isolation family](cpu-backward-isolation-results-v1.md) restores
entry stack reservations but fails its sole small-case screen. Both stay archived;
neither is promoted or receives additional samples.

Freeze this [new protocol](../benchmarks/protocols/cpu-prepared-backward-v1.json)
before implementation. CPU plans already know their operation and contraction
strategy, while current `run` entries still dispatch operations and contain general
contraction machinery. Select an immutable function pointer at construction:
pointwise Empty/EmptyReduction, scalar/row Copy, Tree32 or Tree256, with operation
and operand specialization. Reduction adjoints similarly select empty, scalar/row
and Sum/Mean scaling paths at construction. Keep the sixteen-element row boundary.
This changes prepared dispatch, not the derivative formulas or ordered lane trees.

Validate exact bound layouts and required payloads before invoking a selected
kernel. Preserve empty accumulation, NaN/copy/zero policies, output holes, shared
plans, saved versions and native complete forward/backward/SGD semantics. No forward,
precision, vendor, CUDA-kernel or application change belongs to this intervention.

The hypothesis is that separating known Copy/tree and operation paths removes
avoidable dispatch and large contraction frames on small gradients while retaining
row traversal on large gradients. Static code size alone cannot qualify it. An
insufficient complete-step gain or any unresolved guarded case retains control.

## Control, resources and gates

Retain producer `3b6635754637ca81e897e6431ae88695b8c51496` and executable
`121f8931a976d349e315d6b4f74be7f4330473fba25f43338ab6c4c4b5cbcef5` as the
fixed control. Use all twelve unchanged procedural training cases, original worker
windows, i5-13420H CPU 0, one MKL thread, strict FP32, Rust compiler and release flags.
No baseline reset or fresh Torch timing is used.

Before confirmation require full portable/MKL/CUDA debug/release, Clippy/formatting,
targeted signed-layout Miri and Python regressions, the existing independent local/
composed references, and a supplemental long exceptional-value/layout probe. A full
numerical preflight checks eight independent updates and exact control/candidate
payloads. Every warmed window retains evolving FP64/eager controls, exact pair and
reset/repeat agreement, identical explicit dynamic resources and zero hot allocations.

Existing operator and tape resource contracts explicitly exclude fixed control
headers. Record those separately rather than silently counting them as free:
`UnaryBackward`, `BinaryBackward` and `ReduceBackward` may each grow by at most
one eight-byte function pointer on this measured x86_64 ABI. Compare baseline and
candidate `size_of` probes built with the same compiler/features. Extra allocator
usable-size, code-page and process peaks remain outside this logical byte model.
No element-sized plan tables, additional payload buffers or per-run metadata upload
are allowed. The baseline header probe verifies all 71 compiled sources against the
original control before implementation.

## One candidate and fixed confirmation

There is one source candidate, **zero timing screens** and no measured revision.
Freeze exact source/build/executable, collector, assessor and candidate admission
after all functional/numerical/resource/header gates pass. Then run exactly twelve
fresh contexts. One noisy single-context point gate cannot select or stop this
already declared candidate; only the complete fixed confirmation decides performance.

Each case/context uses the original six balanced paired blocks, alternating startup/
execution order by context and catalogue index. Rotate/reverse case order, retain
all observations and run no concurrent local Cargo or benchmark work. Inner updates
and resident blocks are not independent contexts.

Each case/context effect is the mean paired log control/candidate wall latency
ratio. The primary context effect equally averages `wave-multichannel`, `wave-long`,
`spectrum-medium` and `spectrum-large`. Across all twelve contexts, use one-sided
df=11 Student-t lower bounds with `alpha=0.05/13`. Require primary lower bound at
least `log(1.10)` and every case lower bound at least `-log(1.05)`, after all hard
gates pass. Preserve the fixed anchor to prevent cumulative tolerated regressions.

The bounds assume approximately normal independent context effects on this pinned
host. Their error-control scope is this one candidate's thirteen bounds, not either
previous family, unlimited project search, unseen inputs or deployment populations.
No distribution-free or framework-wide confidence claim follows.

The budget is 2400 measurement/diagnostic seconds, at most 420 seconds per context.
Charge sixty seconds conservatively for bounded premeasurement static diagnostics.
No partial confirmation interval changes the candidate, sample count or thresholds.
A failed/invalid context or exhausted budget stops the campaign and remains archived;
do not replace it. Unspent budget does not authorize an extra candidate.

## Preservation and follow-up

Preserve declaration, both negative predecessors, exact source/build/header probes,
immutable collector copies, raw transports/payloads/order/environment, functional
logs, failures and the independently recomputable decision. Read-only symbol/prologue
inspection and at most one separate bounded waveform profile pair can explain a
residual; the uninstrumented complete step decides acceptance. Restore the scalar
source on rejected or unresolved acceptance. NVIDIA functional checks do not qualify
NVIDIA latency. Required derivative/operator breadth and memory/stream/runtime work
remain open, and year one continues without new applications.

## Qualified outcome

The [complete twelve-context result](cpu-prepared-backward-results-v1.md) passes
all hard gates and mean slowdown guards. Primary speedup is 1.5648 with lower bound
1.5341; the tightest case lower bound is 0.9969, above 0.952381. Each fixed CPU plan
header grows by exactly eight bytes, with identical reported dynamic resources and
zero warmed Rust allocations. Keep the prepared source as the qualified CPU default
and retain the original anchor for cumulative guards. No timing screen, measured
revision or extra context is used; broader framework and NVIDIA performance remain open.
