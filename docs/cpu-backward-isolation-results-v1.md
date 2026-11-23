# CPU backward isolation: screening rejection

## Decision

The [isolation experiment](cpu-backward-isolation-v1.md) stops at its only screen.
The four primary audio cases have paired geometric-mean speedup **1.5376**, but
`dense-tiny` and `wave-tiny` have point speedups **0.9347** and **0.9152** against
the unchanged `1/1.05 = 0.952381` admission guard. The candidate is **not promoted**.
No selection, code revision, retry or confirmation is performed. The original scalar
source is restored as the default, preserving both this result and the earlier
[unresolved traversal family](cpu-backward-traversal-results-v1.md).

This is one screening context, not a confirmed gain or a statistical claim of a
roughly 7%/9% deployment regression. The independent
[evidence assessment](../benchmarks/results/2026-10-20-cpu-isolation-assessment.json)
checks full numerical/resource/byte evidence, actual process completion and the
required early stop. All twelve cases remain visible.

## Declaration and exact artifacts

Declaration `cd9ad8a` precedes implementation. Its original 32-block provision
cannot run on the frozen control, which caps every worker at six blocks.
Correction `a883707` preserves that unexecuted declaration and freezes
[v2](../benchmarks/protocols/cpu-backward-isolation-v2.json) before any timing:
retain all original six-block windows and use twelve independent contexts if the
sole screen qualifies. Its SHA-256 is
`bb87bfaf3f4117ed2666f6865fb9edf42fb417b67b08d452565f5a6e83b2c227`.
The primary, all five-percent guards, fixed anchor and numerical/resource gates
remain unchanged. The planned twelve confirmations are never entered.

| Role | Producer | Exact executable SHA-256 |
| --- | --- | --- |
| Original control | `3b6635754637ca81e897e6431ae88695b8c51496` | `121f8931a976d349e315d6b4f74be7f4330473fba25f43338ab6c4c4b5cbcef5` |
| Isolated row candidate | `d8e535863020089ca2f6c8a49d1dda829af6564e` | `a5cd0a910a908303ebfa60b1e99bcacec75369fcd7abdc77264edc29afdb406b` |

The candidate keeps the original scalar bodies and routes non-contracted pointwise
VJPs and reduction adjoints with at least sixteen elements to non-inlined row bodies.
Local formulas, contraction trees, public API and explicit resources remain unchanged.
The i5-13420H uses CPU 0, one MKL thread, strict FP32, Rust 1.98.1 and the original
release flags. No fresh Torch or NVIDIA timing is collected.

## All screening cases

Wall latencies are microseconds. Each median describes six windows. Screening uses
the mean of paired log ratios, **not** the ratio of pooled medians. In particular,
the tiny dense median ratio and paired estimate have different signs. Every valid
slow sample remains; changing estimands would change the declared decision.

| Case | Control median | Candidate median | Paired speedup | Point guard |
| --- | ---: | ---: | ---: | --- |
| dense-tiny | 2.073 | 2.042 | 0.9347 | fail |
| dense-tail | 18.325 | 10.516 | 1.9664 | pass |
| dense-projection | 1957.274 | 1101.806 | 1.8217 | pass |
| dense-large | 12312.628 | 9032.694 | 1.3804 | pass |
| wave-tiny | 9.528 | 11.330 | 0.9152 | fail |
| wave-block | 32.389 | 19.336 | 1.7202 | pass |
| wave-multichannel | 3994.988 | 2585.666 | 1.5154 | pass |
| wave-long | 4783.098 | 2567.227 | 1.8601 | pass |
| spectrum-tiny | 15.980 | 11.806 | 1.4131 | pass |
| spectrum-block | 171.737 | 111.751 | 1.5622 | pass |
| spectrum-medium | 7212.353 | 5066.835 | 1.4107 | pass |
| spectrum-large | 47697.349 | 34465.182 | 1.4056 | pass |

The [raw screen](../benchmarks/results/2026-10-20-cpu-isolation-screen/comparison.json),
[independent screen audit](../benchmarks/results/2026-10-20-cpu-isolation-screen-assessment.json),
[CSV](../benchmarks/results/2026-10-20-cpu-isolation-all-cases.csv) and
[derived summary](../benchmarks/results/2026-10-20-cpu-isolation-summary.json)
retain all values and orders. No confirmation interval is computed from this screen.

## Functional checks and mechanism evidence

The exact candidate passes portable/MKL debug/release (103 Rust tests and 16 doctests
each), CUDA/cuDNN debug/release (152 and 17), all-target Clippy, formatting, two
strict-provenance signed-layout Miri checks, 50 Python checks and seven separate
experiment guards. The [functional qualification](../benchmarks/results/2026-10-20-cpu-isolation-functional-assessment.json)
binds reports, raw logs, the 166-file producer inventory and exact builds.

A supplemental [public-API probe](../benchmarks/results/2026-10-20-cpu-isolation-exceptional-assessment.json)
checks 80 bindings and 2720 gradient elements across all local formulas, overwrite/
accumulate, long contiguous and signed/aliased rows, output holes and unchanged
sources. It connects the new long path to the already verified scalar special-value
policy. Defined NaN/copy bits and non-NaN results match exactly; otherwise NaN class
is checked. This differential probe complements the independent scalar/FP64 oracles;
it is not a new independent mathematical reference.

The preflight and screen check 384 full validation snapshots, including evolving
FP64/eager and actual-update controls. The 144 measured windows preserve full exact
control/candidate and reset/repeat agreement. Explicit resources stay identical and
warmed Rust allocations stay zero. All 53 recorded build/runner/collector/worker
processes are terminal, including the independently checked NVIDIA runner.

[Read-only code-generation evidence](../benchmarks/results/2026-10-20-cpu-isolation-codegen-observation.json)
confirms the intended isolation: unary, binary and reduction entry stack reservations
return to the control's 1320, 1192 and 24 bytes, excluding pushes. Eighteen separate
row symbols remain. Restoring those prologues is insufficient to pass the complete-
step small-case screen; frame size alone does not explain its outcome. No additional
performance profile or replacement timing is used.

## Preservation, restoration and next work

The screen takes 102.6 seconds; a conservative sixty-second charge covers bounded
premeasurement code-generation diagnostics, for 162.6 seconds against the fixed
2400-second budget. Unspent time does not permit a retry. Preserve the invalid
producer-argument bootstrap rejected before compilation, the initial demangled-symbol
extraction failure, the corrected scripts and all immutable launch copies. Planned
selection/confirmation tooling remains unexecuted and provides no result evidence.

Source archives, compressed exact executables, raw transports and content-addressed
payloads remain recoverable. No isolated rebuild is claimed. Recompute the screen
without rerunning timings, using a fresh output path:

```sh
.venv/bin/python benchmarks/results/2026-10-20-cpu-isolation-assess.py --contexts benchmarks/results/2026-10-20-cpu-isolation-screen --output target/cpu-isolation-reassessment.json
```

The [restoration assessment](../benchmarks/results/2026-10-20-cpu-isolation-full-restoration-assessment.json)
checks all 71 library/native/build/worker source files against the original control
and binds 24 restored-source regressions. Both failed/unresolved candidate families
remain archived; existing operator/model champions stay unchanged.

Before another CPU intervention, examine prepared operation/strategy selection:
the current scalar entry still includes general contraction machinery even when a
plan already knows it is Copy. Any such trial needs a fresh declaration, explicit
accounting for plan metadata and enough independent tiny-case evidence. It must keep
the original anchor and both negative outcomes. Required gradient/operator breadth,
CPU spectral/nonlinear costs, GPU reduction/replay and memory/stream contracts remain
open. Year one continues to exclude new applications.
