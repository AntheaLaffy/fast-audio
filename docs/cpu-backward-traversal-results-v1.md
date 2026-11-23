# CPU backward traversal: useful large-case gains, unresolved tiny guard

## Decision

The [predeclared experiment](cpu-backward-traversal-v1.md) is complete and **not
promoted**. The selected candidate's primary paired speedup is **1.5571**, with
simultaneous one-sided lower bound **1.5153**, above the required 1.10. Eleven
individual guards pass. `dense-tiny` has speedup estimate **0.9803** and lower
bound **0.9248**, below the required `1/1.05 = 0.952381`.

This leaves the small-case guard unresolved; it does not prove an 8% regression.
Its point estimate corresponds to about 2% slowdown, but the fixed five-context
evidence cannot exclude slowdown beyond 5%. The original scalar backward source
is restored as the default. No extra confirmations, discarded blocks, changed
thresholds or champion reset are used.

The [full evidence audit](../benchmarks/results/2026-10-20-cpu-backward-assessment.json)
binds functional checks, exact builds, two screens, the frozen selection, all five
confirmations, payload/transport recovery and separate profiles. The
[decision assessment](../benchmarks/results/2026-10-20-cpu-backward-confirmation-assessment.json)
recomputes the numerical gates and statistical decision. This is a native CPU
optimization experiment, with no fresh Torch timing or NVIDIA latency qualification.

## Frozen artifacts and sampling

| Role | Producer | Exact executable SHA-256 |
| --- | --- | --- |
| Fixed control | `3b6635754637ca81e897e6431ae88695b8c51496` | `121f8931a976d349e315d6b4f74be7f4330473fba25f43338ab6c4c4b5cbcef5` |
| First row candidate | `cd8a5eb4f58851c1dec65b421b47ba15f1828b2f` | `982da038394f85079794106a7ab8efdd058ee0f6b0fd8a21f154ad299d18b664` |
| Selected small-payload revision | `bb6761db2ca5fe2bd8485126f38a3f99250852c9` | `9ed7134cb7bd83f13cbdacf5ddfbf9dd4d0e2c46c2e767f6525bf15370331313` |

Declaration `c94ebaa` precedes implementation. The immutable protocol SHA-256 is
`6fff152f08eda39b71830d59bb83a9fc3e16980b46173fa70eca92443e680be6`.
Both candidates reuse prepared signed/coalesced rows for non-contracted pointwise
VJPs and reduction adjoints. The single permitted revision keeps scalar traversal
below sixteen logical FP32 elements. Formulas, contraction trees and payload resources
remain unchanged. Candidate source/ELF archives remain preserved after restoration.

The first full screen has primary point speedup 1.5250 and a nearly exhausted
small-case guard. The second has 1.5732 and passes every screening point gate. Freeze
the [selection](../benchmarks/results/2026-10-20-cpu-backward-selection.json) before
collecting exactly five fresh contexts. Each context contains all twelve cases and
six paired blocks, with balanced control/candidate order and rotated/reversed case
order. Inputs, warmups, timed updates, resets and repeats retain the original worker
protocol. All samples and their actual order remain available.

The i5-13420H uses CPU 0, one thread, MKL and strict FP32, with the same Rust compiler
and release flags on both alternatives. Fresh worker pairs provide context-level
replication. Inner steps and resident blocks are not extra independent samples.
The bounds use the declared df=4 Student-t model with `alpha=0.05/13` for the one
primary and twelve guard bounds. Approximate independent normal context effects are
an assumption; these are not distribution-free, production-population or unlimited-
campaign confidence guarantees.

## Complete confirmation results

All wall latencies are microseconds. Pooled medians describe the thirty windows per
case/mode. Acceptance uses the paired geometric mean and its lower bound, **not**
the ratio of pooled medians. This explains the different signs in the tiny row;
changing to a favorable median statistic would change the frozen estimand.

| Case | Control median | Candidate median | Paired speedup | Lower bound | Guard |
| --- | ---: | ---: | ---: | ---: | --- |
| dense-tiny | 2.403 | 2.390 | 0.9803 | 0.9248 | unresolved |
| dense-tail | 18.075 | 10.715 | 1.6676 | 1.4795 | pass |
| dense-projection | 1859.725 | 1060.879 | 1.7536 | 1.5947 | pass |
| dense-large | 11781.318 | 8737.759 | 1.3399 | 1.2298 | pass |
| wave-tiny | 9.475 | 8.036 | 1.2145 | 1.0805 | pass |
| wave-block | 31.249 | 19.294 | 1.6071 | 1.5134 | pass |
| wave-multichannel | 3879.875 | 2374.759 | 1.6468 | 1.5410 | pass |
| wave-long | 4640.288 | 2500.146 | 1.8551 | 1.6554 | pass |
| spectrum-tiny | 12.738 | 10.422 | 1.2566 | 1.2311 | pass |
| spectrum-block | 153.867 | 106.817 | 1.3957 | 1.1046 | pass |
| spectrum-medium | 7760.268 | 5485.744 | 1.4086 | 1.3091 | pass |
| spectrum-large | 48672.673 | 35877.082 | 1.3659 | 1.3565 | pass |

![Paired speedups and simultaneous lower bounds; red marks the unresolved guard](../benchmarks/results/2026-10-20-cpu-backward-all-cases.svg)

Purple rows belong to the primary audio set; all rows are guards. Each horizontal
segment connects a lower bound to its estimate and is not a symmetric interval.
The red dashed line is the per-case guard. The [CSV](../benchmarks/results/2026-10-20-cpu-backward-all-cases.csv)
and [derived summary](../benchmarks/results/2026-10-20-cpu-backward-summary.json)
retain every case. No aggregate hides the tiny guard or qualifies the whole framework.

## Correctness, resources and diagnostics

Both candidates pass complete portable/MKL debug/release regressions (103 Rust tests
and 16 doctests per build), CUDA/cuDNN debug/release regressions (152 and 17), all-target
Clippy and formatting. Two targeted Miri checks cover signed/aliased row bounds; the
revised pointwise fixture includes 34 elements beyond the scalar dispatch boundary.
Fifty Python guards and seven separate experiment decision/order/archive guards pass.

The campaign checks 1728 full validation snapshots across two preflights, two screens
and five confirmations, plus 1008 measured windows and their exact re-executions.
Full outputs, losses, gradients, parameters and unchanged sources satisfy independent
FP64/eager controls and actual-update checks, agree bitwise between alternatives and
repeat exactly. Explicit resources remain equal and warmed Rust allocations are zero.
These scopes exclude allocator/vendor/process peaks and do not prove all possible
layouts or global concurrency behavior.

Separate 99 Hz frame-pointer waveform profiles execute 8192 updates per alternative;
their final payloads agree bitwise. Both report zero lost samples, with roughly two
thousand control and one thousand candidate samples. The control has 11.82% sampled
self cycles in group-offset decoding and 5.40% in mapped element offsets. Neither
symbol appears in the candidate sample summary. Unary-backward self share falls
from 22.04% to 8.39%. Remaining contraction and `tanhf32` work occupy larger relative
shares. These are sampled proportions, not isolated absolute cost differences; frame
unwinds and sparse samples leave blind spots.

[Read-only disassembly](../benchmarks/results/2026-10-20-cpu-backward-codegen-observation.json)
also records larger candidate backward entries and stack reservations. This motivates
separating scalar and row execution bodies in a later experiment. Code size and
prologue differences alone do not establish the cause of the tiny-case effect.

## Preservation, restoration and next action

Measurement/diagnostics consume about 753.4 seconds of the fixed 2100-second budget.
Both screens and all five confirmations are retained. A collector bootstrap failure
rejects a valid sealed-manifest control proof before any worker starts; its failure
log and explicitly reconstructed source are preserved. The corrected tool validates
the sealed inventory against the committed Git archive and keeps the original control.
Subsequent launches preserve immutable collector copies and hashes before execution.

A separate assessor variable-shadowing fix keeps source hashes distinct from the
explicit profile-recovery selector. Re-auditing the unchanged complete Intel context
and default readiness path verifies the fix; old reports remain unchanged.

Source/ELF archives, content-addressed FP32 objects, raw request/reply logs and compressed
perf streams are recoverable. No isolated rebuild is claimed. For example, from the
project root, use a fresh output path to recompute the full decision:

```sh
.venv/bin/python benchmarks/results/2026-10-20-cpu-backward-assess.py --contexts benchmarks/results/2026-10-20-cpu-backward-candidate2-confirmation1 benchmarks/results/2026-10-20-cpu-backward-candidate2-confirmation2 benchmarks/results/2026-10-20-cpu-backward-candidate2-confirmation3 benchmarks/results/2026-10-20-cpu-backward-candidate2-confirmation4 benchmarks/results/2026-10-20-cpu-backward-candidate2-confirmation5 --output target/cpu-backward-reassessment.json
```

Each `trace.perf.data.gz` losslessly recovers its original profiler stream; the profile
receipt records both hashes. `fa-training.gz` similarly recovers each exact executable.
The [default-source restoration](../benchmarks/results/2026-10-20-cpu-backward-default-source-restoration.json)
records byte equality with the original control's three backward/copy source files;
the [restoration assessment](../benchmarks/results/2026-10-20-cpu-backward-default-restoration-assessment.json)
checks all 71 compiled library/native/build source files and 24 targeted restored-source
regressions. Candidate commits, binaries and evidence
remain available, and existing operator/model champions are unchanged.

The next CPU experiment should isolate scalar/row entry code and declare enough tiny-
case precision before collecting new data. Keep this unresolved result and fixed anchor;
do not extend its completed confirmations. Forward spectral reduction, CPU nonlinear
math, GPU reduction/broadcast/replay and required operator/backward breadth remain
separate foundation work. Year one continues without new applications.
