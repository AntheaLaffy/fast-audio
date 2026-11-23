# CPU backward traversal experiment

## Decision and mechanism

The [complete-step baseline](training-characterization-results-v1.md) exposes large
waveform/spectral losses and repeated address decoding in local backward plans.
`Geometry::group_offsets` accounts for 11.47% of waveform sampled self cycles;
mapped element offsets add 5.51%. Spectral profiles show the same mechanism beside
substantial forward-reduction cost. Those percentages are diagnostic, not a forecast.

Before changing implementation, freeze this [experiment protocol](../benchmarks/protocols/cpu-backward-traversal-v1.json).
Traverse already coalesced rows for CPU pointwise adjoints without contraction and
CPU reduction adjoints. Decode outer coordinates once per row, then use the exact
signed inner strides. Retain the general ordered contraction path and all existing
local derivative formulas. This is one addressing intervention at two measured sites.
Forward math, nonlinear approximations, tree orders, fusion and CUDA kernels are
outside this intervention.

Falsifying observations include unchanged address-decoding samples without an
end-to-end gain, substantial regression in any guarded case, or any bit/resource
change. A gain in one symbol does not close the complete-step gate.

## Fixed control and complete scope

Keep producer `3b6635754637ca81e897e6431ae88695b8c51496` and Intel executable SHA-256
`121f8931a976d349e315d6b4f74be7f4330473fba25f43338ab6c4c4b5cbcef5` as the control.
Its existing build/source/executable receipts are verified; routine optimization
does not replace it with a new baseline. Remeasure both alternatives on the current
i5-13420H with MKL, one thread, CPU 0, strict FP32 and the same release flags.

Use all twelve unchanged procedural training cases and original worker windows.
The primary is the equally weighted geometric-mean control/candidate speedup of
`wave-multichannel`, `wave-long`, `spectrum-medium` and `spectrum-large`.
The minimum useful speedup is 1.10: a small local addressing change should remove
at least a ten percent complete-step speedup to justify its extra execution path.
Every case independently guards against candidate/control slowdown above 1.05.
Tiny, odd-tail and dense regimes cannot disappear into the audio aggregate.

Both workers execute the original complete forward/backward/SGD protocol, including
eight validation updates and all six measured/reset/repeat blocks. Independently
evolving FP64/eager controls and the actual-update four-ULP check remain unchanged.
Additionally require every exposed control/candidate payload to agree bitwise,
identical explicit resources and zero warmed Rust allocations. Preparation and first
execution stay separate. No fresh Torch timing or comparative Torch promotion is made.

## Evaluation, uncertainty and stopping

Allow at most two full screening evaluations of the declared addressing mechanism,
including at most one documented revision. A screen needs all hard gates, primary
point speedup at least 1.10 and every per-case point speedup at least `1/1.05` before
the candidate can proceed. Freeze one exact source/build/executable before confirmation.

Run exactly five fresh confirmation contexts of that candidate/control pair. Each
case/context has six paired blocks, with three executions of each order. The protocol
fixes case/startup/block order; rotate/reverse case order across contexts to reduce
systematic thermal placement. Reset state before every window, run workers sequentially
and retain all blocks. No concurrent compilation or other local benchmark job.

For each case/context, average the six paired log latency ratios. Average the four
primary case effects to obtain that context's primary effect. The five fresh contexts
are the uncertainty units; inner updates and resident blocks are not extra independent
replicates. Inspect the preserved series and environment for drift or instability.

Compute one-sided Student-t lower bounds on mean log effects with four degrees of
freedom, using `alpha=0.05/13` for each of the one primary and twelve guard bounds.
Accept only if the primary lower bound reaches `log(1.10)` and every guard lower bound
reaches `-log(1.05)`, after all hard gates pass. This model assumes approximately
normal independent context effects on the pinned host. It provides no distribution-free,
production-population or campaign-across-unlimited-retries guarantee.

The budget is two screens, one selected candidate and five confirmations, at most
420 seconds per context and 2100 seconds of measurement/diagnostics in this campaign.
Do not add confirmations, discard valid slow blocks or revise thresholds after results.
Reject or mark unresolved evidence explicitly and retain the original control.

## Correctness, resources and diagnostic evidence

Before performance admission, pass portable/Intel debug/release regressions, targeted
signed-layout Miri and CUDA functional regressions. Exercise all current 524 pointwise
VJP geometries, arbitrary-stride reduction adjoints, exceptional/empty cases, output
holes, alias composition, saved versions, failure-before-write and prepared resources.
Existing independent references must cover the new row traversal; add a regression
only where a concrete geometry or contract gap remains.

Collect separate control/candidate waveform profiles when permitted, reusing the
worker's 8192-step diagnostic loop. Use frame-pointer call graphs and 99 Hz sampling
to bound trace size. Record lost/unresolved samples and profiling limitations. These
diagnostics explain a mechanism and never replace the primary result.

Preserve protocol bytes, producer commits, source archives, exact executables, raw
request/reply logs, complete payload identities, block order, environment, failures and
the independently recomputable decision. CPU qualification does not qualify NVIDIA
latency. Forward reductions, nonlinear cost, GPU broadcast/reduction/replay and broader
foundation coverage remain open regardless of this experiment's outcome.

## Completed outcome

The [full results](cpu-backward-traversal-results-v1.md) retain both permitted screens,
the single payload-size revision and exactly five fresh confirmations. Both candidates
pass all frozen functional and bit/resource gates. The selected candidate reaches
1.5571 primary speedup with lower bound 1.5153, but `dense-tiny` has lower bound 0.9248,
below its unchanged 0.952381 guard. The campaign is not promoted and receives no extra
confirmations or changed threshold.

The scalar control source is restored as the default. Candidate commits, exact binaries,
all measurements, source/transport/payload recovery and separate profiles remain archived.
The larger signed/aliased Miri fixture remains as an independent bounds regression.
Any later scalar/row code-generation or tiny-case precision work requires a new declared
experiment; this result does not qualify broader audio or NVIDIA performance.
