# Complete native training-step characterization

## Decision and scope

The verified native tape establishes composition correctness, not efficient complete-step
execution. Before selecting liveness, fusion, scheduling or replay changes, characterize
the current prepared implementation on Intel and NVIDIA. This is a baseline campaign
with no optimization promotion or framework-wide performance claim.

The [immutable protocol](../benchmarks/protocols/training-characterization-v1.json)
declares twelve procedural tensor objectives: four dense MSE cases, four multichannel
waveform objectives and four real/imaginary-pair spectral-power objectives. Cover tiny
dispatch, odd tails, projection-sized matrices, short/long time axes, channel/bin
broadcasts and contiguous versus reversed/permuted predictions. These are framework
workloads, with no application, dataset, model import or FFT derivative claim.

## Equations and state

Dense prediction is `X @ W`. Waveform prediction is `tanh(X * gain + bias)`, with
per-channel parameters. Spectral-pair prediction sums `(X * gain + bias)^2` over the
real/imaginary component axis, with channel/bin gains and two component biases.
Reversed cases reverse time and move time next to batch. Every loss is the mean
squared prediction error. Differentiate all declared parameters and update every
parameter once with plain FP32 SGD at rate `0.01`; no momentum, decay or scaling.

Use exact dyadic fixtures: input element `((i*7+3)%29-14)/32`; dense weights
`((i*13+5)%31-15)/256`; audio gains `0.5+(i%7)/32`; waveform bias
`((i*3+2)%7-3)/64`; spectral bias `[-1/16,1/8]`. Dense/waveform targets are
`((i*5+1)%23-11)/32`, and spectral targets are `((i*3+2)%17+4)/64`.

The native program returns loss and a detached prediction root. Its second root has
a broadcast-zero scalar seed and performs no reverse work. Before every validation,
repeat and timing window, reset parameters to exactly the same initial FP32 bits.
Warm-up updates count toward the trajectory. Never reset inside a timed step or rebase
the reference to native parameters. Timed steps always include forward, overwrite
backward and SGD. Keep inputs, targets, parameters, scratch and seeds resident.

After a timed window, synchronize, then perform one untimed forward at the updated
parameters: the checked prediction/loss belongs to the post-update state. Gradients
are the last pre-update backward result, and parameters are post-update. This extra
diagnostic forward is identical on both frameworks and never enters reported latency.
It respects the native API's invalidation of old outputs after an update.

## Execution modes and measurement

Intel uses native MKL with one thread, Torch eager autograd and full-graph Inductor.
NVIDIA additionally requires complete eager and compiled CUDA graph replays. Compiled
training uses `torch.func.grad_and_value` over the same objective, followed by in-place
SGD, returning loss, prediction and gradients so the complete requested computation
remains observable. Eager uses normal backward and explicit overwrite gradients.
Capture includes forward, backward and updates; all captured owners stay alive.
No graph break, eager fallback, partial-step capture or hidden precision conversion.

Enforce highest FP32 matmul precision and disable TF32/fast math; record actual
Inductor C++ contraction/unsafe-math/CUDA flags. Retain the pinned CPU/GPU Torch
environments from the tape verification. Record preparation (including required
compilation/capture executions) and the first post-preparation step separately.
The primary metric is warmed wall microseconds per complete step, including final
device completion. CUDA event intervals are secondary and can include dispatch gaps.
Transfers, resetting, dumps, reference work and profiling stay outside timing.

Use the declared six Intel / ten NVIDIA counterbalanced blocks: rotate mode order
by case and block, then reverse the second rotation. Native and Torch workers remain
resident per case, run sequentially and share the declared affinity. Preserve every
sample and actual order. Inner iterations are not independent samples. Report every
case and its strongest compatible Torch mode; aggregate ratios are descriptive only.
No fixed latency advantage or minimum useful optimization gain is selected here.
Any later promotion requires a separate frozen candidate, gain/regression thresholds
and fresh paired confirmation in a matched context.

## Admission and evidence

Check complete predictions, scalar losses, all parameter gradients and all updated
parameters against independently evolving analytic FP64 and eager FP32 controls,
using the original tape `2e-5` absolute/relative gate. Validate eight updates and the
actual warm-up-plus-timed trajectory. Source/target bits must remain unchanged.
Require exact same-mode reset/re-execution payloads; do not discard nonrepeatable
windows. A failed mode, gate, worker or incomplete case set prevents a complete result.

During each of the eight validation updates, also compare the exposed gradient and
the actual preceding FP32 parameters with the new parameters. Allow at most four FP32
representable steps from the separate multiply/subtract SGD result, covering compiled
fusion/reassociation. This tighter update-consistency check detects skipped/doubled
updates whose small displacement could hide inside the absolute trajectory tolerance.
It supplements, and never replaces or relaxes, the fixed independent numerical gates.

Freeze the full producer, exact executable, source manifest/archive and protocol
bytes. Preserve readiness, raw request/response transport, sample order, complete
payload identities, summaries and terminal worker receipts. Use exclusive output
directories and bounded worker/context deadlines; preserve failures. Separate CPU
and GPU profiles for the declared cases explain mechanisms, not acceptance. Record
tool permission failures and missing resources explicitly; sampled RSS/NVML and
explicit plan bytes are different scopes and are not exact process/device peaks.

Existing operator/model champions, failed screens and prior evidence stay immutable.

## Tooling and verification state

`tools/build_training.py` seals locked/offline hardware builds from an exact full
producer ID. A mirror without Git accepts a separately sealed producer inventory;
the assessor still checks its bytes against the real committed Git archive.
`fa-training` and `tools/training_torch_worker.py` are persistent native/Torch workers;
`tools/training_benchmark.py` collects full controls, raw requests/replies, content-addressed
payloads, windows and separate profiles. `tools/assess_training.py` rechecks recovered
payloads against the independent controls and rejects partial or validation-only
evidence when asked to assess a characterization.

The [completed baseline](training-characterization-results-v1.md) verifies both frozen
readiness contexts and all 216 Intel / 600 NVIDIA timing windows. The NVIDIA collector
retains its original profiler-export failure; a separate copied-input diagnostic
recovery and explicit assessment close that evidence without replacing timing windows.
Tests exercise analytic/autograd/finite differences, real full-graph Inductor updates,
reset/snapshot semantics, mutating importers and corrupt/partial evidence. Substantial
CPU/GPU losses remain; characterization is not optimization acceptance.
