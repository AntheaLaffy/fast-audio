# cuDNN convolution policy and repeatability validation v1

The mapped-access GPU screen failed its exact-output gate. A separate probe then
found small fresh-process differences in both the unchanged control and candidate,
first visible at a transposed-convolution block. Keep that outcome unchanged.
Strict FP32 plan selection currently excludes reduced-precision engines but does
not exclude cuDNN's nondeterministic numerical note.

This foundation change adds an explicit convolution planning policy and exposes
the actual selected engine metadata. It preserves the existing permissive default;
the deterministic policy excludes engines carrying
`CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC`. If no strict-FP32 plan meets policy and
workspace limits, preparation returns an unsupported-capability error. The fused
and unfused searches must use the same requested policy. No silent policy fallback.

The policy is immutable for a CUDA session and inherited by prepared Conv1d,
Conv2d/transposed convolution and graph-contained convolutions. It controls cuDNN
convolution selection, not RNG, all future operators or cross-device bit identity.
The direct cuBLAS pointwise path and existing GRU/FFT implementations stay in use;
their complete-graph repeatability is checked rather than inferred from a flag.

NVIDIA documents the [engine numerical-note query](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/api/cudnn-graph-library.html#cudnnbackendnumericalnote-t)
and limits [bitwise reproducibility across architectures](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/developer/misc.html#reproducibility-determinism).
Use the installed 9.3 headers as the ABI authority. An engine ID/flag does not
establish identical arithmetic across different selected plans or software stacks.

## Declared validation budget

The authoritative fixture/counts are in
[`cudnn-repeatability-v1.json`](../benchmarks/protocols/cudnn-repeatability-v1.json).
Use the existing four frozen GTCRN numerical fixtures as a complete-graph regression
test, with all 15 model traces and three final tensors checked per case. This adds
no model integration or application feature to year one.

- Two policies: allow nondeterministic engines and require deterministic engines.
- Two warm-up regimes: zero and twenty calls per case.
- Three fresh native processes per policy/warm-up context, each covering all four
  cases with one measured call after the CLI's initial call; traces enabled and
  pipeline-only execution. Twelve native commands total, with no replacement runs.
- Every process must pass the original `2e-4 + 2e-4*abs(expected)` full-output gate.
  Exact same-context comparisons include every trace and final tensor. Require bit
  identity only for the requested deterministic policy. Record permissive-policy
  differences without requiring either repeatability or non-repeatability.
- Record selected engine metadata and compare its stability across fresh processes.
  Require every cuDNN convolution to satisfy the deterministic flag when requested.
  A graph with no cuDNN convolution does not exercise this gate.
- Seal source, executable and reference hashes before collection and verify them
  after every command. Preserve raw outputs on the execution host, source archives,
  exact measured executable and failed/incomplete reports. Stop on worker failure.

The native CLI emits mandatory timing fields; retain them without summarizing
latency or drawing performance conclusions. This budget is correctness validation,
with no screen, confirmation, champion promotion or reinterpretation of the old
GPU screen. Tests must additionally cover numerical oracles, repeated calls,
changed buffers, policy inheritance and workspace/error behavior of the public API.

## Collected results

Implementation `343b3be` follows declaration `4d088b5`. All twelve fresh commands
complete on the RTX 3060 with CUDA 12.4.131 and cuDNN 9.3.0. Every case has ten
cuDNN convolution plans. The observed selections and exact comparisons are:

| Warm-up calls | Convolution policy | Flagged nondeterministic plans per case | Bit-identical tensor comparisons |
| --- | --- | ---: | ---: |
| 0 | Allow nondeterministic | 2 | 114 / 144 |
| 0 | Require deterministic | 0 | 144 / 144 |
| 20 | Allow nondeterministic | 2 | 114 / 144 |
| 20 | Require deterministic | 0 | 144 / 144 |

All 864 full-output numerical checks pass the original tolerance, covering 48
complete-case results and 720 model traces. Deterministic mode passes all 288
required bit comparisons. Metadata is stable within each context. On this stack,
the two flagged plans use engine 25 under the default policy and engine 1 when
determinism is required; other selected engine indices are 0 and 28. These are
version-specific observations, not portable engine-selection rules.

Planned graph storage is equal between modes for each case. That is an allocation
plan observation, not measured process/device peaks. The collection provides no
latency or throughput acceptance. Do not use its emitted timer fields as a new
optimization screen or treat the old mapped-access screen as passing.

- [Complete correctness collection](../benchmarks/results/2026-10-17-cudnn-repeatability-nvidia/validation.json).
- [Fixed-budget collector](../benchmarks/results/2026-10-17-cudnn-repeatability-check.py).
- [Independent raw-output audit](../benchmarks/results/2026-10-17-cudnn-repeatability-assessment.json)
  and [audit runner](../benchmarks/results/2026-10-17-cudnn-repeatability-audit.py).
- [Functional source-recovery record](../benchmarks/results/2026-10-17-cudnn-repeatability-recovery.json).

The archive contains the same 85 source hashes on both hosts. All thirteen source
archives verify. Isolated offline recovery passes 26 Intel / 35 NVIDIA release
tests and 19 Python tests per host. Compressed executable recovery verifies the
exact collected SHA-256; rebuilt-byte identity is not established. Raw procedural
outputs/reference captures remain on the NVIDIA host; the audit preserves their
output digests. Cross-device/SDK, arbitrary shapes and concurrent streams remain
outside this twelve-command claim.
