# Complete native training-step baseline

The measurements below retain their original producer and context. Subsequent
[prepared CPU backward qualification](cpu-prepared-backward-results-v1.md) confirms
a 1.5648 primary audio gain against the frozen native control; it does not rewrite
these historical Torch comparisons or qualify NVIDIA latency.

## Outcome and scope

Producer `3b6635754637ca81e897e6431ae88695b8c51496` implements the predeclared
[twelve-case protocol](training-characterization-v1.md). Intel completes all 216
timing windows. NVIDIA completes all 600 windows; its immutable collector report
retains a profiler-export failure, resolved by an explicit diagnostic recovery.
The [Intel assessment](../benchmarks/results/2026-10-20-training-intel-characterization-3b66357-assessment.json)
and [recovered NVIDIA assessment](../benchmarks/results/2026-10-20-training-nvidia-characterization-3b66357-assessment.json)
audit the full measurements separately from descriptive performance results.

This closes the declared baseline, not a performance optimization. Native CPU wins
five of twelve cases against the measured Torch modes, while large spectral cases
trail by about thirteen times. Native NVIDIA trails complete compiled CUDA graphs
on every case. No champion changes, application work or framework-wide speed claim
follows from these measurements.

Both contexts use strict FP32, highest matmul precision, TF32/fast math disabled,
one CPU thread and affinity CPU 0. Intel uses an i5-13420H, MKL 2026.0 and Torch
2.14.1+cpu; NVIDIA uses an RTX 3060, Pentium G5420 host, driver 595.71.05 and Torch
2.14.0+cu130. Native NVIDIA uses the installed CUDA Toolkit rather than Torch's
bundled stack. Sources, full environments and exact release executables remain
recoverable. No isolated rebuild is claimed.

## Per-case warmed wall latency

Every sample includes forward, overwrite backward, plain SGD and final completion.
Parameters, inputs and work buffers remain resident. Reset, reference checks,
post-update diagnostic forward, preparation and profiling are outside timing.
These are process-resident block medians, not independent confirmation runs.

The fastest measured Intel Torch mode is full-graph Inductor in every case. The
fastest measured NVIDIA mode is full-graph compiled CUDA graph replay in every case.
The [complete CSV](../benchmarks/results/2026-10-20-training-all-cases.csv) lists every
case; the [derived summary](../benchmarks/results/2026-10-20-training-summary.json)
retains all eager/compiled/captured mode medians. Preparation and first-step observations remain
in each case's readiness metadata in the raw reports.

All latencies below are microseconds. `Native/Torch` greater than one is a native loss.

| Intel case | Native | Inductor | Native/Torch |
| --- | ---: | ---: | ---: |
| dense-tiny | 2.271 | 65.865 | 0.034 |
| dense-tail | 23.044 | 96.911 | 0.238 |
| dense-projection | 2127.566 | 1113.077 | 1.911 |
| dense-large | 13029.118 | 8195.867 | 1.590 |
| wave-tiny | 13.200 | 91.538 | 0.144 |
| wave-block | 33.519 | 76.280 | 0.439 |
| wave-multichannel | 4826.092 | 952.937 | 5.064 |
| wave-long | 6447.039 | 956.028 | 6.744 |
| spectrum-tiny | 17.594 | 92.373 | 0.190 |
| spectrum-block | 237.700 | 125.595 | 1.893 |
| spectrum-medium | 10469.936 | 807.443 | 12.967 |
| spectrum-large | 58543.877 | 4466.603 | 13.107 |

| NVIDIA case | Native | Compiled CUDA graph | Native/Torch |
| --- | ---: | ---: | ---: |
| dense-tiny | 34.624 | 9.202 | 3.763 |
| dense-tail | 30.562 | 10.984 | 2.782 |
| dense-projection | 49.702 | 23.242 | 2.138 |
| dense-large | 167.673 | 98.179 | 1.708 |
| wave-tiny | 35.331 | 4.030 | 8.768 |
| wave-block | 33.871 | 3.579 | 9.465 |
| wave-multichannel | 96.238 | 17.578 | 5.475 |
| wave-long | 98.075 | 13.378 | 7.331 |
| spectrum-tiny | 40.051 | 4.987 | 8.030 |
| spectrum-block | 37.564 | 9.576 | 3.923 |
| spectrum-medium | 169.881 | 17.565 | 9.671 |
| spectrum-large | 1161.089 | 99.065 | 11.721 |

![Native latency relative to the fastest measured compatible Torch mode](../benchmarks/results/2026-10-20-training-all-cases.svg)

The tiny CPU results include the different framework dispatch boundaries. No C++
Torch/AOTInductor deployment baseline was measured. The comparison therefore covers
the declared eager, full-graph and captured frontends, not every possible Torch
deployment. CUDA event observations can contain host dispatch gaps. No language-level
comparison, universal multiplier, streaming deadline or production workload claim is made.

## Numerical, repeatability and resource gates

Each full context validates eight independently evolving updates per case and mode:
288 Intel and 480 NVIDIA snapshots. Full predictions, losses, parameter gradients and
updated parameters satisfy both analytic FP64 and independent eager FP32 controls at
the unchanged `2e-5` absolute/relative gates. Every validation update also satisfies
the four-ULP actual preceding-parameter/gradient SGD check; observed maxima are zero
ULP on Intel and one on NVIDIA. Both standalone frozen readiness campaigns pass too.

Every warmed window is checked against its complete warm-up-plus-timed trajectory.
Same-mode reset/re-execution and across-window payloads agree exactly; source/target
bits remain unchanged. Native warmed Rust allocation counts are zero. Reported plan,
bound buffer and fixture bytes have explicit scopes; they exclude allocator headers,
vendor scratch and process/device peaks. This campaign does not establish arena reuse,
an allocator pool, peak-memory limits or concurrent stream support.

## Separate CPU and GPU diagnostics

CPU profiles run 8192 complete steps per declared case. They include command startup,
although repeated work dominates. The following are sampled **self** percentages;
inclusive call rows overlap and must not be added.

| CPU profile | Selected self samples |
| --- | --- |
| dense-projection | MKL SGEMM kernels 25.05% and 11.28%; unary backward 14.97%; group offsets 11.76%; mapped element offsets 11.19% |
| wave-multichannel | Binary backward 25.91%; unary backward 21.25%; `tanhf32` 16.06%; group offsets 11.47%; mapped element offsets 5.51% |
| spectrum-medium | Forward reduction 33.14%; binary backward 21.11%; mapped element offsets 11.09%; unary backward 9.73%; group offsets 7.82% |

Dense and spectral profiles report zero lost samples; waveform reports 113 lost
samples among approximately 111 thousand recorded samples. Unresolved unwind/address
rows remain in the raw reports. These samples motivate addressing/traversal and
reduction work; they are not exact isolated kernel costs or a proof that changing one
symbol removes its entire sampled share.

Recovered Nsight SQLite events verify successful runtime calls, unique launch/kernel
correlations, one nonoverlapping device stream and all expected SGD instances.

| GPU diagnostic | Steps | Kernels/step | SGD instances | Kernel sum, us/step | GPU envelope, us/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-tiny | 1024 | 10 | 1024 | 15.110 | 80.487 |
| wave-multichannel | 256 | 16 | 512 | 88.002 | 136.053 |
| spectrum-large | 128 | 17 | 256 | 1205.302 | 1219.062 |

The tiny dense stream has substantial intervals between kernels. Waveform broadcast
backward kernels account for about 40.6% of its kernel time. Large spectral
`reduce_warp` accounts for 44.8%, followed by broadcast backward and block reduction.
This supports separate launch/replay, broadcast and small-axis reduction hypotheses.
It does not quantify a future optimization's gain. Runtime API and GPU intervals
overlap and must not be summed. Instrumented envelopes differ from the primary warmed
measurements; no profiler overhead correction or hardware-counter attribution is claimed.

## Preserved export failures and recovery

The original NVIDIA report remains `failed` with error `training profile summary
failed`, SHA-256 `08d1d1b51a56bdac96b4c0c4d3ba08912ccc929b1e49906eb2f55f8420f18e0b`.
Its complete numerical/timing cases are unchanged. The packaged Nsight CLI captures
the first diagnostic but cannot locate the installed host importer. Both installed
capture/import tools are version `2023.4.4.54-234433681190v0`.

The [first recovery](../benchmarks/results/2026-10-20-training-nvidia-profile-recovery-3b66357/recovery.json)
imports the original dense stream directly and fails its immutability check: the
importer changes headers and appends metadata. A pre-import lossless gzip already
exists. The [restoration receipt](../benchmarks/results/2026-10-20-training-original-stream-restoration.json)
records full-file recovery to original SHA-256
`22cf3d8a9f75803a2d1bb2b81b3754e3155a742275283bd51eb15b14e09288a6`;
the modified variant is retained. No prefix/trailer-only assumption is used.
The first recovery's stored counter `new_instrumented_profile_executions: 2` was a
planned count: it failed before either missing workload ran, so its actual new count
is zero. Its failed receipt is preserved rather than rewritten.

The [second recovery](../benchmarks/results/2026-10-20-training-nvidia-profile-recovery2-3b66357/recovery.json)
reuses the dense capture, runs only the two missing diagnostics from the same frozen
executable, and imports copies. It performs zero new timing windows. The maintained
collector now discovers a matching host importer and copies input before importing;
the [real SDK check](../benchmarks/results/2026-10-20-training-importer-fix/verification.json)
confirms the original stream remains unchanged without running a workload.
The explicit recovered-profile assessment still checks every original timing,
trajectory and transport gate; it cannot admit a numerical failure or partial case set.

## Artifact recovery and next action

The [CPU archive receipt](../benchmarks/results/2026-10-20-training-intel-characterization-3b66357/profile-archives.json)
maps original perf streams to lossless gzip files. GPU gzip streams are retained as
36 content-addressed chunks of at most 32 MiB, with aliases, full stream hashes and
bounded sparse patches for the importer-modified copies in the
[GPU archive receipt](../benchmarks/results/2026-10-20-training-gpu-profile-archives.json).
The [recovery-chain assessment](../benchmarks/results/2026-10-20-training-recovery-assessment.json)
checks failure/restoration/recovery identities and archive recovery. Compiler caches
and raw generated FP32 files are excluded; checked payloads remain recoverable from
hashed `objects/*.f32.gz`. No original report or sealed log is normalized.

From the project root, restore exact profiler streams before re-auditing:

```sh
python3 benchmarks/results/2026-10-20-training-pack-cpu-profiles.py --directory benchmarks/results/2026-10-20-training-intel-characterization-3b66357 --restore
python3 benchmarks/results/2026-10-20-training-pack-gpu-profiles.py --restore
.venv/bin/python tools/assess_training.py --directory benchmarks/results/2026-10-20-training-nvidia-characterization-3b66357 --profile-recovery benchmarks/results/2026-10-20-training-nvidia-profile-recovery2-3b66357/recovery.json --output target/training-reassessment.json
```

Use a fresh assessment output path. The pinned CPU Torch environment is needed to
recompute the independent controls. Source archives are checked against the committed
producer and exact executable bytes, without claiming an isolated rebuild.

Next, declare a bounded CPU backward traversal experiment against the retained native
control, preserving arithmetic/tree order, saved-value checks and all layout/resource
gates. Keep forward spectral reductions, CPU nonlinear cost, GPU small-axis reduction,
broadcast and complete replay as separate measured gaps. Allocator/stream lifetimes,
liveness planning and required backward breadth remain year-one foundation work.
