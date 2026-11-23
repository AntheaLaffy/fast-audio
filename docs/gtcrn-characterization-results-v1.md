# GTCRN complete-model characterization results v1

The first complete native model beats the installed eager Torch GPU stack on these
fixtures, but trails the explicitly prepared Torch CUDA Graph baseline and trails
eager Torch across the equally weighted Intel workload. These are three exploratory
characterization contexts, with no optimization candidate or promotion. Existing
operator champions remain unchanged. The next intervention should target generic
tensor execution on both hosts before assuming launch reduction will close the gap.

## Workload and numerical evidence

Producer commit: `a7caf75`. The [frozen protocol](gtcrn-characterization-v1.md) specifies
four fixed procedural inputs, seven counterbalanced fresh-process blocks, one CPU
thread on core 0, strict FP32, complete warmed STFT/model/inverse-STFT wall latency,
and the original `2e-4 + 2e-4*abs(expected)` gate. Inputs and weights remain resident;
preparation, transfers and serialization are outside the primary timer. Intel and
NVIDIA use different hardware/software stacks and are separate strata.

All 84 pairs completed, along with 16 separate GPU memory workers. The 804 recorded
full-tensor reference/paired checks pass without tolerance changes. Maximum
error/tolerance ratios are 0.0223 on Intel and 0.7525 on NVIDIA. Four separate profile
workers also pass all three final tensor checks. These checks cover numerical behavior
on tones/noise, not speech enhancement quality or unseen-input generalization.

The original `cuda-graph` compatibility probe failed at the synchronization inside
`torch.istft`. Its failure is retained. `cuda-graph-prepared-inverse` is a separately
named baseline: it checks/caches the fixed NOLA envelope during setup and captures
Torch inverse FFT, windowing, overlap fold and normalization. Tests compare it with
the original inverse for all four geometries, modified spectra and NOLA failures.
It does not represent capture of the unmodified `torch.istft` API.

## Completed pipeline latency

Times below are medians in milliseconds. Each NVIDIA context independently remeasures
the same frozen native artifact; its two columns must not be pooled or selected after
looking at their results.

| Input, at 16 kHz | Intel native | Intel eager Torch | GPU native, eager context | GPU eager Torch | GPU native, graph context | GPU prepared Torch Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1, 4,096 samples | 8.436 | 12.193 | 1.138 | 8.178 | 1.120 | 0.661 |
| B1, 16,000 samples | 29.977 | 26.351 | 1.748 | 8.283 | 1.718 | 0.878 |
| B1, 64,000 samples | 121.786 | 90.378 | 3.892 | 8.369 | 3.857 | 1.876 |
| B2, 16,000 samples | 64.000 | 37.433 | 2.270 | 8.727 | 2.228 | 1.063 |

Ratios are geometric means of paired `Torch/native` measurements, equally weighted
across four cases. They are not ratios of the table's medians. An interval resamples
complete block log-ratios with the fixed seed; seven blocks from one execution do not
constitute independent confirmation.

| Context | Paired ratio | Exploratory 95% interval | Largest command RSS, native / Torch, MiB |
| --- | ---: | --- | ---: |
| Intel / eager | 0.833 | [0.809, 0.857] | 123.7 / 343.7 |
| NVIDIA / eager | 4.217 | [4.094, 4.396] | 449.8 / 1336.9 |
| NVIDIA / prepared CUDA Graph | 0.520 | [0.511, 0.532] | 449.8 / 1339.0 |

Intel's four-second native samples range from 82.9 to 137.4 ms; eager samples range
from 56.9 to 118.0 ms. Affinity and thread count are fixed, but powersave, system load
and frequency vary. Preserve every block. A later candidate needs a declared paired
control campaign and better operating-condition evidence; this variation cannot be
resolved by discarding slow samples. GPU clocks also remain driver-managed.

Raw contexts:
[Intel](../benchmarks/results/2026-10-17-gtcrn-characterization-intel-eager/comparison.json),
[NVIDIA eager](../benchmarks/results/2026-10-17-gtcrn-characterization-nvidia-eager/comparison.json),
[NVIDIA prepared graph](../benchmarks/results/2026-10-17-gtcrn-characterization-nvidia-capture/comparison.json).

## Process memory

GNU time RSS includes fresh-process import, preparation, vendor context and output I/O.
It is not inference-only memory. Native planned graph workspace and Torch allocator
counters have different coverage and must not substitute for process observations.

Separate 200-iteration replays sample only each worker's process group. Actual median
poll intervals are about 50–59 ms despite requesting 20 ms; no sampler errors were
recorded. These high-water observations are lower bounds, including setup/context,
and can miss short-lived peaks. They do not establish a precise memory ranking or a
memory-limit acceptance gate.

| Input | Native sampled MiB, both contexts | Eager Torch sampled MiB | Prepared Torch Graph sampled MiB |
| --- | ---: | ---: | ---: |
| B1, 4,096 | 362 | 182 | 248 |
| B1, 16,000 | 374 | 184 | 250 |
| B1, 64,000 | 454 | 372 | 260 |
| B2, 16,000 | 400 | 190 | 256 |

The native executor retains intermediates for tracing even when the example disables
trace downloads. Buffer lifetime planning remains a plausible memory intervention;
these observations do not measure its potential gain.

## Separate native profiles

Each host used the declared single ordinary control and single instrumented run for
B1/64,000, 20 warm-ups and 100 timed calls. Profile timings do not enter characterization.

Intel perf 7.2.10 samples user-space cycles at 199 Hz with DWARF stacks. CPU 0 uses the
core PMU; the hybrid atom PMU contributes no samples. No samples were lost. The flat
self-period report places 72.91% in `Host::execute`, including its inlined generic
tensor kernels, 6.25% in `tanhf32`, 4.92% in `Conv2d::run` and 3.41% in `Gru::run`.
These are symbol self shares over the whole command, including setup/warm-up, rather
than exclusive model-stage times. Assembly annotation shows substantial indexing-loop
work; it does not identify an exact per-operation percentage. Control/instrumented
latencies are 81.85/131.07 ms. Their observed ratio of 1.60 combines instrumentation
and uncontrolled operating-condition changes; it is not an isolated overhead estimate.

NVIDIA Nsight Systems 2023.4.4 records CUDA without CPU sampling/context switches. The
target-only package initially produced only `trace.qdstrm`. A checksum-verified official
package supplied the matching importer, extracted under ignored `target/profilers`.
The existing trace was imported/exported without another model execution. NVIDIA's
[import instructions](https://docs.nvidia.com/nsight-systems/2023.2/pdf/UserGuide.pdf)
require matching collector/importer versions. Initial importer and stats errors remain
in the artifact; successful analysis reads the existing SQLite export directly.

Exactly 121 framing/overlap-add pairs identify one initial call, 20 warm-ups and the
last 100 timed pipelines. The timed portion contains 26,400 kernels: 264 per pipeline.
Its device envelope averages 3.996 ms, versus 3.393 ms of unioned recorded GPU activity;
15.1% of the envelope has no recorded GPU activity from this process. Summed kernel
time is 3.431 ms per call and slightly overlaps, so it must not be added to gaps or
runtime API durations. Generic `elementwise` kernels account for 1.715 ms per call,
50.0% of summed kernel time; persistent GRU kernels account for 1.049 ms, 30.6%.

No H2D/D2H copies, memsets or CUDA allocation APIs occur inside that extracted timed
envelope. Across the complete command, the trace records 7,541,992 H2D bytes and
1,288,112 D2H bytes for setup/input and final outputs. Control/instrumented warmed
wall times are 3.889/3.997 ms, an observed ratio of 1.028. CPU API calls overlap GPU
work and their durations are not added to the device envelope.

Eliminating every recorded gap while leaving kernel work unchanged would still leave
about 3.393 ms of recorded GPU activity, above the 1.876-ms prepared Torch Graph median
in its separate context. This is a diagnostic bound under unchanged work, not a speedup
prediction. Generic tensor indexing/layout specialization is the next focused mechanism
test on both targets. CUDA replay and buffer reuse remain separate possible interventions.
Declare their candidate budget and confirmation before collecting outcome data.

## Artifact recovery and verification

The [audit](../benchmarks/results/2026-10-17-gtcrn-characterization-assessment.json)
recomputes every paired summary/order, checks worker counts/features, asserts recorded
numerical gates and verifies all five source archives. Every archive contains the same
78 source hashes. The characterization archives were restored into empty directories,
rebuilt offline and passed 22 Intel / 29 NVIDIA Rust release tests plus 15 Python tests
on each host. This establishes functional recovery, not byte-identical rebuilding.

Exact measured binaries are additionally preserved as `gtcrn-infer.gz` in the
[Intel profile](../benchmarks/results/2026-10-17-gtcrn-profile-intel/profile.json) and
[NVIDIA profile](../benchmarks/results/2026-10-17-gtcrn-profile-nvidia/profile.json)
directories. Their decompressed SHA-256 matches the frozen identity in every relevant
comparison. Profile data is retained as `profile.perf.data.gz`, `trace.qdstrm.gz`,
`trace.nsys-rep` and `trace.sqlite.gz`; decompression preserves recorded raw bytes.
The NVIDIA [postprocessing companion](../benchmarks/results/2026-10-17-gtcrn-profile-nvidia/postprocessing.json)
inventories its intermediate/imported traces; the original profile report predates that
import. Uncompressed profiler files and procedural FP32 outputs stay ignored, while
compressed diagnostic artifacts, reports, commands and recoverable source are tracked.

To recompute the audit without rerunning any benchmark:

```sh
.venv/bin/python benchmarks/results/2026-10-17-gtcrn-characterization-analysis.py \
  --output target/gtcrn-characterization-audit.json
```

The archived runner and analysis are one-context evidence under the results boundary,
not additional maintained benchmark entry points. The prior rejected integration check
and its [successful frozen rerun](../benchmarks/results/2026-10-17-gtcrn-harness-mkl-v2/validation.json)
also remain available. General optimizing-compiler/vendor baselines, licensed speech
quality, streaming state, exact peak memory and training remain open.
