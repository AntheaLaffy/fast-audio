# Pointwise v2 characterization results

## Outcome and scope

Both declared 56-case contexts completed on 2026-10-19 from producer
`5d42e351750421714fede4888e79822f3d517b30`. Intel records 1,008 timing windows
(six blocks, three modes); NVIDIA records 2,800 (ten blocks, five modes). Every
readiness/window check passes the original FP64 `atol=2e-6, rtol=3e-6` gate, full
input-bit/output-hole preservation and exact same-mode output repeats. Both
controllers and all benchmark workers exited successfully. No context was restarted,
extended, trimmed or substituted after collection began.

This is a completed baseline, with substantial remaining gaps. It contains zero
optimization candidates, zero independent confirmations and no promotion. Native
execution is faster in some regimes and slower in others; neither the suite mean
nor the large uncaptured-compiled ratios establish framework-wide superiority.
Existing champions, failed v1 readiness attempts and older unresolved guards remain
unchanged. Applications and training are outside this increment.

The [v2 declaration](pointwise-characterization-v2.md) and immutable protocol define
the cases, order, budgets and frontend. Eager uses caller-owned out-parameters;
compiled modes fuse functional math and strided `copy_` in a full Inductor graph.
Strict FP32/no-fast-math policy, one CPU thread, affinity `{0}`, resident buffers and
prepared plans apply. Preparation, compilation and transfers stay outside warmed
wall timing and are recorded separately. CUDA wall timing waits for completion;
stream-event windows can include host submission gaps and are not kernel-only time.

| Context | Tested machine and stack | Native route |
| --- | --- | --- |
| Intel | Core i5-13420H; Torch 2.14.1+cpu | Common Rust pointwise plans, release, `-C target-cpu=native`, no vendor features |
| NVIDIA | RTX 3060 12 GiB, Pentium G5420 host, driver 595.71.05; Torch 2.14.0+cu130 | Custom CUDA kernels, Toolkit 12.4, sm_86, g++-13; uncaptured native calls |

These are separate hardware contexts, with different CPU/software environments.
Do not compare their absolute latencies as a hardware or language experiment. The
protocol's fixed finite fixtures also do not establish application quality, every
floating-point category, all input distributions or cross-stack bit repeatability.

## All-case results and important losses

The ratio below is `Torch wall latency / native wall latency`: above one favors
native. First take the geometric mean of paired block ratios for a case, then weight
all 56 cases equally. These are descriptive within-worker observations, with no
independent confidence intervals. Counts above/below one have the same descriptive
scope and are not acceptance decisions.

| Host | Torch mode | Equal-case ratio | Cases above / below one |
| --- | --- | ---: | ---: |
| Intel | eager | 1.33895 | 31 / 25 |
| Intel | compiled | 3.91698 | 39 / 17 |
| NVIDIA | eager | 1.58719 | 31 / 25 |
| NVIDIA | compiled | 9.11930 | 52 / 4 |
| NVIDIA | eager CUDA Graph | 1.07640 | 25 / 31 |
| NVIDIA | compiled CUDA Graph | 1.04702 | 28 / 28 |

The [complete table](../benchmarks/results/2026-10-19-pointwise-v2-all-cases.csv)
contains all 336 host/case/comparator rows, wall medians/ranges, paired ratios,
stream-event medians and preparation/first-call costs. The
[analysis receipt](../benchmarks/results/2026-10-19-pointwise-v2-analysis.json)
also reports each size/layout regime. No fastest-mode oracle replaces the individual
comparators. The figure preserves every case; colors clip at 1/16x and 16x while
labels and the table retain the ratios.

![Every pointwise case and Torch comparison mode](../benchmarks/results/2026-10-19-pointwise-v2-all-cases.png)

[Vector figure](../benchmarks/results/2026-10-19-pointwise-v2-all-cases.svg).

The following losses show why the aggregate is insufficient. Latencies are warmed
wall medians in microseconds; the final column remains the paired geometric ratio,
which need not equal the ratio of those two medians.

| Host / case | Comparator | Native us | Torch us | Torch/native |
| --- | --- | ---: | ---: | ---: |
| Intel bulk-log | eager | 9,097.50 | 1,393.40 | 0.15082 |
| Intel bulk-silu | compiled | 32,609.93 | 6,364.06 | 0.19300 |
| Intel bulk-softplus | compiled | 37,576.18 | 12,160.93 | 0.32593 |
| Intel transpose-write-sigmoid | compiled | 4,660.74 | 497.70 | 0.10802 |
| NVIDIA channel-add | compiled CUDA Graph | 12.74356 | 3.20627 | 0.25152 |
| NVIDIA channel-mul | compiled CUDA Graph | 12.74455 | 3.26258 | 0.25640 |
| NVIDIA transpose-write-add | compiled CUDA Graph | 21.98364 | 10.29205 | 0.46610 |

Intel bulk ratios are 0.75173 against eager and 0.84998 against compiled, despite
faster basic arithmetic. All six bulk transcendental/compound activation cases lose
against both comparators. CPU transposed-write and transposed-read families also
remain weak. Some CPU blocks vary materially: native bulk-SiLU ranges from 29.82 to
39.19 ms. Retain those windows; they do not supply an independent confirmation.

NVIDIA's compiled CUDA Graph comparator is faster in all sixteen bulk cases, although
many gaps are small; its bulk ratio is 0.97768. Channel broadcast is a much larger
loss, with a 0.25395 family ratio. Transposed writes have a 0.54597 family ratio.
Tiny/scalar cases favor native in this context, including the captured comparators;
Python replay/submission overhead and host hardware remain part of the declared wall
boundary. Large compiled-wrapper ratios do not demonstrate equivalently faster
device kernels. Native generic graph replay is still missing.

Preparation costs matter separately. For example, the first compiled `tiny-neg` call
takes about 3.27 s on Intel and 3.01 s on NVIDIA in these ordered contexts, while the
warmed native CPU call is about 0.088 us. Later cases reuse the dedicated compiler
cache. These observations are not universally cold compilation or a general
break-even guarantee.

## Separate diagnostic profiles

All five declared Intel `perf` commands and reports complete, with zero reported lost
samples. They sample the whole native command, including setup, warm-up and dumping,
not only the timing loop. In `bulk-softplus`, `logp1f32` accounts for about 79.08%
self cycles; other libm addresses and the unary loop account for most remaining work.
`tiny-sigmoid` spends about 61.32% self cycles in the unary plan and 37.80% across four
libm address entries. Binary profiles concentrate in the prepared binary loop.
Call-chain unwinding includes unknown frames; do not treat inclusive rows as disjoint
costs or infer vectorization/bandwidth from samples alone. These profiles motivate a
strict-error-budget SIMD math experiment and separate layout traversal work.

All five original Nsight Systems 2023.4.4.54 attempts execute the requested 1,000
iterations and emit QDSTRM streams, but automatic import fails: the multiarch CLI
does not find the installed host importer. Exit code zero and `available=true` do
not mean that an analyzable report was produced. The original logs/receipts remain
unchanged. The same-version `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`
converts those exact streams afterward, followed by `nsys stats`; no workload is
rerun. This follows NVIDIA's [same-version postprocessing requirement](https://docs.nvidia.com/nsight-systems/2023.2/pdf/UserGuide.pdf).

The [profile audit](../benchmarks/results/2026-10-19-pointwise-v2-profile-audit.json)
verifies original/converted/compressed hashes and directly queries recovered SQLite
events. Each capture has exactly 1,000 pointwise kernels correlated with 1,000
successful `cudaLaunchKernel_v7000` calls on one stream, with no overlapping kernels.

| Native GPU profile | Kernel median us | First-kernel-start to last-kernel-end / 1,000, us |
| --- | ---: | ---: |
| bulk-add | 77.1820 | 77.8689 |
| bulk-softplus | 54.1420 | 54.8151 |
| transpose-read-mul | 14.6875 | 15.3492 |
| channel-mul | 12.8640 | 13.5366 |
| tiny-sigmoid | 1.2800 | 7.0791 |

The tiny profile exposes submission gaps under instrumentation. Its uninstrumented
native wall median is only 1.77960 us; do not use the profiled 7.0791-us interval to
predict an equivalent replay speedup. Several profiled launch APIs have a single
2.5–3.0-ms outlier. API intervals and GPU work overlap. The channel kernel itself is
expensive, which supports investigating address calculation/traversal as well as
submission. Hardware-counter attribution and a measured bandwidth ceiling remain
missing; 64-bit division and row layout are hypotheses, not demonstrated sole causes.

## Evidence, recovery and next work

The [complete assessor](../benchmarks/results/2026-10-19-pointwise-v2-assessment.json)
checks declared cases/modes/counts/order, recorded numerical/repeat gates, source
archives, exact compressed binaries, build receipts, worker logs and profiles.
The [producer/transcript audit](../benchmarks/results/2026-10-19-pointwise-v2-analysis.py)
additionally matches both 115-file source inventories to the committed producer and
all 1,232 Intel / 3,024 NVIDIA ready/measurement/shutdown messages to their receipts.
The plotted image and table hashes are sealed in its JSON result and the image was
visually inspected. There are no runtime source changes in this evidence closeout.

- [Intel original report](../benchmarks/results/2026-10-19-pointwise-v2-intel-characterization/comparison.json).
- [NVIDIA original report](../benchmarks/results/2026-10-19-pointwise-v2-nvidia-characterization/comparison.json).
- [Original-stream postprocessing receipt](../benchmarks/results/2026-10-19-pointwise-v2-nvidia-profile-recovery/postprocessing.json)
  and [recovery source](../benchmarks/results/2026-10-19-pointwise-v2-recover-profiles.py).
- [Independent event/hash audit source](../benchmarks/results/2026-10-19-pointwise-v2-profile-audit.py).

Source archives, exact measured executables, CPU perf data and original/converted GPU
traces are retained compressed. Raw FP32 arrays remain on the collecting hosts.
The audits check recorded array gates and hashes without recomputing omitted arrays;
they do not claim an isolated archive rebuild, exact memory peaks, application
quality, training coverage or performance acceptance.

Recheck complete characterization receipts with a new output path:

```sh
.venv/bin/python tools/assess_pointwise.py \
  benchmarks/results/2026-10-19-pointwise-v2-intel-characterization/comparison.json \
  benchmarks/results/2026-10-19-pointwise-v2-nvidia-characterization/comparison.json \
  --output /tmp/fast-audio-pointwise-v2-recheck.json
```

Next foundation work retains this baseline and all older guards. Advance strided
reductions/local derivatives and the tape contracts; in parallel with that sequence,
declare bounded CPU SIMD/transcendental and GPU broadcast/layout mechanism experiments
before editing their hot paths. Keep original special-value/error rules, full-layout
checks, cumulative regression anchors and fresh confirmation. Test native replay at
the runtime layer separately. Do not spend an unbounded search on the suite mean or
defer required differentiation/training until every microbenchmark is optimal.
