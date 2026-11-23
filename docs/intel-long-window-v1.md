# Intel long-window confirmation: predeclared protocol

This is a new measurement context after the inconclusive campaign in `7f7affd`.
The implementation and frozen binaries stay unchanged. The previous 50-iteration
short-STFT block lasted about 0.7 ms, leaving the 5% regression guard unresolved.
Longer fixed windows and warm-up address that precision problem. No old run is removed
or relabeled, and changing the protocol is not an optimization gain.

The authoritative configuration is
[`intel-long-window-v1.json`](../benchmarks/protocols/intel-long-window-v1.json).
Its candidate/anchor SHA-256 identities must match before collection. Use one thread
on logical CPU 0, a performance core of the i5-13420H. Do not compile or run another
CPU benchmark during collection. Library development can use a separate checkout;
the collector must use an immutable source checkout and copied binaries.

| Case | Warm-up iterations | Timed iterations |
| --- | ---: | ---: |
| gemm-stream | 500 | 1000 |
| gemm-projection | 20 | 50 |
| conv-audio | 50 | 100 |
| conv-pointwise | 20 | 50 |
| conv-grouped | 1000 | 2000 |
| stft-stereo | 500 | 1000 |
| stft-stream | 10000 | 20000 |

Counts are fixed using the previous latency ranges, targeting roughly 0.1 seconds or
more for the short measurement windows. No pilot-based resizing is allowed. All seven
workloads, fixtures, strict-FP32 precision and full-output error bounds remain unchanged.
Both Rust binaries and PyTorch use the same per-case counts. First/second ordering
alternates by execution ID, case and block. PyTorch checks follow each Rust pair.

Run exactly three fresh harness executions, each with seven paired blocks. Use distinct
execution IDs 1, 2 and 3 and new result directories. Stop on correctness failures. Do not
add runs or change counts after inspecting outcomes. Preserve raw samples, profile bytes,
source archives and hashes. A planned collection budget of 12 minutes is for scheduling,
not a censoring rule: valid slow samples remain part of the result.

Acceptance is unchanged: recompute each execution's paired log-ratios, use the execution
as the uncertainty unit, and apply two-sided Student t 95% intervals with df=2. The primary
geometric-mean ratio lower bound must be at least `1/0.95`; every stratum's lower bound
must be at least `1/1.05`. If any gate remains unresolved, retain the original Intel
anchor. This is one confirmation campaign for one already selected implementation.

Archive verification also checks the saved `protocol.json` hash and actual timing counts
for all three implementations. Restored source can use that companion file with
`--protocol`. `--execution-id` identifies the counterbalancing schedule; it is not a
request to resume an incomplete prior execution.
