# Pointwise characterization v2: compiled strided writeback

## Reason and inherited scope

The [v1 validation](pointwise-characterization-v1.md#validation-outcome-direct-compiled-out-parameter-path-unavailable)
found the same Dynamo non-contiguous out-parameter limit on both tested hosts. V1 is
preserved as a failed declaration, with no admitted performance context or summary.
V2 changes the compiled frontend explicitly before characterization; it does not
remove a layout, waive a required mode or substitute eager execution for compilation.

The [v2 protocol](../benchmarks/protocols/pointwise-characterization-v2.json) retains
all 56 cases and exact geometry, FP32/input policy, `atol=2e-6, rtol=3e-6`, input/hole
preservation and same-mode repeat checks. Mode lists, order, six Intel / ten NVIDIA
blocks, fixed warm-up/iteration counts, affinity/thread/build settings, two-hour budget,
worker deadlines, five separate profiles and no-promotion rule are identical to v1.
Apart from the name, the only protocol field added is `compile.writeback`.
The original [timing, evidence and interpretation contract](pointwise-characterization-v1.md)
applies with the frontend difference below. No v1 validation window enters v2 results.

## Explicit compiled function

Eager modes still call reusable out-parameter primitives, preferring public bindings.
Both compiled modes instead compile the complete functional expression and caller-view
writeback together:

```python
def unary(source, unused, destination):
    return destination.copy_(functional_operation(source))

def binary(left, right, destination):
    return destination.copy_(functional_operation(left, right))
```

The function is passed to Inductor with `fullgraph=True`, `dynamic=False` and
`mode="max-autotune-no-cudagraphs"`. Select the functional operation/arity before
compilation. Sigmoid, SiLU and Softplus preserve the original finite fixture equations;
Softplus uses beta 1 and threshold 20. There is no graph break, eager fallback or
benchmark-side input materialization, contiguous-output substitution, cast or transfer.
The original caller output view, including its strides and offset, is an argument
and remains the observed result. Shapes/dtypes/devices are fixed and equal for the
writeback; the general broadcasting/conversion options of
[Tensor.copy_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.copy_.html)
are not used. Full backing checks still observe every hole and tail.

Inductor may fuse the expression/store into its generated kernel. Any backend-internal
intermediate or store work remains inside the timed compiled call; it cannot be moved
outside the boundary or omitted from output checks. Record the actual eager versus
compiled endpoint and writeback policy in readiness metadata, plus available generated
kernel counts as diagnostics. A generated-kernel count is not a performance gate or
a claim about all backend-internal memory. The compiled CUDA graph captures this same
already-prepared function after the original side-stream warm-ups and retains all
owners; each replay still performs exactly one requested pointwise result.

Enforce the inherited no-fast-math policy in Inductor: global/CUDA fast math and C++
unsafe math are disabled, and C++ floating-point contraction is off. Record the actual
policy, compiler and generated C++ math flag list rather than assuming process defaults.
The finite numerical gate and original native exceptional-value tests keep their
distinct scopes; compiler flags do not establish cross-library signed-zero/NaN bits.

## Implementation and verification sequence

The active worker/collector/native CLI accept v2 only. V1 source recovery uses its
archived producer and exact executable, rather than keeping an unused live compatibility
branch. Preserve the original protocol bytes and failed receipts. Require tests that
reject v1 in the current runner, changed v2 fields, a missing writeback policy and
unavailable/partial modes. Exercise real full-graph unary/binary writes into non-contiguous
outputs, checking source bits, holes, result identity and repeats. Validate both devices
before starting the complete 56-case context; validation-only output remains inadmissible
for characterization or speed claims.

Use `tools/build_pointwise.py`, `tools/pointwise_benchmark.py` and
`tools/assess_pointwise.py` as in the v1 command examples; their default protocol and
native CLI now select v2. Pass the exact full producer commit to the build helper and
use new build, compiler-cache and result directories. Required-mode readiness failures
still terminate a context. A new comparator frontend is not a native optimization gain,
does not reset an existing accepted guard and does not close the training foundation.

## Verified harness and completed collection

Producer `5d42e351750421714fede4888e79822f3d517b30` implements declaration `ecea37d`.
Portable regression passes 68 Rust tests, thirteen doctests, thirty Python checks and
all-target Clippy/format checks. Real full-graph unary/binary regression tests preserve
the non-contiguous destination's pointer/strides, holes, source bits and repeated results.
The active parser rejects v1 and changed v2 gates; builds require a full producer ID.

Both hardware-specific release builds complete. The same 24-case readiness/transport
suite then passes on Intel and NVIDIA, including the original `gapped-add` failure,
both arities and scalar broadcast orders. All 72 Intel and 120 NVIDIA validation-only
windows pass the original FP64/input/hole/repeat checks. Worker shutdown receipts and
logs are verified. The [harness audit](../benchmarks/results/2026-10-19-pointwise-v2-harness-audit.json)
and [runner](../benchmarks/results/2026-10-19-pointwise-v2-harness-audit.py) verify both
115-file archives against the committed producer, exact compressed executables and
sealed build/worker logs. Raw arrays remain on the collecting hosts and are not
recomputed from copied receipts.

An earlier [CPU capability probe](../benchmarks/results/2026-10-19-pointwise-copyout-probe.log)
and [GPU probe](../benchmarks/results/2026-10-19-pointwise-copyout-gpu-probe.log)
cover all eight gapped/transposed-output Add/Mul/Sigmoid/Softplus cases, each reporting
one generated kernel. GPU capture/replay passes three repeats per case with preserved
input/holes. The [GPU probe source](../benchmarks/results/2026-10-19-pointwise-copyout-gpu-probe.py)
is self-hash-verified; these are capability diagnostics, not latency measurements.

The complete 56-case contexts finish after the harness audit, with source and
executables frozen. The [completed characterization](pointwise-characterization-results-v2.md)
verifies all 1,008 Intel and 2,800 NVIDIA timing windows, original numerical/repeat
gates, source/exact binary identities and worker transcripts. All five CPU profiles
complete. The five original GPU QDSTRM captures are converted with the installed
same-version importer after automatic import fails; no workload is rerun. Validation
windows are not pooled into those contexts. Per-case losses remain explicit, with
no isolated rebuild, exact memory-peak or performance-promotion claim.
