# Prepared view adjoints

## Contract

Implement the [view-adjoint requirement](saved-values-v1.md#view-gradients-required-by-a-future-tape)
as `cpu::ViewBackward` and `cuda::ViewBackward`. Preparation takes the exact dense
base shape, forward view layout, upstream gradient layout and an explicit index-table
budget. The two layouts must have the same logical shape. The forward layout must
fit the base's initialized element count; an upstream view must bind its own checked
initialized storage at execution. Scalars, empty axes, signed strides, permutations,
slices, broadcasts, arbitrary read overlap and logical ranks beyond four are included.
The initial execution contract was declared in `88c2500` before implementation.

For logical index `j`, forward address `p(j)` and upstream address `q(j)`, the result is:

```text
contribution[i] = sum(dy[q(j)] for all j with p(j) == i)
Overwrite: dx[i] = contribution[i]
Accumulate: dx[i] = old_dx[i] + contribution[i] for selected i
```

`GradientMode::Overwrite` writes every base element, with exact positive zeros at
unselected addresses. `Accumulate` preserves every unselected bit, including for an
empty view. An empty view overwrites a nonempty base with zeros in overwrite mode.
Execution validates exact upstream geometry, base shape and CUDA session before
writing. An equal element count with different axes is not accepted. Safe borrowed
inputs and exclusive outputs prevent aliasing. No source payload from the forward
operation is required or retained, and there is no implicit materialization/transfer.

## Initial strategies

Geometry chooses a visible, immutable `ViewBackwardStrategy` before execution:

| Strategy | Mapping and work |
| --- | --- |
| Empty | Clear the base for overwrite; accumulate does no payload work |
| Contiguous | Both mappings are contiguous; direct span copy or elementwise add |
| InjectiveStrided | Forward injectivity is proved; coalesced row/scatter traversal has one writer per target |
| Broadcast | Removing zero-stride repeated axes leaves a proved injective mapping; compute unique targets and reduction axes without element-sized index tables |
| Indexed | General address enumeration proves all targets distinct; gather from prepared source offsets |
| Grouped | Remaining arbitrary overlap; sorted target groups gather every contribution once |

The general route sorts `(target, logical_index)` pairs, preserving ascending logical
index within each target without a stable-sort allocation. It stores only selected
targets and their source offsets/group boundaries, not a table over the entire base.
Ordinary broadcasting uses metadata proportional to rank, even for a large logical
view. Initial rules are algorithmic choices, not a measured optimal dispatch claim.

When the unique target count equals the forward span length, every address of that
span is written. Overwrite clears only its prefix/suffix; otherwise it clears the
whole base before selected writes. This proof also applies to permutations/reversals
and grouped maps with a fully covered target interval.

## Numerical order and repeatability

FP32 is explicit. Injective/one-contribution overwrite paths copy the upstream bits;
accumulation adds once to the previous base value. Reductions use 32 lanes when the
largest target group has at most 32 contributions, otherwise 256 lanes. For each target:

1. Initialize every lane to positive zero.
2. Lane `t` adds contributions `t, t+lanes, ...` in ascending logical order within
   that target, rounding each addition to FP32.
3. Reduce lane pairs at distances `lanes/2, lanes/4, ..., 1`.
4. Overwrite with lane zero, or add that completed sum once to the previous base.

CPU execution uses the same lane/tree order. CUDA uses explicit round-to-nearest-even
adds and no floating-point atomic additions. Warp groups have all 32 lanes participate;
block reductions use explicit shared-memory barriers. These follow the
[CUDA 12.4 shuffle participation rules](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_C_Programming_Guide.pdf)
and [rounding operation](https://docs.nvidia.com/cuda/archive/12.4.1/libdevice-users-guide/__nv_fadd_rn.html).

FP64 oracle/adjoint/finite-difference fixtures have magnitudes at most one under the
validated default rounding environment. Separate rounding-order fixtures check bits
under cancellation rather than impose that mathematical error bound. Reduction
padding can change a negative zero to positive zero;
there is no general NaN-payload or cross-stack bit guarantee for arithmetic. Fixed
strategy/input repeats must be exact on the tested stack. This policy is separate
from cuDNN convolution selection and does not claim framework-wide determinism.

## Resources and lifetime

`index_limit_bytes` bounds requested explicit index-table work, excluding geometry
proportional to rank, allocator overhead and process/driver observations. In general
preparation it covers the simultaneous pair vector and retained index tables. CUDA
also checks the overlap of host tables and their uploaded device copy before upload.
Report `index_bytes`, `preparation_index_bytes`, `host_metadata_bytes` and device
`workspace_bytes` separately. Budget failure is explicit and never selects an
unbounded or nondeterministic fallback. Check byte arithmetic before reserving tables;
use fallible index allocations. No exact RSS/device-memory peak is inferred.
Host metadata reports prepared Rust heap data, including boxed geometry, but excludes
fixed plan headers/native control objects. Geometry uses the existing layout allocation
policy; the index ceiling is not a recoverable allocator or a process memory limit.

Run-time traversal/reduction allocates no Rust heap or device-global scratch. CPU
reductions need at most 256 FP32 lane accumulators on the stack. CUDA reports actual
static shared memory for its selected block kernel separately from global metadata.
Immutable plans can be reused with changing inputs; CPU execution instances may
share a plan with independent outputs. CUDA retains the original single-session
stream contract. Descriptor/index uploads finish before host metadata borrows end;
destruction waits for pending uses before freeing metadata. No host payload staging
or per-run metadata upload is allowed.

## Verification gates

- Compare overwrite and accumulation to an independent Cartesian-coordinate FP64
  oracle; include negative/padded upstream views, gaps, interleaved broadcasts,
  genuine nonzero-stride overlap, unproved injective layouts, high ranks, scalar,
  empty views and empty bases. Check untouched accumulation bits and source preservation.
- Check `<view(x), dy> = <x, backward(dy)>` and central FP64 finite differences of
  the independently specified linear objective, with step `1e-4`. Native/reference,
  adjoint and finite-difference checks use `atol=2e-5, rtol=2e-5` on bounded fixtures.
- Check fixed 32/256-lane rounding-sensitive reductions, exact repeats, changed
  bindings/data, accumulation across calls, and original geometry/session rejection.
- Verify index-budget rejection before large table work, zero index storage for
  analytical broadcasts, declared index byte accounting and no warmed allocations.
- Run portable/Intel/NVIDIA regression suites and relevant Clippy/format checks.
  Exercise actual GPU tail/block/warp cases and memory/synchronization instrumentation
  when available. Preserve failures and unavailable checks explicitly.

This adds one backward primitive. A tape, general strided math, other backward families,
optimizers and training execution remain subsequent foundation work. No application
integration or performance promotion follows from correctness/resource checks alone.

## Verified implementation

Producer `af7fdc2` follows declaration `88c2500`. Portable/Intel/NVIDIA suites pass
62/64/83 Rust tests and 12/12/13 doctests. Both hardware hosts pass nineteen Python
checks. All-target Clippy with warnings denied passes for portable, `mkl,onednn`,
`cuda` and `cudnn`; formatting is checked. Portable/Miri results were observed before
the archived backend collections and are not recollected by the assessment runner.

Independent Cartesian-coordinate enumeration supplies the FP64 oracle rather than
reusing prepared mapping helpers. Bounded fixtures pass both gradient modes, the
adjoint identity and central differences under the declared `atol=rtol=2e-5` and
`1e-4` step. Forward copy execution is included in the adjoint checks. Cases cover
negative/padded upstream strides, broadcast/overlap, gaps, unproved injective maps,
high ranks, scalars and empties. Separate cancellation cases verify the exact fixed
tree, while untouched accumulation and injective overwrite check bit preservation.
CPU plans are also shared across actual threads with independent outputs.

CUDA cases pass the independent oracle, CPU bit agreement, exact repeats, changed
bindings and session/geometry rejection before writing. Compute Sanitizer 2024.1.1.0
runs all four CUDA tests separately under memcheck, racecheck and synccheck: all pass,
with zero errors and zero reported race hazards. The cases include 32-lane warp and
256-lane block reductions, tails and grid-stride reuse beyond the launch-grid cap.
The exact executable used by all three commands is retained compressed, with binary
and compressed hashes checked against its receipt. Targeted Miri passes
`cpu_strided_add_keeps_exclusive_bounds_under_miri`; that nightly reports the existing
graph `fetch_update` naming deprecation, while stable Clippy checks are clean.

Tested warmed execution performs zero Rust heap allocations. Exact index-budget
boundaries pass, and a billion-element logical broadcast prepares with no element
index tables. CUDA accounts for simultaneous host/device tables and reports actual
static shared memory separately. These checks do not measure vendor internals or
process/device peaks. The CPU adjoint is the common Rust implementation in portable
and Intel builds; it has no vendor-specific performance claim.

Both completed backend collections freeze the same 101 source hashes. The assessment
verifies producer bytes, both archives and all command logs without replacing results
or rerunning completed commands:

- [Intel checks](../benchmarks/results/2026-10-18-view-backward-intel-validation/validation.json).
- [NVIDIA checks](../benchmarks/results/2026-10-18-view-backward-nvidia-validation/validation.json).
- [Evidence assessment](../benchmarks/results/2026-10-18-view-backward-assessment.json)
  and [assessment runner](../benchmarks/results/2026-10-18-view-backward-assessment.py).

There is no isolated archive rebuilding, fresh model collection, latency comparison,
exact memory-peak acceptance or performance promotion. Finite tested/instrumented
cases are not an exhaustive proof or a cross-stack bit guarantee. A tape, general
strided math, other backward families, optimizers and training execution remain open.
