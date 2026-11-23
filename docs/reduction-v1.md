# Prepared strided Sum/Mean and local adjoints

## Declared contract

Add `ReduceOp::{Sum, Mean}`, `ReduceSpec`, `ReduceStrategy` and prepared
`cpu::ReducePlan` / `cuda::ReducePlan`, plus `ReduceBackward` on both backends.
This declaration precedes implementation. These are reusable FP32 primitives, not
a complete reduction family, reverse-mode tape or training API. Maximum/minimum,
products, stable log-sum-exp and other reduction/normalization families remain open.

`ReduceSpec::new(operation, input_shape, axes, keep_dims)` takes explicit unsigned
axes. Reject duplicates/out-of-range axes and canonicalize valid axes in ascending
original-axis order. Empty axes mean identity, including a scalar; reducing all axes
requires listing them. `keep_dims` retains each selected axis with size one;
otherwise remove it. The spec exposes original/result shapes, axes, operation,
keep-dimension policy and the reduction extent. A zero reduced dimension gives
extent zero. An unrepresentable extent is permitted only when the result is empty,
reported as `None`; no traversal or count conversion is performed for that case.

A forward plan binds the spec and exact source/destination layouts. Source shape
must match the original shape; destination must match the result shape and have a
proved injective map. Read sources may broadcast or overlap. Include signed strides,
offsets, permutations, padded axes, singleton-heavy/high ranks, scalars and empties.
Outputs overwrite caller-owned logical addresses only; preserve all source bits and
every output hole. No implicit dtype conversion, payload staging, host fallback,
materialization, in-place aliasing or output allocation is allowed. Reject changed
layout bindings and foreign CUDA sessions before writing, including empty operations.

## Numerical order and empty behavior

For each output coordinate, enumerate reduced axes in ascending original-axis order,
with the last reduced axis varying fastest. Coalescing may preserve that flattened
address sequence but may not reorder/reverse contributions for locality.

- An empty result performs no payload work.
- Sum over zero contributions writes exact positive zero. Mean writes canonical
  quiet NaN bits `0x7fc00000`. Other source axes/strides cannot trigger reads in this
  route; unused geometry may be arbitrarily large under the empty-layout contract.
- One contribution, including an identity, copies its bits for both operations.
- Otherwise choose 32 lanes for at most 32 contributions, or 256 lanes for larger
  extents. Each lane starts at positive zero and adds `lane, lane+lanes, ...` in
  ascending reduced logical order. Reduce lane pairs at distances `lanes/2, ..., 1`.
- Mean divides the completed FP32 sum once by the extent rounded to FP32 using the
  default round-to-nearest environment. Its derivative divides upstream once by the
  same rounded count, rather than multiplying a precomputed approximate reciprocal.

CPU and CUDA follow the same lane/tree order. CUDA uses explicit rounded additions
and division, no floating-point atomics, and one writer per logical output. Respect
full-warp participation and block barriers, following the
[CUDA 12.4 shuffle rules](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#warp-shuffle-functions).
NaN/infinity classification follows these arithmetic operations; no general NaN-payload
or cross-stack arithmetic-bit guarantee is made. Fixed-plan repeats must be exact
on the tested stack. One-contribution copies retain signed zero and quiet NaN bits;
multi-contribution padding/addition can change a negative zero to positive zero.

## Explicit local backward

`ReduceBackward::new(spec, upstream_layout, destination_layout)` takes no forward
payload. Upstream shape is the reduced result; destination shape is the original
logical input and must be injective. For each original logical coordinate, select
its upstream coordinate by dropping/zeroing reduced axes. Sum copies that value;
Mean divides it by the rounded count, except the one-contribution identity route.
`GradientMode::Overwrite` writes each logical destination once; `Accumulate` adds
the completed contribution once to the previous value. Preserve destination holes
and all upstream bits. An empty original shape writes nothing in either mode.

This is the local derivative with respect to distinct logical input entries. If the
forward input is an overlapping/broadcast read view, its storage derivative requires
composition with the existing `ViewBackward`; local gradients must not race by writing
directly into an overlapping destination. A future tape needs only shape/axes policy
for this derivative, without saving or version-checking an unrelated forward payload.

## Preparation, resources and lifetime

Store geometry proportional to rank, not element counts. Separate kept-axis mapping
from the ordered reduced-axis mapping; skip unused/singleton metadata and coalesce
only equivalent address sequences. Expose `Empty`, `EmptyReduction`, `Copy`, `Tree32`
and `Tree256` strategies and selected reduction lanes. These are initial deterministic
routes, not measured optimal dispatch claims. Backward prepares a broadcast mapping.

Report retained dynamic Rust `host_metadata_bytes`, persistent device-global
`workspace_bytes` and actual CUDA `shared_memory_bytes_per_block` separately. Exclude fixed
control headers, caller payloads, allocator/driver overhead and process observations.
Forward/backward execution must allocate no Rust heap or device-global scratch and
perform no per-run metadata upload. Reduction CPU stack storage is at most 256 FP32
accumulators. No element-sized indices or nondeterministic/unbounded fallback.
Geometry allocation retains the existing layout allocator policy; byte accounting
does not establish recoverable allocation for all metadata or an exact memory peak.
CPU adjoints preselect empty/scalar/row and Sum/Mean scaling at construction. The
[qualified native CPU experiment](cpu-prepared-backward-results-v1.md) separately
records the one-pointer fixed-header growth; dynamic byte queries and the ordered
forward/backward mathematical contracts remain unchanged.

Plans rebind changed payloads with exact original layouts. CPU plans may be shared
among threads with independent outputs. CUDA retains the single-session stream,
finishes borrowed descriptor uploads before preparation returns, and waits for queued
uses before freeing metadata. Asynchronous errors keep the existing session contract;
geometry checks do not make a failed device execution transactional.

## Verification before capability claims

Use an independent Cartesian-coordinate FP64 oracle, without reusing layout-index,
prepared mapping or reduction helpers. Bounded fixtures use `atol=rtol=2e-5` for
forward/local gradients and adjoint/central-difference checks (step `1e-4`). Test both
operations, both keep-dimension policies, axis-order canonicalization, unsorted axes,
multi-axis/inner/outer reductions, both gradient modes and changed bindings/data.
Check source/hole bits, exact repeats and CPU/CUDA agreement on the tested stack.

Include negative/gapped/overlap/broadcast layouts, true high-rank maps, scalar/no-axis,
empty result versus empty contraction, singleton axes and huge unused empty geometry.
Separate exact cancellation/tree, signed-zero/NaN/infinity and rounded-count checks
from the bounded finite oracle gate. Reject invalid specs, unsupported destinations,
wrong geometry and foreign sessions before writes. Check O(rank) preparation, no hot
allocations, no device coordinate storage for scalar cases, and reported resources.
Exercise warp/block tails, grid-stride reuse, accumulation and actual host sharing.

Run portable/Intel/NVIDIA regressions, relevant all-target Clippy/format checks,
targeted Miri for any new unsafe host traversal, and GPU memcheck/racecheck/synccheck.
Preserve actual command outcomes and recoverable producer/executable identities.
No timing win or performance promotion follows from functional/resource checks;
new performance work requires its own declared budget and matched baselines.

## Verified implementation

Producer `1c87db4460a41170a83135e3d812a221f0c56e3a` follows declaration `907b931`.
Portable/Intel/NVIDIA debug and release suites pass 74/76/103 Rust tests and
14/14/15 doctests per build. Each platform passes thirty Python regression checks,
formatting and relevant all-target Clippy with warnings denied. CPU reduction code
is the common Rust route in portable/Intel builds; no vendor-specific speed is claimed.

The two reduction targets exercise 80 geometry/keep-policy fixtures under both
operations and two changed payload bindings. Independent Cartesian FP64 forward and
local-gradient references, adjoint identities and central differences pass the
original `atol=rtol=2e-5` / `1e-4` step gates. CPU/CUDA bits agree on tested fixtures;
same-plan repeats, read-source bits, destination holes and both gradient modes pass.
Signed/gapped/permuted, genuine high-rank, broadcast/overlapping reads, scalar/no-axis,
empty-output/empty-contraction and huge unused empty geometry are included. Forward
empty contractions also run with actual zero-storage host/device sources.

Separate cancellation cases verify the fixed lane/tree order and canonical axis
order rather than a reordered sum. One-contribution copies preserve signed zero,
quiet NaN payloads and infinities. NaN/infinity categories, positive-zero padding and
canonical empty-Mean NaNs pass. A real `16,777,217`-element local backward checks
the count rounded to FP32, including a result differing from division by the exact
FP64 integer. It also traverses beyond the grid cap. GPU warp/block reductions cover
524,289 two-element groups and 65,537 thirty-three-element groups with grid reuse.

Prepared calls allocate zero Rust heap storage on tested successful paths. Billion-
element broadcast preparation has bounded O(rank) metadata; scalar CUDA plans use
zero coordinate workspace. Static shared memory is queried from the selected native
kernel and kept separate from global descriptors. CPU forward/backward plans execute
across actual host threads with independent outputs. Local gradients compose with
`ViewBackward` for both zero-stride broadcasts and nonzero-stride overlap, matching
independent storage-gradient and adjoint checks.

Targeted Miri passes `cpu_reduction_signed_stride_bounds_under_miri`, covering
negative/gapped forward/backward accesses and unused empty strides. Nightly emits
the preexisting graph `fetch_update` naming deprecation; stable Clippy is clean.
Compute Sanitizer 2024.1.1.0 runs each of the four GPU tests separately under memcheck,
racecheck and synccheck: twelve commands, all passing, with zero reported errors or
hazards. The exact release test executable used by all twelve commands is preserved
compressed. These checks do not prove every input/layout, cross-stack arithmetic bits
or absence of every possible global-memory race.

The initial NVIDIA collection completes both full Rust builds, then fails at Clippy:
Cargo's external subcommand does not receive the global vendored-source options, so
offline resolution cannot find `rustfft`. Preserve that failed receipt and diagnostic.
A separate continuation places `--config` after `clippy`, reuses the exact completed
regression logs/binaries, and executes only the missing checks. It retains the original
5400-second deadline and every numerical/resource gate. No completed full regression
is rerun, and the original failed collection is not relabeled successful.

- [Portable checks and Miri](../benchmarks/results/2026-10-19-reduction-portable-validation/validation.json).
- [Intel checks](../benchmarks/results/2026-10-19-reduction-intel-validation/validation.json).
- [Preserved NVIDIA failure](../benchmarks/results/2026-10-19-reduction-nvidia-validation/validation.json)
  and [completed remaining checks](../benchmarks/results/2026-10-19-reduction-nvidia-continuation/validation.json).
- [Collector](../benchmarks/results/2026-10-19-reduction-check.py),
  [continuation](../benchmarks/results/2026-10-19-reduction-nvidia-continuation.py),
  [evidence assessment](../benchmarks/results/2026-10-19-reduction-assessment.json)
  and [audit source](../benchmarks/results/2026-10-19-reduction-assess.py).

The assessment matches all three 122-file source inventories to the committed
producer and verifies every command log, failure/continuation lineage and exact
compressed test binary. No isolated archive rebuild, fresh model collection, latency
comparison, exact memory peak or performance promotion is claimed. Pointwise/dense/
convolution/spectral derivatives, the tape, remaining reduction families, general
owning tensors and native training remain open foundation work.
