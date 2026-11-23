# Tensor layout and borrowed storage contract v1

This is year-one framework work. Separate logical geometry from initialized storage
without adding applications or changing the equations of existing audio operators.

## Geometry and bounds

`Shape` includes rank-zero scalars (one element) and zero-sized axes (no elements).
An empty product is zero even if another dimension would overflow a nonempty product.
Nonempty FP32 element counts remain bounded by addressable Rust slice bytes.

`Layout` describes a shape, signed element strides and an element offset from a
storage base. It caches the complete reachable span and a conservative non-overlap
proof during preparation. Bounds use initialized storage length, never capacity.
Negative strides are valid when their complete span stays in bounds. Empty layouts
touch no element; their offset must be at most the bound storage length.

Permutation, counted slicing (axis/start/count/nonzero signed step), broadcasting
and compatible reshape transform metadata only. Reshape preserves logical element
order and may merge/split stride-contiguous chunks, including negative and zero
strides. It fails explicitly when the requested reshape requires materialization.
Singleton stride values do not constrain contiguity. Empty derived layouts retain
a valid existing origin instead of constructing an unused out-of-bounds pointer.

## Borrowing and aliases

The owned `Tensor` remains contiguous FP32 storage with a dense shape. Preparing a
layout is separate from binding it to a tensor. `Tensor::view(&layout)` and
`view_mut(&layout)` borrow storage and metadata without allocation or data movement.
The same layout can bind different allocations with sufficient initialized length.
The layout's logical shape may differ from the owned tensor's original dense shape.

Read views may overlap, including zero-stride broadcasting. Mutable views require
an injectivity proof and an exclusive storage borrow. The proof sorts non-singleton
axes by absolute stride and checks disjoint covered spans. It admits dense,
permuted, reversed and ordinary stepped slices. A failed proof can mean overlap or
an unproved exotic layout; such mutable binding fails instead of promising arbitrary
injectivity. No mutable broadcast with repeated elements is admitted.

Element references/iterators cannot outlive their storage/layout borrows. Safe Rust
prevents an owned tensor from being resized, moved or mutated while read views are
in use, and prevents shared access while a mutable view is in use. An immutable
reborrow of a mutable view temporarily suspends its mutation. Owned cloning remains
a deep storage copy. These view types remain borrowed; [shared retained values](saved-values-v1.md)
own storage/layouts separately. Device pointers remain private.

## Prepared data movement

CPU and CUDA copy plans bind exact source/destination layouts of equal logical shape.
They validate rebinding and ownership and perform no Rust allocation during `run`.
Preparation coalesces axes compatible in both layouts. CPU traverses rows rather than
dividing per element; contiguous rows copy directly and broadcast rows repeat a value.
CUDA uses same-stream device copies for contiguous spans and a prepared signed-stride
kernel for other layouts. Device metadata is uploaded once and accounted as workspace.
Every FP32 bit is preserved, including NaN payloads, signed zero and infinities.

CUDA views retain the existing single-session/thread ownership contract. A plan
rejects views from another session. GPU copies never stage payloads through host
memory. Empty device tensors own no allocation and empty copies enqueue no kernel.
Rank limits derive from the descriptor/SDK integer range, not the old graph rank four.

## Existing operators and empty storage

Owned dense matmul defines empty outputs as no work and a zero-width contraction
as an overwritten zero result. No zero-dimension GEMM is passed to vendor libraries.
Convolution, GRU and spectral geometry retain their explicit positive-size rules.
The current prepared graph rejects scalar/empty values until its kernel ABI and
operator semantics support them. Strided views are initially consumed by copy plans;
existing dense operator APIs remain explicit rather than silently materializing views.

## Training and memory requirements

Borrowed views establish alias/bounds rules; they are not an autodiff tape. The
[tracked owner](storage-versions-v1.md) now supplies opt-in identity and mutation
generations for host/device storage. Writes through that owner invalidate saved
versions before mutable access, including access used for queued device writes.
Layout-only reads preserve the version; a newly tracked deep copy gets an independent
identity. `SavedValue` now retains a strong owner and the exact prepared layout;
the version token alone retains neither. Training-aware packed parameters must refresh
after an update; existing inference snapshots keep their original semantics.

Arena reuse additionally requires the last reader, pending stream completion and
saved-gradient lifetimes to end. Logical non-overlap does not prove asynchronous
completion. Allocator alignment, pooling, typed storage and parameter-version
integration remain subsequent foundation work.

## Verification boundary

Use independent coordinate enumeration to check layout addresses and overlap proofs,
including scalar/empty/high-rank, singleton, negative, broadcast, sliced and overflow
cases. Check copy bytes against separately generated coordinate fixtures, changed
bindings and untouched destination gaps. Test immutable/mutable iterator lifetimes,
allocation behavior, CPU/CUDA parity and foreign sessions. Exercise unsafe host
iteration under Miri when the toolchain is available. Existing primitive and complete
graph numerical gates remain active; no performance victory follows from these tests.

## Verified implementation

Implementation `8b89df6` follows contract declaration `6a9176a`. Portable, Intel and
NVIDIA checks pass 35/37/48 Rust tests, respectively, plus three doctests per tested
runtime configuration. Nineteen Python checks pass on both hosts. The address-set
oracle verifies that every accepted non-overlap proof is injective on its sampled
layouts; an independent affine-basis oracle checks reshape acceptance/completeness
for the small exhaustive cases. These are bounded tests, not an exhaustive proof
over every possible shape.

Miri passes the mutable double-ended iterator/reference test and the targeted
negative/broadcast host-copy test. The installed nightly reports a deprecation
warning for the pre-existing graph identity atomic method; stable Clippy passes
with warnings denied. Miri does not check vendor or CUDA execution.

The existing complete graph passes all four output cases and sixty intermediate
outputs on each backend under the unchanged numerical gate, from the same 93
source hashes. The GPU check requests the existing deterministic convolution policy.
This is a single numerical regression per host, without a new cross-process
repeatability or performance claim.

- [Intel complete-graph check](../benchmarks/results/2026-10-17-layout-views-mkl-validation/validation.json).
- [NVIDIA complete-graph check](../benchmarks/results/2026-10-17-layout-views-cudnn-validation/validation.json).
- [Evidence assessment](../benchmarks/results/2026-10-18-layout-views-assessment.json)
  and [assessment runner](../benchmarks/results/2026-10-18-layout-views-assessment.py).
- [Completed Intel recovery](../benchmarks/results/2026-10-18-layout-recovery-intel-corrected.json)
  and [NVIDIA recovery](../benchmarks/results/2026-10-18-layout-recovery-nvidia.json).

The isolated restored trees pass 37/48 release tests, three doctests and nineteen
Python tests per host. Intel's first recovery runner incorrectly resolved the
virtualenv Python symlink to its base interpreter and failed to import Torch; that
[failed receipt](../benchmarks/results/2026-10-18-layout-recovery-intel.json) and
runner version remain preserved in `610054c`. The correction keeps the requested
entry point and completes Python checks while retaining the already verified Rust
result. No model-collection command was repeated to replace an outcome.

Both source archives and compressed collected executables verify. Recovery is
functional; byte-identical rebuilding is not established. No latency comparison,
exact device-memory peak or champion promotion is admitted. The specialized
positive-shape graph and dense math APIs remain in place. At this implementation's
checkpoint, allocators, owning aliases, mutation generations and strided math consumers
remained subsequent work; the tracked-storage contract records the following increment.
