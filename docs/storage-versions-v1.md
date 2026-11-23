# Tracked storage and mutation versions

## Purpose and boundary

Reverse-mode differentiation and packed-parameter caches must reject a saved value
after its storage may have changed. Pointer equality cannot establish this: moves,
empty buffers, allocator reuse and replacement can preserve or change an address
independently of the logical value. A layout's shape also says nothing about writes.

Introduce explicit `Tracked<T>` ownership for `Tensor` and, with CUDA, `DeviceTensor`.
Ordinary inference tensors, views, operator signatures and hot paths keep their
existing representation. Tracking adds no tape and does not claim autodiff support.
The initial contract was declared in `74ddd0c` before implementation. The compact
error variants were refined during implementation to preserve the common error size.

## Identity and access

Each tracked owner receives a process-local, non-recycled `StorageId`, independent
of payload addresses, tensor shape and backend. A `StorageVersion` pairs that identity
with a modification generation, initially zero. Moving an owner preserves both;
new owners, including empty buffers and deep copies, receive distinct identities.
Identifiers are not checkpoint keys or stable identifiers across processes.

`Tracked::read` exposes only a shared reference. `Tracked::write` advances the
generation before returning an exclusive reference to the existing storage type.
There is no mutable dereference, shallow clone or other safe untracked write route.
A sealed `Trackable` bound permits only audited, exclusively owned storage types;
user-defined interior-mutable or shallow-cloned types cannot opt themselves in.

Generations describe permission to mutate, not a count of changed elements. A write
borrow invalidates every prior version even if it touches no bytes, uses an empty
view, changes only part of a buffer, fails a later operator validation, or unwinds.
One borrow may perform several writes. Layout transformations and read borrows do
not change a generation; all views of one tracked owner share its version boundary.

The exclusive reference can replace the whole inner tensor. The identity therefore
names a tracked ownership lifetime, not a permanently fixed physical allocation.
Replacement invalidates earlier versions through the already advanced generation.
`into_inner` consumes and retires this identity; wrapping the returned tensor creates
a fresh one even when its address is unchanged. No version can be reset or forged
through the public API. Identity/generation exhaustion must return an error before
exposing write access, never wrap around and accept an old version again.

## Saved versions and device ordering

`check_version` returns a shared storage reference only when both identity and
generation match. Different owners return `StorageIdentityMismatch` with both IDs;
a changed generation returns `StorageModified` with that ID and both generations.
Equal shapes/bytes cannot bypass either check. Separate errors avoid placing two full
versions in the common error enum and increasing untracked inference's result size.
A version token retains neither storage nor layout. A future tape must retain the
owner and exact saved layout separately; these tokens alone cannot recover a dropped
activation. [Shared retained values](saved-values-v1.md) now provide that strong
ownership and checked payload access as a separate opt-in layer.

CUDA writes use `write` before the operation obtains its device destination. Thus a
queued or partially failed operation invalidates earlier versions on the host before
device completion. The existing private pointers, single-session stream ordering and
synchronous destruction still govern lifetime safety. Version checks are not events,
synchronization, deferred-error checks or evidence of successful GPU execution.
`Tracked<DeviceTensor>` inherits the original non-Send/non-Sync session contract.

Existing prepared inference plans still snapshot their parameters. Tracking does not
silently refresh them. Training-aware packing will require keys containing versions
and explicit refresh/invalidations; stale snapshots remain valid immutable snapshots
under the current inference contract.

## Resource and verification gates

Identity reservation uses one process-wide atomic counter during opt-in construction.
Write/version/check operations use owner-local integer state and require no atomic
operation, payload copy, heap allocation, CUDA call or synchronization. These are
structural/resource requirements, not a measured latency claim. Ordinary untracked
inference remains usable independently. On the tested 64-bit targets, retain the
existing 48-byte host tensor, 40-byte device tensor and 32-byte common error budgets.

Verification must exercise:

- Unique identities across owners, backends, empty buffers, deep copies, moves,
  replacement, unwrap/rewrap and concurrent host construction.
- An independent access-sequence model for saved-version acceptance, reads,
  potential writes, partial writes/errors and unwinding, plus exhaustion boundaries.
- Borrowed layout reads/mutations, prepared strided copies and dense math on tracked
  storage, with exact/independent numerical expectations and frozen parameter snapshots.
- Compile-time exclusion of external storage types and mutable access through reads;
  retained Rust borrow rules for element/view references.
- Allocation-free successful tracking/binding/execution after preparation; portable
  and Intel execution, and actual NVIDIA queued writes, copies, failed session checks,
  scalar/empty storage and foreign identities.
- Existing backend regression suites and relevant Clippy/format checks. There is no
  performance promotion, finite-difference claim or completed training-runtime claim.

Shared retention is specified and implemented by the retained-value contract. General
owning strided tensor values, allocation pools, asynchronous reuse, a tape and
parameter-gradient/update APIs remain subsequent foundation work.

## Verified implementation

Producer `1bc77b7` passes 47 portable, 49 Intel and 62 NVIDIA Rust tests, plus seven
doctests per tested configuration. Five doctests require compilation to fail,
including external trait implementation, writes through reads and a retained view
across another write borrow. The two runtime doctests also pass. Both hardware
hosts pass nineteen Python checks. All-target Clippy passes with warnings denied
for portable, `mkl,onednn`, `cuda` and `cudnn` builds.

The tests cover 512 concurrent host identities, a bounded independent access-history
model, moves, replacement, deep copies, retirement, exhausted counters, partial writes
and unwinding. Prepared GEMM/layout copies check actual results and untouched source
versions. Convolution checks distinguish immutable old snapshots from explicit
refresh. NVIDIA checks observe invalidation before synchronization and verify payloads
by download, including scalar/empty/zero-contraction and rejected foreign-session work.
Allocation guards and 48/40/32-byte untracked host/device/error size budgets pass.
A draft error carrying two full versions exceeded its budget at 40 bytes; compact
identity/generation variants restored it before this producer was committed.

Both hardware collections use the same archived 95 source hashes. The initial Intel
attempt was interrupted during compilation and has no successful Rust exit. NVIDIA
finished its 69 Rust/doctest checks and cuDNN Clippy before a disconnected observation
pipe raised `BrokenPipeError`. Those original receipts, logs and runner are retained
in `feec4d6`; their states have not been rewritten. After confirming neither worker
remained alive, a new Intel receipt completes its full suite. NVIDIA's continuation
verifies and inherits the completed commands by hash, then performs only CUDA-only
Clippy and Python checks. Detached observation logs prevent the previous pipe failure.

- [Intel completed checks](../benchmarks/results/2026-10-18-storage-versions-intel-resumed-validation/validation.json).
- [NVIDIA completed checks](../benchmarks/results/2026-10-18-storage-versions-nvidia-resumed-validation/validation.json).
- [Evidence assessment](../benchmarks/results/2026-10-18-storage-versions-assessment.json)
  and [assessment runner](../benchmarks/results/2026-10-18-storage-versions-assessment.py).

The source archives verify against the producer; isolated archive rebuilding is not
claimed for this increment. Subsequent map/documentation maintenance does not change
the tested executable sources. No latency comparison, new model collection, exact
memory peak or performance promotion follows from these functional/resource checks.
Tokens still require retained owners/layouts in a future tape; mutation tracking is
not activation retention, shared owning aliases or automatic parameter refresh.
