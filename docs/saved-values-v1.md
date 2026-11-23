# Retained values and view-gradient contract

## Purpose

A [storage version](storage-versions-v1.md) detects mutation but does not keep an
activation alive. Add opt-in shared ownership and retained values for host/device
storage. Ordinary inference tensors and borrowed views keep their existing APIs and
representation. This is storage infrastructure; a differentiation tape and executable
view-backward kernels remain separate implementation work.
The initial contract was declared in `45e980d` before implementation.

## Shared ownership and access

`Shared<T>` contains one reference-counted `Tracked<T>` protected by a read/write
lock. Cloning a shared owner preserves storage identity and generation and copies no
payload. `from_tracked` preserves an existing tracking lifetime. `try_unwrap` returns
that tracked owner only when it has no other strong owners, including retained values;
otherwise it returns the shared handle without consuming the underlying storage.

Read/write acquisition is nonblocking. Concurrent readers are allowed; a conflicting
writer or reader returns `StorageBorrowConflict`, rather than blocking or deadlocking
the caller. A successful write guard advances the generation exactly once before
exposing any mutable payload access. Reborrowing or repeatedly dereferencing that
guard does not advance it again. A denied acquisition or exhausted generation exposes
no writer and leaves the generation unchanged. Existing saved-version checks and
exhaustion rules still apply.
A writable guard exposes no current-version getter. Saves and current versions need
read access, so a new snapshot key cannot be issued midway through a writable borrow.

Guards borrow their shared owner. References and host/device views derived from them
cannot outlive that borrow. No public mutable reference to tracking state, raw pointer,
unchecked lock access or reference-count-dependent mutation shortcut is introduced.
`Shared<Tensor>` can be shared across host threads; `Shared<DeviceTensor>` inherits
the device tensor's original non-Send/non-Sync session restriction.

Write unwinding leaves an already advanced version and possibly partially updated
data. The sealed owned storage types remain valid Rust values, so poisoned lock guards
are recovered for inspection and future explicit writes. This does not roll back
updates or certify native execution success. Old retained values fail their version
checks; a fresh read observes the current possibly partial value. No generic external
interior-mutable type can opt into this recovery policy.

## Retained values

Prepare immutable layout metadata once in `Arc<Layout>`. `Shared::save` validates that
layout against initialized storage while holding a read guard, and captures a strong
owner, the layout and the current version together. Broadcast/overlapping read layouts
are allowed. Saving and cloning a retained value share payload/layout ownership;
they perform no deep copy, device transfer or warmed Rust heap allocation.
A saved slice retains its entire owning allocation. A compact snapshot remains an
explicit materialization rather than an automatic copy.

`SavedValue::read` acquires a read guard and checks the captured identity/generation
before exposing storage. Its typed `view` uses the captured layout. No lock is held
between reads, so a later write may invalidate a retained value instead of preventing
all parameter/activation updates for its entire saved lifetime. During an active read,
conflicting writes fail without advancing the generation. Invalid layouts and failed
version/borrow checks expose no payload reference.

Dropping the original facade, tracked handle or other aliases does not release saved
storage. The last shared/saved owner releases it after derived guards end. CUDA keeps
its existing session, stream ordering and synchronous allocation destruction rules;
a version check is not a completion event or permission for asynchronous pool reuse.
There is no implicit activation snapshot or automatic parameter repacking.

Reference-count and lock operations occur at ownership/borrow boundaries, not per
element. Prepared consumers can reuse layouts and hold guards across execution. Their
actual latency cost needs later measurement; allocation-free behavior is not a speed
claim, and untracked inference does not acquire these locks.
Initial owner/layout reference-count allocations use Rust's standard allocation-failure
policy; recoverable metadata allocation and custom/pool allocators are not provided here.

## View gradients required by a future tape

For a dense owning storage vector `x` and a validated layout address function `p`,
a read view is `y[j] = x[p(j)]`, with `j` in logical row-major order. Its adjoint is:

```text
dx[i] = sum(dy[j] for all j with p(j) == i)
```

Permutation and negative strides reverse that indexing; slicing leaves unselected
positions at zero. Broadcasting and every other overlapping read layout must sum
repeated addresses, rather than overwrite one contribution. A gradient accumulator
adds this result to its previous value; overwrite mode initializes every owning-storage
element, including untouched elements. An empty view contributes zero, including when
its owner is nonempty. Scalars follow the same one-element rule.

The upstream gradient must have the exact logical view shape. The base gradient has
the exact dense owning shape, with checked bounds and an injective destination. A
future implementation must specify accumulation order, numerical/repeatability policy
and resource limits before choosing scatter atomics or deterministic gather/reduction.
Validate it with an independent coordinate oracle, inner-product adjoint identities
and finite differences; retention tests alone do not establish gradient correctness.
The subsequent [view-backward implementation](view-backward-v1.md) defines those
execution rules and its own checks separately from retained-value evidence.

View backward usually needs the mapping and gradient destination identity, not the
original payload. A future tape must distinguish metadata-only dependencies from
saved-data dependencies, retaining and version-checking payload only when a derivative
actually needs it. This avoids forcing unnecessary activation retention or rejecting
mathematically valid updates. Other backward families, accumulation scheduling and
parameter update contracts still need their own explicit requirements.
The tape must retain the original base shape and logical value identity even when it
omits payload retention; replacing an owner must not redirect an old gradient to a
differently shaped value.

## Verification gates

- Keep a host/device value and its exact layout usable after other owners are dropped;
  distinguish shallow shared aliases from deep tensor copies and unique extraction.
- Check mutation/replacement, repeated reads, failed/held write guards, read conflicts,
  unwinding/poison recovery and immutable captured metadata. Exercise simultaneous
  host readers and a real cross-thread writer without timing-dependent assertions.
- Check scalar/empty/sliced/reversed/broadcast and invalid layouts, source versions,
  ordinary inference size budgets, and allocation-free capture/clone/borrow after
  preparation. Cover counter exhaustion before exposing a mutable guard.
- Use actual prepared CPU and CUDA consumers, including queued writes, same/foreign
  sessions, exact payload checks and dropping external device ownership.
- Require borrow/retention compile-fail examples and the existing backend regressions,
  format and Clippy gates. Do not claim autodiff, a view backward implementation,
  asynchronous storage reuse, latency gains or completed training from this increment.

## Verified implementation

Producer `490737b` follows declaration `45e980d`. Portable/Intel/NVIDIA suites pass
56/58/73 Rust tests. Portable and Intel pass eleven doctests (three runtime and eight
compile-fail cases); NVIDIA passes twelve, adding the negative device-owner Send case.
Both hardware hosts pass nineteen Python checks. All-target Clippy with warnings
denied passes for portable, `mkl,onednn`, `cuda` and `cudnn`; formatting is checked.

The added cases verify payload addresses and retention of immutable prepared layouts,
alias versus deep-copy identity, unique extraction, partial writes and poison recovery.
Barriers coordinate actual shared host readers and a writer without timing-based
assertions. Counter exhaustion still permits the last valid write guard exactly once.
Prepared CPU GEMM/copies and CUDA copies consume retained values after external owners
are dropped. CUDA also exercises a last source owner released with work potentially
queued, stale saves after queued/rejected writes, and scalar/empty views.

Prepared capture/clone/borrow and tested execution loops perform zero warmed Rust heap
allocations. Existing 48/40/32-byte ordinary host/device/error budgets still pass.
These observations exclude vendor/OS internals and do not measure lock/refcount latency.
Both hosts freeze the same 96 source hashes; source archives, logs and producer bytes
verify without replacing outcomes or rerunning completed commands.

- [Intel checks](../benchmarks/results/2026-10-18-saved-values-intel-validation/validation.json).
- [NVIDIA checks](../benchmarks/results/2026-10-18-saved-values-nvidia-validation/validation.json).
- [Evidence assessment](../benchmarks/results/2026-10-18-saved-values-assessment.json)
  and [assessment runner](../benchmarks/results/2026-10-18-saved-values-assessment.py).

No isolated archive rebuilding, new model collection, latency comparison, exact memory
peak or performance promotion is claimed. The view-adjoint section is a specification;
no gradient/finite-difference gate or reverse-mode execution is implemented by these
retention checks. The subsequent view-backward contract tracks that primitive;
general strided math and the differentiation tape remain foundation work.
