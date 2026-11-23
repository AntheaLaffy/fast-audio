# Prepared reverse-mode differentiation foundation

## Objective and scope

This contract governs the initial native single-device forward/backward/SGD
implementation, following the verified [pointwise local VJPs](pointwise-backward-v1.md),
[reductions](reduction-v1.md), [view adjoints](view-backward-v1.md) and
[retained values](saved-values-v1.md). First-year work remains framework-only.

The first compiled tape must compose all twelve unary operations, Add/Sub/Mul/Div,
Sum/Mean, metadata views/detach and rank-two matmul, with multiple leaves and roots,
branched reuse, tied parameter storage and explicit gradient accumulation. Close a
procedural dense regression and waveform/spectral-shaped tensor loss on CPU/CUDA.
The [prepared matmul prerequisite](matmul-v1.md#verified-implementation) now passes
its frozen functional/resource and independent reference gates.
Convolution/spectral derivatives, batched/vector matmul, broader operators/dtypes,
higher derivatives, dynamic control flow and optimizer/RNG/checkpoint hardening
remain required later capabilities; this tape is not the entire year-one gate.

Public names are `autograd::{Builder, Program, Value}` and `cpu::Tape` /
`cuda::Tape`. The selected API was declared in `862f2b2` before implementation. Existing inference
APIs keep their current contracts and costs; using a training tape is explicit.

## Selected API and execution boundaries

`Builder::new()` returns a checked builder. `leaf(shape, requires_grad)`,
`unary(operation, input)`, `binary(operation, left, right)`,
`reduce(operation, input, axes, keep_dims)` and `matmul(left, right)` append values.
`permute`, `slice`, `broadcast_to`, `reshape` and `detach` append metadata aliases.
`view(input, &layout)` explicitly supplies an absolute layout over the input's storage
root, checked against that root's initialized extent; detach still cuts that use.
`finish(roots)` consumes the builder and returns `Program`; roots are an ordered,
nonempty slice of branded values. `Program` exposes roots, leaves, logical layouts,
shapes and gradient requirements without exposing mutable topology.

`Binding::new(value, Shared<T>)` binds every declared leaf exactly once, including
unused/non-gradient leaves. `cpu::Tape::new(cpu, &program, &bindings)` and
`cuda::Tape::new(&cuda, &program, &bindings)` use default options. `with_options`
accepts `PrepareOptions { seed_layouts, view_index_limit_bytes }`: seed layouts are
fixed per root and default to contiguous logical shapes. The default combined
retained view-projection index budget is 16 MiB; analytical projections consume no
index budget. The same ceiling bounds already retained indices plus the current
projection's transient index preparation. Caller buffers remain explicit.

`forward()` executes the prepared program. `output(root)` returns a guarded
`ReadValue<T>` exposing storage, logical layout and a borrowed CPU/device view;
the reader borrows the tape and cannot survive a mutable execution call.
`backward(&[Seed::new(root, &tensor, &layout)], mode)` validates all seeds in root
order before writes. `backward_scalar(mode)` supplies the prepared one only for a
single scalar root with its default scalar seed layout.

`synchronize()` is an explicit completion/error boundary, a no-op on CPU. It adds no
hidden synchronization to forward/backward/update; a reported execution failure
invalidates forward/gradient/update state under the existing CUDA limits.

`gradient_status(leaf)` returns `Unused`, `Uncomputed`, `Valid` or `Invalid`.
`gradient(leaf)` returns `None` for an unused/non-gradient owner or a borrowed dense
canonical gradient when valid; uncomputed/invalid gradients are errors. A
differentiable leaf alias queries the combined gradient of its bound owner when any
differentiable occurrence of that owner is gradient-reachable. Non-gradient aliases
remain unused. Intermediate/view-gradient queries are unsupported.

`zero_grad()` clears active leaf gradients and makes accumulation recoverable.
`sgd(&[leaves], learning_rate)` applies plain FP32 updates with a finite nonnegative
rate passed as a scalar kernel argument. It deduplicates tied owners. A successful
backward since the latest forward is required; a fresh forward or zeroing preserves
gradient bytes where specified but does not make an old gradient update-ready.
An empty selection is a no-op after rate validation. Preflight write-availability
checks do not advance versions or promise a transactional lease; execution failure
invalidates gradient/update state and may leave earlier parameter writes visible.

`resources()` reports prepared activation, canonical gradient, logical edge scratch,
scalar seed, operator workspace and view-index payload bytes, operator host metadata,
and maximum static GPU shared bytes per block. Caller leaf/seed tensors, fixed control
objects, allocator overhead, vendor internals and process peaks remain separate.
Operator host metadata does not include graph/ownership bookkeeping or the caller's
shared `Program`; the query is not a complete host allocator footprint.
The resource query is not a performance-acceptance claim. V1 retains required
activations and optimizes identity gradient projections without hidden view copies.

## Values, ownership and aliasing

- Brand values with a checked builder identity and index. Reject foreign/unknown
  values and exhausted identities without wrapping. Validate operations before
  appending nodes so failed construction cannot leave partial edges.
- Declare leaf shapes and gradient requirements separately from backend binding.
  Preparation binds exact-shape `Shared<Tensor>` or `Shared<DeviceTensor>` owners;
  CUDA validates every original session, including empty/non-differentiated leaves.
  Keep the bound owner identities fixed for an execution instance. Feed new data
  through explicit writes/transfers; preparing a new instance permits new owners.
- Represent computed storage separately from logical values. Permute, slice,
  broadcast and permitted no-copy reshape compose `Layout` metadata over the same
  storage root. Reject a reshape requiring copying unless an explicit materializing
  operation is requested. A view is not an implicit forward copy.
- An active view's gradient is the consumer's logical input gradient projected to
  its dense storage-root gradient with `ViewBackward`. Preserve the checked overlap
  index budget and analytical broadcasts. Do not write a gradient through an
  overlapping mutable view. Intermediate view gradients are not retained by default;
  querying a supported leaf gradient must not imply arbitrary intermediate retention.
- `detach` shares forward storage and layout while cutting only that use's gradient
  path. Propagate gradient requirements through actual edges, not storage identity.
- Deduplicate differentiable leaf gradient buffers by bound `StorageId`. Repeated
  operands and separate leaf handles bound to the same owner must contribute to one
  physical parameter gradient. A detached/non-gradient alias can still be a saved
  coefficient; it does not become differentiable because another alias is trainable.

## Preparation and forward execution

`Program` is a checked DAG with fixed shapes and declared output roots. Prepare only
forward-reachable nodes and gradient-reachable edges. Multiple/duplicate roots have
explicit seed order. Support scalar/empty values and arbitrary-rank existing tensor
operations; matmul keeps its separately declared rank-two capability.

Prepare operator plans, storage-root activation/gradient buffers, scratch, logical
input-gradient staging where a view needs projection, seed buffers and per-buffer
initialization flags. Report them separately. V1 may retain all required computed
storage in a bounded prepared arena; liveness reuse/recomputation requires later
equivalence and resource gates. Aliased views do not count as separate payloads.

Forward executes nodes in stable topological order and captures versions only for
payloads queried by active local derivatives. Neg/Add/Sub and payload-free reduction/
view derivatives do not save an unnecessary input. Empty/no-contribution needs follow
the local plan's effective query. Saved values retain owners/layouts without copying
their payload; computed storage must outlive every queued consumer.
Capture leaf versions before execution and recheck them after all nodes, rejecting
interleaved writes even when no local VJP needs the original payload. This is
optimistic validation, not a long-lived transaction lease. Cached computed outputs
represent the successful forward epoch; later external writes do not refresh them.

A forward attempt invalidates the previous active forward epoch. Only successful
completion exposes current outputs and a backward-ready epoch. A new forward refreshes
saved versions; it preserves explicitly accumulated leaf gradients. Output readers
cannot outlive the borrow needed to rerun the instance or update its state.

## Backward and accumulation

Backward requires a successful forward and exact-shape seeds for the declared roots.
An implicit seed of one is only available for a scalar root. Prepare its storage
ahead of execution. Validate root/seed identities, shapes, layouts and sessions
before gradient writes; reject an unsupported root/gradient request explicitly.

Initialize internal gradient scratch afresh for every backward. Insert seeds in root
order, then traverse nodes in reverse topological order and operands in fixed order.
Track first writes by canonical gradient buffer, including tied leaves. Local plans
use overwrite for a first internal contribution and accumulate for later ones;
`GradientMode` controls preservation of existing leaf gradients between backward
calls. Zero-contribution accumulation retains the local primitive's bit rules.
Structurally empty local contributions initialize an uncomputed canonical gradient
to positive zeros or preserve an already initialized gradient without arithmetic.
They require neither logical gradient scratch nor view-projection indices, even
when the original read view overlaps. Numerical zeros on nonempty paths retain the
primitive's ordinary arithmetic rules.

Apply the same rules when both binary operands are the same value, when two views
overlap, when roots share a storage source, and when one root is an ancestor of
another. Do not overwrite a pending branch contribution or add an old internal
gradient during a repeated backward. Retained saved values permit repeated backward
on the same successful forward until a new forward or invalidating update.

Distinguish a statically unreachable leaf gradient from a computed zero gradient.
Expose gradient validity and explicit zeroing; never substitute a missing derivative
with zeros. Preserve a separate capability/error for unsupported differentiation.

## Mutation, concurrency and failures

Precheck needed saved versions before execution and acquire checked read guards
again around each use, retaining them through the enqueued primitive calls. This
avoids unchecked reads after an external alias writes. It does not establish one
transactional lease over every value for the entire pass. A concurrent write may
be detected after earlier gradient work; on execution failure mark all affected
gradient state invalid and prohibit reading/updating from partial results.

After a failed accumulating pass, require explicit zeroing or a successful overwrite
pass before accepting further accumulation. Deterministic argument failures detected
before execution preserve existing gradient bytes and their previously valid state.
Native asynchronous failures retain the existing device error limits; no rollback
or recoverable-device guarantee is implied.

One mutable tape instance serializes execution and owns scratch. Independent CPU
instances can run on host threads; CUDA instances preserve their original session/
host-thread restrictions. External leaf writers use the existing nonblocking shared
guards. A saved guard is host ownership/version evidence, not a device event; retain
same-stream ordering and existing synchronized destruction until explicit event
ownership is implemented.

The versioned-save design is consistent with the relevant
[PyTorch mechanism](https://docs.pytorch.org/docs/2.14/notes/autograd.html#in-place-correctness-checks),
but this framework declares its own failure/accumulation and lifetime behavior.

## Minimal parameter update

Provide explicit SGD for a caller-selected set of differentiable leaves. Deduplicate
tied owners so a parameter is updated once. Validate gradient state, original forward
parameter versions, shapes, sessions, finite update configuration and borrow access
before starting writes. A parameter mutation invalidates the current forward epoch;
packed parameter caches must refresh or invalidate rather than serve stale values.

Plain SGD is `p = p - learning_rate * grad` in the declared FP32 order. No implicit
weight decay, momentum, clipping, precision conversion or gradient clearing. A
successful update followed by a new forward must use the new parameter payload.
Do not claim atomic multi-parameter rollback on a device failure. Define partial
failure validity before exposing the implementation.

## Verification and performance gates

- Independently evaluate composed analytic FP64 forward/VJP objectives and central
  differences at leaves; match pinned Torch gradients and complete parameter updates.
  Include all local operation families, branches, duplicate roots/operands, ancestor
  roots, detach, tied leaves, overlapping views and changed inputs/parameters.
- Check unused versus zero gradients, empty broadcasts/contractions, scalar seeds,
  repeated backward and cross-forward accumulation. Retain local non-finite/tail and
  exact rounding contracts without replacing them with a graph-level loose tolerance.
- Reject foreign values/bindings/seeds/sessions and stale saves; test external writes,
  guard conflicts, failed construction/execution and invalid-gradient recovery. Verify
  that incomplete/partial gradients cannot reach SGD.
- Match a procedural training trajectory over repeated updates on both hardware
  paths, including real tied parameters and noncontiguous views. Keep full outputs,
  gradients, parameters, source bits and resource reports.
- Measure preparation, first call, warmed forward/backward/update, transfers, Rust
  allocations, explicit persistent/scratch memory and vendor/process observations
  separately. Prepared framework execution must avoid new hot Rust heap buffers and
  metadata uploads; vendor internals need their own measured scope.
- Freeze full regressions, targeted Miri, GPU tools, exact sources/executables/logs
  before capability claims. Latency against compatible compiled/captured PyTorch
  modes is a separate campaign. Existing inference/model champions stay unchanged.

## Verified initial implementation

The [frozen assessment](../benchmarks/results/2026-10-20-autograd-assessment.json)
verifies producer `160070d3332fd90ce88b5e09928d5289a862331a` against the declared API
and campaign. All three 155-file source archives match that committed producer;
release test executables, command logs and losslessly compressed raw exports are
hash-verified. All recorded collector/command processes are confirmed terminal.

| Context | Rust tests per debug/release build | Doctests per build | Python guards | Torch records across profiles |
| --- | --- | --- | --- | --- |
| Portable CPU | 103 | 16 | 41 | 114 |
| Intel reference/MKL + oneDNN regressions | 105 | 16 | 41 | 228 |
| NVIDIA reference CPU/CUDA + cuDNN regressions | 152 | 17 | 41 | 228 |

All-target Clippy, formatting, the targeted signed/overlapping-view Miri check and
24 separate GPU instrumentation commands (eight tests under each of memcheck,
racecheck and synccheck) pass. The pinned Torch comparison covers 32 raw files and
570 records under the unchanged precision/tolerance policy below.

The public builder/program and CPU/CUDA tapes now implement the selected operation
surface, metadata aliases, selective per-node saves, canonical tied gradients,
ordered seeds/reverse edges, guarded outputs and explicit plain SGD. Private shared
writer preflight does not advance versions; actual updates validate the saved version
under each write guard. Immutable prepared scalar seeds survive execution unwinding.
Existing inference graphs and their measured routes are retained.

Eleven CPU/MKL integration tests and eight RTX 3060 CUDA tests pass under both profiles.
Sixteen shared composition fixtures exercise all twelve unary and four binary
operations under two payload bindings, independent analytic FP64 and finite-difference
controls, both gradient modes and exact repeats. Dense native training runs eight
updates against an independently evolving FP64 control. Additional cases check tied/
detached/overlapping branches, duplicate/ancestor roots, signed seeds, stale/foreign/
borrowed owners, real CPU threads, strided matrix VJPs, explicit byte budgets, a
six-billion-element virtual root, zero-contribution bits and SGD grid reuse.

Four unit checks delegate arithmetic to real CPU plans and deterministically mutate
external owners between forward nodes, after partial backward/update work, or trigger
execution unwinding. They verify snapshot rejection, invalid-gradient barriers,
per-writer version rechecks and both overwrite/zero recovery.
The CPU API example and guarded-output compile-fail check pass.

Two additional procedural objectives exercise `[2, 2, 17]` multichannel waveforms and
`[2, 2, 3, 5, 2]` real/imaginary-pair spectral storage. Channel/bin gains and biases train
for eight updates through broadcast, nonlinear/power operations, reductions and reversed/
permuted prediction roots. Independent physical-coordinate FP64 equations and all-parameter
central differences check their derivatives. Full prediction, loss, gradient and parameter
payloads match independently evolving pinned Torch controls. The spectral
fixture exercises tensor composition; it does not implement or claim FFT/STFT derivatives.

`FAST_AUDIO_AUTOGRAD_EXPORT_DIR` enables exclusive raw-bit exports outside hot allocation
guards. `tools/check_autograd.py` requires complete operation/binding/backend/profile scope,
fixed source/seed/shape catalogues and ordered eight-step trajectories. Six checker tests
use separate scalar equations and reject corrupted predictions/gradients/updates, schema
changes, missing cases, duplicate records and reordered steps.

This closes the functional/resource gate for the declared initial tape and plain SGD
surface. Runtime memory optimization and performance against compatible compiled/captured
Torch modes remain separate milestones. Liveness arenas, fusion/replay, optimizer state,
checkpoint/resume and broader operator/dtype coverage remain foundation work. Recovery is
audited without an isolated rebuild; finite fixtures and targeted instrumentation are
not universal proofs, and racecheck does not establish absence of every global-memory
race. No application, complete training stack, exact memory peak or speedup is claimed.

## Frozen functional campaign

Freeze the implementation and reference-checking sources before collection. Each backend
and profile must export 32 composition records (16 operations, two changed bindings), eight
dense SGD steps, sixteen audio-shaped SGD steps (eight per fixture) and one tied/overlap/
detach/multiple-root record: 57 records. Portable checks the reference CPU; Intel checks
reference and MKL; NVIDIA checks reference CPU and CUDA. Debug/release therefore require
114/228/228 raw records respectively, with CPU Torch `2.14.1+cpu` and NVIDIA-host Torch
`2.14.0+cu130`, highest matmul precision, TF32 disabled and unchanged `2e-5` absolute/relative
tolerances. Do not rebase reference parameters to native parameters between updates.

Require all Rust regressions and doctests under both profiles, all-target Clippy (including
CUDA-only), formatting, all Python guards, the targeted signed-view Miri test and separate
memcheck/racecheck/synccheck commands for every one of the eight CUDA tape tests. Preserve
the exact committed source inventory, terminal command logs, release test executables,
raw export bytes and all hashes. Verify process termination and recoverability separately
from navigation fingerprints. These are functional/resource gates, with no timing promotion.
