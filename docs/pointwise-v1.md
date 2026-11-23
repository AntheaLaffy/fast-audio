# Prepared strided pointwise operators

## Contract and scope

Add independent `cpu::UnaryPlan` / `cpu::BinaryPlan` and corresponding CUDA plans.
They consume borrowed FP32 views, overwrite caller-owned injective output views and
retain geometry rather than input payloads. The public `UnaryOp`, `BinaryOp` and
`PointwiseStrategy` enums identify the immutable operation and traversal strategy.
This declaration precedes implementation. These are forward primitives for a future
tape; derivatives, in-place APIs and tape execution are not supplied by this increment.

A unary plan takes an operation, source layout and destination layout with exactly
equal logical shapes. A binary plan takes an operation, left/right original layouts
and a destination layout. Infer the common shape by aligning input axes on the right:
equal dimensions remain equal; otherwise exactly one dimension must be one and expands
to the other dimension. Missing leading axes act as ones. In particular, broadcasting
one to zero yields zero, not one. The destination must have exactly that common shape,
including every axis. Broadcasting is explicit in preparation; execution accepts only
the exact original input layouts and prepared destination, not alternative broadcasts.

Read inputs may overlap, including each other. Outputs require the layout's existing
injectivity proof. Safe exclusive destination borrows prevent source/output aliasing.
Preserve all input bits and every destination address outside the logical output.
Validate all geometry and CUDA sessions before writing, including for empty outputs.
Rank-zero scalars, zero-sized axes, signed strides, arbitrary read overlap, offsets,
permutations and ranks beyond four are included. There are no implicit materializations,
casts, host fallbacks, payload transfers or output allocations.

## Operations and numerical rules

`BinaryOp` includes `Add`, `Sub`, `Mul` and `Div`, in left/right operand order.
Arithmetic is FP32 under the tested default round-to-nearest environment. Division is
ordinary division, without epsilon/clamping or reciprocal approximation. CUDA uses
explicit round-to-nearest arithmetic rather than fast-math intrinsics.

| Unary operation | Definition |
| --- | --- |
| Neg | Toggle the FP32 sign bit |
| Abs | Clear the FP32 sign bit |
| Square | `x*x` |
| Reciprocal | `1/x` |
| Sqrt | Principal square root |
| Exp | Natural exponential |
| Log | Natural logarithm |
| Relu | Positive zero if `x < 0`, otherwise the original value |
| Sigmoid | Stable logistic: `1/(1+exp(-x))` for nonnegative x; `exp(x)/(1+exp(x))` otherwise |
| Tanh | Hyperbolic tangent |
| Silu | `x*sigmoid(x)`, with the limiting value negative zero at negative infinity |
| Softplus | Stable `max(x,0)+log1p(exp(-abs(x)))` |

Neg/Abs preserve all other bits, including quiet NaN payloads; Relu preserves both
zeros and propagates a NaN rather than suppressing it. Other arithmetic promises NaN
classification rather than payload bits. Infinities, signed zeros and domain failures
follow these definitions; negative finite Sqrt/Log yield NaN, Log of either zero is
negative infinity, and division/reciprocal of signed zero yields signed infinity when
the numerator is finite nonzero. There is no global exception/flush-mode guarantee.
CUDA uses the normal `expf`, `logf`, `log1pf`, `tanhf` and `sqrtf` implementations,
without a cross-library bit-equality promise for transcendental functions.

Independent FP64 forward references use `atol=2e-6, rtol=3e-6` on the finite fixture
domain. Include Exp on [-80,80], small/large finite Sigmoid/Softplus inputs, and positive
Log/Sqrt inputs; separate category/sign/bit tests cover exceptional values. Fixed-plan
repeats must be exact on the same tested backend/stack. An arithmetic error budget is
not a backward, cross-stack determinism or application-fidelity claim.

## Traversal and resources

Prepare geometry once. Remove singleton axes, reverse negative output axes while
adjusting all input/output origins, and order axes by output stride to favor adjacent
writes. Coalesce axes only when every participating input and output stride preserves
the same address map. These reorderings are valid because each output is independent;
they must not be reused for a reduction whose rounding order is observable.

Expose `Empty`, `Contiguous`, `ContiguousRows` and `Strided` strategies, plus coalesced
rank/inner length. A contiguous strategy is a single physically linear traversal,
which may come from logically permuted/reversed views. Contiguous rows include
zero-stride input broadcasts and dense input/output columns. CPU dispatches once per
operation and row, retaining vectorizable dense/broadcast loops without per-element
coordinate division. General rows use checked-bound, injective signed-stride traversal.

CUDA passes scalar/one-axis geometry as kernel parameters, with no device coordinate
array. Additional coalesced outer axes occupy one 32-byte device descriptor each.
The initial general kernel uses 64-bit coordinates and signed offsets; do not narrow
counts or silently limit public layouts to rank four. Select operation/flat mapping
outside the device element loop. These are initial strategies, not measured optimal
dispatch rules. Preserve the existing fixed-graph kernels and numerical behavior;
their model consumers and measured addressing specialization remain valid until
separate equivalence/performance gates justify migration.

Execution allocates no Rust heap, device-global scratch or per-call metadata upload.
Report `host_metadata_bytes` and device `workspace_bytes` separately. Host bytes count
retained dynamic Rust geometry, excluding fixed headers/native control objects;
device workspace counts persistent coordinate descriptors, not caller payloads.
Storage is O(rank), with no element-sized index table. Preparation follows the existing
layout allocator policy; these byte reports do not establish recoverable allocation
for every metadata allocation or an exact process/device peak.

Plans rebind changed buffers with the same layouts. CPU plans may be shared among
threads with independent destinations. CUDA retains the single-session stream, copies
and synchronizes borrowed metadata before preparation returns, and waits for pending
uses before destroying coordinate storage. Asynchronous runtime errors retain the
existing CUDA session contract; geometry rejection does not imply transactional
recovery after an actual device execution error.

## Verification gates

- Enumerate coordinates independently of `Layout::index`, offsets iterators and
  prepared traversal helpers. Check all operations against FP64 references, source
  preservation and untouched output holes. Include mixed-rank broadcasting in both
  operand orders, overlapping reads, negative source/output strides, permutations,
  singleton axes, high ranks, scalars, empties, prime/tail lengths and generated cases.
- Test stable finite extremes and exceptional categories/signs/declared exact bits.
  Check exact same-backend repeats, changed input/output bindings and CPU plan sharing.
- Reject incompatible broadcasts, wrong unary/output axes, unproved mutable layouts,
  changed bindings and foreign sessions before writes, including empty cases.
- Check coalescing/resource strategy boundaries, O(rank) preparation for large logical
  broadcasts, zero warmed Rust allocations and zero device descriptors for flat cases.
- Run portable/Intel/NVIDIA regressions, relevant all-target Clippy and formatting,
  targeted Miri for the unsafe signed-stride loops, and GPU memory/synchronization
  instrumentation when available. Archive raw outcomes and exact producer sources.

No latency claim or performance promotion follows from these functional/resource
gates. Profile and declare matched per-stratum timing work separately before tuning
dispatch or replacing existing graph kernels. General reductions, gather/scatter,
other backward families, the reverse-mode tape and training remain foundation work.

## Verified implementation

Producer `7a637c7` follows declaration `12e1d2a`. Portable/Intel/NVIDIA suites pass
68/70/93 Rust tests and 13/13/14 doctests; both hardware hosts pass nineteen Python
checks. All-target Clippy with warnings denied passes for portable, `mkl,onednn`,
`cuda` and `cudnn`; formatting is checked. Portable/Miri results were observed before
the archived backend collections and are not recollected by the assessment runner.

Independent coordinate enumeration tests all twelve unary and four ordered binary
operations over 79 CPU and 27 GPU geometry cases, with two distinct input/output
bindings per operation/case. The cases include generated signed/padded/permuted maps,
mixed-rank broadcasts in both operand orders, overlapping reads, genuinely nontrivial
rank-five layouts, singleton-heavy ranks, scalar origins and huge empty axes. Finite
outputs pass the declared FP64 `atol=2e-6, rtol=3e-6` gate. Separate tests check stable
extremes, subnormal positive Sigmoid/Softplus at -90, exceptional categories/signs and
declared bit rules. Sources and output holes retain their original bits; repeated
same-backend outputs are exact on the tested stack.

Prepared successful calls allocate zero Rust heap storage in the tested paths. CPU
plans execute across actual host threads with independent destinations. Strategy and
metadata checks cover physical coalescing of permuted/reversed layouts and preparation
above `u32::MAX` logical elements without element-sized index tables. Flat CUDA plans
use zero coordinate workspace; general plans report 32 bytes per coalesced outer axis.
Targeted Miri passes `strided_pointwise_bounds_and_read_aliases_under_miri`, including
negative/gapped traversal and aliased read inputs. Its nightly reports the existing
graph `fetch_update` naming deprecation; stable Clippy checks remain clean.

Compute Sanitizer 2024.1.1.0 runs all four CUDA tests separately under memcheck,
racecheck and synccheck. Every command passes with zero reported errors/hazards.
The tested flat kernel covers a tail beyond `65535*256` elements, exercising grid-stride
reuse. The exact instrumented executable is retained compressed, with binary and
compressed hashes checked against the receipt. Finite/instrumented cases do not prove
all layouts or cross-stack bit repeatability, and racecheck's reported hazard scope
is not a general proof of absence of global races.

Both completed backend collections freeze the same 106 source hashes. The assessment
verifies producer bytes, archives, command logs and the instrumented binary without
rerunning completed commands or replacing outcomes:

- [Intel checks](../benchmarks/results/2026-10-18-pointwise-intel-validation/validation.json).
- [NVIDIA checks](../benchmarks/results/2026-10-18-pointwise-nvidia-validation/validation.json).
- [Evidence assessment](../benchmarks/results/2026-10-18-pointwise-assessment.json)
  and [assessment runner](../benchmarks/results/2026-10-18-pointwise-assessment.py).

Both backend collections completed without retry. This is functional/resource evidence,
not an isolated archive rebuild, fresh model collection, latency comparison, exact
memory peak or performance promotion. CPU pointwise code is the common Rust route in
portable/Intel builds, without a vendor-specific speed claim. Pointwise gradients,
general reductions, the tape and training remain open; existing graph kernels stay
unchanged. The next performance work needs a matched per-stratum baseline before tuning.
