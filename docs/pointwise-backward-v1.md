# Prepared pointwise local gradients and broadcast contraction

## Contract and dependencies

`cpu::UnaryBackward` / `cuda::UnaryBackward` implement all twelve existing unary
operations and `BinaryBackward` on both backends for Add/Sub/Mul/Div. A public
`BinaryOperand::{Left, Right}` selects the differentiated binary operand. This
contract was declared before implementation. These are explicit local vector-Jacobian
products, not a reverse-mode tape or complete training API.

Unary preparation takes the operation, original input layout, upstream layout and
gradient layout. All logical shapes must match. Binary preparation takes operation,
selected operand, original left/right layouts, upstream layout and gradient layout.
Use the same right-aligned broadcast inference as forward, including one-to-zero
broadcasts. Upstream must have that exact common shape; the gradient must have the
selected original operand's exact shape. Require injective gradient outputs. Read
inputs/upstream may overlap and have signed strides, offsets, padding or permutations.
Include scalars, empty axes, high ranks and changed payloads with fixed exact layouts.

Runtime takes optional forward input views, a required upstream view, caller-owned
gradient view and `GradientMode`. `UnaryBackward::input_required()` and
`BinaryBackward::input_dependencies()` expose effective payload needs:

| Operation / selected operand | Required forward payload |
| --- | --- |
| Unary Neg | None |
| Other unary operations | Original input |
| Binary Add/Sub, either operand | None |
| Binary Mul, left / right | Right / left input respectively |
| Binary Div, left / right | Right / both inputs respectively |

If no logical contribution exists (empty target or empty broadcast domain), all
forward payload needs are false. Missing required inputs fail before any write.
Every supplied view still requires the original exact layout and CUDA session,
even if unused or empty. Validate upstream/output geometry and sessions before
empty returns as well. Preserve all read bits and every gradient-output hole.
No implicit casts, materialization, host fallback, output allocation or in-place aliasing.

Plans retain geometry, not payloads. Standalone calls differentiate supplied values;
they cannot establish that those values are the original forward execution's values.
A future tape must save only the queried dependencies with `SavedValue` and obtain
checked read guards before calling the primitive. Mutation rejection then happens
before gradient writes. No forward output is required by this version.

## FP32 local equations

Let `d` be upstream, `x` unary input, `a,b` binary inputs and `neg` an FP32 sign-bit
toggle. Evaluate the explicitly parenthesized operations in FP32 without FMA or
fast-math substitution. Division is rounded division, without epsilon/clamping.
Differentiate the analytic forward equations, not a discrete floating-point mapping;
an accurately computed gradient may be nonzero after a forward output rounds to
saturation. Outside differentiable domains, expressions define arithmetic behavior,
not mathematical validity of a derivative.

Define stable `s(x)` as the existing sign-split sigmoid. With
`q = exp(-abs(x))`, `t = 1+q`, define `h(x) = (q/t)/t`. For Tanh use
`u = (2*q)/(1+q*q)` and derivative factor `u*u`. These avoid subtracting a saturated
sigmoid/tanh output. Include separate positive-tail tests beyond the finite-difference
fixtures so an absolute tolerance cannot silently admit zero gradients there.

| Unary operation | Local contribution |
| --- | --- |
| Neg | `neg(d)`; preserve other bits |
| Abs | `d` for x>0, `neg(d)` for x<0, positive zero at either zero; canonical quiet NaN for NaN x |
| Square | `(2*x)*d` |
| Reciprocal | `neg((d/x)/x)` |
| Sqrt | `d/(2*sqrt(x))` |
| Exp | `exp(x)*d` |
| Log | `d/x` |
| Relu | `d` for x>0; positive zero for x<=0; canonical quiet NaN for NaN x |
| Sigmoid | `h(x)*d` |
| Tanh | `(u*u)*d` |
| Silu | `(s(x)+x*h(x))*d`; use factors 1 / negative zero at positive / negative infinity |
| Softplus | `s(x)*d` |

Abs/Relu inactive branches mask upstream even if it is NaN/infinite. Other listed
arithmetic propagates exceptional categories according to its FP32 expression.
For example Log's expression can be finite for a negative input although forward
Log is outside its real domain. NaN arithmetic does not promise payload bits; the
explicit canonical NaN above is `0x7fc00000`. No global exception/flush-mode guarantee.

| Binary operation | Left contribution | Right contribution |
| --- | --- | --- |
| Add | `d` | `d` |
| Sub | `d` | `neg(d)` |
| Mul | `d*b` | `d*a` |
| Div | `d/b` | `neg((d/b)*(a/b))` |

The two quotients in right-Div avoid constructing `b*b`; they do not guarantee
avoidance of every intermediate overflow for arbitrary input magnitudes.

## Contraction, aliasing and rounding

For each selected operand coordinate, sum contributions over missing leading axes
and expanded size-one axes of the common broadcast shape. Use `ReduceSpec` Sum
shape/axis rules and the existing 32/256-lane ordered tree. Canonical reduced axes
are in original common-axis order, last reduced axis fastest. Coalescing must preserve
every participating read address and reduced sequence; do not reorder/reverse it
to favor writes. Fuse local arithmetic with contraction, with no element-sized
intermediate gradient or index table. One-contribution routes store the local value
without adding padding zero.

Overwrite writes every logical operand-gradient coordinate. Accumulate adds each
completed contraction once to its previous value. With zero contributions, overwrite
writes positive zero and accumulate preserves every bit without addition. Empty
gradient layouts perform no payload work. For overlapping/broadcast original input
storage, compose the logical operand gradient with `ViewBackward`; never write into
an overlapping gradient destination. Separate operand plans can accumulate into one
base storage gradient through that checked composition when forward inputs alias.

CPU and CUDA use the same reduction order, with no floating-point atomics. Fixed-plan
repeats are exact on the tested stack; transcendental local equations do not promise
cross-library bits. Preserve existing forward/reduction contracts and old graph/model
consumers until separate equivalence/performance gates justify replacing them.

## Resources and verification

Prepare immutable O(rank) metadata. Execution allocates no Rust heap/device-global
scratch and uploads no metadata. CPU needs at most 256 FP32 lane accumulators;
report retained dynamic host bytes, persistent device descriptor bytes and actual
static shared memory separately. Exclude fixed control headers, caller data and
allocator/driver/process overhead. Geometry allocations retain the existing layout
allocator policy; these reports are not recoverable-allocation or exact-peak claims.
CPU plans may share immutable metadata across threads with independent destinations.
CPU plans preselect operation/operand and Empty/Copy/Tree32/Tree256 execution at
construction. Scalar/row Copy stays separate from contraction bodies. The
[qualified native CPU experiment](cpu-prepared-backward-results-v1.md) records its
one-pointer fixed-header cost separately from these unchanged dynamic queries;
the mathematical and layout contracts above remain unchanged.
CUDA retains its original session/stream, completes borrowed descriptor uploads before
returning, and waits for queued uses before destruction. Existing asynchronous-error
semantics remain; geometry rejection does not make device failures transactional.

Before capability claims, independently enumerate logical coordinates and compare
every operation/operand to FP64 VJP equations and central differences of the analytic
forward objective (`step=1e-4`, `atol=rtol=2e-5` on bounded nonsingular fixtures).
Cover scalar/channel/outer/mixed-rank/empty broadcasts, both operands, signed/gapped/
permuted/high-rank/overlapping reads and outputs, changed bindings and both modes.
Reject unproved mutable maps, wrong equal-size shapes, missing dependencies and
foreign sessions before writes. Check source/hole bits, exact repeats, saved-version
integration, payload-free calls, view-gradient composition and real plan sharing.

Separately verify nonsmooth choices, singular/non-finite categories, stable extreme
tails, exact arithmetic/tree cancellation and zero-contribution accumulation bits.
Exercise warp/block/grid reuse and large logical preparation without element-sized
tables. Run portable/Intel/NVIDIA debug/release regressions, existing Python checks,
relevant Clippy/format, targeted Miri and separate GPU memcheck/racecheck/synccheck
commands. Preserve actual source/executable/log identities and every failed outcome.
Matched performance characterization and the complete native training foundation
remain separate open gates.

## Verified implementation

CPU and CUDA implementations now provide every declared unary/operand local VJP.
The shared CPU/CUDA suite contains 524 operation/geometry combinations with two
payload bindings, independent Cartesian FP64 VJPs, central differences at selected
original logical input coordinates, both gradient modes, complete source/hole checks
and exact repeats.
Additional checks exercise stable tails, singular/nonsmooth/bit rules, cancellation
trees, empty accumulation, dependency/layout/session rejection, bounded large
preparation, saved versions and composition of aliased forward inputs. CPU unary and
binary plans are actually shared across host threads; GPU element/warp/block grids
reuse work across their maximum grid boundary.

The [initial 496-case assessment](../benchmarks/results/2026-10-19-pointwise-backward-assessment.json)
retains the complete producer `40c928e` evidence. Coverage review added 28 cases
coupling signed/read-overlap upstream gradients with permuted/gapped gradient
destinations, including both contraction tree sizes, and extended the Miri fixture.
The [expanded assessment](../benchmarks/results/2026-10-19-pointwise-backward-layout-assessment.json)
verifies producer `50b16b8`: only the declared test/map files differ from the
predecessor source inventory. Native implementation and numerical gates are unchanged.

Portable/Intel/NVIDIA debug and release builds each pass 81/83/116 Rust tests plus
14/14/15 doctests. Every platform passes thirty Python checks, relevant Clippy and
formatting. Extended targeted Miri passes; all six GPU tests pass separately under
memcheck/racecheck/synccheck, with eighteen zero-error/hazard commands. Three 129-file
source archives match the expanded commit; exact compressed test executables and
all command logs pass the independent audit. All collecting processes and recorded
commands are terminal. The original 496-case artifacts remain immutable.

No relative latency, optimization promotion, isolated archive rebuild, exact memory
peak, reverse-mode tape or complete native training is established by these checks.
