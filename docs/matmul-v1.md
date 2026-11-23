# Prepared strided matrix multiplication and local derivatives

## Contract declared before implementation

Add `cpu::MatmulPlan` / `cuda::MatmulPlan` for rank-two FP32 borrowed layouts and
`MatmulBackward` for either `BinaryOperand`. This closes the dense local primitive
needed by the [native training tape](autodiff-v1.md). Batched/vector matmul and other
dtypes remain published foundation gaps; old dense-owned matmul/graph consumers
retain their existing ABI until separate equivalence/performance gates justify reuse.

For `A[M,K]` and `B[K,N]`, forward writes exact-shape `Y[M,N]`. Local VJPs are
`dA = dY @ B^T` and `dB = A^T @ dY`. Do not retain/physically transpose parameters
just to compute these derivatives. Read layouts may have offsets, signed strides,
padding, permutation or repeated addresses; every output/gradient needs a proved
injective layout. No implicit shape promotion, transfers, casts, output allocation
or in-place aliasing. Different logical shapes with equal element counts fail.

CPU preparation takes `Cpu` and three exact layouts; CUDA preparation takes `Cuda`
and the same layouts. Forward `run` always overwrites. Backward preparation takes
backend, operand, original A/B layouts, upstream and gradient layouts; `run` takes
optional A/B views, required upstream/destination views and `GradientMode`.
The gradient shape matches the selected original operand exactly. Expose layout,
operand/dependency, route, host-metadata/workspace and GPU static-shared-memory
queries. CPU execution takes mutable plan access for its bounded packing scratch.

`MatmulBackward::input_dependencies()` needs only B for dA and only A for dB when
contributions exist. No forward output is required. Empty targets or zero-length
local contractions need neither forward payload. All supplied views, upstream and
destination still require exact original geometry and CUDA session before any write
or empty return. A future tape obtains checked saved reads; standalone calls operate
on caller-supplied values and cannot establish the original forward version.

## Empty, accumulation and numerical policy

Empty outputs do no payload work. Zero contraction with a nonempty destination writes
positive zeros in overwrite mode and preserves every bit in accumulate mode, without
addition. The local contraction is N for dA and M for dB; this can be zero even when
the selected original operand is nonempty. Preserve source/gradient holes and all
read bits. For overlapping forward storage, project logical gradients through the
existing `ViewBackward` rather than writing overlapping mutable gradients.

Nonempty matmul uses FP32 inputs/accumulation and finite-input numerical gates.
Keep strict CUDA precision; do not silently enable TF32 or reduced precision.
Vendor FMA/reassociation and blocked accumulation need not match reference bits.
`Accumulate` mathematically adds the matmul result to existing gradients; direct
vendor GEMM may use beta=1, so no separate post-rounded-add order is promised.
This differs from the fixed-tree pointwise/reduction contracts. Overwrite must not
depend on prior selected destination values, including a NaN-initialized destination.

Check same-plan repeated results exactly on the tested stack, without claiming
cross-library/hardware/SDK bits or universal vendor repeatability. NaN/infinite
ordinary products do not gain a new payload/category guarantee from this version;
empty/zero-contraction bit rules remain exact.

## Routes and bounded resources

Publish `MatmulStrategy` with empty, zero, portable reference, direct MKL, tiled
packed MKL, direct cuBLAS and strided CUDA routes. Select during preparation:

1. Use vendor transpose and leading-dimension parameters for compatible positive
   row/column layouts, including padded leading dimensions and singleton axes with
   irrelevant strides. A column-compatible destination can use the transposed
   equation with swapped operands. Preserve its original exact binding geometry.
2. Direct vendor dimensions/leading dimensions must fit its integer ABI. Other
   valid CPU MKL layouts use bounded packing, not full-matrix transposes. Pack at
   most a 32x128 A tile, 128x64 B tile and 32x64 output tile: at most 57,344 FP32
   payload bytes. Smaller extents use smaller prepared tiles. Default explicit
   workspace limit is 64 KiB; provide CPU preparation with a caller-selected limit
   and reject insufficient space before execution. Portable reference needs no
   packing workspace. Large logical/read-overlap shapes must not create element
   index tables or matrix-sized preparation buffers.
3. Other valid CUDA layouts use a 16x16 shared-memory tiled kernel with signed
   64-bit addresses, bounded grid reuse and unique destination writes. Handle
   zero contractions separately. Descriptors fit kernel parameters; do not upload
   metadata or allocate explicit device-global scratch during execution.

The initial routing/tiles are correctness/resource candidates, not measured winners.
Keep every route visible so future profiling can improve the measured regimes.

Report retained dynamic layout metadata separately from explicit CPU packing bytes,
GPU descriptor/global bytes and actual static shared-memory bytes. Exclude fixed
control headers and caller tensors. Vendor-owned CPU/GPU temporary allocation,
lazy initialization, context memory and process peaks are separate observations;
explicit workspace=0 is not a zero-total-memory claim. New warmed framework calls
allocate no Rust heap buffers or framework device-global scratch. Native library
internals do not acquire an unmeasured zero-allocation guarantee.

CUDA retains its original context/stream and validates every supplied owner. Matrix
metadata is passed by value. Keep existing queued-use/destruction and asynchronous
error contracts; validation errors precede writes but device errors are not
transactional. Independent CPU instances own independent scratch for concurrent
execution. No new inference/model migration or performance promotion is implied.

Vendor references: [CUDA 12.4 GEMM](https://docs.nvidia.com/cuda/archive/12.4.1/cublas/index.html#cublas-t-gemm)
specifies column-major operation/leading dimensions and beta handling; the adapter
uses the transposed row-major equation. Compile the Intel adapter against installed
MKL headers and retain scoped calling-thread settings.

## Verification gates

Independently enumerate affine coordinates and evaluate every forward result and
both operand VJPs in FP64. On bounded finite fixtures use `atol=rtol=2e-5` and
central differences with `step=1e-4`. Compare composed scalar objectives, changed
bindings, overwrite/accumulate and repeated updates against the pinned Torch stack;
do not infer training support from these local plans alone.

Cover all transpose combinations, positive padded leading dimensions, offsets,
negative/gapped/permuted/repeated reads, aliased A/B, injective permuted gradients
and aliased upstream. Include matrix/vector-shaped singleton axes, zero M/N/K,
partial tile extents and contraction tails around 16/32/64/128/257. Check complete
source/hole bits, exact repeats, missing dependencies, wrong equal-size shapes,
foreign sessions even when unused/empty, empty accumulation bits, NaN-initialized
overwrite, saved versions and overlapping view-adjoint composition.

Assert actual route, workspace cap/budget rejection, GPU static shared bytes and
bounded large logical preparation. Verify no hot Rust allocation, independent
concurrent CPU instances, CUDA tile/grid reuse and untouched storage. Run complete
portable/Intel/NVIDIA debug/release and Python regressions, Clippy/format, targeted
Miri, separate GPU memcheck/racecheck/synccheck and source/executable/log recovery.
Keep all outcomes and original numerical gates. Relative latency and exact memory
peaks require separate measured campaigns.

## Verified implementation

CPU and CUDA plans implement the declared rank-two forward and either-operand local
VJPs. CPU dispatches to the portable reference, direct transpose/leading-dimension
MKL or bounded packed MKL; CUDA dispatches to strict cuBLAS or signed-stride shared
tiles. Retained dynamic layout metadata is 96 forward/128 backward bytes on the
validated 64-bit targets. Maximum explicit CPU packing is 57,344 bytes; the tiled
CUDA kernel reports 2,048 static shared bytes and no explicit global workspace.
These are resource observations, not performance acceptance or total memory peaks.

Development checks pass seven CPU/MKL tests and six RTX 3060 CUDA tests. The shared
389-case catalogue covers two changed payload bindings, independent FP64 forward/
VJP/adjoint/finite-difference controls, full source/hole bits, exact repeats and both
gradient modes. Separate cases cover saved versions, aliased operand/view-adjoint
composition, rejected bindings/sessions, empty bits, bounded large preparation,
explicit packing budgets, host threads, Miri-focused bounds and CUDA grid reuse.

An eight-step native primitive chain composes matmul, pointwise and Mean forward/
backward plans with explicit SGD updates. Every loss, weight gradient and updated
parameter agrees with an independently evolving FP64 control. A separate pinned
Torch process verifies all exported matrix/VJP/seeded-accumulation results and the
evolving training trajectory at the original `2e-5` gates. These development checks
preceded the frozen campaign below; their losslessly compressed exports and the older
CUDA preflight log remain in the result boundary.

The [independent assessment](../benchmarks/results/2026-10-19-matmul-assessment.json)
verifies producer `b786aaf`, following declaration `a64faf9`, on 2026-10-20:

| Platform | Rust tests per debug/release build | Doctests per build | Python checks | Export records checked against Torch |
| --- | --- | --- | --- | --- |
| Portable CPU | 88 | 14 | 35 | 1,572 |
| Intel MKL/oneDNN | 90 | 14 | 35 | 3,144 |
| NVIDIA CUDA/cuDNN | 129 | 15 | 35 | 3,144 |

All-target Clippy and formatting pass on each declared configuration; NVIDIA also
passes CUDA-only Clippy. Targeted Miri passes the signed/overlap/permuted-bound paths.
Its nightly emits the preexisting graph `fetch_update` naming deprecation; validated
stable Clippy is clean. Each of the six CUDA tests passes separately under memcheck,
racecheck and synccheck: eighteen commands with zero reported errors/hazards.

Strict reference controls use Torch 2.14.1+cpu locally and 2.14.0+cu130 remotely,
highest FP32 matmul precision and disabled TF32. Sixteen debug/release export files
contain 7,860 matrix/VJP/manual-update records. The maximum observed matrix/VJP/
accumulation absolute error is `7.62939453125e-6`; training gradient and parameter
absolute errors are zero on these fixtures, with loss error at most `1.862645149230957e-9`.
The declared mixed absolute/relative gate remains `2e-5`; these observations do not
change it or promise universal agreement.

All three 138-file source archives match the committed producer. Exact compressed
test executables, command logs, launch receipts and lossless native export archives
pass the independent audit. The [process observation](../benchmarks/results/2026-10-20-matmul-process-state.json)
confirms every recorded collector/command handle is absent. Sources and executables
are recoverable; no isolated archive rebuild or exact memory peak is claimed.

The chain is explicitly assembled by its test. No reverse-mode tape, public optimizer,
batched matmul, relative latency advantage or new application is established here.
