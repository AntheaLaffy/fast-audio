# Execution and numerical contract

## Storage and validation

`Tensor` owns contiguous row-major FP32 host storage. `Shape` supports rank-zero
scalars and zero-sized axes; nonempty products reject multiplication/byte overflow.
`Layout` separately prepares signed element strides, a storage origin, reachable
bounds and a conservative injectivity proof. Host/device views borrow storage and
layout metadata without allocation or copying. Mutable binding requires exclusive
storage access and a proved non-overlapping layout; read-only broadcasts may overlap.
See the [layout/storage contract](tensor-layout-v1.md) for reshape and training limits.
`Tracked<Tensor>` and `Tracked<DeviceTensor>` add opt-in owner identities and modification
generations. Shared reads preserve a version; exclusive access advances it before a
possible write, even if an operation fails or touches no bytes. Saved-version checks
distinguish foreign storage from mutation. Ordinary inference storage carries no such
metadata. Tokens retain neither memory nor layout, and do not prove GPU completion;
see the [tracked-storage contract](storage-versions-v1.md).
`Shared<T>` and `SavedValue<T>` supply opt-in strong retention and immutable saved
layouts. Nonblocking guards allow concurrent host readers, serialize potential writes
and check versions before saved payload access. Capture holds no long-lived lock.
See the [retained-value contract](saved-values-v1.md), including poison recovery and
the distinction between metadata-only view derivatives and saved payload dependencies.
Vendor operations also validate integer dimension limits before crossing FFI.
There are no implicit reshapes, transfers, casts, or hidden outputs. Binary pointwise
broadcasting is explicit in preparation and fixed when binding views.
`ReduceSpec` fixes explicit Sum/Mean axes and keep-dimension policy. Prepared forward
plans preserve reduced logical order through fixed FP32 lane trees; payload-free
local adjoints broadcast upstream and divide Mean gradients by the same rounded
count. An overlapping forward view needs composition with `ViewBackward` to accumulate
storage gradients. Empty/no-axis/single-contribution rules, resources and current
verification limits are in the [reduction contract](reduction-v1.md).
Vendor numerical validation targets finite FP32 inputs. The dynamic reference/MKL ReLU
also preserves NaNs for diagnostics; prepared vendor fusion is validated for finite inputs.

| Operator | Input and parameters | Output and semantics |
| --- | --- | --- |
| Matmul | A `[M,K]`, B `[K,N]` | `[M,N]`, overwrite with `A @ B` |
| Conv1d | X `[B,Cin,L]`, W `[Cout,Cin/groups,K]`, optional bias `[Cout]` | `[B,Cout,Lout]`; cross-correlation, symmetric zero padding, groups/stride/dilation, optional ReLU |
| Conv2d | X NCHW, W `[Cout,Cin/groups,Kh,Kw]`, optional bias `[Cout]` | NCHW cross-correlation; per-axis padding/stride/dilation, groups and optional ReLU |
| Transposed Conv2d | X NCHW, W `[Cin,Cout/groups,Kh,Kw]` | NCHW scatter sum; same groups, optional bias/ReLU and per-axis output padding |
| STFT | X `[B,L]`, window of length `N` (periodic Hann by default), positive hop | `[B,frames,N/2+1,2]`; real/imaginary components, one-sided, unnormalized, no centering, complete frames only |
| Centered STFT | X `[B,L]`, validated `CenteredStftSpec` | `[B,frames,N/2+1,2]`; same transform with `floor(N/2)` reflection padding at each end |
| Inverse STFT | Centered spectrum `[B,frames,N/2+1,2]` | `[B,L]`; inverse FFT, window synthesis and overlap-envelope normalization, fixed validated length |
| GRU | X `[B,T,I]`, initial H `[D,B,H]`, snapshot weights/biases | Y `[B,T,D*H]`, final H `[D,B,H]`; one layer, reset-after, reset/update/new gate order, optional reverse direction |
| Layout copy | Equal logical source/destination shapes with arbitrary signed strides | Exact FP32 bits into an injective destination; prepared CPU rows or same-stream CUDA copy/kernel |
| View adjoint | Original dense base shape, forward read layout and exact-shape upstream gradient layout | Explicit overwrite/accumulate; repeated-address contributions reduce into a dense base gradient |
| Unary pointwise | Equal logical source/output shapes with signed-stride views | Twelve prepared unary operations, including stable nonlinearities; injective output, untouched holes and explicit exceptional-value semantics |
| Binary pointwise | Signed-stride left/right views and an explicit common output shape | Add/subtract/multiply/divide with fixed right-aligned input broadcasting; injective output and untouched holes |

Dense matmul skips empty outputs and overwrites a zero-width contraction with positive
zeros, without sending zero dimensions to vendor GEMM. Empty CUDA tensors own no
device allocation; empty transfers/copies enqueue no payload work. Convolution, GRU
and spectral geometry retain their positive-size contracts. The prepared graph also
rejects scalar/empty values, including scalar normalization suffixes.

`cpu::CopyPlan` / `cuda::CopyPlan` prepare coordinate coalescing once. Run-time view
binding checks initialized length and exact geometry; execution allocates no Rust
scratch. CUDA metadata uploads occur in preparation and its device bytes are reported
as workspace. These plans and the pointwise plans consume strided views; vendor
spatial/spectral/recurrent math still requires owned dense tensors. Prepared
`cpu::MatmulPlan` / `cuda::MatmulPlan` and `MatmulBackward` additionally accept rank-two
strided matrix views and expose either operand's local VJP. They preserve exact binding
geometry, query selective forward-payload dependencies, use direct vendor transpose/
leading dimensions where compatible and bound layout fallback resources. The
[matrix contract](matmul-v1.md#verified-implementation) records complete frozen
CPU/CUDA numerical, resource, Torch, Miri and GPU-tool verification. Existing
dense-owned and graph consumers retain their earlier routes.
Graph rank-four mapped kernels remain specialized consumers of
their older fixed-output/multiply ABI until numerical and performance gates justify
migration; general view copying does not supersede that measured specialization.

`cpu::ViewBackward` / `cuda::ViewBackward` implement the read view's linear adjoint
without retaining the original forward payload. Injective mappings use unique writes;
broadcasts reduce analytically over zero-stride axes; remaining overlap uses bounded
sorted index tables. Fixed 32/256-lane FP32 trees avoid atomic reduction order. See
the [view-backward contract](view-backward-v1.md) for initialization, byte accounting,
repeatability limits and independent gradient gates. The initial native tape composes
these adjoints; its declared initial scope now passes the frozen campaign described below.

`cpu::UnaryPlan` / `cpu::BinaryPlan` and their CUDA counterparts prepare general
pointwise math independently of the fixed graph ABI. Output strides determine axis
order; negative output axes are reversed with every input, then compatible axes
coalesce. CPU dispatches dense/broadcast rows or signed-stride loops. CUDA keeps flat
geometry in kernel parameters and stores only additional outer-axis descriptors.
Preparation is O(rank) in storage; execution has no heap/global scratch allocation or
metadata upload. See the [pointwise contract](pointwise-v1.md) for exact operations,
FP32 category/bit rules, resource scope and independent forward gates. These plans do
not supply derivatives or in-place execution. The [coverage matrix](operator-coverage.md)
tracks forward/backward gaps without treating a family as complete.

`cpu::UnaryBackward` / `cpu::BinaryBackward` and their CUDA counterparts provide
separate local VJPs for the same operations. Binary plans select one original operand
with `BinaryOperand` and fuse its broadcast contraction with the local derivative.
Dependency queries identify which optional forward inputs a caller must retain;
plans store geometry and validate supplied layouts/sessions before writes. Compose
logical input gradients with `ViewBackward` when forward storage overlaps. See the
[pointwise-backward contract](pointwise-backward-v1.md) for stable derivative, ordered
tree, empty/accumulation and resource rules. Frozen portable/Intel/NVIDIA full-suite,
Miri and GPU instrumentation checks pass, including aliased upstream/permuted
gradient destinations. These explicit calls do not implement a reverse-mode tape or
enforce saved-value versions automatically.

`autograd::Builder` / `Program` and `cpu::Tape` / `cuda::Tape` add explicit prepared
training composition over guarded shared leaves. Metadata aliases retain their
storage roots; logical gradients project to canonical dense root gradients, with
tied leaf owners deduplicated by storage identity. Local dependency queries select
per-node saved versions. Forward epochs, root seeds, reverse edge order, gradient
validity and checked parameter writers prevent partial or stale gradients from
reaching plain SGD. Normal execution reuses prepared buffers and plans; CUDA uses
same-stream zeroing and a by-value learning-rate update kernel. See the
[training contract](autodiff-v1.md#verified-initial-implementation) for exact API,
failure/resource limits and verified Torch/audio-layout/frozen scope. This first implementation retains required
activations; arena liveness reuse, fusion/replay and broader training mechanics remain
later foundation work.

Convolution uses `Lout = floor((L + 2*padding - dilation*(K-1) - 1)/stride) + 1`.
STFT uses `frames = 1 + floor((L-N)/hop)` and defaults to
`window[t] = 0.5 - 0.5*cos(2*pi*t/N)`.
For non-centered STFT an input shorter than one FFT frame is an error. Odd FFT sizes are supported.
Output axes are checked individually; an equal number of elements with a different
shape is not accepted.

## Centered audio reconstruction

`StftSpec::centered()` returns a validated `CenteredStftSpec` without changing the
existing non-centered plan. Reflection requires `L > floor(N/2)` and excludes the
boundary sample from its reflected copy, matching PyTorch's `pad_mode="reflect"`.
The centered frame count is `1 + floor((L + 2*floor(N/2) - N)/hop)`; the formula also
covers odd FFT sizes. `input_shape` and `output_shape` describe the forward transform;
the inverse takes those shapes in reverse order. Use `cpu::CenteredStft` and
`cpu::Istft`, or the identically shaped `cuda` plans. FFT size, hop, batch and waveform
length are fixed during preparation.

All three forward/centered/inverse plan types expose `with_window` on CPU and CUDA.
The supplied host tensor must have exactly shape `[N]` and finite coefficients; preparation
snapshots them, including upload for CUDA. Source mutation cannot change a prepared plan.
Analysis and synthesis must use the same window for a round trip. GTCRN passes its exact
FP32 square-root Hann coefficients instead of substituting the default Hann window.
The inverse computes its envelope from those actual coefficients; a zero window is
valid for a forward transform but fails the inverse NOLA check.

For output sample `t`, let `q = t + floor(N/2) - frame*hop`, retaining terms with
`0 <= q < N`. The inverse sums `raw_inverse_fft(spectrum[frame])[q] * window[q]`
and divides by `N * sum(window[q]^2)`. Preparation caches the reciprocal envelope
and rejects any requested sample with an envelope below `1e-11` (NOLA). Gaps or an
uncovered final tail are errors; the inverse does not silently synthesize a zero tail.
The original length remains explicit, including a final partial hop. The periodic Hann
has a zero first sample, so the original non-centered transform is not offered as a
lossless inverse pair.

Spectra remain one-sided, interleaved FP32 rather than a new complex tensor dtype.
For real synthesis, DC and the even-size Nyquist bin use only their real components;
the highest odd-size bin retains its imaginary component. Input spectra are preserved
on every backend. oneMKL and cuFFT use plan-owned spectrum copies; cuFFT C2R may overwrite
its input. CUDA overlap-add assigns each output sample to one thread and accumulates
frames in order, avoiding atomic addition. Framing reflects indices directly without
allocating a padded waveform. Repeated execution reuses all explicit workspace.

CPU `Istft::workspace_bytes()` counts explicit execution buffers, excluding FFT
implementation internals. CUDA also includes cuFFT's reported work area. Caller input
and output allocations are excluded. These transforms implement fixed offline chunks;
streaming overlap state, alternative normalization and differentiable
audio graphs remain future work. The [example](../examples/audio-roundtrip.rs) and
`tools/check_audio.py` check full spectra, round trips and arbitrary modified spectra
against PyTorch, without making a latency claim.

Reference contracts: [PyTorch inverse STFT](https://docs.pytorch.org/docs/2.14/generated/torch.istft.html),
[oneMKL transform distances](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/dfti-input-distance-dfti-output-distance.html),
and [CUDA 12.4 cuFFT layouts](https://docs.nvidia.com/cuda/archive/12.4.1/cufft/index.html#data-layout).

## CPU route

`Cpu::Reference` uses a scalar GEMM and RustFFT, and remains an independent portable
control for vendor integration. It is not the performance backend.
`Cpu::Mkl` uses row-major oneMKL SGEMM and batched real/complex DFTI descriptors.
The C adapter includes the installed vendor headers so Rust does not duplicate the
variadic DFTI ABI or the selected C integer typedefs.

Conv1d currently lowers to reusable im2col storage and GEMM, processing each batch/group
on the CPU. Pointwise convolution directly uses input storage and needs no im2col
workspace. The bias/ReLU pass follows GEMM. This implementation remains the dynamic-parameter
control and the direct pointwise path. Plan construction validates
geometry and allocates scratch; repeated execution performs no Rust heap allocations.
Tests verify that property after warm-up. Vendor-internal allocations are outside the
Rust allocator's visibility and must be profiled separately.

With `onednn`, `onednn::Conv1d` snapshots weights/bias, packs weights once, owns a user
scratchpad, and prepares required NCL/vendor-layout reorders. Bias/ReLU are fused into
the convolution primitive. Input/output reorders are included in execution timing;
parameter packing belongs to preparation. `workspace_bytes` includes internal input,
output and scratch storage; `parameter_bytes` reports packed parameters separately.
The adapter restores the caller's OpenMP thread-count/dynamic settings after creation
and execution and requires an OpenMP oneDNN build.

`cpu::PreparedConv1d` owns constant parameters. With `mkl,onednn`, temporal/grouped
convolution uses oneDNN while pointwise convolution keeps direct oneMKL GEMM to avoid
reorder overhead. Otherwise it snapshots parameters for the existing CPU route.
Source parameter changes do not affect a prepared plan; construct a new plan to update
parameters. The original `cpu::Conv1d` reads weights/bias on every call. The Intel
prepared route remains opt-in pending resolution of the Intel full-suite timing guard;
see progress.

## Spatial convolution

`Conv2dSpec` describes a fixed NCHW input, two spatial axes and constant parameters.
Both channel counts must be divisible by `groups`. Each forward output axis uses the
Conv1d formula above. A transposed axis uses
`(input-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1`.
Output padding is zero for forward convolution; for transpose it must be smaller
than stride or dilation on each axis. Preparation checks positive dimensions, storage
arithmetic, vendor/GEMM integer limits and exact parameter shapes before FFI.

`cpu::Conv2d` snapshots weights/bias and reuses im2col or col2im scratch. Transpose
prepacks weights for GEMM, then scatters columns into the caller's output. With
`mkl,onednn`, non-pointwise plans use oneDNN forward/deconvolution with cached packing,
layout reorders and user scratchpad. With `mkl` alone they use MKL GEMM. Pointwise
plans in either direction normalize weights once and need no column scratch.
`onednn::Conv2d` also exposes the vendor plan directly.

`cuda::Conv2d` requires `cudnn`, snapshots host parameters to owned device storage,
and uses strict-FP32 forward/backward-data graphs, retaining direct cuBLAS pointwise
execution. This differs from the dynamic device parameters of `cuda::Conv1d`.
Bias/ReLU follow the graph's existing fusion/fallback policy. Plans retain their
session and reuse immutable variant packs; input/output rebinding follows the same
synchronization rule as Conv1d. Execution never stages activations on the host.

Vendor deconvolution represents output padding below stride. For valid output padding
at least as large as stride but below dilation, plans extend the high edges of the input
with zeros. This preserves the requested output and all original convolution sums.
CPU copies and the CUDA padding kernel execute into reusable storage inside `run`.
Workspace metadata includes this extension plus vendor scratch/reorders; the CUDA
limit bounds their total. Packed parameters and caller tensors are counted separately.
Tests compare grouped, dilated, depthwise and transposed cases against an independent
FP64 gather oracle, including zero-padding extensions and hot Rust allocations.

## Recurrent contract

`GruSpec` validates nonzero batch/steps/input/hidden sizes, i32 projection dimensions
and all storage bounds. `GruWeights` supplies input `[D,3*H,I]`, recurrent `[D,3*H,H]`
and two bias `[D,3*H]` tensors. Plans snapshot them during preparation. Let
`u = W_ih*x + b_ih` and `v = W_hh*h + b_hh`, with rows ordered reset, update, new:
`r = sigmoid(u_r+v_r)`, `z = sigmoid(u_z+v_z)`, `n = tanh(u_n+r*v_n)`, and
`h_next = (1-z)*n+z*h`. The reset multiplies the recurrent candidate bias as well.
This matches PyTorch's reset-after equation. Both direction outputs use the original
time index; final reverse state corresponds to the beginning of the sequence.

Plans hold no implicit carry state between calls. Forward chunks can pass the previous
final state explicitly; bidirectional chunks cannot in general reproduce a full future-aware
sequence. Fixed full-length batches are supported. Packed variable-length input, layers
above one, dropout, gradients and mixed precision are not implemented by this primitive.

The CPU route batches all input projections and reuses state/recurrent scratch during
step updates. With `mkl`, projections use oneMKL. CUDA requires `cudnn` and uses owned
vendor weight space with one stream. It selects the persistent small-hidden algorithm
for SM 6+ and `H <= 192`, otherwise standard mode, always using `CUDNN_FMA_MATH`.
`GruOptions` exposes standard/automatic algorithm and internal batch/sequence layout;
public tensors remain batch-first. Sequence-major execution skips transforms for batch
one and uses preallocated GPU transpose buffers otherwise. Copies are inside execution.
cuDNN additionally limits `T` to 65,535; Rust rejects larger plans before native setup.

`workspace_bytes` excludes caller buffers and packed parameters. It includes explicit
CPU scratch or the cuDNN work area plus GPU layout buffers. `parameter_bytes` reports
the prepared storage, including vendor padding. Vendor descriptor/driver allocations
are not a peak-memory measurement. Both outputs are caller-owned and must have exact
shapes and the correct session. Plans preserve input and initial state and wait for
pending device uses before releasing native storage.

See [GTCRN work](gtcrn.md) for captured real inputs and the remaining whole-model gaps.

## NVIDIA route

The Rust API owns a native session containing a nonblocking CUDA stream and a cuBLAS
handle. Tensor allocations, convolution plans, FFT plans, and events retain that
session using `Rc`. Public handles remain on one host thread; no `Send`/`Sync`
implementation is supplied. Cloning `Cuda` shares its session. Tensors from separately
created sessions are rejected, including when they use the same physical GPU.

All uses of an allocation are ordered on its session's stream. Device pointers are
private. Host copies synchronize before borrowed slices can expire. Dropping a device
allocation uses synchronous `cudaFree`; FFT plans synchronize before releasing scratch.
Dropping the facade before a tensor is safe because the tensor retains the session.
Explicit synchronization reports deferred GPU errors; destructors cannot report errors.
This initial ownership model favors correctness and steady-state reuse. Stream pools,
pinned transfers, async allocation pools, and graph replay require a later explicit API.

GEMM calls cuBLAS in strict FP32 mode. Row-major `C=A*B` is interpreted as the column-major
`C^T=B^T*A^T` without a transpose kernel. Conv1d uses a GPU im2col kernel, strided batched
SGEMM with broadcast weights, and a fused bias/ReLU epilogue. Pointwise convolution skips
im2col. With `cudnn`, `Conv1d::with_cudnn` prepares a cuDNN graph; `Conv1d::planned`
keeps direct pointwise GEMM and chooses cuDNN for temporal/grouped shapes. Engine
configurations advertising tensor cores, input down-conversion or reduced-precision
reduction are excluded from this strict-FP32 policy. Heuristics A/fallback search is
bounded by the caller's workspace limit (64 MiB in the benchmark). Unsupported graph
fusion uses a cuDNN convolution graph and the existing CUDA epilogue. Implementation
metadata identifies the selected engine and whether activation/bias fusion succeeded.

`Cuda::new` retains `ConvolutionPolicy::AllowNonDeterministic`. With `cudnn`,
`Cuda::with_convolution_policy(device, ConvolutionPolicy::RequireDeterministic)`
creates a session whose convolution policy is immutable and shared by clones.
Both fused and unfused cuDNN searches also reject engines advertising the
`NONDETERMINISTIC` numerical note. `Conv1d`, `Conv2d` and graph-contained convolutions
inherit the session policy. Other operators and the direct pointwise cuBLAS route
retain their own algorithms; this is not a global RNG or cross-stack bit guarantee.

`cudnn_plan_info()` returns actual engine index, numerical-note mask, workspace and
fusion/determinism flags, or None for direct cuBLAS. `Graph::cudnn_convolution_plans()`
reports those selections by graph step. No timing-loop metadata query is needed.
When no strict-FP32 engine satisfies policy and workspace, preparation returns
`Error::Unsupported` with the policy/limit; it never silently relaxes determinism.
The private 32-byte C/Rust metadata layout and negative no-plan status are distinct
from vendor failures. The [repeatability protocol](cudnn-repeatability-v1.md) checks
the existing complete graph under both policies and warm-up regimes.

cuDNN parameters remain device inputs, so values may change without repacking. Prepared
bias presence is fixed. Finalized variant packs are immutable: stable buffer addresses
reuse a cached pack; changed addresses synchronize old uses and rebuild the pack.
This rebinding cost matters to applications rotating buffers. The plan retains its
session and waits before freeing graph/workspace resources. Host pointers never enter
a CUDA variant pack.

STFT fuses framing/window multiplication into one kernel, followed by a reusable
batched cuFFT R2C plan. Its non-centered kernel specialization excludes reflection work.
The inverse uses a C2R plan and overlap-add as described above. Plans, uploads, allocations, and first-use initialization are
outside the warmed execution loop.

The implementation follows the vendor contracts for [cuBLAS storage](https://docs.nvidia.com/cuda/cublas/index.html#data-layout)
and [cuFFT planning](https://docs.nvidia.com/cuda/cufft/index.html),
[oneDNN convolution](https://uxlfoundation.github.io/oneDNN/dev_guide_convolution.html),
[user scratchpads](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_scratchpad.html),
and the [cuDNN 9.3 graph API](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/api/cudnn-graph-library.html).
Prefer installed headers
and versioned documentation when changing FFI. Native status codes are partitioned as
`1000 + CUDA`, `2000 + cuBLAS`, `3000 + cuFFT`, and `4000 + cuDNN`; zero indicates success.

## Prepared inference graphs

`graph::Builder` has one fixed-shape FP32 input. Each value carries a builder identity;
foreign values, unknown edges and incompatible shapes fail during construction. Nodes
can reference only earlier values, making execution acyclic. `finish` requires the final
single-output operation as the caller-visible result. `Program` is consumed by
`cpu::Graph::new` or, with `cudnn`, `cuda::Graph::new`. Preparation snapshots parameters,
packs linear weights once and prepares every vendor primitive.

| Operation | Explicit contract |
| --- | --- |
| Layout | Slice ranges, axis permutation and output reshape in one materialization; input rank at most four |
| Linear | Last-axis `[out,in]` projection with optional bias and explicit output reshape; checked GEMM row/column limits |
| Elementwise | Equal-shape addition, sigmoid/tanh/scalar PReLU; explicit right-aligned second-input multiplication broadcasting up to rank four |
| Concat | Checked concatenation on one axis, or alternation of equal-sized axis entries for channel shuffle |
| Padding/unfold | Zero padding on an explicit axis; odd-width, stride-one, zero-padded NCHW frequency neighborhoods |
| Reduction | Mean square along the last axis, dropping that axis |
| Layer normalization | Mean/biased variance over the parameter-shaped suffix, positive finite epsilon and snapshot affine parameters |
| Complex features | `[B,T,F,2]` to NCHW magnitude/real/imaginary with explicit magnitude epsilon |
| Complex mask | NCHW two-channel mask times `[B,T,F,2]`, yielding `[B,T,F,2]` |
| Convolution/GRU | Existing checked prepared primitives; GRU initial and final states are distinct graph values |

Execution preserves caller input, writes the caller's exact-shape output directly, and
reuses every explicit intermediate buffer. Both backends perform no warmed Rust heap
allocation. CUDA validates external session ownership and executes all kernels/vendor
calls on the retained stream. Its normalization/reduction kernels accumulate in FP32;
linear/convolution/recurrent primitives retain their strict-FP32 policies. Activations
stay on the device throughout the model. Uploads and parameter packing occur during
preparation. The prepared Rust sequence is independent of future NVIDIA CUDA graph
capture/replay support.

Strict FP32 numerical filtering is separate from deterministic execution. Default
cuDNN selection permits nondeterministic notes; the explicit convolution policy
above rejects them when requested. The earlier
[mapped-access repeatability diagnosis](mapped-access-screen-v1.md) observes tiny
fresh-process differences even in the unchanged native control. Original numerical
tolerances remain applicable; whole-model bit repeatability is not a public guarantee.

Layout and broadcast multiplication prepare coalesced coordinate mappings. CPU walks
mapped rows, copying contiguous regions and repeating broadcast factors without
per-element division. CUDA uses a private 60-byte descriptor when every dimension,
stride, output count and largest mapped source index fits `u32`. Precomputed reciprocal
quotients plus one remainder correction give exact integer addressing; grid-stride
iteration stays 64-bit to prevent wraparound. Larger mappings retain the existing
112-byte descriptor and 64-bit kernel. No output buffers, floating-point equations or
reduction orders change. Descriptor metadata remains outside reported tensor workspace.

The initial executor retains all intermediate values, enabling complete trace inspection
through `intermediate`. Layouts materialize a contiguous output; linear projection may
reshape its own output without a separate copy. Liveness-based buffer reuse and transform
fusion are future profiling work. `workspace_bytes` sums graph intermediates and primitive
scratch; `parameter_bytes` includes constants and prepared parameters. Caller input/output,
descriptor metadata and vendor/driver allocations outside reported work areas are excluded.
Each CUDA convolution graph has a 64-MiB workspace allowance, separate from other nodes.
These are planned component counts, not observed peak memory. The complete-model example's
`inverse_workspace_bytes` field reports inverse-plan storage only (historical snapshots
called it `spectral_workspace_bytes`). Forward FFT storage is not included there.
The complete-model characterization records whole-command RSS and sampled GPU process
high-water values separately; exact GPU peaks and complete allocation attribution remain open.

`models::gtcrn::build(batch, frames, parameters)` lowers the pinned MIT architecture
from named host tensors. Inference BatchNorm is folded into OIHW or grouped IOHW weights
and bias once. ERB projections, SFE, grouped temporal attention, both dual-path GRNNs,
decoder skips/shuffles and complex masking then use the shared graph. All recurrent
states start at zero for each offline invocation. Preparation must be repeated for
changed weights or geometry. Streaming carry state and differentiable model execution
remain separate contracts. The original attribution is preserved in
[`models/gtcrn/LICENSE`](../models/gtcrn/LICENSE).

`--capture-model` exports the reference's named FP32 parameters, skips integer training
counters and saves 15 block outputs per input. The complete-model checker verifies the
locked model identity, bundle/parameter/activation hashes and full native outputs before
claiming numerical success. Rust runtime execution uses exported tensors and native
libraries; the example's file loader validates tensor byte/shape geometry, while the
checker supplies provenance verification. Source and binary are frozen across validation.

## Extension rules

An operator needs an explicit layout, precision policy, independent numerical oracle,
shape/boundary rejection, and the same ownership guarantees before performance tuning.
Keep preparation outside execution and report required workspace. An optimization must
survive full-output validation and the benchmark contract. Do not hide a host fallback,
host/device copy, precision reduction, or synchronization behind a tensor operation.
