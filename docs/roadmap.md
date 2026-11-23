# Foundation-first roadmap

## Direction and year-one boundary

Build a general audio deep learning framework in Rust for Intel CPUs and NVIDIA GPUs.
Performance is the primary objective in `target.md`; numerical correctness, ownership
and reproducible measurement constrain every optimization.

The user's foundation-first direction, reaffirmed on 2026-10-19, supersedes the
earlier application-led plan:
**year one is dedicated to framework foundations, with no application delivery or new
model integration.** Music processing, source separation and audio synthesis remain
the first application priorities after the foundation gates pass. Speech and other
audio domains remain important; non-audio applications remain outside the main focus.

Year-one work includes storage, operators, differentiation, graph execution, device
scheduling, audio DSP primitives, training mechanics and developer tooling. It excludes
new separators/generators/codecs, task-specific training recipes, model import projects,
application demos, DAW plugins and product interfaces. Keep the existing GTCRN path as
a numerical/performance regression workload. The completed [DAC inventory](dac.md) is
archived research; checkpoint acquisition and native integration are deferred.

This boundary does not remove workload validation. Procedural signals, independently
specified operator/subgraph suites and the existing frozen graph exercise realistic
audio geometry without adding applications. No model-specific feature should enter
the core merely to expand a model list.

Admit year-one work only when it closes a reusable framework capability, a correctness/
resource gap or a measured performance bottleneck. An early working inference or
training graph does not reopen application delivery. Keep model acquisition, task
datasets and product milestones out of the year-one critical path.

### What a solid foundation means

The first-year deliverable is a coherent single-device training and inference stack
on Intel and NVIDIA. It must connect tensor ownership, reusable forward/backward
operators, differentiation, execution planning and parameter updates. Start with
strict FP32 and make additional dtypes, layouts and numerical policies explicit
capabilities.

Keep the general audio scope in the foundation itself: long time axes, short blocks,
small batches, multichannel and time/frequency layouts, complex spectra and persistent
DSP state. These are reusable tensor, numerical and runtime requirements. Music,
separation and synthesis guide this geometry without introducing task models.

Use the following order when work competes for time:

| Priority | Work to close | Completion rule |
| --- | --- | --- |
| Blocking foundation | Storage/alias/version contracts; pointwise/reduction/dense gradients; reverse-mode tape; a native forward/backward/update loop | Complete and independently check a small procedural training graph on both hardware paths before widening the graph API |
| Required audio core | Convolution, normalization, spectral/complex derivatives; typed indices/masks; audio layouts and state; reusable operator coverage | Close the declared year-one capability matrix, including input/parameter gradients and composed audio subgraphs |
| Required runtime and performance | Memory planning, asynchronous ownership, thread/stream scheduling, fusion/replay and hardware dispatch | Begin contracts and profiling in Q1; integrate optimizations as prerequisites pass, with measured gains and resource checks |
| Foundation breadth | Remaining reduction/indexing, sequence/attention, DSP and precision capabilities in the matrix | Implement in dependency order; retain unsupported cases and unfinished gates explicitly rather than counting a partial family as complete |

Every required gap remains in the [coverage matrix](operator-coverage.md). If the
required work exceeds the first-year window, continue foundation work and revise
the estimate. Application work does not
start automatically when the calendar advances. Comprehensive audio operator, layout,
precision and performance coverage remains a multi-year objective.

## Starting point and unresolved evidence

There is a working inference substrate, not a finished foundation: checked contiguous
FP32 storage, separate host/device ownership, prepared GEMM, grouped/spatial/transposed
convolution, GRU, centered STFT/inverse-STFT and a fixed-shape inference graph. The graph
has one external input/output, retains intermediates, limits layout rank to four and
snapshots immutable parameters. The [borrowed-layout foundation](tensor-layout-v1.md)
now supports FP32 signed-stride host/device views, exact layout copying and scalar/
empty storage. [Tracked ownership](storage-versions-v1.md) adds opt-in identities and
potential-write generations without changing ordinary inference tensor metadata.
[Shared retained values](saved-values-v1.md) add guarded ownership and saved layouts.
[View adjoints](view-backward-v1.md) provide the first backward primitive.
[Prepared pointwise math](pointwise-v1.md) adds strided unary/binary forward execution;
the verified [Sum/Mean plans](reduction-v1.md) add ordered general-stride reductions
and explicit payload-free local adjoints. Verified [pointwise local VJPs](pointwise-backward-v1.md)
add selective saved-value dependencies and fused broadcast contraction. The
[prepared rank-two matmul/local VJPs](matmul-v1.md) now verify dense derivatives and
an eight-update primitive chain against independent FP64/Torch controls. The [prepared strided 1D/2D convolutions and local VJPs](conv-backward-v1.md) now verify
non-transposed and transposed grouped/strided/dilated/padded cross-correlation and
deconvolution forward (`Conv1dPlan`, `Conv2dPlan`, `ConvTranspose1dPlan`,
`ConvTranspose2dPlan`), `dX`/`dW` adjoints, and `autograd::Builder::{conv1d, conv2d,
conv_transpose1d, conv_transpose2d}` tape training across portable CPU, Intel MKL/oneDNN,
and NVIDIA CUDA/cuDNN against independent FP64, Miri, `compute-sanitizer`, and strict-FP32
PyTorch controls. The [differentiable strided STFT, centered STFT, and NOLA inverse STFT](stft-backward-v1.md)
verify spectral forward/local VJPs, post-spectral whole-step fusion, and full CUDA Graph
capture. The [parameter-free normalization](norm-v1.md), [stable softmax/log-softmax](softmax-v1.md),
and [nonlinear reductions](reduction-v1.md) (`Max`, `Min`, `LogSumExp`) verify numerical
blocks and their local VJPs across CPU and CUDA. The [broadcast batched matrix multiplication](bmm-v1.md)
and [strict-FP32 O(T)-memory Flash Scaled Dot-Product Attention](sdpa-v1.md) (MHA/GQA,
causal/non-causal, zero-atomic `dQ`/`dK`/`dV`) verify sequence attention and batched dense
training. The [structural padding and concatenation primitives](structural-v1.md) (`Pad`
with `Constant`/`Reflect`/`Replicate` and `Cat`/`cat_all`) and [native fused AdamW,
deterministic global L2 gradient clipping, single-CUDA-Graph dynamic-LR training, and
checksummed cross-device checkpointing](optim-v1.md) together with exact tape activation
liveness buffer reuse and runtime AVX2+FMA / 2D L1 cache-blocked CPU whole-step fusion
close the core single-device audio training and checkpointing loop. The
[coverage matrix](operator-coverage.md) records remaining derivative families (such as
GRU backward, gather/scatter, pooling, RNG/dropout, and opt-in reduced precision).

The [complete pointwise baseline](pointwise-characterization-results-v2.md) now
verifies all 56 cases on both hosts. CPU transcendental/nonlinear and transposed
layouts, GPU channel broadcast and transposed writes retain substantial losses
against compatible Torch modes. Independent profiling identifies CPU libm work and
expensive GPU broadcast kernels; the suite averages do not close these gaps or
constitute performance acceptance. Required differentiation work continues alongside
bounded experiments on these measured mechanisms.

The [complete-graph characterization](gtcrn-characterization-results-v1.md) shows both
opportunities and gaps. Native CUDA beats eager Torch on these fixtures but trails
prepared Torch CUDA Graph execution; Intel trails eager Torch. The subsequent
[mapped-address screen](mapped-access-screen-v1.md) reduces Intel latency by about
22.2% against the retained native control, without independent confirmation or fresh
Torch timing. Its GPU exact-output gate fails; a separate probe exposes repeatability
limits in both old and new binaries. Original numerical tolerances still pass.

The subsequent [explicit convolution policy](cudnn-repeatability-v1.md) verifies
all 288 required exact trace/output comparisons across its twelve-command suite.
It provides a tested deterministic control on the current GPU stack; the default
remains permissive and its performance tradeoff is unmeasured.

Keep that failed screen, the unresolved Intel projection-GEMM guard and existing
champions unchanged. Sampled memory observations are not exact peaks. These are active
engineering gaps, not grounds for a framework-wide performance claim.

## Current status assessment (2026-11-23)

`fast-audio` has reached its **strict-FP32, single-device, static-shape audio training MVP
core**, verified across portable CPU + Miri (0 UB), Colab Intel CPU (`--features mkl,onednn`,
132 tests + 16 doctests, 35/35 PyTorch cross-checks), and Colab NVIDIA L4 (`CUDA_ARCH=89
--features cuda,cudnn`, 195 tests + 17 doctests, 35/35 PyTorch cross-checks, 12/12
`compute-sanitizer` `memcheck`/`racecheck`/`synccheck` runs with 0 errors and 0 hazards). See
[progress.md](progress.md) for raw measurements.

Within this narrow strict-FP32 fixed-shape audio slice, prepared zero-allocation dispatch,
2D L1 cache-blocked CPU whole-step fusion, NVRTC CUDA fusion, single-CUDA-Graph dynamic-LR
`AdamW`, and custom 64×64 SIMT SDPA GEMM kernels outperform PyTorch `2.14.1` strict FP32 on
Colab Intel CPU (**11/11** sweep vs Eager/Inductor, 1.70× geomean vs best) and NVIDIA L4
(**9/9** vs Eager at 2.79× geomean; **8/9** vs best at 1.84× geomean, with stateless 2-kernel
`softmax-256` at 0.98× vs Inductor's single fused Triton kernel).

**This is a validated foundation MVP, not framework-wide superiority over PyTorch.** Four major
engineering gaps separate the current stack from a competitive general audio training framework:

1. **No mixed precision or Tensor Core execution (`BF16`/`FP16`/`TF32`/`FP8`):** Strict FP32
   only today. On modern NVIDIA GPUs (Ampere/Ada/Hopper), PyTorch with `autocast(bfloat16)` +
   Tensor Core GEMM/Conv/FlashAttention-2/3 delivers 4–8× higher throughput than strict FP32.
   Closing this gap is the highest-leverage performance priority.
2. **Incomplete dtype and audio-model operator surface:** General `I64`/`U32`/`Bool`/`Complex64`
   tensors do not exist yet. Key training primitives are still missing: `Embedding`/`Gather`/
   `ScatterAdd`, `GRU`/`LSTM` backward, `Pooling`, arbitrary attention masks (beyond causal/none),
   reproducible `RNG`/`Dropout`, reusable losses (`CrossEntropy`, multi-resolution STFT loss),
   and differentiable stateful chunked DSP (resampling, filterbanks).
3. **Single-device, fixed-shape, single-thread scaling limits:** Only fixed static shapes are
   supported (no variable-length audio bucketing); CPU benchmarks currently qualify 1-thread
   execution rather than multi-core thread pools; long-sequence GPU SDPA (`T > 1024`) still
   falls back to scalar-register `BR=32, BC=32` FlashAttention; and there is no general pooled
   VRAM allocator, async host-to-device staging pipeline, or multi-GPU distributed runtime.
4. **No full-scale multi-epoch convergence proof yet:** Current end-to-end training benchmarks
   exercise single-block Transformer and 2-layer spectral-vocoder steps. Full multi-layer audio
   architectures have not yet been trained to convergence on real audio workloads against
   matched PyTorch `bf16 + compile + cudagraph` runs.

## First-year delivery sequence

Quarters describe the intended order within the first development year, not calendar
release promises. Each gate controls advancement; correctness and performance work
continue throughout. Design training contracts in Q1 and implement gradients in Q2,
before inference-only assumptions become entrenched.

The critical dependency chain is checked storage and saved values, local derivatives,
reverse-mode composition, a native training step, and then optimized whole-step
planning. Allocator/stream design and performance measurement run alongside this
chain from the beginning. Q3 is the integration milestone for runtime work, not its
starting date. No phase is complete merely because its planned quarter has ended.

For every required capability, track functional verification, runtime integration and
performance qualification separately. A reference implementation or a passing local
gradient test closes only its declared functional gate. Begin profiling and bounded
optimization once that gate passes; do not postpone all tuning until every operator
exists. Conversely, a fast forward kernel does not close missing backward, ownership
or integration work. Publish those statuses in the capability and performance reports.

| Phase | Main work | Required evidence |
| --- | --- | --- |
| Q1: tensor, memory and measurement contracts | Storage/views/aliasing, dtype and shape rules, device/stream ownership, allocation accounting, parameter versions and repeatability policy | Independent storage/layout oracles, ownership/error checks, reproducible CPU/GPU baseline matrix and an explicit gradient contract |
| Q2: operator and differentiation core | Reusable operators, reverse-mode tape, backward kernels, a minimal SGD training loop and hardware dispatch | Published coverage, gradient/adjoint checks, native forward/backward/update agreement, layout/tail cases and per-operator resource/performance reports |
| Q3: efficient graph and runtime | Multiple inputs/outputs, lifetime reuse, fusion, shape specialization, asynchronous scheduling and prepared training steps | Subgraph equivalence, verified memory reuse, no unexpected warmed allocations, concurrency checks and confirmed gains on affected strata |
| Q4: training foundation and hardening | Optimizers, RNG/checkpoints, precision modes, state primitives, recovery and regression tooling | Matched training steps/resume, resource budgets and independently confirmed inference/training performance on the declared suite |

### Q1: tensor, memory and execution contracts

- Separate storage ownership from tensor metadata. Specify shape, stride, offset,
  alignment, contiguity, zero-sized dimensions, slicing, broadcasting and alias rules.
  Remove the architectural rank-four assumption; expose implementation limits explicitly.
  Validate overflow and bounds before selecting an unsafe hot path.
- Keep FP32 as the reference policy. Define typed integer/index, boolean, complex and
  reduced-precision storage/conversion contracts, with a capability query for operations
  actually implemented. No silent precision change or host fallback.
- Define allocator/pool lifetimes, persistent versus temporary storage, workspace
  ceilings and allocation-failure behavior. Separate caller buffers, parameters, graph
  arenas, vendor scratch and process observations in resource reports.
- Specify context/device/stream/event ownership, asynchronous errors, transfer
  completion and concurrent execution instances. Define trainable parameter versions,
  saved values, mutation checks and inference parameter snapshots together.
- Declare numerical/repeatability policies before selecting vendor plans. Test
  same-process and fresh-process repeats under the exact warm-up/algorithm regime.
  Deterministic and nondeterministic performance modes need distinct contracts; strict
  FP32 alone is not a bit-repeatability guarantee.
- Extend the benchmark catalogue and record preparation, first execution, warmed
  execution and transfers separately. Improve CPU affinity/thread/frequency/load
  controls before another Intel confirmation.

**Gate:** written storage/autodiff/stream contracts exercised by portable, Intel and
NVIDIA tests; allocation and ownership checks; a declared baseline matrix; explicit
unsupported cases. A design document alone does not finish this phase.

### Q2: operator coverage and reverse-mode differentiation

Implement a reusable operator surface rather than new model adapters. Prioritize the
following families; publish exact supported shapes, layouts, dtypes, forward/backward
routes and hardware in a capability matrix before calling a family complete.

| Family | Foundation scope |
| --- | --- |
| Tensor algebra | Elementwise arithmetic/activations, broadcast/reduce, gather/scatter, concatenation, padding, views and layout conversion |
| Dense and spatial | GEMM/batched GEMM; grouped, strided, dilated and transposed Conv1d/2d; bias/activation epilogues and pooling |
| Numerical blocks | Stable softmax/log-sum-exp, normalization, reductions, masking and their gradients |
| Audio transforms | FFT/real FFT and adjoints, complex arithmetic, STFT/inverse-STFT, overlap-add, windows, magnitude/phase and multiresolution transforms |
| Audio DSP | Resampling, filterbanks, FIR/IIR state, delay accounting and differentiable paths where specified |
| Sequence primitives | Recurrent cells, scaled dot-product attention, masks, positional transforms and bounded state/cache interfaces |
| Training primitives | Seeded random/dropout operations, scalar waveform/spectral losses, accumulation and parameter-gradient buffers |

Each family needs an independent reference, awkward shapes and error behavior before
tuning. Check FFT normalization/conjugation, STFT boundary/window envelopes, stereo
independence, resampling delay/alias rejection and filter state explicitly. Define
derivatives at nonsmooth points and unstable/undefined phase cases rather than relying
on accidental vendor behavior.

Build reverse-mode autodiff with saved-activation lifetimes, broadcast reductions,
view/alias gradients, gradient accumulation, detach and parameter mutation checks.
Inference must not allocate or maintain a training tape. Validate local gradients with
FP64 finite differences/adjoint identities, and compositions against a pinned reference.
Close a minimal native SGD step on a procedural tensor graph in this phase, including
repeated parameter updates. Use it to check the tape and trainable-parameter contract
before expanding graph execution. Full optimizer, RNG and recovery hardening remains
the Q4 milestone.

Choose MKL/oneDNN, cuBLAS/cuDNN/cuFFT or custom kernels using measured shape regimes.
SIMD, unsafe Rust, CUDA specialization and fusion are implementation tools with explicit
preconditions and fallbacks. Do not replace a faster vendor route merely to own a kernel.
Only claim ISA or GPU support actually tested on available hardware.

**Gate:** declared core coverage passes CPU/CUDA forward and backward checks, including
parameter/input gradients and repeated updates. Unsupported DSP/sequence derivatives
remain visible and block claims for those capabilities. Basic tensor/dense/convolution/
normalization/spectral gradients are required for the year-one training foundation.

### Q3: graph planning, scheduling and measured optimization

- Extend the graph to multiple inputs/outputs, typed values, explicit state and
  versioned parameters. Preserve operator semantics across eager/reference execution
  and lowered plans; add shape buckets and precise cache invalidation keys.
- Compute liveness and reuse arenas subject to alias, saved-gradient and asynchronous
  completion constraints. Retain traces only on request; independently verify reuse
  with adversarial overlapping lifetimes and changing inputs/parameters.
- Specialize layouts, contiguous regions, reductions and producer/consumer chains
  where profiles show useful work to remove. Account for fusion's register/scratch
  cost and preserve full subgraph outputs and gradients.
- Evaluate CUDA replay, launch reduction, asynchronous/pinned transfers, overlap and
  bounded workspace as separate interventions. Replays must honor pointer lifetimes,
  shape changes and parameter updates. Keep a checked uncaptured path.
- Define CPU thread pools, affinity and vendor threading without oversubscription.
  Separate immutable shared plans/weights from per-instance mutable scratch/state.
  Verify stream isolation, event ordering and repeated/concurrent executions.
- Plan forward, backward and update execution together. Add bounded activation
  recomputation only after gradient and memory-lifetime correctness. Introduce a
  compiler/code generator only when a measured bottleneck justifies its complexity.

**Gate:** identical declared semantics before/after planning, gradient agreement,
resource ceilings, memory accounting and independent performance confirmation for
affected hardware/workloads. Report preparation/compilation break-even separately;
one fast kernel does not establish a fast execution graph.

### Q4: training mechanics, state and foundation release

- Deliver SGD/AdamW, gradient clipping, accumulation, train/eval state and reproducible
  RNG streams. Parameter updates must invalidate or refresh packed inference weights.
- Checkpoint parameters, optimizer, RNG, counters, schema and relevant audio/execution
  configuration. Validate uninterrupted versus resumed synthetic training trajectories
  and reject incompatible or corrupted state explicitly.
- Add opt-in FP16/BF16/TF32 modes only with declared accumulation, casts, loss scaling,
  hardware support and error budgets. Keep strict FP32 controls and cost conversions.
  Quantization remains conditional research after the required floating-point core.
- Provide low-level chunk/process/flush/reset state for recurrent, convolution,
  resampling and overlap operations. Verify full/chunk equivalence where causal,
  partial tails, independent streams, bounded buffers and declared delay. These are
  runtime/DSP primitives, not an application streaming pipeline.
- Stabilize capability/error reporting, Rust module/parameter APIs, build feature
  combinations and supported SDK documentation. Add correctness and performance
  regression routes with recoverable sources and exact measured artifacts.

**Gate:** small procedural tensor/audio graphs train through native forward/backward/
update on Intel and NVIDIA, match the reference and resume correctly. Report separate
inference and training latency/throughput/memory. No task model, dataset pipeline or
claimed audio application is needed for this gate. Distributed/multi-GPU training and
additional hardware are later extensions after single-device fundamentals are verified.

## Performance work throughout the year

The goal is sustained, independently verified advantage over the strongest compatible
PyTorch execution mode on declared audio workloads. Building in Rust or completing a
feature list does not establish that advantage. "Maximum performance" means repeatedly
removing measured bottlenecks under fixed correctness/resource constraints, not a
claim that no faster implementation can exist.

Performance leadership must come from better kernels, less memory traffic and lower
execution overhead. The competing PyTorch path includes native kernels, vendor
libraries and compiler/runtime optimizations.
PyTorch's [compiler modes](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)
include autotuning and CUDA Graph options, and its
[tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)
describes operator fusion. Use the fastest compatible measured baseline in each
declared regime, while publishing every required mode and its setup cost. Keep the
version and configuration pinned for each campaign; do not rewrite historical results
when the reference stack changes.

### Where to seek an advantage

| Cost or regime | Foundation work | Evidence before claiming an improvement |
| --- | --- | --- |
| Short audio blocks and small batches | Prepared dispatch, fewer launches, fusion/replay, bounded scheduling overhead | Uninstrumented host/device latency, warmed allocations and relevant arrival-pattern tails |
| Long waveforms and spectra | Contiguous/coalesced traversal, SIMD, layout propagation and fewer intermediate buffers | Bytes moved, measured bandwidth, full output/gradient checks and memory accounting |
| Dense/convolution-heavy regions | Shape-based vendor/custom dispatch, packing reuse, tiling and explicit precision modes | Best compatible vendor/Torch control, packing break-even and input/parameter-gradient correctness |
| Training steps | Fused backward contractions/updates, saved-value lifetimes, arena reuse and optional recomputation | Full forward/backward/update latency, training trajectories and persistent/temporary storage budgets |
| Chunked audio primitives | Persistent bounded state, process/flush/reset rules and explicit delay | Full/chunk equivalence, independent streams, tail behavior and scheduling/resource limits |

For each priority workload, keep the remaining bottleneck, observed hardware/vendor
ceiling, current cost, next hypothesis and stopping condition visible. Declare
hot-path allocation, synchronization and workspace budgets before tuning. Prepared
paths should avoid repeated allocation, metadata upload, hidden host synchronization
and redundant copies; record unavoidable vendor costs instead of excluding them
silently. Use unsafe Rust, intrinsics and specialized CUDA when their preconditions
are checked and their measured benefit justifies the maintenance cost.

### Performance qualification sequence

Earn performance claims at increasing execution scopes. Keep every scope specific to
the tested hardware, precision, shape/layout catalogue and compatible reference modes.
The ambition is broad audio leadership; the sequence makes progress checkable without
promising universal superiority from a language choice or a completed API.

| Scope | Year-one target | Qualification evidence |
| --- | --- | --- |
| Operators | Competitive tensor, dense/convolution and spectral forward/backward routes across declared audio geometries | Per-case latency/throughput, allocations/workspace, numerical/gradient checks and fresh confirmation against the fastest compatible measured control |
| Reusable subgraphs | Remove avoidable layout conversions, intermediate traffic, launches and scheduling overhead | Complete forward/backward equivalence and measured gains over matched eager, compiled and captured controls where applicable; include preparation break-even |
| Native execution steps | Efficient inference and complete forward/backward/update steps on Intel and NVIDIA | Whole-step timing, changing parameters, steady-state resources and training agreement; an optimizer update or synchronization cannot be omitted from only one side |
| Foundation suite | Sustained advantage across the required catalogue without concealing weak cases | Independent repeated confirmation, explicit per-regime regressions, coverage gaps and resource limits; aggregate speedups alone do not qualify the suite |

For each losing or unmeasured required regime, retain a next experiment or an explicit
deferred status. Distinguish bandwidth-bound, compute-bound and overhead-bound costs
before selecting SIMD, custom kernels, fusion or replay. Use vendor implementations
when they remain the best measured route; framework ownership does not require
rewriting every BLAS, convolution or FFT implementation.

### Measurement and acceptance

1. **Cover the workload space.** Declare short/long windows, small/large batches,
   mono/stereo/multichannel, 16/44.1/48-kHz geometry, non-hop-aligned tails, silence,
   impulses, tones and procedural transients. Include tensor, operator, subgraph and
   existing complete-graph cases. Shapes and signal generators are fixtures, not apps.
2. **Match the execution contract.** Compare eager, compiled/captured and applicable
   vendor baselines with precision, shapes, algorithm policy, threads and I/O scope
   recorded. Include Rust preparation/packing/compilation when measuring cold work;
   include transfers/synchronization where the declared boundary requires them.
3. **Measure more than medians.** Record setup, first-call and warmed latency,
   throughput, allocations, persistent/scratch memory and process observations.
   Declare p95/p99/deadline tests with their arrival pattern when testing runtime
   scheduling. Sampling cannot prove an exact GPU peak or a real-time guarantee.
4. **Explain the remaining cost.** Measure effective bandwidth, compute utilization,
   layout/transfer work, launch gaps and synchronization. Compare with measured
   hardware/vendor ceilings to choose the next bottleneck; theoretical peak FLOPS
   alone is not an achievable graph-latency target.
5. **Run bounded experiments.** Profile separately, choose one mechanism, freeze the
   candidate/control, declare screening and confirmation budgets, keep every outcome
   and stop at the budget. Preserve the existing numerical gates and unresolved guards.
   Use the existing 5% useful-latency-gain and 5% per-case-regression rules in their
   original contexts; these are minimum useful-change gates, not the final ambition.
   New contexts need explicit gates before collection.
6. **Confirm and report gaps.** Use fresh counterbalanced confirmation, per-case and
   per-hardware results, resource ceilings and artifact recovery. Training and mixed
   precision need their own contracts. Approximate DSP needs declared signal-fidelity
   checks; later application claims still require licensed quality/listening evidence.

An optimization may be complete when its declared acceptance gate passes. A slow
regime remains an explicit issue with a next hypothesis; do not keep tuning one
microbenchmark indefinitely while required foundations remain missing. Preserve
measured regressions rather than hiding them behind a geometric mean.

## Year-one exit and later applications

The foundation checkpoint requires the tested storage/runtime contracts, published
operator/backward coverage, native synthetic training/resume, CPU/NVIDIA regression
suites, resource accounting and independently confirmed performance reports above.
All required gaps remain visible until closed. This is a defined framework milestone,
not a declaration that every possible low-level capability is finished.

Keep functional readiness and competitive readiness separate in the exit report:
operator/dtype/layout/backward coverage, native training/resume, resource behavior,
and latency/throughput against each compatible Torch mode all need their own status.
List every losing or unmeasured regime. A universal speedup or fixed multiplier is
not a first-year promise; broad, sustained audio performance leadership remains the
multi-year objective. A fast kernel, a fast eager comparison or a favorable aggregate
cannot close a missing training path or an unresolved performance gate.

After those gates pass and year one ends, reopen music processing, source separation
and synthesis together as the first application wave. Add complete models and licensed
quality fixtures to validate the foundation's generality, followed by other important
audio domains. Model bundles/importers, task data pipelines, DAW/C interfaces and
application examples belong to that later stage. Keep the original multi-year scope.

## Next-stage plan: comprehensive PyTorch leadership

Now that the strict-FP32 single-device MVP core is verified, the active competitive objective
(adjusted by user direction on 2026-11-23) is **comprehensive, end-to-end leadership over
PyTorch `2.14+` across Intel CPUs and NVIDIA GPUs**, covering **both training and inference**:
1. **Training:** Comprehensively beat **PyTorch `2.14+`** (`autocast(bfloat16/float16) +
   torch.compile + CUDA Graph` on GPU; single- and multi-threaded `Inductor + oneDNN/MKL` on CPU)
   on step throughput, peak host/device memory, and wall-clock `Time-to-Train (TTT)` to matched
   convergence quality.
2. **Inference:** Comprehensively beat **PyTorch `2.14+`** (`torch.inference_mode()` + Eager,
   `torch.compile(inductor)`, `autocast(bfloat16/float16)`, static KV-cache, and CUDA Graph on
   GPU; multi-threaded `Inductor + oneDNN/MKL` on CPU) on latency, `RTFx`/`tokens/s`,
   `p95`/`p99` tails, cold-start cost, and memory footprint across offline, chunked/streaming,
   and autoregressive audio workloads.

**Explicit Scope Boundary:** Comparisons and parity gates against other ML frameworks and
standalone C++ inference engines are **deferred** until comprehensive leadership over PyTorch is
achieved and independently confirmed across the Intel and NVIDIA strata.

Execute these four phases in strict priority order, enforcing a mandatory **"Measure Every Step"**
discipline: every single ticket (`A0–D5`) must record a controlled before/after measurement receipt
(cold/warm/p95/p99 latency, roofline bandwidth/TFLOPS, 0 hot allocations, exact memory breakdown,
and same-session paired PyTorch comparisons) before advancing.

The PyTorch competitive analysis and architectural positioning live in
[competitive-analysis-v1.md](competitive-analysis-v1.md). The full **25-ticket Work Breakdown
Structure (`A0–A6`, `B1–B7`, `C1–C6`, `D1–D5`)**, dependency DAG, module ownership, hardware
strata (`fa-cpu`, `L4`, `A100`, `G4`, `T4`), and per-step measurement protocol live in
[plan-phases-a-d.md](plan-phases-a-d.md).
The [benchmark contract](benchmarking.md#current-external-baseline) defines how to match
workloads and select PyTorch modes for both paths.

Six architectural rules govern Phases A–D:
1. **Unify mixed precision inside `autograd::Tape` (`A1`/`A5`):** Do not maintain a permanent
   split between general FP32 `Tape` and hardcoded BF16 trainers.
2. **Qualify uncommitted prototypes first (`A0`):** Verify, measure, and commit existing prototype
   files on Miri, Colab `fa-cpu`, and Colab `L4` (`compute-sanitizer` + PyTorch BF16 cross-checks)
   before starting new Phase A–D development.
3. **Precision order aligned to verified GPUs:** Deliver **`BF16 + TF32`** first (`sm_80`/`sm_89`/
   `sm_120`) and isolate **`FP16 + zero-sync GradScaler`** in `A6` (`sm_75`/`sm_89`). Defer
   training-time `FP8` and post-training weight-only quantized inference (`W8A16`/`W4A16`/INT8)
   until after comprehensive PyTorch leadership passes in Phase D.
4. **Vendor ceilings + custom audio/runtime bottlenecks:** Delegate dense GEMM (`M > 8`) to
   `cublasLt` and BF16/FP16 convolutions to cuDNN v9 Graph; own custom PTX/CUDA/SIMD kernels for
   audio-masked `mma.sync`+`cp.async` FlashAttention-2, `T_q=1` Split-K `FlashDecoding`, low-batch
   `M=1..8` BF16/FP16 GEMV, fused norm/SwiGLU/CE/RoPE/spectral kernels, NVRTC BF16 fusion, and
   repacked multi-core 2D CPU fusion.
5. **One model definition, two lowered runtimes (`TrainingTape` + `FrozenInferencePlan`):** Lower
   the same `Builder`/`nn::Module` graph into either a full mixed-precision training tape or a
   zero-autograd frozen inference engine (`C6`: Norm/BN folding, horizontal `QKV`/`Gate+Up`
   fusion, load-time weight repacking, single-buffer contiguous byte arena, `mmap` weights, static
   `d_kv_len` KV-Cache, and Cargo feature-trimmed minimal runtime) to beat PyTorch in both
   training and inference.
6. **Measure every step:** No ticket closes without a verified Step Measurement Receipt (before/after
   delta, median/p95/p99, bandwidth/TFLOPS, exact memory, Miri/`compute-sanitizer`, and paired
   PyTorch training/inference baselines).

### Phase A (Priority 1): Explicit Mixed Precision (`BF16`/`FP16`/`TF32`) and Tensor Core Stack

*Why first:* On modern NVIDIA GPUs (`A100 sm_80`, `L4 sm_89`, `G4 sm_120`), PyTorch with
`torch.amp.autocast(bfloat16)` and Tensor Core FlashAttention-2 runs 4–8× faster than strict FP32.
Closing this gap is required for real-world GPU training and inference leadership over PyTorch.

| Ticket | Work Package | Primary Deliverable |
| --- | --- | --- |
| `A0` | Uncommitted prototype audit & gate | Verify, measure, and commit existing Phase A/B prototype code across Miri, Colab `fa-cpu`, and Colab `L4` (`compute-sanitizer` + PyTorch BF16 baseline) |
| `A1` | Unified typed storage, frozen weights & mixed `Tape` | Explicit `DType::{F32, Bf16, F16}` on storage/views/`SavedValue`, differentiable `Cast`, zero-tracking `FrozenWeight` layout abstraction |
| `A2` | `cublasLt` TC `Matmul`/`Bmm`, epilogues & low-batch GEMV | Heuristic-cached `cublasLtMatmul` (`BF16`/`TF32`), fused Bias/SiLU/GELU epilogues, and dedicated `M=1..8` warp-reduced BF16 GEMV for low-batch inference |
| `A3` | cuDNN v9 Graph `BF16`/`TF32` Conv1d/2d & Transpose | Channels-last (`NHWC`/`NLC`) & `NCHW`/`NCL` Tensor Core forward/`dX`/`dW` graphs + inference epilogues with determinism filter and workspace cap |
| `A4` | High-throughput `BF16` FlashAttention-2 | `cp.async`-pipelined `mma.sync` kernel (`D ∈ {32,64,80,96,128,256}`, `T=256..4096`), zero-LSE inference path, optional cuDNN SDPA ceiling route |
| `A5` | Mixed-precision NVRTC fusion, `AdamW` & CPU `BF16` | BF16 I/O + FP32-acc NVRTC fusion, unified FP32-master/BF16-mirror `AdamW` in single CUDA Graph, Intel oneDNN BF16/AMX CPU route |
| `A6` | IEEE `FP16` Tensor Core & zero-sync `GradScaler` | Complete FP16 dense/GEMV/conv/SDPA routes + device-side overflow skip inside CUDA Graph replay (`T4 sm_75` + `L4 sm_89`) |

- **Phase A Gate:** Matched 50-step training trajectories vs PyTorch `autocast(bfloat16)`,
  **≥ 1.15× geomean** GPU training win against PyTorch `2.14.1` `bf16 + torch.compile + CUDA Graph`
  on Colab `L4`/`A100`, `M=1..4` BF16 GEMV at ≥ 80% HBM bandwidth beating PyTorch low-batch
  inference, 0 sanitizer errors/hazards, and Step Measurement Receipts for every ticket `A0–A6`.

### Phase B (Priority 2): Audio-Model Operator, Dtype, and Differentiable DSP Closure

*Why second:* Real audio architectures (discrete neural codecs like DAC/EnCodec/Mimi, source
separators like HTDemucs, streaming enhancers like GTCRN, and speech/music Transformers)
require operators, dtypes, and stateful zero-copy DSP beyond pure float dense/conv/STFT blocks.

| Ticket | Work Package | Primary Deliverable |
| --- | --- | --- |
| `B1` | Complete typed storage & `Tape` aux inputs | `U32`/`Bool`/`Complex64` storage & views, differentiable complex math, typed non-differentiable `IndexInput`/`MaskInput` on `Builder` |
| `B2` | Token & codebook primitives on `Tape` | `Embedding`, arbitrary-axis `Gather`, `IndexSelect`, and deterministic zero-atomic `ScatterAdd` (RVQ / discrete audio tokenizers) |
| `B3` | Arbitrary audio masks, Sequence Packing, device-offset `RoPE` & block-sparse `Sdpa` | `2D`/`4D` `Bool`/bias masks + block-skipping sliding/chunked/`cu_seqlens` packed audio masks; interleaved/NTK/YaRN `RoPE` with per-segment reset & device `d_pos_offset` |
| `B4` | Differentiable `GRU` & `LSTM` backward | Full `dX`/`dH0`/`dW` on CPU (MKL/AVX2) and CUDA (cuDNN v9 RNN + fused small-hidden `H<=192` kernel for GTCRN) on `Tape` |
| `B5` | Pooling, `Tape` `Philox Dropout` & `Prod` | `AvgPool1d/2d`, `MaxPool1d/2d`, CUDA-Graph-safe device-counter `Philox4x32-10` `Dropout` on `Tape`, and `Prod` reduction |
| `B6` | Audio loss suite & Fused Chunked `Linear+CrossEntropy` | `Tape` fused `CrossEntropy` + chunked `Linear+CE` (0 `[B*T,V]` logits allocation), `MultiResolutionStftLoss` (1-CUDA-Graph safe), `LogMel`, `SiSdrLoss` |
| `B7` | Stateful ring-buffer chunked differentiable DSP | Zero-`memmove` ring-buffer `process`/`flush`/`reset` contracts for causal `ChunkedConv1d`, `PolyphaseResample`, and `StreamingStft`/`StreamingIstft` |

- **Phase B Gate:** Every ticket passes FP64/PyTorch cross-checks, Miri (0 UB), `compute-sanitizer`
  (0 errors/hazards), full-vs-chunk DSP equivalence (`max_abs <= 1e-5` FP32), beats compatible
  PyTorch baselines on CPU and GPU, and records its Step Measurement Receipt (`B1–B7`).

### Phase C (Priority 3): System Scale, Static Memory Arenas, Multi-Core CPU, and Frozen Inference

*Why third:* Comprehensive leadership over PyTorch requires dynamic-length audio bucketing
without recompilation stalls, zero-fragmentation static arenas, selective rematerialization,
multi-core CPU parallelism, long-context SDPA/KV-cache decode, and zero-autograd frozen inference
lowering.

| Ticket | Work Package | Primary Deliverable |
| --- | --- | --- |
| `C1` | Single-buffer contiguous Arena & Buffer Donation | Single contiguous byte arena sliced by aligned static offsets on CPU & CUDA, explicit Buffer Donation, and exact componentized peak VRAM reporting |
| `C2` | Variable-length audio bucketing, Sequence Packing & shared-arena graphs | `BucketedTapeCache` sharing one parameter store and one max-bucket byte arena + `cu_seqlens` audio sequence packing (`< 5%` padding waste, `< 20 us` switch) |
| `C3` | Policy-based selective activation checkpointing (`Remat`) | `Builder::checkpoint_scope(RematPolicy::{SaveDotsOnly, FullBlock})` inside static arena & single CUDA Graph (≥ 45% peak activation VRAM reduction at ≤ 10–25% time cost) |
| `C4` | Spin-futex multi-core Intel CPU pool & weight repacking | Low-jitter spin-then-futex thread pool coordinated with MKL/oneDNN OpenMP, load-time microkernel weight repacking (`pack_b`), parallel audio kernels |
| `C5` | Long-context SDPA, Split-K `FlashDecoding`, `d_kv_len` & async H2D | `T=2k..16k` O(T) SDPA, `T_q=1` Split-K decode + static/ring KV-Cache with device `d_kv_len` in 1 CUDA Graph, pinned async `copy_stream` staging |
| `C6` | Frozen inference lowering, `mmap` weights & minimal runtime | `FrozenInferencePlan` (Norm/BN folding, horizontal `QKV`/`Gate+Up` fusion, epilogue sinking, repacking, 64B-aligned `mmap` + `safetensors`, feature-trimmed build) |

- **Phase C Gate:** 0 leaks across 1,000 bucket switches, ≥ 45% remat VRAM savings vs uncheckpointed
  baseline, monotonic multi-core CPU scaling beating multi-threaded PyTorch across ≥ 90% of suite,
  `nsys`-verified ≥ 85% H2D overlap, and Step Measurement Receipts confirming clear wins over
  PyTorch `2.14.1` strongest compatible training and inference modes (`Eager` / `torch.compile` /
  `CUDA Graph`) across `C1–C6`.

### Phase D (Priority 4): Full-Model PyTorch Leadership Proofs (Train & Infer), Ergonomics, and Multi-GPU

*Why fourth:* Single-block microbenchmarks prove kernel efficiency, not end-to-end training
viability or comprehensive whole-model victory over PyTorch.

| Ticket | Work Package | Primary Deliverable |
| --- | --- | --- |
| `D1` | Ergonomic `nn` API, homogeneous layer-plan sharing, schedulers, grad accum & `Checkpoint v2` | `Module`/`ParamRegistry`, $O(1)$ homogeneous layer-plan sharing, device-scalar LR schedulers, `grad_accum_steps`, `Checkpoint v2` + `FrozenInferencePlan` export |
| `D2` | Full Model 1: Multi-layer Causal Audio Transformer | 12-layer (~50–100M) discrete audio Transformer on unified `Tape` + `FrozenInferencePlan` (beat PyTorch on training tok/s, VRAM & wall-clock `Time-to-Train`, plus autoregressive decode latency/`tokens/s`/`p99`) |
| `D3` | Full Model 2: Multi-scale Spectral Enhancement / Separation U-Net | Trainable GTCRN + deeper multi-scale Spectral U-Net (beat PyTorch `bf16+compile` on training throughput & `Time-to-Train`, and PyTorch `inference_mode+compile+CUDA Graph` on offline/chunked inference) |
| `D4` | Full Model 3: Neural Audio Codec / Vocoder | DAC/EnCodec/Mimi-class codec + RVQ + `MultiResolutionStftLoss` + streaming chunked inference (beat PyTorch on training throughput & `Time-to-Train`, plus offline/streaming inference latency & `RTFx`) |
| `D5` | Multi-GPU Data Parallelism (`DDP` via NCCL) | Feature-gated (`--features nccl`) bucketed async `ncclAllReduce` overlapped with backward; gated strictly after single-device PyTorch proofs pass |

- **Phase D Gate:** All three full-scale audio architectures (1) converge within `<= 1.0%` loss
  and **beat PyTorch `2.14.1` `bf16 + compile + cudagraph` on step throughput, peak VRAM, AND
  wall-clock `Time-to-Train (TTT)` to target quality** (including preparation, compilation, data
  staging, and sequence packing), (2) **comprehensively beat PyTorch `2.14.1` inference modes**
  (`inference_mode` + `bf16`/`fp32` + `torch.compile` + CUDA Graph + static KV-cache) on latency,
  `RTFx`/`tokens/s`, `p99`, and memory across Intel CPU and NVIDIA GPU, (3) pass exact checkpoint
  resume, and (4) achieve ≥ 85% weak scaling on multi-GPU DDP (`D5`) beating PyTorch DDP, with
  archived Step Measurement Receipts for every ticket.

### Deferred Post-PyTorch Scope

After Phases A–D confirm comprehensive victory over PyTorch on both Intel CPUs and NVIDIA GPUs,
subsequent milestones may reopen:
- Benchmarking and parity comparisons against other ML frameworks and standalone C++ inference
  engines;
- Post-training weight-only quantized inference (`W8A16`/`W4A16` CUDA GEMV and CPU `VNNI`/`AMX`
  INT8/INT4);
- Training-time `FP8` precision routes;
- Application-wave audio model suites and deployment interfaces.

Historical baselines, failed/inconclusive screens, and original `sm_86` champions remain immutable
regression anchors throughout all four phases.
