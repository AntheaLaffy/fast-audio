# fast-audio

A performance-first Rust audio deep learning framework for Intel CPUs and NVIDIA GPUs.

Explore the [project website](https://anthealaffy.github.io/fast-audio/) in English, Chinese or Japanese for installation, tutorials, API signatures and interactive benchmarks. The [website guide](website/README.md) covers building and hosting the site.

## Use the Rust crate

Try the first preview with `cargo add fast-audio@=0.1.0-alpha.1`. The portable CPU build needs Rust 1.87 or later. Select vendor features when the corresponding SDK is installed. The [compatibility policy](https://github.com/AntheaLaffy/fast-audio/blob/main/docs/api-stability.md) and [release process](https://github.com/AntheaLaffy/fast-audio/blob/main/docs/releasing.md) describe how a preview qualifies for promotion.

**Outrun PyTorch. Measure every step.**

Year one is dedicated to the framework foundation: tensors and memory, audio operators,
autodiff, graph execution, training mechanics and verified Intel/NVIDIA performance.
New model integrations and application delivery follow that foundation. Music processing,
source separation and audio synthesis remain the first later application priorities;
speech and other audio domains also matter. See the [foundation-first roadmap](docs/roadmap.md).
The first-year checkpoint is a coherent single-device training and inference stack,
with measured performance against compatible eager, compiled and captured PyTorch
paths. Calendar milestones do not bypass incomplete foundation gates.
Performance qualification progresses from operators to reusable subgraphs and complete
inference/training steps, with per-regime losses and unmeasured cases kept visible.

## Current status (2026-11-23)

`fast-audio` has completed its **strict-FP32, single-device, static-shape audio training MVP
core** on both Intel CPUs and NVIDIA GPUs. See [progress.md](docs/progress.md),
[operator-coverage.md](docs/operator-coverage.md), and the [foundation roadmap](docs/roadmap.md)
for exact verified scope, raw numbers, and remaining gaps.

Implemented and verified across portable CPU + Miri (`0` UB), Colab Intel CPU (`--features
mkl,onednn`, 132 tests + 16 doctests, 35/35 PyTorch `2.14.1` strict-FP32 cross-checks), and
Colab NVIDIA L4 (`CUDA_ARCH=89 --features cuda,cudnn`, 195 tests + 17 doctests, 35/35 PyTorch
cross-checks, 12/12 `compute-sanitizer` `memcheck`/`racecheck`/`synccheck` runs with `0` errors
and `0` hazards):

- **Checked storage, signed-stride views, and retained training lifetimes:** Arbitrary-rank
  signed-stride `Layout`, `TensorView`/`DeviceView`, opt-in `Tracked<T>` mutation versions,
  `Shared<T>` / `SavedValue<T>`, and prepared `ViewBackward` adjoints.
- **Prepared FP32 forward & local-VJP operator suite (CPU & CUDA):** Unary/binary pointwise
  with broadcast contraction, ordered reductions (`Sum`/`Mean`/`Max`/`Min`/`LogSumExp`),
  `LayerNorm`/`RmsNorm`/`GroupNorm`, `Softmax`/`LogSoftmax` (`run` and stateless
  `run_from_output`), `Matmul`, arbitrary-rank broadcast `Bmm`, MHA/GQA causal/non-causal
  `Sdpa` (L1-tiled AVX2+FMA CPU FlashAttention-2; CUDA 64×64 SIMT SDPA GEMM + `float4` softmax
  + `O(T)` FlashAttention-2 fallback), grouped/strided/dilated `Conv1d`/`Conv2d` and
  `ConvTranspose1d`/`ConvTranspose2d`, structural `Pad` (`Constant`/`Reflect`/`Replicate`) and
  `Cat`, and differentiable `Stft`/`CenteredStft`/`Istft`.
- **Native reverse-mode tape, fusion, optimizers, and checkpoints:** Exact activation liveness
  slot reuse, 2D L1 cache-blocked CPU whole-step fusion + runtime AVX2+FMA kernels, NVRTC
  strict-FP32 CUDA whole-step fusion, decoupled `AdamW` with deterministic FP64 global L2
  gradient clipping, zero-recapture dynamic-LR single-CUDA-Graph training
  (`step_scalar_adamw_captured`), and FNV-1a checksummed cross-device `Checkpoint` save/load.
- **Strict-FP32 benchmark status vs PyTorch `2.14.1`:** Clean **11/11** sweep on Colab Intel
  CPU (1 thread, 1.70× geomean vs best / 2.66× vs Eager) and **9/9** vs Eager (2.79× geomean) /
  **8/9** vs best (1.84× geomean, including 1.20× over PyTorch CUDA Graph on Audio Transformer
  + AdamW and 7.47× on Spectral Vocoder + AdamW) on NVIDIA L4.
- **Explicit non-claims & next priorities:** Only strict FP32 is supported today — mixed
  precision (`BF16`/`FP16`/`TF32`) and Tensor Core GEMM/Conv/FlashAttention, integer/bool/complex
  dtypes, `Embedding`/`Gather`/`Scatter`, `GRU` backward, `RNG`/`Dropout`, arbitrary attention
  masks, variable-length shape bucketing, multi-core CPU scaling, and full-model convergence
  validation are the active next-stage goals in [docs/roadmap.md](docs/roadmap.md).

The existing [GTCRN speech-enhancement reference](docs/gtcrn.md) remains a regression workload
whose complete offline waveform inference path runs natively on CPU/CUDA. Historical baseline
and characterization records ([gtcrn-characterization-results-v1.md](docs/gtcrn-characterization-results-v1.md),
[pointwise-characterization-results-v2.md](docs/pointwise-characterization-results-v2.md),
[training-characterization-results-v1.md](docs/training-characterization-results-v1.md),
[cpu-prepared-backward-results-v1.md](docs/cpu-prepared-backward-results-v1.md)) remain preserved
and immutable.

## Build and run

Validated on Linux x86-64 with Rust `1.98–1.100`, Intel oneMKL `2026.0`, oneDNN `3.11.3`,
CUDA Toolkit `12.4` / `13.0`, and cuDNN `9.x` (tested on RTX 3060 `SM 8.6` and NVIDIA L4
`SM 8.9`). The default build needs no Intel or NVIDIA SDK and uses a portable CPU reference.

```sh
cargo test --locked
cargo test --locked --features mkl
cargo build --locked --release --features mkl --bin fa-bench
target/release/fa-bench --backend mkl --case conv-audio --threads 1
```

The `mkl` feature needs oneMKL headers and `libmkl_rt` on the runtime loader path.
`MKLROOT` defaults to `/opt/intel/oneapi/mkl/latest`. Thread counts are explicit;
GEMM temporarily sets and restores oneMKL's calling-thread setting.

`onednn` adds OpenMP oneDNN convolution with packed immutable parameters. Build with
`--features mkl,onednn` and use `--backend intel` to evaluate shape-based inference
dispatch, or `--backend onednn` to force oneDNN convolution. The Intel candidate improves
convolution, but remains opt-in because its full-suite confirmation was inconclusive.

For NVIDIA, install CUDA development headers/libraries, a compatible host C++ compiler,
and an NVIDIA driver. The validated target is an RTX 3060 with CUDA Toolkit 12.4.
`nvcc` must be on `PATH`; `NVCC`, `CUDA_ROOT`, and `CUDA_HOST_CXX` can override discovery.

```sh
CUDA_ARCH=86 CUDA_HOST_CXX=g++-13 cargo test --locked --features cuda
CUDA_ARCH=86 CUDA_HOST_CXX=g++-13 cargo build --locked --release --features cuda --bin fa-bench
target/release/fa-bench --backend cuda --case stft-stereo
```

For the validated prepared NVIDIA convolution route, install cuDNN 9 development headers
and build with `--features cudnn` (which includes `cuda`):

```sh
CUDA_ARCH=86 CUDA_HOST_CXX=g++-13 cargo build --locked --release --features cudnn --bin fa-bench
target/release/fa-bench --backend nvidia --case conv-audio
```

The `nvidia` route uses a bounded cuDNN graph for temporal/grouped convolution and keeps
direct cuBLAS pointwise GEMM. It passed the seven-case regression contract against this
project's original CUDA implementation. `--backend cudnn` forces the graph route.

`CUDA_ARCH` defaults to `86`; select the actual target GPU when rebuilding. SASS and PTX
are both emitted for that compute capability. SDK features are explicit: a missing
vendor library produces a build error rather than a silent fallback.

## Rust API

```rust
use fast_audio::{Tensor, cpu::Cpu};

fn main() -> Result<(), fast_audio::Error> {
    let cpu = Cpu::Reference; // Cpu::mkl(1)? with the `mkl` feature
    let a = Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.])?;
    let b = Tensor::from_vec([3, 1], vec![1., 0., -1.])?;
    let mut output = Tensor::zeros([2, 1])?;
    cpu.matmul(&a, &b, &mut output)?;
    assert_eq!(output.data(), &[-2., -2.]);
    Ok(())
}
```

Host tensors and device tensors have distinct types. Outputs are caller-owned. Convolution
and FFT plans own reusable workspace. CUDA operations enqueue onto one owned stream;
downloads and `Cuda::synchronize` wait for completion and surface asynchronous errors.
See the [execution contract](docs/architecture.md) before adding an operator.

`Layout` prepares signed strides, slices, permutations, broadcasts and compatible
no-copy reshapes independently of storage. `Tensor::view(&layout)` / `view_mut`
bind without allocation; mutable views require a non-overlap proof. Prepared
`cpu::CopyPlan` / `cuda::CopyPlan` move logical elements with exact FP32 bits and no
warmed Rust allocation. See the [layout contract](docs/tensor-layout-v1.md) and
[tested host example](src/copy.rs). Dense scalar/empty tensors are supported; math
and graph capabilities retain the explicit limits documented there.

`Tracked<Tensor>` / `Tracked<DeviceTensor>` opt into storage identity and mutation
versions without adding metadata to ordinary inference tensors. `write()` invalidates
saved versions before granting mutable access; `check_version()` rejects a changed
or different owner. These are [training-storage foundations](docs/storage-versions-v1.md),
not an autodiff tape or a GPU completion check.

`Shared<T>` adds opt-in shared host/device ownership with nonblocking read/write
guards. `SavedValue<T>` retains that owner, a prepared `Arc<Layout>` and its version
without copying payloads; reads reject later mutation. See the
[retained-value contract](docs/saved-values-v1.md) for borrowing, lifetime and future
view-gradient rules. Ordinary inference tensors use their existing allocation model.

`cpu::ViewBackward` / `cuda::ViewBackward` prepare the adjoint of a read layout.
They support overwrite/accumulate modes, analytical broadcasts and budgeted arbitrary
overlap, with fixed FP32 reduction trees. The [view-backward contract](docs/view-backward-v1.md)
defines numerical order, resources and gradient checks. This primitive does not yet
provide an autodiff tape or other operator backward families.

With `cudnn`, `Cuda::with_convolution_policy(0, ConvolutionPolicy::RequireDeterministic)`
excludes cuDNN convolution engines marked nondeterministic. The default remains
`AllowNonDeterministic`. Prepared convolutions and graph-contained convolutions inherit
the session policy and expose actual selected engine metadata. See the
[policy contract](docs/cudnn-repeatability-v1.md) for its scope and validation budget.

`cpu::PreparedConv1d` snapshots parameters; `onednn::Conv1d` exposes the packed oneDNN
plan directly. `cuda::Conv1d::planned` selects cuDNN or direct pointwise GEMM, with explicit
bias presence and workspace limit. Existing dynamic-parameter plans remain available.

`cpu::Conv2d` and `cuda::Conv2d` (with `cudnn`) snapshot NCHW convolution parameters.
`Conv2dSpec` covers groups, stride, dilation and transposed output padding. Forward
weights are OIHW; transposed weights are IOHW. Pointwise plans use direct GEMM; other
Intel/cuDNN plans use vendor convolution. CPU builds without oneDNN use reusable
im2col/col2im scratch. Preparation and parameter uploads stay outside execution.

`cpu::Gru` and `cuda::Gru` (with `cudnn`) prepare one-layer, batch-first reset-after
GRUs with explicit initial/final states and immutable packed parameters. Bidirectional
output stays in chronological order. CUDA exposes algorithm/layout controls through
`GruOptions`; warmed execution includes any internal layout transforms.

To capture the real model and check its GRU layers:

```sh
uv pip install --python .venv/bin/python -r models/gtcrn/requirements-reference.txt
taskset -c 0 .venv/bin/python tools/gtcrn_reference.py --device cpu \
  --output-dir benchmarks/results/local-gtcrn-reference
cargo build --locked --release --features mkl --example gru-reference
taskset -c 0 .venv/bin/python tools/check_gru.py --backend mkl \
  --binary target/release/examples/gru-reference \
  --reference benchmarks/results/local-gtcrn-reference \
  --output-dir benchmarks/results/local-gtcrn-gru
```

The reference downloads only the hash-locked checkpoint when absent. Install the CPU
PyTorch environment below first. For NVIDIA use a CUDA PyTorch build, `--device cuda`,
build the example with `--features cudnn` and choose `--backend cudnn` in the checker.
The fixed input fixtures cover numerical behavior; speech quality remains unevaluated.

To export named FP32 parameters and validate the complete native model:

```sh
taskset -c 0 .venv/bin/python tools/gtcrn_reference.py --device cpu --capture-model \
  --warmup 1 --samples 1 --iterations 1 \
  --output-dir benchmarks/results/local-gtcrn-model-reference
cargo build --locked --release --features mkl,onednn --example gtcrn-infer
taskset -c 0 .venv/bin/python tools/check_gtcrn.py --backend mkl \
  --binary target/release/examples/gtcrn-infer \
  --reference benchmarks/results/local-gtcrn-model-reference \
  --output-dir benchmarks/results/local-gtcrn-native
```

For NVIDIA, capture with `--device cuda`, build with `--features cudnn`, and select
`--backend cudnn`. PyTorch exports weights and supplies the numerical reference;
the Rust executable runs STFT, the complete network and inverse STFT through native
plans. The checker verifies every parameter/activation hash and all four final
spectra/waveforms plus 60 model-block outputs. Its native-only timing is exploratory.

`graph::Builder` prepares fixed-shape operation sequences with explicit layouts,
broadcasting, linear projections, convolution, recurrent state, normalization and
complex audio operations. `cpu::Graph` / `cuda::Graph` own intermediate storage and
parameter snapshots. `models::gtcrn::build` lowers named tensors to such a program;
callers retain ownership of spectrum input/output. See the [graph contract](docs/architecture.md#prepared-inference-graphs).

Add `--capture-conv` to a fresh reference run to save every convolution input and
parameter. Build `--example gtcrn-stages --features mkl,onednn` and run
`tools/check_gtcrn_stages.py` with the same `--binary`, `--reference`, `--backend mkl`
and a new `--output-dir` to check 88 actual convolution instances and four spectral
cases. On NVIDIA, build with `cudnn` and choose `--backend cudnn`. The checker validates
isolated stages against captured activations. The complete-model checker above connects
the native stages and compares its final enhanced audio.

For waveform reconstruction, validate centered geometry with `StftSpec::centered()`
and prepare `cpu::CenteredStft` / `cpu::Istft` or their `cuda` equivalents. Both use
caller-owned buffers and a fixed waveform length. The inverse preserves its spectrum
input and rejects gaps in the overlap envelope. See the [audio contract](docs/architecture.md#centered-audio-reconstruction)
and [complete-output example](examples/audio-roundtrip.rs).

`Stft::with_window`, `CenteredStft::with_window` and `Istft::with_window` snapshot a
finite host tensor of shape `[fft_size]` on either backend. `new` defaults to periodic
Hann. Use the same coefficients for analysis and synthesis; GTCRN uses square-root Hann.

```sh
cargo build --locked --release --features mkl --example audio-roundtrip
.venv/bin/python tools/check_audio.py --binary target/release/examples/audio-roundtrip \
  --backend mkl --output-dir benchmarks/results/local-audio-mkl
```

Build with `--features cuda` and use `--backend cuda` for the device path. This check
compares complete spectra, reconstructed waveforms and modified-spectrum inverses
with PyTorch; it does not measure latency.

## Compare with PyTorch

PyTorch is a measurement dependency only. It is not linked into the Rust library.

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python tools/compare_torch.py --backend mkl --threads 1 \
  --blocks 7 --iterations 50 --warmup 20 \
  --output-dir benchmarks/results/local-cpu-run
```

Use a CUDA-enabled PyTorch environment and `--backend cuda` for GPU comparisons.
Every measured case checks the entire output, then records wall latency, CUDA event
latency where applicable, plan construction time, raw samples, environment versions,
binary/source hashes, and a source archive. Use a new output directory for each run.

The [measurement contract](docs/benchmarking.md) explains matching precision, synchronization,
allocation differences, vendor-library versions, and the limits of these operator results.
Initial results and remaining gaps are tracked in [progress](docs/progress.md).
Start code navigation at [CODEMAP.md](CODEMAP.md).

For the declared complete-model characterization, use the exported model reference
and a release `gtcrn-infer` binary:

```sh
taskset -c 0 .venv/bin/python tools/gtcrn_benchmark.py \
  --binary target/release/examples/gtcrn-infer \
  --reference benchmarks/results/local-gtcrn-model-reference --stratum intel \
  --torch-mode eager --output-dir benchmarks/results/local-gtcrn-characterization
```

Build with `mkl,onednn` for Intel or `cudnn` for NVIDIA. The collector keeps a read-only
SHA-named binary copy, validates every paired output and records fresh-process completed
pipeline latency. NVIDIA has named eager and CUDA graph baselines, including
`--stratum nvidia --torch-mode cuda-graph-prepared-inverse` for the inverse with its fixed
envelope checked during preparation. The unmodified inverse failed the capture probe;
its result is preserved in the characterization documentation.
Separate memory replays sample the worker process group; those samples do not enter
latency summaries. See the [declared scope and limits](docs/gtcrn-characterization-v1.md).
