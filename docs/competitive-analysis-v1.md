# PyTorch Competitive Analysis and Architectural Positioning (v1)

**Date:** 2026-11-23  
**Scope:** Focused competitive analysis of **PyTorch `2.14+`** across Intel CPUs and NVIDIA GPUs for audio deep learning (both training and inference), documenting PyTorch's strongest execution modes, structural audio bottlenecks, and the systems/kernel architecture `fast-audio` uses to comprehensively beat PyTorch. Comparisons against other ML frameworks and standalone C++ inference engines are explicitly deferred.  
**Related:** [target.md](../target.md), [roadmap.md](roadmap.md), [plan-phases-a-d.md](plan-phases-a-d.md), [progress.md](progress.md), [operator-coverage.md](operator-coverage.md), [architecture.md](architecture.md), [colab-hardware-evaluation.md](colab-hardware-evaluation.md).

---

## 1. Executive Summary and PyTorch-First Mandate

`fast-audio` has completed its strict-FP32 single-device training and inference MVP ([progress.md](progress.md)). Per the 2026-11-23 objective adjustment, the active mandate is **unambiguous and single-target**:

- **Comprehensively Beat PyTorch `2.14+` First:** Outperform the strongest compatible PyTorch execution mode across **both training and inference** on **Intel CPUs** and **NVIDIA GPUs** — covering strict FP32 and mixed precision (`BF16`/`FP16`/`TF32`), isolated operators, reusable subgraphs, streaming/chunked audio DSP, variable-length audio batches, and full multi-layer audio models — on latency, throughput (`steps/s`, `tokens/s`, `RTFx`), tail latency (`p95`/`p99`), peak host/device memory, and wall-clock `Time-to-Train (TTT)` to matched convergence quality.
- **No Cross-Framework Distractions For Now:** Benchmarking and parity gates against other ML frameworks or standalone C++ inference engines are **deferred** until comprehensive PyTorch leadership is confirmed across Phases A–D.
- **Execution Discipline (`Measure Every Step`):** Every ticket (`A0–D5` in [plan-phases-a-d.md](plan-phases-a-d.md)) requires a controlled before/after **Step Measurement Receipt** anchored against the strongest compatible PyTorch `2.14+` mode on the same hardware session.

### Why Rust + `nvcc`/NVRTC + Vendor Libraries Can Comprehensively Beat PyTorch

`fast-audio` matches PyTorch's hardware ceilings on dense vendor primitives while eliminating PyTorch's Python/ATen runtime overhead, graph-capture breaks, dynamic-shape recompilation stalls, and memory allocator fragmentation:

- **Device execution:** All GPU kernels are native CUDA C++17 and inline PTX (`mma.sync`, `cp.async`) compiled by `nvcc` (`CUDA 13.0`), runtime `NVRTC`, and direct C ABI links to `cublasLt`, `cuDNN 9.14`, and `cuFFT` ([build.rs](../build.rs)). Where PyTorch hits hardware ceilings via `cublasLt` or cuDNN v9, `fast-audio` calls the exact same vendor engines directly with zero Python/ATen dispatch and tighter workspace/epilogue control.
- **Host execution:** Rust compiles through LLVM to native x86-64 machine code with strict `&mut` aliasing guarantees (`noalias`), static pre-planned buffers, runtime AVX2/AVX-512/oneMKL/oneDNN dispatch, and 2D L1 cache-blocked audio micro-tiling — avoiding interpreter dispatch, allocator churn, and thread-wakeup jitter.

---

## 2. Audio Workload Taxonomy and Physical Bottlenecks

General deep learning workloads in PyTorch are tuned primarily for fixed-length large-batch dense compute. Audio deep learning spans three physically distinct regimes that expose structural bottlenecks in PyTorch:

| Audio Regime | Representative Architectures | Primary Physical Bottlenecks | Where PyTorch Loses |
| --- | --- | --- | --- |
| **1. Full-Sequence Training & Prefill** | Audio Transformers / Conformers, Multi-scale Spectral U-Nets, Neural Codecs (DAC / EnCodec / Mimi) | BF16/TF32 Tensor Core throughput (`GEMM`, `Conv1d/2d`, `SDPA`), activation VRAM on long sequences (`T = 1k..16k`) & large token codebooks (`[B*T, V]`), **30–60% padding waste on variable-duration audio clips** | Naive padding wastes up to half of GPU FLOPs; variable durations trigger `torch.compile` recompilation stalls or break CUDA Graph Trees; `torch.istft` invalidates CUDA Graphs and blocks Inductor fusion |
| **2. Autoregressive Decode & Real-Time Low-Batch Streaming** | Audio token decode (`BS = 1..4`, `T_q = 1`), 48 kHz real-time enhancement (`hop = 256` -> `5.33 ms` frame budget) | **Memory bandwidth bound (HBM / DDR)** reading weights once per token/frame; host launch latency (20–40% of step time without CUDA Graphs); OS thread wakeup jitter (20–50 us on CPU) | Standard GEMM underutilizes Tensor Cores at `M = 1`; Python/ATen dispatch destroys microsecond frames; generic thread pools stall on OS sleep/wake; dynamic KV lengths complicate graph capture |
| **3. Stateful Spectral & Causal DSP Pipelines** | GTCRN, Streaming Codecs, Polyphase Resamplers, Multi-Resolution STFT Loss | Kernel launch gaps across many small spectral/pointwise stages, ring-buffer memory movement (`memmove`), small-hidden recurrent loops (`GRU` with `H <= 192`), CPU L1/L2 cache thrashing | Separate eager launches per FFT/window/mask/OLA stage; lack of zero-`memmove` stateful chunk contracts; CPU Inductor omits 2D L1 audio micro-tiling |

---

## 3. Deep Dive: PyTorch `2.14+` Execution Modes to Beat

To make every performance claim watertight, `fast-audio` never cherry-picks slow PyTorch Eager baselines when a stronger compatible PyTorch mode exists. Every benchmark stratum pairs against **all compatible PyTorch `2.14+` execution modes**:

### 3.1 GPU Training Modes
1. **Eager (`fp32` / `autocast(bfloat16)` / `autocast(float16) + GradScaler`):** Reference numerical and baseline throughput control.
2. **`torch.compile(backend="inductor")`:** Dynamo graph capture + AOTAutograd + TorchInductor Triton kernel fusion + `cublasLt` / cuDNN v9 / FlashAttention-2 / FlexAttention dispatch.
3. **`torch.compile(mode="reduce-overhead")` / Explicit CUDA Graph Capture + Fused `AdamW`:** PyTorch's fastest fixed-shape GPU training mode via CUDA Graph Trees and allocator state checkpointing.
4. **Activation Checkpointing (`torch.utils.checkpoint`) & `DDP`:** Memory-constrained deep-model training and multi-GPU data-parallel baselines.

### 3.2 GPU & CPU Inference Modes
1. **GPU Inference:** `torch.inference_mode()` across Eager, `torch.compile(inductor)`, `autocast(bfloat16/float16)`, `sdpa(flash/cudnn)`, static KV-cache decode, and CUDA Graph replay.
2. **Intel CPU Training & Inference:** Single-threaded (`threads = 1`) and multi-threaded (`threads = 2, 4, 8`) PyTorch Eager and `torch.compile(inductor)` C++/OpenMP + oneDNN/oneMKL (including `bfloat16`/AMX where supported).

### 3.3 Verified Structural Weaknesses of PyTorch on Audio Workloads
1. **Dynamic Audio Length vs. CUDA Graphs Conflict:** Variable clip durations either trigger Dynamo/Inductor recompilation guards (`dynamic=True`) or disable CUDA Graph capture (`reduce-overhead`).
2. **Spectral / DSP Graph Breaks:** `torch.istft` fails CUDA Graph capture (`cudaErrorStreamCaptureInvalidated`) and forces graph breaks in Inductor (`fast-audio` already beats PyTorch by **7.47x** on strict-FP32 `spectral-vocoder-step-adamw`).
3. **Compile & Warmup Wall-Clock Tax:** Inductor autotuning and compilation add tens of seconds to `Time-to-Train` and cold-start inference, especially across multiple audio duration buckets.
4. **Materialized Output Logits & Allocator Overhead:** Standard PyTorch `Linear + CrossEntropy` materializes giant `[B*T, V]` logits tensors in HBM; `CUDACachingAllocator` retains fragmented pools compared to a single static byte arena with buffer donation.
5. **Small-Batch Dispatch, Generic CPU Loops & Small-Hidden RNNs:** Python/ATen dispatch penalizes short audio frames and `BS=1..4` decode; CPU Inductor lacks 2D L1 cache-blocked audio micro-tiling (`fast-audio` already sweeps CPU Inductor **11/11** at 1 thread) and spin-then-futex thread wakeups; small-hidden `GRU` (`H <= 192`) incurs launch overhead per step.

---

## 4. `fast-audio` Baseline and Gap Audit (2026-11-23)

### 4.1 Verified Strict-FP32 MVP Strengths (`HEAD = 1843e42`)
- **Colab Intel CPU (`fa-cpu`, `--features mkl,onednn`, 1 thread):** **11/11 sweep** vs PyTorch `2.14.1` Eager and Inductor (2.66x geomean vs Eager, **1.70x geomean vs Best**, peaking at **6.77x** on `spectrum-large` and **4.00x** on `spectrum-medium` via 2D L1 cache-blocked fusion).
- **Colab NVIDIA L4 (`fa-gpu`, `sm_89`, CUDA `13.0`, cuDNN `9.14.0`):** **9/9 sweep** vs PyTorch Eager (**2.79x geomean**); **8/9 wins** vs PyTorch Best (**1.84x geomean**), including **5.95x vs Eager / 1.20x vs CUDA Graph** on `audio-transformer-step-adamw` and **7.47x vs Eager** on `spectral-vocoder-step-adamw`.
- **Verification Discipline:** 0 UB under Miri, 35/35 CPU & 35/35 CUDA PyTorch cross-checks, and 12/12 `compute-sanitizer` (`memcheck`/`racecheck`/`synccheck`) runs with 0 errors and 0 hazards.

### 4.2 Hardware Strata Reality
Verified Colab entitlements ([colab-hardware-evaluation.md](colab-hardware-evaluation.md)):
- **Available GPUs:** `L4 (sm_89)` (primary daily GPU, ~300 GB/s), `A100 (sm_80)` (HBM2e ~1,935 GB/s scaling), `G4 (sm_120)` (Blackwell compute scaling), `T4 (sm_75)` (FP16 floor; no native BF16 Tensor Cores). `H100 (sm_90)` is ineligible.
- **CPU Runtime:** `fa-cpu --high-mem` (8 vCPUs, ~51 GB RAM, MKL + oneDNN) strictly isolated from GPU instances.

### 4.3 Uncommitted Prototype Audit and the "Two Worlds" Architectural Risk
Inspection of the working tree revealed uncommitted Phase A/B prototype code (`src/dtype.rs`, `src/embedding.rs`, `src/rope.rs`, `src/dropout.rs`, `src/loss.rs`, `src/cuda/mixed.rs`, `src/cuda/audio_ops.rs`, `native/bf16_tc.cu`, `native/sdpa_bf16.cu`, `native/audio_ops.cu`, `tests/phase_ab_cpu.rs`):
- **Strengths already prototyped:** IEEE RNE BF16/FP16 scalar conversions, `mma.m16n8k16` BF16 FlashAttention-2 (`D = 64|128`), `Philox4x32-10` Dropout, fused CrossEntropy, BF16 RMSNorm/Residual+RMSNorm/SwiGLU/AdamW/`CudaGradScaler` (100% sync-free), and a single-CUDA-Graph `Bf16AudioTransformerTrainer` (initial L4 screen recorded 1.11x over best observed PyTorch BF16 mode on a 6-layer step, with long-sequence `T=1024..2048` SDPA remaining a clear gap).
- **Critical Risk ("Two Worlds Split"):** `src/cuda/mixed.rs` bypasses `autograd::Tape`, `Layout` strided views, `Shared`/`SavedValue` versioning, and NVRTC fusion.
- **Resolution:** Ticket `A0` audits, verifies, measures, and commits the prototype cleanly; Tickets `A1`–`A5` unify mixed precision inside `autograd::Tape`; Ticket `C6` lowers the exact same model graph into `FrozenInferencePlan` to crush PyTorch inference modes as well.

---

## 5. Core Engineering Levers to Comprehensively Beat PyTorch

### 5.1 Training Systems & Kernel Levers

| # | Engineering Mechanism | Why It Beats PyTorch on Audio Training | Mapped Ticket(s) |
| --- | --- | --- | --- |
| 1 | **Explicit Mixed Precision (`BF16`/`TF32` & `FP16`) + Tensor Cores** | Matches PyTorch `cublasLt`/cuDNN v9 dense/conv ceilings (`A2`/`A3`) and beats PyTorch `sdpa`/pointwise via `cp.async` FlashAttention-2 (`A4`) and BF16 NVRTC fusion (`A5`) with zero Python/ATen tax | `A1–A6` |
| 2 | **Audio Sequence Packing (`cu_seqlens` + Block-Diagonal SDPA + Per-Segment RoPE Reset)** | Eliminates 30–60% padding FLOP waste on variable-duration audio clips while keeping static buffer shapes for CUDA Graphs (**1.4x–2.2x Time-to-Train win**) | `B3`, `C2` |
| 3 | **Fused Chunked `Linear + CrossEntropy` (Zero Full-Logits Materialization)** | Tiles `[B*T, D] @ [D, V] + CE` over `BT` chunks so giant `[B*T, V]` audio token/codebook logits never materialize in HBM (**30–50% peak activation VRAM reduction vs PyTorch**) | `B6`, `D2` |
| 4 | **Single-Buffer Contiguous Byte Arena + Explicit Buffer Donation** | Eliminates per-tensor allocation overhead and `CUDACachingAllocator` fragmentation; aliases dead inputs/intermediates onto outputs with zero extra copy | `C1`, `C2` |
| 5 | **Policy-Based Selective Rematerialization (`SaveDotsOnly` vs `FullBlock`)** | Retains expensive Tensor Core GEMM/Conv/SDPA outputs while recomputing cheap bandwidth-bound Norm/SwiGLU/RoPE/pointwise ops inside 1 CUDA Graph, beating `torch.utils.checkpoint` | `C3` |
| 6 | **Homogeneous Layer-Plan Sharing + 100% Sync-Free Single-CUDA-Graph Training** | $O(1)$ preparation time across $L$ identical layers (eliminating Inductor compile stalls); keeps `d_lr`, `d_step`, `GradScaler` `found_inf`, and `Philox` RNG step entirely on device | `A5`, `A6`, `B5`, `D1–D5` |
| 7 | **CUDA-Graph-Safe Differentiable Spectral & Ring-Buffer DSP** | Keeps `Stft`/`CenteredStft`/`Istft`/`MultiResolutionStftLoss` and chunked DSP 100% inside captured CUDA Graphs with zero `memmove`, where PyTorch `torch.istft` breaks graph capture | `B6`, `B7`, `D3`, `D4` |

### 5.2 Inference & Multi-Core CPU Levers

| # | Engineering Mechanism | Why It Beats PyTorch on Audio Inference & CPU | Mapped Ticket(s) |
| --- | --- | --- | --- |
| 1 | **Zero-Autograd `FrozenInferencePlan` Lowering** | Automatic Norm/BN folding, horizontal `QKV`/`Gate+Up` fusion, epilogue sinking, and Cargo feature-trimmed runtime — eliminating Python/ATen/autograd overhead | `C6` |
| 2 | **Spin-Then-Futex Persistent Multi-Core CPU Pool + Weight Repacking (`pack_b`)** | Eliminates 20–50 us OS thread wakeup jitter on short audio frames, avoids MKL/oneDNN OpenMP oversubscription, and streams packed weights into AVX2/AVX-512 microkernels | `C4`, `C6` |
| 3 | **Low-Batch `M=1..8` Warp-Reduced GEMV + Split-K `FlashDecoding` + Device `d_kv_len` KV-Cache** | Hits >= 80% HBM bandwidth on `BS=1..4` autoregressive audio decode and runs `T_k = 1..16384` inside **one CUDA Graph with 0 recapture**, crushing PyTorch decode latency and `p99` | `A2`, `B3`, `C5` |
| 4 | **64B-Aligned Zero-Copy `mmap` Weights + Async Pinned H2D Staging** | Millisecond cold starts (`< 50 ms`) without PyTorch/Inductor startup stalls, plus `>= 85%` H2D transfer overlap | `C5`, `C6` |

---

## 6. PyTorch-Only Benchmark Anchor Matrix (`Measure Every Step`)

Every workload below is paired **strictly against PyTorch `2.14+`** (strongest compatible mode per regime) on identical hardware sessions:

| Workload Class | Training vs PyTorch `2.14.1` (Must Beat on Step Speed, VRAM & Wall-Clock `TTT`) | Inference vs PyTorch `2.14.1` (Must Beat on Latency, `RTFx`/`tokens/s`, `p99` & Memory) | Hardware Strata |
| --- | --- | --- | --- |
| **Causal Audio Transformer (`D2`, `A2/A4/B6/C5`)** | `bf16 + torch.compile + CUDA Graph` (`>= 1.15x` tok/s, `<= 0.85x` VRAM, `>= 1.25x` faster wall-clock `TTT` to target loss); multi-threaded CPU Inductor | `torch.inference_mode()` + `bf16`/`fp32` + `torch.compile` + static KV-cache + CUDA Graph (`BS=1..4` decode `tokens/s`, `TTFT`, `p99`) | Colab `fa-cpu` (1–8T), `L4`, `A100`, `G4` |
| **Spectral Enhancement / Separation U-Net & GTCRN (`D3`, `A3/B4/B6`)** | `bf16 + torch.compile` (`>= 1.5x–3.0x` via single CUDA Graph, `>= 1.5x` faster `TTT` to target SI-SDR); multi-threaded CPU Inductor | `torch.inference_mode()` + Eager / `torch.compile` / CUDA Graph (offline + chunked `RTFx` and `p99` frame latency) | Colab `fa-cpu` (1–8T), `L4`, `A100` |
| **Neural Audio Codec / Vocoder (`D4`, `B2/B7/C6`)** | `bf16 + torch.compile` (`>= 1.25x` throughput, `>= 1.25x` faster `TTT` to target multi-res spectral loss); multi-threaded CPU Inductor | `torch.inference_mode()` + Eager / `torch.compile` / CUDA Graph (offline + causal streaming chunk `p99` and `RTFx`) | Colab `fa-cpu` (1–8T), `L4`, `A100`, `T4` (FP16) |

---

## 7. Deferred Scope (Post-PyTorch Leadership)

To maintain sharp focus on beating PyTorch across all audio training and inference regimes first, the following comparisons and extensions are explicitly deferred until after Phase D gates pass:
- Cross-framework benchmarks and parity targets against other ML frameworks (e.g., JAX/OpenXLA, Burn, Candle, Megatron-Core/TransformerEngine) or standalone C++ inference engines (e.g., `ggml`/`llama.cpp`/`whisper.cpp`, `CTranslate2`, `ONNX Runtime`, `OpenVINO`, `TensorRT`, `FlashInfer`, `XNNPACK`/`ncnn`).
- Post-training weight-only quantized inference (`W8A16`/`W4A16` / `Q8_0`/`Q4_0` / CPU INT8/INT4).
- Training-time `FP8` precision paths.
