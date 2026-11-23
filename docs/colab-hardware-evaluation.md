# Google Colab hardware, `colab` CLI, and Google Drive I/O workflow

## Executive verdict and instance-isolation rule

Google Colab is `fast-audio`'s cloud execution environment for multi-architecture NVIDIA GPU and cloud CPU workloads, operated programmatically through the **`colab` CLI** with **Google Drive (`/content/drive/MyDrive/fast-audio-colab`)** acting as a persistent high-speed I/O cache.

To prevent unnecessary Compute Unit (CU) burn while maintaining high I/O throughput, enforce **strict CPU vs. GPU instance separation**:

1. **Rent CPU instances for CPU workflows (`colab new -s fa-cpu` or `colab new -s fa-cpu --high-mem`):**
   - Run all portable Rust (`cargo test`), Intel CPU (`--features mkl,onednn`), CPU benchmark (`fa-bench`, `fa-pointwise`, `fa-training`), and Google Drive cache-population tasks on a **CPU runtime**.
   - **Never run CPU compilation/testing workflows, CPU benchmarks, or toolchain/Drive bootstrapping on a GPU instance.**
2. **Rent GPU instances strictly for GPU workflows (`colab new -s fa-gpu --gpu <L4|T4|A100|G4>`):**
   - Allocate a GPU runtime only when GPU compilation (`--features cuda,cudnn`), GPU functional verification, or GPU benchmarking is ready to execute.
   - Mount Google Drive (`colab drivemount -s fa-gpu`), pull the pre-staged repository and cached toolchain from `/content/drive/MyDrive/fast-audio-colab` into local `/content/fast-audio`, run the GPU workload, sync sealed artifacts back to Google Drive / local disk, and **immediately terminate the GPU session (`colab stop -s fa-gpu`)**.
3. **Two-tier confirmation strategy:**
   - **Tier A — Hosted Colab via `colab` CLI (Daily Cloud Battleground):** Use CPU sessions for CPU verification/screening and GPU sessions (`L4` primary, `A100` for HBM2e bandwidth/launch-overhead studies, `G4` for Blackwell `sm_120` compute scaling, `T4` for `sm_75` floor checks) for same-session A/B screening and `nsys` timeline profiling.
   - **Tier B — Dedicated Cloud VM / Local Intel Host (Authoritative 5% Promotion & Hardware Counters):** Because multi-tenant Colab containers block clock locking (`nvidia-smi -lgc`, `cpupower`) and privileged PMU counters (`ncu` `ERR_NVGPUCTRPERM`, `perf`), run formal 5% CI champion promotions and `ncu` hardware-counter attribution on an on-demand root-privileged cloud VM (or the local Intel Core i5-13420H for local Intel CPU confirmation).

> **Stratum transition note:** Because the historical local RTX 3060 12 GB (`sm_86`) machine is retired, preserve existing `sm_86` baseline records (`benchmarks/champions.json` and historical `benchmarks/results/`) untouched per [the benchmark contract](benchmarking.md), and establish fresh anchor baselines on the cloud GPU stratum (**NVIDIA L4 / `sm_89`**) inside a single live session before evaluating new GPU optimization candidates.

---

## Live-verified Colab environment probe (2026-11-20)

Probing the live Colab runtime via `colab usage` (`/tun/m/ccu-info`) and `colab exec` / `colab ssh` on 2026-11-20 establishes the exact container software and entitlement profile:

- **Account accelerator entitlement (`ccu-info`):**
  - `eligibleGpus`: `["G4", "A100", "L4", "T4"]`
  - `ineligibleGpus`: `["H100"]`
  - `eligibleTpus`: `["V6E1", "V5E1"]` (TPUs remain outside Year-1 Intel+NVIDIA scope)
- **Container OS and compiler toolchain:**
  - **OS:** Ubuntu 24.04.4 LTS (`Noble Numbat`)
  - **C++ compiler:** `/usr/bin/g++` and `/usr/bin/g++-13` (`GCC 13.3.0`) preinstalled on `PATH` (directly compatible with `"cuda_host_cxx": "g++-13"` in existing benchmark protocols).
  - **CUDA & cuDNN:** NVIDIA Driver `580.82.07`, **CUDA Toolkit 13.0** (`nvcc V13.0.88`, supporting `sm_75` through Blackwell `sm_120`), and **cuDNN 9.14.0** (`/usr/include/x86_64-linux-gnu/cudnn_version.h`).
  - **Python & PyTorch:** Python `3.13.16`, PyTorch `2.14.1+cu130` (`torch.version.cuda == "13.0"`, `cudnn == 92800`).
  - **Preinstalled profilers & CPU headers:** Nsight Systems `2025.3.1.0` (`/usr/local/bin/nsys`), Nsight Compute (`/usr/local/cuda/bin/ncu`), GNU `gprofng` (`/usr/bin/gprofng`), and Intel MKL headers (`/usr/include/mkl`) are preinstalled; Linux `perf` is not installed.
  - **Storage layout:** `236 GB` local container overlay disk (`/content`, `/tmp`) and `236 GB` Google Drive FUSE mount (`/content/drive`).

---

## `colab` CLI and Google Drive high-speed I/O architecture

### Why stage code and caches on Google Drive (`/content/drive/MyDrive/fast-audio-colab`)

Colab container disks (`/content`) are wiped whenever a session stops (`colab stop`), whereas uploading toolchains, crates, and repositories from a local machine on every session is bottlenecked by client upload bandwidth. Conversely, compiling Cargo projects (`target/`) or writing profiler streams directly onto the `/content/drive` FUSE mount is slow due to per-file metadata round-trips.

Use a **two-layer storage split**:

| Layer | Path | Role |
| --- | --- | --- |
| **Persistent Drive Cache** | `/content/drive/MyDrive/fast-audio-colab/` | Stores `fa-source.tar.gz` (~485 KB tracked source/docs without historical `benchmarks/results/`), `fa-vendor.tar.gz` (~1.7 MB offline Cargo vendor bundle), `producer-manifest.json`, cached Rust toolchain/packages (`rust-cache.tar.gz`), and completed benchmark result archives (`results/`). |
| **Ephemeral Fast Scratch** | `/content/fast-audio/` & `/content/target/` | Local container overlay SSD where archives are unpacked, `cargo build` / `cargo test` execute at full local disk speed, and raw benchmark outputs are generated before archiving back to Drive. |

Transferring compressed archives between `/content/drive/MyDrive/fast-audio-colab/` and `/content/fast-audio/` runs entirely inside Google's datacenter network, cutting session cold-start I/O to seconds.

### Standard `colab` CLI workflow

#### 1. Package lightweight source and vendor archives locally

Keep historical `benchmarks/results/` out of daily upload archives. Package the tracked source manifest (~485 KB) and offline Cargo vendor directory (~1.7 MB) locally:

```sh
# 1. Vendor Cargo crates offline once (only needed when Cargo.lock changes)
cargo vendor --offline /tmp/fa-vendor
tar -czf /tmp/fa-vendor.tar.gz -C /tmp fa-vendor

# 2. Package tracked source, tests, tools, protocols, and docs (~485 KB)
.venv/bin/python -c '
import tarfile
from pathlib import Path
from tools.compare_torch import source_manifest
root = Path(".").resolve()
manifest = source_manifest()
extra = [p for p in root.rglob("*") if p.is_file() and p.relative_to(root).parts[0] in ("docs", "AGENTS.md", "CODEMAP.md", "target.md", "reference.md", ".gitignore", ".gitattributes") and "__pycache__" not in p.parts]
for_archive = sorted(set(manifest.keys()) | {str(p.relative_to(root)) for p in extra})
with tarfile.open("/tmp/fa-source.tar.gz", "w:gz") as tf:
    for rel in for_archive:
        tf.add(root / rel, arcname=rel, recursive=False)
'
```

#### 2. CPU workflow: rent a CPU instance (`fa-cpu`)

Use a CPU session (`--high-mem` for 8 vCPUs / ~51 GB RAM, or standard CPU for light tasks) to sync Google Drive, build/cache the Rust toolchain, and run all CPU tests and CPU benchmarks:

```sh
# 1. Allocate a CPU session (never use --gpu for CPU workflows)
colab new -s fa-cpu --high-mem

# 2. Mount Google Drive at /content/drive
colab drivemount -s fa-cpu

# 3. Upload source delta (and vendor tarball if updated)
colab upload -s fa-cpu /tmp/fa-source.tar.gz content/fa-source.tar.gz
colab upload -s fa-cpu /tmp/fa-vendor.tar.gz content/fa-vendor.tar.gz

# 4. Sync archives to Google Drive and unpack into local /content/fast-audio via colab ssh
colab ssh -s fa-cpu -- 'mkdir -p /content/drive/MyDrive/fast-audio-colab /content/fast-audio && cp /content/fa-source.tar.gz /content/fa-vendor.tar.gz /content/drive/MyDrive/fast-audio-colab/ && tar -xzf /content/fa-source.tar.gz -C /content/fast-audio && tar -xzf /content/fa-vendor.tar.gz -C /content'

# 5. Run CPU tests / benchmarks on local /content/fast-audio
colab ssh -s fa-cpu -- 'cd /content/fast-audio && cargo test --offline'

# 6. Stop the CPU session when finished
colab stop -s fa-cpu
```

#### 3. GPU workflow: rent a GPU instance (`fa-gpu`) only for GPU execution

Once code and Drive caches are updated, spin up a GPU session strictly for the GPU build, test, or benchmark window:

```sh
# 1. Allocate the target GPU session (L4 for daily GPU work, T4 for sm_75 floor, A100/G4 for scaling)
colab new -s fa-gpu --gpu L4

# 2. Mount Google Drive
colab drivemount -s fa-gpu

# 3. Bootstrap directly from Google Drive cache into local /content/fast-audio (fast internal I/O)
colab ssh -s fa-gpu -- 'mkdir -p /content/fast-audio && tar -xzf /content/drive/MyDrive/fast-audio-colab/fa-source.tar.gz -C /content/fast-audio && tar -xzf /content/drive/MyDrive/fast-audio-colab/fa-vendor.tar.gz -C /content'

# 4. Build and run CUDA/cuDNN suites on local disk (set CUDA_ARCH=89 for L4, 75 for T4, 80 for A100, 120 for G4)
colab ssh -s fa-gpu -- 'cd /content/fast-audio && CUDA_ARCH=89 cargo test --offline --features cuda,cudnn'

# 5. Archive any benchmark results back to Google Drive and download locally
colab ssh -s fa-gpu -- 'cp -r /content/fast-audio/benchmarks/results/<run-id> /content/drive/MyDrive/fast-audio-colab/results/'

# 6. Immediately terminate the GPU session to stop burning Compute Units
colab stop -s fa-gpu
```

> **`colab ssh` vs. `colab exec` operational note:** After running `colab drivemount -s <name>`, always prefer `colab ssh -s <name> -- '<cmd>'` (along with `colab upload` / `colab download`) for automation. `colab drivemount` spawns an auxiliary Jupyter kernel session for OAuth credential propagation, whereas `colab ssh` tunnels directly over `/colab/ssh` using `~/.ssh/id_ed25519.pub` and returns deterministic command exit codes without `jupyter_kernel_client`'s 10-second WebSocket handshake timeout.

---

## Google Colab GPU comparison vs historical baseline

All GPUs below are evaluated under **`fast-audio`'s strict-FP32 execution contract** (`TF32` disabled, no implicit FP16/BF16 downcast). Non-Tensor FP32 TFLOPS, L2 cache capacity, and DRAM bandwidth govern strict-FP32 operator latency.

| Platform | GPU Model | Arch / `CUDA_ARCH` | SMs (CUDA Cores) | Peak FP32 (TFLOPS) | Tensor TF32 / FP16 (Dense, TFLOPS) | VRAM & Type | Memory Bandwidth | L2 Cache | TDP / Form Factor | Typical Colab Burn (CU/h) |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| **Historical Baseline (Retired Local)** | GeForce RTX 3060 | Ampere / `86` | 28 (3,584) | 12.7 | 12.7 (25.5 sparse) / 25.5 (51 sparse) | 12 GB GDDR6 | 360 GB/s | 3 MB | 170 W / PCIe | — |
| **Colab Eligible (`T4`)** | Tesla T4 | Turing / `75` | 40 (2,560) | 8.1 | N/A / 65 (FP16/32 mix) | 16 GB GDDR6 (~15.0 usable) | 300 GB/s | 4 MB | 70 W / PCIe Gen3 x16 | ~1.19–1.58 |
| **Colab Primary (`L4`)** | NVIDIA L4 | Ada Lovelace / `89` | 58 (7,424) | 30.3 | ~60 (120 sparse) / 121 | 24 GB GDDR6 (~22.5 usable) | 300 GB/s | **48 MB** | 72 W / PCIe Gen4 x16 | ~1.71–3.00 |
| **Colab Eligible (`A100`)** | NVIDIA A100 (40G / 80G) | Ampere / `80` | 108 (6,912) | 19.5 | 156 / 312 | 40 or 80 GB HBM2e | 1,555 GB/s (40G) / ~2,039 GB/s (80G) | 40 MB | 400 W / SXM4 | ~5.37–10.59 |
| **Colab Eligible (`G4`)** | RTX PRO 6000 Blackwell Server Ed. (`G4`) | Blackwell / `120` | 188 (24,064) | **117–120** | Blackwell 5th-gen (FP4/FP6/FP8/FP16/TF32) | 96 GB GDDR7 ECC (~95.6 usable) | ~1,597 GB/s | **128 MB** | 600 W / PCIe Gen5 x16 | ~6.40–8.71 |
| **Ineligible on Current Plan (`H100`)** | NVIDIA H100 80GB HBM3 | Hopper / `90` | 132 (16,896) | **67.0** (SXM5) | ~495 (989 sparse) / ~990 (1,979 sparse) | 80 GB HBM3 (~79.6 usable) | **3,350 GB/s** | 50 MB | 700 W / SXM5 | ~18.05 |

### Architectural significance for `fast-audio` GPU workloads

| Device | FP32 / Bandwidth Ratio (FLOPs/Byte) | Primary Diagnostic Value for `fast-audio` |
| --- | ---: | --- |
| **RTX 3060** (historical) | 35.3 | Balanced entry Ampere (`sm_86`), small 3 MB L2 exposes un-fused DRAM traffic. |
| **Tesla T4** (`--gpu T4`) | 27.0 | Legacy `sm_75` compatibility floor; 4 MB L2; low CU burn (~1.2–1.6 CU/h). |
| **NVIDIA L4** (`--gpu L4`) | **101.0** | **Primary daily Colab GPU.** 300 GB/s GDDR6 paired with 30.3 TFLOPS FP32 and a **48 MB L2 cache** (`16x` RTX 3060). Penalizes wasted DRAM round-trips while rewarding kernel fusion and L2 residency at modest ~1.7–3.0 CU/h burn. |
| **NVIDIA A100** (`--gpu A100`) | **12.5** | HBM2e bandwidth abundance (`4.3x` RTX 3060). Separates DRAM-bandwidth-bound kernels from host-dispatch / CUDA launch-gap bottlenecks. |
| **RTX PRO 6000** (`--gpu G4`) | **73.3–75.1** | Highest strict-FP32 compute density (`~9.2x` RTX 3060 FP32) plus **128 MB L2** and ~1.6 TB/s GDDR7; supported natively by the container's CUDA 13.0 (`CUDA_ARCH=120`). |

---

## Google Colab CPU instances vs local baseline

When running CPU workloads on Colab, **always provision a CPU runtime (`colab new -s fa-cpu --high-mem` or `colab new -s fa-cpu`)** rather than running CPU jobs on a GPU runtime:

| Runtime | Command | CPU Microarchitecture | vCPUs (Approx. Physical Cores) | System RAM | Role & Constraints |
| --- | --- | --- | ---: | ---: | --- |
| **Local Baseline** | Local workspace | Intel Core i5-13420H (Raptor Lake-H, 4P+4E) | 12 threads (8 cores: 4P+4E) | Local | AVX2, AVX-VNNI (no AVX-512). Historical Intel control. |
| **Colab CPU High-RAM** | `colab new -s fa-cpu --high-mem` | Intel Xeon or AMD EPYC 7B12 (live-probed 2026-11-20: 8 vCPU, 4C/8T, AVX2+FMA) | 8 (~4 physical cores w/ SMT) | ~50–51 GB | **Primary Colab CPU instance** for parallel Cargo compilation, CPU correctness suites, MKL/oneDNN tests, and exploratory CPU benchmarks. Verify `lscpu` per session for Intel vs. AMD vendor dispatch. |
| **Colab CPU Standard** | `colab new -s fa-cpu` | Intel Xeon `@ 2.20 GHz` (Broadwell/Skylake pool) | 2 (~1 physical core w/ SMT) | ~12.7 GB | Lightweight Drive cache management and single-thread (`--threads 1`) functional checks. Do not run multi-thread benchmarks on 2-vCPU SMT. |
| **GPU Host CPUs (`L4`/`A100`/`G4`)** | *(Attached to GPU VMs)* | Intel Cascade Lake (`L4`/`A100`, 12 vCPU, AVX-512) or AMD EPYC Turin (`G4`, 48 vCPU) | 12 or 48 | 52–177 GB | **Host dispatch only.** Never rent a GPU VM to run CPU workflows; `G4` also uses AMD EPYC rather than Intel. |

---

## Compatibility audit with `fast-audio` tooling and protocols

| `fast-audio` Requirement / Tool | Hosted Colab Behavior (Ubuntu 24.04 / CUDA 13.0) | Required Adaptation |
| --- | --- | --- |
| `Cargo.toml` / `build.rs` (`--features cuda,cudnn`) | Container ships `nvcc` 13.0, `g++-13` 13.3.0, and `cudnn_version.h` 9.14.0. | Set `CUDA_ARCH` to match the rented GPU (`89` for L4, `75` for T4, `80` for A100, `120` for G4). |
| `Cargo.toml` / `build.rs` (`--features mkl,onednn`) | `/usr/include/mkl` is present; `libmkl-dev` / `libdnnl-dev` headers/libs may need APT or cached `.deb` install on fresh VMs. | Install/cache `libmkl-dev` and `libdnnl-dev` on the CPU instance and create `/opt/intel/oneapi/mkl/latest/include -> /usr/include/mkl` (`mkdir -p /opt/intel/oneapi/mkl/latest && ln -sfn /usr/include/mkl /opt/intel/oneapi/mkl/latest/include`) so `build.rs` resolves `mkl.h` without modifying `build.rs`. |
| `tools/compare_torch.py` / `fa-bench` | Works out of the box once binary and Rust toolchain are present. | Run anchor and candidate back-to-back inside the **exact same live Colab session**; never compare latencies across separate session allocations. |
| `tools/build_pointwise.py` & `tools/build_training.py` | Frozen `sm_86` protocols specify `"cuda_arch": "86"`, `"cuda_host_cxx": "g++-13"`, and `--offline`. | Pass `--vendor-dir /content/fa-vendor` and `--producer-manifest`; use stratum-matched `cuda_arch` (`89` on L4) for cloud GPU campaigns while keeping `sm_86` historical reports immutable. |
| `nsys` (`training_benchmark.py` / `pointwise_benchmark.py`) | `/usr/local/bin/nsys` (`2025.3.1.0`) is preinstalled on `PATH` and supports `--sample=none --cpuctxsw=none`. | Copy `.nsys-rep` / SQLite exports to `/content/drive/MyDrive/fast-audio-colab/results/` before `colab stop`. |
| `gprofng` / `perf` | `/usr/bin/gprofng` is preinstalled; `perf` is absent. | `training_benchmark.py` automatically falls back to `gprofng` on CPU runs. |
