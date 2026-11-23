#!/usr/bin/env python3
"""Counterbalanced, full-output comparisons against eager PyTorch. See docs/benchmarking.md."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

import torch
import torch.nn.functional as F

try:
    from .benchmark_protocol import case_timing, load_profile, validate_artifacts
except ImportError:
    from benchmark_protocol import case_timing, load_profile, validate_artifacts

ROOT = Path(__file__).resolve().parents[1]


def fixture(shape, salt, device):
    size = math.prod(shape)
    state = (torch.arange(size, dtype=torch.int64) * 747796405 + salt) & 0xFFFFFFFF
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & 0xFFFFFFFF
    bits = ((word >> 22) ^ word) & 65535
    return ((bits.to(torch.float32) - 32768) / 32768).reshape(shape).to(device)


def prepare(case, device):
    op = case["op"]
    if op == "matmul":
        a = fixture([case["m"], case["k"]], 11, device)
        b = fixture([case["k"], case["n"]], 29, device)
        y = torch.empty((case["m"], case["n"]), device=device)
        return lambda: torch.mm(a, b, out=y)
    if op == "conv1d":
        x = fixture([case["batch"], case["in_channels"], case["input_len"]], 11, device)
        w = fixture([case["out_channels"], case["in_channels"] // case["groups"], case["kernel"]], 29, device)
        bias = fixture([case["out_channels"]], 47, device) if case["bias"] else None

        def conv():
            y = F.conv1d(x, w, bias, case["stride"], case["padding"], case["dilation"], case["groups"])
            return y.relu_() if case["relu"] else y

        return conv
    if op == "stft":
        n = case["fft_size"]
        x = fixture([case["batch"], case["input_len"]], 11, device)
        window = torch.tensor([0.5 - 0.5 * math.cos(math.tau * i / n) for i in range(n)], dtype=torch.float32, device=device)
        return lambda: torch.stft(x, n_fft=n, hop_length=case["hop"], win_length=n,
                                  window=window, center=False, normalized=False,
                                  onesided=True, return_complex=True)
    raise ValueError(f"unknown operator {op}")


def torch_sample(run, case, args, device):
    iterations, warmup = case_timing(args.profile, case["name"], args.iterations, args.warmup)
    for _ in range(warmup):
        result = run()
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Prime lazy event initialization outside the measured region.
        start.record()
        end.record()
        end.synchronize()
        begin = time.perf_counter_ns()
        start.record()
        for _ in range(iterations):
            result = run()
        end.record()
        end.synchronize()
        wall_us = (time.perf_counter_ns() - begin) / 1000 / iterations
        device_us = start.elapsed_time(end) * 1000 / iterations
    else:
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            result = run()
        wall_us = (time.perf_counter_ns() - begin) / 1000 / iterations
        device_us = None
    # Canonicalize layout outside timing; torch.stft exposes a frequency-major view.
    if case["op"] == "stft":
        result = torch.view_as_real(result.transpose(-1, -2))
    return {"wall_us": wall_us, "device_us": device_us, "iterations":iterations, "warmup":warmup}, result.cpu().contiguous()


def rust_sample(case, args, output, binary=None, backend=None):
    iterations, warmup = case_timing(args.profile, case["name"], args.iterations, args.warmup)
    command = [str(binary or args.binary), "--backend", backend or args.backend, "--case", case["name"],
               "--threads", str(args.threads), "--iterations", str(iterations),
               "--warmup", str(warmup), "--samples", "1", "--output", str(output),
               "--workloads", str(args.workloads)]
    raw = subprocess.run(command, check=True, text=True, capture_output=True)
    report = json.loads(raw.stdout)
    if (report["iterations"],report["warmup"]) != (iterations,warmup) or report["case"] != case or report["backend"] != (backend or args.backend):
        raise RuntimeError("benchmark execution differs from requested case/backend/timing counts")
    blob = bytearray(output.read_bytes())
    if len(blob) != math.prod(report["output_shape"]) * 4:
        raise RuntimeError("output file byte size does not match reported shape")
    result = torch.frombuffer(blob, dtype=torch.float32).clone().reshape(report["output_shape"])
    return report, result


def errors(actual, expected):
    if list(actual.shape) != list(expected.shape):
        return {"passed": False, "shape_actual": list(actual.shape), "shape_expected": list(expected.shape)}
    diff = (actual - expected).abs()
    tolerance = 2e-4 + 2e-4 * expected.abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {"passed": finite and bool((diff <= tolerance).all()), "finite": finite,
            "elements": actual.numel(), "max_abs": diff.max().item(),
            "rms": diff.square().mean().sqrt().item(),
            "max_tolerance_ratio": (diff / tolerance).max().item(), "atol": 2e-4, "rtol": 2e-4}


def summary(blocks):
    # Blocks, rather than inner iterations, are the resampling unit. Exploratory CI only.
    logs = [math.log(b["torch"]["wall_us"] / b["rust"]["samples"][0]["wall_us"]) for b in blocks]
    rng = random.Random(1701)
    bootstrap = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(5000))
    result = {"rust_wall_us_median": statistics.median(b["rust"]["samples"][0]["wall_us"] for b in blocks),
            "torch_wall_us_median": statistics.median(b["torch"]["wall_us"] for b in blocks),
            "wall_speedup_geomean": math.exp(statistics.mean(logs)),
            "wall_speedup_exploratory_95pct_ci": [bootstrap[125], bootstrap[4874]]}
    if "anchor" in blocks[0]:
        logs = [math.log(b["anchor"]["samples"][0]["wall_us"] / b["rust"]["samples"][0]["wall_us"]) for b in blocks]
        result["anchor_wall_us_median"] = statistics.median(b["anchor"]["samples"][0]["wall_us"] for b in blocks)
        result["anchor_wall_speedup_geomean"] = math.exp(statistics.mean(logs))
    return result


def command_output(command):
    try:
        return subprocess.run(command, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"unavailable": str(exc)}


def cpu_identity():
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return platform.processor()
    fields = dict(line.split(":", 1) for line in path.read_text().split("\n\n",1)[0].splitlines() if ":" in line)
    return {k.strip(): v.strip() for k,v in fields.items() if k.strip() in
            {"vendor_id", "cpu family", "model", "model name", "stepping", "microcode"}}


def source_manifest():
    paths = [ROOT / name for name in ["Cargo.toml", "Cargo.lock", "build.rs", "benchmarks/workloads.json"]]
    for directory in ["src", "native", "tests", "tools", "examples", "models", "benchmarks/protocols"]:
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/fa-bench")
    parser.add_argument("--workloads", type=Path, default=ROOT / "benchmarks/workloads.json")
    parser.add_argument("--anchor-binary", type=Path)
    parser.add_argument("--anchor-backend", choices=["reference", "mkl", "onednn", "intel", "cuda", "cudnn", "nvidia"])
    parser.add_argument("--anchor-report", type=Path)
    parser.add_argument("--backend", choices=["mkl", "onednn", "intel", "cuda", "cudnn", "nvidia", "reference"], required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--execution-id", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.profile = None
    profile_hash = None
    suite = json.loads(args.workloads.read_text())
    if suite["schema_version"] != 1:
        raise ValueError("unsupported workload schema")
    if args.protocol:
        args.profile, profile_hash = load_profile(args.protocol, [c["name"] for c in suite["cases"]])
        if args.iterations is not None or args.warmup is not None or args.cases:
            parser.error("a declared timing protocol supplies all case windows; do not override them")
        if not args.anchor_binary:
            parser.error("a declared timing protocol requires its retained anchor")
        if args.execution_id not in range(1,args.profile["executions"]+1):
            parser.error("--execution-id must identify one of the three declared executions")
        for field in ["threads", "blocks"]:
            if getattr(args,field) is not None and getattr(args,field) != args.profile[field]:
                parser.error(f"--{field} conflicts with the declared protocol")
            setattr(args,field,args.profile[field])
    elif args.execution_id is not None:
        parser.error("--execution-id requires --protocol")
    for key,value in {"threads":1,"iterations":50,"warmup":20,"blocks":7}.items():
        if getattr(args,key) is None: setattr(args,key,value)
    if any([args.anchor_binary, args.anchor_backend, args.anchor_report]) and not all([args.anchor_binary, args.anchor_backend, args.anchor_report]):
        parser.error("anchor comparison needs --anchor-binary, --anchor-backend and --anchor-report together")
    if min(args.threads, args.iterations, args.blocks) < 1 or args.warmup < 0:
        parser.error("threads, iterations, blocks must be positive; warmup must be nonnegative")
    if sys.byteorder != "little":
        parser.error("the comparison reader currently requires a little-endian host")
    args.binary = args.binary.resolve()
    args.workloads = args.workloads.resolve()
    if args.anchor_binary:
        args.anchor_binary = args.anchor_binary.resolve()
        args.anchor_report = args.anchor_report.resolve()
        original = json.loads(args.anchor_report.read_text())
        if original["status"] != "verified_exploratory_baseline" or original["protocol"]["backend"] != args.anchor_backend:
            raise ValueError("anchor report is unverified or names a different backend")
        anchor = original["environment"]
        if hashlib.sha256(args.anchor_binary.read_bytes()).hexdigest() != anchor["binary_sha256"]:
            raise ValueError("anchor binary differs from its original report")
        if hashlib.sha256(args.workloads.read_bytes()).hexdigest() != anchor["workloads_sha256"]:
            raise ValueError("workloads differ from the anchor contract")
    if args.profile:
        candidate_hash = hashlib.sha256(args.binary.read_bytes()).hexdigest()
        anchor_hash = hashlib.sha256(args.anchor_binary.read_bytes()).hexdigest()
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None
        validate_artifacts(args.profile,candidate_hash,anchor_hash,args.backend,args.anchor_backend,affinity)
        device = "cuda" if args.backend in ("cuda","cudnn","nvidia") else "cpu"
        if args.profile["device"] != device:
            parser.error("backend device differs from the declared protocol")
    if (args.output_dir / "comparison.json").exists():
        parser.error("output directory already contains a run; use a new directory to preserve evidence")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = suite["cases"]
    if args.cases:
        if set(args.cases) - {c["name"] for c in cases}:
            parser.error("unknown case name")
        cases = [c for c in cases if c["name"] in args.cases]
    device = "cuda" if args.backend in ("cuda", "cudnn", "nvidia") else "cpu"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA comparison requires a working PyTorch CUDA installation")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    sources = source_manifest()
    archive = args.output_dir / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sources:
            bundle.add(ROOT / path, arcname=path)
    environment = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "torch_cudnn": torch.backends.cudnn.version(),
        "torch_config": torch.__config__.show(), "rustc": command_output(["rustc", "-Vv"]),
        "cpu": command_output(["lscpu"]),
        "cpu_identity": cpu_identity(),
        "gpu_identity": command_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"]),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pstate,temperature.gpu,clocks.sm", "--format=csv"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "native_linkage": command_output(["ldd", str(args.binary)]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_DYNAMIC", "MKL_INTERFACE_LAYER", "MKL_THREADING_LAYER", "CUDA_VISIBLE_DEVICES", "CUDA_ARCH", "CUDA_HOST_CXX", "CUDA_ROOT", "RUSTFLAGS", "LD_LIBRARY_PATH"]},
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "workloads_sha256": hashlib.sha256(args.workloads.read_bytes()).hexdigest(),
        "source_sha256": sources, "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    if args.profile:
        shutil.copyfile(args.protocol,args.output_dir / "protocol.json")
        environment["protocol_sha256"] = profile_hash
    if args.anchor_binary:
        old_archive = args.anchor_report.parent / "source.tar.gz"
        if hashlib.sha256(old_archive.read_bytes()).hexdigest() != anchor["source_archive_sha256"]:
            raise ValueError("original anchor archive differs from its recorded hash")
        shutil.copyfile(old_archive, args.output_dir / "anchor-source.tar.gz")
        shutil.copyfile(args.anchor_report, args.output_dir / "anchor-report.json")
        environment["anchor"] = {"binary_sha256": anchor["binary_sha256"], "backend": args.anchor_backend,
            "source_sha256": anchor["source_sha256"], "source_archive_sha256": anchor["source_archive_sha256"],
            "original_report_sha256": hashlib.sha256(args.anchor_report.read_bytes()).hexdigest(),
            "native_linkage": command_output(["ldd", str(args.anchor_binary)])}
    report = {"schema_version": 1, "status": "running", "environment": environment,
              "protocol": {"backend": args.backend, "threads": args.threads, "blocks": args.blocks,
                           "iterations": args.iterations, "warmup": args.warmup,
                           "primary": "geometric mean of paired torch/Rust wall latency ratios across cases",
                           "precision": "strict FP32, TF32 disabled", "torch_mode": "eager inference",
                           "order": "alternating Rust-first and torch-first by case and block",
                           "scope": "warm prepared operator execution, inputs resident, setup/transfers excluded",
                           "allocation": "Rust preallocates outputs; torch.mm uses out=; conv1d/stft allocate outputs",
                           "inference": "exploratory; repeated warm blocks are not independent deployment trials"}, "cases": []}
    destination = args.output_dir / "comparison.json"
    if args.anchor_binary:
        report["protocol"]["primary"] = "geometric mean of paired retained-anchor/Rust wall latency ratios across cases"
        report["protocol"]["order"] = "alternating anchor-first and candidate-first; torch check follows each pair"
        report["protocol"]["parameters"] = "identical constant parameters; oneDNN snapshots and packs them during preparation"
    if args.profile:
        report["execution_id"] = args.execution_id
        report["protocol"]["timing_profile"] = args.profile
        report["protocol"]["iterations"] = None
        report["protocol"]["warmup"] = None
        report["protocol"]["order"] = "alternating by execution/case/block; torch check follows each pair"
    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        with torch.inference_mode():
            for index, case in enumerate(cases):
                run = prepare(case, device)
                record = {"case": case, "blocks": []}
                report["cases"].append(record)
                for block in range(args.blocks):
                    output = args.output_dir / f"{case['name']}.f32"
                    if args.anchor_binary:
                        anchor_output = args.output_dir / f"{case['name']}-anchor.f32"
                        if ((args.execution_id or 1)-1 + index + block) % 2 == 0:
                            anchor_report, control = rust_sample(case, args, anchor_output, args.anchor_binary, args.anchor_backend)
                            rust_report, actual = rust_sample(case, args, output)
                            order = "anchor,rust,torch"
                        else:
                            rust_report, actual = rust_sample(case, args, output)
                            anchor_report, control = rust_sample(case, args, anchor_output, args.anchor_binary, args.anchor_backend)
                            order = "rust,anchor,torch"
                        torch_report, expected = torch_sample(run, case, args, device)
                    elif (index + block) % 2 == 0:
                        rust_report, actual = rust_sample(case, args, output)
                        torch_report, expected = torch_sample(run, case, args, device)
                        order = "rust,torch"
                    else:
                        torch_report, expected = torch_sample(run, case, args, device)
                        rust_report, actual = rust_sample(case, args, output)
                        order = "torch,rust"
                    check = errors(actual, expected)
                    result = {"block": block, "order": order, "rust": rust_report,
                              "torch": torch_report, "correctness": check}
                    if args.anchor_binary:
                        result["anchor"] = anchor_report
                        result["anchor_correctness"] = errors(control, expected)
                    record["blocks"].append(result)
                    save()
                    if not check["passed"]:
                        raise RuntimeError(f"{case['name']} failed full-output correctness: {check}")
                    if args.anchor_binary and not result["anchor_correctness"]["passed"]:
                        raise RuntimeError(f"{case['name']} anchor failed correctness")
                record["summary"] = summary(record["blocks"])
                print(f"{case['name']}: Rust {record['summary']['rust_wall_us_median']:.2f} us, "
                      f"torch {record['summary']['torch_wall_us_median']:.2f} us, "
                      f"torch/Rust {record['summary']['wall_speedup_geomean']:.3f}x" +
                      (f", anchor/Rust {record['summary']['anchor_wall_speedup_geomean']:.3f}x" if args.anchor_binary else ""), flush=True)
                save()
        report["wall_speedup_geomean"] = math.exp(statistics.mean(
            math.log(c["summary"]["wall_speedup_geomean"]) for c in report["cases"]))
        report["status"] = "verified_exploratory_baseline"
        if args.anchor_binary:
            report["anchor_wall_speedup_geomean"] = math.exp(statistics.mean(
                math.log(c["summary"]["anchor_wall_speedup_geomean"]) for c in report["cases"]))
        save()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
