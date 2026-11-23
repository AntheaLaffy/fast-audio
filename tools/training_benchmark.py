#!/usr/bin/env python3
"""Counterbalanced complete-training baseline with full independently evolving controls."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.training_protocol import PROTOCOL, load_protocol, sha, require, order, control, check_snapshot, check_profile_snapshot, fixtures, snapshot_update
from tools.pointwise_benchmark import Peer, log_command
from tools.compare_torch import source_manifest, cpu_identity, command_output
from tools.gtcrn_benchmark import telemetry


class TrainingPeer(Peer):
    def __init__(self, *args):
        super().__init__(*args)
        try:
            self.requests = (self.directory / "stdin.jsonl").open("xb")
        except Exception:
            super().close()
            raise

    def request(self, message, timeout):
        if hasattr(self, "requests"):
            self.requests.write((json.dumps(message) + "\n").encode())
            self.requests.flush()
        return super().request(message, timeout)

    def close(self):
        try:
            result = super().close()
        finally:
            self.requests.close()
        result["log_sha256"]["stdin.jsonl"] = sha(self.directory / "stdin.jsonl")
        return result


def selected_cases(p, names, validation_only):
    if names is None:
        return p["cases"]
    require(validation_only, "partial case selection is validation-only")
    require(len(names) == len(set(names)) and bool(names), "duplicate/empty validation case selection")
    cases = {c["name"]: c for c in p["cases"]}
    require(set(names) <= set(cases), "unknown training case selection")
    return [cases[name] for name in names]


def seal_snapshot(snapshot, checked, destination):
    objects = destination / "objects"
    objects.mkdir(exist_ok=True)
    for name, record in checked["payloads"].items():
        raw = Path(snapshot[name]["file"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == record["sha256"], "snapshot changed after checking")
        packed = objects / (record["sha256"] + ".f32.gz")
        if not packed.exists():
            with packed.open("xb") as stream:
                stream.write(gzip.compress(raw, mtime=0))
        require(hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest() == record["sha256"], "snapshot compression differs")
        record.update(object=str(packed.relative_to(destination)), gzip_sha256=sha(packed))
    return checked


def validate_ready(native, torch, case, p, platform):
    require(native["status"] == torch["status"] == "ready" and native["engine"] == "native" and torch["engine"] == "torch", "training workers not ready")
    require(native["case"] == torch["case"] == case["name"] and native["platform"] == platform, "worker case/platform differs")
    require(native["features"]["mkl" if platform=="intel" else "cuda"] is True, "required native training backend missing")
    expected = p["strata"][platform]["modes"][1:]
    require(set(torch["modes"]) == set(expected), "required training mode missing")
    for mode, item in torch["modes"].items():
        require(item["complete_step"] is True and item["capture"] == mode.endswith("cuda-graph"), "partial training/capture mode")
        require(item["frontend"] == ("torch.func.grad_and_value" if mode.startswith("compiled") else "autograd.backward"), "training frontend differs")
    require(torch["torch"] == ("2.14.0+cu130" if platform == "nvidia" else "2.14.1+cpu"), "pinned Torch differs")
    require(torch["precision"] == {"matmul": "highest", "allow_tf32": False,"fp32_precision":"ieee"} and torch["threads"] == torch["interop_threads"] == 1, "training precision/thread policy differs")
    require(torch["affinity"] == p["affinity"], "Torch training affinity differs")
    require(torch["compiler"]["policy"] == {"use_fast_math": False, "cpp_unsafe_math": False, "cpp_contract": "off", "cuda_fast_math": False}, "compiled math policy differs")
    require(not torch["dynamo_counters"].get("graph_break"), "compiled training graph break reported")
    require(native["seed_payload_bytes"] == 8 and native["resources"]["view_index_bytes"] == 0, "unexpected training seed/view index resources")
    for name in ("preparation_us", "first_step_us"):
        require(math.isfinite(native[name]) and native[name] > 0 and all(math.isfinite(item[name]) and item[name] > 0 for item in torch["modes"].values()), "missing preparation/first-step timing")


def validate_close(result, p, platform, validation_only, native):
    reply = result.get("reply", {})
    require(result["returncode"] == 0 and "shutdown_error" not in result and reply.get("status") == "stopped", "training worker did not stop cleanly")
    counts = [reply["validation_steps"]] if native else list(reply["validation_steps"].values())
    blocks = [reply["measured_blocks"]] if native else list(reply["measured_blocks"].values())
    repeat = [reply["pending_repeat"]] if native else list(reply["pending_repeat"].values())
    expected = 0 if validation_only else p["strata"][platform]["blocks"]
    require(all(v == p["validation_steps"] for v in counts) and all(v == expected for v in blocks) and all(v is None for v in repeat), "training shutdown scope incomplete")


def import_nsys_trace(tool, prefix, environment, timeout, capture_version):
    """Some packaged CLIs capture successfully but cannot locate their host importer."""
    if prefix.with_suffix(".nsys-rep").is_file():
        return None
    explicit = environment.get("NSYS_QDSTRM_IMPORTER")
    candidates = [Path(explicit)] if explicit else [
        Path(found) for found in [shutil.which("QdstrmImporter", path=environment.get("PATH"))] if found
    ] + [path.parent.parent / "host-linux-x64/QdstrmImporter" for path in (Path(tool), Path(tool).resolve())] + [
        Path("/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter")
    ]
    importer = next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
    require(importer is not None, "Nsight host importer unavailable; set NSYS_QDSTRM_IMPORTER")
    version = command_output([str(importer), "--version"])
    token = r"\d+\.\d+\.\d+\.\d+(?:-[\w.]+)?"
    captured = re.search(token, capture_version) if isinstance(capture_version, str) else None
    imported = re.search(token, version) if isinstance(version, str) else None
    require(captured is not None and imported is not None and captured.group() == imported.group(), "Nsight capture/importer versions differ")
    raw = prefix.with_suffix(".qdstrm")
    require(raw.is_file(), "Nsight raw stream missing")
    original = sha(raw)
    copied = prefix.with_name("import-input.qdstrm")
    require(not copied.exists(), "preserve previous Nsight import input")
    shutil.copyfile(raw, copied)
    require(sha(copied) == original, "Nsight input copy differs")
    # The importer can rewrite headers as well as append metadata. Keep the capture intact.
    try:
        execution = log_command([str(importer), "--input-file", str(copied), "--output-file", str(prefix.with_suffix(".nsys-rep"))],
                                prefix.parent / "import", timeout, environment)
    finally:
        require(sha(raw) == original, "original Nsight stream changed during copied import")
    require(execution["returncode"] == 0 and not execution["timeout"] and prefix.with_suffix(".nsys-rep").is_file(), "Nsight copied-input import failed")
    return {"tool": str(importer), "tool_sha256": sha(importer), "version": version,
            "original_sha256": original, "copied_input_after_sha256": sha(copied), "execution": execution}


def profiles(destination, build, p, platform, environment, budget, unchanged):
    result = {"scope": "Separate instrumented complete-step loops; startup/dumps excluded from GPU range; CPU command profiles include startup",
              "overhead": "Not isolated from startup and different loop counts; these timings do not enter primary samples"}
    native = build["binary"]
    if platform == "nvidia":
        tool = shutil.which("nsys")
        if not tool:
            return {**result, "available": False, "reason": "nsys not on PATH"}
        engine = "nsys"
    else:
        tool = shutil.which("perf")
        if tool:
            probe = log_command([tool, "stat", "-e", "cycles:u", "--", "true"], destination / "profile-permission", budget("request_seconds"), environment)
            result["perf_permission_probe"] = probe
            if probe["returncode"] != 0:
                tool = None
        engine = "perf"
        if not tool:
            tool = shutil.which("gprofng"); engine = "gprofng"
        if not tool:
            return {**result, "available": False, "reason": "perf unavailable/denied and gprofng not on PATH"}
    result.update(available=True, tool=tool, engine=engine, version=command_output([tool, "--version"]), cases={})
    for name, count in p["profiles"][platform].items():
        unchanged()
        directory = destination / "profiles" / name
        worker = destination / "profile-workers" / name
        command = [native, "--protocol", str(PROTOCOL), "--platform", platform, "--case", name, "--output-dir", str(worker), "--profile"]
        prefix = directory / "trace"
        if engine == "nsys":
            launch = [tool, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=false", "--output", str(prefix), *command]
        elif engine == "perf":
            launch = [tool, "record", "-e", "cycles:u", "-g", "--call-graph", "dwarf", "-o", str(prefix.with_suffix(".data")), "--", *command]
        else:
            launch = [tool, "collect", "app", "-O", str(prefix.with_suffix(".er")), *command]
        execution = log_command(launch, directory, budget("worker_ready_seconds"), environment)
        require(execution["returncode"] == 0 and not execution["timeout"], "native training profiler command failed")
        replies = [json.loads(line) for line in (directory / "stdout.log").read_text().splitlines() if line.startswith('{')]
        require(len(replies) == 2 and replies[0]["status"] == "ready" and replies[1]["status"] == "profiled" and replies[1]["iterations"] == count, "profiled training scope differs")
        entry = {"execution": execution, "iterations": count, "worker": str(worker), "summary": None}
        case = next(c for c in p["cases"] if c["name"] == name)
        entry["checked"] = seal_snapshot(replies[1]["snapshot"], check_profile_snapshot(replies[1]["snapshot"], worker, case), destination)
        if engine == "nsys":
            entry["import"] = import_nsys_trace(tool, prefix, environment, budget("worker_ready_seconds"), result["version"])
            summary = [tool, "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum", "--format", "csv", str(prefix.with_suffix(".nsys-rep"))]
        elif engine == "perf":
            summary = [tool, "report", "--stdio", "-i", str(prefix.with_suffix(".data"))]
        else:
            summary = [tool, "display", "text", "-functions", str(prefix.with_suffix(".er"))]
        entry["summary"] = log_command(summary, directory / "summary", budget("worker_ready_seconds"), environment)
        require(entry["summary"]["returncode"] == 0 and not entry["summary"]["timeout"], "training profile summary failed")
        entry["files"] = {str(f.relative_to(directory)): sha(f) for f in directory.rglob('*') if f.is_file()}
        result["cases"][name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["intel", "nvidia"], required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    p = load_protocol()
    cases = selected_cases(p, args.cases, args.validate_only)
    require(not args.output_dir.exists(), "use a new characterization directory; preserve failures")
    build = json.loads(args.build_receipt.read_text())
    manifest = source_manifest()
    require(build["status"] == "built" and build["platform"] == args.platform and build["source_sha256"] == manifest and sha(build["binary"]) == build["binary_sha256"], "native build identity/source differs")
    destination = args.output_dir.absolute(); destination.mkdir(parents=True)
    native_build = destination / "native-build"; native_build.mkdir()
    for name in ("build-receipt.json", "build.log", "source.tar.gz", "fa-training.gz"):
        shutil.copyfile(args.build_receipt.parent / name, native_build / name)
    if build["producer_proof"]["kind"]=="sealed_manifest":
        shutil.copyfile(args.build_receipt.parent/build["producer_proof"]["file"],native_build/build["producer_proof"]["file"])
    report_path = destination / "comparison.json"
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=p["precision"]["cublas_workspace_config"])
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(destination / "compiler-cache")
    os.sched_setaffinity(0, p["affinity"])
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    report = {"schema_version": 1, "status": "running", "pid": os.getpid(), "producer_commit": build["producer_commit"],
              "platform": args.platform, "validation_only": args.validate_only, "protocol_sha256": sha(PROTOCOL),
              "build_receipt": {"file": "native-build/build-receipt.json", "sha256": sha(args.build_receipt)},
              "source_sha256": manifest, "binary_sha256": build["binary_sha256"], "environment": {"cpu": cpu_identity(),
              "python": sys.executable, "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
              "gpu_identity": command_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader"]) if args.platform=="nvidia" else None,
              "nvcc": command_output(["nvcc", "--version"]) if args.platform == "nvidia" else None}, "cases": [],
              "promotion": False, "limitations": ["Fixed procedural workloads, shared hardware and process-resident blocks are characterization, not independent optimization confirmation", "CUDA event intervals can include dispatch gaps; explicit bytes are not process/vendor/device peaks"]}
    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n"); temporary.replace(report_path)
    started = time.monotonic()
    def unchanged():
        require(source_manifest() == manifest and sha(build["binary"]) == build["binary_sha256"], "training source/binary changed")
        require(time.monotonic() - started < p["limits"]["context_seconds"], "training context budget exhausted")
    def budget(name):
        return min(p["limits"][name], max(1, p["limits"]["context_seconds"] - (time.monotonic() - started)))
    save()
    try:
        for index, case in enumerate(cases):
            unchanged()
            case_dir = destination / case["name"]; case_dir.mkdir()
            record = {"case": case, "readiness": {}, "workers": {}, "validation": [], "blocks": [], "telemetry_before": telemetry(p["strata"][args.platform]["device"])}
            report["cases"].append(record); save()
            peers = {}
            try:
                for engine in ("native", "torch"):
                    output = case_dir / (engine + "-data")
                    command = ([build["binary"]] if engine == "native" else [sys.executable, "tools/training_torch_worker.py"])
                    command += ["--protocol", str(PROTOCOL), "--platform", args.platform, "--case", case["name"], "--output-dir", str(output)]
                    peer = TrainingPeer(command, case_dir / (engine + "-transport"), environment)
                    peers[engine] = peer
                    record["workers"][engine] = {"command": command, "pid": peer.process.pid}
                    save()
                    ready = peer.receive(budget("worker_ready_seconds"))
                    record["readiness"][engine] = ready
                    record["workers"][engine]["startup_ready_us"] = (time.monotonic_ns() - peer.started) / 1000
                    require(sorted(os.sched_getaffinity(peer.process.pid)) == p["affinity"], "native/Torch worker affinity differs")
                    save()
                validate_ready(record["readiness"]["native"], record["readiness"]["torch"], case, p, args.platform)
                previous = {mode:fixtures(case)[2] for mode in p["strata"][args.platform]["modes"]}
                for step in range(p["validation_steps"]):
                    unchanged()
                    controls = [control(case, step + 1, fp64, p["rate"]) for fp64 in (True, False)]
                    for mode in p["strata"][args.platform]["modes"]:
                        engine = "native" if mode == "native" else "torch"
                        request = {"action": "validate", "step": step, **({"mode": mode} if engine == "torch" else {})}
                        reply = peers[engine].request(request, budget("request_seconds"))
                        require(reply["status"] == "validated" and reply["step"] == step, "training validation response differs")
                        checked = check_snapshot(reply["snapshot"], case_dir / (engine + "-data"), controls)
                        update, previous[mode] = snapshot_update(reply["snapshot"], previous[mode], p["rate"])
                        checked["sgd_update"] = update
                        record["validation"].append({"mode": mode, "step": step, "checked": seal_snapshot(reply["snapshot"], checked, destination)})
                    save()
                if not args.validate_only:
                    window = p["windows"][case["regime"]]
                    controls = [control(case, window["warmup"] + window["iterations"], fp64, p["rate"]) for fp64 in (True, False)]
                    anchors = {}
                    for block in range(p["strata"][args.platform]["blocks"]):
                        unchanged()
                        modes = order(p, args.platform, index, block)
                        item = {"block": block, "order": modes, "modes": {},"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}; record["blocks"].append(item); save()
                        for mode in modes:
                            engine = "native" if mode == "native" else "torch"
                            extra = {"mode": mode} if engine == "torch" else {}
                            reply = peers[engine].request({"action": "measure", "block": block, **extra}, budget("request_seconds"))
                            require(reply["status"] == "measured" and reply["block"] == block and reply["warmup"] == window["warmup"] and reply["iterations"] == window["iterations"], "training timing window differs")
                            require(math.isfinite(reply["timing"]["wall_us"]) and reply["timing"]["wall_us"] > 0, "invalid training timing")
                            device_time = reply["timing"]["device_us"]
                            require((device_time is None) if args.platform == "intel" else (device_time is not None and math.isfinite(device_time) and device_time > 0), "training device timing scope differs")
                            if mode == "native":
                                require(reply["timing"]["hot_rust_allocations"] == 0, "hot native Rust allocations")
                            checked = check_snapshot(reply["snapshot"], case_dir / (engine + "-data"), controls)
                            first = seal_snapshot(reply["snapshot"], checked, destination)
                            identity = {k: v["sha256"] for k, v in first["payloads"].items()}
                            require(mode not in anchors or anchors[mode] == identity, "training mode changed bits between reset windows")
                            anchors[mode] = identity
                            repeated = peers[engine].request({"action": "repeat", "block": block, **extra}, budget("request_seconds"))
                            require(repeated["status"] == "repeated" and repeated["block"] == block, "training repeat response differs")
                            checked_repeat = check_snapshot(repeated["snapshot"], case_dir / (engine + "-data"), controls)
                            require({k: v["sha256"] for k, v in first["payloads"].items()} == {k: v["sha256"] for k, v in checked_repeat["payloads"].items()}, "training reset/re-execution is not bit-repeatable")
                            item["modes"][mode] = {"timing": reply["timing"], "checked": first,
                                                   "repeat_payloads": checked_repeat["payloads"], "exact_repeat": True,
                                                   "context_elapsed_seconds":time.monotonic()-started}
                            save()
                        print(args.platform, case["name"], "block", block, "passed", flush=True)
            finally:
                for engine, peer in peers.items():
                    record["workers"][engine]["shutdown"] = peer.close()
                save()
            for engine in ("native", "torch"):
                validate_close(record["workers"][engine]["shutdown"], p, args.platform, args.validate_only, engine == "native")
            record["telemetry_after"] = telemetry(p["strata"][args.platform]["device"])
            if not args.validate_only:
                medians = {m: statistics.median(b["modes"][m]["timing"]["wall_us"] for b in record["blocks"]) for m in p["strata"][args.platform]["modes"]}
                best = min((m for m in medians if m != "native"), key=lambda m: medians[m])
                record["summary"] = {"median_wall_us": medians, "strongest_torch_mode": best, "strongest_torch_over_native": medians[best] / medians["native"]}
            save(); print(args.platform, case["name"], "complete", flush=True)
        unchanged()
        if not args.validate_only:
            report["profiles"] = profiles(destination, build, p, args.platform, environment, budget, unchanged)
            save()
        unchanged()
        report["status"] = "verified_training_readiness_only" if args.validate_only else "complete_training_characterization"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc()); save(); raise
    report["elapsed_seconds"] = time.monotonic() - started
    save(); print(report["status"], flush=True)


if __name__ == "__main__":
    main()
