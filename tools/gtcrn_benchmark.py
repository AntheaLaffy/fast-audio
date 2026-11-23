#!/usr/bin/env python3
"""Characterize paired complete-model latency and separate observed process memory."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = [("b1-l4096", 1, 4096), ("b1-l16000", 1, 16000),
         ("b1-l64000", 1, 64000), ("b2-l16000", 2, 16000)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path):
    p = json.loads(path.read_text())
    if p["schema_version"] != 1 or p["name"] != "gtcrn-characterization-v1":
        raise ValueError("unknown model characterization protocol")
    if (p["atol"], p["rtol"], p["threads"], p["affinity"], p["blocks"], p["warmup"], p["samples_per_worker"]) != (2e-4, 2e-4, 1, [0], 7, 20, 1):
        raise ValueError("characterization numerical/timing contract changed")
    if [(c["name"], c["batch"], c["input_len"]) for c in p["cases"]] != CASES:
        raise ValueError("model shape distribution changed")
    if p["budget"]["promotion"] is not False or p["budget"]["optimization_candidates"] != 0:
        raise ValueError("characterization cannot promote an optimization")
    if p["model_revision"] != "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73" or p["precision"] != "FP32; TF32 disabled":
        raise ValueError("model/precision identity differs")
    for stratum in p["strata"].values():
        if set(stratum["iterations"]) != {c[0] for c in CASES} or any(type(n) is not int or n < 1 for n in stratum["iterations"].values()):
            raise ValueError("invalid per-case windows")
    expected = {"intel": ("mkl", "cpu", ["eager"]), "nvidia": ("cudnn", "cuda", ["eager", "cuda-graph", "cuda-graph-prepared-inverse"])}
    if set(p["strata"]) != set(expected):
        raise ValueError("hardware strata changed")
    for name, (backend, device, modes) in expected.items():
        if (p["strata"][name]["backend"], p["strata"][name]["device"], p["strata"][name]["torch_modes"]) != (backend, device, modes):
            raise ValueError("backend/device/mode contract changed")
    return p


def load_screen_protocol(path):
    p = json.loads(path.read_text())
    if p["schema_version"] != 1 or p["name"] != "mapped-access-screen-v1":
        raise ValueError("unknown native screen protocol")
    if (p["atol"], p["rtol"], p["threads"], p["affinity"], p["blocks"], p["warmup"], p["samples_per_worker"]) != (2e-4, 2e-4, 1, [0], 7, 20, 1):
        raise ValueError("screen numerical/timing contract changed")
    if [(c["name"], c["batch"], c["input_len"]) for c in p["cases"]] != CASES:
        raise ValueError("screen shape distribution changed")
    if p["budget"] != {"candidate_implementations": 1, "screen_executions_per_host": 1, "confirmation_executions": 0, "promotion": False}:
        raise ValueError("screen budget changed")
    if p["model_revision"] != "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73" or p["precision"] != "FP32; TF32 disabled":
        raise ValueError("screen model/precision changed")
    if p["strata"]["intel"]["iterations"] != dict(zip([c[0] for c in CASES], [200, 100, 50, 100])) or p["strata"]["nvidia"]["iterations"] != dict.fromkeys([c[0] for c in CASES], 100):
        raise ValueError("screen windows changed")
    if p["memory"]["cuda_replay_iterations"] != 200 or p["memory"]["requested_poll_seconds"] != 0.02:
        raise ValueError("screen memory sampling changed")
    if set(p["strata"]) != {"intel", "nvidia"}:
        raise ValueError("screen hardware strata changed")
    for name, backend, device in [("intel", "mkl", "cpu"), ("nvidia", "cudnn", "cuda")]:
        expected = {"mkl": name == "intel", "onednn": name == "intel", "cuda": name == "nvidia", "cudnn": name == "nvidia"}
        if (p["strata"][name]["backend"], p["strata"][name]["device"], p["strata"][name]["compiled_features"]) != (backend, device, expected):
            raise ValueError("screen backend/features changed")
    return p


def freeze_binary(binary, artifact_root):
    digest = sha(binary)
    directory = artifact_root / digest
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / "gtcrn-infer"
    if not frozen.exists():
        with binary.open("rb") as source, frozen.open("xb") as out:
            shutil.copyfileobj(source, out)
        frozen.chmod(0o555)
    if sha(frozen) != digest:
        raise ValueError("frozen executable has changed")
    return frozen, digest


def validate_reference(folder, p):
    reference = json.loads((folder / "reference.json").read_text())
    if reference["status"] != "verified_pytorch_reference" or [(c["name"], c["batch"], c["input_len"]) for c in reference["cases"]] != CASES:
        raise ValueError("reference coverage differs")
    bundle_path = folder / reference["parameter_bundle"]["file"]
    if sha(bundle_path) != reference["parameter_bundle"]["sha256"]:
        raise ValueError("parameter bundle changed")
    bundle = json.loads(bundle_path.read_text())
    pin = json.loads((ROOT / "models/gtcrn/upstream.json").read_text())
    if bundle["format"] != "fast-audio-gtcrn-f32-v1" or bundle["model"] != pin or reference["model"] != pin or pin["revision"] != p["model_revision"]:
        raise ValueError("model identity changed")
    for blob in bundle["parameters"].values():
        if sha(folder / blob["file"]) != blob["sha256"]:
            raise ValueError("parameter bytes changed")
    for case in reference["cases"]:
        if set(case["tensors"]) != {"input", "window", "spectrum", "enhanced", "waveform"}:
            raise ValueError("incomplete audio reference")
        for blob in case["tensors"].values():
            if sha(folder / case["name"] / blob["file"]) != blob["sha256"]:
                raise ValueError("audio fixture changed")
    return reference


def process_group(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[2])
    except (OSError, ValueError, IndexError):
        return None


def gpu_sample(group):
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True)
    if query.returncode:
        return {"error": query.stderr.strip(), "processes": [], "mib": None}
    rows = []
    for row in csv.reader(query.stdout.splitlines()):
        if len(row) != 2:
            continue
        try:
            pid, mib = int(row[0].strip()), float(row[1].strip())
        except ValueError:
            continue
        if process_group(pid) == group:
            rows.append({"pid": pid, "mib": mib})
    return {"processes": rows, "mib": sum(r["mib"] for r in rows)}


def telemetry(device):
    snapshot = {"timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    for name, path in {"cpu0_frequency_khz": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
                       "cpu0_governor": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                       "load_average": "/proc/loadavg"}.items():
        try:
            snapshot[name] = Path(path).read_text().strip()
        except OSError:
            snapshot[name] = None
    if device == "cuda":
        query = subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw", "--format=csv,noheader"], text=True, capture_output=True)
        snapshot["gpu"] = query.stdout.strip() if query.returncode == 0 else query.stderr.strip()
    return snapshot


def run_command(command, directory, engine, sample_gpu, poll_seconds, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    resource_path = directory / f"{engine}-resource.json"
    wrapped = ["/usr/bin/time", "-f", '{"max_rss_kib":%M,"user_seconds":%U,"system_seconds":%S,"command_wall_seconds":%e}',
               "-o", str(resource_path), *command]
    samples = []
    with (directory / f"{engine}-stdout.txt").open("w") as stdout, (directory / f"{engine}-stderr.txt").open("w") as stderr:
        begin = time.monotonic()
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        process = subprocess.Popen(wrapped, stdout=stdout, stderr=stderr, start_new_session=True, env=environment)
        try:
            if sample_gpu:
                while process.poll() is None:
                    if time.monotonic() - begin > timeout:
                        raise subprocess.TimeoutExpired(command, timeout)
                    samples.append({"elapsed_seconds": time.monotonic() - begin, **gpu_sample(process.pid)})
                    time.sleep(poll_seconds)
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
    if code:
        raise RuntimeError(f"{engine} worker exited {code}; see {engine}-stderr.txt in {directory}")
    resource = json.loads(resource_path.read_text())
    observation = None
    if sample_gpu:
        valid = [s["mib"] for s in samples if s["mib"] is not None and s["processes"]]
        intervals = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])]
        observation = {"scope": "sampled compute-process-group memory; lower bound on peak; includes CUDA context/vendor overhead",
                       "peak_sampled_mib": max(valid) if valid else None, "samples": samples,
                       "requested_poll_seconds": poll_seconds,
                       "maximum_observed_interval_seconds": max(intervals) if intervals else None}
        (directory / f"{engine}-gpu-memory.json").write_text(json.dumps(observation, indent=2) + "\n")
    return {"command": wrapped, "resource": resource, "gpu_memory": observation}


def check_execution(report, engine, case, warmup, iterations, backend, mode):
    counts = report["protocol"]
    if report["backend"] != (backend if engine == "native" else "torch") or len(report["cases"]) != 1 or report["cases"][0]["case"] != case:
        raise ValueError("worker execution identity differs")
    if (counts["warmup"], counts["iterations"], counts["samples"], counts["traces"], counts["pipeline_only"]) != (warmup, iterations, 1, False, True):
        raise ValueError("worker counts or execution scope differ")
    if engine == "torch" and report["mode"] != mode:
        raise ValueError("PyTorch mode differs")
    if engine == "native":
        expected = {"mkl": backend == "mkl", "onednn": backend == "mkl", "cuda": backend == "cudnn", "cudnn": backend == "cudnn"}
        if report.get("compiled_features") != expected:
            raise ValueError("native compiled feature set differs")
    values = report["cases"][0]["timing"]["pipeline"]["wall_us"]
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
        raise ValueError("invalid completed wall latency")


def interval(logs, seed):
    rng = random.Random(seed)
    bootstrap = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(5000))
    return [bootstrap[125], bootstrap[4874]]


def summarize(blocks, candidate="native", comparator="torch"):
    if {r["block"] for r in blocks} != set(range(7)):
        raise ValueError("summary needs all seven declared blocks")
    for block in range(7):
        records = [r for r in blocks if r["block"] == block]
        if len(records) != 4 or {r["case"] for r in records} != {c[0] for c in CASES}:
            raise ValueError("missing or duplicated case in a block")
    cases = {}
    for name, _, _ in CASES:
        records = [b for b in blocks if b["case"] == name]
        logs = [math.log(r[comparator]["record"]["timing"]["pipeline"]["wall_us"][0] / r[candidate]["record"]["timing"]["pipeline"]["wall_us"][0]) for r in records]
        cases[name] = {"paired_ratio_geomean": math.exp(statistics.mean(logs)), "exploratory_95pct_interval": interval(logs, 1701),
                       **{f"{engine}_wall_us_median": statistics.median(r[engine]["record"]["timing"]["pipeline"]["wall_us"][0] for r in records) for engine in [candidate, comparator]},
                       **{f"{engine}_max_command_rss_kib": max(r[engine]["resource"]["max_rss_kib"] for r in records) for engine in [candidate, comparator]}}
    block_logs = []
    for block in sorted({r["block"] for r in blocks}):
        records = [r for r in blocks if r["block"] == block]
        if len(records) != len(CASES):
            raise ValueError("cannot summarize an incomplete block")
        block_logs.append(statistics.mean(math.log(r[comparator]["record"]["timing"]["pipeline"]["wall_us"][0] / r[candidate]["record"]["timing"]["pipeline"]["wall_us"][0]) for r in records))
    return {"cases": cases, "four_case_paired_ratio_geomean": math.exp(statistics.mean(block_logs)),
            "exploratory_95pct_interval": interval(block_logs, 1729), "promotion": False}


def collect(args):
    try:
        from .compare_torch import command_output, errors, source_manifest
        from .check_gru import load
    except ImportError:
        from compare_torch import command_output, errors, source_manifest
        from check_gru import load
    import torch
    screen = args.control_binary is not None
    p = load_screen_protocol(args.protocol) if screen else load_protocol(args.protocol)
    torch.set_num_threads(p["threads"])
    torch.set_num_interop_threads(1)
    selected = p["strata"][args.stratum]
    if not screen and args.torch_mode not in selected["torch_modes"]:
        raise ValueError("mode is outside the declared stratum")
    if screen and args.torch_mode != "eager":
        raise ValueError("a native screen does not select a Torch mode")
    if sorted(os.sched_getaffinity(0)) != p["affinity"]:
        raise ValueError("run the collector under declared CPU affinity {0}")
    if args.output_dir.exists():
        raise ValueError("use a new output directory")
    reference = validate_reference(args.reference, p)
    frozen, digest = freeze_binary(args.binary, args.artifact_root)
    candidate, comparator = ("candidate", "control") if screen else ("native", "torch")
    control_binary = control_digest = control_report = control_path = None
    if screen:
        try:
            from .verify_archive import verify
        except ImportError:
            from verify_archive import verify
        control_binary, control_digest = freeze_binary(args.control_binary, args.artifact_root)
        control_path = ROOT / selected["control_report"]
        control_report = json.loads(control_path.read_text())
        verify(control_path)
        if control_digest != selected["control_binary_sha256"] or control_digest != control_report["environment"]["binary_sha256"] or digest == control_digest:
            raise ValueError("screen control/candidate identity differs")
        if control_report["status"] != "verified_characterization" or control_report["environment"]["reference_sha256"] != sha(args.reference / "reference.json"):
            raise ValueError("screen control/reference context differs")
    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.protocol, args.output_dir / "protocol.json")
    source = source_manifest()
    archive = args.output_dir / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, checksum in source.items():
            if sha(ROOT / name) != checksum:
                raise ValueError("source changed before archive")
            bundle.add(ROOT / name, arcname=name)
    report = {"schema_version": 1, "status": "running", "protocol": {"name": p["name"], "stratum": args.stratum, "torch_mode": args.torch_mode,
              "scope": p["scope"], "counts": p, "profile_sha256": sha(args.protocol)},
              "environment": {"timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), "python": sys.version,
                              "torch": torch.__version__, "torch_config": torch.__config__.show(),
                              "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu": command_output(["lscpu"]),
                              "native_linkage": command_output(["ldd", str(frozen.resolve())]),
                              "gpu": command_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version,clocks.current.sm,temperature.gpu,power.draw", "--format=csv,noheader"]),
                              "binary_sha256": digest, "frozen_binary": str(frozen.resolve()), "reference_sha256": sha(args.reference / "reference.json"),
                              "source_revision": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) if (ROOT / ".git").exists() else None,
                              "source_sha256": source, "source_archive_sha256": sha(archive)}, "blocks": [], "memory": []}
    destination = args.output_dir / "comparison.json"
    if screen:
        report["protocol"].pop("torch_mode")
        report["protocol"]["comparator"] = "frozen_native_control"
        report["environment"]["anchor"] = {**control_report["environment"], "native_linkage": command_output(["ldd", str(control_binary)])}
        shutil.copyfile(control_path, args.output_dir / "anchor-comparison.json")
        shutil.copyfile(control_path.parent / "source.tar.gz", args.output_dir / "anchor-source.tar.gz")
        for name, binary in [("gtcrn-infer.gz", frozen), ("anchor-gtcrn-infer.gz", control_binary)]:
            (args.output_dir / name).write_bytes(gzip.compress(binary.read_bytes(), mtime=0))
        (args.output_dir / "artifact-seal.json").write_text(json.dumps({"protocol_sha256": sha(args.protocol), "candidate_binary_sha256": digest,
            "control_binary_sha256": control_digest, "candidate_source_sha256": source, "control_report_sha256": sha(control_path)}, indent=2) + "\n")

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def execute(engine, case, directory, iterations, sample_gpu=False):
        folder = directory / engine
        if engine == candidate or screen:
            binary = control_binary if engine == "control" else frozen
            command = [str(binary.resolve()), selected["backend"], str(args.reference.resolve()), str(folder.resolve()),
                       str(p["warmup"]), "1", str(iterations), "--case", case, "--no-trace", "--pipeline-only"]
        else:
            command = [sys.executable, str(ROOT / "tools/gtcrn_torch_worker.py"), "--reference", str(args.reference.resolve()),
                       "--case", case, "--device", selected["device"], "--mode", args.torch_mode, "--warmup", str(p["warmup"]),
                       "--iterations", str(iterations), "--output-dir", str(folder.resolve())]
        metadata = run_command(command, directory, engine, sample_gpu, p["memory"]["requested_poll_seconds"], p["worker_timeout_seconds"])
        native = json.loads((folder / "inference.json").read_text())
        check_execution(native, "native" if screen else engine, case, p["warmup"], iterations, selected["backend"], args.torch_mode)
        record = native["cases"][0]
        expected = next(c for c in reference["cases"] if c["name"] == case)
        checks = {}
        for key in ["spectrum", "enhanced", "waveform"]:
            wanted = load(args.reference / case, expected["tensors"][key])
            if key != "waveform":
                wanted = wanted.transpose(1, 2)
            checks[key] = errors(load(folder, record[key]), wanted)
        if not all(c["passed"] for c in checks.values()):
            (directory / f"{engine}-numerical-failure.json").write_text(json.dumps(checks, indent=2) + "\n")
            raise ValueError("worker failed original numerical gate")
        if sha(frozen) != digest or (screen and sha(control_binary) != control_digest) or source_manifest() != source:
            raise ValueError("frozen source or executable changed during collection")
        return {**metadata, "record": record, "checks": checks,
                "worker_environment": native.get("environment"), "relative_folder": str(folder.relative_to(args.output_dir))}

    def compare_pair(pair):
        left, right = pair[candidate], pair[comparator]
        left_folder = args.output_dir / left["relative_folder"]
        right_folder = args.output_dir / right["relative_folder"]
        pair["paired_checks"] = {key: errors(load(left_folder, left["record"][key]), load(right_folder, right["record"][key])) for key in ["spectrum", "enhanced", "waveform"]}
        if not all(c["passed"] for c in pair["paired_checks"].values()):
            raise ValueError("paired outputs failed original numerical gate")
        if screen:
            pair["paired_bit_identical"] = {key: (left_folder / left["record"][key]["file"]).read_bytes() == (right_folder / right["record"][key]["file"]).read_bytes() for key in ["spectrum", "enhanced", "waveform"]}
            pair["planned_storage_equal"] = all(left["record"][key] == right["record"][key] for key in ["graph_workspace_bytes", "parameter_bytes", "inverse_workspace_bytes"])
            if not all(pair["paired_bit_identical"].values()) or not pair["planned_storage_equal"]:
                raise ValueError("screen exact-output or planned-storage gate failed")

    save()
    try:
        for block in range(p["blocks"]):
            names = [c[0] for c in CASES]
            order = names[block % len(names):] + names[:block % len(names)]
            for name in order:
                engines = [candidate, comparator] if block % 2 == 0 else [comparator, candidate]
                if screen:
                    engines.reverse()
                pair = {"block": block, "case": name, "order": engines, "case_order": order,
                        "telemetry_before": telemetry(selected["device"])}
                directory = args.output_dir / "latency" / f"block-{block}" / name
                report["blocks"].append(pair)
                for engine in engines:
                    pair[engine] = execute(engine, name, directory, selected["iterations"][name])
                pair["telemetry_after"] = telemetry(selected["device"])
                compare_pair(pair)
                save()
                print(args.stratum, "native-control" if screen else args.torch_mode, "block", block, name, flush=True)
        report["summary"] = summarize(report["blocks"], candidate, comparator)
        if selected["device"] == "cuda":
            for name, _, _ in CASES:
                directory = args.output_dir / "memory" / name
                result = {"case": name, "scope": "Separate instrumentation; timing excluded from latency summary"}
                report["memory"].append(result)
                for engine in [candidate, comparator]:
                    result[engine] = execute(engine, name, directory, p["memory"]["cuda_replay_iterations"], True)
                if screen:
                    compare_pair(result)
                save()
                print("memory", args.torch_mode, name, flush=True)
        validate_reference(args.reference, p)
        if sha(args.reference / "reference.json") != report["environment"]["reference_sha256"]:
            raise ValueError("reference identity changed during collection")
        report["memory_status"] = ("sampled_lower_bound" if all(entry[engine]["gpu_memory"]["peak_sampled_mib"] is not None for entry in report["memory"] for engine in [candidate, comparator])
                                   else "partially_unobserved") if selected["device"] == "cuda" else "command_rss_only"
        report["status"] = "verified_native_screen" if screen else "verified_characterization"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise
    save()
    print(report["status"], report["summary"]["four_case_paired_ratio_geomean"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--control-binary", type=Path, help="Select the declared paired native mechanism screen")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--stratum", choices=["intel", "nvidia"], required=True)
    parser.add_argument("--torch-mode", choices=["eager", "cuda-graph", "cuda-graph-prepared-inverse"], default="eager")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks/protocols/gtcrn-characterization-v1.json")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "target/benchmark-artifacts/gtcrn")
    parser.add_argument("--output-dir", type=Path, required=True)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
