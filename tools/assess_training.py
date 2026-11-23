#!/usr/bin/env python3
"""Audit training source/executable/transport/payload scope without promoting a result."""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.training_protocol import PROTOCOL, load_protocol, sha, require, order, shapes, fixtures, tensor_bytes, control, sgd_update
from tools.training_benchmark import validate_ready, validate_close


def verify_objects(checked, directory, case=None, controls=None):
    require(isinstance(checked.get("payloads"), dict) and bool(checked["payloads"]), "missing checked payloads")
    expected = None
    if case is not None:
        xs, ts, ps = shapes(case)
        expected = {"prediction": ts, "loss": [], "owner-0": xs, "owner-1": ts}
        expected.update({"owner-"+str(i+2): shape for i, shape in enumerate(ps)})
        expected.update({"gradient-"+str(i): shape for i, shape in enumerate(ps)})
        require(set(checked["payloads"]) == set(expected), "sealed training payload scope differs")
        x, target, _ = fixtures(case)
    for name, record in checked["payloads"].items():
        digest = record["sha256"]
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and record["object"] == "objects/" + digest + ".f32.gz", "object identity differs")
        path = directory / record["object"]
        require(sha(path) == record["gzip_sha256"], "compressed training payload changed")
        raw = gzip.decompress(path.read_bytes())
        require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == digest, "raw training payload changed")
        if expected is not None:
            import torch
            require(record["shape"] == expected[name] and all(type(v) is int for v in record["shape"]), "sealed payload shape differs")
            require(len(raw) == math.prod(expected[name])*4, "sealed payload extent differs")
            actual = torch.frombuffer(bytearray(raw), dtype=torch.float32).double().reshape(expected[name])
            require(bool(torch.isfinite(actual).all()), "sealed training payload non-finite")
            if name in ("owner-0", "owner-1"):
                require(raw == tensor_bytes(x if name == "owner-0" else target), "sealed training sources changed")
            for values in controls or []:
                reference = values[name].detach().double().cpu()
                require(bool(torch.isfinite(reference).all()) and bool(((actual-reference).abs() <= 2e-5+2e-5*reference.abs()).all()), "recovered training numerical control failed: "+name)


def verify_gpu_events(database, iterations, parameter_count):
    connection = sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True)
    try:
        kernels = connection.execute("SELECT k.start,k.end,s.value,k.correlationId,k.deviceId,k.streamId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start").fetchall()
        runtime = connection.execute("SELECT a.start,a.end,s.value,a.correlationId,a.returnValue "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId ORDER BY a.start").fetchall()
    finally:
        connection.close()
    launches = [a for a in runtime if a[2].startswith("cudaLaunchKernel")]
    require(bool(kernels) and len(kernels) == len(launches), "GPU kernel/launch scope differs")
    require(all(k[1] > k[0] for k in kernels) and all(a[1] >= a[0] and a[4] == 0 for a in runtime), "GPU duration/runtime error")
    require(len({k[3] for k in kernels}) == len(kernels) and {k[3] for k in kernels} == {a[3] for a in launches}, "GPU launch correlations differ")
    require(len({(k[4], k[5]) for k in kernels}) == 1 and all(a[1] <= b[0] for a, b in zip(kernels, kernels[1:])), "GPU stream/overlap scope differs")
    updates = [k for k in kernels if "::sgd(" in k[2]]
    require(len(updates) == iterations * parameter_count and len(kernels) % iterations == 0, "GPU SGD/complete-step counts differ")
    names = sorted({k[2] for k in kernels})
    total = sum(k[1]-k[0] for k in kernels)
    groups = [{"name": name, "instances": sum(k[2] == name for k in kernels),
               "gpu_us_per_step": sum(k[1]-k[0] for k in kernels if k[2] == name)/1000/iterations,
               "fraction_of_kernel_time": sum(k[1]-k[0] for k in kernels if k[2] == name)/total} for name in names]
    return {"kernel_instances": len(kernels), "kernels_per_step": len(kernels)//iterations,
            "sgd_instances": len(updates), "launch_correlations_verified": True,
            "single_nonoverlapping_device_stream": True, "runtime_errors": 0,
            "gpu_kernel_us_per_step": total/1000/iterations,
            "gpu_envelope_us_per_step": (kernels[-1][1]-kernels[0][0])/1000/iterations,
            "launch_api_us_per_step": sum(a[1]-a[0] for a in launches)/1000/iterations,
            "kernels": sorted(groups, key=lambda item: item["gpu_us_per_step"], reverse=True),
            "scope": "Instrumented diagnostics; runtime/GPU intervals overlap and are not additive or primary latency samples"}


def verify_recovered_profiles(path, directory, result, p):
    recovery = json.loads(path.read_text()); root = path.parent
    require(result["platform"] == "nvidia" and result["status"] == "failed" and result.get("error") == "training profile summary failed", "only the known training profile export failure is recoverable")
    require(recovery["status"] == "recovered_complete_training_profiles" and recovery["original_comparison_sha256"] == sha(directory/"comparison.json"), "profile recovery status/report differs")
    require(all(recovery[key] == result[key] for key in ("producer_commit", "protocol_sha256", "binary_sha256")), "profile recovery producer differs")
    require(recovery["timing_windows_rerun"] == 0 and set(recovery["cases"]) == set(p["profiles"]["nvidia"]), "profile recovery scope differs")
    tools = recovery["tools"]; token = r"\d+\.\d+\.\d+\.\d+(?:-[\w.]+)?"
    versions = [re.search(token, tools[name]["version"]) for name in ("nsys", "importer")]
    require(all(versions) and versions[0].group() == versions[1].group() and all(re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) for value in tools.values()), "profile recovery SDK identity differs")
    reused = recovery["reused_existing_capture"]
    require(reused in recovery["cases"], "profile recovery reused case missing")
    new = 0; evidence = {}
    for name, row in recovery["cases"].items():
        count = p["profiles"]["nvidia"][name]
        require(row["iterations"] == count, "recovered profile loop count differs")
        case = next(c for c in p["cases"] if c["name"] == name)
        verify_objects(row["checked"], root, case)
        phases = ["import", "summary"]
        if name == reused:
            require(row["origin"] == "existing_capture" and row["original_qdstrm_sha256"] == row["qdstrm"]["raw_sha256"], "reused capture identity differs")
            for file, digest in row["original_logs"].items():
                require(sha(directory/"profiles"/name/file) == digest, "original profile capture log changed")
        else:
            require(row["origin"] == "new_missing_diagnostic_capture", "recovered capture origin differs")
            command = row["execution"]["command"]
            flags = [tools["nsys"]["path"], "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                     "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=false", "--output"]
            require(len(command) == 20 and command[:9] == flags and Path(command[10]).name == "fa-training" and Path(command[10]).parent.name == result["binary_sha256"] and command[11] == "--protocol" and Path(command[12]).name == PROTOCOL.name and command[13:17] == ["--platform", "nvidia", "--case", name] and command[17] == "--output-dir" and command[19] == "--profile", "recovered native command differs")
            phases += ["capture"]; new += 1
        for phase in phases:
            command = row["execution" if phase == "capture" else phase]
            require(command["returncode"] == 0 and not command["timeout"] and set(command["log_sha256"]) == {"stdout.log", "stderr.log"}, "recovered profile command failed")
            for file, digest in command["log_sha256"].items():
                require(sha(root/name/phase/file) == digest, "recovered profile command log changed")
        imported = row["import"]["command"]
        require(imported[0] == tools["importer"]["path"] and imported[1::2] == ["--input-file", "--output-file"] and Path(imported[2]).name == "import-input.qdstrm" and Path(imported[4]).name == "trace.nsys-rep", "recovery must import a copied input")
        summarized = row["summary"]["command"]
        require(summarized[:-1] == [tools["nsys"]["path"], "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum", "--format", "csv"] and summarized[-1] == imported[4], "recovered summary command differs")
        capture = directory/"profiles"/name/"stdout.log" if name == reused else root/name/"capture/stdout.log"
        replies = [json.loads(line) for line in capture.read_text().splitlines() if line.startswith("{")]
        require(len(replies) == 2 and replies[0]["status"] == "ready" and replies[1]["status"] == "profiled" and replies[1]["iterations"] == count, "recovered native profile replies differ")
        for kind, suffix in (("qdstrm", "qdstrm"), ("nsys_rep", "nsys-rep"), ("sqlite", "sqlite")):
            blob = row[kind]
            require(blob["file"] == name+"/trace."+suffix+".gz" and sha(root/blob["file"]) == blob["sha256"], "recovered compressed profile changed")
            digest = hashlib.sha256(); size = 0
            with gzip.open(root/blob["file"], "rb") as stream:
                while data := stream.read(4*1024*1024):
                    digest.update(data); size += len(data)
            require(digest.hexdigest() == blob["raw_sha256"] and size == blob["raw_bytes"], "recovered raw profile changed")
        with tempfile.TemporaryDirectory(prefix="fa-training-profile-audit-") as temporary:
            database = Path(temporary)/"trace.sqlite"
            database.write_bytes(gzip.decompress((root/row["sqlite"]["file"]).read_bytes()))
            evidence[name] = verify_gpu_events(database, count, len(shapes(case)[2]))
    require(new == recovery["new_instrumented_profile_executions"] == recovery["planned_new_profile_executions"] == len(recovery["cases"])-1, "recovery execution count differs")
    return {"receipt_sha256": sha(path), "original_collection_status": result["status"],
            "timing_windows_rerun": 0, "new_diagnostic_captures": new, "cases": evidence}


def assess(directory, readiness=False, profile_recovery=None):
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    p = load_protocol()
    path = directory / "comparison.json"; result = json.loads(path.read_text())
    expected_status = "verified_training_readiness_only" if readiness else "complete_training_characterization"
    recovered = profile_recovery is not None
    status_ok = result["status"] == expected_status if not recovered else not readiness and result["status"] == "failed" and result.get("error") == "training profile summary failed" and result["platform"] == "nvidia"
    require(status_ok and result["validation_only"] is readiness and result["promotion"] is False, "incomplete/wrong-kind training context")
    require(result["protocol_sha256"] == sha(PROTOCOL) and result["platform"] in p["strata"], "training context/protocol differs")
    platform = result["platform"]; spec = p["strata"][platform]
    build_path = directory / result["build_receipt"]["file"]
    require(result["build_receipt"]["file"] == "native-build/build-receipt.json" and sha(build_path) == result["build_receipt"]["sha256"], "build receipt changed")
    build = json.loads(build_path.read_text()); manifest = result["source_sha256"]
    require(build["returncode"]==0 and build["timeout"] is False, "failed/timed-out native training build")
    require(build["status"] == "built" and build["platform"] == platform and build["producer_commit"] == result["producer_commit"] and build["source_sha256"] == manifest and build["binary_sha256"] == result["binary_sha256"], "build/producer/source differs")
    require(sha(build_path.parent / "build.log") == build["log_sha256"] and sha(build_path.parent / "source.tar.gz") == build["source_archive_sha256"], "build log/source archive changed")
    with tarfile.open(build_path.parent / "source.tar.gz") as archive:
        recovered_sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
    committed_raw = subprocess.check_output(["git", "archive", result["producer_commit"], *manifest], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(committed_raw)) as archive:
        committed = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
    require(recovered_sources == manifest == committed, "training source does not match committed producer")
    proof=build["producer_proof"]
    require(proof["kind"] in ("git_archive","sealed_manifest"), "unknown producer proof")
    if proof["kind"]=="sealed_manifest":
        sealed=build_path.parent/proof["file"]
        require(proof["file"]=="producer-manifest.json" and sha(sealed)==proof["sha256"], "sealed producer inventory changed")
        content=json.loads(sealed.read_text())
        require(content["producer_commit"]==result["producer_commit"] and content["source_sha256"]==committed, "sealed mirror does not match Git producer")
    binary = build_path.parent / build["binary_gzip"]
    require(sha(binary) == build["binary_gzip_sha256"] and hashlib.sha256(gzip.decompress(binary.read_bytes())).hexdigest() == build["binary_sha256"], "exact training executable archive changed")
    require(build["environment"]["RUSTFLAGS"] == p["build"]["rustflags"], "native training build flags differ")
    prefix = ["cargo","build","--release","--locked","--offline","--bin","fa-training","--features",",".join(p["build"]["features"][platform])]
    require(build["command"][:len(prefix)] == prefix, "native training build command differs")
    if platform == "nvidia":
        require(build["environment"]["CUDA_ARCH"] == p["build"]["cuda_arch"] and build["environment"]["CUDA_HOST_CXX"] == p["build"]["cuda_host_cxx"], "native CUDA build policy differs")
    require([c["case"] for c in result["cases"]] == p["cases"], "training case catalogue incomplete/reordered")
    object_count = set(); validation_count = 0; windows = 0
    for index, case_result in enumerate(result["cases"]):
        case = case_result["case"]; case_dir = directory / case["name"]
        validate_ready(case_result["readiness"]["native"], case_result["readiness"]["torch"], case, p, platform)
        expected = {(mode, step) for mode in spec["modes"] for step in range(p["validation_steps"])}
        actual = [(item["mode"], item["step"]) for item in case_result["validation"]]
        require(len(actual) == len(set(actual)) and set(actual) == expected, "training validation trajectory scope differs")
        require(actual == [(mode,step) for step in range(p["validation_steps"]) for mode in spec["modes"]], "training validation trajectory order differs")
        previous = {mode:fixtures(case)[2] for mode in spec["modes"]}
        for item in case_result["validation"]:
            mode=item["mode"]; gradients=[];updated=[]
            for i,before in enumerate(previous[mode]):
                for target,key in ((gradients,"gradient-"+str(i)),(updated,"owner-"+str(i+2))):
                    raw=gzip.decompress((directory/item["checked"]["payloads"][key]["object"]).read_bytes())
                    target.append(torch.frombuffer(bytearray(raw),dtype=torch.float32).reshape(before.shape))
            require(item["checked"]["sgd_update"] == sgd_update(previous[mode],gradients,updated,p["rate"]), "recovered SGD update proof differs")
            previous[mode]=updated
        for engine in ("native", "torch"):
            worker = case_result["workers"][engine]; shutdown = worker["shutdown"]
            require(worker["pid"] == shutdown["pid"], "different stopped training worker")
            validate_close(shutdown, p, platform, readiness, engine == "native")
            for file, digest in shutdown["log_sha256"].items():
                require(sha(case_dir / (engine + "-transport") / file) == digest, "training transport log changed")
            replies = [json.loads(line) for line in (case_dir / (engine + "-transport") / "stdout.jsonl").read_text().splitlines()]
            require(replies[0] == case_result["readiness"][engine] and replies[-1] == shutdown["reply"], "recorded training readiness/shutdown differs from raw replies")
            modes = ["native"] if engine == "native" else spec["modes"][1:]
            count = len(modes)*(p["validation_steps"] + (0 if readiness else 2*spec["blocks"])) + 2
            require(len(replies) == count and all(r["status"] in ("ready", "validated", "measured", "repeated", "stopped") for r in replies), "raw training response count/scope differs")
            requests = [json.loads(line) for line in (case_dir / (engine + "-transport") / "stdin.jsonl").read_text().splitlines()]
            wanted = [{"action": "validate", "step": step, **({"mode": mode} if engine == "torch" else {})}
                      for step in range(p["validation_steps"]) for mode in spec["modes"] if mode in modes]
            for block in case_result["blocks"]:
                for mode in block["order"]:
                    if mode in modes:
                        wanted += [{"action": action, "block": block["block"], **({"mode": mode} if engine == "torch" else {})}
                                   for action in ("measure", "repeat")]
            wanted.append({"action": "quit"})
            require(requests == wanted and len(replies) == len(requests)+1, "raw training requests differ from declared order/counts")
            for request, reply in zip(requests, replies[1:]):
                action = request["action"]
                require(reply["status"] == {"validate": "validated", "measure": "measured", "repeat": "repeated", "quit": "stopped"}[action], "raw training response kind differs")
                if action == "validate":
                    require(reply["step"] == request["step"], "raw training validation step differs")
                elif action in ("measure", "repeat"):
                    require(reply["block"] == request["block"], "raw training response block differs")
                    if action == "measure":
                        sample = case_result["blocks"][request["block"]]["modes"][request.get("mode", "native")]
                        window = p["windows"][case["regime"]]
                        require(reply["timing"] == sample["timing"] and reply["warmup"] == window["warmup"] and reply["iterations"] == window["iterations"], "raw training timing/window differs")
        checks = [(item["checked"],item["step"]+1) for item in case_result["validation"]]
        validation_count += len(checks)
        require(len(case_result["blocks"]) == (0 if readiness else spec["blocks"]), "missing training windows")
        anchors = {}
        for block, item in enumerate(case_result["blocks"]):
            require(item["block"] == block and item["order"] == order(p, platform, index, block) and set(item["modes"]) == set(spec["modes"]), "training order/modes differ")
            for mode, sample in item["modes"].items():
                require(sample["exact_repeat"] is True and math.isfinite(sample["timing"]["wall_us"]) and sample["timing"]["wall_us"] > 0, "training repeat/timing gate incomplete")
                identity = {k: v["sha256"] for k, v in sample["checked"]["payloads"].items()}
                require(identity == {k: v["sha256"] for k, v in sample["repeat_payloads"].items()} and (mode not in anchors or anchors[mode] == identity), "training reset/repeat identities differ")
                require(sample["repeat_payloads"] == {k:{field:v[field] for field in ("shape","sha256","bytes")} for k,v in sample["checked"]["payloads"].items()}, "training repeated payload metadata differs")
                anchors[mode] = identity
                device = sample["timing"]["device_us"]
                require((device is None) if platform == "intel" else (device is not None and math.isfinite(device) and device > 0), "training device timing scope differs")
                if mode == "native":
                    require(sample["timing"]["hot_rust_allocations"] == 0, "training hot allocations")
                window = p["windows"][case["regime"]]
                checks.append((sample["checked"],window["warmup"]+window["iterations"])); windows += 1
        controls = {}
        for checked, count in checks:
            if count not in controls:
                controls[count] = [control(case,count,fp64,p["rate"]) for fp64 in (True,False)]
            verify_objects(checked, directory, case, controls[count])
            object_count.update(item["object"] for item in checked["payloads"].values())
            require(set(checked["field_max_abs_error"]) == {"analytic_fp64", "eager_fp32"}, "independent training controls missing")
            for errors in checked["field_max_abs_error"].values():
                require(set(errors)==set(checked["payloads"]) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in errors.values()), "recorded training control error scope differs")
        if not readiness:
            medians = {m: statistics.median(b["modes"][m]["timing"]["wall_us"] for b in case_result["blocks"]) for m in spec["modes"]}
            best = min((m for m in medians if m != "native"), key=lambda m: medians[m])
            require(case_result["summary"] == {"median_wall_us": medians, "strongest_torch_mode": best, "strongest_torch_over_native": medians[best]/medians["native"]}, "training descriptive summary differs")
    profile_evidence = None
    if recovered:
        profile_evidence = verify_recovered_profiles(profile_recovery, directory, result, p)
    elif not readiness:
        profiles = result["profiles"]
        if profiles["available"]:
            require(set(profiles["cases"]) == set(p["profiles"][platform]), "training profile scope incomplete")
            for name, profile in profiles["cases"].items():
                require(profile["iterations"] == p["profiles"][platform][name] and profile["execution"]["returncode"] == profile["summary"]["returncode"] == 0, "failed/wrong training profile")
                for file, digest in profile["files"].items():
                    require(sha(directory / "profiles" / name / file) == digest, "training profile file changed")
                case = next(c for c in p["cases"] if c["name"] == name)
                verify_objects(profile["checked"], directory, case)
        else:
            require(bool(profiles["reason"]), "missing profile limitation")
    return {"status": "verified_training_readiness_evidence" if readiness else "verified_training_characterization_with_recovered_profiles" if recovered else "verified_training_characterization_evidence",
            "platform": platform, "producer_commit": result["producer_commit"], "report_sha256": sha(path),
            "source_files": len(manifest), "source_matches_committed_producer": True, "exact_binary_and_logs_verified": True,
            "cases": len(result["cases"]), "trajectory_snapshots": validation_count, "timed_windows": windows,
            "payload_objects": len(object_count), "promotion": False,
            "profile_recovery": profile_evidence,
            "limitations": ["Recovery audited without isolated rebuild; fixed procedural scope is not all audio training", "Persistent blocks do not establish independent confirmation; profiler/resource scopes remain separate"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--profile-recovery", type=Path, help="explicit successful diagnostic recovery for an immutable Nsight export failure")
    args = parser.parse_args()
    require(not args.output.exists(), "preserve previous training assessment")
    result = assess(args.directory.absolute(), args.readiness, args.profile_recovery.absolute() if args.profile_recovery else None)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"], result["cases"], "cases", flush=True)


if __name__ == "__main__":
    main()
