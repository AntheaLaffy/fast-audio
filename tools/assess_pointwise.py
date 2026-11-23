#!/usr/bin/env python3
"""Audit complete pointwise characterization receipts without accepting optimizations."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.pointwise_protocol import COMPILER_POLICY, load_protocol, sha, order, summarize
from tools.verify_archive import verify


def command_logs(record,directory):
    for name,digest in record["log_sha256"].items():
        if Path(name).name!=name or sha(directory/name)!=digest: raise ValueError("command log changed")


def validation(value,case,expected_hash=None):
    if value["passed"] is not True or value["finite"] is not True or value["holes_preserved"] is not True:
        raise ValueError("failed backing/numerical gate")
    if value["input_sha256"]!=value["expected_input_sha256"]:
        raise ValueError("input bits changed")
    if (value["atol"],value["rtol"],value["elements"])!=(2e-6,3e-6,math.prod(case["output"]["dims"])):
        raise ValueError("numerical scope/tolerance differs")
    if not math.isfinite(value["max_tolerance_ratio"]) or not 0<=value["max_tolerance_ratio"]<=1:
        raise ValueError("numerical error exceeds declared gate")
    if expected_hash is not None and (not value["repeat_exact"] or value["output_sha256"]!=expected_hash):
        raise ValueError("same-mode repeatability failed")


def assess(path):
    report=json.loads(path.read_text())
    if report["status"]!="characterized_pointwise_context" or report["protocol"]["validation_only"]:
        raise ValueError("a failed, partial or validation-only run is not characterization")
    p=load_protocol(path.parent/"protocol.json")
    if report["protocol"]["parameters"]!=p or report["protocol"]["sha256"]!=sha(path.parent/"protocol.json") or report["protocol"]["promotion"] is not False:
        raise ValueError("protocol identity or no-promotion rule differs")
    platform=report["protocol"]["platform"]
    spec=p["strata"][platform]
    source_files=verify(path)
    environment=report["environment"]
    if sha(path.parent/"fa-pointwise.gz")!=environment["binary_gzip_sha256"]:
        raise ValueError("compressed native binary changed")
    if hashlib.sha256(gzip.decompress((path.parent/"fa-pointwise.gz").read_bytes())).hexdigest()!=environment["binary_sha256"]:
        raise ValueError("instrumented native executable bytes differ")
    build_path=path.parent/"build-receipt.json"
    if sha(build_path)!=environment["build_receipt_sha256"]:
        raise ValueError("build receipt changed")
    build=json.loads(build_path.read_text())
    if build["status"]!="built" or build["platform"]!=platform or build["returncode"]!=0 or build["source_sha256"]!=environment["source_sha256"] or build["binary_sha256"]!=environment["binary_sha256"] or build["environment"]["RUSTFLAGS"]!=p["build"]["rustflags"] or sha(path.parent/"build.log")!=build["log_sha256"]:
        raise ValueError("build/source/flag evidence differs")
    if [r["case"] for r in report["cases"]]!=p["cases"]:
        raise ValueError("missing, reordered or changed cases")
    windows=0
    for index,row in enumerate(report["cases"]):
        case=row["case"]
        if set(row["readiness"])!=set(spec["modes"]): raise ValueError("missing ready mode")
        for mode in spec["modes"]: validation(row["readiness"][mode],case)
        if len(row["blocks"])!=spec["blocks"]: raise ValueError("block budget differs")
        for engine,worker in row["workers"].items():
            expected=spec["blocks"] if engine=="native" else dict.fromkeys(spec["modes"][1:],spec["blocks"])
            if worker["shutdown"]["returncode"]!=0 or worker["shutdown"]["reply"]!={"status":"stopped","measured_blocks":expected}:
                raise ValueError("worker terminal receipt differs")
            command_logs(worker["shutdown"],path.parent/"workers"/case["name"]/engine)
        if set(row["workers"])!={"native","torch"}: raise ValueError("worker pair differs")
        native=row["workers"]["native"]["ready"]
        torch=row["workers"]["torch"]["ready"]
        if native["compiled_features"]!=spec["compiled_features"] or native["debug_assertions"] or native["cpu_allowed_list"]!="0":
            raise ValueError("native build/affinity differs")
        metadata=native["metadata"]
        expected_workspace=0 if spec["device"]=="cpu" else max(metadata["coalesced_rank"]-1,0)*32
        if metadata["workspace_bytes"]!=expected_workspace: raise ValueError("coordinate workspace differs")
        if torch["affinity"]!=p["affinity"] or (torch["threads"],torch["interop_threads"])!=(1,1) or torch["compile"]!=p["compile"]:
            raise ValueError("Torch compile/thread/affinity differs")
        if torch["compiler"]["policy"]!=COMPILER_POLICY: raise ValueError("compiler precision policy differs")
        if set(torch["modes"])!=set(spec["modes"])-{"native"}: raise ValueError("Torch readiness modes differ")
        for mode,metadata in torch["modes"].items():
            expected_writeback=p["compile"]["writeback"] if mode in ("compiled","compiled-cuda-graph") else "out-parameter"
            if metadata["writeback"]!=expected_writeback: raise ValueError("compiled writeback differs")
        for worker in (native,torch):
            if worker["case"]!=case or worker["device"]!=spec["device"] or worker["window"]!=spec["windows"][case["regime"]]:
                raise ValueError("ready geometry/device/window differs")
        for block_index,block in enumerate(row["blocks"]):
            if block["index"]!=block_index or block["order"]!=order(p,platform,index,block_index) or set(block["samples"])!=set(spec["modes"]):
                raise ValueError("block order or mode coverage differs")
            for mode,sample in block["samples"].items():
                if (sample["status"],sample["case"],sample["mode"],sample["block"],sample["window"])!=("measured",case["name"],mode,block_index,spec["windows"][case["regime"]]):
                    raise ValueError("measured command metadata differs")
                validation(sample["validation"],case,row["readiness"][mode]["output_sha256"])
                for metric in ("wall_us","device_us"):
                    value=sample["timing"][metric]
                    if value is None and metric=="device_us" and spec["device"]=="cpu": continue
                    if value is None or not math.isfinite(value) or value<=0: raise ValueError("invalid timing metric")
                windows+=1
        if summarize(row["blocks"],spec["modes"])!=row["summary"]: raise ValueError("descriptive summary differs")
    recomputed={mode:math.exp(statistics.mean(math.log(row["summary"][mode]["torch_over_native_descriptive_geomean"]) for row in report["cases"])) for mode in spec["modes"]}
    if recomputed!=report["summary"]: raise ValueError("suite summary differs")
    profiles=report["profiles"]
    if "permission_probe" in profiles: command_logs(profiles["permission_probe"],path.parent/"profile-permission")
    if profiles.get("available") and set(profiles["cases"])!=set(p["profiles"][platform]): raise ValueError("profile budget differs")
    for name,profile in profiles.get("cases",{}).items():
        if profile["iterations"]!=p["profiles"][platform][name]: raise ValueError("profile counts differ")
        command_logs(profile,path.parent/"profiles"/name)
        if "summary" in profile: command_logs(profile["summary"],path.parent/"profiles"/name/"summary")
        if "gzip_file" in profile:
            compressed=path.parent/profile["gzip_file"]
            if sha(compressed)!=profile["gzip_sha256"] or hashlib.sha256(gzip.decompress(compressed.read_bytes())).hexdigest()!=profile["raw_sha256"]:
                raise ValueError("profile archive differs")
    return {"platform":platform,"report_sha256":sha(path),"source_files":source_files,"cases":len(report["cases"]),
            "latency_windows":windows,"all_recorded_checks_passed":True,"archive_and_exact_binary_verified":True,
            "descriptive_torch_over_native":recomputed,"profiles":report["profiles"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error("use a new assessment file")
    results=[assess(p) for p in args.reports]
    if len({r["platform"] for r in results})!=len(results): raise ValueError("duplicate hardware context")
    result={"status":"verified_recorded_pointwise_characterization","contexts":results,"promotion":False,
        "audit_sha256":sha(Path(__file__)),"limitations":["Audits recorded full-output checks and hashes without recomputing omitted raw arrays",
            "Within-worker blocks do not provide independent confirmation or formal confidence intervals",
            "Captured Torch modes and uncaptured native mode retain different execution strategies",
            "No isolated rebuild, exact memory peak, optimization acceptance or framework-wide claim"]}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result["status"],[(r["platform"],r["cases"],r["latency_windows"]) for r in results])


if __name__=="__main__": main()
