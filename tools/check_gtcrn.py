#!/usr/bin/env python3
"""Verify the complete native GTCRN spectrum and waveform path against pinned captures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import torch

try:
    from .compare_torch import ROOT, errors, source_manifest
    from .check_gru import load, sha
except ImportError:
    from compare_torch import ROOT, errors, source_manifest
    from check_gru import load, sha


def validate_convolution_policy(native, requested):
    """Check actual selected plan metadata, rather than trusting an echoed CLI flag."""
    if native["backend"] != "cudnn":
        if requested != "allow_nondeterministic":
            raise ValueError("convolution policy requires cuDNN")
        return
    if native["protocol"].get("convolution_policy") != requested:
        raise ValueError("native convolution policy differs")
    for case in native["cases"]:
        plans = case.get("convolution_plans")
        if not isinstance(plans, list) or not plans:
            raise ValueError("native convolution metadata is absent")
        steps = set()
        for plan in plans:
            if not isinstance(plan, dict):
                raise ValueError("invalid native convolution metadata")
            if (type(plan.get("step")) is not int or plan["step"] < 0
                    or plan["step"] in steps
                    or type(plan.get("engine_index")) is not int or plan["engine_index"] < 0
                    or type(plan.get("workspace_bytes")) is not int
                    or not 0 <= plan["workspace_bytes"] <= 64 << 20
                    or type(plan.get("numerical_notes")) is not int
                    or not 0 <= plan["numerical_notes"] < 1 << 64
                    or type(plan.get("deterministic")) is not bool
                    or type(plan.get("fused_epilogue")) is not bool):
                raise ValueError("invalid native convolution metadata")
            steps.add(plan["step"])
            if requested == "require_deterministic" and not plan["deterministic"]:
                raise ValueError("native convolution engine violates deterministic policy")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--backend",choices=["reference","mkl","cudnn"],required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--samples",type=int,default=3)
    parser.add_argument("--iterations",type=int,default=5)
    parser.add_argument("--convolution-policy",choices=["allow_nondeterministic","require_deterministic"],default="allow_nondeterministic")
    parser.add_argument("--pipeline-only",action="store_true")
    args=parser.parse_args()
    if args.output_dir.exists():parser.error("use a new output directory")
    if args.warmup<0 or min(args.samples,args.iterations)<1:parser.error("invalid timing counts")
    if args.backend!="cudnn" and args.convolution_policy!="allow_nondeterministic":parser.error("convolution policy requires cuDNN")
    reference_path=args.reference/"reference.json"
    reference=json.loads(reference_path.read_text())
    if reference["status"]!="verified_pytorch_reference":raise ValueError("incomplete reference")
    expected_names={"b1-l4096","b1-l16000","b1-l64000","b2-l16000"}
    if len(reference["cases"])!=4 or {c["name"] for c in reference["cases"]}!=expected_names:raise ValueError("reference strata changed")
    bundle_path=args.reference/reference["parameter_bundle"]["file"]
    if sha(bundle_path)!=reference["parameter_bundle"]["sha256"]:raise ValueError("parameter manifest changed")
    bundle=json.loads(bundle_path.read_text())
    pin=json.loads((ROOT/"models/gtcrn/upstream.json").read_text())
    if bundle["format"]!="fast-audio-gtcrn-f32-v1" or bundle["model"]!=reference["model"] or bundle["model"]!=pin:raise ValueError("model identity differs")
    for blob in bundle["parameters"].values():
        if sha(args.reference/blob["file"])!=blob["sha256"]:raise ValueError("parameter blob changed")
    for case in reference["cases"]:
        if len(case["model_stages"])!=15 or len({x["name"] for x in case["model_stages"]})!=15:raise ValueError("model trace incomplete")
        for blob in [*case["tensors"].values(),*(x["output"] for x in case["model_stages"])]:
            if sha(args.reference/case["name"]/blob["file"])!=blob["sha256"]:raise ValueError("reference activation changed")
    # Archive first. The example requires an absent output directory, so use a temporary
    # sibling for the archive and move it only after successful native execution.
    manifest=source_manifest()
    archive=args.output_dir.with_name(args.output_dir.name+"-source.tar.gz")
    if archive.exists():raise ValueError("source archive already exists")
    with tarfile.open(archive,"w:gz") as tar:
        for name,digest in manifest.items():
            if sha(ROOT/name)!=digest:raise ValueError("source changed before archive")
            tar.add(ROOT/name,arcname=name)
    binary_sha=sha(args.binary)
    command=[str(args.binary.resolve()),args.backend,str(args.reference.resolve()),str(args.output_dir.resolve()),
        str(args.warmup),str(args.samples),str(args.iterations)]
    if args.convolution_policy=="require_deterministic":command.append("--deterministic-convolution")
    if args.pipeline_only:command.append("--pipeline-only")
    subprocess.run(command,check=True)
    if sha(args.binary)!=binary_sha or source_manifest()!=manifest:raise ValueError("execution artifact changed during validation")
    archive.rename(args.output_dir/"source.tar.gz")
    native=json.loads((args.output_dir/"inference.json").read_text())
    validate_convolution_policy(native,args.convolution_policy)
    if native["backend"]!=args.backend or len(native["cases"])!=4 or {c["case"] for c in native["cases"]}!=expected_names:raise ValueError("native coverage differs")
    expected={c["name"]:c for c in reference["cases"]}
    report={"schema_version":1,"status":"running","protocol":{**native["protocol"],"backend":args.backend,"precision":"FP32; TF32 disabled",
        "scope":"complete native offline waveform pipeline; procedural numerical fixtures, no speech-quality acceptance",
        "timing":"native-only exploratory timing, no performance promotion"},
        "environment":{"timestamp_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"torch":torch.__version__,
            "cpu_affinity":sorted(os.sched_getaffinity(0)),"binary_sha256":binary_sha,"reference_sha256":sha(reference_path),
            "parameter_bundle_sha256":sha(bundle_path),"source_sha256":manifest,"source_archive_sha256":sha(args.output_dir/"source.tar.gz")},"cases":[]}
    for record in native["cases"]:
        case=expected[record["case"]];folder=args.reference/case["name"]
        stages={x["name"]:x["output"] for x in case["model_stages"]}
        if record["model_stages"].keys()!=stages.keys():raise ValueError("native model trace differs")
        checks={name:errors(load(args.output_dir,record[name]),load(folder,case["tensors"][name]).transpose(1,2) if name in ["spectrum","enhanced"] else load(folder,case["tensors"][name]))
            for name in ["spectrum","enhanced","waveform"]}
        checks.update({name:errors(load(args.output_dir,blob),load(folder,stages[name])) for name,blob in record["model_stages"].items()})
        report["cases"].append({**record,"checks":checks})
    report["status"]="verified_native_gtcrn" if all(v["passed"] for c in report["cases"] for v in c["checks"].values()) else "failed"
    (args.output_dir/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(args.backend,report["status"],"4 complete pipelines and 60 model stage outputs")
    if report["status"]=="failed":raise RuntimeError("complete model numerical gate failed")


if __name__=="__main__":main()
