#!/usr/bin/env python3
"""Validate exact sqrt-Hann front/back processing and all captured 2D model convolutions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import torch

try:
    from .compare_torch import ROOT,errors,source_manifest
    from .check_gru import load,sha
except ImportError:
    from compare_torch import ROOT,errors,source_manifest
    from check_gru import load,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--backend",choices=["reference","mkl","cudnn"],required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists(): parser.error("use a new output directory")
    reference_path=args.reference/"reference.json"
    reference=json.loads(reference_path.read_text())
    if reference["status"]!="verified_pytorch_reference" or len(reference["cases"])!=4:
        raise ValueError("expected the complete four-case reference")
    expected={}
    for case in reference["cases"]:
        conv=[layer for layer in case["leaf_operators"] if layer["type"] in ["Conv2d","ConvTranspose2d"]]
        if len(conv)!=22: raise ValueError("missing captured convolutions")
        for layer in conv:
            for blob in [layer["input"],layer["output"],*layer["parameters"].values()]:
                if sha(args.reference/case["name"]/blob["file"])!=blob["sha256"]: raise ValueError("convolution capture changed")
            expected[(case["name"],layer["name"])]=layer
        for blob in case["tensors"].values():
            if sha(args.reference/case["name"]/blob["file"])!=blob["sha256"]: raise ValueError("audio capture changed")
    subprocess.run([str(args.binary.resolve()),args.backend,str(args.reference.resolve()),str(args.output_dir.resolve())],check=True)
    native=json.loads((args.output_dir/"stages.json").read_text())
    if native["backend"]!=args.backend or len(native["audio"])!=4 or len(native["layers"])!=88:
        raise ValueError("incomplete native stage execution")
    if {(record["case"],record["layer"]) for record in native["layers"]}!=expected.keys():
        raise ValueError("convolution coverage changed")
    audio={case["name"]:case for case in reference["cases"]}
    if {record["case"] for record in native["audio"]}!=audio.keys(): raise ValueError("audio strata changed")
    report={"schema_version":1,"status":"running","protocol":{"backend":args.backend,
        "scope":"real model stages; model middle remains a PyTorch reference; no end-to-end acceptance",
        "precision":"FP32, TF32 disabled","timing":"native-only exploratory warmed layer timings; 20 warmups, 7 x 50 iterations"},
        "environment":{"torch":torch.__version__,"binary_sha256":sha(args.binary),
            "reference_sha256":sha(reference_path),"source_sha256":source_manifest()},"audio":[],"layers":[]}
    archive=args.output_dir/"source.tar.gz"
    with tarfile.open(archive,"w:gz") as bundle:
        for name,digest in report["environment"]["source_sha256"].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError("source changed before archive")
            bundle.add(ROOT/name,arcname=name)
    report["environment"]["source_archive_sha256"]=sha(archive)
    for record in native["audio"]:
        case=audio[record["case"]];folder=args.reference/case["name"]
        spectrum=load(folder,case["tensors"]["spectrum"]).transpose(1,2)
        waveform=load(folder,case["tensors"]["waveform"])
        report["audio"].append({"case":case["name"],"checks":{"spectrum":errors(load(args.output_dir,record["spectrum"]),spectrum),
            "enhanced_inverse":errors(load(args.output_dir,record["waveform"]),waveform)}})
    for record in native["layers"]:
        layer=expected[(record["case"],record["layer"])];folder=args.reference/record["case"]
        check=errors(load(args.output_dir,record["output"]),load(folder,layer["output"]))
        report["layers"].append({**record,"correctness":check})
    passed=all(c["correctness"]["passed"] for c in report["layers"]) and all(check["passed"] for c in report["audio"] for check in c["checks"].values())
    report["status"]="verified_gtcrn_stages" if passed else "failed"
    (args.output_dir/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(args.backend,report["status"],"88 convolutions and 4 exact-window spectral cases")
    if not passed: raise RuntimeError("native model stage numerical gate failed")


if __name__=="__main__": main()
