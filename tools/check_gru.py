#!/usr/bin/env python3
"""Validate prepared GRUs against all actual GTCRN layer inputs, outputs and states."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import torch

try:
    from .compare_torch import errors, source_manifest, ROOT
except ImportError:
    from compare_torch import errors, source_manifest, ROOT


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder,description):
    path=folder/description["file"]
    if description.get("sha256") and sha(path)!=description["sha256"]:
        raise ValueError(f"reference tensor changed: {path.name}")
    return torch.frombuffer(bytearray(path.read_bytes()),dtype=torch.float32).reshape(description["shape"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--backend",choices=["reference","mkl","cudnn"],required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists(): parser.error("use a new output directory")
    source=args.reference/"reference.json"
    reference=json.loads(source.read_text())
    if reference["status"]!="verified_pytorch_reference" or len(reference["cases"])!=4:
        raise ValueError("expected the complete four-case model reference")
    for case in reference["cases"]:
        layers=[layer for layer in case["leaf_operators"] if layer["type"]=="GRU"]
        if len(layers)!=14: raise ValueError("model reference must contain every GRU")
        for layer in layers:
            for blob in [layer["input"],layer["initial"],layer["output"],layer["final"],*layer["parameters"].values()]:
                if sha(args.reference/case["name"]/blob["file"])!=blob["sha256"]: raise ValueError("reference tensor hash mismatch")
    subprocess.run([str(args.binary.resolve()),args.backend,str(args.reference.resolve()),str(args.output_dir.resolve())],check=True)
    native=json.loads((args.output_dir/"gru.json").read_text())
    if native["backend"]!=args.backend or len(native["cases"])!=56: raise ValueError("incomplete GRU execution")
    records={(case["name"],layer["name"]):layer for case in reference["cases"] for layer in case["leaf_operators"] if layer["type"]=="GRU"}
    if {(record["case"],record["layer"]) for record in native["cases"]}!=records.keys(): raise ValueError("GRU layer coverage changed")
    device="cuda" if args.backend=="cudnn" else "cpu"
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report={"schema_version":1,"status":"running","protocol":{"backend":args.backend,
        "scope":"real-layer correctness and exploratory warm timings; not full-model acceptance",
        "warmup":20,"iterations":50,"samples":7,"order":"all Rust layers, then all Torch layers; drift is uncontrolled",
        "precision":"FP32, TF32 disabled"},"environment":{"torch":torch.__version__,"torch_cuda":torch.version.cuda,
        "binary_sha256":sha(args.binary),"reference_sha256":sha(source),"source_sha256":source_manifest()},"cases":[]}
    archive=args.output_dir/"source.tar.gz"
    with tarfile.open(archive,"w:gz") as bundle:
        for name in report["environment"]["source_sha256"]: bundle.add(ROOT/name,arcname=name)
    report["environment"]["source_archive_sha256"]=sha(archive)
    def sync():
        if device=="cuda": torch.cuda.synchronize()
    with torch.inference_mode():
        for record in native["cases"]:
            layer=records[(record["case"],record["layer"])];folder=args.reference/record["case"]
            x=load(folder,layer["input"]).to(device);initial=load(folder,layer["initial"]).to(device)
            module=torch.nn.GRU(layer["input_size"],layer["hidden_size"],batch_first=True,bidirectional=layer["bidirectional"]).to(device).eval()
            module.load_state_dict({name:load(folder,description).to(device) for name,description in layer["parameters"].items()})
            for _ in range(20): expected=module(x,initial)
            sync();samples=[]
            for _ in range(7):
                begin=time.perf_counter_ns()
                for _ in range(50): expected=module(x,initial)
                sync();samples.append((time.perf_counter_ns()-begin)/1000/50)
            checks={}
            for index,key in enumerate(["output","final"]):
                actual=load(args.output_dir,record[key]);saved=load(folder,layer[key])
                checks[key]=errors(actual,saved)
                checks[key+"_torch_replay"]=errors(expected[index].cpu(),saved)
            report["cases"].append({**record,"checks":checks,"torch_wall_us":samples})
    report["status"]="verified_real_gru_layers" if all(check["passed"] for case in report["cases"] for check in case["checks"].values()) else "failed"
    (args.output_dir/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(args.backend,report["status"],len(report["cases"]),"layer/shape cases")
    if report["status"]=="failed": raise RuntimeError("GRU numerical gate failed")


if __name__=="__main__": main()
