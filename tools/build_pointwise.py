#!/usr/bin/env python3
"""Build the declared pointwise worker and seal its source, flags, log and binary."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.pointwise_protocol import load_protocol, sha
from tools.compare_torch import source_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform",choices=["intel","nvidia"],required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--vendor-dir",type=Path)
    parser.add_argument("--producer",required=True)
    args=parser.parse_args()
    if args.output_dir.exists(): parser.error("use a new build directory")
    if not re.fullmatch(r"[0-9a-f]{40}",args.producer): parser.error("producer must be an exact full 40-character commit")
    p=load_protocol()
    env=os.environ.copy()
    env["RUSTFLAGS"]=p["build"]["rustflags"]
    if args.platform=="nvidia":
        env["CUDA_ARCH"]=p["build"]["cuda_arch"]
        env["CUDA_HOST_CXX"]=p["build"]["cuda_host_cxx"]
    command=["cargo","build","--release","--locked","--offline","--bin","fa-pointwise"]
    if args.platform=="nvidia": command.extend(["--features","cuda"])
    if args.vendor_dir:
        command.extend(["--config",'source.crates-io.replace-with="vendored-sources"',"--config",
                        f'source.vendored-sources.directory="{args.vendor_dir.absolute()}"'])
    manifest=source_manifest()
    args.output_dir.mkdir(parents=True)
    receipt=args.output_dir/"build-receipt.json"
    log=args.output_dir/"build.log"
    result={"schema_version":1,"status":"running","producer_commit":args.producer,
        "platform":args.platform,"command":command,"cwd":str(ROOT),"source_sha256":manifest,
        "environment":{k:env.get(k) for k in ("RUSTFLAGS","CUDA_ARCH","CUDA_HOST_CXX","CARGO_HOME")},
        "log":log.name,"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    def save():
        tmp=receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(result,indent=2)+"\n")
        tmp.replace(receipt)
    save()
    begin=time.monotonic()
    with log.open("xb") as output:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        result["pid"]=process.pid
        save()
        code=process.wait()
    result.update(returncode=code,elapsed_seconds=time.monotonic()-begin,log_sha256=sha(log))
    if source_manifest()!=manifest:
        result.update(status="failed",error="source changed during build")
        save()
        raise RuntimeError(result["error"])
    if code:
        result["status"]="failed"
        save()
        raise RuntimeError("pointwise build failed; retained log")
    binary=ROOT/"target/release/fa-pointwise"
    result.update(status="built",binary=str(binary),binary_sha256=sha(binary))
    save()
    print("built",args.platform,result["binary_sha256"],flush=True)


if __name__=="__main__": main()
