#!/usr/bin/env python3
"""Counterbalanced resident-buffer pointwise characterization, without promotion."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import select
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.pointwise_protocol import PROTOCOL, COMPILER_POLICY, load_protocol, sha, order, reference, check_buffers, summarize
from tools.compare_torch import source_manifest, cpu_identity, command_output
from tools.gtcrn_benchmark import telemetry


class Peer:
    def __init__(self,command,directory,environment):
        self.command=command
        self.directory=directory
        directory.mkdir(parents=True)
        self.error=(directory/"stderr.log").open("xb")
        self.transcript=(directory/"stdout.jsonl").open("xb")
        self.pending=b""
        self.started=time.monotonic_ns()
        self.process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,stderr=self.error,start_new_session=True,bufsize=0)

    def receive(self,timeout):
        deadline=time.monotonic()+timeout
        while b"\n" not in self.pending:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([self.process.stdout],[],[],remaining)[0]:
                raise TimeoutError(f"declared worker timeout: pid {self.process.pid}")
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:
                raise RuntimeError(f"worker stdout ended: pid {self.process.pid}, code {self.process.poll()}; see {self.directory/'stderr.log'}")
            self.transcript.write(chunk)
            self.transcript.flush()
            self.pending+=chunk
        line,self.pending=self.pending.split(b"\n",1)
        return json.loads(line)

    def request(self,message,timeout):
        self.process.stdin.write((json.dumps(message)+"\n").encode())
        self.process.stdin.flush()
        return self.receive(timeout)

    def close(self):
        result={"pid":self.process.pid}
        if self.process.poll() is None:
            try:
                result["reply"]=self.request({"action":"quit"},10)
                self.process.wait(timeout=10)
            except Exception as exc:
                result["shutdown_error"]=str(exc)
                if self.process.poll() is None:
                    try: os.killpg(self.process.pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(self.process.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                        self.process.wait(timeout=10)
        result["returncode"]=self.process.returncode
        self.process.stdin.close()
        self.process.stdout.close()
        self.error.close()
        self.transcript.close()
        result["log_sha256"]={name:sha(self.directory/name) for name in ("stderr.log","stdout.jsonl")}
        return result


def freeze(binary,root):
    digest=sha(binary)
    destination=root/digest/"fa-pointwise"
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        with destination.open("xb") as output: output.write(binary.read_bytes())
        destination.chmod(0o555)
    if sha(destination)!=digest: raise ValueError("frozen executable changed")
    return destination,digest


def log_command(command,directory,timeout,env=None):
    directory.mkdir(parents=True,exist_ok=False)
    begin=time.time()
    timed_out=False
    with (directory/"stdout.log").open("xb") as out,(directory/"stderr.log").open("xb") as err:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        try: code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=10)
            code=process.returncode
            timed_out=True
    return {"command":command,"returncode":code,"timeout":timed_out,"elapsed_seconds":time.time()-begin,
            "log_sha256":{name:sha(directory/name) for name in ("stdout.log","stderr.log")}}


def profiles(args,p,frozen,environment,check_unchanged,remaining):
    results={}
    tool=shutil.which("perf" if args.platform=="intel" else "nsys")
    if not tool: return {"available":False,"reason":"profile tool not on PATH"}
    results["tool"]=tool
    results["version"]=command_output([tool,"--version"])
    if args.platform=="intel":
        probe=log_command([tool,"stat","-e","cycles:u","--","true"],args.output_dir/"profile-permission",remaining(30),environment)
        results["permission_probe"]=probe
        if probe["returncode"]!=0: return {**results,"available":False,"reason":"perf permission/tool probe failed; retained logs"}
    results["available"]=True
    results["cases"]={}
    for name,count in p["profiles"][args.platform].items():
        check_unchanged()
        directory=args.output_dir/"profiles"/name
        worker=args.output_dir/"profile-workers"/name
        command=[str(frozen),"--protocol",str(args.protocol),"--device",p["strata"][args.platform]["device"],
                 "--case",name,"--output-dir",str(worker),"--profile"]
        if args.platform=="intel":
            raw=directory/"samples.perf.data"
            command=[tool,"record","-e","cycles:u","-F","997","--call-graph","dwarf","-o",str(raw),"--",*command]
        else:
            raw=directory/"trace.nsys-rep"
            command=[tool,"profile","--sample=none","--trace=cuda","--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                     "--force-overwrite=false","-o",str(directory/"trace"),*command]
        record=log_command(command,directory,remaining(120),environment)
        record["iterations"]=count
        if record["returncode"]==0 and raw.exists():
            record["raw_sha256"]=sha(raw)
            compressed=raw.with_name(raw.name+".gz")
            with raw.open("rb") as source,gzip.open(compressed,"xb") as out: shutil.copyfileobj(source,out)
            record["gzip_sha256"]=sha(compressed)
            record["gzip_file"]=str(compressed.relative_to(args.output_dir))
            if args.platform=="intel": summary=[tool,"report","--stdio","--percent-limit","0.5","-i",str(raw)]
            else: summary=[tool,"stats","--report","cuda_gpu_kern_sum,cuda_api_sum","--format","csv",str(raw)]
            record["summary"]=log_command(summary,directory/"summary",remaining(120),environment)
        results["cases"][name]=record
        check_unchanged()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--build-receipt",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--platform",choices=["intel","nvidia"],required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--case",action="append")
    args=parser.parse_args()
    if args.output_dir.exists() or (args.case and not args.validate_only): parser.error("use a new directory; subsets are validation-only")
    args.output_dir=args.output_dir.absolute()
    args.protocol=args.protocol.absolute()
    p=load_protocol(args.protocol)
    spec=p["strata"][args.platform]
    selected=[c for c in p["cases"] if not args.case or c["name"] in args.case]
    if not selected or (args.case and set(args.case)!={c["name"] for c in selected}): parser.error("unknown/empty validation case set")
    os.sched_setaffinity(0,set(p["affinity"]))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",TORCHINDUCTOR_COMPILE_THREADS="1")
    cache=ROOT/"target/pointwise-inductor"/(args.platform+"-"+args.output_dir.name)
    if cache.exists(): raise ValueError("compiler context cache already exists")
    manifest=source_manifest()
    build=json.loads(args.build_receipt.read_text())
    if build["status"]!="built" or build["platform"]!=args.platform or build["returncode"]!=0 or build["source_sha256"]!=manifest or build["binary_sha256"]!=sha(args.binary) or build["environment"]["RUSTFLAGS"]!=p["build"]["rustflags"]:
        raise ValueError("build artifact/source/flags receipt differs")
    build_log=args.build_receipt.parent/build["log"]
    if sha(build_log)!=build["log_sha256"]: raise ValueError("build log hash differs")
    cache.mkdir(parents=True)
    environment["TORCHINDUCTOR_CACHE_DIR"]=str(cache)
    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.build_receipt,args.output_dir/"build-receipt.json")
    shutil.copyfile(build_log,args.output_dir/"build.log")
    shutil.copyfile(args.protocol,args.output_dir/"protocol.json")
    frozen,digest=freeze(args.binary.absolute(),ROOT/"target/pointwise-artifacts")
    with frozen.open("rb") as source,gzip.open(args.output_dir/"fa-pointwise.gz","xb") as out: shutil.copyfileobj(source,out)
    archive=args.output_dir/"source.tar.gz"
    with tarfile.open(archive,"x:gz") as bundle:
        for name in manifest: bundle.add(ROOT/name,arcname=name,recursive=False)
    report={"schema_version":1,"status":"running","timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "protocol":{"name":p["name"],"platform":args.platform,"parameters":p,"sha256":sha(args.protocol),"validation_only":args.validate_only,"promotion":False},
        "environment":{"source_sha256":manifest,"source_archive_sha256":sha(archive),"binary_sha256":digest,
            "binary_gzip_sha256":sha(args.output_dir/"fa-pointwise.gz"),"frozen_binary":str(frozen),"build_receipt_sha256":sha(args.build_receipt),
            "rustc":command_output(["rustc","--version"]),"cpu":cpu_identity(),"torch":torch.__version__,"python":sys.executable,
            "gpu_identity":command_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader"]) if spec["device"]=="cuda" else None,
            "nvcc":command_output(["nvcc","--version"]) if spec["device"]=="cuda" else None,
            "cxx_version":command_output([environment.get("CXX","c++"),"--version"]),
            "native_linkage":command_output(["ldd",str(frozen)]),"affinity":sorted(os.sched_getaffinity(0)),"compiler_cache":str(cache),
            "selected_env":{k:environment.get(k) for k in ["RUSTFLAGS","CXX","CUDA_HOST_CXX","CUDA_ARCH","CUDA_ROOT","OMP_NUM_THREADS","MKL_NUM_THREADS","TORCHINDUCTOR_COMPILE_THREADS"]}},
        "cases":[],"profiles":{},"limitations":["One persistent native/Torch worker pair per case; within-worker windows are not independent confirmation",
            "Capture is a separate Torch execution mode; native generic replay is not implemented",
            "Raw FP32 arrays and generated compiler caches remain on the collecting host; copied receipts alone do not recompute them",
            "No exact process/device peak, promotion, application or training inference"]}
    destination=args.output_dir/"comparison.json"
    def save():
        tmp=destination.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        tmp.replace(destination)
    def unchanged():
        if source_manifest()!=manifest or sha(frozen)!=digest: raise ValueError("frozen source or executable changed")
    started=time.monotonic()
    def remaining(limit):
        budget=p["budget"]["context_timeout_seconds"]-(time.monotonic()-started)
        if budget<=0: raise TimeoutError("declared context budget exhausted")
        return min(budget,limit)
    save()
    try:
        for case in selected:
            unchanged()
            index=p["cases"].index(case)
            case_record={"case":case,"telemetry_before":telemetry(spec["device"]),"workers":{},"readiness":{},"blocks":[]}
            report["cases"].append(case_record)
            save()
            print("case",case["name"],"prepare",flush=True)
            expected_inputs,expected=reference(case)
            active=[]
            first={}
            case_directory=args.output_dir/"workers"/case["name"]
            try:
                commands={"native":[str(frozen),"--protocol",str(args.protocol),"--device",spec["device"],"--case",case["name"],"--output-dir",str(case_directory/"native/data")],
                          "torch":[sys.executable,str(ROOT/"tools/pointwise_torch_worker.py"),"--protocol",str(args.protocol),"--platform",args.platform,"--case",case["name"],"--output-dir",str(case_directory/"torch/data")]}
                peers={}
                for engine,command in commands.items():
                    peer=Peer(command,case_directory/engine,environment)
                    active.append(peer)
                    peers[engine]=peer
                    case_record["workers"][engine]={"pid":peer.process.pid,"command":command}
                    save()
                    ready=peer.receive(remaining(p["budget"]["worker_prepare_timeout_seconds"]))
                    case_record["workers"][engine]["ready_wall_us"]=(time.monotonic_ns()-peer.started)/1000
                    case_record["workers"][engine]["ready"]=ready
                    if ready["status"]!="ready" or ready["case"]!=case or ready["window"]!=spec["windows"][case["regime"]] or ready["device"]!=spec["device"]: raise ValueError("worker readiness metadata differs")
                    if engine=="native":
                        if ready["debug_assertions"] or ready["compiled_features"]!=spec["compiled_features"]: raise ValueError("native build features/profile differ")
                        if ready["cpu_allowed_list"]!="0": raise ValueError("native worker affinity differs")
                    elif ready["affinity"]!=p["affinity"] or (ready["threads"],ready["interop_threads"])!=(1,1) or set(ready["modes"])!=set(spec["modes"])-{"native"}:
                        raise ValueError("Torch modes/environment differ")
                    if engine=="torch":
                        if ready["compiler"]["policy"]!=COMPILER_POLICY: raise ValueError("Torch compiler precision differs")
                        for mode,metadata in ready["modes"].items():
                            expected_writeback=p["compile"]["writeback"] if mode in ("compiled","compiled-cuda-graph") else "out-parameter"
                            if metadata["writeback"]!=expected_writeback: raise ValueError("Torch compiled writeback differs")
                    for mode in (["native"] if engine=="native" else spec["modes"][1:]):
                        validation=check_buffers(case,peer.directory/"data",mode,expected_inputs,expected)
                        validation["output_file"]=str(Path(validation["output_file"]).relative_to(args.output_dir))
                        case_record["readiness"][mode]=validation
                        first[mode]=validation["output_sha256"]
                        save()
                        if not validation["passed"]: raise ValueError("readiness full-output/input/hole gate failed")
                for block in range(1 if args.validate_only else spec["blocks"]):
                    row={"index":block,"order":order(p,args.platform,index,block),"samples":{}}
                    case_record["blocks"].append(row)
                    for mode in row["order"]:
                        peer=peers["native" if mode=="native" else "torch"]
                        sample=peer.request({"action":"measure","mode":mode,"block":block},remaining(p["budget"]["worker_command_timeout_seconds"]))
                        row["samples"][mode]=sample
                        if (sample["status"],sample["case"],sample["mode"],sample["block"],sample["window"])!=("measured",case["name"],mode,block,spec["windows"][case["regime"]]): raise ValueError("measured request/count metadata differs")
                        validation=check_buffers(case,peer.directory/"data",mode,expected_inputs,expected)
                        validation["output_file"]=str(Path(validation["output_file"]).relative_to(args.output_dir))
                        validation["repeat_exact"]=validation["output_sha256"]==first[mode]
                        sample["validation"]=validation
                        if not validation["passed"] or not validation["repeat_exact"]: raise ValueError("measured correctness/repeatability gate failed")
                    unchanged()
                    save()
                    print("case",case["name"],"block",block+1,"passed",flush=True)
                if not args.validate_only: case_record["summary"]=summarize(case_record["blocks"],spec["modes"])
            finally:
                for peer in active: case_record["workers"]["native" if peer is active[0] else "torch"]["shutdown"]=peer.close()
                save()
            if any(w["shutdown"]["returncode"]!=0 for w in case_record["workers"].values()): raise ValueError("worker did not exit successfully")
            count=1 if args.validate_only else spec["blocks"]
            for engine,worker in case_record["workers"].items():
                reply=worker["shutdown"].get("reply",{})
                expected=count if engine=="native" else dict.fromkeys(spec["modes"][1:],count)
                if reply.get("status")!="stopped" or reply.get("measured_blocks")!=expected:
                    raise ValueError("worker completed-block receipt differs")
            case_record["telemetry_after"]=telemetry(spec["device"])
            unchanged()
            save()
        if not args.validate_only:
            report["profiles"]=profiles(args,p,frozen,environment,unchanged,remaining)
            report["summary"]={mode:math.exp(statistics.mean(math.log(c["summary"][mode]["torch_over_native_descriptive_geomean"]) for c in report["cases"])) for mode in spec["modes"]}
        unchanged()
        report["status"]="validated_pointwise_harness" if args.validate_only else "characterized_pointwise_context"
        report["elapsed_seconds"]=time.monotonic()-started
        save()
    except Exception as exc:
        report.update(status="failed",error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        save()
        raise
    print(report["status"],flush=True)


if __name__=="__main__": main()
