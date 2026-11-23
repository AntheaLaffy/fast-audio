#!/usr/bin/env python3
"""Build and freeze the exact committed native complete-training worker."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.training_protocol import load_protocol, sha, require
from tools.compare_torch import source_manifest, command_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["intel", "nvidia"], required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--producer-manifest", type=Path, help="Sealed local Git inventory for a source mirror without .git")
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.producer), "use a full committed producer ID")
    require(not args.output_dir.exists(), "use a new build directory; preserve failures")
    p = load_protocol()
    manifest = source_manifest()
    if args.producer_manifest:
        sealed=json.loads(args.producer_manifest.read_text())
        require(sealed["producer_commit"]==args.producer, "sealed producer identity differs")
        committed=sealed["source_sha256"]
    else:
        archive = subprocess.check_output(["git", "archive", args.producer, *manifest], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            committed = {m.name: hashlib.sha256(source.extractfile(m).read()).hexdigest() for m in source.getmembers() if m.isfile()}
    require(committed == manifest, "working implementation differs from committed producer")
    environment = os.environ.copy()
    environment.update(RUSTFLAGS=p["build"]["rustflags"], CARGO_TERM_COLOR="never")
    if args.platform == "nvidia":
        environment.update(CUDA_ARCH=p["build"]["cuda_arch"], CUDA_HOST_CXX=p["build"]["cuda_host_cxx"])
    command = ["cargo", "build", "--release", "--locked", "--offline", "--bin", "fa-training",
               "--features", ",".join(p["build"]["features"][args.platform])]
    if args.vendor_dir:
        command += ["--config", 'source.crates-io.replace-with="vendored-sources"', "--config",
                    'source.vendored-sources.directory="' + str(args.vendor_dir.absolute()) + '"']
    directory = args.output_dir.absolute()
    directory.mkdir(parents=True)
    receipt = directory / "build-receipt.json"
    result = {"schema_version": 1, "status": "running", "producer_commit": args.producer,
              "platform": args.platform, "command": command, "cwd": str(ROOT), "source_sha256": manifest,
              "environment": {k: environment.get(k) for k in ("RUSTFLAGS", "CUDA_ARCH", "CUDA_HOST_CXX", "CXX", "CARGO_HOME", "LD_LIBRARY_PATH")},
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log": "build.log"}
    result["producer_proof"]={"kind":"git_archive"}
    if args.producer_manifest:
        shutil.copyfile(args.producer_manifest,directory/"producer-manifest.json")
        result["producer_proof"]={"kind":"sealed_manifest","file":"producer-manifest.json","sha256":sha(directory/"producer-manifest.json")}
    def save():
        tmp = receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(receipt)
    save()
    with tarfile.open(directory / "source.tar.gz", "x:gz") as source:
        for name in manifest:
            source.add(ROOT / name, arcname=name, recursive=False)
    result["source_archive_sha256"] = sha(directory / "source.tar.gz")
    begin = time.monotonic()
    with (directory / "build.log").open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        result["pid"] = process.pid; save()
        timed_out=False
        try:
            code = process.wait(timeout=p["limits"]["context_seconds"])
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);code=process.wait(timeout=10)
    result["timeout"]=timed_out
    result.update(returncode=code, elapsed_seconds=time.monotonic() - begin, log_sha256=sha(directory / "build.log"))
    if code or timed_out or source_manifest() != manifest:
        result.update(status="failed", error="build failed or source changed"); save()
        raise RuntimeError(result["error"])
    binary = ROOT / "target/release/fa-training"
    frozen = ROOT / "target/benchmarks/training" / sha(binary) / "fa-training"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        shutil.copyfile(binary, frozen); frozen.chmod(0o555)
    require(sha(frozen) == sha(binary), "frozen executable differs")
    with gzip.open(directory / "fa-training.gz", "xb") as stream:
        stream.write(frozen.read_bytes())
    result.update(status="built", binary=str(frozen), binary_sha256=sha(frozen),
                  binary_gzip="fa-training.gz", binary_gzip_sha256=sha(directory / "fa-training.gz"),
                  rustc=command_output(["rustc", "--version"]), native_linkage=command_output(["ldd", str(frozen)]))
    save()
    print("built", args.platform, result["binary_sha256"], flush=True)


if __name__ == "__main__":
    main()
