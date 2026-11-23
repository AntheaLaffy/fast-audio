#!/usr/bin/env python3
"""Verify a comparison report's source archive and optionally restore it to an empty directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

try:
    from .benchmark_protocol import verify_report_profile
except ImportError:
    from benchmark_protocol import verify_report_profile


def verify(report_path, restore=None, anchor=False):
    report = json.loads(report_path.read_text())
    profile = None
    if report["protocol"].get("timing_profile") is not None:
        profile = verify_report_profile(report, report_path)
    environment = report["environment"]["anchor"] if anchor else report["environment"]
    manifest = environment["source_sha256"]
    archive = report_path.parent / ("anchor-source.tar.gz" if anchor else "source.tar.gz")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != environment["source_archive_sha256"]:
        raise ValueError("source archive SHA-256 mismatch")
    contents = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or member.name in contents:
                raise ValueError(f"unsafe or duplicate archive member: {member.name}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            contents[member.name] = stream.read()
    if contents.keys() != manifest.keys():
        raise ValueError("archive inventory differs from source manifest")
    for name, content in contents.items():
        if hashlib.sha256(content).hexdigest() != manifest[name]:
            raise ValueError(f"source hash mismatch: {name}")
    if restore is not None:
        if restore.exists() and (not restore.is_dir() or any(restore.iterdir())):
            raise ValueError("restore destination must be absent or an empty directory")
        restore.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            path = restore / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if profile is not None:
            blob = (report_path.parent / "protocol.json").read_bytes()
            # Early timing-profile snapshots kept this hashed fixture only as a
            # companion. Restore its canonical path too so archived tests can run.
            for relative in ["protocol.json", f"benchmarks/protocols/{profile['name']}.json"]:
                path = restore / relative
                if path.exists() and path.read_bytes() != blob:
                    raise ValueError("companion timing profile conflicts with archived source")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
    return len(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--anchor", action="store_true")
    args = parser.parse_args()
    count = verify(args.report, args.restore, args.anchor)
    print(f"Verified archive and {count} source hashes" + (f"; restored to {args.restore}" if args.restore else ""))


if __name__ == "__main__":
    main()
