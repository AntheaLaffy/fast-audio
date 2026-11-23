"""Validate declared per-case measurement windows without importing PyTorch."""
import hashlib
import json
from pathlib import Path
import re

BACKENDS = {"reference", "mkl", "onednn", "intel", "cuda", "cudnn", "nvidia"}


def validate_profile(profile, case_names):
    if profile.get("schema_version") != 1 or not isinstance(profile.get("name"), str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", profile["name"]):
        raise ValueError("invalid timing profile schema/name")
    if profile.get("executions") != 3 or profile.get("blocks") != 7:
        raise ValueError("confirmation profile must declare three executions and seven blocks")
    if profile.get("device") not in {"cpu", "cuda"}:
        raise ValueError("profile must declare cpu or cuda")
    if type(profile.get("threads")) is not int or not 1 <= profile["threads"] <= 65535:
        raise ValueError("invalid profile thread count")
    affinity = profile.get("cpu_affinity")
    if not isinstance(affinity, list) or not affinity or any(type(v) is not int or v < 0 for v in affinity) or affinity != sorted(set(affinity)):
        raise ValueError("profile must declare a sorted, nonempty CPU affinity")
    if profile.get("acceptance") != {"min_latency_reduction": 0.05, "max_latency_regression": 0.05}:
        raise ValueError("timing profiles cannot relax the 5% acceptance contract")
    for key in ["candidate", "anchor"]:
        artifact = profile.get(key, {})
        digest = artifact.get("binary_sha256", "")
        if artifact.get("backend") not in BACKENDS or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid {key} artifact identity")
    cases = profile.get("cases", {})
    if len(case_names) != 7 or set(cases) != set(case_names):
        raise ValueError("timing profile must cover exactly the seven workload cases")
    for name, timing in cases.items():
        if set(timing) != {"iterations", "warmup"} or any(type(v) is not int for v in timing.values()):
            raise ValueError(f"invalid timing fields for {name}")
        if timing["iterations"] < 50 or timing["warmup"] < 20:
            raise ValueError("confirmation windows cannot be shorter than the original 50/20 policy")
    return profile


def load_profile(path, case_names):
    blob = Path(path).read_bytes()
    return validate_profile(json.loads(blob), case_names), hashlib.sha256(blob).hexdigest()


def case_timing(profile, name, iterations=50, warmup=20):
    if profile is None:
        return iterations, warmup
    timing = profile["cases"][name]
    return timing["iterations"], timing["warmup"]


def validate_artifacts(profile, candidate_hash, anchor_hash, candidate_backend, anchor_backend, affinity):
    if profile["candidate"] != {"backend":candidate_backend, "binary_sha256":candidate_hash}:
        raise ValueError("candidate differs from the predeclared timing profile")
    if profile["anchor"] != {"backend":anchor_backend, "binary_sha256":anchor_hash}:
        raise ValueError("anchor differs from the predeclared timing profile")
    if profile["cpu_affinity"] != affinity:
        raise ValueError("CPU affinity differs from the predeclared timing profile")


def verify_report_profile(report, report_path):
    profile = report["protocol"].get("timing_profile")
    if profile is None:
        if (report["protocol"]["blocks"], report["protocol"]["iterations"], report["protocol"]["warmup"]) != (7,50,20):
            raise ValueError("legacy confirmation protocol differs from 7/50/20")
        return None
    names = [case["case"]["name"] for case in report["cases"]]
    saved, digest = load_profile(Path(report_path).parent / "protocol.json", names)
    if saved != profile or digest != report["environment"].get("protocol_sha256"):
        raise ValueError("saved timing profile differs from its report/hash")
    validate_artifacts(profile, report["environment"]["binary_sha256"],
        report["environment"]["anchor"]["binary_sha256"], report["protocol"]["backend"],
        report["environment"]["anchor"]["backend"], report["environment"]["cpu_affinity"])
    if report["protocol"]["threads"] != profile["threads"] or report["protocol"]["blocks"] != profile["blocks"]:
        raise ValueError("reported thread/block counts differ from the timing profile")
    execution = report.get("execution_id")
    if type(execution) is not int or not 1 <= execution <= profile["executions"]:
        raise ValueError("invalid declared execution id")
    for index, case in enumerate(report["cases"]):
        iterations, warmup = case_timing(profile, case["case"]["name"])
        for block_id, block in enumerate(case["blocks"]):
            expected_order = "anchor,rust,torch" if (execution-1+index+block_id)%2 == 0 else "rust,anchor,torch"
            if block["block"] != block_id or block["order"] != expected_order:
                raise ValueError("paired order differs from the declared counterbalancing")
            for key in ["rust", "anchor", "torch"]:
                if (block[key]["iterations"], block[key]["warmup"]) != (iterations,warmup):
                    raise ValueError("a measured implementation used different timing counts")
    return profile
