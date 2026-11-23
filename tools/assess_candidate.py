#!/usr/bin/env python3
"""Assess one frozen candidate using three fresh, paired seven-case confirmation runs."""
import argparse
import json
import math
import re
from pathlib import Path
import statistics

try:
    from .benchmark_protocol import verify_report_profile
except ImportError:
    from benchmark_protocol import verify_report_profile


def interval(logs):
    # Two-sided Student t 95% CI, df=2. The independent unit is a fresh harness
    # execution; inner iterations/blocks are deliberately not pooled as replicates.
    mean = statistics.mean(logs)
    margin = 4.302652729911275 * statistics.stdev(logs) / math.sqrt(3)
    return {"ratio": math.exp(mean), "ci95": [math.exp(mean-margin), math.exp(mean+margin)]}


def assess(paths):
    if len(paths) != 3 or len({p.resolve() for p in paths}) != 3:
        raise ValueError("confirmation requires three distinct harness executions")
    runs = [json.loads(p.read_text()) for p in paths]
    first = runs[0]
    cases = [c["case"]["name"] for c in first["cases"]]
    if len({run["environment"]["timestamp_utc"] for run in runs}) != 3:
        raise ValueError("confirmation reports must be from distinct execution start times")
    if len(cases) != 7 or len(set(cases)) != 7:
        raise ValueError("confirmation must cover all seven declared strata")
    profiles = [verify_report_profile(run,path) for run,path in zip(runs,paths)]
    if profiles[0] is not None and {run.get("execution_id") for run in runs} != {1,2,3}:
        raise ValueError("confirmation must contain each declared execution exactly once")
    for run in runs:
        if run["status"] != "verified_exploratory_baseline":
            raise ValueError("a run has not passed its numerical gates")
        required = {"binary_sha256", "source_sha256", "workloads_sha256", "cpu_affinity", "cpu_identity",
                    "gpu_identity", "rustc", "anchor", "torch", "torch_cuda", "torch_cudnn", "env", "native_linkage"}
        if not required.issubset(run["environment"]):
            raise ValueError("confirmation environment metadata is incomplete")
        for field in ["binary_sha256", "source_sha256", "workloads_sha256", "cpu_affinity", "cpu_identity", "gpu_identity", "rustc", "anchor", "torch", "torch_cuda", "torch_cudnn", "env", "protocol_sha256"]:
            # ASLR addresses vary in ldd output; the nested anchor linkage is
            # compared separately by its immutable binary digest below.
            if field == "anchor":
                for key in ["binary_sha256", "backend", "source_sha256", "source_archive_sha256"]:
                    if run["environment"][field][key] != first["environment"][field][key]:
                        raise ValueError(f"anchor {key} changed during confirmation")
            elif run["environment"].get(field) != first["environment"].get(field):
                raise ValueError(f"context/candidate field {field} changed during confirmation")
        for environment in ["candidate", "anchor"]:
            current = run["environment"] if environment == "candidate" else run["environment"]["anchor"]
            original = first["environment"] if environment == "candidate" else first["environment"]["anchor"]
            normalize = lambda value: re.sub(r"0x[0-9a-fA-F]+", "<address>", value)
            if normalize(current["native_linkage"]) != normalize(original["native_linkage"]):
                raise ValueError("native linkage changed during confirmation")
        if run["protocol"] != first["protocol"]:
            raise ValueError("confirmation protocol changed between executions")
        if [c["case"]["name"] for c in run["cases"]] != cases:
            raise ValueError("strata changed during confirmation")
        for case in run["cases"]:
            if len(case["blocks"]) != 7 or not all(b["correctness"]["passed"] and b["anchor_correctness"]["passed"] for b in case["blocks"]):
                raise ValueError("numerical validation is incomplete")
    def case_log(run, i):
        values=[]
        for block in run["cases"][i]["blocks"]:
            anchor=block["anchor"]["samples"][0]["wall_us"]
            candidate=block["rust"]["samples"][0]["wall_us"]
            if not all(math.isfinite(v) and v>0 for v in [anchor,candidate]):
                raise ValueError("invalid raw latency sample")
            values.append(math.log(anchor/candidate))
        value=statistics.mean(values)
        if not math.isclose(math.exp(value),run["cases"][i]["summary"]["anchor_wall_speedup_geomean"],rel_tol=1e-12):
            raise ValueError("case summary differs from raw paired measurements")
        return value
    logs=[[case_log(run,i) for i in range(len(cases))] for run in runs]
    per_case = {}
    for i,name in enumerate(cases):
        gate = interval([execution[i] for execution in logs])
        gate["guard_passed"] = gate["ci95"][0] >= 1/1.05
        per_case[name] = gate
    primary = interval([statistics.mean(execution) for execution in logs])
    primary["gain_passed"] = primary["ci95"][0] >= 1/0.95
    accepted = primary["gain_passed"] and all(c["guard_passed"] for c in per_case.values())
    return {"schema_version":1, "status":"accepted_operator_candidate" if accepted else "inconclusive_or_rejected_keep_anchor",
            "scope":"one frozen candidate, one confirmation campaign, approximate log-normal execution ratios",
            "primary":primary, "cases":per_case, "reports":[str(p) for p in paths]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.reports)
    if args.output.exists():
        raise ValueError("assessment output already exists")
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
