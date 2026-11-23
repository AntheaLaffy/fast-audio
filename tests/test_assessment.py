"""Performance claims need both a primary win and every latency regression guard."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tools.assessment", ROOT / "tools/assess_candidate.py")
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)


def report(index, ratios):
    anchor = {"binary_sha256":"anchor", "backend":"mkl", "source_sha256":{"a.rs":"a"},
              "source_archive_sha256":"archive", "native_linkage":"libc.so (0x1234)"}
    environment = {"timestamp_utc":str(index), "anchor":anchor, "binary_sha256":"candidate",
        "source_sha256":{"b.rs":"b"}, "workloads_sha256":"workloads", "cpu_affinity":[0],
        "cpu_identity":"Intel", "gpu_identity":"none", "rustc":"1.98.1", "torch":"2.14.1",
        "torch_cuda":None, "torch_cudnn":None, "env":{}, "native_linkage":"libc.so (0xabcd)"}
    cases = []
    for i,ratio in enumerate(ratios):
        block = {"anchor":{"samples":[{"wall_us":100*ratio}]}, "rust":{"samples":[{"wall_us":100}]},
                 "correctness":{"passed":True}, "anchor_correctness":{"passed":True}}
        cases.append({"case":{"name":str(i)}, "blocks":[copy.deepcopy(block) for _ in range(7)],
                      "summary":{"anchor_wall_speedup_geomean":ratio}})
    return {"status":"verified_exploratory_baseline", "environment":environment, "cases":cases,
            "protocol":{"blocks":7,"iterations":50,"warmup":20}}


class AssessmentTests(unittest.TestCase):
    def assess(self, reports):
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            if reports[0]["protocol"].get("timing_profile"):
                (Path(folder)/"protocol.json").write_bytes((ROOT/"benchmarks/protocols/intel-long-window-v1.json").read_bytes())
            for i,value in enumerate(reports):
                path=Path(folder)/f"run-{i}.json"
                path.write_text(json.dumps(value))
                paths.append(path)
            return assessment.assess(paths)

    def test_positive_primary_and_all_guards(self):
        result=self.assess([report(i,[1.2]*7) for i in range(3)])
        self.assertEqual(result["status"],"accepted_operator_candidate")
        self.assertAlmostEqual(result["primary"]["ratio"],1.2)

    def test_large_primary_gain_cannot_hide_one_regression(self):
        result=self.assess([report(i,[0.8]+[2.0]*6) for i in range(3)])
        self.assertTrue(result["primary"]["gain_passed"])
        self.assertFalse(result["cases"]["0"]["guard_passed"])
        self.assertNotEqual(result["status"],"accepted_operator_candidate")

    def test_changed_candidate_is_not_pooled(self):
        runs=[report(i,[1.2]*7) for i in range(3)]
        runs[1]["environment"]["binary_sha256"]="different"
        with self.assertRaises(ValueError): self.assess(runs)

    def test_raw_samples_override_a_misleading_summary(self):
        runs=[report(i,[1.2]*7) for i in range(3)]
        runs[1]["cases"][0]["blocks"][0]["rust"]["samples"][0]["wall_us"]=1000
        with self.assertRaises(ValueError): self.assess(runs)

    def test_declared_windows_require_all_three_execution_ids(self):
        from tools.benchmark_protocol import load_profile
        names=[case["name"] for case in json.loads((ROOT/"benchmarks/workloads.json").read_text())["cases"]]
        profile,digest=load_profile(ROOT/"benchmarks/protocols/intel-long-window-v1.json",
                                   names)
        runs=[report(i,[1.2]*7) for i in range(3)]
        for index,run in enumerate(runs):
            run["execution_id"]=index+1
            run["protocol"].update(timing_profile=profile,backend="intel",threads=1,iterations=None,warmup=None)
            run["environment"].update(binary_sha256=profile["candidate"]["binary_sha256"],protocol_sha256=digest)
            run["environment"]["anchor"]["binary_sha256"]=profile["anchor"]["binary_sha256"]
            for case_id,(case,name) in enumerate(zip(run["cases"],profile["cases"])):
                case["case"]["name"]=name
                for block_id,block in enumerate(case["blocks"]):
                    block.update(block=block_id,order="anchor,rust,torch" if (index+case_id+block_id)%2==0 else "rust,anchor,torch")
                    block["torch"]={}
                    for key in ["rust","anchor","torch"]:
                        block[key].update(profile["cases"][name])
        self.assertEqual(self.assess(runs)["status"],"accepted_operator_candidate")
        # Reusing one execution's evidence cannot supply a third independent run.
        runs[2]=copy.deepcopy(runs[1])
        runs[2]["environment"]["timestamp_utc"]="a later invocation"
        with self.assertRaises(ValueError): self.assess(runs)


if __name__ == "__main__":
    unittest.main()
