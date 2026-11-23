import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.gtcrn_benchmark import CASES, check_execution, freeze_binary, gpu_sample, load_protocol, load_screen_protocol, summarize

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "benchmarks/protocols/gtcrn-characterization-v1.json"


class ModelCharacterizationTests(unittest.TestCase):
    def test_native_screen_preserves_its_fixed_budget_and_counts(self):
        path = ROOT / "benchmarks/protocols/mapped-access-screen-v1.json"
        original = load_screen_protocol(path)
        variants = []
        for key, value in [("atol", 0.001), ("rtol", 0.001), ("blocks", 6), ("warmup", 1)]:
            p = copy.deepcopy(original)
            p[key] = value
            variants.append(p)
        for path_keys, value in [(("budget", "promotion"), True), (("budget", "candidate_implementations"), 2),
                                 (("strata", "intel", "iterations", "b1-l64000"), 5),
                                 (("strata", "nvidia", "compiled_features", "mkl"), True)]:
            p = copy.deepcopy(original)
            target = p
            for key in path_keys[:-1]:
                target = target[key]
            target[path_keys[-1]] = value
            variants.append(p)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "protocol.json"
            for p in variants:
                target.write_text(json.dumps(p))
                with self.assertRaises(ValueError):
                    load_screen_protocol(target)

    def test_protocol_rejects_relaxed_gate_missing_shape_and_promotion(self):
        self.assertEqual(load_protocol(PROFILE)["blocks"], 7)
        original = json.loads(PROFILE.read_text())
        variants = []
        for key, value in [("atol", 0.001), ("rtol", 0.001), ("blocks", 8), ("warmup", 0)]:
            p = copy.deepcopy(original)
            p[key] = value
            variants.append(p)
        p = copy.deepcopy(original)
        p["cases"].pop()
        variants.append(p)
        p = copy.deepcopy(original)
        p["budget"]["promotion"] = True
        variants.append(p)
        p = copy.deepcopy(original)
        p["strata"]["intel"]["iterations"]["b1-l4096"] = 0
        variants.append(p)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for p in variants:
                path.write_text(json.dumps(p))
                with self.assertRaises(ValueError):
                    load_protocol(path)

    def test_frozen_executable_survives_source_rebuild_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "binary"
            source.write_bytes(b"first executable")
            frozen, first_hash = freeze_binary(source, root / "artifacts")
            source.write_bytes(b"rebuilt executable")
            second, second_hash = freeze_binary(source, root / "artifacts")
            self.assertNotEqual(first_hash, second_hash)
            self.assertNotEqual(frozen, second)
            self.assertEqual(frozen.read_bytes(), b"first executable")
            frozen.chmod(0o755)
            frozen.write_bytes(b"corrupted executable")
            source.write_bytes(b"first executable")
            with self.assertRaisesRegex(ValueError, "frozen executable"):
                freeze_binary(source, root / "artifacts")

    def test_worker_counts_scope_and_compiled_features_are_checked(self):
        report = {"backend": "mkl", "compiled_features": {"mkl": True, "onednn": True, "cuda": False, "cudnn": False},
                  "protocol": {"warmup": 20, "iterations": 50, "samples": 1, "traces": False, "pipeline_only": True},
                  "cases": [{"case": "b1-l4096", "timing": {"pipeline": {"wall_us": [100.0]}}}]}
        check_execution(report, "native", "b1-l4096", 20, 50, "mkl", "eager")
        for key, value in [("iterations", 1), ("samples", 2), ("traces", True), ("pipeline_only", False)]:
            changed = copy.deepcopy(report)
            changed["protocol"][key] = value
            with self.assertRaises(ValueError):
                check_execution(changed, "native", "b1-l4096", 20, 50, "mkl", "eager")
        report["compiled_features"]["onednn"] = False
        with self.assertRaisesRegex(ValueError, "compiled feature"):
            check_execution(report, "native", "b1-l4096", 20, 50, "mkl", "eager")

    def test_nvml_observation_excludes_other_process_groups(self):
        completed = type("Result", (), {"returncode": 0, "stdout": "11, 200\n22, 300\n33, 400\n", "stderr": ""})()
        with patch("tools.gtcrn_benchmark.subprocess.run", return_value=completed), patch("tools.gtcrn_benchmark.process_group", side_effect=lambda pid: 7 if pid in [11, 33] else 8):
            observation = gpu_sample(7)
        self.assertEqual(observation["mib"], 600)
        self.assertEqual([r["pid"] for r in observation["processes"]], [11, 33])

    def test_summary_uses_all_shapes_and_rejects_incomplete_or_duplicate_blocks(self):
        blocks = []
        for block in range(7):
            for name, _, _ in CASES:
                def engine(wall):
                    return {"record": {"timing": {"pipeline": {"wall_us": [wall]}}}, "resource": {"max_rss_kib": 123}}
                blocks.append({"block": block, "case": name, "native": engine(10), "torch": engine(20)})
        result = summarize(blocks)
        self.assertAlmostEqual(result["four_case_paired_ratio_geomean"], 2.0)
        self.assertFalse(result["promotion"])
        native_pairs = [{"block": r["block"], "case": r["case"], "candidate": r["native"], "control": r["torch"]} for r in blocks]
        screen = summarize(native_pairs, "candidate", "control")
        self.assertAlmostEqual(screen["four_case_paired_ratio_geomean"], 2.0)
        self.assertIn("control_wall_us_median", screen["cases"][CASES[0][0]])
        self.assertNotIn("torch_wall_us_median", screen["cases"][CASES[0][0]])
        with self.assertRaises(ValueError):
            summarize(blocks[:-1])
        changed = copy.deepcopy(blocks)
        changed[0]["case"] = changed[1]["case"]
        with self.assertRaises(ValueError):
            summarize(changed)


if __name__ == "__main__":
    unittest.main()
