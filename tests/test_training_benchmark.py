import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import torch

from tools import training_protocol as protocol
from tools.training_torch_worker import State, compiled_step, configure_compiler
from tools.training_benchmark import selected_cases, seal_snapshot, import_nsys_trace
from tools.assess_training import verify_objects, assess, verify_gpu_events, verify_recovered_profiles


class TrainingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        configure_compiler()
        cls.p = protocol.load_protocol()
        cls.rate = torch.tensor(cls.p["rate"], dtype=torch.float32).item()
        cls.cases = [next(c for c in cls.p["cases"] if c["name"] == name) for name in ("dense-tiny", "wave-tiny", "spectrum-tiny")]

    def test_protocol_identity_and_full_counterbalanced_scope(self):
        self.assertEqual(len(self.p["cases"]), 12)
        self.assertEqual({c["family"] for c in self.p["cases"]}, {"dense", "waveform", "spectrum"})
        for platform, spec in self.p["strata"].items():
            for index in range(12):
                orders = [protocol.order(self.p, platform, index, b) for b in range(spec["blocks"])]
                for position in range(len(spec["modes"])):
                    for mode in spec["modes"]:
                        self.assertEqual(sum(o[position] == mode for o in orders), 2)
            with self.assertRaises(ValueError):
                protocol.order(self.p, platform, 0, spec["blocks"])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "protocol.json"
            changed = copy.deepcopy(self.p); changed["windows"]["large"]["iterations"] = 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                protocol.load_protocol(path)

    def test_fp64_physical_equations_match_autograd_and_finite_differences(self):
        for case in self.cases:
            x, target, parameters = protocol.fixtures(case, dtype=torch.float64)
            for p in parameters:
                p.requires_grad_()
            loss, gradients = protocol.analytic(case, x, target, parameters)
            actual_loss, _ = protocol.objective(case, x, target, *parameters)
            expected = torch.autograd.grad(actual_loss, parameters)
            self.assertAlmostEqual(float(loss.detach()), float(actual_loss.detach()), places=14)
            for side, (actual, reference) in enumerate(zip(gradients, expected)):
                self.assertTrue(torch.allclose(actual, reference, atol=1e-12, rtol=1e-12))
                for i in range(actual.numel()):
                    with torch.no_grad():
                        original = parameters[side].reshape(-1)[i].item()
                        parameters[side].reshape(-1)[i] = original + 1e-4
                        plus = protocol.objective(case, x, target, *parameters)[0].item()
                        parameters[side].reshape(-1)[i] = original - 1e-4
                        minus = protocol.objective(case, x, target, *parameters)[0].item()
                        parameters[side].reshape(-1)[i] = original
                    self.assertAlmostEqual((plus-minus)/2e-4, actual.reshape(-1)[i].item(), delta=2e-5)

    def test_real_fullgraph_training_and_post_update_snapshot_trajectories(self):
        for case in self.cases:
            compiled = torch.compile(compiled_step(case, self.rate), backend="inductor", fullgraph=True, dynamic=False)
            for frontend in (None, compiled):
                state = State(case, "cpu", self.rate, frontend)
                previous = [p.clone() for p in state.initial]
                with tempfile.TemporaryDirectory() as name:
                    directory = Path(name); (directory / "snapshot").mkdir()
                    for i in range(1, 9):
                        state.run()
                        controls = [protocol.control(case, i, fp64, self.rate) for fp64 in (True, False)]
                        snapshot = state.dump(directory / "snapshot")
                        checked = protocol.check_snapshot(snapshot, directory, controls)
                        update, previous = protocol.snapshot_update(snapshot, previous, self.rate)
                        self.assertEqual(len(update["max_parameter_ulps"]),len(previous))
                        self.assertEqual(set(checked["payloads"]), set(controls[0]))
                    old = checked["payloads"]
                    state.reset()
                    for _ in range(8):
                        state.run()
                    repeated = protocol.check_snapshot(state.dump(directory / "snapshot"), directory, controls)
                    self.assertEqual(repeated["payloads"], old)

    def test_snapshot_corruption_and_metadata_changes_cannot_pass(self):
        case = self.cases[1]; state = State(case, "cpu", self.rate); state.run()
        controls = [protocol.control(case, 1, fp64, self.rate) for fp64 in (True, False)]
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name); (directory / "snapshot").mkdir()
            initial = state.dump(directory / "snapshot")
            for field in initial:
                snapshot = state.dump(directory / "snapshot")
                path = Path(snapshot[field]["file"]); raw = path.read_bytes()
                path.write_bytes(torch.tensor([99.], dtype=torch.float32).numpy().tobytes() + raw[4:])
                with self.subTest(field=field), self.assertRaises(ValueError):
                    protocol.check_snapshot(snapshot, directory, controls)
            for mutation in ("scope", "shape", "path", "extent"):
                snapshot = state.dump(directory / "snapshot")
                if mutation == "scope":
                    del snapshot["gradient-1"]
                elif mutation == "shape":
                    snapshot["prediction"]["shape"] = [68]
                elif mutation == "path":
                    snapshot["loss"]["file"] = str(directory / "wrong.f32")
                else:
                    Path(snapshot["loss"]["file"]).write_bytes(b"")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    protocol.check_snapshot(snapshot, directory, controls)

    def test_partial_evidence_and_corrupt_sealed_payloads_cannot_be_admitted(self):
        for names, validation in ((["dense-tiny"], False), (["dense-tiny"]*2, True), (["unknown"], True)):
            with self.assertRaises(ValueError):
                selected_cases(self.p, names, validation)
        self.assertEqual(len(selected_cases(self.p, None, False)), 12)
        case = self.cases[0]; state = State(case, "cpu", self.rate); state.run()
        controls = [protocol.control(case, 1, fp64, self.rate) for fp64 in (True, False)]
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name); (directory / "snapshot").mkdir()
            snapshot = state.dump(directory / "snapshot")
            checked = protocol.check_snapshot(snapshot, directory, controls)
            sealed = seal_snapshot(snapshot, checked, directory)
            verify_objects(sealed, directory)
            item = next(iter(sealed["payloads"].values()))
            path = directory / item["object"]; path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError, "compressed.*changed"):
                verify_objects(sealed, directory)
            (directory / "comparison.json").write_text(json.dumps({"status":"verified_training_readiness_only","validation_only":True,"promotion":False}))
            with self.assertRaisesRegex(ValueError, "wrong-kind"):
                assess(directory)

    def test_update_consistency_catches_skips_inside_absolute_tolerance(self):
        previous = [torch.tensor([0.05,0.1])]; gradients = [torch.tensor([0.001,-0.002])]
        actual = [previous[0]-gradients[0]*self.rate]
        self.assertEqual(protocol.sgd_update(previous,gradients,actual,self.rate)["max_parameter_ulps"],[0])
        for wrong in (previous,[previous[0]-2*gradients[0]*self.rate]):
            self.assertTrue(torch.allclose(wrong[0],actual[0],atol=2e-5,rtol=2e-5))
            with self.assertRaisesRegex(ValueError,"update missing"):
                protocol.sgd_update(previous,gradients,wrong,self.rate)
        wrong=actual[0].clone()
        for _ in range(5):
            wrong=torch.nextafter(wrong,torch.full_like(wrong,float('inf')))
        with self.assertRaises(ValueError):
            protocol.sgd_update(previous,gradients,[wrong],self.rate)

    def test_mutating_nsys_importer_only_receives_a_version_matched_copy(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name); prefix = directory/"trace"
            raw = prefix.with_suffix(".qdstrm"); original = b"original header and captured events"
            raw.write_bytes(original)
            importer = directory/"QdstrmImporter"
            importer.write_text("#!"+sys.executable+"\nimport sys\nfrom pathlib import Path\n"
                "if sys.argv[1:] == ['--version']:\n print('NVIDIA Nsight Systems 2023.4.4.54-build')\n"
                "else:\n source=Path(sys.argv[2]); source.write_bytes(b'changed header'+source.read_bytes()+b'metadata')\n"
                " Path(sys.argv[4]).write_bytes(b'imported report')\n")
            importer.chmod(0o755)
            environment = {**os.environ, "NSYS_QDSTRM_IMPORTER": str(importer)}
            with self.assertRaisesRegex(ValueError, "versions differ"):
                import_nsys_trace("/not/installed/nsys", prefix, environment, 10, "2024.1.1.1-build")
            self.assertFalse((directory/"import-input.qdstrm").exists())
            result = import_nsys_trace("/not/installed/nsys", prefix, environment, 10, "NVIDIA Nsight Systems version 2023.4.4.54-build")
            self.assertEqual(raw.read_bytes(), original)
            self.assertNotEqual(result["original_sha256"], result["copied_input_after_sha256"])
            self.assertEqual(result["execution"]["returncode"], 0)
            self.assertIsNone(import_nsys_trace("/not/installed/nsys", prefix, environment, 10, "unused"))

    def test_profile_recovery_cannot_admit_other_failures_or_rerun_windows(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path = root/"recovery.json"
            report = {"platform": "nvidia", "status": "failed", "error": "training profile summary failed",
                      "producer_commit": "producer", "protocol_sha256": "protocol", "binary_sha256": "binary"}
            (root/"comparison.json").write_text(json.dumps(report))
            recovered = {key:report[key] for key in ("producer_commit", "protocol_sha256", "binary_sha256")}
            recovered.update(status="recovered_complete_training_profiles", original_comparison_sha256=protocol.sha(root/"comparison.json"),
                             timing_windows_rerun=0, cases={name:{} for name in self.p["profiles"]["nvidia"]})
            for mutation, message in (("report", "status/report"), ("producer", "producer differs"), ("windows", "scope differs"), ("failure", "known training")):
                changed = copy.deepcopy(recovered); source = copy.deepcopy(report)
                if mutation == "report":
                    changed["original_comparison_sha256"] = "0"*64
                elif mutation == "producer":
                    changed["binary_sha256"] = "changed"
                elif mutation == "windows":
                    changed["timing_windows_rerun"] = 1
                else:
                    source["error"] = "numerical validation failed"
                path.write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                    verify_recovered_profiles(path, root, source, self.p)

    def test_gpu_event_audit_rejects_skipped_updates_errors_and_invalid_correlations(self):
        with tempfile.TemporaryDirectory() as name:
            database = Path(name)/"trace.sqlite"; connection = sqlite3.connect(database)
            connection.executescript("CREATE TABLE StringIds(id INTEGER,value TEXT);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,correlationId INTEGER,deviceId INTEGER,streamId INTEGER);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,correlationId INTEGER,returnValue INTEGER);")
            connection.executemany("INSERT INTO StringIds VALUES (?,?)", [(1,"cudaLaunchKernel_v7000"),(2,"<unnamed>::sgd(unsigned long, float *, const float *, float)"),(3,"pointwise")])
            for i in range(6):
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)", (i*1000+500,i*1000+600,3 if i%3==2 else 2,i+1,0,7))
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", (i*1000,i*1000+200,1,i+1,0))
            connection.commit(); connection.close(); original = database.read_bytes()
            evidence = verify_gpu_events(database, 2, 2)
            self.assertEqual((evidence["kernels_per_step"], evidence["sgd_instances"]), (3,4))
            changes = ["UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1 WHERE correlationId=1",
                       "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=1 WHERE correlationId=2",
                       "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=8 WHERE correlationId=1",
                       "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=99999 WHERE correlationId=1",
                       "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=3 WHERE correlationId=1",
                       "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=1"]
            for sql in changes:
                database.write_bytes(original); connection = sqlite3.connect(database)
                connection.execute(sql); connection.commit(); connection.close()
                with self.subTest(sql=sql), self.assertRaises(ValueError):
                    verify_gpu_events(database, 2, 2)


if __name__ == "__main__":
    unittest.main()
