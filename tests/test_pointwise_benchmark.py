import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

from tools.pointwise_protocol import PROTOCOL, COMPILER_POLICY, UNARY, BINARY, load_protocol, order, reference, storage, extent, view, tensor_bytes, check_buffers, summarize
from tools.pointwise_benchmark import Peer
from tools.assess_pointwise import assess
from tools.pointwise_torch_worker import compiled_writeback, configure_compiler


class PointwiseProtocolTests(unittest.TestCase):
    def test_declared_cases_cover_operations_and_layout_regimes(self):
        p=load_protocol()
        self.assertEqual(len(p["cases"]),56)
        for regime in ("tiny","bulk"):
            self.assertEqual({c["operation"] for c in p["cases"] if c["regime"]==regime},set(UNARY+BINARY))
        self.assertEqual({c["regime"] for c in p["cases"]},
            {"tiny","bulk","channel","outer","transpose-read","transpose-write","gapped","overlap","scalar-left","scalar-right"})
        for case in p["cases"]:
            self.assertEqual(len(case["inputs"]),1 if case["operation"] in UNARY else 2)
            self.assertGreater(extent(case["output"]),0)

    def test_every_mode_occupies_every_position_in_both_order_directions(self):
        p=load_protocol()
        for platform in p["strata"]:
            modes=p["strata"][platform]["modes"]
            for index in range(56):
                orders=[order(p,platform,index,b) for b in range(2*len(modes))]
                for position in range(len(modes)):
                    for mode in modes: self.assertEqual(sum(o[position]==mode for o in orders),2)
                self.assertEqual(orders[len(modes):],[o[::-1] for o in orders[:len(modes)]])
            with self.assertRaises(ValueError): order(p,platform,0,p["strata"][platform]["blocks"])

    def test_changed_gates_cases_windows_or_budget_cannot_reuse_v2(self):
        original=load_protocol()
        mutations=[]
        for path,value in [("atol",1e-3),("rtol",1e-3),("threads",2)]:
            p=copy.deepcopy(original);p[path]=value;mutations.append(p)
        p=copy.deepcopy(original);p["cases"].pop();mutations.append(p)
        p=copy.deepcopy(original);p["strata"]["intel"]["windows"]["tiny"]["iterations"]=1;mutations.append(p)
        p=copy.deepcopy(original);p["budget"]["promotion"]=True;mutations.append(p)
        p=copy.deepcopy(original);del p["compile"]["writeback"];mutations.append(p)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"protocol.json"
            for p in mutations:
                path.write_text(json.dumps(p))
                with self.assertRaises(ValueError): load_protocol(path)

    def test_v1_runner_is_retired_and_v2_preserves_its_entire_workload(self):
        path=PROTOCOL.with_name("pointwise-characterization-v1.json")
        old=json.loads(path.read_text())
        new=copy.deepcopy(load_protocol())
        new["name"]=old["name"]
        del new["compile"]["writeback"]
        self.assertEqual(new,old)
        with self.assertRaises(ValueError): load_protocol(path)

    def test_fullgraph_writeback_keeps_noncontiguous_destination_and_source_bits(self):
        import torch
        from torch._inductor import config
        torch.set_num_threads(1)
        backing=torch.arange(16,dtype=torch.float32)/8-1
        source=backing.as_strided([2,3],[7,2],1)
        other=torch.tensor([0.25,0.5,0.75])
        before=tensor_bytes(backing)
        for name,arity in [("add",2),("softplus",1)]:
            destination_storage=torch.full([24],0x7fc12345,dtype=torch.int32).view(torch.float32)
            destination=destination_storage.as_strided([2,3],[9,2],2)
            mask=torch.zeros(24,dtype=torch.bool)
            mask.as_strided([2,3],[9,2],2).fill_(True)
            expected=source.double()+other.double() if arity==2 else torch.logaddexp(source.double(),torch.zeros_like(source,dtype=torch.float64))
            with config.patch(compile_threads=1),torch.inference_mode():
                function=torch.compile(compiled_writeback(name,arity),fullgraph=True,dynamic=False,mode="max-autotune-no-cudagraphs")
                for _ in range(2):
                    result=function(source,other if arity==2 else None,destination)
                    self.assertEqual(result.data_ptr(),destination.data_ptr())
                    self.assertEqual(result.stride(),destination.stride())
                    self.assertTrue(bool(((destination.double()-expected).abs()<=2e-6+3e-6*expected.abs()).all()))
                    self.assertTrue(bool((destination_storage.view(torch.int32)[~mask]==0x7fc12345).all()))
            self.assertEqual(tensor_bytes(backing),before)
        with self.assertRaises(ValueError): compiled_writeback("add",1)

    def test_build_producer_must_be_a_full_commit_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"build"
            result=subprocess.run([sys.executable,str(PROTOCOL.parents[2]/"tools/build_pointwise.py"),"--platform","intel",
                "--producer","HEAD","--output-dir",str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("exact full 40-character commit",result.stderr)
            self.assertFalse(output.exists())

    def test_declared_compiler_precision_overrides_relaxed_process_settings(self):
        from torch._inductor import config
        with config.patch({"use_fast_math":True,"cpp.enable_unsafe_math_opt_flag":True,
                           "cpp.enable_floating_point_contract_flag":"fast","cuda.use_fast_math":True}):
            self.assertEqual(configure_compiler()["policy"],COMPILER_POLICY)

    def test_full_backing_gate_detects_holes_input_mutation_and_wrong_math(self):
        import torch
        torch.set_num_threads(1)
        case=next(c for c in load_protocol()["cases"] if c["name"]=="gapped-add")
        expected_inputs,expected=reference(case)
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            for i in range(2): (directory/f"input-{i}.f32").write_bytes(tensor_bytes(storage(case,i)))
            output=torch.full([extent(case["output"])],0x7fc12345,dtype=torch.int32).view(torch.float32)
            view(output,case["output"]).copy_(expected.float())
            path=directory/"native-out.f32"
            good=tensor_bytes(output);path.write_bytes(good)
            self.assertTrue(check_buffers(case,directory,"native",expected_inputs,expected)["passed"])
            output[-1]=0;path.write_bytes(tensor_bytes(output))
            result=check_buffers(case,directory,"native",expected_inputs,expected)
            self.assertFalse(result["passed"]);self.assertFalse(result["holes_preserved"])
            path.write_bytes(good)
            source=directory/"input-0.f32";data=source.read_bytes();source.write_bytes(bytes(4)+data[4:])
            self.assertFalse(check_buffers(case,directory,"native",expected_inputs,expected)["passed"])
            source.write_bytes(data)
            view(output,case["output"])[0,0]=float("nan");path.write_bytes(tensor_bytes(output))
            result=check_buffers(case,directory,"native",expected_inputs,expected)
            self.assertFalse(result["finite"]);self.assertFalse(result["passed"])

    def test_statistics_reject_missing_modes_failed_checks_and_invalid_latencies(self):
        modes=["native","eager","compiled"]
        row={"samples":{mode:{"timing":{"wall_us":value,"device_us":None},
            "validation":{"passed":True,"repeat_exact":True}} for mode,value in zip(modes,[10,20,5])}}
        summary=summarize([row],modes)
        self.assertAlmostEqual(summary["eager"]["torch_over_native_descriptive_geomean"],2)
        self.assertAlmostEqual(summary["compiled"]["torch_over_native_descriptive_geomean"],0.5)
        bad=copy.deepcopy(row);del bad["samples"]["compiled"]
        with self.assertRaises(ValueError): summarize([bad],modes)
        for key in ("passed","repeat_exact"):
            bad=copy.deepcopy(row);bad["samples"]["eager"]["validation"][key]=False
            with self.assertRaises(ValueError): summarize([bad],modes)
        for value in (0,-1,float("nan"),float("inf")):
            bad=copy.deepcopy(row);bad["samples"]["native"]["timing"]["wall_us"]=value
            with self.assertRaises(ValueError): summarize([bad],modes)

    def test_peer_preserves_fragmented_json_and_reaps_its_worker(self):
        script="import sys,time;sys.stdout.write('{\\\"sta');sys.stdout.flush();time.sleep(.02);print('tus\\\":\\\"ready\\\"}',flush=True);line=sys.stdin.readline();print('{\\\"status\\\":\\\"stopped\\\"}',flush=True)"
        # JSON literals are code arguments, not shell text; no quoting layer is involved.
        script=script.replace('\\"','"')
        with tempfile.TemporaryDirectory() as folder:
            peer=Peer([sys.executable,"-c",script],Path(folder)/"worker",os.environ.copy())
            try: self.assertEqual(peer.receive(5),{"status":"ready"})
            finally: result=peer.close()
            self.assertEqual(result["returncode"],0)
            self.assertIsNotNone(peer.process.poll())
            self.assertIn(b'"ready"',(peer.directory/"stdout.jsonl").read_bytes())

    def test_partial_failed_and_validation_runs_cannot_be_assessed_as_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"comparison.json"
            for status in ("running","failed","validated_pointwise_harness"):
                path.write_text(json.dumps({"status":status,"protocol":{"validation_only":True}}))
                with self.assertRaises(ValueError): assess(path)


if __name__=="__main__": unittest.main()
