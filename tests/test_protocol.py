"""Declared timing windows must be applied to every measured implementation."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.benchmark_protocol import load_profile, validate_profile, validate_artifacts, verify_report_profile
from tools.verify_archive import verify

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "benchmarks/protocols/intel-long-window-v1.json"
PROFILE = json.loads(PROFILE_PATH.read_text())
NAMES = list(PROFILE["cases"])


class ProtocolTests(unittest.TestCase):
    def test_profile_preserves_full_scope_and_thresholds(self):
        self.assertEqual(validate_profile(PROFILE,NAMES),PROFILE)
        for edit in [lambda p:p["cases"].pop("stft-stream"),
                     lambda p:p["acceptance"].update(min_latency_reduction=0.01),
                     lambda p:p.update(name="../escape"),
                     lambda p:p["cases"]["stft-stream"].update(iterations=True)]:
            changed=copy.deepcopy(PROFILE); edit(changed)
            with self.assertRaises(ValueError): validate_profile(changed,NAMES)

    def test_frozen_artifacts_and_affinity_are_required(self):
        validate_artifacts(PROFILE,PROFILE["candidate"]["binary_sha256"],PROFILE["anchor"]["binary_sha256"],"intel","mkl",[0])
        for candidate,affinity in [("0"*64,[0]),(PROFILE["candidate"]["binary_sha256"],[0,1])]:
            with self.assertRaises(ValueError):
                validate_artifacts(PROFILE,candidate,PROFILE["anchor"]["binary_sha256"],"intel","mkl",affinity)

    def test_report_checks_actual_counts_and_counterbalanced_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"comparison.json"
            (path.parent/"protocol.json").write_bytes(PROFILE_PATH.read_bytes())
            profile,digest=load_profile(path.parent/"protocol.json",NAMES)
            report={"execution_id":2,"protocol":{"timing_profile":profile,"backend":"intel","threads":1,"blocks":7},
                    "environment":{"binary_sha256":profile["candidate"]["binary_sha256"],
                        "anchor":{"binary_sha256":profile["anchor"]["binary_sha256"],"backend":"mkl"},
                        "cpu_affinity":[0],"protocol_sha256":digest},"cases":[]}
            for index,name in enumerate(NAMES):
                blocks=[]
                for b in range(7):
                    order="anchor,rust,torch" if (1+index+b)%2==0 else "rust,anchor,torch"
                    blocks.append({"block":b,"order":order,**{key:dict(profile["cases"][name]) for key in ["rust","anchor","torch"]}})
                report["cases"].append({"case":{"name":name},"blocks":blocks})
            self.assertEqual(verify_report_profile(report,path),profile)
            archive=path.parent/"source.tar.gz"
            content=b"archived source fixture\n"
            with tarfile.open(archive,"w:gz") as bundle:
                member=tarfile.TarInfo("source.rs"); member.size=len(content)
                bundle.addfile(member,io.BytesIO(content))
            report["environment"].update(source_sha256={"source.rs":hashlib.sha256(content).hexdigest()},
                source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            path.write_text(json.dumps(report))
            restored=path.parent/"restored"
            self.assertEqual(verify(path,restore=restored),1)
            self.assertEqual((restored/"benchmarks/protocols/intel-long-window-v1.json").read_bytes(),PROFILE_PATH.read_bytes())
            report["cases"][-1]["blocks"][0]["torch"]["iterations"]=50
            with self.assertRaises(ValueError): verify_report_profile(report,path)
            report["cases"][-1]["blocks"][0]["torch"]["iterations"]=profile["cases"]["stft-stream"]["iterations"]
            report["cases"][0]["blocks"][0]["order"]="anchor,rust,torch"
            with self.assertRaises(ValueError): verify_report_profile(report,path)


if __name__ == "__main__":
    unittest.main()
