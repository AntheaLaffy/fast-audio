"""Reject a requested repeatability mode without actual compatible engine evidence."""
import copy
import unittest

from tools.check_gtcrn import validate_convolution_policy


class ConvolutionPolicyEvidence(unittest.TestCase):
    def record(self):
        return {"backend": "cudnn", "protocol": {"convolution_policy": "require_deterministic"},
                "cases": [{"convolution_plans": [{"step": 2, "engine_index": 1,
                    "workspace_bytes": 4096, "numerical_notes": 0, "deterministic": True,
                    "fused_epilogue": False}]}]}

    def test_actual_engine_required_even_with_correct_policy_label(self):
        for plans in (None, [], [None], [{"deterministic": True}]):
            record = self.record()
            record["cases"][0]["convolution_plans"] = plans
            with self.assertRaises(ValueError):
                validate_convolution_policy(record, "require_deterministic")

    def test_a_single_nondeterministic_engine_rejects_the_case(self):
        record = self.record()
        bad = copy.deepcopy(record["cases"][0]["convolution_plans"][0])
        bad.update(step=3, deterministic=False)
        record["cases"][0]["convolution_plans"].append(bad)
        with self.assertRaises(ValueError):
            validate_convolution_policy(record, "require_deterministic")
        record["protocol"]["convolution_policy"] = "allow_nondeterministic"
        validate_convolution_policy(record, "allow_nondeterministic")

    def test_policy_mismatch_and_malformed_metadata_are_not_accepted(self):
        record = self.record()
        validate_convolution_policy(record, "require_deterministic")
        with self.assertRaises(ValueError):
            validate_convolution_policy(record, "allow_nondeterministic")
        for key, value in [("engine_index", -1), ("deterministic", "true"),
                           ("workspace_bytes", 1 << 30), ("step", True),
                           ("numerical_notes", -1)]:
            bad = self.record()
            bad["cases"][0]["convolution_plans"][0][key] = value
            with self.assertRaises(ValueError):
                validate_convolution_policy(bad, "require_deterministic")


if __name__ == "__main__":
    unittest.main()
