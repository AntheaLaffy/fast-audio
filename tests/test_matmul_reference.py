import copy
import struct
import unittest

from tools import check_matmul as checker


def bits(values):
    return [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]


def record():
    return {"schema_version": 1, "kind": "matrix", "backend": "reference",
            "case_index": 0, "binding": 0, "dims": [1, 1, 1],
            "a": bits([2.]), "b": bits([3.]), "upstream": bits([4.]), "output": bits([6.]),
            "left_gradient": bits([12.]), "right_gradient": bits([8.]),
            "left_seed": bits([1.]), "right_seed": bits([1.]),
            "left_accumulate": bits([13.]), "right_accumulate": bits([9.])}


def training_records():
    # Analytic scalar-loop control, independent of Torch autograd and native kernels.
    rounded = lambda values: [struct.unpack("<f", struct.pack("<f", v))[0] for v in values]
    x, target, weights = map(rounded, [checker.TRAIN_X, checker.TRAIN_TARGET, checker.TRAIN_INITIAL])
    rate = rounded([0.05])[0]
    rows = []
    for step in range(8):
        residual = [sum(x[i*2+p]*weights[p*2+j] for p in range(2)) - target[i*2+j] for i in range(3) for j in range(2)]
        gradient = [sum(x[i*2+p]*2*residual[i*2+j]/6 for i in range(3)) for p in range(2) for j in range(2)]
        updated = rounded([w-rate*g for w, g in zip(weights, gradient)])
        rows.append({"schema_version": 1, "kind": "sgd", "backend": "reference", "step": step, "steps": 8, "dims": [3, 2, 2],
                     "x": bits(x), "target": bits(target), "weights": bits(weights), "rate": bits([rate]),
                     "loss": bits([sum(v*v for v in residual)/6]), "gradient": bits(gradient), "updated_weights": bits(updated)})
        weights = updated
    return rows


class MatmulReferenceTests(unittest.TestCase):
    def test_real_autograd_checks_both_operands_and_accumulation(self):
        result = checker.verify_matrix(record())
        self.assertEqual(set(result), {"output", "left_gradient", "right_gradient", "left_accumulate", "right_accumulate"})
        self.assertTrue(all(v == 0 for v in result.values()))
        training = checker.verify_training_records(training_records(), {"reference"})
        self.assertEqual(training["records"], 8)

    def test_corrupted_output_or_accumulated_gradient_is_rejected(self):
        for field in ("output", "left_gradient", "right_gradient", "left_accumulate", "right_accumulate"):
            with self.subTest(field=field):
                item = record()
                item[field] = bits([99.])
                with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                    checker.verify_matrix(item)
        trajectory = training_records()
        trajectory[-1]["weights"] = bits([99.] * 4)
        with self.assertRaisesRegex(ValueError, "pre-update weights numerical mismatch"):
            checker.verify_training_records(trajectory, {"reference"})

    def test_schema_shape_bits_and_nonfinite_values_are_rejected(self):
        bad = []
        item = record(); item["dims"] = [1, 1, 2]; bad.append(item)
        item = record(); item["a"] = [True]; bad.append(item)
        item = record(); item["a"] = [0x7fc00000]; bad.append(item)
        item = record(); item["binding"] = True; bad.append(item)
        item = record(); item["schema_version"] = True; bad.append(item)
        item = record(); item["unexpected"] = 1; bad.append(item)
        for item in bad:
            with self.assertRaises(ValueError):
                checker.validate_record(item)

    def test_duplicate_or_partial_scope_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checker.verify_records([record(), copy.deepcopy(record())], {"reference"})
        with self.assertRaisesRegex(ValueError, "missing/extra"):
            checker.verify_records([record()], {"reference"})
        with self.assertRaisesRegex(ValueError, "unexpected backend"):
            checker.verify_records([record()], {"mkl"})
        with self.assertRaisesRegex(ValueError, "step order/completeness"):
            checker.verify_training_records(training_records()[::-1], {"reference"})

    def test_partial_json_array_and_wrong_lengths_are_rejected(self):
        item = record(); item["right_seed"] = []
        with self.assertRaisesRegex(ValueError, "extent"):
            checker.validate_record(item)
        with self.assertRaisesRegex(ValueError, "complete list"):
            checker.verify_records(iter([record()]), {"reference"})


if __name__ == "__main__":
    unittest.main()
