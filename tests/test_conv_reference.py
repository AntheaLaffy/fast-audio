import copy
import struct
import unittest

import torch
import torch.nn.functional as F
from tools import check_conv as checker


def bits(values):
    return [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]


def record_1d():
    return {
        "schema_version": 1,
        "kind": "conv1d",
        "backend": "reference",
        "case_index": 0,
        "binding": 0,
        "dims": [1, 1, 1, 1, 1, 1, 0, 1, 1],
        "x": bits([2.0]),
        "w": bits([3.0]),
        "upstream": bits([4.0]),
        "output": bits([6.0]),
        "left_gradient": bits([12.0]),
        "right_gradient": bits([8.0]),
        "left_seed": bits([1.0]),
        "right_seed": bits([1.0]),
        "left_accumulate": bits([13.0]),
        "right_accumulate": bits([9.0]),
    }


def record_2d():
    return {
        "schema_version": 1,
        "kind": "conv2d",
        "backend": "reference",
        "case_index": 0,
        "binding": 0,
        "dims": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        "x": bits([2.0]),
        "w": bits([3.0]),
        "upstream": bits([4.0]),
        "output": bits([6.0]),
        "left_gradient": bits([12.0]),
        "right_gradient": bits([8.0]),
        "left_seed": bits([1.0]),
        "right_seed": bits([1.0]),
        "left_accumulate": bits([13.0]),
        "right_accumulate": bits([9.0]),
    }


def training_records():
    checker.configure()
    x1 = torch.tensor(checker.TRAIN_X1, dtype=torch.float32).reshape(2, 2, 5)
    x2 = torch.tensor(checker.TRAIN_X2, dtype=torch.float32).reshape(2, 2, 2, 3)
    w1 = torch.tensor(checker.TRAIN_W1, dtype=torch.float32).reshape(2, 2, 3).requires_grad_()
    b1 = torch.tensor(checker.TRAIN_B1, dtype=torch.float32).reshape(2).requires_grad_()
    w2 = torch.tensor(checker.TRAIN_W2, dtype=torch.float32).reshape(2, 2, 2, 2).requires_grad_()
    b2 = torch.tensor(checker.TRAIN_B2, dtype=torch.float32).reshape(2).requires_grad_()
    rate = torch.tensor(0.05, dtype=torch.float32)
    rows = []
    for step in range(8):
        z1 = F.conv1d(x1, w1, bias=b1, stride=1, padding=1, dilation=1, groups=1)
        z2 = F.conv2d(x2, w2, bias=b2, stride=1, padding=0, dilation=1, groups=1)
        loss = z1.square().mean() + z2.square().mean()
        loss.backward()
        with torch.no_grad():
            nw1 = w1 - rate * w1.grad
            nb1 = b1 - rate * b1.grad
            nw2 = w2 - rate * w2.grad
            nb2 = b2 - rate * b2.grad
        rows.append({
            "schema_version": 1,
            "kind": "sgd",
            "backend": "reference",
            "step": step,
            "steps": 8,
            "w1": bits(w1.detach().reshape(-1).tolist()),
            "b1": bits(b1.detach().reshape(-1).tolist()),
            "w2": bits(w2.detach().reshape(-1).tolist()),
            "b2": bits(b2.detach().reshape(-1).tolist()),
            "loss": bits([float(loss.detach())]),
            "gw1": bits(w1.grad.detach().reshape(-1).tolist()),
            "gb1": bits(b1.grad.detach().reshape(-1).tolist()),
            "gw2": bits(w2.grad.detach().reshape(-1).tolist()),
            "gb2": bits(b2.grad.detach().reshape(-1).tolist()),
            "updated_w1": bits(nw1.detach().reshape(-1).tolist()),
            "updated_b1": bits(nb1.detach().reshape(-1).tolist()),
            "updated_w2": bits(nw2.detach().reshape(-1).tolist()),
            "updated_b2": bits(nb2.detach().reshape(-1).tolist()),
        })
        w1 = nw1.detach().requires_grad_()
        b1 = nb1.detach().requires_grad_()
        w2 = nw2.detach().requires_grad_()
        b2 = nb2.detach().requires_grad_()
    return rows


class ConvReferenceTests(unittest.TestCase):
    def test_real_autograd_checks_1d_2d_operands_and_accumulation(self):
        for rec in (record_1d(), record_2d()):
            result = checker.verify_conv(rec)
            self.assertEqual(
                set(result),
                {"output", "left_gradient", "right_gradient", "left_accumulate", "right_accumulate"},
            )
            self.assertTrue(all(v == 0 for v in result.values()))
        training = checker.verify_training_records(training_records(), {"reference"})
        self.assertEqual(training["records"], 8)

    def test_corrupted_output_or_accumulated_gradient_is_rejected(self):
        for rec_fn in (record_1d, record_2d):
            for field in ("output", "left_gradient", "right_gradient", "left_accumulate", "right_accumulate"):
                with self.subTest(kind=rec_fn()["kind"], field=field):
                    item = rec_fn()
                    item[field] = bits([99.0])
                    with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                        checker.verify_conv(item)
        trajectory = training_records()
        trajectory[-1]["w1"] = bits([99.0] * 12)
        with self.assertRaisesRegex(ValueError, "pre-update w1 numerical mismatch"):
            checker.verify_training_records(trajectory, {"reference"})

    def test_schema_shape_bits_and_nonfinite_values_are_rejected(self):
        bad = []
        item = record_1d(); item["dims"] = [1, 1, 1, 2, 1, 1, 0, 1, 1]; bad.append(item)
        item = record_2d(); item["dims"] = [1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]; bad.append(item)
        item = record_1d(); item["x"] = [True]; bad.append(item)
        item = record_1d(); item["x"] = [0x7fc00000]; bad.append(item)
        item = record_1d(); item["binding"] = True; bad.append(item)
        item = record_1d(); item["schema_version"] = True; bad.append(item)
        item = record_1d(); item["unexpected"] = 1; bad.append(item)
        for item in bad:
            with self.assertRaises(ValueError):
                checker.validate_record(item)

    def test_duplicate_or_partial_scope_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checker.verify_records([record_1d(), copy.deepcopy(record_1d())], {"reference"})
        with self.assertRaisesRegex(ValueError, "missing/extra"):
            checker.verify_records([record_1d(), record_2d()], {"reference"})
        with self.assertRaisesRegex(ValueError, "unexpected backend"):
            checker.verify_records([record_1d()], {"mkl"})
        with self.assertRaisesRegex(ValueError, "step order/completeness"):
            checker.verify_training_records(training_records()[::-1], {"reference"})


if __name__ == "__main__":
    unittest.main()
