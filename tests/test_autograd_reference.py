import copy
import math
import unittest

from tools import check_autograd as checker


def rounded(values):
    return list(checker.decode(checker.bits(values), len(values), "test fixture"))


def tensor(shape, values):
    return {"shape": shape, "bits": checker.bits(values)}


def composition():
    # Scalar-loop Add -> Square -> Sum; both broadcast derivatives and accumulation.
    left, right = rounded([-0.4, 0.9]), rounded([0.5, 1.3, 2.1])
    output = sum((a + b)**2 for a in left for b in right)
    gradients = [[sum(2 * (a + b) for b in right) for a in left],
                 [sum(2 * (a + b) for a in left) for b in right]]
    return {"schema_version": 1, "kind": "composition", "backend": "reference", "case_index": 12, "binding": 0,
            "inputs": [tensor([2], left), tensor([3], right)], "seed": tensor([], [1.]), "output": tensor([], [output]),
            "gradients": [tensor(s, v) for s, v in zip([[2], [3]], gradients)],
            "accumulated": [tensor(s, [2*x for x in v]) for s, v in zip([[2], [3]], gradients)]}


def dense_records():
    x, target, weights = map(rounded, [checker.TRAIN_X, checker.TRAIN_TARGET, checker.TRAIN_INITIAL])
    rate = rounded([0.05])[0]
    rows = []
    for step in range(8):
        residual = [sum(x[i*2+k]*weights[k*2+j] for k in range(2))-target[i*2+j] for i in range(3) for j in range(2)]
        gradient = [sum(2*x[i*2+k]*residual[i*2+j]/6 for i in range(3)) for k in range(2) for j in range(2)]
        updated = rounded([w-rate*g for w, g in zip(weights, gradient)])
        rows.append({"schema_version": 1, "kind": "dense_sgd", "backend": "reference", "step": step, "steps": 8,
                     "input": tensor([3, 2], x), "target": tensor([3, 2], target), "parameters": [tensor([2, 2], weights)],
                     "rate": checker.bits([rate])[0], "loss": tensor([], [sum(r*r for r in residual)/6]),
                     "gradients": [tensor([2, 2], gradient)], "updated": [tensor([2, 2], updated)]})
        weights = updated
    return rows


def branches():
    gradient = [8.2, 28., 26.4]
    accumulated = [2*g for g in gradient]
    return {"schema_version": 1, "kind": "branches", "backend": "reference", "input": tensor([3], [1., 2., 3.]),
            "outputs": [tensor([], [v]) for v in [18., 18., 14., 14., 18.]] + [tensor([2, 2], [1., 4., 4., 9.])],
            "gradient": tensor([3], gradient), "accumulated": tensor([3], accumulated),
            "updated": tensor([3], [w-rounded([0.01])[0]*g for w, g in zip([1., 2., 3.], accumulated)]),
            "rate": checker.bits([0.01])[0]}


def audio_records():
    # Independent scalar physical-coordinate equations, without Torch/autograd.
    rows = []
    for name in ("waveform", "spectrum"):
        xs, ts, shapes, x, target, parameters = checker.audio_catalog(name)
        parameters = [rounded(v) for v in parameters]
        rate = rounded([0.05])[0]
        for step in range(8):
            prediction = [0.] * math.prod(ts)
            gradients = [[0.]*len(v) for v in parameters]
            for batch in range(2):
                for time in range(17 if name == "waveform" else 5):
                    for channel in range(2):
                        if name == "waveform":
                            v = x[(batch*2+channel)*17+16-time]
                            y = math.tanh(v*parameters[0][channel]+parameters[1][channel])
                            i = (batch*17+time)*2+channel
                            prediction[i] = y
                            dy = 2*(y-target[i])/len(prediction)*(1-y*y)
                            gradients[0][channel] += dy*v
                            gradients[1][channel] += dy
                        else:
                            for frequency in range(3):
                                offset = (((batch*2+channel)*3+frequency)*5+4-time)*2
                                coeff = channel*3+frequency
                                z = [x[offset+r]*parameters[0][coeff]+parameters[1][r] for r in range(2)]
                                y = sum(v*v for v in z)
                                i = ((batch*5+time)*2+channel)*3+frequency
                                prediction[i] = y
                                for r in range(2):
                                    dz = 4*(y-target[i])*z[r]/len(prediction)
                                    gradients[0][coeff] += dz*x[offset+r]
                                    gradients[1][r] += dz
            loss = sum((a-b)**2 for a, b in zip(prediction, target))/len(prediction)
            updated = [rounded([p-rate*g for p, g in zip(ps, gs)]) for ps, gs in zip(parameters, gradients)]
            rows.append({"schema_version": 1, "kind": "audio_sgd", "backend": "reference", "fixture": name,
                         "step": step, "steps": 8, "input": tensor(xs, x), "target": tensor(ts, target),
                         "parameters": [tensor(s, v) for s, v in zip(shapes, parameters)],
                         "rate": checker.bits([rate])[0], "loss": tensor([], [loss]), "prediction": tensor(ts, prediction),
                         "gradients": [tensor(s, v) for s, v in zip(shapes, gradients)],
                         "updated": [tensor(s, v) for s, v in zip(shapes, updated)]})
            parameters = updated
    return rows


class AutogradReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checker.configure()

    def test_real_torch_composition_and_tied_detached_roots(self):
        self.assertTrue(all(v < 2e-5 for v in checker.verify_composition(composition()).values()))
        result = checker.verify_branches([branches()], {"reference"})
        self.assertEqual(result["records"], 1)
        self.assertEqual(set(result["field_max_abs_error"]), {"outputs", "gradient", "accumulated", "updated"})

    def test_independently_evolving_dense_and_audio_trajectories(self):
        self.assertEqual(checker.verify_training(dense_records(), {"reference"})["records"], 8)
        result = checker.verify_training(audio_records(), {"reference"}, True)
        self.assertEqual(result["records"], 16)
        self.assertEqual(set(result["field_max_abs_error"]), {"parameters", "prediction", "loss", "gradients", "updated"})

    def test_corrupted_outputs_and_both_gradient_modes_are_rejected(self):
        for field in ("output", "gradients", "accumulated"):
            row = composition()
            value = row[field] if field == "output" else row[field][1]
            value["bits"][0] = checker.bits([99.])[0]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "numerical mismatch"):
                checker.verify_composition(row)
        for field in ("outputs", "gradient", "accumulated", "updated"):
            row = branches()
            value = row[field][0] if field == "outputs" else row[field]
            value["bits"][0] = checker.bits([99.])[0]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "numerical mismatch"):
                checker.verify_branch(row)

    def test_trajectory_corruption_cannot_rebase_the_reference(self):
        for audio in (False, True):
            for field in ("parameters", "gradients", "updated", "loss") + (("prediction",) if audio else ()):
                rows = audio_records() if audio else dense_records()
                row = rows[-1]
                value = row[field][0] if field in ("parameters", "gradients", "updated") else row[field]
                value["bits"][0] = checker.bits([99.])[0]
                with self.subTest(audio=audio, field=field), self.assertRaisesRegex(ValueError, "numerical mismatch"):
                    checker.verify_training(rows, {"reference"}, audio)

    def test_schema_shape_source_seed_and_bit_corruption_are_rejected(self):
        edits = [("case_index", True), ("binding", True), ("schema_version", True), ("unexpected", 1)]
        for field, value in edits:
            row = composition(); row[field] = value
            with self.assertRaises(ValueError):
                checker.validate_composition(row)
        for field in ("inputs", "seed", "gradients"):
            for corruption in (True, 0x7fc00000, 0xffffffff, -1, checker.bits([99.])[0]):
                row = composition()
                value = row[field][0] if field in ("inputs", "gradients") else row[field]
                value["bits"][0] = corruption
                if field == "gradients" and corruption == checker.bits([99.])[0]:
                    continue  # Finite output corruption belongs to the numerical gate.
                with self.assertRaises(ValueError):
                    checker.validate_composition(row)
        row = composition(); row["output"]["shape"] = [1]
        with self.assertRaisesRegex(ValueError, "shape"):
            checker.validate_composition(row)

    def test_complete_scope_is_required_before_numerical_work(self):
        row = composition()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checker.verify_compositions([row, copy.deepcopy(row)], {"reference"})
        with self.assertRaisesRegex(ValueError, "missing/extra"):
            checker.verify_compositions([row], {"reference"})
        with self.assertRaisesRegex(ValueError, "unexpected backend"):
            checker.verify_compositions([row], {"mkl"})
        with self.assertRaisesRegex(ValueError, "complete list"):
            checker.verify_compositions(iter([row]), {"reference"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checker.verify_training(dense_records()*2, {"reference"})
        with self.assertRaisesRegex(ValueError, "missing/extra"):
            checker.verify_training(audio_records()[:8], {"reference"}, True)
        with self.assertRaisesRegex(ValueError, "order/completeness"):
            checker.verify_training(dense_records()[::-1], {"reference"})
        with self.assertRaisesRegex(ValueError, "branch backend"):
            checker.verify_branches([branches()]*2, {"reference"})


if __name__ == "__main__":
    unittest.main()
