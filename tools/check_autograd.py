#!/usr/bin/env python3
"""Check complete native tape compositions and evolving SGD with pinned Torch."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import torch

ATOL = RTOL = 2e-5
STEPS = 8
BACKENDS = {"reference", "mkl", "cuda"}
MAX_FILE_BYTES = 16 * 1024 * 1024
BASE = {"schema_version", "kind", "backend"}
COMPOSITION_KEYS = BASE | {"case_index", "binding", "inputs", "seed", "output", "gradients", "accumulated"}
TRAINING_KEYS = BASE | {"step", "steps", "input", "target", "parameters", "rate", "loss", "gradients", "updated"}
AUDIO_KEYS = TRAINING_KEYS | {"fixture", "prediction"}
BRANCH_KEYS = BASE | {"input", "outputs", "gradient", "accumulated", "updated", "rate"}
TRAIN_X = [0.2, -0.3, 0.4, 0.1, -0.2, 0.5]
TRAIN_TARGET = [0.05, -0.1, 0.2, 0.15, -0.05, 0.25]
TRAIN_INITIAL = [0.1, -0.2, 0.3, 0.4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bits(values):
    return list(struct.unpack("<" + str(len(values)) + "I", struct.pack("<" + str(len(values)) + "f", *values)))


def decode(values, count, label):
    require(isinstance(values, list) and len(values) == count, label + " extent differs")
    require(all(type(v) is int and 0 <= v <= 0xffffffff for v in values), label + " invalid FP32 bits")
    result = struct.unpack("<" + str(count) + "f", struct.pack("<" + str(count) + "I", *values))
    require(all(math.isfinite(v) for v in result), label + " requires finite values")
    return result


def tensor(record, shape, label):
    require(isinstance(record, dict) and set(record) == {"shape", "bits"}, label + " tensor schema differs")
    actual = record["shape"]
    require(isinstance(actual, list) and all(type(d) is int for d in actual) and actual == shape,
            label + " shape differs")
    return decode(record["bits"], math.prod(shape), label)


def tensors(records, shapes, label):
    require(isinstance(records, list) and len(records) == len(shapes), label + " tensor count differs")
    return [tensor(record, shape, label) for record, shape in zip(records, shapes)]


def fixed(record, shape, values, label):
    tensor(record, shape, label)
    require(record["bits"] == bits(values), label + " fixture differs")


def base(record, kind, keys):
    require(isinstance(record, dict) and set(record) == keys, kind + " record schema differs")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1 and record["kind"] == kind,
            "record version/kind differs")
    require(record["backend"] in BACKENDS, "backend differs")


def composition_catalog(index, binding):
    unary = index < 12
    left = ([0.4, 0.8, 1.2, 1.6, 2., 2.4] if index in (4, 6) else
            [-0.8, -0.3, 0.2, 0.7, 1.2, 1.7]) if unary else [-0.4, 0.9]
    right = [0.1, 0.2, 0.3] if unary else [0.5, 1.3, 2.1]
    if binding:
        change = decode(bits([0.07]), 1, "fixture change")[0]
        left = [v + change for v in decode(bits(left), len(left), "fixture left")]
        right = [v + change for v in decode(bits(right), len(right), "fixture right")]
    return ([[2, 3], [3]] if unary else [[2], [3]], left, right,
            [3] if unary else [], [0.5, -0.75, 1.25] if unary else [1.])


def validate_composition(record):
    base(record, "composition", COMPOSITION_KEYS)
    index, binding = record["case_index"], record["binding"]
    require(type(index) is int and 0 <= index < 16, "composition case differs")
    require(type(binding) is int and binding in (0, 1), "composition binding differs")
    shapes, left, right, output_shape, seed = composition_catalog(index, binding)
    inputs = tensors(record["inputs"], shapes, "input")
    for item, shape, values in zip(record["inputs"], shapes, [left, right]):
        fixed(item, shape, values, "input")
    fixed(record["seed"], output_shape, seed, "seed")
    return {"inputs": inputs, "seed": tensor(record["seed"], output_shape, "seed"),
            "output": tensor(record["output"], output_shape, "output"),
            "gradients": tensors(record["gradients"], shapes, "gradients"),
            "accumulated": tensors(record["accumulated"], shapes, "accumulated")}


def device_for(backend):
    device = "cuda" if backend == "cuda" else "cpu"
    require(device != "cuda" or torch.cuda.is_available(), "CUDA reference unavailable; no host fallback")
    return device


def to_torch(values, shape, device, gradient=False):
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(shape).requires_grad_(gradient)


def compare(values, expected, label):
    expected = expected.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    require(len(values) == expected.numel() and bool(torch.isfinite(expected).all()), label + " reference differs")
    delta = (torch.tensor(values, dtype=torch.float64) - expected).abs()
    require(bool(torch.all(delta <= ATOL + RTOL * expected.abs())), label + " numerical mismatch")
    return float(delta.max()) if len(values) else 0.


def verify_composition(record):
    values = validate_composition(record)
    index = record["case_index"]
    shapes, _, _, output_shape, _ = composition_catalog(index, record["binding"])
    device = device_for(record["backend"])
    x, right = [to_torch(v, s, device, True) for v, s in zip(values["inputs"], shapes)]
    if index < 12:
        operations = [torch.neg, torch.abs, torch.square, torch.reciprocal, torch.sqrt, torch.exp,
                      torch.log, torch.relu, torch.sigmoid, torch.tanh, torch.nn.functional.silu,
                      torch.nn.functional.softplus]
        y = operations[index](x).transpose(0, 1)
        output = (y * y + right[:, None]).mean(dim=1)
    else:
        y = [torch.add, torch.sub, torch.mul, torch.div][index - 12](x[:, None], right)
        output = y.square().sum()
    seed = to_torch(values["seed"], output_shape, device)
    output.backward(seed, retain_graph=True)
    errors = {"output": compare(values["output"], output, "output")}
    for i, leaf in enumerate([x, right]):
        errors["gradient_" + str(i)] = compare(values["gradients"][i], leaf.grad, "gradient")
    output.backward(seed)
    for i, leaf in enumerate([x, right]):
        errors["accumulated_" + str(i)] = compare(values["accumulated"][i], leaf.grad, "accumulated")
    return errors


def complete_backends(backends):
    require(isinstance(backends, set) and bool(backends) and backends <= BACKENDS, "expected backend scope differs")


def maxima(result, errors):
    for name, value in errors.items():
        result[name] = max(result.get(name, 0.), value)


def verify_compositions(records, backends):
    complete_backends(backends)
    require(isinstance(records, list), "records must be a complete list")
    keys = []
    for record in records:
        validate_composition(record)
        require(record["backend"] in backends, "unexpected backend")
        keys.append((record["backend"], record["case_index"], record["binding"]))
    require(len(keys) == len(set(keys)), "duplicate composition record")
    require(set(keys) == {(b, i, n) for b in backends for i in range(16) for n in (0, 1)},
            "missing/extra composition operation/binding records")
    errors = {}
    for record in records:
        maxima(errors, verify_composition(record))
    return {"records": len(records), "backends": sorted(backends), "field_max_abs_error": errors}


def audio_catalog(name):
    require(name in ("waveform", "spectrum"), "audio fixture differs")
    waveform = name == "waveform"
    input_shape = [2, 2, 17] if waveform else [2, 2, 3, 5, 2]
    target_shape = [2, 17, 2] if waveform else [2, 5, 2, 3]
    parameter_shapes = [[2], [2]] if waveform else [[2, 3], [2]]
    inputs = [((i * 7 + 3) % 29) / 32 - 0.5 for i in range(math.prod(input_shape))]
    target = [((i * 5 + 1) % 23) / 32 - 0.375 if waveform else
              ((i * 3 + 2) % 17) / 64 + 0.0625 for i in range(math.prod(target_shape))]
    parameters = [[0.5 + i / 16 for i in range(math.prod(parameter_shapes[0]))], [-0.0625, 0.125]]
    return input_shape, target_shape, parameter_shapes, inputs, target, parameters


def validate_training(record, audio):
    base(record, "audio_sgd" if audio else "dense_sgd", AUDIO_KEYS if audio else TRAINING_KEYS)
    require(type(record["step"]) is int and 0 <= record["step"] < STEPS and
            type(record["steps"]) is int and record["steps"] == STEPS, "training step differs")
    if audio:
        input_shape, target_shape, shapes, inputs, target, _ = audio_catalog(record["fixture"])
    else:
        input_shape = target_shape = [3, 2]
        shapes, inputs, target = [[2, 2]], TRAIN_X, TRAIN_TARGET
    fixed(record["input"], input_shape, inputs, "input")
    fixed(record["target"], target_shape, target, "target")
    require(type(record["rate"]) is int and record["rate"] == bits([0.05])[0], "SGD rate differs")
    result = {"parameters": tensors(record["parameters"], shapes, "parameters"),
              "gradients": tensors(record["gradients"], shapes, "gradients"),
              "updated": tensors(record["updated"], shapes, "updated"),
              "loss": tensor(record["loss"], [], "loss")}
    if audio:
        result["prediction"] = tensor(record["prediction"], target_shape, "prediction")
    return result


def verify_training(records, backends, audio=False):
    complete_backends(backends)
    require(isinstance(records, list), "records must be a complete list")
    for record in records:
        validate_training(record, audio)
        require(record["backend"] in backends, "unexpected backend")
    fixtures = ["waveform", "spectrum"] if audio else ["dense"]
    required = {(b, f, i) for b in backends for f in fixtures for i in range(STEPS)}
    keys = [(r["backend"], r["fixture"] if audio else "dense", r["step"]) for r in records]
    require(len(keys) == len(set(keys)), "duplicate training record")
    require(set(keys) == required, "missing/extra training fixture/step records")
    errors = {}
    for backend in sorted(backends):
        for name in fixtures:
            rows = [r for r in records if r["backend"] == backend and (not audio or r["fixture"] == name)]
            require([r["step"] for r in rows] == list(range(STEPS)), "training step order/completeness differs")
            if audio:
                xs, ts, shapes, inputs, target, initial = audio_catalog(name)
            else:
                xs = ts = [3, 2]
                shapes, inputs, target, initial = [[2, 2]], TRAIN_X, TRAIN_TARGET, [TRAIN_INITIAL]
            device = device_for(backend)
            x, target = to_torch(inputs, xs, device), to_torch(target, ts, device)
            parameters = [to_torch(v, s, device, True) for v, s in zip(initial, shapes)]
            rate = decode(bits([0.05]), 1, "rate")[0]
            for row in rows:
                values = validate_training(row, audio)
                for actual, p in zip(values["parameters"], parameters):
                    maxima(errors, {"parameters": compare(actual, p, "pre-update parameters")})
                if name == "dense":
                    prediction = x @ parameters[0]
                elif name == "waveform":
                    prediction = torch.flip((x * parameters[0][None, :, None] + parameters[1][None, :, None]).tanh(), [2]).permute(0, 2, 1)
                else:
                    square = (x * parameters[0][None, :, :, None, None] + parameters[1]).square()
                    prediction = torch.flip(square.sum(-1), [3]).permute(0, 3, 1, 2)
                loss = (prediction - target).square().mean()
                loss.backward()
                maxima(errors, {"loss": compare(values["loss"], loss, "loss")})
                if audio:
                    maxima(errors, {"prediction": compare(values["prediction"], prediction, "prediction")})
                for actual, p in zip(values["gradients"], parameters):
                    maxima(errors, {"gradients": compare(actual, p.grad, "gradient")})
                # Independent evolving FP32 trajectory: never rebase to native weights.
                with torch.no_grad():
                    for actual, p in zip(values["updated"], parameters):
                        p.sub_(p.grad * rate)
                        maxima(errors, {"updated": compare(actual, p, "updated parameters")})
                        p.grad = None
    return {"records": len(records), "backends": sorted(backends), "steps_per_fixture": STEPS,
            "fixtures": fixtures, "field_max_abs_error": errors}


def validate_branch(record):
    base(record, "branches", BRANCH_KEYS)
    fixed(record["input"], [3], [1., 2., 3.], "input")
    require(type(record["rate"]) is int and record["rate"] == bits([0.01])[0], "SGD rate differs")
    return {"outputs": tensors(record["outputs"], [[], [], [], [], [], [2, 2]], "outputs"),
            **{key: tensor(record[key], [3], key) for key in ("gradient", "accumulated", "updated")}}


def verify_branch(record):
    values = validate_branch(record)
    device = device_for(record["backend"])
    x = to_torch([1., 2., 3.], [3], device, True)
    view = x.as_strided((2, 2), (1, 1))
    square = view.square()
    total = square.sum()
    roots = [total, total, (x * x).sum(), (x * x.detach()).sum(), (view * view.detach()).sum(), square]
    seeds = [to_torch([v], [], device) for v in [0.5, 1.5, 1., 1., 1.]] + [to_torch([0.1, 0.2, 0.3, 0.4], [2, 2], device)]
    errors = {}
    for actual, expected in zip(values["outputs"], roots):
        maxima(errors, {"outputs": compare(actual, expected, "root output")})
    torch.autograd.backward(roots, seeds, retain_graph=True)
    maxima(errors, {"gradient": compare(values["gradient"], x.grad, "branch gradient")})
    torch.autograd.backward(roots, seeds)
    maxima(errors, {"accumulated": compare(values["accumulated"], x.grad, "branch accumulated")})
    updated = x.detach() - x.grad * decode(bits([0.01]), 1, "rate")[0]
    maxima(errors, {"updated": compare(values["updated"], updated, "branch updated")})
    return errors


def verify_branches(records, backends):
    complete_backends(backends)
    require(isinstance(records, list), "records must be a complete list")
    for record in records:
        validate_branch(record)
    require(len(records) == len(backends) and {r["backend"] for r in records} == backends,
            "missing/duplicate branch backend records")
    errors = {}
    for record in records:
        maxima(errors, verify_branch(record))
    return {"records": len(records), "backends": sorted(backends), "field_max_abs_error": errors}


def configure():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify_exports(directory, platform):
    configure()
    require(torch.__version__ == ("2.14.0+cu130" if platform == "nvidia" else "2.14.1+cpu"), "pinned Torch version differs")
    cpu = {"mkl", "reference"} if platform == "intel" else {"reference"}
    roles = {"composition": verify_compositions, "training": verify_training,
             "branches": verify_branches, "audio": lambda r, b: verify_training(r, b, True)}
    files = []
    for profile in ("debug", "release"):
        for suffix, backends in ([("cpu", cpu), ("cuda", {"cuda"})] if platform == "nvidia" else [("cpu", cpu)]):
            for role, verify in roles.items():
                raw = directory / ("exports-" + profile) / (role + "-" + suffix + ".jsonl")
                packed = raw.with_suffix(".jsonl.gz")
                require(raw.exists() != packed.exists(), "missing/ambiguous export: " + str(raw))
                path = raw if raw.exists() else packed
                require(path.stat().st_size <= MAX_FILE_BYTES, "oversized export")
                with (gzip.open(path, "rb") if path == packed else path.open("rb")) as stream:
                    data = stream.read(MAX_FILE_BYTES + 1)
                require(len(data) <= MAX_FILE_BYTES, "oversized expanded export")
                records = [json.loads(line) for line in data.splitlines()]
                files.append({"path": str(raw.relative_to(directory)), "profile": profile,
                              "raw_sha256": hashlib.sha256(data).hexdigest(), **verify(records, backends)})
    return {"status": "verified_native_tape_torch_comparison", "platform": platform,
            "atol": ATOL, "rtol": RTOL, "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "torch": torch.__version__,
            "python": sys.executable, "files": files, "performance_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--platform", choices=["portable", "intel", "nvidia"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve previous comparison; use a new output path")
    report = verify_exports(args.directory.absolute(), args.platform)
    require(report["matmul_precision"] == "highest" and report["allow_tf32"] is False, "reference precision differs")
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], sum(f["records"] for f in report["files"]), "native tape/SGD records")


if __name__ == "__main__":
    main()
