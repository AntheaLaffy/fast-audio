#!/usr/bin/env python3
"""Check complete native FP32 matrix/VJP exports against a separate Torch process."""
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
MAX_FILE_BYTES = 128 * 1024 * 1024
SHAPES = [
    [1, 1, 1], [1, 7, 3], [5, 1, 9], [3, 4, 1], [2, 3, 4],
    [3, 5, 7], [5, 17, 16], [17, 3, 33], [9, 5, 65], [3, 7, 127],
    [2, 5, 128], [3, 2, 129], [7, 3, 257], [2, 1, 257],
    [0, 3, 4], [3, 0, 4], [3, 4, 0], [0, 3, 0], [3, 0, 0], [0, 0, 3],
    [3, 5, 15], [3, 5, 17], [3, 5, 31], [3, 5, 32], [3, 5, 63], [3, 5, 64],
    [3, 5, 255], [3, 5, 256], [33, 65, 17], [129, 2, 3], [2, 129, 3],
]
CASE_COUNT = 389
FIELDS = {"a", "b", "upstream", "output", "left_gradient", "right_gradient",
          "left_seed", "right_seed", "left_accumulate", "right_accumulate"}
KEYS = FIELDS | {"schema_version", "kind", "backend", "case_index", "binding", "dims"}
TRAIN_X = [0.2, -0.3, 0.4, 0.1, -0.2, 0.5]
TRAIN_TARGET = [0.05, -0.1, 0.2, 0.15, -0.05, 0.25]
TRAIN_INITIAL = [0.1, -0.2, 0.3, 0.4]
TRAIN_KEYS = {"schema_version", "kind", "backend", "step", "steps", "dims", "x", "target",
              "weights", "rate", "loss", "gradient", "updated_weights"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def decode(values, count, name):
    require(isinstance(values, list) and len(values) == count, name + " extent differs")
    require(all(type(v) is int and 0 <= v <= 0xffffffff for v in values), name + " invalid FP32 bits")
    result = struct.unpack("<" + str(count) + "f", struct.pack("<" + str(count) + "I", *values))
    require(all(math.isfinite(v) for v in result), name + " requires finite validation values")
    return result


def validate_record(record):
    require(isinstance(record, dict) and set(record) == KEYS, "matrix record schema differs")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1 and record["kind"] == "matrix", "record version/kind differs")
    index, binding = record["case_index"], record["binding"]
    require(type(index) is int and 0 <= index < CASE_COUNT, "case index differs")
    require(type(binding) is int and binding in (0, 1), "payload binding differs")
    require(record["backend"] in ("reference", "mkl", "cuda"), "backend differs")
    expected = (SHAPES[index // 12] if index < 372 else
                [[3, 5, 7], [5, 17, 33]][(index - 372) // 8] if index < 388 else [33, 65, 129])
    dims = record["dims"]
    require(isinstance(dims, list) and all(type(d) is int for d in dims) and dims == expected,
            "matrix shape/case catalogue differs")
    m, n, k = dims
    lengths = {"a": m*k, "b": k*n, "upstream": m*n, "output": m*n,
               "left_gradient": m*k, "right_gradient": k*n,
               "left_seed": m*k, "right_seed": k*n,
               "left_accumulate": m*k, "right_accumulate": k*n}
    return {name: decode(record[name], length, name) for name, length in lengths.items()}


def compare(values, expected, label):
    expected = expected.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    require(expected.numel() == len(values), label + " reference extent differs")
    require(bool(torch.isfinite(expected).all()), label + " reference is non-finite")
    if not values:
        return 0.0
    actual = torch.tensor(values, dtype=torch.float64)
    delta = (actual - expected).abs()
    require(bool(torch.all(delta <= ATOL + RTOL * expected.abs())), label + " numerical mismatch")
    return float(delta.max())


def verify_matrix(record):
    values = validate_record(record)
    m, n, k = record["dims"]
    device = "cuda" if record["backend"] == "cuda" else "cpu"
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA reference unavailable; no host fallback")
    a = torch.tensor(values["a"], dtype=torch.float32, device=device).reshape(m, k).requires_grad_()
    b = torch.tensor(values["b"], dtype=torch.float32, device=device).reshape(k, n).requires_grad_()
    dy = torch.tensor(values["upstream"], dtype=torch.float32, device=device).reshape(m, n)
    y = a @ b
    y.backward(dy, retain_graph=True)
    require(a.grad is not None and b.grad is not None, "reference derivative missing")
    errors = {"output": compare(values["output"], y, "output"),
              "left_gradient": compare(values["left_gradient"], a.grad, "left gradient"),
              "right_gradient": compare(values["right_gradient"], b.grad, "right gradient")}
    a.grad = torch.tensor(values["left_seed"], dtype=torch.float32, device=device).reshape(m, k)
    b.grad = torch.tensor(values["right_seed"], dtype=torch.float32, device=device).reshape(k, n)
    y.backward(dy)
    errors["left_accumulate"] = compare(values["left_accumulate"], a.grad, "left accumulate")
    errors["right_accumulate"] = compare(values["right_accumulate"], b.grad, "right accumulate")
    for extent, name in [(k, "output"), (n, "left_gradient"), (m, "right_gradient")]:
        if extent == 0:
            require(all(bits == 0 for bits in record[name]), "zero-contraction overwrite bits differ")
    return errors


def verify_records(records, backends):
    require(isinstance(records, list), "records must be a complete list")
    keys = set()
    for record in records:
        validate_record(record)
        key = (record["backend"], record["case_index"], record["binding"])
        require(key not in keys, "duplicate matrix record")
        require(key[0] in backends, "unexpected backend")
        keys.add(key)
    required = {(backend, index, binding) for backend in backends for index in range(CASE_COUNT) for binding in (0, 1)}
    require(keys == required, "missing/extra matrix operation/layout/binding records")
    maxima = {}
    for record in records:
        for name, value in verify_matrix(record).items():
            maxima[name] = max(maxima.get(name, 0.0), value)
    return {"records": len(records), "backends": sorted(backends), "field_max_abs_error": maxima}


def encode(values):
    return list(struct.unpack("<" + str(len(values)) + "I", struct.pack("<" + str(len(values)) + "f", *values)))


def validate_training_record(record):
    require(isinstance(record, dict) and set(record) == TRAIN_KEYS, "training record schema differs")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1 and record["kind"] == "sgd", "training version/kind differs")
    require(type(record["step"]) is int and 0 <= record["step"] < 8 and type(record["steps"]) is int and record["steps"] == 8,
            "training step extent differs")
    require(record["backend"] in ("reference", "mkl", "cuda"), "training backend differs")
    require(record["dims"] == [3, 2, 2] and all(type(d) is int for d in record["dims"]), "training geometry differs")
    lengths = {"x": 6, "target": 6, "weights": 4, "rate": 1, "loss": 1, "gradient": 4, "updated_weights": 4}
    values = {name: decode(record[name], length, name) for name, length in lengths.items()}
    require(record["x"] == encode(TRAIN_X) and record["target"] == encode(TRAIN_TARGET) and record["rate"] == encode([0.05]),
            "training fixture/rate differs")
    if record["step"] == 0:
        require(record["weights"] == encode(TRAIN_INITIAL), "initial training parameters differ")
    return values


def verify_training_records(records, backends):
    require(isinstance(records, list), "training records must be a complete list")
    groups = {backend: [] for backend in backends}
    for record in records:
        validate_training_record(record)
        require(record["backend"] in groups, "unexpected training backend")
        groups[record["backend"]].append(record)
    maxima = {}
    for backend, rows in groups.items():
        require([row["step"] for row in rows] == list(range(8)), "training step order/completeness differs")
        device = "cuda" if backend == "cuda" else "cpu"
        if device == "cuda":
            require(torch.cuda.is_available(), "CUDA training reference unavailable; no host fallback")
        initial = validate_training_record(rows[0])
        x = torch.tensor(initial["x"], dtype=torch.float32, device=device).reshape(3, 2)
        target = torch.tensor(initial["target"], dtype=torch.float32, device=device).reshape(3, 2)
        rate = torch.tensor(initial["rate"][0], dtype=torch.float32, device=device)
        weights = torch.tensor(initial["weights"], dtype=torch.float32, device=device).reshape(2, 2).requires_grad_()
        for row in rows:
            values = validate_training_record(row)
            errors = {"weights": compare(values["weights"], weights, "pre-update weights")}
            loss = ((x @ weights - target).square()).mean()
            loss.backward()
            require(weights.grad is not None, "training parameter gradient missing")
            errors["loss"] = compare(values["loss"], loss, "training loss")
            errors["gradient"] = compare(values["gradient"], weights.grad, "training gradient")
            with torch.no_grad():
                updated = weights - rate * weights.grad
            errors["updated_weights"] = compare(values["updated_weights"], updated, "updated weights")
            weights = updated.detach().requires_grad_()
            for name, value in errors.items():
                maxima[name] = max(maxima.get(name, 0.0), value)
    return {"records": len(records), "backends": sorted(backends), "steps_per_backend": 8,
            "field_max_abs_error": maxima}


def export_bytes(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
    else:
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
    require(len(data) <= MAX_FILE_BYTES, "export exceeds fixed verification size budget")
    return data


def verify_exports(directory, platform):
    configure()
    cpu = {"reference", "mkl"} if platform == "intel" else {"reference"}
    roles = {"matmul-cpu": (cpu, verify_records), "training-cpu": (cpu, verify_training_records)}
    if platform == "nvidia":
        roles["matmul-cuda"] = ({"cuda"}, verify_records)
        roles["training-cuda"] = ({"cuda"}, verify_training_records)
    files = []
    for profile in ("debug", "release"):
        for stem, (backends, verify) in roles.items():
            plain = directory / ("exports-" + profile) / (stem + ".jsonl")
            compressed = plain.with_suffix(plain.suffix + ".gz")
            require(plain.exists() != compressed.exists(), "missing/ambiguous export: " + str(plain))
            path = plain if plain.exists() else compressed
            data = export_bytes(path)
            records = [json.loads(line) for line in data.splitlines()]
            result = verify(records, backends)
            files.append({"path": str(path.relative_to(directory)), "raw_sha256": hashlib.sha256(data).hexdigest(),
                          "profile": profile, **result})
    return {"status": "verified_prepared_matmul_torch_comparison", "platform": platform,
            "atol": ATOL, "rtol": RTOL, "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch": torch.__version__, "python": sys.executable, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=["portable", "intel", "nvidia"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve previous comparison; use a new output path")
    report = verify_exports(args.directory.absolute(), args.platform)
    require(report["matmul_precision"] == "highest" and report["allow_tf32"] is False, "reference precision differs")
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], sum(f["records"] for f in report["files"]), "matrix/VJP/manual-SGD records")


if __name__ == "__main__":
    main()
