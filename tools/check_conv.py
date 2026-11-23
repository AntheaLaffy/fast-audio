#!/usr/bin/env python3
"""Check complete native FP32 1D/2D convolution/VJP and tape SGD exports against PyTorch."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import torch
import torch.nn.functional as F

ATOL = RTOL = 2e-5
MAX_FILE_BYTES = 128 * 1024 * 1024

SPECS_1D = [
    [1, 1, 1, 1, 1, 1, 0, 1, 1],
    [2, 3, 4, 5, 1, 1, 0, 1, 1],
    [2, 3, 4, 7, 3, 1, 1, 1, 1],
    [2, 4, 6, 9, 3, 2, 1, 2, 2],
    [3, 3, 3, 11, 5, 2, 2, 1, 3],
    [1, 2, 17, 16, 3, 1, 1, 1, 1],
    [1, 5, 3, 33, 7, 1, 3, 1, 1],
]
SPECS_2D = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [2, 4, 6, 3, 4, 1, 1, 1, 1, 0, 0, 1, 1, 2],
    [2, 3, 4, 4, 5, 3, 3, 1, 1, 1, 1, 1, 1, 1],
    [2, 4, 6, 5, 6, 2, 3, 2, 1, 1, 0, 1, 2, 2],
    [1, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 3],
    [1, 2, 17, 4, 4, 3, 2, 1, 1, 1, 0, 1, 1, 1],
]
CASE_COUNT_1D = 64
CASE_COUNT_2D = 56
FIELDS = {
    "x", "w", "upstream", "output",
    "left_gradient", "right_gradient",
    "left_seed", "right_seed",
    "left_accumulate", "right_accumulate",
}
KEYS = FIELDS | {"schema_version", "kind", "backend", "case_index", "binding", "dims"}

TRAIN_X1 = [(((i * 7 + 3) % 19) - 9.0) / 16.0 for i in range(20)]
TRAIN_W1 = [(((i * 5 + 1) % 13) - 6.0) / 16.0 for i in range(12)]
TRAIN_B1 = [0.125, -0.1875]
TRAIN_X2 = [(((i * 11 + 2) % 23) - 11.0) / 32.0 for i in range(24)]
TRAIN_W2 = [(((i * 3 + 7) % 17) - 8.0) / 16.0 for i in range(16)]
TRAIN_B2 = [-0.0625, 0.125]
TRAIN_KEYS = {
    "schema_version", "kind", "backend", "step", "steps",
    "w1", "b1", "w2", "b2", "loss",
    "gw1", "gb1", "gw2", "gb2",
    "updated_w1", "updated_b1", "updated_w2", "updated_b2",
}


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


def encode(values):
    return list(struct.unpack("<" + str(len(values)) + "I", struct.pack("<" + str(len(values)) + "f", *values)))


def expected_dims_1d(index):
    if index < 56:
        return SPECS_1D[index // 8]
    return SPECS_1D[(index - 56) // 4]


def expected_dims_2d(index):
    if index < 48:
        return SPECS_2D[index // 8]
    return SPECS_2D[(index - 48) // 4]


def validate_record(record):
    require(isinstance(record, dict) and set(record) == KEYS, "convolution record schema differs")
    require(
        type(record["schema_version"]) is int
        and record["schema_version"] == 1
        and record["kind"] in ("conv1d", "conv2d"),
        "record version/kind differs",
    )
    index, binding = record["case_index"], record["binding"]
    require(type(binding) is int and binding in (0, 1), "payload binding differs")
    require(record["backend"] in ("reference", "mkl", "cuda"), "backend differs")
    dims = record["dims"]
    require(isinstance(dims, list) and all(type(d) is int for d in dims), "invalid dims list")
    if record["kind"] == "conv1d":
        require(type(index) is int and 0 <= index < CASE_COUNT_1D, "conv1d case index differs")
        require(dims == expected_dims_1d(index), "conv1d case catalogue differs")
        n, cin, cout, l, k, s, p, d, g = dims
        out_len = (l + 2 * p - ((k - 1) * d + 1)) // s + 1
        x_len = n * cin * l
        w_len = cout * (cin // g) * k
        y_len = n * cout * out_len
    else:
        require(type(index) is int and 0 <= index < CASE_COUNT_2D, "conv2d case index differs")
        require(dims == expected_dims_2d(index), "conv2d case catalogue differs")
        n, cin, cout, ih, iw, kh, kw, sh, sw, ph, pw, dh, dw, g = dims
        oh = (ih + 2 * ph - ((kh - 1) * dh + 1)) // sh + 1
        ow = (iw + 2 * pw - ((kw - 1) * dw + 1)) // sw + 1
        x_len = n * cin * ih * iw
        w_len = cout * (cin // g) * kh * kw
        y_len = n * cout * oh * ow
    lengths = {
        "x": x_len, "w": w_len, "upstream": y_len, "output": y_len,
        "left_gradient": x_len, "right_gradient": w_len,
        "left_seed": x_len, "right_seed": w_len,
        "left_accumulate": x_len, "right_accumulate": w_len,
    }
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


def verify_conv(record):
    values = validate_record(record)
    device = "cuda" if record["backend"] == "cuda" else "cpu"
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA reference unavailable; no host fallback")
    if record["kind"] == "conv1d":
        n, cin, cout, l, k, s, p, d, g = record["dims"]
        out_len = (l + 2 * p - ((k - 1) * d + 1)) // s + 1
        x = torch.tensor(values["x"], dtype=torch.float32, device=device).reshape(n, cin, l).requires_grad_()
        w = torch.tensor(values["w"], dtype=torch.float32, device=device).reshape(cout, cin // g, k).requires_grad_()
        dy = torch.tensor(values["upstream"], dtype=torch.float32, device=device).reshape(n, cout, out_len)
        y = F.conv1d(x, w, stride=s, padding=p, dilation=d, groups=g)
        left_shape = (n, cin, l)
        right_shape = (cout, cin // g, k)
    else:
        n, cin, cout, ih, iw, kh, kw, sh, sw, ph, pw, dh, dw, g = record["dims"]
        oh = (ih + 2 * ph - ((kh - 1) * dh + 1)) // sh + 1
        ow = (iw + 2 * pw - ((kw - 1) * dw + 1)) // sw + 1
        x = torch.tensor(values["x"], dtype=torch.float32, device=device).reshape(n, cin, ih, iw).requires_grad_()
        w = torch.tensor(values["w"], dtype=torch.float32, device=device).reshape(cout, cin // g, kh, kw).requires_grad_()
        dy = torch.tensor(values["upstream"], dtype=torch.float32, device=device).reshape(n, cout, oh, ow)
        y = F.conv2d(x, w, stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw), groups=g)
        left_shape = (n, cin, ih, iw)
        right_shape = (cout, cin // g, kh, kw)
    y.backward(dy, retain_graph=True)
    require(x.grad is not None and w.grad is not None, "reference derivative missing")
    errors = {
        "output": compare(values["output"], y, "output"),
        "left_gradient": compare(values["left_gradient"], x.grad, "left gradient"),
        "right_gradient": compare(values["right_gradient"], w.grad, "right gradient"),
    }
    x.grad = torch.tensor(values["left_seed"], dtype=torch.float32, device=device).reshape(*left_shape)
    w.grad = torch.tensor(values["right_seed"], dtype=torch.float32, device=device).reshape(*right_shape)
    y.backward(dy)
    errors["left_accumulate"] = compare(values["left_accumulate"], x.grad, "left accumulate")
    errors["right_accumulate"] = compare(values["right_accumulate"], w.grad, "right accumulate")
    return errors


def verify_records(records, backends):
    require(isinstance(records, list), "records must be a complete list")
    keys = set()
    for record in records:
        validate_record(record)
        key = (record["kind"], record["backend"], record["case_index"], record["binding"])
        require(key not in keys, "duplicate convolution record")
        require(key[1] in backends, "unexpected backend")
        keys.add(key)
    required_1d = {
        ("conv1d", backend, index, binding)
        for backend in backends
        for index in range(CASE_COUNT_1D)
        for binding in (0, 1)
    }
    required_2d = {
        ("conv2d", backend, index, binding)
        for backend in backends
        for index in range(CASE_COUNT_2D)
        for binding in (0, 1)
    }
    require(keys == (required_1d | required_2d), "missing/extra convolution operation/layout/binding records")
    maxima = {}
    for record in records:
        for name, value in verify_conv(record).items():
            maxima[name] = max(maxima.get(name, 0.0), value)
    return {"records": len(records), "backends": sorted(backends), "field_max_abs_error": maxima}


def validate_training_record(record):
    require(isinstance(record, dict) and set(record) == TRAIN_KEYS, "training record schema differs")
    require(
        type(record["schema_version"]) is int
        and record["schema_version"] == 1
        and record["kind"] == "sgd",
        "training version/kind differs",
    )
    require(
        type(record["step"]) is int
        and 0 <= record["step"] < 8
        and type(record["steps"]) is int
        and record["steps"] == 8,
        "training step extent differs",
    )
    require(record["backend"] in ("reference", "mkl", "cuda"), "training backend differs")
    lengths = {
        "w1": 12, "b1": 2, "w2": 16, "b2": 2, "loss": 1,
        "gw1": 12, "gb1": 2, "gw2": 16, "gb2": 2,
        "updated_w1": 12, "updated_b1": 2, "updated_w2": 16, "updated_b2": 2,
    }
    values = {name: decode(record[name], length, name) for name, length in lengths.items()}
    if record["step"] == 0:
        require(
            record["w1"] == encode(TRAIN_W1)
            and record["b1"] == encode(TRAIN_B1)
            and record["w2"] == encode(TRAIN_W2)
            and record["b2"] == encode(TRAIN_B2),
            "initial training parameters differ",
        )
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
        x1 = torch.tensor(TRAIN_X1, dtype=torch.float32, device=device).reshape(2, 2, 5)
        x2 = torch.tensor(TRAIN_X2, dtype=torch.float32, device=device).reshape(2, 2, 2, 3)
        w1 = torch.tensor(initial["w1"], dtype=torch.float32, device=device).reshape(2, 2, 3).requires_grad_()
        b1 = torch.tensor(initial["b1"], dtype=torch.float32, device=device).reshape(2).requires_grad_()
        w2 = torch.tensor(initial["w2"], dtype=torch.float32, device=device).reshape(2, 2, 2, 2).requires_grad_()
        b2 = torch.tensor(initial["b2"], dtype=torch.float32, device=device).reshape(2).requires_grad_()
        rate = torch.tensor(0.05, dtype=torch.float32, device=device)
        for row in rows:
            values = validate_training_record(row)
            errors = {
                "w1": compare(values["w1"], w1, "pre-update w1"),
                "b1": compare(values["b1"], b1, "pre-update b1"),
                "w2": compare(values["w2"], w2, "pre-update w2"),
                "b2": compare(values["b2"], b2, "pre-update b2"),
            }
            z1 = F.conv1d(x1, w1, bias=b1, stride=1, padding=1, dilation=1, groups=1)
            z2 = F.conv2d(x2, w2, bias=b2, stride=1, padding=0, dilation=1, groups=1)
            loss = z1.square().mean() + z2.square().mean()
            loss.backward()
            errors["loss"] = compare(values["loss"], loss, "training loss")
            errors["gw1"] = compare(values["gw1"], w1.grad, "training gw1")
            errors["gb1"] = compare(values["gb1"], b1.grad, "training gb1")
            errors["gw2"] = compare(values["gw2"], w2.grad, "training gw2")
            errors["gb2"] = compare(values["gb2"], b2.grad, "training gb2")
            with torch.no_grad():
                nw1 = w1 - rate * w1.grad
                nb1 = b1 - rate * b1.grad
                nw2 = w2 - rate * w2.grad
                nb2 = b2 - rate * b2.grad
            errors["updated_w1"] = compare(values["updated_w1"], nw1, "updated w1")
            errors["updated_b1"] = compare(values["updated_b1"], nb1, "updated b1")
            errors["updated_w2"] = compare(values["updated_w2"], nw2, "updated w2")
            errors["updated_b2"] = compare(values["updated_b2"], nb2, "updated b2")
            w1 = nw1.detach().requires_grad_()
            b1 = nb1.detach().requires_grad_()
            w2 = nw2.detach().requires_grad_()
            b2 = nb2.detach().requires_grad_()
            for name, value in errors.items():
                maxima[name] = max(maxima.get(name, 0.0), value)
    return {
        "records": len(records),
        "backends": sorted(backends),
        "steps_per_backend": 8,
        "field_max_abs_error": maxima,
    }


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
    roles = {}
    if platform in ("portable", "intel"):
        roles["conv-cpu"] = (cpu, verify_records)
        roles["training-cpu"] = (cpu, verify_training_records)
    elif platform == "nvidia":
        roles["conv-cuda"] = ({"cuda"}, verify_records)
        roles["training-cuda"] = ({"cuda"}, verify_training_records)
    files = []
    for stem, (backends, verify) in roles.items():
        plain = directory / (stem + ".jsonl")
        compressed = plain.with_suffix(plain.suffix + ".gz")
        require(plain.exists() != compressed.exists(), "missing/ambiguous export: " + str(plain))
        path = plain if plain.exists() else compressed
        data = export_bytes(path)
        records = [json.loads(line) for line in data.splitlines()]
        result = verify(records, backends)
        files.append({
            "path": str(path.relative_to(directory)),
            "raw_sha256": hashlib.sha256(data).hexdigest(),
            **result,
        })
    return {
        "status": "verified_prepared_conv_torch_comparison",
        "platform": platform,
        "atol": ATOL,
        "rtol": RTOL,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch": torch.__version__,
        "python": sys.executable,
        "files": files,
    }


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
    print(report["status"], sum(f["records"] for f in report["files"]), "conv1d/conv2d/VJP/tape-SGD records")


if __name__ == "__main__":
    main()
