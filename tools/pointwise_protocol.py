"""Fixed pointwise suite, independent full-output checks and descriptive statistics."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/protocols/pointwise-characterization-v2.json"
PROTOCOL_SHA256 = "2193a48de4d53ef062e8be4773b4ddb55e5bc8eed96bfde43f7e4b3365af04d8"
UNARY = ("neg", "abs", "square", "reciprocal", "sqrt", "exp", "log", "relu", "sigmoid", "tanh", "silu", "softplus")
BINARY = ("add", "sub", "mul", "div")
COMPILER_POLICY = {"use_fast_math":False,"cpp_unsafe_math":False,"cpp_contract":"off","cuda_fast_math":False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path=PROTOCOL):
    if sha(path) != PROTOCOL_SHA256:
        raise ValueError("pointwise v2 protocol bytes changed; declare a new context")
    p = json.loads(Path(path).read_text())
    if (p["name"], p["schema_version"], p["atol"], p["rtol"], p["threads"], p["affinity"]) != ("pointwise-characterization-v2", 1, 2e-6, 3e-6, 1, [0]):
        raise ValueError("pointwise numerical/environment contract differs")
    if p["compile"]["writeback"]!="functional-copy_":
        raise ValueError("compiled writeback contract differs")
    names = [c["name"] for c in p["cases"]]
    if len(names) != len(set(names)) or len(names) != 56:
        raise ValueError("pointwise coverage differs")
    for regime in ("tiny", "bulk"):
        if {c["operation"] for c in p["cases"] if c["regime"] == regime} != set(UNARY+BINARY):
            raise ValueError("missing required operation stratum")
    return p


def order(p, platform, case_index, block):
    spec = p["strata"][platform]
    modes = spec["modes"]
    if not 0 <= block < spec["blocks"]:
        raise ValueError("block is outside declared budget")
    start = (block+case_index) % len(modes)
    result = modes[start:]+modes[:start]
    return result[::-1] if block >= len(modes) else result


def extent(geometry):
    dims, strides, offset = geometry["dims"], geometry["strides"], geometry["offset"]
    if len(dims) != len(strides) or offset < 0 or any(d < 1 for d in dims) or any(s < 0 for s in strides):
        raise ValueError("invalid characterization geometry")
    return offset + sum((d-1)*s for d, s in zip(dims, strides)) + 1 + 2


def storage(case, index):
    from .compare_torch import fixture
    x = fixture([extent(case["inputs"][index])], (11, 29)[index], "cpu")
    if index == 1:
        return 0.5+x.abs()
    return 0.125+x.abs() if case["operation"] in ("reciprocal", "sqrt", "log") else 4*x


def view(tensor, geometry):
    return tensor.as_strided(geometry["dims"], geometry["strides"], geometry["offset"])


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()


def reference(case):
    import torch
    inputs = [storage(case, i) for i in range(len(case["inputs"]))]
    values = [view(x, geometry).double() for x, geometry in zip(inputs, case["inputs"])]
    x, op = values[0], case["operation"]
    if op == "neg": result = -x
    elif op == "abs": result = x.abs()
    elif op == "square": result = x*x
    elif op == "reciprocal": result = 1/x
    elif op == "sqrt": result = x.sqrt()
    elif op == "exp": result = x.exp()
    elif op == "log": result = x.log()
    elif op == "relu": result = x.clamp_min(0)
    elif op == "sigmoid": result = 1/(1+(-x).exp())
    elif op == "tanh": result = x.tanh()
    elif op == "silu": result = x/(1+(-x).exp())
    elif op == "softplus": result = torch.logaddexp(x, torch.zeros_like(x))
    elif op == "add": result = x+values[1]
    elif op == "sub": result = x-values[1]
    elif op == "mul": result = x*values[1]
    elif op == "div": result = x/values[1]
    else: raise ValueError("unknown pointwise reference")
    return [hashlib.sha256(tensor_bytes(x)).hexdigest() for x in inputs], result


def check_buffers(case, directory, mode, expected_inputs, expected):
    import torch
    directory = Path(directory)
    inputs = [sha(directory/f"input-{i}.f32") for i in range(len(case["inputs"]))]
    path = directory/f"{mode}-out.f32"
    data = path.read_bytes()
    if len(data) != extent(case["output"])*4:
        raise ValueError("output backing byte count differs")
    backing = torch.frombuffer(bytearray(data), dtype=torch.float32)
    actual = view(backing, case["output"]).double()
    if list(actual.shape) != list(expected.shape):
        raise ValueError("output logical shape differs")
    finite = bool(actual.isfinite().all() and expected.isfinite().all())
    diff = (actual-expected).abs()
    tolerance = 2e-6+3e-6*expected.abs()
    numerical = finite and bool((diff <= tolerance).all())
    mask = torch.zeros(len(backing), dtype=torch.bool)
    view(mask, case["output"]).fill_(True)
    holes = bool((backing.view(torch.int32)[~mask] == 0x7fc12345).all())
    return {"passed": numerical and holes and inputs == expected_inputs,
            "input_sha256": inputs, "expected_input_sha256": expected_inputs,
            "output_sha256": hashlib.sha256(data).hexdigest(), "output_file": str(path),
            "finite": finite, "holes_preserved": holes, "elements": actual.numel(),
            "atol": 2e-6, "rtol": 3e-6,
            "max_abs": diff.max().item() if finite else None,
            "rms": diff.square().mean().sqrt().item() if finite else None,
            "max_tolerance_ratio": (diff/tolerance).max().item() if finite else None}


def summarize(records, modes):
    if not records:
        raise ValueError("cannot summarize empty measurements")
    for record in records:
        if set(record["samples"]) != set(modes):
            raise ValueError("incomplete mode coverage")
        for sample in record["samples"].values():
            if not sample["validation"]["passed"] or not sample["validation"]["repeat_exact"]:
                raise ValueError("cannot summarize failed numerical/repeat gate")
            if not math.isfinite(sample["timing"]["wall_us"]) or sample["timing"]["wall_us"] <= 0:
                raise ValueError("invalid latency")
    result = {}
    for mode in modes:
        wall = [r["samples"][mode]["timing"]["wall_us"] for r in records]
        ratios = [r["samples"][mode]["timing"]["wall_us"]/r["samples"]["native"]["timing"]["wall_us"] for r in records]
        result[mode] = {"wall_us_median": statistics.median(wall), "wall_us_min": min(wall),
                        "wall_us_max": max(wall), "torch_over_native_descriptive_geomean": math.exp(statistics.mean(map(math.log,ratios)))}
        device = [r["samples"][mode]["timing"]["device_us"] for r in records]
        if all(x is not None for x in device):
            result[mode]["device_us_median"] = statistics.median(device)
    return result
