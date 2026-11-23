"""Immutable training workload, independent analytic trajectories and full-buffer gates."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/protocols/training-characterization-v1.json"
PROTOCOL_SHA256 = "ba5cd58d8928da60791321ffaea15ed61ff30320a8024fde17e4539de5d2678d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path=PROTOCOL):
    require(sha(path) == PROTOCOL_SHA256, "training v1 protocol bytes changed; declare a new context")
    return json.loads(Path(path).read_text())


def order(p, platform, index, block):
    spec = p["strata"][platform]
    require(type(block) is int and 0 <= block < spec["blocks"], "training block exceeds declared budget")
    modes = spec["modes"]
    offset = (index + block) % len(modes)
    result = modes[offset:] + modes[:offset]
    return result[::-1] if block >= len(modes) else result


def shapes(case):
    dims, family, contiguous = case["dims"], case["family"], case["layout"] == "contiguous"
    if family == "dense":
        m, k, n = dims
        return [m, k], [m, n], [[k, n]]
    if family == "waveform":
        b, c, t = dims
        return [b, c, t], [b, c, t] if contiguous else [b, t, c], [[c], [c]]
    require(family == "spectrum", "unknown training family")
    b, c, f, t = dims
    return [b, c, f, t, 2], [b, c, f, t] if contiguous else [b, t, c, f], [[c, f], [2]]


def fixtures(case, device="cpu", dtype=None):
    import torch
    if dtype is None:
        dtype = torch.float32
    xs, ts, ps = shapes(case)

    def recipe(shape, multiplier, add, modulus, offset, divisor):
        i = torch.arange(math.prod(shape), dtype=torch.int64, device="cpu")
        return (((i * multiplier + add) % modulus).to(dtype) + offset).div(divisor).reshape(shape).to(device)

    x = recipe(xs, 7, 3, 29, -14, 32)
    target = recipe(ts, 3, 2, 17, 4, 64) if case["family"] == "spectrum" else recipe(ts, 5, 1, 23, -11, 32)
    if case["family"] == "dense":
        parameters = [recipe(ps[0], 13, 5, 31, -15, 256)]
    else:
        parameters = [recipe(ps[0], 1, 0, 7, 16, 32),
                      recipe(ps[1], 3, 2, 7, -3, 64) if case["family"] == "waveform" else
                      torch.tensor([-0.0625, 0.125], dtype=dtype, device=device)]
    return x, target, parameters


def transform(case, prediction, inverse=False):
    import torch
    if case["layout"] == "contiguous":
        return prediction
    if case["family"] == "waveform":
        return torch.flip(prediction.permute(0, 2, 1), [2]) if inverse else torch.flip(prediction, [2]).permute(0, 2, 1)
    return torch.flip(prediction.permute(0, 2, 3, 1), [3]) if inverse else torch.flip(prediction, [3]).permute(0, 3, 1, 2)


def prediction(case, x, parameters):
    if case["family"] == "dense":
        return x @ parameters[0]
    if case["family"] == "waveform":
        result = (x * parameters[0][None, :, None] + parameters[1][None, :, None]).tanh()
    else:
        result = (x * parameters[0][None, :, :, None, None] + parameters[1]).square().sum(-1)
    return transform(case, result)


def objective(case, x, target, *parameters):
    result = prediction(case, x, parameters)
    return (result - target).square().mean(), result


def analytic(case, x, target, parameters):
    """Explicit FP64 physical-parameter equations, without autograd or a native tape."""
    raw_target = transform(case, target, True) if case["family"] != "dense" else target
    n = target.numel()
    if case["family"] == "dense":
        result = x @ parameters[0]
        residual = result - target
        gradients = [x.T @ (2 * residual / n)]
    elif case["family"] == "waveform":
        raw = (x * parameters[0][None, :, None] + parameters[1][None, :, None]).tanh()
        dy = (2 / n) * (raw - raw_target) * (1 - raw.square())
        gradients = [(dy * x).sum((0, 2)), dy.sum((0, 2))]
        result = transform(case, raw)
    else:
        z = x * parameters[0][None, :, :, None, None] + parameters[1]
        raw = z.square().sum(-1)
        dz = (4 / n) * (raw - raw_target).unsqueeze(-1) * z
        gradients = [(dz * x).sum((0, 3, 4)), dz.sum((0, 1, 2, 3))]
        result = transform(case, raw)
    return (result - target).square().mean(), gradients


def payload(case, x, target, parameters, gradients):
    loss, output = objective(case, x, target, *parameters)
    result = {"prediction": output.detach(), "loss": loss.detach(), "owner-0": x, "owner-1": target}
    result.update({"owner-" + str(i + 2): p.detach() for i, p in enumerate(parameters)})
    result.update({"gradient-" + str(i): g.detach() for i, g in enumerate(gradients)})
    return result


def control(case, steps, analytic_fp64, rate=0.01):
    import torch
    require(type(steps) is int and steps > 0, "control needs a positive step count")
    dtype = torch.float64 if analytic_fp64 else torch.float32
    x, target, parameters = fixtures(case, dtype=dtype)
    rate = torch.tensor(rate, dtype=torch.float32).item()
    for p in parameters:
        p.requires_grad_(not analytic_fp64)
    for _ in range(steps):
        if analytic_fp64:
            _, gradients = analytic(case, x, target, parameters)
        else:
            for p in parameters:
                p.grad = None
            loss, _ = objective(case, x, target, *parameters)
            loss.backward()
            gradients = [p.grad for p in parameters]
        with torch.no_grad():
            for p, g in zip(parameters, gradients):
                p.sub_(g * rate)
    return payload(case, x, target, parameters, gradients)


def tensor_bytes(tensor):
    return tensor.detach().to(device="cpu").contiguous().numpy().astype("<f4", copy=False).tobytes()


def sgd_update(previous, gradients, updated, rate):
    """Expose skipped/doubled updates even when an absolute trajectory gate is loose."""
    import torch
    require(len(previous) == len(gradients) == len(updated) and bool(previous), "SGD parameter/gradient count differs")
    rate = torch.tensor(rate, dtype=torch.float32)
    require(bool(torch.isfinite(rate)) and float(rate)>=0, "invalid SGD rate")
    maxima = []
    for before, gradient, actual in zip(previous, gradients, updated):
        require(before.shape == gradient.shape == actual.shape and all(bool(torch.isfinite(v).all()) for v in (before,gradient,actual)), "SGD shape/finite policy differs")
        expected = before.float() - gradient.float() * rate
        def ordered(values):
            words = values.contiguous().view(torch.int32).to(torch.int64)
            return torch.where(words < 0, 0x80000000 - (words & 0x7fffffff), 0x80000000 + words)
        delta = (ordered(expected) - ordered(actual.float())).abs()
        maximum = int(delta.max())
        require(maximum <= 4, "plain SGD update missing, duplicated or inconsistent with exposed gradient")
        maxima.append(maximum)
    return {"max_parameter_ulps": maxima, "allowed_parameter_ulps": 4}


def snapshot_update(snapshot, previous, rate):
    import torch
    gradients, updated = [], []
    for i, before in enumerate(previous):
        shape = list(before.shape)
        for target, key in ((gradients,"gradient-"+str(i)),(updated,"owner-"+str(i+2))):
            item = snapshot[key]
            require(item["shape"] == shape, "SGD parameter shape differs")
            target.append(torch.frombuffer(bytearray(Path(item["file"]).read_bytes()),dtype=torch.float32).reshape(shape))
    return sgd_update(previous,gradients,updated,rate), updated


def check_snapshot(snapshot, directory, controls):
    import torch
    directory = Path(directory).absolute()
    require(isinstance(snapshot, dict) and set(snapshot) == set(controls[0]) == set(controls[1]), "training payload scope differs")
    hashes, errors = {}, {"analytic_fp64": {}, "eager_fp32": {}}
    for name, item in snapshot.items():
        require(isinstance(item, dict) and set(item) == {"file", "shape"}, "training payload descriptor differs")
        path = Path(item["file"]).absolute()
        require(path.parent == directory / "snapshot" and path.name == name + ".f32", "training payload path differs")
        expected_shape = list(controls[0][name].shape)
        require(item["shape"] == expected_shape and all(type(d) is int for d in item["shape"]), "training payload shape differs")
        raw = path.read_bytes()
        require(len(raw) == math.prod(expected_shape) * 4, "training payload length differs")
        actual = torch.frombuffer(bytearray(raw), dtype=torch.float32).double().reshape(expected_shape)
        require(bool(torch.isfinite(actual).all()), "training payload is non-finite")
        if name in ("owner-0", "owner-1"):
            require(raw == tensor_bytes(controls[1][name]), "immutable training source/target changed")
        for label, values in zip(errors, controls):
            expected = values[name].detach().to(device="cpu", dtype=torch.float64)
            delta = (actual - expected).abs()
            require(bool(torch.isfinite(expected).all()) and bool((delta <= 2e-5 + 2e-5 * expected.abs()).all()), label + " training numerical gate failed: " + name)
            errors[label][name] = float(delta.max())
        hashes[name] = {"shape": expected_shape, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return {"payloads": hashes, "field_max_abs_error": errors}


def check_profile_snapshot(snapshot, directory, case):
    """Profiles have longer diagnostic loops; only their finite/source scope is asserted."""
    import torch
    x, target, parameters = fixtures(case)
    expected = {"prediction": list(target.shape), "loss": [], "owner-0": list(x.shape), "owner-1": list(target.shape)}
    expected.update({"owner-" + str(i+2): list(p.shape) for i, p in enumerate(parameters)})
    expected.update({"gradient-" + str(i): list(p.shape) for i, p in enumerate(parameters)})
    require(set(snapshot) == set(expected), "profile payload scope differs")
    result = {}
    for name, shape in expected.items():
        item = snapshot[name]; path = Path(item["file"]).absolute()
        require(set(item) == {"shape", "file"} and item["shape"] == shape and path.parent == Path(directory).absolute() / "snapshot" and path.name == name + ".f32", "profile payload descriptor differs")
        raw = path.read_bytes()
        require(len(raw) == math.prod(shape)*4 and bool(torch.isfinite(torch.frombuffer(bytearray(raw), dtype=torch.float32)).all()), "non-finite/truncated profiled payload")
        if name in ("owner-0", "owner-1"):
            require(raw == tensor_bytes(x if name == "owner-0" else target), "profile source/target changed")
        result[name] = {"shape": shape, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return {"payloads": result, "scope": "finite diagnostic payloads and unchanged sources; trajectory controls apply to unprofiled validation/timing windows"}
