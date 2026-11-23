#!/usr/bin/env python3
"""Resident eager, full-graph differentiated and complete CUDA-graph training steps."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.training_protocol import load_protocol, fixtures, objective, payload, tensor_bytes
from tools.pointwise_torch_worker import configure_compiler


def compiled_step(case, rate):
    count = 1 if case["family"] == "dense" else 2
    derivative = torch.func.grad_and_value(lambda x, target, *ps: objective(case, x, target, *ps),
                                         argnums=tuple(range(2, 2 + count)), has_aux=True)

    def step(x, target, *parameters):
        gradients, (loss, prediction) = derivative(x, target, *parameters)
        with torch.no_grad():
            for p, g in zip(parameters, gradients):
                p.sub_(g * rate)
        return loss, prediction, gradients
    return step


class State:
    def __init__(self, case, device, rate, compiled=None):
        self.case, self.device, self.rate, self.compiled = case, device, rate, compiled
        self.x, self.target, initial = fixtures(case)
        self.initial = initial
        self.x, self.target = self.x.to(device), self.target.to(device)
        self.parameters = [p.to(device).clone().requires_grad_(compiled is None) for p in initial]
        self.last = None
        self.graph = None

    def reset(self):
        with torch.no_grad():
            for p, initial in zip(self.parameters, self.initial):
                p.copy_(initial)
        self.sync()

    def sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def run_uncaptured(self):
        self.last = None
        if self.compiled is not None:
            self.last = self.compiled(self.x, self.target, *self.parameters)
        else:
            for p in self.parameters:
                p.grad = None
            loss, result = objective(self.case, self.x, self.target, *self.parameters)
            loss.backward()
            gradients = tuple(p.grad for p in self.parameters)
            with torch.no_grad():
                for p, g in zip(self.parameters, gradients):
                    p.sub_(g * self.rate)
            self.last = loss, result, gradients

    def run(self):
        if self.graph is None:
            self.run_uncaptured()
        else:
            self.graph.replay()

    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                self.run_uncaptured()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.run_uncaptured()
        self.graph = graph
        self.sync()

    def dump(self, directory):
        # Identical untimed post-update forward to the native diagnostic boundary.
        with torch.no_grad():
            values = payload(self.case, self.x, self.target, self.parameters, self.last[2])
        self.sync()
        result = {}
        for name, tensor in values.items():
            path = directory / (name + ".f32")
            path.write_bytes(tensor_bytes(tensor))
            result[name] = {"file": str(path), "shape": list(tensor.shape)}
        return result


def respond(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--platform", choices=["intel", "nvidia"], required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    p = load_protocol(args.protocol)
    if args.output_dir.exists():
        parser.error("use a new worker directory")
    args.output_dir = args.output_dir.absolute()
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    compiler = configure_compiler()
    spec = p["strata"][args.platform]
    device = spec["device"]
    if torch.__version__ != ("2.14.0+cu130" if device == "cuda" else "2.14.1+cpu"):
        raise RuntimeError("pinned training Torch version differs")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference unavailable; no host fallback")
    if device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != p["precision"]["cublas_workspace_config"]:
        raise RuntimeError("NVIDIA training requires the declared CUBLAS_WORKSPACE_CONFIG")
    case = next(c for c in p["cases"] if c["name"] == args.case)
    rate = torch.tensor(p["rate"], dtype=torch.float32).item()
    shared_begin = time.perf_counter_ns()
    step = compiled_step(case, rate)
    compiled = torch.compile(step, backend=p["compile"]["backend"], mode=p["compile"]["mode"],
                             fullgraph=True, dynamic=False)
    shared_preparation = (time.perf_counter_ns() - shared_begin) / 1000
    states, readiness = {}, {}
    for mode in spec["modes"][1:]:
        begin = time.perf_counter_ns()
        state = State(case, device, rate, compiled if mode.startswith("compiled") else None)
        if mode.startswith("compiled"):
            state.run_uncaptured()  # Compilation is preparation, with its execution recorded.
            state.sync()
        if mode.endswith("cuda-graph"):
            state.capture()
        preparation = (time.perf_counter_ns() - begin) / 1000
        state.reset()
        begin = time.perf_counter_ns()
        state.run()
        state.sync()
        first = (time.perf_counter_ns() - begin) / 1000
        states[mode] = state
        readiness[mode] = {"preparation_us": preparation, "first_step_us": first,
                           "prepare_includes_execution": mode != "eager", "complete_step": True,
                           "frontend": "torch.func.grad_and_value" if mode.startswith("compiled") else "autograd.backward",
                           "capture": state.graph is not None}
    directory = args.output_dir / "snapshot"
    directory.mkdir()
    respond({"status": "ready", "engine": "torch", "case": case["name"], "modes": readiness,
             "compiler": compiler, "precision": {"matmul": torch.get_float32_matmul_precision(), "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),"fp32_precision":torch.backends.cuda.matmul.fp32_precision},
             "torch": torch.__version__, "torch_config": torch.__config__.show(), "cuda_version": torch.version.cuda,
             "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
             "shared_frontend_preparation_us": shared_preparation,
             "dynamo_counters": {name: dict(values) for name, values in torch._dynamo.utils.counters.items() if name in ("stats", "graph_break")},
             "affinity": sorted(os.sched_getaffinity(0)), "reset_source": "host_initial_fp32",
             "snapshot_semantics": "post-update forward; last pre-update gradient"})
    validation = {m: 0 for m in states}
    blocks = {m: 0 for m in states}
    repeats = {m: None for m in states}
    window = p["windows"][case["regime"]]
    for line in sys.stdin:
        request = json.loads(line)
        expected = {"action"} if request.get("action") == "quit" else {"action", "mode", "step"} if request.get("action") == "validate" else {"action", "mode", "block"}
        if set(request) != expected:
            raise ValueError("training request fields differ")
        if request["action"] == "quit":
            respond({"status": "stopped", "validation_steps": validation, "measured_blocks": blocks, "pending_repeat": repeats})
            return
        mode = request.get("mode")
        if mode not in states:
            raise ValueError("unexpected training mode")
        state = states[mode]
        if request["action"] == "validate" and type(request.get("step")) is int and request["step"] == validation[mode] < p["validation_steps"] and blocks[mode] == 0:
            if request["step"] == 0:
                state.reset()
            state.run()
            state.sync()
            respond({"status": "validated", "step": request["step"], "snapshot": state.dump(directory)})
            validation[mode] += 1
        elif request["action"] == "measure" and validation[mode] == p["validation_steps"] and repeats[mode] is None and type(request.get("block")) is int and request["block"] == blocks[mode] < spec["blocks"]:
            state.reset()
            for _ in range(window["warmup"]):
                state.run()
            state.sync()
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter_ns()
                start.record()
                for _ in range(window["iterations"]):
                    state.run()
                end.record()
                end.synchronize()
                wall = (time.perf_counter_ns() - begin) / 1000 / window["iterations"]
                stream = start.elapsed_time(end) * 1000 / window["iterations"]
            else:
                begin = time.perf_counter_ns()
                for _ in range(window["iterations"]):
                    state.run()
                wall = (time.perf_counter_ns() - begin) / 1000 / window["iterations"]
                stream = None
            respond({"status": "measured", "block": request["block"], **window,
                     "timing": {"wall_us": wall, "device_us": stream}, "snapshot": state.dump(directory)})
            repeats[mode] = request["block"]
            blocks[mode] += 1
        elif request["action"] == "repeat" and repeats[mode] is not None and type(request.get("block")) is int and request["block"] == repeats[mode]:
            state.reset()
            for _ in range(window["warmup"] + window["iterations"]):
                state.run()
            state.sync()
            respond({"status": "repeated", "block": request["block"], "snapshot": state.dump(directory)})
            repeats[mode] = None
        else:
            raise ValueError("unexpected training action/order or declared budget exceeded")
    raise RuntimeError("training worker input closed without quit")


if __name__ == "__main__":
    main()
