#!/usr/bin/env python3
"""One fresh-process, device-resident GTCRN pipeline measurement for the paired collector."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

try:
    from .gtcrn_reference import ROOT, load_model, store_tensor
    from .check_gru import load
except ImportError:
    from gtcrn_reference import ROOT, load_model, store_tensor
    from check_gru import load


class PreparedInverse:
    """Fixed 512/256 real synthesis, with the exact requested NOLA envelope checked once."""
    def __init__(self, window, frames, length):
        if tuple(window.shape) != (512,) or window.dtype != torch.float32 or frames < 1 or length < 1:
            raise ValueError("invalid prepared GTCRN inverse geometry")
        self.frames, self.length = frames, length
        self.covered = (frames - 1) * 256 + 512
        if 256 + length > self.covered:
            raise ValueError("uncovered inverse tail")
        self.window = window.clone().view(1, 512, 1)
        envelope = F.fold(self.window.square().expand(1, 512, frames),
                          (1, self.covered), (1, 512), stride=(1, 256)).reshape(1, self.covered)
        self.envelope = envelope[:, 256:256 + length].contiguous()
        if not self.window.isfinite().all().item() or self.envelope.abs().min().item() < 1e-11:
            raise ValueError("inverse window fails NOLA")

    def __call__(self, spectrum):
        if spectrum.ndim != 3 or spectrum.dtype != torch.complex64 or tuple(spectrum.shape[1:]) != (257, self.frames):
            raise ValueError("prepared inverse spectrum shape differs")
        columns = torch.fft.irfft(spectrum, n=512, dim=1) * self.window
        covered = F.fold(columns, (1, self.covered), (1, 512), stride=(1, 256)).reshape(spectrum.shape[0], self.covered)
        return covered[:, 256:256 + self.length] / self.envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--mode", choices=["eager", "cuda-graph", "cuda-graph-prepared-inverse"], required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.iterations < 1 or args.warmup < 0:
        parser.error("use a new output directory and valid counts")
    if args.mode != "eager" and args.device != "cuda":
        parser.error("CUDA graph replay needs CUDA")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    reference = json.loads((args.reference / "reference.json").read_text())
    case = next(c for c in reference["cases"] if c["name"] == args.case)
    folder = args.reference / args.case
    x = load(folder, case["tensors"]["input"]).to(args.device)
    window = load(folder, case["tensors"]["window"]).to(args.device)
    begin_setup = time.perf_counter_ns()
    model, _ = load_model(args.device, ROOT / "target/models/gtcrn/model_trained_on_dns3.tar")
    inverse = PreparedInverse(window, case["frames"], case["input_len"]) if args.mode == "cuda-graph-prepared-inverse" else None

    def pipeline():
        spectrum = torch.stft(x, 512, 256, 512, window, center=True,
                              pad_mode="reflect", normalized=False, return_complex=True)
        enhanced = model(torch.view_as_real(spectrum))
        complex_enhanced = torch.view_as_complex(enhanced.contiguous())
        waveform = inverse(complex_enhanced) if inverse is not None else torch.istft(
            complex_enhanced, 512, 256, 512, window, center=True, length=case["input_len"])
        return spectrum, enhanced, waveform

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        result = pipeline()
        sync()
        run = pipeline
        graph = None
        if args.mode != "eager":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    pipeline()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = pipeline()

            def replay():
                graph.replay()
                return captured

            run = replay
        sync()
        setup_us = (time.perf_counter_ns() - begin_setup) / 1000
        for _ in range(args.warmup):
            result = run()
        sync()
        start = end = None
        if args.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            end.record()
            end.synchronize()
        begin = time.perf_counter_ns()
        if start is not None:
            start.record()
        for _ in range(args.iterations):
            result = run()
        if end is not None:
            end.record()
            end.synchronize()
        wall_us = (time.perf_counter_ns() - begin) / 1000 / args.iterations
        device_us = start.elapsed_time(end) * 1000 / args.iterations if start is not None else None
        spectrum, enhanced, waveform = result
        args.output_dir.mkdir(parents=True)
        tensors = {"spectrum": store_tensor(args.output_dir, "spectrum", torch.view_as_real(spectrum).transpose(1, 2)),
                   "enhanced": store_tensor(args.output_dir, "enhanced", enhanced.transpose(1, 2)),
                   "waveform": store_tensor(args.output_dir, "waveform", waveform)}
        record = {"case": args.case, **tensors, "setup_us": setup_us,
                  "timing": {"pipeline": {"wall_us": [wall_us], "device_us": [device_us]}},
                  "allocator": {"peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None}}
        report = {"schema_version": 1, "backend": "torch", "mode": args.mode,
                  "protocol": {"warmup": args.warmup, "iterations": args.iterations, "samples": 1,
                               "traces": False, "pipeline_only": True},
                  "environment": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                                  "torch_cudnn": torch.backends.cudnn.version(), "torch_config": torch.__config__.show(),
                                  "inverse_policy": "prechecked irfft/fold" if inverse is not None else "torch.istft"},
                  "cases": [record]}
        (args.output_dir / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.case, args.mode, round(wall_us, 3), flush=True)


if __name__ == "__main__":
    main()
