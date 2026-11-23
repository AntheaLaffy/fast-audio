#!/usr/bin/env python3
"""Head-to-head strict-FP32 benchmark: fast-audio vs PyTorch (Eager, Inductor, CUDA Graph).

Usage:
    python tools/bench_foundation_vs_pytorch.py cpu [--binary PATH] [--out results.json]
    python tools/bench_foundation_vs_pytorch.py cuda [--binary PATH] [--out results.json]
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.training_protocol import load_protocol
from tools.training_torch_worker import configure_compiler, compiled_step, State


def pattern(shape, seed, device="cpu"):
    n = math.prod(shape)
    vals = [(((i * 37 + seed * 19 + 5) % 29) - 14.0) / 32.0 for i in range(n)]
    return torch.tensor(vals, dtype=torch.float32, device=device).reshape(shape)


def time_torch_cpu(fn, warmup=6, iters=32):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    return (time.perf_counter_ns() - t0) / 1000.0 / iters


def time_torch_cuda(fn, warmup=16, iters=64, use_graph=False):
    if use_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        run = g.replay
    else:
        run = fn
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter_ns()
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    wall_us = (time.perf_counter_ns() - t0) / 1000.0 / iters
    device_us = s.elapsed_time(e) * 1000.0 / iters
    return wall_us, device_us


def bench_pytorch_cpu(filter_str=""):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configure_compiler()
    p = load_protocol()
    out = {}
    keep = lambda name: (not filter_str) or (filter_str in name)

    # 1. training-v1 spectrum + waveform (Eager & Inductor)
    for name, iters in [("spectrum-medium", 64), ("spectrum-large", 32), ("wave-multichannel", 32), ("wave-long", 32)]:
        if not keep(name):
            continue
        case = next(c for c in p["cases"] if c["name"] == name)
        st_e = State(case, "cpu", p["rate"], None)
        eager_us = time_torch_cpu(st_e.run, warmup=4, iters=iters)
        step = compiled_step(case, p["rate"])
        comp = torch.compile(step, backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
        st_c = State(case, "cpu", p["rate"], comp)
        st_c.run_uncaptured()
        st_c.reset()
        ind_us = time_torch_cpu(st_c.run, warmup=4, iters=iters)
        out[name] = {"eager_us": eager_us, "inductor_us": ind_us}

    # 2. LayerNorm & RmsNorm fwd+bwd
    for name, kind, shape, iters in [
        ("norm-layernorm-256", "ln", [8, 128, 256], 64),
        ("norm-rmsnorm-512", "rms", [4, 128, 512], 64),
    ]:
        if not keep(name):
            continue
        x = pattern(shape, 1)
        dy = pattern(shape, 2)
        def eager_fn():
            xi = x.detach().requires_grad_(True)
            y = F.layer_norm(xi, (shape[-1],), eps=1e-5) if kind == "ln" else xi * torch.rsqrt(xi.pow(2).mean(-1, keepdim=True) + 1e-5)
            torch.autograd.grad(y, xi, dy)
        def raw_step(xi, dyi):
            def f(z):
                return F.layer_norm(z, (shape[-1],), eps=1e-5) if kind == "ln" else z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5)
            y, vjp = torch.func.vjp(f, xi)
            return y, vjp(dyi)[0]
        comp = torch.compile(raw_step, backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
        comp(x, dy)
        out[name] = {
            "eager_us": time_torch_cpu(eager_fn, 6, iters),
            "inductor_us": time_torch_cpu(lambda: comp(x, dy), 6, iters),
        }

    # 3. Softmax fwd+bwd
    if keep("softmax-256"):
        x = pattern([4, 8, 128, 128], 3)
        dy = pattern([4, 8, 128, 128], 4)
        def eager_fn():
            xi = x.detach().requires_grad_(True)
            y = F.softmax(xi, dim=-1)
            torch.autograd.grad(y, xi, dy)
        def raw_step(xi, dyi):
            y, vjp = torch.func.vjp(lambda z: F.softmax(z, dim=-1), xi)
            return y, vjp(dyi)[0]
        comp = torch.compile(raw_step, backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
        comp(x, dy)
        out["softmax-256"] = {
            "eager_us": time_torch_cpu(eager_fn, 4, 32),
            "inductor_us": time_torch_cpu(lambda: comp(x, dy), 4, 32),
        }

    # 4. BMM linear fwd+bwd
    if keep("bmm-linear"):
        a = pattern([4, 128, 256], 5)
        b = pattern([256, 256], 6)
        dy = pattern([4, 128, 256], 7)
        def eager_fn():
            ai = a.detach().requires_grad_(True)
            bi = b.detach().requires_grad_(True)
            y = torch.matmul(ai, bi)
            torch.autograd.grad(y, (ai, bi), dy)
        def raw_step(ai, bi, dyi):
            y, vjp = torch.func.vjp(torch.matmul, ai, bi)
            return y, vjp(dyi)
        comp = torch.compile(raw_step, backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
        comp(a, b, dy)
        out["bmm-linear"] = {
            "eager_us": time_torch_cpu(eager_fn, 4, 24),
            "inductor_us": time_torch_cpu(lambda: comp(a, b, dy), 4, 24),
        }

    # 5. SDPA causal fwd+bwd
    if keep("sdpa-causal-t128-d64"):
        q = pattern([2, 4, 128, 64], 8)
        k = pattern([2, 4, 128, 64], 9)
        v = pattern([2, 4, 128, 64], 10)
        dy = pattern([2, 4, 128, 64], 11)
        def eager_fn():
            qi = q.detach().requires_grad_(True)
            ki = k.detach().requires_grad_(True)
            vi = v.detach().requires_grad_(True)
            o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
            torch.autograd.grad(o, (qi, ki, vi), dy)
        out["sdpa-causal-t128-d64"] = {"eager_us": time_torch_cpu(eager_fn, 2, 16)}

    # 6. Full Audio Transformer Step + AdamW
    if keep("audio-transformer-step-adamw"):
        bs, t, h, d = 2, 64, 4, 32
        dm = h * d
        x = pattern([bs, t, dm], 1)
        wq = pattern([dm, dm], 2).requires_grad_(True)
        wk = pattern([dm, dm], 3).requires_grad_(True)
        wv = pattern([dm, dm], 4).requires_grad_(True)
        wo = pattern([dm, dm], 5).requires_grad_(True)
        tgt = pattern([bs, t, dm], 6)
        opt = torch.optim.AdamW([wq, wk, wv, wo], lr=1e-3, weight_decay=0.01)
        def eager_fn():
            opt.zero_grad(set_to_none=True)
            xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
            q = (xn @ wq).reshape(bs, t, h, d).permute(0, 2, 1, 3)
            k = (xn @ wk).reshape(bs, t, h, d).permute(0, 2, 1, 3)
            v = (xn @ wv).reshape(bs, t, h, d).permute(0, 2, 1, 3)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_c = F.silu(attn.permute(0, 2, 1, 3)).reshape(bs, t, dm)
            res = attn_c @ wo + x
            loss = (res - tgt).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([wq, wk, wv, wo], 1.0)
            opt.step()
        out["audio-transformer-step-adamw"] = {"eager_us": time_torch_cpu(eager_fn, 2, 16)}

    # 7. Spectral-Vocoder U-Net Step + AdamW
    if keep("spectral-vocoder-step-adamw"):
        bs, samples = 2, 4096
        wav = pattern([bs, samples], 1)
        tgt = pattern([bs, samples], 2)
        w_enc = pattern([16, 257, 3], 3).requires_grad_(True)
        w_dec = pattern([16, 257, 4], 4).requires_grad_(True)
        win = torch.hann_window(512, periodic=True)
        opt = torch.optim.AdamW([w_enc, w_dec], lr=1e-3, weight_decay=0.01)
        def eager_fn():
            opt.zero_grad(set_to_none=True)
            spec = torch.view_as_real(torch.stft(wav, 512, hop_length=128, window=win, center=True, return_complex=True))
            # spec: [bs, 257, frames, 2]
            re = spec[..., 0]
            im = spec[..., 1]
            re_pad = F.pad(re, (2, 0), mode="reflect")
            enc = F.conv1d(re_pad, w_enc, stride=2)
            act = F.silu(enc * torch.rsqrt(enc.pow(2).mean(-1, keepdim=True) + 1e-5))
            dec = F.conv_transpose1d(act, w_dec, stride=2, padding=1)[..., : re.shape[-1]]
            merged = torch.complex(dec, im)
            recon = torch.istft(merged, 512, hop_length=128, window=win, center=True, length=samples)
            loss = (recon - tgt).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w_enc, w_dec], 1.0)
            opt.step()
        out["spectral-vocoder-step-adamw"] = {"eager_us": time_torch_cpu(eager_fn, 2, 16)}

    return out


def bench_pytorch_cuda(filter_str=""):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    out = {}
    keep = lambda name: (not filter_str) or (filter_str in name)

    # 1. LayerNorm & RmsNorm
    for name, kind, shape in [
        ("norm-layernorm-256", "ln", [16, 256, 256]),
        ("norm-rmsnorm-512", "rms", [16, 256, 512]),
    ]:
        if not keep(name):
            continue
        x = pattern(shape, 1, dev)
        dy = pattern(shape, 2, dev)
        def eager_fn():
            xi = x.detach().requires_grad_(True)
            y = F.layer_norm(xi, (shape[-1],), eps=1e-5) if kind == "ln" else xi * torch.rsqrt(xi.pow(2).mean(-1, keepdim=True) + 1e-5)
            torch.autograd.grad(y, xi, dy)
        def raw_step(xi, dyi):
            def f(z):
                return F.layer_norm(z, (shape[-1],), eps=1e-5) if kind == "ln" else z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5)
            y, vjp = torch.func.vjp(f, xi)
            return y, vjp(dyi)[0]
        comp = torch.compile(raw_step, backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
        comp(x, dy)
        torch.cuda.synchronize()
        ew, ed = time_torch_cuda(eager_fn, 16, 128, False)
        iw, id_ = time_torch_cuda(lambda: comp(x, dy), 16, 128, False)
        out[name] = {"eager_us": ew, "eager_dev_us": ed, "inductor_us": iw, "inductor_dev_us": id_}

    # 2. Softmax
    if keep("softmax-256"):
        x = pattern([8, 8, 256, 256], 3, dev)
        dy = pattern([8, 8, 256, 256], 4, dev)
        def eager_fn():
            xi = x.detach().requires_grad_(True)
            y = F.softmax(xi, dim=-1)
            torch.autograd.grad(y, xi, dy)
        def raw_step(xi, dyi):
            y, vjp = torch.func.vjp(lambda z: F.softmax(z, dim=-1), xi)
            return y, vjp(dyi)[0]
        comp = torch.compile(raw_step, backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
        comp(x, dy)
        torch.cuda.synchronize()
        ew, ed = time_torch_cuda(eager_fn, 16, 128, False)
        iw, id_ = time_torch_cuda(lambda: comp(x, dy), 16, 128, False)
        out["softmax-256"] = {"eager_us": ew, "eager_dev_us": ed, "inductor_us": iw, "inductor_dev_us": id_}

    # 3. BMM linear
    if keep("bmm-linear"):
        a = pattern([8, 256, 256], 5, dev)
        b = pattern([256, 512], 6, dev)
        dy = pattern([8, 256, 512], 7, dev)
        def eager_fn():
            ai = a.detach().requires_grad_(True)
            bi = b.detach().requires_grad_(True)
            y = torch.matmul(ai, bi)
            torch.autograd.grad(y, (ai, bi), dy)
        def raw_step(ai, bi, dyi):
            y, vjp = torch.func.vjp(torch.matmul, ai, bi)
            return y, vjp(dyi)
        comp = torch.compile(raw_step, backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
        comp(a, b, dy)
        torch.cuda.synchronize()
        ew, ed = time_torch_cuda(eager_fn, 16, 128, False)
        iw, id_ = time_torch_cuda(lambda: comp(a, b, dy), 16, 128, False)
        out["bmm-linear"] = {"eager_us": ew, "eager_dev_us": ed, "inductor_us": iw, "inductor_dev_us": id_}

    # 4. Strict-FP32 SDPA (Causal T=256, Non-Causal T=512, GQA Causal T=512)
    for name, bs, h, hkv, t, d, causal in [
        ("sdpa-causal-t256-d64", 4, 8, 8, 256, 64, True),
        ("sdpa-noncausal-t512-d64", 2, 8, 8, 512, 64, False),
        ("sdpa-gqa-causal-t512-d64", 2, 8, 2, 512, 64, True),
    ]:
        if not keep(name):
            continue
        q = pattern([bs, h, t, d], 8, dev)
        k = pattern([bs, hkv, t, d], 9, dev)
        v = pattern([bs, hkv, t, d], 10, dev)
        dy = pattern([bs, h, t, d], 11, dev)
        gqa = h != hkv
        def eager_fn():
            qi = q.detach().requires_grad_(True)
            ki = k.detach().requires_grad_(True)
            vi = v.detach().requires_grad_(True)
            o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, enable_gqa=gqa)
            torch.autograd.grad(o, (qi, ki, vi), dy)
        ew, ed = time_torch_cuda(eager_fn, 8, 64, False)
        out[name] = {"eager_us": ew, "eager_dev_us": ed}

    # 5. Full Audio Transformer Step + AdamW (Eager + CUDA Graph!)
    if keep("audio-transformer-step-adamw"):
        bs, t, h, d = 4, 256, 4, 64
        dm = h * d
        x = pattern([bs, t, dm], 1, dev)
        wq = pattern([dm, dm], 2, dev).requires_grad_(True)
        wk = pattern([dm, dm], 3, dev).requires_grad_(True)
        wv = pattern([dm, dm], 4, dev).requires_grad_(True)
        wo = pattern([dm, dm], 5, dev).requires_grad_(True)
        tgt = pattern([bs, t, dm], 6, dev)
        opt = torch.optim.AdamW([wq, wk, wv, wo], lr=1e-3, weight_decay=0.01, capturable=True)
        def step_fn():
            opt.zero_grad(set_to_none=False)
            xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
            q = (xn @ wq).reshape(bs, t, h, d).permute(0, 2, 1, 3)
            k = (xn @ wk).reshape(bs, t, h, d).permute(0, 2, 1, 3)
            v = (xn @ wv).reshape(bs, t, h, d).permute(0, 2, 1, 3)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_c = F.silu(attn.permute(0, 2, 1, 3)).reshape(bs, t, dm)
            res = attn_c @ wo + x
            loss = (res - tgt).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([wq, wk, wv, wo], 1.0)
            opt.step()
        ew, ed = time_torch_cuda(step_fn, 8, 64, False)
        gw, gd = time_torch_cuda(step_fn, 8, 64, True)
        out["audio-transformer-step-adamw"] = {
            "eager_us": ew, "eager_dev_us": ed, "cudagraph_us": gw, "cudagraph_dev_us": gd,
        }

    # 6. Full Spectral-Vocoder U-Net Step + AdamW (Eager + CUDA Graph!)
    if keep("spectral-vocoder-step-adamw"):
        bs, samples = 4, 16000
        wav = pattern([bs, samples], 1, dev)
        tgt = pattern([bs, samples], 2, dev)
        w_enc = pattern([32, 257, 3], 3, dev).requires_grad_(True)
        w_dec = pattern([32, 257, 4], 4, dev).requires_grad_(True)
        win = torch.hann_window(512, periodic=True, device=dev)
        opt = torch.optim.AdamW([w_enc, w_dec], lr=1e-3, weight_decay=0.01, capturable=True)
        def step_fn():
            opt.zero_grad(set_to_none=False)
            spec = torch.view_as_real(torch.stft(wav, 512, hop_length=128, window=win, center=True, return_complex=True))
            re = spec[..., 0]
            im = spec[..., 1]
            re_pad = F.pad(re, (2, 0), mode="reflect")
            enc = F.conv1d(re_pad, w_enc, stride=2)
            act = F.silu(enc * torch.rsqrt(enc.pow(2).mean(-1, keepdim=True) + 1e-5))
            dec = F.conv_transpose1d(act, w_dec, stride=2, padding=1)[..., : re.shape[-1]]
            merged = torch.complex(dec, im)
            recon = torch.istft(merged, 512, hop_length=128, window=win, center=True, length=samples)
            loss = (recon - tgt).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w_enc, w_dec], 1.0)
            opt.step()
        ew, ed = time_torch_cuda(step_fn, 8, 64, False)
        try:
            gw, gd = time_torch_cuda(step_fn, 8, 64, True)
            out["spectral-vocoder-step-adamw"] = {
                "eager_us": ew, "eager_dev_us": ed, "cudagraph_us": gw, "cudagraph_dev_us": gd,
            }
        except Exception as exc:
            out["spectral-vocoder-step-adamw"] = {
                "eager_us": ew, "eager_dev_us": ed, "cudagraph_error": str(exc),
            }

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--filter", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.binary is not None:
        cmd = [str(args.binary), "--device", args.device, "--mode", "bench", "--filter", args.filter]
    else:
        feat = ["--features", "cuda"] if args.device == "cuda" else []
        cmd = ["cargo", "run", "--release", *feat, "--example", "audio-foundation-bench", "--", "--device", args.device, "--mode", "bench", "--filter", args.filter]
    fa_raw = subprocess.check_output(cmd, cwd=ROOT, text=True)
    fa_data = json.loads(fa_raw)
    fa_map = {r["name"]: r for r in fa_data["results"]}

    pt_map = bench_pytorch_cpu(args.filter) if args.device == "cpu" else bench_pytorch_cuda(args.filter)

    print(f"\n=== fast-audio vs PyTorch {torch.__version__} ({args.device.upper()}, strict FP32) ===")
    print(f"{'Workload':30s} | {'fast-audio':>11s} | {'PT Eager':>11s} | {'PT Best':>11s} | {'Speedup vs Best':>15s}")
    print("-" * 88)
    summary = []
    for name, fa in fa_map.items():
        pt = pt_map.get(name, {})
        fa_us = fa["wall_us"]
        eager_us = pt.get("eager_us")
        cands = [v for k, v in pt.items() if k in ("eager_us", "inductor_us", "cudagraph_us") and isinstance(v, (int, float))]
        best_us = min(cands) if cands else None
        speedup = (best_us / fa_us) if best_us else None
        e_str = f"{eager_us:8.1f} us" if eager_us else "        n/a"
        b_str = f"{best_us:8.1f} us" if best_us else "        n/a"
        s_str = f"{speedup:6.2f}x" if speedup else "n/a"
        print(f"{name:30s} | {fa_us:8.1f} us | {e_str} | {b_str} | {s_str:>15s}")
        summary.append({"name": name, "fast_audio": fa, "pytorch": pt, "speedup_vs_best": speedup})

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "device": args.device,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "results": summary,
        }, indent=2) + "\n")
        print(f"\nSaved JSON report to {args.out}")


if __name__ == "__main__":
    main()
