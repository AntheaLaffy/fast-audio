#!/usr/bin/env python3
"""Head-to-head BF16 Tensor Core benchmark: fast-audio vs PyTorch 2.14 (autocast BF16 Eager / Compile / CUDA Graph)."""

import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def distribution(samples):
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p95_us": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p99_us": ordered[math.ceil(0.99 * len(ordered)) - 1],
    }


def bench_cuda_us(fn, warmup, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Materialize PyTorch's lazy event handles outside the first-call measurement.
    start.record()
    end.record()
    end.synchronize()

    def measure(count):
        torch.cuda.synchronize()
        wall_start = time.perf_counter_ns()
        start.record()
        for _ in range(count):
            fn()
        end.record()
        end.synchronize()
        wall_us = (time.perf_counter_ns() - wall_start) / 1000 / count
        return start.elapsed_time(end) * 1000 / count, wall_us

    first_device, first_wall = measure(1)
    for _ in range(warmup):
        fn()
    device_us, wall_us = measure(iters)
    device_samples, wall_samples = zip(*(measure(1) for _ in range(100)))
    return {
        "device_us": device_us,
        "wall_us": wall_us,
        "first_device_us": first_device,
        "first_wall_us": first_wall,
        "device_samples_us": device_samples,
        "wall_samples_us": wall_samples,
        "device_latency": distribution(device_samples),
        "wall_latency": distribution(wall_samples),
    }


def apply_rope_bhtd(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class TorchAudioTransformer(nn.Module):
    def __init__(self, vocab=4096, d_model=512, hq=8, hkv=2, ffn=1536, layers=6, seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.hq = hq
        self.hkv = hkv
        self.hd = d_model // hq
        self.emb = nn.Embedding(vocab, d_model)
        self.rms1 = nn.ModuleList([nn.RMSNorm(d_model, eps=1e-5) for _ in range(layers)])
        self.qkv = nn.ModuleList([nn.Linear(d_model, (hq + 2 * hkv) * self.hd, bias=False) for _ in range(layers)])
        self.wo = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(layers)])
        self.rms2 = nn.ModuleList([nn.RMSNorm(d_model, eps=1e-5) for _ in range(layers)])
        self.wgu = nn.ModuleList([nn.Linear(d_model, 2 * ffn, bias=False) for _ in range(layers)])
        self.wdn = nn.ModuleList([nn.Linear(ffn, d_model, bias=False) for _ in range(layers)])
        self.final_rms = nn.RMSNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        inv_freq = 10000.0 ** (-torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd)
        pos = torch.arange(seq_len, dtype=torch.float32)
        ang = torch.outer(pos, inv_freq)
        self.register_buffer("cos", ang.cos()[None, None, :, :].to(torch.bfloat16))
        self.register_buffer("sin", ang.sin()[None, None, :, :].to(torch.bfloat16))

    def ordered_parameters(self):
        yield self.emb.weight
        for modules in zip(self.rms1, self.qkv, self.wo, self.rms2, self.wgu, self.wdn):
            yield from (module.weight for module in modules)
        yield self.final_rms.weight
        yield self.lm_head.weight

    def forward(self, ids):
        B, T = ids.shape
        x = self.emb(ids)
        for l in range(len(self.rms1)):
            h = self.rms1[l](x)
            qkv = self.qkv[l](h).view(B, T, self.hq + 2 * self.hkv, self.hd)
            q = apply_rope_bhtd(qkv[:, :, :self.hq, :].transpose(1, 2), self.cos, self.sin)
            k = apply_rope_bhtd(qkv[:, :, self.hq:self.hq + self.hkv, :].transpose(1, 2), self.cos, self.sin)
            v = qkv[:, :, self.hq + self.hkv:, :].transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.hq != self.hkv))
            attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)
            x = x + self.wo[l](attn)
            h2 = self.rms2[l](x)
            g, u = self.wgu[l](h2).chunk(2, dim=-1)
            x = x + self.wdn[l](F.silu(g) * u)
        return self.lm_head(self.final_rms(x))


def write_transformer_weights(directory):
    model = TorchAudioTransformer()
    generator = torch.Generator().manual_seed(351)
    with torch.no_grad():
        for index, weight in enumerate(model.ordered_parameters()):
            if weight.ndim == 1:
                weight.fill_(1.0)
            else:
                weight.normal_(std=0.02, generator=generator)
            weight.numpy().astype("<f4", copy=False).tofile(directory / f"tr_w_{index}.bin")
    return model.state_dict()


def run_pytorch_bf16_benchmarks(transformer_weights):
    pt = {}

    # 1. BF16 Linear fwd+bwd [2048, 1024] x [1024, 1024]
    x = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    def lin_step():
        x.grad = None; w.grad = None
        y = F.linear(x, w)
        y.backward(dy)
    pt["bf16-linear-2048x1024-fwd-bwd"] = {"eager_us": bench_cuda_us(lin_step, 10, 50)}

    # 2. BF16 SDPA (PyTorch FlashAttention-2 via F.scaled_dot_product_attention!)
    for name, b, hq, hkv, t, d, causal in [
        ("bf16-sdpa-causal-t512-d64", 4, 8, 8, 512, 64, True),
        ("bf16-sdpa-causal-t1024-d64", 4, 8, 8, 1024, 64, True),
        ("bf16-sdpa-gqa-causal-t2048-d64", 2, 8, 2, 2048, 64, True),
        ("bf16-sdpa-causal-t1024-d128", 2, 8, 8, 1024, 128, True),
    ]:
        q = torch.randn(b, hq, t, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, hkv, t, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, hkv, t, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn(b, hq, t, d, device="cuda", dtype=torch.bfloat16)
        gqa = (hq != hkv)
        def fwd_only():
            with torch.no_grad():
                F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa)
        def fwd_bwd():
            q.grad = None; k.grad = None; v.grad = None
            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa)
            o.backward(do)
        pt[f"{name}-fwd"] = {"eager_us": bench_cuda_us(fwd_only, 5, 30)}
        pt[f"{name}-fwd-bwd"] = {"eager_us": bench_cuda_us(fwd_bwd, 5, 25)}

    # 3. BF16 Residual + RmsNorm fwd+bwd [2048, 1024]
    a = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b_t = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    rms = nn.RMSNorm(1024, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    dy_r = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    def rms_eager():
        a.grad = None; rms.weight.grad = None
        y = rms(a + b_t)
        y.backward(dy_r)
    rms_compiled = torch.compile(lambda x1, x2: rms(x1 + x2))
    def rms_comp():
        a.grad = None; rms.weight.grad = None
        y = rms_compiled(a, b_t)
        y.backward(dy_r)
    pt["bf16-rmsnorm-2048x1024-fwd-bwd"] = {
        "eager_us": bench_cuda_us(rms_eager, 10, 50),
        "compile_us": bench_cuda_us(rms_comp, 5, 50),
    }

    # 4. BF16 SwiGLU fwd+bwd [2048, 1536]
    gu = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    do_sw = torch.randn(2048, 1536, device="cuda", dtype=torch.bfloat16)
    def sw_fn(t):
        g, u = t.chunk(2, dim=-1)
        return F.silu(g) * u
    sw_compiled = torch.compile(sw_fn)
    def sw_eager():
        gu.grad = None; sw_fn(gu).backward(do_sw)
    def sw_comp():
        gu.grad = None; sw_compiled(gu).backward(do_sw)
    pt["bf16-swiglu-2048x1536-fwd-bwd"] = {
        "eager_us": bench_cuda_us(sw_eager, 10, 50),
        "compile_us": bench_cuda_us(sw_comp, 5, 50),
    }

    # 5. BF16 RoPE fwd+bwd [4, 8, 1024, 64]
    rx = torch.randn(4, 8, 1024, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    rdy = torch.randn(4, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
    inv_f = 10000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64)
    ang = torch.outer(torch.arange(1024, dtype=torch.float32, device="cuda"), inv_f)
    cos = ang.cos()[None, None, :, :].to(torch.bfloat16)
    sin = ang.sin()[None, None, :, :].to(torch.bfloat16)
    rope_compiled = torch.compile(apply_rope_bhtd)
    def rope_eager():
        rx.grad = None; apply_rope_bhtd(rx, cos, sin).backward(rdy)
    def rope_comp():
        rx.grad = None; rope_compiled(rx, cos, sin).backward(rdy)
    pt["bf16-rope-t1024-d64-fwd-bwd"] = {
        "eager_us": bench_cuda_us(rope_eager, 10, 50),
        "compile_us": bench_cuda_us(rope_comp, 5, 50),
    }

    # 6. BF16 CrossEntropy fwd+bwd [2048, 4096]
    logits = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, 4096, (2048,), device="cuda", dtype=torch.long)
    ce_compiled = torch.compile(lambda l, t: F.cross_entropy(l, t))
    def ce_eager():
        logits.grad = None; F.cross_entropy(logits, targets).backward()
    def ce_comp():
        logits.grad = None; ce_compiled(logits, targets).backward()
    pt["bf16-cross-entropy-2048x4096-fwd-bwd"] = {
        "eager_us": bench_cuda_us(ce_eager, 10, 50),
        "compile_us": bench_cuda_us(ce_comp, 5, 50),
    }

    # 7. Flagship 6-Layer GQA Causal Audio Transformer (autocast BF16 Eager + torch.compile + CUDA Graph!)
    positions = torch.arange(2048, device="cuda", dtype=torch.long)
    ids = ((positions * 19) % 4096).view(4, 512)
    tgt = ((positions + 1) * 19) % 4096

    def model_from_fixture():
        model = TorchAudioTransformer()
        model.load_state_dict(transformer_weights)
        return model.cuda()

    # Eager autocast BF16 + fused AdamW
    m_eager = model_from_fixture()
    opt_eager = torch.optim.AdamW(m_eager.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    def tr_eager():
        opt_eager.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m_eager(ids).view(-1, 4096), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_eager.parameters(), 1.0)
        opt_eager.step()
    eager_us = bench_cuda_us(tr_eager, 5, 30)

    # torch.compile(mode="default") autocast BF16 + fused AdamW
    m_comp = model_from_fixture()
    m_comp_fn = torch.compile(m_comp, mode="default")
    opt_comp = torch.optim.AdamW(m_comp.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    def tr_comp():
        opt_comp.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m_comp_fn(ids).view(-1, 4096), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_comp.parameters(), 1.0)
        opt_comp.step()
    compile_us = bench_cuda_us(tr_comp, 5, 30)

    # Explicit CUDA Graph + torch.compile (PyTorch absolute ceiling!)
    m_cg = model_from_fixture()
    m_cg_fn = torch.compile(m_cg, mode="default")
    opt_cg = torch.optim.AdamW(m_cg.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01, fused=True, capturable=True)
    def tr_cg():
        opt_cg.zero_grad(set_to_none=False)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m_cg_fn(ids).view(-1, 4096), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_cg.parameters(), 1.0)
        opt_cg.step()

    # AccumulateGrad retains its warmup stream, so reuse it for capture.
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            tr_cg()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    capture_start = time.perf_counter_ns()
    with torch.cuda.graph(g, stream=capture_stream):
        tr_cg()
    capture_wall_us = (time.perf_counter_ns() - capture_start) / 1000
    cudagraph_us = bench_cuda_us(lambda: g.replay(), 5, 30)
    cudagraph_us["capture_wall_us"] = capture_wall_us

    pt["bf16-audio-transformer-6layer-adamw"] = {
        "eager_us": eager_us,
        "compile_us": compile_us,
        "cudagraph_us": cudagraph_us,
    }
    for timings in pt["bf16-audio-transformer-6layer-adamw"].values():
        timings["tokens_per_second"] = 2048 * 1e6 / timings["wall_us"]
    return pt


def main():
    bin_path = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else "/tmp/bf16_vs_pytorch_l4.json"
    with tempfile.TemporaryDirectory(prefix="fa-bf16-") as directory:
        transformer_weights = write_transformer_weights(Path(directory))
        out = subprocess.check_output([bin_path, "bench-cuda", directory], text=True)
    fa_rows = json.loads(out)
    # Retain native samples if a compiler or graph mode aborts the comparison.
    Path(f"{out_json}.native.json").write_text(out, encoding="utf-8")
    pt_map = run_pytorch_bf16_benchmarks(transformer_weights)

    summary = {
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformer_weight_seed": 351,
        "speedup_metric": "synchronized batch wall time per step",
        "latency_samples": "100 serial synchronized calls after batched timing",
        "benchmarks": [],
    }
    print(f"\n=== BF16 Tensor Core Head-to-Head on {summary['device']} (PyTorch {summary['pytorch']}) ===")
    print(f"{'Workload':38s} | {'fast-audio':>10s} | {'PT Eager':>10s} | {'PT Best':>10s} | {'vs Eager':>8s} | {'vs Best':>8s}")
    print("-" * 100)
    wins_eager = 0
    wins_best = 0
    for row in fa_rows:
        name = row["name"]
        fa_us = row["device_us"]
        pt_info = pt_map[name]
        fa_wall = row["wall_us"]
        if name == "bf16-audio-transformer-6layer-adamw":
            row["tokens_per_second"] = 2048 * 1e6 / fa_wall
        pt_eager = pt_info["eager_us"]["wall_us"]
        best_mode, best = min(pt_info.items(), key=lambda mode: mode[1]["wall_us"])
        pt_best = best["wall_us"]
        sp_eager = pt_eager / fa_wall
        sp_best = pt_best / fa_wall
        if sp_eager >= 1.0:
            wins_eager += 1
        if sp_best >= 1.0:
            wins_best += 1
        print(f"{name:38s} | {fa_wall:8.1f}us | {pt_eager:8.1f}us | {pt_best:8.1f}us | {sp_eager:7.2f}x | {sp_best:7.2f}x")
        summary["benchmarks"].append({
            "name": name,
            "fast_audio_us": fa_us,
            "fast_audio_wall_us": fa_wall,
            "fast_audio": {
                **row,
                "device_latency": distribution(row["device_samples_us"]),
                "wall_latency": distribution(row["wall_samples_us"]),
            },
            "pytorch": pt_info,
            "pytorch_best_wall_mode": best_mode,
            "speedup_vs_eager": sp_eager,
            "speedup_vs_best": sp_best,
        })
    print("-" * 100)
    print(f"Score: {wins_eager}/{len(fa_rows)} vs PyTorch BF16 Eager | {wins_best}/{len(fa_rows)} vs PyTorch BF16 Best (Compile/Graph)")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved report to {out_json}")


if __name__ == "__main__":
    main()
