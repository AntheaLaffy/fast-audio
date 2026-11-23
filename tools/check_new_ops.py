#!/usr/bin/env python3
"""Strict-FP32 PyTorch cross-check for all Milestone 1-3 foundation operators & AdamW.

Usage:
    python tools/check_new_ops.py cpu [--binary PATH]
    python tools/check_new_ops.py cuda [--binary PATH]
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def pattern(shape, seed, device="cpu"):
    n = math.prod(shape)
    vals = [(((i * 37 + seed * 19 + 5) % 29) - 14.0) / 32.0 for i in range(n)]
    return torch.tensor(vals, dtype=torch.float32, device=device).reshape(shape)


def check_close(name, got_list, expected_tensor, tol=3e-5):
    got = torch.tensor(got_list, dtype=torch.float32)
    exp = expected_tensor.detach().cpu().contiguous().reshape(-1).float()
    assert got.numel() == exp.numel(), f"{name}: length {got.numel()} != {exp.numel()}"
    diff = (got - exp).abs().max().item()
    assert math.isfinite(diff) and diff <= tol, f"{name}: max_abs_diff={diff:.3e} > {tol:.1e}"
    print(f"  [PASS] {name:28s} max_abs_diff = {diff:.3e} (tol {tol:.1e})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("--binary", type=Path, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    if args.binary is not None:
        cmd = [str(args.binary), "--device", dev, "--mode", "verify-json"]
    else:
        feat = ["--features", "cuda"] if dev == "cuda" else []
        cmd = ["cargo", "run", "--release", *feat, "--example", "audio-foundation-bench", "--", "--device", dev, "--mode", "verify-json"]
    raw = subprocess.check_output(cmd, cwd=ROOT, text=True)
    data = json.loads(raw)
    assert data["device"] == dev

    print(f"=== Strict-FP32 PyTorch {torch.__version__} Cross-Check ({dev.upper()}) ===")

    # 1. LayerNorm & RmsNorm
    nx = pattern([2, 4, 16], 1, dev).requires_grad_(True)
    ndy = pattern([2, 4, 16], 2, dev)
    ln_y = F.layer_norm(nx, (16,), eps=1e-5)
    ln_dx, = torch.autograd.grad(ln_y, nx, ndy)
    check_close("layernorm.y", data["layernorm"]["y"], ln_y)
    check_close("layernorm.dx", data["layernorm"]["dx"], ln_dx)

    nx2 = pattern([2, 4, 16], 1, dev).requires_grad_(True)
    rms_y = nx2 * torch.rsqrt(nx2.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    rms_dx, = torch.autograd.grad(rms_y, nx2, ndy)
    check_close("rmsnorm.y", data["rmsnorm"]["y"], rms_y)
    check_close("rmsnorm.dx", data["rmsnorm"]["dx"], rms_dx)

    # 2. Softmax & LogSoftmax
    smx = pattern([2, 5, 7], 3, dev).requires_grad_(True)
    smdy = pattern([2, 5, 7], 4, dev)
    sm_y = F.softmax(smx, dim=1)
    sm_dx, = torch.autograd.grad(sm_y, smx, smdy)
    check_close("softmax.y", data["softmax"]["y"], sm_y)
    check_close("softmax.dx", data["softmax"]["dx"], sm_dx)

    smx2 = pattern([2, 5, 7], 3, dev).requires_grad_(True)
    lsm_y = F.log_softmax(smx2, dim=1)
    lsm_dx, = torch.autograd.grad(lsm_y, smx2, smdy)
    check_close("log_softmax.y", data["log_softmax"]["y"], lsm_y)
    check_close("log_softmax.dx", data["log_softmax"]["dx"], lsm_dx)

    # 3. Nonlinear reductions (Max, Min, LogSumExp)
    rdy = pattern([2, 4], 6, dev)
    for key, fn in [("reduce_max", torch.amax), ("reduce_min", torch.amin), ("reduce_lse", torch.logsumexp)]:
        rx = pattern([2, 4, 6], 5, dev).requires_grad_(True)
        ry = fn(rx, dim=2)
        rdx, = torch.autograd.grad(ry, rx, rdy)
        check_close(f"{key}.y", data[key]["y"], ry)
        check_close(f"{key}.dx", data[key]["dx"], rdx)

    # 4. Broadcast Bmm
    ba = pattern([2, 3, 5, 7], 7, dev).requires_grad_(True)
    bb = pattern([1, 3, 7, 4], 8, dev).requires_grad_(True)
    bdy = pattern([2, 3, 5, 4], 9, dev)
    by = torch.matmul(ba, bb)
    bda, bdb = torch.autograd.grad(by, (ba, bb), bdy)
    check_close("bmm.y", data["bmm"]["y"], by)
    check_close("bmm.da", data["bmm"]["da"], bda)
    check_close("bmm.db", data["bmm"]["db"], bdb)

    # 5. SDPA GQA causal
    q = pattern([2, 4, 12, 16], 10, dev).requires_grad_(True)
    k = pattern([2, 2, 12, 16], 11, dev).requires_grad_(True)
    v = pattern([2, 2, 12, 16], 12, dev).requires_grad_(True)
    sdy = pattern([2, 4, 12, 16], 13, dev)
    so = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    sdq, sdk, sdv = torch.autograd.grad(so, (q, k, v), sdy)
    check_close("sdpa_gqa_causal.o", data["sdpa_gqa_causal"]["o"], so)
    check_close("sdpa_gqa_causal.dq", data["sdpa_gqa_causal"]["dq"], sdq)
    check_close("sdpa_gqa_causal.dk", data["sdpa_gqa_causal"]["dk"], sdk)
    check_close("sdpa_gqa_causal.dv", data["sdpa_gqa_causal"]["dv"], sdv)

    # 6. ConvTranspose1d & ConvTranspose2d
    ct1_x = pattern([2, 4, 7], 14, dev).requires_grad_(True)
    ct1_w = pattern([4, 3, 4], 15, dev).requires_grad_(True)
    ct1_dy = pattern([2, 6, 15], 16, dev)
    ct1_y = F.conv_transpose1d(ct1_x, ct1_w, stride=2, padding=1, output_padding=1, dilation=1, groups=2)
    ct1_dx, ct1_dw = torch.autograd.grad(ct1_y, (ct1_x, ct1_w), ct1_dy)
    check_close("conv_transpose1d.y", data["conv_transpose1d"]["y"], ct1_y)
    check_close("conv_transpose1d.dx", data["conv_transpose1d"]["dx"], ct1_dx)
    check_close("conv_transpose1d.dw", data["conv_transpose1d"]["dw"], ct1_dw)

    ct2_x = pattern([2, 4, 4, 5], 17, dev).requires_grad_(True)
    ct2_w = pattern([4, 2, 3, 2], 18, dev).requires_grad_(True)
    ct2_dy = pattern([2, 4, 8, 12], 19, dev)
    ct2_y = F.conv_transpose2d(ct2_x, ct2_w, stride=(2, 2), padding=(1, 0), output_padding=(1, 1), dilation=(1, 2), groups=2)
    ct2_dx, ct2_dw = torch.autograd.grad(ct2_y, (ct2_x, ct2_w), ct2_dy)
    check_close("conv_transpose2d.y", data["conv_transpose2d"]["y"], ct2_y)
    check_close("conv_transpose2d.dx", data["conv_transpose2d"]["dx"], ct2_dx)
    check_close("conv_transpose2d.dw", data["conv_transpose2d"]["dw"], ct2_dw)

    # 7. Pad (Reflect & Replicate) & Cat
    pdy = pattern([2, 6, 8], 21, dev)
    for key, mode in [("pad_reflect", "reflect"), ("pad_replicate", "replicate")]:
        px = pattern([2, 3, 5], 20, dev).requires_grad_(True)
        # F.pad uses reverse axis order: (last_before, last_after, prev_before, prev_after)
        py = F.pad(px.unsqueeze(0), (2, 1, 1, 2), mode=mode).squeeze(0)
        pdx, = torch.autograd.grad(py, px, pdy)
        check_close(f"{key}.y", data[key]["y"], py)
        check_close(f"{key}.dx", data[key]["dx"], pdx)

    ca = pattern([2, 3, 4], 22, dev).requires_grad_(True)
    cb = pattern([2, 2, 4], 23, dev).requires_grad_(True)
    cdy = pattern([2, 5, 4], 24, dev)
    cy = torch.cat([ca, cb], dim=1)
    cda, cdb = torch.autograd.grad(cy, (ca, cb), cdy)
    check_close("cat.y", data["cat"]["y"], cy)
    check_close("cat.da", data["cat"]["da"], cda)
    check_close("cat.db", data["cat"]["db"], cdb)

    # 8. 5-step AdamW + global L2 grad clip + dynamic LR
    ax = pattern([2, 2, 12], 30, dev)
    aw = pattern([4, 2, 3], 31, dev).requires_grad_(True)
    ag = pattern([1, 4, 1], 32, dev).requires_grad_(True)
    ab = pattern([1, 4, 1], 33, dev).requires_grad_(True)
    at = pattern([2, 4, 12], 34, dev)
    opt = torch.optim.AdamW([aw, ag, ab], lr=0.02, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.02)
    for step in range(5):
        lr = 0.02 * (1.0 - 0.1 * step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        h = F.conv1d(ax, aw, padding=1)
        hn = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        out = F.silu(hn) * ag + ab
        loss = (out - at).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([aw, ag, ab], 0.25)
        opt.step()
    check_close("adamw_5step.w", data["adamw_5step"]["w"], aw, tol=5e-5)
    check_close("adamw_5step.gamma", data["adamw_5step"]["gamma"], ag, tol=5e-5)
    check_close("adamw_5step.bias", data["adamw_5step"]["bias"], ab, tol=5e-5)

    print(f"ALL 35 STRICT-FP32 PYTORCH CROSS-CHECKS PASSED ON {dev.upper()}!")


if __name__ == "__main__":
    main()
