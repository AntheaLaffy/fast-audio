#!/usr/bin/env python3
"""Numerical parity & 30-step 6-layer Audio Transformer convergence cross-check vs PyTorch."""

import os
import subprocess
import sys
import tempfile
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def write_f32(path, arr):
    np.asarray(arr, dtype=np.float32).tofile(path)


def write_i64(path, arr):
    np.asarray(arr, dtype=np.int64).tofile(path)


def read_f32(path, shape=None):
    arr = np.fromfile(path, dtype=np.float32)
    return arr.reshape(shape) if shape is not None else arr


def apply_rope_torch(x, base=10000.0, seq_axis=2):
    # x: [B, H, T, D] if seq_axis==2 else [B, T, H, D]
    D = x.shape[-1]
    half = D // 2
    T = x.shape[seq_axis]
    inv_freq = torch.tensor(
        [base ** (-2.0 * i / D) for i in range(half)],
        dtype=torch.float64,
        device=x.device,
    )
    pos = torch.arange(T, dtype=torch.float64, device=x.device)
    angles = torch.outer(pos, inv_freq)
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    if seq_axis == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def rmsnorm_torch(x, w, eps=1e-5):
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv).to(x.dtype) * w


class RefAudioTransformer(nn.Module):
    def __init__(self, vocab, d_model, num_heads, num_kv_heads, ffn_hidden, num_layers, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.hq = num_heads
        self.hkv = num_kv_heads
        self.hd = d_model // num_heads
        self.eps = eps
        self.tok_emb = nn.Parameter(torch.empty(vocab, d_model))
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ParameterList([
                nn.Parameter(torch.ones(d_model)),
                nn.Parameter(torch.empty((num_heads + 2 * num_kv_heads) * self.hd, d_model)),
                nn.Parameter(torch.empty(d_model, d_model)),
                nn.Parameter(torch.ones(d_model)),
                nn.Parameter(torch.empty(2 * ffn_hidden, d_model)),
                nn.Parameter(torch.empty(d_model, ffn_hidden)),
            ]))
        self.final_rms = nn.Parameter(torch.ones(d_model))
        self.lm_head = nn.Parameter(torch.empty(vocab, d_model))

    def all_params(self):
        ps = [self.tok_emb]
        for layer in self.layers:
            ps.extend(list(layer))
        ps.extend([self.final_rms, self.lm_head])
        return ps

    def forward(self, ids):
        B, T = ids.shape
        x = F.embedding(ids, self.tok_emb)
        for layer in self.layers:
            rms1, w_qkv, w_o, rms2, w_gu, w_dn = layer
            h = rmsnorm_torch(x, rms1, self.eps)
            qkv = F.linear(h, w_qkv).view(B, T, self.hq + 2 * self.hkv, self.hd)
            q = apply_rope_torch(qkv[:, :, :self.hq, :], seq_axis=1).transpose(1, 2)
            k = apply_rope_torch(qkv[:, :, self.hq:self.hq + self.hkv, :], seq_axis=1).transpose(1, 2)
            v = qkv[:, :, self.hq + self.hkv:, :].transpose(1, 2)
            if self.hq != self.hkv:
                rep = self.hq // self.hkv
                k_rep = k.repeat_interleave(rep, dim=1)
                v_rep = v.repeat_interleave(rep, dim=1)
            else:
                k_rep, v_rep = k, v
            attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)
            x = x + F.linear(attn, w_o)
            h2 = rmsnorm_torch(x, rms2, self.eps)
            gu = F.linear(h2, w_gu)
            g, u = gu.chunk(2, dim=-1)
            x = x + F.linear(F.silu(g) * u, w_dn)
        h_fin = rmsnorm_torch(x, self.final_rms, self.eps)
        return F.linear(h_fin, self.lm_head)


def check_err(name, a, b, max_tol, mean_tol=None):
    diff = np.abs(a - b)
    mx = float(diff.max())
    mn = float(diff.mean())
    ok = mx <= max_tol and (mean_tol is None or mn <= mean_tol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:36s} max_err={mx:.3e} mean_err={mn:.3e} (tol={max_tol:.1e})")
    if not ok:
        raise AssertionError(f"{name} failed: max_err={mx}, mean_err={mn}")


def run_cpu(bin_path):
    print(f"=== Phase B CPU Cross-Check vs PyTorch {torch.__version__} ===")
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as d:
        # Embedding
        w = rng.standard_normal((32, 64)).astype(np.float32) * 0.5
        idx = rng.integers(0, 32, size=(4, 16), dtype=np.int64)
        dy = rng.standard_normal((4, 16, 64)).astype(np.float32) * 0.3
        write_f32(f"{d}/emb_w.bin", w)
        write_i64(f"{d}/emb_idx.bin", idx)
        write_f32(f"{d}/emb_dy.bin", dy)

        # RoPE
        rx = rng.standard_normal((2, 4, 16, 32)).astype(np.float32) * 0.7
        rdy = rng.standard_normal((2, 4, 16, 32)).astype(np.float32) * 0.5
        write_f32(f"{d}/rope_x.bin", rx)
        write_f32(f"{d}/rope_dy.bin", rdy)

        # CrossEntropy
        logits = rng.standard_normal((16, 64)).astype(np.float32) * 1.5
        targets = rng.integers(0, 64, size=(16,), dtype=np.int64)
        targets[3] = -100
        targets[11] = -100
        write_f32(f"{d}/ce_logits.bin", logits)
        write_i64(f"{d}/ce_targets.bin", targets)

        subprocess.run([bin_path, "verify-cpu", d], check=True)

        # PyTorch references
        tw = torch.tensor(w, requires_grad=True)
        ti = torch.tensor(idx, dtype=torch.long)
        to = F.embedding(ti, tw, padding_idx=0)
        to.backward(torch.tensor(dy))
        check_err("cpu-embedding-fwd", read_f32(f"{d}/emb_out_fa.bin", to.shape), to.detach().numpy(), 1e-6)
        check_err("cpu-embedding-bwd", read_f32(f"{d}/emb_dw_fa.bin", tw.shape), tw.grad.numpy(), 1e-5)

        tx = torch.tensor(rx, requires_grad=True)
        ty = apply_rope_torch(tx, base=10000.0, seq_axis=2)
        ty.backward(torch.tensor(rdy))
        check_err("cpu-rope-fwd", read_f32(f"{d}/rope_y_fa.bin", ty.shape), ty.detach().numpy(), 1e-5)
        check_err("cpu-rope-bwd", read_f32(f"{d}/rope_dx_fa.bin", tx.shape), tx.grad.numpy(), 1e-5)

        for tag, ls in [("ce0", 0.0), ("ce1", 0.1)]:
            tl = torch.tensor(logits, requires_grad=True)
            tt = torch.tensor(targets, dtype=torch.long)
            loss = F.cross_entropy(tl, tt, ignore_index=-100, label_smoothing=ls)
            loss.backward()
            check_err(f"cpu-{tag}-loss", read_f32(f"{d}/{tag}_loss_fa.bin"), np.array([loss.item()], dtype=np.float32), 1e-5)
            check_err(f"cpu-{tag}-grad", read_f32(f"{d}/{tag}_grad_fa.bin", tl.shape), tl.grad.numpy(), 1e-5)


def run_cuda(bin_path):
    print(f"=== Phase A+B CUDA BF16 Tensor Core Cross-Check vs PyTorch {torch.__version__} ({torch.cuda.get_device_name(0)}) ===")
    rng = np.random.default_rng(123)
    with tempfile.TemporaryDirectory() as d:
        # 1. SDPA cases
        sdpa_inputs = {}
        for tag, b, hq, hkv, tq, tk, dim, causal in [
            ("sdpa64", 2, 8, 2, 96, 96, 64, True),
            ("sdpa128", 1, 4, 4, 128, 128, 128, False),
            ("sdpa48", 1, 4, 2, 64, 80, 48, True),
            ("sdpa129", 1, 2, 1, 29, 21, 129, True),
        ]:
            sdpa_rng = rng if dim in (64, 128) else np.random.default_rng(123 + dim)
            q = (sdpa_rng.standard_normal((b, hq, tq, dim)) * 0.4).astype(np.float32)
            k = (sdpa_rng.standard_normal((b, hkv, tk, dim)) * 0.4).astype(np.float32)
            v = (sdpa_rng.standard_normal((b, hkv, tk, dim)) * 0.4).astype(np.float32)
            do = (sdpa_rng.standard_normal((b, hq, tq, dim)) * 0.3).astype(np.float32)
            write_f32(f"{d}/{tag}_q.bin", q)
            write_f32(f"{d}/{tag}_k.bin", k)
            write_f32(f"{d}/{tag}_v.bin", v)
            write_f32(f"{d}/{tag}_do.bin", do)
            sdpa_inputs[tag] = (q, k, v, do, hq, hkv, causal)

        # 2. Norm, SwiGLU, CE, Emb
        rms_a = (rng.standard_normal((32, 128)) * 0.8).astype(np.float32)
        rms_b = (rng.standard_normal((32, 128)) * 0.5).astype(np.float32)
        rms_w = (rng.uniform(0.7, 1.3, size=(128,))).astype(np.float32)
        rms_dy = (rng.standard_normal((32, 128)) * 0.4).astype(np.float32)
        sw_do = (rng.standard_normal((32, 64)) * 0.4).astype(np.float32)
        bf16_ce_t = rng.integers(0, 128, size=(32,), dtype=np.int64)
        bf16_ce_t[4] = -100
        emb_w = (rng.standard_normal((32, 64)) * 0.5).astype(np.float32)
        emb_idx = rng.integers(0, 32, size=(4, 16), dtype=np.int64)
        emb_dy = (rng.standard_normal((4, 16, 64)) * 0.3).astype(np.float32)
        write_f32(f"{d}/rms_a.bin", rms_a)
        write_f32(f"{d}/rms_b.bin", rms_b)
        write_f32(f"{d}/rms_w.bin", rms_w)
        write_f32(f"{d}/rms_dy.bin", rms_dy)
        write_f32(f"{d}/sw_do.bin", sw_do)
        write_i64(f"{d}/bf16_ce_t.bin", bf16_ce_t)
        write_f32(f"{d}/emb_w.bin", emb_w)
        write_i64(f"{d}/emb_idx.bin", emb_idx)
        write_f32(f"{d}/emb_dy.bin", emb_dy)

        # 3. 30-step 6-Layer GQA Causal Audio Transformer weights & batch
        model = RefAudioTransformer(256, 256, 4, 2, 512, 6, eps=1e-5).cuda()
        with torch.no_grad():
            for idx_p, p in enumerate(model.all_params()):
                if p.ndim == 1:
                    p.fill_(1.0)
                else:
                    arr = (rng.standard_normal(p.shape) * 0.03).astype(np.float32)
                    p.copy_(torch.tensor(arr, device="cuda"))
                write_f32(f"{d}/tr_w_{idx_p}.bin", p.cpu().numpy())
        # Structured repeating token pattern for crisp loss decay
        pat = np.arange(2 * 128, dtype=np.int64)
        tr_in = ((pat * 7 + (pat // 16) * 5) % 256).reshape(2, 128)
        tr_tgt = (((pat + 1) * 7 + (pat // 16) * 5) % 256)
        write_i64(f"{d}/tr_in.bin", tr_in)
        write_i64(f"{d}/tr_tgt.bin", tr_tgt)

        subprocess.run([bin_path, "verify-cuda", d], check=True)

        # Verify SDPA vs FP32 exact reference on GPU
        for tag, (q, k, v, do, hq, hkv, causal) in sdpa_inputs.items():
            tq = torch.tensor(q, device="cuda", requires_grad=True)
            tk = torch.tensor(k, device="cuda", requires_grad=True)
            tv = torch.tensor(v, device="cuda", requires_grad=True)
            if hq != hkv:
                rep = hq // hkv
                kr = tk.repeat_interleave(rep, dim=1)
                vr = tv.repeat_interleave(rep, dim=1)
            else:
                kr, vr = tk, tv
            to = F.scaled_dot_product_attention(tq, kr, vr, is_causal=causal)
            to.backward(torch.tensor(do, device="cuda"))
            check_err(f"bf16-{tag}-O", read_f32(f"{d}/{tag}_o_fa.bin", to.shape), to.detach().cpu().numpy(), 3e-2, 4e-3)
            check_err(f"bf16-{tag}-dQ", read_f32(f"{d}/{tag}_dq_fa.bin", tq.shape), tq.grad.cpu().numpy(), 3e-2, 4e-3)
            check_err(f"bf16-{tag}-dK", read_f32(f"{d}/{tag}_dk_fa.bin", tk.shape), tk.grad.cpu().numpy(), 3e-2, 4e-3)
            check_err(f"bf16-{tag}-dV", read_f32(f"{d}/{tag}_dv_fa.bin", tv.shape), tv.grad.cpu().numpy(), 3e-2, 4e-3)

        # Verify Residual+RmsNorm
        ta = torch.tensor(rms_a, device="cuda", dtype=torch.bfloat16)
        tb = torch.tensor(rms_b, device="cuda", dtype=torch.bfloat16)
        tres = (ta + tb).detach().float().requires_grad_(True)
        tw = torch.tensor(rms_w, device="cuda", requires_grad=True)
        ty = rmsnorm_torch(tres, tw, 1e-5)
        ty.backward(torch.tensor(rms_dy, device="cuda"))
        check_err("bf16-residual-rmsnorm-y", read_f32(f"{d}/rms_y_fa.bin", ty.shape), ty.detach().cpu().numpy(), 3e-2, 5e-3)
        check_err("bf16-residual-rmsnorm-dx", read_f32(f"{d}/rms_dx_fa.bin", tres.shape), tres.grad.cpu().numpy(), 3e-2, 5e-3)
        check_err("bf16-residual-rmsnorm-dw", read_f32(f"{d}/rms_dw_fa.bin", tw.shape), tw.grad.cpu().numpy(), 6e-2, 1.5e-2)

        # Verify SwiGLU
        tgu = torch.tensor(rms_a, device="cuda", requires_grad=True)
        g, u = tgu.chunk(2, dim=-1)
        tsw = F.silu(g) * u
        tsw.backward(torch.tensor(sw_do, device="cuda"))
        check_err("bf16-swiglu-fwd", read_f32(f"{d}/sw_o_fa.bin", tsw.shape), tsw.detach().cpu().numpy(), 2e-2, 3e-3)
        check_err("bf16-swiglu-bwd", read_f32(f"{d}/sw_dgu_fa.bin", tgu.shape), tgu.grad.cpu().numpy(), 2e-2, 3e-3)

        # Verify BF16 CrossEntropy
        tl = torch.tensor(rms_a, device="cuda", dtype=torch.bfloat16).float().requires_grad_(True)
        tt = torch.tensor(bf16_ce_t, device="cuda", dtype=torch.long)
        tloss = F.cross_entropy(tl, tt, ignore_index=-100, label_smoothing=0.1)
        tloss.backward()
        check_err("bf16-cross-entropy-loss", read_f32(f"{d}/bf16_ce_l_fa.bin"), np.array([tloss.item()], dtype=np.float32), 2e-2)
        check_err("bf16-cross-entropy-grad", read_f32(f"{d}/bf16_ce_g_fa.bin", tl.shape), tl.grad.cpu().numpy(), 2e-3, 3e-4)

        # Verify BF16 Embedding
        tew = torch.tensor(emb_w, device="cuda", dtype=torch.bfloat16).float().requires_grad_(True)
        tei = torch.tensor(emb_idx, device="cuda", dtype=torch.long)
        teo = F.embedding(tei, tew, padding_idx=0)
        teo.backward(torch.tensor(emb_dy, device="cuda", dtype=torch.bfloat16).float())
        check_err("bf16-embedding-fwd", read_f32(f"{d}/bf16_emb_o_fa.bin", teo.shape), teo.detach().cpu().numpy(), 1e-5)
        check_err("bf16-embedding-bwd-f32", read_f32(f"{d}/bf16_emb_dw_fa.bin", tew.shape), tew.grad.cpu().numpy(), 1e-4)

        # Verify 30-step 6-Layer GQA Audio Transformer loss convergence vs PyTorch autocast(bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
        tin = torch.tensor(tr_in, device="cuda", dtype=torch.long)
        ttgt = torch.tensor(tr_tgt, device="cuda", dtype=torch.long)
        pt_losses = []
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tin).view(-1, 256)
                loss = F.cross_entropy(logits, ttgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pt_losses.append(float(loss.item()))
        fa_losses = read_f32(f"{d}/tr_losses_fa.bin").tolist()
        max_loss_diff = max(abs(a - b) for a, b in zip(fa_losses, pt_losses))
        print(f"  30-step 6-layer GQA Transformer convergence:")
        print(f"    PyTorch BF16: step1={pt_losses[0]:.4f} -> step15={pt_losses[14]:.4f} -> step30={pt_losses[-1]:.4f}")
        print(f"    fast-audio:   step1={fa_losses[0]:.4f} -> step15={fa_losses[14]:.4f} -> step30={fa_losses[-1]:.4f}")
        print(f"    max |loss_fa - loss_pt| across 30 steps = {max_loss_diff:.4f}")
        assert all(math.isfinite(x) for x in fa_losses), "non-finite fast-audio loss!"
        assert fa_losses[-1] < fa_losses[0] * 0.25, "fast-audio failed to converge!"
        assert max_loss_diff < 0.15, f"30-step BF16 trajectory diverged: max_diff={max_loss_diff}"
        print("  [PASS] 30-step 6-layer GQA Audio Transformer BF16 convergence matches PyTorch!")


if __name__ == "__main__":
    mode = sys.argv[1]
    bin_path = sys.argv[2]
    if mode == "cpu":
        run_cpu(bin_path)
    elif mode == "cuda":
        run_cuda(bin_path)
    else:
        raise SystemExit(f"unknown mode: {mode}")
