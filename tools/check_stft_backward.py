#!/usr/bin/env python3
"""Independent strict-FP32 PyTorch autograd oracle for STFT/CenteredSTFT/iSTFT and spectral Tape training."""
import math
import sys
import torch


def signal(n: int, seed: int) -> list[float]:
    s = (seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    out = []
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        u = float(s >> 33) / float(1 << 31) - 0.5
        out.append(float(u * 1.4 + 0.25 * math.sin(i * 0.37)))
    return out


def custom_window(n: int) -> list[float]:
    return [
        float(
            0.65
            - 0.35 * math.cos(math.tau * i / n)
            + 0.18 * math.sin(math.tau * i / n)
            - 0.05 * (i / n)
        )
        for i in range(n)
    ]


def check_torch_spectral_autograd(device: str) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(device)

    for center in (False, True):
        for batch, length, n_fft, hop in ((2, 32, 16, 8), (2, 19, 9, 4)):
            win = torch.tensor(custom_window(n_fft), dtype=torch.float32, device=dev)
            x = torch.tensor(
                signal(batch * length, 100 + n_fft), dtype=torch.float32, device=dev
            ).reshape(batch, length)
            x.requires_grad_(True)
            spec = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=win,
                center=center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            # Shape [B, F, K, 2]
            spec_ri = torch.view_as_real(spec).transpose(1, 2).contiguous()
            dy = torch.tensor(
                signal(spec_ri.numel(), 200 + n_fft), dtype=torch.float32, device=dev
            ).reshape_as(spec_ri)
            (spec_ri * dy).sum().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            # Adjoint check <dy, STFT(x)> == <dX, x>
            lhs = (spec_ri.detach().double() * dy.double()).sum().item()
            rhs = (x.grad.detach().double() * x.detach().double()).sum().item()
            assert abs(lhs - rhs) <= 2e-5 * (1.0 + max(abs(lhs), abs(rhs))), (lhs, rhs)

            if center:
                s_in = torch.tensor(
                    signal(spec_ri.numel(), 300 + n_fft),
                    dtype=torch.float32,
                    device=dev,
                ).reshape_as(spec_ri)
                s_in.requires_grad_(True)
                s_c = torch.view_as_complex(s_in.transpose(1, 2).contiguous())
                y = torch.istft(
                    s_c,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=n_fft,
                    window=win,
                    center=True,
                    normalized=False,
                    onesided=True,
                    length=length,
                )
                dy_w = torch.tensor(
                    signal(batch * length, 400 + n_fft),
                    dtype=torch.float32,
                    device=dev,
                ).reshape(batch, length)
                (y * dy_w).sum().backward()
                assert s_in.grad is not None and torch.isfinite(s_in.grad).all()
                lhs2 = (y.detach().double() * dy_w.double()).sum().item()
                rhs2 = (s_in.grad.detach().double() * s_in.detach().double()).sum().item()
                assert abs(lhs2 - rhs2) <= 2e-5 * (1.0 + max(abs(lhs2), abs(rhs2))), (
                    lhs2,
                    rhs2,
                )
    print(f"check_stft_backward ({device}): OK")


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    check_torch_spectral_autograd(dev)
