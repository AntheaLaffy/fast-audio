---
source: progress.md
lines: 1656
generated_at: 2026-11-23
---

Progress navigation by milestone; evidence remains in the source and linked artifacts.

## Feature Index

| Intent | Lines | Notes |
| --- | --- | --- |
| 2026-11-23: objective refocused on comprehensive PyTorch leadership | 1637–1656 | Direction adjustment deferring non-PyTorch framework/C++ inference engine comparisons; preserves historical evidence |
| 2026-11-23: BF16 SDPA fallback and GTCRN C++ inference screen | 1554–1635 | Release A/B, resource traces, CUDA safety and direct CPU ORT comparison |
| 2026-11-23: A0 prototype screen | 1474–1552 | Paired CPU cross-entropy measurements, matched L4 training screen, sanitizer race reproduction and verification |
| 2026-11-23: audio deep learning foundation suite (Norm, Softmax, nonlinear reductions, broadcast BMM, strict-FP32 SDPA, ConvTranspose1d/2d, Pad/Cat, liveness reuse, 2D CPU fusion, AdamW, and Checkpoints) | 3–91 | Read linked contracts and raw reports for scope and limits |
| 2026-11-20: differentiable strided STFT, reflect-centered STFT, and NOLA inverse STFT with whole-step fusion and CUDA Graph capture | 93–151 | Read linked contracts and raw reports for scope and limits |
| 2026-11-20: general whole-step tape fusion compiler (CPU L1-tiled fusion + CUDA NVRTC strict-FP32 fusion + CUDA Graph replay) | 153–225 | Read linked contracts and raw reports for scope and limits |
| 2026-11-20: Intel oneMKL VML transcendentals, multi-SM CUDA split reductions, and guarded CUDA Graph step capture/replay | 227–288 | Read linked contracts and raw reports for scope and limits |
| 2026-11-20: bit-exact CPU/CUDA small-axis reduction, broadcast-backward, and pointwise indexing optimization | 290–332 | Read linked contracts and raw reports for scope and limits |
| 2026-11-20: prepared strided 1D/2D convolutions, local VJPs and tape training verified | 334–359 | Read linked contracts and raw reports for scope and limits |
| 2026-10-20: prepared CPU backward dispatch passes fixed confirmation | 361–390 | Read linked contracts and raw reports for scope and limits |
| 2026-10-20: isolated CPU backward entries fail screening admission | 392–419 | Read linked contracts and raw reports for scope and limits |
| 2026-10-20: CPU backward gains leave the tiny guard unresolved | 421–449 | Read linked contracts and raw reports for scope and limits |
| 2026-10-20: complete-step baseline exposes CPU and GPU foundation gaps | 451–483 | Read linked contracts and raw reports for scope and limits |
| 2026-10-20: complete-step training characterization enters harness verification | 485–498 | Read linked contracts and raw reports for scope and limits |
| 2026-10-20: native tape and procedural training pass frozen acceptance | 500–528 | Read linked contracts and raw reports for scope and limits |
| 2026-10-20: native reverse-mode tapes and checked SGD enter composed verification | 530–558 | Read linked contracts and raw reports for scope and limits |
| 2026-10-20: strided matrices, both local VJPs and primitive updates verified | 560–583 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: prepared strided matrices and local VJPs enter frozen verification | 585–600 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: year-one admission and performance qualification made explicit | 602–615 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: full pointwise VJPs and expanded layout gates verified | 617–638 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: pointwise gradient layout coverage expanded after initial verification | 640–658 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: full pointwise local gradients enter frozen verification | 660–677 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: foundation priorities and performance readiness clarified | 679–693 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: strided Sum/Mean and local adjoints verified on CPU/CUDA | 695–724 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: ordered reductions and local derivatives enter frozen verification | 726–744 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: complete pointwise baselines expose nonlinear and layout gaps | 746–775 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: compiled strided writeback passes both hardware harnesses | 777–797 | Read linked contracts and raw reports for scope and limits |
| 2026-10-19: pointwise baseline validation preserves a compiler capability failure | 799–815 | Read linked contracts and raw reports for scope and limits |
| 2026-10-18: general strided unary/binary forward operators verified | 817–849 | Read linked contracts and raw reports for scope and limits |
| 2026-10-18: first prepared CPU/CUDA backward primitive verified | 851–882 | Read linked contracts and raw reports for scope and limits |
| 2026-10-18: retained host/device values and guarded shared storage verified | 884–910 | Read linked contracts and raw reports for scope and limits |
| 2026-10-18: opt-in tracked ownership and mutation versions verified | 912–937 | Read linked contracts and raw reports for scope and limits |
| 2026-10-18: borrowed-layout evidence and source recovery completed | 939–960 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: signed-stride layouts and borrowed host/device views | 962–986 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: explicit cuDNN convolution selection policy | 988–1015 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: first year reserved for framework foundations | 1017–1037 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: prepared mapped addressing and synthesis inventory | 1039–1081 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: audio application priorities confirmed | 1083–1089 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: complete-model latency and memory characterization | 1091–1142 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: complete native GTCRN inference | 1144–1196 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: exact model windows and spatial convolution | 1198–1238 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: real-model reference and prepared GRU | 1240–1281 | Read linked contracts and raw reports for scope and limits |
| 2026-10-17: centered audio reconstruction | 1283–1324 | Read linked contracts and raw reports for scope and limits |
| 2026-10-16: Intel long-window follow-up | 1326–1349 | Read linked contracts and raw reports for scope and limits |
| 2026-10-16: prepared convolution campaign | 1351–1410 | Read linked contracts and raw reports for scope and limits |
| 2026-10-16: initial foundation | 1412–1472 | Read linked contracts and raw reports for scope and limits |

## Logical Sections

| Lines | Content |
| --- | --- |
| 3–583 | Audio deep learning foundation suite (Norm/Softmax/Max/Min/LogSumExp/BMM/SDPA/ConvTranspose/Pad/Cat/liveness reuse/2D CPU fusion/AdamW/Checkpoints), differentiable STFT/CenteredSTFT/iSTFT, whole-step tape fusion compiler, MKL VML/CUDA Graph optimization, bit-exact reduction/backward optimization, verified 1D/2D convolution VJPs, prepared CPU dispatch, native training measurements, tape verification and dense derivatives |
| 585–775 | Foundation admission, pointwise derivatives, reductions and their measured gaps |
| 777–937 | Forward math, view adjoints, retained ownership and tracked versions |
| 939–1037 | Borrowed layouts, repeatability policy, first-year boundary and retained synthesis research |
| 1039–1324 | Earlier priorities and frozen model/GRU/spectral regression evidence |
| 1326–1410 | Original optimization confirmation and unresolved Intel guard |
| 1412–1472 | Initial storage, vendor plans, hardware baselines and Git setup |
| 1474–1552 | A0 prototype correctness, latency, and remaining qualification |
| 1554–1635 | BF16 fallback training and forward attention, and the GTCRN CPU inference comparison |
| 1637–1656 | Direction adjustment refocusing Phases A–D strictly on comprehensive PyTorch leadership |
