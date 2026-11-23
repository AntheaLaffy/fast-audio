# Deferred DAC source inventory

This inventory was collected while synthesis integration was under consideration.
The user's subsequent foundation-first direction defers all new model integration
until after year one. Keep these source/shape facts as research evidence; DAC is not
an active implementation milestone or a dependency of the foundation roadmap.

The reference is [Descript Audio Codec at
`c7cfc5d2647e26471dc394f95846a0830e7bec34`](https://github.com/descriptinc/descript-audio-codec/tree/c7cfc5d2647e26471dc394f95846a0830e7bec34).
Five unchanged files, including MIT attribution, are checksummed and archived in the
[source companion](../benchmarks/results/2026-10-17-dac-artifacts.json).
The [inventory](../benchmarks/results/2026-10-17-dac-inventory.json) records every
default decoder parameter name/shape and the inspected source hashes.

## Verified architecture facts

The original layer module was loaded unchanged. The original `ResidualUnit`,
`DecoderBlock` and `Decoder` class definitions were selected from the pinned AST and
instantiated using real PyTorch modules on the meta device. No external dependency
stubs were used. This isolates the decoder from unrelated codec I/O dependencies.
Meta execution verifies shape propagation, not numerical values or audio quality.

| Default decoder property | Verified value |
| --- | --- |
| Class-default sample rate | 44,100 Hz; checkpoint configuration not yet verified |
| Input latent | `[B,1024,frames]` |
| Initial decoder width | 1,536 channels |
| Upsampling rates | 8, 8, 4, 2 |
| Output | `[B,1,512*frames]` |
| Conv1d / ConvTranspose1d | 26 / 4 modules |
| Snake1d / Tanh | 29 / 1 modules |
| Residual units | 12 |
| Parameter elements | 54,104,162, including weight-normalization `g`/`v` and Snake parameters |

Real meta forwards verify B1 at 1/3/17 latent frames and B2 at 17 frames. The
512-sample hop is an upsampling ratio, approximately 11.61 ms at 44.1 kHz; it does
not establish algorithmic delay, streaming causality or observed execution latency.
The core default decoder is mono. Independent-channel wrappers and jointly trained
stereo synthesis require explicit separate evaluation.

## Deferred integration questions

The inventory identified effective weight-normalization export, per-channel Snake
(`x + reciprocal(alpha + 1e-9) * sin(alpha*x)^2`) and checked transposed-convolution
lowering as possible runtime consumers. It does not establish those capabilities or
queue DAC-specific work for year one. Encoder/quantizer support, integer codes,
checkpoint configuration, stereo behavior and codec I/O also remain unverified.

Any later selection requires a new explicit model/reference/quality contract under
the [roadmap's application stage](roadmap.md#year-one-exit-and-later-applications).
Procedural latents and meta forwards cannot establish musical output quality.

## Recovery

The archived [inventory runner](../benchmarks/results/2026-10-17-dac-inventory.py) accepts
`--source-root` and `--output`. Restore the five source archive members to an empty
directory and write `manifest.json` using the inventory's `repository`, `revision`
and `source_sha256` fields. The runner checks every source hash before meta execution.
The source archive and member checksums have been verified. No model checkpoint was
downloaded, and no real CPU/GPU decoder waveform was computed in this inventory.
