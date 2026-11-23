//! Deterministic counter-based `Philox4x32-10` Dropout (bit-identical masks on CPU and CUDA,
//! zero saved mask memory, 100% CUDA Graph capture safe).

use crate::{GradientMode, Layout, Result, Shape, TensorView, TensorViewMut, error::invalid};

/// Validated specification for inverted `Dropout` with counter-based Philox4x32-10 RNG.
#[derive(Clone, Debug, PartialEq)]
pub struct DropoutSpec {
    shape: Shape,
    p: f32,
    seed: u64,
}

impl DropoutSpec {
    pub fn new(shape: &Shape, p: f32, seed: u64) -> Result<Self> {
        if !p.is_finite() || !(0.0..1.0).contains(&p) {
            return Err(invalid("dropout probability p must lie in [0.0, 1.0)"));
        }
        Ok(Self {
            shape: shape.clone(),
            p,
            seed,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn p(&self) -> f32 {
        self.p
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn inv_keep_prob(&self) -> f32 {
        1.0f32 / (1.0f32 - self.p)
    }
}

/// Exact Salmon et al. (2011) `Philox4x32-10` stateless block generator.
///
/// Produces four uniform `u32` words from `(counter, key)`. Identical bit-for-bit to the CUDA
/// device implementation in `native/audio_ops.cu` and `native/bf16_tc.cu`.
#[inline]
pub fn philox4x32_10(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    const M0: u64 = 0xD251_1F53;
    const M1: u64 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;

    for _ in 0..10 {
        let p0 = M0.wrapping_mul(ctr[0] as u64);
        let p1 = M1.wrapping_mul(ctr[2] as u64);
        let hi0 = (p0 >> 32) as u32;
        let lo0 = p0 as u32;
        let hi1 = (p1 >> 32) as u32;
        let lo1 = p1 as u32;
        ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
        key[0] = key[0].wrapping_add(W0);
        key[1] = key[1].wrapping_add(W1);
    }
    ctr
}

/// Maps a uniform `u32` draw to `[0.0, 1.0)` identically on CPU and CUDA (`24`-bit mantissa).
#[inline]
pub fn u32_to_unit_f32(word: u32) -> f32 {
    ((word >> 8) as f32) * (1.0f32 / 16_777_216.0f32)
}

/// Prepared deterministic CPU `Dropout` forward plan (zero hot allocations).
pub struct DropoutPlan {
    spec: DropoutSpec,
    input_layout: Layout,
    output_layout: Layout,
}

impl DropoutPlan {
    pub fn new(spec: &DropoutSpec, input: &Layout, output: &Layout) -> Result<Self> {
        if input.shape() != spec.shape() || output.shape() != spec.shape() {
            return Err(invalid("dropout layout shape differs from spec"));
        }
        if !input.is_contiguous() || !output.is_contiguous() {
            return Err(crate::Error::Unsupported(
                "dropout requires contiguous layouts".into(),
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            input_layout: input.clone(),
            output_layout: output.clone(),
        })
    }

    pub fn spec(&self) -> &DropoutSpec {
        &self.spec
    }

    pub fn run(
        &self,
        input: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        step: u64,
    ) -> Result<()> {
        if input.layout() != &self.input_layout || output.layout() != &self.output_layout {
            return Err(invalid("dropout layouts differ from prepared plan"));
        }
        let n = self.spec.shape().len();
        let x_off = input.layout().offset();
        let y_off = output.layout().offset();
        let x = &input.data[x_off..x_off + n];
        let y = &mut output.data[y_off..y_off + n];
        if self.spec.p == 0.0 {
            y.copy_from_slice(x);
            return Ok(());
        }
        let p = self.spec.p;
        let scale = self.spec.inv_keep_prob();
        let key = [self.spec.seed as u32, (self.spec.seed >> 32) as u32];
        let step_lo = step as u32;
        let step_hi = (step >> 32) as u32;

        let mut i = 0usize;
        while i < n {
            let block = (i >> 2) as u64;
            let r = philox4x32_10([block as u32, (block >> 32) as u32, step_lo, step_hi], key);
            let rem = (n - i).min(4);
            for (k, &word) in r.iter().enumerate().take(rem) {
                y[i + k] = if u32_to_unit_f32(word) >= p {
                    x[i + k] * scale
                } else {
                    0.0
                };
            }
            i += 4;
        }
        Ok(())
    }
}

/// Prepared deterministic CPU `Dropout` local VJP (`input_required() == false`, zero saved mask).
pub struct DropoutBackward {
    spec: DropoutSpec,
    upstream_layout: Layout,
    output_layout: Layout,
}

impl DropoutBackward {
    pub fn new(spec: &DropoutSpec, upstream: &Layout, output: &Layout) -> Result<Self> {
        if upstream.shape() != spec.shape() || output.shape() != spec.shape() {
            return Err(invalid("dropout backward layout shape differs from spec"));
        }
        if !upstream.is_contiguous() || !output.is_contiguous() {
            return Err(crate::Error::Unsupported(
                "dropout backward requires contiguous layouts".into(),
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            upstream_layout: upstream.clone(),
            output_layout: output.clone(),
        })
    }

    pub fn spec(&self) -> &DropoutSpec {
        &self.spec
    }
    pub fn input_required(&self) -> bool {
        false
    }

    pub fn run(
        &self,
        upstream: &TensorView<'_>,
        output: &mut TensorViewMut<'_>,
        step: u64,
        mode: GradientMode,
    ) -> Result<()> {
        if upstream.layout() != &self.upstream_layout || output.layout() != &self.output_layout {
            return Err(invalid(
                "dropout backward layouts differ from prepared plan",
            ));
        }
        let n = self.spec.shape().len();
        let dy_off = upstream.layout().offset();
        let dx_off = output.layout().offset();
        let dy = &upstream.data[dy_off..dy_off + n];
        let dx = &mut output.data[dx_off..dx_off + n];
        let acc = mode == GradientMode::Accumulate;
        if self.spec.p == 0.0 {
            if acc {
                for (d, &u) in dx.iter_mut().zip(dy) {
                    *d += u;
                }
            } else {
                dx.copy_from_slice(dy);
            }
            return Ok(());
        }
        let p = self.spec.p;
        let scale = self.spec.inv_keep_prob();
        let key = [self.spec.seed as u32, (self.spec.seed >> 32) as u32];
        let step_lo = step as u32;
        let step_hi = (step >> 32) as u32;

        let mut i = 0usize;
        while i < n {
            let block = (i >> 2) as u64;
            let r = philox4x32_10([block as u32, (block >> 32) as u32, step_lo, step_hi], key);
            let rem = (n - i).min(4);
            for (k, &word) in r.iter().enumerate().take(rem) {
                let v = if u32_to_unit_f32(word) >= p {
                    dy[i + k] * scale
                } else {
                    0.0
                };
                if acc {
                    dx[i + k] += v;
                } else {
                    dx[i + k] = v;
                }
            }
            i += 4;
        }
        Ok(())
    }
}
