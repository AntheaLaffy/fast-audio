//! Explicit storage data types, IEEE-754 round-to-nearest-even FP16/BF16 conversions,
//! execution precision policies, and typed contiguous host tensors (`Bf16Tensor`, `IndexTensor`).

use crate::{Error, Result, Shape, Tensor, error::invalid};

/// Explicit element storage data type (no silent implicit precision conversions).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DType {
    #[default]
    F32 = 0,
    Bf16 = 1,
    F16 = 2,
    I64 = 3,
}

impl DType {
    /// Element storage size in bytes.
    #[inline]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 | Self::F16 => 2,
            Self::I64 => 8,
        }
    }
}

/// Explicit opt-in hardware execution precision policy.
///
/// Defaults to [`PrecisionPolicy::StrictFp32`] so all existing strict-FP32 paths remain
/// bit-for-bit unaffected unless a caller explicitly requests Tensor Core acceleration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum PrecisionPolicy {
    /// Strict IEEE-754 FP32 throughout (TF32 disabled, no implicit downcasts).
    #[default]
    StrictFp32 = 0,
    /// Opt-in TF32 Tensor Core math for FP32 GEMM/BMM/Conv (FP32 storage and accumulation).
    Tf32 = 1,
    /// Opt-in BF16 Tensor Core math with FP32 accumulation and FP32 master weights.
    Bf16Mixed = 2,
}

/// Exact IEEE-754 round-to-nearest-even (`RNE`) conversion from `f32` to `bfloat16` (`u16` bits).
///
/// Matches CUDA `__float2bfloat16_rn` and PyTorch `tensor.to(torch.bfloat16)` across normal,
/// subnormal, zero, infinity, and quiet NaN values.
#[inline]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let exp = bits & 0x7f80_0000;
    let frac = bits & 0x007f_ffff;
    if exp == 0x7f80_0000 {
        if frac != 0 {
            // Preserve quiet NaN payload and ensure the quiet bit is set.
            return ((bits >> 16) as u16) | 0x0040;
        }
        return (bits >> 16) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x0000_7fff + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Exact lossless widening from `bfloat16` (`u16` bits) to `f32`.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Exact IEEE-754 round-to-nearest-even (`RNE`) conversion from `f32` to IEEE binary16 (`f16` bits).
///
/// Matches CUDA `__float2half_rn` and PyTorch `tensor.to(torch.float16)` across normal,
/// subnormal, overflow-to-infinity, signed-zero, and quiet NaN values.
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 255 {
        if frac != 0 {
            // Quiet NaN with preserved upper payload bits.
            let payload = (frac >> 13) as u16;
            return sign | 0x7e00 | payload;
        }
        return sign | 0x7c00;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow to signed infinity.
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        // Subnormal or underflow to signed zero.
        if unbiased < -25 {
            return sign;
        }
        let mant = 0x0080_0000u32 | frac;
        let shift = (-14 - unbiased) as u32 + 13; // in 14..=24
        let half = 1u32 << (shift - 1);
        let mask = (1u32 << shift) - 1;
        let rem = mant & mask;
        let mut out = mant >> shift;
        if rem > half || (rem == half && (out & 1) != 0) {
            out += 1;
        }
        return sign | (out as u16);
    }

    let f16_exp = (unbiased + 15) as u32;
    let rem = frac & 0x1fff;
    let mut f16_mant = frac >> 13;
    if rem > 0x1000 || (rem == 0x1000 && (f16_mant & 1) != 0) {
        f16_mant += 1;
        if f16_mant == 0x0400 {
            // Mantissa carried into exponent.
            let next_exp = f16_exp + 1;
            if next_exp >= 31 {
                return sign | 0x7c00;
            }
            return sign | ((next_exp as u16) << 10);
        }
    }
    sign | ((f16_exp as u16) << 10) | (f16_mant as u16)
}

/// Exact lossless widening from IEEE binary16 (`f16` bits) to `f32`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        // Normalize subnormal f16 mantissa.
        let shift = frac.leading_zeros() - 21; // frac < 1024 -> leading_zeros in 22..=31 -> shift in 1..=10
        let mant = (frac << shift) & 0x03ff;
        let f32_exp = 127 - 14 - shift;
        return f32::from_bits(sign | (f32_exp << 23) | (mant << 13));
    }
    if exp == 31 {
        if frac == 0 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        return f32::from_bits(sign | 0x7fc0_0000 | (frac << 13));
    }
    let f32_exp = exp + (127 - 15);
    f32::from_bits(sign | (f32_exp << 23) | (frac << 13))
}

/// Owned, initialized, contiguous `bfloat16` host tensor (`u16` bit storage, 2 bytes/element).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bf16Tensor {
    shape: Shape,
    data: Vec<u16>,
}

impl Bf16Tensor {
    pub fn from_bits(dims: impl AsRef<[usize]>, data: Vec<u16>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.len() {
            return Err(invalid("BF16 data length does not match shape"));
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(dims: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let mut data = Vec::new();
        data.try_reserve_exact(shape.len())
            .map_err(|e| Error::Allocation(e.to_string()))?;
        data.resize(shape.len(), 0);
        Ok(Self { shape, data })
    }

    /// Explicitly converts an FP32 [`Tensor`] to `bfloat16` using IEEE round-to-nearest-even.
    pub fn from_f32(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.shape().clone();
        let mut data = Vec::new();
        data.try_reserve_exact(shape.len())
            .map_err(|e| Error::Allocation(e.to_string()))?;
        for &v in tensor.data() {
            data.push(f32_to_bf16(v));
        }
        Ok(Self { shape, data })
    }

    /// Explicitly widens this `bfloat16` tensor to an FP32 [`Tensor`].
    pub fn to_f32(&self) -> Result<Tensor> {
        let mut out = Vec::new();
        out.try_reserve_exact(self.data.len())
            .map_err(|e| Error::Allocation(e.to_string()))?;
        for &b in &self.data {
            out.push(bf16_to_f32(b));
        }
        Tensor::from_vec(self.shape.dims(), out)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        DType::Bf16
    }
    pub fn bits(&self) -> &[u16] {
        &self.data
    }
    pub fn bits_mut(&mut self) -> &mut [u16] {
        &mut self.data
    }
    pub fn into_bits(self) -> Vec<u16> {
        self.data
    }
}

/// Owned, initialized, contiguous `i64` host index tensor for `Embedding`, `CrossEntropy`,
/// and `Gather`/`Scatter` primitives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexTensor {
    shape: Shape,
    data: Vec<i64>,
}

impl IndexTensor {
    pub fn from_vec(dims: impl AsRef<[usize]>, data: Vec<i64>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.len() {
            return Err(invalid("index data length does not match shape"));
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(dims: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let mut data = Vec::new();
        data.try_reserve_exact(shape.len())
            .map_err(|e| Error::Allocation(e.to_string()))?;
        data.resize(shape.len(), 0);
        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        DType::I64
    }
    pub fn data(&self) -> &[i64] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [i64] {
        &mut self.data
    }
    pub fn into_vec(self) -> Vec<i64> {
        self.data
    }
}
