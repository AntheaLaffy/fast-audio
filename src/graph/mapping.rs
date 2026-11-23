use super::Kernel;

/// Coalesce only adjacent coordinates whose source addresses are contiguous.
/// Zero strides represent broadcasting; preparation owns all shape arithmetic.
pub(crate) struct Mapping {
    dims: [usize; 4],
    strides: [usize; 4],
    offset: usize,
    len: usize,
    multiply: bool,
}

impl Mapping {
    pub(crate) fn new(k: Kernel) -> Self {
        debug_assert!(matches!(k.kind, 0 | 3));
        let mut dims = [1; 4];
        let mut strides = [0; 4];
        let mut rank = 0;
        for a in 0..k.rank as usize {
            let d = k.dims[a] as usize;
            let s = k.strides[a] as usize;
            if d == 1 {
                continue;
            }
            if rank > 0 && s.checked_mul(d) == Some(strides[rank - 1]) {
                dims[rank - 1] *= d;
                strides[rank - 1] = s;
            } else {
                dims[rank] = d;
                strides[rank] = s;
                rank += 1;
            }
        }
        // Left padding makes the inner row and all three outer loops unconditional.
        dims.rotate_right(4 - rank);
        strides.rotate_right(4 - rank);
        Self {
            dims,
            strides,
            offset: k.offset as usize,
            len: k.len as usize,
            multiply: k.kind == 3,
        }
    }

    pub(crate) fn run(&self, x: &[f32], y: Option<&[f32]>, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.len);
        let [a, b, c, n] = self.dims;
        let [sa, sb, sc, step] = self.strides;
        let mut rows = out.chunks_exact_mut(n);
        let mut at = 0;
        for i in 0..a {
            for j in 0..b {
                let base = self.offset + i * sa + j * sb;
                for k in 0..c {
                    let source = base + k * sc;
                    let dest = rows.next().unwrap();
                    if self.multiply {
                        let rhs = y.unwrap();
                        let lhs = &x[at..at + n];
                        if step == 0 {
                            let factor = rhs[source];
                            for (d, &v) in dest.iter_mut().zip(lhs) {
                                *d = v * factor;
                            }
                        } else {
                            for ((d, &v), &r) in dest
                                .iter_mut()
                                .zip(lhs)
                                .zip(rhs[source..].iter().step_by(step))
                            {
                                *d = v * r;
                            }
                        }
                    } else if step == 1 {
                        dest.copy_from_slice(&x[source..source + n]);
                    } else if step == 0 {
                        dest.fill(x[source]);
                    } else {
                        for (d, &v) in dest.iter_mut().zip(x[source..].iter().step_by(step)) {
                            *d = v;
                        }
                    }
                    at += n;
                }
            }
        }
    }

    #[cfg(any(feature = "cudnn", test))]
    pub(crate) fn cuda32(&self) -> Option<Map32> {
        let mut dims = [1; 4];
        let mut strides = [0; 4];
        for a in 0..4 {
            dims[a] = u32::try_from(self.dims[a]).ok()?;
            strides[a] = u32::try_from(self.strides[a]).ok()?;
        }
        let max = self
            .dims
            .iter()
            .zip(self.strides)
            .try_fold(self.offset, |at, (&d, s)| {
                at.checked_add((d - 1).checked_mul(s)?)
            })?;
        u32::try_from(max).ok()?;
        Some(Map32 {
            dims,
            strides,
            reciprocal: dims.map(|d| {
                if d == 1 {
                    0
                } else {
                    ((1u64 << 32) / d as u64) as u32
                }
            }),
            len: u32::try_from(self.len).ok()?,
            offset: u32::try_from(self.offset).ok()?,
            multiply: u32::from(self.multiply),
        })
    }
}

#[cfg(any(feature = "cudnn", test))]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Map32 {
    dims: [u32; 4],
    strides: [u32; 4],
    reciprocal: [u32; 4],
    len: u32,
    offset: u32,
    multiply: u32,
}
#[cfg(any(feature = "cudnn", test))]
const _: [(); 60] = [(); size_of::<Map32>()];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_indices_match_integer_division_at_extremes_and_random_inputs() {
        let divisors = [
            2,
            3,
            5,
            7,
            17,
            31,
            32,
            33,
            257,
            65535,
            65536,
            (1u32 << 31) - 1,
            1u32 << 31,
            u32::MAX,
        ];
        let mut state = 0x531714b9u32;
        for d in divisors {
            let m = ((1u64 << 32) / d as u64) as u32;
            for n in [0, 1, d - 1, d, d.saturating_add(1), u32::MAX - 1, u32::MAX]
                .into_iter()
                .chain((0..1000).map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state
                }))
            {
                let mut q = ((n as u64 * m as u64) >> 32) as u32;
                let mut r = n - q * d;
                if r >= d {
                    q += 1;
                    r -= d;
                }
                assert_eq!((q, r), (n / d, n % d), "n={n}, d={d}");
            }
        }
    }

    #[test]
    fn cuda_mapping_bounds_reject_large_source_or_output_without_allocating() {
        let mut m = Mapping {
            dims: [1, 1, 1, 2],
            strides: [0, 0, 0, 1],
            offset: u32::MAX as usize - 1,
            len: 2,
            multiply: false,
        };
        assert!(m.cuda32().is_some());
        m.offset += 1;
        assert!(m.cuda32().is_none());
        m.offset = 0;
        m.len = u32::MAX as usize + 1;
        assert!(m.cuda32().is_none());
        m.len = 2;
        m.strides[3] = u32::MAX as usize + 1;
        assert!(m.cuda32().is_none());
    }
}
