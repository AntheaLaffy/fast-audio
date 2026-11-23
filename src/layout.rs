use crate::{Error, Result, Shape, error::invalid, tensor::MAX_ELEMENTS};
use std::{iter::FusedIterator, ops::Range};

/// Prepared signed element strides over initialized storage, independent of its base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Box<[isize]>,
    offset: usize,
    span: Option<(usize, usize)>,
    contiguous: bool,
    non_overlapping: bool,
}
impl Layout {
    pub fn contiguous(dims: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let mut strides = vec![0; shape.dims().len()];
        if !shape.is_empty() {
            let mut step = 1;
            for axis in (0..strides.len()).rev() {
                strides[axis] = step as isize;
                step *= shape.dims()[axis];
            }
        }
        Self::make(shape, strides, 0)
    }
    pub fn from_strides(
        dims: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        offset: usize,
    ) -> Result<Self> {
        Self::make(Shape::new(dims)?, strides.as_ref().to_vec(), offset)
    }
    fn make(shape: Shape, strides: Vec<isize>, offset: usize) -> Result<Self> {
        if strides.len() != shape.dims().len() || offset > MAX_ELEMENTS {
            return Err(invalid("layout rank or storage origin is invalid"));
        }
        let span = if shape.is_empty() {
            None
        } else {
            let (mut low, mut high) = (offset as isize, offset as isize);
            for (&dim, &stride) in shape.dims().iter().zip(&strides) {
                if dim <= 1 {
                    continue;
                }
                let delta = stride
                    .checked_mul((dim - 1) as isize)
                    .ok_or_else(|| invalid("layout stride extent overflows"))?;
                if delta < 0 {
                    low = low
                        .checked_add(delta)
                        .ok_or_else(|| invalid("layout lower bound overflows"))?;
                } else {
                    high = high
                        .checked_add(delta)
                        .ok_or_else(|| invalid("layout upper bound overflows"))?;
                }
            }
            if low < 0 || high >= MAX_ELEMENTS as isize {
                return Err(invalid("layout span exceeds addressable FP32 storage"));
            }
            Some((low as usize, high as usize + 1))
        };
        let mut expected = 1;
        let mut contiguous = true;
        if !shape.is_empty() {
            for (&dim, &stride) in shape.dims().iter().zip(&strides).rev() {
                if dim > 1 && stride != expected as isize {
                    contiguous = false;
                }
                expected *= dim;
            }
        }
        let non_overlapping = if shape.len() <= 1 {
            true
        } else {
            let mut axes: Vec<_> = shape
                .dims()
                .iter()
                .zip(&strides)
                .filter(|(dim, _)| **dim > 1)
                .map(|(&dim, &stride)| (stride.unsigned_abs(), dim))
                .collect();
            axes.sort_unstable();
            let mut covered = 1usize;
            axes.into_iter().all(|(stride, dim)| {
                if stride < covered {
                    return false;
                }
                if let Some(next) = (dim - 1)
                    .checked_mul(stride)
                    .and_then(|n| covered.checked_add(n))
                {
                    covered = next;
                    true
                } else {
                    false
                }
            })
        };
        Ok(Self {
            shape,
            strides: strides.into_boxed_slice(),
            offset,
            span,
            contiguous,
            non_overlapping,
        })
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn span(&self) -> Option<Range<usize>> {
        self.span.map(|(a, b)| a..b)
    }
    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }
    /// A sufficient injectivity proof. False also includes layouts not proved by it.
    pub fn is_non_overlapping(&self) -> bool {
        self.non_overlapping
    }
    pub(crate) fn validate_storage(&self, len: usize) -> Result<()> {
        if self.span.map_or(self.offset > len, |(_, end)| end > len) {
            Err(invalid("layout exceeds initialized storage length"))
        } else {
            Ok(())
        }
    }
    pub fn index(&self, coordinates: &[usize]) -> Option<usize> {
        if self.shape.is_empty() || coordinates.len() != self.strides.len() {
            return None;
        }
        let mut index = self.offset as isize;
        for ((&coordinate, &dim), &stride) in
            coordinates.iter().zip(self.shape.dims()).zip(&self.strides)
        {
            if coordinate >= dim {
                return None;
            }
            index += coordinate as isize * stride;
        }
        Some(index as usize)
    }
    pub fn linear_index(&self, flat: usize) -> Option<usize> {
        (flat < self.shape.len()).then(|| self.linear_valid(flat))
    }
    pub(crate) fn linear_valid(&self, mut flat: usize) -> usize {
        debug_assert!(flat < self.shape.len());
        if self.contiguous {
            return self.offset + flat;
        }
        let mut index = self.offset as isize;
        for (&dim, &stride) in self.shape.dims().iter().zip(&self.strides).rev() {
            index += (flat % dim) as isize * stride;
            flat /= dim;
        }
        index as usize
    }
    pub fn offsets(&self) -> LayoutOffsets<'_> {
        LayoutOffsets {
            layout: self,
            front: 0,
            back: self.shape.len(),
        }
    }
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.strides.len() {
            return Err(invalid("permutation rank differs"));
        }
        let mut seen = vec![false; axes.len()];
        for &axis in axes {
            if axis >= axes.len() || seen[axis] {
                return Err(invalid("axes must be a permutation"));
            }
            seen[axis] = true;
        }
        let dims: Vec<_> = axes.iter().map(|&i| self.shape.dims()[i]).collect();
        let strides = axes.iter().map(|&i| self.strides[i]).collect();
        Self::make(Shape::new(dims)?, strides, self.offset)
    }
    /// Select `count` elements from `start`, advancing by a nonzero signed `step`.
    pub fn slice(&self, axis: usize, start: usize, count: usize, step: isize) -> Result<Self> {
        let dim = *self
            .shape
            .dims()
            .get(axis)
            .ok_or_else(|| invalid("slice axis is out of range"))?;
        if step == 0 || start > dim || (count > 0 && start == dim) {
            return Err(invalid("invalid slice start or step"));
        }
        if count > 0 {
            let last = (count - 1) as i128 * step as i128 + start as i128;
            if last < 0 || last >= dim as i128 {
                return Err(invalid("slice exceeds its axis"));
            }
        }
        let mut dims = self.shape.dims().to_vec();
        dims[axis] = count;
        let shape = Shape::new(dims)?;
        let mut strides = self.strides.to_vec();
        strides[axis] = match strides[axis].checked_mul(step) {
            Some(value) => value,
            None if shape.is_empty() || count <= 1 => 0,
            None => return Err(invalid("slice stride overflows")),
        };
        let offset = if shape.is_empty() {
            self.offset
        } else {
            let value = self.offset as i128 + start as i128 * self.strides[axis] as i128;
            usize::try_from(value).map_err(|_| invalid("slice origin overflows"))?
        };
        Self::make(shape, strides, offset)
    }
    pub fn broadcast_to(&self, dims: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let rank = shape.dims().len();
        let original = self.shape.dims();
        if original.len() > rank {
            return Err(invalid("broadcast cannot remove axes"));
        }
        let prefix = rank - original.len();
        let mut strides = vec![0; rank];
        for (axis, (&dim, &target)) in original.iter().zip(&shape.dims()[prefix..]).enumerate() {
            if dim != target && dim != 1 {
                return Err(invalid("broadcast dimensions differ"));
            }
            if dim == target {
                strides[axis + prefix] = self.strides[axis];
            }
        }
        Self::make(shape, strides, self.offset)
    }
    /// Reshape stride-contiguous chunks without changing logical element order.
    pub fn reshape(&self, dims: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if shape.len() != self.shape.len() {
            return Err(invalid("reshape element count differs"));
        }
        if shape == self.shape {
            return Ok(self.clone());
        }
        if self.shape.len() <= 1 {
            let layout = Self::contiguous(shape.dims())?;
            return Self::make(shape, layout.strides.into_vec(), self.offset);
        }
        let mut chunks: Vec<(usize, isize)> = Vec::new();
        for (&dim, &stride) in self.shape.dims().iter().zip(&self.strides).rev() {
            if dim == 1 {
                continue;
            }
            if let Some((size, step)) = chunks.last_mut() {
                if step.checked_mul(*size as isize) == Some(stride) {
                    *size *= dim;
                    continue;
                }
            }
            chunks.push((dim, stride));
        }
        let mut axis = shape.dims().len();
        let mut strides = vec![0; axis];
        for (size, step) in chunks {
            let mut product = 1;
            while axis > 0 && (product < size || shape.dims()[axis - 1] == 1) {
                axis -= 1;
                strides[axis] = step
                    .checked_mul(product as isize)
                    .ok_or_else(|| invalid("reshape stride overflows"))?;
                product *= shape.dims()[axis];
            }
            if product != size {
                return Err(Error::Unsupported(
                    "reshape crosses a stride discontinuity; materialize explicitly".into(),
                ));
            }
        }
        if axis != 0 {
            return Err(Error::Unsupported(
                "reshape cannot preserve strided element order".into(),
            ));
        }
        Self::make(shape, strides, self.offset)
    }
}

/// Allocation-free traversal of storage indices in logical row-major order.
#[derive(Clone)]
pub struct LayoutOffsets<'a> {
    layout: &'a Layout,
    front: usize,
    back: usize,
}
impl Iterator for LayoutOffsets<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.front == self.back {
            return None;
        }
        let flat = self.front;
        self.front += 1;
        Some(self.layout.linear_valid(flat))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}
impl DoubleEndedIterator for LayoutOffsets<'_> {
    fn next_back(&mut self) -> Option<usize> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        Some(self.layout.linear_valid(self.back))
    }
}
impl ExactSizeIterator for LayoutOffsets<'_> {}
impl FusedIterator for LayoutOffsets<'_> {}
