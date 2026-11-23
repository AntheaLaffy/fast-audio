use crate::{Error, Result, error::invalid};

pub(crate) const MAX_ELEMENTS: usize = isize::MAX as usize / size_of::<f32>();

/// Logical FP32 shape. Rank zero is a scalar; any zero-sized axis is empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: Box<[usize]>,
    len: usize,
}

impl Shape {
    pub fn new(dims: impl AsRef<[usize]>) -> Result<Self> {
        let dims = dims.as_ref();
        let len = if dims.contains(&0) {
            0
        } else {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|&n| n <= MAX_ELEMENTS)
                .ok_or_else(|| invalid("tensor byte size overflows addressable memory"))?
        };
        Ok(Self {
            dims: dims.into(),
            len,
        })
    }
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub(crate) fn expect(&self, dims: &[usize]) -> Result<()> {
        if self.dims() == dims {
            Ok(())
        } else {
            Err(invalid(format!(
                "expected shape {dims:?}, got {:?}",
                self.dims()
            )))
        }
    }
}

/// Owned, initialized, contiguous FP32 host storage. Views borrow an explicit layout.
#[derive(Clone, Debug)]
pub struct Tensor {
    shape: Shape,
    data: Vec<f32>,
}

impl Tensor {
    pub fn from_vec(dims: impl AsRef<[usize]>, data: Vec<f32>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.len() {
            return Err(invalid("data length does not match shape"));
        }
        Ok(Self { shape, data })
    }
    pub fn zeros(dims: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let mut data = Vec::new();
        data.try_reserve_exact(shape.len())
            .map_err(|e| Error::Allocation(e.to_string()))?;
        data.resize(shape.len(), 0.0);
        Ok(Self { shape, data })
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }
    pub fn view<'a>(&'a self, layout: &'a crate::Layout) -> Result<crate::TensorView<'a>> {
        crate::TensorView::from_slice(&self.data, layout)
    }
    pub fn view_mut<'a>(
        &'a mut self,
        layout: &'a crate::Layout,
    ) -> Result<crate::TensorViewMut<'a>> {
        crate::TensorViewMut::from_slice(&mut self.data, layout)
    }
}

pub(crate) fn matmul_dims(a: &Shape, b: &Shape, out: &Shape) -> Result<(usize, usize, usize)> {
    let [m, k] = *a.dims() else {
        return Err(invalid("matmul A must be rank 2"));
    };
    let [bk, n] = *b.dims() else {
        return Err(invalid("matmul B must be rank 2"));
    };
    if k != bk {
        return Err(invalid("matmul contraction dimensions differ"));
    }
    out.expect(&[m, n])?;
    Ok((m, n, k))
}
