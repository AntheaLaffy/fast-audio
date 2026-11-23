use crate::{Error, Layout, Result, TensorView, TensorViewMut, error::invalid};

pub(crate) struct MappingSpec {
    pub source: Layout,
    pub destination: Layout,
    pub dims: Vec<usize>,
    pub source_strides: Vec<isize>,
    pub destination_strides: Vec<isize>,
    pub len: usize,
    pub inner: usize,
    pub rows: usize,
    pub contiguous: bool,
}
impl MappingSpec {
    pub fn new(source: &Layout, destination: &Layout) -> Result<Self> {
        if source.shape() != destination.shape() {
            return Err(invalid("mapped layouts must have the same logical shape"));
        }
        let len = source.shape().len();
        let mut axes: Vec<(usize, isize, isize)> = Vec::new();
        if len > 0 {
            for ((&dim, &x), &y) in source
                .shape()
                .dims()
                .iter()
                .zip(source.strides())
                .zip(destination.strides())
                .rev()
            {
                if dim == 1 {
                    continue;
                }
                if let Some((inner, sx, sy)) = axes.last_mut() {
                    if sx.checked_mul(*inner as isize) == Some(x)
                        && sy.checked_mul(*inner as isize) == Some(y)
                    {
                        *inner *= dim;
                        continue;
                    }
                }
                axes.push((dim, x, y));
            }
            axes.reverse();
        }
        let inner = if len == 0 {
            0
        } else {
            axes.last().map_or(1, |a| a.0)
        };
        Ok(Self {
            source: source.clone(),
            destination: destination.clone(),
            dims: axes.iter().map(|a| a.0).collect(),
            source_strides: axes.iter().map(|a| a.1).collect(),
            destination_strides: axes.iter().map(|a| a.2).collect(),
            len,
            inner,
            rows: if len == 0 { 0 } else { len / inner },
            contiguous: source.is_contiguous() && destination.is_contiguous(),
        })
    }
    pub fn for_copy(source: &Layout, destination: &Layout) -> Result<Self> {
        if source.shape() != destination.shape() {
            return Err(invalid("copy layouts must have the same logical shape"));
        }
        if !destination.is_non_overlapping() {
            return Err(Error::Unsupported(
                "copy destination must be proved non-overlapping".into(),
            ));
        }
        Self::new(source, destination)
    }
    pub fn metadata_bytes(&self) -> usize {
        (self.source.shape().dims().len() + self.destination.shape().dims().len())
            * (size_of::<usize>() + size_of::<isize>())
            + self.dims.capacity() * size_of::<usize>()
            + (self.source_strides.capacity() + self.destination_strides.capacity())
                * size_of::<isize>()
    }
    pub fn element_offsets(&self, mut flat: usize) -> (usize, usize) {
        if self.contiguous {
            return (
                self.source.offset() + flat,
                self.destination.offset() + flat,
            );
        }
        let (mut x, mut y) = (
            self.source.offset() as isize,
            self.destination.offset() as isize,
        );
        for axis in (1..self.dims.len()).rev() {
            let coordinate = (flat % self.dims[axis]) as isize;
            flat /= self.dims[axis];
            x += coordinate * self.source_strides[axis];
            y += coordinate * self.destination_strides[axis];
        }
        if !self.dims.is_empty() {
            x += flat as isize * self.source_strides[0];
            y += flat as isize * self.destination_strides[0];
        }
        (x as usize, y as usize)
    }
    pub fn validate(&self, source: &Layout, destination: &Layout) -> Result<()> {
        if !(std::ptr::eq(source, &self.source) || source == &self.source)
            || !(std::ptr::eq(destination, &self.destination) || destination == &self.destination)
        {
            Err(invalid(
                "copy view geometry differs from the prepared layouts",
            ))
        } else {
            Ok(())
        }
    }
    pub(crate) fn row_offsets(&self, mut row: usize) -> (isize, isize) {
        let (mut x, mut y) = (
            self.source.offset() as isize,
            self.destination.offset() as isize,
        );
        for axis in (0..self.dims.len().saturating_sub(1)).rev() {
            let coordinate = (row % self.dims[axis]) as isize;
            row /= self.dims[axis];
            x += coordinate * self.source_strides[axis];
            y += coordinate * self.destination_strides[axis];
        }
        (x, y)
    }
}

/// Prepared host data movement. Row traversal and execution do not allocate scratch.
///
/// ```
/// use fast_audio::{Layout, Tensor, cpu};
/// let source = Tensor::from_vec([2, 3], vec![0., 1., 2., 3., 4., 5.])?;
/// let transpose = Layout::contiguous([2, 3])?.permute(&[1, 0])?;
/// let destination = Layout::contiguous([3, 2])?;
/// let plan = cpu::CopyPlan::new(&transpose, &destination)?;
/// let mut output = Tensor::zeros([3, 2])?;
/// plan.run(&source.view(plan.source_layout())?,
///          &mut output.view_mut(plan.destination_layout())?)?;
/// assert_eq!(output.data(), &[0., 3., 1., 4., 2., 5.]);
/// # Ok::<(), fast_audio::Error>(())
/// ```
pub struct CopyPlan {
    spec: MappingSpec,
}
impl CopyPlan {
    pub fn new(source: &Layout, destination: &Layout) -> Result<Self> {
        Ok(Self {
            spec: MappingSpec::for_copy(source, destination)?,
        })
    }
    pub fn source_layout(&self) -> &Layout {
        &self.spec.source
    }
    pub fn destination_layout(&self) -> &Layout {
        &self.spec.destination
    }
    pub fn workspace_bytes(&self) -> usize {
        0
    }
    pub fn run(&self, source: &TensorView<'_>, destination: &mut TensorViewMut<'_>) -> Result<()> {
        self.spec.apply::<false>(source, destination)
    }
}
impl MappingSpec {
    pub(crate) fn apply<const ADD: bool>(
        &self,
        source: &TensorView<'_>,
        destination: &mut TensorViewMut<'_>,
    ) -> Result<()> {
        let spec = self;
        spec.validate(source.layout, destination.layout)?;
        if spec.len == 0 {
            return Ok(());
        }
        if spec.contiguous {
            let x = spec.source.offset();
            let y = spec.destination.offset();
            if ADD {
                for (out, &value) in destination.data[y..y + spec.len]
                    .iter_mut()
                    .zip(&source.data[x..x + spec.len])
                {
                    *out += value;
                }
            } else {
                destination.data[y..y + spec.len].copy_from_slice(&source.data[x..x + spec.len]);
            }
            return Ok(());
        }
        let sx = spec.source_strides.last().copied().unwrap_or(1);
        let sy = spec.destination_strides.last().copied().unwrap_or(1);
        for row in 0..spec.rows {
            let (x, y) = spec.row_offsets(row);
            if sx == 1 && sy == 1 {
                if ADD {
                    for (out, &value) in destination.data[y as usize..][..spec.inner]
                        .iter_mut()
                        .zip(&source.data[x as usize..][..spec.inner])
                    {
                        *out += value;
                    }
                } else {
                    destination.data[y as usize..][..spec.inner]
                        .copy_from_slice(&source.data[x as usize..][..spec.inner]);
                }
            } else if sx == 0 && sy == 1 {
                if ADD {
                    for out in &mut destination.data[y as usize..][..spec.inner] {
                        *out += source.data[x as usize];
                    }
                } else {
                    destination.data[y as usize..][..spec.inner].fill(source.data[x as usize]);
                }
            } else {
                for column in 0..spec.inner {
                    // SAFETY: bound views validated complete spans; coalescing preserves
                    // every original coordinate. The exclusive destination borrow cannot
                    // alias the shared source, and destination indices are injective.
                    unsafe {
                        let input = source.data.as_ptr().offset(x + column as isize * sx);
                        let output = destination
                            .data
                            .as_mut_ptr()
                            .offset(y + column as isize * sy);
                        if ADD {
                            output.write(output.read() + input.read());
                        } else {
                            output.write(input.read());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
