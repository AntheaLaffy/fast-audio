use crate::{Error, Layout, LayoutOffsets, Result, Tensor};
use std::{iter::FusedIterator, marker::PhantomData};

/// Read-only borrowed FP32 storage with separately prepared logical geometry.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) layout: &'a Layout,
}
impl<'a> TensorView<'a> {
    pub fn from_slice(data: &'a [f32], layout: &'a Layout) -> Result<Self> {
        layout.validate_storage(data.len())?;
        Ok(Self { data, layout })
    }
    pub fn layout(&self) -> &Layout {
        self.layout
    }
    pub fn get(&self, coordinates: &[usize]) -> Option<&'a f32> {
        self.layout.index(coordinates).map(|i| &self.data[i])
    }
    pub fn as_slice(&self) -> Option<&'a [f32]> {
        self.layout
            .is_contiguous()
            .then(|| &self.data[self.layout.offset()..][..self.layout.shape().len()])
    }
    pub fn iter(&self) -> Elements<'a> {
        Elements {
            data: self.data,
            offsets: self.layout.offsets(),
        }
    }
    /// Explicitly materialize logical element order into independent dense storage.
    pub fn to_tensor(&self) -> Result<Tensor> {
        let mut output = Tensor::zeros(self.layout.shape().dims())?;
        if let Some(source) = self.as_slice() {
            output.data_mut().copy_from_slice(source);
        } else {
            for (out, value) in output.data_mut().iter_mut().zip(self.iter()) {
                *out = *value;
            }
        }
        Ok(output)
    }
}

/// Exclusive borrowed view. Overlapping or unproved layouts cannot bind mutably.
///
/// ```compile_fail
/// use fast_audio::{Layout, Tensor};
/// let layout = Layout::contiguous([4]).unwrap();
/// let mut tensor = Tensor::zeros([4]).unwrap();
/// let view = tensor.view(&layout).unwrap();
/// tensor.data_mut()[0] = 1.0;
/// assert_eq!(view.get(&[0]), Some(&0.0));
/// ```
/// ```compile_fail
/// use fast_audio::{Layout, Tensor};
/// let layout = Layout::contiguous([4]).unwrap();
/// let mut tensor = Tensor::zeros([4]).unwrap();
/// let mut view = tensor.view_mut(&layout).unwrap();
/// let borrowed = view.as_view();
/// *view.get_mut(&[0]).unwrap() = 1.0;
/// assert_eq!(borrowed.get(&[0]), Some(&0.0));
/// ```
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    pub(crate) data: &'a mut [f32],
    pub(crate) layout: &'a Layout,
}
impl<'a> TensorViewMut<'a> {
    pub fn from_slice(data: &'a mut [f32], layout: &'a Layout) -> Result<Self> {
        layout.validate_storage(data.len())?;
        if !layout.is_non_overlapping() {
            return Err(Error::Unsupported(
                "mutable views require a proved non-overlapping layout".into(),
            ));
        }
        Ok(Self { data, layout })
    }
    pub fn layout(&self) -> &Layout {
        self.layout
    }
    pub fn as_view(&self) -> TensorView<'_> {
        TensorView {
            data: self.data,
            layout: self.layout,
        }
    }
    pub fn get_mut(&mut self, coordinates: &[usize]) -> Option<&mut f32> {
        self.layout.index(coordinates).map(|i| &mut self.data[i])
    }
    pub fn as_slice_mut(&mut self) -> Option<&mut [f32]> {
        self.layout
            .is_contiguous()
            .then(|| &mut self.data[self.layout.offset()..][..self.layout.shape().len()])
    }
    pub fn iter_mut(&mut self) -> ElementsMut<'_> {
        ElementsMut {
            base: self.data.as_mut_ptr(),
            offsets: self.layout.offsets(),
            storage: PhantomData,
        }
    }
}
pub struct Elements<'a> {
    data: &'a [f32],
    offsets: LayoutOffsets<'a>,
}
impl<'a> Iterator for Elements<'a> {
    type Item = &'a f32;
    fn next(&mut self) -> Option<Self::Item> {
        self.offsets.next().map(|i| &self.data[i])
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}
impl DoubleEndedIterator for Elements<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.offsets.next_back().map(|i| &self.data[i])
    }
}
impl ExactSizeIterator for Elements<'_> {}
impl FusedIterator for Elements<'_> {}

pub struct ElementsMut<'a> {
    base: *mut f32,
    offsets: LayoutOffsets<'a>,
    storage: PhantomData<&'a mut f32>,
}
impl<'a> Iterator for ElementsMut<'a> {
    type Item = &'a mut f32;
    fn next(&mut self) -> Option<Self::Item> {
        self.offsets.next().map(|index| {
            // SAFETY: binding proved bounds and injectivity over an exclusive slice.
            // Front/back cursors yield each logical element at most once.
            unsafe { &mut *self.base.add(index) }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}
impl DoubleEndedIterator for ElementsMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.offsets.next_back().map(|index| {
            // SAFETY: same unique-address and exclusive-borrow proof as forward iteration;
            // the shared front/back cursor range cannot return an earlier element again.
            unsafe { &mut *self.base.add(index) }
        })
    }
}
impl ExactSizeIterator for ElementsMut<'_> {}
impl FusedIterator for ElementsMut<'_> {}
