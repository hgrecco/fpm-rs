//! Private representations of validated C-contiguous ndarray storage.
//!
//! These types carry only a layout invariant. Domain-specific shape and value
//! invariants remain the responsibility of their owning model types.
#![allow(dead_code)] // The complete invariant-preserving accessor set is intentional.

use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2};

use crate::{Error, Result};

pub(crate) fn checked_len_2d(shape: (usize, usize)) -> Result<usize> {
    shape
        .0
        .checked_mul(shape.1)
        .ok_or_else(|| Error::ShapeOverflow {
            shape: vec![shape.0, shape.1],
        })
}

pub(crate) fn checked_len_3d(shape: (usize, usize, usize)) -> Result<usize> {
    shape
        .0
        .checked_mul(shape.1)
        .and_then(|length| length.checked_mul(shape.2))
        .ok_or_else(|| Error::ShapeOverflow {
            shape: vec![shape.0, shape.1, shape.2],
        })
}

fn nonstandard_2d<T>(context: &'static str, view: &ArrayView2<'_, T>) -> Error {
    Error::NonStandardLayout {
        context,
        shape: view.shape().to_vec(),
        strides: view.strides().to_vec(),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StandardArray2<T> {
    inner: Array2<T>,
}

impl<T> TryFrom<Array2<T>> for StandardArray2<T> {
    type Error = Error;

    fn try_from(inner: Array2<T>) -> Result<Self> {
        if !inner.is_standard_layout() {
            return Err(nonstandard_2d("owned array", &inner.view()));
        }
        Ok(Self { inner })
    }
}

impl<T> StandardArray2<T> {
    pub(crate) fn from_shape_vec(shape: (usize, usize), data: Vec<T>) -> Result<Self> {
        let expected = checked_len_2d(shape)?;
        if data.len() != expected {
            return Err(Error::LengthMismatch {
                actual: data.len(),
                expected,
                shape,
            });
        }
        Ok(Self {
            inner: Array2::from_shape_vec(shape, data)?,
        })
    }

    pub(crate) fn dim(&self) -> (usize, usize) {
        self.inner.dim()
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub(crate) fn view(&self) -> StandardView2<'_, T> {
        StandardView2 {
            view: self.inner.view(),
        }
    }

    pub(crate) fn view_mut(&mut self) -> StandardViewMut2<'_, T> {
        StandardViewMut2 {
            view: self.inner.view_mut(),
        }
    }

    pub(crate) fn ndarray_view(&self) -> ArrayView2<'_, T> {
        self.inner.view()
    }

    pub(crate) fn ndarray_view_mut(&mut self) -> ArrayViewMut2<'_, T> {
        self.inner.view_mut()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: every constructor establishes standard row-major layout.
        // Such an array represents exactly `len` consecutive initialized
        // elements beginning at `as_ptr()`. No structural mutation is exposed.
        unsafe { std::slice::from_raw_parts(self.inner.as_ptr(), self.inner.len()) }
    }

    pub(crate) fn as_slice_mut(&mut self) -> &mut [T] {
        let length = self.inner.len();
        // SAFETY: as above, with exclusive access to the wrapper ensuring that
        // no mutable alias can be produced while this slice is borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.inner.as_mut_ptr(), length) }
    }

    pub(crate) fn into_inner(self) -> Array2<T> {
        self.inner
    }
}

impl<T: Clone> StandardArray2<T> {
    pub(crate) fn filled(shape: (usize, usize), value: T) -> Result<Self> {
        let length = checked_len_2d(shape)?;
        Self::from_shape_vec(shape, vec![value; length])
    }
}

impl<T: Clone + Default> StandardArray2<T> {
    pub(crate) fn zeros(shape: (usize, usize)) -> Result<Self> {
        Self::filled(shape, T::default())
    }
}

#[derive(Debug)]
pub(crate) struct StandardView2<'a, T> {
    view: ArrayView2<'a, T>,
}

impl<T> Copy for StandardView2<'_, T> {}

impl<T> Clone for StandardView2<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> TryFrom<ArrayView2<'a, T>> for StandardView2<'a, T> {
    type Error = Error;

    fn try_from(view: ArrayView2<'a, T>) -> Result<Self> {
        if !view.is_standard_layout() {
            return Err(nonstandard_2d("array view", &view));
        }
        Ok(Self { view })
    }
}

impl<'a, T> StandardView2<'a, T> {
    pub(crate) fn dim(self) -> (usize, usize) {
        self.view.dim()
    }

    pub(crate) fn len(self) -> usize {
        self.view.len()
    }

    pub(crate) fn as_slice(self) -> &'a [T] {
        // SAFETY: construction validates standard layout, and immutable views
        // cannot change their shape or strides. The returned lifetime is the
        // lifetime of the original ndarray borrow.
        unsafe { std::slice::from_raw_parts(self.view.as_ptr(), self.view.len()) }
    }

    pub(crate) fn ndarray_view(self) -> ArrayView2<'a, T> {
        self.view
    }
}

#[derive(Debug)]
pub(crate) struct StandardViewMut2<'a, T> {
    view: ArrayViewMut2<'a, T>,
}

impl<'a, T> TryFrom<ArrayViewMut2<'a, T>> for StandardViewMut2<'a, T> {
    type Error = Error;

    fn try_from(view: ArrayViewMut2<'a, T>) -> Result<Self> {
        if !view.is_standard_layout() {
            return Err(Error::NonStandardLayout {
                context: "mutable array view",
                shape: view.shape().to_vec(),
                strides: view.strides().to_vec(),
            });
        }
        Ok(Self { view })
    }
}

impl<'a, T> StandardViewMut2<'a, T> {
    pub(crate) fn dim(&self) -> (usize, usize) {
        self.view.dim()
    }

    pub(crate) fn as_slice_mut(&mut self) -> &mut [T] {
        let length = self.view.len();
        // SAFETY: construction validates standard layout. Exclusive access to
        // the wrapper prevents mutable aliasing for the returned borrow.
        unsafe { std::slice::from_raw_parts_mut(self.view.as_mut_ptr(), length) }
    }

    pub(crate) fn into_ndarray_view(self) -> ArrayViewMut2<'a, T> {
        self.view
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StandardArray3<T> {
    inner: Array3<T>,
}

impl<T> TryFrom<Array3<T>> for StandardArray3<T> {
    type Error = Error;

    fn try_from(inner: Array3<T>) -> Result<Self> {
        if !inner.is_standard_layout() {
            return Err(Error::NonStandardLayout {
                context: "owned three-dimensional array",
                shape: inner.shape().to_vec(),
                strides: inner.strides().to_vec(),
            });
        }
        Ok(Self { inner })
    }
}

impl<T> StandardArray3<T> {
    pub(crate) fn from_shape_vec(shape: (usize, usize, usize), data: Vec<T>) -> Result<Self> {
        let expected = checked_len_3d(shape)?;
        if data.len() != expected {
            return Err(Error::ArrayLengthMismatch {
                actual: data.len(),
                expected,
                shape: vec![shape.0, shape.1, shape.2],
            });
        }
        Ok(Self {
            inner: Array3::from_shape_vec(shape, data)?,
        })
    }

    pub(crate) fn dim(&self) -> (usize, usize, usize) {
        self.inner.dim()
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub(crate) fn ndarray_view(&self) -> ndarray::ArrayView3<'_, T> {
        self.inner.view()
    }

    pub(crate) fn ndarray_view_mut(&mut self) -> ndarray::ArrayViewMut3<'_, T> {
        self.inner.view_mut()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: every constructor establishes standard row-major layout.
        unsafe { std::slice::from_raw_parts(self.inner.as_ptr(), self.inner.len()) }
    }

    pub(crate) fn as_slice_mut(&mut self) -> &mut [T] {
        let length = self.inner.len();
        // SAFETY: exclusive wrapper access prevents mutable aliases.
        unsafe { std::slice::from_raw_parts_mut(self.inner.as_mut_ptr(), length) }
    }

    pub(crate) fn into_inner(self) -> Array3<T> {
        self.inner
    }
}

impl<T: Clone> StandardArray3<T> {
    pub(crate) fn filled(shape: (usize, usize, usize), value: T) -> Result<Self> {
        let length = checked_len_3d(shape)?;
        Self::from_shape_vec(shape, vec![value; length])
    }
}

impl<T: Clone + Default> StandardArray3<T> {
    pub(crate) fn zeros(shape: (usize, usize, usize)) -> Result<Self> {
        Self::filled(shape, T::default())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3, ShapeBuilder, array, s};

    use crate::Error;

    use super::{StandardArray2, StandardArray3, StandardView2, StandardViewMut2};

    #[test]
    fn standard_owned_array_preserves_allocation_and_views() {
        let array = array![[1, 2, 3], [4, 5, 6]];
        let pointer = array.as_ptr();
        let mut standard = StandardArray2::try_from(array).unwrap();
        assert_eq!(standard.as_slice(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(standard.ndarray_view().as_ptr(), pointer);
        standard.view_mut().as_slice_mut()[1] = 9;
        assert_eq!(standard.view().as_slice(), &[1, 9, 3, 4, 5, 6]);
        let inner = standard.into_inner();
        assert_eq!(inner.as_ptr(), pointer);
    }

    #[test]
    fn nonstandard_owned_and_borrowed_layouts_are_rejected() {
        let matrix = array![[1, 2, 3, 4], [5, 6, 7, 8]];
        assert!(StandardView2::try_from(matrix.t()).is_err());
        assert!(StandardView2::try_from(matrix.slice(s![.., ..;2])).is_err());
        assert!(StandardView2::try_from(matrix.slice(s![..;-1, ..])).is_err());
        let fortran = Array2::from_shape_vec((2, 4).f(), (0..8).collect()).unwrap();
        assert!(StandardArray2::try_from(fortran).is_err());
    }

    #[test]
    fn mutable_standard_view_updates_the_source() {
        let mut matrix = Array2::zeros((2, 3));
        {
            let mut view = StandardViewMut2::try_from(matrix.view_mut()).unwrap();
            view.as_slice_mut()[4] = 7;
        }
        assert_eq!(matrix[(1, 1)], 7);
    }

    #[test]
    fn shape_construction_is_checked_for_two_and_three_dimensions() {
        assert!(StandardArray2::<u8>::from_shape_vec((2, 3), vec![0; 5]).is_err());
        assert!(StandardArray3::<u8>::from_shape_vec((2, 3, 4), vec![0; 23]).is_err());
        let stack = StandardArray3::from_shape_vec((2, 2, 2), (0..8).collect()).unwrap();
        assert_eq!(stack.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn standard_owned_stack_preserves_allocation_and_rejects_fortran_layout() {
        let stack = Array3::from_shape_vec((2, 2, 2), (0..8).collect()).unwrap();
        let pointer = stack.as_ptr();
        let standard = StandardArray3::try_from(stack).unwrap();
        assert_eq!(standard.ndarray_view().as_ptr(), pointer);
        assert_eq!(standard.dim(), (2, 2, 2));
        assert_eq!(standard.into_inner().as_ptr(), pointer);

        let fortran = Array3::from_shape_vec((2, 2, 2).f(), (0..8).collect()).unwrap();
        assert!(StandardArray3::try_from(fortran).is_err());
    }

    #[test]
    fn nonstandard_owned_stack_layouts_are_rejected() {
        let permuted = Array3::from_shape_vec((2, 3, 4), (0..24).collect())
            .unwrap()
            .permuted_axes([1, 0, 2]);
        assert!(matches!(
            StandardArray3::try_from(permuted),
            Err(Error::NonStandardLayout { .. })
        ));

        let stepped = Array3::from_shape_vec((2, 3, 8), (0..48).collect())
            .unwrap()
            .slice_move(s![.., .., ..;2]);
        assert!(matches!(
            StandardArray3::try_from(stepped),
            Err(Error::NonStandardLayout { .. })
        ));

        let reversed = Array3::from_shape_vec((2, 3, 4), (0..24).collect())
            .unwrap()
            .slice_move(s![.., ..;-1, ..]);
        assert!(matches!(
            StandardArray3::try_from(reversed),
            Err(Error::NonStandardLayout { .. })
        ));
    }
}
