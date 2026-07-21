//! Explicit persistent representations for ndarray values.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, array_layout::checked_len_2d};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Array2Data<T> {
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) data: Vec<T>,
}

impl<T: Clone> Array2Data<T> {
    pub(crate) fn from_view(view: ArrayView2<'_, T>) -> Self {
        Self {
            height: view.nrows(),
            width: view.ncols(),
            data: view.iter().cloned().collect(),
        }
    }
}

impl<T> Array2Data<T> {
    pub(crate) fn into_array(self) -> Result<Array2<T>> {
        let shape = (self.height, self.width);
        let expected = checked_len_2d(shape)?;
        if self.data.len() != expected {
            return Err(Error::LengthMismatch {
                actual: self.data.len(),
                expected,
                shape,
            });
        }
        Ok(Array2::from_shape_vec(shape, self.data)?)
    }
}
