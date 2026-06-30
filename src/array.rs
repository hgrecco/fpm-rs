use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use std::ops::{Index, IndexMut};

use crate::error::{Error, Result};

/// A small, row-major, contiguous 2-D array used by the computational core.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Array2<T> {
    height: usize,
    width: usize,
    data: Vec<T>,
}

impl<'de, T> Deserialize<'de> for Array2<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Representation<T> {
            height: usize,
            width: usize,
            data: Vec<T>,
        }

        let representation = Representation::deserialize(deserializer)?;
        Self::from_vec(
            (representation.height, representation.width),
            representation.data,
        )
        .map_err(D::Error::custom)
    }
}

impl<T> Array2<T> {
    pub fn from_vec(shape: (usize, usize), data: Vec<T>) -> Result<Self> {
        let expected = shape
            .0
            .checked_mul(shape.1)
            .ok_or_else(|| Error::InvalidShape(format!("shape {shape:?} overflows")))?;
        if shape.0 == 0 || shape.1 == 0 {
            return Err(Error::InvalidShape(format!(
                "dimensions must be non-zero, got {shape:?}"
            )));
        }
        if data.len() != expected {
            return Err(Error::LengthMismatch {
                actual: data.len(),
                expected,
                shape,
            });
        }
        Ok(Self {
            height: shape.0,
            width: shape.1,
            data,
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.height && col < self.width {
            self.data.get(row * self.width + col)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.height && col < self.width {
            self.data.get_mut(row * self.width + col)
        } else {
            None
        }
    }

    pub fn map<U>(&self, mut function: impl FnMut(&T) -> U) -> Array2<U> {
        Array2 {
            height: self.height,
            width: self.width,
            data: self.data.iter().map(&mut function).collect(),
        }
    }
}

impl<T: Clone> Array2<T> {
    pub fn filled(shape: (usize, usize), value: T) -> Result<Self> {
        let len = shape
            .0
            .checked_mul(shape.1)
            .ok_or_else(|| Error::InvalidShape(format!("shape {shape:?} overflows")))?;
        Self::from_vec(shape, vec![value; len])
    }
}

impl<T: Default + Clone> Array2<T> {
    pub fn zeros(shape: (usize, usize)) -> Result<Self> {
        Self::filled(shape, T::default())
    }
}

impl<T> Index<(usize, usize)> for Array2<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.width + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Array2<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.width + index.1]
    }
}
