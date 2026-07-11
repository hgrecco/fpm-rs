use serde::{Deserialize, Serialize};

use num_complex::Complex64;

use crate::{Array2, Result, error::Error};

/// Fractional Fourier-grid displacement relative to an integer crop origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FourierOffset {
    pub row: f64,
    pub column: f64,
}

impl FourierOffset {
    pub const fn new(row: f64, column: f64) -> Self {
        Self { row, column }
    }

    pub fn is_zero(self) -> bool {
        self.row.abs() <= 1e-12 && self.column.abs() <= 1e-12
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FourierCrop {
    pub start_row: usize,
    pub start_col: usize,
    pub height: usize,
    pub width: usize,
}

impl FourierCrop {
    pub fn new(start_row: usize, start_col: usize, height: usize, width: usize) -> Self {
        Self {
            start_row,
            start_col,
            height,
            width,
        }
    }

    pub fn validate_inside(&self, shape: (usize, usize)) -> Result<()> {
        let end_row = self.start_row.checked_add(self.height);
        let end_col = self.start_col.checked_add(self.width);
        if self.height == 0
            || self.width == 0
            || end_row.is_none_or(|end| end > shape.0)
            || end_col.is_none_or(|end| end > shape.1)
        {
            return Err(Error::InvalidModel(format!(
                "crop {self:?} lies outside reconstruction shape {shape:?}"
            )));
        }
        Ok(())
    }

    pub fn extract<T: Copy>(&self, source: &Array2<T>, destination: &mut [T]) -> Result<()> {
        self.validate_inside(source.shape())?;
        if destination.len() != self.height * self.width {
            return Err(Error::LengthMismatch {
                actual: destination.len(),
                expected: self.height * self.width,
                shape: (self.height, self.width),
            });
        }
        for row in 0..self.height {
            let source_start = (self.start_row + row) * source.width() + self.start_col;
            let destination_start = row * self.width;
            destination[destination_start..destination_start + self.width]
                .copy_from_slice(&source.as_slice()[source_start..source_start + self.width]);
        }
        Ok(())
    }

    pub fn validate_subpixel_inside(
        &self,
        shape: (usize, usize),
        offset: FourierOffset,
    ) -> Result<()> {
        interpolation_axis(self.start_row, self.height, shape.0, offset.row)?;
        interpolation_axis(self.start_col, self.width, shape.1, offset.column)?;
        Ok(())
    }

    /// Bilinearly samples a potentially fractional crop from `source`.
    ///
    /// This is a local Fourier-grid interpolation, not a bandlimited shift
    /// operator. Its approximation error grows with grid-frequency content and
    /// fractional displacement; low-bandwidth objects are the intended regime.
    pub fn extract_subpixel(
        &self,
        source: &Array2<Complex64>,
        destination: &mut [Complex64],
        offset: FourierOffset,
    ) -> Result<()> {
        if destination.len() != self.height * self.width {
            return Err(Error::LengthMismatch {
                actual: destination.len(),
                expected: self.height * self.width,
                shape: (self.height, self.width),
            });
        }
        if offset.is_zero() {
            return self.extract(source, destination);
        }
        let rows = interpolation_axis(self.start_row, self.height, source.height(), offset.row)?;
        let columns =
            interpolation_axis(self.start_col, self.width, source.width(), offset.column)?;
        for row in 0..self.height {
            let lower_row = rows.lower_start + row;
            let upper_row = rows.upper_start + row;
            for column in 0..self.width {
                let lower_column = columns.lower_start + column;
                let upper_column = columns.upper_start + column;
                destination[row * self.width + column] = source[(lower_row, lower_column)]
                    * (rows.lower_weight * columns.lower_weight)
                    + source[(lower_row, upper_column)]
                        * (rows.lower_weight * columns.upper_weight)
                    + source[(upper_row, lower_column)]
                        * (rows.upper_weight * columns.lower_weight)
                    + source[(upper_row, upper_column)]
                        * (rows.upper_weight * columns.upper_weight);
            }
        }
        Ok(())
    }

    /// Adds the exact adjoint of [`Self::extract_subpixel`] to `destination`.
    pub fn insert_subpixel_adjoint(
        &self,
        destination: &mut Array2<Complex64>,
        update: &[Complex64],
        scale: f64,
        offset: FourierOffset,
    ) -> Result<()> {
        let destination_shape = destination.shape();
        self.insert_subpixel_adjoint_slice(
            destination.as_mut_slice(),
            destination_shape,
            update,
            scale,
            offset,
        )
    }

    pub(crate) fn insert_subpixel_adjoint_slice(
        &self,
        destination: &mut [Complex64],
        destination_shape: (usize, usize),
        update: &[Complex64],
        scale: f64,
        offset: FourierOffset,
    ) -> Result<()> {
        if destination.len() != destination_shape.0 * destination_shape.1 {
            return Err(Error::LengthMismatch {
                actual: destination.len(),
                expected: destination_shape.0 * destination_shape.1,
                shape: destination_shape,
            });
        }
        if update.len() != self.height * self.width {
            return Err(Error::LengthMismatch {
                actual: update.len(),
                expected: self.height * self.width,
                shape: (self.height, self.width),
            });
        }
        if !scale.is_finite() {
            return Err(Error::InvalidParameter {
                name: "scale",
                reason: "must be finite".into(),
            });
        }
        if offset.is_zero() {
            self.validate_inside(destination_shape)?;
            for row in 0..self.height {
                for column in 0..self.width {
                    destination
                        [(self.start_row + row) * destination_shape.1 + self.start_col + column] +=
                        scale * update[row * self.width + column];
                }
            }
            return Ok(());
        }
        let rows =
            interpolation_axis(self.start_row, self.height, destination_shape.0, offset.row)?;
        let columns = interpolation_axis(
            self.start_col,
            self.width,
            destination_shape.1,
            offset.column,
        )?;
        for row in 0..self.height {
            let lower_row = rows.lower_start + row;
            let upper_row = rows.upper_start + row;
            for column in 0..self.width {
                let lower_column = columns.lower_start + column;
                let upper_column = columns.upper_start + column;
                let value = scale * update[row * self.width + column];
                destination[lower_row * destination_shape.1 + lower_column] +=
                    value * (rows.lower_weight * columns.lower_weight);
                destination[lower_row * destination_shape.1 + upper_column] +=
                    value * (rows.lower_weight * columns.upper_weight);
                destination[upper_row * destination_shape.1 + lower_column] +=
                    value * (rows.upper_weight * columns.lower_weight);
                destination[upper_row * destination_shape.1 + upper_column] +=
                    value * (rows.upper_weight * columns.upper_weight);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct AxisInterpolation {
    lower_start: usize,
    upper_start: usize,
    lower_weight: f64,
    upper_weight: f64,
}

fn interpolation_axis(
    start: usize,
    length: usize,
    bound: usize,
    offset: f64,
) -> Result<AxisInterpolation> {
    if length == 0 || bound == 0 || !offset.is_finite() {
        return Err(Error::InvalidModel(
            "subpixel crop dimensions and offset must be finite and non-zero".into(),
        ));
    }
    let nearest = offset.round();
    let (integer_offset, fraction) = if (offset - nearest).abs() <= 1e-12 {
        (nearest, 0.0)
    } else {
        let floor = offset.floor();
        (floor, offset - floor)
    };
    if integer_offset < isize::MIN as f64 || integer_offset > isize::MAX as f64 {
        return Err(Error::InvalidModel(
            "subpixel crop offset is outside the supported index range".into(),
        ));
    }
    let start = isize::try_from(start).map_err(|_| {
        Error::InvalidModel("crop origin is outside the supported index range".into())
    })?;
    let lower_start = start
        .checked_add(integer_offset as isize)
        .filter(|value| *value >= 0)
        .ok_or_else(|| Error::InvalidModel("subpixel crop starts outside the grid".into()))?;
    let lower_start = usize::try_from(lower_start)
        .map_err(|_| Error::InvalidModel("subpixel crop starts outside the grid".into()))?;
    let lower_end = lower_start
        .checked_add(length)
        .ok_or_else(|| Error::InvalidModel("subpixel crop dimensions overflow the grid".into()))?;
    let needs_upper = fraction > 0.0;
    if lower_end > bound || (needs_upper && lower_end >= bound) {
        return Err(Error::InvalidModel(
            "subpixel crop interpolation stencil lies outside the grid".into(),
        ));
    }
    Ok(AxisInterpolation {
        lower_start,
        upper_start: lower_start + usize::from(needs_upper),
        lower_weight: 1.0 - fraction,
        upper_weight: fraction,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CropIndices {
    pub crops: Vec<FourierCrop>,
}

impl CropIndices {
    pub fn new(crops: Vec<FourierCrop>) -> Self {
        Self { crops }
    }

    pub fn len(&self) -> usize {
        self.crops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.crops.is_empty()
    }

    pub fn get(&self, frame: usize) -> Result<FourierCrop> {
        self.crops
            .get(frame)
            .copied()
            .ok_or(Error::FrameOutOfRange {
                index: frame,
                frames: self.crops.len(),
            })
    }
}
