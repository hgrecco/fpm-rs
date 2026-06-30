use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Array2, Result,
    error::Error,
    experiment::{IlluminationSource, KVector, MultiplexingMatrix, Optics},
};

use super::{CropIndices, FourierCrop, FourierOffset, Pupil, Sampling};

/// Algorithm-facing image-plane FPM model. It contains no LED or camera geometry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImagePlaneModel {
    /// One transverse wave vector per illumination source.
    pub k_vectors: Vec<KVector>,
    pub pupil: Pupil,
    pub crop_indices: CropIndices,
    /// Fractional `(row, column)` Fourier-grid offsets relative to each crop.
    #[serde(default)]
    pub subpixel_offsets: Option<Vec<FourierOffset>>,
    pub sampling: Sampling,
    pub image_shape: (usize, usize),
    pub reconstruction_shape: (usize, usize),
    pub frame_gains: Option<Vec<f64>>,
    pub background: Option<Vec<f64>>,
    /// Optional measured-frame rows of `(source_index, incoherent_weight)`.
    pub multiplexing_matrix: Option<MultiplexingMatrix>,
}

impl ImagePlaneModel {
    pub fn new(
        k_vectors: Vec<KVector>,
        pupil: Pupil,
        crop_indices: CropIndices,
        sampling: Sampling,
        image_shape: (usize, usize),
        reconstruction_shape: (usize, usize),
    ) -> Result<Self> {
        let model = Self {
            k_vectors,
            pupil,
            crop_indices,
            subpixel_offsets: None,
            sampling,
            image_shape,
            reconstruction_shape,
            frame_gains: None,
            background: None,
            multiplexing_matrix: None,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn from_experiment<I: IlluminationSource>(
        optics: &Optics,
        illumination: &I,
        image_shape: (usize, usize),
        reconstruction_shape: (usize, usize),
    ) -> Result<Self> {
        optics.validate()?;
        if image_shape.0 == 0
            || image_shape.1 == 0
            || reconstruction_shape.0 < image_shape.0
            || reconstruction_shape.1 < image_shape.1
        {
            return Err(Error::InvalidShape(format!(
                "image shape {image_shape:?} must be non-zero and fit reconstruction shape {reconstruction_shape:?}"
            )));
        }
        let scale_y = reconstruction_shape.0 as f64 / image_shape.0 as f64;
        let scale_x = reconstruction_shape.1 as f64 / image_shape.1 as f64;
        if (scale_x - scale_y).abs() > 1e-9 * scale_x.max(scale_y) {
            return Err(Error::InvalidShape(
                "reconstruction must use the same scale factor in both dimensions".into(),
            ));
        }
        let low_res_pixel_size = optics.object_pixel_size();
        let mut sampling = Sampling::new(
            low_res_pixel_size,
            low_res_pixel_size / scale_x,
            std::f64::consts::TAU / (image_shape.1 as f64 * low_res_pixel_size),
            std::f64::consts::TAU / (image_shape.0 as f64 * low_res_pixel_size),
        )?;
        sampling.wavelength = Some(optics.wavelength);
        let k_vectors = illumination.k_vectors(optics)?;
        if k_vectors.is_empty() {
            return Err(Error::InvalidModel(
                "illumination must contain at least one frame".into(),
            ));
        }
        let maximum_illumination_na = k_vectors
            .iter()
            .map(|vector| vector.kx.hypot(vector.ky) * optics.wavelength / std::f64::consts::TAU)
            .fold(0.0, f64::max);
        sampling.synthetic_na = Some(optics.objective_na + maximum_illumination_na);
        let pupil = Pupil::circular(image_shape, &sampling, optics)?;
        let mut offsets = Vec::with_capacity(k_vectors.len());
        let crops = k_vectors
            .iter()
            .map(|vector| {
                let (crop, offset) =
                    Self::crop_for_k_vector(vector, &sampling, image_shape, reconstruction_shape)?;
                offsets.push(offset);
                Ok(crop)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut model = Self::new(
            k_vectors,
            pupil,
            CropIndices::new(crops),
            sampling,
            image_shape,
            reconstruction_shape,
        )?;
        model.frame_gains = illumination.frame_gains()?;
        model.multiplexing_matrix = illumination.multiplexing_matrix()?;
        model.subpixel_offsets = Some(offsets);
        model.validate()?;
        Ok(model)
    }

    pub fn source_count(&self) -> usize {
        self.k_vectors.len()
    }

    pub fn frame_count(&self) -> usize {
        self.multiplexing_matrix
            .as_ref()
            .map_or_else(|| self.source_count(), Vec::len)
    }

    pub fn is_multiplexed(&self) -> bool {
        self.multiplexing_matrix.is_some()
    }

    pub fn with_multiplexing(mut self, matrix: MultiplexingMatrix) -> Result<Self> {
        self.multiplexing_matrix = Some(matrix);
        self.validate()?;
        Ok(self)
    }

    pub fn with_subpixel_offsets(mut self, offsets: Vec<FourierOffset>) -> Result<Self> {
        self.subpixel_offsets = Some(offsets);
        self.validate()?;
        Ok(self)
    }

    pub fn source_offset(&self, source: usize) -> Result<FourierOffset> {
        self.crop_indices.get(source)?;
        match &self.subpixel_offsets {
            None => Ok(FourierOffset::default()),
            Some(offsets) => offsets.get(source).copied().ok_or_else(|| {
                Error::InvalidModel("subpixel offset count does not match source count".into())
            }),
        }
    }

    pub fn extract_patch(
        &self,
        object_spectrum: &Array2<Complex64>,
        source: usize,
        destination: &mut [Complex64],
    ) -> Result<()> {
        self.extract_patch_at_offset(
            object_spectrum,
            source,
            self.source_offset(source)?,
            destination,
        )
    }

    pub fn extract_patch_at_offset(
        &self,
        object_spectrum: &Array2<Complex64>,
        source: usize,
        offset: FourierOffset,
        destination: &mut [Complex64],
    ) -> Result<()> {
        if object_spectrum.shape() != self.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "object spectrum shape {:?} does not match {:?}",
                object_spectrum.shape(),
                self.reconstruction_shape
            )));
        }
        let crop = self.crop_indices.get(source)?;
        crop.extract_subpixel(object_spectrum, destination, offset)
    }

    pub fn insert_patch_adjoint(
        &self,
        destination: &mut Array2<Complex64>,
        source: usize,
        update: &[Complex64],
        scale: f64,
    ) -> Result<()> {
        self.insert_patch_adjoint_at_offset(
            destination,
            source,
            update,
            scale,
            self.source_offset(source)?,
        )
    }

    pub fn insert_patch_adjoint_at_offset(
        &self,
        destination: &mut Array2<Complex64>,
        source: usize,
        update: &[Complex64],
        scale: f64,
        offset: FourierOffset,
    ) -> Result<()> {
        if destination.shape() != self.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "object spectrum shape {:?} does not match {:?}",
                destination.shape(),
                self.reconstruction_shape
            )));
        }
        let crop = self.crop_indices.get(source)?;
        crop.insert_subpixel_adjoint(destination, update, scale, offset)
    }

    pub(crate) fn insert_patch_adjoint_slice_at_offset(
        &self,
        destination: &mut [Complex64],
        source: usize,
        update: &[Complex64],
        scale: f64,
        offset: FourierOffset,
    ) -> Result<()> {
        let crop = self.crop_indices.get(source)?;
        crop.insert_subpixel_adjoint_slice(
            destination,
            self.reconstruction_shape,
            update,
            scale,
            offset,
        )
    }

    pub fn validate_source_offset(&self, source: usize, offset: FourierOffset) -> Result<()> {
        self.crop_indices
            .get(source)?
            .validate_subpixel_inside(self.reconstruction_shape, offset)
    }

    pub(crate) fn crop_for_k_vector(
        vector: &KVector,
        sampling: &Sampling,
        image_shape: (usize, usize),
        reconstruction_shape: (usize, usize),
    ) -> Result<(FourierCrop, FourierOffset)> {
        let continuous_row = vector.ky / sampling.dky;
        let continuous_column = vector.kx / sampling.dkx;
        let (shift_row, offset_row) = checked_grid_shift(continuous_row)?;
        let (shift_column, offset_column) = checked_grid_shift(continuous_column)?;
        let reconstruction_center_row = isize::try_from(reconstruction_shape.0 / 2)
            .map_err(|_| Error::InvalidShape("reconstruction height is too large".into()))?;
        let reconstruction_center_column = isize::try_from(reconstruction_shape.1 / 2)
            .map_err(|_| Error::InvalidShape("reconstruction width is too large".into()))?;
        let image_half_height = isize::try_from(image_shape.0 / 2)
            .map_err(|_| Error::InvalidShape("image height is too large".into()))?;
        let image_half_width = isize::try_from(image_shape.1 / 2)
            .map_err(|_| Error::InvalidShape("image width is too large".into()))?;
        let start_row = reconstruction_center_row
            .checked_sub(image_half_height)
            .and_then(|value| value.checked_add(shift_row));
        let start_column = reconstruction_center_column
            .checked_sub(image_half_width)
            .and_then(|value| value.checked_add(shift_column));
        let (Some(start_row), Some(start_column)) = (start_row, start_column) else {
            return Err(Error::InvalidModel(format!(
                "illumination vector {vector:?} overflows the reconstruction grid"
            )));
        };
        if start_row < 0 || start_column < 0 {
            return Err(Error::InvalidModel(format!(
                "illumination vector {vector:?} produces a crop outside the reconstruction grid"
            )));
        }
        let crop = FourierCrop::new(
            usize::try_from(start_row).map_err(|_| {
                Error::InvalidModel("crop row cannot be represented as an index".into())
            })?,
            usize::try_from(start_column).map_err(|_| {
                Error::InvalidModel("crop column cannot be represented as an index".into())
            })?,
            image_shape.0,
            image_shape.1,
        );
        let offset = FourierOffset::new(offset_row, offset_column);
        crop.validate_subpixel_inside(reconstruction_shape, offset)?;
        Ok((crop, offset))
    }

    pub fn frame_gain(&self, frame: usize) -> Result<f64> {
        if frame >= self.frame_count() {
            return Err(Error::FrameOutOfRange {
                index: frame,
                frames: self.frame_count(),
            });
        }
        Ok(self
            .frame_gains
            .as_ref()
            .map_or(1.0, |values| values[frame]))
    }

    pub fn background_value(&self, frame: usize, pixel: usize) -> Result<f64> {
        if frame >= self.frame_count() {
            return Err(Error::FrameOutOfRange {
                index: frame,
                frames: self.frame_count(),
            });
        }
        let image_len = self.image_shape.0 * self.image_shape.1;
        if pixel >= image_len {
            return Err(Error::InvalidParameter {
                name: "pixel",
                reason: format!("index {pixel} is outside an image with {image_len} pixels"),
            });
        }
        Ok(self.background.as_ref().map_or(0.0, |values| {
            values[if values.len() == image_len {
                pixel
            } else {
                frame * image_len + pixel
            }]
        }))
    }

    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        if self.k_vectors.is_empty() || self.source_count() != self.crop_indices.len() {
            return Err(Error::InvalidModel(format!(
                "{} source k-vectors and {} crops; counts must be equal and non-zero",
                self.source_count(),
                self.crop_indices.len()
            )));
        }
        if self
            .k_vectors
            .iter()
            .any(|vector| !vector.kx.is_finite() || !vector.ky.is_finite())
        {
            return Err(Error::InvalidModel(
                "k-vectors must contain finite values".into(),
            ));
        }
        if self.pupil.shape() != self.image_shape {
            return Err(Error::InvalidModel(format!(
                "pupil shape {:?} does not match image shape {:?}",
                self.pupil.shape(),
                self.image_shape
            )));
        }
        if self.pupil.support.len() != self.pupil.values.len()
            || self
                .pupil
                .values
                .as_slice()
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::InvalidModel(
                "pupil support or numeric values are invalid".into(),
            ));
        }
        if self.reconstruction_shape.0 < self.image_shape.0
            || self.reconstruction_shape.1 < self.image_shape.1
        {
            return Err(Error::InvalidModel(
                "reconstruction shape must contain a low-resolution crop".into(),
            ));
        }
        for crop in &self.crop_indices.crops {
            if (crop.height, crop.width) != self.image_shape {
                return Err(Error::InvalidModel(format!(
                    "crop shape {:?} does not match image shape {:?}",
                    (crop.height, crop.width),
                    self.image_shape
                )));
            }
            crop.validate_inside(self.reconstruction_shape)?;
        }
        if let Some(offsets) = &self.subpixel_offsets {
            if offsets.len() != self.source_count() {
                return Err(Error::InvalidModel(
                    "subpixel offset count does not match source count".into(),
                ));
            }
            for (crop, &offset) in self.crop_indices.crops.iter().zip(offsets) {
                crop.validate_subpixel_inside(self.reconstruction_shape, offset)?;
            }
        }
        if let Some(values) = &self.frame_gains {
            if values.len() != self.frame_count() {
                return Err(Error::InvalidModel(
                    "frame gain count does not match frame count".into(),
                ));
            }
            if values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(Error::InvalidModel(
                    "frame gains must be finite and positive".into(),
                ));
            }
        }
        let image_len = self.image_shape.0 * self.image_shape.1;
        if self.background.as_ref().is_some_and(|values| {
            values.len() != image_len && values.len() != image_len * self.frame_count()
        }) {
            return Err(Error::InvalidModel(
                "background must be one image or one image per frame".into(),
            ));
        }
        if self
            .background
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::InvalidModel(
                "background values must be finite".into(),
            ));
        }
        if let Some(matrix) = &self.multiplexing_matrix {
            if matrix.is_empty() {
                return Err(Error::InvalidModel(
                    "multiplexing matrix must contain at least one measured frame".into(),
                ));
            }
            for (row_index, row) in matrix.iter().enumerate() {
                let mut seen = vec![false; self.source_count()];
                let invalid = row.is_empty()
                    || row.iter().any(|&(source, weight)| {
                        let duplicate = source < self.source_count() && seen[source];
                        if source < self.source_count() {
                            seen[source] = true;
                        }
                        source >= self.source_count()
                            || !weight.is_finite()
                            || weight <= 0.0
                            || duplicate
                    });
                if invalid {
                    return Err(Error::InvalidModel(format!(
                        "multiplexing row {row_index} is empty or contains a duplicate/invalid source or weight"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn checked_grid_shift(value: f64) -> Result<(isize, f64)> {
    if !value.is_finite() {
        return Err(Error::InvalidModel(
            "illumination shift must be finite".into(),
        ));
    }
    let rounded = value.round();
    if rounded < isize::MIN as f64 || rounded > isize::MAX as f64 {
        return Err(Error::InvalidModel(
            "illumination shift is outside the supported index range".into(),
        ));
    }
    Ok((rounded as isize, value - rounded))
}
