use serde::{Deserialize, Serialize};

use crate::{experiment::KVector, model::ImagePlaneModel};

/// Pixel bounds and center of one source crop in the high-resolution Fourier grid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CropIndexDiagnostics {
    /// Zero-based individual illumination-source index.
    pub illumination_index: usize,

    /// Inclusive first Fourier column (`x`).
    pub x_start: usize,
    /// Exclusive last Fourier column (`x`).
    pub x_end: usize,

    /// Inclusive first Fourier row (`y`).
    pub y_start: usize,
    /// Exclusive last Fourier row (`y`).
    pub y_end: usize,

    /// Crop center along the Fourier column axis, in grid pixels.
    pub center_x: f64,
    /// Crop center along the Fourier row axis, in grid pixels.
    pub center_y: f64,
}

/// Summary of individual-source coverage on the high-resolution Fourier grid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FourierCoverageDiagnostics {
    /// Dimensionless synthetic numerical aperture, when wavelength metadata is available.
    pub synthetic_na: Option<f64>,

    /// Approximate circular-pupil radius in low-resolution Fourier-grid pixels.
    pub pupil_radius_px: f64,

    /// Source crop centers as `[x_column, y_row]` in high-resolution grid pixels.
    pub pupil_centers_px: Vec<[f64; 2]>,

    /// Dimensionless illumination NA for each individual source.
    pub illumination_na: Vec<f64>,

    /// Integer crop diagnostics in individual source order.
    pub crop_indices: Vec<CropIndexDiagnostics>,

    /// High-resolution `[height, width]` of pixels covered by at least two pupils.
    pub overlap_shape: Option<[usize; 2]>,
}

/// Validates `model` and derives source centers, crop bounds, NA, and overlap coverage.
pub fn compute_fourier_coverage(
    model: &ImagePlaneModel,
) -> crate::Result<FourierCoverageDiagnostics> {
    model.validate()?;
    let wavelength = model.sampling.wavelength.unwrap_or(0.0);
    let illumination_na = model
        .k_vectors
        .iter()
        .map(|vector| k_vector_na(vector, wavelength))
        .collect();
    let crop_indices = model
        .crop_indices
        .crops
        .iter()
        .enumerate()
        .map(|(illumination_index, crop)| {
            let offset = model.source_offset(illumination_index)?;
            Ok(CropIndexDiagnostics {
                illumination_index,
                x_start: crop.start_col,
                x_end: crop.start_col + crop.width,
                y_start: crop.start_row,
                y_end: crop.start_row + crop.height,
                center_x: crop.start_col as f64 + crop.width as f64 / 2.0 + offset.column,
                center_y: crop.start_row as f64 + crop.height as f64 / 2.0 + offset.row,
            })
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let pupil_radius_px = pupil_radius_px(&model.pupil);
    let pupil_centers_px = crop_indices
        .iter()
        .map(|crop| [crop.center_x, crop.center_y])
        .collect();
    Ok(FourierCoverageDiagnostics {
        synthetic_na: model.sampling.synthetic_na,
        pupil_radius_px,
        pupil_centers_px,
        illumination_na,
        crop_indices,
        overlap_shape: None,
    })
}

fn k_vector_na(vector: &KVector, wavelength: f64) -> f64 {
    if wavelength <= 0.0 {
        0.0
    } else {
        vector.kx.hypot(vector.ky) * wavelength / std::f64::consts::TAU
    }
}

fn pupil_radius_px(pupil: &crate::model::Pupil) -> f64 {
    let shape = pupil.shape();
    let center_y = shape.0 as f64 / 2.0;
    let center_x = shape.1 as f64 / 2.0;
    let mut max_radius: f64 = 0.0;
    for row in 0..shape.0 {
        for col in 0..shape.1 {
            let index = row * shape.1 + col;
            if pupil.support.as_slice()[index] != 0 {
                let radius = (row as f64 - center_y).hypot(col as f64 - center_x);
                max_radius = max_radius.max(radius);
            }
        }
    }
    max_radius
}
