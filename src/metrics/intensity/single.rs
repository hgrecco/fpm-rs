//! Statistics calculated from one intensity image.

use serde::{Deserialize, Serialize};

/// Summary statistics calculated from one intensity image.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntensityStats {
    /// Arithmetic mean intensity.
    pub mean: f64,
    /// Population standard deviation of intensity.
    pub std: f64,
    /// Minimum intensity.
    pub min: f64,
    /// Maximum intensity.
    pub max: f64,
    /// Sum of intensities.
    pub sum: f64,
    /// Pixels at or above the optional saturation threshold, or zero when absent.
    pub saturated_pixels: usize,
    /// Pixels exactly equal to zero.
    pub zero_pixels: usize,
}

/// Calculate summary statistics for a non-empty intensity image.
pub fn stats(values: &[f64], saturation_value: Option<f64>) -> crate::Result<IntensityStats> {
    if values.is_empty() {
        return Err(crate::Error::InvalidShape(
            "intensity statistics require non-empty values".into(),
        ));
    }
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut saturated_pixels = 0;
    let mut zero_pixels = 0;
    for &value in values {
        if !value.is_finite() {
            return Err(crate::Error::Numerical(
                "intensity statistics contain a non-finite value".into(),
            ));
        }
        sum += value;
        sum_squares += value * value;
        min = min.min(value);
        max = max.max(value);
        zero_pixels += usize::from(value == 0.0);
        saturated_pixels += usize::from(saturation_value.is_some_and(|limit| value >= limit));
    }
    let count = values.len() as f64;
    let mean = sum / count;
    Ok(IntensityStats {
        mean,
        std: (sum_squares / count - mean * mean).max(0.0).sqrt(),
        min,
        max,
        sum,
        saturated_pixels,
        zero_pixels,
    })
}
