use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameDiagnostics {
    /// Reconstruction iteration that produced this summary. Standalone helper
    /// calls leave it unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration: Option<usize>,
    pub frame_index: usize,
    pub illumination_index: usize,

    pub measured_sum: f64,
    pub predicted_sum: f64,

    pub residual_l1: f64,
    pub residual_l2: f64,
    pub residual_mean: f64,
    pub residual_std: f64,
    pub residual_max_abs: f64,

    pub normalized_l2: f64,

    pub saturated_pixels: Option<usize>,
}

pub fn compute_frame_diagnostics(
    frame_index: usize,
    illumination_index: usize,
    measured: &[f64],
    predicted: &[f64],
    saturation_value: Option<f64>,
) -> crate::Result<FrameDiagnostics> {
    compute_frame_diagnostics_with_mask(
        frame_index,
        illumination_index,
        measured,
        predicted,
        None,
        saturation_value,
    )
}

pub fn compute_frame_diagnostics_with_mask(
    frame_index: usize,
    illumination_index: usize,
    measured: &[f64],
    predicted: &[f64],
    mask: Option<&[u8]>,
    saturation_value: Option<f64>,
) -> crate::Result<FrameDiagnostics> {
    if measured.len() != predicted.len() || measured.is_empty() {
        return Err(crate::Error::InvalidShape(format!(
            "frame diagnostics inputs have lengths {} and {}",
            measured.len(),
            predicted.len()
        )));
    }
    if mask.is_some_and(|mask| mask.len() != measured.len()) {
        return Err(crate::Error::LengthMismatch {
            actual: mask.map_or(0, |mask| mask.len()),
            expected: measured.len(),
            shape: (1, measured.len()),
        });
    }
    let mut measured_sum = 0.0;
    let mut predicted_sum = 0.0;
    let mut residual_l1 = 0.0;
    let mut residual_l2 = 0.0;
    let mut residual_sum = 0.0;
    let mut residual_sum_squares = 0.0;
    let mut residual_max_abs: f64 = 0.0;
    let mut measured_l2 = 0.0;
    let mut saturated_pixels = 0usize;
    let mut valid_pixels = 0usize;
    for (pixel, (&measured, &predicted)) in measured.iter().zip(predicted).enumerate() {
        if mask.is_some_and(|mask| mask[pixel] == 0) {
            continue;
        }
        if !measured.is_finite() || !predicted.is_finite() {
            return Err(crate::Error::Numerical(
                "frame diagnostics input contains a non-finite value".into(),
            ));
        }
        measured_sum += measured;
        predicted_sum += predicted;
        let residual = predicted - measured;
        residual_l1 += residual.abs();
        residual_l2 += residual * residual;
        residual_sum += residual;
        residual_sum_squares += residual * residual;
        residual_max_abs = residual_max_abs.max(residual.abs());
        measured_l2 += measured * measured;
        if saturation_value.is_some_and(|limit| measured >= limit) {
            saturated_pixels += 1;
        }
        valid_pixels += 1;
    }
    let count = valid_pixels as f64;
    let residual_mean = if valid_pixels == 0 {
        0.0
    } else {
        residual_sum / count
    };
    let residual_std = if valid_pixels == 0 {
        0.0
    } else {
        (residual_sum_squares / count - residual_mean * residual_mean)
            .max(0.0)
            .sqrt()
    };
    let normalized_l2 = if valid_pixels == 0 {
        0.0
    } else {
        residual_l2.sqrt() / (measured_l2.sqrt() + f64::EPSILON)
    };
    Ok(FrameDiagnostics {
        iteration: None,
        frame_index,
        illumination_index,
        measured_sum,
        predicted_sum,
        residual_l1,
        residual_l2: residual_l2.sqrt(),
        residual_mean,
        residual_std,
        residual_max_abs,
        normalized_l2,
        saturated_pixels: saturation_value.map(|_| saturated_pixels),
    })
}
