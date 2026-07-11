use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawFrameStats {
    pub frame_index: usize,

    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub sum: f64,

    pub saturated_pixels: usize,
    pub zero_pixels: usize,
}

pub fn compute_raw_frame_stats(
    frame_index: usize,
    frame: &[f64],
    saturation_value: Option<f64>,
) -> crate::Result<RawFrameStats> {
    if frame.is_empty() {
        return Err(crate::Error::InvalidShape(
            "raw frame statistics require a non-empty frame".into(),
        ));
    }
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut saturated_pixels = 0usize;
    let mut zero_pixels = 0usize;
    for &value in frame {
        if !value.is_finite() {
            return Err(crate::Error::Numerical(
                "raw frame contains a non-finite value".into(),
            ));
        }
        sum += value;
        sum_squares += value * value;
        min = min.min(value);
        max = max.max(value);
        if value == 0.0 {
            zero_pixels += 1;
        }
        if saturation_value.is_some_and(|limit| value >= limit) {
            saturated_pixels += 1;
        }
    }
    let count = frame.len() as f64;
    let mean = sum / count;
    let variance = (sum_squares / count - mean * mean).max(0.0);
    Ok(RawFrameStats {
        frame_index,
        mean,
        std: variance.sqrt(),
        min,
        max,
        sum,
        saturated_pixels,
        zero_pixels,
    })
}
