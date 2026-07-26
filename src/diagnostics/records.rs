//! Run-scoped metadata attached to reusable metric values.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::metrics::intensity::{IntensityComparisonMetrics, IntensityStats};

#[derive(Clone, Debug)]
pub struct FrameDiagnosticRecord {
    pub iteration: Option<usize>,
    pub frame_index: usize,
    pub illumination_index: usize,
    pub metrics: IntensityComparisonMetrics,
}

#[derive(Serialize)]
struct FrameDiagnosticRecordRef {
    #[serde(skip_serializing_if = "Option::is_none")]
    iteration: Option<usize>,
    frame_index: usize,
    illumination_index: usize,
    reference_sum: f64,
    estimate_sum: f64,
    residual_l1: f64,
    residual_l2: f64,
    residual_mean: f64,
    residual_std: f64,
    residual_max_abs: f64,
    normalized_l2: f64,
    saturated_pixels: Option<usize>,
}

#[derive(Deserialize)]
struct FrameDiagnosticRecordOwned {
    #[serde(default)]
    iteration: Option<usize>,
    frame_index: usize,
    illumination_index: usize,
    reference_sum: f64,
    estimate_sum: f64,
    residual_l1: f64,
    residual_l2: f64,
    residual_mean: f64,
    residual_std: f64,
    residual_max_abs: f64,
    normalized_l2: f64,
    saturated_pixels: Option<usize>,
}

impl Serialize for FrameDiagnosticRecord {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        FrameDiagnosticRecordRef {
            iteration: self.iteration,
            frame_index: self.frame_index,
            illumination_index: self.illumination_index,
            reference_sum: self.metrics.reference_sum,
            estimate_sum: self.metrics.estimate_sum,
            residual_l1: self.metrics.residual_l1,
            residual_l2: self.metrics.residual_l2,
            residual_mean: self.metrics.residual_mean,
            residual_std: self.metrics.residual_std,
            residual_max_abs: self.metrics.residual_max_abs,
            normalized_l2: self.metrics.normalized_l2,
            saturated_pixels: self.metrics.saturated_pixels,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FrameDiagnosticRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let record = FrameDiagnosticRecordOwned::deserialize(deserializer)?;
        Ok(Self {
            iteration: record.iteration,
            frame_index: record.frame_index,
            illumination_index: record.illumination_index,
            metrics: IntensityComparisonMetrics {
                reference_sum: record.reference_sum,
                estimate_sum: record.estimate_sum,
                residual_l1: record.residual_l1,
                residual_l2: record.residual_l2,
                residual_mean: record.residual_mean,
                residual_std: record.residual_std,
                residual_max_abs: record.residual_max_abs,
                normalized_l2: record.normalized_l2,
                saturated_pixels: record.saturated_pixels,
            },
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawFrameStatisticsRecord {
    pub frame_index: usize,
    #[serde(flatten)]
    pub metrics: IntensityStats,
}
