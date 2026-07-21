//! Run-scoped metadata attached to reusable metric values.

use serde::{Deserialize, Serialize};

use crate::metrics::intensity::{IntensityComparisonMetrics, IntensityStatistics};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameDiagnosticRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration: Option<usize>,
    pub frame_index: usize,
    pub illumination_index: usize,
    #[serde(flatten)]
    pub metrics: IntensityComparisonMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawFrameStatisticsRecord {
    pub frame_index: usize,
    #[serde(flatten)]
    pub metrics: IntensityStatistics,
}
