use ndarray::Array2;

use super::{FrameDiagnosticRecord, RawFrameStatisticsRecord};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticRequest {
    Objective,
    PerFrameError,
    RawFrameStats,
    FrameSummaries,
    ObjectAmplitude,
    ObjectPhase,
    Pupil,
    ResidualImages,
}

#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub objective: Option<f64>,
    pub per_frame_objective: Option<Vec<f64>>,
    pub raw_frame_statistics: Option<Vec<RawFrameStatisticsRecord>>,
    pub frame_diagnostics: Option<Vec<FrameDiagnosticRecord>>,
    pub object_amplitude: Option<Array2<f64>>,
    pub object_phase: Option<Array2<f64>>,
    pub pupil_amplitude: Option<Array2<f64>>,
    pub pupil_phase: Option<Array2<f64>>,
    pub residual_images: Option<Vec<Array2<f64>>>,
}
