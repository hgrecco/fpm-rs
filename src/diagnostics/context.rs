use ndarray::Array2;

use super::{FrameDiagnosticRecord, RawFrameStatisticsRecord};

/// Diagnostic quantity a callback asks the runner to compute for a hook.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticRequest {
    /// Weight-normalized scalar objective for the current completed work.
    Objective,
    /// Objective value for every acquisition frame.
    PerFrameError,
    /// Raw measured-intensity distribution statistics for each frame.
    RawFrameStats,
    /// Per-frame reference-versus-prediction comparison metrics.
    FrameSummaries,
    /// High-resolution reconstructed object amplitude snapshot.
    ObjectAmplitude,
    /// High-resolution wrapped reconstructed object phase in radians.
    ObjectPhase,
    /// Low-resolution pupil amplitude and wrapped phase snapshots.
    Pupil,
    /// Predicted-minus-measured intensity image for every acquisition frame.
    ResidualImages,
}

/// Optional diagnostics computed for one callback hook.
#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    /// Weight-normalized objective, when requested.
    pub objective: Option<f64>,
    /// Objective in acquisition-frame order, when requested.
    pub per_frame_objective: Option<Vec<f64>>,
    /// Raw frame statistics in acquisition-frame order, when requested.
    pub raw_frame_statistics: Option<Vec<RawFrameStatisticsRecord>>,
    /// Frame comparison records in acquisition-frame order, when requested.
    pub frame_diagnostics: Option<Vec<FrameDiagnosticRecord>>,
    /// Owned high-resolution `(row, column)` object amplitude snapshot.
    pub object_amplitude: Option<Array2<f64>>,
    /// Owned high-resolution wrapped object phase snapshot in radians.
    pub object_phase: Option<Array2<f64>>,
    /// Owned low-resolution pupil amplitude snapshot.
    pub pupil_amplitude: Option<Array2<f64>>,
    /// Owned low-resolution wrapped pupil phase snapshot in radians.
    pub pupil_phase: Option<Array2<f64>>,
    /// Owned predicted-minus-measured intensity images in acquisition-frame order.
    pub residual_images: Option<Vec<Array2<f64>>>,
}
