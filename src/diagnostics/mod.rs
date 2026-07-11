mod coverage;
mod frame;
mod history;
mod iteration;
mod loss;
mod metrics;
mod raw_stack;
mod recorder;
mod serialization;

pub use coverage::{CropIndexDiagnostics, FourierCoverageDiagnostics, compute_fourier_coverage};
pub use frame::{FrameDiagnostics, compute_frame_diagnostics, compute_frame_diagnostics_with_mask};
pub use history::{IterationRecord, ReconstructionHistory};
pub use iteration::IterationDiagnostics;
pub(crate) use loss::point_loss;
pub use loss::{LossType, loss};
pub use metrics::{
    DiagnosticRequest, Diagnostics, GroundTruthMetrics, StepDiagnostics,
    compute_ground_truth_metrics,
};
pub use raw_stack::{RawFrameStats, compute_raw_frame_stats};
pub use recorder::{DiagnosticRecorder, DiagnosticRecorderConfig};
pub use serialization::ReconstructionDiagnostics;
