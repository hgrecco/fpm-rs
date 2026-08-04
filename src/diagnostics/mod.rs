//! Structured convergence, coverage, and per-frame reconstruction diagnostics.
//!
//! [`crate::diagnostics::DiagnosticRecorder`] is a callback that gathers requested data
//! during a run. It produces [`crate::diagnostics::ReconstructionDiagnostics`], while
//! [`crate::diagnostics::compute_fourier_coverage`] describes which high-resolution
//! Fourier pixels are sampled by a compiled model.

mod context;
mod convergence;
mod coverage;
mod recorder;
mod records;
mod serialization;

pub use context::{DiagnosticRequest, Diagnostics};
pub use convergence::IterationDiagnostics;
pub use coverage::{CropIndexDiagnostics, FourierCoverageDiagnostics, compute_fourier_coverage};
pub use recorder::{DiagnosticRecorder, DiagnosticRecorderConfig};
pub use records::{FrameDiagnosticRecord, RawFrameStatisticsRecord};
pub use serialization::ReconstructionDiagnostics;
