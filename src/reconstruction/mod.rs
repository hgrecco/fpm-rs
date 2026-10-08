//! Reconstruction inputs, execution state, schedules, checkpoints, and results.
//!
//! Pair a compiled model and measurements in
//! [`crate::reconstruction::ReconstructionProblem`], then run an
//! [`crate::algorithms::ReconstructionAlgorithm`] directly or configure
//! [`crate::reconstruction::Runner`] with [`crate::reconstruction::RunOptions`],
//! callbacks, and a [`crate::reconstruction::FrameSchedule`]. The owned
//! [`crate::reconstruction::ReconstructionResult`] contains the reconstructed field and
//! trace.

mod batch;
mod checkpoint;
mod opd;
mod options;
mod problem;
mod result;
mod runner;
mod schedule;
mod spectral;
pub(crate) mod spectral_checkpoint;
mod state;
mod trace;

#[cfg(feature = "parquet")]
pub use crate::tabular::parquet::{
    BUNDLE_FORMAT_VERSION, BundleArrays, BundleArtifact, BundleExportOptions, BundlePreviews,
    BundleTables, BundleVerificationResult, ResultBundle, SPECTRAL_BUNDLE_FORMAT_VERSION,
    SpectralResultBundle, read_bundle, read_spectral_bundle,
};
pub use batch::Batch;
pub use checkpoint::{CHECKPOINT_FORMAT_VERSION, ReconstructionCheckpoint};
pub use opd::{
    MultiWavelengthReconstructionResult, OpticalPathDifferenceResult, PhaseReference,
    SyntheticWavelengthUnwrapper,
};
pub use options::RunOptions;
pub use problem::ReconstructionProblem;
pub use result::{ReconstructionResult, RuntimeInfo};
pub(crate) use result::{save_grayscale, save_signed_grayscale, state_object};
pub use runner::Runner;
pub use schedule::FrameSchedule;
pub use spectral::{
    SpectralChannelResult, SpectralFrameSchedule, SpectralReconstructionProblem,
    SpectralReconstructionResult, SpectralReconstructionState, SpectralRunner,
};
pub use spectral_checkpoint::{
    SPECTRAL_CHECKPOINT_FORMAT_VERSION, SpectralCheckpointOptions, SpectralReconstructionCheckpoint,
};
pub use state::{
    AdaptiveAlternatingProjectionAuxiliaryState, AdmmAuxiliaryState, AlgorithmAuxiliaryState,
    MpieAuxiliaryState, ReconstructionState,
};
pub use trace::{AlgorithmMetricRecord, IterationRecord, ReconstructionTrace};
