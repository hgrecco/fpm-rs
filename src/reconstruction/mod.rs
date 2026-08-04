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
mod options;
mod problem;
mod result;
mod runner;
mod schedule;
mod state;
mod trace;

#[cfg(feature = "parquet")]
pub use crate::tabular::parquet::{
    BUNDLE_FORMAT_VERSION, BundleArrays, BundleArtifact, BundleExportOptions, BundlePreviews,
    BundleTables, BundleVerificationResult, ResultBundle, read_bundle,
};
pub use batch::Batch;
pub use checkpoint::{CHECKPOINT_FORMAT_VERSION, ReconstructionCheckpoint};
pub use options::RunOptions;
pub use problem::ReconstructionProblem;
pub use result::{ReconstructionResult, RuntimeInfo};
pub(crate) use result::{save_grayscale, save_signed_grayscale, state_object};
pub use runner::Runner;
pub use schedule::FrameSchedule;
pub use state::{AdmmAuxiliaryState, AlgorithmAuxiliaryState, ReconstructionState};
pub use trace::{AlgorithmMetricRecord, IterationRecord, ReconstructionTrace};
