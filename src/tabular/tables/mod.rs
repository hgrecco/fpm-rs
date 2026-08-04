mod algorithm_metrics;
mod benchmark;
mod calibration;
mod diagnostics;
mod evaluation;
mod history;
mod summary;

use polars::prelude::DataFrame;

pub use algorithm_metrics::algorithm_metrics_dataframe;
pub use benchmark::{
    benchmark_artifacts_dataframe, benchmark_frames_dataframe, benchmark_metadata_dataframe,
    benchmark_runs_dataframe,
};
pub use calibration::{frame_calibration_dataframe, illumination_calibration_dataframe};
pub use diagnostics::{
    frame_diagnostics_dataframe, iteration_diagnostics_dataframe, metadata_dataframe,
    raw_frame_statistics_dataframe, scalar_diagnostics_dataframe,
};
pub use evaluation::frame_evaluation_dataframe;
pub use history::history_dataframe;
pub use summary::{COMMON_RUN_COLUMNS, summary_dataframe};

use crate::{
    Result, diagnostics::ReconstructionDiagnostics, evaluation::ReconstructionEvaluation,
    reconstruction::ReconstructionResult,
};

/// All stable reconstruction tables materialized together on explicit request.
pub struct ReconstructionTables {
    /// Required one-row run summary.
    pub summary: DataFrame,
    /// Required one-row-per-completed-iteration trace.
    pub history: DataFrame,
    /// Optional algorithm-specific scalar metrics.
    pub algorithm_metrics: Option<DataFrame>,
    /// Optional convergence diagnostics by iteration.
    pub iteration_diagnostics: Option<DataFrame>,
    /// Optional comparison metrics by frame and iteration.
    pub frame_diagnostics: Option<DataFrame>,
    /// Optional raw measured-intensity statistics by frame.
    pub raw_frame_statistics: Option<DataFrame>,
    /// Optional predicted-versus-measured evaluation by frame.
    pub frame_evaluation: Option<DataFrame>,
    /// Optional per-source Fourier-grid calibration corrections.
    pub illumination_calibration: Option<DataFrame>,
    /// Optional acquisition-frame gains and backgrounds.
    pub frame_calibration: Option<DataFrame>,
    /// Optional key/value scalar diagnostics.
    pub scalar_diagnostics: Option<DataFrame>,
    /// Optional key/value string metadata.
    pub metadata: Option<DataFrame>,
}

/// Materializes all required and non-empty optional tables for one result and run ID.
pub fn reconstruction_tables(
    run_id: &str,
    result: &ReconstructionResult,
    diagnostics: Option<&ReconstructionDiagnostics>,
    evaluation: Option<&ReconstructionEvaluation>,
) -> Result<ReconstructionTables> {
    Ok(ReconstructionTables {
        summary: summary_dataframe(run_id, result)?,
        history: history_dataframe(run_id, &result.trace)?,
        algorithm_metrics: (!result.trace.algorithm_metrics.is_empty())
            .then(|| algorithm_metrics_dataframe(run_id, &result.trace))
            .transpose()?,
        iteration_diagnostics: diagnostics
            .filter(|value| !value.iteration_diagnostics.is_empty())
            .map(|value| iteration_diagnostics_dataframe(run_id, value))
            .transpose()?,
        frame_diagnostics: diagnostics
            .filter(|value| !value.frame_diagnostics.is_empty())
            .map(|value| frame_diagnostics_dataframe(run_id, value))
            .transpose()?,
        raw_frame_statistics: diagnostics
            .filter(|value| !value.raw_frame_statistics.is_empty())
            .map(|value| raw_frame_statistics_dataframe(run_id, value))
            .transpose()?,
        frame_evaluation: evaluation
            .and_then(|value| value.intensity.as_ref())
            .filter(|value| !value.per_frame.is_empty())
            .map(|value| frame_evaluation_dataframe(run_id, value))
            .transpose()?,
        illumination_calibration: result
            .calibrated_illumination
            .as_ref()
            .map(|_| illumination_calibration_dataframe(run_id, result))
            .transpose()?,
        frame_calibration: (result.recovered_frame_gains.is_some()
            || result.recovered_background.is_some())
        .then(|| frame_calibration_dataframe(run_id, result))
        .transpose()?,
        scalar_diagnostics: (!result.scalar_diagnostics.is_empty())
            .then(|| scalar_diagnostics_dataframe(run_id, &result.scalar_diagnostics))
            .transpose()?,
        metadata: (!result.metadata.is_empty())
            .then(|| metadata_dataframe(run_id, &result.metadata))
            .transpose()?,
    })
}
