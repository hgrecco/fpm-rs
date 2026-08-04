//! On-demand Polars conversions for reconstruction and benchmark records.
//!
//! Domain types remain independent of Polars. These free functions allocate
//! one column vector per output column only when a caller explicitly requests a
//! table or writes a bundle.

mod tables;

pub use tables::{
    COMMON_RUN_COLUMNS, ReconstructionTables, algorithm_metrics_dataframe,
    benchmark_artifacts_dataframe, benchmark_frames_dataframe, benchmark_metadata_dataframe,
    benchmark_runs_dataframe, frame_calibration_dataframe, frame_diagnostics_dataframe,
    frame_evaluation_dataframe, history_dataframe, illumination_calibration_dataframe,
    iteration_diagnostics_dataframe, metadata_dataframe, raw_frame_statistics_dataframe,
    reconstruction_tables, scalar_diagnostics_dataframe, summary_dataframe,
};

#[cfg(feature = "parquet")]
/// Parquet/NPY result and benchmark bundle persistence.
pub mod parquet;
