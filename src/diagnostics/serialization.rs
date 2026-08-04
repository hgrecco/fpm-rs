use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use serde::{Deserialize, Serialize};

use super::{FrameDiagnosticRecord, IterationDiagnostics, RawFrameStatisticsRecord};

/// Serializable diagnostic collections emitted by [`super::DiagnosticRecorder`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReconstructionDiagnostics {
    /// Convergence summaries in completed-iteration order.
    pub iteration_diagnostics: Vec<IterationDiagnostics>,

    /// Per-frame comparison records in capture order.
    pub frame_diagnostics: Vec<FrameDiagnosticRecord>,

    /// Raw measured-frame statistics in acquisition order.
    pub raw_frame_statistics: Vec<RawFrameStatisticsRecord>,

    /// Optional compiled-model Fourier coverage summary.
    pub coverage: Option<super::FourierCoverageDiagnostics>,

    /// Optional ground-truth complex-object metrics.
    pub ground_truth_metrics: Option<crate::metrics::complex_field::ComplexFieldComparisonMetrics>,
}

impl ReconstructionDiagnostics {
    /// Serializes diagnostics as pretty-printed JSON.
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Deserializes reconstruction diagnostics from JSON.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Ok(serde_json::from_reader(reader)?)
    }
}
