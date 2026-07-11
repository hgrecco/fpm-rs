use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use serde::{Deserialize, Serialize};

use super::{FrameDiagnostics, IterationDiagnostics, RawFrameStats};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReconstructionDiagnostics {
    pub iteration_history: Vec<IterationDiagnostics>,

    pub frame_diagnostics: Vec<FrameDiagnostics>,

    pub raw_frame_stats: Vec<RawFrameStats>,

    pub coverage: Option<super::FourierCoverageDiagnostics>,

    pub ground_truth_metrics: Option<super::GroundTruthMetrics>,
}

impl ReconstructionDiagnostics {
    pub fn to_json_file<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn from_json_file<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Ok(serde_json::from_reader(reader)?)
    }
}
