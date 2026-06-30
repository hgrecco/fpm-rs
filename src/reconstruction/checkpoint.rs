use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Array2, Result, diagnostics::ReconstructionHistory, error::Error,
    measurements::MeasurementRead, model::Pupil,
};

use super::{AlgorithmAuxiliaryState, ReconstructionProblem, ReconstructionState};

pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// Serializable algorithm state used to resume a reconstruction exactly.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReconstructionCheckpoint {
    pub format_version: u32,
    pub completed_iterations: usize,
    pub object_spectrum: Array2<Complex64>,
    pub pupil: Pupil,
    /// Per-source `(row, column)` corrections in Fourier-grid pixels.
    pub illumination_corrections: Option<Vec<(f64, f64)>>,
    pub frame_gains: Option<Vec<f64>>,
    pub background: Option<Vec<f64>>,
    #[serde(default)]
    pub algorithm_auxiliary: Option<AlgorithmAuxiliaryState>,
    pub history: ReconstructionHistory,
}

impl ReconstructionCheckpoint {
    pub fn capture(
        completed_iterations: usize,
        state: &ReconstructionState,
        history: &ReconstructionHistory,
    ) -> Self {
        Self {
            format_version: CHECKPOINT_FORMAT_VERSION,
            completed_iterations,
            object_spectrum: state.object_spectrum.clone(),
            pupil: state.pupil.clone(),
            illumination_corrections: state.illumination_corrections.clone(),
            frame_gains: state.frame_gains.clone(),
            background: state.background.clone(),
            algorithm_auxiliary: state.algorithm_auxiliary.clone(),
            history: history.clone(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        let checkpoint: Self = serde_json::from_reader(reader)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Loads a checkpoint and verifies all dimensions and calibration counts
    /// against the problem that will resume it.
    pub fn load_for_problem<M: MeasurementRead>(
        path: impl AsRef<Path>,
        problem: &ReconstructionProblem<M>,
    ) -> Result<Self> {
        let checkpoint = Self::load(path)?;
        checkpoint.validate_for_problem(problem)?;
        Ok(checkpoint)
    }

    /// Validates integrity that does not depend on a reconstruction problem.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != CHECKPOINT_FORMAT_VERSION {
            return Err(Error::InvalidParameter {
                name: "checkpoint format_version",
                reason: format!(
                    "expected {CHECKPOINT_FORMAT_VERSION}, got {}",
                    self.format_version
                ),
            });
        }
        if self.pupil.support.len() != self.pupil.values.len() {
            return Err(Error::InvalidShape(
                "checkpoint pupil support and values have different lengths".into(),
            ));
        }
        if self
            .object_spectrum
            .as_slice()
            .iter()
            .chain(self.pupil.values.as_slice())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::InvalidModel(
                "checkpoint contains non-finite complex values".into(),
            ));
        }
        if self
            .illumination_corrections
            .as_ref()
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|&(row, column)| !row.is_finite() || !column.is_finite())
            })
        {
            return Err(Error::InvalidModel(
                "checkpoint illumination corrections contain non-finite values".into(),
            ));
        }
        if self.frame_gains.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(Error::InvalidModel(
                "checkpoint frame gains must be finite and positive".into(),
            ));
        }
        if self
            .background
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::InvalidModel(
                "checkpoint background contains non-finite values".into(),
            ));
        }
        if self
            .algorithm_auxiliary
            .as_ref()
            .is_some_and(|auxiliary| match auxiliary {
                AlgorithmAuxiliaryState::Admm(admm) => {
                    admm.auxiliary_fields.len() != admm.dual_fields.len()
                        || admm
                            .auxiliary_fields
                            .iter()
                            .chain(&admm.dual_fields)
                            .any(|value| !value.re.is_finite() || !value.im.is_finite())
                }
            })
        {
            return Err(Error::InvalidModel(
                "checkpoint algorithm auxiliary state is inconsistent or non-finite".into(),
            ));
        }
        let records = &self.history.iterations;
        if records.len() != self.completed_iterations
            || records.iter().enumerate().any(|(index, record)| {
                record.iteration != index + 1
                    || !record.loss.is_finite()
                    || !record.elapsed_seconds.is_finite()
                    || record.elapsed_seconds < 0.0
            })
            || records
                .windows(2)
                .any(|pair| pair[1].elapsed_seconds < pair[0].elapsed_seconds)
        {
            return Err(Error::InvalidModel(
                "checkpoint history is incomplete, non-finite, or non-monotonic".into(),
            ));
        }
        Ok(())
    }

    /// Validates checkpoint dimensions and calibration variables for `problem`.
    pub fn validate_for_problem<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<()> {
        problem.validate()?;
        self.validate()?;
        if self.object_spectrum.shape() != problem.model.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "checkpoint spectrum shape {:?} differs from reconstruction shape {:?}",
                self.object_spectrum.shape(),
                problem.model.reconstruction_shape
            )));
        }
        if self.pupil.shape() != problem.model.image_shape {
            return Err(Error::InvalidShape(
                "checkpoint pupil does not match the model image shape".into(),
            ));
        }
        if self
            .illumination_corrections
            .as_ref()
            .is_some_and(|values| values.len() != problem.model.source_count())
        {
            return Err(Error::InvalidModel(
                "checkpoint illumination correction count does not match the model".into(),
            ));
        }
        if self
            .frame_gains
            .as_ref()
            .is_some_and(|values| values.len() != problem.model.frame_count())
        {
            return Err(Error::InvalidModel(
                "checkpoint frame gain count does not match the model".into(),
            ));
        }
        let image_len = problem.measurements.frame_len();
        let stack_len = image_len
            .checked_mul(problem.model.frame_count())
            .ok_or_else(|| Error::InvalidShape("checkpoint stack length overflows".into()))?;
        if self
            .background
            .as_ref()
            .is_some_and(|values| values.len() != image_len && values.len() != stack_len)
        {
            return Err(Error::InvalidModel(
                "checkpoint background dimensions do not match the model".into(),
            ));
        }
        let mode_count = problem.model.multiplexing_matrix.as_ref().map_or_else(
            || Ok(problem.model.frame_count()),
            |matrix| {
                matrix.iter().try_fold(0_usize, |count, row| {
                    count.checked_add(row.len()).ok_or_else(|| {
                        Error::InvalidShape("checkpoint source mode count overflows".into())
                    })
                })
            },
        )?;
        let auxiliary_len = image_len.checked_mul(mode_count).ok_or_else(|| {
            Error::InvalidShape("checkpoint auxiliary length overflows".into())
        })?;
        if self
            .algorithm_auxiliary
            .as_ref()
            .is_some_and(|auxiliary| match auxiliary {
                AlgorithmAuxiliaryState::Admm(admm) => admm.auxiliary_fields.len() != auxiliary_len,
            })
        {
            return Err(Error::InvalidModel(
                "checkpoint algorithm auxiliary dimensions do not match the model".into(),
            ));
        }
        Ok(())
    }
}
