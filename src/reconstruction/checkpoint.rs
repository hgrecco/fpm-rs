use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Result, array_serde::Array2Data, error::Error, measurements::MeasurementRead, model::Pupil,
};

use super::{
    AlgorithmAuxiliaryState, ReconstructionProblem, ReconstructionState, ReconstructionTrace,
};

/// Current JSON checkpoint serialization format version.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// Serializable algorithm state used to resume a reconstruction exactly.
#[derive(Clone, Debug)]
pub struct ReconstructionCheckpoint {
    pub(crate) format_version: u32,
    pub(crate) completed_iterations: usize,
    pub(crate) object_spectrum: Array2<Complex64>,
    pub(crate) pupil: Pupil,
    /// Per-source `(row, column)` corrections in Fourier-grid pixels.
    pub(crate) illumination_corrections: Option<Vec<(f64, f64)>>,
    pub(crate) frame_gains: Option<Vec<f64>>,
    pub(crate) background: Option<Vec<f64>>,
    pub(crate) algorithm_auxiliary: Option<AlgorithmAuxiliaryState>,
    pub(crate) trace: ReconstructionTrace,
}

impl ReconstructionCheckpoint {
    /// Clones resumable state and trace after `completed_iterations` complete passes.
    pub fn capture(
        completed_iterations: usize,
        state: &ReconstructionState,
        trace: &ReconstructionTrace,
    ) -> Self {
        Self {
            format_version: CHECKPOINT_FORMAT_VERSION,
            completed_iterations,
            object_spectrum: state.object_spectrum.clone().into_inner(),
            pupil: state.pupil.clone(),
            illumination_corrections: state.illumination_corrections.clone(),
            frame_gains: state.frame_gains.clone(),
            background: state.background.clone(),
            algorithm_auxiliary: state.algorithm_auxiliary.clone(),
            trace: trace.clone(),
        }
    }

    /// Returns the serialized checkpoint format version.
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns the number of complete iterations represented by this state.
    pub const fn completed_iterations(&self) -> usize {
        self.completed_iterations
    }

    /// Borrows the centered high-resolution complex object spectrum.
    pub fn object_spectrum(&self) -> ArrayView2<'_, Complex64> {
        self.object_spectrum.view()
    }

    /// Borrows the low-resolution recovered pupil state.
    pub fn pupil(&self) -> &Pupil {
        &self.pupil
    }

    /// Borrows optional per-source `(row, column)` corrections in Fourier-grid pixels.
    pub fn illumination_corrections(&self) -> Option<&[(f64, f64)]> {
        self.illumination_corrections.as_deref()
    }

    /// Borrows optional positive calibration gains in acquisition-frame order.
    pub fn frame_gains(&self) -> Option<&[f64]> {
        self.frame_gains.as_deref()
    }

    /// Borrows optional additive intensity backgrounds in acquisition-frame order.
    pub fn background(&self) -> Option<&[f64]> {
        self.background.as_deref()
    }

    /// Borrows optional solver-specific resumable state.
    pub fn algorithm_auxiliary(&self) -> Option<&AlgorithmAuxiliaryState> {
        self.algorithm_auxiliary.as_ref()
    }

    /// Borrows the iteration trace accumulated before capture.
    pub const fn trace(&self) -> &ReconstructionTrace {
        &self.trace
    }

    /// Validates and serializes this checkpoint as JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Deserializes and validates a JSON checkpoint independently of a problem.
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
        let records = &self.trace.iterations;
        if records.len() != self.completed_iterations
            || records.iter().enumerate().any(|(index, record)| {
                record.iteration != index + 1
                    || !record.objective.is_finite()
                    || !record.elapsed_seconds.is_finite()
                    || record.elapsed_seconds < 0.0
            })
            || records
                .windows(2)
                .any(|pair| pair[1].elapsed_seconds < pair[0].elapsed_seconds)
        {
            return Err(Error::InvalidModel(
                "checkpoint trace is incomplete, non-finite, or non-monotonic".into(),
            ));
        }
        if self.trace.algorithm_metrics.iter().any(|record| {
            record.iteration == 0
                || record.iteration > self.completed_iterations
                || record.namespace.is_empty()
                || record.metric.is_empty()
                || !record.value.is_finite()
        }) {
            return Err(Error::InvalidModel(
                "checkpoint algorithm metrics are invalid".into(),
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
        if self.object_spectrum.dim() != problem.model.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "checkpoint spectrum shape {:?} differs from reconstruction shape {:?}",
                self.object_spectrum.dim(),
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
        let auxiliary_len = image_len
            .checked_mul(mode_count)
            .ok_or_else(|| Error::InvalidShape("checkpoint auxiliary length overflows".into()))?;
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

impl Serialize for ReconstructionCheckpoint {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Representation<'a> {
            format_version: u32,
            completed_iterations: usize,
            object_spectrum: Array2Data<Complex64>,
            pupil: &'a Pupil,
            illumination_corrections: &'a Option<Vec<(f64, f64)>>,
            frame_gains: &'a Option<Vec<f64>>,
            background: &'a Option<Vec<f64>>,
            algorithm_auxiliary: &'a Option<AlgorithmAuxiliaryState>,
            trace: &'a ReconstructionTrace,
        }

        Representation {
            format_version: self.format_version,
            completed_iterations: self.completed_iterations,
            object_spectrum: Array2Data::from_view(self.object_spectrum.view()),
            pupil: &self.pupil,
            illumination_corrections: &self.illumination_corrections,
            frame_gains: &self.frame_gains,
            background: &self.background,
            algorithm_auxiliary: &self.algorithm_auxiliary,
            trace: &self.trace,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ReconstructionCheckpoint {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Representation {
            format_version: u32,
            completed_iterations: usize,
            object_spectrum: Array2Data<Complex64>,
            pupil: Pupil,
            illumination_corrections: Option<Vec<(f64, f64)>>,
            frame_gains: Option<Vec<f64>>,
            background: Option<Vec<f64>>,
            #[serde(default)]
            algorithm_auxiliary: Option<AlgorithmAuxiliaryState>,
            trace: ReconstructionTrace,
        }

        let representation = Representation::deserialize(deserializer)?;
        let checkpoint = Self {
            format_version: representation.format_version,
            completed_iterations: representation.completed_iterations,
            object_spectrum: representation
                .object_spectrum
                .into_array()
                .map_err(D::Error::custom)?,
            pupil: representation.pupil,
            illumination_corrections: representation.illumination_corrections,
            frame_gains: representation.frame_gains,
            background: representation.background,
            algorithm_auxiliary: representation.algorithm_auxiliary,
            trace: representation.trace,
        };
        checkpoint.validate().map_err(D::Error::custom)?;
        Ok(checkpoint)
    }
}
