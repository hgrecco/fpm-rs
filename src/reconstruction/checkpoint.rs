use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    array_layout::checked_len_2d,
    array_serde::Array2Data,
    error::Error,
    illumination_calibration::IlluminationCalibrationState,
    measurements::MeasurementRead,
    model::{ImagePlaneModel, Pupil},
};

use super::{
    AlgorithmAuxiliaryState, ReconstructionProblem, ReconstructionState, ReconstructionTrace,
};

/// Current JSON checkpoint serialization format version.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;

/// Serializable algorithm state used to resume a reconstruction exactly.
///
/// Built-in pupil-recovering algorithms capture object and pupil arrays after
/// their iteration-boundary gauge projection. Older valid checkpoints are
/// projected immediately after restoration. Problem-aware validation requires
/// the stored pupil support to equal the compiled model support, not only to
/// have the same shape. Format version 2 uses `algorithm_auxiliary` as an
/// extension point for solver state. Readers predating a particular auxiliary
/// enum variant cannot deserialize checkpoints containing that variant.
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
    pub(crate) physical_illumination_calibration: Option<IlluminationCalibrationState>,
    pub(crate) calibrated_model: Option<ImagePlaneModel>,
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
            physical_illumination_calibration: state.physical_illumination_calibration.clone(),
            calibrated_model: state.calibrated_model.clone(),
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

    /// Borrows physical planar-array calibration state, when joint calibration is active.
    pub fn physical_illumination_calibration(&self) -> Option<&IlluminationCalibrationState> {
        self.physical_illumination_calibration.as_ref()
    }

    /// Borrows the illumination-refreshed model used at checkpoint capture.
    pub fn calibrated_model(&self) -> Option<&ImagePlaneModel> {
        self.calibrated_model.as_ref()
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

    /// Loads a checkpoint and verifies all dimensions, the exact pupil support,
    /// and calibration counts against the problem that will resume it.
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
        if self.physical_illumination_calibration.is_some() != self.calibrated_model.is_some() {
            return Err(Error::InvalidModel(
                "checkpoint physical calibration and calibrated model must be present together"
                    .into(),
            ));
        }
        if let Some(model) = &self.calibrated_model {
            model.validate()?;
        }
        if let Some(calibration) = &self.physical_illumination_calibration {
            calibration.validate()?;
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
                AlgorithmAuxiliaryState::Mpie(mpie) => {
                    mpie.velocity.len() != mpie.anchor.len()
                        || mpie
                            .velocity
                            .iter()
                            .chain(&mpie.anchor)
                            .any(|value| !value.re.is_finite() || !value.im.is_finite())
                        || !mpie.object_step.is_finite()
                        || mpie.object_step <= 0.0
                        || !mpie.stability.is_finite()
                        || !(0.0..=1.0).contains(&mpie.stability)
                        || !mpie.epsilon.is_finite()
                        || mpie.epsilon <= 0.0
                        || mpie.momentum_interval == 0
                        || mpie.effective_frames_since_momentum >= mpie.momentum_interval
                        || !mpie.momentum_friction.is_finite()
                        || !(0.0..1.0).contains(&mpie.momentum_friction)
                        || !mpie.momentum_feedback.is_finite()
                        || !(0.0..=1.0).contains(&mpie.momentum_feedback)
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
        if self.pupil.support.as_slice() != problem.model.pupil().support.as_slice() {
            return Err(Error::InvalidModel(
                "checkpoint pupil support differs from the reconstruction problem".into(),
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
        if let Some(model) = &self.calibrated_model
            && (model.image_shape() != problem.model.image_shape()
                || model.reconstruction_shape() != problem.model.reconstruction_shape()
                || model.source_count() != problem.model.source_count()
                || model.frame_count() != problem.model.frame_count())
        {
            return Err(Error::InvalidModel(
                "checkpoint calibrated model topology differs from the reconstruction problem"
                    .into(),
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
        let object_len = checked_len_2d(problem.model.reconstruction_shape())?;
        if self
            .algorithm_auxiliary
            .as_ref()
            .is_some_and(|auxiliary| match auxiliary {
                AlgorithmAuxiliaryState::Admm(admm) => admm.auxiliary_fields.len() != auxiliary_len,
                AlgorithmAuxiliaryState::Mpie(mpie) => {
                    mpie.velocity.len() != object_len || mpie.anchor.len() != object_len
                }
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
            physical_illumination_calibration: &'a Option<IlluminationCalibrationState>,
            calibrated_model: &'a Option<ImagePlaneModel>,
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
            physical_illumination_calibration: &self.physical_illumination_calibration,
            calibrated_model: &self.calibrated_model,
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
            physical_illumination_calibration: Option<IlluminationCalibrationState>,
            calibrated_model: Option<ImagePlaneModel>,
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
            physical_illumination_calibration: representation.physical_illumination_calibration,
            calibrated_model: representation.calibrated_model,
            algorithm_auxiliary: representation.algorithm_auxiliary,
            trace: representation.trace,
        };
        checkpoint.validate().map_err(D::Error::custom)?;
        Ok(checkpoint)
    }
}
