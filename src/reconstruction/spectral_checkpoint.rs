//! Typed iteration-boundary persistence for spectral AP and shared-OPD descent.

use super::{
    OpticalPathDifferenceResult, ReconstructionTrace, SpectralFrameSchedule,
    SpectralReconstructionProblem,
};
use crate::{
    Error, Result, array_serde::Array2Data, measurements::MeasurementRead,
    model::SpectralImagePlaneModel,
};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

/// Version of the separately tagged spectral checkpoint format.
/// Ordinary reconstruction checkpoint versions are independent.
pub const SPECTRAL_CHECKPOINT_FORMAT_VERSION: u32 = 1;
const KIND: &str = "fpm_rs.spectral_checkpoint";

/// Optional periodic checkpoint output at accepted iteration boundaries.
/// A checkpoint is also retained in supported solver results when no directory is set.
#[derive(Clone, Debug)]
pub struct SpectralCheckpointOptions {
    /// Optional output directory, created as needed. Existing checkpoint files with
    /// the same solver kind and iteration number are atomically replaced.
    pub directory: Option<PathBuf>,
    /// Positive interval in complete passes or accepted OPD updates; default one.
    pub every: usize,
}
impl Default for SpectralCheckpointOptions {
    fn default() -> Self {
        Self {
            directory: None,
            every: 1,
        }
    }
}
impl SpectralCheckpointOptions {
    /// Rejects a zero interval or an empty directory path.
    pub fn validate(&self) -> Result<()> {
        if self.every == 0
            || self
                .directory
                .as_ref()
                .is_some_and(|p| p.as_os_str().is_empty())
        {
            return Err(invalid(
                "checkpoint_options",
                "interval must be positive and directory nonempty",
            ));
        }
        Ok(())
    }
    pub(crate) fn write(
        &self,
        checkpoint: &SpectralReconstructionCheckpoint,
        final_state: bool,
    ) -> Result<()> {
        if let Some(directory) = &self.directory
            && (final_state || checkpoint.completed_iterations.is_multiple_of(self.every))
        {
            fs::create_dir_all(directory)?;
            let prefix = match checkpoint.state {
                StoredSolver::Spectral { .. } => "spectral",
                StoredSolver::JointOpd { .. } => "joint_opd",
            };
            checkpoint.save(directory.join(format!(
                "{prefix}_checkpoint_{:05}.json",
                checkpoint.completed_iterations
            )))?;
        }
        Ok(())
    }
}

/// Validated snapshot of a compiled spectral problem and one solver's numerical state.
///
/// Spectral state stores centered spectra directly. Joint state stores amplitudes,
/// OPD bounds and the fixed gauge, and retains automatic initialization records.
/// Resume checks the exact ordered compiled model and a streaming SHA-256 of
/// detector values, masks and weights. A changed total iteration target is allowed;
/// stepping options and explicitly selected schedules must match.
/// Scientific arrays and objectives resume exactly; elapsed wall times accumulate.
/// The format rejects ordinary checkpoints and other spectral solver kinds.
///
/// # Example
/// ```
/// use fpm_rs::{Result, algorithms::SpectralAlternatingProjection,
///     measurements::MeasurementRead, reconstruction::{SpectralRunner,
///     SpectralReconstructionCheckpoint, SpectralReconstructionProblem,
///     SpectralReconstructionResult}};
/// fn resume<M: MeasurementRead>(problem: &SpectralReconstructionProblem<M>,
///     checkpoint: SpectralReconstructionCheckpoint) -> Result<SpectralReconstructionResult> {
///     SpectralRunner::new(SpectralAlternatingProjection::default().iterations(50))
///         .with_checkpoint(checkpoint).run(problem)
/// }
/// ```
/// The example requires saved default stepping options and a target of at least
/// the completed pass count. The saved schedule is inherited.

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralReconstructionCheckpoint {
    kind: String,
    format_version: u32,
    pub(crate) model: SpectralImagePlaneModel,
    model_fingerprint: String,
    measurement_fingerprint: String,
    pub(crate) completed_iterations: usize,
    pub(crate) elapsed_seconds: f64,
    pub(crate) trace: ReconstructionTrace,
    pub(crate) state: StoredSolver,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum StoredSolver {
    Spectral {
        algorithm: String,
        configuration: String,
        spectra: Vec<Array2Data<Complex64>>,
        schedule: SpectralFrameSchedule,
    },
    JointOpd {
        configuration: String,
        opd: Array2Data<f64>,
        amplitudes: Vec<Array2Data<f64>>,
        opd_range_m: (f64, f64),
        gauge: crate::algorithms::Gauge,
        initialization_opd: Option<Box<StoredOpd>>,
        initialization_trace: Option<ReconstructionTrace>,
    },
}

/// JSON representation of OPD diagnostics. Null is reserved for invalid NaN pixels.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredOpd {
    opd_m: Array2Data<Option<f64>>,
    valid_mask: Array2Data<u8>,
    phase_residual_rad: Array2Data<Option<f64>>,
    fringe_orders: Vec<Array2Data<i64>>,
    phase_offsets_rad: Vec<f64>,
    wavelengths_vacuum_m: Vec<f64>,
    wavelength_ladder_m: Vec<f64>,
}
impl StoredOpd {
    pub(crate) fn from_result(result: &OpticalPathDifferenceResult) -> Self {
        let optional = |a: &Array2<f64>| Array2Data {
            height: a.nrows(),
            width: a.ncols(),
            data: a
                .iter()
                .map(|&v| if v.is_nan() { None } else { Some(v) })
                .collect(),
        };
        Self {
            opd_m: optional(&result.opd_m),
            valid_mask: Array2Data::from_view(result.valid_mask.view()),
            phase_residual_rad: optional(&result.phase_residual_rad),
            fringe_orders: result
                .fringe_orders
                .iter()
                .map(|a| Array2Data::from_view(a.view()))
                .collect(),
            phase_offsets_rad: result.phase_offsets_rad.clone(),
            wavelengths_vacuum_m: result.wavelengths_vacuum_m.clone(),
            wavelength_ladder_m: result.wavelength_ladder_m.clone(),
        }
    }
    pub(crate) fn into_result(self) -> Result<OpticalPathDifferenceResult> {
        let result = OpticalPathDifferenceResult {
            opd_m: self.opd_m.into_array()?.mapv(|v| v.unwrap_or(f64::NAN)),
            valid_mask: self.valid_mask.into_array()?,
            phase_residual_rad: self
                .phase_residual_rad
                .into_array()?
                .mapv(|v| v.unwrap_or(f64::NAN)),
            fringe_orders: self
                .fringe_orders
                .into_iter()
                .map(Array2Data::into_array)
                .collect::<Result<_>>()?,
            phase_offsets_rad: self.phase_offsets_rad,
            wavelengths_vacuum_m: self.wavelengths_vacuum_m,
            wavelength_ladder_m: self.wavelength_ladder_m,
        };
        validate_opd(&result)?;
        Ok(result)
    }
}

pub(crate) fn validate_opd(result: &OpticalPathDifferenceResult) -> Result<()> {
    let shape = result.opd_m.dim();
    crate::array_layout::checked_len_2d(shape)?;
    let count = result.wavelengths_vacuum_m.len();
    if count < 2
        || result.valid_mask.dim() != shape
        || result.phase_residual_rad.dim() != shape
        || result.fringe_orders.len() != count
        || result.phase_offsets_rad.len() != count
        || result.fringe_orders.iter().any(|a| a.dim() != shape)
    {
        return Err(Error::InvalidShape(
            "OPD persistence arrays and channel records disagree".into(),
        ));
    }
    let mut wavelengths = result.wavelengths_vacuum_m.clone();
    wavelengths.sort_by(f64::total_cmp);
    if wavelengths.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || wavelengths.windows(2).any(|w| w[0] == w[1])
        || result.phase_offsets_rad.iter().any(|v| !v.is_finite())
        || result.wavelength_ladder_m.is_empty()
        || result
            .wavelength_ladder_m
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        || result.wavelength_ladder_m.windows(2).any(|w| w[0] <= w[1])
    {
        return Err(invalid(
            "opd_metadata",
            "wavelengths, ladder, and phase offsets are invalid",
        ));
    }
    for ((&opd, &valid), &residual) in result
        .opd_m
        .iter()
        .zip(&result.valid_mask)
        .zip(&result.phase_residual_rad)
    {
        if valid > 1
            || (valid == 1 && (!opd.is_finite() || !residual.is_finite() || residual < 0.0))
            || (valid == 0 && (!opd.is_nan() || !residual.is_nan()))
        {
            return Err(invalid(
                "opd_arrays",
                "valid pixels must be finite and invalid pixels must be NaN",
            ));
        }
    }
    Ok(())
}

impl SpectralReconstructionCheckpoint {
    pub(crate) fn capture(
        model: SpectralImagePlaneModel,
        fingerprints: &(String, String),
        completed_iterations: usize,
        elapsed_seconds: f64,
        trace: ReconstructionTrace,
        state: StoredSolver,
    ) -> Self {
        Self {
            kind: KIND.into(),
            format_version: SPECTRAL_CHECKPOINT_FORMAT_VERSION,
            model,
            model_fingerprint: fingerprints.0.clone(),
            measurement_fingerprint: fingerprints.1.clone(),
            completed_iterations,
            elapsed_seconds,
            trace,
            state,
        }
    }
    /// Returns the separately versioned format number.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }
    /// Returns completed passes or accepted joint updates, including those before resume.
    pub fn completed_iterations(&self) -> usize {
        self.completed_iterations
    }
    /// Returns the solver name; it determines which resume entry point may consume this snapshot.
    pub fn algorithm(&self) -> &str {
        match &self.state {
            StoredSolver::Spectral { algorithm, .. } => {
                algorithm.rsplit("::").next().unwrap_or(algorithm)
            }
            StoredSolver::JointOpd { .. } => "MultiWavelengthGradientDescent",
        }
    }
    /// Borrows the exact ordered compiled model, including fixed pupils and detector calibration.
    pub fn model(&self) -> &SpectralImagePlaneModel {
        &self.model
    }
    /// Borrows the accumulated objective trace; timing includes earlier runs.
    pub fn trace(&self) -> &ReconstructionTrace {
        &self.trace
    }
    /// Borrows the authoritative joint OPD map in metres, or returns None for spectral AP.
    pub fn opd_m(&self) -> Option<ArrayView2<'_, f64>> {
        match &self.state {
            StoredSolver::JointOpd { opd, .. } => Some(
                ArrayView2::from_shape((opd.height, opd.width), &opd.data)
                    .expect("validated checkpoint OPD"),
            ),
            _ => None,
        }
    }
    /// Validates and atomically saves JSON, preserving binary64 values through float-roundtrip parsing.
    /// A failed write leaves an existing destination intact.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let temporary = parent.join(format!(".spectral-checkpoint-{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            // Close the file before publishing it: Windows does not permit a
            // rename while a writer that disallows delete sharing is open.
            // This also keeps checkpoint publication independent of platform
            // file-sharing defaults.
            {
                let mut writer = BufWriter::new(File::create(&temporary)?);
                serde_json::to_writer(&mut writer, self)?;
                writer.flush()?;
                writer.get_ref().sync_all()?;
            }
            fs::rename(&temporary, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }
    /// Loads a spectral checkpoint, rejecting unknown fields, wrong kind/version, and invalid state.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let checkpoint: Self = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    /// Loads and verifies the model plus a streaming fingerprint of measurements, masks, and weights.
    pub fn load_for_problem<M: MeasurementRead>(
        path: impl AsRef<Path>,
        problem: &SpectralReconstructionProblem<M>,
    ) -> Result<Self> {
        let checkpoint = Self::load(path)?;
        checkpoint.validate_for_problem(problem)?;
        Ok(checkpoint)
    }
    /// Checks metadata, array layouts/shapes, finite values, gauge, and contiguous trace numbering.
    pub fn validate(&self) -> Result<()> {
        if self.kind != KIND || self.format_version != SPECTRAL_CHECKPOINT_FORMAT_VERSION {
            return Err(invalid(
                "spectral_checkpoint",
                "unsupported checkpoint kind or version",
            ));
        }
        self.model.validate()?;
        if self.model_fingerprint != model_fingerprint(&self.model)?
            || !is_hash(&self.measurement_fingerprint)
        {
            return Err(invalid(
                "spectral_checkpoint",
                "model or measurement fingerprint is invalid",
            ));
        }
        if !self.elapsed_seconds.is_finite() || self.elapsed_seconds < 0.0 {
            return Err(invalid("elapsed_seconds", "must be finite and nonnegative"));
        }
        let shape = self.model.reconstruction_shape();
        let joint = matches!(self.state, StoredSolver::JointOpd { .. });
        let expected = self
            .completed_iterations
            .checked_add(usize::from(joint))
            .ok_or_else(|| invalid("completed_iterations", "trace length overflow"))?;
        if self.trace.iterations.len() != expected {
            return Err(invalid(
                "spectral_trace",
                "trace length disagrees with completed iterations",
            ));
        }
        let mut previous_time = 0.0;
        for (index, row) in self.trace.iterations.iter().enumerate() {
            if row.iteration != index + usize::from(!joint)
                || !row.objective.is_finite()
                || !row.elapsed_seconds.is_finite()
                || row.elapsed_seconds < previous_time
                || row.elapsed_seconds > self.elapsed_seconds
            {
                return Err(invalid(
                    "spectral_trace",
                    "trace numbering, objectives, or times are invalid",
                ));
            }
            previous_time = row.elapsed_seconds;
        }
        if self
            .trace
            .algorithm_metrics
            .iter()
            .any(|m| !m.value.is_finite() || m.iteration > self.completed_iterations)
        {
            return Err(invalid("spectral_trace", "algorithm metrics are invalid"));
        }
        match &self.state {
            StoredSolver::Spectral {
                algorithm,
                configuration,
                spectra,
                schedule,
            } => {
                if algorithm.trim().is_empty()
                    || serde_json::from_str::<serde_json::Value>(configuration).is_err()
                    || spectra.len() != self.model.object_count()
                {
                    return Err(invalid(
                        "spectral_state",
                        "invalid algorithm, options, or spectrum count",
                    ));
                }
                schedule.order(self.model.frame_count(), self.completed_iterations)?;
                for a in spectra {
                    validate_array(a, shape, |v| v.re.is_finite() && v.im.is_finite())?;
                }
            }
            StoredSolver::JointOpd {
                configuration,
                opd,
                amplitudes,
                opd_range_m,
                gauge,
                initialization_opd,
                initialization_trace,
            } => {
                let options: crate::algorithms::MultiWavelengthGradientDescent =
                    serde_json::from_str(configuration)?;
                options.validate()?;
                if self.model.object_coupling() != crate::model::ObjectCoupling::Independent
                    || self.model.channels().len() < 2
                    || amplitudes.len() != self.model.channels().len()
                {
                    return Err(invalid(
                        "joint_state",
                        "requires independent channel amplitudes and at least two channels",
                    ));
                }
                super::SyntheticWavelengthUnwrapper::new(*opd_range_m)?;
                validate_array(opd, shape, |v| {
                    v.is_finite() && *v >= opd_range_m.0 && *v < opd_range_m.1
                })?;
                for a in amplitudes {
                    validate_array(a, shape, |v| v.is_finite() && *v >= options.epsilon.sqrt())?;
                }
                gauge.validate(opd, *opd_range_m)?;
                if initialization_opd.is_some() != initialization_trace.is_some() {
                    return Err(invalid(
                        "initialization",
                        "OPD diagnostics and AP trace must be present together",
                    ));
                }
                if let Some(stored) = initialization_opd {
                    let result = stored.clone().into_result()?;
                    if result.opd_m.dim() != shape
                        || result.valid_mask.iter().any(|&v| v != 1)
                        || result.wavelengths_vacuum_m
                            != self
                                .model
                                .channels()
                                .iter()
                                .map(|c| c.model.sampling().wavelength.unwrap())
                                .collect::<Vec<_>>()
                    {
                        return Err(invalid(
                            "initialization_opd",
                            "initialization must be valid on the common grid and channel order",
                        ));
                    }
                }
                if let Some(trace) = initialization_trace
                    && (trace.iterations.len() != options.initialization_iterations
                        || trace.iterations.iter().enumerate().any(|(i, r)| {
                            r.iteration != i + 1
                                || !r.objective.is_finite()
                                || !r.elapsed_seconds.is_finite()
                                || r.elapsed_seconds < 0.0
                        }))
                {
                    return Err(invalid(
                        "initialization_trace",
                        "initialization trace is invalid",
                    ));
                }
            }
        }
        Ok(())
    }
    /// Rejects changed ordered kernels, acquisition, detector data, masks, or weights.
    /// Fingerprinting streams one detector frame at a time and performs no reconstruction.
    pub fn validate_for_problem<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
    ) -> Result<()> {
        self.validate()?;
        let current = fingerprints(problem)?;
        if current.0 != self.model_fingerprint || current.1 != self.measurement_fingerprint {
            return Err(invalid(
                "spectral_checkpoint",
                "compiled spectral model or detector measurements changed",
            ));
        }
        Ok(())
    }
    pub(crate) fn fingerprints(&self) -> (String, String) {
        (
            self.model_fingerprint.clone(),
            self.measurement_fingerprint.clone(),
        )
    }
}

pub(crate) fn fingerprints<M: MeasurementRead>(
    problem: &SpectralReconstructionProblem<M>,
) -> Result<(String, String)> {
    problem.validate()?;
    let mut hash = Sha256::new();
    hash.update(b"fpm_rs.spectral_measurements.v1");
    hash.update((problem.measurements.frame_count() as u64).to_le_bytes());
    let shape = problem.measurements.image_shape();
    hash.update((shape.0 as u64).to_le_bytes());
    hash.update((shape.1 as u64).to_le_bytes());
    for frame in 0..problem.measurements.frame_count() {
        hash.update(problem.measurements.frame_weight(frame)?.to_le_bytes());
        let values = problem.measurements.frame(frame)?;
        if values.len() != shape.0 * shape.1 || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidMeasurements(format!(
                "frame {frame} must have the declared shape and finite detector values"
            )));
        }
        for &value in values.iter() {
            hash.update(value.to_le_bytes());
        }
        match problem.measurements.frame_mask(frame)? {
            None => hash.update([0]),
            Some(mask) => {
                hash.update([1]);
                hash.update(mask);
            }
        }
    }
    Ok((
        model_fingerprint(&problem.model)?,
        format!("{:x}", hash.finalize()),
    ))
}
fn model_fingerprint(model: &SpectralImagePlaneModel) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(model)?)))
}
fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn validate_array<T>(
    a: &Array2Data<T>,
    shape: (usize, usize),
    finite: impl Fn(&T) -> bool,
) -> Result<()> {
    let len = crate::array_layout::checked_len_2d(shape)?;
    if (a.height, a.width) != shape || a.data.len() != len || a.data.iter().any(|v| !finite(v)) {
        return Err(invalid(
            "spectral_state_arrays",
            "arrays must have the common shape and valid finite values",
        ));
    }
    Ok(())
}
fn invalid(name: &'static str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name,
        reason: reason.into(),
    }
}
