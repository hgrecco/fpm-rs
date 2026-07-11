use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use image::{GrayImage, Luma};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Array2, Result,
    backend::FftDirection,
    complex,
    diagnostics::ReconstructionHistory,
    error::Error,
    model::{Pupil, ifftshift_copy},
};

use super::ReconstructionState;

pub const RESULT_BUNDLE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub elapsed_seconds: f64,
    pub completed_iterations: usize,
    pub stopped_early: bool,
    pub algorithm: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReconstructionResult {
    pub object: Array2<Complex64>,
    pub amplitude: Array2<f64>,
    pub phase: Array2<f64>,
    pub object_spectrum: Array2<Complex64>,
    pub recovered_pupil: Pupil,
    /// Per-source `(row, column)` corrections in Fourier-grid pixels.
    pub calibrated_illumination: Option<Vec<(f64, f64)>>,
    pub recovered_frame_gains: Option<Vec<f64>>,
    pub recovered_background: Option<Vec<f64>>,
    pub history: ReconstructionHistory,
    pub diagnostics: BTreeMap<String, f64>,
    pub runtime: RuntimeInfo,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize)]
struct ReconstructionResultBundle {
    format_version: u32,
    result: ReconstructionResult,
}

impl ReconstructionResult {
    pub(crate) fn from_state(
        state: &mut ReconstructionState,
        history: ReconstructionHistory,
        runtime: RuntimeInfo,
    ) -> Result<Self> {
        let object = state_object(state)?;
        let amplitude = complex::amplitude(&object);
        let phase = complex::phase(&object);
        let mut diagnostics = BTreeMap::new();
        if let Some(loss) = history.final_loss() {
            diagnostics.insert("final_loss".into(), loss);
        }
        if let Some(record) = history.iterations.last() {
            if let Some(residual) = record.admm_primal_residual_rms {
                diagnostics.insert("final_admm_primal_residual_rms".into(), residual);
            }
            if let Some(residual) = record.admm_dual_residual_rms {
                diagnostics.insert("final_admm_dual_residual_rms".into(), residual);
            }
        }
        Ok(Self {
            object,
            amplitude,
            phase,
            object_spectrum: state.object_spectrum.clone(),
            recovered_pupil: state.pupil.clone(),
            calibrated_illumination: state.illumination_corrections.clone(),
            recovered_frame_gains: state.frame_gains.clone(),
            recovered_background: state.background.clone(),
            history,
            diagnostics,
            runtime,
            metadata: BTreeMap::new(),
        })
    }

    pub fn save_amplitude(&self, path: impl AsRef<Path>) -> Result<()> {
        save_grayscale(&self.amplitude, path, false)
    }

    pub fn save_phase(&self, path: impl AsRef<Path>) -> Result<()> {
        save_grayscale(&self.phase, path, true)
    }

    pub fn save_complex_object(&self, path: impl AsRef<Path>) -> Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(writer, &self.object)?;
        Ok(())
    }

    pub fn save_pupil(&self, path: impl AsRef<Path>) -> Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(writer, &self.recovered_pupil)?;
        Ok(())
    }

    pub fn save_loss_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record([
            "iteration",
            "loss",
            "elapsed_seconds",
            "admm_primal_residual_rms",
            "admm_dual_residual_rms",
        ])?;
        for record in &self.history.iterations {
            writer.serialize((
                record.iteration,
                record.loss,
                record.elapsed_seconds,
                record.admm_primal_residual_rms,
                record.admm_dual_residual_rms,
            ))?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Saves every result array, calibration value, diagnostic, history entry,
    /// runtime field, and metadata value in a versioned JSON bundle.
    pub fn save_bundle(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(
            writer,
            &ReconstructionResultBundle {
                format_version: RESULT_BUNDLE_FORMAT_VERSION,
                result: self.clone(),
            },
        )?;
        Ok(())
    }

    pub fn load_bundle(path: impl AsRef<Path>) -> Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        let bundle: ReconstructionResultBundle = serde_json::from_reader(reader)?;
        if bundle.format_version != RESULT_BUNDLE_FORMAT_VERSION {
            return Err(Error::InvalidParameter {
                name: "result bundle format_version",
                reason: format!(
                    "expected {RESULT_BUNDLE_FORMAT_VERSION}, got {}",
                    bundle.format_version
                ),
            });
        }
        bundle.result.validate()?;
        Ok(bundle.result)
    }

    pub fn validate(&self) -> Result<()> {
        let shape = self.object.shape();
        if self.amplitude.shape() != shape
            || self.phase.shape() != shape
            || self.object_spectrum.shape() != shape
        {
            return Err(Error::InvalidShape(
                "result object, amplitude, phase, and spectrum shapes must match".into(),
            ));
        }
        if self.recovered_pupil.support.len() != self.recovered_pupil.values.len() {
            return Err(Error::InvalidShape(
                "result pupil support and values have different lengths".into(),
            ));
        }
        if self
            .object
            .as_slice()
            .iter()
            .chain(self.object_spectrum.as_slice())
            .chain(self.recovered_pupil.values.as_slice())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
            || self
                .amplitude
                .as_slice()
                .iter()
                .chain(self.phase.as_slice())
                .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidModel(
                "result arrays contain non-finite values".into(),
            ));
        }
        if self.calibrated_illumination.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|&(row, column)| !row.is_finite() || !column.is_finite())
        }) || self.recovered_frame_gains.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        }) || self
            .recovered_background
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::InvalidModel(
                "result calibration values are invalid".into(),
            ));
        }
        if self.diagnostics.values().any(|value| !value.is_finite()) {
            return Err(Error::InvalidModel(
                "result diagnostics contain non-finite values".into(),
            ));
        }
        if !self.runtime.elapsed_seconds.is_finite()
            || self.runtime.elapsed_seconds < 0.0
            || self.runtime.algorithm.is_empty()
            || self.runtime.completed_iterations != self.history.iterations.len()
            || self
                .history
                .iterations
                .iter()
                .enumerate()
                .any(|(index, record)| {
                    record.iteration != index + 1
                        || !record.loss.is_finite()
                        || !record.elapsed_seconds.is_finite()
                        || record.elapsed_seconds < 0.0
                        || record
                            .admm_primal_residual_rms
                            .is_some_and(|value| !value.is_finite() || value < 0.0)
                        || record
                            .admm_dual_residual_rms
                            .is_some_and(|value| !value.is_finite() || value < 0.0)
                })
            || self
                .history
                .iterations
                .windows(2)
                .any(|pair| pair[1].elapsed_seconds < pair[0].elapsed_seconds)
            || self
                .history
                .iterations
                .last()
                .is_some_and(|record| record.elapsed_seconds > self.runtime.elapsed_seconds)
        {
            return Err(Error::InvalidModel(
                "result runtime and history are inconsistent".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn state_object(state: &mut ReconstructionState) -> Result<Array2<Complex64>> {
    if let Some(cached) = &state.object_real_space_cache {
        return Ok(cached.clone());
    }
    let shape = state.object_spectrum.shape();
    let mut unshifted = vec![Complex64::default(); state.object_spectrum.len()];
    ifftshift_copy(state.object_spectrum.as_slice(), &mut unshifted, shape);
    state.backend.fft2(
        &mut unshifted,
        shape,
        FftDirection::Inverse,
        &mut state.scratch.column,
    )?;
    let object = Array2::from_vec(shape, unshifted)?;
    state.object_real_space_cache = Some(object.clone());
    Ok(object)
}

pub(crate) fn save_grayscale(
    values: &Array2<f64>,
    path: impl AsRef<Path>,
    phase: bool,
) -> Result<()> {
    let range = if phase {
        (-std::f64::consts::PI, std::f64::consts::PI)
    } else {
        let minimum = values
            .as_slice()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let maximum = values
            .as_slice()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        (minimum, maximum)
    };
    save_grayscale_with_range(values, path, range)
}

pub(crate) fn save_signed_grayscale(values: &Array2<f64>, path: impl AsRef<Path>) -> Result<()> {
    let maximum_absolute = values
        .as_slice()
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(f64::EPSILON);
    save_grayscale_with_range(values, path, (-maximum_absolute, maximum_absolute))
}

fn save_grayscale_with_range(
    values: &Array2<f64>,
    path: impl AsRef<Path>,
    (minimum, maximum): (f64, f64),
) -> Result<()> {
    let width = u32::try_from(values.width()).map_err(|_| {
        Error::InvalidShape("image width does not fit the PNG dimension type".into())
    })?;
    let height = u32::try_from(values.height()).map_err(|_| {
        Error::InvalidShape("image height does not fit the PNG dimension type".into())
    })?;
    let range = (maximum - minimum).max(f64::EPSILON);
    let mut image = GrayImage::new(width, height);
    for row in 0..values.height() {
        for column in 0..values.width() {
            let normalized = ((values[(row, column)] - minimum) / range).clamp(0.0, 1.0);
            image.put_pixel(
                column as u32,
                row as u32,
                Luma([(normalized * 255.0).round() as u8]),
            );
        }
    }
    image.save(path)?;
    Ok(())
}
