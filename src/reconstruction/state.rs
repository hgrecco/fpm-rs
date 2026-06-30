use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    Array2, Result,
    backend::{Backend, CpuBackend, FftDirection},
    error::Error,
    measurements::MeasurementRead,
    model::{FourierOffset, ImagePlaneModel, Pupil, fftshift_copy},
};

use super::{ReconstructionCheckpoint, ReconstructionProblem};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdmmAuxiliaryState {
    pub auxiliary_fields: Vec<Complex64>,
    pub dual_fields: Vec<Complex64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AlgorithmAuxiliaryState {
    Admm(AdmmAuxiliaryState),
}

#[derive(Clone, Debug)]
pub struct ReconstructionScratch {
    pub patch: Vec<Complex64>,
    pub exit_spectrum: Vec<Complex64>,
    pub field: Vec<Complex64>,
    pub projected_field: Vec<Complex64>,
    pub projected_spectrum: Vec<Complex64>,
    pub difference: Vec<Complex64>,
    /// High-resolution accumulator used by mini-batch algorithms.
    pub object_gradient: Vec<Complex64>,
    pub regularization_field: Vec<Complex64>,
    pub pupil_gradient: Vec<Complex64>,
    pub calibration_reference: Vec<f64>,
    pub illumination_gradient: Vec<(f64, f64)>,
    pub illumination_curvature: Vec<(f64, f64)>,
    pub illumination_weight: Vec<f64>,
    /// Per-source low-resolution fields for an incoherently multiplexed frame.
    pub multiplex_fields: Vec<Complex64>,
    /// Matching pre-update object patches for multiplexed projection updates.
    pub multiplex_patches: Vec<Complex64>,
    pub multiplex_offsets: Vec<FourierOffset>,
    pub column: Vec<Complex64>,
}

impl ReconstructionScratch {
    fn new(low_shape: (usize, usize), high_shape: (usize, usize)) -> Self {
        let low_len = low_shape.0 * low_shape.1;
        Self {
            patch: vec![Complex64::default(); low_len],
            exit_spectrum: vec![Complex64::default(); low_len],
            field: vec![Complex64::default(); low_len],
            projected_field: vec![Complex64::default(); low_len],
            projected_spectrum: vec![Complex64::default(); low_len],
            difference: vec![Complex64::default(); low_len],
            object_gradient: Vec::new(),
            regularization_field: Vec::new(),
            pupil_gradient: vec![Complex64::default(); low_len],
            calibration_reference: vec![0.0; low_len],
            illumination_gradient: Vec::new(),
            illumination_curvature: Vec::new(),
            illumination_weight: Vec::new(),
            multiplex_fields: Vec::new(),
            multiplex_patches: Vec::new(),
            multiplex_offsets: Vec::new(),
            column: vec![Complex64::default(); low_shape.0.max(high_shape.0)],
        }
    }
}

#[derive(Clone)]
pub struct ReconstructionState {
    pub object_spectrum: Array2<Complex64>,
    pub object_real_space_cache: Option<Array2<Complex64>>,
    pub pupil: Pupil,
    /// Per-source `(row, column)` corrections in Fourier-grid pixels.
    pub illumination_corrections: Option<Vec<(f64, f64)>>,
    pub frame_gains: Option<Vec<f64>>,
    pub background: Option<Vec<f64>>,
    pub algorithm_auxiliary: Option<AlgorithmAuxiliaryState>,
    pub scratch: ReconstructionScratch,
    pub(crate) backend: Arc<dyn Backend>,
}

impl std::fmt::Debug for ReconstructionState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReconstructionState")
            .field("object_spectrum", &self.object_spectrum)
            .field("pupil", &self.pupil)
            .field("illumination_corrections", &self.illumination_corrections)
            .field("frame_gains", &self.frame_gains)
            .field("background", &self.background)
            .field("algorithm_auxiliary", &self.algorithm_auxiliary)
            .finish_non_exhaustive()
    }
}

impl ReconstructionState {
    pub fn effective_source_offset(
        &self,
        model: &ImagePlaneModel,
        source: usize,
    ) -> Result<FourierOffset> {
        let base = model.source_offset(source)?;
        let correction = match &self.illumination_corrections {
            None => (0.0, 0.0),
            Some(values) => *values.get(source).ok_or_else(|| {
                Error::InvalidModel(
                    "illumination correction count does not match source count".into(),
                )
            })?,
        };
        if !correction.0.is_finite() || !correction.1.is_finite() {
            return Err(Error::InvalidModel(
                "illumination corrections must be finite".into(),
            ));
        }
        let effective = FourierOffset::new(base.row + correction.0, base.column + correction.1);
        model.validate_source_offset(source, effective)?;
        Ok(effective)
    }

    pub fn initialize<M: MeasurementRead>(problem: &ReconstructionProblem<M>) -> Result<Self> {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(
            problem.model.image_shape,
            problem.model.reconstruction_shape,
        )?);
        Self::initialize_with_backend(problem, backend)
    }

    pub fn initialize_with_backend<M: MeasurementRead>(
        problem: &ReconstructionProblem<M>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        problem.validate()?;
        let low_shape = problem.model.image_shape;
        let high_shape = problem.model.reconstruction_shape;
        let low_len = low_shape.0 * low_shape.1;
        let mut average_amplitude = vec![0.0; low_len];
        let mut amplitude_weight = vec![0.0; low_len];
        let mut total_amplitude = 0.0;
        let mut total_weight = 0.0;
        for frame_index in 0..problem.measurements.frame_count() {
            let frame = problem.measurements.frame(frame_index)?;
            let frame_weight = problem.measurements.frame_weight(frame_index)?;
            if frame_weight == 0.0 {
                continue;
            }
            let mask = problem.measurements.frame_mask(frame_index)?;
            let gain = problem.model.frame_gain(frame_index)?;
            for (pixel, (average, &intensity)) in
                average_amplitude.iter_mut().zip(frame.iter()).enumerate()
            {
                if mask.is_some_and(|values| values[pixel] == 0) {
                    continue;
                }
                let background = problem.model.background_value(frame_index, pixel)?;
                let amplitude = ((intensity - background) / gain).max(0.0).sqrt();
                *average += frame_weight * amplitude;
                amplitude_weight[pixel] += frame_weight;
                total_amplitude += frame_weight * amplitude;
                total_weight += frame_weight;
            }
        }
        if total_weight == 0.0 {
            return Err(Error::InvalidMeasurements(
                "no positive-weight, unmasked measurements are available".into(),
            ));
        }
        let fallback_amplitude = total_amplitude / total_weight;
        for (value, &weight) in average_amplitude.iter_mut().zip(&amplitude_weight) {
            *value = if weight > 0.0 {
                *value / weight
            } else {
                fallback_amplitude
            };
        }
        let mut object = vec![Complex64::default(); high_shape.0 * high_shape.1];
        for row in 0..high_shape.0 {
            let low_row = row * low_shape.0 / high_shape.0;
            for column in 0..high_shape.1 {
                let low_column = column * low_shape.1 / high_shape.1;
                object[row * high_shape.1 + column] =
                    Complex64::new(average_amplitude[low_row * low_shape.1 + low_column], 0.0);
            }
        }
        let mut column = vec![Complex64::default(); low_shape.0.max(high_shape.0)];
        backend.fft2(&mut object, high_shape, FftDirection::Forward, &mut column)?;
        let mut centered = vec![Complex64::default(); object.len()];
        fftshift_copy(&object, &mut centered, high_shape);
        let object_spectrum = Array2::from_vec(high_shape, centered)?;
        Ok(Self {
            object_spectrum,
            object_real_space_cache: None,
            pupil: problem.model.pupil.clone(),
            illumination_corrections: None,
            frame_gains: problem.model.frame_gains.clone(),
            background: problem.model.background.clone(),
            algorithm_auxiliary: None,
            scratch: ReconstructionScratch::new(low_shape, high_shape),
            backend,
        })
    }

    pub fn from_object<M: MeasurementRead>(
        problem: &ReconstructionProblem<M>,
        object: Array2<Complex64>,
    ) -> Result<Self> {
        if object.shape() != problem.model.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "initial object shape {:?} differs from reconstruction shape {:?}",
                object.shape(),
                problem.model.reconstruction_shape
            )));
        }
        let mut state = Self::initialize(problem)?;
        let mut raw_spectrum = object.into_vec();
        state.backend.fft2(
            &mut raw_spectrum,
            problem.model.reconstruction_shape,
            FftDirection::Forward,
            &mut state.scratch.column,
        )?;
        fftshift_copy(
            &raw_spectrum,
            state.object_spectrum.as_mut_slice(),
            problem.model.reconstruction_shape,
        );
        Ok(state)
    }

    pub fn from_checkpoint<M: MeasurementRead>(
        problem: &ReconstructionProblem<M>,
        checkpoint: &ReconstructionCheckpoint,
    ) -> Result<Self> {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(
            problem.model.image_shape,
            problem.model.reconstruction_shape,
        )?);
        Self::from_checkpoint_with_backend(problem, checkpoint, backend)
    }

    pub fn from_checkpoint_with_backend<M: MeasurementRead>(
        problem: &ReconstructionProblem<M>,
        checkpoint: &ReconstructionCheckpoint,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        checkpoint.validate_for_problem(problem)?;
        let low_shape = problem.model.image_shape;
        let high_shape = problem.model.reconstruction_shape;
        Ok(Self {
            object_spectrum: checkpoint.object_spectrum.clone(),
            object_real_space_cache: None,
            pupil: checkpoint.pupil.clone(),
            illumination_corrections: checkpoint.illumination_corrections.clone(),
            frame_gains: checkpoint.frame_gains.clone(),
            background: checkpoint.background.clone(),
            algorithm_auxiliary: checkpoint.algorithm_auxiliary.clone(),
            scratch: ReconstructionScratch::new(low_shape, high_shape),
            backend,
        })
    }
}
