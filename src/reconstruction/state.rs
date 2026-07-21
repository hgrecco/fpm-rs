use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    Result,
    array_layout::{StandardArray2, StandardView2, checked_len_2d},
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
pub(crate) struct ReconstructionScratch {
    pub(crate) patch: Vec<Complex64>,
    pub(crate) exit_spectrum: Vec<Complex64>,
    pub(crate) field: Vec<Complex64>,
    pub(crate) projected_field: Vec<Complex64>,
    pub(crate) projected_spectrum: Vec<Complex64>,
    pub(crate) difference: Vec<Complex64>,
    /// High-resolution accumulator used by mini-batch algorithms.
    pub(crate) object_gradient: Vec<Complex64>,
    pub(crate) regularization_field: Vec<Complex64>,
    pub(crate) pupil_gradient: Vec<Complex64>,
    pub(crate) calibration_reference: Vec<f64>,
    pub(crate) illumination_gradient: Vec<(f64, f64)>,
    pub(crate) illumination_curvature: Vec<(f64, f64)>,
    pub(crate) illumination_weight: Vec<f64>,
    /// Per-source low-resolution fields for an incoherently multiplexed frame.
    pub(crate) multiplex_fields: Vec<Complex64>,
    /// Matching pre-update object patches for multiplexed projection updates.
    pub(crate) multiplex_patches: Vec<Complex64>,
    pub(crate) multiplex_offsets: Vec<FourierOffset>,
    pub(crate) column: Vec<Complex64>,
}

impl ReconstructionScratch {
    fn new(low_shape: (usize, usize), high_shape: (usize, usize)) -> Result<Self> {
        let low_len = checked_len_2d(low_shape)?;
        checked_len_2d(high_shape)?;
        Ok(Self {
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
        })
    }
}

#[derive(Clone)]
pub struct ReconstructionState {
    pub(crate) object_spectrum: StandardArray2<Complex64>,
    pub(crate) object_real_space_cache: Option<StandardArray2<Complex64>>,
    pub(crate) pupil: Pupil,
    /// Per-source `(row, column)` corrections in Fourier-grid pixels.
    pub(crate) illumination_corrections: Option<Vec<(f64, f64)>>,
    pub(crate) frame_gains: Option<Vec<f64>>,
    pub(crate) background: Option<Vec<f64>>,
    pub(crate) algorithm_auxiliary: Option<AlgorithmAuxiliaryState>,
    pub(crate) scratch: ReconstructionScratch,
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
    pub fn object_spectrum(&self) -> ArrayView2<'_, Complex64> {
        self.object_spectrum.ndarray_view()
    }

    pub fn object_spectrum_mut(&mut self) -> ArrayViewMut2<'_, Complex64> {
        self.object_real_space_cache = None;
        self.object_spectrum.ndarray_view_mut()
    }

    pub fn pupil(&self) -> &Pupil {
        &self.pupil
    }

    pub fn illumination_corrections(&self) -> Option<&[(f64, f64)]> {
        self.illumination_corrections.as_deref()
    }

    pub fn frame_gains(&self) -> Option<&[f64]> {
        self.frame_gains.as_deref()
    }

    pub fn background(&self) -> Option<&[f64]> {
        self.background.as_deref()
    }

    /// Most recently accumulated per-source illumination gradient.
    pub fn illumination_gradient(&self) -> &[(f64, f64)] {
        &self.scratch.illumination_gradient
    }

    pub(crate) fn object_spectrum_standard_view(&self) -> StandardView2<'_, Complex64> {
        self.object_spectrum.view()
    }

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
        let low_len = checked_len_2d(low_shape)?;
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
        let high_len = checked_len_2d(high_shape)?;
        let mut object = vec![Complex64::default(); high_len];
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
        let object_spectrum = StandardArray2::from_shape_vec(high_shape, centered)?;
        Ok(Self {
            object_spectrum,
            object_real_space_cache: None,
            pupil: problem.model.pupil.clone(),
            illumination_corrections: None,
            frame_gains: problem.model.frame_gains.clone(),
            background: problem.model.background.clone(),
            algorithm_auxiliary: None,
            scratch: ReconstructionScratch::new(low_shape, high_shape)?,
            backend,
        })
    }

    pub fn from_object<M: MeasurementRead>(
        problem: &ReconstructionProblem<M>,
        object: Array2<Complex64>,
    ) -> Result<Self> {
        let mut object = StandardArray2::try_from(object)?;
        if object.dim() != problem.model.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "initial object shape {:?} differs from reconstruction shape {:?}",
                object.dim(),
                problem.model.reconstruction_shape
            )));
        }
        let mut state = Self::initialize(problem)?;
        state.backend.fft2(
            object.as_slice_mut(),
            problem.model.reconstruction_shape,
            FftDirection::Forward,
            &mut state.scratch.column,
        )?;
        fftshift_copy(
            object.as_slice(),
            state.object_spectrum.as_slice_mut(),
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
            object_spectrum: StandardArray2::try_from(checkpoint.object_spectrum.clone())?,
            object_real_space_cache: None,
            pupil: checkpoint.pupil.clone(),
            illumination_corrections: checkpoint.illumination_corrections.clone(),
            frame_gains: checkpoint.frame_gains.clone(),
            background: checkpoint.background.clone(),
            algorithm_auxiliary: checkpoint.algorithm_auxiliary.clone(),
            scratch: ReconstructionScratch::new(low_shape, high_shape)?,
            backend,
        })
    }
}
