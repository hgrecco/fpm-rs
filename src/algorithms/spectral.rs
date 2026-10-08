//! Object-only amplitude projection for sparse narrowband spectral exposures.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    backend::FftDirection,
    error::Error,
    measurements::MeasurementRead,
    model::fftshift_copy,
    reconstruction::{
        Batch, SpectralReconstructionProblem, SpectralReconstructionResult,
        SpectralReconstructionState, SpectralRunner,
    },
};

use super::{
    AlgorithmIterationMetrics, NoIterationMetrics, StepOutput, StepSummary,
    objective::{LossType, point_loss},
};

/// Separate solver contract for compiled narrowband spectral models.
///
/// Implementors consume only the compiled spectral model and scalar
/// [`MeasurementRead`], never physical illumination geometry. Ordinary
/// [`super::ReconstructionAlgorithm`] implementations are intentionally incompatible.
pub trait SpectralReconstructionAlgorithm {
    /// Algorithm-owned metrics reduced in exact global detector schedule order.
    type IterationMetrics: AlgorithmIterationMetrics;

    /// Validates solver parameters before state initialization.
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Validates algorithm-specific requirements on a spectral problem.
    fn validate_problem<M: MeasurementRead>(
        &self,
        _problem: &SpectralReconstructionProblem<M>,
    ) -> Result<()> {
        Ok(())
    }

    /// Creates default CPU spectral state using mask-aware amplitude initialization.
    fn initialize<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
    ) -> Result<SpectralReconstructionState> {
        SpectralReconstructionState::initialize(problem)
    }

    /// Updates a batch of physical detector exposures in the supplied order.
    fn step<M: MeasurementRead>(
        &mut self,
        problem: &SpectralReconstructionProblem<M>,
        state: &mut SpectralReconstructionState,
        batch: &Batch,
        iteration: usize,
    ) -> Result<StepOutput<Self::IterationMetrics>>;

    /// Returns the positive number of complete global schedule passes.
    fn iterations(&self) -> usize;

    /// Returns a positive batch size; implementations still stream individual frames.
    fn batch_size(&self) -> usize {
        1
    }

    /// Returns canonical JSON stepping options for a stateless checkpointable solver.
    /// `None` (the default) disables capture/resume for custom implementations.
    /// Implementors opting in must keep every persistent numerical variable in
    /// the spectral state and exclude only the total iteration target here.
    fn checkpoint_configuration(&self) -> Option<String> {
        None
    }

    /// Runs with sequential physical detector order and default initialization.
    fn run<M: MeasurementRead>(
        self,
        problem: &SpectralReconstructionProblem<M>,
    ) -> Result<SpectralReconstructionResult>
    where
        Self: Sized,
    {
        SpectralRunner::new(self).run(problem)
    }
}

/// Object-only alternating projection for separate or multiplexed spectral data.
///
/// Every coherent source mode in an exposure is evaluated before any object
/// update. One mask-aware amplitude ratio from their weighted intensity sum is
/// applied to all modes. Canonical channel crop adjoints accumulate into the
/// independent objects or one explicitly shared object. Mode updates use their
/// effective intensity weight divided by the exposure's total mode weight,
/// preserving ordinary alternating-projection behavior when there is one
/// channel. Pupils, source powers, and detector calibration remain fixed.
///
/// This generalizes narrowband state decomposition to typed channel/local-frame
/// composition. It does not implement finite bandwidth or mixed-state coherence.
///
/// # References
///
/// S. Dong, R. Shiradkar, P. Nanda, and G. Zheng,
/// [“Spectral multiplexing and coherent-state decomposition in Fourier ptychographic imaging”](https://doi.org/10.1364/BOE.5.001757),
/// *Biomedical Optics Express* **5**(6), 1757–1767 (2014).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralAlternatingProjection {
    /// Positive number of complete physical acquisition passes.
    pub iterations: usize,
    /// Finite positive relaxation applied to object corrections.
    pub object_step: f64,
    /// Positive number of exposures per step; does not change per-frame update order.
    pub batch_size: usize,
    /// Finite positive denominator floor and dark-prediction threshold.
    pub epsilon: f64,
    /// Mask-aware diagnostic loss; the update always projects measured amplitude.
    pub loss_type: LossType,
}

impl Default for SpectralAlternatingProjection {
    fn default() -> Self {
        Self {
            iterations: 50,
            object_step: 1.0,
            batch_size: 1,
            epsilon: 1e-10,
            loss_type: LossType::AmplitudeMse,
        }
    }
}

impl SpectralAlternatingProjection {
    /// Sets the positive number of complete acquisition passes.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets a finite positive object relaxation, checked before execution.
    pub fn object_step(mut self, object_step: f64) -> Self {
        self.object_step = object_step;
        self
    }

    /// Sets positive batching without changing the sequential per-frame updates.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl SpectralReconstructionAlgorithm for SpectralAlternatingProjection {
    type IterationMetrics = NoIterationMetrics;

    fn validate(&self) -> Result<()> {
        for (name, value) in [("object_step", self.object_step), ("epsilon", self.epsilon)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidParameter {
                    name,
                    reason: "must be finite and positive".into(),
                });
            }
        }
        if self.iterations == 0 {
            return Err(Error::InvalidParameter {
                name: "iterations",
                reason: "must be positive".into(),
            });
        }
        if self.batch_size == 0 {
            return Err(Error::InvalidParameter {
                name: "batch_size",
                reason: "must be positive".into(),
            });
        }
        Ok(())
    }

    fn step<M: MeasurementRead>(
        &mut self,
        problem: &SpectralReconstructionProblem<M>,
        state: &mut SpectralReconstructionState,
        batch: &Batch,
        _iteration: usize,
    ) -> Result<StepOutput<Self::IterationMetrics>> {
        let model = &problem.model;
        let shape = model.image_shape();
        let mut summary = StepSummary::default();
        for &frame in &batch.indices {
            let row = model
                .acquisition()
                .frames()
                .get(frame)
                .ok_or(Error::FrameOutOfRange {
                    index: frame,
                    frames: model.frame_count(),
                })?;
            let frame_weight = problem.measurements.frame_weight(frame)?;
            if frame_weight == 0.0 {
                summary.push_frame(frame, 0.0, 0.0);
                continue;
            }
            state.evaluate_frame(model, frame)?;
            let weight_sum: f64 = state.modes.iter().map(|mode| mode.weight).sum();
            let measured = problem.measurements.frame(frame)?;
            let mask = problem.measurements.frame_mask(frame)?;
            let mut loss = 0.0;
            let mut valid = 0;
            for pixel in 0..measured.len() {
                if mask.is_some_and(|values| values[pixel] == 0) {
                    state.projection[pixel] = Complex64::new(1.0, 0.0);
                    continue;
                }
                valid += 1;
                let prediction = state.predicted[pixel];
                loss += point_loss(
                    state.effective_gain * prediction + row.background,
                    measured[pixel],
                    self.loss_type,
                );
                let target = ((measured[pixel] - row.background) / state.effective_gain).max(0.0);
                state.projection[pixel] = if prediction > self.epsilon {
                    Complex64::new((target / prediction).sqrt(), 0.0)
                } else {
                    Complex64::new(0.0, (target / weight_sum).sqrt())
                };
            }
            if valid == 0 {
                return Err(Error::InvalidMeasurements(format!(
                    "frame {frame} has no unmasked pixels"
                )));
            }
            summary.push_frame(frame, loss / valid as f64, frame_weight);
            for mode in &state.modes {
                let kernel = &model.channels()[mode.channel].model;
                for pixel in 0..measured.len() {
                    state.field[pixel] = if mask.is_some_and(|values| values[pixel] == 0) {
                        mode.field[pixel]
                    } else if state.predicted[pixel] > self.epsilon {
                        mode.field[pixel] * state.projection[pixel].re
                    } else {
                        Complex64::new(state.projection[pixel].im, 0.0)
                    };
                }
                state.backend.fft2(
                    &mut state.field,
                    shape,
                    FftDirection::Forward,
                    &mut state.column,
                )?;
                fftshift_copy(&state.field, &mut state.centered, shape);
                for (pixel, pupil) in kernel.pupil().values().iter().enumerate() {
                    state.update[pixel] = pupil.conj() * (state.centered[pixel] - mode.exit[pixel])
                        / (pupil.norm_sqr() + self.epsilon);
                }
                let object_index = model.object_index(mode.channel)?;
                kernel.insert_patch_adjoint(
                    state.spectra[object_index].ndarray_view_mut(),
                    mode.source,
                    &state.update,
                    frame_weight * self.object_step * (mode.weight / weight_sum),
                )?;
            }
            // Keep only this exposure's fields; memory is independent of stack size.
            state.modes.clear();
        }
        Ok(summary.into())
    }

    fn iterations(&self) -> usize {
        self.iterations
    }
    fn batch_size(&self) -> usize {
        self.batch_size
    }
    fn checkpoint_configuration(&self) -> Option<String> {
        let mut configuration = self.clone();
        configuration.iterations = 1;
        serde_json::to_string(&configuration).ok()
    }
}
