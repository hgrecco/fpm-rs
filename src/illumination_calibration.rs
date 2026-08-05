//! Bounded physical calibration of planar LED-array illumination.
//!
//! This module optimizes realizable [`PlanarLedArray`](crate::experiment::PlanarLedArray)
//! parameters against the same image-plane forward model and measurement losses used by
//! reconstruction. It is deliberately separate from the generic per-source Fourier-grid
//! corrections implemented by [`crate::algorithms::GradientDescent`].

use std::collections::{BTreeMap, BTreeSet};

use ndarray::ArrayView2;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    algorithms::objective::{LossType, point_loss},
    error::Error,
    experiment::{
        AcquisitionPlan, ArrayPose, Illumination, IlluminationFrame, Optics, PlanarLedArray,
        SourceCalibration, SourceGeometry,
    },
    measurements::MeasurementRead,
    model::{ForwardModel, ImagePlaneModel, Pupil},
};

/// Bounds, differencing scale, optimization scale, and optional quadratic prior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationParameterSpec {
    /// Inclusive lower bound in the parameter's documented physical unit.
    pub lower_bound: f64,
    /// Inclusive upper bound in the parameter's documented physical unit.
    pub upper_bound: f64,
    /// Positive finite-difference displacement in the physical unit.
    pub finite_difference_step: f64,
    /// Positive scale mapping an absolute value to a dimensionless optimizer variable.
    pub scale: f64,
    /// Optional absolute center of a quadratic prior, in the physical unit.
    pub prior_center: Option<f64>,
    /// Non-negative coefficient of the dimensionless quadratic prior.
    pub regularization_strength: f64,
}

impl CalibrationParameterSpec {
    /// Creates a bounded parameter specification with scale-aware default differencing.
    pub fn new(lower_bound: f64, upper_bound: f64, scale: f64) -> Self {
        Self {
            lower_bound,
            upper_bound,
            finite_difference_step: scale * 1e-3,
            scale,
            prior_center: None,
            regularization_strength: 0.0,
        }
    }

    /// Replaces the positive physical finite-difference displacement.
    pub fn finite_difference_step(mut self, step: f64) -> Self {
        self.finite_difference_step = step;
        self
    }

    /// Adds a quadratic prior centered at `center` with non-negative `strength`.
    pub fn prior(mut self, center: f64, strength: f64) -> Self {
        self.prior_center = Some(center);
        self.regularization_strength = strength;
        self
    }

    /// Validates bounds, scale, finite-difference step, and prior values.
    pub fn validate(&self, name: &'static str) -> Result<()> {
        if !self.lower_bound.is_finite()
            || !self.upper_bound.is_finite()
            || self.lower_bound >= self.upper_bound
        {
            return Err(invalid(name, "bounds must be finite and strictly ordered"));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(invalid(
                name,
                "optimization scale must be finite and positive",
            ));
        }
        if !self.finite_difference_step.is_finite() || self.finite_difference_step <= 0.0 {
            return Err(invalid(
                name,
                "finite-difference step must be finite and positive",
            ));
        }
        if self.prior_center.is_some_and(|value| !value.is_finite())
            || !self.regularization_strength.is_finite()
            || self.regularization_strength < 0.0
        {
            return Err(invalid(
                name,
                "prior center must be finite and regularization non-negative",
            ));
        }
        Ok(())
    }
}

/// Explicit selection and numerical configuration of planar-array parameters.
///
/// `None` means inactive. Position-offset entries are keyed by stable row-major
/// physical source index and contain `(x, y, z)` specifications in metres.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarArrayCalibrationParameters {
    /// Optional `(tx, ty, tz)` specifications in metres.
    pub translation: [Option<CalibrationParameterSpec>; 3],
    /// Optional active-extrinsic `(rx, ry, rz)` specifications in radians.
    pub rotation: [Option<CalibrationParameterSpec>; 3],
    /// Optional `(pitch_x, pitch_y)` specifications in metres.
    pub pitch: [Option<CalibrationParameterSpec>; 2],
    /// Optional fractional `(reference_column, reference_row)` specifications.
    pub reference_index: [Option<CalibrationParameterSpec>; 2],
    /// Explicit source-indexed local `(offset_x, offset_y, offset_z)` specifications.
    pub position_offsets: BTreeMap<usize, [CalibrationParameterSpec; 3]>,
    /// One optional specification shared by all stable relative source powers.
    pub relative_source_power: Option<CalibrationParameterSpec>,
    /// One optional specification shared by all acquisition-frame gains.
    pub frame_gains: Option<CalibrationParameterSpec>,
}

impl Default for PlanarArrayCalibrationParameters {
    fn default() -> Self {
        Self {
            translation: std::array::from_fn(|_| None),
            rotation: std::array::from_fn(|_| None),
            pitch: std::array::from_fn(|_| None),
            reference_index: std::array::from_fn(|_| None),
            position_offsets: BTreeMap::new(),
            relative_source_power: None,
            frame_gains: None,
        }
    }
}

impl PlanarArrayCalibrationParameters {
    /// Starts a builder with every parameter inactive.
    pub fn builder() -> PlanarArrayCalibrationParametersBuilder {
        PlanarArrayCalibrationParametersBuilder::default()
    }

    /// Returns whether at least one physical or multiplicative parameter is active.
    pub fn has_active_parameters(&self) -> bool {
        self.translation.iter().any(Option::is_some)
            || self.rotation.iter().any(Option::is_some)
            || self.pitch.iter().any(Option::is_some)
            || self.reference_index.iter().any(Option::is_some)
            || !self.position_offsets.is_empty()
            || self.relative_source_power.is_some()
            || self.frame_gains.is_some()
    }

    /// Validates specifications, selected sources, supported geometry, and gauges.
    pub fn validate_for(&self, illumination: &Illumination) -> Result<()> {
        let geometry = planar_geometry(illumination)?;
        for spec in self
            .translation
            .iter()
            .chain(&self.rotation)
            .chain(&self.pitch)
            .chain(&self.reference_index)
            .flatten()
        {
            spec.validate("calibration parameter")?;
        }
        for (&source, specs) in &self.position_offsets {
            if source >= geometry.source_count() {
                return Err(invalid(
                    "position_offsets",
                    format!(
                        "source index {source} must be less than {}",
                        geometry.source_count()
                    ),
                ));
            }
            for spec in specs {
                spec.validate("position offset")?;
            }
        }
        if self
            .pitch
            .iter()
            .flatten()
            .any(|spec| spec.lower_bound <= 0.0)
        {
            return Err(invalid(
                "pitch",
                "pitch lower bounds must be strictly positive",
            ));
        }
        if let Some(spec) = &self.relative_source_power {
            spec.validate("relative_source_power")?;
            if spec.lower_bound <= 0.0 {
                return Err(invalid(
                    "relative_source_power",
                    "the lower bound must be strictly positive",
                ));
            }
        }
        if let Some(spec) = &self.frame_gains {
            spec.validate("frame_gains")?;
            if spec.lower_bound <= 0.0 {
                return Err(invalid(
                    "frame_gains",
                    "the lower bound must be strictly positive",
                ));
            }
        }
        if self.translation[0].is_some() && self.reference_index[0].is_some() {
            return Err(invalid(
                "calibration parameters",
                "tx and reference_column cannot be optimized together because they describe the same lateral gauge",
            ));
        }
        if self.translation[1].is_some() && self.reference_index[1].is_some() {
            return Err(invalid(
                "calibration parameters",
                "ty and reference_row cannot be optimized together because they describe the same lateral gauge",
            ));
        }
        if self.relative_source_power.is_some() && self.frame_gains.is_some() {
            return Err(invalid(
                "calibration parameters",
                "relative source power and frame gains cannot be optimized together because their product has an unresolved scale gauge",
            ));
        }
        if !self.position_offsets.is_empty()
            && self.translation.iter().any(Option::is_some)
            && self.position_offsets.len() < 2
        {
            return Err(invalid(
                "position_offsets",
                "at least two selected source offsets are required with global translation so their mean can be constrained to zero",
            ));
        }
        Ok(())
    }
}

/// Builder for explicit planar-array calibration selection.
#[derive(Clone, Debug, Default)]
pub struct PlanarArrayCalibrationParametersBuilder {
    parameters: PlanarArrayCalibrationParameters,
    position_offset_indices: Vec<usize>,
}

impl PlanarArrayCalibrationParametersBuilder {
    /// Activates selected `(tx, ty, tz)` components with metre-scale defaults.
    pub fn translation(mut self, active: [bool; 3]) -> Self {
        self.parameters.translation =
            std::array::from_fn(|axis| active[axis].then(|| default_translation_spec(axis)));
        self
    }

    /// Replaces optional translation specifications directly.
    pub fn translation_specs(mut self, specs: [Option<CalibrationParameterSpec>; 3]) -> Self {
        self.parameters.translation = specs;
        self
    }

    /// Activates selected `(rx, ry, rz)` components with radian defaults.
    pub fn rotation(mut self, active: [bool; 3]) -> Self {
        self.parameters.rotation =
            std::array::from_fn(|axis| active[axis].then(default_rotation_spec));
        self
    }

    /// Replaces optional rotation specifications directly.
    pub fn rotation_specs(mut self, specs: [Option<CalibrationParameterSpec>; 3]) -> Self {
        self.parameters.rotation = specs;
        self
    }

    /// Activates selected `(pitch_x, pitch_y)` components with metre defaults.
    pub fn pitch(mut self, active: [bool; 2]) -> Self {
        self.parameters.pitch = std::array::from_fn(|axis| active[axis].then(default_pitch_spec));
        self
    }

    /// Replaces optional pitch specifications directly.
    pub fn pitch_specs(mut self, specs: [Option<CalibrationParameterSpec>; 2]) -> Self {
        self.parameters.pitch = specs;
        self
    }

    /// Activates selected fractional `(reference_column, reference_row)` components.
    pub fn reference_index(mut self, active: [bool; 2]) -> Self {
        self.parameters.reference_index =
            std::array::from_fn(|axis| active[axis].then(default_reference_spec));
        self
    }

    /// Replaces optional reference-index specifications directly.
    pub fn reference_index_specs(mut self, specs: [Option<CalibrationParameterSpec>; 2]) -> Self {
        self.parameters.reference_index = specs;
        self
    }

    /// Selects stable source indices whose local XYZ offsets are all optimized.
    pub fn position_offsets(mut self, indices: impl IntoIterator<Item = usize>) -> Self {
        self.position_offset_indices = indices.into_iter().collect();
        self
    }

    /// Replaces explicit source-indexed XYZ offset specifications.
    pub fn position_offset_specs(
        mut self,
        specs: BTreeMap<usize, [CalibrationParameterSpec; 3]>,
    ) -> Self {
        self.parameters.position_offsets = specs;
        self.position_offset_indices.clear();
        self
    }

    /// Enables or disables optimization of all stable relative source powers.
    pub fn relative_source_power(mut self, active: bool) -> Self {
        self.parameters.relative_source_power = active.then(default_multiplicative_spec);
        self
    }

    /// Replaces the optional source-power specification directly.
    pub fn relative_source_power_spec(mut self, spec: Option<CalibrationParameterSpec>) -> Self {
        self.parameters.relative_source_power = spec;
        self
    }

    /// Enables or disables optimization of all acquisition-frame gains.
    pub fn frame_gains(mut self, active: bool) -> Self {
        self.parameters.frame_gains = active.then(default_multiplicative_spec);
        self
    }

    /// Replaces the optional frame-gain specification directly.
    pub fn frame_gain_spec(mut self, spec: Option<CalibrationParameterSpec>) -> Self {
        self.parameters.frame_gains = spec;
        self
    }

    /// Validates duplicate selections and returns the inspectable configuration.
    pub fn build(mut self) -> Result<PlanarArrayCalibrationParameters> {
        if !self.position_offset_indices.is_empty() {
            let unique: BTreeSet<_> = self.position_offset_indices.iter().copied().collect();
            if unique.len() != self.position_offset_indices.len() {
                return Err(invalid(
                    "position_offsets",
                    "selected source indices must be unique",
                ));
            }
            self.parameters.position_offsets = unique
                .into_iter()
                .map(|index| (index, std::array::from_fn(|_| default_offset_spec())))
                .collect();
        }
        Ok(self.parameters)
    }
}

/// Settings for the deterministic bounded finite-difference optimizer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedFiniteDifferenceOptimizer {
    /// Maximum accepted or rejected trial steps per illumination phase.
    pub max_steps: usize,
    /// Relative objective improvement below which a phase is converged.
    pub relative_tolerance: f64,
    /// Initial dimensionless step length in scaled parameter space.
    pub initial_step_size: f64,
    /// Smallest line-search step considered before reporting rejection.
    pub minimum_step_size: f64,
    /// Multiplicative line-search reduction in the open interval `(0, 1)`.
    pub step_reduction: f64,
}

impl Default for BoundedFiniteDifferenceOptimizer {
    fn default() -> Self {
        Self {
            max_steps: 2,
            relative_tolerance: 1e-6,
            initial_step_size: 0.25,
            minimum_step_size: 1e-6,
            step_reduction: 0.5,
        }
    }
}

impl BoundedFiniteDifferenceOptimizer {
    /// Validates step counts, tolerances, and line-search controls.
    pub fn validate(&self) -> Result<()> {
        if self.max_steps == 0 {
            return Err(invalid("max_steps", "must be greater than zero"));
        }
        if !self.relative_tolerance.is_finite() || self.relative_tolerance < 0.0 {
            return Err(invalid(
                "relative_tolerance",
                "must be finite and non-negative",
            ));
        }
        if !self.initial_step_size.is_finite() || self.initial_step_size <= 0.0 {
            return Err(invalid("initial_step_size", "must be finite and positive"));
        }
        if !self.minimum_step_size.is_finite()
            || self.minimum_step_size <= 0.0
            || self.minimum_step_size > self.initial_step_size
        {
            return Err(invalid(
                "minimum_step_size",
                "must be positive and no larger than initial_step_size",
            ));
        }
        if !self.step_reduction.is_finite() || !(0.0..1.0).contains(&self.step_reduction) {
            return Err(invalid(
                "step_reduction",
                "must be finite and strictly between zero and one",
            ));
        }
        Ok(())
    }
}

/// Serializable physical illumination-calibration configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IlluminationCalibration {
    /// Explicit physical and multiplicative parameter selection.
    pub parameters: PlanarArrayCalibrationParameters,
    /// Narrow deterministic bounded optimizer settings.
    pub optimizer: BoundedFiniteDifferenceOptimizer,
    /// Canonical measurement-domain data loss; amplitude MSE is the default.
    pub loss_type: LossType,
}

impl IlluminationCalibration {
    /// Creates a calibrator using bounded finite differences and amplitude MSE.
    pub fn new(parameters: PlanarArrayCalibrationParameters) -> Self {
        Self {
            parameters,
            optimizer: BoundedFiniteDifferenceOptimizer::default(),
            loss_type: LossType::AmplitudeMse,
        }
    }

    /// Replaces bounded optimizer settings.
    pub fn optimizer(mut self, optimizer: BoundedFiniteDifferenceOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Replaces the reused reconstruction measurement-domain loss.
    pub fn loss_type(mut self, loss_type: LossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Validates configuration against an initial illumination.
    pub fn validate_for(&self, illumination: &Illumination) -> Result<()> {
        self.optimizer.validate()?;
        self.parameters.validate_for(illumination)?;
        if !self.parameters.has_active_parameters() {
            return Err(invalid(
                "parameters",
                "at least one calibration parameter must be active",
            ));
        }
        Ok(())
    }

    /// Creates checkpointable absolute and normalized state for `illumination`.
    pub fn initialize(
        &self,
        illumination: &Illumination,
        optics: &Optics,
    ) -> Result<IlluminationCalibrationState> {
        self.validate_for(illumination)?;
        let mut values = PlanarArrayParameterValues::from_illumination(illumination)?;
        apply_gauge_constraints(&self.parameters, &mut values)?;
        validate_values(&self.parameters, &values)?;
        let current_illumination = values.to_illumination(illumination)?;
        validate_physical_sources(&current_illumination, optics)?;
        let active = active_parameters(&self.parameters, &values);
        let names = active.iter().map(|value| value.name()).collect();
        let normalized_variables = vec![0.0; active.len()];
        let constraints = applied_constraints(&self.parameters);
        let mut initial_conditioning = CalibrationConditioning::default();
        if self.parameters.translation[2].is_some()
            && self.parameters.pitch.iter().any(Option::is_some)
        {
            initial_conditioning.warnings.push(
                "pitch and axial translation are jointly active and may be weakly identifiable"
                    .into(),
            );
        }
        Ok(IlluminationCalibrationState {
            initial_illumination: current_illumination.clone(),
            current_illumination,
            initial_parameters: values.clone(),
            current_parameters: values,
            parameter_names: names,
            normalized_variables,
            applied_constraints: constraints,
            parameter_history: Vec::new(),
            loss_history: Vec::new(),
            convergence_reason: None,
            conditioning: initial_conditioning,
            geometry_recompilations: 0,
            multiplicative_updates: 0,
            rejected_steps: 0,
        })
    }

    /// Calibrates physical illumination for a fixed reconstructed object and pupil.
    ///
    /// `model` must have been compiled from `initial_illumination`; it is refreshed in
    /// place and can be reused immediately. `outer_steps` repeats the configured bounded
    /// phase and must be positive. Joint object/illumination reconstruction is provided by
    /// [`crate::algorithms::JointReconstruction`].
    #[allow(clippy::too_many_arguments)]
    pub fn calibrate<M: MeasurementRead>(
        &self,
        measurements: &M,
        optics: &Optics,
        initial_illumination: &Illumination,
        object_spectrum: ArrayView2<'_, Complex64>,
        pupil: &Pupil,
        model: &mut ImagePlaneModel,
        outer_steps: usize,
    ) -> Result<IlluminationCalibrationState> {
        if outer_steps == 0 {
            return Err(invalid("outer_steps", "must be greater than zero"));
        }
        let expected = ImagePlaneModel::from_experiment(
            optics,
            initial_illumination,
            model.image_shape(),
            crate::model::ReconstructionShape::Exact(model.reconstruction_shape()),
        )?;
        if expected.k_vectors() != model.k_vectors()
            || expected.source_count() != model.source_count()
            || expected.frame_count() != model.frame_count()
        {
            return Err(Error::InvalidModel(
                "calibration model was not compiled from the supplied initial illumination".into(),
            ));
        }
        let mut state = self.initialize(initial_illumination, optics)?;
        self.synchronize_model(optics, model, initial_illumination, &state)?;
        for outer_iteration in 1..=outer_steps {
            self.optimize(
                measurements,
                optics,
                object_spectrum,
                pupil,
                model,
                &mut state,
                outer_iteration,
            )?;
        }
        Ok(state)
    }

    pub(crate) fn synchronize_model(
        &self,
        optics: &Optics,
        model: &mut ImagePlaneModel,
        original_illumination: &Illumination,
        state: &IlluminationCalibrationState,
    ) -> Result<()> {
        let original = PlanarArrayParameterValues::from_illumination(original_illumination)?;
        if geometry_values_equal(&original, &state.current_parameters) {
            model.update_intensity_calibration(
                &state.current_parameters.relative_source_power,
                state.current_illumination.acquisition(),
            )
        } else {
            model.update_illumination_geometry(optics, &state.current_illumination)
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the calibration step deliberately receives each canonical model input explicitly"
    )]
    pub(crate) fn optimize<M: MeasurementRead>(
        &self,
        measurements: &M,
        optics: &Optics,
        object_spectrum: ArrayView2<'_, Complex64>,
        pupil: &Pupil,
        model: &mut ImagePlaneModel,
        state: &mut IlluminationCalibrationState,
        outer_iteration: usize,
    ) -> Result<CalibrationObjective> {
        self.validate_for(&state.initial_illumination)?;
        state.validate()?;
        let active = active_parameters(&self.parameters, &state.current_parameters);
        if active.is_empty() {
            return Err(invalid("parameters", "no active parameters remain"));
        }
        let active_names: Vec<_> = active.iter().map(ActiveParameter::name).collect();
        if state.parameter_names != active_names {
            return Err(Error::InvalidModel(
                "checkpoint physical parameter selection differs from the joint algorithm".into(),
            ));
        }
        let mut current = calibration_objective(
            measurements,
            object_spectrum,
            pupil,
            model,
            &self.parameters,
            &state.initial_parameters,
            &state.current_parameters,
            &active,
            self.loss_type,
        )?;
        let mut latest_sensitivities = vec![0.0; active.len()];
        let mut latest_curvatures = vec![0.0; active.len()];
        let mut final_reason = CalibrationConvergenceReason::MaximumSteps;
        let mut update_counters = UpdateCounters::default();

        for optimizer_step in 0..self.optimizer.max_steps {
            let mut scaled_gradient = vec![0.0; active.len()];
            for (parameter_index, parameter) in active.iter().enumerate() {
                let finite_difference = finite_difference_parameter(
                    measurements,
                    optics,
                    object_spectrum,
                    pupil,
                    model,
                    state,
                    &self.parameters,
                    &active,
                    parameter_index,
                    current.total_loss,
                    self.loss_type,
                    &mut update_counters,
                )?;
                scaled_gradient[parameter_index] =
                    finite_difference.gradient * parameter.spec.scale;
                latest_sensitivities[parameter_index] = scaled_gradient[parameter_index].abs();
                latest_curvatures[parameter_index] =
                    finite_difference.curvature * parameter.spec.scale * parameter.spec.scale;
            }
            let gradient_norm = scaled_gradient
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if !gradient_norm.is_finite() {
                return Err(Error::Numerical(
                    "physical calibration gradient is non-finite".into(),
                ));
            }
            if gradient_norm <= f64::EPSILON {
                final_reason = CalibrationConvergenceReason::NegligibleGradient;
                break;
            }

            let mut step_size = self.optimizer.initial_step_size;
            let mut accepted = None;
            while step_size >= self.optimizer.minimum_step_size {
                let mut candidate = state.current_parameters.clone();
                for (parameter, gradient) in active.iter().zip(&scaled_gradient) {
                    let delta = -step_size * parameter.spec.scale * gradient / gradient_norm;
                    let value = (parameter.value(&candidate) + delta)
                        .clamp(parameter.spec.lower_bound, parameter.spec.upper_bound);
                    parameter.set(&mut candidate, value);
                }
                apply_gauge_constraints(&self.parameters, &mut candidate)?;
                if candidate == state.current_parameters {
                    step_size *= self.optimizer.step_reduction;
                    continue;
                }
                if let Ok((candidate_model, candidate_illumination, candidate_objective)) =
                    evaluate_candidate(
                        measurements,
                        optics,
                        object_spectrum,
                        pupil,
                        model,
                        &state.initial_illumination,
                        &self.parameters,
                        &state.current_parameters,
                        &state.initial_parameters,
                        &candidate,
                        &active,
                        self.loss_type,
                        &mut update_counters,
                    )
                    && candidate_objective.total_loss < current.total_loss
                {
                    accepted = Some((
                        candidate,
                        candidate_model,
                        candidate_illumination,
                        candidate_objective,
                    ));
                    break;
                }
                step_size *= self.optimizer.step_reduction;
            }

            if let Some((candidate, candidate_model, illumination, objective)) = accepted {
                let previous_loss = current.total_loss;
                state.current_parameters = candidate;
                state.current_illumination = illumination;
                *model = candidate_model;
                current = objective;
                state
                    .parameter_history
                    .push(CalibrationParameterHistoryEntry {
                        outer_iteration,
                        optimizer_step: optimizer_step + 1,
                        accepted: true,
                        step_size,
                        normalized_values: normalized_values(
                            &active,
                            &state.initial_parameters,
                            &state.current_parameters,
                        ),
                    });
                state.loss_history.push(CalibrationLossHistoryEntry {
                    outer_iteration,
                    optimizer_step: optimizer_step + 1,
                    total_loss: current.total_loss,
                    data_loss: current.data_loss,
                    regularization_loss: current.regularization_loss,
                    accepted: true,
                });
                let improvement =
                    (previous_loss - current.total_loss) / previous_loss.abs().max(f64::EPSILON);
                if improvement <= self.optimizer.relative_tolerance {
                    final_reason = CalibrationConvergenceReason::RelativeTolerance;
                    break;
                }
            } else {
                state.rejected_steps += 1;
                state
                    .parameter_history
                    .push(CalibrationParameterHistoryEntry {
                        outer_iteration,
                        optimizer_step: optimizer_step + 1,
                        accepted: false,
                        step_size: 0.0,
                        normalized_values: normalized_values(
                            &active,
                            &state.initial_parameters,
                            &state.current_parameters,
                        ),
                    });
                state.loss_history.push(CalibrationLossHistoryEntry {
                    outer_iteration,
                    optimizer_step: optimizer_step + 1,
                    total_loss: current.total_loss,
                    data_loss: current.data_loss,
                    regularization_loss: current.regularization_loss,
                    accepted: false,
                });
                final_reason = CalibrationConvergenceReason::LineSearchFailed;
                break;
            }
        }
        state.normalized_variables = normalized_values(
            &active,
            &state.initial_parameters,
            &state.current_parameters,
        );
        state.geometry_recompilations += update_counters.geometry;
        state.multiplicative_updates += update_counters.multiplicative;
        state.conditioning = conditioning(
            &active,
            &state.current_parameters,
            &latest_sensitivities,
            &latest_curvatures,
            &self.parameters,
            state.rejected_steps,
        );
        state.convergence_reason = Some(final_reason);
        Ok(current)
    }
}

/// Absolute physical and multiplicative parameter values in canonical units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarArrayParameterValues {
    /// Array-pose translation `(tx, ty, tz)` in metres.
    pub translation_m: [f64; 3],
    /// Active-extrinsic fixed-axis `(rx, ry, rz)` rotation in radians.
    pub rotation_rad: [f64; 3],
    /// Lattice `(pitch_x, pitch_y)` in metres.
    pub pitch_m: [f64; 2],
    /// Fractional `(reference_column, reference_row)` coordinate.
    pub reference_index: [f64; 2],
    /// Full row-major source-offset array in local XYZ metres.
    pub position_offsets_m: Vec<[f64; 3]>,
    /// Stable dimensionless relative source powers in source order.
    pub relative_source_power: Vec<f64>,
    /// Dimensionless frame gains in acquisition order.
    pub frame_gains: Vec<f64>,
}

impl PlanarArrayParameterValues {
    /// Extracts explicit absolute values, expanding implicit unit powers and zero offsets.
    pub fn from_illumination(illumination: &Illumination) -> Result<Self> {
        let geometry = planar_geometry(illumination)?;
        geometry.validate()?;
        let source_count = geometry.source_count();
        let mut offsets = geometry.position_offsets_m().to_vec();
        if offsets.is_empty() {
            offsets.resize(source_count, [0.0; 3]);
        }
        let relative_source_power = illumination
            .calibration()
            .relative_power()
            .map_or_else(|| vec![1.0; source_count], <[f64]>::to_vec);
        let frame_gains = illumination
            .acquisition()
            .frames()
            .iter()
            .map(|frame| frame.gain)
            .collect();
        Ok(Self {
            translation_m: geometry.pose().translation_m,
            rotation_rad: geometry.pose().rotation_rad,
            pitch_m: [geometry.pitch_m().0, geometry.pitch_m().1],
            reference_index: [geometry.reference_index().0, geometry.reference_index().1],
            position_offsets_m: offsets,
            relative_source_power,
            frame_gains,
        })
    }

    /// Applies these values to the topology and sparse weights of `template`.
    pub fn to_illumination(&self, template: &Illumination) -> Result<Illumination> {
        let geometry = planar_geometry(template)?;
        let pose = ArrayPose::from_translation_and_extrinsic_xyz_radians(
            self.translation_m,
            self.rotation_rad,
        );
        let calibrated_geometry = PlanarLedArray::new(
            geometry.shape(),
            (self.pitch_m[0], self.pitch_m[1]),
            (self.reference_index[0], self.reference_index[1]),
            pose,
        )
        .with_position_offsets_m(self.position_offsets_m.clone());
        let frames = template
            .acquisition()
            .frames()
            .iter()
            .zip(&self.frame_gains)
            .map(|(frame, &gain)| IlluminationFrame::new(frame.contributions.clone(), gain))
            .collect();
        let acquisition = AcquisitionPlan::from_sparse(frames)?;
        Ok(Illumination::new(
            calibrated_geometry.into(),
            SourceCalibration::new(Some(self.relative_source_power.clone())),
            acquisition,
        ))
    }
}

/// Scalar decomposition of one physical-calibration objective evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationObjective {
    /// Weighted measurement-domain loss.
    pub data_loss: f64,
    /// Sum of configured dimensionless quadratic priors.
    pub regularization_loss: f64,
    /// `data_loss + regularization_loss`.
    pub total_loss: f64,
}

/// Why the most recent illumination-update phase stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationConvergenceReason {
    /// Configured optimizer-step limit was reached.
    MaximumSteps,
    /// Relative objective improvement met the configured tolerance.
    RelativeTolerance,
    /// Every scaled finite-difference sensitivity was negligible.
    NegligibleGradient,
    /// No bounded line-search candidate improved the objective.
    LineSearchFailed,
}

/// Practical finite-difference conditioning indicators, not statistical uncertainty.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationConditioning {
    /// Active parameter names in the same order as sensitivity arrays.
    pub parameter_names: Vec<String>,
    /// Absolute derivatives with respect to normalized optimizer variables.
    pub scaled_sensitivities: Vec<f64>,
    /// Approximate diagonal curvature in normalized variables.
    pub scaled_diagonal_curvature: Vec<f64>,
    /// Ratio of largest to smallest useful positive diagonal curvature, if defined.
    pub diagonal_condition_estimate: Option<f64>,
    /// Parameters within a numerical tolerance of either configured bound.
    pub parameters_at_bounds: Vec<String>,
    /// Number of trial steps rejected over the complete joint run.
    pub rejected_steps: usize,
    /// Human-readable weak-identifiability or negligible-influence warnings.
    pub warnings: Vec<String>,
}

/// One accepted or rejected bounded optimizer trial.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationParameterHistoryEntry {
    /// One-based joint outer iteration.
    pub outer_iteration: usize,
    /// One-based optimizer step inside the illumination phase.
    pub optimizer_step: usize,
    /// Whether the bounded trial decreased the complete objective.
    pub accepted: bool,
    /// Accepted dimensionless line-search length, or zero for rejection.
    pub step_size: f64,
    /// Normalized absolute parameter values in [`IlluminationCalibrationState::parameter_names`] order.
    pub normalized_values: Vec<f64>,
}

/// Objective history for one physical illumination trial.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationLossHistoryEntry {
    /// One-based joint outer iteration.
    pub outer_iteration: usize,
    /// One-based optimizer step inside the illumination phase.
    pub optimizer_step: usize,
    /// Complete data-plus-regularization objective.
    pub total_loss: f64,
    /// Measurement-domain component.
    pub data_loss: f64,
    /// Quadratic-prior component.
    pub regularization_loss: f64,
    /// Whether the associated bounded trial was accepted.
    pub accepted: bool,
}

/// Checkpointable physical calibration values, histories, constraints, and diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IlluminationCalibrationState {
    /// Gauge-normalized illumination at calibration initialization.
    pub initial_illumination: Illumination,
    /// Current normal reusable calibrated illumination.
    pub current_illumination: Illumination,
    /// Immutable initial absolute parameter values.
    pub initial_parameters: PlanarArrayParameterValues,
    /// Current absolute parameter values.
    pub current_parameters: PlanarArrayParameterValues,
    /// Stable active parameter names, including source and frame indices.
    pub parameter_names: Vec<String>,
    /// Current `(value - initial) / scale` values in parameter-name order.
    pub normalized_variables: Vec<f64>,
    /// Gauge and bound constraints applied to the parameterization.
    pub applied_constraints: Vec<String>,
    /// Accepted and rejected normalized parameter trials.
    pub parameter_history: Vec<CalibrationParameterHistoryEntry>,
    /// Data, regularization, and total loss history.
    pub loss_history: Vec<CalibrationLossHistoryEntry>,
    /// Stop reason for the most recent illumination phase.
    pub convergence_reason: Option<CalibrationConvergenceReason>,
    /// Practical finite-difference conditioning diagnostics.
    pub conditioning: CalibrationConditioning,
    /// Geometry trial evaluations that recompiled positions, vectors, and crops.
    pub geometry_recompilations: usize,
    /// Source-power or frame-gain trial evaluations that retained geometry and crops.
    pub multiplicative_updates: usize,
    /// Total bounded line-search failures or rejected optimizer steps.
    pub rejected_steps: usize,
}

impl IlluminationCalibrationState {
    /// Validates serialized physical values, illumination consistency, and history shapes.
    pub fn validate(&self) -> Result<()> {
        let initial_geometry = planar_geometry(&self.initial_illumination)?;
        let current_geometry = planar_geometry(&self.current_illumination)?;
        initial_geometry.validate()?;
        current_geometry.validate()?;
        if initial_geometry.shape() != current_geometry.shape()
            || self.initial_illumination.acquisition().frame_count()
                != self.current_illumination.acquisition().frame_count()
        {
            return Err(Error::InvalidModel(
                "physical calibration changed source or acquisition topology".into(),
            ));
        }
        let source_count = current_geometry.source_count();
        let frame_count = self.current_illumination.acquisition().frame_count();
        for (name, values) in [
            (
                "initial relative source power",
                self.initial_parameters.relative_source_power.as_slice(),
            ),
            (
                "current relative source power",
                self.current_parameters.relative_source_power.as_slice(),
            ),
        ] {
            if values.len() != source_count
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(Error::InvalidModel(format!(
                    "{name} must contain {source_count} finite non-negative values"
                )));
            }
        }
        for (name, values) in [
            (
                "initial frame gains",
                self.initial_parameters.frame_gains.as_slice(),
            ),
            (
                "current frame gains",
                self.current_parameters.frame_gains.as_slice(),
            ),
        ] {
            if values.len() != frame_count
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(Error::InvalidModel(format!(
                    "{name} must contain {frame_count} finite non-negative values"
                )));
            }
        }
        if [
            &self.initial_parameters.position_offsets_m,
            &self.current_parameters.position_offsets_m,
        ]
        .into_iter()
        .any(|values| {
            values.len() != source_count || values.iter().flatten().any(|value| !value.is_finite())
        }) {
            return Err(Error::InvalidModel(format!(
                "physical position offsets must contain {source_count} finite XYZ values"
            )));
        }
        let all_absolute_values = |values: &PlanarArrayParameterValues| {
            values
                .translation_m
                .iter()
                .chain(&values.rotation_rad)
                .chain(&values.pitch_m)
                .chain(&values.reference_index)
                .all(|value| value.is_finite())
                && values.pitch_m.iter().all(|value| *value > 0.0)
        };
        if !all_absolute_values(&self.initial_parameters)
            || !all_absolute_values(&self.current_parameters)
        {
            return Err(Error::InvalidModel(
                "physical calibration parameters must be finite with positive pitch".into(),
            ));
        }
        if PlanarArrayParameterValues::from_illumination(&self.initial_illumination)?
            != self.initial_parameters
            || PlanarArrayParameterValues::from_illumination(&self.current_illumination)?
                != self.current_parameters
        {
            return Err(Error::InvalidModel(
                "physical calibration illumination and absolute values disagree".into(),
            ));
        }
        if self.parameter_names.len() != self.normalized_variables.len()
            || self
                .normalized_variables
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidModel(
                "physical calibration normalized variables do not match parameter names".into(),
            ));
        }
        if self.parameter_history.iter().any(|entry| {
            entry.outer_iteration == 0
                || entry.optimizer_step == 0
                || !entry.step_size.is_finite()
                || entry.step_size < 0.0
                || entry.normalized_values.len() != self.parameter_names.len()
                || entry
                    .normalized_values
                    .iter()
                    .any(|value| !value.is_finite())
        }) {
            return Err(Error::InvalidModel(
                "physical calibration parameter history is malformed".into(),
            ));
        }
        if self.loss_history.iter().any(|entry| {
            entry.outer_iteration == 0
                || entry.optimizer_step == 0
                || [entry.total_loss, entry.data_loss, entry.regularization_loss]
                    .iter()
                    .any(|value| !value.is_finite())
        }) || self.loss_history.len() != self.parameter_history.len()
        {
            return Err(Error::InvalidModel(
                "physical calibration loss history is malformed".into(),
            ));
        }
        let conditioning_len = self.conditioning.parameter_names.len();
        if self.conditioning.scaled_sensitivities.len() != conditioning_len
            || self.conditioning.scaled_diagonal_curvature.len() != conditioning_len
            || self
                .conditioning
                .scaled_sensitivities
                .iter()
                .chain(&self.conditioning.scaled_diagonal_curvature)
                .any(|value| !value.is_finite())
            || self
                .conditioning
                .diagonal_condition_estimate
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            || (conditioning_len != 0 && self.conditioning.parameter_names != self.parameter_names)
        {
            return Err(Error::InvalidModel(
                "physical calibration conditioning diagnostics are malformed".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ActiveParameter {
    kind: ParameterKind,
    spec: CalibrationParameterSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParameterKind {
    Translation(usize),
    Rotation(usize),
    Pitch(usize),
    Reference(usize),
    Offset(usize, usize),
    SourcePower(usize),
    FrameGain(usize),
}

impl ActiveParameter {
    fn name(&self) -> String {
        match self.kind {
            ParameterKind::Translation(axis) => ["tx_m", "ty_m", "tz_m"][axis].into(),
            ParameterKind::Rotation(axis) => ["rx_rad", "ry_rad", "rz_rad"][axis].into(),
            ParameterKind::Pitch(axis) => ["pitch_x_m", "pitch_y_m"][axis].into(),
            ParameterKind::Reference(axis) => ["reference_column", "reference_row"][axis].into(),
            ParameterKind::Offset(source, axis) => {
                format!("position_offset_{}_m[{source}]", ["x", "y", "z"][axis])
            }
            ParameterKind::SourcePower(source) => format!("relative_source_power[{source}]"),
            ParameterKind::FrameGain(frame) => format!("frame_gain[{frame}]"),
        }
    }

    fn value(&self, values: &PlanarArrayParameterValues) -> f64 {
        match self.kind {
            ParameterKind::Translation(axis) => values.translation_m[axis],
            ParameterKind::Rotation(axis) => values.rotation_rad[axis],
            ParameterKind::Pitch(axis) => values.pitch_m[axis],
            ParameterKind::Reference(axis) => values.reference_index[axis],
            ParameterKind::Offset(source, axis) => values.position_offsets_m[source][axis],
            ParameterKind::SourcePower(source) => values.relative_source_power[source],
            ParameterKind::FrameGain(frame) => values.frame_gains[frame],
        }
    }

    fn set(&self, values: &mut PlanarArrayParameterValues, value: f64) {
        match self.kind {
            ParameterKind::Translation(axis) => values.translation_m[axis] = value,
            ParameterKind::Rotation(axis) => values.rotation_rad[axis] = value,
            ParameterKind::Pitch(axis) => values.pitch_m[axis] = value,
            ParameterKind::Reference(axis) => values.reference_index[axis] = value,
            ParameterKind::Offset(source, axis) => values.position_offsets_m[source][axis] = value,
            ParameterKind::SourcePower(source) => values.relative_source_power[source] = value,
            ParameterKind::FrameGain(frame) => values.frame_gains[frame] = value,
        }
    }

    fn update_class(&self) -> UpdateClass {
        match self.kind {
            ParameterKind::SourcePower(_) | ParameterKind::FrameGain(_) => {
                UpdateClass::Multiplicative
            }
            _ => UpdateClass::Geometry,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UpdateClass {
    Geometry,
    Multiplicative,
}

#[derive(Clone, Copy, Debug, Default)]
struct FiniteDifference {
    gradient: f64,
    curvature: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct UpdateCounters {
    geometry: usize,
    multiplicative: usize,
}

impl UpdateCounters {
    fn record(&mut self, update: UpdateClass) {
        match update {
            UpdateClass::Geometry => self.geometry += 1,
            UpdateClass::Multiplicative => self.multiplicative += 1,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finite_difference_parameter<M: MeasurementRead>(
    measurements: &M,
    optics: &Optics,
    object_spectrum: ArrayView2<'_, Complex64>,
    pupil: &Pupil,
    model: &ImagePlaneModel,
    state: &IlluminationCalibrationState,
    parameters: &PlanarArrayCalibrationParameters,
    active: &[ActiveParameter],
    parameter_index: usize,
    center_loss: f64,
    loss_type: LossType,
    update_counters: &mut UpdateCounters,
) -> Result<FiniteDifference> {
    let parameter = &active[parameter_index];
    let center_value = parameter.value(&state.current_parameters);
    let requested = parameter.spec.finite_difference_step;
    let plus_value = (center_value + requested).min(parameter.spec.upper_bound);
    let minus_value = (center_value - requested).max(parameter.spec.lower_bound);
    let mut evaluate = |value: f64| -> Option<(f64, f64)> {
        if value == center_value {
            return None;
        }
        let mut candidate = state.current_parameters.clone();
        parameter.set(&mut candidate, value);
        apply_gauge_constraints(parameters, &mut candidate).ok()?;
        let actual = parameter.value(&candidate);
        let (_, _, objective) = evaluate_candidate(
            measurements,
            optics,
            object_spectrum,
            pupil,
            model,
            &state.initial_illumination,
            parameters,
            &state.current_parameters,
            &state.initial_parameters,
            &candidate,
            active,
            loss_type,
            update_counters,
        )
        .ok()?;
        Some((actual - center_value, objective.total_loss))
    };
    let plus = evaluate(plus_value);
    let minus = evaluate(minus_value);
    match (plus, minus) {
        (Some((hp, fp)), Some((negative_hm, fm))) if hp > 0.0 && negative_hm < 0.0 => {
            let hm = -negative_hm;
            let denominator = hp * hm * (hp + hm);
            Ok(FiniteDifference {
                gradient: (hm * hm * (fp - center_loss) + hp * hp * (center_loss - fm))
                    / denominator,
                curvature: 2.0 * (hm * fp - (hp + hm) * center_loss + hp * fm) / denominator,
            })
        }
        (Some((h, value)), _) if h != 0.0 => Ok(FiniteDifference {
            gradient: (value - center_loss) / h,
            curvature: 0.0,
        }),
        (_, Some((h, value))) if h != 0.0 => Ok(FiniteDifference {
            gradient: (value - center_loss) / h,
            curvature: 0.0,
        }),
        _ => Ok(FiniteDifference::default()),
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_candidate<M: MeasurementRead>(
    measurements: &M,
    optics: &Optics,
    object_spectrum: ArrayView2<'_, Complex64>,
    pupil: &Pupil,
    base_model: &ImagePlaneModel,
    template: &Illumination,
    parameters: &PlanarArrayCalibrationParameters,
    base_values: &PlanarArrayParameterValues,
    initial_values: &PlanarArrayParameterValues,
    candidate_values: &PlanarArrayParameterValues,
    active: &[ActiveParameter],
    loss_type: LossType,
    update_counters: &mut UpdateCounters,
) -> Result<(ImagePlaneModel, Illumination, CalibrationObjective)> {
    validate_values(parameters, candidate_values)?;
    let illumination = candidate_values.to_illumination(template)?;
    validate_physical_sources(&illumination, optics)?;
    let mut model = base_model.clone();
    let update = changed_update_class(active, base_values, candidate_values);
    update_counters.record(update);
    match update {
        UpdateClass::Geometry => model.update_illumination_geometry(optics, &illumination)?,
        UpdateClass::Multiplicative => model.update_intensity_calibration(
            &candidate_values.relative_source_power,
            illumination.acquisition(),
        )?,
    }
    let objective = calibration_objective(
        measurements,
        object_spectrum,
        pupil,
        &model,
        parameters,
        initial_values,
        candidate_values,
        active,
        loss_type,
    )?;
    Ok((model, illumination, objective))
}

fn changed_update_class(
    active: &[ActiveParameter],
    previous: &PlanarArrayParameterValues,
    candidate: &PlanarArrayParameterValues,
) -> UpdateClass {
    if active.iter().any(|parameter| {
        parameter.update_class() == UpdateClass::Geometry
            && parameter.value(previous) != parameter.value(candidate)
    }) {
        UpdateClass::Geometry
    } else {
        UpdateClass::Multiplicative
    }
}

fn geometry_values_equal(
    left: &PlanarArrayParameterValues,
    right: &PlanarArrayParameterValues,
) -> bool {
    left.translation_m == right.translation_m
        && left.rotation_rad == right.rotation_rad
        && left.pitch_m == right.pitch_m
        && left.reference_index == right.reference_index
        && left.position_offsets_m == right.position_offsets_m
}

#[allow(clippy::too_many_arguments)]
fn calibration_objective<M: MeasurementRead>(
    measurements: &M,
    object_spectrum: ArrayView2<'_, Complex64>,
    pupil: &Pupil,
    model: &ImagePlaneModel,
    parameters: &PlanarArrayCalibrationParameters,
    initial_values: &PlanarArrayParameterValues,
    values: &PlanarArrayParameterValues,
    active: &[ActiveParameter],
    loss_type: LossType,
) -> Result<CalibrationObjective> {
    let forward = ForwardModel::new(model)?;
    let mut workspace = forward.workspace()?;
    let mut predicted = vec![0.0; measurements.frame_len()];
    let mut objective_sum = 0.0;
    let mut weight_sum = 0.0;
    for frame in 0..measurements.frame_count() {
        let weight = measurements.frame_weight(frame)?;
        if weight == 0.0 {
            continue;
        }
        forward.forward_intensity_into(
            object_spectrum,
            pupil,
            frame,
            &mut workspace,
            &mut predicted,
        )?;
        let measured = measurements.frame(frame)?;
        let mask = measurements.frame_mask(frame)?;
        let mut loss_sum = 0.0;
        let mut valid = 0;
        for pixel in 0..predicted.len() {
            if mask.is_none_or(|values| values[pixel] != 0) {
                loss_sum += point_loss(predicted[pixel], measured[pixel], loss_type);
                valid += 1;
            }
        }
        if valid == 0 {
            return Err(Error::InvalidMeasurements(format!(
                "frame {frame} has no unmasked pixels"
            )));
        }
        objective_sum += weight * loss_sum / valid as f64;
        weight_sum += weight;
    }
    if weight_sum == 0.0 {
        return Err(Error::InvalidMeasurements(
            "physical calibration has no positive-weight frames".into(),
        ));
    }
    let data_loss = objective_sum / weight_sum;
    let regularization_loss = regularization(parameters, initial_values, values, active);
    Ok(CalibrationObjective {
        data_loss,
        regularization_loss,
        total_loss: data_loss + regularization_loss,
    })
}

fn regularization(
    _parameters: &PlanarArrayCalibrationParameters,
    initial: &PlanarArrayParameterValues,
    values: &PlanarArrayParameterValues,
    active: &[ActiveParameter],
) -> f64 {
    active
        .iter()
        .map(|parameter| {
            let center = parameter
                .spec
                .prior_center
                .unwrap_or_else(|| parameter.value(initial));
            let residual = (parameter.value(values) - center) / parameter.spec.scale;
            0.5 * parameter.spec.regularization_strength * residual * residual
        })
        .sum()
}

fn active_parameters(
    parameters: &PlanarArrayCalibrationParameters,
    values: &PlanarArrayParameterValues,
) -> Vec<ActiveParameter> {
    let mut active = Vec::new();
    for axis in 0..3 {
        if let Some(spec) = &parameters.translation[axis] {
            active.push(ActiveParameter {
                kind: ParameterKind::Translation(axis),
                spec: spec.clone(),
            });
        }
    }
    for axis in 0..3 {
        if let Some(spec) = &parameters.rotation[axis] {
            active.push(ActiveParameter {
                kind: ParameterKind::Rotation(axis),
                spec: spec.clone(),
            });
        }
    }
    for axis in 0..2 {
        if let Some(spec) = &parameters.pitch[axis] {
            active.push(ActiveParameter {
                kind: ParameterKind::Pitch(axis),
                spec: spec.clone(),
            });
        }
    }
    for axis in 0..2 {
        if let Some(spec) = &parameters.reference_index[axis] {
            active.push(ActiveParameter {
                kind: ParameterKind::Reference(axis),
                spec: spec.clone(),
            });
        }
    }
    for (&source, specs) in &parameters.position_offsets {
        for (axis, spec) in specs.iter().enumerate() {
            active.push(ActiveParameter {
                kind: ParameterKind::Offset(source, axis),
                spec: spec.clone(),
            });
        }
    }
    if let Some(spec) = &parameters.relative_source_power {
        for source in 0..values.relative_source_power.len() {
            active.push(ActiveParameter {
                kind: ParameterKind::SourcePower(source),
                spec: spec.clone(),
            });
        }
    }
    if let Some(spec) = &parameters.frame_gains {
        for frame in 0..values.frame_gains.len() {
            active.push(ActiveParameter {
                kind: ParameterKind::FrameGain(frame),
                spec: spec.clone(),
            });
        }
    }
    active
}

fn apply_gauge_constraints(
    parameters: &PlanarArrayCalibrationParameters,
    values: &mut PlanarArrayParameterValues,
) -> Result<()> {
    if parameters.relative_source_power.is_some() {
        normalize_mean_one(&mut values.relative_source_power, "relative_source_power")?;
    }
    if parameters.frame_gains.is_some() {
        normalize_mean_one(&mut values.frame_gains, "frame_gains")?;
    }
    if !parameters.position_offsets.is_empty() && parameters.translation.iter().any(Option::is_some)
    {
        for axis in 0..3 {
            let mean = parameters
                .position_offsets
                .keys()
                .map(|&source| values.position_offsets_m[source][axis])
                .sum::<f64>()
                / parameters.position_offsets.len() as f64;
            for &source in parameters.position_offsets.keys() {
                values.position_offsets_m[source][axis] -= mean;
            }
        }
    }
    Ok(())
}

fn normalize_mean_one(values: &mut [f64], name: &'static str) -> Result<()> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return Err(invalid(name, "positive finite mean is required"));
    }
    for value in values {
        *value /= mean;
    }
    Ok(())
}

fn validate_values(
    parameters: &PlanarArrayCalibrationParameters,
    values: &PlanarArrayParameterValues,
) -> Result<()> {
    for parameter in active_parameters(parameters, values) {
        let value = parameter.value(values);
        if !value.is_finite()
            || value < parameter.spec.lower_bound - f64::EPSILON
            || value > parameter.spec.upper_bound + f64::EPSILON
        {
            return Err(invalid(
                "calibration parameter",
                format!(
                    "{}={value} lies outside [{}, {}]",
                    parameter.name(),
                    parameter.spec.lower_bound,
                    parameter.spec.upper_bound
                ),
            ));
        }
    }
    Ok(())
}

fn validate_physical_sources(illumination: &Illumination, optics: &Optics) -> Result<()> {
    let resolved = illumination.resolve(optics)?;
    if resolved
        .positions_m()
        .is_some_and(|positions| positions.iter().any(|position| position[2].abs() <= 1e-9))
    {
        return Err(invalid(
            "illumination geometry",
            "calibration cannot place a source on the sample plane",
        ));
    }
    Ok(())
}

fn normalized_values(
    active: &[ActiveParameter],
    initial: &PlanarArrayParameterValues,
    current: &PlanarArrayParameterValues,
) -> Vec<f64> {
    active
        .iter()
        .map(|parameter| {
            (parameter.value(current) - parameter.value(initial)) / parameter.spec.scale
        })
        .collect()
}

fn conditioning(
    active: &[ActiveParameter],
    values: &PlanarArrayParameterValues,
    sensitivities: &[f64],
    curvatures: &[f64],
    parameters: &PlanarArrayCalibrationParameters,
    rejected_steps: usize,
) -> CalibrationConditioning {
    let useful: Vec<_> = curvatures
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 1e-14)
        .collect();
    let diagonal_condition_estimate = (!useful.is_empty()).then(|| {
        useful.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            / useful.iter().copied().fold(f64::INFINITY, f64::min)
    });
    let parameters_at_bounds = active
        .iter()
        .filter(|parameter| {
            let value = parameter.value(values);
            let tolerance = 1e-9 * parameter.spec.scale.max(1.0);
            (value - parameter.spec.lower_bound).abs() <= tolerance
                || (value - parameter.spec.upper_bound).abs() <= tolerance
        })
        .map(ActiveParameter::name)
        .collect();
    let mut warnings = Vec::new();
    if parameters.translation[2].is_some() && parameters.pitch.iter().any(Option::is_some) {
        warnings.push(
            "pitch and axial translation are jointly active and may be weakly identifiable".into(),
        );
    }
    if diagonal_condition_estimate.is_some_and(|value| value > 1e8) {
        warnings.push("scaled diagonal curvature condition estimate exceeds 1e8".into());
    }
    for (parameter, &sensitivity) in active.iter().zip(sensitivities) {
        if sensitivity <= 1e-12 {
            warnings.push(format!(
                "{} has negligible finite-difference influence",
                parameter.name()
            ));
        }
    }
    CalibrationConditioning {
        parameter_names: active.iter().map(ActiveParameter::name).collect(),
        scaled_sensitivities: sensitivities.to_vec(),
        scaled_diagonal_curvature: curvatures.to_vec(),
        diagonal_condition_estimate,
        parameters_at_bounds,
        rejected_steps,
        warnings,
    }
}

fn applied_constraints(parameters: &PlanarArrayCalibrationParameters) -> Vec<String> {
    let mut constraints = vec!["inclusive parameter bounds".into()];
    if parameters.relative_source_power.is_some() {
        constraints.push("mean(relative_source_power) = 1".into());
    }
    if parameters.frame_gains.is_some() {
        constraints.push("mean(frame_gain) = 1".into());
    }
    if !parameters.position_offsets.is_empty() && parameters.translation.iter().any(Option::is_some)
    {
        constraints.push("mean(selected position offset) = 0 on every axis".into());
    }
    constraints
}

fn planar_geometry(illumination: &Illumination) -> Result<&PlanarLedArray> {
    match illumination.geometry() {
        SourceGeometry::PlanarArray(value) => Ok(value),
        _ => Err(Error::Unsupported(
            "physical illumination calibration supports only PlanarLedArray geometry; use generic k-vector correction for other geometries".into(),
        )),
    }
}

fn default_translation_spec(axis: usize) -> CalibrationParameterSpec {
    if axis == 2 {
        CalibrationParameterSpec::new(-1.0, -1e-5, 1e-3).finite_difference_step(1e-5)
    } else {
        CalibrationParameterSpec::new(-0.1, 0.1, 1e-3).finite_difference_step(1e-5)
    }
}

fn default_rotation_spec() -> CalibrationParameterSpec {
    CalibrationParameterSpec::new(
        -std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_4,
        0.01,
    )
    .finite_difference_step(1e-4)
}

fn default_pitch_spec() -> CalibrationParameterSpec {
    CalibrationParameterSpec::new(1e-6, 0.1, 1e-4).finite_difference_step(1e-6)
}

fn default_reference_spec() -> CalibrationParameterSpec {
    CalibrationParameterSpec::new(-10_000.0, 10_000.0, 0.1).finite_difference_step(1e-3)
}

fn default_offset_spec() -> CalibrationParameterSpec {
    CalibrationParameterSpec::new(-0.01, 0.01, 1e-4)
        .finite_difference_step(1e-6)
        .prior(0.0, 1e-6)
}

fn default_multiplicative_spec() -> CalibrationParameterSpec {
    CalibrationParameterSpec::new(1e-3, 1e3, 0.1)
        .finite_difference_step(1e-3)
        .prior(1.0, 1e-6)
}

fn invalid(name: &'static str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name,
        reason: reason.into(),
    }
}
