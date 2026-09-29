use ndarray::ArrayView2;
use num_complex::Complex64;

use crate::{
    Result,
    algorithms::{AlgorithmIterationMetrics, StepOutput, StepSummary},
    array_layout::{StandardView2, checked_len_2d},
    backend::FftDirection,
    error::Error,
    measurements::MeasurementRead,
    model::{FourierOffset, ImagePlaneModel, fftshift_copy, ifftshift_copy},
    reconstruction::{Batch, ReconstructionProblem, ReconstructionState},
};

use super::ReconstructionAlgorithm;

/// Numerical work reported by one global Gauss–Newton object update.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalGaussNewtonIterationMetrics {
    conjugate_gradient_iterations: usize,
    linear_residual_ratio: f64,
    line_search_evaluations: usize,
    accepted_step_scale: f64,
    gradient_norm: f64,
}

impl GlobalGaussNewtonIterationMetrics {
    /// Returns the number of matrix-free conjugate-gradient iterations used.
    pub fn conjugate_gradient_iterations(&self) -> usize {
        self.conjugate_gradient_iterations
    }

    /// Returns the final linear residual norm divided by its initial norm.
    pub fn linear_residual_ratio(&self) -> f64 {
        self.linear_residual_ratio
    }

    /// Returns the number of full-data trial-objective evaluations.
    pub fn line_search_evaluations(&self) -> usize {
        self.line_search_evaluations
    }

    /// Returns the accepted multiplier applied to the Gauss–Newton direction.
    pub fn accepted_step_scale(&self) -> f64 {
        self.accepted_step_scale
    }

    /// Returns the Euclidean norm of `J^T r` at the pre-update object.
    pub fn gradient_norm(&self) -> f64 {
        self.gradient_norm
    }
}

impl AlgorithmIterationMetrics for GlobalGaussNewtonIterationMetrics {
    fn merge(&mut self, other: Self) {
        self.conjugate_gradient_iterations += other.conjugate_gradient_iterations;
        self.linear_residual_ratio = other.linear_residual_ratio;
        self.line_search_evaluations += other.line_search_evaluations;
        self.accepted_step_scale = other.accepted_step_scale;
        self.gradient_norm = other.gradient_norm;
    }

    fn append_records(
        &self,
        iteration: usize,
        output: &mut Vec<crate::reconstruction::AlgorithmMetricRecord>,
    ) {
        for (metric, value) in [
            (
                "conjugate_gradient_iterations",
                self.conjugate_gradient_iterations as f64,
            ),
            ("linear_residual_ratio", self.linear_residual_ratio),
            (
                "line_search_evaluations",
                self.line_search_evaluations as f64,
            ),
            ("accepted_step_scale", self.accepted_step_scale),
            ("gradient_norm", self.gradient_norm),
        ] {
            output.push(crate::reconstruction::AlgorithmMetricRecord {
                iteration,
                namespace: "global_gauss_newton".into(),
                metric: metric.into(),
                value,
            });
        }
    }
}

/// Matrix-free damped Gauss–Newton reconstruction of a fixed-pupil FPM object.
///
/// # Method
///
/// The solver minimizes the full-stack, frame-weighted mean amplitude-MSE
/// objective in intrinsic intensity units. Known detector gain and background
/// are removed before forming residuals, masks omit pixels, and zero-weight
/// frames contribute neither residuals nor derivatives. Incoherently
/// multiplexed frames use one detector residual whose analytic derivative
/// contains every contributing coherent source mode.
///
/// Each outer iteration linearizes the amplitude residual at the current
/// centered object spectrum and solves
/// `(J^T J + damping * diag(C)) d = -J^T r` with preconditioned conjugate
/// gradients. `J` and `J^T` are applied analytically under the real inner
/// product on complex arrays. `C` is a floored pupil-power Fourier-coverage
/// approximation used for both damping and Jacobi preconditioning. Armijo
/// backtracking accepts a step only when the same full-data objective
/// decreases sufficiently.
///
/// Frames are processed for every gradient, normal-operator, and line-search
/// evaluation. The implementation stores a fixed number of object-sized
/// vectors and only the coherent fields of the current multiplexed frame; it
/// never forms a Jacobian or Hessian and does not retain curvature across outer
/// iterations. Consequently it supports lazy measurements and exact
/// iteration-boundary checkpoint resume, but each iteration is substantially
/// more expensive than a sequential projection or one gradient pass.
///
/// The pupil, frame response, generic source corrections, and physical model
/// are treated as fixed during one step. The absence of persistent curvature
/// permits use as the object phase of
/// [`crate::algorithms::JointReconstruction`], whose subsequent physical phase
/// recompiles the model before the next fresh linearization.
///
/// # Failure behavior
///
/// Configuration validation rejects non-positive limits or damping and
/// tolerances outside their documented open intervals. A step also rejects an
/// incomplete global batch or unrelated algorithm auxiliary state. Non-finite
/// products, non-positive conjugate-gradient curvature, a non-descent
/// direction, or an exhausted line search return an error without committing a
/// trial object.
///
/// # Example
///
/// ```no_run
/// use fpm_rs::{
///     Result,
///     algorithms::{GlobalGaussNewton, ReconstructionAlgorithm},
///     measurements::MeasurementRead,
///     reconstruction::ReconstructionProblem,
/// };
///
/// # fn reconstruct<M: MeasurementRead>(problem: &ReconstructionProblem<M>) -> Result<()> {
/// let result = GlobalGaussNewton::default()
///     .iterations(10)
///     .damping(1e-3)
///     .maximum_cg_iterations(8)
///     .run(problem)?;
/// assert_eq!(result.trace.iterations.len(), 10);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// [L.-H. Yeh, J. Dong, J. Zhong, L. Tian, M. Chen, G. Tang,
/// M. Soltanolkotabi, and L. Waller, “Experimental robustness of Fourier
/// ptychography phase retrieval algorithms,” *Optics Express* **23**(26),
/// 33214–33240 (2015)](https://doi.org/10.1364/OE.23.033214). That work forms
/// exact CR-calculus Hessians for several objectives; this implementation keeps
/// only the positive-semidefinite Gauss–Newton part of the amplitude residual
/// and applies it without materializing a matrix.
///
/// Matrix-free second-order ptychographic optimization is also demonstrated by
/// [S. Kandel, S. Maddali, Y. S. G. Nashed, S. O. Hruszkewycz, C. Jacobsen,
/// and M. Allain, “Efficient ptychographic phase retrieval via a matrix-free
/// Levenberg–Marquardt algorithm,” *Optics Express* **29**(15), 23019–23055
/// (2021)](https://doi.org/10.1364/OE.422768). That work treats
/// diffraction-plane ptychography with automatic differentiation, whereas
/// this solver uses analytic products for the crate's image-plane FPM model.
#[derive(Clone, Debug)]
pub struct GlobalGaussNewton {
    /// Number of complete global object updates.
    pub iterations: usize,
    /// Positive coverage-scaled diagonal damping coefficient.
    pub damping: f64,
    /// Maximum matrix-free conjugate-gradient iterations per outer update.
    pub maximum_cg_iterations: usize,
    /// Relative linear-residual tolerance for conjugate-gradient termination.
    pub cg_relative_tolerance: f64,
    /// Maximum full-data trial-objective evaluations per outer update.
    pub maximum_line_search_steps: usize,
    /// Multiplicative trial-step reduction in `(0, 1)`.
    pub line_search_reduction: f64,
    /// Armijo sufficient-decrease coefficient in `(0, 1)`.
    pub line_search_sufficient_decrease: f64,
    /// Positive floor for dark-field derivatives and Fourier coverage.
    pub epsilon: f64,
}

impl Default for GlobalGaussNewton {
    fn default() -> Self {
        Self {
            iterations: 20,
            damping: 1e-3,
            maximum_cg_iterations: 12,
            cg_relative_tolerance: 1e-3,
            maximum_line_search_steps: 8,
            line_search_reduction: 0.5,
            line_search_sufficient_decrease: 1e-4,
            epsilon: 1e-10,
        }
    }
}

impl GlobalGaussNewton {
    /// Sets the positive number of global object updates.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the finite positive coverage-scaled damping coefficient.
    pub fn damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Sets the positive maximum conjugate-gradient iteration count.
    pub fn maximum_cg_iterations(mut self, iterations: usize) -> Self {
        self.maximum_cg_iterations = iterations;
        self
    }

    /// Sets the relative conjugate-gradient residual tolerance in `(0, 1)`.
    pub fn cg_relative_tolerance(mut self, tolerance: f64) -> Self {
        self.cg_relative_tolerance = tolerance;
        self
    }

    /// Sets the positive maximum number of trial-objective evaluations.
    pub fn maximum_line_search_steps(mut self, steps: usize) -> Self {
        self.maximum_line_search_steps = steps;
        self
    }

    /// Sets the multiplicative backtracking reduction in `(0, 1)`.
    pub fn line_search_reduction(mut self, reduction: f64) -> Self {
        self.line_search_reduction = reduction;
        self
    }

    /// Sets the Armijo sufficient-decrease coefficient in `(0, 1)`.
    pub fn line_search_sufficient_decrease(mut self, coefficient: f64) -> Self {
        self.line_search_sufficient_decrease = coefficient;
        self
    }

    /// Sets the finite positive numerical floor.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl ReconstructionAlgorithm for GlobalGaussNewton {
    type IterationMetrics = GlobalGaussNewtonIterationMetrics;

    fn validate(&self) -> Result<()> {
        if self.iterations == 0 {
            return Err(Error::InvalidParameter {
                name: "iterations",
                reason: "must be greater than zero".into(),
            });
        }
        if !self.damping.is_finite() || self.damping <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "damping",
                reason: "must be finite and positive".into(),
            });
        }
        if self.maximum_cg_iterations == 0 {
            return Err(Error::InvalidParameter {
                name: "maximum_cg_iterations",
                reason: "must be greater than zero".into(),
            });
        }
        if !self.cg_relative_tolerance.is_finite()
            || self.cg_relative_tolerance <= 0.0
            || self.cg_relative_tolerance >= 1.0
        {
            return Err(Error::InvalidParameter {
                name: "cg_relative_tolerance",
                reason: "must be finite and in (0, 1)".into(),
            });
        }
        if self.maximum_line_search_steps == 0 {
            return Err(Error::InvalidParameter {
                name: "maximum_line_search_steps",
                reason: "must be greater than zero".into(),
            });
        }
        if !self.line_search_reduction.is_finite()
            || self.line_search_reduction <= 0.0
            || self.line_search_reduction >= 1.0
        {
            return Err(Error::InvalidParameter {
                name: "line_search_reduction",
                reason: "must be finite and in (0, 1)".into(),
            });
        }
        if !self.line_search_sufficient_decrease.is_finite()
            || self.line_search_sufficient_decrease <= 0.0
            || self.line_search_sufficient_decrease >= 1.0
        {
            return Err(Error::InvalidParameter {
                name: "line_search_sufficient_decrease",
                reason: "must be finite and in (0, 1)".into(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                reason: "must be finite and positive".into(),
            });
        }
        Ok(())
    }

    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        _iteration: usize,
    ) -> Result<StepOutput<Self::IterationMetrics>> {
        validate_global_batch(problem.model.frame_count(), batch)?;
        if state.algorithm_auxiliary.is_some() {
            return Err(Error::InvalidModel(
                "global Gauss–Newton cannot interpret state owned by another algorithm".into(),
            ));
        }

        let object = state.object_spectrum.as_slice().to_vec();
        let mut workspace = GaussNewtonWorkspace::new(&problem.model)?;
        let coverage = coverage_diagonal(problem, state, self.epsilon, &mut workspace)?;
        let (objective, current_summary, gradient) =
            objective_and_gradient(problem, state, &object, self.epsilon, &mut workspace)?;
        let gradient_norm = real_norm(&gradient);
        if !gradient_norm.is_finite() {
            return Err(Error::Numerical(
                "global Gauss–Newton gradient norm is non-finite".into(),
            ));
        }
        if gradient_norm <= self.epsilon {
            return Ok(StepOutput {
                summary: current_summary,
                metrics: GlobalGaussNewtonIterationMetrics {
                    conjugate_gradient_iterations: 0,
                    linear_residual_ratio: 0.0,
                    line_search_evaluations: 0,
                    accepted_step_scale: 0.0,
                    gradient_norm,
                },
            });
        }

        let (direction, cg_iterations, linear_residual_ratio) = solve_direction(
            problem,
            state,
            &object,
            &gradient,
            &coverage,
            self,
            &mut workspace,
        )?;
        let directional_derivative = real_dot(&gradient, &direction);
        if !directional_derivative.is_finite() || directional_derivative >= 0.0 {
            return Err(Error::Numerical(
                "global Gauss–Newton produced a non-descent direction".into(),
            ));
        }

        let mut step_scale = 1.0;
        let mut accepted = None;
        let mut evaluations = 0;
        let mut candidate = vec![Complex64::default(); object.len()];
        for _ in 0..self.maximum_line_search_steps {
            evaluations += 1;
            let mut finite = true;
            for index in 0..object.len() {
                candidate[index] = object[index] + step_scale * direction[index];
                finite &= candidate[index].re.is_finite() && candidate[index].im.is_finite();
            }
            if finite {
                match objective_only(problem, state, &candidate, &mut workspace) {
                    Ok((trial_objective, trial_summary)) => {
                        let armijo_bound = objective
                            + 2.0
                                * self.line_search_sufficient_decrease
                                * step_scale
                                * directional_derivative;
                        if trial_objective <= armijo_bound {
                            accepted = Some(trial_summary);
                            break;
                        }
                    }
                    Err(Error::Numerical(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            step_scale *= self.line_search_reduction;
        }
        let summary = accepted.ok_or_else(|| {
            Error::Numerical(format!(
                "global Gauss–Newton line search failed after {} evaluations",
                self.maximum_line_search_steps
            ))
        })?;
        state.object_spectrum.as_slice_mut().copy_from_slice(&candidate);
        state.object_real_space_cache = None;

        Ok(StepOutput {
            summary,
            metrics: GlobalGaussNewtonIterationMetrics {
                conjugate_gradient_iterations: cg_iterations,
                linear_residual_ratio,
                line_search_evaluations: evaluations,
                accepted_step_scale: step_scale,
                gradient_norm,
            },
        })
    }

    fn iterations(&self) -> usize {
        self.iterations
    }

    fn batch_size(&self) -> usize {
        usize::MAX
    }
}

struct GaussNewtonWorkspace {
    patch: Vec<Complex64>,
    centered: Vec<Complex64>,
    field: Vec<Complex64>,
    detector: Vec<Complex64>,
    mode_fields: Vec<Complex64>,
    predicted: Vec<f64>,
    denominator: Vec<f64>,
    residual: Vec<f64>,
    directional: Vec<f64>,
    column: Vec<Complex64>,
}

impl GaussNewtonWorkspace {
    fn new(model: &ImagePlaneModel) -> Result<Self> {
        let low_len = checked_len_2d(model.image_shape)?;
        Ok(Self {
            patch: vec![Complex64::default(); low_len],
            centered: vec![Complex64::default(); low_len],
            field: vec![Complex64::default(); low_len],
            detector: vec![Complex64::default(); low_len],
            mode_fields: Vec::new(),
            predicted: vec![0.0; low_len],
            denominator: vec![0.0; low_len],
            residual: vec![0.0; low_len],
            directional: vec![0.0; low_len],
            column: vec![
                Complex64::default();
                model.image_shape.0.max(model.reconstruction_shape.0)
            ],
        })
    }

    fn resize_modes(&mut self, modes: usize, low_len: usize) -> Result<()> {
        let length = modes
            .checked_mul(low_len)
            .ok_or_else(|| Error::InvalidShape("multiplexed field storage overflows".into()))?;
        self.mode_fields.resize(length, Complex64::default());
        Ok(())
    }
}

fn validate_global_batch(frame_count: usize, batch: &Batch) -> Result<()> {
    if batch.indices.len() != frame_count {
        return Err(Error::InvalidParameter {
            name: "batch",
            reason: format!(
                "global Gauss–Newton requires all {frame_count} frames in one step"
            ),
        });
    }
    let mut seen = vec![false; frame_count];
    for &frame in &batch.indices {
        if frame >= frame_count || seen[frame] {
            return Err(Error::InvalidParameter {
                name: "batch",
                reason: "must contain every frame exactly once".into(),
            });
        }
        seen[frame] = true;
    }
    Ok(())
}

fn positive_weight_sum<M: MeasurementRead>(problem: &ReconstructionProblem<M>) -> Result<f64> {
    let mut total = 0.0;
    for frame in 0..problem.model.frame_count() {
        total += problem.measurements.frame_weight(frame)?;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::InvalidMeasurements(
            "global Gauss–Newton requires positive finite frame weight".into(),
        ));
    }
    Ok(total)
}

fn valid_pixel_count<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    frame: usize,
) -> Result<usize> {
    let mask = problem.measurements.frame_mask(frame)?;
    let count = mask.map_or(problem.measurements.frame_len(), |values| {
        values.iter().filter(|&&value| value != 0).count()
    });
    if count == 0 {
        return Err(Error::InvalidMeasurements(format!(
            "positive-weight frame {frame} has no unmasked pixels"
        )));
    }
    Ok(count)
}

fn coverage_diagonal<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &ReconstructionState,
    epsilon: f64,
    workspace: &mut GaussNewtonWorkspace,
) -> Result<Vec<f64>> {
    let model = &problem.model;
    let total_weight = positive_weight_sum(problem)?;
    let mut coverage = vec![Complex64::default(); state.object_spectrum.len()];
    for pixel in 0..workspace.centered.len() {
        workspace.centered[pixel] =
            Complex64::new(state.pupil.values.as_slice()[pixel].norm_sqr(), 0.0);
    }
    for frame in 0..model.frame_count() {
        let frame_weight = problem.measurements.frame_weight(frame)?;
        if frame_weight == 0.0 {
            continue;
        }
        let valid_pixels = valid_pixel_count(problem, frame)? as f64;
        let single_source = [(frame, 1.0)];
        let sources = model
            .multiplexing_matrix
            .as_ref()
            .map_or(single_source.as_slice(), |matrix| matrix[frame].as_slice());
        for &(source, source_weight) in sources {
            let offset = state.effective_source_offset(model, source)?;
            model.insert_patch_adjoint_slice_at_offset(
                &mut coverage,
                source,
                &workspace.centered,
                frame_weight * source_weight / (valid_pixels * total_weight),
                offset,
            )?;
        }
    }
    let maximum = coverage
        .iter()
        .map(|value| value.re)
        .fold(0.0_f64, f64::max);
    if !maximum.is_finite() || maximum <= 0.0 {
        return Err(Error::InvalidModel(
            "global Gauss–Newton Fourier coverage is empty or non-finite".into(),
        ));
    }
    Ok(coverage
        .into_iter()
        .map(|value| (value.re / maximum).max(epsilon))
        .collect())
}

fn objective_and_gradient<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &ReconstructionState,
    object: &[Complex64],
    epsilon: f64,
    workspace: &mut GaussNewtonWorkspace,
) -> Result<(f64, StepSummary, Vec<Complex64>)> {
    let model = &problem.model;
    let low_len = checked_len_2d(model.image_shape)?;
    let total_weight = positive_weight_sum(problem)?;
    let mut gradient = vec![Complex64::default(); object.len()];
    let mut summary = StepSummary::default();
    for frame in 0..model.frame_count() {
        let frame_weight = problem.measurements.frame_weight(frame)?;
        if frame_weight == 0.0 {
            summary.push_frame(frame, 0.0, 0.0);
            continue;
        }
        let valid_pixels = valid_pixel_count(problem, frame)?;
        let single_source = [(frame, 1.0)];
        let sources = model
            .multiplexing_matrix
            .as_ref()
            .map_or(single_source.as_slice(), |matrix| matrix[frame].as_slice());
        predict_frame(problem, state, object, sources, workspace)?;

        let measured = problem.measurements.frame(frame)?;
        let mask = problem.measurements.frame_mask(frame)?;
        let gain = frame_gain(state, frame)?;
        let residual_scale =
            (frame_weight / (valid_pixels as f64 * total_weight)).sqrt();
        let mut frame_loss = 0.0;
        for pixel in 0..low_len {
            if mask.is_some_and(|values| values[pixel] == 0) {
                workspace.residual[pixel] = 0.0;
                workspace.denominator[pixel] = epsilon.sqrt();
                continue;
            }
            let target = ((measured[pixel] - background_value(state, frame, pixel, low_len))
                / gain)
                .max(0.0);
            let predicted_amplitude = workspace.predicted[pixel].max(0.0).sqrt();
            let residual = predicted_amplitude - target.sqrt();
            frame_loss += residual * residual;
            workspace.residual[pixel] = residual_scale * residual;
            workspace.denominator[pixel] = predicted_amplitude.max(epsilon.sqrt());
        }
        summary.push_frame(frame, frame_loss / valid_pixels as f64, frame_weight);

        for (mode, &(source, source_weight)) in sources.iter().enumerate() {
            let start = mode * low_len;
            let mode_field = &workspace.mode_fields[start..start + low_len];
            for pixel in 0..low_len {
                workspace.detector[pixel] = if mask.is_some_and(|values| values[pixel] == 0) {
                    Complex64::default()
                } else {
                    mode_field[pixel]
                        * (source_weight * residual_scale * workspace.residual[pixel]
                            / workspace.denominator[pixel])
                };
            }
            let offset = state.effective_source_offset(model, source)?;
            adjoint_source(
                model,
                state,
                source,
                offset,
                &workspace.detector,
                &mut gradient,
                &mut workspace.field,
                &mut workspace.centered,
                &mut workspace.column,
            )?;
        }
    }
    let objective = summary.mean_objective().ok_or_else(|| {
        Error::InvalidMeasurements("global objective has no positive frame weight".into())
    })?;
    if !objective.is_finite() || gradient.iter().any(|v| !complex_is_finite(*v)) {
        return Err(Error::Numerical(
            "global Gauss–Newton objective or gradient is non-finite".into(),
        ));
    }
    Ok((objective, summary, gradient))
}

fn objective_only<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &ReconstructionState,
    object: &[Complex64],
    workspace: &mut GaussNewtonWorkspace,
) -> Result<(f64, StepSummary)> {
    let model = &problem.model;
    let low_len = checked_len_2d(model.image_shape)?;
    let mut summary = StepSummary::default();
    for frame in 0..model.frame_count() {
        let frame_weight = problem.measurements.frame_weight(frame)?;
        if frame_weight == 0.0 {
            summary.push_frame(frame, 0.0, 0.0);
            continue;
        }
        let valid_pixels = valid_pixel_count(problem, frame)?;
        let single_source = [(frame, 1.0)];
        let sources = model
            .multiplexing_matrix
            .as_ref()
            .map_or(single_source.as_slice(), |matrix| matrix[frame].as_slice());
        predict_frame(problem, state, object, sources, workspace)?;
        let measured = problem.measurements.frame(frame)?;
        let mask = problem.measurements.frame_mask(frame)?;
        let gain = frame_gain(state, frame)?;
        let mut frame_loss = 0.0;
        for pixel in 0..low_len {
            if mask.is_some_and(|values| values[pixel] == 0) {
                continue;
            }
            let target = ((measured[pixel] - background_value(state, frame, pixel, low_len))
                / gain)
                .max(0.0);
            let residual = workspace.predicted[pixel].max(0.0).sqrt() - target.sqrt();
            frame_loss += residual * residual;
        }
        summary.push_frame(frame, frame_loss / valid_pixels as f64, frame_weight);
    }
    let objective = summary.mean_objective().ok_or_else(|| {
        Error::InvalidMeasurements("global objective has no positive frame weight".into())
    })?;
    if !objective.is_finite() {
        return Err(Error::Numerical(
            "global Gauss–Newton trial objective is non-finite".into(),
        ));
    }
    Ok((objective, summary))
}

#[allow(clippy::too_many_arguments)]
fn solve_direction<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &ReconstructionState,
    object: &[Complex64],
    gradient: &[Complex64],
    coverage: &[f64],
    algorithm: &GlobalGaussNewton,
    workspace: &mut GaussNewtonWorkspace,
) -> Result<(Vec<Complex64>, usize, f64)> {
    let length = object.len();
    let mut solution = vec![Complex64::default(); length];
    let mut residual: Vec<_> = gradient.iter().map(|value| -*value).collect();
    let initial_norm = real_norm(&residual);
    if initial_norm == 0.0 {
        return Ok((solution, 0, 0.0));
    }
    let mut preconditioned = vec![Complex64::default(); length];
    apply_preconditioner(
        &residual,
        coverage,
        algorithm.damping,
        &mut preconditioned,
    );
    let mut direction = preconditioned.clone();
    let mut residual_product = real_dot(&residual, &preconditioned);
    if !residual_product.is_finite() || residual_product <= 0.0 {
        return Err(Error::Numerical(
            "global Gauss–Newton preconditioned residual is not positive".into(),
        ));
    }
    let mut ratio = 1.0;
    let mut completed = 0;
    for iteration in 0..algorithm.maximum_cg_iterations {
        let operator_direction = apply_normal_operator(
            problem,
            state,
            object,
            &direction,
            coverage,
            algorithm.damping,
            algorithm.epsilon,
            workspace,
        )?;
        let curvature = real_dot(&direction, &operator_direction);
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err(Error::Numerical(
                "global Gauss–Newton conjugate-gradient curvature is not positive".into(),
            ));
        }
        let step = residual_product / curvature;
        if !step.is_finite() {
            return Err(Error::Numerical(
                "global Gauss–Newton conjugate-gradient step is non-finite".into(),
            ));
        }
        for index in 0..length {
            solution[index] += step * direction[index];
            residual[index] -= step * operator_direction[index];
        }
        completed = iteration + 1;
        ratio = real_norm(&residual) / initial_norm;
        if !ratio.is_finite() {
            return Err(Error::Numerical(
                "global Gauss–Newton linear residual is non-finite".into(),
            ));
        }
        if ratio <= algorithm.cg_relative_tolerance {
            break;
        }
        apply_preconditioner(
            &residual,
            coverage,
            algorithm.damping,
            &mut preconditioned,
        );
        let next_product = real_dot(&residual, &preconditioned);
        if !next_product.is_finite() || next_product <= 0.0 {
            return Err(Error::Numerical(
                "global Gauss–Newton conjugate-gradient residual broke down".into(),
            ));
        }
        let beta = next_product / residual_product;
        for index in 0..length {
            direction[index] = preconditioned[index] + beta * direction[index];
        }
        residual_product = next_product;
    }
    if solution.iter().any(|value| !complex_is_finite(*value)) {
        return Err(Error::Numerical(
            "global Gauss–Newton direction is non-finite".into(),
        ));
    }
    Ok((solution, completed, ratio))
}

fn apply_preconditioner(
    input: &[Complex64],
    coverage: &[f64],
    damping: f64,
    output: &mut [Complex64],
) {
    for index in 0..input.len() {
        output[index] = input[index] / ((1.0 + damping) * coverage[index]);
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_normal_operator<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &ReconstructionState,
    object: &[Complex64],
    vector: &[Complex64],
    coverage: &[f64],
    damping: f64,
    epsilon: f64,
    workspace: &mut GaussNewtonWorkspace,
) -> Result<Vec<Complex64>> {
    let model = &problem.model;
    let low_len = checked_len_2d(model.image_shape)?;
    let total_weight = positive_weight_sum(problem)?;
    let mut output = vec![Complex64::default(); object.len()];
    for frame in 0..model.frame_count() {
        let frame_weight = problem.measurements.frame_weight(frame)?;
        if frame_weight == 0.0 {
            continue;
        }
        let valid_pixels = valid_pixel_count(problem, frame)?;
        let mask = problem.measurements.frame_mask(frame)?;
        let residual_scale =
            (frame_weight / (valid_pixels as f64 * total_weight)).sqrt();
        let single_source = [(frame, 1.0)];
        let sources = model
            .multiplexing_matrix
            .as_ref()
            .map_or(single_source.as_slice(), |matrix| matrix[frame].as_slice());
        predict_frame(problem, state, object, sources, workspace)?;
        workspace.directional.fill(0.0);
        for (mode, &(source, source_weight)) in sources.iter().enumerate() {
            let offset = state.effective_source_offset(model, source)?;
            forward_source(
                model,
                state,
                vector,
                source,
                offset,
                &mut workspace.patch,
                &mut workspace.centered,
                &mut workspace.field,
                &mut workspace.column,
            )?;
            let start = mode * low_len;
            let mode_field = &workspace.mode_fields[start..start + low_len];
            for pixel in 0..low_len {
                workspace.directional[pixel] += source_weight
                    * (mode_field[pixel].conj() * workspace.field[pixel]).re;
            }
        }
        for pixel in 0..low_len {
            let amplitude = workspace.predicted[pixel].max(0.0).sqrt();
            workspace.denominator[pixel] = amplitude.max(epsilon.sqrt());
            workspace.residual[pixel] = if mask.is_some_and(|values| values[pixel] == 0) {
                0.0
            } else {
                residual_scale * workspace.directional[pixel] / workspace.denominator[pixel]
            };
        }
        for (mode, &(source, source_weight)) in sources.iter().enumerate() {
            let start = mode * low_len;
            let mode_field = &workspace.mode_fields[start..start + low_len];
            for pixel in 0..low_len {
                workspace.detector[pixel] = if mask.is_some_and(|values| values[pixel] == 0) {
                    Complex64::default()
                } else {
                    mode_field[pixel]
                        * (source_weight * residual_scale * workspace.residual[pixel]
                            / workspace.denominator[pixel])
                };
            }
            let offset = state.effective_source_offset(model, source)?;
            adjoint_source(
                model,
                state,
                source,
                offset,
                &workspace.detector,
                &mut output,
                &mut workspace.field,
                &mut workspace.centered,
                &mut workspace.column,
            )?;
        }
    }
    for index in 0..output.len() {
        output[index] += damping * coverage[index] * vector[index];
    }
    if output.iter().any(|value| !complex_is_finite(*value)) {
        return Err(Error::Numerical(
            "global Gauss–Newton normal-operator product is non-finite".into(),
        ));
    }
    Ok(output)
}

fn predict_frame<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &ReconstructionState,
    object: &[Complex64],
    sources: &[(usize, f64)],
    workspace: &mut GaussNewtonWorkspace,
) -> Result<()> {
    let low_len = checked_len_2d(problem.model.image_shape)?;
    workspace.resize_modes(sources.len(), low_len)?;
    workspace.predicted.fill(0.0);
    for (mode, &(source, source_weight)) in sources.iter().enumerate() {
        let offset = state.effective_source_offset(&problem.model, source)?;
        forward_source(
            &problem.model,
            state,
            object,
            source,
            offset,
            &mut workspace.patch,
            &mut workspace.centered,
            &mut workspace.field,
            &mut workspace.column,
        )?;
        let start = mode * low_len;
        workspace.mode_fields[start..start + low_len].copy_from_slice(&workspace.field);
        for pixel in 0..low_len {
            workspace.predicted[pixel] += source_weight * workspace.field[pixel].norm_sqr();
        }
    }
    if workspace
        .predicted
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(Error::Numerical(
            "global Gauss–Newton forward prediction is non-finite".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_source(
    model: &ImagePlaneModel,
    state: &ReconstructionState,
    object: &[Complex64],
    source: usize,
    offset: FourierOffset,
    patch: &mut [Complex64],
    centered: &mut [Complex64],
    field: &mut [Complex64],
    column: &mut [Complex64],
) -> Result<()> {
    let view = StandardView2::try_from(ArrayView2::from_shape(
        model.reconstruction_shape,
        object,
    )?)?;
    model.extract_patch_at_offset(view, source, offset, patch)?;
    for pixel in 0..patch.len() {
        centered[pixel] = patch[pixel] * state.pupil.values.as_slice()[pixel];
    }
    ifftshift_copy(centered, field, model.image_shape);
    state
        .backend
        .fft2(field, model.image_shape, FftDirection::Inverse, column)
}

#[allow(clippy::too_many_arguments)]
fn adjoint_source(
    model: &ImagePlaneModel,
    state: &ReconstructionState,
    source: usize,
    offset: FourierOffset,
    detector: &[Complex64],
    destination: &mut [Complex64],
    field: &mut [Complex64],
    centered: &mut [Complex64],
    column: &mut [Complex64],
) -> Result<()> {
    field.copy_from_slice(detector);
    state
        .backend
        .fft2(field, model.image_shape, FftDirection::Forward, column)?;
    fftshift_copy(field, centered, model.image_shape);
    for pixel in 0..centered.len() {
        centered[pixel] *= state.pupil.values.as_slice()[pixel].conj();
    }
    model.insert_patch_adjoint_slice_at_offset(
        destination,
        source,
        centered,
        checked_len_2d(model.image_shape)? as f64,
        offset,
    )
}

fn frame_gain(state: &ReconstructionState, frame: usize) -> Result<f64> {
    let gain = state.frame_gains.as_ref().map_or(1.0, |values| values[frame]);
    if !gain.is_finite() || gain <= 0.0 {
        return Err(Error::InvalidModel(format!(
            "state frame {frame} has invalid gain {gain}"
        )));
    }
    Ok(gain)
}

fn background_value(
    state: &ReconstructionState,
    frame: usize,
    pixel: usize,
    image_len: usize,
) -> f64 {
    state.background.as_ref().map_or(0.0, |values| {
        values[if values.len() == image_len {
            pixel
        } else {
            frame * image_len + pixel
        }]
    })
}

fn real_dot(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (left.conj() * right).re)
        .sum()
}

fn real_norm(values: &[Complex64]) -> f64 {
    values.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt()
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::presets::noiseless_mixed_fpm;

    fn deterministic_vector(length: usize, phase: usize) -> Vec<Complex64> {
        (0..length)
            .map(|index| {
                let real = ((index + 3 * phase) % 17) as f64 - 8.0;
                let imaginary = ((5 * index + phase) % 19) as f64 - 9.0;
                Complex64::new(real / 17.0, imaginary / 19.0)
            })
            .collect()
    }

    #[test]
    fn compiled_source_forward_and_adjoint_obey_the_real_dot_product() {
        let simulation = noiseless_mixed_fpm(17).unwrap();
        let problem =
            ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
                .unwrap();
        let state = ReconstructionState::initialize(&problem).unwrap();
        let mut workspace = GaussNewtonWorkspace::new(&problem.model).unwrap();
        let object = deterministic_vector(state.object_spectrum.len(), 1);
        let detector = deterministic_vector(workspace.field.len(), 2);
        let offset = state.effective_source_offset(&problem.model, 0).unwrap();
        forward_source(
            &problem.model,
            &state,
            &object,
            0,
            offset,
            &mut workspace.patch,
            &mut workspace.centered,
            &mut workspace.field,
            &mut workspace.column,
        )
        .unwrap();
        let field = workspace.field.clone();
        let mut adjoint = vec![Complex64::default(); object.len()];
        adjoint_source(
            &problem.model,
            &state,
            0,
            offset,
            &detector,
            &mut adjoint,
            &mut workspace.field,
            &mut workspace.centered,
            &mut workspace.column,
        )
        .unwrap();

        let forward_dot = real_dot(&field, &detector);
        let adjoint_dot = real_dot(&object, &adjoint);
        let scale = forward_dot.abs().max(adjoint_dot.abs()).max(1.0);
        assert!((forward_dot - adjoint_dot).abs() <= 1e-11 * scale);
    }

    #[test]
    fn analytic_global_gradient_matches_a_centered_objective_difference() {
        let simulation = noiseless_mixed_fpm(23).unwrap();
        let problem =
            ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
                .unwrap();
        let state = ReconstructionState::initialize(&problem).unwrap();
        let object = state.object_spectrum.as_slice().to_vec();
        let direction = deterministic_vector(object.len(), 3);
        let mut workspace = GaussNewtonWorkspace::new(&problem.model).unwrap();
        let (_, _, gradient) =
            objective_and_gradient(&problem, &state, &object, 1e-10, &mut workspace).unwrap();
        let step = 1e-6;
        let plus: Vec<_> = object
            .iter()
            .zip(&direction)
            .map(|(&value, &delta)| value + step * delta)
            .collect();
        let minus: Vec<_> = object
            .iter()
            .zip(&direction)
            .map(|(&value, &delta)| value - step * delta)
            .collect();
        let plus_objective = objective_only(&problem, &state, &plus, &mut workspace)
            .unwrap()
            .0;
        let minus_objective = objective_only(&problem, &state, &minus, &mut workspace)
            .unwrap()
            .0;
        let finite_difference = (plus_objective - minus_objective) / (2.0 * step);
        let analytic = 2.0 * real_dot(&gradient, &direction);
        let scale = finite_difference.abs().max(analytic.abs()).max(1.0);
        assert!((finite_difference - analytic).abs() <= 5e-5 * scale);
    }

    #[test]
    fn damped_normal_operator_is_real_symmetric_and_positive() {
        let simulation = noiseless_mixed_fpm(31).unwrap();
        let problem =
            ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
                .unwrap();
        let state = ReconstructionState::initialize(&problem).unwrap();
        let object = state.object_spectrum.as_slice().to_vec();
        let left = deterministic_vector(object.len(), 4);
        let right = deterministic_vector(object.len(), 5);
        let coverage = vec![1.0; object.len()];
        let mut workspace = GaussNewtonWorkspace::new(&problem.model).unwrap();
        let normal_left = apply_normal_operator(
            &problem,
            &state,
            &object,
            &left,
            &coverage,
            1e-3,
            1e-10,
            &mut workspace,
        )
        .unwrap();
        let normal_right = apply_normal_operator(
            &problem,
            &state,
            &object,
            &right,
            &coverage,
            1e-3,
            1e-10,
            &mut workspace,
        )
        .unwrap();

        let left_right = real_dot(&left, &normal_right);
        let right_left = real_dot(&normal_left, &right);
        let scale = left_right.abs().max(right_left.abs()).max(1.0);
        assert!((left_right - right_left).abs() <= 1e-10 * scale);
        assert!(real_dot(&left, &normal_left) > 0.0);
        assert!(real_dot(&right, &normal_right) > 0.0);
    }
}
