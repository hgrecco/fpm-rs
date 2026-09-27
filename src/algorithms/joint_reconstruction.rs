use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    Result,
    callbacks::Callback,
    error::Error,
    experiment::{Illumination, Optics},
    illumination_calibration::{
        CalibrationConditioning, CalibrationConvergenceReason, CalibrationLossHistoryEntry,
        CalibrationParameterHistoryEntry, IlluminationCalibration, IlluminationCalibrationState,
        PlanarArrayParameterValues,
    },
    measurements::MeasurementRead,
    model::{ImagePlaneModel, ReconstructionShape},
    reconstruction::{
        AlgorithmMetricRecord, Batch, ReconstructionCheckpoint, ReconstructionProblem,
        ReconstructionResult, ReconstructionState, RunOptions, Runner,
    },
};

use super::{AlgorithmIterationMetrics, ReconstructionAlgorithm, StepOutput, StepSummary};

/// Scalar metrics emitted after one alternating object/illumination outer iteration.
#[derive(Clone, Debug, Default)]
pub struct JointIterationMetrics {
    data_loss: Option<f64>,
    regularization_loss: Option<f64>,
    total_loss: Option<f64>,
    geometry_recompilations: usize,
    multiplicative_updates: usize,
    rejected_steps: usize,
    object_metrics: Vec<(String, String, f64)>,
}

impl AlgorithmIterationMetrics for JointIterationMetrics {
    fn merge(&mut self, other: Self) {
        self.data_loss = other.data_loss.or(self.data_loss);
        self.regularization_loss = other.regularization_loss.or(self.regularization_loss);
        self.total_loss = other.total_loss.or(self.total_loss);
        self.geometry_recompilations += other.geometry_recompilations;
        self.multiplicative_updates += other.multiplicative_updates;
        self.rejected_steps += other.rejected_steps;
        self.object_metrics.extend(other.object_metrics);
    }

    fn append_records(&self, iteration: usize, output: &mut Vec<AlgorithmMetricRecord>) {
        for (metric, value) in [
            ("data_loss", self.data_loss),
            ("regularization_loss", self.regularization_loss),
            ("total_loss", self.total_loss),
        ] {
            if let Some(value) = value {
                output.push(AlgorithmMetricRecord {
                    iteration,
                    namespace: "physical_illumination".into(),
                    metric: metric.into(),
                    value,
                });
            }
        }
        for (metric, value) in [
            (
                "geometry_recompilations",
                self.geometry_recompilations as f64,
            ),
            ("multiplicative_updates", self.multiplicative_updates as f64),
            ("rejected_steps", self.rejected_steps as f64),
        ] {
            output.push(AlgorithmMetricRecord {
                iteration,
                namespace: "physical_illumination".into(),
                metric: metric.into(),
                value,
            });
        }
        output.extend(
            self.object_metrics
                .iter()
                .map(|(namespace, metric, value)| AlgorithmMetricRecord {
                    iteration,
                    namespace: format!("object_update.{namespace}"),
                    metric: metric.clone(),
                    value: *value,
                }),
        );
    }
}

/// Alternates an existing object or object/pupil algorithm with physical LED calibration.
///
/// One runner iteration is one outer iteration. The complete acquisition order is supplied
/// to the wrapped algorithm `object_iterations_per_outer` times, then bounded physical
/// updates refresh only illumination-dependent model state. Generic Fourier-grid correction
/// remains an independent feature of [`crate::algorithms::GradientDescent`].
///
/// The joint-estimation motivation follows [J. Sun, Q. Chen, Y. Zhang, and C. Zuo,
/// “Efficient positional misalignment correction method for Fourier ptychographic
/// microscopy,” *Biomedical Optics Express* **7**(4), 1336–1350
/// (2016)](https://doi.org/10.1364/BOE.7.001336). This implementation uses bounded,
/// scaled finite differences rather than that work's simulated annealing and nonlinear
/// regression. It does not implement the brightfield circle detection or spectral
/// correlation of [R. Eckert, Z. F. Phillips, and L. Waller, “Efficient illumination
/// angle self-calibration in Fourier ptychography,” *Applied Optics* **57**(19),
/// 5434–5442 (2018)](https://doi.org/10.1364/AO.57.005434).
#[derive(Clone, Debug)]
pub struct JointReconstruction<A> {
    /// Existing analytic object or object/pupil update algorithm.
    pub object_algorithm: A,
    /// Optical configuration used to resolve physical source positions.
    pub optics: Optics,
    /// Nominal reusable physical illumination.
    pub initial_illumination: Illumination,
    /// Physical parameter selection, priors, loss, and bounded optimizer.
    pub illumination_calibration: IlluminationCalibration,
    /// Number of alternating outer iterations.
    pub outer_iterations: usize,
    /// Complete object-update passes before each physical phase.
    pub object_iterations_per_outer: usize,
    /// Repetitions of the configured bounded physical phase per outer iteration.
    pub illumination_steps_per_outer: usize,
}

impl<A> JointReconstruction<A> {
    /// Creates an alternating reconstruction with one object pass and one calibration phase.
    pub fn new(
        object_algorithm: A,
        optics: Optics,
        initial_illumination: Illumination,
        illumination_calibration: IlluminationCalibration,
        outer_iterations: usize,
    ) -> Self {
        Self {
            object_algorithm,
            optics,
            initial_illumination,
            illumination_calibration,
            outer_iterations,
            object_iterations_per_outer: 1,
            illumination_steps_per_outer: 1,
        }
    }

    /// Sets the positive number of complete object passes in each outer iteration.
    pub fn object_iterations_per_outer(mut self, iterations: usize) -> Self {
        self.object_iterations_per_outer = iterations;
        self
    }

    /// Sets the positive number of bounded illumination phases in each outer iteration.
    pub fn illumination_steps_per_outer(mut self, steps: usize) -> Self {
        self.illumination_steps_per_outer = steps;
        self
    }
}

impl<A: ReconstructionAlgorithm> JointReconstruction<A> {
    /// Runs alternating reconstruction without callbacks and returns structured physical output.
    pub fn run<M: MeasurementRead>(
        self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<JointReconstructionResult> {
        let options = RunOptions {
            max_iterations: self.outer_iterations,
            batch_size: usize::MAX,
            ..RunOptions::default()
        };
        JointReconstructionResult::from_reconstruction(Runner::new(self, options).run(problem)?)
    }

    /// Runs alternating reconstruction with ordinary runner callbacks.
    ///
    /// Outer-iteration callbacks receive physical phase metrics in the
    /// `physical_illumination` namespace, and checkpoint callbacks capture the complete
    /// physical state and refreshed model.
    pub fn run_with_callbacks<M: MeasurementRead>(
        self,
        problem: &ReconstructionProblem<M>,
        callbacks: Vec<Box<dyn Callback>>,
    ) -> Result<JointReconstructionResult> {
        let options = RunOptions {
            max_iterations: self.outer_iterations,
            batch_size: usize::MAX,
            ..RunOptions::default()
        };
        JointReconstructionResult::from_reconstruction(
            Runner::new(self, options)
                .with_callbacks(callbacks)
                .run(problem)?,
        )
    }

    /// Resumes alternating reconstruction from a checkpoint containing physical state.
    pub fn run_from_checkpoint<M: MeasurementRead>(
        self,
        problem: &ReconstructionProblem<M>,
        checkpoint: ReconstructionCheckpoint,
    ) -> Result<JointReconstructionResult> {
        let options = RunOptions {
            max_iterations: self.outer_iterations,
            batch_size: usize::MAX,
            ..RunOptions::default()
        };
        JointReconstructionResult::from_reconstruction(
            Runner::new(self, options)
                .resume_from(checkpoint)
                .run(problem)?,
        )
    }
}

impl<A: ReconstructionAlgorithm> ReconstructionAlgorithm for JointReconstruction<A> {
    type IterationMetrics = JointIterationMetrics;

    fn validate(&self) -> Result<()> {
        self.object_algorithm.validate()?;
        self.illumination_calibration
            .validate_for(&self.initial_illumination)?;
        self.optics.validate()?;
        if self.outer_iterations == 0
            || self.object_iterations_per_outer == 0
            || self.illumination_steps_per_outer == 0
        {
            return Err(Error::InvalidParameter {
                name: "joint iteration counts",
                reason: "outer, object, and illumination iteration counts must be positive".into(),
            });
        }
        Ok(())
    }

    fn validate_problem<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<()> {
        self.validate()?;
        self.object_algorithm.validate_problem(problem)?;
        let expected = ImagePlaneModel::from_experiment(
            &self.optics,
            &self.initial_illumination,
            problem.model.image_shape(),
            ReconstructionShape::Exact(problem.model.reconstruction_shape()),
        )?;
        if expected.source_count() != problem.model.source_count()
            || expected.frame_count() != problem.model.frame_count()
            || expected.k_vectors() != problem.model.k_vectors()
        {
            return Err(Error::InvalidModel(
                "joint reconstruction problem model was not compiled from the supplied initial illumination"
                    .into(),
            ));
        }
        Ok(())
    }

    fn initialize<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
    ) -> Result<ReconstructionState> {
        let mut state = self.object_algorithm.initialize(problem)?;
        let calibration_state = self
            .illumination_calibration
            .initialize(&self.initial_illumination, &self.optics)?;
        let mut calibrated_model = problem.model.clone();
        self.illumination_calibration.synchronize_model(
            &self.optics,
            &mut calibrated_model,
            &self.initial_illumination,
            &calibration_state,
        )?;
        state.frame_gains = calibrated_model.frame_gains().map(<[f64]>::to_vec);
        state.physical_illumination_calibration = Some(calibration_state);
        state.calibrated_model = Some(calibrated_model);
        Ok(state)
    }

    fn initialize_with_backend<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        backend: std::sync::Arc<dyn crate::backend::Backend>,
    ) -> Result<ReconstructionState> {
        let mut state = self
            .object_algorithm
            .initialize_with_backend(problem, backend)?;
        let calibration_state = self
            .illumination_calibration
            .initialize(&self.initial_illumination, &self.optics)?;
        let mut calibrated_model = problem.model.clone();
        self.illumination_calibration.synchronize_model(
            &self.optics,
            &mut calibrated_model,
            &self.initial_illumination,
            &calibration_state,
        )?;
        state.frame_gains = calibrated_model.frame_gains().map(<[f64]>::to_vec);
        state.physical_illumination_calibration = Some(calibration_state);
        state.calibrated_model = Some(calibrated_model);
        Ok(state)
    }

    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
        batch: &Batch,
        iteration: usize,
    ) -> Result<StepOutput<Self::IterationMetrics>> {
        let mut effective_model = state
            .calibrated_model
            .clone()
            .ok_or_else(|| Error::InvalidModel("joint calibrated model is missing".into()))?;
        let local_problem = ReconstructionProblem {
            measurements: &problem.measurements,
            model: effective_model.clone(),
            name: problem.name.clone(),
        };
        let mut summary = StepSummary::default();
        let mut object_metrics = Vec::new();
        for _ in 0..self.object_iterations_per_outer {
            let output = self
                .object_algorithm
                .step(&local_problem, state, batch, iteration)?;
            self.object_algorithm
                .canonicalize_state(&local_problem, state)?;
            summary.merge(output.summary);
            let mut records = Vec::new();
            output.metrics.append_records(iteration + 1, &mut records);
            object_metrics.extend(
                records
                    .into_iter()
                    .map(|record| (record.namespace, record.metric, record.value)),
            );
        }

        let geometry_before = state
            .physical_illumination_calibration
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidModel("joint physical calibration state is missing".into())
            })?
            .geometry_recompilations;
        let multiplicative_before = state
            .physical_illumination_calibration
            .as_ref()
            .expect("checked above")
            .multiplicative_updates;
        let rejected_before = state
            .physical_illumination_calibration
            .as_ref()
            .expect("checked above")
            .rejected_steps;
        let mut objective = None;
        for _ in 0..self.illumination_steps_per_outer {
            objective = Some(
                self.illumination_calibration.optimize(
                    &problem.measurements,
                    &self.optics,
                    state.object_spectrum.ndarray_view(),
                    &state.pupil,
                    &mut effective_model,
                    state
                        .physical_illumination_calibration
                        .as_mut()
                        .expect("checked above"),
                    iteration + 1,
                )?,
            );
        }
        state.frame_gains = effective_model.frame_gains().map(<[f64]>::to_vec);
        state.calibrated_model = Some(effective_model);
        let calibration = state
            .physical_illumination_calibration
            .as_ref()
            .expect("checked above");
        let objective = objective.expect("positive illumination phase count was validated");
        Ok(StepOutput {
            summary,
            metrics: JointIterationMetrics {
                data_loss: Some(objective.data_loss),
                regularization_loss: Some(objective.regularization_loss),
                total_loss: Some(objective.total_loss),
                geometry_recompilations: calibration.geometry_recompilations - geometry_before,
                multiplicative_updates: calibration.multiplicative_updates - multiplicative_before,
                rejected_steps: calibration.rejected_steps - rejected_before,
                object_metrics,
            },
        })
    }

    fn canonicalize_state<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        state: &mut ReconstructionState,
    ) -> Result<()> {
        let model = state
            .calibrated_model
            .clone()
            .unwrap_or_else(|| problem.model.clone());
        let local_problem = ReconstructionProblem {
            measurements: &problem.measurements,
            model,
            name: problem.name.clone(),
        };
        self.object_algorithm
            .canonicalize_state(&local_problem, state)
    }

    fn iterations(&self) -> usize {
        self.outer_iterations
    }

    fn batch_size(&self) -> usize {
        usize::MAX
    }
}

/// Structured joint result with a normal reconstruction and reusable physical illumination.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointReconstructionResult {
    /// Ordinary reconstruction fields, trace, diagnostics, and calibration state.
    pub reconstruction: ReconstructionResult,
    /// Gauge-normalized initial physical illumination.
    pub initial_illumination: Illumination,
    /// Final reusable physical illumination.
    pub calibrated_illumination: Illumination,
    /// Final image-plane model refreshed from the calibrated illumination.
    pub calibrated_model: ImagePlaneModel,
    /// Immutable initial absolute physical and multiplicative values.
    pub initial_parameters: PlanarArrayParameterValues,
    /// Final absolute physical and multiplicative values.
    pub final_parameters: PlanarArrayParameterValues,
    /// Accepted and rejected parameter trials.
    pub parameter_history: Vec<CalibrationParameterHistoryEntry>,
    /// Data, regularization, and total objective history.
    pub loss_history: Vec<CalibrationLossHistoryEntry>,
    /// Stop reason for the last illumination phase.
    pub convergence_reason: Option<CalibrationConvergenceReason>,
    /// Practical finite-difference conditioning indicators.
    pub conditioning: CalibrationConditioning,
    /// Complete checkpointable physical state and update counters.
    pub diagnostics: IlluminationCalibrationState,
}

impl JointReconstructionResult {
    /// Builds the structured view from a runner result containing physical state.
    pub fn from_reconstruction(reconstruction: ReconstructionResult) -> Result<Self> {
        let diagnostics = reconstruction
            .physical_illumination_calibration
            .clone()
            .ok_or_else(|| {
                Error::InvalidModel("joint result has no physical calibration".into())
            })?;
        let calibrated_model = reconstruction
            .calibrated_model
            .clone()
            .ok_or_else(|| Error::InvalidModel("joint result has no calibrated model".into()))?;
        Ok(Self {
            initial_illumination: diagnostics.initial_illumination.clone(),
            calibrated_illumination: diagnostics.current_illumination.clone(),
            calibrated_model,
            initial_parameters: diagnostics.initial_parameters.clone(),
            final_parameters: diagnostics.current_parameters.clone(),
            parameter_history: diagnostics.parameter_history.clone(),
            loss_history: diagnostics.loss_history.clone(),
            convergence_reason: diagnostics.convergence_reason,
            conditioning: diagnostics.conditioning.clone(),
            diagnostics,
            reconstruction,
        })
    }

    /// Serializes the complete structured joint result as JSON.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Loads and validates a structured joint JSON result.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let result: Self = serde_json::from_reader(reader)?;
        result.reconstruction.validate()?;
        result.calibrated_model.validate()?;
        Ok(result)
    }

    /// Writes the ordinary and physical result through the established bundle path.
    #[cfg(feature = "parquet")]
    pub fn write_bundle(
        &self,
        path: impl AsRef<Path>,
        options: crate::reconstruction::BundleExportOptions,
    ) -> Result<crate::reconstruction::ResultBundle> {
        self.reconstruction.write_bundle(path, options)
    }
}
