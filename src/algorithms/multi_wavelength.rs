//! Joint real-parameter optimization of nondispersive OPD and channel amplitudes.

use std::{f64::consts::TAU, time::Instant};

use ndarray::Array2;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Error, Result,
    array_layout::StandardView2,
    array_serde::Array2Data,
    backend::FftDirection,
    measurements::MeasurementRead,
    model::{ObjectCoupling, fftshift_copy, ifftshift_copy},
    reconstruction::{
        IterationRecord, OpticalPathDifferenceResult, PhaseReference, ReconstructionTrace,
        RuntimeInfo, SpectralChannelResult, SpectralCheckpointOptions,
        SpectralReconstructionCheckpoint, SpectralReconstructionProblem,
        SpectralReconstructionResult, SpectralReconstructionState, SyntheticWavelengthUnwrapper,
        spectral_checkpoint::{StoredOpd, StoredSolver, fingerprints},
    },
};

use super::{SpectralAlternatingProjection, SpectralReconstructionAlgorithm};

/// Joint OPD fit, derived wavelength fields, and optional initialization diagnostics.
#[derive(Clone, Debug)]
pub struct MultiWavelengthSolverResult {
    /// Optimized nondispersive OPD in metres on the common `(row, column)` grid.
    pub opd_m: Array2<f64>,
    /// Derived channel fields and fixed pupils, with the joint objective trace.
    /// The model uses independent field storage; this solver couples their phases
    /// through `opd_m`. Trace iteration zero is the constrained initial objective,
    /// followed by post-update smoothed amplitude objectives.
    pub spectral: SpectralReconstructionResult,
    /// Phase-mixing diagnostics before joint optimization, absent for explicit OPD starts.
    pub initialization_opd: Option<OpticalPathDifferenceResult>,
    /// Independent-field AP trace used by automatic initialization, otherwise absent.
    pub initialization_trace: Option<ReconstructionTrace>,
    /// Final accepted OPD/amplitude state, fixed gauge, and initialization records for exact resume.
    pub checkpoint: SpectralReconstructionCheckpoint,
}

/// Fits one shared nondispersive OPD and a positive amplitude for each wavelength.
///
/// Every forward evaluation uses `O_c = A_c exp(i 2π d / λ_c)`, with vacuum
/// wavelengths in metres. All fixed channel kernels and sparse intensity mixtures
/// come from the compiled spectral model. The objective is the frame-weighted
/// mean of valid-pixel `(sqrt(prediction + epsilon) - sqrt(max(measurement, 0) + epsilon))²`,
/// including known detector gain/background. Analytic adjoints differentiate the
/// objective into amplitudes and the real shared OPD; there is no independent
/// channel phase update. A full-data backtracking step projects amplitudes to
/// at least `sqrt(epsilon)` and OPD to its explicit half-open interval.
///
/// The phase coordinate for descent is `q = 2π d / min(λ)`. Both real gradients
/// are multiplied by the common-grid pixel count before applying their separate
/// step multipliers. Backtracking halves one common multiplier until the full
/// objective is non-increasing. Failure to find such a step stops cleanly and is
/// reported by `runtime.stopped_early`; this nonconvex solver cannot guarantee
/// recovery of a branch missed by initialization.
///
/// Automatic initialization reconstructs independent fields with spectral AP,
/// then uses referenced synthetic-wavelength unwrapping. Every initial OPD pixel
/// must be valid. An explicit OPD/amplitude start can bypass that initializer.
/// A reference region is fixed pixelwise at its known OPD throughout optimization;
/// explicit piston offsets instead fix the initial spatial mean OPD. Absolute
/// reference, nondispersive OPD, registered grids, and matched effective resolution
/// remain assumptions. Pupils, source powers, geometry, gain, and background are fixed.
///
/// # Example
/// ```
/// use fpm_rs::{Result, algorithms::{MultiWavelengthGradientDescent, MultiWavelengthSolverResult},
///     measurements::MeasurementRead, reconstruction::{PhaseReference,
///     SpectralReconstructionProblem, SyntheticWavelengthUnwrapper}};
/// fn solve<M: MeasurementRead>(problem: &SpectralReconstructionProblem<M>,
///     reference: &PhaseReference) -> Result<MultiWavelengthSolverResult> {
///     let unwrapper = SyntheticWavelengthUnwrapper::new((-1e-6, 2e-6))?;
///     MultiWavelengthGradientDescent::default().run(problem, &unwrapper, reference)
/// }
/// ```
///
/// # References
/// L. Bian, J. Suo, G. Zheng, K. Guo, F. Chen, and Q. Dai,
/// [“Fourier ptychographic reconstruction using Wirtinger flow optimization”](https://doi.org/10.1364/OE.23.004856),
/// *Optics Express* **23**(4), 4856–4866 (2015), for FPM loss-gradient optimization.
/// The shared-OPD chain rule, smoothed amplitude loss, box/gauge projections, and
/// monotone full-data backtracking are implementation extensions, not that paper's solver.
/// See [`SyntheticWavelengthUnwrapper`] for the initialization method and reference.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiWavelengthGradientDescent {
    /// Positive maximum number of accepted full-data joint updates.
    pub iterations: usize,
    /// Positive number of spectral AP passes when automatic initialization is used.
    pub initialization_iterations: usize,
    /// Finite positive multiplier for the pixel-count-scaled real amplitude gradient.
    pub amplitude_step: f64,
    /// Finite positive multiplier for the pixel-count-scaled phase-coordinate gradient.
    pub opd_step: f64,
    /// Positive maximum number of step trials per update, halving the common multiplier.
    pub max_backtracks: usize,
    /// Finite positive intensity smoothing and squared minimum amplitude.
    pub epsilon: f64,
}

impl Default for MultiWavelengthGradientDescent {
    fn default() -> Self {
        Self {
            iterations: 100,
            initialization_iterations: 50,
            amplitude_step: 1.0,
            opd_step: 1.0,
            max_backtracks: 30,
            epsilon: 1e-10,
        }
    }
}

struct Parameters {
    opd: Array2<f64>,
    amplitudes: Vec<Array2<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Gauge {
    Region { pixels: Vec<usize>, opd_m: f64 },
    Mean(f64),
}

struct Execution {
    parameters: Parameters,
    gauge: Gauge,
    opd_range_m: (f64, f64),
    trace: ReconstructionTrace,
    completed: usize,
    elapsed_offset: f64,
    signature: (String, String),
    initialization_opd: Option<OpticalPathDifferenceResult>,
    initialization_trace: Option<ReconstructionTrace>,
}

struct Gradient {
    amplitude: Vec<Array2<f64>>,
    phase: Array2<f64>,
}

fn invalid(name: &'static str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name,
        reason: reason.into(),
    }
}

impl MultiWavelengthGradientDescent {
    /// Checks positive iteration/trial counts, step multipliers, and numerical smoothing.
    /// `initialization_iterations` is checked only by the automatic `run` path.
    pub fn validate(&self) -> Result<()> {
        if self.iterations == 0 || self.max_backtracks == 0 {
            return Err(invalid("iterations/max_backtracks", "must be positive"));
        }
        for (name, value) in [
            ("amplitude_step", self.amplitude_step),
            ("opd_step", self.opd_step),
            ("epsilon", self.epsilon),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(name, "must be finite and positive"));
            }
        }
        Ok(())
    }

    /// Reconstructs independent fields, unwraps their phases, then jointly refines OPD/amplitudes.
    ///
    /// Requires independent model storage and at least two wavelengths. Rejects
    /// any invalid phase-unwrapped initialization pixel; use `run_from_opd` for
    /// an externally supplied finite branch when phase fusion cannot initialize it.
    /// The result retains the pre-fit unwrapping diagnostics and independent AP trace.
    pub fn run<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        unwrapper: &SyntheticWavelengthUnwrapper,
        reference: &PhaseReference,
    ) -> Result<MultiWavelengthSolverResult> {
        self.run_with_options(
            problem,
            unwrapper,
            reference,
            &SpectralCheckpointOptions::default(),
        )
    }

    /// Runs automatic AP/phase initialization with optional accepted-update checkpoint files.
    /// The final checkpoint retains initialization records and never repeats them on resume.
    pub fn run_with_options<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        unwrapper: &SyntheticWavelengthUnwrapper,
        reference: &PhaseReference,
        options: &SpectralCheckpointOptions,
    ) -> Result<MultiWavelengthSolverResult> {
        options.validate()?;
        self.validate_problem(problem, unwrapper.opd_range_m)?;
        if self.initialization_iterations == 0 {
            return Err(invalid(
                "initialization_iterations",
                "must be positive for automatic initialization",
            ));
        }
        let initial = SpectralAlternatingProjection::default()
            .iterations(self.initialization_iterations)
            .run(problem)?;
        let opd = initial.unwrap_opd(unwrapper, reference, None)?;
        if opd.valid_mask.iter().any(|&v| v == 0) {
            return Err(invalid(
                "initialization_opd",
                "phase unwrapping left invalid pixels; supply an explicit finite OPD/amplitude start or improve the reference/acquisition",
            ));
        }
        let amplitudes = initial
            .channels
            .iter()
            .map(|c| c.amplitude.clone())
            .collect();
        self.start_from_opd(
            problem,
            unwrapper.opd_range_m,
            reference,
            opd.opd_m.clone(),
            amplitudes,
            Some(opd),
            Some(initial.trace),
            options,
        )
    }

    /// Jointly refines owned finite C-contiguous OPD and channel amplitudes on the common grid.
    ///
    /// OPD is in metres and must be inside `opd_range_m = [lower, upper)`.
    /// Amplitudes must be finite and nonnegative, one array per channel, and are
    /// floored at `sqrt(epsilon)`. This bypasses phase unwrapping, so the interval
    /// may exceed a synthetic period; the caller supplies the correct fringe branch.
    /// A region reference fixes selected pixels to its known in-range OPD. Offsets
    /// must contain one finite piston per channel, and retain the initial OPD mean;
    /// no phase is subtracted from the already unwrapped explicit start.
    pub fn run_from_opd<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        opd_range_m: (f64, f64),
        reference: &PhaseReference,
        opd_m: Array2<f64>,
        amplitudes: Vec<Array2<f64>>,
    ) -> Result<MultiWavelengthSolverResult> {
        self.run_from_opd_with_options(
            problem,
            opd_range_m,
            reference,
            opd_m,
            amplitudes,
            &SpectralCheckpointOptions::default(),
        )
    }

    /// Fits explicit OPD/amplitude starts while saving periodic accepted-state checkpoints.
    /// Scientific input and gauge contracts are identical to `run_from_opd`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_from_opd_with_options<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        opd_range_m: (f64, f64),
        reference: &PhaseReference,
        opd_m: Array2<f64>,
        amplitudes: Vec<Array2<f64>>,
        options: &SpectralCheckpointOptions,
    ) -> Result<MultiWavelengthSolverResult> {
        self.start_from_opd(
            problem,
            opd_range_m,
            reference,
            opd_m,
            amplitudes,
            None,
            None,
            options,
        )
    }

    /// Resumes the saved joint OPD state without initialization or another gauge projection.
    /// `iterations` is the total target; all other solver options, detector values,
    /// masks/weights, and ordered compiled model metadata must match the checkpoint.
    pub fn run_from_checkpoint<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        checkpoint: SpectralReconstructionCheckpoint,
    ) -> Result<MultiWavelengthSolverResult> {
        self.run_from_checkpoint_with_options(
            problem,
            checkpoint,
            &SpectralCheckpointOptions::default(),
        )
    }

    /// Resumes joint state and optionally continues periodic accepted-update checkpoint output.
    pub fn run_from_checkpoint_with_options<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        checkpoint: SpectralReconstructionCheckpoint,
        options: &SpectralCheckpointOptions,
    ) -> Result<MultiWavelengthSolverResult> {
        self.validate()?;
        options.validate()?;
        checkpoint.validate_for_problem(problem)?;
        let signature = checkpoint.fingerprints();
        let StoredSolver::JointOpd {
            configuration,
            opd,
            amplitudes,
            opd_range_m,
            gauge,
            initialization_opd,
            initialization_trace,
        } = checkpoint.state
        else {
            return Err(invalid(
                "resume_from",
                "spectral AP state cannot resume joint OPD descent",
            ));
        };
        if configuration != self.configuration()?
            || self.iterations < checkpoint.completed_iterations
        {
            return Err(invalid(
                "resume_from",
                "stepping options changed or total target precedes the saved iteration",
            ));
        }
        self.validate_problem(problem, opd_range_m)?;
        let execution = Execution {
            parameters: Parameters {
                opd: opd.into_array()?,
                amplitudes: amplitudes
                    .into_iter()
                    .map(Array2Data::into_array)
                    .collect::<Result<_>>()?,
            },
            gauge,
            opd_range_m,
            trace: checkpoint.trace,
            completed: checkpoint.completed_iterations,
            elapsed_offset: checkpoint.elapsed_seconds,
            signature,
            initialization_opd: initialization_opd.map(|o| (*o).into_result()).transpose()?,
            initialization_trace,
        };
        self.execute(problem, execution, options)
    }

    #[allow(clippy::too_many_arguments)]
    fn start_from_opd<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        opd_range_m: (f64, f64),
        reference: &PhaseReference,
        opd_m: Array2<f64>,
        amplitudes: Vec<Array2<f64>>,
        initialization_opd: Option<OpticalPathDifferenceResult>,
        initialization_trace: Option<ReconstructionTrace>,
        options: &SpectralCheckpointOptions,
    ) -> Result<MultiWavelengthSolverResult> {
        options.validate()?;
        self.validate_problem(problem, opd_range_m)?;
        let started = Instant::now();
        let shape = problem.model.reconstruction_shape();
        if opd_m.dim() != shape || amplitudes.len() != problem.model.channels().len() {
            return Err(Error::InvalidShape(
                "joint OPD/amplitudes must match the common grid and channel count".into(),
            ));
        }
        StandardView2::try_from(opd_m.view())?;
        if opd_m
            .iter()
            .any(|&d| !d.is_finite() || d < opd_range_m.0 || d >= opd_range_m.1)
        {
            return Err(invalid(
                "initial_opd_m",
                "must be finite and inside the half-open OPD interval",
            ));
        }
        for amplitude in &amplitudes {
            if amplitude.dim() != shape {
                return Err(Error::InvalidShape(
                    "every amplitude must match the common grid".into(),
                ));
            }
            StandardView2::try_from(amplitude.view())?;
            if amplitude.iter().any(|&v| !v.is_finite() || v < 0.0) {
                return Err(invalid(
                    "initial_amplitudes",
                    "must be finite and nonnegative",
                ));
            }
        }
        let mut parameters = Parameters {
            opd: opd_m,
            amplitudes,
        };
        for amplitude in &mut parameters.amplitudes {
            amplitude.mapv_inplace(|v| v.max(self.epsilon.sqrt()));
        }
        let gauge = Gauge::new(reference, &parameters, opd_range_m)?;
        gauge.project(&mut parameters.opd, opd_range_m)?;
        let signature = fingerprints(problem)?;
        let execution = Execution {
            parameters,
            gauge,
            opd_range_m,
            trace: ReconstructionTrace::default(),
            completed: 0,
            elapsed_offset: started.elapsed().as_secs_f64(),
            signature,
            initialization_opd,
            initialization_trace,
        };
        self.execute(problem, execution, options)
    }

    fn execute<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        execution: Execution,
        options: &SpectralCheckpointOptions,
    ) -> Result<MultiWavelengthSolverResult> {
        let started = Instant::now();
        let Execution {
            mut parameters,
            gauge,
            opd_range_m,
            mut trace,
            mut completed,
            elapsed_offset,
            signature,
            initialization_opd,
            initialization_trace,
        } = execution;
        let configuration = self.configuration()?;
        let wavelengths: Vec<_> = problem
            .model
            .channels()
            .iter()
            .map(|c| c.model.sampling().wavelength.unwrap())
            .collect();
        let shortest = wavelengths.iter().copied().reduce(f64::min).unwrap();
        let mut state =
            SpectralReconstructionState::from_objects(problem, parameters.objects(&wavelengths))?;
        let (mut objective, _) =
            self.evaluate(problem, &mut state, &parameters, &wavelengths, false)?;
        if trace.iterations.is_empty() {
            trace.iterations.push(IterationRecord {
                iteration: 0,
                objective,
                elapsed_seconds: elapsed_offset + started.elapsed().as_secs_f64(),
            });
        } else if (trace.iterations.last().unwrap().objective - objective).abs()
            > 64.0 * f64::EPSILON * objective.abs().max(1.0)
        {
            return Err(invalid(
                "resume_from",
                "saved objective does not match the joint numerical state",
            ));
        }
        for iteration in (completed + 1)..=self.iterations {
            let (_, gradient) =
                self.evaluate(problem, &mut state, &parameters, &wavelengths, true)?;
            let gradient = gradient.unwrap();
            let mut step = 1.0;
            let mut accepted = None;
            for _ in 0..self.max_backtracks {
                let mut candidate = parameters.updated(&gradient, step, self, shortest);
                if !candidate.is_finite() {
                    step *= 0.5;
                    continue;
                }
                gauge.project(&mut candidate.opd, opd_range_m)?;
                set_fields(&mut state, &candidate.objects(&wavelengths))?;
                match self.evaluate(problem, &mut state, &candidate, &wavelengths, false) {
                    Ok((value, _)) if value <= objective => {
                        accepted = Some((candidate, value));
                        break;
                    }
                    Ok(_) | Err(Error::Numerical(_)) => step *= 0.5,
                    Err(error) => return Err(error),
                }
            }
            let Some((candidate, value)) = accepted else {
                set_fields(&mut state, &parameters.objects(&wavelengths))?;
                break;
            };
            parameters = candidate;
            objective = value;
            completed = iteration;
            trace.iterations.push(IterationRecord {
                iteration,
                objective,
                elapsed_seconds: elapsed_offset + started.elapsed().as_secs_f64(),
            });
            if options.directory.is_some() && iteration.is_multiple_of(options.every) {
                let checkpoint = SpectralReconstructionCheckpoint::capture(
                    problem.model.clone(),
                    &signature,
                    completed,
                    elapsed_offset + started.elapsed().as_secs_f64(),
                    trace.clone(),
                    StoredSolver::JointOpd {
                        configuration: configuration.clone(),
                        opd: Array2Data::from_view(parameters.opd.view()),
                        amplitudes: parameters
                            .amplitudes
                            .iter()
                            .map(|a| Array2Data::from_view(a.view()))
                            .collect(),
                        opd_range_m,
                        gauge: gauge.clone(),
                        initialization_opd: initialization_opd
                            .as_ref()
                            .map(|o| Box::new(StoredOpd::from_result(o))),
                        initialization_trace: initialization_trace.clone(),
                    },
                );
                options.write(&checkpoint, false)?;
            }
        }
        let objects = parameters.objects(&wavelengths);
        let channels = problem
            .model
            .channels()
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let object = objects[i].clone();
                SpectralChannelResult {
                    channel_id: c.channel_id.clone(),
                    wavelength_vacuum_m: wavelengths[i],
                    phase: crate::complex::phase(object.view()),
                    amplitude: parameters.amplitudes[i].clone(),
                    object,
                    object_spectrum: state.spectra[i].clone().into_inner(),
                    pupil: c.model.pupil().clone(),
                }
            })
            .collect();
        let elapsed_seconds = elapsed_offset + started.elapsed().as_secs_f64();
        let checkpoint = SpectralReconstructionCheckpoint::capture(
            problem.model.clone(),
            &signature,
            completed,
            elapsed_seconds,
            trace.clone(),
            StoredSolver::JointOpd {
                configuration,
                opd: Array2Data::from_view(parameters.opd.view()),
                amplitudes: parameters
                    .amplitudes
                    .iter()
                    .map(|a| Array2Data::from_view(a.view()))
                    .collect(),
                opd_range_m,
                gauge,
                initialization_opd: initialization_opd
                    .as_ref()
                    .map(|o| Box::new(StoredOpd::from_result(o))),
                initialization_trace: initialization_trace.clone(),
            },
        );
        options.write(&checkpoint, true)?;
        Ok(MultiWavelengthSolverResult {
            checkpoint,
            opd_m: parameters.opd,
            spectral: SpectralReconstructionResult {
                channels,
                object_coupling: ObjectCoupling::Independent,
                trace,
                checkpoint: None,
                runtime: RuntimeInfo {
                    elapsed_seconds,
                    completed_iterations: completed,
                    stopped_early: completed < self.iterations,
                    algorithm: "MultiWavelengthGradientDescent".into(),
                },
            },
            initialization_opd,
            initialization_trace,
        })
    }

    fn configuration(&self) -> Result<String> {
        let mut configuration = self.clone();
        configuration.iterations = 1;
        Ok(serde_json::to_string(&configuration)?)
    }

    fn validate_problem<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        range: (f64, f64),
    ) -> Result<()> {
        self.validate()?;
        problem.validate()?;
        SyntheticWavelengthUnwrapper::new(range)?;
        if problem.model.object_coupling() != ObjectCoupling::Independent
            || problem.model.channels().len() < 2
        {
            return Err(invalid(
                "object_coupling/channels",
                "joint OPD requires independent field storage and at least two wavelengths",
            ));
        }
        let shortest = problem
            .model
            .channels()
            .iter()
            .map(|c| c.model.sampling().wavelength.unwrap())
            .reduce(f64::min)
            .unwrap();
        if range.0.abs().max(range.1.abs()) / shortest >= (1_u64 << 52) as f64 {
            return Err(invalid(
                "opd_range_m",
                "bounds exceed exact floating-point fringe-order precision",
            ));
        }
        Ok(())
    }

    fn evaluate<M: MeasurementRead>(
        &self,
        problem: &SpectralReconstructionProblem<M>,
        state: &mut SpectralReconstructionState,
        parameters: &Parameters,
        wavelengths: &[f64],
        with_gradient: bool,
    ) -> Result<(f64, Option<Gradient>)> {
        let high_shape = problem.model.reconstruction_shape();
        let low_shape = problem.model.image_shape();
        let mut spectra: Vec<_> = (0..wavelengths.len())
            .map(|_| Array2::zeros(high_shape))
            .collect();
        let total_weight = (0..problem.model.frame_count())
            .map(|f| problem.measurements.frame_weight(f))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<f64>();
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(Error::InvalidMeasurements(
                "joint OPD total frame weight must be finite and positive".into(),
            ));
        }
        let mut objective = 0.0;
        for frame in 0..problem.model.frame_count() {
            let weight = problem.measurements.frame_weight(frame)?;
            if weight == 0.0 {
                continue;
            }
            state.evaluate_frame(&problem.model, frame)?;
            let row = &problem.model.acquisition().frames()[frame];
            let measured = problem.measurements.frame(frame)?;
            let mask = problem.measurements.frame_mask(frame)?;
            let valid = (0..measured.len())
                .filter(|&p| mask.is_none_or(|m| m[p] != 0))
                .count();
            let normalization = weight / total_weight / valid as f64;
            for pixel in 0..measured.len() {
                if mask.is_some_and(|m| m[pixel] == 0) {
                    state.projection[pixel] = Complex64::default();
                    continue;
                }
                let prediction = state.effective_gain * state.predicted[pixel] + row.background;
                if !prediction.is_finite() || prediction < 0.0 {
                    return Err(Error::Numerical(
                        "joint OPD prediction is not finite/nonnegative".into(),
                    ));
                }
                let root = (prediction + self.epsilon).sqrt();
                let target = (measured[pixel].max(0.0) + self.epsilon).sqrt();
                objective += normalization * (root - target).powi(2);
                state.projection[pixel] = Complex64::new(
                    normalization * state.effective_gain * (1.0 - target / root),
                    0.0,
                );
            }
            if with_gradient {
                for mode in &state.modes {
                    let kernel = &problem.model.channels()[mode.channel].model;
                    for p in 0..measured.len() {
                        state.field[p] = mode.field[p] * (mode.weight * state.projection[p].re);
                    }
                    state.backend.fft2(
                        &mut state.field,
                        low_shape,
                        FftDirection::Forward,
                        &mut state.column,
                    )?;
                    fftshift_copy(&state.field, &mut state.centered, low_shape);
                    for (p, &pupil) in kernel.pupil().values().iter().enumerate() {
                        state.update[p] = pupil.conj() * state.centered[p];
                    }
                    kernel.insert_patch_adjoint(
                        spectra[mode.channel].view_mut(),
                        mode.source,
                        &state.update,
                        1.0,
                    )?;
                }
            }
            state.modes.clear();
        }
        if !objective.is_finite() {
            return Err(Error::Numerical("joint OPD objective is nonfinite".into()));
        }
        if !with_gradient {
            return Ok((objective, None));
        }
        let shortest = wavelengths.iter().copied().reduce(f64::min).unwrap();
        let mut gradient = Gradient {
            amplitude: Vec::new(),
            phase: Array2::zeros(high_shape),
        };
        // FFT_norm^* = IFFT / N_high, IFFT^* = N_low FFT_norm.
        // Preserve both normalization factors before the real-parameter chain rule.
        let adjoint_scale =
            (low_shape.0 * low_shape.1) as f64 / (high_shape.0 * high_shape.1) as f64;
        for (channel, spectrum) in spectra.iter().enumerate() {
            let mut field = vec![Complex64::default(); spectrum.len()];
            ifftshift_copy(spectrum.as_slice().unwrap(), &mut field, high_shape);
            state.backend.fft2(
                &mut field,
                high_shape,
                FftDirection::Inverse,
                &mut state.column,
            )?;
            let mut amplitude_gradient = Array2::zeros(high_shape);
            for (p, &value) in field.iter().enumerate() {
                let index = (p / high_shape.1, p % high_shape.1);
                let unit = Complex64::from_polar(
                    1.0,
                    TAU * (parameters.opd[index] / wavelengths[channel]).rem_euclid(1.0),
                );
                let rotated = unit.conj() * value * adjoint_scale;
                amplitude_gradient[index] = 2.0 * rotated.re;
                gradient.phase[index] += 2.0
                    * (shortest / wavelengths[channel])
                    * parameters.amplitudes[channel][index]
                    * rotated.im;
            }
            gradient.amplitude.push(amplitude_gradient);
        }
        if gradient
            .phase
            .iter()
            .chain(gradient.amplitude.iter().flat_map(|a| a.iter()))
            .any(|v| !v.is_finite())
        {
            return Err(Error::Numerical("joint OPD gradient is nonfinite".into()));
        }
        Ok((objective, Some(gradient)))
    }
}

impl Parameters {
    fn objects(&self, wavelengths: &[f64]) -> Vec<Array2<Complex64>> {
        self.amplitudes
            .iter()
            .zip(wavelengths)
            .map(|(a, &w)| {
                Array2::from_shape_fn(self.opd.dim(), |i| {
                    Complex64::from_polar(a[i], TAU * (self.opd[i] / w).rem_euclid(1.0))
                })
            })
            .collect()
    }
    fn is_finite(&self) -> bool {
        self.opd
            .iter()
            .chain(self.amplitudes.iter().flat_map(|a| a.iter()))
            .all(|v| v.is_finite())
    }
    fn updated(
        &self,
        gradient: &Gradient,
        step: f64,
        solver: &MultiWavelengthGradientDescent,
        shortest: f64,
    ) -> Self {
        let scale = step * self.opd.len() as f64;
        let opd = Array2::from_shape_fn(self.opd.dim(), |i| {
            self.opd[i] - scale * solver.opd_step * gradient.phase[i] * shortest / TAU
        });
        let amplitudes = self
            .amplitudes
            .iter()
            .zip(&gradient.amplitude)
            .map(|(a, g)| {
                Array2::from_shape_fn(a.dim(), |i| {
                    (a[i] - scale * solver.amplitude_step * g[i]).max(solver.epsilon.sqrt())
                })
            })
            .collect();
        Self { opd, amplitudes }
    }
}

impl Gauge {
    pub(crate) fn validate(&self, opd: &Array2Data<f64>, range: (f64, f64)) -> Result<()> {
        match self {
            Self::Region { pixels, opd_m } => {
                if pixels.is_empty()
                    || pixels.windows(2).any(|w| w[0] >= w[1])
                    || !opd_m.is_finite()
                    || *opd_m < range.0
                    || *opd_m >= range.1
                    || pixels
                        .iter()
                        .any(|&p| p >= opd.data.len() || opd.data[p] != *opd_m)
                {
                    return Err(invalid(
                        "checkpoint_gauge",
                        "reference pixels must be sorted, nonempty, in range, and fixed at the known OPD",
                    ));
                }
            }
            Self::Mean(target) => {
                let mean: f64 = opd.data.iter().map(|v| v / opd.data.len() as f64).sum();
                let tolerance = 64.0 * f64::EPSILON * range.0.abs().max(range.1.abs());
                if !target.is_finite()
                    || *target < range.0
                    || *target >= range.1
                    || (mean - target).abs() > tolerance
                {
                    return Err(invalid(
                        "checkpoint_gauge",
                        "stored mean OPD disagrees with the fixed gauge",
                    ));
                }
            }
        }
        Ok(())
    }
    fn new(reference: &PhaseReference, parameters: &Parameters, range: (f64, f64)) -> Result<Self> {
        match reference {
            PhaseReference::Offsets(offsets) => {
                if offsets.len() != parameters.amplitudes.len()
                    || offsets.iter().any(|v| !v.is_finite())
                {
                    return Err(invalid(
                        "phase_offsets_rad",
                        "requires one finite piston per channel",
                    ));
                }
                // Sum after dividing to avoid overflow in a finite in-range map.
                Ok(Self::Mean(
                    parameters
                        .opd
                        .iter()
                        .map(|d| d / parameters.opd.len() as f64)
                        .sum(),
                ))
            }
            PhaseReference::Region { mask, opd_m } => {
                if mask.dim() != parameters.opd.dim() {
                    return Err(Error::InvalidShape(
                        "OPD reference mask must match the common grid".into(),
                    ));
                }
                let mask = StandardView2::try_from(mask.view())?;
                if !opd_m.is_finite() || *opd_m < range.0 || *opd_m >= range.1 {
                    return Err(invalid(
                        "reference_opd_m",
                        "must be finite and inside the OPD interval",
                    ));
                }
                let pixels = mask
                    .as_slice()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (v != 0).then_some(i))
                    .collect::<Vec<_>>();
                if pixels.is_empty() {
                    return Err(invalid(
                        "reference_mask",
                        "reference region must be nonempty",
                    ));
                }
                Ok(Self::Region {
                    pixels,
                    opd_m: *opd_m,
                })
            }
        }
    }
    fn project(&self, opd: &mut Array2<f64>, range: (f64, f64)) -> Result<()> {
        let upper = range.1.next_down();
        let data = opd.as_slice_mut().unwrap();
        match self {
            Self::Region { pixels, opd_m } => {
                for d in data.iter_mut() {
                    *d = d.clamp(range.0, upper);
                }
                for &p in pixels {
                    data[p] = *opd_m;
                }
            }
            Self::Mean(target) => {
                // Euclidean projection onto the box with a fixed mean: a common
                // shift followed by clipping, solved by monotone bisection.
                let min = data.iter().copied().reduce(f64::min).unwrap();
                let max = data.iter().copied().reduce(f64::max).unwrap();
                let mut lower = range.0 - max;
                let mut higher = upper - min;
                if !lower.is_finite() || !higher.is_finite() {
                    return Err(Error::Numerical("OPD mean projection overflow".into()));
                }
                for _ in 0..64 {
                    let shift = lower * 0.5 + higher * 0.5;
                    let mean = data
                        .iter()
                        .map(|d| (d + shift).clamp(range.0, upper) / data.len() as f64)
                        .sum::<f64>();
                    if mean < *target {
                        lower = shift;
                    } else {
                        higher = shift;
                    }
                }
                let shift = lower * 0.5 + higher * 0.5;
                for d in data.iter_mut() {
                    *d = (*d + shift).clamp(range.0, upper);
                }
            }
        }
        Ok(())
    }
}

fn set_fields(
    state: &mut SpectralReconstructionState,
    objects: &[Array2<Complex64>],
) -> Result<()> {
    let shape = objects[0].dim();
    for (spectrum, object) in state.spectra.iter_mut().zip(objects) {
        let mut field = object.as_slice().unwrap().to_vec();
        state
            .backend
            .fft2(&mut field, shape, FftDirection::Forward, &mut state.column)?;
        fftshift_copy(&field, spectrum.as_slice_mut(), shape);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        experiment::{
            AcquisitionPlan, DirectionList, IlluminationFrame, Optics, SourceCalibration,
            SourceContribution, SpectralAcquisitionPlan, SpectralChannel, SpectralContribution,
            SpectralFrame, SpectralGeometry,
        },
        measurements::MeasurementStack,
        model::{ReconstructionShape, SpectralImagePlaneModel},
    };

    #[test]
    fn amplitude_and_shared_opd_gradients_match_finite_differences_of_detector_loss() {
        let channels: Vec<_> = [500e-9, 550e-9]
            .into_iter()
            .enumerate()
            .map(|(i, w)| SpectralChannel {
                channel_id: format!("channel{i}"),
                optics: Optics {
                    wavelength_vacuum_m: w,
                    objective_na: 0.12,
                    magnification: 4.0,
                    camera_pixel_size: 6.5e-6,
                    illumination_refractive_index: 1.0,
                    objective_medium_refractive_index: 1.0,
                    defocus_distance: Some(2e-6),
                    pupil_aberration: None,
                },
                calibration: SourceCalibration::new(Some(vec![0.7, 1.3, 0.9])),
                acquisition: AcquisitionPlan::from_sparse(vec![
                    IlluminationFrame::new(
                        vec![SourceContribution {
                            source: 0,
                            intensity_weight: 1.0,
                        }],
                        1.3,
                    ),
                    IlluminationFrame::new(
                        vec![
                            SourceContribution {
                                source: 1,
                                intensity_weight: 0.4,
                            },
                            SourceContribution {
                                source: 2,
                                intensity_weight: 0.6,
                            },
                        ],
                        0.8,
                    ),
                ])
                .unwrap(),
            })
            .collect();
        let model = SpectralImagePlaneModel::from_experiment(
            &channels,
            &SpectralGeometry::Shared(
                DirectionList::from_direction_cosines(vec![
                    [0.0, 0.0],
                    [0.047, 0.031],
                    [-0.063, 0.042],
                ])
                .unwrap()
                .into(),
            ),
            SpectralAcquisitionPlan::multiplexed(
                (0..2)
                    .map(|local_frame| SpectralFrame {
                        contributions: (0..2)
                            .map(|channel| SpectralContribution {
                                channel,
                                local_frame,
                                spectral_weight: 0.6 + 0.2 * channel as f64,
                            })
                            .collect(),
                        gain: 1.7,
                        background: 0.12,
                    })
                    .collect(),
            )
            .unwrap(),
            (6, 8),
            ReconstructionShape::Smooth,
            ObjectCoupling::Independent,
        )
        .unwrap();
        let shape = model.reconstruction_shape();
        let parameters = Parameters {
            opd: Array2::from_shape_fn(shape, |(r, c)| 1.2e-6 + 5e-9 * (r + c) as f64),
            amplitudes: (0..2)
                .map(|i| {
                    Array2::from_shape_fn(shape, |(r, c)| {
                        0.8 + 0.02 * i as f64 + 0.03 * ((r + c) as f64).sin()
                    })
                })
                .collect(),
        };
        let wavelengths = [500e-9, 550e-9];
        let mut frames = Vec::new();
        let temporary = SpectralReconstructionProblem::new(
            MeasurementStack::from_frames(&[vec![1.0; 48], vec![1.0; 48]], (6, 8)).unwrap(),
            model.clone(),
        )
        .unwrap();
        let state =
            SpectralReconstructionState::from_objects(&temporary, parameters.objects(&wavelengths))
                .unwrap();
        for f in 0..2 {
            frames.push(
                model
                    .forward_intensity(&state.object_spectra(), f)
                    .unwrap()
                    .mapv(|v| 0.86 * v + 0.02)
                    .into_raw_vec_and_offset()
                    .0,
            );
        }
        let mut measurements = MeasurementStack::from_frames(&frames, (6, 8))
            .unwrap()
            .with_masks(Array2::from_shape_fn((6, 8), |(r, c)| {
                u8::from((r + c) % 4 != 0)
            }))
            .unwrap();
        measurements.set_frame_weight(0, 0.4).unwrap();
        measurements.set_frame_weight(1, 1.7).unwrap();
        let problem = SpectralReconstructionProblem::new(measurements, model).unwrap();
        let mut state =
            SpectralReconstructionState::from_objects(&problem, parameters.objects(&wavelengths))
                .unwrap();
        let solver = MultiWavelengthGradientDescent::default();
        let (_, gradient) = solver
            .evaluate(&problem, &mut state, &parameters, &wavelengths, true)
            .unwrap();
        let gradient = gradient.unwrap();
        let h = 1e-5;
        for index in [(0, 0), (2, 3), (shape.0 - 1, shape.1 - 2)] {
            for channel in 0..2 {
                let mut plus = Parameters {
                    opd: parameters.opd.clone(),
                    amplitudes: parameters.amplitudes.clone(),
                };
                plus.amplitudes[channel][index] += h;
                set_fields(&mut state, &plus.objects(&wavelengths)).unwrap();
                let high = solver
                    .evaluate(&problem, &mut state, &plus, &wavelengths, false)
                    .unwrap()
                    .0;
                plus.amplitudes[channel][index] -= 2.0 * h;
                set_fields(&mut state, &plus.objects(&wavelengths)).unwrap();
                let low = solver
                    .evaluate(&problem, &mut state, &plus, &wavelengths, false)
                    .unwrap()
                    .0;
                let numerical = (high - low) / (2.0 * h);
                assert!(
                    (gradient.amplitude[channel][index] - numerical).abs()
                        < 1e-8 * (1.0 + numerical.abs()),
                    "amplitude {index:?}: analytic {}, numerical {numerical}",
                    gradient.amplitude[channel][index]
                );
            }
            let mut plus = Parameters {
                opd: parameters.opd.clone(),
                amplitudes: parameters.amplitudes.clone(),
            };
            plus.opd[index] += h * wavelengths[0] / TAU;
            set_fields(&mut state, &plus.objects(&wavelengths)).unwrap();
            let high = solver
                .evaluate(&problem, &mut state, &plus, &wavelengths, false)
                .unwrap()
                .0;
            plus.opd[index] -= 2.0 * h * wavelengths[0] / TAU;
            set_fields(&mut state, &plus.objects(&wavelengths)).unwrap();
            let low = solver
                .evaluate(&problem, &mut state, &plus, &wavelengths, false)
                .unwrap()
                .0;
            let numerical = (high - low) / (2.0 * h);
            assert!(
                (gradient.phase[index] - numerical).abs() < 1e-8 * (1.0 + numerical.abs()),
                "OPD {index:?}: analytic {}, numerical {numerical}",
                gradient.phase[index]
            );
        }
    }
}
