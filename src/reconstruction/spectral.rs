//! Separate execution contracts for narrowband spectral reconstruction.

use std::{sync::Arc, time::Instant};

use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_complex::Complex64;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    algorithms::{AlgorithmIterationMetrics, SpectralReconstructionAlgorithm, StepOutput},
    array_layout::{StandardArray2, checked_len_2d},
    backend::{Backend, CpuBackend, FftDirection},
    complex,
    error::Error,
    measurements::MeasurementRead,
    model::{
        ForwardModel, ForwardWorkspace, ObjectCoupling, Pupil, SpectralImagePlaneModel,
        fftshift_copy, ifftshift_copy,
    },
};

use super::{
    Batch, IterationRecord, ReconstructionTrace, RuntimeInfo, SpectralCheckpointOptions,
    SpectralReconstructionCheckpoint,
    spectral_checkpoint::{StoredSolver, fingerprints},
};

/// Validated pairing of scalar detector measurements and a compiled spectral model.
///
/// Measurements remain `(frame, row, column)` grayscale intensities. Membership
/// comes exclusively from the spectral plan, including for multiplexed frames.
#[derive(Clone, Debug)]
pub struct SpectralReconstructionProblem<M> {
    /// Resident or lazy scalar detector measurement provider.
    pub measurements: M,
    /// Compiled spectral kernels, common grid, and sparse detector composition.
    pub model: SpectralImagePlaneModel,
}

impl<M: MeasurementRead> SpectralReconstructionProblem<M> {
    /// Owns and validates measurements and their compiled spectral model.
    pub fn new(measurements: M, model: SpectralImagePlaneModel) -> Result<Self> {
        let problem = Self {
            measurements,
            model,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Checks shapes/counts and positive-measurement-weight coverage of every local row.
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.measurements.validate()?;
        if self.measurements.frame_count() != self.model.frame_count()
            || self.measurements.image_shape() != self.model.image_shape()
        {
            return Err(Error::InvalidMeasurements(
                "spectral detector frame count and shape must match the compiled model".into(),
            ));
        }
        let mut coverage: Vec<Vec<bool>> = self
            .model
            .channels()
            .iter()
            .map(|channel| vec![false; channel.model.frame_count()])
            .collect();
        for (frame, row) in self.model.acquisition().frames().iter().enumerate() {
            if self.measurements.frame_weight(frame)? == 0.0 {
                continue;
            }
            if self
                .measurements
                .frame_mask(frame)?
                .is_some_and(|mask| mask.iter().all(|&value| value == 0))
            {
                return Err(Error::InvalidMeasurements(format!(
                    "positive-weight frame {frame} has no unmasked pixels"
                )));
            }
            for contribution in &row.contributions {
                coverage[contribution.channel][contribution.local_frame] = true;
            }
        }
        if coverage.iter().flatten().any(|&covered| !covered) {
            return Err(Error::InvalidMeasurements("every spectral channel and local frame must participate in a positive-weight detector frame".into()));
        }
        Ok(())
    }
}

pub(crate) struct SpectralMode {
    pub(crate) channel: usize,
    pub(crate) source: usize,
    pub(crate) weight: f64,
    pub(crate) field: Vec<Complex64>,
    pub(crate) exit: Vec<Complex64>,
}

/// Mutable spectral object state with scratch storage for only the current exposure.
///
/// Independent coupling stores one centered spectrum per channel; shared
/// coupling stores exactly one. Pupils and calibration stay fixed in the model.
pub struct SpectralReconstructionState {
    pub(crate) spectra: Vec<StandardArray2<Complex64>>,
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) workspace: ForwardWorkspace,
    pub(crate) modes: Vec<SpectralMode>,
    pub(crate) predicted: Vec<f64>,
    pub(crate) projection: Vec<Complex64>,
    pub(crate) field: Vec<Complex64>,
    pub(crate) centered: Vec<Complex64>,
    pub(crate) update: Vec<Complex64>,
    pub(crate) column: Vec<Complex64>,
    pub(crate) effective_gain: f64,
}

impl SpectralReconstructionState {
    /// Initializes real non-negative objects from weighted, mask-aware measured amplitudes.
    ///
    /// Separate frames reproduce ordinary per-channel initialization. For mixed
    /// frames the starting amplitude splits measured signal equally per unit
    /// spectral/local-gain weight; this is an initialization heuristic, not
    /// spectral demixing. Fully unseen pixels use the weighted amplitude mean.
    pub fn initialize<M: MeasurementRead>(
        problem: &SpectralReconstructionProblem<M>,
    ) -> Result<Self> {
        problem.validate()?;
        let low_shape = problem.model.image_shape();
        let high_shape = problem.model.reconstruction_shape();
        let low_len = checked_len_2d(low_shape)?;
        let mut objects = Vec::new();
        for object_index in 0..problem.model.object_count() {
            let mut amplitude = vec![0.0; low_len];
            let mut weights = vec![0.0; low_len];
            let mut total_amplitude = 0.0;
            let mut total_weight = 0.0;
            for (frame, row) in problem.model.acquisition().frames().iter().enumerate() {
                if !row
                    .contributions
                    .iter()
                    .any(|c| problem.model.object_index(c.channel).ok() == Some(object_index))
                {
                    continue;
                }
                let frame_weight = problem.measurements.frame_weight(frame)?;
                if frame_weight == 0.0 {
                    continue;
                }
                let gain = row
                    .contributions
                    .iter()
                    .map(|c| {
                        Ok(c.spectral_weight
                            * problem.model.channels()[c.channel]
                                .model
                                .frame_gain(c.local_frame)?)
                    })
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .sum::<f64>()
                    * row.gain;
                if !gain.is_finite() || gain <= 0.0 {
                    return Err(Error::Numerical(
                        "spectral initialization gain must be finite and positive".into(),
                    ));
                }
                let measured = problem.measurements.frame(frame)?;
                let mask = problem.measurements.frame_mask(frame)?;
                for pixel in 0..low_len {
                    if mask.is_some_and(|values| values[pixel] == 0) {
                        continue;
                    }
                    let value = ((measured[pixel] - row.background) / gain).max(0.0).sqrt();
                    amplitude[pixel] += frame_weight * value;
                    weights[pixel] += frame_weight;
                    total_amplitude += frame_weight * value;
                    total_weight += frame_weight;
                }
            }
            let fallback = total_amplitude / total_weight;
            for (value, &weight) in amplitude.iter_mut().zip(&weights) {
                *value = if weight > 0.0 {
                    *value / weight
                } else {
                    fallback
                };
            }
            let object = Array2::from_shape_fn(high_shape, |(row, column)| {
                Complex64::new(
                    amplitude[(row * low_shape.0 / high_shape.0) * low_shape.1
                        + column * low_shape.1 / high_shape.1],
                    0.0,
                )
            });
            objects.push(object);
        }
        Self::from_objects(problem, objects)
    }

    /// Initializes from owned common-grid complex objects in stored-object order.
    ///
    /// Inputs must be finite standard row-major arrays. Independent coupling
    /// requires one object per channel; shared coupling requires exactly one.
    pub fn from_objects<M: MeasurementRead>(
        problem: &SpectralReconstructionProblem<M>,
        objects: Vec<Array2<Complex64>>,
    ) -> Result<Self> {
        problem.validate()?;
        let views: Vec<_> = objects.iter().map(|object| object.view()).collect();
        problem.model.validate_spectra(&views)?;
        let low_shape = problem.model.image_shape();
        let high_shape = problem.model.reconstruction_shape();
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(low_shape, high_shape)?);
        let mut column = vec![Complex64::default(); low_shape.0.max(high_shape.0)];
        let mut spectra = Vec::new();
        for object in objects {
            let mut object = StandardArray2::try_from(object)?;
            backend.fft2(
                object.as_slice_mut(),
                high_shape,
                FftDirection::Forward,
                &mut column,
            )?;
            let mut centered = vec![Complex64::default(); object.len()];
            fftshift_copy(object.as_slice(), &mut centered, high_shape);
            spectra.push(StandardArray2::from_shape_vec(high_shape, centered)?);
        }
        let workspace =
            ForwardModel::with_backend(&problem.model.channels()[0].model, backend.clone())?
                .workspace()?;
        let len = checked_len_2d(low_shape)?;
        Ok(Self {
            spectra,
            backend,
            workspace,
            modes: Vec::new(),
            predicted: vec![0.0; len],
            projection: vec![Complex64::default(); len],
            field: vec![Complex64::default(); len],
            centered: vec![Complex64::default(); len],
            update: vec![Complex64::default(); len],
            column,
            effective_gain: 1.0,
        })
    }

    /// Borrows stored centered spectra, one per independent object or one when shared.
    pub fn object_spectra(&self) -> Vec<ArrayView2<'_, Complex64>> {
        self.spectra
            .iter()
            .map(|array| array.ndarray_view())
            .collect()
    }

    /// Mutably borrows one stored centered spectrum by zero-based object index.
    /// Shared coupling has only index zero; out-of-range indices return an error.
    pub fn object_spectrum_mut(&mut self, index: usize) -> Result<ArrayViewMut2<'_, Complex64>> {
        self.spectra
            .get_mut(index)
            .map(|spectrum| spectrum.ndarray_view_mut())
            .ok_or(Error::InvalidParameter {
                name: "object_index",
                reason: "stored object index is out of range".into(),
            })
    }

    /// Evaluates every coherent mode from the same pre-frame object state.
    pub(crate) fn evaluate_frame(
        &mut self,
        model: &SpectralImagePlaneModel,
        frame: usize,
    ) -> Result<()> {
        let row = &model.acquisition().frames()[frame];
        self.modes.clear();
        self.predicted.fill(0.0);
        self.effective_gain = row.gain;
        let separate = row.contributions.len() == 1;
        if separate {
            let contribution = row.contributions[0];
            self.effective_gain *= contribution.spectral_weight
                * model.channels()[contribution.channel]
                    .model
                    .frame_gain(contribution.local_frame)?;
        }
        for contribution in &row.contributions {
            let kernel = &model.channels()[contribution.channel].model;
            let object = self.spectra[model.object_index(contribution.channel)?].view();
            let single = [(contribution.local_frame, 1.0)];
            let sources = kernel
                .multiplexing_matrix()
                .map_or(single.as_slice(), |matrix| {
                    matrix[contribution.local_frame].as_slice()
                });
            let forward = ForwardModel::with_backend(kernel, self.backend.clone())?;
            for &(source, source_weight) in sources {
                let weight = if separate {
                    source_weight
                } else {
                    contribution.spectral_weight
                        * kernel.frame_gain(contribution.local_frame)?
                        * source_weight
                };
                forward.forward_source_field_standard_into(
                    object,
                    kernel.pupil(),
                    source,
                    &mut self.workspace,
                )?;
                let field = self.workspace.field().to_vec();
                for (prediction, value) in self.predicted.iter_mut().zip(&field) {
                    *prediction += weight * value.norm_sqr();
                }
                let mut exit = vec![Complex64::default(); field.len()];
                kernel.extract_patch_standard(object, source, &mut exit)?;
                for (value, pupil) in exit.iter_mut().zip(kernel.pupil().values()) {
                    *value *= pupil;
                }
                self.modes.push(SpectralMode {
                    channel: contribution.channel,
                    source,
                    weight,
                    field,
                    exit,
                });
            }
        }
        Ok(())
    }
}

/// Ordering of physical detector exposures for a spectral solver.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectralFrameSchedule {
    /// Visit exposures in declared acquisition order.
    #[default]
    Sequential,
    /// Shuffle once per iteration using the same seed rule as ordinary reconstruction.
    RandomShuffle {
        /// Base random seed combined with the zero-based iteration number.
        seed: u64,
    },
    /// An explicit permutation of every detector row, reused each iteration.
    Explicit(Vec<usize>),
}

impl SpectralFrameSchedule {
    /// Returns a permutation after validating explicit orders; never reorders channel modes.
    pub fn order(&self, frame_count: usize, iteration: usize) -> Result<Vec<usize>> {
        let mut order: Vec<_> = (0..frame_count).collect();
        match self {
            Self::Sequential => {}
            Self::RandomShuffle { seed } => order.shuffle(&mut StdRng::seed_from_u64(
                seed.wrapping_add((iteration as u64).wrapping_mul(0x9e3779b97f4a7c15)),
            )),
            Self::Explicit(values) => {
                let mut sorted = values.clone();
                sorted.sort_unstable();
                if sorted != order {
                    return Err(Error::InvalidParameter {
                        name: "spectral_schedule",
                        reason: "explicit order must be a permutation of all detector frames"
                            .into(),
                    });
                }
                order.clone_from(values);
            }
        }
        Ok(order)
    }
}

/// Reconstructed channel metadata and owned arrays, kept together as one record.
#[derive(Clone, Debug)]
pub struct SpectralChannelResult {
    /// Stable channel identifier.
    pub channel_id: String,
    /// Channel's vacuum wavelength in metres.
    pub wavelength_vacuum_m: f64,
    /// Common-grid thin-sample complex transmission.
    pub object: Array2<Complex64>,
    /// Non-negative magnitude of the complex transmission.
    pub amplitude: Array2<f64>,
    /// Wrapped transmission phase in radians in `[-π, π]`.
    pub phase: Array2<f64>,
    /// Centered spectrum of the reported complex object.
    pub object_spectrum: Array2<Complex64>,
    /// Channel-specific fixed sampled pupil, owned independently of other records.
    pub pupil: Pupil,
}

/// Channel-ordered spectral reconstruction output and algorithm-neutral trace.
///
/// Unconstrained spectral AP has independent phase pistons for independent
/// fields, or one for shared-complex coupling. Shared-complex records contain
/// identical final objects. Joint OPD descent additionally constrains channel
/// phases via one referenced OPD; `object_coupling` still describes the compiled
/// model's field storage, not that solver's real-parameter constraints.
#[derive(Clone, Debug)]
pub struct SpectralReconstructionResult {
    /// Ordered channel records pairing IDs/wavelengths with scientific arrays.
    pub channels: Vec<SpectralChannelResult>,
    /// Compiled model's complex-field storage/reuse rule.
    pub object_coupling: ObjectCoupling,
    /// Weighted, mask-aware objectives and elapsed times. Spectral AP records
    /// pre-update pass objectives; joint OPD descent records an iteration-zero
    /// initial objective followed by post-update full-data objectives.
    pub trace: ReconstructionTrace,
    /// Algorithm name, completed iterations, and elapsed seconds.
    pub runtime: RuntimeInfo,
    /// Final iteration-boundary state for supported stateless solvers. Joint
    /// descent keeps its resumable state in the enclosing OPD result instead.
    pub checkpoint: Option<SpectralReconstructionCheckpoint>,
}

/// Executor for a spectral algorithm, distinct from ordinary reconstruction.
///
/// Global detector order is preserved even with batching; scratch fields are
/// released after each exposure. Existing single-object algorithms and their
/// calibration/pupil-recovery contracts cannot be passed to this runner.
pub struct SpectralRunner<A> {
    algorithm: A,
    schedule: SpectralFrameSchedule,
    initial_objects: Option<Vec<Array2<Complex64>>>,
    checkpoint: Option<SpectralReconstructionCheckpoint>,
    checkpoint_options: SpectralCheckpointOptions,
    explicit_schedule: bool,
}

impl<A: SpectralReconstructionAlgorithm> SpectralRunner<A> {
    /// Reconstructs independent wavelength fields, then mixes referenced phases into OPD.
    ///
    /// Assumes nondispersive OPD and registered, resolution-matched fields; this
    /// convenience performs the same operations as `run` followed by
    /// [`SpectralReconstructionResult::unwrap_opd`]. Pupils and calibration stay fixed.
    /// Shared-complex coupling is rejected before reconstruction.
    pub fn run_opd<M: MeasurementRead>(
        self,
        problem: &SpectralReconstructionProblem<M>,
        unwrapper: &super::SyntheticWavelengthUnwrapper,
        reference: &super::PhaseReference,
        mask: Option<ArrayView2<'_, u8>>,
    ) -> Result<super::MultiWavelengthReconstructionResult> {
        if problem.model.object_coupling() != ObjectCoupling::Independent {
            return Err(Error::InvalidParameter {
                name: "object_coupling",
                reason: "OPD phase mixing requires independent wavelength fields".into(),
            });
        }
        unwrapper.validate()?;
        let spectral = self.run(problem)?;
        let opd = spectral.unwrap_opd(unwrapper, reference, mask)?;
        Ok(super::MultiWavelengthReconstructionResult { spectral, opd })
    }

    /// Creates an executor using sequential detector order and amplitude initialization.
    pub fn new(algorithm: A) -> Self {
        Self {
            algorithm,
            schedule: SpectralFrameSchedule::default(),
            initial_objects: None,
            checkpoint: None,
            checkpoint_options: SpectralCheckpointOptions::default(),
            explicit_schedule: false,
        }
    }

    /// Selects physical detector ordering; channel/local-frame order stays canonical.
    pub fn with_schedule(mut self, schedule: SpectralFrameSchedule) -> Self {
        self.schedule = schedule;
        self.explicit_schedule = true;
        self
    }

    /// Sets owned initial complex fields in stored-object order, validated when run.
    pub fn with_initial_objects(mut self, objects: Vec<Array2<Complex64>>) -> Self {
        self.initial_objects = Some(objects);
        self
    }

    /// Restores a validated spectral solver snapshot. Initial objects cannot
    /// also be supplied. Its schedule is inherited unless explicitly selected,
    /// in which case the schedule must match. `iterations` is the total target.
    pub fn with_checkpoint(mut self, checkpoint: SpectralReconstructionCheckpoint) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Configures periodic JSON files after complete passes and at completion.
    pub fn with_checkpoint_options(mut self, options: SpectralCheckpointOptions) -> Self {
        self.checkpoint_options = options;
        self
    }

    /// Validates and streams one exposure at a time, returning owned channel records.
    pub fn run<M: MeasurementRead>(
        mut self,
        problem: &SpectralReconstructionProblem<M>,
    ) -> Result<SpectralReconstructionResult> {
        self.algorithm.validate()?;
        problem.validate()?;
        self.algorithm.validate_problem(problem)?;
        if self.algorithm.iterations() == 0 || self.algorithm.batch_size() == 0 {
            return Err(Error::InvalidParameter {
                name: "iterations/batch_size",
                reason: "must be positive".into(),
            });
        }
        self.checkpoint_options.validate()?;
        let configuration = self.algorithm.checkpoint_configuration();
        let started = Instant::now();
        let algorithm_name = std::any::type_name::<A>().to_string();
        let (mut state, mut trace, completed, elapsed_offset, signature) = match self.checkpoint {
            Some(checkpoint) => {
                if self.initial_objects.is_some() {
                    return Err(Error::InvalidParameter {
                        name: "initial_objects",
                        reason: "cannot combine initial objects with a checkpoint".into(),
                    });
                }
                checkpoint.validate_for_problem(problem)?;
                let signature = checkpoint.fingerprints();
                let StoredSolver::Spectral {
                    algorithm,
                    configuration: stored_configuration,
                    spectra,
                    schedule,
                } = checkpoint.state
                else {
                    return Err(Error::InvalidParameter {
                        name: "resume_from",
                        reason: "joint OPD state cannot resume spectral AP".into(),
                    });
                };
                if algorithm != algorithm_name
                    || configuration.as_ref() != Some(&stored_configuration)
                    || (self.explicit_schedule && self.schedule != schedule)
                {
                    return Err(Error::InvalidParameter {
                        name: "resume_from",
                        reason: "solver stepping options or schedule changed".into(),
                    });
                }
                if self.algorithm.iterations() < checkpoint.completed_iterations {
                    return Err(Error::InvalidParameter {
                        name: "iterations",
                        reason: "total target cannot precede checkpoint iterations".into(),
                    });
                }
                self.schedule = schedule;
                let objects = (0..problem.model.object_count())
                    .map(|_| Array2::zeros(problem.model.reconstruction_shape()))
                    .collect();
                let mut state = SpectralReconstructionState::from_objects(problem, objects)?;
                for (target, stored) in state.spectra.iter_mut().zip(spectra) {
                    *target = StandardArray2::try_from(stored.into_array()?)?;
                }
                (
                    state,
                    checkpoint.trace,
                    checkpoint.completed_iterations,
                    checkpoint.elapsed_seconds,
                    Some(signature),
                )
            }
            None => {
                let signature = configuration
                    .as_ref()
                    .map(|_| fingerprints(problem))
                    .transpose()?;
                let state = match self.initial_objects {
                    Some(objects) => SpectralReconstructionState::from_objects(problem, objects)?,
                    None => self.algorithm.initialize(problem)?,
                };
                (state, ReconstructionTrace::default(), 0, 0.0, signature)
            }
        };
        if self.checkpoint_options.directory.is_some() && configuration.is_none() {
            return Err(Error::InvalidParameter {
                name: "checkpoint_options",
                reason: "this custom spectral algorithm has not opted into stateless checkpointing"
                    .into(),
            });
        }
        self.schedule
            .order(problem.model.frame_count(), completed)?;
        for iteration in completed..self.algorithm.iterations() {
            let order = self
                .schedule
                .order(problem.model.frame_count(), iteration)?;
            let mut output = StepOutput::<A::IterationMetrics>::default();
            for (batch_index, indices) in order.chunks(self.algorithm.batch_size()).enumerate() {
                output.merge(self.algorithm.step(
                    problem,
                    &mut state,
                    &Batch {
                        indices: indices.to_vec(),
                        batch_index,
                    },
                    iteration,
                )?);
            }
            let objective = output.summary.mean_objective().ok_or_else(|| {
                Error::Numerical("spectral iteration has no positive frame weight".into())
            })?;
            if !objective.is_finite() {
                return Err(Error::Numerical(
                    "spectral iteration objective is non-finite".into(),
                ));
            }
            trace.iterations.push(IterationRecord {
                iteration: iteration + 1,
                objective,
                elapsed_seconds: elapsed_offset + started.elapsed().as_secs_f64(),
            });
            output
                .metrics
                .append_records(iteration + 1, &mut trace.algorithm_metrics);
            if self.checkpoint_options.directory.is_some()
                && (iteration + 1).is_multiple_of(self.checkpoint_options.every)
            {
                let checkpoint = SpectralReconstructionCheckpoint::capture(
                    problem.model.clone(),
                    signature.as_ref().unwrap(),
                    iteration + 1,
                    elapsed_offset + started.elapsed().as_secs_f64(),
                    trace.clone(),
                    StoredSolver::Spectral {
                        algorithm: algorithm_name.clone(),
                        configuration: configuration.clone().unwrap(),
                        spectra: state
                            .spectra
                            .iter()
                            .map(|a| crate::array_serde::Array2Data::from_view(a.ndarray_view()))
                            .collect(),
                        schedule: self.schedule.clone(),
                    },
                );
                self.checkpoint_options.write(&checkpoint, false)?;
            }
        }
        let high_shape = problem.model.reconstruction_shape();
        let mut objects = Vec::new();
        for spectrum in &state.spectra {
            let mut object = vec![Complex64::default(); spectrum.len()];
            ifftshift_copy(spectrum.as_slice(), &mut object, high_shape);
            state.backend.fft2(
                &mut object,
                high_shape,
                FftDirection::Inverse,
                &mut state.column,
            )?;
            objects.push(Array2::from_shape_vec(high_shape, object)?);
        }
        let channels = problem
            .model
            .channels()
            .iter()
            .enumerate()
            .map(|(index, channel)| {
                let object_index = problem.model.object_index(index)?;
                let object = objects[object_index].clone();
                Ok(SpectralChannelResult {
                    channel_id: channel.channel_id.clone(),
                    wavelength_vacuum_m: channel.model.sampling().wavelength.unwrap(),
                    amplitude: complex::amplitude(object.view()),
                    phase: complex::phase(object.view()),
                    object,
                    object_spectrum: state.spectra[object_index].clone().into_inner(),
                    pupil: channel.model.pupil().clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let elapsed_seconds = elapsed_offset + started.elapsed().as_secs_f64();
        let checkpoint = configuration.map(|configuration| {
            SpectralReconstructionCheckpoint::capture(
                problem.model.clone(),
                signature.as_ref().unwrap(),
                self.algorithm.iterations(),
                elapsed_seconds,
                trace.clone(),
                StoredSolver::Spectral {
                    algorithm: algorithm_name,
                    configuration,
                    spectra: state
                        .spectra
                        .iter()
                        .map(|a| crate::array_serde::Array2Data::from_view(a.ndarray_view()))
                        .collect(),
                    schedule: self.schedule,
                },
            )
        });
        if let Some(checkpoint) = &checkpoint {
            self.checkpoint_options.write(checkpoint, true)?;
        }
        Ok(SpectralReconstructionResult {
            channels,
            checkpoint,
            object_coupling: problem.model.object_coupling(),
            trace,
            runtime: RuntimeInfo {
                elapsed_seconds,
                completed_iterations: self.algorithm.iterations(),
                stopped_early: false,
                algorithm: std::any::type_name::<A>()
                    .rsplit("::")
                    .next()
                    .unwrap_or("spectral algorithm")
                    .into(),
            },
        })
    }
}
