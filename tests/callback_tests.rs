mod common;

use std::sync::{Arc, Mutex, atomic::Ordering};

use approx::assert_abs_diff_eq;
use fpm_rs::{
    Result,
    algorithms::{AlternatingProjection, ReconstructionAlgorithm},
    callbacks::CheckpointEvery,
    callbacks::{
        Callback, CallbackAction, CallbackHook, CsvLogger, SaveImageEvery, SaveResidualsEvery,
        StepContext, StopOnPlateau,
    },
    diagnostics::DiagnosticRequest,
    measurements::MeasurementStack,
    reconstruction::{
        ReconstructionCheckpoint, ReconstructionProblem, ReconstructionResult, RunOptions, Runner,
    },
    simulation::{Simulator, SyntheticObject},
};

struct Recorder {
    events: Arc<Mutex<Vec<String>>>,
}

impl Callback for Recorder {
    fn on_start(&mut self, _context: &StepContext<'_>) -> Result<CallbackAction> {
        self.events.lock().unwrap().push("start".into());
        Ok(CallbackAction::Continue)
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        self.events
            .lock()
            .unwrap()
            .push(format!("iteration-{}", context.iteration));
        Ok(CallbackAction::Continue)
    }

    fn on_finish(&mut self, _result: &ReconstructionResult) -> Result<()> {
        self.events.lock().unwrap().push("finish".into());
        Ok(())
    }
}

struct PeriodicDiagnosticProbe {
    observations: Arc<Mutex<Vec<(usize, bool)>>>,
}

struct StartResidualProbe;

type FrameObservation = (usize, usize, usize, bool);

struct FrameRecorder {
    observations: Arc<Mutex<Vec<FrameObservation>>>,
}

impl Callback for FrameRecorder {
    fn requires_for(&self, hook: CallbackHook, _iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::FrameEnd {
            vec![DiagnosticRequest::Objective]
        } else {
            Vec::new()
        }
    }

    fn on_frame_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        self.observations.lock().unwrap().push((
            context.iteration,
            context.frame_index.unwrap(),
            context.batch_index.unwrap(),
            context.diagnostics.objective.is_some(),
        ));
        Ok(CallbackAction::Continue)
    }
}

impl Callback for StartResidualProbe {
    fn requires_for(&self, hook: CallbackHook, _iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::Start {
            vec![
                DiagnosticRequest::ResidualImages,
                DiagnosticRequest::PerFrameError,
            ]
        } else {
            Vec::new()
        }
    }

    fn on_start(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        assert_eq!(
            context.diagnostics.residual_images.as_ref().map(Vec::len),
            Some(context.model.frame_count())
        );
        assert_eq!(
            context
                .diagnostics
                .per_frame_objective
                .as_ref()
                .map(Vec::len),
            Some(context.model.frame_count())
        );
        Ok(CallbackAction::Stop)
    }
}

impl Callback for PeriodicDiagnosticProbe {
    fn requires(&self) -> Vec<DiagnosticRequest> {
        vec![DiagnosticRequest::ObjectAmplitude]
    }

    fn requires_for(&self, hook: CallbackHook, iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::IterationEnd && iteration.is_multiple_of(2) {
            self.requires()
        } else {
            Vec::new()
        }
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> Result<CallbackAction> {
        self.observations.lock().unwrap().push((
            context.iteration,
            context.diagnostics.object_amplitude.is_some(),
        ));
        Ok(CallbackAction::Continue)
    }
}

fn problem() -> ReconstructionProblem<MeasurementStack> {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((16, 16)).unwrap())
        .simulate()
        .unwrap();
    ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model).unwrap()
}

#[test]
fn callbacks_execute_in_lifecycle_order() {
    let events = Arc::new(Mutex::new(Vec::new()));
    AlternatingProjection::default()
        .iterations(3)
        .run_with_callbacks(
            &problem(),
            vec![Box::new(Recorder {
                events: events.clone(),
            })],
        )
        .unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        [
            "start",
            "iteration-1",
            "iteration-2",
            "iteration-3",
            "finish"
        ]
    );
}

#[test]
fn periodic_callbacks_only_compute_diagnostics_when_active() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    AlternatingProjection::default()
        .iterations(4)
        .run_with_callbacks(
            &problem(),
            vec![Box::new(PeriodicDiagnosticProbe {
                observations: observations.clone(),
            })],
        )
        .unwrap();
    assert_eq!(
        *observations.lock().unwrap(),
        [(1, false), (2, true), (3, false), (4, true)]
    );
}

#[test]
fn frame_callbacks_fire_once_per_frame_across_batches() {
    let observations = Arc::new(Mutex::new(Vec::new()));
    Runner::new(
        AlternatingProjection::default(),
        RunOptions {
            max_iterations: 1,
            batch_size: 2,
            enable_frame_callbacks: true,
            ..RunOptions::default()
        },
    )
    .with_callback(Box::new(FrameRecorder {
        observations: observations.clone(),
    }))
    .run(&problem())
    .unwrap();
    assert_eq!(
        *observations.lock().unwrap(),
        [
            (1, 0, 0, true),
            (1, 1, 0, true),
            (1, 2, 1, true),
            (1, 3, 1, true),
            (1, 4, 2, true),
        ]
    );
}

#[test]
fn callback_forward_diagnostics_use_the_injected_backend() {
    let problem = problem();
    let frame_count = problem.model.frame_count();
    let (backend, calls) = common::CountingBackend::new(
        problem.model.image_shape(),
        problem.model.reconstruction_shape(),
    )
    .unwrap();
    let result = Runner::new(
        AlternatingProjection::default(),
        RunOptions {
            max_iterations: 1,
            ..RunOptions::default()
        },
    )
    .with_backend(backend)
    .with_callback(Box::new(StartResidualProbe))
    .run(&problem)
    .unwrap();

    assert!(result.runtime.stopped_early);
    assert_eq!(result.runtime.completed_iterations, 0);
    // Initialization and result construction each transform the high-resolution
    // object. Residual images and per-frame errors share one prediction pass.
    assert_eq!(calls.load(Ordering::Relaxed), frame_count + 2);
}

#[test]
fn file_callbacks_obey_frequency_and_log_rows() {
    let directory = tempfile::tempdir().unwrap();
    let image_directory = directory.path().join("images");
    let csv_path = directory.path().join("objective.csv");
    AlternatingProjection::default()
        .iterations(3)
        .run_with_callbacks(
            &problem(),
            vec![
                Box::new(SaveImageEvery::new(2, &image_directory)),
                Box::new(CsvLogger::new(&csv_path)),
            ],
        )
        .unwrap();
    assert!(image_directory.join("amplitude_00002.png").exists());
    assert!(image_directory.join("phase_00002.png").exists());
    assert!(!image_directory.join("amplitude_00001.png").exists());
    let rows = std::fs::read_to_string(csv_path).unwrap();
    assert_eq!(rows.lines().count(), 4);
}

#[test]
fn residual_callback_only_computes_and_saves_at_its_frequency() {
    let problem = problem();
    let (baseline_backend, baseline_calls) = common::CountingBackend::new(
        problem.model.image_shape(),
        problem.model.reconstruction_shape(),
    )
    .unwrap();
    Runner::new(
        AlternatingProjection::default(),
        RunOptions {
            max_iterations: 3,
            ..RunOptions::default()
        },
    )
    .with_backend(baseline_backend)
    .run(&problem)
    .unwrap();

    let directory = tempfile::tempdir().unwrap();
    let residual_directory = directory.path().join("residuals");
    let (callback_backend, callback_calls) = common::CountingBackend::new(
        problem.model.image_shape(),
        problem.model.reconstruction_shape(),
    )
    .unwrap();
    Runner::new(
        AlternatingProjection::default(),
        RunOptions {
            max_iterations: 3,
            ..RunOptions::default()
        },
    )
    .with_backend(callback_backend)
    .with_callback(Box::new(SaveResidualsEvery::new(2, &residual_directory)))
    .run(&problem)
    .unwrap();

    assert_eq!(
        callback_calls.load(Ordering::Relaxed) - baseline_calls.load(Ordering::Relaxed),
        problem.model.frame_count()
    );
    for frame in 0..problem.model.frame_count() {
        assert!(
            residual_directory
                .join(format!("residual_00002_frame_{frame:05}.png"))
                .exists()
        );
        assert!(
            !residual_directory
                .join(format!("residual_00001_frame_{frame:05}.png"))
                .exists()
        );
        assert!(
            !residual_directory
                .join(format!("residual_00003_frame_{frame:05}.png"))
                .exists()
        );
    }
}

#[test]
fn plateau_callback_stops_before_iteration_limit() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .simulate()
        .unwrap();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
            .unwrap();
    let result = AlternatingProjection::default()
        .iterations(20)
        .run_with_callbacks(&problem, vec![Box::new(StopOnPlateau::new(2, 1e-12))])
        .unwrap();
    assert!(result.runtime.stopped_early);
    assert!(result.runtime.completed_iterations < 20);
}

#[test]
fn checkpoint_round_trip_resumes_exactly() {
    let problem = problem();
    let directory = tempfile::tempdir().unwrap();
    AlternatingProjection::default()
        .iterations(2)
        .run_with_callbacks(
            &problem,
            vec![Box::new(CheckpointEvery::new(2, directory.path()))],
        )
        .unwrap();
    let checkpoint =
        ReconstructionCheckpoint::load(directory.path().join("checkpoint_00002.json")).unwrap();
    assert_eq!(checkpoint.completed_iterations(), 2);
    assert_eq!(checkpoint.trace().iterations.len(), 2);

    let resumed = AlternatingProjection::default()
        .iterations(4)
        .run_from_checkpoint(&problem, checkpoint)
        .unwrap();
    let uninterrupted = AlternatingProjection::default()
        .iterations(4)
        .run(&problem)
        .unwrap();
    for (&resumed, &uninterrupted) in resumed
        .object_spectrum
        .iter()
        .zip(uninterrupted.object_spectrum.iter())
    {
        assert_abs_diff_eq!(resumed.re, uninterrupted.re, epsilon = 1e-14);
        assert_abs_diff_eq!(resumed.im, uninterrupted.im, epsilon = 1e-14);
    }
    assert_eq!(
        resumed.recovered_pupil.values(),
        uninterrupted.recovered_pupil.values()
    );
    assert_eq!(resumed.runtime.completed_iterations, 4);
    assert_eq!(resumed.trace.iterations.len(), 4);
}

#[test]
fn checkpoint_io_rejects_corruption_and_problem_mismatch_early() {
    let problem = problem();
    let state = fpm_rs::reconstruction::ReconstructionState::initialize(&problem).unwrap();
    let checkpoint = ReconstructionCheckpoint::capture(
        0,
        &state,
        &fpm_rs::reconstruction::ReconstructionTrace::default(),
    );
    let directory = tempfile::tempdir().unwrap();

    let invalid_path = directory.path().join("invalid.json");
    let mut invalid = serde_json::to_value(&checkpoint).unwrap();
    invalid["completed_iterations"] = serde_json::json!(1);
    std::fs::write(&invalid_path, serde_json::to_vec(&invalid).unwrap()).unwrap();
    assert!(ReconstructionCheckpoint::load(&invalid_path).is_err());

    let valid_path = directory.path().join("valid.json");
    checkpoint.save(&valid_path).unwrap();
    ReconstructionCheckpoint::load_for_problem(&valid_path, &problem).unwrap();

    let support_mismatch_path = directory.path().join("support_mismatch.json");
    let mut support_mismatch = serde_json::to_value(&checkpoint).unwrap();
    support_mismatch["pupil"]["support"]["data"][0] = serde_json::json!(0);
    std::fs::write(
        &support_mismatch_path,
        serde_json::to_vec(&support_mismatch).unwrap(),
    )
    .unwrap();
    let support_mismatch = ReconstructionCheckpoint::load(&support_mismatch_path).unwrap();
    assert!(support_mismatch.validate_for_problem(&problem).is_err());

    let corrupted_path = directory.path().join("corrupted.json");
    let mut serialized = serde_json::to_value(&checkpoint).unwrap();
    serialized["format_version"] = serde_json::json!(999);
    std::fs::write(&corrupted_path, serde_json::to_vec(&serialized).unwrap()).unwrap();
    assert!(ReconstructionCheckpoint::load(&corrupted_path).is_err());

    let incompatible_path = directory.path().join("incompatible.json");
    let mut incompatible = serde_json::to_value(&checkpoint).unwrap();
    incompatible["object_spectrum"]["height"] = serde_json::json!(2);
    incompatible["object_spectrum"]["width"] = serde_json::json!(2);
    incompatible["object_spectrum"]["data"] = serde_json::json!([
        {"re": 0.0, "im": 0.0},
        {"re": 0.0, "im": 0.0},
        {"re": 0.0, "im": 0.0},
        {"re": 0.0, "im": 0.0}
    ]);
    std::fs::write(
        &incompatible_path,
        serde_json::to_vec(&incompatible).unwrap(),
    )
    .unwrap();
    assert!(ReconstructionCheckpoint::load_for_problem(&incompatible_path, &problem).is_err());
}
