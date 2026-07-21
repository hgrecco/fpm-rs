mod common;

use std::sync::{Arc, Mutex};

use fpm_rs::{
    Result,
    algorithms::ReconstructionAlgorithm,
    diagnostics::StepDiagnostics,
    measurements::{FrameMetadata, MeasurementRead, MeasurementStack},
    reconstruction::{
        Batch, FrameSchedule, ReconstructionProblem, ReconstructionState, RunOptions, Runner,
    },
};

#[test]
fn brightfield_and_spiral_schedules_start_at_zero_angle() {
    let model = common::direct_model().unwrap();
    let brightfield = FrameSchedule::BrightfieldFirst.order(&model, 0);
    assert_eq!(brightfield[0], 2);

    let spiral = FrameSchedule::SpiralOut.order(&model, 0);
    assert_eq!(spiral, vec![2, 4, 3, 0, 1]);
}

#[test]
fn random_schedule_is_seeded_per_iteration() {
    let model = common::direct_model().unwrap();
    let schedule = FrameSchedule::RandomShuffle { seed: 42 };
    assert_eq!(schedule.order(&model, 3), schedule.order(&model, 3));
    assert_ne!(schedule.order(&model, 3), schedule.order(&model, 4));

    let mut sorted = schedule.order(&model, 3);
    sorted.sort_unstable();
    assert_eq!(sorted, (0..model.frame_count()).collect::<Vec<_>>());
}

#[test]
fn snr_schedule_uses_measurements_masks_weights_and_stable_ties() {
    let problem = weighted_problem();
    let schedule = FrameSchedule::SnrWeighted;

    assert_eq!(
        schedule.order(&problem.model, 0),
        (0..problem.model.frame_count()).collect::<Vec<_>>()
    );
    assert_eq!(
        schedule.order_for_problem(&problem, 0).unwrap(),
        vec![1, 0, 3, 4, 2]
    );
}

#[test]
fn runner_applies_measurement_aware_snr_order() {
    let problem = weighted_problem();
    let visited = Arc::new(Mutex::new(Vec::new()));
    let algorithm = RecordingAlgorithm {
        visited: visited.clone(),
    };
    Runner::new(
        algorithm,
        RunOptions {
            max_iterations: 1,
            batch_size: 2,
            schedule: FrameSchedule::SnrWeighted,
            ..RunOptions::default()
        },
    )
    .run(&problem)
    .unwrap();

    assert_eq!(*visited.lock().unwrap(), vec![1, 0, 3, 4, 2]);
}

fn weighted_problem() -> ReconstructionProblem<MeasurementStack> {
    let model = common::direct_model().unwrap();
    let frame_len = model.image_shape().0 * model.image_shape().1;
    let mut data = Vec::with_capacity(frame_len * model.frame_count());
    for level in [4.0, 9.0, 100.0, 16.0, 1.0] {
        data.extend(std::iter::repeat_n(level, frame_len));
    }
    // This outlier must not affect frame 1's score because it is masked.
    data[frame_len] = 1e12;
    let mut metadata: Vec<_> = (0..model.frame_count()).map(FrameMetadata::new).collect();
    metadata[2].weight = 0.0;
    metadata[3].weight = 0.5;
    let mut masks = vec![1; data.len()];
    masks[frame_len] = 0;
    masks[2 * frame_len..3 * frame_len].fill(0);
    let masks = ndarray::Array3::from_shape_vec(
        (
            model.frame_count(),
            model.image_shape().0,
            model.image_shape().1,
        ),
        masks,
    )
    .unwrap();
    let measurements = MeasurementStack::from_vec(data, model.image_shape(), metadata)
        .unwrap()
        .with_per_frame_masks(masks)
        .unwrap();
    ReconstructionProblem::new(measurements, model).unwrap()
}

struct RecordingAlgorithm {
    visited: Arc<Mutex<Vec<usize>>>,
}

impl ReconstructionAlgorithm for RecordingAlgorithm {
    fn step<M: MeasurementRead>(
        &mut self,
        problem: &ReconstructionProblem<M>,
        _state: &mut ReconstructionState,
        batch: &Batch,
        _iteration: usize,
    ) -> Result<StepDiagnostics> {
        self.visited.lock().unwrap().extend(&batch.indices);
        let mut diagnostics = StepDiagnostics::default();
        for &frame in &batch.indices {
            diagnostics.push_frame(frame, 0.0, problem.measurements.frame_weight(frame)?);
        }
        Ok(diagnostics)
    }

    fn iterations(&self) -> usize {
        1
    }
}
