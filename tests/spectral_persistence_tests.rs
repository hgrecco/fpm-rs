use fpm_rs::{
    Complex64,
    algorithms::{
        MultiWavelengthGradientDescent, SpectralAlternatingProjection,
        SpectralReconstructionAlgorithm,
    },
    experiment::{
        AcquisitionPlan, DirectionList, Optics, SourceCalibration, SpectralAcquisitionPlan,
        SpectralChannel, SpectralContribution, SpectralFrame, SpectralGeometry,
    },
    measurements::MeasurementStack,
    model::{ObjectCoupling, ReconstructionShape, SpectralImagePlaneModel},
    reconstruction::{
        PhaseReference, SpectralCheckpointOptions, SpectralFrameSchedule,
        SpectralReconstructionCheckpoint, SpectralReconstructionProblem, SpectralRunner,
        SyntheticWavelengthUnwrapper,
    },
};
use ndarray::Array2;
use std::f64::consts::TAU;

fn fixture(
    mixed: bool,
    shared: bool,
) -> (
    SpectralReconstructionProblem<MeasurementStack>,
    Array2<f64>,
    Vec<Array2<f64>>,
) {
    let channels = [500e-9, 550e-9]
        .iter()
        .enumerate()
        .map(|(i, &wavelength)| SpectralChannel {
            channel_id: format!("channel-{i}"),
            optics: Optics {
                wavelength_vacuum_m: wavelength,
                objective_na: 0.1,
                magnification: 4.0,
                camera_pixel_size: 6.5e-6,
                illumination_refractive_index: 1.0,
                objective_medium_refractive_index: 1.0,
                defocus_distance: None,
                pupil_aberration: None,
            },
            calibration: SourceCalibration::unity(),
            acquisition: AcquisitionPlan::all_sources(3).unwrap(),
        })
        .collect::<Vec<_>>();
    let geometry = SpectralGeometry::Shared(
        DirectionList::from_direction_cosines(vec![[0.0, 0.0], [0.08, 0.0], [0.0, 0.08]])
            .unwrap()
            .into(),
    );
    let plan = if mixed {
        SpectralAcquisitionPlan::multiplexed(
            (0..3)
                .map(|local_frame| SpectralFrame {
                    contributions: vec![
                        SpectralContribution {
                            channel: 0,
                            local_frame,
                            spectral_weight: 0.7,
                        },
                        SpectralContribution {
                            channel: 1,
                            local_frame,
                            spectral_weight: 1.2,
                        },
                    ],
                    gain: 1.3,
                    background: 0.04,
                })
                .collect(),
        )
        .unwrap()
    } else {
        SpectralAcquisitionPlan::separate(&[3, 3]).unwrap()
    };
    let coupling = if shared {
        ObjectCoupling::SharedComplex
    } else {
        ObjectCoupling::Independent
    };
    let model = SpectralImagePlaneModel::from_experiment(
        &channels,
        &geometry,
        plan,
        (6, 8),
        ReconstructionShape::Smooth,
        coupling,
    )
    .unwrap();
    let shape = model.reconstruction_shape();
    let opd = Array2::from_shape_fn(shape, |(r, c)| {
        1.2e-6
            + 8e-9 * (TAU * c as f64 / shape.1 as f64).cos()
            + 5e-9 * (TAU * r as f64 / shape.0 as f64).sin()
    });
    let amplitudes = (0..2)
        .map(|i| {
            Array2::from_shape_fn(shape, |(r, c)| {
                0.8 + 0.1 * i as f64 + 0.025 * (TAU * (r + c) as f64 / shape.1 as f64).cos()
            })
        })
        .collect::<Vec<_>>();
    let objects = if shared {
        vec![Array2::from_elem(shape, Complex64::new(0.9, 0.0))]
    } else {
        amplitudes
            .iter()
            .zip([500e-9, 550e-9])
            .map(|(a, w)| {
                Array2::from_shape_fn(shape, |p| Complex64::from_polar(a[p], TAU * opd[p] / w))
            })
            .collect()
    };
    let placeholder =
        MeasurementStack::from_vec(vec![1.0; model.frame_count() * 48], (6, 8), vec![]).unwrap();
    let problem = SpectralReconstructionProblem::new(placeholder, model.clone()).unwrap();
    let state =
        fpm_rs::reconstruction::SpectralReconstructionState::from_objects(&problem, objects)
            .unwrap();
    let mut data = Vec::new();
    for frame in 0..model.frame_count() {
        data.extend(
            model
                .forward_intensity(&state.object_spectra(), frame)
                .unwrap()
                .iter()
                .copied(),
        );
    }
    let measurements = MeasurementStack::from_vec(data, (6, 8), vec![]).unwrap();
    (
        SpectralReconstructionProblem::new(measurements, model).unwrap(),
        opd,
        amplitudes,
    )
}

fn losses(trace: &fpm_rs::reconstruction::ReconstructionTrace) -> Vec<(usize, f64)> {
    trace
        .iterations
        .iter()
        .map(|r| (r.iteration, r.objective))
        .collect()
}

#[test]
fn spectral_checkpoint_resumes_centered_spectra_and_seeded_schedule_exactly() {
    for (mixed, shared) in [(false, false), (true, false), (true, true)] {
        let (problem, _, _) = fixture(mixed, shared);
        let directory = tempfile::tempdir().unwrap();
        let algorithm = SpectralAlternatingProjection::default()
            .iterations(3)
            .batch_size(2);
        let partial = SpectralRunner::new(algorithm.clone())
            .with_schedule(SpectralFrameSchedule::RandomShuffle { seed: 19 })
            .with_checkpoint_options(SpectralCheckpointOptions {
                directory: Some(directory.path().into()),
                every: 2,
            })
            .run(&problem)
            .unwrap();
        assert!(
            directory
                .path()
                .join("spectral_checkpoint_00002.json")
                .is_file()
        );
        let checkpoint = SpectralReconstructionCheckpoint::load_for_problem(
            directory.path().join("spectral_checkpoint_00003.json"),
            &problem,
        )
        .unwrap();
        assert_eq!(checkpoint.algorithm(), "SpectralAlternatingProjection");
        let resumed = SpectralRunner::new(algorithm.clone().iterations(8))
            .with_checkpoint(checkpoint)
            .run(&problem)
            .unwrap();
        let whole = SpectralRunner::new(algorithm.iterations(8))
            .with_schedule(SpectralFrameSchedule::RandomShuffle { seed: 19 })
            .run(&problem)
            .unwrap();
        for (a, b) in resumed.channels.iter().zip(&whole.channels) {
            assert_eq!(a.object_spectrum, b.object_spectrum);
            assert_eq!(a.object, b.object);
            assert_eq!(a.amplitude, b.amplitude);
            assert_eq!(a.phase, b.phase);
        }
        assert_eq!(losses(&resumed.trace), losses(&whole.trace));
        assert_eq!(resumed.runtime.completed_iterations, 8);
        assert!(
            resumed.trace.iterations[3].elapsed_seconds
                >= partial.trace.iterations[2].elapsed_seconds
        );
    }
}

#[test]
fn joint_checkpoint_restores_the_original_region_or_mean_gauge_without_projection() {
    for mixed in [false, true] {
        for region in [false, true] {
            let (problem, opd, amplitudes) = fixture(mixed, false);
            let shape = opd.dim();
            let reference = if region {
                PhaseReference::Region {
                    mask: Array2::from_shape_fn(shape, |p| u8::from(p == (0, 0))),
                    opd_m: opd[(0, 0)],
                }
            } else {
                PhaseReference::Offsets(vec![0.0; 2])
            };
            let start = Array2::from_shape_fn(shape, |(r, c)| {
                opd[(r, c)] + 11e-9 * (TAU * (r + c) as f64 / shape.1 as f64).sin()
            });
            let amplitudes = amplitudes
                .iter()
                .map(|a| a.mapv(|v| v * 1.05))
                .collect::<Vec<_>>();
            let directory = tempfile::tempdir().unwrap();
            let algorithm = MultiWavelengthGradientDescent {
                iterations: 3,
                ..Default::default()
            };
            let partial = algorithm
                .run_from_opd_with_options(
                    &problem,
                    (0.0, 2e-6),
                    &reference,
                    start.clone(),
                    amplitudes.clone(),
                    &SpectralCheckpointOptions {
                        directory: Some(directory.path().into()),
                        every: 2,
                    },
                )
                .unwrap();
            let checkpoint = SpectralReconstructionCheckpoint::load_for_problem(
                directory.path().join("joint_opd_checkpoint_00003.json"),
                &problem,
            )
            .unwrap();
            assert_eq!(checkpoint.opd_m().unwrap(), partial.opd_m.view());
            let algorithm = MultiWavelengthGradientDescent {
                iterations: 8,
                ..algorithm
            };
            let resumed = algorithm.run_from_checkpoint(&problem, checkpoint).unwrap();
            let whole = algorithm
                .run_from_opd(&problem, (0.0, 2e-6), &reference, start, amplitudes)
                .unwrap();
            assert_eq!(resumed.opd_m, whole.opd_m);
            assert_eq!(
                losses(&resumed.spectral.trace),
                losses(&whole.spectral.trace)
            );
            for (a, b) in resumed
                .spectral
                .channels
                .iter()
                .zip(&whole.spectral.channels)
            {
                assert_eq!(a.object, b.object);
                assert_eq!(a.amplitude, b.amplitude);
                assert_eq!(a.object_spectrum, b.object_spectrum);
            }
        }
    }
}

#[test]
fn spectral_checkpoints_reject_changed_data_masks_options_schedules_and_solver_kinds() {
    let (problem, opd, amplitudes) = fixture(true, false);
    let partial = SpectralAlternatingProjection::default()
        .iterations(2)
        .run(&problem)
        .unwrap();
    let checkpoint = partial.checkpoint.unwrap();
    let algorithm = SpectralAlternatingProjection::default().iterations(4);
    assert!(
        SpectralRunner::new(algorithm.clone().object_step(0.5))
            .with_checkpoint(checkpoint.clone())
            .run(&problem)
            .is_err()
    );
    assert!(
        SpectralRunner::new(algorithm.clone())
            .with_schedule(SpectralFrameSchedule::RandomShuffle { seed: 2 })
            .with_checkpoint(checkpoint.clone())
            .run(&problem)
            .is_err()
    );
    let mut measurements = problem.measurements.clone();
    measurements.frame_mut(0).unwrap()[0] += 1.0;
    let changed = SpectralReconstructionProblem::new(measurements, problem.model.clone()).unwrap();
    assert!(checkpoint.validate_for_problem(&changed).is_err());
    let mut mask = Array2::from_elem((6, 8), 1_u8);
    mask[(0, 0)] = 0;
    let changed = SpectralReconstructionProblem::new(
        problem.measurements.clone().with_masks(mask).unwrap(),
        problem.model.clone(),
    )
    .unwrap();
    assert!(checkpoint.validate_for_problem(&changed).is_err());
    let reference = PhaseReference::Offsets(vec![0.0; 2]);
    let joint = MultiWavelengthGradientDescent {
        iterations: 2,
        ..Default::default()
    };
    assert!(
        joint
            .run_from_checkpoint(&problem, checkpoint.clone())
            .is_err()
    );
    let result = joint
        .run_from_opd(&problem, (0.0, 2e-6), &reference, opd, amplitudes)
        .unwrap();
    assert!(
        SpectralRunner::new(algorithm)
            .with_checkpoint(result.checkpoint.clone())
            .run(&problem)
            .is_err()
    );
    let mut changed_joint = joint.clone();
    changed_joint.iterations = 4;
    changed_joint.opd_step = 0.5;
    assert!(
        changed_joint
            .run_from_checkpoint(&problem, result.checkpoint)
            .is_err()
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("checkpoint.json");
    checkpoint.save(&path).unwrap();
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["model"]["channels"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(SpectralReconstructionCheckpoint::load(&path).is_err());
    value["kind"] = "ordinary_checkpoint".into();
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(SpectralReconstructionCheckpoint::load(&path).is_err());
}

#[test]
fn automatic_joint_checkpoint_keeps_phase_mixing_records_on_resume() {
    let channels = [500e-9, 550e-9]
        .iter()
        .enumerate()
        .map(|(i, &w)| SpectralChannel {
            channel_id: format!("c{i}"),
            optics: Optics {
                wavelength_vacuum_m: w,
                objective_na: 0.1,
                magnification: 4.0,
                camera_pixel_size: 6.5e-6,
                illumination_refractive_index: 1.0,
                objective_medium_refractive_index: 1.0,
                defocus_distance: None,
                pupil_aberration: None,
            },
            calibration: SourceCalibration::unity(),
            acquisition: AcquisitionPlan::all_sources(1).unwrap(),
        })
        .collect::<Vec<_>>();
    let model = SpectralImagePlaneModel::from_experiment(
        &channels,
        &SpectralGeometry::Shared(
            DirectionList::from_direction_cosines(vec![[0.0, 0.0]])
                .unwrap()
                .into(),
        ),
        SpectralAcquisitionPlan::separate(&[1, 1]).unwrap(),
        (4, 6),
        ReconstructionShape::Exact((4, 6)),
        ObjectCoupling::Independent,
    )
    .unwrap();
    let measurements = MeasurementStack::from_vec(vec![0.81; 48], (4, 6), vec![]).unwrap();
    let problem = SpectralReconstructionProblem::new(measurements, model).unwrap();
    let solver = MultiWavelengthGradientDescent {
        iterations: 2,
        initialization_iterations: 4,
        ..Default::default()
    };
    let reference = PhaseReference::Region {
        mask: Array2::from_elem((4, 6), 1),
        opd_m: 1.2e-6,
    };
    let result = solver
        .run(
            &problem,
            &SyntheticWavelengthUnwrapper::new((0.0, 2e-6)).unwrap(),
            &reference,
        )
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("automatic.json");
    result.checkpoint.save(&path).unwrap();
    let checkpoint = SpectralReconstructionCheckpoint::load(&path).unwrap();
    let resumed = MultiWavelengthGradientDescent {
        iterations: 5,
        ..solver
    }
    .run_from_checkpoint(&problem, checkpoint)
    .unwrap();
    assert_eq!(
        resumed.initialization_opd.as_ref().unwrap().opd_m,
        result.initialization_opd.as_ref().unwrap().opd_m
    );
    assert_eq!(
        losses(resumed.initialization_trace.as_ref().unwrap()),
        losses(result.initialization_trace.as_ref().unwrap())
    );
    assert_eq!(
        resumed
            .initialization_trace
            .as_ref()
            .unwrap()
            .iterations
            .len(),
        4
    );
    assert_eq!(resumed.spectral.runtime.completed_iterations, 5);
    #[cfg(feature = "parquet")]
    {
        let bundle = resumed
            .write_bundle(
                directory.path().join("initialized-bundle"),
                fpm_rs::tabular::parquet::BundleExportOptions::default(),
            )
            .unwrap();
        bundle.verify().unwrap();
        let loaded = bundle.joint_result().unwrap().unwrap();
        assert_eq!(
            loaded.initialization_opd.unwrap().opd_m,
            resumed.initialization_opd.unwrap().opd_m
        );
        assert_eq!(
            losses(loaded.initialization_trace.as_ref().unwrap()),
            losses(resumed.initialization_trace.as_ref().unwrap())
        );
    }
}

#[cfg(feature = "parquet")]
#[test]
fn bundles_roundtrip_both_solvers_and_reject_corruption_and_ordinary_interchange() {
    use fpm_rs::tabular::parquet::{BundleExportOptions, ResultBundle, SpectralResultBundle};
    let (problem, opd, amplitudes) = fixture(true, false);
    let dir = tempfile::tempdir().unwrap();
    let ap = SpectralAlternatingProjection::default()
        .iterations(3)
        .run(&problem)
        .unwrap();
    let bundle = ap
        .write_bundle(dir.path().join("spectral"), BundleExportOptions::default())
        .unwrap();
    bundle.verify().unwrap();
    assert!(ResultBundle::read(bundle.path()).is_err());
    let loaded = bundle.result().unwrap();
    assert_eq!(loaded.channels[0].object, ap.channels[0].object);
    assert_eq!(
        loaded.channels[1].object_spectrum,
        ap.channels[1].object_spectrum
    );
    assert_eq!(loaded.channels[0].phase, ap.channels[0].phase);
    assert!(bundle.joint_result().unwrap().is_none());
    let reference = PhaseReference::Offsets(vec![0.0; 2]);
    let solver = MultiWavelengthGradientDescent {
        iterations: 3,
        ..Default::default()
    };
    let joint = solver
        .run_from_opd(&problem, (0.0, 2e-6), &reference, opd, amplitudes)
        .unwrap();
    let bundle2 = joint
        .write_bundle(dir.path().join("joint"), BundleExportOptions::default())
        .unwrap();
    bundle2.verify().unwrap();
    let restored = bundle2.joint_result().unwrap().unwrap();
    assert_eq!(restored.opd_m, joint.opd_m);
    assert_eq!(
        restored.spectral.channels[0].amplitude,
        joint.spectral.channels[0].amplitude
    );
    let resumed = MultiWavelengthGradientDescent {
        iterations: 6,
        ..solver.clone()
    }
    .run_from_checkpoint(&problem, restored.checkpoint)
    .unwrap();
    let expected = MultiWavelengthGradientDescent {
        iterations: 6,
        ..solver
    }
    .run_from_checkpoint(&problem, joint.checkpoint)
    .unwrap();
    assert_eq!(resumed.opd_m, expected.opd_m);
    let artifact = &bundle.artifacts()["channels.0.object"];
    let mut bytes = std::fs::read(&artifact.path).unwrap();
    let end = bytes.len() - 1;
    bytes[end] ^= 1;
    std::fs::write(&artifact.path, bytes).unwrap();
    let reopened = SpectralResultBundle::read(bundle.path()).unwrap();
    assert!(reopened.result().is_err());
    assert!(reopened.verify().is_err());
    let path = bundle2.path().join("manifest.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["artifacts"][0]["relative_path"] = serde_json::json!("../escape.npy");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(SpectralResultBundle::read(bundle2.path()).is_err());
}

#[test]
fn spectral_dataset_preserves_explicit_channels_and_grayscale_order() {
    use fpm_rs::datasets::{DatasetLoader, SpectralDatasetChannel, SpectralDatasetManifest};
    use image::{GrayImage, Luma};
    let (problem, _, _) = fixture(false, false);
    let dir = tempfile::tempdir().unwrap();
    let mut channels = Vec::new();
    for (i, c) in problem.model.channels().iter().enumerate() {
        c.model
            .save_json(dir.path().join(format!("kernel-{i}.json")))
            .unwrap();
        channels.push(SpectralDatasetChannel {
            channel_id: c.channel_id.clone(),
            wavelength_vacuum_m: c.model.sampling().wavelength.unwrap(),
            compiled_model: format!("kernel-{i}.json").into(),
            response_provenance: "generated unity calibration and explicit exposure ordering"
                .into(),
            ground_truth_object: None,
            valid_object_mask: None,
        });
    }
    let mut frames = Vec::new();
    for i in 0..problem.model.frame_count() {
        let shape = problem.model.image_shape();
        let image = GrayImage::from_pixel(shape.1 as u32, shape.0 as u32, Luma([(i + 1) as u8]));
        image
            .save(dir.path().join(format!("frame-{i}.png")))
            .unwrap();
        frames.push(serde_json::json!({"path":format!("frame-{i}.png")}));
    }
    std::fs::write(
        dir.path().join("measurements.json"),
        serde_json::to_vec(&serde_json::json!({"frames":frames})).unwrap(),
    )
    .unwrap();
    let manifest = SpectralDatasetManifest {
        format_version: 2,
        measurement_manifest: "measurements.json".into(),
        channels,
        object_coupling: ObjectCoupling::Independent,
        acquisition: problem.model.acquisition().clone(),
        provenance: Default::default(),
        measurement_units: Some("camera counts".into()),
    };
    let path = dir.path().join("dataset.json");
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let loader = DatasetLoader::new(dir.path()).unwrap();
    assert!(loader.load().is_err());
    let dataset = loader.load_spectral().unwrap();
    assert_eq!(
        dataset.model().channels()[0].channel_id,
        problem.model.channels()[0].channel_id
    );
    assert_eq!(
        dataset.measurements().frame_count(),
        problem.model.frame_count()
    );
    dataset.reconstruction_problem().unwrap();
    assert_eq!(dataset.measurements().frame(1).unwrap()[0], 2.0);
    let mut bad = manifest.clone();
    bad.channels[0].wavelength_vacuum_m = 400e-9;
    std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(loader.load_spectral().is_err());
    let mut bad = manifest.clone();
    bad.channels[0].response_provenance.clear();
    std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(loader.load_spectral().is_err());
    let mut bad = manifest;
    bad.channels[0].compiled_model = "../kernel.json".into();
    std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(loader.load_spectral().is_err());
}
