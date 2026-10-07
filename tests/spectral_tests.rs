use fpm_rs::{
    Complex64,
    algorithms::{
        AlternatingProjection, MultiWavelengthGradientDescent, ReconstructionAlgorithm,
        SpectralAlternatingProjection, SpectralReconstructionAlgorithm,
    },
    experiment::{
        AcquisitionPlan, DirectionList, Illumination, IlluminationFrame, KVector, KVectorList,
        Optics, SourceCalibration, SourceContribution, SourceGeometry, SpectralAcquisitionPlan,
        SpectralChannel, SpectralContribution, SpectralFrame, SpectralGeometry,
    },
    measurements::{FrameMetadata, MeasurementStack},
    model::{
        ForwardModel, ImagePlaneModel, ObjectCoupling, ReconstructionShape, SpectralImagePlaneModel,
    },
    reconstruction::{
        PhaseReference, ReconstructionProblem, SpectralFrameSchedule,
        SpectralReconstructionProblem, SpectralReconstructionState, SpectralRunner,
        SyntheticWavelengthUnwrapper,
    },
};
use ndarray::{Array2, ShapeBuilder};

#[test]
fn joint_opd_solver_refines_a_perturbed_branch_and_preserves_the_shared_phase() {
    let mut channels = channels();
    for (c, w) in channels.iter_mut().zip([500e-9, 532e-9, 550e-9]) {
        c.optics.wavelength_vacuum_m = w;
        c.optics.defocus_distance = None;
        c.calibration = SourceCalibration::unity();
        c.acquisition = AcquisitionPlan::all_sources(9).unwrap();
    }
    let geometry = SpectralGeometry::Shared(
        DirectionList::from_direction_cosines(
            [-0.12, 0.0, 0.12]
                .into_iter()
                .flat_map(|dy| [-0.12, 0.0, 0.12].into_iter().map(move |dx| [dx, dy]))
                .collect(),
        )
        .unwrap()
        .into(),
    );
    for mixed in [false, true] {
        let acquisition = if mixed {
            SpectralAcquisitionPlan::multiplexed(
                (0..9)
                    .flat_map(|local_frame| {
                        (0..3).map(move |dominant| SpectralFrame {
                            contributions: (0..3)
                                .map(|channel| SpectralContribution {
                                    channel,
                                    local_frame,
                                    spectral_weight: if channel == dominant { 1.0 } else { 0.1 },
                                })
                                .collect(),
                            gain: 1.4,
                            background: 0.03,
                        })
                    })
                    .collect(),
            )
            .unwrap()
        } else {
            SpectralAcquisitionPlan::separate(&[9; 3]).unwrap()
        };
        let model = SpectralImagePlaneModel::from_experiment(
            &channels,
            &geometry,
            acquisition,
            (8, 12),
            ReconstructionShape::Smooth,
            ObjectCoupling::Independent,
        )
        .unwrap();
        let shape = model.reconstruction_shape();
        let truth_opd = Array2::from_shape_fn(shape, |(r, c)| {
            1.8e-6
                + 0.03e-6 * (std::f64::consts::TAU * c as f64 / shape.1 as f64).cos()
                + 0.02e-6 * (2.0 * std::f64::consts::TAU * r as f64 / shape.0 as f64).sin()
        });
        let amplitudes: Vec<_> = (0..3)
            .map(|c| {
                Array2::from_shape_fn(shape, |(r, col)| {
                    0.7 + c as f64 * 0.1
                        + 0.04 * (std::f64::consts::TAU * (r + col) as f64 / shape.1 as f64).cos()
                })
            })
            .collect();
        let objects: Vec<_> = channels
            .iter()
            .enumerate()
            .map(|(c, k)| {
                Array2::from_shape_fn(shape, |i| {
                    Complex64::from_polar(
                        amplitudes[c][i],
                        std::f64::consts::TAU * truth_opd[i] / k.optics.wavelength_vacuum_m,
                    )
                })
            })
            .collect();
        let problem =
            SpectralReconstructionProblem::new(predicted_measurements(&model, &objects), model)
                .unwrap();
        let initial_opd = Array2::from_shape_fn(shape, |(r, c)| {
            truth_opd[(r, c)]
                + 12e-9 * (std::f64::consts::TAU * (r + c) as f64 / shape.1 as f64).sin()
        });
        let initial_amplitude: Vec<_> = amplitudes.iter().map(|a| a.mapv(|v| 1.06 * v)).collect();
        let mut reference = Array2::zeros(shape);
        reference[(0, 0)] = 1;
        let solver = MultiWavelengthGradientDescent {
            iterations: 150,
            ..Default::default()
        };
        let result = solver
            .run_from_opd(
                &problem,
                (0.0, 4e-6),
                &PhaseReference::Region {
                    mask: reference,
                    opd_m: truth_opd[(0, 0)],
                },
                initial_opd.clone(),
                initial_amplitude,
            )
            .unwrap();
        let initial_loss = result.spectral.trace.iterations[0].objective;
        let final_loss = result.spectral.trace.final_objective().unwrap();
        assert!(
            final_loss < initial_loss * 0.05,
            "mixed={mixed}, initial={initial_loss}, final={final_loss}"
        );
        assert!(
            result
                .spectral
                .trace
                .iterations
                .windows(2)
                .all(|w| w[1].objective <= w[0].objective)
        );
        assert_eq!(result.opd_m[(0, 0)], truth_opd[(0, 0)]);
        let initial_error = initial_opd
            .iter()
            .zip(&truth_opd)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        let final_error = result
            .opd_m
            .iter()
            .zip(&truth_opd)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        assert!(
            final_error < initial_error * 0.4,
            "mixed={mixed}, OPD squared error ratio {}",
            final_error / initial_error
        );
        for channel in &result.spectral.channels {
            for (&opd, &field) in result.opd_m.iter().zip(&channel.object) {
                let expected = Complex64::from_polar(
                    field.norm(),
                    std::f64::consts::TAU * (opd / channel.wavelength_vacuum_m).rem_euclid(1.0),
                );
                assert!((field - expected).norm() < 1e-14);
            }
        }
    }
}

#[test]
fn multi_wavelength_runner_recovers_referenced_opd_from_intensity_data() {
    let mut channels = channels();
    for (channel, wavelength) in channels.iter_mut().zip([500e-9, 532e-9, 550e-9]) {
        channel.optics.wavelength_vacuum_m = wavelength;
        channel.optics.defocus_distance = None;
        channel.calibration = SourceCalibration::unity();
        channel.acquisition = AcquisitionPlan::all_sources(9).unwrap();
    }
    let geometry = SpectralGeometry::Shared(
        DirectionList::from_direction_cosines(
            [-0.12, 0.0, 0.12]
                .into_iter()
                .flat_map(|dy| [-0.12, 0.0, 0.12].into_iter().map(move |dx| [dx, dy]))
                .collect(),
        )
        .unwrap()
        .into(),
    );
    let model = SpectralImagePlaneModel::from_experiment(
        &channels,
        &geometry,
        SpectralAcquisitionPlan::separate(&[9; 3]).unwrap(),
        (8, 12),
        ReconstructionShape::Smooth,
        ObjectCoupling::Independent,
    )
    .unwrap();
    let shape = model.reconstruction_shape();
    let expected = Array2::from_shape_fn(shape, |(r, c)| {
        let x = std::f64::consts::TAU * c as f64 / shape.1 as f64;
        let y = std::f64::consts::TAU * r as f64 / shape.0 as f64;
        1.8e-6 + 0.03e-6 * x.cos() + 0.02e-6 * (2.0 * y).sin()
    });
    let truth: Vec<_> = channels
        .iter()
        .map(|channel| {
            Array2::from_shape_fn(shape, |(r, c)| {
                let d = expected[(r, c)];
                let x = std::f64::consts::TAU * c as f64 / shape.1 as f64;
                let y = std::f64::consts::TAU * r as f64 / shape.0 as f64;
                Complex64::from_polar(
                    0.8 + 0.1 * (x + y).cos(),
                    std::f64::consts::TAU * d / channel.optics.wavelength_vacuum_m,
                )
            })
        })
        .collect();
    let problem =
        SpectralReconstructionProblem::new(predicted_measurements(&model, &truth), model).unwrap();
    let mut reference_mask = Array2::zeros(shape);
    reference_mask[(0, 0)] = 1;
    let result = SpectralRunner::new(SpectralAlternatingProjection::default().iterations(200))
        .run_opd(
            &problem,
            &SyntheticWavelengthUnwrapper::new((0.0, 4e-6)).unwrap(),
            &PhaseReference::Region {
                mask: reference_mask,
                opd_m: expected[(0, 0)],
            },
            None,
        )
        .unwrap();
    assert!(
        result.spectral.trace.final_objective().unwrap()
            < result.spectral.trace.iterations[0].objective * 0.01
    );
    for (channel, expected_field) in result.spectral.channels.iter().zip(&truth) {
        let error = fpm_rs::metrics::complex_field::nrmse(
            expected_field.view(),
            channel.object.view(),
            None,
            fpm_rs::metrics::complex_field::ComplexAlignment::GlobalPhase,
        )
        .unwrap();
        assert!(error < 0.02, "{} field error {error}", channel.channel_id);
    }
    assert!(
        result.opd.valid_mask.iter().all(|&v| v == 1),
        "valid {} of {}",
        result.opd.valid_mask.iter().filter(|&&v| v == 1).count(),
        result.opd.valid_mask.len()
    );
    let max_error = expected
        .iter()
        .zip(&result.opd.opd_m)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_error < 20e-9, "maximum OPD error {max_error} metres");
    assert!(
        result
            .opd
            .fringe_orders
            .iter()
            .flatten()
            .any(|&v| v.abs() > 1)
    );
    let mut reference_mask = Array2::zeros(shape);
    reference_mask[(0, 0)] = 1;
    let joint = MultiWavelengthGradientDescent {
        iterations: 80,
        initialization_iterations: 200,
        ..Default::default()
    }
    .run(
        &problem,
        &SyntheticWavelengthUnwrapper::new((0.0, 4e-6)).unwrap(),
        &PhaseReference::Region {
            mask: reference_mask,
            opd_m: expected[(0, 0)],
        },
    )
    .unwrap();
    assert!(joint.initialization_opd.is_some());
    assert_eq!(
        joint
            .initialization_trace
            .as_ref()
            .unwrap()
            .iterations
            .len(),
        200
    );
    assert_eq!(joint.spectral.trace.iterations[0].iteration, 0);
    assert!(
        joint.spectral.trace.final_objective().unwrap()
            < joint.spectral.trace.iterations[0].objective
    );
    let max_error = joint
        .opd_m
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_error < 20e-9, "joint automatic OPD error {max_error}");
}

fn channels() -> Vec<SpectralChannel> {
    [450e-9, 532e-9, 630e-9]
        .into_iter()
        .enumerate()
        .map(|(index, wavelength)| SpectralChannel {
            channel_id: ["blue", "green", "red"][index].into(),
            optics: Optics {
                wavelength_vacuum_m: wavelength,
                objective_na: 0.12,
                magnification: 4.0,
                camera_pixel_size: 6.5e-6,
                illumination_refractive_index: 1.0,
                objective_medium_refractive_index: 1.0,
                defocus_distance: Some(1e-6),
                pupil_aberration: None,
            },
            calibration: SourceCalibration::new(Some(vec![1.0, 0.8, 1.2])),
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
                    0.7,
                ),
            ])
            .unwrap(),
        })
        .collect()
}

fn geometry() -> SpectralGeometry {
    SpectralGeometry::Shared(
        DirectionList::from_direction_cosines(vec![[0.0, 0.0], [0.08, 0.03], [-0.05, 0.07]])
            .unwrap()
            .into(),
    )
}

fn compile(
    channels: &[SpectralChannel],
    plan: SpectralAcquisitionPlan,
    coupling: ObjectCoupling,
) -> SpectralImagePlaneModel {
    SpectralImagePlaneModel::from_experiment(
        channels,
        &geometry(),
        plan,
        (8, 12),
        ReconstructionShape::Smooth,
        coupling,
    )
    .unwrap()
}

fn mixed_plan() -> SpectralAcquisitionPlan {
    SpectralAcquisitionPlan::multiplexed(
        (0..2)
            .map(|local_frame| SpectralFrame {
                contributions: (0..3)
                    .map(|channel| SpectralContribution {
                        channel,
                        local_frame,
                        spectral_weight: [0.4, 0.8, 1.1][channel],
                    })
                    .collect(),
                gain: 1.7,
                background: 0.2,
            })
            .collect(),
    )
    .unwrap()
}

fn objects(model: &SpectralImagePlaneModel) -> Vec<Array2<Complex64>> {
    (0..model.object_count())
        .map(|channel| {
            Array2::from_shape_fn(model.reconstruction_shape(), |(row, col)| {
                Complex64::from_polar(
                    0.8 + 0.1 * ((row + col + channel) as f64).sin(),
                    0.1 * (row as f64 * 0.4 + col as f64 * 0.7 + channel as f64).cos(),
                )
            })
        })
        .collect()
}

fn predicted_measurements(
    model: &SpectralImagePlaneModel,
    objects: &[Array2<Complex64>],
) -> MeasurementStack {
    let dummy =
        MeasurementStack::from_vec(vec![1.0; model.frame_count() * 8 * 12], (8, 12), vec![])
            .unwrap();
    let problem = SpectralReconstructionProblem::new(dummy, model.clone()).unwrap();
    let state = SpectralReconstructionState::from_objects(&problem, objects.to_vec()).unwrap();
    let frames: Vec<_> = (0..model.frame_count())
        .map(|frame| {
            model
                .forward_intensity(&state.object_spectra(), frame)
                .unwrap()
                .into_raw_vec_and_offset()
                .0
        })
        .collect();
    MeasurementStack::from_frames(&frames, (8, 12)).unwrap()
}

#[test]
fn wavelength_specific_geometry_pupils_and_union_grid() {
    let channels = channels();
    let model = compile(
        &channels,
        SpectralAcquisitionPlan::separate(&[2, 2, 2]).unwrap(),
        ObjectCoupling::Independent,
    );
    let SpectralGeometry::Shared(shared_geometry) = geometry() else {
        unreachable!()
    };
    for (index, channel) in channels.iter().enumerate() {
        let illumination = Illumination::new(
            shared_geometry.clone(),
            channel.calibration.clone(),
            channel.acquisition.clone(),
        );
        let ordinary = ImagePlaneModel::from_experiment(
            &channel.optics,
            &illumination,
            (8, 12),
            ReconstructionShape::Minimum,
        )
        .unwrap();
        assert!(model.reconstruction_shape().0 >= ordinary.reconstruction_shape().0);
        assert_eq!(
            model.channels()[index].model.k_vectors(),
            ordinary.k_vectors()
        );
        assert_eq!(
            model.channels()[index].model.pupil().values(),
            ordinary.pupil().values()
        );
    }
    let blue = &model.channels()[0].model;
    let red = &model.channels()[2].model;
    assert_ne!(blue.pupil().support(), red.pupil().support());
    assert!((blue.k_vectors()[1].kx / red.k_vectors()[1].kx - 630.0 / 450.0).abs() < 1e-14);
    assert!(
        SpectralImagePlaneModel::from_experiment(
            &channels,
            &geometry(),
            SpectralAcquisitionPlan::separate(&[2, 2, 2]).unwrap(),
            (8, 12),
            ReconstructionShape::Exact((8, 12)),
            ObjectCoupling::Independent
        )
        .is_err()
    );
}

#[test]
fn single_channel_matches_ordinary_ap_with_local_multiplexing_and_gains() {
    let channels = vec![channels()[0].clone()];
    let model = compile(
        &channels,
        SpectralAcquisitionPlan::separate(&[2]).unwrap(),
        ObjectCoupling::Independent,
    );
    let measurements = predicted_measurements(&model, &objects(&model));
    let ordinary =
        ReconstructionProblem::new(measurements.clone(), model.channels()[0].model.clone())
            .unwrap();
    let spectral = SpectralReconstructionProblem::new(measurements, model).unwrap();
    let expected = AlternatingProjection::default()
        .iterations(4)
        .run(&ordinary)
        .unwrap();
    let actual = SpectralAlternatingProjection::default()
        .iterations(4)
        .run(&spectral)
        .unwrap();
    assert_eq!(actual.channels[0].object_spectrum, expected.object_spectrum);
    assert_eq!(actual.channels[0].object, expected.object);
    assert_eq!(
        actual
            .trace
            .iterations
            .iter()
            .map(|row| row.objective)
            .collect::<Vec<_>>(),
        expected
            .trace
            .iterations
            .iter()
            .map(|row| row.objective)
            .collect::<Vec<_>>()
    );
}

#[test]
fn independent_separate_runs_match_per_channel_ap_and_batching() {
    let model = compile(
        &channels(),
        SpectralAcquisitionPlan::separate(&[2, 2, 2]).unwrap(),
        ObjectCoupling::Independent,
    );
    let measurements = predicted_measurements(&model, &objects(&model));
    let problem = SpectralReconstructionProblem::new(measurements.clone(), model.clone()).unwrap();
    let result = SpectralAlternatingProjection::default()
        .iterations(4)
        .batch_size(4)
        .run(&problem)
        .unwrap();
    for (index, channel) in model.channels().iter().enumerate() {
        let start = index * 2 * 8 * 12;
        let local = MeasurementStack::from_vec(
            measurements.as_slice()[start..start + 2 * 8 * 12].to_vec(),
            (8, 12),
            vec![],
        )
        .unwrap();
        let ordinary = ReconstructionProblem::new(local, channel.model.clone()).unwrap();
        let expected = AlternatingProjection::default()
            .iterations(4)
            .run(&ordinary)
            .unwrap();
        assert_eq!(
            result.channels[index].object_spectrum,
            expected.object_spectrum
        );
    }
    let unbatched = SpectralAlternatingProjection::default()
        .iterations(4)
        .run(&problem)
        .unwrap();
    assert_eq!(result.channels[0].object, unbatched.channels[0].object);
}

#[test]
fn mixed_intensity_matches_manual_channel_sum_and_detector_calibration() {
    let model = compile(&channels(), mixed_plan(), ObjectCoupling::Independent);
    let measurements = predicted_measurements(&model, &objects(&model));
    let problem = SpectralReconstructionProblem::new(measurements, model.clone()).unwrap();
    let state = SpectralReconstructionState::from_objects(&problem, objects(&model)).unwrap();
    for (frame, row) in model.acquisition().frames().iter().enumerate() {
        let mut sum = Array2::zeros(model.image_shape());
        for contribution in &row.contributions {
            let kernel = &model.channels()[contribution.channel].model;
            let local = ForwardModel::new(kernel)
                .unwrap()
                .forward_intensity(
                    state.object_spectra()[contribution.channel],
                    kernel.pupil(),
                    contribution.local_frame,
                )
                .unwrap();
            sum.scaled_add(contribution.spectral_weight, &local);
        }
        sum.mapv_inplace(|value| row.gain * value + row.background);
        assert_eq!(
            model
                .forward_intensity(&state.object_spectra(), frame)
                .unwrap(),
            sum
        );
    }
}

#[test]
fn shared_object_updates_accumulate_and_return_identical_channel_fields() {
    let model = compile(&channels(), mixed_plan(), ObjectCoupling::SharedComplex);
    let measurements = predicted_measurements(&model, &objects(&model));
    let problem = SpectralReconstructionProblem::new(measurements, model).unwrap();
    let result = SpectralAlternatingProjection::default()
        .iterations(8)
        .run(&problem)
        .unwrap();
    assert_eq!(result.channels[0].object, result.channels[1].object);
    assert_eq!(
        result.channels[1].object_spectrum,
        result.channels[2].object_spectrum
    );
    assert!(result.trace.final_objective().unwrap() < result.trace.iterations[0].objective);
    assert_ne!(
        result.channels[0].pupil.values(),
        result.channels[1].pupil.values()
    );
}

#[test]
fn canonical_plan_and_invalid_configurations() {
    let plan = SpectralAcquisitionPlan::multiplexed(vec![SpectralFrame {
        contributions: vec![
            SpectralContribution {
                channel: 0,
                local_frame: 0,
                spectral_weight: 0.2,
            },
            SpectralContribution {
                channel: 0,
                local_frame: 0,
                spectral_weight: 0.3,
            },
            SpectralContribution {
                channel: 9,
                local_frame: 4,
                spectral_weight: 0.0,
            },
        ],
        gain: 1.0,
        background: 0.0,
    }])
    .unwrap();
    assert_eq!(
        plan.frames()[0].contributions,
        vec![SpectralContribution {
            channel: 0,
            local_frame: 0,
            spectral_weight: 0.5
        }]
    );
    let mut channels = channels();
    let build = |channels: &[SpectralChannel], geometry: &SpectralGeometry| {
        SpectralImagePlaneModel::from_experiment(
            channels,
            geometry,
            SpectralAcquisitionPlan::separate(&[2, 2, 2]).unwrap(),
            (8, 12),
            ReconstructionShape::Minimum,
            ObjectCoupling::Independent,
        )
    };
    channels[1].channel_id = channels[0].channel_id.clone();
    assert!(build(&channels, &geometry()).is_err());
    channels[1].channel_id = "green".into();
    channels[1].optics.wavelength_vacuum_m = channels[0].optics.wavelength_vacuum_m;
    assert!(build(&channels, &geometry()).is_err());
    channels[1].optics.wavelength_vacuum_m = 532e-9;
    channels[1].optics.magnification = 5.0;
    assert!(build(&channels, &geometry()).is_err());
    channels[1].optics.magnification = 4.0;
    let direct = SpectralGeometry::Shared(SourceGeometry::KVectors(KVectorList::new(vec![
        KVector::new(0.0, 0.0),
    ])));
    assert!(build(&channels, &direct).is_err());
    assert!(
        SpectralImagePlaneModel::from_experiment(
            &channels,
            &geometry(),
            plan,
            (8, 12),
            ReconstructionShape::Minimum,
            ObjectCoupling::Independent
        )
        .is_err()
    );
    let model = compile(&channels, mixed_plan(), ObjectCoupling::Independent);
    let mut metadata = vec![FrameMetadata::new(0), FrameMetadata::new(1)];
    metadata[1].weight = 0.0;
    let measurements =
        MeasurementStack::from_vec(vec![1.0; 2 * 8 * 12], (8, 12), metadata).unwrap();
    assert!(SpectralReconstructionProblem::new(measurements, model).is_err());
}

#[test]
fn rejects_nonstandard_initial_objects_and_invalid_schedules() {
    let model = compile(&channels(), mixed_plan(), ObjectCoupling::Independent);
    let problem = SpectralReconstructionProblem::new(
        predicted_measurements(&model, &objects(&model)),
        model.clone(),
    )
    .unwrap();
    let shape = model.reconstruction_shape();
    let initial = Array2::from_elem(shape.f(), Complex64::new(1.0, 0.0));
    assert!(SpectralReconstructionState::from_objects(&problem, vec![initial; 3]).is_err());
    assert!(
        SpectralRunner::new(SpectralAlternatingProjection::default())
            .with_schedule(SpectralFrameSchedule::Explicit(vec![0, 0]))
            .run(&problem)
            .is_err()
    );
    let result = SpectralRunner::new(SpectralAlternatingProjection::default().iterations(2))
        .with_schedule(SpectralFrameSchedule::RandomShuffle { seed: 42 })
        .run(&problem)
        .unwrap();
    assert_eq!(result.runtime.completed_iterations, 2);
}

#[test]
fn compiled_spectral_serialization_revalidates_channels_and_references() {
    let model = compile(&channels(), mixed_plan(), ObjectCoupling::Independent);
    let encoded = serde_json::to_string(&model).unwrap();
    let restored: SpectralImagePlaneModel = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        restored.channels()[0].model.k_vectors(),
        model.channels()[0].model.k_vectors()
    );
    assert_eq!(restored.acquisition(), model.acquisition());
    let mut invalid: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    invalid["channels"][1]["channel_id"] = invalid["channels"][0]["channel_id"].clone();
    assert!(serde_json::from_value::<SpectralImagePlaneModel>(invalid).is_err());
    let mut invalid: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    invalid["acquisition"]["frames"][0]["contributions"][0]["local_frame"] = 99.into();
    assert!(serde_json::from_value::<SpectralImagePlaneModel>(invalid).is_err());
}

#[test]
fn masked_detector_pixels_do_not_affect_any_spectral_object() {
    let model = compile(&channels(), mixed_plan(), ObjectCoupling::Independent);
    let measurements = predicted_measurements(&model, &objects(&model));
    let mut changed = measurements.as_slice().to_vec();
    let mut mask = ndarray::Array3::from_elem((model.frame_count(), 8, 12), 1u8);
    for frame in 0..model.frame_count() {
        mask[(frame, 3, 5)] = 0;
        changed[frame * 8 * 12 + 3 * 12 + 5] = 1e6;
    }
    let original = measurements.with_per_frame_masks(mask.clone()).unwrap();
    let changed = MeasurementStack::from_vec(changed, (8, 12), vec![])
        .unwrap()
        .with_per_frame_masks(mask)
        .unwrap();
    let original = SpectralReconstructionProblem::new(original, model.clone()).unwrap();
    let changed = SpectralReconstructionProblem::new(changed, model).unwrap();
    let solver = SpectralAlternatingProjection::default().iterations(3);
    let expected = solver.clone().run(&original).unwrap();
    let actual = solver.run(&changed).unwrap();
    for (left, right) in expected.channels.iter().zip(&actual.channels) {
        assert_eq!(left.object_spectrum, right.object_spectrum);
    }
    assert_eq!(
        expected.trace.final_objective(),
        actual.trace.final_objective()
    );
}

#[test]
fn permuting_channel_records_preserves_mixed_predictions_and_recovery() {
    let original_channels = channels();
    let original = compile(
        &original_channels,
        mixed_plan(),
        ObjectCoupling::Independent,
    );
    let truth = objects(&original);
    let measurements = predicted_measurements(&original, &truth);
    let order = [2, 0, 1];
    let reordered_channels: Vec<_> = order
        .iter()
        .map(|&index| original_channels[index].clone())
        .collect();
    let mut frames = mixed_plan().frames().to_vec();
    for row in &mut frames {
        for contribution in &mut row.contributions {
            contribution.channel = order
                .iter()
                .position(|&index| index == contribution.channel)
                .unwrap();
        }
    }
    let reordered = compile(
        &reordered_channels,
        SpectralAcquisitionPlan::multiplexed(frames).unwrap(),
        ObjectCoupling::Independent,
    );
    let reordered_truth: Vec<_> = order.iter().map(|&index| truth[index].clone()).collect();
    let reordered_measurements = predicted_measurements(&reordered, &reordered_truth);
    for (&left, &right) in measurements
        .as_slice()
        .iter()
        .zip(reordered_measurements.as_slice())
    {
        assert!((left - right).abs() < 1e-14);
    }
    let original_problem =
        SpectralReconstructionProblem::new(measurements.clone(), original).unwrap();
    let reordered_problem = SpectralReconstructionProblem::new(measurements, reordered).unwrap();
    let result = SpectralAlternatingProjection::default()
        .iterations(3)
        .run(&original_problem)
        .unwrap();
    let permuted = SpectralAlternatingProjection::default()
        .iterations(3)
        .run(&reordered_problem)
        .unwrap();
    for (channel, &original_index) in order.iter().enumerate() {
        assert_eq!(
            permuted.channels[channel].channel_id,
            result.channels[original_index].channel_id
        );
        for (&left, &right) in result.channels[original_index]
            .object
            .iter()
            .zip(&permuted.channels[channel].object)
        {
            assert!((left - right).norm() < 1e-13);
        }
    }
}

#[test]
fn bounded_rgb_multiplexed_reconstruction_recovers_distinct_transmissions() {
    // An implementation-derived narrowband synthetic example inspired by the
    // state decomposition discussed in SpectralAlternatingProjection's reference.
    // All three channels contribute to each exposure; three fixed weight rows
    // are repeated at each source, with a distinct dominant channel in each row.
    let mut channels = channels();
    for channel in &mut channels {
        channel.acquisition = AcquisitionPlan::all_sources(9).unwrap();
        channel.calibration = SourceCalibration::unity();
        channel.optics.defocus_distance = None;
    }
    let rows = (0..9)
        .flat_map(|local_frame| {
            (0..3).map(move |dominant| SpectralFrame {
                contributions: (0..3)
                    .map(|channel| SpectralContribution {
                        channel,
                        local_frame,
                        spectral_weight: if channel == dominant { 1.0 } else { 0.1 },
                    })
                    .collect(),
                gain: 1.0,
                background: 0.0,
            })
        })
        .collect();
    let spectral_geometry = SpectralGeometry::Shared(
        DirectionList::from_direction_cosines(
            [-0.12, 0.0, 0.12]
                .into_iter()
                .flat_map(|dy| [-0.12, 0.0, 0.12].into_iter().map(move |dx| [dx, dy]))
                .collect(),
        )
        .unwrap()
        .into(),
    );
    let model = SpectralImagePlaneModel::from_experiment(
        &channels,
        &spectral_geometry,
        SpectralAcquisitionPlan::multiplexed(rows).unwrap(),
        (8, 12),
        ReconstructionShape::Smooth,
        ObjectCoupling::Independent,
    )
    .unwrap();
    let shape = model.reconstruction_shape();
    let truth: Vec<_> = (0..3)
        .map(|channel| {
            Array2::from_shape_fn(shape, |(row, col)| {
                let x = std::f64::consts::TAU * col as f64 / shape.1 as f64;
                let y = std::f64::consts::TAU * row as f64 / shape.0 as f64;
                Complex64::from_polar(
                    0.65 + channel as f64 * 0.12 + 0.08 * (x + channel as f64).cos(),
                    0.08 * (2.0 * y + channel as f64).sin(),
                )
            })
        })
        .collect();
    let problem =
        SpectralReconstructionProblem::new(predicted_measurements(&model, &truth), model).unwrap();
    let result = SpectralAlternatingProjection::default()
        .iterations(150)
        .run(&problem)
        .unwrap();
    assert!(result.trace.final_objective().unwrap() < result.trace.iterations[0].objective * 0.01);
    for (channel, expected) in result.channels.iter().zip(&truth) {
        let error = fpm_rs::metrics::complex_field::nrmse(
            expected.view(),
            channel.object.view(),
            None,
            fpm_rs::metrics::complex_field::ComplexAlignment::GlobalPhase,
        )
        .unwrap();
        assert!(
            error < 0.03,
            "{} normalized complex error: {error}",
            channel.channel_id
        );
    }
}
