use std::collections::BTreeMap;

use fpm_rs::{
    algorithms::{
        AdaptiveAlternatingProjection, Fpie, JointReconstruction, Mpie, ReconstructionAlgorithm,
    },
    callbacks::CheckpointEvery,
    experiment::{
        ArrayPose, DirectionList, Illumination, Optics, PlanarLedArray, SourceCalibration,
    },
    illumination_calibration::{
        BoundedFiniteDifferenceOptimizer, CalibrationParameterSpec, IlluminationCalibration,
        PlanarArrayCalibrationParameters, PlanarArrayParameterValues,
    },
    measurements::MeasurementStack,
    metrics::complex_field::compare_complex_fields,
    model::{ForwardModel, ImagePlaneModel, ReconstructionShape},
    reconstruction::{ReconstructionCheckpoint, ReconstructionProblem, ReconstructionState},
    simulation::{Simulator, SyntheticObject},
};

fn optics() -> Optics {
    Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    }
}

fn array(translation: [f64; 3]) -> PlanarLedArray {
    PlanarLedArray::new(
        (3, 3),
        (4e-3, 4e-3),
        (1.0, 1.0),
        ArrayPose::from_translation(translation),
    )
}

fn simulate_pair(
    true_illumination: &Illumination,
    nominal_illumination: &Illumination,
) -> (
    MeasurementStack,
    ImagePlaneModel,
    ndarray::Array2<fpm_rs::Complex64>,
) {
    let optics = optics();
    let true_model = ImagePlaneModel::from_experiment(
        &optics,
        true_illumination,
        (12, 12),
        ReconstructionShape::Exact((36, 36)),
    )
    .unwrap();
    let nominal_model = ImagePlaneModel::from_experiment(
        &optics,
        nominal_illumination,
        (12, 12),
        ReconstructionShape::Exact((36, 36)),
    )
    .unwrap();
    let simulation = Simulator::new(true_model)
        .object(SyntheticObject::resolution_target((36, 36)).unwrap())
        .reconstruction_model(nominal_model.clone())
        .seed(17)
        .simulate()
        .unwrap();
    (
        simulation.measurements,
        nominal_model,
        simulation.ground_truth_object,
    )
}

fn fixed_truth_state(
    measurements: MeasurementStack,
    model: ImagePlaneModel,
    object: ndarray::Array2<fpm_rs::Complex64>,
) -> (ReconstructionProblem<MeasurementStack>, ReconstructionState) {
    let problem = ReconstructionProblem::new(measurements, model).unwrap();
    let state = ReconstructionState::from_object(&problem, object).unwrap();
    (problem, state)
}

fn amplitude_loss(
    measurements: &MeasurementStack,
    model: &ImagePlaneModel,
    state: &ReconstructionState,
) -> f64 {
    let forward = ForwardModel::new(model).unwrap();
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for frame in 0..measurements.frame_count() {
        let measured = measurements.frame(frame).unwrap();
        let weight = measurements.frame_weight(frame).unwrap();
        sum += weight
            * forward
                .frame_loss(
                    state.object_spectrum(),
                    state.pupil(),
                    frame,
                    measured,
                    fpm_rs::algorithms::objective::LossType::AmplitudeMse,
                )
                .unwrap();
        weight_sum += weight;
    }
    sum / weight_sum
}

#[test]
fn parameter_selection_is_explicit_and_rejects_gauges_and_duplicates() {
    let default = PlanarArrayCalibrationParameters::default();
    assert!(!default.has_active_parameters());

    assert!(
        PlanarArrayCalibrationParameters::builder()
            .position_offsets([1, 1])
            .build()
            .is_err()
    );
    let ambiguous = PlanarArrayCalibrationParameters::builder()
        .translation([true, false, false])
        .reference_index([true, false])
        .build()
        .unwrap();
    let illumination = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    assert!(ambiguous.validate_for(&illumination).is_err());

    let scale_ambiguous = PlanarArrayCalibrationParameters::builder()
        .relative_source_power(true)
        .frame_gains(true)
        .build()
        .unwrap();
    assert!(scale_ambiguous.validate_for(&illumination).is_err());

    let mut offsets = vec![[0.0; 3]; 9];
    offsets[1] = [0.2e-3, -0.1e-3, 0.05e-3];
    offsets[7] = [0.4e-3, 0.3e-3, -0.15e-3];
    let illumination_with_offsets =
        Illumination::from_geometry(array([0.0, 0.0, -80e-3]).with_position_offsets_m(offsets))
            .unwrap();
    let constrained = PlanarArrayCalibrationParameters::builder()
        .translation([true, false, false])
        .position_offsets([1, 7])
        .build()
        .unwrap();
    let state = IlluminationCalibration::new(constrained)
        .initialize(&illumination_with_offsets, &optics())
        .unwrap();
    for axis in 0..3 {
        let mean = (state.current_parameters.position_offsets_m[1][axis]
            + state.current_parameters.position_offsets_m[7][axis])
            / 2.0;
        assert!(mean.abs() < 1e-15);
    }
    assert!(
        state
            .applied_constraints
            .iter()
            .any(|constraint| constraint.contains("mean(selected position offset)"))
    );
}

#[test]
fn specifications_and_unsupported_geometry_are_validated() {
    let invalid = CalibrationParameterSpec::new(1.0, 0.0, 1.0);
    assert!(invalid.validate("test").is_err());
    assert!(
        CalibrationParameterSpec::new(0.0, 1.0, 0.0)
            .validate("test")
            .is_err()
    );
    assert!(
        CalibrationParameterSpec::new(0.0, 1.0, 1.0)
            .finite_difference_step(0.0)
            .validate("test")
            .is_err()
    );
    assert!(
        CalibrationParameterSpec::new(0.0, 1.0, 1.0)
            .prior(0.5, -1.0)
            .validate("test")
            .is_err()
    );
    let directions = DirectionList::from_unit_vectors(vec![[0.0, 0.0, 1.0]]).unwrap();
    let unsupported = Illumination::from_geometry(directions).unwrap();
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation([true, false, false])
        .build()
        .unwrap();
    assert!(parameters.validate_for(&unsupported).is_err());
    let planar = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let bad_source = PlanarArrayCalibrationParameters::builder()
        .position_offsets([9])
        .build()
        .unwrap();
    assert!(bad_source.validate_for(&planar).is_err());
    let nonphysical_pitch = PlanarArrayCalibrationParameters::builder()
        .pitch_specs([Some(CalibrationParameterSpec::new(-1e-3, 1e-2, 1e-4)), None])
        .build()
        .unwrap();
    assert!(nonphysical_pitch.validate_for(&planar).is_err());
}

#[test]
fn intensity_only_model_updates_preserve_geometry_and_pupil() {
    let illumination = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let mut model = ImagePlaneModel::from_experiment(
        &optics(),
        &illumination,
        (12, 12),
        ReconstructionShape::Exact((36, 36)),
    )
    .unwrap();
    let vectors = model.k_vectors().to_vec();
    let crops = model.crop_indices().clone();
    let offsets = model.subpixel_offsets().unwrap().to_vec();
    let pupil = model.pupil().clone();
    let powers = vec![1.0; model.source_count()];
    let gains = vec![0.5, 0.75, 1.0, 1.25, 1.5, 0.8, 0.9, 1.1, 1.2];
    let frames = illumination
        .acquisition()
        .frames()
        .iter()
        .zip(gains)
        .map(|(frame, gain)| {
            fpm_rs::experiment::IlluminationFrame::new(frame.contributions.clone(), gain)
        })
        .collect();
    let acquisition = fpm_rs::experiment::AcquisitionPlan::from_sparse(frames).unwrap();
    model
        .update_intensity_calibration(&powers, &acquisition)
        .unwrap();
    assert_eq!(model.k_vectors(), vectors);
    assert_eq!(model.crop_indices().as_slice(), crops.as_slice());
    assert_eq!(model.subpixel_offsets().unwrap(), offsets);
    assert_eq!(model.pupil().values(), pupil.values());
    assert_eq!(model.frame_gains().unwrap()[0], 0.5);
}

#[test]
fn geometry_model_updates_change_vectors_but_preserve_static_optics() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let translated = Illumination::from_geometry(array([0.3e-3, -0.2e-3, -79e-3])).unwrap();
    let mut model = ImagePlaneModel::from_experiment(
        &optics(),
        &nominal,
        (12, 12),
        ReconstructionShape::Exact((36, 36)),
    )
    .unwrap();
    let vectors = model.k_vectors().to_vec();
    let pupil = model.pupil().clone();
    let sampling = model.sampling().clone();
    model
        .update_illumination_geometry(&optics(), &translated)
        .unwrap();
    assert_ne!(model.k_vectors(), vectors);
    assert_eq!(model.pupil().values(), pupil.values());
    assert_eq!(model.pupil().support(), pupil.support());
    assert_eq!(
        model.sampling().low_res_pixel_size,
        sampling.low_res_pixel_size
    );
    assert_eq!(model.image_shape(), (12, 12));
    assert_eq!(model.reconstruction_shape(), (36, 36));
}

#[test]
fn bounded_translation_calibration_reduces_loss_and_moves_toward_truth() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let true_illumination = Illumination::from_geometry(array([0.45e-3, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, object) = simulate_pair(&true_illumination, &nominal);
    let (problem, state) = fixed_truth_state(measurements, nominal_model, object);
    let mut calibrated_model = problem.model.clone();
    let initial_loss = amplitude_loss(&problem.measurements, &calibrated_model, &state);
    let translation_spec =
        CalibrationParameterSpec::new(-2e-3, 2e-3, 5e-4).finite_difference_step(2e-5);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(translation_spec), None, None])
        .build()
        .unwrap();
    let calibration =
        IlluminationCalibration::new(parameters).optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 4,
            initial_step_size: 0.5,
            ..BoundedFiniteDifferenceOptimizer::default()
        });
    let result = calibration
        .calibrate(
            &problem.measurements,
            &optics(),
            &nominal,
            state.object_spectrum(),
            state.pupil(),
            &mut calibrated_model,
            3,
        )
        .unwrap();
    let final_loss = amplitude_loss(&problem.measurements, &calibrated_model, &state);
    assert!(final_loss < initial_loss, "{final_loss} !< {initial_loss}");
    assert!((result.current_parameters.translation_m[0] - 0.45e-3).abs() < 0.45e-3);
    assert!(result.geometry_recompilations > 0);
    assert_eq!(result.multiplicative_updates, 0);
    assert!(!result.loss_history.is_empty());
}

fn assert_geometry_parameter_recovers(
    name: &str,
    true_geometry: PlanarLedArray,
    parameters: PlanarArrayCalibrationParameters,
    value: fn(&PlanarArrayParameterValues) -> f64,
    truth: f64,
) {
    let nominal_geometry = array([0.0, 0.0, -80e-3]);
    let nominal = Illumination::from_geometry(nominal_geometry).unwrap();
    let true_illumination = Illumination::from_geometry(true_geometry).unwrap();
    let (measurements, nominal_model, object) = simulate_pair(&true_illumination, &nominal);
    let (problem, state) = fixed_truth_state(measurements, nominal_model, object);
    let mut calibrated_model = problem.model.clone();
    let initial_loss = amplitude_loss(&problem.measurements, &calibrated_model, &state);
    let initial_parameters = PlanarArrayParameterValues::from_illumination(&nominal).unwrap();
    let initial_error = (value(&initial_parameters) - truth).abs();
    let calibration =
        IlluminationCalibration::new(parameters).optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 4,
            initial_step_size: 0.5,
            relative_tolerance: 0.0,
            ..BoundedFiniteDifferenceOptimizer::default()
        });
    let result = calibration
        .calibrate(
            &problem.measurements,
            &optics(),
            &nominal,
            state.object_spectrum(),
            state.pupil(),
            &mut calibrated_model,
            3,
        )
        .unwrap();
    let final_loss = amplitude_loss(&problem.measurements, &calibrated_model, &state);
    let final_error = (value(&result.current_parameters) - truth).abs();
    assert!(
        final_loss < initial_loss,
        "{name}: {final_loss} !< {initial_loss}"
    );
    assert!(
        final_error < initial_error,
        "{name}: parameter error {final_error} !< {initial_error}"
    );
}

#[test]
fn every_geometry_parameter_group_recovers_independently() {
    let translation_spec =
        || CalibrationParameterSpec::new(-2e-3, 2e-3, 4e-4).finite_difference_step(2e-5);
    assert_geometry_parameter_recovers(
        "ty",
        array([0.0, 0.4e-3, -80e-3]),
        PlanarArrayCalibrationParameters::builder()
            .translation_specs([None, Some(translation_spec()), None])
            .build()
            .unwrap(),
        |values| values.translation_m[1],
        0.4e-3,
    );

    let axial_spec =
        CalibrationParameterSpec::new(-90e-3, -70e-3, 2e-3).finite_difference_step(0.1e-3);
    assert_geometry_parameter_recovers(
        "tz",
        array([0.0, 0.0, -78e-3]),
        PlanarArrayCalibrationParameters::builder()
            .translation_specs([None, None, Some(axial_spec)])
            .build()
            .unwrap(),
        |values| values.translation_m[2],
        -78e-3,
    );

    for (axis, name) in ["rx", "ry", "rz"].into_iter().enumerate() {
        let mut rotation = [0.0; 3];
        rotation[axis] = 0.012;
        let rotation_spec =
            CalibrationParameterSpec::new(-0.04, 0.04, 0.01).finite_difference_step(2e-4);
        let mut specs = [None, None, None];
        specs[axis] = Some(rotation_spec);
        assert_geometry_parameter_recovers(
            name,
            array([0.0, 0.0, -80e-3]).with_pose(
                ArrayPose::from_translation_and_extrinsic_xyz_radians([0.0, 0.0, -80e-3], rotation),
            ),
            PlanarArrayCalibrationParameters::builder()
                .rotation_specs(specs)
                .build()
                .unwrap(),
            match axis {
                0 => |values: &PlanarArrayParameterValues| values.rotation_rad[0],
                1 => |values: &PlanarArrayParameterValues| values.rotation_rad[1],
                _ => |values: &PlanarArrayParameterValues| values.rotation_rad[2],
            },
            0.012,
        );
    }

    for (axis, name) in ["pitch_x", "pitch_y"].into_iter().enumerate() {
        let mut pitch = [4e-3, 4e-3];
        pitch[axis] = 4.15e-3;
        let pitch_spec =
            CalibrationParameterSpec::new(3.5e-3, 4.5e-3, 0.15e-3).finite_difference_step(2e-6);
        let mut specs = [None, None];
        specs[axis] = Some(pitch_spec);
        assert_geometry_parameter_recovers(
            name,
            array([0.0, 0.0, -80e-3]).with_pitch_m((pitch[0], pitch[1])),
            PlanarArrayCalibrationParameters::builder()
                .pitch_specs(specs)
                .build()
                .unwrap(),
            if axis == 0 {
                |values: &PlanarArrayParameterValues| values.pitch_m[0]
            } else {
                |values: &PlanarArrayParameterValues| values.pitch_m[1]
            },
            4.15e-3,
        );
    }

    for (axis, name) in ["reference_column", "reference_row"]
        .into_iter()
        .enumerate()
    {
        let mut reference = [1.0, 1.0];
        reference[axis] = 1.06;
        let reference_spec =
            CalibrationParameterSpec::new(0.8, 1.2, 0.06).finite_difference_step(1e-3);
        let mut specs = [None, None];
        specs[axis] = Some(reference_spec);
        assert_geometry_parameter_recovers(
            name,
            array([0.0, 0.0, -80e-3]).with_reference_index((reference[0], reference[1])),
            PlanarArrayCalibrationParameters::builder()
                .reference_index_specs(specs)
                .build()
                .unwrap(),
            if axis == 0 {
                |values: &PlanarArrayParameterValues| values.reference_index[0]
            } else {
                |values: &PlanarArrayParameterValues| values.reference_index[1]
            },
            1.06,
        );
    }

    let mut offsets = vec![[0.0; 3]; 9];
    offsets[0][0] = 0.25e-3;
    let offset_spec =
        CalibrationParameterSpec::new(-0.6e-3, 0.6e-3, 0.25e-3).finite_difference_step(2e-5);
    assert_geometry_parameter_recovers(
        "selected_offset_x",
        array([0.0, 0.0, -80e-3]).with_position_offsets_m(offsets),
        PlanarArrayCalibrationParameters::builder()
            .position_offset_specs(BTreeMap::from([(
                0,
                [offset_spec.clone(), offset_spec.clone(), offset_spec],
            )]))
            .build()
            .unwrap(),
        |values| values.position_offsets_m[0][0],
        0.25e-3,
    );
}

#[test]
fn combined_identifiable_geometry_calibration_reduces_scaled_parameter_error() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let truth = [0.32e-3, -0.24e-3, 0.009, 4.12e-3];
    let true_geometry = array([0.0, 0.0, -80e-3])
        .with_pose(ArrayPose::from_translation_and_extrinsic_xyz_radians(
            [truth[0], truth[1], -80e-3],
            [0.0, 0.0, truth[2]],
        ))
        .with_pitch_m((truth[3], 4e-3));
    let true_illumination = Illumination::from_geometry(true_geometry).unwrap();
    let (measurements, nominal_model, object) = simulate_pair(&true_illumination, &nominal);
    let (problem, state) = fixed_truth_state(measurements, nominal_model, object);
    let mut model = problem.model.clone();
    let translation =
        CalibrationParameterSpec::new(-1e-3, 1e-3, 0.3e-3).finite_difference_step(2e-5);
    let rotation = CalibrationParameterSpec::new(-0.03, 0.03, 0.01).finite_difference_step(2e-4);
    let pitch = CalibrationParameterSpec::new(3.6e-3, 4.4e-3, 0.12e-3).finite_difference_step(2e-6);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(translation.clone()), Some(translation), None])
        .rotation_specs([None, None, Some(rotation)])
        .pitch_specs([Some(pitch), None])
        .build()
        .unwrap();
    let calibration =
        IlluminationCalibration::new(parameters).optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 5,
            initial_step_size: 0.5,
            relative_tolerance: 0.0,
            ..BoundedFiniteDifferenceOptimizer::default()
        });
    let result = calibration
        .calibrate(
            &problem.measurements,
            &optics(),
            &nominal,
            state.object_spectrum(),
            state.pupil(),
            &mut model,
            4,
        )
        .unwrap();
    let initial_error = (truth[0] / 0.3e-3).powi(2)
        + (truth[1] / 0.3e-3).powi(2)
        + (truth[2] / 0.01).powi(2)
        + ((truth[3] - 4e-3) / 0.12e-3).powi(2);
    let values = result.current_parameters;
    let final_error = ((values.translation_m[0] - truth[0]) / 0.3e-3).powi(2)
        + ((values.translation_m[1] - truth[1]) / 0.3e-3).powi(2)
        + ((values.rotation_rad[2] - truth[2]) / 0.01).powi(2)
        + ((values.pitch_m[0] - truth[3]) / 0.12e-3).powi(2);
    assert!(
        final_error < initial_error,
        "{final_error} !< {initial_error}"
    );
}

#[test]
fn one_sided_differences_are_deterministic_at_bounds_and_invalid_perturbations() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let true_illumination = Illumination::from_geometry(array([0.3e-3, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, object) = simulate_pair(&true_illumination, &nominal);
    let (problem, state) = fixed_truth_state(measurements, nominal_model, object);
    let spec = CalibrationParameterSpec::new(0.0, 1e-3, 0.3e-3)
        .finite_difference_step(2e-5)
        .prior(0.3e-3, 1e-8);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(spec), None, None])
        .build()
        .unwrap();
    let calibration =
        IlluminationCalibration::new(parameters).optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 3,
            initial_step_size: 0.5,
            relative_tolerance: 0.0,
            ..BoundedFiniteDifferenceOptimizer::default()
        });
    let run = || {
        let mut model = problem.model.clone();
        calibration
            .calibrate(
                &problem.measurements,
                &optics(),
                &nominal,
                state.object_spectrum(),
                state.pupil(),
                &mut model,
                2,
            )
            .unwrap()
    };
    let first = run();
    let second = run();
    assert!(first.current_parameters.translation_m[0] > 0.0);
    assert_eq!(first.current_parameters, second.current_parameters);
    assert_eq!(first.parameter_history, second.parameter_history);

    let axial_truth = Illumination::from_geometry(array([0.0, 0.0, -75e-3])).unwrap();
    let (measurements, nominal_model, object) = simulate_pair(&axial_truth, &nominal);
    let (problem, state) = fixed_truth_state(measurements, nominal_model, object);
    let initial_loss = amplitude_loss(&problem.measurements, &problem.model, &state);
    let axial = CalibrationParameterSpec::new(-0.1, 0.0, 5e-3).finite_difference_step(80e-3);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([None, None, Some(axial)])
        .build()
        .unwrap();
    let mut model = problem.model.clone();
    IlluminationCalibration::new(parameters)
        .optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 3,
            initial_step_size: 0.5,
            relative_tolerance: 0.0,
            ..BoundedFiniteDifferenceOptimizer::default()
        })
        .calibrate(
            &problem.measurements,
            &optics(),
            &nominal,
            state.object_spectrum(),
            state.pupil(),
            &mut model,
            2,
        )
        .unwrap();
    assert!(amplitude_loss(&problem.measurements, &model, &state) < initial_loss);
}

#[test]
fn calibration_objective_honors_frame_weights_and_pixel_masks() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let true_illumination = Illumination::from_geometry(array([0.3e-3, 0.0, -80e-3])).unwrap();
    let (mut reference, nominal_model, object) = simulate_pair(&true_illumination, &nominal);
    reference.set_frame_weight(0, 0.0).unwrap();
    let mut mask = ndarray::Array2::ones((12, 12));
    mask[(0, 0)] = 0;
    let reference = reference.with_masks(mask).unwrap();
    let mut corrupted = reference.clone();
    corrupted.frame_mut(0).unwrap().fill(1e12);
    for frame in 1..corrupted.frame_count() {
        corrupted.frame_mut(frame).unwrap()[0] = 1e12;
    }
    let spec = CalibrationParameterSpec::new(-1e-3, 1e-3, 0.3e-3).finite_difference_step(2e-5);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(spec), None, None])
        .build()
        .unwrap();
    let calibration = IlluminationCalibration::new(parameters);
    let run = |measurements: &MeasurementStack| {
        let problem = ReconstructionProblem::new(measurements, nominal_model.clone()).unwrap();
        let state = ReconstructionState::from_object(&problem, object.clone()).unwrap();
        let mut model = nominal_model.clone();
        calibration
            .calibrate(
                measurements,
                &optics(),
                &nominal,
                state.object_spectrum(),
                state.pupil(),
                &mut model,
                2,
            )
            .unwrap()
    };
    let clean_result = run(&reference);
    let corrupted_result = run(&corrupted);
    assert_eq!(
        clean_result.current_parameters,
        corrupted_result.current_parameters
    );
    assert_eq!(clean_result.loss_history, corrupted_result.loss_history);
}

#[test]
fn quadratic_prior_uses_configured_center_scale_and_strength() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, object) = simulate_pair(&nominal, &nominal);
    let (problem, state) = fixed_truth_state(measurements, nominal_model, object);
    let spec = CalibrationParameterSpec::new(-1e-3, 1e-3, 0.4e-3)
        .finite_difference_step(2e-5)
        .prior(0.4e-3, 1.0);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(spec), None, None])
        .build()
        .unwrap();
    let mut model = problem.model.clone();
    let result = IlluminationCalibration::new(parameters)
        .optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 4,
            initial_step_size: 0.5,
            relative_tolerance: 0.0,
            ..BoundedFiniteDifferenceOptimizer::default()
        })
        .calibrate(
            &problem.measurements,
            &optics(),
            &nominal,
            state.object_spectrum(),
            state.pupil(),
            &mut model,
            2,
        )
        .unwrap();
    assert!(result.current_parameters.translation_m[0] > 0.0);
    let final_loss = result.loss_history.last().unwrap();
    assert!(final_loss.total_loss < 0.5);
    assert!(final_loss.regularization_loss < 0.5);
}

#[test]
fn source_power_calibration_is_normalized_and_avoids_geometry_recompilation() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let true_power = vec![0.7, 0.8, 0.9, 1.0, 1.3, 1.2, 1.1, 1.05, 0.95];
    let mean = true_power.iter().sum::<f64>() / true_power.len() as f64;
    let true_power: Vec<_> = true_power.into_iter().map(|value| value / mean).collect();
    let true_illumination = nominal
        .clone()
        .with_calibration(SourceCalibration::new(Some(true_power.clone())));
    let (measurements, nominal_model, object) = simulate_pair(&true_illumination, &nominal);
    let (problem, state) = fixed_truth_state(measurements, nominal_model, object);
    let mut calibrated_model = problem.model.clone();
    let vectors = calibrated_model.k_vectors().to_vec();
    let initial_loss = amplitude_loss(&problem.measurements, &calibrated_model, &state);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .relative_source_power(true)
        .build()
        .unwrap();
    let calibration =
        IlluminationCalibration::new(parameters).optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 3,
            initial_step_size: 0.4,
            ..BoundedFiniteDifferenceOptimizer::default()
        });
    let result = calibration
        .calibrate(
            &problem.measurements,
            &optics(),
            &nominal,
            state.object_spectrum(),
            state.pupil(),
            &mut calibrated_model,
            2,
        )
        .unwrap();
    let recovered_mean = result
        .current_parameters
        .relative_source_power
        .iter()
        .sum::<f64>()
        / result.current_parameters.relative_source_power.len() as f64;
    assert!((recovered_mean - 1.0).abs() < 1e-12);
    assert_eq!(result.geometry_recompilations, 0);
    assert!(result.multiplicative_updates > 0);
    assert_eq!(calibrated_model.k_vectors(), vectors);
    assert!(amplitude_loss(&problem.measurements, &calibrated_model, &state) < initial_loss);
    let initial_error = true_power
        .iter()
        .map(|truth| (1.0 - truth).powi(2))
        .sum::<f64>();
    let final_error = result
        .current_parameters
        .relative_source_power
        .iter()
        .zip(&true_power)
        .map(|(value, truth)| (value - truth).powi(2))
        .sum::<f64>();
    assert!(final_error < initial_error);
}

#[test]
fn frame_gain_calibration_is_normalized_and_avoids_geometry_recompilation() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let raw_gains = [0.7, 0.8, 0.9, 1.0, 1.3, 1.2, 1.1, 1.05, 0.95];
    let mean = raw_gains.iter().sum::<f64>() / raw_gains.len() as f64;
    let frames = nominal
        .acquisition()
        .frames()
        .iter()
        .zip(raw_gains)
        .map(|(frame, gain)| {
            fpm_rs::experiment::IlluminationFrame::new(frame.contributions.clone(), gain / mean)
        })
        .collect();
    let true_illumination = nominal
        .clone()
        .with_acquisition(fpm_rs::experiment::AcquisitionPlan::from_sparse(frames).unwrap());
    let (measurements, nominal_model, object) = simulate_pair(&true_illumination, &nominal);
    let (problem, state) = fixed_truth_state(measurements, nominal_model, object);
    let mut calibrated_model = problem.model.clone();
    let vectors = calibrated_model.k_vectors().to_vec();
    let initial_loss = amplitude_loss(&problem.measurements, &calibrated_model, &state);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .frame_gains(true)
        .build()
        .unwrap();
    let calibration =
        IlluminationCalibration::new(parameters).optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 3,
            initial_step_size: 0.4,
            ..BoundedFiniteDifferenceOptimizer::default()
        });
    let result = calibration
        .calibrate(
            &problem.measurements,
            &optics(),
            &nominal,
            state.object_spectrum(),
            state.pupil(),
            &mut calibrated_model,
            2,
        )
        .unwrap();
    let recovered_mean = result.current_parameters.frame_gains.iter().sum::<f64>()
        / result.current_parameters.frame_gains.len() as f64;
    assert!((recovered_mean - 1.0).abs() < 1e-12);
    assert_eq!(result.geometry_recompilations, 0);
    assert!(result.multiplicative_updates > 0);
    assert_eq!(calibrated_model.k_vectors(), vectors);
    assert!(amplitude_loss(&problem.measurements, &calibrated_model, &state) < initial_loss);
    let true_gains: Vec<_> = raw_gains.into_iter().map(|gain| gain / mean).collect();
    let initial_error = true_gains
        .iter()
        .map(|truth| (1.0 - truth).powi(2))
        .sum::<f64>();
    let final_error = result
        .current_parameters
        .frame_gains
        .iter()
        .zip(true_gains)
        .map(|(value, truth)| (value - truth).powi(2))
        .sum::<f64>();
    assert!(final_error < initial_error);
}

#[test]
fn pitch_and_axial_distance_emit_identifiability_warning() {
    let illumination = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation([false, false, true])
        .pitch([true, false])
        .build()
        .unwrap();
    let calibration = IlluminationCalibration::new(parameters);
    let initialized = calibration.initialize(&illumination, &optics()).unwrap();
    assert!(initialized.conditioning.warnings[0].contains("weakly identifiable"));
}

#[test]
fn joint_reconstruction_returns_reusable_physical_state() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let true_illumination = Illumination::from_geometry(array([0.25e-3, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, _) = simulate_pair(&true_illumination, &nominal);
    let problem = ReconstructionProblem::new(measurements, nominal_model).unwrap();
    let spec = CalibrationParameterSpec::new(-1e-3, 1e-3, 2.5e-4).finite_difference_step(2e-5);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(spec), None, None])
        .build()
        .unwrap();
    let joint = JointReconstruction::new(
        Fpie::default().iterations(1),
        optics(),
        nominal,
        IlluminationCalibration::new(parameters),
        1,
    );
    let result = joint.run(&problem).unwrap();
    result.calibrated_illumination.resolve(&optics()).unwrap();
    result.calibrated_model.validate().unwrap();
    assert!(
        result
            .reconstruction
            .physical_illumination_calibration
            .is_some()
    );
    let encoded = serde_json::to_vec(&result).unwrap();
    let decoded: fpm_rs::algorithms::JointReconstructionResult =
        serde_json::from_slice(&encoded).unwrap();
    assert_eq!(
        decoded.final_parameters.translation_m,
        result.final_parameters.translation_m
    );
}

#[test]
fn joint_reconstruction_rejects_mpie_model_recompilation() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, _) = simulate_pair(&nominal, &nominal);
    let problem = ReconstructionProblem::new(measurements, nominal_model).unwrap();
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation([true, false, false])
        .build()
        .unwrap();
    let error = JointReconstruction::new(
        Mpie::default().iterations(1),
        optics(),
        nominal,
        IlluminationCalibration::new(parameters),
        1,
    )
    .run(&problem)
    .unwrap_err();
    assert!(matches!(
        error,
        fpm_rs::Error::InvalidParameter {
            name: "object_algorithm",
            ..
        }
    ));
}

#[test]
fn joint_reconstruction_rejects_adaptive_projection_feedback_history() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, _) = simulate_pair(&nominal, &nominal);
    let problem = ReconstructionProblem::new(measurements, nominal_model).unwrap();
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation([true, false, false])
        .build()
        .unwrap();
    let error = JointReconstruction::new(
        AdaptiveAlternatingProjection::default().iterations(1),
        optics(),
        nominal,
        IlluminationCalibration::new(parameters),
        1,
    )
    .run(&problem)
    .unwrap_err();
    assert!(matches!(
        error,
        fpm_rs::Error::InvalidParameter {
            name: "object_algorithm",
            ..
        }
    ));
}

#[test]
fn joint_calibration_improves_object_quality_over_the_nominal_model() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let true_illumination = Illumination::from_geometry(array([0.7e-3, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, truth) = simulate_pair(&true_illumination, &nominal);
    let problem = ReconstructionProblem::new(measurements, nominal_model).unwrap();
    let baseline = Fpie::default().iterations(6).run(&problem).unwrap();
    let spec = CalibrationParameterSpec::new(-1.5e-3, 1.5e-3, 0.5e-3).finite_difference_step(2e-5);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(spec), None, None])
        .build()
        .unwrap();
    let calibrated = JointReconstruction::new(
        Fpie::default().iterations(1),
        optics(),
        nominal,
        IlluminationCalibration::new(parameters).optimizer(BoundedFiniteDifferenceOptimizer {
            max_steps: 2,
            initial_step_size: 0.5,
            relative_tolerance: 0.0,
            ..BoundedFiniteDifferenceOptimizer::default()
        }),
        6,
    )
    .run(&problem)
    .unwrap();
    let baseline_error = compare_complex_fields(truth.view(), baseline.object.view())
        .unwrap()
        .complex_nrmse;
    let calibrated_error =
        compare_complex_fields(truth.view(), calibrated.reconstruction.object.view())
            .unwrap()
            .complex_nrmse;
    assert!(
        calibrated_error < baseline_error,
        "calibrated object error {calibrated_error} !< nominal {baseline_error}"
    );
}

#[test]
fn joint_calibration_resumes_deterministically_from_checkpoint() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let true_illumination = Illumination::from_geometry(array([0.25e-3, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, _) = simulate_pair(&true_illumination, &nominal);
    let problem = ReconstructionProblem::new(measurements, nominal_model).unwrap();
    let make_joint = |outer_iterations| {
        let spec = CalibrationParameterSpec::new(-1e-3, 1e-3, 2.5e-4).finite_difference_step(2e-5);
        let parameters = PlanarArrayCalibrationParameters::builder()
            .translation_specs([Some(spec), None, None])
            .build()
            .unwrap();
        JointReconstruction::new(
            Fpie::default().iterations(1),
            optics(),
            nominal.clone(),
            IlluminationCalibration::new(parameters),
            outer_iterations,
        )
    };
    let uninterrupted = make_joint(2).run(&problem).unwrap();
    let directory = tempfile::tempdir().unwrap();
    make_joint(1)
        .run_with_callbacks(
            &problem,
            vec![Box::new(CheckpointEvery::new(1, directory.path()))],
        )
        .unwrap();
    let checkpoint =
        ReconstructionCheckpoint::load(directory.path().join("checkpoint_00001.json")).unwrap();
    assert!(checkpoint.physical_illumination_calibration().is_some());
    assert!(checkpoint.calibrated_model().is_some());
    let mismatched_parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([
            None,
            Some(CalibrationParameterSpec::new(-1e-3, 1e-3, 2.5e-4).finite_difference_step(2e-5)),
            None,
        ])
        .build()
        .unwrap();
    assert!(
        JointReconstruction::new(
            Fpie::default().iterations(1),
            optics(),
            nominal.clone(),
            IlluminationCalibration::new(mismatched_parameters),
            2,
        )
        .run_from_checkpoint(&problem, checkpoint.clone())
        .is_err()
    );
    let resumed = make_joint(2)
        .run_from_checkpoint(&problem, checkpoint)
        .unwrap();
    assert_eq!(
        resumed.final_parameters.translation_m,
        uninterrupted.final_parameters.translation_m
    );
    let maximum_difference = resumed
        .reconstruction
        .object_spectrum
        .iter()
        .zip(&uninterrupted.reconstruction.object_spectrum)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max);
    assert!(maximum_difference < 1e-15, "{maximum_difference}");
}

#[cfg(feature = "parquet")]
#[test]
fn physical_calibration_round_trips_through_result_bundle() {
    let nominal = Illumination::from_geometry(array([0.0, 0.0, -80e-3])).unwrap();
    let true_illumination = Illumination::from_geometry(array([0.2e-3, 0.0, -80e-3])).unwrap();
    let (measurements, nominal_model, _) = simulate_pair(&true_illumination, &nominal);
    let problem = ReconstructionProblem::new(measurements, nominal_model).unwrap();
    let spec = CalibrationParameterSpec::new(-1e-3, 1e-3, 2.5e-4).finite_difference_step(2e-5);
    let parameters = PlanarArrayCalibrationParameters::builder()
        .translation_specs([Some(spec), None, None])
        .build()
        .unwrap();
    let result = JointReconstruction::new(
        Fpie::default().iterations(1),
        optics(),
        nominal,
        IlluminationCalibration::new(parameters),
        1,
    )
    .run(&problem)
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let bundle = result
        .write_bundle(
            directory.path().join("joint-bundle"),
            fpm_rs::reconstruction::BundleExportOptions::default(),
        )
        .unwrap();
    let reopened = bundle.result().unwrap();
    assert!(reopened.physical_illumination_calibration.is_some());
    assert!(reopened.calibrated_model.is_some());
    assert_eq!(
        reopened
            .physical_illumination_calibration
            .as_ref()
            .unwrap()
            .current_parameters
            .translation_m,
        result.final_parameters.translation_m
    );
}
