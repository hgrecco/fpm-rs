mod common;

use fpm_rs::reconstruction::ReconstructionProblem;
use fpm_rs::simulation::{CameraModel, IlluminationAcquisitionErrors, Simulator, SyntheticObject};
use fpm_rs::{
    algorithms::{AlternatingProjection, ReconstructionAlgorithm},
    experiment::{LEDArray, Optics, PupilAberration},
    model::{ForwardModel, ImagePlaneModel, ReconstructionShape},
};
use image::GrayImage;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn ideal_simulation_uses_shared_forward_model() {
    let model = common::direct_model().unwrap();
    let object = SyntheticObject::phase_disk((16, 16), 4.0, 0.7).unwrap();
    let simulation = Simulator::ideal(model.clone())
        .object(object)
        .seed(7)
        .simulate()
        .unwrap();
    assert_eq!(simulation.measurements.frame_count(), model.frame_count());
    assert_eq!(simulation.measurements.image_shape(), model.image_shape);
    assert_eq!(simulation.true_model.pupil.values, model.pupil.values);
    assert_eq!(
        simulation.reconstruction_model.pupil.values,
        model.pupil.values
    );
    assert!(
        simulation
            .measurements
            .as_slice()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
}

#[test]
fn poisson_gaussian_noise_is_reproducible() {
    let model = common::direct_model().unwrap();
    let object = SyntheticObject::resolution_target((16, 16)).unwrap();
    let camera = CameraModel::new()
        .photons_per_pixel(200.0)
        .shot_noise(true)
        .read_noise_electrons(1.5)
        .bit_depth(12);
    let first = Simulator::new(model.clone())
        .object(object.clone())
        .camera(camera.clone())
        .seed(1234)
        .simulate()
        .unwrap();
    let second = Simulator::new(model)
        .object(object)
        .camera(camera)
        .seed(1234)
        .simulate()
        .unwrap();
    assert_eq!(
        first.measurements.as_slice(),
        second.measurements.as_slice()
    );
}

#[test]
fn camera_quantizes_and_saturates() {
    let model = common::direct_model().unwrap();
    let object = SyntheticObject::constant((16, 16), 10.0, 0.0).unwrap();
    let simulation = Simulator::new(model)
        .object(object)
        .camera(
            CameraModel::new()
                .photons_per_pixel(1_000.0)
                .bit_depth(8)
                .saturation(200.0),
        )
        .simulate()
        .unwrap();
    assert!(
        simulation
            .measurements
            .as_slice()
            .iter()
            .all(|value| value.fract() == 0.0 && *value <= 200.0)
    );
    assert!(simulation.measurements.as_slice().contains(&200.0));
}

#[test]
fn ideal_camera_is_an_identity_detector() {
    let camera = CameraModel::ideal();
    let mut frame = vec![0.0, 0.5, 2.0];
    camera
        .measure_frame(&mut frame, &mut StdRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(frame, vec![0.0, 0.5, 2.0]);
}

#[test]
fn camera_converts_intensity_with_pixel_sensitivity_and_electronics() {
    let camera = CameraModel::ideal()
        .photons_per_pixel(10.0)
        .pixel_sensitivity(vec![1.0, 0.5])
        .dark_current_electrons(3.0)
        .gain(2.0)
        .offset_counts(5.0);
    let mut frame = vec![2.0, 2.0];
    camera
        .measure_frame(&mut frame, &mut StdRng::seed_from_u64(2))
        .unwrap();
    assert_eq!(frame, vec![51.0, 31.0]);
}

#[test]
fn camera_noise_is_seeded_and_owned_by_the_camera() {
    let camera = CameraModel::ideal()
        .photons_per_pixel(50.0)
        .shot_noise(true)
        .read_noise_electrons(1.5);
    let mut first = vec![1.0; 16];
    let mut second = first.clone();
    camera
        .measure_frame(&mut first, &mut StdRng::seed_from_u64(3))
        .unwrap();
    camera
        .measure_frame(&mut second, &mut StdRng::seed_from_u64(3))
        .unwrap();
    assert_eq!(first, second);
    assert!(first.iter().any(|&value| value != 50.0));
}

#[test]
fn shot_noise_moments_match_the_poisson_model() {
    let expected_electrons = 100.0;
    let camera = CameraModel::ideal()
        .photons_per_pixel(expected_electrons)
        .shot_noise(true);
    let mut frame = vec![1.0; 65_536];
    camera
        .measure_frame(&mut frame, &mut StdRng::seed_from_u64(4))
        .unwrap();
    let (mean, variance) = sample_mean_and_variance(&frame);
    assert!((mean - expected_electrons).abs() < 0.5, "mean={mean}");
    assert!(
        (variance - expected_electrons).abs() < 4.0,
        "variance={variance}"
    );
}

#[test]
fn read_noise_variance_scales_through_detector_gain() {
    let expected_electrons = 50.0;
    let read_noise_electrons = 4.0;
    let gain = 2.0;
    let offset = 7.0;
    let camera = CameraModel::ideal()
        .dark_current_electrons(expected_electrons)
        .read_noise_electrons(read_noise_electrons)
        .gain(gain)
        .offset_counts(offset);
    let mut frame = vec![0.0; 65_536];
    camera
        .measure_frame(&mut frame, &mut StdRng::seed_from_u64(5))
        .unwrap();
    let (mean, variance) = sample_mean_and_variance(&frame);
    assert!((mean - (gain * expected_electrons + offset)).abs() < 0.35);
    assert!((variance - (gain * read_noise_electrons).powi(2)).abs() < 3.0);
}

#[test]
fn poisson_gaussian_moments_add_before_detector_gain() {
    let expected_electrons = 80.0;
    let read_noise_electrons = 3.0;
    let gain = 1.5;
    let offset = 11.0;
    let camera = CameraModel::ideal()
        .photons_per_pixel(expected_electrons)
        .shot_noise(true)
        .read_noise_electrons(read_noise_electrons)
        .gain(gain)
        .offset_counts(offset);
    let mut frame = vec![1.0; 65_536];
    camera
        .measure_frame(&mut frame, &mut StdRng::seed_from_u64(6))
        .unwrap();
    let (mean, variance) = sample_mean_and_variance(&frame);
    let expected_mean = gain * expected_electrons + offset;
    let expected_variance = gain * gain * (expected_electrons + read_noise_electrons.powi(2));
    assert!((mean - expected_mean).abs() < 0.6, "mean={mean}");
    assert!(
        (variance - expected_variance).abs() < 5.0,
        "variance={variance}"
    );
}

#[test]
fn camera_applies_clipping_then_quantization_before_bad_pixel_override() {
    let camera = CameraModel::ideal()
        .photons_per_pixel(10.0)
        .gain(2.0)
        .offset_counts(1.0)
        .bit_depth(4)
        .saturation(10.0)
        .quantize(true)
        .bad_pixels(vec![1], 7.4);
    let mut frame = vec![-1.0, 0.12, 0.55, 1.0];
    camera
        .measure_frame(&mut frame, &mut StdRng::seed_from_u64(7))
        .unwrap();
    assert_eq!(frame, vec![1.0, 7.0, 10.0, 10.0]);
}

fn sample_mean_and_variance(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (count - 1.0);
    (mean, variance)
}

#[test]
fn camera_validates_pixel_maps_and_bad_pixel_indices_for_the_frame() {
    let mut frame = vec![1.0; 2];
    assert!(
        CameraModel::ideal()
            .pixel_sensitivity(vec![1.0])
            .measure_frame(&mut frame, &mut StdRng::seed_from_u64(4))
            .is_err()
    );
    assert!(
        CameraModel::ideal()
            .bad_pixels(vec![2], 0.0)
            .measure_frame(&mut frame, &mut StdRng::seed_from_u64(4))
            .is_err()
    );
}

#[test]
fn illumination_mismatch_keeps_true_and_reconstruction_models_distinct() {
    let optics = Optics {
        wavelength: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        medium_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let assumed_array = LEDArray::new()
        .grid_shape((1, 3))
        .pitch(4.0e-3)
        .distance(90.0e-3)
        .center((1.0, 0.0));
    let true_array = LEDArray::new()
        .grid_shape((1, 3))
        .pitch(4.05e-3)
        .distance(89.5e-3)
        .center((1.08, -0.04))
        .rotation_deg(0.7);
    let true_model = ImagePlaneModel::from_experiment(
        &optics,
        &true_array,
        (8, 8),
        ReconstructionShape::Exact((16, 16)),
    )
    .unwrap();
    let reconstruction_model = ImagePlaneModel::from_experiment(
        &optics,
        &assumed_array,
        (8, 8),
        ReconstructionShape::Exact((16, 16)),
    )
    .unwrap();

    let simulation = Simulator::new(true_model.clone())
        .reconstruction_model(reconstruction_model.clone())
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .simulate()
        .unwrap();
    assert_ne!(true_model.k_vectors, reconstruction_model.k_vectors);
    assert_eq!(simulation.true_model.k_vectors, true_model.k_vectors);
    assert_eq!(
        simulation.reconstruction_model.k_vectors,
        reconstruction_model.k_vectors
    );
    assert!(simulation.illumination_acquisition_errors.is_none());
}

#[test]
fn pupil_mismatch_comes_from_the_provided_models() {
    let assumed_optics = Optics {
        wavelength: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        medium_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let true_optics = Optics {
        defocus_distance: Some(-12e-6),
        pupil_aberration: Some(PupilAberration {
            astigmatism: 0.2,
            spherical: 0.1,
            edge_apodization: 0.3,
            ..PupilAberration::default()
        }),
        ..assumed_optics.clone()
    };
    let illumination = LEDArray::new();
    let true_model = ImagePlaneModel::from_experiment(
        &true_optics,
        &illumination,
        (8, 8),
        ReconstructionShape::Exact((16, 16)),
    )
    .unwrap();
    let reconstruction_model = ImagePlaneModel::from_experiment(
        &assumed_optics,
        &illumination,
        (8, 8),
        ReconstructionShape::Exact((16, 16)),
    )
    .unwrap();

    let simulation = Simulator::new(true_model.clone())
        .reconstruction_model(reconstruction_model.clone())
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .simulate()
        .unwrap();

    assert_ne!(true_model.pupil.values, reconstruction_model.pupil.values);
    assert_eq!(simulation.true_model.pupil.values, true_model.pupil.values);
    assert_eq!(
        simulation.reconstruction_model.pupil.values,
        reconstruction_model.pupil.values
    );
}

#[test]
fn source_weights_encode_angle_dependent_transmission() {
    let mut reconstruction_model = common::direct_model().unwrap();
    reconstruction_model.frame_gains = Some(vec![2.0; reconstruction_model.frame_count()]);
    reconstruction_model.validate().unwrap();
    let mut true_model = reconstruction_model.clone();
    true_model.frame_gains = Some(vec![1.0, 1.0, 2.0, 1.0, 1.0]);
    true_model.validate().unwrap();
    let object = SyntheticObject::mixed_test_pattern((16, 16)).unwrap();
    let baseline = Simulator::ideal(reconstruction_model.clone())
        .object(object.clone())
        .simulate()
        .unwrap();
    let ordinary = Simulator::new(true_model)
        .reconstruction_model(reconstruction_model)
        .object(object)
        .simulate()
        .unwrap();
    let true_gains = ordinary.true_model.frame_gains.as_ref().unwrap();
    assert!((true_gains[2] - 2.0).abs() < 1e-14);
    for frame in [0, 1, 3, 4] {
        assert!((true_gains[frame] - 1.0).abs() < 1e-14);
        for (&vignetted, &unvignetted) in ordinary
            .measurements
            .frame(frame)
            .unwrap()
            .iter()
            .zip(baseline.measurements.frame(frame).unwrap())
        {
            assert!((vignetted - 0.5 * unvignetted).abs() < 1e-12);
        }
    }
    assert_eq!(
        ordinary.reconstruction_model.frame_gains,
        Some(vec![2.0; 5])
    );

    let reconstruction_model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![vec![(0, 1.0), (2, 1.0)], vec![(1, 0.25), (4, 0.75)]])
        .unwrap();
    let true_model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![vec![(0, 0.5), (2, 1.0)], vec![(1, 0.125), (4, 0.375)]])
        .unwrap();
    let multiplexed = Simulator::new(true_model)
        .reconstruction_model(reconstruction_model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .simulate()
        .unwrap();
    let true_matrix = multiplexed.true_model.multiplexing_matrix.as_ref().unwrap();
    assert!((true_matrix[0][0].1 - 0.5).abs() < 1e-14);
    assert!((true_matrix[0][1].1 - 1.0).abs() < 1e-14);
    assert!((true_matrix[1][0].1 - 0.125).abs() < 1e-14);
    assert!((true_matrix[1][1].1 - 0.375).abs() < 1e-14);
    assert_eq!(
        multiplexed.reconstruction_model.multiplexing_matrix,
        Some(vec![vec![(0, 1.0), (2, 1.0)], vec![(1, 0.25), (4, 0.75)]])
    );
}

#[test]
fn missing_frames_become_zero_weight_dark_frames() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::new(model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .illumination_acquisition_errors(
            IlluminationAcquisitionErrors::new().missing_frames(vec![1, 3]),
        )
        .simulate()
        .unwrap();
    assert_eq!(simulation.parameters.missing_frames, vec![1, 3]);
    for frame in [1, 3] {
        assert_eq!(simulation.measurements.frame_weight(frame).unwrap(), 0.0);
        assert!(
            simulation
                .measurements
                .frame(frame)
                .unwrap()
                .iter()
                .all(|&value| value == 0.0)
        );
    }
    ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model).unwrap();
}

#[test]
fn source_permutation_reorders_true_sources_only() {
    let model = common::direct_model().unwrap();
    let order = vec![4, 3, 2, 1, 0];
    let simulation = Simulator::new(model.clone())
        .object(SyntheticObject::phase_disk((16, 16), 4.0, 0.5).unwrap())
        .illumination_acquisition_errors(
            IlluminationAcquisitionErrors::new().source_permutation(order.clone()),
        )
        .simulate()
        .unwrap();
    for (frame, &source) in order.iter().enumerate() {
        assert_eq!(
            simulation.true_model.k_vectors[frame],
            model.k_vectors[source]
        );
    }
    assert_eq!(simulation.reconstruction_model.k_vectors, model.k_vectors);
    assert!(simulation.illumination_acquisition_errors.is_some());
}

#[test]
fn frame_gain_variation_multiplies_existing_source_gains() {
    let mut weighted_model = common::direct_model().unwrap();
    weighted_model.frame_gains = Some(vec![2.0; weighted_model.frame_count()]);
    weighted_model.validate().unwrap();
    let unweighted_model = common::direct_model().unwrap();
    let object = SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap();
    let errors = IlluminationAcquisitionErrors::new().frame_gain_relative_std(0.2);

    let weighted = Simulator::new(weighted_model)
        .object(object.clone())
        .illumination_acquisition_errors(errors.clone())
        .seed(19)
        .simulate()
        .unwrap();
    let unweighted = Simulator::new(unweighted_model)
        .object(object)
        .illumination_acquisition_errors(errors)
        .seed(19)
        .simulate()
        .unwrap();

    let weighted_gains = weighted.true_model.frame_gains.as_ref().unwrap();
    let unweighted_gains = unweighted.true_model.frame_gains.as_ref().unwrap();
    for (&weighted_gain, &unweighted_gain) in weighted_gains.iter().zip(unweighted_gains) {
        assert!((weighted_gain - 2.0 * unweighted_gain).abs() < 1e-12);
    }
    assert_eq!(
        weighted.reconstruction_model.frame_gains,
        Some(vec![2.0; weighted.reconstruction_model.frame_count()])
    );
}

#[test]
fn illumination_acquisition_errors_reject_invalid_parameters() {
    let object = || SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap();

    assert!(
        Simulator::new(common::direct_model().unwrap())
            .object(object())
            .illumination_acquisition_errors(
                IlluminationAcquisitionErrors::new().frame_gain_relative_std(-0.1),
            )
            .simulate()
            .is_err()
    );
    assert!(
        Simulator::new(common::direct_model().unwrap())
            .object(object())
            .illumination_acquisition_errors(
                IlluminationAcquisitionErrors::new().missing_frames(vec![1, 1]),
            )
            .simulate()
            .is_err()
    );
    assert!(
        Simulator::new(common::direct_model().unwrap())
            .object(object())
            .illumination_acquisition_errors(
                IlluminationAcquisitionErrors::new().source_permutation(vec![0, 1, 2, 3, 3]),
            )
            .simulate()
            .is_err()
    );
}

#[test]
fn camera_applies_dark_current_and_fixed_bad_pixels() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::new(model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .camera(
            CameraModel::new()
                .photons_per_pixel(10.0)
                .dark_current_electrons(7.0)
                .bad_pixels(vec![0, 5], 123.0)
                .bit_depth(16),
        )
        .simulate()
        .unwrap();
    for frame in 0..simulation.measurements.frame_count() {
        let values = simulation.measurements.frame(frame).unwrap();
        assert_eq!(values[0], 123.0);
        assert_eq!(values[5], 123.0);
        assert_eq!(values[1], 17.0);
    }
}

#[test]
fn optical_background_remains_in_the_model_and_detector_offset_stays_in_the_camera() {
    let mut true_model = common::direct_model().unwrap();
    true_model.background = Some(vec![2.0; 64]);
    true_model.validate().unwrap();
    let reconstruction_model = true_model.clone();
    let object = SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap();
    let optical = Simulator::ideal(true_model.clone())
        .object(object.clone())
        .simulate()
        .unwrap();
    let measured = Simulator::new(true_model.clone())
        .reconstruction_model(reconstruction_model)
        .object(object)
        .camera(
            CameraModel::ideal()
                .dark_current_electrons(3.0)
                .offset_counts(5.0),
        )
        .simulate()
        .unwrap();

    for (&counts, &intensity) in measured
        .measurements
        .as_slice()
        .iter()
        .zip(optical.measurements.as_slice())
    {
        assert!((counts - intensity - 8.0).abs() < 1e-12);
    }
    assert_eq!(measured.true_model.background, Some(vec![2.0; 64]));
    assert_eq!(
        measured.reconstruction_model.background,
        Some(vec![10.0; 64])
    );
}

#[test]
fn mixed_and_biological_objects_are_finite_and_reproducible() {
    let mixed = SyntheticObject::mixed_test_pattern((32, 32)).unwrap();
    assert!(
        mixed
            .field
            .as_slice()
            .iter()
            .any(|value| value.norm() < 0.9)
    );
    assert!(
        mixed
            .field
            .as_slice()
            .iter()
            .any(|value| value.arg().abs() > 0.1)
    );

    let first = SyntheticObject::biological_like((32, 32), 12, 99).unwrap();
    let second = SyntheticObject::biological_like((32, 32), 12, 99).unwrap();
    assert_eq!(first.field, second.field);
    assert!(
        first
            .field
            .as_slice()
            .iter()
            .all(|value| { value.re.is_finite() && value.im.is_finite() && value.norm() >= 0.1 })
    );
    assert!(SyntheticObject::biological_like((0, 32), 4, 1).is_err());
    assert!(SyntheticObject::particle_field((32, 0), 4, 1).is_err());
}

#[test]
fn multiplexed_simulation_is_an_incoherent_weighted_intensity_sum() {
    let base_model = common::direct_model().unwrap();
    let object = SyntheticObject::mixed_test_pattern((16, 16)).unwrap();
    let base = Simulator::ideal(base_model.clone())
        .object(object.clone())
        .simulate()
        .unwrap();
    let multiplexed_model = base_model
        .with_multiplexing(vec![vec![(0, 0.25), (2, 0.75)], vec![(1, 0.5), (4, 0.5)]])
        .unwrap();
    assert_eq!(multiplexed_model.source_count(), 5);
    assert_eq!(multiplexed_model.frame_count(), 2);
    let multiplexed = Simulator::ideal(multiplexed_model)
        .object(object)
        .simulate()
        .unwrap();
    for pixel in 0..base.measurements.frame_len() {
        let expected = 0.25 * base.measurements.frame(0).unwrap()[pixel]
            + 0.75 * base.measurements.frame(2).unwrap()[pixel];
        assert!((multiplexed.measurements.frame(0).unwrap()[pixel] - expected).abs() < 1e-12);
    }
    let problem =
        ReconstructionProblem::new(multiplexed.measurements, multiplexed.reconstruction_model)
            .unwrap();
    let reconstruction = AlternatingProjection::default()
        .iterations(2)
        .run(&problem)
        .unwrap();
    assert_eq!(reconstruction.history.iterations.len(), 2);
    let forward = ForwardModel::new(&problem.model).unwrap();
    assert!(
        forward
            .forward_field(
                &fpm_rs::Array2::filled(
                    problem.model.reconstruction_shape,
                    fpm_rs::Complex64::default(),
                )
                .unwrap(),
                &problem.model.pupil,
                0,
            )
            .is_err()
    );
}

#[test]
fn synthetic_object_import_maps_normalized_amplitude_and_phase() {
    let directory = tempfile::tempdir().unwrap();
    let amplitude_path = directory.path().join("amplitude.png");
    let phase_path = directory.path().join("phase.png");
    GrayImage::from_vec(2, 1, vec![0, 255])
        .unwrap()
        .save(&amplitude_path)
        .unwrap();
    GrayImage::from_vec(2, 1, vec![0, 255])
        .unwrap()
        .save(&phase_path)
        .unwrap();
    let object = SyntheticObject::from_amplitude_phase_images(
        &amplitude_path,
        &phase_path,
        std::f64::consts::PI,
    )
    .unwrap();
    assert!(object.field.as_slice()[0].norm() < 1e-12);
    assert!((object.field.as_slice()[1].norm() - 1.0).abs() < 1e-12);
    assert!((object.field.as_slice()[1].arg().abs() - std::f64::consts::PI).abs() < 1e-12);
}
