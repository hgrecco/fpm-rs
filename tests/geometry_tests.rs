use approx::assert_abs_diff_eq;
use fpm_rs::{
    Array2, Complex64,
    experiment::{
        AngleList, CodedIllumination, IlluminationSource, KVector, LEDArray, LEDSphere, Optics,
        PupilAberration, RotatingLEDArc, SphericalLEDArm,
    },
    model::{CropIndices, FourierCrop, FourierOffset, ImagePlaneModel, Pupil, Sampling},
};

fn optics() -> Optics {
    Optics {
        wavelength: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        medium_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    }
}

#[test]
fn led_geometry_is_centered_and_symmetric() {
    let array = LEDArray::new()
        .grid_shape((1, 3))
        .pitch(4e-3)
        .distance(90e-3)
        .center((1.0, 0.0));
    let vectors = array.k_vectors(&optics()).unwrap();
    assert_abs_diff_eq!(vectors[1].kx, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(vectors[0].kx, -vectors[2].kx, epsilon = 1e-9);
    assert!(vectors.iter().all(|vector| vector.ky.abs() < 1e-12));
}

#[test]
fn led_geometry_applies_pitch_distance_and_rotation_directly() {
    let base = LEDArray::new()
        .grid_shape((1, 2))
        .pitch(4e-3)
        .distance(90e-3)
        .center((0.0, 0.0));
    let wider = base.clone().pitch(5e-3);
    let farther = base.clone().distance(120e-3);
    let rotated = base.clone().rotation_deg(90.0);

    let base_vector = base.k_vectors(&optics()).unwrap()[1];
    let wider_vector = wider.k_vectors(&optics()).unwrap()[1];
    let farther_vector = farther.k_vectors(&optics()).unwrap()[1];
    let rotated_vector = rotated.k_vectors(&optics()).unwrap()[1];

    assert!(wider_vector.kx > base_vector.kx);
    assert!(farther_vector.kx < base_vector.kx);
    assert_abs_diff_eq!(rotated_vector.kx, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(rotated_vector.ky, base_vector.kx, epsilon = 1e-9);
}

#[test]
fn led_sphere_uses_polar_and_azimuth_angles() {
    let theta = 0.3;
    let sphere = LEDSphere::new(
        vec![
            (0.0, 0.0),
            (theta, 0.0),
            (theta, std::f64::consts::FRAC_PI_2),
        ],
        100e-3,
    );
    let vectors = sphere.k_vectors(&optics()).unwrap();
    let k = optics().medium_wavenumber();

    assert_abs_diff_eq!(vectors[0].kx, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(vectors[0].ky, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(vectors[1].kx, k * theta.sin(), epsilon = 1e-9);
    assert_abs_diff_eq!(vectors[1].ky, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(vectors[2].kx, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(vectors[2].ky, k * theta.sin(), epsilon = 1e-9);

    let different_radius = LEDSphere::new(vec![(theta, 0.0)], 250e-3)
        .k_vectors(&optics())
        .unwrap();
    assert_abs_diff_eq!(different_radius[0].kx, vectors[1].kx, epsilon = 1e-9);
}

#[test]
fn led_sphere_models_pose_and_individual_placement_errors() {
    let nominal = LEDSphere::new(vec![(0.2, 0.0)], 100e-3);
    let corrected = nominal.clone().angular_corrections(vec![(0.01, 0.02)]);
    let decentered = nominal.clone().center_offset((1e-3, -2e-3, 0.5e-3));
    let rotated = nominal.clone().orientation_deg((0.0, 0.0, 90.0));

    let nominal_vector = nominal.k_vectors(&optics()).unwrap()[0];
    let corrected_vector = corrected.k_vectors(&optics()).unwrap()[0];
    let decentered_vector = decentered.k_vectors(&optics()).unwrap()[0];
    let rotated_vector = rotated.k_vectors(&optics()).unwrap()[0];

    assert_ne!(corrected_vector, nominal_vector);
    assert_ne!(decentered_vector, nominal_vector);
    assert_abs_diff_eq!(rotated_vector.kx, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(rotated_vector.ky, nominal_vector.kx, epsilon = 1e-9);
}

#[test]
fn ideal_spherical_arm_matches_fixed_sphere() {
    let angles = vec![(0.0, 0.0), (0.15, -0.2), (0.3, 0.7)];
    let sphere = LEDSphere::new(angles.clone(), 90e-3);
    let arm = SphericalLEDArm::new(angles, 90e-3);
    let sphere_vectors = sphere.k_vectors(&optics()).unwrap();
    let arm_vectors = arm.k_vectors(&optics()).unwrap();

    for (sphere, arm) in sphere_vectors.iter().zip(arm_vectors) {
        assert_abs_diff_eq!(sphere.kx, arm.kx, epsilon = 1e-9);
        assert_abs_diff_eq!(sphere.ky, arm.ky, epsilon = 1e-9);
    }
}

#[test]
fn spherical_arm_models_encoder_axis_pose_and_backlash_errors() {
    let commands = vec![(0.2, 0.0), (0.3, 0.0), (0.2, 0.0)];
    let nominal = SphericalLEDArm::new(commands.clone(), 100e-3);
    let backlash = SphericalLEDArm::new(commands, 100e-3).backlash_deg(2.0, 0.0);
    let misaligned = nominal
        .clone()
        .encoder_zero_deg(0.5, -0.25)
        .encoder_scale(1.01, 0.99)
        .elevation_axis_tilt_deg(1.0)
        .pivot_offset((0.5e-3, -0.25e-3, 0.0))
        .orientation_deg((0.1, -0.2, 0.3));

    let nominal_vectors = nominal.k_vectors(&optics()).unwrap();
    let backlash_vectors = backlash.k_vectors(&optics()).unwrap();
    let misaligned_vectors = misaligned.k_vectors(&optics()).unwrap();

    // The first point has no known approach direction; increasing/decreasing
    // moves occupy opposite sides of the backlash dead band.
    assert_abs_diff_eq!(
        backlash_vectors[0].kx,
        nominal_vectors[0].kx,
        epsilon = 1e-9
    );
    assert!(backlash_vectors[1].kx > nominal_vectors[1].kx);
    assert!(backlash_vectors[2].kx < nominal_vectors[2].kx);
    assert_ne!(misaligned_vectors, nominal_vectors);
}

#[test]
fn spherical_geometries_validate_angles_and_error_parameters() {
    assert!(LEDSphere::new(Vec::new(), 0.1).validate().is_err());
    assert!(
        LEDSphere::new(vec![(0.1, 0.0)], 0.1)
            .angular_corrections(vec![])
            .validate()
            .is_err()
    );
    assert!(
        SphericalLEDArm::new(vec![(0.1, 0.0)], 0.1)
            .encoder_scale(0.0, 1.0)
            .validate()
            .is_err()
    );
    assert!(
        SphericalLEDArm::new(vec![(0.1, 0.0)], 0.1)
            .backlash_deg(-1.0, 0.0)
            .validate()
            .is_err()
    );
}

#[test]
fn rotating_led_arc_compiles_rotation_major_sources_and_led_gains() {
    let theta = 0.2;
    let arc = RotatingLEDArc::new(
        vec![0.0, theta],
        vec![0.0, std::f64::consts::FRAC_PI_2],
        100e-3,
    )
    .led_intensity_weights(vec![0.5, 2.0]);
    let vectors = arc.k_vectors(&optics()).unwrap();
    let k_transverse = optics().medium_wavenumber() * theta.sin();

    assert_eq!(vectors.len(), 4);
    assert_abs_diff_eq!(vectors[0].kx, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(vectors[0].ky, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(vectors[1].kx, k_transverse, epsilon = 1e-9);
    assert_abs_diff_eq!(vectors[1].ky, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(vectors[2].kx, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(vectors[2].ky, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(vectors[3].kx, 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(vectors[3].ky, k_transverse, epsilon = 1e-9);
    assert_eq!(arc.frame_gains().unwrap(), Some(vec![0.5, 2.0, 0.5, 2.0]));
}

#[test]
fn rotating_led_arc_models_axis_encoder_backlash_and_led_placement_errors() {
    let nominal = RotatingLEDArc::new(vec![0.2], vec![0.0, 0.3, 0.0], 100e-3);
    let backlash = nominal.clone().rotation_backlash_deg(2.0);
    let perturbed = nominal
        .clone()
        .axis_origin_offset((0.5e-3, -0.2e-3, 0.1e-3))
        .axis_tilt_deg(0.2, -0.3)
        .rotation_encoder_zero_deg(0.4)
        .rotation_encoder_scale(1.01)
        .led_angular_corrections(vec![(0.002, -0.001)])
        .led_radial_offsets(vec![0.3e-3]);

    let nominal_vectors = nominal.k_vectors(&optics()).unwrap();
    let backlash_vectors = backlash.k_vectors(&optics()).unwrap();
    let perturbed_vectors = perturbed.k_vectors(&optics()).unwrap();

    assert_eq!(backlash_vectors[0], nominal_vectors[0]);
    assert!(backlash_vectors[1].ky > nominal_vectors[1].ky);
    assert!(backlash_vectors[2].ky < nominal_vectors[2].ky);
    assert_ne!(perturbed_vectors, nominal_vectors);
}

#[test]
fn rotating_led_arc_rejects_invalid_shape_and_calibration_parameters() {
    assert!(
        RotatingLEDArc::new(vec![], vec![0.0], 0.1)
            .validate()
            .is_err()
    );
    assert!(
        RotatingLEDArc::new(vec![0.1], vec![], 0.1)
            .validate()
            .is_err()
    );
    assert!(
        RotatingLEDArc::new(vec![0.1], vec![0.0], 0.1)
            .led_radial_offsets(vec![-0.1])
            .validate()
            .is_err()
    );
    assert!(
        RotatingLEDArc::new(vec![0.1], vec![0.0], 0.1)
            .rotation_encoder_scale(0.0)
            .validate()
            .is_err()
    );
}

#[test]
fn experiment_compiles_to_valid_model() {
    let array = LEDArray::new()
        .grid_shape((3, 3))
        .pitch(4e-3)
        .distance(90e-3)
        .center((1.0, 1.0));
    let model = ImagePlaneModel::from_experiment(&optics(), &array, (16, 16), (32, 32)).unwrap();
    assert_eq!(model.frame_count(), 9);
    assert_eq!(model.pupil.shape(), (16, 16));
    assert!(model.pupil.support.iter().any(|inside| *inside));
    assert!(
        model
            .crop_indices
            .crops
            .iter()
            .all(|crop| crop.validate_inside((32, 32)).is_ok())
    );
}

#[test]
fn pupil_aberration_and_apodization_compile_into_the_pupil() {
    let ideal_optics = optics();
    let mut aberrated_optics = ideal_optics.clone();
    aberrated_optics.defocus_distance = Some(-12e-6);
    aberrated_optics.pupil_aberration = Some(PupilAberration {
        astigmatism: 0.2,
        coma: 0.1,
        spherical: 0.05,
        edge_apodization: 0.4,
    });
    let illumination = LEDArray::new();
    let ideal =
        ImagePlaneModel::from_experiment(&ideal_optics, &illumination, (16, 16), (32, 32)).unwrap();
    let aberrated =
        ImagePlaneModel::from_experiment(&aberrated_optics, &illumination, (16, 16), (32, 32))
            .unwrap();

    assert_eq!(ideal.pupil.support, aberrated.pupil.support);
    assert_ne!(ideal.pupil.values, aberrated.pupil.values);
    assert!(
        ideal
            .pupil
            .values
            .as_slice()
            .iter()
            .zip(aberrated.pupil.values.as_slice())
            .zip(&ideal.pupil.support)
            .all(|((&ideal, &aberrated), &inside)| !inside || aberrated.norm() <= ideal.norm())
    );
    assert!(
        aberrated
            .pupil
            .values
            .as_slice()
            .iter()
            .zip(&aberrated.pupil.support)
            .any(|(&value, &inside)| inside && value.norm() < 0.99)
    );
}

#[test]
fn experiment_compilation_preserves_fractional_fourier_shifts() {
    let optics = optics();
    let low_res_pixel_size = optics.object_pixel_size();
    let dk = std::f64::consts::TAU / (16.0 * low_res_pixel_size);
    let model = ImagePlaneModel::from_experiment(
        &optics,
        &vec![KVector::new(0.25 * dk, -0.4 * dk)],
        (16, 16),
        (32, 32),
    )
    .unwrap();
    let offset = model.source_offset(0).unwrap();
    assert_abs_diff_eq!(offset.row, -0.4, epsilon = 1e-12);
    assert_abs_diff_eq!(offset.column, 0.25, epsilon = 1e-12);
}

#[test]
fn invalid_crop_is_rejected() {
    let sampling = Sampling::new(1.0, 0.5, 1.0, 1.0).unwrap();
    let pupil = Pupil::new(
        Array2::filled((4, 4), Complex64::new(1.0, 0.0)).unwrap(),
        vec![true; 16],
    )
    .unwrap();
    let result = ImagePlaneModel::new(
        vec![KVector::default()],
        pupil,
        CropIndices::new(vec![FourierCrop::new(6, 6, 4, 4)]),
        sampling,
        (4, 4),
        (8, 8),
    );
    assert!(result.is_err());
}

#[test]
fn non_finite_model_parameters_are_rejected() {
    let mut model =
        ImagePlaneModel::from_experiment(&optics(), &LEDArray::new(), (8, 8), (16, 16)).unwrap();
    model.frame_gains = Some(vec![f64::NAN]);
    assert!(model.validate().is_err());

    model.frame_gains = None;
    model.pupil.values.as_mut_slice()[0] = Complex64::new(f64::INFINITY, 0.0);
    assert!(model.validate().is_err());

    let mut model =
        ImagePlaneModel::from_experiment(&optics(), &vec![KVector::default()], (8, 8), (16, 16))
            .unwrap();
    model.subpixel_offsets = Some(vec![FourierOffset::new(f64::NAN, 0.0)]);
    assert!(model.validate().is_err());

    model.subpixel_offsets = Some(Vec::new());
    assert!(model.source_offset(0).is_err());

    assert!(
        ImagePlaneModel::from_experiment(
            &optics(),
            &vec![KVector::new(f64::MAX, 0.0)],
            (8, 8),
            (16, 16),
        )
        .is_err()
    );
}

#[test]
fn led_intensity_weights_compile_in_acquisition_order() {
    let array = LEDArray::new()
        .grid_shape((1, 3))
        .center((1.0, 0.0))
        .illumination_order(vec![2, 0, 1])
        .intensity_weights(vec![0.5, 1.0, 2.0]);
    let model = ImagePlaneModel::from_experiment(&optics(), &array, (16, 16), (32, 32)).unwrap();
    assert_eq!(model.frame_gains, Some(vec![2.0, 0.5, 1.0]));
}

#[test]
fn coded_illumination_compiles_sources_and_measured_frames_separately() {
    let coded = CodedIllumination {
        source_k_vectors: vec![
            KVector::new(0.0, 0.0),
            KVector::new(1.0e5, 0.0),
            KVector::new(0.0, 1.0e5),
        ],
        frame_weights: vec![vec![(0, 0.5), (1, 0.5)], vec![(2, 1.0)]],
    };
    let model = ImagePlaneModel::from_experiment(&optics(), &coded, (16, 16), (32, 32)).unwrap();
    assert_eq!(model.source_count(), 3);
    assert_eq!(model.frame_count(), 2);
    assert_eq!(model.crop_indices.len(), 3);
    assert!(model.is_multiplexed());
}

#[test]
fn experiment_validation_rejects_non_finite_and_non_propagating_inputs() {
    let mut invalid_optics = optics();
    invalid_optics.pupil_aberration = Some(PupilAberration {
        astigmatism: f64::NAN,
        ..PupilAberration::default()
    });
    assert!(invalid_optics.validate().is_err());
    assert!(
        ImagePlaneModel::from_experiment(&invalid_optics, &LEDArray::new(), (8, 8), (16, 16))
            .is_err()
    );

    let mut invalid_optics = optics();
    invalid_optics.pupil_aberration = Some(PupilAberration {
        edge_apodization: -0.1,
        ..PupilAberration::default()
    });
    assert!(invalid_optics.validate().is_err());

    let mut invalid_optics = optics();
    invalid_optics.defocus_distance = Some(f64::NAN);
    assert!(invalid_optics.validate().is_err());

    let optics = optics();
    let non_propagating_angles = AngleList::new(vec![(
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_3,
    )]);
    assert!(non_propagating_angles.k_vectors(&optics).is_err());
    assert!(
        vec![KVector::new(optics.medium_wavenumber() * 1.01, 0.0)]
            .k_vectors(&optics)
            .is_err()
    );
    assert!(Vec::<KVector>::new().k_vectors(&optics).is_err());
}

#[test]
fn coded_illumination_rejects_invalid_and_duplicate_sources() {
    let coded = CodedIllumination {
        source_k_vectors: vec![KVector::default()],
        frame_weights: vec![vec![(0, 0.5), (0, 0.5)]],
    };
    assert!(coded.validate(&optics()).is_err());
    assert!(coded.multiplexing_matrix().is_err());

    let model =
        ImagePlaneModel::from_experiment(&optics(), &vec![KVector::default()], (8, 8), (16, 16))
            .unwrap();
    assert!(
        model
            .with_multiplexing(vec![vec![(0, 0.5), (0, 0.5)]])
            .is_err()
    );
}
