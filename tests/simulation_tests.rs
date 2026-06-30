mod common;

use fpm_rs::reconstruction::ReconstructionProblem;
use fpm_rs::simulation::{
    AberrationModel, CameraModel, Simulator, IlluminationErrorModel, NoiseModel, SyntheticObject,
};
use fpm_rs::{
    algorithms::{AlternatingProjection, ReconstructionAlgorithm},
    model::ForwardModel,
};
use image::GrayImage;

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
        .read_noise_electrons(1.5)
        .bit_depth(12);
    let first = Simulator::new(model.clone())
        .object(object.clone())
        .camera(camera.clone())
        .noise(NoiseModel::PoissonGaussian)
        .seed(1234)
        .simulate()
        .unwrap();
    let second = Simulator::new(model)
        .object(object)
        .camera(camera)
        .noise(NoiseModel::PoissonGaussian)
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
fn illumination_mismatch_keeps_true_and_reconstruction_models_distinct() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::new(model.clone())
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .illumination_errors(IlluminationErrorModel::new().global_shift((0.25, -0.4)))
        .seed(4)
        .simulate()
        .unwrap();
    assert_ne!(
        simulation.true_model.k_vectors[0].kx,
        simulation.reconstruction_model.k_vectors[0].kx
    );
    assert_eq!(simulation.reconstruction_model.k_vectors, model.k_vectors);
    let offset = simulation.true_model.source_offset(0).unwrap();
    assert!((offset.row + 0.4).abs() < 1e-12);
    assert!((offset.column - 0.25).abs() < 1e-12);
}

#[test]
fn illumination_vignetting_attenuates_frames_and_multiplexed_sources() {
    let strength = std::f64::consts::LN_2;
    let mut model = common::direct_model().unwrap();
    model.frame_gains = Some(vec![2.0; model.frame_count()]);
    model.validate().unwrap();
    let object = SyntheticObject::mixed_test_pattern((16, 16)).unwrap();
    let baseline = Simulator::ideal(model.clone())
        .object(object.clone())
        .simulate()
        .unwrap();
    let ordinary = Simulator::new(model)
        .object(object)
        .aberration(AberrationModel::new().illumination_vignetting(strength))
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

    let multiplexed_model = common::direct_model()
        .unwrap()
        .with_multiplexing(vec![vec![(0, 1.0), (2, 1.0)], vec![(1, 0.25), (4, 0.75)]])
        .unwrap();
    let multiplexed = Simulator::new(multiplexed_model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .aberration(AberrationModel::new().illumination_vignetting(strength))
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
    assert!(
        AberrationModel::new()
            .illumination_vignetting(-0.1)
            .validate()
            .is_err()
    );
}

#[test]
fn missing_sources_become_zero_weight_dark_frames() {
    let model = common::direct_model().unwrap();
    let simulation = Simulator::new(model)
        .object(SyntheticObject::constant((16, 16), 1.0, 0.0).unwrap())
        .illumination_errors(IlluminationErrorModel::new().missing_sources(vec![1, 3]))
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
fn source_order_mismatch_reorders_true_geometry_only() {
    let model = common::direct_model().unwrap();
    let order = vec![4, 3, 2, 1, 0];
    let simulation = Simulator::new(model.clone())
        .object(SyntheticObject::phase_disk((16, 16), 4.0, 0.5).unwrap())
        .illumination_errors(IlluminationErrorModel::new().source_order(order.clone()))
        .simulate()
        .unwrap();
    for (frame, &source) in order.iter().enumerate() {
        assert_eq!(
            simulation.true_model.k_vectors[frame],
            model.k_vectors[source]
        );
    }
    assert_eq!(simulation.reconstruction_model.k_vectors, model.k_vectors);
    assert!(simulation.illumination_errors.is_some());
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
        .noise(NoiseModel::None)
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
