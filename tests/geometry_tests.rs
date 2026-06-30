use approx::assert_abs_diff_eq;
use fpm_rs::{
    Array2, Complex64,
    experiment::{
        AngleList, CodedIllumination, IlluminationSource, KVector, LEDArray, Optics,
        PupilAberration,
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
        defocus: None,
        initial_pupil_aberration: None,
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
    invalid_optics.initial_pupil_aberration = Some(PupilAberration {
        defocus: f64::NAN,
        ..PupilAberration::default()
    });
    assert!(invalid_optics.validate().is_err());
    assert!(
        ImagePlaneModel::from_experiment(&invalid_optics, &LEDArray::new(), (8, 8), (16, 16))
            .is_err()
    );

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
