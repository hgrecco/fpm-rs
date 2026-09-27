use fpm_rs::{
    configuration::{CONFIGURATION_FORMAT_VERSION, ExperimentDescription, SimulationConfiguration},
    experiment::{
        AcquisitionPlan, ArrayPose, DirectionList, Illumination, KVector, KVectorList, Optics,
        PlanarLedArray, RotatingLedArc, SourceCalibration, SourcePositionList, SphericalLedArm,
        SphericalLedArray,
    },
    model::ReconstructionShape,
    simulation::{CameraModel, IlluminationAcquisitionErrors},
};
use tempfile::tempdir;

fn optics() -> Optics {
    Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.10,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    }
}

fn illumination_variants() -> Vec<Illumination> {
    let vectors = vec![KVector::new(0.0, 0.0), KVector::new(1.0e4, -2.0e4)];
    vec![
        Illumination::from_geometry(PlanarLedArray::new(
            (1, 2),
            (4e-3, 4e-3),
            (0.5, 0.0),
            ArrayPose::from_translation([0.0, 0.0, -90e-3]),
        ))
        .unwrap(),
        Illumination::from_geometry(
            DirectionList::from_component_angles_radians(vec![[0.0, 0.0], [0.01, -0.02]]).unwrap(),
        )
        .unwrap(),
        Illumination::from_geometry(
            SphericalLedArray::new(vec![(0.0, 0.0), (0.1, 0.3)], 90e-3)
                .center_offset((0.5e-3, 0.0, 0.0))
                .angular_corrections(vec![(0.0, 0.0), (0.001, -0.002)]),
        )
        .unwrap(),
        Illumination::from_geometry(
            SphericalLedArm::new(vec![(0.0, 0.0), (0.1, 0.3)], 90e-3)
                .encoder_zero_deg(0.1, -0.2)
                .backlash_deg(0.05, 0.1),
        )
        .unwrap(),
        Illumination::from_geometry(
            RotatingLedArc::new(vec![0.0, 0.1], vec![0.0, 0.3], 90e-3)
                .axis_origin_offset((0.2e-3, 0.0, 0.0))
                .rotation_backlash_deg(0.05)
                .led_angular_corrections(vec![(0.0, 0.0), (0.001, -0.001)]),
        )
        .unwrap(),
        Illumination::from_geometry(SourcePositionList::new(vec![
            [0.0, 0.0, -0.09],
            [0.001, 0.0, -0.09],
        ]))
        .unwrap(),
        Illumination::from_geometry(KVectorList::new(vectors.clone())).unwrap(),
        Illumination::new(
            KVectorList::new(vectors.clone()).into(),
            SourceCalibration::unity(),
            AcquisitionPlan::from_dense(vec![vec![1.0, 0.0], vec![0.5, 0.5]]).unwrap(),
        ),
        Illumination::new(
            KVectorList::new(vectors).into(),
            SourceCalibration::new(Some(vec![0.8, 1.2])),
            AcquisitionPlan::all_sources(2).unwrap(),
        ),
    ]
}

#[test]
fn every_concrete_illumination_source_round_trips() {
    for illumination in illumination_variants() {
        let description = ExperimentDescription::new(optics(), illumination);
        let configuration = SimulationConfiguration::new(
            description.clone(),
            description,
            (8, 8),
            ReconstructionShape::Exact((16, 16)),
        )
        .unwrap();
        let serialized = serde_json::to_string(&configuration).unwrap();
        let restored: SimulationConfiguration = serde_json::from_str(&serialized).unwrap();
        restored.validate().unwrap();
        assert_eq!(
            restored.compiled_models.true_model.source_count(),
            configuration.compiled_models.true_model.source_count()
        );
        assert_eq!(
            restored.compiled_models.true_model.frame_count(),
            configuration.compiled_models.true_model.frame_count()
        );
    }
}

#[test]
fn versioned_configuration_round_trips_models_camera_and_acquisition() {
    let true_experiment = ExperimentDescription::new(
        optics(),
        Illumination::from_geometry(PlanarLedArray::new(
            (2, 2),
            (4e-3, 4e-3),
            (0.5, 0.5),
            ArrayPose::from_translation([0.0, 0.0, -90e-3]),
        ))
        .unwrap(),
    )
    .with_optical_background(vec![2.0; 8 * 8]);
    let mut assumed_optics = optics();
    assumed_optics.defocus_distance = Some(1e-6);
    let reconstruction_experiment =
        ExperimentDescription::new(assumed_optics, true_experiment.illumination.clone());
    let camera = CameraModel::new()
        .photons_per_pixel(50.0)
        .gain(2.0)
        .offset_counts(10.0)
        .quantize(false);
    let acquisition = IlluminationAcquisitionErrors::new()
        .frame_gain_relative_std(0.02)
        .missing_frames(vec![3]);
    let configuration = SimulationConfiguration::new(
        true_experiment,
        reconstruction_experiment,
        (8, 8),
        ReconstructionShape::Exact((16, 16)),
    )
    .unwrap()
    .with_camera(camera)
    .unwrap()
    .with_illumination_acquisition_errors(acquisition)
    .unwrap()
    .with_random_seed(1234);

    let directory = tempdir().unwrap();
    let path = directory.path().join("experiment.json");
    configuration.save(&path).unwrap();
    let restored = SimulationConfiguration::load(&path).unwrap();

    assert_eq!(restored.format_version, CONFIGURATION_FORMAT_VERSION);
    assert_eq!(restored.random_seed, 1234);
    assert_eq!(restored.image_shape, configuration.image_shape);
    assert_eq!(
        restored.compiled_models.true_model.background(),
        configuration.compiled_models.true_model.background()
    );
    assert_eq!(
        restored
            .illumination_acquisition_errors
            .as_ref()
            .unwrap()
            .missing_frames,
        vec![3]
    );
    let count_model = restored.reconstruction_model_for_counts().unwrap();
    assert_eq!(count_model.frame_gains().unwrap(), &[100.0; 4]);
}

#[test]
fn automatic_configuration_shape_covers_true_and_assumed_geometry() {
    let optics = optics();
    let image_shape = (8, 8);
    let dk = std::f64::consts::TAU / (image_shape.1 as f64 * optics.object_pixel_size());
    let true_experiment = ExperimentDescription::new(
        optics.clone(),
        Illumination::from_geometry(KVectorList::new(vec![KVector::new(2.25 * dk, 0.0)])).unwrap(),
    );
    let reconstruction_experiment = ExperimentDescription::new(
        optics,
        Illumination::from_geometry(KVectorList::new(vec![KVector::new(-2.25 * dk, 0.0)])).unwrap(),
    );
    let configuration = SimulationConfiguration::new(
        true_experiment,
        reconstruction_experiment,
        image_shape,
        ReconstructionShape::Minimum,
    )
    .unwrap();

    assert_eq!(configuration.reconstruction_shape, (14, 14));
    configuration.compiled_models.validate().unwrap();
}

#[test]
fn configuration_rejects_unknown_versions_fields_and_model_drift() {
    let description = ExperimentDescription::new(
        optics(),
        Illumination::from_geometry(KVectorList::new(vec![KVector::new(0.0, 0.0)])).unwrap(),
    );
    let configuration = SimulationConfiguration::new(
        description.clone(),
        description,
        (8, 8),
        ReconstructionShape::Exact((16, 16)),
    )
    .unwrap();

    let mut wrong_version = configuration.clone();
    wrong_version.format_version = 999;
    assert!(wrong_version.validate().is_err());

    let mut undersampled = configuration.clone();
    assert_eq!(undersampled.format_version, CONFIGURATION_FORMAT_VERSION);
    undersampled.true_experiment.optics.camera_pixel_size = 20e-6;
    let directory = tempdir().unwrap();
    let path = directory.path().join("undersampled.json");
    std::fs::write(&path, serde_json::to_vec(&undersampled).unwrap()).unwrap();
    let error = SimulationConfiguration::load(path).unwrap_err();
    assert!(error.to_string().contains("camera_pixel_size"));
    assert!(error.to_string().contains("coherent-field sampling"));

    let mut drifted = configuration.clone();
    let mut vectors = drifted.compiled_models.true_model.k_vectors().to_vec();
    vectors[0].kx = 1e5;
    drifted
        .compiled_models
        .true_model
        .replace_k_vectors(vectors)
        .unwrap();
    assert!(drifted.validate().is_err());

    let mut value = serde_json::to_value(configuration).unwrap();
    value["future_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<SimulationConfiguration>(value).is_err());

    let invalid_calibrated = ExperimentDescription::new(
        optics(),
        Illumination::new(
            KVectorList::new(vec![KVector::new(0.0, 0.0), KVector::new(1.0e4, 0.0)]).into(),
            SourceCalibration::new(Some(vec![1.0])),
            AcquisitionPlan::all_sources(2).unwrap(),
        ),
    );
    assert!(
        SimulationConfiguration::new(
            invalid_calibrated.clone(),
            invalid_calibrated,
            (8, 8),
            ReconstructionShape::Exact((16, 16)),
        )
        .is_err()
    );
}
