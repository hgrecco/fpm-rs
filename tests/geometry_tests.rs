use approx::assert_abs_diff_eq;
use fpm_rs::{
    experiment::{
        AcquisitionPlan, ArrayPose, DirectionList, Illumination, IlluminationFrame, KVector,
        KVectorList, Optics, PlanarLedArray, SourceCalibration, SourceContribution,
        SourcePositionList,
    },
    model::{ImagePlaneModel, ReconstructionShape},
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

fn position_after_pose(pose: ArrayPose, point: [f64; 3]) -> [f64; 3] {
    let geometry = PlanarLedArray::new((1, 1), (1.0, 1.0), (0.0, 0.0), pose)
        .with_position_offsets_m(vec![point]);
    geometry.resolve(&optics()).unwrap().positions_m().unwrap()[0]
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[test]
fn array_pose_identity_translation_axes_and_degree_equivalence() {
    let point = [1.0, 2.0, 3.0];
    assert_eq!(position_after_pose(ArrayPose::identity(), point), point);
    assert_eq!(
        position_after_pose(ArrayPose::from_translation([4.0, -1.0, 0.5]), point),
        [5.0, 1.0, 3.5]
    );

    let x = ArrayPose::from_translation_and_extrinsic_xyz_degrees([0.0; 3], [90.0, 0.0, 0.0]);
    let y = ArrayPose::from_translation_and_extrinsic_xyz_degrees([0.0; 3], [0.0, 90.0, 0.0]);
    let z = ArrayPose::from_translation_and_extrinsic_xyz_degrees([0.0; 3], [0.0, 0.0, 90.0]);
    let px = position_after_pose(x, [0.0, 1.0, 0.0]);
    let py = position_after_pose(y, [0.0, 0.0, 1.0]);
    let pz = position_after_pose(z, [1.0, 0.0, 0.0]);
    assert_abs_diff_eq!(px[2], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(py[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(pz[1], 1.0, epsilon = 1e-12);

    let degrees = ArrayPose::from_translation_and_extrinsic_xyz_degrees(
        [0.1, -0.2, -0.3],
        [10.0, 20.0, 30.0],
    );
    let radians = ArrayPose::from_translation_and_extrinsic_xyz_radians(
        [0.1, -0.2, -0.3],
        [
            10_f64.to_radians(),
            20_f64.to_radians(),
            30_f64.to_radians(),
        ],
    );
    assert_eq!(degrees, radians);
}

#[test]
fn array_pose_is_active_extrinsic_xyz_and_proper_orthogonal() {
    let pose = ArrayPose::from_translation_and_extrinsic_xyz_radians(
        [0.2, -0.1, 0.3],
        [0.31, -0.27, 0.44],
    );
    let origin = position_after_pose(pose.clone(), [0.0; 3]);
    let ex = subtract(position_after_pose(pose.clone(), [1.0, 0.0, 0.0]), origin);
    let ey = subtract(position_after_pose(pose.clone(), [0.0, 1.0, 0.0]), origin);
    let ez = subtract(position_after_pose(pose, [0.0, 0.0, 1.0]), origin);
    for axis in [ex, ey, ez] {
        assert_abs_diff_eq!(dot(axis, axis), 1.0, epsilon = 1e-12);
    }
    assert_abs_diff_eq!(dot(ex, ey), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(dot(ex, ez), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(dot(ey, ez), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(dot(cross(ex, ey), ez), 1.0, epsilon = 1e-12);

    let combined =
        ArrayPose::from_translation_and_extrinsic_xyz_degrees([0.0; 3], [90.0, 90.0, 0.0]);
    let transformed = position_after_pose(combined, [0.0, 1.0, 0.0]);
    assert_abs_diff_eq!(transformed[0], 1.0, epsilon = 1e-12);
}

#[test]
fn planar_array_indexing_pitch_reference_pose_and_offsets_are_explicit() {
    let geometry = PlanarLedArray::new(
        (2, 3),
        (2e-3, 5e-3),
        (0.5, 0.25),
        ArrayPose::from_translation([1e-3, -2e-3, -0.1]),
    )
    .with_position_offsets_m(vec![
        [0.1e-3, 0.2e-3, 0.3e-3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
    ]);
    geometry.validate().unwrap();
    assert_eq!(geometry.source_count(), 6);
    assert_eq!(geometry.source_index(1, 2).unwrap(), 5);
    assert_eq!(geometry.source_row_column(4).unwrap(), (1, 1));
    let resolved = geometry.resolve(&optics()).unwrap();
    let positions = resolved.positions_m().unwrap();
    assert_eq!(positions.len(), 6);
    assert_abs_diff_eq!(positions[0][0], 0.1e-3, epsilon = 1e-15);
    assert_abs_diff_eq!(positions[0][1], -3.05e-3, epsilon = 1e-15);
    assert_abs_diff_eq!(positions[0][2], -99.7e-3, epsilon = 1e-15);
    assert_abs_diff_eq!(positions[5][0], 4e-3, epsilon = 1e-15);
    assert_abs_diff_eq!(positions[5][1], 1.75e-3, epsilon = 1e-15);
}

#[test]
fn planar_array_reference_led_tilt_rotation_and_validation() {
    let pose = ArrayPose::from_translation_and_extrinsic_xyz_degrees(
        [0.01, -0.02, -0.1],
        [5.0, -3.0, 90.0],
    );
    let geometry = PlanarLedArray::new((1, 2), (4e-3, 6e-3), (0.0, 0.0), pose);
    let positions = geometry
        .resolve(&optics())
        .unwrap()
        .positions_m()
        .unwrap()
        .to_vec();
    assert_abs_diff_eq!(positions[0][0], 0.01, epsilon = 1e-15);
    assert_abs_diff_eq!(positions[0][1], -0.02, epsilon = 1e-15);
    assert_abs_diff_eq!(positions[0][2], -0.1, epsilon = 1e-15);
    assert!(positions[1][1] > positions[0][1]);
    assert_ne!(positions[1][2], positions[0][2]);

    assert!(
        PlanarLedArray::new((0, 1), (1.0, 1.0), (0.0, 0.0), ArrayPose::identity())
            .validate()
            .is_err()
    );
    assert!(
        PlanarLedArray::new((1, 1), (0.0, 1.0), (0.0, 0.0), ArrayPose::identity())
            .validate()
            .is_err()
    );
    assert!(
        PlanarLedArray::new((1, 2), (1.0, 1.0), (0.0, 0.0), ArrayPose::identity())
            .with_position_offsets_m(vec![[0.0; 3]])
            .validate()
            .is_err()
    );
}

#[test]
fn positions_resolve_toward_sample_with_positive_z_propagation() {
    let resolved = SourcePositionList::new(vec![[0.01, -0.02, -0.1]])
        .resolve(&optics())
        .unwrap();
    let direction = resolved.directions()[0];
    assert!(direction[0] < 0.0);
    assert!(direction[1] > 0.0);
    assert!(direction[2] > 0.0);
    let norm = dot(direction, direction);
    assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-12);
    assert!(
        SourcePositionList::new(vec![[0.0; 3]])
            .resolve(&optics())
            .is_err()
    );
}

#[test]
fn direction_constructors_and_accessors_round_trip() {
    let cosines = vec![[0.1, -0.2], [0.0, 0.0]];
    let directions = DirectionList::from_direction_cosines(cosines.clone()).unwrap();
    for (actual, expected) in directions.direction_cosines().iter().zip(cosines) {
        assert_abs_diff_eq!(actual[0], expected[0], epsilon = 1e-12);
        assert_abs_diff_eq!(actual[1], expected[1], epsilon = 1e-12);
    }
    let component = DirectionList::from_component_angles_degrees(vec![[10.0, -5.0]]).unwrap();
    let component_round_trip = component.component_angles_deg()[0];
    assert_abs_diff_eq!(component_round_trip[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(component_round_trip[1], -5.0, epsilon = 1e-12);

    let polar = DirectionList::from_polar_angles_degrees(vec![[30.0, 120.0]]).unwrap();
    let polar_round_trip = polar.polar_angles_deg()[0];
    assert_abs_diff_eq!(polar_round_trip[0], 30.0, epsilon = 1e-12);
    assert_abs_diff_eq!(polar_round_trip[1], 120.0, epsilon = 1e-12);

    let normalized = DirectionList::from_vectors(vec![[0.0, 0.0, 2.0]], true).unwrap();
    assert_eq!(normalized.unit_vectors(), &[[0.0, 0.0, 1.0]]);
    assert!(DirectionList::from_unit_vectors(vec![[0.0, 0.0, 2.0]]).is_err());
    assert!(DirectionList::from_direction_cosines(vec![[0.8, 0.8]]).is_err());
    assert!(DirectionList::from_unit_vectors(vec![[0.0, 0.0, -1.0]]).is_err());
}

#[test]
fn direction_and_kvector_resolution_use_vacuum_wavelength_and_illumination_index() {
    let direction = DirectionList::from_direction_cosines(vec![[0.25, -0.5]]).unwrap();
    let air = direction.resolve(&optics()).unwrap();
    let mut glass_optics = optics();
    glass_optics.illumination_refractive_index = 1.5;
    let glass = direction.resolve(&glass_optics).unwrap();
    assert_abs_diff_eq!(
        glass.k_vectors()[0].kx,
        1.5 * air.k_vectors()[0].kx,
        epsilon = 1e-9
    );
    let direct = KVectorList::new(air.k_vectors().to_vec())
        .resolve(&glass_optics)
        .unwrap();
    assert_eq!(direct.k_vectors(), air.k_vectors());
    assert!(
        KVectorList::new(vec![KVector::new(
            glass_optics.illumination_wavenumber() * 1.01,
            0.0,
        )])
        .resolve(&glass_optics)
        .is_err()
    );
}

fn frame(entries: &[(usize, f64)], gain: f64) -> IlluminationFrame {
    IlluminationFrame::new(
        entries
            .iter()
            .map(|&(source, weight)| SourceContribution::new(source, weight))
            .collect(),
        gain,
    )
}

#[test]
fn acquisition_plan_supports_subsets_repetition_and_canonical_sparse_frames() {
    let all = AcquisitionPlan::all_sources(3).unwrap();
    assert_eq!(
        all.dense_weights(3).unwrap(),
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]
    );
    let sequential = AcquisitionPlan::sequential(vec![2, 0, 2]).unwrap();
    assert_eq!(sequential.frame_count(), 3);
    assert_eq!(sequential.frames()[0].contributions[0].source, 2);

    let sparse = AcquisitionPlan::from_sparse(vec![frame(
        &[(2, 0.0), (1, 0.25), (0, 0.5), (1, 0.75)],
        2.0,
    )])
    .unwrap();
    assert_eq!(
        sparse.frames()[0].contributions,
        vec![
            SourceContribution::new(0, 0.5),
            SourceContribution::new(1, 1.0),
        ]
    );
    assert_eq!(sparse.dense_weights(3).unwrap(), vec![vec![0.5, 1.0, 0.0]]);
    assert!(sparse.dense_weights(1).is_err());
}

#[test]
fn acquisition_dense_and_sparse_validation_are_equivalent() {
    let dense = AcquisitionPlan::from_dense(vec![vec![0.5, 0.5], vec![1.0, 0.0]]).unwrap();
    let sparse = AcquisitionPlan::from_sparse(vec![
        frame(&[(0, 0.5), (1, 0.5)], 1.0),
        frame(&[(0, 1.0)], 1.0),
    ])
    .unwrap();
    assert_eq!(dense, sparse);
    assert!(AcquisitionPlan::sequential(Vec::new()).is_err());
    assert!(AcquisitionPlan::from_sparse(vec![frame(&[(0, 0.0)], 1.0)]).is_err());
    assert!(AcquisitionPlan::from_sparse(vec![frame(&[(0, -1.0)], 1.0)]).is_err());
    assert!(AcquisitionPlan::from_sparse(vec![frame(&[(0, 1.0)], f64::NAN)]).is_err());
    assert!(AcquisitionPlan::from_dense(vec![vec![1.0], vec![1.0, 2.0]]).is_err());
}

#[test]
fn complete_resolution_exposes_counts_weights_powers_and_gains() {
    let geometry = KVectorList::new(vec![KVector::new(0.0, 0.0), KVector::new(1e4, -2e4)]);
    let illumination = Illumination::new(
        geometry.into(),
        SourceCalibration::new(Some(vec![0.5, 2.0])),
        AcquisitionPlan::from_sparse(vec![
            frame(&[(1, 1.0)], 0.8),
            frame(&[(0, 0.25), (1, 0.75)], 1.2),
        ])
        .unwrap(),
    );
    let resolved = illumination.resolve(&optics()).unwrap();
    assert_eq!(resolved.source_count(), 2);
    assert_eq!(resolved.frame_count(), 2);
    assert!(resolved.is_multiplexed());
    assert_eq!(resolved.source_power(), &[0.5, 2.0]);
    assert_eq!(resolved.frame_gains(), vec![0.8, 1.2]);
    assert_eq!(
        resolved.dense_weights(),
        vec![vec![0.0, 1.0], vec![0.25, 0.75]]
    );
    assert!(resolved.positions_m().is_none());

    let model = ImagePlaneModel::compile(
        &optics(),
        &resolved,
        (8, 8),
        ReconstructionShape::Exact((16, 16)),
    )
    .unwrap();
    assert_eq!(model.frame_count(), 2);
    assert_eq!(model.frame_gains(), Some(&[0.8, 1.2][..]));
    assert_eq!(
        model.multiplexing_matrix().unwrap(),
        &vec![vec![(1, 2.0)], vec![(0, 0.125), (1, 1.5)]]
    );
}

#[test]
fn complete_resolution_rejects_invalid_indices_and_power() {
    let geometry = KVectorList::new(vec![KVector::new(0.0, 0.0)]);
    let bad_index = Illumination::new(
        geometry.clone().into(),
        SourceCalibration::unity(),
        AcquisitionPlan::sequential(vec![1]).unwrap(),
    );
    assert!(bad_index.resolve(&optics()).is_err());
    let bad_power = Illumination::new(
        geometry.into(),
        SourceCalibration::new(Some(vec![-1.0])),
        AcquisitionPlan::all_sources(1).unwrap(),
    );
    assert!(bad_power.resolve(&optics()).is_err());
}

#[test]
fn model_compilation_preserves_fourier_shift_sign_and_fractional_offsets() {
    let optics = optics();
    let image_shape = (8, 8);
    let dk = std::f64::consts::TAU / (image_shape.1 as f64 * optics.object_pixel_size());
    let illumination =
        Illumination::from_geometry(KVectorList::new(vec![KVector::new(1.25 * dk, -0.5 * dk)]))
            .unwrap();
    let model = ImagePlaneModel::from_experiment(
        &optics,
        &illumination,
        image_shape,
        ReconstructionShape::Exact((16, 16)),
    )
    .unwrap();
    let crop = model.crop_indices().as_slice()[0];
    let central_start = (16 - 8) / 2;
    assert!(crop.start_col > central_start);
    assert!(crop.start_row < central_start);
    let offset = model.subpixel_offsets().unwrap()[0];
    assert_abs_diff_eq!(offset.column, 0.25, epsilon = 1e-12);
    assert_abs_diff_eq!(offset.row, 0.5, epsilon = 1e-12);
}

#[test]
fn new_illumination_schema_round_trips_without_old_variants() {
    let illumination = Illumination::new(
        PlanarLedArray::new(
            (1, 2),
            (4e-3, 5e-3),
            (0.5, 0.0),
            ArrayPose::from_translation([0.0, 0.0, -0.09]),
        )
        .into(),
        SourceCalibration::new(Some(vec![0.8, 1.2])),
        AcquisitionPlan::sequential(vec![1, 0, 1]).unwrap(),
    );
    let json = serde_json::to_string_pretty(&illumination).unwrap();
    assert!(json.contains("planar_led_array"));
    assert!(json.contains("pitch_m"));
    assert!(json.contains("active_extrinsic_xyz"));
    assert!(json.contains("relative_power"));
    assert!(json.contains("contributions"));
    assert!(!json.contains("wavelength_override"));
    assert!(!json.contains("illumination_order"));
    let restored: Illumination = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, illumination);
}
