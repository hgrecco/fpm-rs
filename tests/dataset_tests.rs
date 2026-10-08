use std::fs;

use fpm_rs::{
    Complex64, Result,
    algorithms::AlternatingProjection,
    benchmark::{
        BenchmarkRecord, run_benchmark_subset_case, write_benchmark_csv, write_benchmark_json,
    },
    configuration::{ExperimentDescription, SimulationConfiguration},
    datasets::{Dataset, DatasetLoader, DatasetManifest, FrameSelector},
    experiment::{
        AcquisitionPlan, Illumination, IlluminationFrame, KVector, KVectorList, Optics,
        SourceCalibration, SourceContribution,
    },
    measurements::{FrameSpec, MeasurementSpec, MeasurementStack},
    model::ReconstructionShape,
};
use ndarray::Array2;
use tempfile::TempDir;

fn direct_illumination(vectors: Vec<KVector>) -> Illumination {
    Illumination::from_geometry(KVectorList::new(vectors)).unwrap()
}

fn calibrated_illumination(
    vectors: Vec<KVector>,
    gains: Vec<f64>,
    weights: Vec<Vec<(usize, f64)>>,
) -> Illumination {
    let frames = weights
        .into_iter()
        .zip(gains)
        .map(|(row, gain)| {
            IlluminationFrame::new(
                row.into_iter()
                    .map(|(source, weight)| SourceContribution::new(source, weight))
                    .collect(),
                gain,
            )
        })
        .collect();
    Illumination::new(
        KVectorList::new(vectors).into(),
        SourceCalibration::unity(),
        AcquisitionPlan::from_sparse(frames).unwrap(),
    )
}

#[test]
fn dataset_manifest_rejects_unknown_versions_fields_and_escaping_paths() -> Result<()> {
    let temporary = TempDir::new()?;
    let root = temporary.path().join("bundle");
    fs::create_dir_all(&root)?;
    let derived = &root;
    let mut manifest = DatasetManifest {
        format_version: 1,
        measurement_manifest: "../outside.json".into(),
        configuration: "configuration.json".into(),
        ground_truth_object: None,
        valid_object_mask: None,
        provenance: Default::default(),
        measurement_units: None,
    };
    serde_json::to_writer_pretty(fs::File::create(root.join("dataset.json"))?, &manifest)?;

    let error = DatasetLoader::new(&root)?.load().unwrap_err();
    assert!(error.to_string().contains("safe relative path"));

    manifest.format_version = 2;
    manifest.measurement_manifest = "measurements.json".into();
    serde_json::to_writer_pretty(fs::File::create(root.join("dataset.json"))?, &manifest)?;
    let error = DatasetLoader::new(&root)?.load().unwrap_err();
    assert!(error.to_string().contains("expected 1"));

    fs::write(
        root.join("dataset.json"),
        r#"{
            "format_version": 1,
            "measurement_manifest": "measurements.json",
            "configuration": "configuration.json",
            "unknown": true
        }"#,
    )?;
    let error = DatasetLoader::new(&root)?.load().unwrap_err();
    assert!(error.to_string().contains("unknown field"));

    manifest.format_version = 1;
    manifest.measurement_manifest = "measurements.json".into();
    serde_json::to_writer_pretty(fs::File::create(root.join("dataset.json"))?, &manifest)?;
    MeasurementSpec::new(vec![FrameSpec::new("../../outside.png")])
        .save(derived.join("measurements.json"))?;
    let error = DatasetLoader::new(&root)?.load().unwrap_err();
    assert!(error.to_string().contains("safe relative path"));
    Ok(())
}

#[test]
fn dataset_loader_reads_a_conforming_generic_bundle() -> Result<()> {
    let temporary = TempDir::new()?;
    let root = temporary.path().join("bundle");
    fs::create_dir_all(&root)?;
    let derived = &root;
    for index in 0..2_u8 {
        image::GrayImage::from_raw(4, 4, vec![index + 1; 16])
            .unwrap()
            .save(derived.join(format!("frame-{index}.png")))?;
    }
    MeasurementSpec::new(vec![
        FrameSpec::new("frame-0.png"),
        FrameSpec::new("frame-1.png"),
    ])
    .save(derived.join("measurements.json"))?;
    let optics = Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let illumination = direct_illumination(vec![KVector::new(0.0, 0.0), KVector::new(1000.0, 0.0)]);
    let experiment = ExperimentDescription::new(optics, illumination);
    SimulationConfiguration::new(
        experiment.clone(),
        experiment,
        (4, 4),
        ReconstructionShape::Exact((8, 8)),
    )?
    .save(derived.join("configuration.json"))?;
    let ground_truth = Array2::from_shape_vec(
        (8, 8),
        (0..64)
            .map(|index| Complex64::new(1.0 + index as f64 / 100.0, 0.1))
            .collect(),
    )?;
    serde_json::to_writer_pretty(
        fs::File::create(derived.join("ground-truth.json"))?,
        &serde_json::json!({
            "height": 8,
            "width": 8,
            "data": ground_truth.iter().copied().collect::<Vec<_>>(),
        }),
    )?;
    let mut mask_values = vec![0_u8; 64];
    for row in 2..6 {
        for column in 2..6 {
            mask_values[row * 8 + column] = 1;
        }
    }
    let mask = Array2::from_shape_vec((8, 8), mask_values)?;
    serde_json::to_writer_pretty(
        fs::File::create(derived.join("mask.json"))?,
        &serde_json::json!({
            "height": 8,
            "width": 8,
            "data": mask.iter().copied().collect::<Vec<_>>(),
        }),
    )?;
    let manifest = DatasetManifest {
        format_version: 1,
        measurement_manifest: "measurements.json".into(),
        configuration: "configuration.json".into(),
        ground_truth_object: Some("ground-truth.json".into()),
        valid_object_mask: Some("mask.json".into()),
        provenance: [
            ("source".into(), "generated test fixture".into()),
            ("license".into(), "CC0".into()),
        ]
        .into(),
        measurement_units: Some("arbitrary intensity".into()),
    };
    serde_json::to_writer_pretty(fs::File::create(root.join("dataset.json"))?, &manifest)?;
    let load = || DatasetLoader::new(&root)?.load();
    let dataset = load()?;
    assert_eq!(dataset.measurements().frame_count(), 2);
    assert_eq!(dataset.measurements().as_slice()[0], 1.0);
    assert_eq!(dataset.measurements().as_slice()[16], 2.0);
    assert_eq!(
        dataset.ground_truth_object().unwrap().shape(),
        ground_truth.shape()
    );
    assert!(
        dataset
            .ground_truth_object()
            .unwrap()
            .iter()
            .zip(ground_truth.iter())
            .all(|(loaded, expected)| (*loaded - *expected).norm() < 1e-14)
    );
    assert_eq!(dataset.valid_object_mask().unwrap(), &mask);
    assert_eq!(dataset.measurement_units(), Some("arbitrary intensity"));
    assert_eq!(dataset.provenance()["license"], "CC0");

    let subset = dataset.subset().crop_pixels(1, 1, 2, 2)?.build()?;
    assert_eq!(subset.ground_truth_object().unwrap().dim(), (4, 4));
    assert_eq!(subset.valid_object_mask().unwrap().dim(), (4, 4));
    assert!(
        subset
            .valid_object_mask()
            .unwrap()
            .iter()
            .all(|&value| value == 1)
    );
    assert_eq!(subset.measurement_units(), Some("arbitrary intensity"));
    let (record, result) = run_benchmark_subset_case(
        "local-fixture",
        "iterations=1",
        AlternatingProjection::default().iterations(1),
        &subset,
    )?;
    assert!(record.success, "{:?}", record.error);
    assert!(record.amplitude_rmse.is_some());
    assert_eq!(record.metadata["license"], "CC0");
    assert_eq!(record.metadata["measurement_units"], "arbitrary intensity");
    assert_eq!(result.unwrap().metadata["case_id"], record.case_id);

    serde_json::to_writer_pretty(
        fs::File::create(derived.join("ground-truth.json"))?,
        &serde_json::json!({
            "height": 4,
            "width": 4,
            "data": vec![Complex64::new(1.0, 0.0); 16],
        }),
    )?;
    let error = load().unwrap_err();
    assert!(error.to_string().contains("ground-truth shape"));

    serde_json::to_writer_pretty(
        fs::File::create(derived.join("ground-truth.json"))?,
        &serde_json::json!({
            "height": 8,
            "width": 8,
            "data": ground_truth.iter().copied().collect::<Vec<_>>(),
        }),
    )?;
    serde_json::to_writer_pretty(
        fs::File::create(derived.join("mask.json"))?,
        &serde_json::json!({
            "height": 4,
            "width": 4,
            "data": vec![1_u8; 16],
        }),
    )?;
    let error = load().unwrap_err();
    assert!(error.to_string().contains("valid-object mask shape"));

    let mut missing_manifest = manifest;
    missing_manifest.ground_truth_object = Some("missing-ground-truth.json".into());
    serde_json::to_writer_pretty(
        fs::File::create(root.join("dataset.json"))?,
        &missing_manifest,
    )?;
    let error = load().unwrap_err();
    assert!(error.to_string().contains("failed to open ground-truth"));
    Ok(())
}

fn subset_fixture() -> Result<Dataset> {
    let optics = Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let illumination = direct_illumination(vec![
        KVector::new(0.0, 0.0),
        KVector::new(1000.0, 0.0),
        KVector::new(0.0, 1000.0),
    ]);
    let experiment = ExperimentDescription::new(optics, illumination);
    let configuration = SimulationConfiguration::new(
        experiment.clone(),
        experiment,
        (4, 4),
        ReconstructionShape::Exact((8, 8)),
    )?;
    let measurements =
        MeasurementStack::from_vec((0..48).map(f64::from).collect(), (4, 4), Vec::new())?;
    Dataset::new(measurements, configuration)
}

#[test]
fn deterministic_frame_and_pixel_subset_builds_a_valid_problem() -> Result<()> {
    let dataset = subset_fixture()?;
    let subset = dataset
        .subset()
        .frames(FrameSelector::Indices(vec![0, 2]))
        .crop_pixels(1, 1, 2, 2)?
        .build()?;
    assert_eq!(subset.measurements().frame_count(), 2);
    assert_eq!(subset.measurements().image_shape(), (2, 2));
    assert_eq!(
        subset
            .measurements()
            .frame_metadata()
            .iter()
            .map(|metadata| metadata.original_frame_index)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(2)]
    );
    assert_eq!(
        subset
            .measurements()
            .frame_metadata()
            .iter()
            .map(|metadata| metadata.original_illumination_index)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(2)]
    );
    assert_eq!(
        subset.measurements().as_slice(),
        &[5.0, 6.0, 9.0, 10.0, 37.0, 38.0, 41.0, 42.0]
    );
    assert_eq!(subset.configuration().reconstruction_shape, (4, 4));
    subset.reconstruction_problem()?;
    Ok(())
}

#[test]
fn subset_benchmarks_preserve_selection_in_records_and_artifacts() -> Result<()> {
    let dataset = subset_fixture()?;
    let subset = dataset
        .subset()
        .frames(FrameSelector::Indices(vec![0, 2]))
        .crop_pixels(1, 1, 2, 2)?
        .build()?;
    let reordered = dataset
        .subset()
        .frames(FrameSelector::Indices(vec![2, 0]))
        .crop_pixels(1, 1, 2, 2)?
        .build()?;
    let run = |subset: &fpm_rs::datasets::DatasetSubset| {
        run_benchmark_subset_case(
            "subset-fixture",
            "iterations=1",
            AlternatingProjection::default().iterations(1),
            subset,
        )
    };
    let (record, result) = run(&reordered)?;
    assert!(record.success, "{:?}", record.error);
    assert_eq!(record.spatial_crop, Some([1, 1, 2, 2]));
    assert_eq!(
        record
            .frames
            .iter()
            .map(|f| f.original_frame_index)
            .collect::<Vec<_>>(),
        [2, 0]
    );
    assert_eq!(
        record
            .frames
            .iter()
            .map(|f| f.original_illumination_index)
            .collect::<Vec<_>>(),
        [Some(2), Some(0)]
    );
    let (repeat, _) = run(&reordered)?;
    assert_eq!(record.case_id, repeat.case_id);
    assert_ne!(record.run_id, repeat.run_id);
    let alternate_crop = dataset
        .subset()
        .frames(FrameSelector::Indices(vec![2, 0]))
        .crop_pixels(0, 0, 2, 2)?
        .build()?;
    assert_ne!(record.case_id, run(&alternate_crop)?.0.case_id);
    assert_ne!(record.case_id, run(&subset)?.0.case_id);
    let full_image = dataset.subset().every_nth_frame(2).build()?;
    assert_eq!(run(&full_image)?.0.spatial_crop, Some([0, 0, 4, 4]));

    let (failed, no_result) = run_benchmark_subset_case(
        "subset-fixture",
        "object_step=-1",
        AlternatingProjection::default().object_step(-1.0),
        &reordered,
    )?;
    assert!(!failed.success && no_result.is_none());
    assert_eq!(failed.spatial_crop, record.spatial_crop);
    assert_eq!(failed.frames[0].original_frame_index, 2);
    assert!(failed.frames.iter().all(|f| f.normalized_l2.is_none()));

    let result = result.unwrap();
    let completed = BenchmarkRecord::from_subset_result(
        "explicit-case",
        "subset-fixture",
        "iterations=1",
        &result,
        &reordered,
    )?;
    assert_eq!(completed.case_id, "explicit-case");
    assert_eq!(completed.frames[0].original_frame_index, 2);
    assert!(
        BenchmarkRecord::from_subset_result("case", "fixture", "", &result, &full_image).is_err()
    );

    let directory = tempfile::tempdir()?;
    let records = [record.clone(), failed];
    write_benchmark_json(&records, directory.path().join("records.json"))?;
    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.path().join("records.json"))?)?;
    assert_eq!(
        json["records"][0]["spatial_crop"],
        serde_json::json!([1, 1, 2, 2])
    );
    assert_eq!(json["records"][1]["frames"][0]["original_frame_index"], 2);
    write_benchmark_csv(&records, directory.path().join("records.csv"))?;
    let mut csv = csv::Reader::from_path(directory.path().join("records.csv"))?;
    let columns = csv.headers()?.clone();
    let crop_column = columns.iter().position(|c| c == "spatial_crop").unwrap();
    assert_eq!(&csv.records().next().unwrap()?[crop_column], "1;1;2;2");

    #[cfg(feature = "parquet")]
    {
        use polars::prelude::{ParquetReader, SerReader};
        let results = [(record.run_id.clone(), result)].into();
        let bundle = fpm_rs::benchmark_bundle::write_benchmark_bundle(
            directory.path().join("bundle"),
            "subset-suite",
            &records,
            &results,
            Default::default(),
        )?;
        let runs = ParquetReader::new(fs::File::open(bundle.tables.runs.path)?).finish()?;
        let frames = ParquetReader::new(fs::File::open(bundle.tables.frames.path)?).finish()?;
        assert_eq!(runs.column("crop_row")?.u64()?.get(1), Some(1));
        assert_eq!(runs.column("crop_height")?.u64()?.get(0), Some(2));
        assert_eq!(
            frames.column("original_frame_index")?.u64()?.get(2),
            Some(2)
        );
        assert_eq!(
            frames.column("original_illumination_index")?.u64()?.get(1),
            Some(0)
        );
        let mut invalid = records[0].clone();
        invalid.spatial_crop = Some([0, 0, 3, 2]);
        assert!(
            fpm_rs::benchmark_bundle::write_benchmark_bundle(
                directory.path().join("invalid"),
                "subset-suite",
                &[invalid],
                &results,
                Default::default(),
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn subsets_preserve_frame_gains_backgrounds_and_multiplexing() -> Result<()> {
    let optics = Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.1,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let vectors = vec![
        KVector::new(0.0, 0.0),
        KVector::new(1000.0, 0.0),
        KVector::new(0.0, 1000.0),
    ];
    let background: Vec<f64> = (0..3)
        .flat_map(|frame| vec![10.0 + frame as f64; 16])
        .collect();
    let ordinary_illumination = calibrated_illumination(
        vectors.clone(),
        vec![1.0, 2.0, 3.0],
        vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
    );
    let ordinary_experiment = ExperimentDescription::new(optics.clone(), ordinary_illumination)
        .with_optical_background(vec![9.0; 16]);
    let ordinary_configuration = SimulationConfiguration::new(
        ordinary_experiment.clone(),
        ordinary_experiment,
        (4, 4),
        ReconstructionShape::Exact((8, 8)),
    )?;
    let measurements =
        MeasurementStack::from_vec((0..48).map(f64::from).collect(), (4, 4), Vec::new())?;
    let ordinary = Dataset::new(measurements.clone(), ordinary_configuration)?;
    let ordinary_subset = ordinary
        .subset()
        .frames(FrameSelector::Indices(vec![2, 0]))
        .crop_pixels(1, 1, 2, 2)?
        .build()?;
    let ordinary_model = &ordinary_subset.configuration().compiled_models.true_model;
    assert_eq!(ordinary_model.frame_gains(), Some(&[3.0, 1.0][..]));
    assert_eq!(ordinary_model.background(), Some(&[9.0, 9.0, 9.0, 9.0][..]));
    ordinary_subset.reconstruction_problem()?;

    let multiplexing = vec![
        vec![(0, 1.0)],
        vec![(0, 0.25), (1, 0.75)],
        vec![(1, 0.4), (2, 0.6)],
    ];
    let multiplexed_illumination =
        calibrated_illumination(vectors, vec![1.1, 1.2, 1.3], multiplexing.clone());
    let multiplexed_experiment = ExperimentDescription::new(optics, multiplexed_illumination)
        .with_optical_background(background);
    let multiplexed_configuration = SimulationConfiguration::new(
        multiplexed_experiment.clone(),
        multiplexed_experiment,
        (4, 4),
        ReconstructionShape::Exact((8, 8)),
    )?;
    let multiplexed = Dataset::new(measurements, multiplexed_configuration)?;
    let multiplexed_subset = multiplexed
        .subset()
        .frames(FrameSelector::Indices(vec![1, 2]))
        .build()?;
    let multiplexed_model = &multiplexed_subset
        .configuration()
        .compiled_models
        .reconstruction_model;
    assert_eq!(multiplexed_model.source_count(), 3);
    assert_eq!(multiplexed_model.frame_count(), 2);
    assert_eq!(
        multiplexed_model.multiplexing_matrix().map(Vec::as_slice),
        Some(&multiplexing[1..])
    );
    assert_eq!(multiplexed_model.frame_gains(), Some(&[1.2, 1.3][..]));
    let expected_background: Vec<f64> = [vec![11.0; 16], vec![12.0; 16]].concat();
    assert_eq!(
        multiplexed_model.background(),
        Some(expected_background.as_slice())
    );
    assert!(
        multiplexed_subset
            .measurements()
            .frame_metadata()
            .iter()
            .all(|metadata| metadata.illumination_index.is_none())
    );
    multiplexed_subset.reconstruction_problem()?;
    Ok(())
}
