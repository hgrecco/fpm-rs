use std::{fs, path::Path};

use fpm_rs::{
    Result,
    configuration::{ExperimentDescription, SimulationConfiguration},
    datasets::{
        DatasetArchive, DatasetCitation, DatasetLicense, DatasetRegistry, DatasetRegistryDocument,
        DatasetRegistryEntry, DatasetSource,
    },
    experiment::{Illumination, KVector, KVectorList, Optics},
    measurements::{FrameSpec, MeasurementSpec},
    model::ReconstructionShape,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn write_bundle(root: &Path) -> Result<()> {
    fs::create_dir_all(root.join("frames"))?;
    image::GrayImage::from_raw(4, 4, vec![7_u8; 16])
        .unwrap()
        .save(root.join("frames/frame.png"))?;
    MeasurementSpec::new(vec![FrameSpec::new("frames/frame.png")])
        .save(root.join("measurements.json"))?;
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
    let experiment = ExperimentDescription::new(
        optics,
        Illumination::from_geometry(KVectorList::new(vec![KVector::new(0.0, 0.0)]))?,
    );
    SimulationConfiguration::new(
        experiment.clone(),
        experiment,
        (4, 4),
        ReconstructionShape::Exact((8, 8)),
    )?
    .save(root.join("configuration.json"))?;
    fs::write(
        root.join("dataset.json"),
        r#"{
  "format_version": 1,
  "measurement_manifest": "measurements.json",
  "configuration": "configuration.json",
  "provenance": {"source": "registry test fixture"},
  "measurement_units": "counts"
}"#,
    )?;
    Ok(())
}

fn write_archive(bundle: &Path, archive_path: &Path) -> Result<()> {
    let encoder = zstd::Encoder::new(fs::File::create(archive_path)?, 1)?;
    let mut archive = tar::Builder::new(encoder);
    for relative in [
        "dataset.json",
        "measurements.json",
        "configuration.json",
        "frames/frame.png",
    ] {
        archive.append_path_with_name(bundle.join(relative), relative)?;
    }
    archive.into_inner()?.finish()?;
    Ok(())
}

fn entry(id: &str, archive_path: &Path) -> DatasetRegistryEntry {
    let bytes = fs::read(archive_path).unwrap();
    DatasetRegistryEntry {
        id: id.into(),
        version: "1.0.0".into(),
        title: format!("Fixture {id}"),
        description: "Small generated image-plane FPM bundle".into(),
        format_version: 1,
        archive: DatasetArchive {
            url: archive_path.display().to_string(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            size_bytes: bytes.len() as u64,
        },
        license: DatasetLicense {
            spdx: "CC0-1.0".into(),
            url: "https://creativecommons.org/publicdomain/zero/1.0/".into(),
        },
        citation: DatasetCitation {
            doi: "10.0000/test".into(),
            text: "Generated registry test fixture".into(),
        },
        source: DatasetSource {
            url: "https://example.invalid/source".into(),
            description: "Generated fixture".into(),
        },
        tags: vec!["test".into(), "image-plane-fpm".into()],
    }
}

fn write_registry(path: &Path, entries: Vec<DatasetRegistryEntry>) -> Result<()> {
    serde_json::to_writer_pretty(
        fs::File::create(path)?,
        &DatasetRegistryDocument {
            registry_version: 1,
            datasets: entries,
        },
    )?;
    Ok(())
}

#[test]
fn registry_lists_downloads_opens_and_cleans_without_external_network() -> Result<()> {
    let temporary = TempDir::new()?;
    let bundle = temporary.path().join("bundle");
    let archive = temporary.path().join("fixture.tar.zst");
    let registry_path = temporary.path().join("registry.json");
    let cache = temporary.path().join("cache");
    write_bundle(&bundle)?;
    write_archive(&bundle, &archive)?;
    write_registry(&registry_path, vec![entry("fixture", &archive)])?;

    let registry = DatasetRegistry::new(registry_path.display().to_string(), &cache)?;
    let listing = registry.list()?;
    assert_eq!(listing.len(), 1);
    assert!(!listing[0].cached);

    let installed = registry.download("fixture")?;
    assert!(installed.join("dataset.json").is_file());
    assert!(!cache.join("partial").join("fixture.tar.zst").exists());
    assert!(registry.list()?[0].cached);

    fs::write(installed.join(".fpm-rs-install.json"), b"{}")?;
    assert!(!registry.list()?[0].cached);
    assert_eq!(registry.download("fixture")?, installed);
    assert!(registry.list()?[0].cached);

    let dataset = registry.open("fixture")?;
    assert_eq!(dataset.source_path(), Some(installed.as_path()));
    assert_eq!(dataset.measurements().frame_count(), 1);
    assert_eq!(dataset.measurements().as_slice()[0], 7.0);
    dataset.reconstruction_problem()?;

    fs::remove_file(installed.join("configuration.json"))?;
    let repaired = registry.open("fixture")?;
    assert_eq!(repaired.measurements().frame_count(), 1);
    assert!(installed.join("configuration.json").is_file());

    fs::remove_file(&archive)?;
    let cached = registry.open("fixture")?;
    assert_eq!(cached.measurements().frame_count(), 1);

    assert!(registry.clean("fixture")?);
    assert!(!registry.clean("fixture")?);
    assert!(!registry.list()?[0].cached);
    Ok(())
}

#[test]
fn concurrent_downloads_converge_on_one_complete_cache_entry() -> Result<()> {
    let temporary = TempDir::new()?;
    let bundle = temporary.path().join("bundle");
    let archive = temporary.path().join("fixture.tar.zst");
    let registry_path = temporary.path().join("registry.json");
    let cache = temporary.path().join("cache");
    write_bundle(&bundle)?;
    write_archive(&bundle, &archive)?;
    write_registry(&registry_path, vec![entry("fixture", &archive)])?;
    let registry = DatasetRegistry::new(registry_path.display().to_string(), &cache)?;

    let handles = (0..4)
        .map(|_| {
            let registry = registry.clone();
            std::thread::spawn(move || registry.download("fixture"))
        })
        .collect::<Vec<_>>();
    let paths = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Result<Vec<_>>>()?;
    assert!(paths.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(paths[0].join("dataset.json").is_file());
    assert_eq!(
        fs::read_dir(cache.join("partial"))?.count(),
        0,
        "partial downloads must be cleaned"
    );
    Ok(())
}

#[test]
fn registry_snapshot_is_used_only_after_the_source_becomes_unavailable() -> Result<()> {
    let temporary = TempDir::new()?;
    let bundle = temporary.path().join("bundle");
    let archive = temporary.path().join("fixture.tar.zst");
    let registry_path = temporary.path().join("registry.json");
    write_bundle(&bundle)?;
    write_archive(&bundle, &archive)?;
    write_registry(&registry_path, vec![entry("fixture", &archive)])?;
    let registry = DatasetRegistry::new(
        format!("file://{}", registry_path.display()),
        temporary.path().join("cache"),
    )?;

    assert_eq!(registry.list()?.len(), 1);
    fs::remove_file(registry_path)?;
    assert_eq!(registry.list()?[0].entry.id, "fixture");
    Ok(())
}

#[test]
fn registry_snapshots_are_separated_by_configured_source() -> Result<()> {
    let temporary = TempDir::new()?;
    let first_path = temporary.path().join("first.json");
    let second_path = temporary.path().join("second.json");
    let cache = temporary.path().join("cache");
    write_registry(&first_path, Vec::new())?;
    fs::write(&second_path, r#"{"registry_version":2,"datasets":[]}"#)?;

    let first = DatasetRegistry::new(first_path.display().to_string(), &cache)?;
    let second = DatasetRegistry::new(second_path.display().to_string(), &cache)?;
    assert!(first.list()?.is_empty());
    assert!(second.list().is_err());

    fs::remove_file(first_path)?;
    fs::remove_file(second_path)?;
    assert!(first.list()?.is_empty());
    assert!(
        second.list().is_err(),
        "a different source's snapshot was reused"
    );
    Ok(())
}

#[test]
fn a_new_registry_version_installs_beside_the_old_one_until_cleaned() -> Result<()> {
    let temporary = TempDir::new()?;
    let bundle = temporary.path().join("bundle");
    let archive = temporary.path().join("fixture.tar.zst");
    let registry_path = temporary.path().join("registry.json");
    let cache = temporary.path().join("cache");
    write_bundle(&bundle)?;
    write_archive(&bundle, &archive)?;

    let first = entry("fixture", &archive);
    write_registry(&registry_path, vec![first])?;
    let registry = DatasetRegistry::new(registry_path.display().to_string(), &cache)?;
    let old_path = registry.download("fixture")?;

    let mut current = entry("fixture", &archive);
    current.version = "2.0.0".into();
    write_registry(&registry_path, vec![current])?;
    assert!(!registry.list()?[0].cached);
    let current_path = registry.download("fixture")?;
    assert_ne!(old_path, current_path);
    assert!(old_path.join("dataset.json").is_file());
    assert!(current_path.join("dataset.json").is_file());

    assert!(registry.clean("fixture")?);
    assert!(!old_path.exists());
    assert!(!current_path.exists());
    Ok(())
}

#[test]
fn download_all_and_clean_all_cover_each_current_version() -> Result<()> {
    let temporary = TempDir::new()?;
    let bundle = temporary.path().join("bundle");
    let archive = temporary.path().join("fixture.tar.zst");
    let registry_path = temporary.path().join("registry.json");
    let cache = temporary.path().join("cache");
    write_bundle(&bundle)?;
    write_archive(&bundle, &archive)?;
    write_registry(
        &registry_path,
        vec![entry("first", &archive), entry("second", &archive)],
    )?;
    let registry = DatasetRegistry::new(registry_path.display().to_string(), &cache)?;

    assert_eq!(registry.download_all()?.len(), 2);
    assert!(registry.list()?.iter().all(|listing| listing.cached));
    assert_eq!(registry.clean_all()?, 2);
    assert!(!cache.exists());
    assert_eq!(registry.clean_all()?, 0);
    Ok(())
}

#[test]
fn checksum_size_and_bundle_validation_fail_before_cache_promotion() -> Result<()> {
    let temporary = TempDir::new()?;
    let bundle = temporary.path().join("bundle");
    let archive = temporary.path().join("fixture.tar.zst");
    let registry_path = temporary.path().join("registry.json");
    let cache = temporary.path().join("cache");
    write_bundle(&bundle)?;
    write_archive(&bundle, &archive)?;

    let mut invalid = entry("fixture", &archive);
    invalid.archive.sha256 = "0".repeat(64);
    write_registry(&registry_path, vec![invalid])?;
    let registry = DatasetRegistry::new(registry_path.display().to_string(), &cache)?;
    let error = registry.download("fixture").unwrap_err();
    assert!(error.to_string().contains("SHA-256 mismatch"));
    assert!(!cache.join("datasets/fixture").exists());
    assert_eq!(fs::read_dir(cache.join("partial"))?.count(), 0);

    let mut invalid = entry("fixture", &archive);
    invalid.archive.size_bytes += 1;
    write_registry(&registry_path, vec![invalid])?;
    let error = registry.download("fixture").unwrap_err();
    assert!(error.to_string().contains("size mismatch"));
    assert!(!cache.join("datasets/fixture").exists());
    assert_eq!(fs::read_dir(cache.join("partial"))?.count(), 0);
    Ok(())
}

#[test]
fn registry_schema_rejects_duplicates_unknown_fields_and_invalid_metadata() {
    let mut entry = serde_json::json!({
        "id": "fixture",
        "version": "1.0.0",
        "title": "Fixture",
        "description": "Fixture dataset",
        "format_version": 1,
        "archive": {
            "url": "https://example.invalid/fixture.tar.zst",
            "sha256": "0".repeat(64),
            "size_bytes": 1
        },
        "license": {"spdx": "CC0-1.0", "url": "https://example.invalid/license"},
        "citation": {"doi": "10.0/test", "text": "Fixture"},
        "source": {"url": "https://example.invalid", "description": "Fixture"},
        "tags": ["test"]
    });
    let duplicate = serde_json::json!({
        "registry_version": 1,
        "datasets": [entry.clone(), entry.clone()]
    });
    assert!(
        DatasetRegistryDocument::from_slice(duplicate.to_string().as_bytes())
            .unwrap_err()
            .to_string()
            .contains("duplicate id")
    );

    entry["unknown"] = serde_json::json!(true);
    let unknown = serde_json::json!({"registry_version": 1, "datasets": [entry]});
    assert!(
        DatasetRegistryDocument::from_slice(unknown.to_string().as_bytes())
            .unwrap_err()
            .to_string()
            .contains("unknown field")
    );

    let unsupported = br#"{"registry_version":2,"datasets":[]}"#;
    assert!(
        DatasetRegistryDocument::from_slice(unsupported)
            .unwrap_err()
            .to_string()
            .contains("unsupported dataset registry version")
    );
}

#[test]
fn cleanup_refuses_an_unmarked_directory() -> Result<()> {
    let temporary = TempDir::new()?;
    let cache = temporary.path().join("user-data");
    fs::create_dir(&cache)?;
    fs::write(cache.join("preserved"), b"user data")?;
    let registry = DatasetRegistry::new("missing.json", &cache)?;
    let error = registry.clean_all().unwrap_err();
    assert!(error.to_string().contains("ownership marker"));
    assert_eq!(fs::read(cache.join("preserved"))?, b"user data");
    Ok(())
}
