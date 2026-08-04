//! Normalized benchmark comparison bundles.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Component, Path, PathBuf},
};

use polars::prelude::{DataFrame, DataType, ParquetReader, SerReader};
use serde::{Deserialize, Serialize};

use crate::{
    Error, Result,
    benchmark::BenchmarkRecord,
    reconstruction::{BundleArtifact, BundleExportOptions, ReconstructionResult, ResultBundle},
    tabular::{
        benchmark_artifacts_dataframe, benchmark_frames_dataframe, benchmark_metadata_dataframe,
        benchmark_runs_dataframe,
        parquet::{ParquetWriteOptions, sha256, write_parquet_file},
    },
};

/// Current benchmark-bundle manifest format version.
pub const BENCHMARK_BUNDLE_FORMAT_VERSION: u32 = 1;

const RUNS: &str = "tables.runs";
const FRAMES: &str = "tables.frames";
const ARTIFACTS: &str = "tables.artifacts";
const METADATA: &str = "tables.metadata";

/// Manifest-described Parquet tables in a benchmark bundle.
#[derive(Clone, Debug)]
pub struct BenchmarkBundleTables {
    /// One-row-per-run summary table artifact.
    pub runs: BundleArtifact,
    /// Per-acquisition-frame metrics table artifact.
    pub frames: BundleArtifact,
    /// Generated artifact-path table artifact.
    pub artifacts: BundleArtifact,
    /// Extensible benchmark metadata table artifact.
    pub metadata: BundleArtifact,
}

/// Reopened benchmark suite with table metadata and lazily loaded result bundles.
#[derive(Clone)]
pub struct BenchmarkBundle {
    /// Root directory containing the benchmark bundle.
    pub path: PathBuf,
    /// Path to the benchmark manifest JSON file.
    pub manifest_path: PathBuf,
    /// Stable benchmark-suite name.
    pub name: String,
    /// Optional human-readable benchmark label.
    pub label: Option<String>,
    /// Parquet table artifact descriptors.
    pub tables: BenchmarkBundleTables,
    /// Result bundles keyed by benchmark run ID.
    pub results: BTreeMap<String, ResultBundle>,
}

/// Options controlling benchmark-bundle metadata.
#[derive(Clone, Debug, Default)]
pub struct BenchmarkBundleExportOptions {
    /// Optional human-readable bundle label.
    pub label: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkManifest {
    benchmark_bundle_format_version: u32,
    name: String,
    label: Option<String>,
    crate_version: String,
    artifacts: Vec<BenchmarkManifestArtifact>,
    results: BTreeMap<String, PathBuf>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkManifestArtifact {
    role: String,
    relative_path: PathBuf,
    media_type: String,
    byte_size: u64,
    sha256: String,
}

/// Atomically writes benchmark tables plus referenced reconstruction result bundles.
///
/// Every record must have a unique run ID and every supplied result must correspond to a
/// successful record. Existing complete destinations are rejected.
pub fn write_benchmark_bundle(
    path: impl AsRef<Path>,
    name: impl Into<String>,
    records: &[BenchmarkRecord],
    results: &BTreeMap<String, ReconstructionResult>,
    options: BenchmarkBundleExportOptions,
) -> Result<BenchmarkBundle> {
    validate_records(records, results)?;
    let name = name.into();
    if name.is_empty() {
        return Err(Error::InvalidParameter {
            name: "benchmark name",
            reason: "must be non-empty".into(),
        });
    }
    let (final_path, workspace) = unique_paths(path.as_ref())?;
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&workspace)?;
    fs::create_dir(workspace.join("results"))?;
    write_json(
        &workspace.join("run-state.json"),
        &serde_json::json!({
            "benchmark_bundle_format_version": BENCHMARK_BUNDLE_FORMAT_VERSION,
            "phase": "exporting",
        }),
    )?;

    let export = (|| -> Result<(Vec<BenchmarkManifestArtifact>, BTreeMap<String, PathBuf>)> {
        let mut result_paths = BTreeMap::new();
        for record in records {
            let Some(result) = results.get(&record.run_id) else {
                continue;
            };
            let relative = PathBuf::from("results").join(&record.run_id);
            result.write_bundle(
                workspace.join(&relative),
                BundleExportOptions {
                    run_id: Some(record.run_id.clone()),
                    label: Some(format!("{}: {}", record.case_id, record.algorithm)),
                    include_previews: false,
                },
            )?;
            result_paths.insert(record.run_id.clone(), relative);
        }
        let relative_strings = result_paths
            .iter()
            .map(|(run_id, path)| (run_id.clone(), path.to_string_lossy().into_owned()))
            .collect();
        let mut artifacts = Vec::with_capacity(4);
        let writer_options = ParquetWriteOptions::default();
        write_benchmark_table(
            &workspace,
            "tables/runs.parquet",
            RUNS,
            &name,
            benchmark_runs_dataframe(records)?,
            &writer_options,
            &mut artifacts,
        )?;
        write_benchmark_table(
            &workspace,
            "tables/frames.parquet",
            FRAMES,
            &name,
            benchmark_frames_dataframe(records)?,
            &writer_options,
            &mut artifacts,
        )?;
        write_benchmark_table(
            &workspace,
            "tables/artifacts.parquet",
            ARTIFACTS,
            &name,
            benchmark_artifacts_dataframe(records, &relative_strings)?,
            &writer_options,
            &mut artifacts,
        )?;
        write_benchmark_table(
            &workspace,
            "tables/metadata.parquet",
            METADATA,
            &name,
            benchmark_metadata_dataframe(records)?,
            &writer_options,
            &mut artifacts,
        )?;
        Ok((artifacts, result_paths))
    })();
    let (artifacts, result_paths) = match export {
        Ok(value) => value,
        Err(error) => {
            let _ = write_json(
                &workspace.join("run-state.json"),
                &serde_json::json!({
                    "benchmark_bundle_format_version": BENCHMARK_BUNDLE_FORMAT_VERSION,
                    "phase": "failed",
                }),
            );
            return Err(error);
        }
    };
    fs::remove_file(workspace.join("run-state.json"))?;
    let manifest = BenchmarkManifest {
        benchmark_bundle_format_version: BENCHMARK_BUNDLE_FORMAT_VERSION,
        name,
        label: options.label,
        crate_version: env!("CARGO_PKG_VERSION").into(),
        artifacts,
        results: result_paths,
    };
    write_json(&workspace.join("manifest.json"), &manifest)?;
    fs::rename(workspace, &final_path)?;
    read_benchmark_bundle(final_path)
}

/// Verifies and reopens a complete benchmark bundle and its nested result manifests.
pub fn read_benchmark_bundle(path: impl AsRef<Path>) -> Result<BenchmarkBundle> {
    let path = path.as_ref();
    if path
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.ends_with(".inprogress"))
    {
        return Err(Error::IncompleteBundle(format!(
            "{} is an in-progress benchmark workspace",
            path.display()
        )));
    }
    let manifest_path = path.join("manifest.json");
    let manifest: BenchmarkManifest =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))
            .map_err(|error| Error::InvalidManifest(error.to_string()))?;
    if manifest.benchmark_bundle_format_version != BENCHMARK_BUNDLE_FORMAT_VERSION {
        return Err(Error::UnsupportedBundleVersion {
            actual: manifest.benchmark_bundle_format_version,
            supported: BENCHMARK_BUNDLE_FORMAT_VERSION,
        });
    }
    if manifest.name.is_empty() || manifest.crate_version.is_empty() {
        return Err(Error::InvalidManifest(
            "benchmark name and crate_version must be non-empty".into(),
        ));
    }
    let root = path.canonicalize()?;
    let mut handles = BTreeMap::new();
    for artifact in &manifest.artifacts {
        if ![RUNS, FRAMES, ARTIFACTS, METADATA].contains(&artifact.role.as_str()) {
            return Err(Error::UnsupportedArtifactRole(artifact.role.clone()));
        }
        validate_relative(&artifact.relative_path)?;
        let artifact_path = path.join(&artifact.relative_path);
        let canonical_artifact = artifact_path.canonicalize().map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::MissingArtifact {
                    role: artifact.role.clone(),
                }
            } else {
                Error::Io(error)
            }
        })?;
        if !canonical_artifact.starts_with(&root) {
            return Err(Error::InvalidRelativePath(
                artifact.relative_path.display().to_string(),
            ));
        }
        if fs::metadata(&artifact_path)?.len() != artifact.byte_size {
            return Err(Error::InvalidManifest(format!(
                "benchmark artifact {} has the wrong size",
                artifact.role
            )));
        }
        if artifact.media_type != "application/vnd.apache.parquet"
            || artifact.sha256.len() != 64
            || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::InvalidManifest(format!(
                "benchmark artifact {} has invalid media type or digest metadata",
                artifact.role
            )));
        }
        if sha256(&artifact_path)? != artifact.sha256 {
            return Err(Error::ArtifactHashMismatch {
                role: artifact.role.clone(),
            });
        }
        if handles
            .insert(
                artifact.role.clone(),
                BundleArtifact {
                    role: artifact.role.clone(),
                    path: artifact_path,
                    media_type: artifact.media_type.clone(),
                    byte_size: artifact.byte_size,
                    sha256: artifact.sha256.clone(),
                    dtype: None,
                    shape: None,
                },
            )
            .is_some()
        {
            return Err(Error::InvalidManifest(format!(
                "duplicate benchmark artifact role {}",
                artifact.role
            )));
        }
    }
    let required = |role: &str| {
        handles
            .get(role)
            .cloned()
            .ok_or_else(|| Error::MissingArtifact { role: role.into() })
    };
    let tables = BenchmarkBundleTables {
        runs: required(RUNS)?,
        frames: required(FRAMES)?,
        artifacts: required(ARTIFACTS)?,
        metadata: required(METADATA)?,
    };
    let successful_run_ids = validate_benchmark_tables(&tables, &manifest.results)?;
    let manifest_run_ids = manifest.results.keys().cloned().collect::<BTreeSet<_>>();
    if successful_run_ids != manifest_run_ids {
        return Err(Error::InvalidManifest(
            "successful benchmark runs and nested result bundles differ".into(),
        ));
    }
    let mut results = BTreeMap::new();
    for (run_id, relative) in &manifest.results {
        validate_relative(relative)?;
        let result_path = path.join(relative);
        let canonical_result = result_path.canonicalize().map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::MissingArtifact {
                    role: format!("results.{run_id}"),
                }
            } else {
                Error::Io(error)
            }
        })?;
        if !canonical_result.starts_with(&root) {
            return Err(Error::InvalidRelativePath(relative.display().to_string()));
        }
        let result = ResultBundle::read(result_path)?;
        if result.run_id != *run_id {
            return Err(Error::InconsistentRunId {
                expected: run_id.clone(),
                actual: result.run_id,
            });
        }
        results.insert(run_id.clone(), result);
    }
    Ok(BenchmarkBundle {
        path: path.into(),
        manifest_path,
        name: manifest.name,
        label: manifest.label,
        tables,
        results,
    })
}

fn validate_benchmark_tables(
    tables: &BenchmarkBundleTables,
    result_paths: &BTreeMap<String, PathBuf>,
) -> Result<BTreeSet<String>> {
    let runs = read_table(&tables.runs.path)?;
    validate_schema(
        RUNS,
        &runs,
        &[
            ("run_id", DataType::String, false),
            ("case_id", DataType::String, true),
            ("algorithm", DataType::String, false),
            ("algorithm_configuration", DataType::String, true),
            ("dataset_name", DataType::String, true),
            ("dataset_version", DataType::String, true),
            ("random_seed", DataType::UInt64, true),
            ("frame_count", DataType::UInt64, true),
            ("completed_iterations", DataType::UInt64, true),
            ("elapsed_seconds", DataType::Float64, false),
            ("final_objective", DataType::Float64, true),
            ("success", DataType::Boolean, false),
            ("error", DataType::String, true),
            ("repetition", DataType::UInt64, true),
            ("warmup", DataType::Boolean, true),
            ("benchmark_group", DataType::String, true),
            ("reference_run_id", DataType::String, true),
        ],
    )?;
    let run_ids = runs.column("run_id")?.str()?;
    let case_ids = runs.column("case_id")?.str()?;
    let successes = runs.column("success")?.bool()?;
    let errors = runs.column("error")?.str()?;
    let completed_iterations = runs.column("completed_iterations")?.u64()?;
    let final_objectives = runs.column("final_objective")?.f64()?;
    let frame_counts = runs.column("frame_count")?.u64()?;
    let mut all_run_ids = BTreeSet::new();
    let mut successful_run_ids = BTreeSet::new();
    let mut expected_frame_counts = BTreeMap::new();
    for row in 0..runs.height() {
        let missing = |column: &str| Error::InvalidParquetSchema {
            role: RUNS.into(),
            reason: format!("required benchmark column {column} contains null"),
        };
        let run_id = run_ids.get(row).ok_or_else(|| missing("run_id"))?;
        let case_id = case_ids.get(row).ok_or_else(|| missing("case_id"))?;
        let success = successes.get(row).ok_or_else(|| missing("success"))?;
        if run_id.is_empty() || case_id.is_empty() || !all_run_ids.insert(run_id.to_owned()) {
            return Err(Error::InvalidParquetSchema {
                role: RUNS.into(),
                reason: "run_id and case_id must be non-empty and run_id unique".into(),
            });
        }
        let valid_status = if success {
            errors.get(row).is_none() && completed_iterations.get(row).is_some()
        } else {
            errors.get(row).is_some_and(|value| !value.is_empty())
                && completed_iterations.get(row).is_none()
                && final_objectives.get(row).is_none()
        };
        if !valid_status {
            return Err(Error::InvalidParquetSchema {
                role: RUNS.into(),
                reason: "success, error, and result fields are inconsistent".into(),
            });
        }
        if success {
            successful_run_ids.insert(run_id.to_owned());
        }
        expected_frame_counts.insert(
            run_id.to_owned(),
            frame_counts
                .get(row)
                .ok_or_else(|| missing("frame_count"))?,
        );
    }

    for (artifact, role, schema) in [
        (
            &tables.frames,
            FRAMES,
            &[
                ("run_id", DataType::String, false),
                ("frame_index", DataType::UInt64, false),
                ("original_frame_index", DataType::UInt64, false),
                ("original_illumination_index", DataType::UInt64, true),
                ("normalized_l2", DataType::Float64, true),
            ][..],
        ),
        (
            &tables.artifacts,
            ARTIFACTS,
            &[
                ("run_id", DataType::String, false),
                ("role", DataType::String, false),
                ("relative_path", DataType::String, false),
            ][..],
        ),
        (
            &tables.metadata,
            METADATA,
            &[
                ("run_id", DataType::String, false),
                ("key", DataType::String, false),
                ("value", DataType::String, false),
            ][..],
        ),
    ] {
        let dataframe = read_table(&artifact.path)?;
        validate_schema(role, &dataframe, schema)?;
        for run_id in dataframe.column("run_id")?.str()?.iter().flatten() {
            if !all_run_ids.contains(run_id) {
                return Err(Error::InconsistentRunId {
                    expected: "one of the IDs in tables.runs".into(),
                    actual: run_id.into(),
                });
            }
        }
    }

    let frames = read_table(&tables.frames.path)?;
    let mut next_frame = BTreeMap::<String, u64>::new();
    for (run_id, frame_index) in frames
        .column("run_id")?
        .str()?
        .iter()
        .flatten()
        .zip(frames.column("frame_index")?.u64()?.into_no_null_iter())
    {
        let expected = next_frame.entry(run_id.into()).or_default();
        if frame_index != *expected {
            return Err(Error::InvalidParquetSchema {
                role: FRAMES.into(),
                reason: "frame indexes must be ordered, unique, and zero-based per run".into(),
            });
        }
        *expected += 1;
    }
    if expected_frame_counts
        .iter()
        .any(|(run_id, &count)| next_frame.get(run_id).copied().unwrap_or(0) != count)
    {
        return Err(Error::InvalidParquetSchema {
            role: FRAMES.into(),
            reason: "frame rows must match each run's declared frame_count".into(),
        });
    }

    let artifacts = read_table(&tables.artifacts.path)?;
    let artifact_run_ids = artifacts.column("run_id")?.str()?;
    let artifact_roles = artifacts.column("role")?.str()?;
    let artifact_paths = artifacts.column("relative_path")?.str()?;
    let mut artifact_ids = BTreeSet::new();
    for ((run_id, role), relative_path) in artifact_run_ids
        .iter()
        .flatten()
        .zip(artifact_roles.iter().flatten())
        .zip(artifact_paths.iter().flatten())
    {
        if role != "result_bundle"
            || !artifact_ids.insert(run_id.to_owned())
            || validate_relative(Path::new(relative_path)).is_err()
            || result_paths
                .get(run_id)
                .is_none_or(|expected| expected != Path::new(relative_path))
        {
            return Err(Error::InvalidParquetSchema {
                role: ARTIFACTS.into(),
                reason: "result artifacts must have one safe result_bundle path per run".into(),
            });
        }
    }
    if artifact_ids != successful_run_ids {
        return Err(Error::InvalidParquetSchema {
            role: ARTIFACTS.into(),
            reason: "artifact rows and successful runs differ".into(),
        });
    }

    let metadata = read_table(&tables.metadata.path)?;
    let mut metadata_keys = BTreeSet::new();
    for (run_id, key) in metadata
        .column("run_id")?
        .str()?
        .iter()
        .flatten()
        .zip(metadata.column("key")?.str()?.iter().flatten())
    {
        if !metadata_keys.insert((run_id.to_owned(), key.to_owned())) {
            return Err(Error::InvalidParquetSchema {
                role: METADATA.into(),
                reason: "metadata keys must be unique per run".into(),
            });
        }
    }
    Ok(successful_run_ids)
}

fn read_table(path: &Path) -> Result<DataFrame> {
    Ok(ParquetReader::new(File::open(path)?).finish()?)
}

fn validate_schema(
    role: &str,
    dataframe: &DataFrame,
    expected: &[(&str, DataType, bool)],
) -> Result<()> {
    if dataframe.width() != expected.len() {
        return Err(Error::InvalidParquetSchema {
            role: role.into(),
            reason: format!(
                "found {} columns, expected {}",
                dataframe.width(),
                expected.len()
            ),
        });
    }
    for &(name, ref dtype, nullable) in expected {
        let column = dataframe
            .column(name)
            .map_err(|error| Error::InvalidParquetSchema {
                role: role.into(),
                reason: error.to_string(),
            })?;
        if column.dtype() != dtype || (!nullable && column.null_count() != 0) {
            return Err(Error::InvalidParquetSchema {
                role: role.into(),
                reason: format!(
                    "column {name} has dtype {:?} and {} nulls; expected {dtype:?}",
                    column.dtype(),
                    column.null_count()
                ),
            });
        }
    }
    Ok(())
}

fn validate_records(
    records: &[BenchmarkRecord],
    results: &BTreeMap<String, ReconstructionResult>,
) -> Result<()> {
    if records.is_empty() {
        return Err(Error::InvalidParameter {
            name: "benchmark records",
            reason: "must contain at least one run".into(),
        });
    }
    let mut run_ids = BTreeSet::new();
    for record in records {
        if record.format_version != crate::benchmark::BENCHMARK_RECORD_FORMAT_VERSION
            || record.run_id.is_empty()
            || record.case_id.is_empty()
            || record.dataset_name.is_empty()
            || record.algorithm.is_empty()
            || record.crate_version.is_empty()
            || !run_ids.insert(record.run_id.as_str())
        {
            return Err(Error::InvalidParameter {
                name: "benchmark records",
                reason:
                    "records must use the supported version, required identities must be non-empty, and run_id must be unique"
                        .into(),
            });
        }
        if !record.elapsed_seconds.is_finite()
            || record.elapsed_seconds < 0.0
            || record.frame_count != record.frames.len()
            || record.success != record.error.is_none()
            || record
                .error
                .as_ref()
                .is_some_and(|message| message.is_empty())
        {
            return Err(Error::InvalidParameter {
                name: "benchmark records",
                reason: "run status, timing, frame count, or error message is inconsistent".into(),
            });
        }
        let run_path = Path::new(&record.run_id);
        if run_path.components().count() != 1
            || !matches!(run_path.components().next(), Some(Component::Normal(_)))
        {
            return Err(Error::InvalidParameter {
                name: "benchmark run_id",
                reason: "must be safe as one result-directory name".into(),
            });
        }
        if record.success != results.contains_key(&record.run_id) {
            return Err(Error::InvalidParameter {
                name: "benchmark results",
                reason: format!(
                    "successful run {} must have exactly one ReconstructionResult",
                    record.run_id
                ),
            });
        }
        for (index, frame) in record.frames.iter().enumerate() {
            if frame.frame_index != index {
                return Err(Error::InvalidParameter {
                    name: "benchmark frame records",
                    reason: "frame indexes must be unique, ordered, and zero-based".into(),
                });
            }
        }
    }
    if results
        .keys()
        .any(|run_id| !run_ids.contains(run_id.as_str()))
    {
        return Err(Error::InvalidParameter {
            name: "benchmark results",
            reason: "result map contains an unknown run_id".into(),
        });
    }
    Ok(())
}

fn validate_relative(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::InvalidRelativePath(path.display().to_string()));
    }
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn write_benchmark_table(
    workspace: &Path,
    relative_path: &str,
    role: &str,
    name: &str,
    mut dataframe: DataFrame,
    options: &ParquetWriteOptions,
    artifacts: &mut Vec<BenchmarkManifestArtifact>,
) -> Result<()> {
    let table_path = workspace.join(relative_path);
    write_parquet_file(&table_path, role, name, &mut dataframe, options)?;
    artifacts.push(BenchmarkManifestArtifact {
        role: role.into(),
        relative_path: relative_path.into(),
        media_type: "application/vnd.apache.parquet".into(),
        byte_size: fs::metadata(&table_path)?.len(),
        sha256: sha256(&table_path)?,
    });
    Ok(())
}

fn unique_paths(requested: &Path) -> Result<(PathBuf, PathBuf)> {
    let name = requested
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| Error::InvalidParameter {
            name: "benchmark bundle path",
            reason: "must name a UTF-8 directory".into(),
        })?;
    for suffix in 0_u64.. {
        let final_path = if suffix == 0 {
            requested.to_owned()
        } else {
            requested.with_file_name(format!("{name}-{suffix}"))
        };
        let final_name = final_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| Error::InvalidParameter {
                name: "benchmark bundle path",
                reason: "must name a UTF-8 directory".into(),
            })?;
        let workspace = final_path.with_file_name(format!("{final_name}.inprogress"));
        if !final_path.exists() && !workspace.exists() {
            return Ok((final_path, workspace));
        }
    }
    Err(Error::InvalidParameter {
        name: "benchmark bundle path",
        reason: "could not choose a unique directory".into(),
    })
}
