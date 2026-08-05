use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use ndarray::Array2;
use polars::prelude::{DataFrame, KeyValueMetadata, ParquetWriter};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Error, Result, complex,
    diagnostics::ReconstructionDiagnostics,
    evaluation::ReconstructionEvaluation,
    reconstruction::ReconstructionResult,
    tabular::{
        algorithm_metrics_dataframe, frame_calibration_dataframe, frame_diagnostics_dataframe,
        frame_evaluation_dataframe, history_dataframe, illumination_calibration_dataframe,
        iteration_diagnostics_dataframe, metadata_dataframe, raw_frame_statistics_dataframe,
        scalar_diagnostics_dataframe, summary_dataframe,
    },
};

use super::{
    bundle::ResultBundle,
    manifest::{
        ARRAY_BACKGROUND, ARRAY_FRAME_GAINS, ARRAY_ILLUMINATION_CALIBRATION, ARRAY_OBJECT,
        ARRAY_OBJECT_SPECTRUM, ARRAY_PUPIL, ARRAY_PUPIL_SUPPORT, BUNDLE_FORMAT_VERSION,
        BundleExportOptions, BundleManifest, DOMAIN_DIAGNOSTICS, DOMAIN_EVALUATION,
        DOMAIN_PHYSICAL_ILLUMINATION, DatasetIdentity, ManifestArtifact, PREVIEW_FOURIER_COVERAGE,
        PREVIEW_OBJECT_AMPLITUDE, PREVIEW_OBJECT_PHASE, PREVIEW_PUPIL_AMPLITUDE,
        PREVIEW_PUPIL_PHASE, ResultDescriptor, TABLE_ALGORITHM_METRICS, TABLE_FRAME_CALIBRATION,
        TABLE_FRAME_DIAGNOSTICS, TABLE_FRAME_EVALUATION, TABLE_HISTORY,
        TABLE_ILLUMINATION_CALIBRATION, TABLE_ITERATION_DIAGNOSTICS, TABLE_METADATA,
        TABLE_RAW_FRAME_STATISTICS, TABLE_SCALAR_DIAGNOSTICS, TABLE_SUMMARY,
    },
    npy,
    options::ParquetWriteOptions,
};

// Kept private to this module so a future manifest migration cannot
// accidentally infer dataset identity from arbitrary dynamic metadata.
const DATASET_NAME_KEY: &str = "dataset_name";
const DATASET_VERSION_KEY: &str = "dataset_version";

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct RunState<'a> {
    bundle_format_version: u32,
    run_id: &'a str,
    phase: &'a str,
}

pub(crate) fn write_result_bundle(
    result: &ReconstructionResult,
    requested_path: &Path,
    options: BundleExportOptions,
    diagnostics: Option<&ReconstructionDiagnostics>,
    evaluation: Option<&ReconstructionEvaluation>,
) -> Result<ResultBundle> {
    result.validate()?;
    let run_id = options
        .run_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    validate_run_id(&run_id)?;
    let (final_path, workspace) = unique_paths(requested_path)?;
    let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    fs::create_dir(&workspace)?;
    fs::create_dir(workspace.join("checkpoints"))?;
    write_json_atomic(
        &workspace.join("run-state.json"),
        &RunState {
            bundle_format_version: BUNDLE_FORMAT_VERSION,
            run_id: &run_id,
            phase: "exporting",
        },
    )?;

    let export = (|| -> Result<Vec<ManifestArtifact>> {
        let mut artifacts = Vec::new();
        let parquet_options = ParquetWriteOptions::default();

        write_table(
            &workspace,
            "tables/summary.parquet",
            TABLE_SUMMARY,
            summary_dataframe(&run_id, result)?,
            &run_id,
            &parquet_options,
            &mut artifacts,
        )?;
        write_table(
            &workspace,
            "tables/history.parquet",
            TABLE_HISTORY,
            history_dataframe(&run_id, &result.trace)?,
            &run_id,
            &parquet_options,
            &mut artifacts,
        )?;
        if !result.trace.algorithm_metrics.is_empty() {
            write_table(
                &workspace,
                "tables/algorithm_metrics.parquet",
                TABLE_ALGORITHM_METRICS,
                algorithm_metrics_dataframe(&run_id, &result.trace)?,
                &run_id,
                &parquet_options,
                &mut artifacts,
            )?;
        }
        if let Some(diagnostics) = diagnostics {
            if !diagnostics.iteration_diagnostics.is_empty() {
                write_table(
                    &workspace,
                    "tables/iteration_diagnostics.parquet",
                    TABLE_ITERATION_DIAGNOSTICS,
                    iteration_diagnostics_dataframe(&run_id, diagnostics)?,
                    &run_id,
                    &parquet_options,
                    &mut artifacts,
                )?;
            }
            if !diagnostics.frame_diagnostics.is_empty() {
                write_table(
                    &workspace,
                    "tables/frame_diagnostics.parquet",
                    TABLE_FRAME_DIAGNOSTICS,
                    frame_diagnostics_dataframe(&run_id, diagnostics)?,
                    &run_id,
                    &parquet_options,
                    &mut artifacts,
                )?;
            }
            if !diagnostics.raw_frame_statistics.is_empty() {
                write_table(
                    &workspace,
                    "tables/raw_frame_statistics.parquet",
                    TABLE_RAW_FRAME_STATISTICS,
                    raw_frame_statistics_dataframe(&run_id, diagnostics)?,
                    &run_id,
                    &parquet_options,
                    &mut artifacts,
                )?;
            }
            write_json_artifact(
                &workspace,
                "domain/diagnostics.json",
                DOMAIN_DIAGNOSTICS,
                diagnostics,
                &mut artifacts,
            )?;
        }
        if let Some(evaluation) = evaluation {
            if let Some(frames) = &evaluation.intensity
                && !frames.per_frame.is_empty()
            {
                write_table(
                    &workspace,
                    "tables/frame_evaluation.parquet",
                    TABLE_FRAME_EVALUATION,
                    frame_evaluation_dataframe(&run_id, frames)?,
                    &run_id,
                    &parquet_options,
                    &mut artifacts,
                )?;
            }
            write_json_artifact(
                &workspace,
                "domain/evaluation.json",
                DOMAIN_EVALUATION,
                evaluation,
                &mut artifacts,
            )?;
        }
        if let (Some(calibration), Some(model)) = (
            &result.physical_illumination_calibration,
            &result.calibrated_model,
        ) {
            write_json_artifact(
                &workspace,
                "domain/physical_illumination.json",
                DOMAIN_PHYSICAL_ILLUMINATION,
                &(calibration, model),
                &mut artifacts,
            )?;
        }
        if result.calibrated_illumination.is_some() {
            write_table(
                &workspace,
                "tables/illumination_calibration.parquet",
                TABLE_ILLUMINATION_CALIBRATION,
                illumination_calibration_dataframe(&run_id, result)?,
                &run_id,
                &parquet_options,
                &mut artifacts,
            )?;
        }
        if result.recovered_frame_gains.is_some() || result.recovered_background.is_some() {
            write_table(
                &workspace,
                "tables/frame_calibration.parquet",
                TABLE_FRAME_CALIBRATION,
                frame_calibration_dataframe(&run_id, result)?,
                &run_id,
                &parquet_options,
                &mut artifacts,
            )?;
        }
        if !result.scalar_diagnostics.is_empty() {
            write_table(
                &workspace,
                "tables/scalar_diagnostics.parquet",
                TABLE_SCALAR_DIAGNOSTICS,
                scalar_diagnostics_dataframe(&run_id, &result.scalar_diagnostics)?,
                &run_id,
                &parquet_options,
                &mut artifacts,
            )?;
        }
        let metadata = metadata_dataframe(&run_id, &result.metadata)?;
        if metadata.height() > 0 {
            write_table(
                &workspace,
                "tables/metadata.parquet",
                TABLE_METADATA,
                metadata,
                &run_id,
                &parquet_options,
                &mut artifacts,
            )?;
        }

        write_arrays(&workspace, result, &mut artifacts)?;
        if options.include_previews {
            write_previews(&workspace, result, diagnostics, &mut artifacts)?;
        }
        Ok(artifacts)
    })();

    let artifacts = match export {
        Ok(artifacts) => artifacts,
        Err(error) => {
            let _ = write_json_atomic(
                &workspace.join("run-state.json"),
                &RunState {
                    bundle_format_version: BUNDLE_FORMAT_VERSION,
                    run_id: &run_id,
                    phase: "failed",
                },
            );
            return Err(error);
        }
    };

    for artifact in &artifacts {
        validate_written_artifact(&workspace, artifact)?;
    }
    fs::remove_dir_all(workspace.join("checkpoints"))?;
    fs::remove_file(workspace.join("run-state.json"))?;

    let dataset = result
        .metadata
        .get(DATASET_NAME_KEY)
        .map(|name| DatasetIdentity {
            name: name.clone(),
            version: result.metadata.get(DATASET_VERSION_KEY).cloned(),
        });
    let provenance = result
        .metadata
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("provenance.")
                .map(|key| (key.to_owned(), value.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let manifest = BundleManifest {
        bundle_format_version: BUNDLE_FORMAT_VERSION,
        run_id,
        label: options.label,
        crate_version: env!("CARGO_PKG_VERSION").into(),
        provenance,
        dataset,
        runtime: result.runtime.clone(),
        result: ResultDescriptor {
            reconstruction_shape: [result.object.nrows() as u64, result.object.ncols() as u64],
            image_shape: [
                result.recovered_pupil.shape().0 as u64,
                result.recovered_pupil.shape().1 as u64,
            ],
        },
        artifacts,
    };
    // The manifest is deliberately the final write within the workspace.
    write_json_sync(&workspace.join("manifest.json"), &manifest)?;
    fs::rename(&workspace, &final_path)?;
    ResultBundle::read(&final_path)
}

fn write_table(
    workspace: &Path,
    relative_path: &str,
    role: &str,
    mut dataframe: DataFrame,
    run_id: &str,
    options: &ParquetWriteOptions,
    artifacts: &mut Vec<ManifestArtifact>,
) -> Result<()> {
    let path = workspace.join(relative_path);
    create_parent(&path)?;
    write_parquet_file(&path, role, run_id, &mut dataframe, options)?;
    artifacts.push(describe_artifact(
        workspace,
        role,
        relative_path,
        "application/vnd.apache.parquet",
        None,
        None,
    )?);
    Ok(())
}

pub(crate) fn write_parquet_file(
    path: &Path,
    role: &str,
    bundle_id: &str,
    dataframe: &mut DataFrame,
    options: &ParquetWriteOptions,
) -> Result<()> {
    create_parent(path)?;
    let metadata = KeyValueMetadata::from_static(vec![
        ("fpm.table_role".into(), role.into()),
        (
            "fpm.bundle_format_version".into(),
            BUNDLE_FORMAT_VERSION.to_string(),
        ),
        ("fpm.run_id".into(), bundle_id.into()),
        ("fpm.crate_version".into(), env!("CARGO_PKG_VERSION").into()),
    ]);
    let file = File::create(path)?;
    ParquetWriter::new(file)
        .with_compression(options.compression)
        .with_statistics(options.statistics)
        .with_row_group_size(options.row_group_size)
        .with_data_page_size(options.data_page_size)
        .set_parallel(options.parallel)
        .with_key_value_metadata(Some(metadata))
        .finish(dataframe)?;
    Ok(())
}

fn write_arrays(
    workspace: &Path,
    result: &ReconstructionResult,
    artifacts: &mut Vec<ManifestArtifact>,
) -> Result<()> {
    let array_directory = workspace.join("arrays");
    fs::create_dir_all(&array_directory)?;
    write_array(
        workspace,
        "arrays/object.npy",
        ARRAY_OBJECT,
        "<c16",
        &[result.object.nrows(), result.object.ncols()],
        |path| npy::write_complex2(path, result.object.view()),
        artifacts,
    )?;
    write_array(
        workspace,
        "arrays/object_spectrum.npy",
        ARRAY_OBJECT_SPECTRUM,
        "<c16",
        &[
            result.object_spectrum.nrows(),
            result.object_spectrum.ncols(),
        ],
        |path| npy::write_complex2(path, result.object_spectrum.view()),
        artifacts,
    )?;
    write_array(
        workspace,
        "arrays/pupil.npy",
        ARRAY_PUPIL,
        "<c16",
        &[
            result.recovered_pupil.shape().0,
            result.recovered_pupil.shape().1,
        ],
        |path| npy::write_complex2(path, result.recovered_pupil.values()),
        artifacts,
    )?;
    write_array(
        workspace,
        "arrays/pupil_support.npy",
        ARRAY_PUPIL_SUPPORT,
        "|u1",
        &[
            result.recovered_pupil.shape().0,
            result.recovered_pupil.shape().1,
        ],
        |path| npy::write_u8_2(path, result.recovered_pupil.support()),
        artifacts,
    )?;
    if let Some(values) = &result.calibrated_illumination {
        let flattened: Vec<f64> = values
            .iter()
            .flat_map(|&(row, column)| [row, column])
            .collect();
        let array = ndarray::ArrayView2::from_shape((values.len(), 2), &flattened)?;
        write_array(
            workspace,
            "arrays/illumination_calibration.npy",
            ARRAY_ILLUMINATION_CALIBRATION,
            "<f8",
            &[values.len(), 2],
            |path| npy::write_f64_2(path, array),
            artifacts,
        )?;
    }
    if let Some(values) = &result.recovered_frame_gains {
        write_array(
            workspace,
            "arrays/frame_gains.npy",
            ARRAY_FRAME_GAINS,
            "<f8",
            &[values.len()],
            |path| npy::write_f64_1(path, values),
            artifacts,
        )?;
    }
    if let Some(values) = &result.recovered_background {
        write_array(
            workspace,
            "arrays/background.npy",
            ARRAY_BACKGROUND,
            "<f8",
            &[values.len()],
            |path| npy::write_f64_1(path, values),
            artifacts,
        )?;
    }
    Ok(())
}

fn write_array(
    workspace: &Path,
    relative_path: &str,
    role: &str,
    dtype: &str,
    shape: &[usize],
    writer: impl FnOnce(&Path) -> Result<()>,
    artifacts: &mut Vec<ManifestArtifact>,
) -> Result<()> {
    let path = workspace.join(relative_path);
    create_parent(&path)?;
    writer(&path)?;
    artifacts.push(describe_artifact(
        workspace,
        role,
        relative_path,
        "application/x-npy",
        Some(dtype.into()),
        Some(shape.iter().map(|&value| value as u64).collect()),
    )?);
    Ok(())
}

fn write_previews(
    workspace: &Path,
    result: &ReconstructionResult,
    diagnostics: Option<&ReconstructionDiagnostics>,
    artifacts: &mut Vec<ManifestArtifact>,
) -> Result<()> {
    let directory = workspace.join("previews");
    fs::create_dir_all(&directory)?;
    let previews = [
        (
            "previews/object_amplitude.png",
            PREVIEW_OBJECT_AMPLITUDE,
            result.amplitude.view(),
            false,
        ),
        (
            "previews/object_phase.png",
            PREVIEW_OBJECT_PHASE,
            result.phase.view(),
            true,
        ),
    ];
    for (relative, role, values, signed) in previews {
        crate::reconstruction::save_grayscale(values, workspace.join(relative), signed)?;
        artifacts.push(describe_artifact(
            workspace,
            role,
            relative,
            "image/png",
            None,
            None,
        )?);
    }
    let pupil_amplitude = complex::amplitude(result.recovered_pupil.values());
    let pupil_phase = complex::phase(result.recovered_pupil.values());
    for (relative, role, values, signed) in [
        (
            "previews/pupil_amplitude.png",
            PREVIEW_PUPIL_AMPLITUDE,
            pupil_amplitude.view(),
            false,
        ),
        (
            "previews/pupil_phase.png",
            PREVIEW_PUPIL_PHASE,
            pupil_phase.view(),
            true,
        ),
    ] {
        crate::reconstruction::save_grayscale(values, workspace.join(relative), signed)?;
        artifacts.push(describe_artifact(
            workspace,
            role,
            relative,
            "image/png",
            None,
            None,
        )?);
    }
    if let Some(coverage) = diagnostics.and_then(|value| value.coverage.as_ref()) {
        if !coverage.pupil_radius_px.is_finite()
            || coverage.pupil_radius_px < 0.0
            || coverage
                .pupil_centers_px
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidModel(
                "Fourier coverage preview contains invalid geometry".into(),
            ));
        }
        let (height, width) = result.object.dim();
        let mut counts = Array2::<f64>::zeros((height, width));
        let radius = coverage.pupil_radius_px;
        let radius_squared = radius * radius;
        for &[center_x, center_y] in &coverage.pupil_centers_px {
            let row_start = (center_y - radius)
                .floor()
                .clamp(0.0, height.saturating_sub(1) as f64) as usize;
            let row_end = (center_y + radius)
                .ceil()
                .clamp(0.0, height.saturating_sub(1) as f64) as usize;
            let column_start = (center_x - radius)
                .floor()
                .clamp(0.0, width.saturating_sub(1) as f64) as usize;
            let column_end = (center_x + radius)
                .ceil()
                .clamp(0.0, width.saturating_sub(1) as f64) as usize;
            for row in row_start..=row_end {
                for column in column_start..=column_end {
                    let distance_squared =
                        (row as f64 - center_y).powi(2) + (column as f64 - center_x).powi(2);
                    if distance_squared <= radius_squared {
                        counts[(row, column)] += 1.0;
                    }
                }
            }
        }
        let relative = "previews/fourier_coverage.png";
        crate::reconstruction::save_grayscale(counts.view(), workspace.join(relative), false)?;
        artifacts.push(describe_artifact(
            workspace,
            PREVIEW_FOURIER_COVERAGE,
            relative,
            "image/png",
            None,
            None,
        )?);
    }
    Ok(())
}

fn write_json_artifact(
    workspace: &Path,
    relative_path: &str,
    role: &str,
    value: &impl Serialize,
    artifacts: &mut Vec<ManifestArtifact>,
) -> Result<()> {
    let path = workspace.join(relative_path);
    create_parent(&path)?;
    write_json_sync(&path, value)?;
    artifacts.push(describe_artifact(
        workspace,
        role,
        relative_path,
        "application/json",
        None,
        None,
    )?);
    Ok(())
}

fn create_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn describe_artifact(
    workspace: &Path,
    role: &str,
    relative_path: &str,
    media_type: &str,
    dtype: Option<String>,
    shape: Option<Vec<u64>>,
) -> Result<ManifestArtifact> {
    let path = workspace.join(relative_path);
    Ok(ManifestArtifact {
        role: role.into(),
        relative_path: PathBuf::from(relative_path),
        media_type: media_type.into(),
        byte_size: fs::metadata(&path)?.len(),
        sha256: sha256(&path)?,
        dtype,
        shape,
    })
}

pub(crate) fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_written_artifact(workspace: &Path, artifact: &ManifestArtifact) -> Result<()> {
    let path = workspace.join(&artifact.relative_path);
    if fs::metadata(&path)?.len() != artifact.byte_size {
        return Err(Error::InvalidManifest(format!(
            "artifact {} changed size during export",
            artifact.role
        )));
    }
    if sha256(&path)? != artifact.sha256 {
        return Err(Error::ArtifactHashMismatch {
            role: artifact.role.clone(),
        });
    }
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("tmp");
    write_json_sync(&temporary, value)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_json_sync(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn validate_run_id(run_id: &str) -> Result<()> {
    if run_id.is_empty() || run_id.len() > 256 || run_id.chars().any(char::is_control) {
        return Err(Error::InvalidParameter {
            name: "run_id",
            reason: "must contain 1 to 256 non-control UTF-8 characters".into(),
        });
    }
    Ok(())
}

fn unique_paths(requested: &Path) -> Result<(PathBuf, PathBuf)> {
    if requested.file_name().is_none() {
        return Err(Error::InvalidParameter {
            name: "bundle path",
            reason: "must name a result directory".into(),
        });
    }
    for suffix in 0_u64.. {
        let final_path = if suffix == 0 {
            requested.to_owned()
        } else {
            let name = requested
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| Error::InvalidParameter {
                    name: "bundle path",
                    reason: "directory name must be valid UTF-8".into(),
                })?;
            requested.with_file_name(format!("{name}-{suffix}"))
        };
        let name = final_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| Error::InvalidParameter {
                name: "bundle path",
                reason: "directory name must be valid UTF-8".into(),
            })?;
        let workspace = final_path.with_file_name(format!("{name}.inprogress"));
        if !final_path.exists() && !workspace.exists() {
            return Ok((final_path, workspace));
        }
    }
    Err(Error::InvalidParameter {
        name: "bundle path",
        reason: "could not generate a unique output name".into(),
    })
}
