//! Reproducible single-case reconstruction benchmarks.
//!
//! The runner is generic over the existing reconstruction and measurement
//! traits. Calling it once per concrete algorithm avoids a second algorithm
//! registry or trait-object hierarchy.

#[cfg(feature = "parquet")]
use std::fs;
use std::{
    collections::BTreeMap,
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    time::Instant,
};

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    Complex64, Result,
    algorithms::ReconstructionAlgorithm,
    evaluation::{evaluate_frame_intensity, evaluate_reconstruction_with_problem},
    measurements::MeasurementRead,
    model::ImagePlaneModel,
    reconstruction::{ReconstructionProblem, ReconstructionResult},
};

pub const BENCHMARK_RECORD_FORMAT_VERSION: u32 = 1;
pub const SMOKE_BENCHMARK_PROFILE: &str = "smoke";
pub const CPU_BENCHMARK_PROFILE: &str = "cpu";

/// Stable metadata for a named benchmark profile.
///
/// Profiles describe runtime and output expectations only. Examples still list
/// concrete algorithms explicitly so the benchmark layer does not become an
/// algorithm registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkProfile {
    pub name: &'static str,
    pub description: &'static str,
    pub expected_runtime: &'static str,
    pub output_directory: &'static str,
    pub algorithms: &'static [&'static str],
}

impl BenchmarkProfile {
    pub fn output_path(&self) -> PathBuf {
        PathBuf::from(self.output_directory)
    }
}

pub const BENCHMARK_PROFILES: &[BenchmarkProfile] = &[
    BenchmarkProfile {
        name: SMOKE_BENCHMARK_PROFILE,
        description: "offline synthetic sanity profile for all implemented CPU algorithms",
        expected_runtime: "under 1 minute on a typical laptop CPU",
        output_directory: "target/benchmark-results/smoke",
        algorithms: &[
            "AlternatingProjection",
            "Fpie",
            "Epry",
            "Admm",
            "GradientDescent",
        ],
    },
    BenchmarkProfile {
        name: CPU_BENCHMARK_PROFILE,
        description: "offline synthetic CPU comparison with longer iteration counts",
        expected_runtime: "1-5 minutes on a typical laptop CPU",
        output_directory: "target/benchmark-results/cpu",
        algorithms: &[
            "AlternatingProjection",
            "Fpie",
            "Epry",
            "Admm",
            "GradientDescent",
        ],
    },
];

pub fn benchmark_profile(name: &str) -> Option<&'static BenchmarkProfile> {
    BENCHMARK_PROFILES
        .iter()
        .find(|profile| profile.name == name)
}

/// Adds the selected profile metadata to one benchmark record.
pub fn annotate_benchmark_profile(record: &mut BenchmarkRecord, profile: &BenchmarkProfile) {
    record
        .metadata
        .insert("benchmark_profile".into(), profile.name.into());
    record.metadata.insert(
        "benchmark_profile_expected_runtime".into(),
        profile.expected_runtime.into(),
    );
    record.metadata.insert(
        "benchmark_profile_output_directory".into(),
        profile.output_directory.into(),
    );
}

/// Serializable summary of one algorithm run on one immutable problem.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRecord {
    pub format_version: u32,
    /// Deterministic identity of the immutable case configuration.
    pub case_id: String,
    /// Unique identity of this execution.
    pub run_id: String,
    pub dataset_name: String,
    pub dataset_version: Option<String>,
    pub preset_name: Option<String>,
    pub crate_version: String,
    pub random_seed: Option<u64>,
    /// Original-image crop as `[row, column, height, width]`, when applicable.
    pub spatial_crop: Option<[usize; 4]>,
    pub algorithm: String,
    pub algorithm_configuration: String,
    pub success: bool,
    pub error: Option<String>,
    pub frame_count: usize,
    pub image_shape: [usize; 2],
    pub reconstruction_shape: [usize; 2],
    pub completed_iterations: usize,
    pub elapsed_seconds: f64,
    pub initial_objective: Option<f64>,
    pub final_objective: Option<f64>,
    /// Final objective divided by the initial objective.
    pub final_to_initial_objective_ratio: Option<f64>,
    pub amplitude_rmse: Option<f64>,
    pub phase_rmse: Option<f64>,
    pub complex_field_relative_error: Option<f64>,
    pub fourier_domain_relative_error: Option<f64>,
    pub pupil_amplitude_rmse: Option<f64>,
    pub pupil_phase_rmse: Option<f64>,
    pub illumination_position_rmse: Option<f64>,
    pub per_frame_residual_mean: Option<f64>,
    pub per_frame_residual_max: Option<f64>,
    pub frames: Vec<BenchmarkFrameRecord>,
    pub output_paths: Vec<PathBuf>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkFrameRecord {
    pub frame_index: usize,
    pub original_frame_index: usize,
    pub original_illumination_index: Option<usize>,
    pub normalized_l2: Option<f64>,
}

impl BenchmarkRecord {
    /// Builds a normalized benchmark record for an already completed result.
    ///
    /// Callers provide the deterministic `case_id`; a fresh `run_id` is
    /// generated for this execution.
    pub fn from_result(
        case_id: impl Into<String>,
        dataset_name: impl Into<String>,
        algorithm_configuration: impl Into<String>,
        result: &ReconstructionResult,
    ) -> Self {
        let image_shape = result.recovered_pupil.shape();
        let reconstruction_shape = result.object.dim();
        let frame_count = result
            .metadata
            .get("frame_count")
            .and_then(|value| value.parse().ok())
            .or_else(|| result.recovered_frame_gains.as_ref().map(Vec::len))
            .unwrap_or(0);
        let initial_objective = result
            .trace
            .iterations
            .first()
            .map(|record| record.objective);
        let final_objective = result.trace.final_objective();
        Self {
            format_version: BENCHMARK_RECORD_FORMAT_VERSION,
            case_id: case_id.into(),
            run_id: Uuid::new_v4().to_string(),
            dataset_name: dataset_name.into(),
            dataset_version: result.metadata.get("dataset_version").cloned(),
            preset_name: result.metadata.get("preset_name").cloned(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            random_seed: result
                .metadata
                .get("random_seed")
                .and_then(|value| value.parse().ok()),
            spatial_crop: None,
            algorithm: result.runtime.algorithm.clone(),
            algorithm_configuration: algorithm_configuration.into(),
            success: true,
            error: None,
            frame_count,
            image_shape: [image_shape.0, image_shape.1],
            reconstruction_shape: [reconstruction_shape.0, reconstruction_shape.1],
            completed_iterations: result.runtime.completed_iterations,
            elapsed_seconds: result.runtime.elapsed_seconds,
            initial_objective,
            final_objective,
            final_to_initial_objective_ratio: initial_objective.and_then(|initial| {
                final_objective
                    .filter(|_| initial.abs() > f64::EPSILON)
                    .map(|final_value| final_value / initial)
            }),
            amplitude_rmse: None,
            phase_rmse: None,
            complex_field_relative_error: None,
            fourier_domain_relative_error: None,
            pupil_amplitude_rmse: None,
            pupil_phase_rmse: None,
            illumination_position_rmse: None,
            per_frame_residual_mean: None,
            per_frame_residual_max: None,
            frames: (0..frame_count)
                .map(|frame_index| BenchmarkFrameRecord {
                    frame_index,
                    original_frame_index: frame_index,
                    original_illumination_index: None,
                    normalized_l2: None,
                })
                .collect(),
            output_paths: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}

/// Runs one concrete algorithm and always returns a record. Reconstruction or
/// metric failures are stored in `record.error`; successful reconstruction data
/// is returned separately so callers may inspect or save it.
pub fn run_benchmark_case<A, M>(
    dataset_name: impl Into<String>,
    algorithm_configuration: impl Into<String>,
    algorithm: A,
    problem: &ReconstructionProblem<M>,
    ground_truth: Option<ArrayView2<'_, Complex64>>,
    true_model: Option<&ImagePlaneModel>,
    valid_object_mask: Option<ArrayView2<'_, u8>>,
) -> (BenchmarkRecord, Option<ReconstructionResult>)
where
    A: ReconstructionAlgorithm,
    M: MeasurementRead,
{
    let algorithm_name = short_type_name::<A>().to_owned();
    let image_shape = problem.measurements.image_shape();
    let reconstruction_shape = problem.model.reconstruction_shape;
    let mut record = BenchmarkRecord {
        format_version: BENCHMARK_RECORD_FORMAT_VERSION,
        case_id: String::new(),
        run_id: Uuid::new_v4().to_string(),
        dataset_name: dataset_name.into(),
        dataset_version: None,
        preset_name: None,
        crate_version: env!("CARGO_PKG_VERSION").into(),
        random_seed: None,
        spatial_crop: None,
        algorithm: algorithm_name,
        algorithm_configuration: algorithm_configuration.into(),
        success: false,
        error: None,
        frame_count: problem.measurements.frame_count(),
        image_shape: [image_shape.0, image_shape.1],
        reconstruction_shape: [reconstruction_shape.0, reconstruction_shape.1],
        completed_iterations: 0,
        elapsed_seconds: 0.0,
        initial_objective: None,
        final_objective: None,
        final_to_initial_objective_ratio: None,
        amplitude_rmse: None,
        phase_rmse: None,
        complex_field_relative_error: None,
        fourier_domain_relative_error: None,
        pupil_amplitude_rmse: None,
        pupil_phase_rmse: None,
        illumination_position_rmse: None,
        per_frame_residual_mean: None,
        per_frame_residual_max: None,
        frames: problem
            .measurements
            .frame_metadata()
            .iter()
            .enumerate()
            .map(|(index, metadata)| BenchmarkFrameRecord {
                frame_index: index,
                original_frame_index: metadata.original_frame_index.unwrap_or(index),
                original_illumination_index: metadata
                    .original_illumination_index
                    .or(metadata.illumination_index),
                normalized_l2: None,
            })
            .collect(),
        output_paths: Vec::new(),
        metadata: BTreeMap::new(),
    };
    record.case_id = format!("{:016x}", case_hash(&record));

    let started = Instant::now();
    let result = match algorithm.run(problem) {
        Ok(result) => result,
        Err(error) => {
            record.elapsed_seconds = started.elapsed().as_secs_f64();
            record.error = Some(error.to_string());
            return (record, None);
        }
    };
    record.elapsed_seconds = started.elapsed().as_secs_f64();
    record.algorithm = result.runtime.algorithm.clone();
    record.completed_iterations = result.runtime.completed_iterations;
    record.initial_objective = result.trace.iterations.first().map(|entry| entry.objective);
    record.final_objective = result.trace.final_objective();
    record.final_to_initial_objective_ratio = record.initial_objective.and_then(|initial| {
        record
            .final_objective
            .filter(|_| initial.abs() > f64::EPSILON)
            .map(|final_objective| final_objective / initial)
    });

    let residuals: Vec<f64> = if let Some(truth) = ground_truth {
        let metrics = evaluate_reconstruction_with_problem(
            &result,
            problem,
            truth,
            true_model,
            valid_object_mask,
        );
        match metrics {
            Ok(metrics) => {
                record.amplitude_rmse = Some(metrics.object.amplitude_rmse);
                record.phase_rmse = Some(metrics.object.phase_rmse);
                record.complex_field_relative_error = Some(metrics.object.complex_nrmse);
                record.fourier_domain_relative_error = Some(metrics.object.fourier_nrmse);
                record.pupil_amplitude_rmse =
                    metrics.pupil.as_ref().map(|value| value.amplitude_rmse);
                record.pupil_phase_rmse = metrics.pupil.as_ref().map(|value| value.phase_rmse);
                record.illumination_position_rmse = metrics
                    .illumination
                    .as_ref()
                    .map(|value| value.position_rmse);
                metrics
                    .intensity
                    .map(|value| {
                        value
                            .per_frame
                            .into_iter()
                            .map(|frame| frame.normalized_l2)
                            .collect()
                    })
                    .unwrap_or_default()
            }
            Err(error) => {
                record.error = Some(format!("benchmark metric calculation failed: {error}"));
                return (record, Some(result));
            }
        }
    } else {
        match evaluate_frame_intensity(&result, problem) {
            Ok(metrics) => metrics
                .per_frame
                .into_iter()
                .map(|frame| frame.normalized_l2)
                .collect(),
            Err(error) => {
                record.error = Some(format!("benchmark residual calculation failed: {error}"));
                return (record, Some(result));
            }
        }
    };
    if !residuals.is_empty() {
        record.per_frame_residual_mean =
            Some(residuals.iter().sum::<f64>() / residuals.len() as f64);
        record.per_frame_residual_max = residuals.iter().copied().reduce(f64::max);
    }
    for (frame, residual) in record.frames.iter_mut().zip(residuals) {
        frame.normalized_l2 = Some(residual);
    }
    let mut result = result;
    result
        .metadata
        .insert("case_id".into(), record.case_id.clone());
    result
        .metadata
        .insert("dataset_name".into(), record.dataset_name.clone());
    if let Some(version) = &record.dataset_version {
        result
            .metadata
            .insert("dataset_version".into(), version.clone());
    }
    result.metadata.insert(
        "algorithm_configuration".into(),
        record.algorithm_configuration.clone(),
    );
    result
        .metadata
        .insert("frame_count".into(), record.frame_count.to_string());
    if let Some(seed) = record.random_seed {
        result
            .metadata
            .insert("random_seed".into(), seed.to_string());
    }
    record.success = true;
    (record, Some(result))
}

/// Saves the standard reconstruction artifacts for a benchmark case and adds
/// their paths to the record.
#[cfg(feature = "parquet")]
pub fn save_benchmark_outputs(
    record: &mut BenchmarkRecord,
    result: &ReconstructionResult,
    directory: impl AsRef<Path>,
) -> Result<()> {
    let directory = directory.as_ref();
    fs::create_dir_all(directory)?;
    let stem = format!(
        "{}-{}-{:016x}",
        safe_stem(&record.dataset_name),
        safe_stem(&record.algorithm),
        case_hash(record),
    );
    let outputs = [
        directory.join(format!("{stem}-amplitude.png")),
        directory.join(format!("{stem}-phase.png")),
        directory.join(format!("{stem}-result")),
        directory.join(format!("{stem}-trace.csv")),
    ];
    result.save_amplitude(&outputs[0])?;
    result.save_phase(&outputs[1])?;
    result.write_bundle(
        &outputs[2],
        crate::reconstruction::BundleExportOptions {
            run_id: Some(record.run_id.clone()),
            label: None,
            include_previews: true,
        },
    )?;
    result.save_trace_csv(&outputs[3])?;
    record.output_paths.extend(outputs);
    Ok(())
}

#[cfg(not(feature = "parquet"))]
pub fn save_benchmark_outputs(
    _record: &mut BenchmarkRecord,
    _result: &ReconstructionResult,
    _directory: impl AsRef<Path>,
) -> Result<()> {
    Err(crate::Error::Unsupported(
        "benchmark result bundles require the `parquet` feature".into(),
    ))
}

pub fn write_benchmark_json(records: &[BenchmarkRecord], path: impl AsRef<Path>) -> Result<()> {
    #[derive(Serialize)]
    struct Report<'a> {
        format_version: u32,
        records: &'a [BenchmarkRecord],
    }

    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(
        writer,
        &Report {
            format_version: BENCHMARK_RECORD_FORMAT_VERSION,
            records,
        },
    )?;
    Ok(())
}

pub fn write_benchmark_csv(records: &[BenchmarkRecord], path: impl AsRef<Path>) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "format_version",
        "case_id",
        "run_id",
        "dataset_name",
        "dataset_version",
        "preset_name",
        "crate_version",
        "random_seed",
        "spatial_crop",
        "algorithm",
        "algorithm_configuration",
        "success",
        "error",
        "frame_count",
        "image_height",
        "image_width",
        "reconstruction_height",
        "reconstruction_width",
        "completed_iterations",
        "elapsed_seconds",
        "initial_objective",
        "final_objective",
        "final_to_initial_objective_ratio",
        "amplitude_rmse",
        "phase_rmse",
        "complex_field_relative_error",
        "fourier_domain_relative_error",
        "pupil_amplitude_rmse",
        "pupil_phase_rmse",
        "illumination_position_rmse",
        "per_frame_residual_mean",
        "per_frame_residual_max",
        "frames_json",
        "output_paths",
        "metadata_json",
    ])?;
    for record in records {
        writer.write_record([
            record.format_version.to_string(),
            record.case_id.clone(),
            record.run_id.clone(),
            record.dataset_name.clone(),
            record.dataset_version.clone().unwrap_or_default(),
            record.preset_name.clone().unwrap_or_default(),
            record.crate_version.clone(),
            record
                .random_seed
                .map_or_else(String::new, |seed| seed.to_string()),
            record.spatial_crop.map_or_else(String::new, |crop| {
                crop.iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(";")
            }),
            record.algorithm.clone(),
            record.algorithm_configuration.clone(),
            record.success.to_string(),
            record.error.clone().unwrap_or_default(),
            record.frame_count.to_string(),
            record.image_shape[0].to_string(),
            record.image_shape[1].to_string(),
            record.reconstruction_shape[0].to_string(),
            record.reconstruction_shape[1].to_string(),
            record.completed_iterations.to_string(),
            record.elapsed_seconds.to_string(),
            optional_number(record.initial_objective),
            optional_number(record.final_objective),
            optional_number(record.final_to_initial_objective_ratio),
            optional_number(record.amplitude_rmse),
            optional_number(record.phase_rmse),
            optional_number(record.complex_field_relative_error),
            optional_number(record.fourier_domain_relative_error),
            optional_number(record.pupil_amplitude_rmse),
            optional_number(record.pupil_phase_rmse),
            optional_number(record.illumination_position_rmse),
            optional_number(record.per_frame_residual_mean),
            optional_number(record.per_frame_residual_max),
            serde_json::to_string(&record.frames)?,
            record
                .output_paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(";"),
            serde_json::to_string(&record.metadata)?,
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn short_type_name<T>() -> &'static str {
    std::any::type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap_or("reconstruction algorithm")
}

fn optional_number(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

#[cfg(feature = "parquet")]
fn safe_stem(value: &str) -> String {
    let stem: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if stem.is_empty() {
        "benchmark".into()
    } else {
        stem
    }
}

fn case_hash(record: &BenchmarkRecord) -> u64 {
    // Stable FNV-1a rather than a process-seeded map hasher, so output names are
    // reproducible across runs and platforms.
    let mut hash = 0xcbf29ce484222325_u64;
    let mut update = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    update(record.dataset_name.as_bytes());
    update(
        record
            .dataset_version
            .as_deref()
            .unwrap_or_default()
            .as_bytes(),
    );
    update(record.preset_name.as_deref().unwrap_or_default().as_bytes());
    update(record.algorithm.as_bytes());
    update(record.algorithm_configuration.as_bytes());
    for frame in &record.frames {
        update(&frame.original_frame_index.to_le_bytes());
        update(
            &frame
                .original_illumination_index
                .unwrap_or(usize::MAX)
                .to_le_bytes(),
        );
    }
    if let Some(crop) = record.spatial_crop {
        for value in crop {
            update(&value.to_le_bytes());
        }
    }
    hash
}
