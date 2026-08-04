use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::BufReader,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
};

use ndarray::Array2;
use num_complex::Complex64;
use polars::prelude::{DataFrame, DataType, ParquetReader, SerReader};
use serde::de::DeserializeOwned;

use crate::{
    Error, Result, complex,
    diagnostics::ReconstructionDiagnostics,
    evaluation::ReconstructionEvaluation,
    model::Pupil,
    reconstruction::{
        AlgorithmMetricRecord, IterationRecord, ReconstructionResult, ReconstructionTrace,
    },
};

use super::{
    manifest::{
        ARRAY_BACKGROUND, ARRAY_FRAME_GAINS, ARRAY_ILLUMINATION_CALIBRATION, ARRAY_OBJECT,
        ARRAY_OBJECT_SPECTRUM, ARRAY_PUPIL, ARRAY_PUPIL_SUPPORT, BUNDLE_FORMAT_VERSION,
        BundleManifest, DOMAIN_DIAGNOSTICS, DOMAIN_EVALUATION, KNOWN_ROLES, ManifestArtifact,
        PREVIEW_FOURIER_COVERAGE, PREVIEW_OBJECT_AMPLITUDE, PREVIEW_OBJECT_PHASE,
        PREVIEW_PUPIL_AMPLITUDE, PREVIEW_PUPIL_PHASE, TABLE_ALGORITHM_METRICS,
        TABLE_FRAME_CALIBRATION, TABLE_FRAME_DIAGNOSTICS, TABLE_FRAME_EVALUATION, TABLE_HISTORY,
        TABLE_ILLUMINATION_CALIBRATION, TABLE_ITERATION_DIAGNOSTICS, TABLE_METADATA,
        TABLE_RAW_FRAME_STATISTICS, TABLE_SCALAR_DIAGNOSTICS, TABLE_SUMMARY,
    },
    npy,
    write::sha256,
};

/// Verified manifest metadata for one file inside a result bundle.
#[derive(Clone, Debug)]
pub struct BundleArtifact {
    /// Stable semantic artifact role.
    pub role: String,
    /// Absolute or bundle-root-relative resolved local path.
    pub path: PathBuf,
    /// Declared MIME media type.
    pub media_type: String,
    /// Exact artifact size in bytes.
    pub byte_size: u64,
    /// Lowercase hexadecimal SHA-256 digest.
    pub sha256: String,
    /// Optional NPY element-dtype descriptor.
    pub dtype: Option<String>,
    /// Optional array shape in axis order.
    pub shape: Option<Vec<u64>>,
}

/// Parquet table artifact descriptors in a result bundle.
#[derive(Clone, Debug)]
pub struct BundleTables {
    /// Required one-row run summary table.
    pub summary: BundleArtifact,
    /// Required iteration history table.
    pub history: BundleArtifact,
    /// Optional algorithm-specific scalar metric table.
    pub algorithm_metrics: Option<BundleArtifact>,
    /// Optional convergence diagnostics table.
    pub iteration_diagnostics: Option<BundleArtifact>,
    /// Optional per-frame diagnostic comparison table.
    pub frame_diagnostics: Option<BundleArtifact>,
    /// Optional raw measured-frame statistics table.
    pub raw_frame_statistics: Option<BundleArtifact>,
    /// Optional predicted-versus-measured frame evaluation table.
    pub frame_evaluation: Option<BundleArtifact>,
    /// Optional per-source Fourier-grid correction table.
    pub illumination_calibration: Option<BundleArtifact>,
    /// Optional per-frame gain and background table.
    pub frame_calibration: Option<BundleArtifact>,
    /// Optional named scalar diagnostics table.
    pub scalar_diagnostics: Option<BundleArtifact>,
    /// Optional extensible string metadata table.
    pub metadata: Option<BundleArtifact>,
}

/// Lossless NPY array artifact descriptors in a result bundle.
#[derive(Clone, Debug)]
pub struct BundleArrays {
    /// High-resolution `(height, width)` complex object field.
    pub object: BundleArtifact,
    /// Centered high-resolution complex object spectrum.
    pub object_spectrum: BundleArtifact,
    /// Low-resolution complex pupil values.
    pub pupil: BundleArtifact,
    /// Low-resolution binary pupil support.
    pub pupil_support: BundleArtifact,
    /// Optional source-order `(row, column)` corrections in Fourier-grid pixels.
    pub illumination_calibration: Option<BundleArtifact>,
    /// Optional acquisition-frame gains.
    pub frame_gains: Option<BundleArtifact>,
    /// Optional acquisition-frame additive backgrounds.
    pub background: Option<BundleArtifact>,
}

/// Optional derived PNG preview artifact descriptors.
#[derive(Clone, Debug, Default)]
pub struct BundlePreviews {
    /// Linearly normalized object-amplitude preview.
    pub object_amplitude: Option<BundleArtifact>,
    /// Wrapped object-phase preview.
    pub object_phase: Option<BundleArtifact>,
    /// Linearly normalized pupil-amplitude preview.
    pub pupil_amplitude: Option<BundleArtifact>,
    /// Wrapped pupil-phase preview.
    pub pupil_phase: Option<BundleArtifact>,
    /// Rendered individual-source Fourier-coverage preview.
    pub fourier_coverage: Option<BundleArtifact>,
}

/// Aggregate result of verifying every artifact in a bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleVerificationResult {
    /// Number of manifest-declared artifacts verified.
    pub artifact_count: usize,
    /// Sum of verified artifact sizes in bytes.
    pub total_bytes: u64,
}

#[derive(Default)]
struct BundleCache {
    object: Option<Arc<Array2<Complex64>>>,
    object_spectrum: Option<Arc<Array2<Complex64>>>,
    pupil: Option<Arc<Array2<Complex64>>>,
    pupil_support: Option<Arc<Array2<u8>>>,
    illumination_calibration: Option<Arc<Vec<(f64, f64)>>>,
    frame_gains: Option<Arc<Vec<f64>>>,
    background: Option<Arc<Vec<f64>>>,
    trace: Option<Arc<ReconstructionTrace>>,
    scalar_diagnostics: Option<Arc<BTreeMap<String, f64>>>,
    metadata: Option<Arc<BTreeMap<String, String>>>,
    result: Option<Arc<ReconstructionResult>>,
    diagnostics: Option<Arc<ReconstructionDiagnostics>>,
    evaluation: Option<Arc<ReconstructionEvaluation>>,
}

struct ResultBundleInner {
    manifest: BundleManifest,
    artifacts: BTreeMap<String, BundleArtifact>,
    cache: Mutex<BundleCache>,
}

/// One final, disk-backed reconstruction bundle.
///
/// The manifest and artifact handles are eager. Scientific arrays and domain
/// objects are hash-checked, loaded, and cached on first access.
#[derive(Clone)]
pub struct ResultBundle {
    /// Bundle root directory.
    pub path: PathBuf,
    /// Path to the bundle manifest JSON file.
    pub manifest_path: PathBuf,
    /// Unique run identifier shared by tables and manifest.
    pub run_id: String,
    /// Optional human-readable result label.
    pub label: Option<String>,
    /// Parquet table descriptors.
    pub tables: BundleTables,
    /// Lossless scientific-array descriptors.
    pub arrays: BundleArrays,
    /// Optional derived preview descriptors.
    pub previews: BundlePreviews,
    inner: Arc<ResultBundleInner>,
}

/// Opens and validates a complete bundle manifest without eagerly loading scientific arrays.
pub fn read_bundle(path: impl AsRef<Path>) -> Result<ResultBundle> {
    ResultBundle::read(path)
}

impl ResultBundle {
    /// Opens a complete bundle and validates manifest paths and required artifact metadata.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.ends_with(".inprogress"))
        {
            return Err(Error::IncompleteBundle(format!(
                "{} is an in-progress workspace",
                path.display()
            )));
        }
        let manifest_path = path.join("manifest.json");
        if !manifest_path.is_file() {
            return Err(Error::MissingArtifact {
                role: "manifest".into(),
            });
        }
        let manifest: BundleManifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))
                .map_err(|error| Error::InvalidManifest(error.to_string()))?;
        validate_manifest(path, &manifest)?;
        let artifacts = manifest
            .artifacts
            .iter()
            .map(|artifact| (artifact.role.clone(), public_artifact(path, artifact)))
            .collect::<BTreeMap<_, _>>();
        let required = |role: &str| {
            artifacts
                .get(role)
                .cloned()
                .ok_or_else(|| Error::MissingArtifact { role: role.into() })
        };
        let optional = |role: &str| artifacts.get(role).cloned();
        let tables = BundleTables {
            summary: required(TABLE_SUMMARY)?,
            history: required(TABLE_HISTORY)?,
            algorithm_metrics: optional(TABLE_ALGORITHM_METRICS),
            iteration_diagnostics: optional(TABLE_ITERATION_DIAGNOSTICS),
            frame_diagnostics: optional(TABLE_FRAME_DIAGNOSTICS),
            raw_frame_statistics: optional(TABLE_RAW_FRAME_STATISTICS),
            frame_evaluation: optional(TABLE_FRAME_EVALUATION),
            illumination_calibration: optional(TABLE_ILLUMINATION_CALIBRATION),
            frame_calibration: optional(TABLE_FRAME_CALIBRATION),
            scalar_diagnostics: optional(TABLE_SCALAR_DIAGNOSTICS),
            metadata: optional(TABLE_METADATA),
        };
        let arrays = BundleArrays {
            object: required(ARRAY_OBJECT)?,
            object_spectrum: required(ARRAY_OBJECT_SPECTRUM)?,
            pupil: required(ARRAY_PUPIL)?,
            pupil_support: required(ARRAY_PUPIL_SUPPORT)?,
            illumination_calibration: optional(ARRAY_ILLUMINATION_CALIBRATION),
            frame_gains: optional(ARRAY_FRAME_GAINS),
            background: optional(ARRAY_BACKGROUND),
        };
        let previews = BundlePreviews {
            object_amplitude: optional(PREVIEW_OBJECT_AMPLITUDE),
            object_phase: optional(PREVIEW_OBJECT_PHASE),
            pupil_amplitude: optional(PREVIEW_PUPIL_AMPLITUDE),
            pupil_phase: optional(PREVIEW_PUPIL_PHASE),
            fourier_coverage: optional(PREVIEW_FOURIER_COVERAGE),
        };
        let run_id = manifest.run_id.clone();
        let label = manifest.label.clone();
        Ok(Self {
            path: path.to_owned(),
            manifest_path,
            run_id,
            label,
            tables,
            arrays,
            previews,
            inner: Arc::new(ResultBundleInner {
                manifest,
                artifacts,
                cache: Mutex::new(BundleCache::default()),
            }),
        })
    }

    /// Hash-checks, lazily loads, and caches the high-resolution complex object.
    pub fn object(&self) -> Result<Arc<Array2<Complex64>>> {
        let mut cache = self.cache();
        if let Some(value) = &cache.object {
            return Ok(value.clone());
        }
        let shape = self.reconstruction_shape()?;
        let value = Arc::new(npy::read_complex2(
            &self.checked_artifact(ARRAY_OBJECT)?.path,
            ARRAY_OBJECT,
            shape,
        )?);
        cache.object = Some(value.clone());
        Ok(value)
    }

    /// Hash-checks, lazily loads, and caches the centered complex object spectrum.
    pub fn object_spectrum(&self) -> Result<Arc<Array2<Complex64>>> {
        let mut cache = self.cache();
        if let Some(value) = &cache.object_spectrum {
            return Ok(value.clone());
        }
        let shape = self.reconstruction_shape()?;
        let value = Arc::new(npy::read_complex2(
            &self.checked_artifact(ARRAY_OBJECT_SPECTRUM)?.path,
            ARRAY_OBJECT_SPECTRUM,
            shape,
        )?);
        cache.object_spectrum = Some(value.clone());
        Ok(value)
    }

    /// Hash-checks, lazily loads, and caches low-resolution complex pupil values.
    pub fn pupil(&self) -> Result<Arc<Array2<Complex64>>> {
        let mut cache = self.cache();
        if let Some(value) = &cache.pupil {
            return Ok(value.clone());
        }
        let shape = self.image_shape()?;
        let value = Arc::new(npy::read_complex2(
            &self.checked_artifact(ARRAY_PUPIL)?.path,
            ARRAY_PUPIL,
            shape,
        )?);
        cache.pupil = Some(value.clone());
        Ok(value)
    }

    /// Hash-checks, lazily loads, and caches the low-resolution binary pupil support.
    pub fn pupil_support(&self) -> Result<Arc<Array2<u8>>> {
        let mut cache = self.cache();
        if let Some(value) = &cache.pupil_support {
            return Ok(value.clone());
        }
        let shape = self.image_shape()?;
        let value = Arc::new(npy::read_u8_2(
            &self.checked_artifact(ARRAY_PUPIL_SUPPORT)?.path,
            ARRAY_PUPIL_SUPPORT,
            shape,
        )?);
        cache.pupil_support = Some(value.clone());
        Ok(value)
    }

    /// Lazily reconstructs and caches a validated domain result from lossless artifacts.
    pub fn result(&self) -> Result<Arc<ReconstructionResult>> {
        let mut cache = self.cache();
        if let Some(value) = &cache.result {
            return Ok(value.clone());
        }

        // Build every fallible value locally first. A failed load does not
        // install a partially initialized result.
        let reconstruction_shape = self.reconstruction_shape()?;
        let image_shape = self.image_shape()?;
        let object = cache.object.clone().map_or_else(
            || {
                npy::read_complex2(
                    &self.checked_artifact(ARRAY_OBJECT)?.path,
                    ARRAY_OBJECT,
                    reconstruction_shape,
                )
                .map(Arc::new)
            },
            Ok,
        )?;
        let object_spectrum = cache.object_spectrum.clone().map_or_else(
            || {
                npy::read_complex2(
                    &self.checked_artifact(ARRAY_OBJECT_SPECTRUM)?.path,
                    ARRAY_OBJECT_SPECTRUM,
                    reconstruction_shape,
                )
                .map(Arc::new)
            },
            Ok,
        )?;
        let pupil_values = cache.pupil.clone().map_or_else(
            || {
                npy::read_complex2(
                    &self.checked_artifact(ARRAY_PUPIL)?.path,
                    ARRAY_PUPIL,
                    image_shape,
                )
                .map(Arc::new)
            },
            Ok,
        )?;
        let pupil_support = cache.pupil_support.clone().map_or_else(
            || {
                npy::read_u8_2(
                    &self.checked_artifact(ARRAY_PUPIL_SUPPORT)?.path,
                    ARRAY_PUPIL_SUPPORT,
                    image_shape,
                )
                .map(Arc::new)
            },
            Ok,
        )?;
        let trace = cache
            .trace
            .clone()
            .map_or_else(|| self.load_trace().map(Arc::new), Ok)?;
        let scalar_diagnostics = cache.scalar_diagnostics.clone().map_or_else(
            || self.load_f64_map(TABLE_SCALAR_DIAGNOSTICS).map(Arc::new),
            Ok,
        )?;
        let metadata = cache
            .metadata
            .clone()
            .map_or_else(|| self.load_result_metadata().map(Arc::new), Ok)?;
        let illumination_calibration = cache
            .illumination_calibration
            .clone()
            .map_or_else(|| self.load_illumination_calibration().map(Arc::new), Ok)?;
        let frame_gains = cache.frame_gains.clone().map_or_else(
            || self.load_optional_f64(ARRAY_FRAME_GAINS).map(Arc::new),
            Ok,
        )?;
        let background = cache.background.clone().map_or_else(
            || self.load_optional_f64(ARRAY_BACKGROUND).map(Arc::new),
            Ok,
        )?;

        let recovered_pupil = Pupil::new((*pupil_values).clone(), (*pupil_support).clone())?;
        let amplitude = complex::amplitude(object.view());
        let phase = complex::phase(object.view());
        let result = Arc::new(ReconstructionResult {
            object: (*object).clone(),
            amplitude,
            phase,
            object_spectrum: (*object_spectrum).clone(),
            recovered_pupil,
            calibrated_illumination: (!illumination_calibration.is_empty())
                .then(|| (*illumination_calibration).clone()),
            recovered_frame_gains: (!frame_gains.is_empty()).then(|| (*frame_gains).clone()),
            recovered_background: (!background.is_empty()).then(|| (*background).clone()),
            trace: (*trace).clone(),
            scalar_diagnostics: (*scalar_diagnostics).clone(),
            runtime: self.inner.manifest.runtime.clone(),
            metadata: (*metadata).clone(),
        });
        result.validate()?;
        self.validate_summary(&result)?;

        cache.object = Some(object);
        cache.object_spectrum = Some(object_spectrum);
        cache.pupil = Some(pupil_values);
        cache.pupil_support = Some(pupil_support);
        cache.trace = Some(trace);
        cache.scalar_diagnostics = Some(scalar_diagnostics);
        cache.metadata = Some(metadata);
        cache.illumination_calibration = Some(illumination_calibration);
        cache.frame_gains = Some(frame_gains);
        cache.background = Some(background);
        cache.result = Some(result.clone());
        Ok(result)
    }

    /// Lazily loads and caches optional structured diagnostics JSON.
    pub fn diagnostics(&self) -> Result<Option<Arc<ReconstructionDiagnostics>>> {
        if !self.inner.artifacts.contains_key(DOMAIN_DIAGNOSTICS) {
            return Ok(None);
        }
        let mut cache = self.cache();
        if let Some(value) = &cache.diagnostics {
            return Ok(Some(value.clone()));
        }
        let value: Arc<ReconstructionDiagnostics> =
            Arc::new(self.load_json_artifact(DOMAIN_DIAGNOSTICS)?);
        cache.diagnostics = Some(value.clone());
        Ok(Some(value))
    }

    /// Lazily loads and caches optional structured ground-truth evaluation JSON.
    pub fn evaluation(&self) -> Result<Option<Arc<ReconstructionEvaluation>>> {
        if !self.inner.artifacts.contains_key(DOMAIN_EVALUATION) {
            return Ok(None);
        }
        let mut cache = self.cache();
        if let Some(value) = &cache.evaluation {
            return Ok(Some(value.clone()));
        }
        let value: Arc<ReconstructionEvaluation> =
            Arc::new(self.load_json_artifact(DOMAIN_EVALUATION)?);
        cache.evaluation = Some(value.clone());
        Ok(Some(value))
    }

    /// Drops this bundle's shared in-memory array and domain-object cache.
    pub fn clear_cache(&self) {
        *self.cache() = BundleCache::default();
    }

    /// Verifies existence, byte size, SHA-256, NPY metadata, Parquet schema, and run IDs.
    pub fn verify(&self) -> Result<BundleVerificationResult> {
        let mut total_bytes = 0_u64;
        for artifact in self.inner.artifacts.values() {
            self.verify_artifact(artifact)?;
            total_bytes = total_bytes.checked_add(artifact.byte_size).ok_or_else(|| {
                Error::InvalidManifest("artifact byte-size total overflows u64".into())
            })?;
            if artifact.media_type == "application/vnd.apache.parquet" {
                let dataframe = read_parquet(&artifact.path)?;
                validate_table_schema(&artifact.role, &dataframe, &self.run_id)?;
            }
        }
        // This also validates all authoritative array dtypes, shapes, and
        // cross-artifact reconstruction invariants.
        let result = self.result()?;
        self.validate_summary(&result)?;
        Ok(BundleVerificationResult {
            artifact_count: self.inner.artifacts.len(),
            total_bytes,
        })
    }

    fn validate_summary(&self, result: &ReconstructionResult) -> Result<()> {
        let artifact = self.checked_artifact(TABLE_SUMMARY)?;
        let summary = read_parquet(&artifact.path)?;
        validate_table_schema(TABLE_SUMMARY, &summary, &self.run_id)?;
        let string =
            |name: &str| -> Result<Option<&str>> { Ok(summary.column(name)?.str()?.get(0)) };
        let unsigned =
            |name: &str| -> Result<Option<u64>> { Ok(summary.column(name)?.u64()?.get(0)) };
        let float = |name: &str| -> Result<Option<f64>> { Ok(summary.column(name)?.f64()?.get(0)) };
        let boolean =
            |name: &str| -> Result<Option<bool>> { Ok(summary.column(name)?.bool()?.get(0)) };
        let image_shape = result.recovered_pupil.shape();
        let reconstruction_shape = result.object.dim();
        let expected_dataset = self
            .inner
            .manifest
            .dataset
            .as_ref()
            .map(|value| value.name.as_str());
        let expected_dataset_version = self
            .inner
            .manifest
            .dataset
            .as_ref()
            .and_then(|value| value.version.as_deref());
        let consistent = string("crate_version")?
            == Some(self.inner.manifest.crate_version.as_str())
            && string("dataset_name")? == expected_dataset
            && string("dataset_version")? == expected_dataset_version
            && string("algorithm")? == Some(result.runtime.algorithm.as_str())
            && unsigned("image_width")? == Some(image_shape.1 as u64)
            && unsigned("image_height")? == Some(image_shape.0 as u64)
            && unsigned("reconstruction_width")? == Some(reconstruction_shape.1 as u64)
            && unsigned("reconstruction_height")? == Some(reconstruction_shape.0 as u64)
            && unsigned("completed_iterations")?
                == Some(result.runtime.completed_iterations as u64)
            && float("elapsed_seconds")? == Some(result.runtime.elapsed_seconds)
            && boolean("stopped_early")? == Some(result.runtime.stopped_early)
            && float("final_objective")? == result.trace.final_objective();
        if !consistent {
            return Err(Error::InvalidParquetSchema {
                role: TABLE_SUMMARY.into(),
                reason: "summary values disagree with the manifest or reconstructed result".into(),
            });
        }
        Ok(())
    }

    fn reconstruction_shape(&self) -> Result<(usize, usize)> {
        shape2(
            self.inner.manifest.result.reconstruction_shape,
            ARRAY_OBJECT,
        )
    }

    fn image_shape(&self) -> Result<(usize, usize)> {
        shape2(self.inner.manifest.result.image_shape, ARRAY_PUPIL)
    }

    fn checked_artifact(&self, role: &str) -> Result<BundleArtifact> {
        let artifact = self
            .inner
            .artifacts
            .get(role)
            .ok_or_else(|| Error::MissingArtifact { role: role.into() })?;
        self.verify_artifact(artifact)?;
        Ok(artifact.clone())
    }

    fn verify_artifact(&self, artifact: &BundleArtifact) -> Result<()> {
        let metadata = fs::metadata(&artifact.path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::MissingArtifact {
                    role: artifact.role.clone(),
                }
            } else {
                Error::Io(error)
            }
        })?;
        if metadata.len() != artifact.byte_size {
            return Err(Error::InvalidManifest(format!(
                "artifact {} has size {}, expected {}",
                artifact.role,
                metadata.len(),
                artifact.byte_size
            )));
        }
        if sha256(&artifact.path)? != artifact.sha256 {
            return Err(Error::ArtifactHashMismatch {
                role: artifact.role.clone(),
            });
        }
        Ok(())
    }

    fn load_trace(&self) -> Result<ReconstructionTrace> {
        let history_artifact = self.checked_artifact(TABLE_HISTORY)?;
        let history = read_parquet(&history_artifact.path)?;
        validate_table_schema(TABLE_HISTORY, &history, &self.run_id)?;
        let iterations = history.column("iteration")?.u64()?;
        let objectives = history.column("objective")?.f64()?;
        let elapsed = history.column("elapsed_seconds")?.f64()?;
        let iterations = iterations
            .into_no_null_iter()
            .zip(objectives.into_no_null_iter())
            .zip(elapsed.into_no_null_iter())
            .map(|((iteration, objective), elapsed_seconds)| {
                Ok(IterationRecord {
                    iteration: usize::try_from(iteration).map_err(|_| {
                        Error::InvalidParquetSchema {
                            role: TABLE_HISTORY.into(),
                            reason: "iteration is not addressable as usize".into(),
                        }
                    })?,
                    objective,
                    elapsed_seconds,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let algorithm_metrics =
            if let Some(artifact) = self.inner.artifacts.get(TABLE_ALGORITHM_METRICS) {
                self.verify_artifact(artifact)?;
                let dataframe = read_parquet(&artifact.path)?;
                validate_table_schema(TABLE_ALGORITHM_METRICS, &dataframe, &self.run_id)?;
                let iteration = dataframe.column("iteration")?.u64()?;
                let namespace = dataframe.column("namespace")?.str()?;
                let metric = dataframe.column("metric")?.str()?;
                let value = dataframe.column("value")?.f64()?;
                iteration
                    .into_no_null_iter()
                    .zip(namespace.iter().flatten())
                    .zip(metric.iter().flatten())
                    .zip(value.into_no_null_iter())
                    .map(|(((iteration, namespace), metric), value)| {
                        Ok(AlgorithmMetricRecord {
                            iteration: usize::try_from(iteration).map_err(|_| {
                                Error::InvalidParquetSchema {
                                    role: TABLE_ALGORITHM_METRICS.into(),
                                    reason: "iteration is not addressable as usize".into(),
                                }
                            })?,
                            namespace: namespace.into(),
                            metric: metric.into(),
                            value,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
        Ok(ReconstructionTrace {
            iterations,
            algorithm_metrics,
        })
    }

    fn load_f64_map(&self, role: &str) -> Result<BTreeMap<String, f64>> {
        let Some(artifact) = self.inner.artifacts.get(role) else {
            return Ok(BTreeMap::new());
        };
        self.verify_artifact(artifact)?;
        let dataframe = read_parquet(&artifact.path)?;
        validate_table_schema(role, &dataframe, &self.run_id)?;
        let keys = dataframe.column("key")?.str()?;
        let values = dataframe.column("value")?.f64()?;
        Ok(keys
            .iter()
            .flatten()
            .zip(values.into_no_null_iter())
            .map(|(key, value)| (key.into(), value))
            .collect())
    }

    fn load_string_map(&self, role: &str) -> Result<BTreeMap<String, String>> {
        let Some(artifact) = self.inner.artifacts.get(role) else {
            return Ok(BTreeMap::new());
        };
        self.verify_artifact(artifact)?;
        let dataframe = read_parquet(&artifact.path)?;
        validate_table_schema(role, &dataframe, &self.run_id)?;
        let keys = dataframe.column("key")?.str()?;
        let values = dataframe.column("value")?.str()?;
        Ok(keys
            .iter()
            .flatten()
            .zip(values.iter().flatten())
            .map(|(key, value)| (key.into(), value.into()))
            .collect())
    }

    fn load_result_metadata(&self) -> Result<BTreeMap<String, String>> {
        let mut metadata = self.load_string_map(TABLE_METADATA)?;
        let artifact = self.checked_artifact(TABLE_SUMMARY)?;
        let summary = read_parquet(&artifact.path)?;
        validate_table_schema(TABLE_SUMMARY, &summary, &self.run_id)?;
        for key in [
            "case_id",
            "dataset_name",
            "dataset_version",
            "preset_name",
            "algorithm_configuration",
        ] {
            if let Some(value) = summary.column(key)?.str()?.get(0) {
                metadata.insert(key.into(), value.into());
            }
        }
        for key in ["random_seed", "frame_count"] {
            if let Some(value) = summary.column(key)?.u64()?.get(0) {
                metadata.insert(key.into(), value.to_string());
            }
        }
        Ok(metadata)
    }

    fn load_illumination_calibration(&self) -> Result<Vec<(f64, f64)>> {
        let Some(artifact) = self.inner.artifacts.get(ARRAY_ILLUMINATION_CALIBRATION) else {
            return Ok(Vec::new());
        };
        self.verify_artifact(artifact)?;
        let shape = artifact_shape(artifact)?;
        if shape.len() != 2 || shape[1] != 2 {
            return Err(Error::InvalidArrayShape {
                role: ARRAY_ILLUMINATION_CALIBRATION.into(),
                reason: format!("expected (source_count, 2), got {shape:?}"),
            });
        }
        let values = npy::read_f64(&artifact.path, ARRAY_ILLUMINATION_CALIBRATION, &shape)?;
        Ok(values
            .chunks_exact(2)
            .map(|values| (values[0], values[1]))
            .collect())
    }

    fn load_optional_f64(&self, role: &str) -> Result<Vec<f64>> {
        let Some(artifact) = self.inner.artifacts.get(role) else {
            return Ok(Vec::new());
        };
        self.verify_artifact(artifact)?;
        let shape = artifact_shape(artifact)?;
        if shape.len() != 1 {
            return Err(Error::InvalidArrayShape {
                role: role.into(),
                reason: format!("expected one dimension, got {shape:?}"),
            });
        }
        npy::read_f64(&artifact.path, role, &shape)
    }

    fn load_json_artifact<T: DeserializeOwned>(&self, role: &str) -> Result<T> {
        let artifact = self.checked_artifact(role)?;
        serde_json::from_reader(BufReader::new(File::open(artifact.path)?))
            .map_err(Error::Serialization)
    }

    fn cache(&self) -> MutexGuard<'_, BundleCache> {
        self.inner
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn validate_manifest(root: &Path, manifest: &BundleManifest) -> Result<()> {
    if manifest.bundle_format_version != BUNDLE_FORMAT_VERSION {
        return Err(Error::UnsupportedBundleVersion {
            actual: manifest.bundle_format_version,
            supported: BUNDLE_FORMAT_VERSION,
        });
    }
    if manifest.run_id.is_empty() || manifest.crate_version.is_empty() {
        return Err(Error::InvalidManifest(
            "run_id and crate_version must be non-empty".into(),
        ));
    }
    let canonical_root = root.canonicalize()?;
    let mut roles = BTreeSet::new();
    for artifact in &manifest.artifacts {
        if !KNOWN_ROLES.contains(&artifact.role.as_str()) {
            return Err(Error::UnsupportedArtifactRole(artifact.role.clone()));
        }
        if !roles.insert(artifact.role.as_str()) {
            return Err(Error::InvalidManifest(format!(
                "duplicate artifact role {}",
                artifact.role
            )));
        }
        validate_relative_path(&artifact.relative_path)?;
        let path = root.join(&artifact.relative_path);
        let canonical_path = path.canonicalize().map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::MissingArtifact {
                    role: artifact.role.clone(),
                }
            } else {
                Error::Io(error)
            }
        })?;
        if !canonical_path.starts_with(&canonical_root) {
            return Err(Error::InvalidRelativePath(
                artifact.relative_path.display().to_string(),
            ));
        }
        if artifact.sha256.len() != 64
            || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::InvalidManifest(format!(
                "artifact {} has an invalid SHA-256 encoding",
                artifact.role
            )));
        }
        validate_array_descriptor(artifact, &manifest.result)?;
    }
    for role in [
        TABLE_SUMMARY,
        TABLE_HISTORY,
        ARRAY_OBJECT,
        ARRAY_OBJECT_SPECTRUM,
        ARRAY_PUPIL,
        ARRAY_PUPIL_SUPPORT,
    ] {
        if !roles.contains(role) {
            return Err(Error::MissingArtifact { role: role.into() });
        }
    }
    Ok(())
}

fn validate_array_descriptor(
    artifact: &ManifestArtifact,
    result: &super::manifest::ResultDescriptor,
) -> Result<()> {
    let (expected_dtype, expected_shape): (&str, Option<&[u64]>) = match artifact.role.as_str() {
        ARRAY_OBJECT | ARRAY_OBJECT_SPECTRUM => ("<c16", Some(&result.reconstruction_shape)),
        ARRAY_PUPIL => ("<c16", Some(&result.image_shape)),
        ARRAY_PUPIL_SUPPORT => ("|u1", Some(&result.image_shape)),
        ARRAY_ILLUMINATION_CALIBRATION | ARRAY_FRAME_GAINS | ARRAY_BACKGROUND => ("<f8", None),
        _ => return Ok(()),
    };
    if artifact.dtype.as_deref() != Some(expected_dtype) {
        return Err(Error::InvalidArrayDtype {
            role: artifact.role.clone(),
            actual: artifact.dtype.as_deref().unwrap_or("<missing>").into(),
            expected: expected_dtype.into(),
        });
    }
    let shape = artifact
        .shape
        .as_deref()
        .ok_or_else(|| Error::InvalidArrayShape {
            role: artifact.role.clone(),
            reason: "manifest shape is missing".into(),
        })?;
    if let Some(expected_shape) = expected_shape {
        if shape != expected_shape {
            return Err(Error::InvalidArrayShape {
                role: artifact.role.clone(),
                reason: format!("manifest shape {shape:?}, expected {expected_shape:?}"),
            });
        }
    } else {
        let valid = match artifact.role.as_str() {
            ARRAY_ILLUMINATION_CALIBRATION => shape.len() == 2 && shape[1] == 2,
            ARRAY_FRAME_GAINS | ARRAY_BACKGROUND => shape.len() == 1,
            _ => true,
        };
        if !valid {
            return Err(Error::InvalidArrayShape {
                role: artifact.role.clone(),
                reason: format!("invalid manifest shape {shape:?}"),
            });
        }
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<()> {
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

fn public_artifact(root: &Path, artifact: &ManifestArtifact) -> BundleArtifact {
    BundleArtifact {
        role: artifact.role.clone(),
        path: root.join(&artifact.relative_path),
        media_type: artifact.media_type.clone(),
        byte_size: artifact.byte_size,
        sha256: artifact.sha256.clone(),
        dtype: artifact.dtype.clone(),
        shape: artifact.shape.clone(),
    }
}

fn read_parquet(path: &Path) -> Result<DataFrame> {
    Ok(ParquetReader::new(File::open(path)?).finish()?)
}

fn validate_table_schema(role: &str, dataframe: &DataFrame, run_id: &str) -> Result<()> {
    let expected: &[(&str, DataType, bool)] = match role {
        TABLE_SUMMARY => &[
            ("run_id", DataType::String, false),
            ("case_id", DataType::String, true),
            ("crate_version", DataType::String, false),
            ("dataset_name", DataType::String, true),
            ("dataset_version", DataType::String, true),
            ("preset_name", DataType::String, true),
            ("algorithm", DataType::String, false),
            ("algorithm_configuration", DataType::String, true),
            ("random_seed", DataType::UInt64, true),
            ("frame_count", DataType::UInt64, true),
            ("image_width", DataType::UInt64, false),
            ("image_height", DataType::UInt64, false),
            ("reconstruction_width", DataType::UInt64, false),
            ("reconstruction_height", DataType::UInt64, false),
            ("completed_iterations", DataType::UInt64, true),
            ("elapsed_seconds", DataType::Float64, false),
            ("stopped_early", DataType::Boolean, false),
            ("final_objective", DataType::Float64, true),
            ("success", DataType::Boolean, false),
            ("error", DataType::String, true),
        ],
        TABLE_HISTORY => &[
            ("run_id", DataType::String, false),
            ("iteration", DataType::UInt64, false),
            ("objective", DataType::Float64, false),
            ("elapsed_seconds", DataType::Float64, false),
        ],
        TABLE_ALGORITHM_METRICS => &[
            ("run_id", DataType::String, false),
            ("iteration", DataType::UInt64, false),
            ("namespace", DataType::String, false),
            ("metric", DataType::String, false),
            ("value", DataType::Float64, false),
        ],
        TABLE_ITERATION_DIAGNOSTICS => &[
            ("run_id", DataType::String, false),
            ("iteration", DataType::UInt64, false),
            ("total_objective", DataType::Float64, true),
            ("data_objective", DataType::Float64, true),
            ("regularization_objective", DataType::Float64, true),
            ("object_relative_change", DataType::Float64, true),
            ("pupil_relative_change", DataType::Float64, true),
            ("median_frame_objective", DataType::Float64, true),
            ("worst_frame_objective", DataType::Float64, true),
            ("elapsed_seconds", DataType::Float64, true),
        ],
        TABLE_FRAME_DIAGNOSTICS => &[
            ("run_id", DataType::String, false),
            ("iteration", DataType::UInt64, true),
            ("frame_index", DataType::UInt64, false),
            ("illumination_index", DataType::UInt64, false),
            ("reference_sum", DataType::Float64, false),
            ("estimate_sum", DataType::Float64, false),
            ("residual_l1", DataType::Float64, false),
            ("residual_l2", DataType::Float64, false),
            ("residual_mean", DataType::Float64, false),
            ("residual_std", DataType::Float64, false),
            ("residual_max_abs", DataType::Float64, false),
            ("normalized_l2", DataType::Float64, false),
            ("saturated_pixels", DataType::UInt64, true),
        ],
        TABLE_RAW_FRAME_STATISTICS => &[
            ("run_id", DataType::String, false),
            ("frame_index", DataType::UInt64, false),
            ("mean", DataType::Float64, false),
            ("std", DataType::Float64, false),
            ("min", DataType::Float64, false),
            ("max", DataType::Float64, false),
            ("sum", DataType::Float64, false),
            ("saturated_pixels", DataType::UInt64, false),
            ("zero_pixels", DataType::UInt64, false),
        ],
        TABLE_FRAME_EVALUATION => &[
            ("run_id", DataType::String, false),
            ("frame_index", DataType::UInt64, false),
            ("reference_sum", DataType::Float64, false),
            ("estimate_sum", DataType::Float64, false),
            ("residual_l1", DataType::Float64, false),
            ("residual_l2", DataType::Float64, false),
            ("residual_mean", DataType::Float64, false),
            ("residual_std", DataType::Float64, false),
            ("residual_max_abs", DataType::Float64, false),
            ("normalized_l2", DataType::Float64, false),
            ("saturated_pixels", DataType::UInt64, true),
        ],
        TABLE_ILLUMINATION_CALIBRATION => &[
            ("run_id", DataType::String, false),
            ("source_index", DataType::UInt64, false),
            ("row_correction_pixels", DataType::Float64, false),
            ("column_correction_pixels", DataType::Float64, false),
        ],
        TABLE_FRAME_CALIBRATION => &[
            ("run_id", DataType::String, false),
            ("frame_index", DataType::UInt64, false),
            ("gain", DataType::Float64, true),
            ("background", DataType::Float64, true),
        ],
        TABLE_SCALAR_DIAGNOSTICS => &[
            ("run_id", DataType::String, false),
            ("key", DataType::String, false),
            ("value", DataType::Float64, false),
        ],
        TABLE_METADATA => &[
            ("run_id", DataType::String, false),
            ("key", DataType::String, false),
            ("value", DataType::String, false),
        ],
        _ => &[],
    };
    if !expected.is_empty() {
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
    }
    if let Ok(column) = dataframe.column("run_id") {
        let strings = column.str().map_err(|error| Error::InvalidParquetSchema {
            role: role.into(),
            reason: error.to_string(),
        })?;
        for actual in strings.iter().flatten() {
            if actual != run_id {
                return Err(Error::InconsistentRunId {
                    expected: run_id.into(),
                    actual: actual.into(),
                });
            }
        }
    }
    if role == TABLE_HISTORY {
        let iterations = dataframe.column("iteration")?.u64()?;
        for (index, iteration) in iterations.into_no_null_iter().enumerate() {
            if iteration != index as u64 + 1 {
                return Err(Error::InvalidParquetSchema {
                    role: role.into(),
                    reason: "iterations must be unique, ordered, and one-based".into(),
                });
            }
        }
    }
    if role == TABLE_SUMMARY {
        if dataframe.height() != 1 {
            return Err(Error::InvalidParquetSchema {
                role: role.into(),
                reason: "summary must contain exactly one row".into(),
            });
        }
        if dataframe.column("success")?.bool()?.get(0) != Some(true)
            || dataframe.column("error")?.null_count() != 1
        {
            return Err(Error::InvalidParquetSchema {
                role: role.into(),
                reason: "a result-bundle summary must describe one successful run".into(),
            });
        }
    }
    if role == TABLE_ALGORITHM_METRICS {
        let iterations = dataframe.column("iteration")?.u64()?;
        let namespaces = dataframe.column("namespace")?.str()?;
        let metrics = dataframe.column("metric")?.str()?;
        let mut previous_iteration = 0;
        let mut keys = BTreeSet::new();
        for ((iteration, namespace), metric) in iterations
            .into_no_null_iter()
            .zip(namespaces.iter().flatten())
            .zip(metrics.iter().flatten())
        {
            if iteration == 0
                || iteration < previous_iteration
                || namespace.is_empty()
                || metric.is_empty()
                || !keys.insert((iteration, namespace, metric))
            {
                return Err(Error::InvalidParquetSchema {
                    role: role.into(),
                    reason: "metric keys must be non-empty and unique, with ordered one-based iterations"
                        .into(),
                });
            }
            previous_iteration = iteration;
        }
    }
    if role == TABLE_ITERATION_DIAGNOSTICS {
        let mut previous = 0;
        for iteration in dataframe.column("iteration")?.u64()?.into_no_null_iter() {
            if iteration == 0 || iteration <= previous {
                return Err(Error::InvalidParquetSchema {
                    role: role.into(),
                    reason: "diagnostic iterations must be unique, ordered, and one-based".into(),
                });
            }
            previous = iteration;
        }
    }
    if role == TABLE_FRAME_DIAGNOSTICS {
        let iterations = dataframe.column("iteration")?.u64()?;
        let frames = dataframe.column("frame_index")?.u64()?;
        let mut keys = BTreeSet::new();
        for (iteration, frame) in iterations.iter().zip(frames.into_no_null_iter()) {
            if iteration == Some(0) || !keys.insert((iteration, frame)) {
                return Err(Error::InvalidParquetSchema {
                    role: role.into(),
                    reason: "frame diagnostic keys must be unique and iterations one-based".into(),
                });
            }
        }
    }
    if matches!(
        role,
        TABLE_RAW_FRAME_STATISTICS | TABLE_FRAME_EVALUATION | TABLE_FRAME_CALIBRATION
    ) {
        validate_zero_based_index(role, dataframe, "frame_index")?;
    }
    if role == TABLE_ILLUMINATION_CALIBRATION {
        validate_zero_based_index(role, dataframe, "source_index")?;
    }
    if matches!(role, TABLE_SCALAR_DIAGNOSTICS | TABLE_METADATA) {
        let keys = dataframe.column("key")?.str()?;
        let mut unique = BTreeSet::new();
        if keys
            .iter()
            .flatten()
            .any(|key| key.is_empty() || !unique.insert(key))
        {
            return Err(Error::InvalidParquetSchema {
                role: role.into(),
                reason: "keys must be non-empty and unique within one run".into(),
            });
        }
    }
    Ok(())
}

fn validate_zero_based_index(role: &str, dataframe: &DataFrame, column: &str) -> Result<()> {
    for (expected, actual) in dataframe
        .column(column)?
        .u64()?
        .into_no_null_iter()
        .enumerate()
    {
        if actual != expected as u64 {
            return Err(Error::InvalidParquetSchema {
                role: role.into(),
                reason: format!("{column} must be unique, ordered, and zero-based"),
            });
        }
    }
    Ok(())
}

fn shape2(shape: [u64; 2], role: &str) -> Result<(usize, usize)> {
    Ok((
        usize::try_from(shape[0]).map_err(|_| Error::InvalidArrayShape {
            role: role.into(),
            reason: "height is not addressable".into(),
        })?,
        usize::try_from(shape[1]).map_err(|_| Error::InvalidArrayShape {
            role: role.into(),
            reason: "width is not addressable".into(),
        })?,
    ))
}

fn artifact_shape(artifact: &BundleArtifact) -> Result<Vec<usize>> {
    artifact
        .shape
        .as_ref()
        .ok_or_else(|| Error::InvalidArrayShape {
            role: artifact.role.clone(),
            reason: "manifest shape is missing".into(),
        })?
        .iter()
        .map(|&value| {
            usize::try_from(value).map_err(|_| Error::InvalidArrayShape {
                role: artifact.role.clone(),
                reason: "dimension is not addressable".into(),
            })
        })
        .collect()
}
