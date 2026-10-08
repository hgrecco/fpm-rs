//! Bright-field circle initialization for physical planar LED arrays.
//!
//! This module estimates bright-field illumination wave vectors directly from
//! measured intensity spectra, then fits those observations to the canonical
//! [`PlanarLedArray`](crate::experiment::PlanarLedArray) geometry. It is an
//! optional warm start for reconstruction and
//! [`IlluminationCalibration`](crate::illumination_calibration::IlluminationCalibration),
//! not a reconstruction algorithm or an independent per-source correction path.

use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    Error, Result,
    array_layout::checked_len_2d,
    backend::{Backend, CpuBackend, FftDirection},
    experiment::{Illumination, KVector, Optics, SourceGeometry},
    illumination_calibration::{
        CalibrationParameterSpec, PlanarArrayCalibrationParameters, PlanarArrayParameterValues,
    },
    measurements::MeasurementRead,
    model::{ImagePlaneModel, ReconstructionShape, fftshift_copy},
};

/// Current JSON serialization format for planar-array initialization results.
pub const INITIALIZATION_FORMAT_VERSION: u32 = 1;
/// Current verified directory-bundle format for initialization results.
pub const INITIALIZATION_BUNDLE_FORMAT_VERSION: u32 = 1;
const TAU: f64 = std::f64::consts::TAU;
const RESULT_ROLE: &str = "domain.initialization";
const OBSERVATIONS_ROLE: &str = "tables.observations";
const FIT_HISTORY_ROLE: &str = "tables.fit_history";

/// Numerical and validation controls for bright-field circle initialization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrightfieldCircleOptions {
    /// Optional acquisition-frame subset; `None` selects safe bright-field frames automatically.
    pub frame_indices: Option<Vec<usize>>,
    /// Maximum center displacement from the nominal source, in dimensionless NA.
    pub center_search_radius_na: f64,
    /// Additional positive distance retained inside the objective-NA boundary.
    pub brightfield_margin_na: f64,
    /// Half-width of the fitted pupil-radius search, in dimensionless NA.
    pub pupil_radius_search_na: f64,
    /// Gaussian smoothing standard deviation in Fourier-grid pixels.
    pub gaussian_sigma_pixels: f64,
    /// Number of uniformly spaced angles used for every circular score.
    pub angular_samples: usize,
    /// Radial finite-difference displacement in Fourier-grid pixels.
    pub radial_derivative_step_pixels: f64,
    /// Minimum fraction of angular samples required for a valid score.
    pub minimum_arc_fraction: f64,
    /// Minimum normalized first-derivative edge contrast for acceptance.
    pub minimum_edge_contrast: f64,
    /// Positive relative floor used when dividing by the mean magnitude spectrum.
    pub mean_spectrum_floor: f64,
    /// Huber transition scale for physical-fit residual components, in NA.
    pub robust_residual_scale_na: f64,
    /// Maximum bounded physical-fit steps.
    pub maximum_fit_steps: usize,
    /// Relative physical-fit objective improvement required to continue.
    pub fit_relative_tolerance: f64,
    /// Initial normalized physical-fit line-search step.
    pub fit_initial_step_size: f64,
    /// Smallest normalized line-search step attempted.
    pub fit_minimum_step_size: f64,
    /// Multiplicative line-search reduction in `(0, 1)`.
    pub fit_step_reduction: f64,
    /// Relative pivot threshold used for the observation-Jacobian rank test.
    pub rank_tolerance: f64,
    /// Maximum accepted difference between fitted and configured pupil NA.
    pub pupil_radius_tolerance_na: f64,
}

impl Default for BrightfieldCircleOptions {
    fn default() -> Self {
        Self {
            frame_indices: None,
            center_search_radius_na: 0.02,
            brightfield_margin_na: 0.002,
            pupil_radius_search_na: 0.01,
            gaussian_sigma_pixels: 2.0,
            angular_samples: 180,
            radial_derivative_step_pixels: 1.0,
            minimum_arc_fraction: 0.2,
            minimum_edge_contrast: 0.01,
            mean_spectrum_floor: 1e-8,
            robust_residual_scale_na: 0.002,
            maximum_fit_steps: 100,
            fit_relative_tolerance: 1e-8,
            fit_initial_step_size: 0.5,
            fit_minimum_step_size: 1e-6,
            fit_step_reduction: 0.5,
            rank_tolerance: 1e-8,
            pupil_radius_tolerance_na: 0.02,
        }
    }
}

impl BrightfieldCircleOptions {
    /// Validates finite ranges, search controls, and optimizer settings.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("center_search_radius_na", self.center_search_radius_na),
            ("brightfield_margin_na", self.brightfield_margin_na),
            ("pupil_radius_search_na", self.pupil_radius_search_na),
            ("gaussian_sigma_pixels", self.gaussian_sigma_pixels),
            (
                "radial_derivative_step_pixels",
                self.radial_derivative_step_pixels,
            ),
            ("mean_spectrum_floor", self.mean_spectrum_floor),
            ("robust_residual_scale_na", self.robust_residual_scale_na),
            ("fit_initial_step_size", self.fit_initial_step_size),
            ("fit_minimum_step_size", self.fit_minimum_step_size),
            ("rank_tolerance", self.rank_tolerance),
            ("pupil_radius_tolerance_na", self.pupil_radius_tolerance_na),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(name, "must be finite and positive"));
            }
        }
        if !self.minimum_arc_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_arc_fraction)
            || self.minimum_arc_fraction == 0.0
        {
            return Err(invalid(
                "minimum_arc_fraction",
                "must be finite and in (0, 1]",
            ));
        }
        if !self.minimum_edge_contrast.is_finite() || self.minimum_edge_contrast < 0.0 {
            return Err(invalid(
                "minimum_edge_contrast",
                "must be finite and non-negative",
            ));
        }
        if self.angular_samples < 16 {
            return Err(invalid("angular_samples", "must be at least 16"));
        }
        if self.maximum_fit_steps == 0 {
            return Err(invalid("maximum_fit_steps", "must be greater than zero"));
        }
        if !self.fit_relative_tolerance.is_finite() || self.fit_relative_tolerance < 0.0 {
            return Err(invalid(
                "fit_relative_tolerance",
                "must be finite and non-negative",
            ));
        }
        if self.fit_minimum_step_size > self.fit_initial_step_size {
            return Err(invalid(
                "fit_minimum_step_size",
                "must not exceed fit_initial_step_size",
            ));
        }
        if !self.fit_step_reduction.is_finite() || !(0.0..1.0).contains(&self.fit_step_reduction) {
            return Err(invalid(
                "fit_step_reduction",
                "must be finite and strictly between zero and one",
            ));
        }
        if let Some(frames) = &self.frame_indices {
            if frames.is_empty() {
                return Err(invalid(
                    "frame_indices",
                    "an explicit frame subset must not be empty",
                ));
            }
            let mut sorted = frames.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != frames.len() {
                return Err(invalid(
                    "frame_indices",
                    "explicit frame indices must be unique",
                ));
            }
        }
        Ok(())
    }
}

/// One considered acquisition frame and its circle-localization diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrightfieldCircleObservation {
    /// Zero-based acquisition-frame index.
    pub frame_index: usize,
    /// Stable row-major physical source index.
    pub source_index: usize,
    /// Nominal transverse vector `(kx, ky)` in radians per metre.
    pub nominal_k_rad_per_m: [f64; 2],
    /// Detected transverse vector `(kx, ky)` in radians per metre, when accepted.
    pub detected_k_rad_per_m: Option<[f64; 2]>,
    /// Detected `(NA_x, NA_y)` components, when accepted.
    pub detected_na: Option<[f64; 2]>,
    /// Detected centered Fourier coordinates `(row, column)`, when accepted.
    pub fourier_grid_position: Option<[f64; 2]>,
    /// Best pupil radius for this frame, in dimensionless NA.
    pub fitted_pupil_radius_na: f64,
    /// Normalized first-radial-derivative score.
    pub first_derivative_score: f64,
    /// Normalized second-radial-derivative score.
    pub second_derivative_score: f64,
    /// Combined deterministic detector score.
    pub combined_score: f64,
    /// Score at the conjugate branch corresponding to `-k`.
    pub conjugate_score: f64,
    /// Fraction of requested angular samples contributing to the score.
    pub usable_arc_fraction: f64,
    /// Fixed confidence weight supplied to the physical fit.
    pub confidence: f64,
    /// Fraction of background-corrected spatial samples below zero.
    pub negative_sample_fraction: f64,
    /// Human-readable rejection reason, or `None` for an accepted observation.
    pub rejection_reason: Option<String>,
}

impl BrightfieldCircleObservation {
    /// Returns whether this observation contributes to physical fitting.
    pub fn accepted(&self) -> bool {
        self.detected_k_rad_per_m.is_some() && self.rejection_reason.is_none()
    }
}

/// One accepted or rejected bounded physical-fit step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarArrayInitializationFitRecord {
    /// One-based optimizer step.
    pub step: usize,
    /// Whether a bounded line-search candidate reduced the objective.
    pub accepted: bool,
    /// Accepted normalized step size, or zero after exhaustion.
    pub step_size: f64,
    /// Robust observation data loss.
    pub data_loss: f64,
    /// Quadratic-prior loss.
    pub regularization_loss: f64,
    /// `data_loss + regularization_loss`.
    pub total_loss: f64,
    /// Current normalized physical values in `parameter_names` order.
    pub normalized_values: Vec<f64>,
}

/// Structural and numerical summary of circle detection and physical fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarArrayInitializationDiagnostics {
    /// Number of acquisition frames considered by the detector.
    pub candidate_frames: usize,
    /// Number of accepted circle-center observations.
    pub accepted_observations: usize,
    /// Number of rejected circle-center observations.
    pub rejected_observations: usize,
    /// Configured objective pupil radius in NA.
    pub configured_pupil_radius_na: f64,
    /// Confidence-weighted detected pupil radius in NA.
    pub fitted_pupil_radius_na: f64,
    /// Number of independent data-Jacobian columns before applying priors.
    pub jacobian_rank: usize,
    /// Number of active physical parameters.
    pub active_parameter_count: usize,
    /// Squared largest-to-smallest accepted rank pivot ratio.
    pub jacobian_condition_estimate: Option<f64>,
    /// Confidence-weighted initial transverse-vector residual RMS, in NA.
    pub initial_residual_rms_na: f64,
    /// Confidence-weighted final transverse-vector residual RMS, in NA.
    pub final_residual_rms_na: f64,
    /// Additional inspectable warnings that do not invalidate the result.
    pub warnings: Vec<String>,
}

/// Progress stage reported by a bright-field initializer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanarArrayInitializationStage {
    /// Accumulating the mean measured magnitude spectrum.
    MeanSpectrum,
    /// Detecting a circle center in one frame.
    CircleDetection,
    /// Updating bounded physical parameters.
    PhysicalFit,
    /// Initialization completed successfully.
    Complete,
}

/// Immutable progress record for initializer-specific callbacks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarArrayInitializationProgress {
    /// Current stage.
    pub stage: PlanarArrayInitializationStage,
    /// Number of completed units in the current stage.
    pub completed: usize,
    /// Total units in the current stage.
    pub total: usize,
    /// Current acquisition frame, when the stage is frame-indexed.
    pub frame_index: Option<usize>,
}

/// Action returned by a planar-array initialization callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarArrayInitializationAction {
    /// Continue initialization.
    Continue,
    /// Cancel before the next stage boundary without returning a partial result.
    Cancel,
}

/// Callback interface dedicated to pre-reconstruction planar-array initialization.
pub trait PlanarArrayInitializationCallback: Send {
    /// Receives progress at deterministic frame, fit-step, and completion boundaries.
    fn on_progress(
        &mut self,
        progress: &PlanarArrayInitializationProgress,
    ) -> Result<PlanarArrayInitializationAction>;
}

/// Runtime metadata for one completed initialization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarArrayInitializationRuntime {
    /// Wall-clock seconds spent detecting and fitting.
    pub elapsed_seconds: f64,
    /// Number of complete measurement passes.
    pub measurement_passes: usize,
    /// Number of physical objective evaluations.
    pub physical_objective_evaluations: usize,
}

/// Complete serializable output of bright-field planar-array initialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarArrayInitializationResult {
    /// Serialization format version.
    pub format_version: u32,
    /// Nominal illumination supplied to the initializer.
    pub nominal_illumination: Illumination,
    /// Reusable illumination containing the fitted physical geometry.
    pub initialized_illumination: Illumination,
    /// Reusable model atomically refreshed from [`Self::initialized_illumination`].
    pub initialized_model: ImagePlaneModel,
    /// Physical parameter selection and bounds used by the fit.
    pub parameters: PlanarArrayCalibrationParameters,
    /// Detection and optimizer controls used by this run.
    pub options: BrightfieldCircleOptions,
    /// Absolute nominal physical and multiplicative values.
    pub initial_parameters: PlanarArrayParameterValues,
    /// Absolute initialized physical and unchanged multiplicative values.
    pub initialized_parameters: PlanarArrayParameterValues,
    /// Stable active physical parameter names.
    pub parameter_names: Vec<String>,
    /// Detector result for every considered acquisition frame.
    pub observations: Vec<BrightfieldCircleObservation>,
    /// Accepted and rejected physical-fit steps.
    pub fit_history: Vec<PlanarArrayInitializationFitRecord>,
    /// Detection, rank, and residual diagnostics.
    pub diagnostics: PlanarArrayInitializationDiagnostics,
    /// Timing and evaluation counts.
    pub runtime: PlanarArrayInitializationRuntime,
}

impl PlanarArrayInitializationResult {
    /// Validates version, physical state, model consistency, and finite diagnostics.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != INITIALIZATION_FORMAT_VERSION {
            return Err(Error::InvalidModel(format!(
                "planar-array initialization format version {} is unsupported",
                self.format_version
            )));
        }
        self.initialized_model.validate()?;
        if PlanarArrayParameterValues::from_illumination(&self.nominal_illumination)?
            != self.initial_parameters
            || PlanarArrayParameterValues::from_illumination(&self.initialized_illumination)?
                != self.initialized_parameters
        {
            return Err(Error::InvalidModel(
                "initialization illumination and absolute parameter values disagree".into(),
            ));
        }
        if self.parameter_names.is_empty()
            || self
                .observations
                .iter()
                .any(|value| !observation_is_finite(value))
            || !diagnostics_are_finite(&self.diagnostics)
            || !self.runtime.elapsed_seconds.is_finite()
            || self.runtime.elapsed_seconds < 0.0
        {
            return Err(Error::InvalidModel(
                "planar-array initialization result contains invalid diagnostics".into(),
            ));
        }
        BrightfieldCircleInitializer {
            parameters: self.parameters.clone(),
            options: self.options.clone(),
        }
        .validate_for(&self.nominal_illumination)?;
        let active = active_parameters(&self.parameters);
        let expected_names: Vec<_> = active.iter().map(ActiveParameter::name).collect();
        if self.parameter_names != expected_names
            || self.diagnostics.active_parameter_count != active.len()
            || self.diagnostics.jacobian_rank != active.len()
            || self.runtime.measurement_passes != 2
            || self.runtime.physical_objective_evaluations == 0
        {
            return Err(Error::InvalidModel(
                "initialization parameter names or rank diagnostics disagree".into(),
            ));
        }
        if self.diagnostics.candidate_frames != self.observations.len()
            || self.diagnostics.accepted_observations
                != self
                    .observations
                    .iter()
                    .filter(|value| value.accepted())
                    .count()
            || self.diagnostics.rejected_observations
                != self
                    .observations
                    .iter()
                    .filter(|value| !value.accepted())
                    .count()
        {
            return Err(Error::InvalidModel(
                "planar-array initialization observation counts disagree".into(),
            ));
        }
        if self.observations.iter().any(|observation| {
            observation.frame_index >= self.initialized_model.frame_count()
                || observation.source_index >= self.initialized_model.source_count()
                || (observation.rejection_reason.is_none()
                    != observation.detected_k_rad_per_m.is_some())
                || (observation.detected_k_rad_per_m.is_some() != observation.detected_na.is_some())
                || (observation.detected_na.is_some()
                    != observation.fourier_grid_position.is_some())
        }) {
            return Err(Error::InvalidModel(
                "initialization observation indices or acceptance fields disagree".into(),
            ));
        }
        if self.fit_history.iter().enumerate().any(|(index, record)| {
            let loss_scale = record
                .total_loss
                .abs()
                .max(record.data_loss.abs())
                .max(record.regularization_loss.abs())
                .max(1.0);
            record.step != index + 1
                || record.normalized_values.len() != active.len()
                || !record.step_size.is_finite()
                || record.step_size < 0.0
                || (record.accepted && record.step_size == 0.0)
                || (!record.accepted && record.step_size != 0.0)
                || !record.data_loss.is_finite()
                || record.data_loss < 0.0
                || !record.regularization_loss.is_finite()
                || record.regularization_loss < 0.0
                || !record.total_loss.is_finite()
                || (record.total_loss - record.data_loss - record.regularization_loss).abs()
                    > 16.0 * f64::EPSILON * loss_scale
                || record
                    .normalized_values
                    .iter()
                    .any(|value| !value.is_finite())
        }) {
            return Err(Error::InvalidModel(
                "initialization fit history is inconsistent or non-finite".into(),
            ));
        }
        if self.initial_parameters.position_offsets_m
            != self.initialized_parameters.position_offsets_m
            || self.initial_parameters.relative_source_power
                != self.initialized_parameters.relative_source_power
            || self.initial_parameters.frame_gains != self.initialized_parameters.frame_gains
        {
            return Err(Error::InvalidModel(
                "circle initialization changed unsupported per-source or multiplicative values"
                    .into(),
            ));
        }
        if self.initialized_model.source_count()
            != self.initialized_parameters.relative_source_power.len()
            || self.initialized_model.frame_count() != self.initialized_parameters.frame_gains.len()
        {
            return Err(Error::InvalidModel(
                "initialized model and illumination topology disagree".into(),
            ));
        }
        Ok(())
    }

    /// Writes the complete initialization result as JSON.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        serde_json::to_writer(BufWriter::new(File::create(path)?), self)?;
        Ok(())
    }

    /// Loads and validates a complete initialization result from JSON.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let result: Self = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        result.validate()?;
        Ok(result)
    }

    /// Atomically writes a verified directory bundle with JSON state and CSV tables.
    ///
    /// The complete JSON result is authoritative. The observation and fit-history
    /// tables are normalized, analysis-friendly projections of the same state.
    /// Existing destinations and incomplete sibling workspaces are never overwritten.
    pub fn write_bundle(&self, path: impl AsRef<Path>) -> Result<InitializationBundle> {
        write_initialization_bundle(path.as_ref(), self)
    }
}

/// Manifest-verified file inside a planar-array initialization bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitializationBundleArtifact {
    /// Stable semantic role.
    pub role: String,
    /// Resolved local artifact path.
    pub path: PathBuf,
    /// Declared MIME media type.
    pub media_type: String,
    /// Exact byte size.
    pub byte_size: u64,
    /// Lowercase hexadecimal SHA-256 digest.
    pub sha256: String,
}

/// Aggregate result of re-verifying an initialization bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitializationBundleVerificationResult {
    /// Number of manifest-declared artifacts verified.
    pub artifact_count: usize,
    /// Sum of verified artifact sizes in bytes.
    pub total_bytes: u64,
}

/// Eagerly verified initialization bundle and its authoritative physical result.
#[derive(Clone, Debug)]
pub struct InitializationBundle {
    /// Bundle root directory.
    pub path: PathBuf,
    /// Bundle manifest path.
    pub manifest_path: PathBuf,
    /// Complete initialization JSON artifact.
    pub result_artifact: InitializationBundleArtifact,
    /// Per-frame circle-observation CSV artifact.
    pub observations_artifact: InitializationBundleArtifact,
    /// Bounded physical-fit history CSV artifact.
    pub fit_history_artifact: InitializationBundleArtifact,
    /// Validated authoritative initialization result.
    pub result: PlanarArrayInitializationResult,
}

impl InitializationBundle {
    /// Opens a complete bundle, verifies every declared artifact, and loads its result.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        read_initialization_bundle(path)
    }

    /// Recomputes the size and SHA-256 digest of every declared artifact.
    pub fn verify(&self) -> Result<InitializationBundleVerificationResult> {
        let manifest = read_initialization_manifest(&self.path)?;
        verify_initialization_artifacts(&self.path, &manifest)
    }
}

/// Opens and verifies a planar-array initialization bundle.
pub fn read_initialization_bundle(path: impl AsRef<Path>) -> Result<InitializationBundle> {
    let path = path.as_ref();
    if path
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.ends_with(".inprogress"))
    {
        return Err(Error::IncompleteBundle(format!(
            "{} is an in-progress initialization workspace",
            path.display()
        )));
    }
    let manifest = read_initialization_manifest(path)?;
    verify_initialization_artifacts(path, &manifest)?;
    let artifact = |role: &str| -> Result<InitializationBundleArtifact> {
        let value = manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.role == role)
            .ok_or_else(|| Error::MissingArtifact { role: role.into() })?;
        Ok(InitializationBundleArtifact {
            role: value.role.clone(),
            path: path.join(&value.relative_path),
            media_type: value.media_type.clone(),
            byte_size: value.byte_size,
            sha256: value.sha256.clone(),
        })
    };
    let result_artifact = artifact(RESULT_ROLE)?;
    let result = PlanarArrayInitializationResult::load_json(&result_artifact.path)?;
    Ok(InitializationBundle {
        path: path.to_owned(),
        manifest_path: path.join("manifest.json"),
        result_artifact,
        observations_artifact: artifact(OBSERVATIONS_ROLE)?,
        fit_history_artifact: artifact(FIT_HISTORY_ROLE)?,
        result,
    })
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InitializationBundleManifest {
    initialization_bundle_format_version: u32,
    crate_version: String,
    artifacts: Vec<InitializationManifestArtifact>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InitializationManifestArtifact {
    role: String,
    relative_path: PathBuf,
    media_type: String,
    byte_size: u64,
    sha256: String,
}

fn write_initialization_bundle(
    requested: &Path,
    result: &PlanarArrayInitializationResult,
) -> Result<InitializationBundle> {
    result.validate()?;
    if requested.exists() {
        return Err(invalid(
            "initialization bundle path",
            format!("{} already exists", requested.display()),
        ));
    }
    let name = requested
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("initialization bundle path", "must name a UTF-8 directory"))?;
    let workspace = requested.with_file_name(format!("{name}.inprogress"));
    if workspace.exists() {
        return Err(Error::IncompleteBundle(format!(
            "{} already exists",
            workspace.display()
        )));
    }
    if let Some(parent) = requested.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&workspace)?;
    fs::create_dir(workspace.join("domain"))?;
    fs::create_dir(workspace.join("tables"))?;

    let result_path = workspace.join("domain/initialization.json");
    result.save_json(&result_path)?;
    write_observations_csv(
        &workspace.join("tables/observations.csv"),
        &result.observations,
    )?;
    write_fit_history_csv(
        &workspace.join("tables/fit_history.csv"),
        &result.fit_history,
    )?;
    let artifacts = vec![
        describe_initialization_artifact(
            &workspace,
            RESULT_ROLE,
            "domain/initialization.json",
            "application/json",
        )?,
        describe_initialization_artifact(
            &workspace,
            OBSERVATIONS_ROLE,
            "tables/observations.csv",
            "text/csv",
        )?,
        describe_initialization_artifact(
            &workspace,
            FIT_HISTORY_ROLE,
            "tables/fit_history.csv",
            "text/csv",
        )?,
    ];
    write_pretty_json(
        &workspace.join("manifest.json"),
        &InitializationBundleManifest {
            initialization_bundle_format_version: INITIALIZATION_BUNDLE_FORMAT_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").into(),
            artifacts,
        },
    )?;
    fs::rename(&workspace, requested)?;
    read_initialization_bundle(requested)
}

fn read_initialization_manifest(path: &Path) -> Result<InitializationBundleManifest> {
    let manifest_path = path.join("manifest.json");
    if !manifest_path.is_file() {
        return Err(Error::MissingArtifact {
            role: "manifest".into(),
        });
    }
    let manifest: InitializationBundleManifest =
        serde_json::from_reader(BufReader::new(File::open(manifest_path)?))
            .map_err(|error| Error::InvalidManifest(error.to_string()))?;
    if manifest.initialization_bundle_format_version != INITIALIZATION_BUNDLE_FORMAT_VERSION {
        return Err(Error::UnsupportedBundleVersion {
            actual: manifest.initialization_bundle_format_version,
            supported: INITIALIZATION_BUNDLE_FORMAT_VERSION,
        });
    }
    if manifest.crate_version.is_empty() || manifest.artifacts.len() != 3 {
        return Err(Error::InvalidManifest(
            "initialization bundle must declare a crate version and exactly three artifacts".into(),
        ));
    }
    Ok(manifest)
}

fn verify_initialization_artifacts(
    root: &Path,
    manifest: &InitializationBundleManifest,
) -> Result<InitializationBundleVerificationResult> {
    let expected = [
        (
            RESULT_ROLE,
            "domain/initialization.json",
            "application/json",
        ),
        (OBSERVATIONS_ROLE, "tables/observations.csv", "text/csv"),
        (FIT_HISTORY_ROLE, "tables/fit_history.csv", "text/csv"),
    ];
    let mut total_bytes = 0;
    for (role, relative, media_type) in expected {
        let artifact = manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.role == role)
            .ok_or_else(|| Error::MissingArtifact { role: role.into() })?;
        if artifact.relative_path != Path::new(relative)
            || artifact.media_type != media_type
            || artifact.sha256.len() != 64
            || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::InvalidManifest(format!(
                "initialization artifact {role} has invalid path, media type, or digest metadata"
            )));
        }
        let artifact_path = root.join(&artifact.relative_path);
        let metadata = fs::metadata(&artifact_path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::MissingArtifact { role: role.into() }
            } else {
                Error::Io(error)
            }
        })?;
        if metadata.len() != artifact.byte_size {
            return Err(Error::InvalidManifest(format!(
                "initialization artifact {role} has the wrong byte size"
            )));
        }
        if file_sha256(&artifact_path)? != artifact.sha256 {
            return Err(Error::ArtifactHashMismatch { role: role.into() });
        }
        total_bytes += artifact.byte_size;
    }
    Ok(InitializationBundleVerificationResult {
        artifact_count: expected.len(),
        total_bytes,
    })
}

fn describe_initialization_artifact(
    root: &Path,
    role: &str,
    relative_path: &str,
    media_type: &str,
) -> Result<InitializationManifestArtifact> {
    let path = root.join(relative_path);
    Ok(InitializationManifestArtifact {
        role: role.into(),
        relative_path: relative_path.into(),
        media_type: media_type.into(),
        byte_size: fs::metadata(&path)?.len(),
        sha256: file_sha256(&path)?,
    })
}

fn write_observations_csv(
    path: &Path,
    observations: &[BrightfieldCircleObservation],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "frame_index",
        "source_index",
        "nominal_kx_rad_per_m",
        "nominal_ky_rad_per_m",
        "detected_kx_rad_per_m",
        "detected_ky_rad_per_m",
        "detected_na_x",
        "detected_na_y",
        "fourier_row",
        "fourier_column",
        "fitted_pupil_radius_na",
        "first_derivative_score",
        "second_derivative_score",
        "combined_score",
        "conjugate_score",
        "usable_arc_fraction",
        "confidence",
        "negative_sample_fraction",
        "accepted",
        "rejection_reason",
    ])?;
    for value in observations {
        let optional = |pair: Option<[f64; 2]>, axis: usize| {
            pair.map(|pair| pair[axis].to_string()).unwrap_or_default()
        };
        writer.write_record([
            value.frame_index.to_string(),
            value.source_index.to_string(),
            value.nominal_k_rad_per_m[0].to_string(),
            value.nominal_k_rad_per_m[1].to_string(),
            optional(value.detected_k_rad_per_m, 0),
            optional(value.detected_k_rad_per_m, 1),
            optional(value.detected_na, 0),
            optional(value.detected_na, 1),
            optional(value.fourier_grid_position, 0),
            optional(value.fourier_grid_position, 1),
            value.fitted_pupil_radius_na.to_string(),
            value.first_derivative_score.to_string(),
            value.second_derivative_score.to_string(),
            value.combined_score.to_string(),
            value.conjugate_score.to_string(),
            value.usable_arc_fraction.to_string(),
            value.confidence.to_string(),
            value.negative_sample_fraction.to_string(),
            value.accepted().to_string(),
            value.rejection_reason.clone().unwrap_or_default(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_fit_history_csv(
    path: &Path,
    history: &[PlanarArrayInitializationFitRecord],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "step",
        "accepted",
        "step_size",
        "data_loss",
        "regularization_loss",
        "total_loss",
        "normalized_values_json",
    ])?;
    for value in history {
        writer.write_record([
            value.step.to_string(),
            value.accepted.to_string(),
            value.step_size.to_string(),
            value.data_loss.to_string(),
            value.regularization_loss.to_string(),
            value.total_loss.to_string(),
            serde_json::to_string(&value.normalized_values)?,
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_pretty_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String> {
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

/// Bright-field detector and bounded physical planar-array fitter.
///
/// The detector uses circular pupil edges in centered intensity spectra only
/// to obtain a physical warm start. It then fits those observed centers to the
/// canonical [`PlanarLedArray`](crate::experiment::PlanarLedArray) mapping.
/// It does not reconstruct an object and does not produce independent source
/// shifts. Measurement-loss refinement remains the responsibility of
/// [`IlluminationCalibration`](crate::illumination_calibration::IlluminationCalibration).
///
/// # Assumptions
///
/// Selected frames must be single-source, strictly bright-field exposures of
/// a thin coherent specimen with enough reference interference and texture to
/// expose the circular pupil edge. The compiled pupil support must be circular
/// and shift invariant. The initializer uses two deterministic streaming
/// measurement passes and preserves pupil values, backgrounds, source powers,
/// frame gains, sampling, and acquisition topology in its returned model.
///
/// # Example
///
/// ```no_run
/// use fpm_rs::{
///     Result,
///     experiment::{Illumination, Optics},
///     illumination_calibration::{
///         CalibrationParameterSpec, PlanarArrayCalibrationParameters,
///     },
///     illumination_initialization::BrightfieldCircleInitializer,
///     measurements::MeasurementStack,
///     model::ImagePlaneModel,
/// };
///
/// # fn initialize(
/// #     measurements: &MeasurementStack,
/// #     optics: &Optics,
/// #     nominal: &Illumination,
/// #     model: &ImagePlaneModel,
/// # ) -> Result<()> {
/// let lateral = CalibrationParameterSpec::new(-1e-3, 1e-3, 0.2e-3)
///     .finite_difference_step(1e-6);
/// let parameters = PlanarArrayCalibrationParameters::builder()
///     .translation_specs([Some(lateral.clone()), Some(lateral), None])
///     .build()?;
/// let initialized = BrightfieldCircleInitializer::new(parameters)
///     .initialize(measurements, optics, nominal, model)?;
/// let warm_model = initialized.initialized_model;
/// # let _ = warm_model;
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// - J. Sun, Q. Chen, Y. Zhang, and C. Zuo,
///   [“Efficient positional misalignment correction method for Fourier
///   ptychographic microscopy,”](https://doi.org/10.1364/BOE.7.001336)
///   *Biomedical Optics Express* **7**(4), 1336–1350 (2016). Sun et al. search
///   independent apertures during reconstruction and subsequently regress a
///   planar misalignment model; this initializer instead fits detected circle
///   centers directly to the crate's bounded physical geometry.
/// - R. Eckert, Z. F. Phillips, and L. Waller,
///   [“Efficient illumination angle self-calibration in Fourier
///   ptychography,”](https://doi.org/10.1364/AO.57.005434) *Applied Optics*
///   **57**(19), 5434–5442 (2018). This implementation adopts only the
///   bright-field circular-edge initialization concept, not their iterative
///   spectral-correlation stage or three-dimensional variants.
#[derive(Clone, Debug)]
pub struct BrightfieldCircleInitializer {
    /// Global physical parameters selected for fitting.
    pub parameters: PlanarArrayCalibrationParameters,
    /// Detector, rank, and fit controls.
    pub options: BrightfieldCircleOptions,
}

impl BrightfieldCircleInitializer {
    /// Creates an initializer for explicitly selected global physical parameters.
    pub fn new(parameters: PlanarArrayCalibrationParameters) -> Self {
        Self {
            parameters,
            options: BrightfieldCircleOptions::default(),
        }
    }

    /// Replaces circle-detection and physical-fit options.
    pub fn options(mut self, options: BrightfieldCircleOptions) -> Self {
        self.options = options;
        self
    }

    /// Validates this initializer for the nominal planar-array illumination.
    pub fn validate_for(&self, illumination: &Illumination) -> Result<()> {
        self.options.validate()?;
        self.parameters.validate_for(illumination)?;
        if !self.parameters.has_active_parameters() {
            return Err(invalid(
                "parameters",
                "at least one global physical parameter must be active",
            ));
        }
        if !self.parameters.position_offsets.is_empty() {
            return Err(Error::Unsupported(
                "bright-field circle initialization does not fit per-source position offsets"
                    .into(),
            ));
        }
        if self.parameters.relative_source_power.is_some() || self.parameters.frame_gains.is_some()
        {
            return Err(Error::Unsupported(
                "bright-field circle initialization does not fit source powers or frame gains"
                    .into(),
            ));
        }
        match illumination.geometry() {
            SourceGeometry::PlanarArray(_) => Ok(()),
            _ => Err(Error::Unsupported(
                "bright-field circle initialization supports only PlanarLedArray geometry".into(),
            )),
        }
    }

    /// Detects bright-field circles and fits a physical warm start using the CPU backend.
    pub fn initialize<M: MeasurementRead>(
        &self,
        measurements: &M,
        optics: &Optics,
        nominal_illumination: &Illumination,
        model: &ImagePlaneModel,
    ) -> Result<PlanarArrayInitializationResult> {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(
            model.image_shape(),
            model.reconstruction_shape(),
        )?);
        self.initialize_with_backend(measurements, optics, nominal_illumination, model, backend)
    }

    /// Detects and fits using an explicit FFT backend.
    pub fn initialize_with_backend<M: MeasurementRead>(
        &self,
        measurements: &M,
        optics: &Optics,
        nominal_illumination: &Illumination,
        model: &ImagePlaneModel,
        backend: Arc<dyn Backend>,
    ) -> Result<PlanarArrayInitializationResult> {
        self.initialize_internal(
            measurements,
            optics,
            nominal_illumination,
            model,
            backend,
            None,
        )
    }

    /// Detects and fits with an explicit backend and initializer-specific callback.
    pub fn initialize_with_callback<M: MeasurementRead>(
        &self,
        measurements: &M,
        optics: &Optics,
        nominal_illumination: &Illumination,
        model: &ImagePlaneModel,
        backend: Arc<dyn Backend>,
        callback: &mut dyn PlanarArrayInitializationCallback,
    ) -> Result<PlanarArrayInitializationResult> {
        self.initialize_internal(
            measurements,
            optics,
            nominal_illumination,
            model,
            backend,
            Some(callback),
        )
    }

    fn initialize_internal<M: MeasurementRead>(
        &self,
        measurements: &M,
        optics: &Optics,
        nominal_illumination: &Illumination,
        model: &ImagePlaneModel,
        backend: Arc<dyn Backend>,
        mut callback: Option<&mut dyn PlanarArrayInitializationCallback>,
    ) -> Result<PlanarArrayInitializationResult> {
        let started = Instant::now();
        self.validate_for(nominal_illumination)?;
        validate_inputs(measurements, optics, nominal_illumination, model)?;
        let active = active_parameters(&self.parameters);
        let initial_parameters =
            PlanarArrayParameterValues::from_illumination(nominal_illumination)?;
        validate_initial_values(&active, &initial_parameters)?;
        let candidates = select_candidates(
            measurements,
            optics,
            nominal_illumination,
            model,
            &self.options,
        )?;
        if candidates.len().saturating_mul(2) < active.len() {
            return Err(Error::InvalidMeasurements(format!(
                "{} candidate circle centers provide fewer than {} scalar constraints",
                candidates.len(),
                active.len()
            )));
        }
        let image_len = checked_len_2d(model.image_shape())?;
        let mut mean_spectrum = vec![0.0; image_len];
        let mut negative_fractions = vec![0.0; candidates.len()];
        let mut spectrum = vec![Complex64::default(); image_len];
        let mut shifted = vec![Complex64::default(); image_len];
        let mut column = vec![Complex64::default(); model.image_shape().0];
        for (position, candidate) in candidates.iter().enumerate() {
            let negative_fraction = frame_spectrum(
                measurements,
                model,
                candidate.frame,
                candidate.scalar,
                backend.as_ref(),
                &mut spectrum,
                &mut shifted,
                &mut column,
            )?;
            negative_fractions[position] = negative_fraction;
            for (mean, value) in mean_spectrum.iter_mut().zip(&shifted) {
                *mean += value.norm();
            }
            emit_progress(
                &mut callback,
                PlanarArrayInitializationStage::MeanSpectrum,
                position + 1,
                candidates.len(),
                Some(candidate.frame),
            )?;
        }
        for value in &mut mean_spectrum {
            *value /= candidates.len() as f64;
        }
        let mean_max = mean_spectrum.iter().copied().fold(0.0, f64::max);
        if !mean_max.is_finite() || mean_max <= 0.0 {
            return Err(Error::InvalidMeasurements(
                "candidate frames have no finite positive Fourier magnitude".into(),
            ));
        }
        let mean_floor = self.options.mean_spectrum_floor * mean_max;
        let mut observations = Vec::with_capacity(candidates.len());
        let mut normalized = vec![0.0; image_len];
        for (position, candidate) in candidates.iter().enumerate() {
            frame_spectrum(
                measurements,
                model,
                candidate.frame,
                candidate.scalar,
                backend.as_ref(),
                &mut spectrum,
                &mut shifted,
                &mut column,
            )?;
            for pixel in 0..image_len {
                normalized[pixel] = shifted[pixel].norm() / mean_spectrum[pixel].max(mean_floor);
            }
            gaussian_blur_in_place(
                &mut normalized,
                model.image_shape(),
                self.options.gaussian_sigma_pixels,
            );
            observations.push(detect_circle(
                &normalized,
                model.image_shape(),
                model,
                optics,
                candidate,
                negative_fractions[position],
                &self.options,
            ));
            emit_progress(
                &mut callback,
                PlanarArrayInitializationStage::CircleDetection,
                position + 1,
                candidates.len(),
                Some(candidate.frame),
            )?;
        }
        let accepted: Vec<_> = observations
            .iter()
            .filter(|value| value.accepted())
            .collect();
        if accepted.len().saturating_mul(2) < active.len() {
            return Err(Error::InvalidMeasurements(format!(
                "{} accepted circle centers provide fewer than {} scalar constraints",
                accepted.len(),
                active.len()
            )));
        }
        let fitted_pupil_radius_na = weighted_radius(&accepted)?;
        if (fitted_pupil_radius_na - optics.objective_na).abs()
            > self.options.pupil_radius_tolerance_na
        {
            return Err(Error::InvalidMeasurements(format!(
                "fitted pupil radius {fitted_pupil_radius_na:.6e} NA differs from configured objective NA {:.6e} by more than {:.6e}",
                optics.objective_na, self.options.pupil_radius_tolerance_na
            )));
        }
        let (rank, condition) = jacobian_rank(
            optics,
            nominal_illumination,
            &initial_parameters,
            &active,
            &accepted,
            self.options.rank_tolerance,
        )?;
        if rank != active.len() {
            return Err(Error::InvalidParameter {
                name: "parameters",
                reason: format!(
                    "accepted circle observations give data-Jacobian rank {rank} for {} active physical parameters",
                    active.len()
                ),
            });
        }
        let initial_rms =
            residual_rms(optics, nominal_illumination, &initial_parameters, &accepted)?;
        let mut evaluations = 0;
        let (initialized_parameters, fit_history) = fit_parameters(
            optics,
            nominal_illumination,
            &initial_parameters,
            &active,
            &accepted,
            &self.options,
            &mut evaluations,
            &mut callback,
        )?;
        let initialized_illumination =
            initialized_parameters.to_illumination(nominal_illumination)?;
        let mut initialized_model = model.clone();
        initialized_model.update_illumination_geometry(optics, &initialized_illumination)?;
        validate_preserved_model_state(model, &initialized_model)?;
        let final_rms = residual_rms(
            optics,
            nominal_illumination,
            &initialized_parameters,
            &accepted,
        )?;
        let diagnostics = PlanarArrayInitializationDiagnostics {
            candidate_frames: observations.len(),
            accepted_observations: accepted.len(),
            rejected_observations: observations.len() - accepted.len(),
            configured_pupil_radius_na: optics.objective_na,
            fitted_pupil_radius_na,
            jacobian_rank: rank,
            active_parameter_count: active.len(),
            jacobian_condition_estimate: condition,
            initial_residual_rms_na: initial_rms,
            final_residual_rms_na: final_rms,
            warnings: Vec::new(),
        };
        emit_progress(
            &mut callback,
            PlanarArrayInitializationStage::Complete,
            1,
            1,
            None,
        )?;
        let result = PlanarArrayInitializationResult {
            format_version: INITIALIZATION_FORMAT_VERSION,
            nominal_illumination: nominal_illumination.clone(),
            initialized_illumination,
            initialized_model,
            parameters: self.parameters.clone(),
            options: self.options.clone(),
            initial_parameters,
            initialized_parameters,
            parameter_names: active.iter().map(ActiveParameter::name).collect(),
            observations,
            fit_history,
            diagnostics,
            runtime: PlanarArrayInitializationRuntime {
                elapsed_seconds: started.elapsed().as_secs_f64(),
                measurement_passes: 2,
                physical_objective_evaluations: evaluations,
            },
        };
        result.validate()?;
        Ok(result)
    }
}

#[derive(Clone, Copy)]
struct CandidateFrame {
    frame: usize,
    source: usize,
    scalar: f64,
    nominal: KVector,
}

fn select_candidates<M: MeasurementRead>(
    measurements: &M,
    optics: &Optics,
    illumination: &Illumination,
    model: &ImagePlaneModel,
    options: &BrightfieldCircleOptions,
) -> Result<Vec<CandidateFrame>> {
    let resolved = illumination.resolve(optics)?;
    let requested = options.frame_indices.as_ref().map(|values| {
        let mut values = values.clone();
        values.sort_unstable();
        values
    });
    let indices: Vec<_> = requested
        .clone()
        .unwrap_or_else(|| (0..resolved.frame_count()).collect());
    let mut candidates = Vec::new();
    for frame in indices {
        if frame >= resolved.frame_count() {
            return Err(Error::FrameOutOfRange {
                index: frame,
                frames: resolved.frame_count(),
            });
        }
        let explicit = requested.is_some();
        let weight = measurements.frame_weight(frame)?;
        let resolved_frame = &resolved.frames()[frame];
        let eligible = resolved_frame.contributions().len() == 1
            && weight > 0.0
            && resolved_frame.gain() > 0.0;
        if !eligible {
            if explicit {
                return Err(Error::InvalidMeasurements(format!(
                    "selected frame {frame} must have positive measurement weight and exactly one positive source contribution"
                )));
            }
            continue;
        }
        if measurements
            .frame_mask(frame)?
            .is_some_and(|mask| mask.contains(&0))
        {
            if explicit {
                return Err(Error::InvalidMeasurements(format!(
                    "selected frame {frame} has invalid pixels; circle detection requires an all-valid frame"
                )));
            }
            continue;
        }
        let contribution = resolved_frame.contributions()[0];
        let power = resolved.source_power()[contribution.source];
        let scalar = resolved_frame.gain() * contribution.intensity_weight * power;
        if !scalar.is_finite() || scalar <= 0.0 {
            if explicit {
                return Err(Error::InvalidMeasurements(format!(
                    "selected frame {frame} has a non-positive complete intensity factor"
                )));
            }
            continue;
        }
        let nominal = model.k_vectors()[contribution.source];
        let nominal_na = k_to_na(optics, nominal.kx.hypot(nominal.ky));
        if nominal_na + options.center_search_radius_na + options.brightfield_margin_na
            >= optics.objective_na
        {
            if explicit {
                return Err(Error::InvalidMeasurements(format!(
                    "selected frame {frame} is not safely inside the bright-field boundary"
                )));
            }
            continue;
        }
        candidates.push(CandidateFrame {
            frame,
            source: contribution.source,
            scalar,
            nominal,
        });
    }
    if candidates.is_empty() {
        return Err(Error::InvalidMeasurements(
            "no safe single-source bright-field frames are available".into(),
        ));
    }
    candidates.sort_by_key(|value| (value.source, value.frame));
    Ok(candidates)
}

fn validate_inputs<M: MeasurementRead>(
    measurements: &M,
    optics: &Optics,
    illumination: &Illumination,
    model: &ImagePlaneModel,
) -> Result<()> {
    measurements.validate()?;
    optics.validate()?;
    model.validate()?;
    if measurements.frame_count() != model.frame_count()
        || measurements.image_shape() != model.image_shape()
    {
        return Err(Error::InvalidShape(
            "measurements and initialization model must have matching frame and image shapes"
                .into(),
        ));
    }
    let expected = ImagePlaneModel::from_experiment(
        optics,
        illumination,
        model.image_shape(),
        ReconstructionShape::Exact(model.reconstruction_shape()),
    )?;
    if expected.k_vectors() != model.k_vectors()
        || expected.crop_indices().as_slice() != model.crop_indices().as_slice()
        || expected.subpixel_offsets() != model.subpixel_offsets()
        || expected.multiplexing_matrix() != model.multiplexing_matrix()
        || expected.frame_gains() != model.frame_gains()
        || expected
            .pupil()
            .support()
            .iter()
            .ne(model.pupil().support().iter())
        || expected.sampling().dkx != model.sampling().dkx
        || expected.sampling().dky != model.sampling().dky
        || expected.sampling().low_res_pixel_size != model.sampling().low_res_pixel_size
    {
        return Err(Error::InvalidModel(
            "initialization model was not compiled from the supplied optics and nominal illumination"
                .into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn frame_spectrum<M: MeasurementRead>(
    measurements: &M,
    model: &ImagePlaneModel,
    frame: usize,
    scalar: f64,
    backend: &dyn Backend,
    spectrum: &mut [Complex64],
    shifted: &mut [Complex64],
    column: &mut [Complex64],
) -> Result<f64> {
    let measured = measurements.frame(frame)?;
    let shape = model.image_shape();
    let mut negative = 0usize;
    for row in 0..shape.0 {
        let row_window = hann(row, shape.0);
        for col in 0..shape.1 {
            let pixel = row * shape.1 + col;
            let corrected = (measured[pixel] - model.background_value(frame, pixel)?) / scalar;
            if !corrected.is_finite() {
                return Err(Error::InvalidMeasurements(format!(
                    "frame {frame} has a non-finite corrected intensity at pixel {pixel}"
                )));
            }
            negative += usize::from(corrected < 0.0);
            spectrum[pixel] = Complex64::new(corrected * row_window * hann(col, shape.1), 0.0);
        }
    }
    backend.fft2(spectrum, shape, FftDirection::Forward, column)?;
    fftshift_copy(spectrum, shifted, shape);
    Ok(negative as f64 / measured.len() as f64)
}

fn hann(index: usize, length: usize) -> f64 {
    if length <= 1 {
        1.0
    } else {
        0.5 - 0.5 * (TAU * index as f64 / (length - 1) as f64).cos()
    }
}

fn gaussian_blur_in_place(values: &mut [f64], shape: (usize, usize), sigma: f64) {
    let radius = (3.0 * sigma).ceil() as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp())
        .collect();
    let sum = kernel.iter().sum::<f64>();
    for value in &mut kernel {
        *value /= sum;
    }
    let mut temporary = vec![0.0; values.len()];
    for row in 0..shape.0 {
        for col in 0..shape.1 {
            let mut total = 0.0;
            for (kernel_index, &weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                let source = (col as isize + offset).clamp(0, shape.1 as isize - 1) as usize;
                total += weight * values[row * shape.1 + source];
            }
            temporary[row * shape.1 + col] = total;
        }
    }
    for row in 0..shape.0 {
        for col in 0..shape.1 {
            let mut total = 0.0;
            for (kernel_index, &weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                let source = (row as isize + offset).clamp(0, shape.0 as isize - 1) as usize;
                total += weight * temporary[source * shape.1 + col];
            }
            values[row * shape.1 + col] = total;
        }
    }
}

#[derive(Clone, Copy, Default)]
struct CircleScore {
    first: f64,
    second: f64,
    combined: f64,
    arc_fraction: f64,
}

#[allow(clippy::too_many_arguments)]
fn detect_circle(
    normalized: &[f64],
    shape: (usize, usize),
    model: &ImagePlaneModel,
    optics: &Optics,
    candidate: &CandidateFrame,
    negative_fraction: f64,
    options: &BrightfieldCircleOptions,
) -> BrightfieldCircleObservation {
    let na_per_k = optics.wavelength_vacuum_m / TAU;
    let dkx_na = model.sampling().dkx * na_per_k;
    let dky_na = model.sampling().dky * na_per_k;
    let pixel_na = dkx_na.min(dky_na);
    let center_search = options.center_search_radius_na;
    let nominal_na = [
        candidate.nominal.kx * na_per_k,
        candidate.nominal.ky * na_per_k,
    ];
    let separation = 2.0 * nominal_na[0].hypot(nominal_na[1]);
    let conjugate_overlap = separation <= 2.0 * center_search;
    let radius_steps = (options.pupil_radius_search_na / pixel_na).ceil() as isize;
    let x_steps = (center_search / dkx_na).ceil() as isize;
    let y_steps = (center_search / dky_na).ceil() as isize;
    let mut best = None::<([f64; 2], f64, CircleScore)>;
    for radius_step in -radius_steps..=radius_steps {
        let radius_na = optics.objective_na + radius_step as f64 * pixel_na;
        if radius_na <= 0.0
            || (radius_na - optics.objective_na).abs() > options.pupil_radius_search_na
        {
            continue;
        }
        for row_step in -y_steps..=y_steps {
            for col_step in -x_steps..=x_steps {
                let center = [
                    nominal_na[0] + col_step as f64 * dkx_na,
                    nominal_na[1] + row_step as f64 * dky_na,
                ];
                if (center[0] - nominal_na[0]).hypot(center[1] - nominal_na[1]) > center_search {
                    continue;
                }
                let score = circle_score(
                    normalized, shape, center, radius_na, dkx_na, dky_na, options,
                );
                if best.is_none_or(|value| score.combined > value.2.combined) {
                    best = Some((center, radius_na, score));
                }
            }
        }
    }
    let Some((mut center, mut radius_na, mut score)) = best else {
        return rejected_observation(
            candidate,
            optics.objective_na,
            negative_fraction,
            "no finite circle candidate was available",
        );
    };
    for scale in [0.5, 0.25] {
        let origin = center;
        let origin_radius = radius_na;
        for radius_step in -1..=1 {
            for row_step in -1..=1 {
                for col_step in -1..=1 {
                    let trial_center = [
                        origin[0] + col_step as f64 * dkx_na * scale,
                        origin[1] + row_step as f64 * dky_na * scale,
                    ];
                    if (trial_center[0] - nominal_na[0]).hypot(trial_center[1] - nominal_na[1])
                        > center_search
                    {
                        continue;
                    }
                    let trial_radius = origin_radius + radius_step as f64 * pixel_na * scale;
                    if trial_radius <= 0.0
                        || (trial_radius - optics.objective_na).abs()
                            > options.pupil_radius_search_na
                    {
                        continue;
                    }
                    let trial_score = circle_score(
                        normalized,
                        shape,
                        trial_center,
                        trial_radius,
                        dkx_na,
                        dky_na,
                        options,
                    );
                    if trial_score.combined > score.combined {
                        center = trial_center;
                        radius_na = trial_radius;
                        score = trial_score;
                    }
                }
            }
        }
    }
    let conjugate_score = circle_score(
        normalized,
        shape,
        [-center[0], -center[1]],
        radius_na,
        dkx_na,
        dky_na,
        options,
    )
    .combined;
    let nominal_radius = nominal_na[0].hypot(nominal_na[1]);
    let rejection = if nominal_radius <= pixel_na {
        Some("on-axis magnitude spectra do not identify a signed source center".to_string())
    } else if conjugate_overlap {
        Some("nominal and conjugate center-search regions overlap".to_string())
    } else if score.arc_fraction < options.minimum_arc_fraction {
        Some("insufficient usable circular arc".to_string())
    } else if score.first < options.minimum_edge_contrast {
        Some("normalized circular-edge contrast is below the configured minimum".to_string())
    } else if !score.combined.is_finite() {
        Some("circle score is non-finite".to_string())
    } else {
        None
    };
    let accepted = rejection.is_none();
    let k = [center[0] / na_per_k, center[1] / na_per_k];
    let grid = [
        shape.0 as f64 / 2.0 + k[1] / model.sampling().dky,
        shape.1 as f64 / 2.0 + k[0] / model.sampling().dkx,
    ];
    BrightfieldCircleObservation {
        frame_index: candidate.frame,
        source_index: candidate.source,
        nominal_k_rad_per_m: [candidate.nominal.kx, candidate.nominal.ky],
        detected_k_rad_per_m: accepted.then_some(k),
        detected_na: accepted.then_some(center),
        fourier_grid_position: accepted.then_some(grid),
        fitted_pupil_radius_na: radius_na,
        first_derivative_score: score.first,
        second_derivative_score: score.second,
        combined_score: score.combined,
        conjugate_score,
        usable_arc_fraction: score.arc_fraction,
        confidence: if accepted {
            (score.first - options.minimum_edge_contrast).clamp(1e-6, 1.0)
        } else {
            0.0
        },
        negative_sample_fraction: negative_fraction,
        rejection_reason: rejection,
    }
}

#[allow(clippy::too_many_arguments)]
fn circle_score(
    values: &[f64],
    shape: (usize, usize),
    center_na: [f64; 2],
    radius_na: f64,
    dkx_na: f64,
    dky_na: f64,
    options: &BrightfieldCircleOptions,
) -> CircleScore {
    let delta_na = options.radial_derivative_step_pixels * dkx_na.min(dky_na);
    let second_radius = radius_na + options.gaussian_sigma_pixels * dkx_na.min(dky_na);
    // A point-wise denominator makes tiny Gaussian tails look like perfect edges.
    // Normalize by one spectrum-wide scale so an aligned circumference is rewarded
    // for sustained contrast rather than isolated relative changes in the background.
    let intensity_scale = values.iter().copied().fold(0.0, f64::max).max(f64::EPSILON);
    let conjugate = [-center_na[0], -center_na[1]];
    let mut first = 0.0;
    let mut second = 0.0;
    let mut valid = 0usize;
    for sample in 0..options.angular_samples {
        let angle = TAU * sample as f64 / options.angular_samples as f64;
        let (sin, cos) = angle.sin_cos();
        let edge = [
            center_na[0] + radius_na * cos,
            center_na[1] + radius_na * sin,
        ];
        if (edge[0] - conjugate[0]).hypot(edge[1] - conjugate[1]) < radius_na
            && center_na[0].hypot(center_na[1]) > delta_na
        {
            continue;
        }
        let inner = sample_na(
            values,
            shape,
            [
                center_na[0] + (radius_na - delta_na) * cos,
                center_na[1] + (radius_na - delta_na) * sin,
            ],
            dkx_na,
            dky_na,
        );
        let outer = sample_na(
            values,
            shape,
            [
                center_na[0] + (radius_na + delta_na) * cos,
                center_na[1] + (radius_na + delta_na) * sin,
            ],
            dkx_na,
            dky_na,
        );
        let second_inner = sample_na(
            values,
            shape,
            [
                center_na[0] + (second_radius - delta_na) * cos,
                center_na[1] + (second_radius - delta_na) * sin,
            ],
            dkx_na,
            dky_na,
        );
        let second_mid = sample_na(
            values,
            shape,
            [
                center_na[0] + second_radius * cos,
                center_na[1] + second_radius * sin,
            ],
            dkx_na,
            dky_na,
        );
        let second_outer = sample_na(
            values,
            shape,
            [
                center_na[0] + (second_radius + delta_na) * cos,
                center_na[1] + (second_radius + delta_na) * sin,
            ],
            dkx_na,
            dky_na,
        );
        let (Some(inner), Some(outer), Some(second_inner), Some(second_mid), Some(second_outer)) =
            (inner, outer, second_inner, second_mid, second_outer)
        else {
            continue;
        };
        first += (inner - outer) / intensity_scale;
        second += (second_inner - 2.0 * second_mid + second_outer).abs() / (4.0 * intensity_scale);
        valid += 1;
    }
    if valid == 0 {
        return CircleScore::default();
    }
    let first = first / valid as f64;
    let second = second / valid as f64;
    CircleScore {
        first,
        second,
        combined: first.max(0.0) + 0.25 * second,
        arc_fraction: valid as f64 / options.angular_samples as f64,
    }
}

fn sample_na(
    values: &[f64],
    shape: (usize, usize),
    coordinate_na: [f64; 2],
    dkx_na: f64,
    dky_na: f64,
) -> Option<f64> {
    let column = shape.1 as f64 / 2.0 + coordinate_na[0] / dkx_na;
    let row = shape.0 as f64 / 2.0 + coordinate_na[1] / dky_na;
    if row < 0.0 || column < 0.0 || row > (shape.0 - 1) as f64 || column > (shape.1 - 1) as f64 {
        return None;
    }
    let row0 = row.floor() as usize;
    let col0 = column.floor() as usize;
    let row1 = (row0 + 1).min(shape.0 - 1);
    let col1 = (col0 + 1).min(shape.1 - 1);
    let tr = row - row0 as f64;
    let tc = column - col0 as f64;
    let a = values[row0 * shape.1 + col0] * (1.0 - tc) + values[row0 * shape.1 + col1] * tc;
    let b = values[row1 * shape.1 + col0] * (1.0 - tc) + values[row1 * shape.1 + col1] * tc;
    Some(a * (1.0 - tr) + b * tr)
}

fn rejected_observation(
    candidate: &CandidateFrame,
    radius_na: f64,
    negative_sample_fraction: f64,
    reason: impl Into<String>,
) -> BrightfieldCircleObservation {
    BrightfieldCircleObservation {
        frame_index: candidate.frame,
        source_index: candidate.source,
        nominal_k_rad_per_m: [candidate.nominal.kx, candidate.nominal.ky],
        detected_k_rad_per_m: None,
        detected_na: None,
        fourier_grid_position: None,
        fitted_pupil_radius_na: radius_na,
        first_derivative_score: 0.0,
        second_derivative_score: 0.0,
        combined_score: 0.0,
        conjugate_score: 0.0,
        usable_arc_fraction: 0.0,
        confidence: 0.0,
        negative_sample_fraction,
        rejection_reason: Some(reason.into()),
    }
}

#[derive(Clone)]
struct ActiveParameter {
    kind: ParameterKind,
    spec: CalibrationParameterSpec,
}

#[derive(Clone, Copy)]
enum ParameterKind {
    Translation(usize),
    Rotation(usize),
    Pitch(usize),
    Reference(usize),
}

impl ActiveParameter {
    fn name(&self) -> String {
        match self.kind {
            ParameterKind::Translation(axis) => ["tx_m", "ty_m", "tz_m"][axis].into(),
            ParameterKind::Rotation(axis) => ["rx_rad", "ry_rad", "rz_rad"][axis].into(),
            ParameterKind::Pitch(axis) => ["pitch_x_m", "pitch_y_m"][axis].into(),
            ParameterKind::Reference(axis) => ["reference_column", "reference_row"][axis].into(),
        }
    }

    fn value(&self, values: &PlanarArrayParameterValues) -> f64 {
        match self.kind {
            ParameterKind::Translation(axis) => values.translation_m[axis],
            ParameterKind::Rotation(axis) => values.rotation_rad[axis],
            ParameterKind::Pitch(axis) => values.pitch_m[axis],
            ParameterKind::Reference(axis) => values.reference_index[axis],
        }
    }

    fn set(&self, values: &mut PlanarArrayParameterValues, value: f64) {
        match self.kind {
            ParameterKind::Translation(axis) => values.translation_m[axis] = value,
            ParameterKind::Rotation(axis) => values.rotation_rad[axis] = value,
            ParameterKind::Pitch(axis) => values.pitch_m[axis] = value,
            ParameterKind::Reference(axis) => values.reference_index[axis] = value,
        }
    }
}

fn active_parameters(parameters: &PlanarArrayCalibrationParameters) -> Vec<ActiveParameter> {
    let mut active = Vec::new();
    for axis in 0..3 {
        if let Some(spec) = &parameters.translation[axis] {
            active.push(ActiveParameter {
                kind: ParameterKind::Translation(axis),
                spec: spec.clone(),
            });
        }
    }
    for axis in 0..3 {
        if let Some(spec) = &parameters.rotation[axis] {
            active.push(ActiveParameter {
                kind: ParameterKind::Rotation(axis),
                spec: spec.clone(),
            });
        }
    }
    for axis in 0..2 {
        if let Some(spec) = &parameters.pitch[axis] {
            active.push(ActiveParameter {
                kind: ParameterKind::Pitch(axis),
                spec: spec.clone(),
            });
        }
    }
    for axis in 0..2 {
        if let Some(spec) = &parameters.reference_index[axis] {
            active.push(ActiveParameter {
                kind: ParameterKind::Reference(axis),
                spec: spec.clone(),
            });
        }
    }
    active
}

fn validate_initial_values(
    active: &[ActiveParameter],
    values: &PlanarArrayParameterValues,
) -> Result<()> {
    for parameter in active {
        let value = parameter.value(values);
        if value < parameter.spec.lower_bound || value > parameter.spec.upper_bound {
            return Err(Error::InvalidParameter {
                name: "parameters",
                reason: format!(
                    "initial {}={value:.6e} lies outside [{:.6e}, {:.6e}]",
                    parameter.name(),
                    parameter.spec.lower_bound,
                    parameter.spec.upper_bound
                ),
            });
        }
    }
    Ok(())
}

fn predicted_vectors(
    optics: &Optics,
    template: &Illumination,
    values: &PlanarArrayParameterValues,
) -> Result<Vec<KVector>> {
    Ok(values
        .to_illumination(template)?
        .resolve(optics)?
        .k_vectors()
        .to_vec())
}

#[allow(clippy::too_many_arguments)]
fn objective(
    optics: &Optics,
    template: &Illumination,
    initial: &PlanarArrayParameterValues,
    values: &PlanarArrayParameterValues,
    active: &[ActiveParameter],
    observations: &[&BrightfieldCircleObservation],
    robust_scale: f64,
    evaluations: &mut usize,
) -> Result<(f64, f64, f64)> {
    *evaluations += 1;
    let vectors = predicted_vectors(optics, template, values)?;
    let mut data_loss = 0.0;
    let mut weight_sum = 0.0;
    for observation in observations {
        let detected = observation.detected_na.ok_or_else(|| {
            Error::InvalidMeasurements("accepted observation has no detected center".into())
        })?;
        let predicted = vectors[observation.source_index];
        let residuals = [
            k_to_na(optics, predicted.kx) - detected[0],
            k_to_na(optics, predicted.ky) - detected[1],
        ];
        let weight = observation.confidence.max(1e-12);
        for residual in residuals {
            data_loss += weight * huber(residual, robust_scale);
            weight_sum += weight;
        }
    }
    data_loss /= weight_sum.max(f64::EPSILON);
    let mut regularization = 0.0;
    for parameter in active {
        if let Some(center) = parameter.spec.prior_center {
            let normalized = (parameter.value(values) - center) / parameter.spec.scale;
            regularization +=
                0.5 * parameter.spec.regularization_strength * normalized * normalized;
        }
        let initial_value = parameter.value(initial);
        if !initial_value.is_finite() {
            return Err(Error::InvalidModel(
                "initial physical parameter is non-finite".into(),
            ));
        }
    }
    Ok((data_loss, regularization, data_loss + regularization))
}

fn huber(residual: f64, scale: f64) -> f64 {
    let magnitude = residual.abs();
    if magnitude <= scale {
        0.5 * residual * residual
    } else {
        scale * (magnitude - 0.5 * scale)
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_parameters(
    optics: &Optics,
    template: &Illumination,
    initial: &PlanarArrayParameterValues,
    active: &[ActiveParameter],
    observations: &[&BrightfieldCircleObservation],
    options: &BrightfieldCircleOptions,
    evaluations: &mut usize,
    callback: &mut Option<&mut dyn PlanarArrayInitializationCallback>,
) -> Result<(
    PlanarArrayParameterValues,
    Vec<PlanarArrayInitializationFitRecord>,
)> {
    let mut current = initial.clone();
    let mut current_objective = objective(
        optics,
        template,
        initial,
        &current,
        active,
        observations,
        options.robust_residual_scale_na,
        evaluations,
    )?;
    let mut history = Vec::new();
    for step in 1..=options.maximum_fit_steps {
        let mut gradient = vec![0.0; active.len()];
        for (index, parameter) in active.iter().enumerate() {
            let value = parameter.value(&current);
            let plus_value =
                (value + parameter.spec.finite_difference_step).min(parameter.spec.upper_bound);
            let minus_value =
                (value - parameter.spec.finite_difference_step).max(parameter.spec.lower_bound);
            let derivative = if plus_value > value && minus_value < value {
                let mut plus = current.clone();
                parameter.set(&mut plus, plus_value);
                let plus_loss = objective(
                    optics,
                    template,
                    initial,
                    &plus,
                    active,
                    observations,
                    options.robust_residual_scale_na,
                    evaluations,
                )?
                .2;
                let mut minus = current.clone();
                parameter.set(&mut minus, minus_value);
                let minus_loss = objective(
                    optics,
                    template,
                    initial,
                    &minus,
                    active,
                    observations,
                    options.robust_residual_scale_na,
                    evaluations,
                )?
                .2;
                (plus_loss - minus_loss) / (plus_value - minus_value)
            } else if plus_value > value {
                let mut plus = current.clone();
                parameter.set(&mut plus, plus_value);
                (objective(
                    optics,
                    template,
                    initial,
                    &plus,
                    active,
                    observations,
                    options.robust_residual_scale_na,
                    evaluations,
                )?
                .2 - current_objective.2)
                    / (plus_value - value)
            } else if minus_value < value {
                let mut minus = current.clone();
                parameter.set(&mut minus, minus_value);
                (current_objective.2
                    - objective(
                        optics,
                        template,
                        initial,
                        &minus,
                        active,
                        observations,
                        options.robust_residual_scale_na,
                        evaluations,
                    )?
                    .2)
                    / (value - minus_value)
            } else {
                0.0
            };
            gradient[index] = derivative * parameter.spec.scale;
        }
        let norm = gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() {
            return Err(Error::Numerical(
                "planar-array initialization gradient is non-finite".into(),
            ));
        }
        if norm <= f64::EPSILON {
            break;
        }
        let mut step_size = options.fit_initial_step_size;
        let mut accepted = None;
        while step_size >= options.fit_minimum_step_size {
            let mut candidate = current.clone();
            for (parameter, &component) in active.iter().zip(&gradient) {
                let value = parameter.value(&candidate)
                    - step_size * parameter.spec.scale * component / norm;
                parameter.set(
                    &mut candidate,
                    value.clamp(parameter.spec.lower_bound, parameter.spec.upper_bound),
                );
            }
            if candidate == current {
                step_size *= options.fit_step_reduction;
                continue;
            }
            if let Ok(candidate_objective) = objective(
                optics,
                template,
                initial,
                &candidate,
                active,
                observations,
                options.robust_residual_scale_na,
                evaluations,
            ) && candidate_objective.2 < current_objective.2
            {
                accepted = Some((candidate, candidate_objective));
                break;
            }
            step_size *= options.fit_step_reduction;
        }
        if let Some((candidate, candidate_objective)) = accepted {
            let relative = (current_objective.2 - candidate_objective.2)
                / current_objective.2.abs().max(f64::EPSILON);
            current = candidate;
            current_objective = candidate_objective;
            history.push(PlanarArrayInitializationFitRecord {
                step,
                accepted: true,
                step_size,
                data_loss: current_objective.0,
                regularization_loss: current_objective.1,
                total_loss: current_objective.2,
                normalized_values: normalized_values(active, initial, &current),
            });
            emit_progress(
                callback,
                PlanarArrayInitializationStage::PhysicalFit,
                step,
                options.maximum_fit_steps,
                None,
            )?;
            if relative <= options.fit_relative_tolerance {
                break;
            }
        } else {
            history.push(PlanarArrayInitializationFitRecord {
                step,
                accepted: false,
                step_size: 0.0,
                data_loss: current_objective.0,
                regularization_loss: current_objective.1,
                total_loss: current_objective.2,
                normalized_values: normalized_values(active, initial, &current),
            });
            emit_progress(
                callback,
                PlanarArrayInitializationStage::PhysicalFit,
                step,
                options.maximum_fit_steps,
                None,
            )?;
            break;
        }
    }
    Ok((current, history))
}

fn normalized_values(
    active: &[ActiveParameter],
    initial: &PlanarArrayParameterValues,
    current: &PlanarArrayParameterValues,
) -> Vec<f64> {
    active
        .iter()
        .map(|parameter| {
            (parameter.value(current) - parameter.value(initial)) / parameter.spec.scale
        })
        .collect()
}

fn residual_rms(
    optics: &Optics,
    template: &Illumination,
    values: &PlanarArrayParameterValues,
    observations: &[&BrightfieldCircleObservation],
) -> Result<f64> {
    let vectors = predicted_vectors(optics, template, values)?;
    let mut total = 0.0;
    let mut weight = 0.0;
    for observation in observations {
        let detected = observation.detected_na.unwrap();
        let predicted = vectors[observation.source_index];
        let confidence = observation.confidence.max(1e-12);
        total += confidence
            * ((k_to_na(optics, predicted.kx) - detected[0]).powi(2)
                + (k_to_na(optics, predicted.ky) - detected[1]).powi(2));
        weight += 2.0 * confidence;
    }
    Ok((total / weight.max(f64::EPSILON)).sqrt())
}

fn jacobian_rank(
    optics: &Optics,
    template: &Illumination,
    values: &PlanarArrayParameterValues,
    active: &[ActiveParameter],
    observations: &[&BrightfieldCircleObservation],
    tolerance: f64,
) -> Result<(usize, Option<f64>)> {
    let rows = observations.len() * 2;
    let mut columns = Vec::with_capacity(active.len());
    for parameter in active {
        let value = parameter.value(values);
        let plus_value =
            (value + parameter.spec.finite_difference_step).min(parameter.spec.upper_bound);
        let minus_value =
            (value - parameter.spec.finite_difference_step).max(parameter.spec.lower_bound);
        if plus_value == minus_value {
            columns.push(vec![0.0; rows]);
            continue;
        }
        let mut plus = values.clone();
        parameter.set(&mut plus, plus_value);
        let plus_vectors = predicted_vectors(optics, template, &plus)?;
        let mut minus = values.clone();
        parameter.set(&mut minus, minus_value);
        let minus_vectors = predicted_vectors(optics, template, &minus)?;
        let mut column = Vec::with_capacity(rows);
        for observation in observations {
            let weight = observation.confidence.max(1e-12).sqrt();
            let plus = plus_vectors[observation.source_index];
            let minus = minus_vectors[observation.source_index];
            let denominator = plus_value - minus_value;
            column.push(
                weight * parameter.spec.scale * k_to_na(optics, plus.kx - minus.kx) / denominator,
            );
            column.push(
                weight * parameter.spec.scale * k_to_na(optics, plus.ky - minus.ky) / denominator,
            );
        }
        columns.push(column);
    }
    let mut pivots = Vec::new();
    let mut remaining: Vec<_> = (0..columns.len()).collect();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    while !remaining.is_empty() {
        let (position, residual, norm) = remaining
            .iter()
            .enumerate()
            .map(|(position, &column)| {
                let mut residual = columns[column].clone();
                for vector in &basis {
                    let projection = dot(&residual, vector);
                    for (value, &basis_value) in residual.iter_mut().zip(vector) {
                        *value -= projection * basis_value;
                    }
                }
                let norm = dot(&residual, &residual).sqrt();
                (position, residual, norm)
            })
            .max_by(|left, right| left.2.total_cmp(&right.2))
            .unwrap();
        let reference = pivots.first().copied().unwrap_or(norm);
        if !norm.is_finite() || norm <= tolerance * reference.max(f64::EPSILON) {
            break;
        }
        let mut normalized = residual;
        for value in &mut normalized {
            *value /= norm;
        }
        basis.push(normalized);
        pivots.push(norm);
        remaining.remove(position);
    }
    let condition = if pivots.is_empty() {
        None
    } else {
        Some((pivots[0] / pivots[pivots.len() - 1]).powi(2))
    };
    Ok((pivots.len(), condition))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn validate_preserved_model_state(
    nominal: &ImagePlaneModel,
    initialized: &ImagePlaneModel,
) -> Result<()> {
    if nominal.image_shape() != initialized.image_shape()
        || nominal.reconstruction_shape() != initialized.reconstruction_shape()
        || nominal.pupil() != initialized.pupil()
        || nominal.background() != initialized.background()
        || nominal.frame_gains() != initialized.frame_gains()
        || nominal.multiplexing_matrix() != initialized.multiplexing_matrix()
    {
        return Err(Error::InvalidModel(
            "physical initialization changed non-geometric compiled model state".into(),
        ));
    }
    Ok(())
}

fn weighted_radius(observations: &[&BrightfieldCircleObservation]) -> Result<f64> {
    let weight = observations
        .iter()
        .map(|value| value.confidence)
        .sum::<f64>();
    if !weight.is_finite() || weight <= 0.0 {
        return Err(Error::InvalidMeasurements(
            "accepted circle observations have no positive confidence".into(),
        ));
    }
    Ok(observations
        .iter()
        .map(|value| value.confidence * value.fitted_pupil_radius_na)
        .sum::<f64>()
        / weight)
}

fn emit_progress(
    callback: &mut Option<&mut dyn PlanarArrayInitializationCallback>,
    stage: PlanarArrayInitializationStage,
    completed: usize,
    total: usize,
    frame_index: Option<usize>,
) -> Result<()> {
    if let Some(callback) = callback.as_deref_mut()
        && callback.on_progress(&PlanarArrayInitializationProgress {
            stage,
            completed,
            total,
            frame_index,
        })? == PlanarArrayInitializationAction::Cancel
    {
        return Err(Error::Numerical(
            "planar-array initialization was cancelled".into(),
        ));
    }
    Ok(())
}

fn k_to_na(optics: &Optics, value: f64) -> f64 {
    value * optics.wavelength_vacuum_m / TAU
}

fn observation_is_finite(value: &BrightfieldCircleObservation) -> bool {
    value
        .nominal_k_rad_per_m
        .iter()
        .chain(value.detected_k_rad_per_m.iter().flatten())
        .chain(value.detected_na.iter().flatten())
        .chain(value.fourier_grid_position.iter().flatten())
        .chain(
            [
                value.fitted_pupil_radius_na,
                value.first_derivative_score,
                value.second_derivative_score,
                value.combined_score,
                value.conjugate_score,
                value.usable_arc_fraction,
                value.confidence,
                value.negative_sample_fraction,
            ]
            .iter(),
        )
        .all(|value| value.is_finite())
}

fn diagnostics_are_finite(value: &PlanarArrayInitializationDiagnostics) -> bool {
    [
        value.configured_pupil_radius_na,
        value.fitted_pupil_radius_na,
        value.initial_residual_rms_na,
        value.final_residual_rms_na,
    ]
    .iter()
    .all(|value| value.is_finite())
        && value
            .jacobian_condition_estimate
            .is_none_or(|condition| condition.is_finite() && condition >= 0.0)
}

fn invalid(name: &'static str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{ArrayPose, PlanarLedArray};

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

    fn illumination(translation: [f64; 3]) -> Illumination {
        Illumination::from_geometry(PlanarLedArray::new(
            (5, 5),
            (4e-3, 4e-3),
            (2.0, 2.0),
            ArrayPose::from_translation(translation),
        ))
        .unwrap()
    }

    fn observations_for(
        optics: &Optics,
        illumination: &Illumination,
    ) -> Vec<BrightfieldCircleObservation> {
        illumination
            .resolve(optics)
            .unwrap()
            .k_vectors()
            .iter()
            .enumerate()
            .map(|(source, vector)| BrightfieldCircleObservation {
                frame_index: source,
                source_index: source,
                nominal_k_rad_per_m: [vector.kx, vector.ky],
                detected_k_rad_per_m: Some([vector.kx, vector.ky]),
                detected_na: Some([k_to_na(optics, vector.kx), k_to_na(optics, vector.ky)]),
                fourier_grid_position: Some([0.0, 0.0]),
                fitted_pupil_radius_na: optics.objective_na,
                first_derivative_score: 1.0,
                second_derivative_score: 1.0,
                combined_score: 1.0,
                conjugate_score: 1.0,
                usable_arc_fraction: 1.0,
                confidence: 1.0,
                negative_sample_fraction: 0.0,
                rejection_reason: None,
            })
            .collect()
    }

    #[test]
    fn physical_fit_recovers_lateral_translation_from_detected_vectors() {
        let optics = optics();
        let nominal = illumination([0.0, 0.0, -80e-3]);
        let truth = illumination([0.8e-3, -0.6e-3, -80e-3]);
        let observations = observations_for(&optics, &truth);
        let accepted: Vec<_> = observations.iter().collect();
        let spec = CalibrationParameterSpec::new(-2e-3, 2e-3, 1e-3).finite_difference_step(1e-6);
        let parameters = PlanarArrayCalibrationParameters::builder()
            .translation_specs([Some(spec.clone()), Some(spec), None])
            .build()
            .unwrap();
        let active = active_parameters(&parameters);
        let initial = PlanarArrayParameterValues::from_illumination(&nominal).unwrap();
        let options = BrightfieldCircleOptions {
            maximum_fit_steps: 200,
            fit_initial_step_size: 0.25,
            ..Default::default()
        };
        let mut evaluations = 0;
        let mut callback = None;
        let (fitted, _) = fit_parameters(
            &optics,
            &nominal,
            &initial,
            &active,
            &accepted,
            &options,
            &mut evaluations,
            &mut callback,
        )
        .unwrap();
        assert!((fitted.translation_m[0] - 0.8e-3).abs() < 5e-5);
        assert!((fitted.translation_m[1] + 0.6e-3).abs() < 5e-5);
    }

    #[test]
    fn rank_test_rejects_axial_distance_pitch_scale_gauge() {
        let optics = optics();
        let nominal = illumination([0.0, 0.0, -80e-3]);
        let observations = observations_for(&optics, &nominal);
        let accepted: Vec<_> = observations.iter().collect();
        let distance =
            CalibrationParameterSpec::new(-0.12, -0.04, 1e-3).finite_difference_step(1e-5);
        let pitch = CalibrationParameterSpec::new(3e-3, 5e-3, 1e-4).finite_difference_step(1e-6);
        let parameters = PlanarArrayCalibrationParameters::builder()
            .translation_specs([None, None, Some(distance)])
            .pitch_specs([Some(pitch.clone()), Some(pitch)])
            .build()
            .unwrap();
        let active = active_parameters(&parameters);
        let values = PlanarArrayParameterValues::from_illumination(&nominal).unwrap();
        let (rank, _) =
            jacobian_rank(&optics, &nominal, &values, &active, &accepted, 1e-7).unwrap();
        assert!(rank < active.len());
    }

    #[test]
    fn physical_fit_recovers_global_components_without_changing_fixed_values() {
        // Exact canonical vectors isolate the physical fit from pixel-level
        // circle-localization error. Each scale-gauge partner stays fixed.
        let optics = optics();
        let nominal = illumination([0.0, 0.0, -80e-3]);
        let initial = PlanarArrayParameterValues::from_illumination(&nominal).unwrap();
        for component in 0..8 {
            let mut expected = initial.clone();
            let mut builder = PlanarArrayCalibrationParameters::builder();
            let tolerance;
            if component < 3 {
                let mut specs = [None, None, None];
                specs[component] = Some(
                    CalibrationParameterSpec::new(-0.15, 0.15, 0.05).finite_difference_step(1e-5),
                );
                builder = builder.rotation_specs(specs);
                expected.rotation_rad[component] = 0.04;
                tolerance = 2e-4;
            } else if component < 5 {
                let axis = component - 3;
                let mut specs = [None, None];
                specs[axis] = Some(
                    CalibrationParameterSpec::new(3e-3, 5e-3, 0.2e-3).finite_difference_step(1e-7),
                );
                builder = builder.pitch_specs(specs);
                expected.pitch_m[axis] = 4.2e-3;
                tolerance = 1e-6;
            } else if component == 5 {
                builder = builder.translation_specs([
                    None,
                    None,
                    Some(
                        CalibrationParameterSpec::new(-0.1, -0.06, 4e-3)
                            .finite_difference_step(1e-6),
                    ),
                ]);
                expected.translation_m[2] = -84e-3;
                tolerance = 2e-5;
            } else {
                let axis = component - 6;
                let mut specs = [None, None];
                specs[axis] =
                    Some(CalibrationParameterSpec::new(1.0, 3.0, 0.2).finite_difference_step(1e-5));
                builder = builder.reference_index_specs(specs);
                expected.reference_index[axis] = 2.2;
                tolerance = 1e-3;
            }
            let parameters = builder.build().unwrap();
            let active = active_parameters(&parameters);
            let truth = expected.to_illumination(&nominal).unwrap();
            let observations = observations_for(&optics, &truth);
            let accepted: Vec<_> = observations.iter().collect();
            let (rank, condition) =
                jacobian_rank(&optics, &nominal, &initial, &active, &accepted, 1e-8).unwrap();
            assert_eq!(rank, 1, "component {component}");
            assert!(condition.unwrap().is_finite());
            let options = BrightfieldCircleOptions {
                maximum_fit_steps: 200,
                fit_initial_step_size: 0.25,
                ..Default::default()
            };
            let (fitted, history) = fit_parameters(
                &optics, &nominal, &initial, &active, &accepted, &options, &mut 0, &mut None,
            )
            .unwrap();
            assert!(
                (active[0].value(&fitted) - active[0].value(&expected)).abs() < tolerance,
                "{}: expected {}, fitted {}",
                active[0].name(),
                active[0].value(&expected),
                active[0].value(&fitted)
            );
            let mut fixed = fitted.clone();
            active[0].set(&mut fixed, active[0].value(&initial));
            assert_eq!(fixed, initial);
            assert!(history.iter().any(|entry| entry.accepted));
            assert!(residual_rms(&optics, &nominal, &fitted, &accepted).unwrap() < 1e-6);
        }
    }

    #[test]
    fn rank_test_rejects_collinear_centers_even_with_priors() {
        let optics = optics();
        let nominal = illumination([0.0, 0.0, -80e-3]);
        let observations = observations_for(&optics, &nominal);
        // Only the central row: its pitch_y derivative is identically zero.
        let accepted: Vec<_> = observations[10..15].iter().collect();
        let pitch = CalibrationParameterSpec::new(3e-3, 5e-3, 1e-4)
            .finite_difference_step(1e-6)
            .prior(4e-3, 1.0);
        let parameters = PlanarArrayCalibrationParameters::builder()
            .pitch_specs([Some(pitch.clone()), Some(pitch)])
            .build()
            .unwrap();
        let active = active_parameters(&parameters);
        let initial = PlanarArrayParameterValues::from_illumination(&nominal).unwrap();
        let (rank, _) =
            jacobian_rank(&optics, &nominal, &initial, &active, &accepted, 1e-8).unwrap();
        assert_eq!(rank, 1);
    }

    #[test]
    fn explicit_frame_selection_rejects_masks_zero_weights_and_cutoff_frames() {
        use crate::measurements::MeasurementStack;

        let optics = optics();
        let nominal = illumination([0.0, 0.0, -80e-3]);
        let model = ImagePlaneModel::from_experiment(
            &optics,
            &nominal,
            (48, 56),
            ReconstructionShape::Exact((96, 112)),
        )
        .unwrap();
        let mut measurements =
            MeasurementStack::from_frames(&vec![vec![1.0; 48 * 56]; 25], (48, 56)).unwrap();
        let options = BrightfieldCircleOptions {
            center_search_radius_na: 0.012,
            ..Default::default()
        };
        let frames = select_candidates(&measurements, &optics, &nominal, &model, &options).unwrap();
        assert!(frames.iter().all(|f| f.frame == f.source));
        assert!(frames.iter().any(|f| f.source == 11));
        assert!(!frames.iter().any(|f| f.source == 0));
        let mut explicit = options.clone();
        explicit.frame_indices = Some(vec![0]);
        assert!(
            matches!(select_candidates(&measurements, &optics, &nominal, &model, &explicit),
            Err(Error::InvalidMeasurements(message)) if message.contains("bright-field boundary"))
        );
        measurements.set_frame_weight(11, 0.0).unwrap();
        explicit.frame_indices = Some(vec![11]);
        assert!(
            matches!(select_candidates(&measurements, &optics, &nominal, &model, &explicit),
            Err(Error::InvalidMeasurements(message)) if message.contains("positive measurement weight"))
        );
        measurements.set_frame_weight(11, 1.0).unwrap();
        let mut mask = ndarray::Array3::from_elem((25, 48, 56), 1u8);
        mask[(11, 0, 0)] = 0;
        let measurements = measurements.with_per_frame_masks(mask).unwrap();
        assert!(
            matches!(select_candidates(&measurements, &optics, &nominal, &model, &explicit),
            Err(Error::InvalidMeasurements(message)) if message.contains("invalid pixels"))
        );
        let frames = select_candidates(&measurements, &optics, &nominal, &model, &options).unwrap();
        assert!(!frames.iter().any(|f| f.source == 11));
    }

    #[test]
    fn detector_rejects_unsigned_centers_and_missing_edges() {
        let optics = optics();
        let nominal = illumination([0.0, 0.0, -80e-3]);
        let shape = (64, 80);
        let model = ImagePlaneModel::from_experiment(
            &optics,
            &nominal,
            shape,
            ReconstructionShape::Exact((128, 160)),
        )
        .unwrap();
        let candidate = |source| CandidateFrame {
            frame: source,
            source,
            scalar: 1.0,
            nominal: model.k_vectors()[source],
        };
        let values = vec![1.0; shape.0 * shape.1];
        let options = BrightfieldCircleOptions::default();
        let on_axis = detect_circle(
            &values,
            shape,
            &model,
            &optics,
            &candidate(12),
            0.0,
            &options,
        );
        assert!(on_axis.rejection_reason.unwrap().contains("on-axis"));
        let overlapping = BrightfieldCircleOptions {
            center_search_radius_na: 0.06,
            ..options.clone()
        };
        let unsigned = detect_circle(
            &values,
            shape,
            &model,
            &optics,
            &candidate(11),
            0.0,
            &overlapping,
        );
        assert!(unsigned.rejection_reason.unwrap().contains("conjugate"));
        let no_edge = detect_circle(
            &values,
            shape,
            &model,
            &optics,
            &candidate(11),
            0.0,
            &options,
        );
        assert!(no_edge.rejection_reason.unwrap().contains("contrast"));
    }

    #[test]
    fn detector_localizes_an_analytic_circular_edge_on_rectangular_sampling() {
        let shape = (64, 80);
        let dkx_na = 0.004;
        let dky_na = 0.005;
        let center = [0.02, -0.015];
        let radius = 0.1;
        let mut values = vec![0.0; shape.0 * shape.1];
        for row in 0..shape.0 {
            for col in 0..shape.1 {
                let coordinate = [
                    (col as f64 - shape.1 as f64 / 2.0) * dkx_na,
                    (row as f64 - shape.0 as f64 / 2.0) * dky_na,
                ];
                let positive = (coordinate[0] - center[0]).hypot(coordinate[1] - center[1]);
                let negative = (coordinate[0] + center[0]).hypot(coordinate[1] + center[1]);
                values[row * shape.1 + col] = if positive <= radius || negative <= radius {
                    1.0
                } else {
                    0.0
                };
            }
        }
        gaussian_blur_in_place(&mut values, shape, 1.0);
        let options = BrightfieldCircleOptions {
            gaussian_sigma_pixels: 1.0,
            ..Default::default()
        };
        let true_score = circle_score(&values, shape, center, radius, dkx_na, dky_na, &options);
        let wrong_score = circle_score(
            &values,
            shape,
            [center[0] + 0.02, center[1]],
            radius,
            dkx_na,
            dky_na,
            &options,
        );
        assert!(
            true_score.combined > wrong_score.combined,
            "true={:?}/{:?}/{:?}, wrong={:?}/{:?}/{:?}",
            true_score.first,
            true_score.second,
            true_score.combined,
            wrong_score.first,
            wrong_score.second,
            wrong_score.combined
        );
        assert!(true_score.arc_fraction >= options.minimum_arc_fraction);
    }
}
