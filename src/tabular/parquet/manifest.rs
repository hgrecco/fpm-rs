use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::reconstruction::RuntimeInfo;

/// Current reconstruction result-bundle manifest format version.
pub const BUNDLE_FORMAT_VERSION: u32 = 1;

/// Options controlling result-bundle identity and preview generation.
#[derive(Clone, Debug, Default)]
pub struct BundleExportOptions {
    /// Optional caller-defined unique run ID; a UUID is generated when absent.
    pub run_id: Option<String>,
    /// Optional human-readable result label.
    pub label: Option<String>,
    /// Whether to write derived PNG previews in addition to lossless arrays and tables.
    pub include_previews: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BundleManifest {
    pub bundle_format_version: u32,
    pub run_id: String,
    pub label: Option<String>,
    pub crate_version: String,
    pub provenance: BTreeMap<String, String>,
    pub dataset: Option<DatasetIdentity>,
    pub runtime: RuntimeInfo,
    pub result: ResultDescriptor,
    pub artifacts: Vec<ManifestArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatasetIdentity {
    pub name: String,
    pub version: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResultDescriptor {
    pub reconstruction_shape: [u64; 2],
    pub image_shape: [u64; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifestArtifact {
    pub role: String,
    pub relative_path: PathBuf,
    pub media_type: String,
    pub byte_size: u64,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<u64>>,
}

pub(crate) const TABLE_SUMMARY: &str = "tables.summary";
pub(crate) const TABLE_HISTORY: &str = "tables.history";
pub(crate) const TABLE_ALGORITHM_METRICS: &str = "tables.algorithm_metrics";
pub(crate) const TABLE_ITERATION_DIAGNOSTICS: &str = "tables.iteration_diagnostics";
pub(crate) const TABLE_FRAME_DIAGNOSTICS: &str = "tables.frame_diagnostics";
pub(crate) const TABLE_RAW_FRAME_STATISTICS: &str = "tables.raw_frame_statistics";
pub(crate) const TABLE_FRAME_EVALUATION: &str = "tables.frame_evaluation";
pub(crate) const TABLE_ILLUMINATION_CALIBRATION: &str = "tables.illumination_calibration";
pub(crate) const TABLE_FRAME_CALIBRATION: &str = "tables.frame_calibration";
pub(crate) const TABLE_SCALAR_DIAGNOSTICS: &str = "tables.scalar_diagnostics";
pub(crate) const TABLE_METADATA: &str = "tables.metadata";

pub(crate) const ARRAY_OBJECT: &str = "arrays.object";
pub(crate) const ARRAY_OBJECT_SPECTRUM: &str = "arrays.object_spectrum";
pub(crate) const ARRAY_PUPIL: &str = "arrays.pupil";
pub(crate) const ARRAY_PUPIL_SUPPORT: &str = "arrays.pupil_support";
pub(crate) const ARRAY_ILLUMINATION_CALIBRATION: &str = "arrays.illumination_calibration";
pub(crate) const ARRAY_FRAME_GAINS: &str = "arrays.frame_gains";
pub(crate) const ARRAY_BACKGROUND: &str = "arrays.background";

pub(crate) const PREVIEW_OBJECT_AMPLITUDE: &str = "previews.object_amplitude";
pub(crate) const PREVIEW_OBJECT_PHASE: &str = "previews.object_phase";
pub(crate) const PREVIEW_PUPIL_AMPLITUDE: &str = "previews.pupil_amplitude";
pub(crate) const PREVIEW_PUPIL_PHASE: &str = "previews.pupil_phase";
pub(crate) const PREVIEW_FOURIER_COVERAGE: &str = "previews.fourier_coverage";

pub(crate) const DOMAIN_DIAGNOSTICS: &str = "domain.diagnostics";
pub(crate) const DOMAIN_EVALUATION: &str = "domain.evaluation";

pub(crate) const KNOWN_ROLES: &[&str] = &[
    TABLE_SUMMARY,
    TABLE_HISTORY,
    TABLE_ALGORITHM_METRICS,
    TABLE_ITERATION_DIAGNOSTICS,
    TABLE_FRAME_DIAGNOSTICS,
    TABLE_RAW_FRAME_STATISTICS,
    TABLE_FRAME_EVALUATION,
    TABLE_ILLUMINATION_CALIBRATION,
    TABLE_FRAME_CALIBRATION,
    TABLE_SCALAR_DIAGNOSTICS,
    TABLE_METADATA,
    ARRAY_OBJECT,
    ARRAY_OBJECT_SPECTRUM,
    ARRAY_PUPIL,
    ARRAY_PUPIL_SUPPORT,
    ARRAY_ILLUMINATION_CALIBRATION,
    ARRAY_FRAME_GAINS,
    ARRAY_BACKGROUND,
    PREVIEW_OBJECT_AMPLITUDE,
    PREVIEW_OBJECT_PHASE,
    PREVIEW_PUPIL_AMPLITUDE,
    PREVIEW_PUPIL_PHASE,
    PREVIEW_FOURIER_COVERAGE,
    DOMAIN_DIAGNOSTICS,
    DOMAIN_EVALUATION,
];
