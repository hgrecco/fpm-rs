//! Error categories returned by validation, numerical, I/O, and bundle operations.
//!
//! Public functions return [`Result`], whose error type is [`enum@Error`]. Match variants
//! when a caller can recover from a specific category; otherwise propagate the error.

use thiserror::Error;

/// Crate-wide result type using [`enum@Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by fpm-rs public operations.
#[derive(Debug, Error)]
pub enum Error {
    /// A supplied or derived array shape is empty, inconsistent, or unsupported.
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    /// A named scalar or option falls outside its accepted domain.
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter {
        /// Stable parameter name.
        name: &'static str,
        /// Explanation of the rejected value or relationship.
        reason: String,
    },
    /// A compiled image-plane model violates a model invariant.
    #[error("model validation failed: {0}")]
    InvalidModel(String),
    /// Measurements or their metadata are incompatible or invalid.
    #[error("measurement validation failed: {0}")]
    InvalidMeasurements(String),
    /// A flat two-dimensional buffer length does not match its `(height, width)` shape.
    #[error("array length {actual} does not match shape {shape:?} (expected {expected})")]
    LengthMismatch {
        /// Number of supplied elements.
        actual: usize,
        /// Number of elements required by `shape`.
        expected: usize,
        /// Requested `(height, width)` shape.
        shape: (usize, usize),
    },
    /// A flat buffer length does not match a general-dimensional shape.
    #[error("array length {actual} does not match shape {shape:?} (expected {expected})")]
    ArrayLengthMismatch {
        /// Number of supplied elements.
        actual: usize,
        /// Number of required elements.
        expected: usize,
        /// Requested axis lengths.
        shape: Vec<usize>,
    },
    /// Multiplying the dimensions of an array shape overflowed addressable storage.
    #[error("array shape {shape:?} overflows addressable storage")]
    ShapeOverflow {
        /// Axis lengths whose product overflowed.
        shape: Vec<usize>,
    },
    /// An ndarray input is not C-contiguous standard row-major storage.
    #[error(
        "{context} with shape {shape:?} and strides {strides:?} is not C-contiguous standard row-major layout"
    )]
    NonStandardLayout {
        /// User-facing name of the offending value.
        context: &'static str,
        /// Logical axis lengths.
        shape: Vec<usize>,
        /// Element strides reported by ndarray.
        strides: Vec<isize>,
    },
    /// ndarray rejected a requested shape or storage layout.
    #[error("ndarray shape construction failed: {0}")]
    NdarrayShape(#[from] ndarray::ShapeError),
    /// A requested acquisition-frame index is outside `0..frames`.
    #[error("frame index {index} is out of range for {frames} frames")]
    FrameOutOfRange {
        /// Requested zero-based frame index.
        index: usize,
        /// Available frame count.
        frames: usize,
    },
    /// A numerical operation produced an invalid or unusable value.
    #[error("numerical error: {0}")]
    Numerical(String),
    /// The requested operation is not implemented for the supplied input.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    /// Dataset discovery, download, verification, or loading failed.
    #[error("dataset error: {0}")]
    Dataset(String),
    /// A result or benchmark bundle uses an unsupported manifest version.
    #[error("unsupported bundle format version {actual}; supported version is {supported}")]
    UnsupportedBundleVersion {
        /// Version found in the manifest.
        actual: u32,
        /// Version supported by this build.
        supported: u32,
    },
    /// A bundle manifest is malformed or violates the bundle contract.
    #[error("invalid bundle manifest: {0}")]
    InvalidManifest(String),
    /// A manifest-declared bundle artifact is absent.
    #[error("bundle artifact `{role}` is missing")]
    MissingArtifact {
        /// Stable artifact role from the manifest.
        role: String,
    },
    /// A bundle artifact's bytes do not match its declared SHA-256 digest.
    #[error("bundle artifact `{role}` has an invalid SHA-256 digest")]
    ArtifactHashMismatch {
        /// Stable artifact role from the manifest.
        role: String,
    },
    /// A Parquet artifact does not have the schema required for its role.
    #[error("bundle artifact `{role}` has an invalid Parquet schema: {reason}")]
    InvalidParquetSchema {
        /// Stable artifact role from the manifest.
        role: String,
        /// Description of the schema mismatch.
        reason: String,
    },
    /// Artifacts within one bundle disagree about the run identifier.
    #[error("bundle artifact has run ID `{actual}`, expected `{expected}`")]
    InconsistentRunId {
        /// Run identifier required by the bundle manifest.
        expected: String,
        /// Run identifier found in an artifact.
        actual: String,
    },
    /// A stored NPY array has a shape inconsistent with its artifact role.
    #[error("bundle array `{role}` has invalid shape: {reason}")]
    InvalidArrayShape {
        /// Stable array role from the manifest.
        role: String,
        /// Description of the shape mismatch.
        reason: String,
    },
    /// A stored NPY array has the wrong element dtype for its role.
    #[error("bundle array `{role}` has invalid dtype `{actual}`, expected `{expected}`")]
    InvalidArrayDtype {
        /// Stable array role from the manifest.
        role: String,
        /// Dtype encoded by the array file.
        actual: String,
        /// Dtype required for the artifact role.
        expected: String,
    },
    /// A bundle manifest names an artifact role unknown to this version.
    #[error("unsupported bundle artifact role `{0}")]
    UnsupportedArtifactRole(String),
    /// A bundle artifact path is absolute or escapes the bundle directory.
    #[error("invalid relative bundle artifact path `{0}")]
    InvalidRelativePath(String),
    /// A temporary bundle workspace is missing required output before publication.
    #[error("bundle workspace is incomplete: {0}")]
    IncompleteBundle(String),
    /// An operation delegated to Polars failed.
    #[cfg(feature = "tabular")]
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
    /// A filesystem or stream operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Reading or writing CSV data failed.
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    /// Decoding or encoding a general image failed.
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    /// Reading a TIFF stack failed.
    #[error("TIFF error: {0}")]
    Tiff(#[from] tiff::TiffError),
    /// JSON serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
