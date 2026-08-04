//! Parquet tables, NPY arrays, manifests, and lazy reopening for result bundles.
//!
//! Write bundles through [`crate::reconstruction::ReconstructionResult::write_bundle`]
//! and reopen them with [`crate::tabular::parquet::read_bundle`].
//! [`crate::tabular::parquet::ResultBundle`] verifies artifact hashes and loads large
//! arrays lazily on first access.

mod bundle;
mod manifest;
mod npy;
mod options;
mod write;

pub use bundle::{
    BundleArrays, BundleArtifact, BundlePreviews, BundleTables, BundleVerificationResult,
    ResultBundle, read_bundle,
};
pub use manifest::{BUNDLE_FORMAT_VERSION, BundleExportOptions};
pub(crate) use options::ParquetWriteOptions;
pub(crate) use write::{sha256, write_parquet_file, write_result_bundle};
