//! Loading, discovery, caching, and subsetting for converted dataset bundles.
//!
//! The bundle format is specified in the repository's `dataset_spec.md`.
//! [`DatasetLoader`] is strictly local and never performs network access or
//! source-specific conversion. [`DatasetRegistry`] adds explicit registry
//! access, verified downloads, and managed caching for callers that opt in.

mod loader;
mod subset;

pub use loader::{DATASET_FORMAT_VERSION, Dataset, DatasetLoader, DatasetManifest};
pub use subset::{DatasetSubset, DatasetSubsetBuilder, FrameSelector, Rect};
