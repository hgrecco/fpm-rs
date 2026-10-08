//! Loading, discovery, caching, and subsetting for converted dataset bundles.
//!
//! The bundle format is specified in the repository's `dataset_spec.md`.
//! [`DatasetLoader`] is strictly local and never performs network access or
//! source-specific conversion. [`DatasetRegistry`] adds explicit registry
//! access, verified downloads, and managed caching for callers that opt in.

mod loader;
mod registry;
mod spectral;
mod subset;

pub use loader::{DATASET_FORMAT_VERSION, Dataset, DatasetLoader, DatasetManifest};
pub use registry::{
    DATASET_CACHE_DIR_ENV, DATASET_REGISTRY_URL_ENV, DATASET_REGISTRY_VERSION,
    DEFAULT_DATASET_REGISTRY_URL, DatasetArchive, DatasetCitation, DatasetLicense, DatasetListing,
    DatasetRegistry, DatasetRegistryDocument, DatasetRegistryEntry, DatasetSource, open_dataset,
};
pub use subset::{DatasetSubset, DatasetSubsetBuilder, FrameSelector, Rect};

pub use spectral::{
    SPECTRAL_DATASET_FORMAT_VERSION, SpectralDataset, SpectralDatasetChannel,
    SpectralDatasetManifest,
};
