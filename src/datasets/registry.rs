use std::{
    collections::BTreeSet,
    env, fs,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

use super::{DATASET_FORMAT_VERSION, Dataset, DatasetLoader};

/// Registry document version understood by this release.
pub const DATASET_REGISTRY_VERSION: u32 = 1;
/// Default registry source used when no API or environment override is set.
pub const DEFAULT_DATASET_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/hgrecco/fpm-rs/main/dataset_registry.json";
/// Environment variable overriding [`DEFAULT_DATASET_REGISTRY_URL`].
pub const DATASET_REGISTRY_URL_ENV: &str = "FPM_RS_DATASET_REGISTRY_URL";
/// Environment variable overriding the platform dataset cache directory.
pub const DATASET_CACHE_DIR_ENV: &str = "FPM_RS_DATASET_CACHE_DIR";

const CACHE_MARKER: &str = ".fpm-rs-dataset-cache";
const CACHE_MARKER_CONTENTS: &[u8] = b"fpm-rs managed dataset cache v1\n";
const INSTALL_METADATA: &str = ".fpm-rs-install.json";
const MAX_REGISTRY_BYTES: usize = 16 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: u64 = 100_000;
const MAX_EXTRACTED_BYTES: u64 = 256 * 1024 * 1024 * 1024;

/// Immutable compressed bundle information from a registry entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetArchive {
    /// HTTPS or file URL for the immutable compressed tar archive.
    pub url: String,
    /// Lowercase hexadecimal SHA-256 digest of the archive bytes.
    pub sha256: String,
    /// Exact compressed archive size in bytes.
    pub size_bytes: u64,
}

/// License metadata from a registry entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetLicense {
    /// SPDX license identifier for the distributed dataset.
    pub spdx: String,
    /// Authoritative license text or record URL.
    pub url: String,
}

/// Preferred citation metadata from a registry entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetCitation {
    /// Publication DOI identifier recorded by the registry contract.
    pub doi: String,
    /// Complete human-readable bibliographic citation.
    pub text: String,
}

/// Original-source provenance from a registry entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSource {
    /// Authoritative original dataset or project URL.
    pub url: String,
    /// Description of source acquisition and external conversion provenance.
    pub description: String,
}

/// One current immutable dataset version advertised by a registry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRegistryEntry {
    /// Stable path-safe dataset identifier.
    pub id: String,
    /// Immutable path-safe dataset version.
    pub version: String,
    /// Short human-readable dataset title.
    pub title: String,
    /// Human-readable scientific and acquisition summary.
    pub description: String,
    /// Dataset bundle format version expected after extraction.
    pub format_version: u32,
    /// Immutable download location, digest, and byte size.
    pub archive: DatasetArchive,
    /// License identifier and authoritative URL.
    pub license: DatasetLicense,
    /// Preferred publication citation metadata.
    pub citation: DatasetCitation,
    /// Original-source provenance.
    pub source: DatasetSource,
    /// Search and filtering tags.
    pub tags: Vec<String>,
}

impl DatasetRegistryEntry {
    /// Validates all version-1 entry invariants.
    pub fn validate(&self) -> Result<()> {
        validate_component("dataset id", &self.id)?;
        validate_component("dataset version", &self.version)?;
        for (label, value) in [
            ("dataset title", self.title.as_str()),
            ("dataset description", self.description.as_str()),
            ("archive URL", self.archive.url.as_str()),
            ("license SPDX identifier", self.license.spdx.as_str()),
            ("license URL", self.license.url.as_str()),
            ("citation DOI", self.citation.doi.as_str()),
            ("citation text", self.citation.text.as_str()),
            ("source URL", self.source.url.as_str()),
            ("source description", self.source.description.as_str()),
        ] {
            validate_nonempty(label, value)?;
        }
        if self.format_version != DATASET_FORMAT_VERSION {
            return Err(Error::Dataset(format!(
                "dataset '{}' declares unsupported format version {}; expected {}",
                self.id, self.format_version, DATASET_FORMAT_VERSION
            )));
        }
        if self.archive.size_bytes == 0 {
            return Err(Error::Dataset(format!(
                "dataset '{}' archive size must be positive",
                self.id
            )));
        }
        if self.archive.sha256.len() != 64
            || !self
                .archive
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::Dataset(format!(
                "dataset '{}' SHA-256 must contain 64 hexadecimal characters",
                self.id
            )));
        }
        let archive_path = self
            .archive
            .url
            .split(['?', '#'])
            .next()
            .unwrap_or_default();
        if !archive_path.ends_with(".tar.zst") {
            return Err(Error::Dataset(format!(
                "dataset '{}' archive URL must identify a .tar.zst file",
                self.id
            )));
        }
        if self.tags.is_empty() {
            return Err(Error::Dataset(format!(
                "dataset '{}' must contain at least one tag",
                self.id
            )));
        }
        let mut tags = BTreeSet::new();
        for tag in &self.tags {
            validate_nonempty("dataset tag", tag)?;
            if !tags.insert(tag) {
                return Err(Error::Dataset(format!(
                    "dataset '{}' contains duplicate tag '{tag}'",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

/// Strict versioned dataset registry document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRegistryDocument {
    /// Registry schema version; must equal [`DATASET_REGISTRY_VERSION`].
    pub registry_version: u32,
    /// Current immutable dataset versions, with unique [`DatasetRegistryEntry::id`] values.
    pub datasets: Vec<DatasetRegistryEntry>,
}

impl DatasetRegistryDocument {
    /// Parses and validates a registry JSON document.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let document: Self = serde_json::from_slice(bytes)?;
        document.validate()?;
        Ok(document)
    }

    /// Validates the registry version, entry metadata, and ID uniqueness.
    pub fn validate(&self) -> Result<()> {
        if self.registry_version != DATASET_REGISTRY_VERSION {
            return Err(Error::Dataset(format!(
                "unsupported dataset registry version {}; expected {}",
                self.registry_version, DATASET_REGISTRY_VERSION
            )));
        }
        let mut identifiers = BTreeSet::new();
        for entry in &self.datasets {
            entry.validate()?;
            if !identifiers.insert(&entry.id) {
                return Err(Error::Dataset(format!(
                    "dataset registry contains duplicate id '{}'",
                    entry.id
                )));
            }
        }
        Ok(())
    }

    fn entry(&self, id: &str) -> Result<&DatasetRegistryEntry> {
        self.datasets
            .iter()
            .find(|entry| entry.id == id)
            .ok_or_else(|| Error::Dataset(format!("dataset registry has no entry '{id}'")))
    }
}

/// A registry entry annotated with its current managed-cache status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetListing {
    /// Validated registry metadata.
    pub entry: DatasetRegistryEntry,
    /// Whether the current version is present and valid in the managed cache.
    pub cached: bool,
    /// Installed version directory when `cached` is true.
    pub cache_path: Option<PathBuf>,
}

/// Registry client with verified downloads and a managed local cache.
#[derive(Clone, Debug)]
pub struct DatasetRegistry {
    registry_url: String,
    cache_dir: PathBuf,
}

impl DatasetRegistry {
    /// Resolves the registry source and cache directory from environment and
    /// platform defaults.
    pub fn from_defaults() -> Result<Self> {
        let registry_url = env::var(DATASET_REGISTRY_URL_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_DATASET_REGISTRY_URL.to_owned());
        let cache_dir = env::var_os(DATASET_CACHE_DIR_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                dirs::cache_dir().map(|directory| directory.join("fpm-rs").join("datasets"))
            })
            .ok_or_else(|| {
                Error::Dataset("could not determine the platform cache directory".into())
            })?;
        Self::new(registry_url, cache_dir)
    }

    /// Creates a registry using explicit source and cache values.
    pub fn new(registry_url: impl Into<String>, cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let registry_url = registry_url.into();
        validate_nonempty("dataset registry URL", &registry_url)?;
        let cache_dir = cache_dir.into();
        if cache_dir.as_os_str().is_empty() {
            return Err(Error::Dataset(
                "dataset cache directory must not be empty".into(),
            ));
        }
        Ok(Self {
            registry_url,
            cache_dir,
        })
    }

    /// Returns the configured registry URL, file URL, or filesystem path.
    pub fn registry_url(&self) -> &str {
        &self.registry_url
    }

    /// Returns the root of the managed cache.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Fetches the registry and reports whether each current version is cached.
    pub fn list(&self) -> Result<Vec<DatasetListing>> {
        let registry = self.load_registry()?;
        let _lock = self.lock_cache()?;
        registry
            .datasets
            .into_iter()
            .map(|entry| {
                let path = self.dataset_path(&entry);
                let cached = self.is_cached(&entry)?;
                Ok(DatasetListing {
                    entry,
                    cached,
                    cache_path: cached.then_some(path),
                })
            })
            .collect()
    }

    /// Downloads, verifies, validates, and caches the current version of `id`.
    ///
    /// An already-valid current installation is returned unchanged.
    pub fn download(&self, id: &str) -> Result<PathBuf> {
        validate_component("dataset id", id)?;
        let registry = self.load_registry()?;
        let entry = registry.entry(id)?.clone();
        self.install(&entry)
    }

    /// Downloads every registry entry sequentially and stops on the first error.
    pub fn download_all(&self) -> Result<Vec<PathBuf>> {
        let registry = self.load_registry()?;
        registry
            .datasets
            .iter()
            .map(|entry| self.install(entry))
            .collect()
    }

    /// Opens `id`, downloading it first when necessary.
    ///
    /// A corrupt current installation is removed and installed once more.
    pub fn open(&self, id: &str) -> Result<Dataset> {
        let path = self.download(id)?;
        match self.load_cached_dataset(&path) {
            Ok(dataset) => Ok(dataset),
            Err(first_error) => {
                let registry = self.load_registry()?;
                let entry = registry.entry(id)?.clone();
                {
                    let _lock = self.lock_cache()?;
                    if path.exists() {
                        self.validate_managed_tree(&path)?;
                        fs::remove_dir_all(&path)?;
                    }
                }
                let repaired = self.install(&entry)?;
                self.load_cached_dataset(&repaired).map_err(|second_error| {
                    Error::Dataset(format!(
                        "cached dataset '{id}' was invalid ({first_error}); reinstall also failed: {second_error}"
                    ))
                })
            }
        }
    }

    fn load_cached_dataset(&self, path: &Path) -> Result<Dataset> {
        let _lock = self.lock_cache()?;
        DatasetLoader::new(path)?.load()
    }

    /// Removes every cached version of `id`.
    pub fn clean(&self, id: &str) -> Result<bool> {
        validate_component("dataset id", id)?;
        let _lock = self.lock_cache()?;
        if !self.cache_dir.exists() {
            return Ok(false);
        }
        self.validate_cache_root()?;
        let path = self.cache_dir.join("datasets").join(id);
        if !path.exists() {
            return Ok(false);
        }
        self.validate_managed_tree(&path)?;
        fs::remove_dir_all(path)?;
        Ok(true)
    }

    /// Removes the complete marked managed cache and returns the version count.
    pub fn clean_all(&self) -> Result<usize> {
        let _lock = self.lock_cache()?;
        if !self.cache_dir.exists() {
            return Ok(0);
        }
        self.validate_cache_root()?;
        self.validate_managed_tree(&self.cache_dir)?;
        let count = count_cached_versions(&self.cache_dir.join("datasets"))?;
        fs::remove_dir_all(&self.cache_dir)?;
        Ok(count)
    }

    fn load_registry(&self) -> Result<DatasetRegistryDocument> {
        match read_source_limited(&self.registry_url, MAX_REGISTRY_BYTES as u64) {
            Ok(bytes) => {
                let registry = DatasetRegistryDocument::from_slice(&bytes)?;
                self.write_registry_snapshot(&bytes)?;
                Ok(registry)
            }
            Err(fetch_error) => match self.read_registry_snapshot() {
                Ok(bytes) => DatasetRegistryDocument::from_slice(&bytes),
                Err(snapshot_error) => Err(Error::Dataset(format!(
                    "failed to fetch registry '{}': {fetch_error}; no usable cached snapshot: {snapshot_error}",
                    self.registry_url
                ))),
            },
        }
    }

    fn install(&self, entry: &DatasetRegistryEntry) -> Result<PathBuf> {
        let _lock = self.lock_cache()?;
        self.ensure_cache_root()?;
        let destination = self.dataset_path(entry);
        if self.is_cached(entry)? {
            return Ok(destination);
        }
        if destination.exists() {
            self.validate_managed_tree(&destination)?;
            fs::remove_dir_all(&destination)?;
        }

        let partial_root = self.cache_dir.join("partial");
        fs::create_dir_all(&partial_root)?;
        let nonce = unique_nonce();
        let archive_path = partial_root.join(format!("{}-{nonce}.tar.zst", entry.id));
        let staging_path = partial_root.join(format!("{}-{nonce}.bundle", entry.id));
        fs::create_dir(&staging_path)?;

        let result = (|| {
            download_verified_archive(entry, &archive_path)?;
            extract_archive(&archive_path, &staging_path)?;
            if !staging_path.join("dataset.json").is_file() {
                return Err(Error::Dataset(format!(
                    "dataset '{}' archive must contain dataset.json at its root",
                    entry.id
                )));
            }
            DatasetLoader::new(&staging_path)?.load()?;
            let metadata = InstallMetadata {
                cache_format_version: 1,
                id: entry.id.clone(),
                version: entry.version.clone(),
                archive_sha256: entry.archive.sha256.to_ascii_lowercase(),
            };
            serde_json::to_writer_pretty(
                File::create(staging_path.join(INSTALL_METADATA))?,
                &metadata,
            )?;
            let parent = destination
                .parent()
                .ok_or_else(|| Error::Dataset("invalid destination for cached dataset".into()))?;
            fs::create_dir_all(parent)?;
            fs::rename(&staging_path, &destination)?;
            Ok(destination.clone())
        })();

        let _ = fs::remove_file(&archive_path);
        if staging_path.exists() {
            let _ = fs::remove_dir_all(&staging_path);
        }
        result
    }

    fn dataset_path(&self, entry: &DatasetRegistryEntry) -> PathBuf {
        self.cache_dir
            .join("datasets")
            .join(&entry.id)
            .join(&entry.version)
    }

    fn is_cached(&self, entry: &DatasetRegistryEntry) -> Result<bool> {
        let path = self.dataset_path(entry);
        if !path.is_dir() || !path.join("dataset.json").is_file() {
            return Ok(false);
        }
        self.validate_managed_tree(&path)?;
        let metadata_path = path.join(INSTALL_METADATA);
        if !metadata_path.is_file() {
            return Ok(false);
        }
        let Ok(metadata) =
            serde_json::from_reader::<_, InstallMetadata>(File::open(metadata_path)?)
        else {
            return Ok(false);
        };
        Ok(metadata.cache_format_version == 1
            && metadata.id == entry.id
            && metadata.version == entry.version
            && metadata.archive_sha256 == entry.archive.sha256.to_ascii_lowercase())
    }

    fn snapshot_path(&self) -> PathBuf {
        let digest = format!("{:x}", Sha256::digest(self.registry_url.as_bytes()));
        self.cache_dir
            .join("registries")
            .join(format!("{digest}.json"))
    }

    fn write_registry_snapshot(&self, bytes: &[u8]) -> Result<()> {
        let _lock = self.lock_cache()?;
        self.ensure_cache_root()?;
        let path = self.snapshot_path();
        let parent = path
            .parent()
            .ok_or_else(|| Error::Dataset("invalid registry snapshot path".into()))?;
        fs::create_dir_all(parent)?;
        let temporary = path.with_extension(format!("json.{}.partial", unique_nonce()));
        fs::write(&temporary, bytes)?;
        replace_file(&temporary, &path)?;
        Ok(())
    }

    fn read_registry_snapshot(&self) -> Result<Vec<u8>> {
        if !self.cache_dir.exists() {
            return Err(Error::Dataset("dataset cache does not exist".into()));
        }
        self.validate_cache_root()?;
        fs::read(self.snapshot_path()).map_err(Into::into)
    }

    fn ensure_cache_root(&self) -> Result<()> {
        if !self.cache_dir.exists() {
            fs::create_dir_all(&self.cache_dir)?;
        }
        let metadata = fs::symlink_metadata(&self.cache_dir)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Error::Dataset(format!(
                "dataset cache root must be a real directory: {}",
                self.cache_dir.display()
            )));
        }
        let marker = self.cache_dir.join(CACHE_MARKER);
        if marker.exists() {
            return validate_marker(&marker);
        }
        if fs::read_dir(&self.cache_dir)?.next().is_some() {
            return Err(Error::Dataset(format!(
                "refusing to manage non-empty unmarked cache directory {}",
                self.cache_dir.display()
            )));
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(marker)?;
        file.write_all(CACHE_MARKER_CONTENTS)?;
        file.sync_all()?;
        Ok(())
    }

    fn validate_cache_root(&self) -> Result<()> {
        let metadata = fs::symlink_metadata(&self.cache_dir)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Error::Dataset(format!(
                "dataset cache root must be a real directory: {}",
                self.cache_dir.display()
            )));
        }
        validate_marker(&self.cache_dir.join(CACHE_MARKER))
    }

    fn validate_managed_tree(&self, path: &Path) -> Result<()> {
        let relative = path.strip_prefix(&self.cache_dir).map_err(|_| {
            Error::Dataset(format!(
                "managed dataset path escapes cache root: {}",
                path.display()
            ))
        })?;
        if relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(Error::Dataset(format!(
                "managed dataset path is unsafe: {}",
                path.display()
            )));
        }
        reject_symlinks(path)
    }

    fn lock_cache(&self) -> Result<File> {
        let parent = self.cache_dir.parent().ok_or_else(|| {
            Error::Dataset("dataset cache root must have a parent directory".into())
        })?;
        fs::create_dir_all(parent)?;
        let name = self
            .cache_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("datasets");
        let path = parent.join(format!(".{name}.lock"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        FileExt::lock_exclusive(&file)?;
        Ok(file)
    }
}

/// Opens a registry dataset using environment and platform defaults.
pub fn open_dataset(id: &str) -> Result<Dataset> {
    DatasetRegistry::from_defaults()?.open(id)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InstallMetadata {
    cache_format_version: u32,
    id: String,
    version: String,
    archive_sha256: String,
}

fn download_verified_archive(entry: &DatasetRegistryEntry, destination: &Path) -> Result<()> {
    let mut reader = open_source_reader(&entry.archive.url)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| Error::Dataset("downloaded archive size overflowed".into()))?;
        if total > entry.archive.size_bytes {
            return Err(Error::Dataset(format!(
                "dataset '{}' archive exceeds declared size {} bytes",
                entry.id, entry.archive.size_bytes
            )));
        }
        digest.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
    }
    output.sync_all()?;
    if total != entry.archive.size_bytes {
        return Err(Error::Dataset(format!(
            "dataset '{}' archive size mismatch: expected {}, got {total}",
            entry.id, entry.archive.size_bytes
        )));
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != entry.archive.sha256.to_ascii_lowercase() {
        return Err(Error::Dataset(format!(
            "dataset '{}' SHA-256 mismatch: expected {}, got {actual}",
            entry.id, entry.archive.sha256
        )));
    }
    Ok(())
}

fn extract_archive(archive_path: &Path, destination: &Path) -> Result<()> {
    let decoder = zstd::Decoder::new(File::open(archive_path)?)?;
    let mut archive = tar::Archive::new(decoder);
    let mut paths = BTreeSet::new();
    let mut remaining_entries = MAX_ARCHIVE_ENTRIES;
    let mut remaining_bytes = MAX_EXTRACTED_BYTES;
    for entry in archive.entries()? {
        let mut entry = entry?;
        remaining_entries = remaining_entries.checked_sub(1).ok_or_else(|| {
            Error::Dataset("dataset archive exceeds the extraction entry limit".into())
        })?;
        let relative = entry.path()?.into_owned();
        validate_archive_path(&relative)?;
        if !paths.insert(relative.clone()) {
            return Err(Error::Dataset(format!(
                "dataset archive contains duplicate path {}",
                relative.display()
            )));
        }
        let entry_type = entry.header().entry_type();
        let target = destination.join(&relative);
        if entry_type.is_dir() {
            fs::create_dir_all(target)?;
            continue;
        }
        if !entry_type.is_file() {
            return Err(Error::Dataset(format!(
                "dataset archive entry is not a regular file or directory: {}",
                relative.display()
            )));
        }
        let size = entry.size();
        remaining_bytes = remaining_bytes.checked_sub(size).ok_or_else(|| {
            Error::Dataset("dataset archive exceeds the extraction byte limit".into())
        })?;
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)?;
        let copied = std::io::copy(&mut entry, &mut output)?;
        if copied != size {
            return Err(Error::Dataset(format!(
                "dataset archive entry size mismatch for {}",
                relative.display()
            )));
        }
    }
    Ok(())
}

fn read_source_limited(source: &str, limit: u64) -> Result<Vec<u8>> {
    let mut reader = open_source_reader(source)?;
    let mut bytes = Vec::new();
    let copied = std::io::copy(&mut reader.by_ref().take(limit + 1), &mut bytes)?;
    if copied > limit {
        return Err(Error::Dataset(format!(
            "source exceeds the {limit}-byte limit: {source}"
        )));
    }
    Ok(bytes)
}

fn open_source_reader(source: &str) -> Result<Box<dyn Read>> {
    if source.starts_with("http://") || source.starts_with("https://") {
        let response = ureq::get(source)
            .call()
            .map_err(|error| Error::Dataset(format!("request failed for {source}: {error}")))?;
        return Ok(Box::new(response.into_body().into_reader()));
    }
    let path = source.strip_prefix("file://").unwrap_or(source);
    Ok(Box::new(File::open(path).map_err(|error| {
        Error::Dataset(format!("failed to open source {source}: {error}"))
    })?))
}

fn validate_archive_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::Dataset(format!(
            "dataset archive entry path is unsafe: {}",
            path.display()
        )));
    }
    Ok(())
}

fn validate_component(label: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(Error::Dataset(format!(
            "{label} must contain only letters, digits, '.', '_', or '-'"
        )));
    }
    Ok(())
}

fn validate_nonempty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::Dataset(format!("{label} must not be empty")));
    }
    Ok(())
}

fn validate_marker(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        Error::Dataset(format!(
            "dataset cache ownership marker is missing at {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(Error::Dataset(format!(
            "dataset cache ownership marker must be a regular file: {}",
            path.display()
        )));
    }
    if fs::read(path)? != CACHE_MARKER_CONTENTS {
        return Err(Error::Dataset(format!(
            "dataset cache ownership marker is invalid: {}",
            path.display()
        )));
    }
    Ok(())
}

fn reject_symlinks(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(Error::Dataset(format!(
            "managed cache path contains a symlink: {}",
            path.display()
        )));
    }
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            reject_symlinks(&entry?.path())?;
        }
    }
    Ok(())
}

fn count_cached_versions(root: &Path) -> Result<usize> {
    if !root.is_dir() {
        return Ok(0);
    }
    let mut count = 0;
    for dataset in fs::read_dir(root)? {
        let dataset = dataset?;
        if dataset.file_type()?.is_dir() {
            count += fs::read_dir(dataset.path())?
                .filter_map(std::result::Result::ok)
                .filter_map(|entry| entry.file_type().ok())
                .filter(|kind| kind.is_dir())
                .count();
        }
    }
    Ok(count)
}

fn unique_nonce() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!("{}-{nanos}", std::process::id())
}

fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(error)
            if destination.exists()
                && matches!(
                    error.kind(),
                    std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied
                ) =>
        {
            fs::remove_file(destination)?;
            fs::rename(source, destination)?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_paths_reject_escape_and_non_normal_components() {
        for path in [
            "",
            ".",
            "./dataset.json",
            "../dataset.json",
            "/dataset.json",
        ] {
            assert!(
                validate_archive_path(Path::new(path)).is_err(),
                "accepted {path:?}"
            );
        }
        assert!(validate_archive_path(Path::new("frames/frame.tiff")).is_ok());
    }

    #[test]
    fn archive_extraction_rejects_links_and_duplicate_paths() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let link_archive = temporary.path().join("link.tar.zst");
        {
            let encoder = zstd::Encoder::new(File::create(&link_archive)?, 1)?;
            let mut archive = tar::Builder::new(encoder);
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::Symlink);
            header.set_size(0);
            header.set_mode(0o777);
            header.set_link_name("outside")?;
            header.set_cksum();
            archive.append_data(&mut header, "link", std::io::empty())?;
            archive.into_inner()?.finish()?;
        }
        let destination = temporary.path().join("link-output");
        fs::create_dir(&destination)?;
        assert!(
            extract_archive(&link_archive, &destination)
                .unwrap_err()
                .to_string()
                .contains("not a regular file or directory")
        );

        let duplicate_archive = temporary.path().join("duplicate.tar.zst");
        {
            let encoder = zstd::Encoder::new(File::create(&duplicate_archive)?, 1)?;
            let mut archive = tar::Builder::new(encoder);
            for payload in [b"first".as_slice(), b"second".as_slice()] {
                let mut header = tar::Header::new_gnu();
                header.set_size(payload.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                archive.append_data(&mut header, "duplicate", payload)?;
            }
            archive.into_inner()?.finish()?;
        }
        let destination = temporary.path().join("duplicate-output");
        fs::create_dir(&destination)?;
        assert!(
            extract_archive(&duplicate_archive, &destination)
                .unwrap_err()
                .to_string()
                .contains("duplicate path")
        );
        Ok(())
    }
}
