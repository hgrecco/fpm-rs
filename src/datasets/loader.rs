use std::{
    collections::BTreeMap,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    Complex64, Result,
    array_serde::Array2Data,
    configuration::SimulationConfiguration,
    error::Error,
    measurements::{ImageSet, MeasurementSpec, MeasurementStack},
    reconstruction::ReconstructionProblem,
};

/// Current version of the language-neutral dataset bundle format.
pub const DATASET_FORMAT_VERSION: u32 = 1;

/// The `dataset.json` entry point defined by `dataset_spec.md`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifest {
    pub format_version: u32,
    pub measurement_manifest: PathBuf,
    pub configuration: PathBuf,
    /// Optional JSON-serialized `Array2<Complex64>` in reconstruction space.
    #[serde(default)]
    pub ground_truth_object: Option<PathBuf>,
    /// Optional JSON-serialized binary `Array2<u8>` in reconstruction space.
    #[serde(default)]
    pub valid_object_mask: Option<PathBuf>,
    #[serde(default)]
    pub provenance: BTreeMap<String, String>,
    #[serde(default)]
    pub measurement_units: Option<String>,
}

/// A validated dataset loaded from a bundle conforming to `dataset_spec.md`.
#[derive(Clone, Debug)]
pub struct Dataset {
    source_path: Option<PathBuf>,
    measurements: MeasurementStack,
    configuration: SimulationConfiguration,
    ground_truth_object: Option<Array2<Complex64>>,
    valid_object_mask: Option<Array2<u8>>,
    provenance: BTreeMap<String, String>,
    measurement_units: Option<String>,
}

impl Dataset {
    /// Constructs a dataset from already-loaded values.
    pub fn new(
        measurements: MeasurementStack,
        configuration: SimulationConfiguration,
    ) -> Result<Self> {
        measurements.validate()?;
        configuration.validate()?;
        ReconstructionProblem::new(
            measurements.clone(),
            configuration.compiled_models.reconstruction_model.clone(),
        )?;
        Ok(Self {
            source_path: None,
            measurements,
            configuration,
            ground_truth_object: None,
            valid_object_mask: None,
            provenance: BTreeMap::new(),
            measurement_units: None,
        })
    }

    pub fn measurements(&self) -> &MeasurementStack {
        &self.measurements
    }

    /// Root directory of the loaded bundle, or `None` for programmatic data.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    pub fn configuration(&self) -> &SimulationConfiguration {
        &self.configuration
    }

    pub fn ground_truth_object(&self) -> Option<&Array2<Complex64>> {
        self.ground_truth_object.as_ref()
    }

    pub fn valid_object_mask(&self) -> Option<&Array2<u8>> {
        self.valid_object_mask.as_ref()
    }

    pub fn provenance(&self) -> &BTreeMap<String, String> {
        &self.provenance
    }

    pub fn measurement_units(&self) -> Option<&str> {
        self.measurement_units.as_deref()
    }

    pub fn reconstruction_problem(&self) -> Result<ReconstructionProblem<MeasurementStack>> {
        ReconstructionProblem::new(
            self.measurements.clone(),
            self.configuration
                .compiled_models
                .reconstruction_model
                .clone(),
        )
    }

    pub fn subset(&self) -> super::DatasetSubsetBuilder<'_> {
        super::DatasetSubsetBuilder::new(self)
    }

    fn with_metadata(
        mut self,
        ground_truth_object: Option<Array2<Complex64>>,
        valid_object_mask: Option<Array2<u8>>,
        provenance: BTreeMap<String, String>,
        measurement_units: Option<String>,
    ) -> Result<Self> {
        validate_dataset_metadata(
            &self.configuration,
            ground_truth_object.as_ref(),
            valid_object_mask.as_ref(),
            &provenance,
            measurement_units.as_deref(),
        )?;
        self.ground_truth_object = ground_truth_object;
        self.valid_object_mask = valid_object_mask;
        self.provenance = provenance;
        self.measurement_units = measurement_units;
        Ok(self)
    }
}

/// Loads an already-converted dataset bundle from a local directory.
#[derive(Clone, Debug)]
pub struct DatasetLoader {
    root: PathBuf,
}

impl DatasetLoader {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        if !root.is_dir() {
            return Err(Error::Dataset(format!(
                "dataset path does not exist or is not a directory: {}",
                root.display()
            )));
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("dataset.json")
    }

    pub fn load(&self) -> Result<Dataset> {
        let manifest_path = self.manifest_path();
        let manifest: DatasetManifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
        if manifest.format_version != DATASET_FORMAT_VERSION {
            return Err(Error::Dataset(format!(
                "unsupported dataset format version {} in {}; expected {}",
                manifest.format_version,
                manifest_path.display(),
                DATASET_FORMAT_VERSION
            )));
        }
        validate_dataset_manifest_paths(&manifest)?;
        let measurement_manifest_path = self.root.join(&manifest.measurement_manifest);
        let measurement_spec = MeasurementSpec::load(&measurement_manifest_path)?;
        validate_measurement_paths(&measurement_spec)?;
        let measurement_base = measurement_manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."));
        let measurements =
            MeasurementStack::from_manifest_definition(measurement_spec, measurement_base)?;
        let configuration = SimulationConfiguration::load(self.root.join(&manifest.configuration))?;
        let ground_truth_object = manifest
            .ground_truth_object
            .as_ref()
            .map(|path| load_json_array(self.root.join(path), "ground-truth object"))
            .transpose()?;
        let valid_object_mask = manifest
            .valid_object_mask
            .as_ref()
            .map(|path| load_json_array(self.root.join(path), "valid-object mask"))
            .transpose()?;
        let mut dataset = Dataset::new(measurements, configuration)?;
        dataset.source_path = Some(self.root.clone());
        dataset.with_metadata(
            ground_truth_object,
            valid_object_mask,
            manifest.provenance,
            manifest.measurement_units,
        )
    }
}

fn validate_dataset_manifest_paths(manifest: &DatasetManifest) -> Result<()> {
    validate_contained_path("measurement manifest", &manifest.measurement_manifest)?;
    validate_contained_path("configuration", &manifest.configuration)?;
    if let Some(path) = &manifest.ground_truth_object {
        validate_contained_path("ground-truth object", path)?;
    }
    if let Some(path) = &manifest.valid_object_mask {
        validate_contained_path("valid-object mask", path)?;
    }
    Ok(())
}

fn validate_measurement_paths(spec: &MeasurementSpec) -> Result<()> {
    for frame in &spec.frames {
        validate_contained_path("measurement frame", &frame.path)?;
    }
    if let Some(path) = &spec.dark_frame {
        validate_contained_path("dark frame", path)?;
    }
    if let Some(path) = &spec.flat_field {
        validate_contained_path("flat field", path)?;
    }
    if let Some(images) = &spec.background {
        validate_image_set_paths("background", images)?;
    }
    if let Some(images) = &spec.mask {
        validate_image_set_paths("mask", images)?;
    }
    Ok(())
}

fn validate_image_set_paths(label: &str, images: &ImageSet) -> Result<()> {
    match images {
        ImageSet::Single(path) => validate_contained_path(label, path),
        ImageSet::PerFrame(paths) => paths
            .iter()
            .try_for_each(|path| validate_contained_path(label, path)),
    }
}

fn validate_contained_path(label: &str, path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(Error::Dataset(format!(
            "{label} must be a safe relative path contained by its manifest directory: {}",
            path.display()
        )));
    }
    Ok(())
}

fn load_json_array<T>(path: PathBuf, label: &str) -> Result<Array2<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let file = File::open(&path).map_err(|error| {
        Error::Dataset(format!(
            "failed to open {label} at {}: {error}",
            path.display()
        ))
    })?;
    let representation: Array2Data<T> =
        serde_json::from_reader(BufReader::new(file)).map_err(|error| {
            Error::Dataset(format!(
                "failed to parse {label} at {}: {error}",
                path.display()
            ))
        })?;
    representation
        .into_array()
        .map_err(|error| Error::Dataset(format!("invalid {label} at {}: {error}", path.display())))
}

fn validate_dataset_metadata(
    configuration: &SimulationConfiguration,
    ground_truth_object: Option<&Array2<Complex64>>,
    valid_object_mask: Option<&Array2<u8>>,
    provenance: &BTreeMap<String, String>,
    measurement_units: Option<&str>,
) -> Result<()> {
    if let Some(ground_truth) = ground_truth_object {
        if ground_truth.dim() != configuration.reconstruction_shape {
            return Err(Error::Dataset(format!(
                "ground-truth shape {:?} differs from reconstruction shape {:?}",
                ground_truth.dim(),
                configuration.reconstruction_shape
            )));
        }
        if ground_truth
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::Dataset(
                "ground-truth object contains non-finite values".into(),
            ));
        }
    }
    if let Some(mask) = valid_object_mask {
        if ground_truth_object.is_none() {
            return Err(Error::Dataset(
                "a valid-object mask requires a ground-truth object".into(),
            ));
        }
        if mask.dim() != configuration.reconstruction_shape {
            return Err(Error::Dataset(format!(
                "valid-object mask shape {:?} differs from reconstruction shape {:?}",
                mask.dim(),
                configuration.reconstruction_shape
            )));
        }
        if mask.iter().any(|&value| value > 1) || !mask.iter().any(|&value| value == 1) {
            return Err(Error::Dataset(
                "valid-object mask must contain only zero and one and select at least one pixel"
                    .into(),
            ));
        }
    }
    if provenance
        .iter()
        .any(|(key, value)| key.trim().is_empty() || value.trim().is_empty())
    {
        return Err(Error::Dataset(
            "dataset provenance keys and values must not be empty".into(),
        ));
    }
    if measurement_units.is_some_and(|units| units.trim().is_empty()) {
        return Err(Error::Dataset(
            "measurement units must not be empty when provided".into(),
        ));
    }
    Ok(())
}
