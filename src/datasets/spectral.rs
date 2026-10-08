//! Explicit version-two narrowband datasets; source-specific conversion stays external.
use super::DatasetLoader;
use crate::{
    Complex64, Error, Result,
    experiment::SpectralAcquisitionPlan,
    measurements::{ImageSet, MeasurementSpec, MeasurementStack},
    model::{ImagePlaneModel, ObjectCoupling, SpectralImagePlaneModel, SpectralModelChannel},
    reconstruction::SpectralReconstructionProblem,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

/// Version of the explicit spectral dataset profile; ordinary bundles remain version one.
pub const SPECTRAL_DATASET_FORMAT_VERSION: u32 = 2;
/// Converter-supplied channel record in stable solver order.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralDatasetChannel {
    /// Nonempty unique channel ID.
    pub channel_id: String,
    /// Explicit positive distinct vacuum wavelength in metres; must match the kernel.
    pub wavelength_vacuum_m: f64,
    /// Safe relative path to JSON-serialized compiled `ImagePlaneModel`.
    pub compiled_model: PathBuf,
    /// Required nonempty description of spectral response, weights and calibration origins.
    pub response_provenance: String,
    /// Optional common-grid complex truth using height/width/data JSON array encoding.
    #[serde(default)]
    pub ground_truth_object: Option<PathBuf>,
    /// Optional common-grid binary mask using height/width/data JSON array encoding.
    #[serde(default)]
    pub valid_object_mask: Option<PathBuf>,
}
/// `dataset.json` for the version-two spectral profile, never inferred from RGB images.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralDatasetManifest {
    /// Must equal [`SPECTRAL_DATASET_FORMAT_VERSION`].
    pub format_version: u32,
    /// Safe relative path to the ordinary grayscale measurement manifest.
    pub measurement_manifest: PathBuf,
    /// Stable ordered wavelength records with externally compiled kernels.
    pub channels: Vec<SpectralDatasetChannel>,
    /// Explicit complex field reuse rule.
    pub object_coupling: ObjectCoupling,
    /// Sparse physical exposure order, including gains/background and channel-local references.
    pub acquisition: SpectralAcquisitionPlan,
    /// Converter provenance metadata.
    #[serde(default)]
    pub provenance: BTreeMap<String, String>,
    /// Optional semantic detector units, such as camera counts.
    #[serde(default)]
    pub measurement_units: Option<String>,
}
/// Resident spectral measurements, compiled model, converter metadata and optional per-channel truth.
#[derive(Clone, Debug)]
pub struct SpectralDataset {
    source_path: PathBuf,
    measurements: MeasurementStack,
    model: SpectralImagePlaneModel,
    manifest: SpectralDatasetManifest,
    truth: Vec<Option<Array2<Complex64>>>,
    masks: Vec<Option<Array2<u8>>>,
}
impl SpectralDataset {
    /// Borrows the local bundle directory.
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }
    /// Borrows resident scalar detector data and masks/weights.
    pub fn measurements(&self) -> &MeasurementStack {
        &self.measurements
    }
    /// Borrows the ordered compiled spectral model.
    pub fn model(&self) -> &SpectralImagePlaneModel {
        &self.model
    }
    /// Borrows converter metadata and response provenance in channel order.
    pub fn manifest(&self) -> &SpectralDatasetManifest {
        &self.manifest
    }
    /// Borrows optional per-channel common-grid ground truths.
    pub fn ground_truth_objects(&self) -> &[Option<Array2<Complex64>>] {
        &self.truth
    }
    /// Borrows optional per-channel binary truth-validity masks.
    pub fn valid_object_masks(&self) -> &[Option<Array2<u8>>] {
        &self.masks
    }
    /// Copies resident data/model into an independently owned validated spectral problem.
    pub fn reconstruction_problem(
        &self,
    ) -> Result<SpectralReconstructionProblem<MeasurementStack>> {
        SpectralReconstructionProblem::new(self.measurements.clone(), self.model.clone())
    }
}
impl DatasetLoader {
    /// Loads a version-two spectral profile entirely offline.
    /// Requires explicit wavelength/channel metadata and response provenance; validates
    /// common grids, sparse references, scalar frames, truth/masks and safe contained paths.
    /// Ordinary version-one data must use `load` instead. No registration or RGB demixing occurs.
    ///
    /// # Example
    /// ```no_run
    /// use fpm_rs::{Result, datasets::DatasetLoader,
    ///     algorithms::{SpectralAlternatingProjection, SpectralReconstructionAlgorithm}};
    /// fn main() -> Result<()> {
    ///     let dataset = DatasetLoader::new("converted-spectral")?.load_spectral()?;
    ///     let result = SpectralAlternatingProjection::default().run(&dataset.reconstruction_problem()?)?;
    ///     assert_eq!(result.channels.len(), dataset.model().channels().len());
    ///     Ok(())
    /// }
    /// ```
    pub fn load_spectral(&self) -> Result<SpectralDataset> {
        let root = self.root();
        let manifest_path = contained(root, Path::new("dataset.json"))?;
        let manifest: SpectralDatasetManifest =
            serde_json::from_reader(BufReader::new(File::open(manifest_path)?))?;
        if manifest.format_version != SPECTRAL_DATASET_FORMAT_VERSION {
            return Err(Error::Dataset(
                "load_spectral requires dataset format_version 2; use load for version 1".into(),
            ));
        }
        if manifest
            .measurement_units
            .as_ref()
            .is_some_and(|v| v.trim().is_empty())
            || manifest
                .provenance
                .iter()
                .any(|(k, v)| k.trim().is_empty() || v.trim().is_empty())
        {
            return Err(Error::Dataset(
                "units and provenance entries must be nonempty".into(),
            ));
        }
        let measurement_path = contained(root, &manifest.measurement_manifest)?;
        let spec = MeasurementSpec::load(&measurement_path)?;
        let base = measurement_path.parent().unwrap_or(root);
        validate_measurement_files(base, &spec)?;
        let measurements = MeasurementStack::from_manifest_definition(spec, base)?;
        let mut channels = Vec::new();
        let mut truth = Vec::new();
        let mut masks = Vec::new();
        for channel in &manifest.channels {
            if channel.response_provenance.trim().is_empty() {
                return Err(Error::Dataset(format!(
                    "channel '{}' requires response_provenance",
                    channel.channel_id
                )));
            }
            let model: ImagePlaneModel = serde_json::from_reader(BufReader::new(File::open(
                contained(root, &channel.compiled_model)?,
            )?))?;
            if model.sampling().wavelength != Some(channel.wavelength_vacuum_m) {
                return Err(Error::Dataset(format!(
                    "channel '{}' wavelength differs from compiled kernel",
                    channel.channel_id
                )));
            }
            if channel.valid_object_mask.is_some() && channel.ground_truth_object.is_none() {
                return Err(Error::Dataset(
                    "spectral valid_object_mask requires ground_truth_object".into(),
                ));
            }
            let object = channel
                .ground_truth_object
                .as_ref()
                .map(|p| super::loader::load_json_array(contained(root, p)?, "spectral truth"))
                .transpose()?;
            let mask = channel
                .valid_object_mask
                .as_ref()
                .map(|p| super::loader::load_json_array(contained(root, p)?, "spectral truth mask"))
                .transpose()?;
            if object.as_ref().is_some_and(|a: &Array2<Complex64>| {
                a.dim() != model.reconstruction_shape()
                    || a.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
            }) || mask.as_ref().is_some_and(|a: &Array2<u8>| {
                a.dim() != model.reconstruction_shape() || a.iter().any(|&v| v > 1)
            }) {
                return Err(Error::Dataset(format!(
                    "channel '{}' truth or mask has invalid shape/values",
                    channel.channel_id
                )));
            }
            channels.push(SpectralModelChannel {
                channel_id: channel.channel_id.clone(),
                model,
            });
            truth.push(object);
            masks.push(mask);
        }
        let model = SpectralImagePlaneModel::from_compiled_channels(
            channels,
            manifest.acquisition.clone(),
            manifest.object_coupling,
        )?;
        SpectralReconstructionProblem::new(&measurements, model.clone())?;
        Ok(SpectralDataset {
            source_path: root.to_owned(),
            measurements,
            model,
            manifest,
            truth,
            masks,
        })
    }
    /// Reads the profile version without interpreting the rest of the manifest.
    pub fn format_version(&self) -> Result<u32> {
        let value: serde_json::Value =
            serde_json::from_reader(BufReader::new(File::open(self.manifest_path())?))?;
        value
            .get("format_version")
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| Error::Dataset("dataset.json requires an integer format_version".into()))
    }
    pub(super) fn validate_profile(&self, expected: u32) -> Result<()> {
        if self.format_version()? != expected {
            return Err(Error::Dataset(
                "registry format_version differs from dataset.json".into(),
            ));
        }
        match expected {
            1 => {
                self.load()?;
            }
            2 => {
                self.load_spectral()?;
            }
            _ => return Err(Error::Dataset("unsupported dataset profile".into())),
        };
        Ok(())
    }
}
fn contained(root: &Path, path: &Path) -> Result<PathBuf> {
    super::loader::validate_contained_path("spectral dataset artifact", path)?;
    let resolved = root.join(path).canonicalize()?;
    if !resolved.starts_with(root.canonicalize()?) {
        return Err(Error::Dataset(format!(
            "spectral artifact escapes its manifest directory: {}",
            path.display()
        )));
    }
    Ok(resolved)
}
fn validate_measurement_files(base: &Path, spec: &MeasurementSpec) -> Result<()> {
    for frame in &spec.frames {
        contained(base, &frame.path)?;
    }
    for path in [&spec.dark_frame, &spec.flat_field].into_iter().flatten() {
        contained(base, path)?;
    }
    for set in [&spec.background, &spec.mask].into_iter().flatten() {
        match set {
            ImageSet::Single(p) => {
                contained(base, p)?;
            }
            ImageSet::PerFrame(paths) => {
                for p in paths {
                    contained(base, p)?;
                }
            }
        }
    }
    Ok(())
}
