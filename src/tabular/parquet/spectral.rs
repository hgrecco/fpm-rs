//! Separately tagged spectral result bundles with lossless per-channel arrays.
use super::{
    BundleArtifact, BundleExportOptions, BundleVerificationResult,
    manifest::ManifestArtifact,
    npy,
    write::{sha256, unique_paths, validate_run_id, write_json_atomic},
};
use crate::{
    Error, Result,
    algorithms::MultiWavelengthSolverResult,
    model::{ObjectCoupling, Pupil},
    reconstruction::{
        MultiWavelengthReconstructionResult, OpticalPathDifferenceResult, ReconstructionTrace,
        RuntimeInfo, SpectralChannelResult, SpectralReconstructionCheckpoint,
        SpectralReconstructionResult,
        spectral_checkpoint::{StoredOpd, StoredSolver},
    },
};
use ndarray::Array2;
use polars::prelude::{DataFrame, KeyValueMetadata, ParquetReader, ParquetWriter, SerReader};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::BufReader,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Version of the separately tagged spectral result-bundle manifest.
pub const SPECTRAL_BUNDLE_FORMAT_VERSION: u32 = 1;
const KIND: &str = "fpm_rs.spectral_result_bundle";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Channel {
    channel_id: String,
    wavelength_vacuum_m: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    kind: String,
    bundle_format_version: u32,
    run_id: String,
    label: Option<String>,
    crate_version: String,
    channels: Vec<Channel>,
    object_coupling: ObjectCoupling,
    image_shape: (usize, usize),
    reconstruction_shape: (usize, usize),
    runtime: RuntimeInfo,
    trace: ReconstructionTrace,
    has_joint_opd: bool,
    has_unwrapped_opd: bool,
    artifacts: Vec<ManifestArtifact>,
}
/// Reopened spectral bundle. Manifest metadata is eager; arrays/checkpoint are verified
/// when requested. `result` caches an immutable owned result shared through `Arc`.
/// Ordinary bundles must use `ResultBundle`. Failed exports keep an `.inprogress`
/// directory and never publish a complete manifest at the requested destination.
///
/// # Example
/// ```no_run
/// use fpm_rs::{Result, reconstruction::SpectralReconstructionResult,
///     tabular::parquet::{BundleExportOptions, read_spectral_bundle}};
/// fn persist(result: &SpectralReconstructionResult) -> Result<()> {
///     let bundle = result.write_bundle("spectral-result", BundleExportOptions::default())?;
///     let reopened = read_spectral_bundle(bundle.path())?;
///     reopened.verify()?;
///     assert_eq!(reopened.result()?.channels.len(), result.channels.len());
///     Ok(())
/// }
/// ```

#[derive(Debug)]
pub struct SpectralResultBundle {
    root: PathBuf,
    manifest: Manifest,
    artifacts: BTreeMap<String, BundleArtifact>,
    cache: Mutex<Option<Arc<SpectralReconstructionResult>>>,
}
/// Opens a spectral manifest without loading scientific arrays; rejects ordinary bundles.
pub fn read_spectral_bundle(path: impl AsRef<Path>) -> Result<SpectralResultBundle> {
    SpectralResultBundle::read(path)
}
impl SpectralResultBundle {
    /// Validates kind/version, unique safe artifact paths, shapes, metadata and required roles.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().canonicalize()?;
        let manifest: Manifest = serde_json::from_reader(BufReader::new(File::open(safe_file(
            &root,
            Path::new("manifest.json"),
        )?)?))?;
        if manifest.kind != KIND || manifest.bundle_format_version != SPECTRAL_BUNDLE_FORMAT_VERSION
        {
            return Err(invalid("unsupported spectral bundle kind/version"));
        }
        validate_run_id(&manifest.run_id)?;
        crate::array_layout::checked_len_2d(manifest.image_shape)?;
        crate::array_layout::checked_len_2d(manifest.reconstruction_shape)?;
        if manifest.channels.is_empty()
            || manifest.image_shape.0 == 0
            || manifest.image_shape.1 == 0
            || manifest.reconstruction_shape.0 == 0
            || manifest.reconstruction_shape.1 == 0
        {
            return Err(invalid("empty channel list or grid"));
        }
        let mut ids = BTreeSet::new();
        let mut wavelengths = Vec::new();
        for c in &manifest.channels {
            if c.channel_id.trim().is_empty()
                || !ids.insert(&c.channel_id)
                || !c.wavelength_vacuum_m.is_finite()
                || c.wavelength_vacuum_m <= 0.0
                || wavelengths.contains(&c.wavelength_vacuum_m)
            {
                return Err(invalid("invalid ordered channel metadata"));
            }
            wavelengths.push(c.wavelength_vacuum_m);
        }
        let expected = expected_roles(&manifest);
        let mut artifacts = BTreeMap::new();
        let mut paths = BTreeSet::new();
        for a in &manifest.artifacts {
            let path = safe_file(&root, &a.relative_path)?;
            let descriptor = expected
                .get(&a.role)
                .ok_or_else(|| invalid("unknown artifact role"))?;
            if artifacts.contains_key(&a.role)
                || !paths.insert(path.clone())
                || a.dtype != descriptor.0
                || a.shape != descriptor.1
                || a.media_type != descriptor.2
                || a.sha256.len() != 64
                || !a
                    .sha256
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            {
                return Err(invalid(
                    "duplicate artifact or invalid dtype/shape/hash/media type",
                ));
            }
            artifacts.insert(
                a.role.clone(),
                BundleArtifact {
                    role: a.role.clone(),
                    path,
                    media_type: a.media_type.clone(),
                    byte_size: a.byte_size,
                    sha256: a.sha256.clone(),
                    dtype: a.dtype.clone(),
                    shape: a.shape.clone(),
                },
            );
        }
        if artifacts.len() != expected.len() {
            return Err(invalid("required spectral artifacts missing"));
        }
        Ok(Self {
            root,
            manifest,
            artifacts,
            cache: Mutex::new(None),
        })
    }
    /// Borrows the canonical published bundle directory.
    pub fn path(&self) -> &Path {
        &self.root
    }
    /// Borrows the run identity preserved in manifest and tables.
    pub fn run_id(&self) -> &str {
        &self.manifest.run_id
    }
    /// Borrows an optional caller label.
    pub fn label(&self) -> Option<&str> {
        self.manifest.label.as_deref()
    }
    /// Borrows verified manifest descriptors, sorted by semantic role.
    pub fn artifacts(&self) -> &BTreeMap<String, BundleArtifact> {
        &self.artifacts
    }
    fn artifact(&self, role: &str) -> Result<&BundleArtifact> {
        let a = self
            .artifacts
            .get(role)
            .ok_or_else(|| invalid("missing artifact"))?;
        // Recheck containment on every access; a replaced symlink cannot escape the root.
        safe_file(
            &self.root,
            a.path
                .strip_prefix(&self.root)
                .map_err(|_| invalid("artifact outside root"))?,
        )?;
        if fs::metadata(&a.path)?.len() != a.byte_size || sha256(&a.path)? != a.sha256 {
            return Err(invalid(format!("artifact size/hash mismatch: {role}")));
        }
        Ok(a)
    }
    /// Loads the lossless resumable state, checking hash and exact manifest metadata.
    pub fn checkpoint(&self) -> Result<SpectralReconstructionCheckpoint> {
        let checkpoint =
            SpectralReconstructionCheckpoint::load(&self.artifact("state.checkpoint")?.path)?;
        let m = checkpoint.model();
        if m.image_shape() != self.manifest.image_shape
            || m.reconstruction_shape() != self.manifest.reconstruction_shape
            || m.object_coupling() != self.manifest.object_coupling
            || m.channels().len() != self.manifest.channels.len()
            || m.channels()
                .iter()
                .zip(&self.manifest.channels)
                .any(|(a, b)| {
                    a.channel_id != b.channel_id
                        || a.model.sampling().wavelength != Some(b.wavelength_vacuum_m)
                })
            || checkpoint.completed_iterations() != self.manifest.runtime.completed_iterations
            || checkpoint.algorithm() != self.manifest.runtime.algorithm
            || serde_json::to_value(checkpoint.trace())?
                != serde_json::to_value(&self.manifest.trace)?
            || checkpoint.elapsed_seconds != self.manifest.runtime.elapsed_seconds
            || matches!(checkpoint.state, StoredSolver::JointOpd { .. })
                != self.manifest.has_joint_opd
        {
            return Err(invalid("checkpoint and manifest disagree"));
        }
        Ok(checkpoint)
    }
    /// Loads per-channel NPY fields, amplitudes, phases, spectra and fixed pupils.
    /// Returned arrays preserve saved binary64 values and are cached together.
    pub fn result(&self) -> Result<Arc<SpectralReconstructionResult>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| invalid("result cache poisoned"))?;
        if let Some(result) = &*cache {
            return Ok(result.clone());
        }
        let checkpoint = self.checkpoint()?;
        let mut channels = Vec::new();
        for (i, c) in self.manifest.channels.iter().enumerate() {
            let role = |name: &str| format!("channels.{i}.{name}");
            let complex = |name: &str, shape| {
                npy::read_complex2(&self.artifact(&role(name))?.path, &role(name), shape)
            };
            let real = |name: &str| -> Result<Array2<f64>> {
                Ok(Array2::from_shape_vec(
                    self.manifest.reconstruction_shape,
                    npy::read_f64(
                        &self.artifact(&role(name))?.path,
                        &role(name),
                        &[
                            self.manifest.reconstruction_shape.0,
                            self.manifest.reconstruction_shape.1,
                        ],
                    )?,
                )?)
            };
            let object = complex("object", self.manifest.reconstruction_shape)?;
            let amplitude = real("amplitude")?;
            let phase = real("phase")?;
            let object_spectrum = complex("object_spectrum", self.manifest.reconstruction_shape)?;
            let pupil = Pupil::new(
                complex("pupil", self.manifest.image_shape)?,
                npy::read_u8_2(
                    &self.artifact(&role("pupil_support"))?.path,
                    &role("pupil_support"),
                    self.manifest.image_shape,
                )?,
            )?;
            channels.push(SpectralChannelResult {
                channel_id: c.channel_id.clone(),
                wavelength_vacuum_m: c.wavelength_vacuum_m,
                object,
                amplitude,
                phase,
                object_spectrum,
                pupil,
            });
        }
        let result = Arc::new(SpectralReconstructionResult {
            channels,
            object_coupling: self.manifest.object_coupling,
            trace: self.manifest.trace.clone(),
            runtime: self.manifest.runtime.clone(),
            checkpoint: if self.manifest.has_joint_opd {
                None
            } else {
                Some(checkpoint.clone())
            },
        });
        validate_result(&result, &checkpoint)?;
        *cache = Some(result.clone());
        Ok(result)
    }
    /// Restores the authoritative joint OPD result and initialization records, or None for AP.
    pub fn joint_result(&self) -> Result<Option<MultiWavelengthSolverResult>> {
        if !self.manifest.has_joint_opd {
            return Ok(None);
        }
        let checkpoint = self.checkpoint()?;
        let opd_m = Array2::from_shape_vec(
            self.manifest.reconstruction_shape,
            npy::read_f64(
                &self.artifact("arrays.opd_m")?.path,
                "arrays.opd_m",
                &[
                    self.manifest.reconstruction_shape.0,
                    self.manifest.reconstruction_shape.1,
                ],
            )?,
        )?;
        let StoredSolver::JointOpd {
            opd,
            initialization_opd,
            initialization_trace,
            ..
        } = &checkpoint.state
        else {
            return Err(invalid("joint state missing"));
        };
        if opd.data != opd_m.iter().copied().collect::<Vec<_>>() {
            return Err(invalid("OPD array differs from checkpoint"));
        }
        Ok(Some(MultiWavelengthSolverResult {
            spectral: (*self.result()?).clone(),
            opd_m,
            initialization_opd: initialization_opd
                .clone()
                .map(|o| (*o).into_result())
                .transpose()?,
            initialization_trace: initialization_trace.clone(),
            checkpoint,
        }))
    }
    /// Restores optional synthetic-wavelength diagnostic arrays, including invalid NaN pixels.
    pub fn unwrapped_opd(&self) -> Result<Option<OpticalPathDifferenceResult>> {
        if !self.manifest.has_unwrapped_opd {
            return Ok(None);
        }
        let stored: StoredOpd = serde_json::from_reader(BufReader::new(File::open(
            &self.artifact("state.unwrapped_opd")?.path,
        )?))?;
        let opd = stored.into_result()?;
        if opd.opd_m.dim() != self.manifest.reconstruction_shape
            || opd.wavelengths_vacuum_m
                != self
                    .manifest
                    .channels
                    .iter()
                    .map(|c| c.wavelength_vacuum_m)
                    .collect::<Vec<_>>()
        {
            return Err(invalid("unwrapped OPD and channels disagree"));
        }
        Ok(Some(opd))
    }
    /// Checks every hash, numerical state and table against authoritative manifest records.
    pub fn verify(&self) -> Result<BundleVerificationResult> {
        let mut bytes = 0_u64;
        for role in self.artifacts.keys() {
            bytes = bytes
                .checked_add(self.artifact(role)?.byte_size)
                .ok_or_else(|| invalid("artifact byte count overflow"))?;
        }
        self.result()?;
        self.joint_result()?;
        self.unwrapped_opd()?;
        for (role, expected) in [
            (
                "tables.history",
                crate::tabular::history_dataframe(self.run_id(), &self.manifest.trace)?,
            ),
            (
                "tables.algorithm_metrics",
                crate::tabular::algorithm_metrics_dataframe(self.run_id(), &self.manifest.trace)?,
            ),
        ] {
            let actual = ParquetReader::new(File::open(&self.artifact(role)?.path)?).finish()?;
            if !actual.equals_missing(&expected) {
                return Err(invalid(format!("table disagrees with trace: {role}")));
            }
        }
        let channels = channel_table(self.run_id(), &self.manifest.channels)?;
        if !ParquetReader::new(File::open(&self.artifact("tables.channels")?.path)?)
            .finish()?
            .equals_missing(&channels)
        {
            return Err(invalid("channel table disagrees with manifest"));
        }
        Ok(BundleVerificationResult {
            artifact_count: self.artifacts.len(),
            total_bytes: bytes,
        })
    }
    /// Releases the cached result; subsequent reads verify and reload its arrays.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            *cache = None;
        }
    }
}
impl SpectralReconstructionResult {
    /// Writes a separate spectral bundle from a supported solver's final checkpoint.
    /// Scientific arrays use lossless NPY, traces/channel metadata use Parquet.
    /// Existing destinations get a numbered sibling; publication is manifest-last.
    /// Preview generation is not supported for this format and must be disabled.
    pub fn write_bundle(
        &self,
        path: impl AsRef<Path>,
        options: BundleExportOptions,
    ) -> Result<SpectralResultBundle> {
        let checkpoint=self.checkpoint.as_ref().ok_or_else(||invalid("spectral result has no resumable checkpoint; write the enclosing joint result instead"))?;
        write(self, checkpoint, None, None, path.as_ref(), options)
    }
}
impl MultiWavelengthSolverResult {
    /// Writes channel fields plus authoritative joint OPD, gauge, bounds and initialization records.
    pub fn write_bundle(
        &self,
        path: impl AsRef<Path>,
        options: BundleExportOptions,
    ) -> Result<SpectralResultBundle> {
        write(
            &self.spectral,
            &self.checkpoint,
            Some(&self.opd_m),
            None,
            path.as_ref(),
            options,
        )
    }
}
impl MultiWavelengthReconstructionResult {
    /// Writes spectral AP fields and synthetic-wavelength diagnostics, preserving invalid NaNs.
    pub fn write_bundle(
        &self,
        path: impl AsRef<Path>,
        options: BundleExportOptions,
    ) -> Result<SpectralResultBundle> {
        let checkpoint = self
            .spectral
            .checkpoint
            .as_ref()
            .ok_or_else(|| invalid("spectral result has no checkpoint"))?;
        write(
            &self.spectral,
            checkpoint,
            None,
            Some(&self.opd),
            path.as_ref(),
            options,
        )
    }
}
fn validate_result(
    result: &SpectralReconstructionResult,
    checkpoint: &SpectralReconstructionCheckpoint,
) -> Result<()> {
    checkpoint.validate()?;
    let model = checkpoint.model();
    let backend =
        crate::backend::CpuBackend::new(model.image_shape(), model.reconstruction_shape())?;
    if result.channels.len() != model.channels().len()
        || result.object_coupling != model.object_coupling()
        || result.runtime.completed_iterations != checkpoint.completed_iterations()
        || result.runtime.algorithm != checkpoint.algorithm()
        || result.runtime.elapsed_seconds != checkpoint.elapsed_seconds
        || serde_json::to_value(&result.trace)? != serde_json::to_value(checkpoint.trace())?
    {
        return Err(invalid("result metadata differs from checkpoint"));
    }
    for (i, (channel, kernel)) in result.channels.iter().zip(model.channels()).enumerate() {
        if channel.channel_id != kernel.channel_id
            || Some(channel.wavelength_vacuum_m) != kernel.model.sampling().wavelength
            || channel.pupil != *kernel.model.pupil()
            || channel.object.dim() != model.reconstruction_shape()
            || channel.object_spectrum.dim() != model.reconstruction_shape()
            || channel.amplitude.dim() != model.reconstruction_shape()
            || channel.phase.dim() != model.reconstruction_shape()
            || !channel.object.is_standard_layout()
            || !channel.object_spectrum.is_standard_layout()
            || !channel.amplitude.is_standard_layout()
            || !channel.phase.is_standard_layout()
            || channel
                .object_spectrum
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid(
                "channel metadata, pupil or array differs from model",
            ));
        }
        for ((v, &a), &p) in channel
            .object
            .iter()
            .zip(&channel.amplitude)
            .zip(&channel.phase)
        {
            if !v.re.is_finite()
                || !v.im.is_finite()
                || !a.is_finite()
                || a < 0.0
                || !p.is_finite()
                || (v.norm() - a).abs() > 64.0 * f64::EPSILON * a.max(1.0)
                || (v.arg() - p).abs() > 64.0 * f64::EPSILON
            {
                return Err(invalid("invalid object/amplitude/phase values"));
            }
        }
        let shape = model.reconstruction_shape();
        let spectrum: Vec<_> = channel.object_spectrum.iter().copied().collect();
        let mut object = vec![crate::Complex64::default(); spectrum.len()];
        crate::model::ifftshift_copy(&spectrum, &mut object, shape);
        let mut column = vec![crate::Complex64::default(); shape.0.max(model.image_shape().0)];
        crate::backend::Backend::fft2(
            &backend,
            &mut object,
            shape,
            crate::backend::FftDirection::Inverse,
            &mut column,
        )?;
        let tolerance = 64.0 * f64::EPSILON * (object.len() as f64).sqrt();
        if object
            .iter()
            .zip(&channel.object)
            .any(|(a, b)| (*a - *b).norm() > tolerance * a.norm().max(b.norm()).max(1.0))
        {
            return Err(invalid("object and centered spectrum disagree"));
        }
        match &checkpoint.state {
            StoredSolver::Spectral { spectra, .. } => {
                if spectra[model.object_index(i)?].data
                    != channel.object_spectrum.iter().copied().collect::<Vec<_>>()
                {
                    return Err(invalid("spectrum differs from checkpoint"));
                }
            }
            StoredSolver::JointOpd {
                opd, amplitudes, ..
            } => {
                if amplitudes[i].data != channel.amplitude.iter().copied().collect::<Vec<_>>() {
                    return Err(invalid("amplitude differs from joint checkpoint"));
                }
                if channel
                    .object
                    .iter()
                    .zip(&opd.data)
                    .zip(&amplitudes[i].data)
                    .any(|((field, &opd), &amplitude)| {
                        let expected = crate::Complex64::from_polar(
                            amplitude,
                            std::f64::consts::TAU * opd / channel.wavelength_vacuum_m,
                        );
                        (*field - expected).norm() > tolerance * amplitude.max(1.0)
                    })
                {
                    return Err(invalid("channel field differs from authoritative OPD"));
                }
            }
        }
    }
    Ok(())
}
fn write(
    result: &SpectralReconstructionResult,
    checkpoint: &SpectralReconstructionCheckpoint,
    opd: Option<&Array2<f64>>,
    unwrapped: Option<&OpticalPathDifferenceResult>,
    path: &Path,
    options: BundleExportOptions,
) -> Result<SpectralResultBundle> {
    validate_result(result, checkpoint)?;
    if options.include_previews {
        return Err(invalid(
            "spectral bundles currently require include_previews=false",
        ));
    }
    let joint = matches!(checkpoint.state, StoredSolver::JointOpd { .. });
    if joint != opd.is_some() {
        return Err(invalid("joint OPD array required for joint checkpoint"));
    }
    if let Some(opd) = opd
        && (!opd.is_standard_layout()
            || checkpoint.opd_m().is_none_or(|stored| stored != opd.view()))
    {
        return Err(invalid("OPD array differs from checkpoint"));
    }
    if let Some(unwrapped) = unwrapped {
        crate::reconstruction::spectral_checkpoint::validate_opd(unwrapped)?;
        if unwrapped.opd_m.dim() != checkpoint.model().reconstruction_shape()
            || unwrapped.wavelengths_vacuum_m
                != result
                    .channels
                    .iter()
                    .map(|c| c.wavelength_vacuum_m)
                    .collect::<Vec<_>>()
        {
            return Err(invalid("unwrapped OPD metadata differs from channels"));
        }
    }
    let run_id = options
        .run_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    validate_run_id(&run_id)?;
    let (destination, workspace) = unique_paths(path)?;
    fs::create_dir_all(destination.parent().unwrap_or(Path::new(".")))?;
    fs::create_dir(&workspace)?;
    let mut manifest = Manifest {
        kind: KIND.into(),
        bundle_format_version: SPECTRAL_BUNDLE_FORMAT_VERSION,
        run_id,
        label: options.label,
        crate_version: env!("CARGO_PKG_VERSION").into(),
        channels: result
            .channels
            .iter()
            .map(|c| Channel {
                channel_id: c.channel_id.clone(),
                wavelength_vacuum_m: c.wavelength_vacuum_m,
            })
            .collect(),
        object_coupling: result.object_coupling,
        image_shape: checkpoint.model().image_shape(),
        reconstruction_shape: checkpoint.model().reconstruction_shape(),
        runtime: result.runtime.clone(),
        trace: result.trace.clone(),
        has_joint_opd: joint,
        has_unwrapped_opd: unwrapped.is_some(),
        artifacts: Vec::new(),
    };
    let descriptors = expected_roles(&manifest);
    for (role, (dtype, shape, media)) in descriptors {
        let relative_path = if role.starts_with("channels.") || role == "arrays.opd_m" {
            PathBuf::from(format!("arrays/{role}.npy"))
        } else if role.starts_with("tables.") {
            PathBuf::from(format!("tables/{role}.parquet"))
        } else {
            PathBuf::from(format!("state/{role}.json"))
        };
        let file = workspace.join(&relative_path);
        fs::create_dir_all(file.parent().unwrap())?;
        if role == "state.checkpoint" {
            checkpoint.save(&file)?;
        } else if role == "state.unwrapped_opd" {
            write_json_atomic(&file, &StoredOpd::from_result(unwrapped.unwrap()))?;
        } else if role == "arrays.opd_m" {
            npy::write_f64_2(&file, opd.unwrap().view())?;
        } else if role.starts_with("tables.") {
            let mut table = match role.as_str() {
                "tables.channels" => channel_table(&manifest.run_id, &manifest.channels)?,
                "tables.history" => {
                    crate::tabular::history_dataframe(&manifest.run_id, &manifest.trace)?
                }
                _ => {
                    crate::tabular::algorithm_metrics_dataframe(&manifest.run_id, &manifest.trace)?
                }
            };
            let mut output = File::create(&file)?;
            {
                ParquetWriter::new(&mut output)
                    .with_key_value_metadata(Some(KeyValueMetadata::from_static(vec![
                        ("fpm.bundle_kind".into(), KIND.into()),
                        (
                            "fpm.bundle_format_version".into(),
                            SPECTRAL_BUNDLE_FORMAT_VERSION.to_string(),
                        ),
                        ("fpm.run_id".into(), manifest.run_id.clone()),
                        ("fpm.table_role".into(), role.clone()),
                    ])))
                    .finish(&mut table)?;
            }
            output.sync_all()?;
        } else {
            let parts: Vec<_> = role.split('.').collect();
            let c = &result.channels[parts[1]
                .parse::<usize>()
                .map_err(|_| invalid("bad channel index"))?];
            match parts[2] {
                "object" => npy::write_complex2(&file, c.object.view())?,
                "object_spectrum" => npy::write_complex2(&file, c.object_spectrum.view())?,
                "amplitude" => npy::write_f64_2(&file, c.amplitude.view())?,
                "phase" => npy::write_f64_2(&file, c.phase.view())?,
                "pupil" => npy::write_complex2(&file, c.pupil.values())?,
                "pupil_support" => npy::write_u8_2(&file, c.pupil.support())?,
                _ => return Err(invalid("unknown channel artifact")),
            };
        }
        manifest.artifacts.push(ManifestArtifact {
            role,
            relative_path,
            media_type: media.into(),
            byte_size: fs::metadata(&file)?.len(),
            sha256: sha256(&file)?,
            dtype,
            shape,
        });
    }
    write_json_atomic(&workspace.join("manifest.json"), &manifest)?;
    fs::rename(&workspace, &destination)?;
    SpectralResultBundle::read(destination)
}
type Descriptor = (Option<String>, Option<Vec<u64>>, &'static str);
fn expected_roles(m: &Manifest) -> BTreeMap<String, Descriptor> {
    let mut roles = BTreeMap::new();
    for i in 0..m.channels.len() {
        for (name, dtype, shape) in [
            ("object", "<c16", m.reconstruction_shape),
            ("object_spectrum", "<c16", m.reconstruction_shape),
            ("amplitude", "<f8", m.reconstruction_shape),
            ("phase", "<f8", m.reconstruction_shape),
            ("pupil", "<c16", m.image_shape),
            ("pupil_support", "|u1", m.image_shape),
        ] {
            roles.insert(
                format!("channels.{i}.{name}"),
                (
                    Some(dtype.into()),
                    Some(vec![shape.0 as u64, shape.1 as u64]),
                    "application/x-npy",
                ),
            );
        }
    }
    for role in [
        "tables.channels",
        "tables.history",
        "tables.algorithm_metrics",
    ] {
        roles.insert(role.into(), (None, None, "application/vnd.apache.parquet"));
    }
    roles.insert("state.checkpoint".into(), (None, None, "application/json"));
    if m.has_joint_opd {
        roles.insert(
            "arrays.opd_m".into(),
            (
                Some("<f8".into()),
                Some(vec![
                    m.reconstruction_shape.0 as u64,
                    m.reconstruction_shape.1 as u64,
                ]),
                "application/x-npy",
            ),
        );
    }
    if m.has_unwrapped_opd {
        roles.insert(
            "state.unwrapped_opd".into(),
            (None, None, "application/json"),
        );
    }
    roles
}
fn channel_table(run_id: &str, channels: &[Channel]) -> Result<DataFrame> {
    use polars::prelude::{Column, NamedFrom, Series};
    Ok(DataFrame::new(
        channels.len(),
        vec![
            Column::from(Series::new("run_id".into(), vec![run_id; channels.len()])),
            Column::from(Series::new(
                "channel_index".into(),
                (0..channels.len() as u64).collect::<Vec<_>>(),
            )),
            Column::from(Series::new(
                "channel_id".into(),
                channels
                    .iter()
                    .map(|c| c.channel_id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Column::from(Series::new(
                "wavelength_vacuum_m".into(),
                channels
                    .iter()
                    .map(|c| c.wavelength_vacuum_m)
                    .collect::<Vec<_>>(),
            )),
        ],
    )?)
}
fn safe_file(root: &Path, path: &Path) -> Result<PathBuf> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(invalid("artifact path must be safely relative"));
    }
    let absolute = root.join(path).canonicalize()?;
    if !absolute.starts_with(root) {
        return Err(invalid("artifact symlink escapes bundle root"));
    }
    if !absolute.is_file() {
        return Err(invalid("artifact is not a regular file"));
    }
    Ok(absolute)
}
fn invalid(reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name: "spectral_bundle",
        reason: reason.into(),
    }
}
