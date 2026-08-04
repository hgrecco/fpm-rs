use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::Result;

use super::PreprocessingConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Serializable specification for loading a measurement stack.
pub struct MeasurementSpec {
    /// Measurement files in acquisition-frame order.
    pub frames: Vec<FrameSpec>,
    #[serde(default)]
    /// Optional shared detector dark image, resolved relative to the manifest.
    pub dark_frame: Option<PathBuf>,
    #[serde(default)]
    /// Optional shared positive flat-field image, resolved relative to the manifest.
    pub flat_field: Option<PathBuf>,
    #[serde(default)]
    /// Optional shared or per-frame additive background images.
    pub background: Option<ImageSet>,
    #[serde(default)]
    /// Optional shared or per-frame binary mask images; non-zero pixels are valid.
    pub mask: Option<ImageSet>,
    #[serde(default)]
    /// Corrections applied when the resident stack is built or a lazy frame is decoded.
    pub preprocessing: PreprocessingConfig,
}

impl MeasurementSpec {
    /// Creates a specification with no correction images and preprocessing disabled.
    pub fn new(frames: Vec<FrameSpec>) -> Self {
        Self {
            frames,
            dark_frame: None,
            flat_field: None,
            background: None,
            mask: None,
            preprocessing: PreprocessingConfig::default(),
        }
    }

    /// Serializes this specification as pretty-printed JSON at `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Deserializes a JSON measurement specification without loading its image files.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Ok(serde_json::from_reader(reader)?)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Path and acquisition metadata for one measurement frame.
pub struct FrameSpec {
    /// Image path, interpreted relative to the manifest directory when not absolute.
    pub path: PathBuf,
    #[serde(default)]
    /// Optional individual source index associated with this acquisition frame.
    pub illumination_index: Option<usize>,
    #[serde(default = "unit_value")]
    /// Positive exposure time in caller-defined units; defaults to `1.0`.
    pub exposure_time: f64,
    #[serde(default = "unit_value")]
    /// Non-negative reconstruction weight; defaults to `1.0`.
    pub weight: f64,
    #[serde(default)]
    /// Optional human-readable acquisition label.
    pub label: Option<String>,
}

impl FrameSpec {
    /// Creates a unit-exposure, unit-weight frame specification for `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            illumination_index: None,
            exposure_time: 1.0,
            weight: 1.0,
            label: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
/// A single image shared by all frames or one image per frame.
pub enum ImageSet {
    /// One image broadcast to every acquisition frame.
    Single(PathBuf),
    /// One image per acquisition frame, in matching order.
    PerFrame(Vec<PathBuf>),
}

fn unit_value() -> f64 {
    1.0
}
