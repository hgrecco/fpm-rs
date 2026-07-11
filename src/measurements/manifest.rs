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
    pub frames: Vec<FrameSpec>,
    #[serde(default)]
    pub dark_frame: Option<PathBuf>,
    #[serde(default)]
    pub flat_field: Option<PathBuf>,
    #[serde(default)]
    pub background: Option<ImageSet>,
    #[serde(default)]
    pub mask: Option<ImageSet>,
    #[serde(default)]
    pub preprocessing: PreprocessingConfig,
}

impl MeasurementSpec {
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

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Ok(serde_json::from_reader(reader)?)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Path and acquisition metadata for one measurement frame.
pub struct FrameSpec {
    pub path: PathBuf,
    #[serde(default)]
    pub illumination_index: Option<usize>,
    #[serde(default = "unit_value")]
    pub exposure_time: f64,
    #[serde(default = "unit_value")]
    pub weight: f64,
    #[serde(default)]
    pub label: Option<String>,
}

impl FrameSpec {
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
    Single(PathBuf),
    PerFrame(Vec<PathBuf>),
}

fn unit_value() -> f64 {
    1.0
}
