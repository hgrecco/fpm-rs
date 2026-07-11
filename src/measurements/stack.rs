use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use std::path::Path;

use crate::{
    error::{Error, Result},
    image_io::{GrayscaleScaling, load_grayscale, load_grayscale_tiff_pages},
};

use super::{FrameMetadata, ImageSet, MeasurementSpec, PreprocessingConfig};

/// An in-memory stack of image-plane intensity measurements.
#[derive(Clone, Debug, Serialize)]
pub struct MeasurementStack {
    data: Vec<f64>,
    image_shape: (usize, usize),
    frames: usize,
    pub frame_metadata: Vec<FrameMetadata>,
    pub dark_frame: Option<Vec<f64>>,
    pub flat_field: Option<Vec<f64>>,
    pub background: Option<Vec<f64>>,
    pub masks: Option<Vec<u8>>,
    pub preprocessing: PreprocessingConfig,
}

impl<'de> Deserialize<'de> for MeasurementStack {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Representation {
            data: Vec<f64>,
            image_shape: (usize, usize),
            frames: usize,
            frame_metadata: Vec<FrameMetadata>,
            dark_frame: Option<Vec<f64>>,
            flat_field: Option<Vec<f64>>,
            background: Option<Vec<f64>>,
            masks: Option<Vec<u8>>,
            preprocessing: PreprocessingConfig,
        }

        let representation = Representation::deserialize(deserializer)?;
        let stack = Self {
            data: representation.data,
            image_shape: representation.image_shape,
            frames: representation.frames,
            frame_metadata: representation.frame_metadata,
            dark_frame: representation.dark_frame,
            flat_field: representation.flat_field,
            background: representation.background,
            masks: representation.masks,
            preprocessing: representation.preprocessing,
        };
        stack.validate().map_err(D::Error::custom)?;
        Ok(stack)
    }
}

impl MeasurementStack {
    pub fn from_vec(
        data: Vec<f64>,
        image_shape: (usize, usize),
        frame_metadata: Vec<FrameMetadata>,
    ) -> Result<Self> {
        let frame_len = image_shape
            .0
            .checked_mul(image_shape.1)
            .ok_or_else(|| Error::InvalidShape("measurement shape overflows".into()))?;
        if frame_len == 0 {
            return Err(Error::InvalidShape(
                "measurement dimensions must be non-zero".into(),
            ));
        }
        if data.is_empty() || !data.len().is_multiple_of(frame_len) {
            return Err(Error::InvalidMeasurements(format!(
                "data length {} is not a positive multiple of frame size {frame_len}",
                data.len()
            )));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidMeasurements(
                "measurements contain non-finite values".into(),
            ));
        }
        let frames = data.len() / frame_len;
        let frame_metadata = if frame_metadata.is_empty() {
            (0..frames).map(FrameMetadata::new).collect()
        } else {
            if frame_metadata.len() != frames {
                return Err(Error::InvalidMeasurements(format!(
                    "{} metadata entries for {frames} frames",
                    frame_metadata.len()
                )));
            }
            frame_metadata
        };
        let stack = Self {
            data,
            image_shape,
            frames,
            frame_metadata,
            dark_frame: None,
            flat_field: None,
            background: None,
            masks: None,
            preprocessing: PreprocessingConfig::default(),
        };
        stack.validate()?;
        Ok(stack)
    }

    pub fn from_frames(frames: &[Vec<f64>], image_shape: (usize, usize)) -> Result<Self> {
        let data = frames.iter().flatten().copied().collect();
        Self::from_vec(data, image_shape, Vec::new())
    }

    /// Loads an in-memory stack from single-channel PNG or TIFF images while
    /// preserving native 8-bit or 16-bit detector counts.
    pub fn from_image_files<P: AsRef<Path>>(
        paths: &[P],
        frame_metadata: Vec<FrameMetadata>,
    ) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::InvalidMeasurements(
                "at least one image path is required".into(),
            ));
        }
        let mut shape = None;
        let mut data = Vec::new();
        for path in paths {
            let frame = load_grayscale(path, GrayscaleScaling::NativeCounts)?;
            if let Some(expected) = shape {
                if frame.shape() != expected {
                    return Err(Error::InvalidMeasurements(format!(
                        "image {} has shape {:?}, expected {expected:?}",
                        path.as_ref().display(),
                        frame.shape()
                    )));
                }
            } else {
                shape = Some(frame.shape());
            }
            data.extend(frame.into_vec());
        }
        let metadata = if frame_metadata.is_empty() {
            paths
                .iter()
                .enumerate()
                .map(|(index, path)| {
                    let mut metadata = FrameMetadata::new(index);
                    metadata.label = Some(path.as_ref().display().to_string());
                    metadata
                })
                .collect()
        } else {
            frame_metadata
        };
        let shape = shape.ok_or_else(|| {
            Error::InvalidMeasurements("image paths did not produce a frame".into())
        })?;
        Self::from_vec(data, shape, metadata)
    }

    /// Loads every page of a grayscale 8-bit or 16-bit TIFF as one frame.
    pub fn from_tiff_stack(
        path: impl AsRef<Path>,
        frame_metadata: Vec<FrameMetadata>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let pages = load_grayscale_tiff_pages(path, GrayscaleScaling::NativeCounts)?;
        let first = pages.first().ok_or_else(|| {
            Error::InvalidMeasurements("TIFF stack does not contain an image".into())
        })?;
        let shape = first.shape();
        if pages.iter().any(|page| page.shape() != shape) {
            return Err(Error::InvalidMeasurements(
                "all TIFF pages must have the same dimensions".into(),
            ));
        }
        let metadata = if frame_metadata.is_empty() {
            (0..pages.len())
                .map(|index| {
                    let mut metadata = FrameMetadata::new(index);
                    metadata.label = Some(format!("{}#page={index}", path.display()));
                    metadata
                })
                .collect()
        } else {
            frame_metadata
        };
        Self::from_vec(
            pages.into_iter().flat_map(|page| page.into_vec()).collect(),
            shape,
            metadata,
        )
    }

    /// Loads a JSON manifest. Relative image paths are resolved against the
    /// manifest's parent directory. Declared preprocessing is configured but is
    /// not applied until [`Self::apply_preprocessing`] is called.
    pub fn from_manifest(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let manifest = MeasurementSpec::load(path)?;
        let base_directory = path.parent().unwrap_or_else(|| Path::new("."));
        Self::from_manifest_definition(manifest, base_directory)
    }

    pub fn from_manifest_definition(
        manifest: MeasurementSpec,
        base_directory: impl AsRef<Path>,
    ) -> Result<Self> {
        let base_directory = base_directory.as_ref();
        let paths: Vec<_> = manifest
            .frames
            .iter()
            .map(|frame| resolve_path(base_directory, &frame.path))
            .collect();
        let metadata = manifest
            .frames
            .iter()
            .enumerate()
            .map(|(index, frame)| FrameMetadata {
                frame_index: index,
                illumination_index: frame.illumination_index,
                original_frame_index: Some(index),
                original_illumination_index: frame.illumination_index,
                exposure_time: frame.exposure_time,
                weight: frame.weight,
                label: frame
                    .label
                    .clone()
                    .or_else(|| Some(frame.path.display().to_string())),
            })
            .collect();
        let mut stack = Self::from_image_files(&paths, metadata)?;
        if let Some(path) = &manifest.dark_frame {
            stack.dark_frame = Some(load_manifest_image(
                base_directory,
                path,
                stack.image_shape,
            )?);
        }
        if let Some(path) = &manifest.flat_field {
            stack.flat_field = Some(load_manifest_image(
                base_directory,
                path,
                stack.image_shape,
            )?);
        }
        if let Some(background) = &manifest.background {
            stack.background = Some(load_manifest_image_set(
                base_directory,
                background,
                stack.image_shape,
                stack.frames,
            )?);
        }
        if let Some(mask) = &manifest.mask {
            stack.masks = Some(
                load_manifest_image_set(base_directory, mask, stack.image_shape, stack.frames)?
                    .into_iter()
                    .map(|value| u8::from(value != 0.0))
                    .collect(),
            );
        }
        stack.preprocessing = manifest.preprocessing;
        stack.validate()?;
        Ok(stack)
    }

    pub fn frame_count(&self) -> usize {
        self.frames
    }

    pub fn image_shape(&self) -> (usize, usize) {
        self.image_shape
    }

    pub fn frame_len(&self) -> usize {
        self.image_shape.0 * self.image_shape.1
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn validate(&self) -> Result<()> {
        let frame_len = self
            .image_shape
            .0
            .checked_mul(self.image_shape.1)
            .ok_or_else(|| Error::InvalidShape("measurement shape overflows".into()))?;
        let expected_len = frame_len.checked_mul(self.frames).ok_or_else(|| {
            Error::InvalidMeasurements("measurement frame count overflows".into())
        })?;
        if frame_len == 0 || self.frames == 0 || self.data.len() != expected_len {
            return Err(Error::InvalidMeasurements(
                "stored frame count, image shape, and data length are inconsistent".into(),
            ));
        }
        if self.data.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidMeasurements(
                "measurements contain non-finite values".into(),
            ));
        }
        if self.frame_metadata.len() != self.frames {
            return Err(Error::InvalidMeasurements(format!(
                "{} metadata entries for {} frames",
                self.frame_metadata.len(),
                self.frames
            )));
        }
        for (index, metadata) in self.frame_metadata.iter().enumerate() {
            if metadata.frame_index != index {
                return Err(Error::InvalidMeasurements(format!(
                    "metadata entry {index} identifies frame {}",
                    metadata.frame_index
                )));
            }
            if !metadata.exposure_time.is_finite() || metadata.exposure_time <= 0.0 {
                return Err(Error::InvalidMeasurements(format!(
                    "frame {index} has invalid exposure {}",
                    metadata.exposure_time
                )));
            }
            if !metadata.weight.is_finite() || metadata.weight < 0.0 {
                return Err(Error::InvalidMeasurements(format!(
                    "frame {index} has invalid weight {}",
                    metadata.weight
                )));
            }
        }
        if let Some(dark) = &self.dark_frame {
            self.validate_correction("dark frame", dark, false)?;
        }
        if let Some(flat) = &self.flat_field {
            self.validate_correction("flat field", flat, false)?;
            if flat.iter().any(|&value| value <= 0.0) {
                return Err(Error::InvalidMeasurements(
                    "flat-field values must be positive".into(),
                ));
            }
        }
        if let Some(background) = &self.background {
            self.validate_correction("background", background, true)?;
        }
        if let Some(masks) = &self.masks
            && masks.len() != frame_len
            && masks.len() != self.data.len()
        {
            return Err(Error::InvalidMeasurements(format!(
                "mask length {} must be {frame_len} or {}",
                masks.len(),
                self.data.len()
            )));
        }
        if self.preprocessing.subtract_dark && self.dark_frame.is_none() {
            return Err(Error::InvalidMeasurements(
                "dark subtraction requested without a dark frame".into(),
            ));
        }
        if self.preprocessing.divide_flat_field && self.flat_field.is_none() {
            return Err(Error::InvalidMeasurements(
                "flat-field correction requested without a flat field".into(),
            ));
        }
        if self.preprocessing.subtract_background && self.background.is_none() {
            return Err(Error::InvalidMeasurements(
                "background subtraction requested without a background".into(),
            ));
        }
        Ok(())
    }

    pub fn frame(&self, index: usize) -> Result<&[f64]> {
        if index >= self.frames {
            return Err(Error::FrameOutOfRange {
                index,
                frames: self.frames,
            });
        }
        let start = index * self.frame_len();
        Ok(&self.data[start..start + self.frame_len()])
    }

    pub fn frame_mut(&mut self, index: usize) -> Result<&mut [f64]> {
        if index >= self.frames {
            return Err(Error::FrameOutOfRange {
                index,
                frames: self.frames,
            });
        }
        let frame_len = self.frame_len();
        let start = index * frame_len;
        Ok(&mut self.data[start..start + frame_len])
    }

    pub fn frame_weight(&self, index: usize) -> Result<f64> {
        self.frame_metadata
            .get(index)
            .map(|metadata| metadata.weight)
            .ok_or(Error::FrameOutOfRange {
                index,
                frames: self.frames,
            })
    }

    /// Returns the mask for a frame. Zero-valued mask entries are excluded.
    /// A single-frame mask is broadcast to every measurement frame.
    pub fn frame_mask(&self, index: usize) -> Result<Option<&[u8]>> {
        if index >= self.frames {
            return Err(Error::FrameOutOfRange {
                index,
                frames: self.frames,
            });
        }
        let Some(masks) = &self.masks else {
            return Ok(None);
        };
        let frame_len = self.frame_len();
        if masks.len() == frame_len {
            Ok(Some(masks))
        } else {
            let start = index * frame_len;
            Ok(Some(&masks[start..start + frame_len]))
        }
    }

    pub fn with_dark_frame(mut self, dark: Vec<f64>) -> Result<Self> {
        self.validate_correction("dark frame", &dark, false)?;
        self.dark_frame = Some(dark);
        self.preprocessing.subtract_dark = true;
        Ok(self)
    }

    pub fn with_flat_field(mut self, flat: Vec<f64>) -> Result<Self> {
        self.validate_correction("flat field", &flat, false)?;
        if flat.iter().any(|&value| value <= 0.0) {
            return Err(Error::InvalidMeasurements(
                "flat-field values must be positive".into(),
            ));
        }
        self.flat_field = Some(flat);
        self.preprocessing.divide_flat_field = true;
        Ok(self)
    }

    pub fn with_background(mut self, background: Vec<f64>) -> Result<Self> {
        self.validate_correction("background", &background, true)?;
        self.background = Some(background);
        self.preprocessing.subtract_background = true;
        Ok(self)
    }

    pub fn with_masks(mut self, masks: Vec<u8>) -> Result<Self> {
        let frame_len = self.frame_len();
        if masks.len() != frame_len && masks.len() != self.data.len() {
            return Err(Error::InvalidMeasurements(format!(
                "mask length {} must be {frame_len} or {}",
                masks.len(),
                self.data.len()
            )));
        }
        self.masks = Some(masks);
        Ok(self)
    }

    pub fn normalize_exposure(mut self) -> Self {
        self.preprocessing.normalize_exposure = true;
        self
    }

    pub fn clamp_negative(mut self) -> Self {
        self.preprocessing.clamp_negative = true;
        self
    }

    pub fn apply_preprocessing(mut self) -> Result<Self> {
        self.validate()?;
        let frame_len = self.frame_len();
        for frame_index in 0..self.frames {
            let exposure = self.frame_metadata[frame_index].exposure_time;
            if self.preprocessing.normalize_exposure && (!exposure.is_finite() || exposure <= 0.0) {
                return Err(Error::InvalidMeasurements(format!(
                    "frame {frame_index} has invalid exposure {exposure}"
                )));
            }
            for pixel in 0..frame_len {
                let index = frame_index * frame_len + pixel;
                let mut value = self.data[index];
                if let Some(dark) = self
                    .preprocessing
                    .subtract_dark
                    .then_some(self.dark_frame.as_deref())
                    .flatten()
                {
                    value -= dark[pixel];
                }
                if let Some(background) = self
                    .preprocessing
                    .subtract_background
                    .then_some(self.background.as_deref())
                    .flatten()
                {
                    value -= background[if background.len() == frame_len {
                        pixel
                    } else {
                        index
                    }];
                }
                if let Some(flat) = self
                    .preprocessing
                    .divide_flat_field
                    .then_some(self.flat_field.as_deref())
                    .flatten()
                {
                    value /= flat[pixel];
                }
                if self.preprocessing.normalize_exposure {
                    value /= exposure;
                }
                if self.preprocessing.clamp_negative {
                    value = value.max(0.0);
                }
                self.data[index] = value;
            }
        }
        Ok(self)
    }

    fn validate_correction(&self, name: &str, values: &[f64], per_frame: bool) -> Result<()> {
        let frame_len = self.frame_len();
        let valid_length =
            values.len() == frame_len || (per_frame && values.len() == self.data.len());
        if !valid_length || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidMeasurements(format!(
                "{name} must contain finite values and have length {frame_len}{}",
                if per_frame {
                    format!(" or {}", self.data.len())
                } else {
                    String::new()
                }
            )));
        }
        Ok(())
    }
}

pub(super) fn resolve_path(base_directory: &Path, path: &Path) -> std::path::PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base_directory.join(path)
    }
}

pub(super) fn load_manifest_image(
    base_directory: &Path,
    path: &Path,
    expected_shape: (usize, usize),
) -> Result<Vec<f64>> {
    let resolved = resolve_path(base_directory, path);
    let image = load_grayscale(&resolved, GrayscaleScaling::NativeCounts)?;
    if image.shape() != expected_shape {
        return Err(Error::InvalidMeasurements(format!(
            "image {} has shape {:?}, expected {expected_shape:?}",
            resolved.display(),
            image.shape()
        )));
    }
    Ok(image.into_vec())
}

pub(super) fn load_manifest_image_set(
    base_directory: &Path,
    images: &ImageSet,
    expected_shape: (usize, usize),
    frame_count: usize,
) -> Result<Vec<f64>> {
    match images {
        ImageSet::Single(path) => load_manifest_image(base_directory, path, expected_shape),
        ImageSet::PerFrame(paths) => {
            if paths.len() != frame_count {
                return Err(Error::InvalidMeasurements(format!(
                    "manifest image set has {} entries for {frame_count} frames",
                    paths.len()
                )));
            }
            let mut values = Vec::with_capacity(expected_shape.0 * expected_shape.1 * frame_count);
            for path in paths {
                values.extend(load_manifest_image(base_directory, path, expected_shape)?);
            }
            Ok(values)
        }
    }
}
