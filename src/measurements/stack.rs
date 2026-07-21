use std::path::Path;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, Axis};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use crate::{
    Result,
    array_layout::{StandardArray2, StandardArray3, checked_len_2d, checked_len_3d},
    error::Error,
    image_io::{GrayscaleScaling, load_grayscale, load_grayscale_tiff_pages},
};

use super::{FrameMetadata, ImageSet, MeasurementSpec, PreprocessingConfig};

#[derive(Clone, Debug)]
enum FrameArray<T> {
    Shared(StandardArray2<T>),
    PerFrame(StandardArray3<T>),
}

impl<T> FrameArray<T> {
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Shared(values) => values.as_slice(),
            Self::PerFrame(values) => values.as_slice(),
        }
    }

    fn frame(&self, index: usize, frame_len: usize) -> &[T] {
        match self {
            Self::Shared(values) => values.as_slice(),
            Self::PerFrame(values) => {
                let start = index * frame_len;
                &values.as_slice()[start..start + frame_len]
            }
        }
    }
}

/// An in-memory stack of image-plane intensity measurements.
///
/// Data are owned as a C-contiguous ndarray with dimension order
/// `(frame, row, column)`. Public ndarray views borrow this allocation without
/// copying. The [`super::MeasurementRead`] boundary continues to expose flat
/// frame slices for reconstruction and lazy-stack parity.
#[derive(Clone, Debug)]
pub struct MeasurementStack {
    data: StandardArray3<f64>,
    frame_metadata: Vec<FrameMetadata>,
    dark_frame: Option<StandardArray2<f64>>,
    flat_field: Option<StandardArray2<f64>>,
    background: Option<FrameArray<f64>>,
    masks: Option<FrameArray<u8>>,
    preprocessing: PreprocessingConfig,
}

impl MeasurementStack {
    pub fn from_vec(
        data: Vec<f64>,
        image_shape: (usize, usize),
        frame_metadata: Vec<FrameMetadata>,
    ) -> Result<Self> {
        let frame_len = checked_len_2d(image_shape)?;
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
            data: StandardArray3::from_shape_vec((frames, image_shape.0, image_shape.1), data)?,
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

    /// Stores an owned standard-layout `(frame,row,column)` array without a copy.
    pub fn new(data: Array3<f64>, frame_metadata: Vec<FrameMetadata>) -> Result<Self> {
        let data = StandardArray3::try_from(data)?;
        let shape = data.dim();
        if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 {
            return Err(Error::InvalidShape(format!(
                "measurement dimensions must be non-zero, got {shape:?}"
            )));
        }
        if data.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidMeasurements(
                "measurements contain non-finite values".into(),
            ));
        }
        let frame_metadata = if frame_metadata.is_empty() {
            (0..shape.0).map(FrameMetadata::new).collect()
        } else if frame_metadata.len() == shape.0 {
            frame_metadata
        } else {
            return Err(Error::InvalidMeasurements(format!(
                "{} metadata entries for {} frames",
                frame_metadata.len(),
                shape.0
            )));
        };
        let stack = Self {
            data,
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
        let frame_len = checked_len_2d(image_shape)?;
        if frames.iter().any(|frame| frame.len() != frame_len) {
            return Err(Error::InvalidMeasurements(
                "every frame must match the declared image shape".into(),
            ));
        }
        let total = frame_len
            .checked_mul(frames.len())
            .ok_or_else(|| Error::ShapeOverflow {
                shape: vec![frames.len(), image_shape.0, image_shape.1],
            })?;
        let mut data = Vec::with_capacity(total);
        data.extend(frames.iter().flatten().copied());
        Self::from_vec(data, image_shape, Vec::new())
    }

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
                if frame.dim() != expected {
                    return Err(Error::InvalidMeasurements(format!(
                        "image {} has shape {:?}, expected {expected:?}",
                        path.as_ref().display(),
                        frame.dim()
                    )));
                }
            } else {
                shape = Some(frame.dim());
            }
            data.extend(frame);
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

    pub fn from_tiff_stack(
        path: impl AsRef<Path>,
        frame_metadata: Vec<FrameMetadata>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let pages = load_grayscale_tiff_pages(path, GrayscaleScaling::NativeCounts)?;
        let first = pages.first().ok_or_else(|| {
            Error::InvalidMeasurements("TIFF stack does not contain an image".into())
        })?;
        let shape = first.dim();
        if pages.iter().any(|page| page.dim() != shape) {
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
        Self::from_vec(pages.into_iter().flatten().collect(), shape, metadata)
    }

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
            stack.dark_frame = Some(StandardArray2::from_shape_vec(
                stack.image_shape(),
                load_manifest_image(base_directory, path, stack.image_shape())?,
            )?);
        }
        if let Some(path) = &manifest.flat_field {
            stack.flat_field = Some(StandardArray2::from_shape_vec(
                stack.image_shape(),
                load_manifest_image(base_directory, path, stack.image_shape())?,
            )?);
        }
        if let Some(background) = &manifest.background {
            let values = load_manifest_image_set(
                base_directory,
                background,
                stack.image_shape(),
                stack.frame_count(),
            )?;
            stack.background = Some(stack.frame_array_from_flat(values, "background")?);
        }
        if let Some(mask) = &manifest.mask {
            let values = load_manifest_image_set(
                base_directory,
                mask,
                stack.image_shape(),
                stack.frame_count(),
            )?
            .into_iter()
            .map(|value| u8::from(value != 0.0))
            .collect();
            stack.masks = Some(stack.frame_array_from_flat(values, "mask")?);
        }
        stack.preprocessing = manifest.preprocessing;
        stack.validate()?;
        Ok(stack)
    }

    pub fn data(&self) -> ArrayView3<'_, f64> {
        self.data.ndarray_view()
    }

    pub fn frame_view(&self, index: usize) -> Result<ArrayView2<'_, f64>> {
        self.check_frame(index)?;
        Ok(self.data.ndarray_view().index_axis_move(Axis(0), index))
    }

    pub fn frame_view_mut(&mut self, index: usize) -> Result<ArrayViewMut2<'_, f64>> {
        self.check_frame(index)?;
        Ok(self.data.ndarray_view_mut().index_axis_move(Axis(0), index))
    }

    pub fn frame_count(&self) -> usize {
        self.data.dim().0
    }

    pub fn image_shape(&self) -> (usize, usize) {
        let shape = self.data.dim();
        (shape.1, shape.2)
    }

    pub fn frame_len(&self) -> usize {
        let shape = self.data.dim();
        shape.1 * shape.2
    }

    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }

    pub fn frame_metadata(&self) -> &[FrameMetadata] {
        &self.frame_metadata
    }

    /// Sets a frame's reconstruction weight after validating it.
    pub fn set_frame_weight(&mut self, index: usize, weight: f64) -> Result<()> {
        self.check_frame(index)?;
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::InvalidMeasurements(format!(
                "frame {index} has invalid weight {weight}"
            )));
        }
        self.frame_metadata[index].weight = weight;
        Ok(())
    }

    pub fn preprocessing(&self) -> &PreprocessingConfig {
        &self.preprocessing
    }

    pub fn validate(&self) -> Result<()> {
        let shape = self.data.dim();
        let expected = checked_len_3d(shape)?;
        if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 || self.data.len() != expected {
            return Err(Error::InvalidMeasurements(
                "stored frame count, image shape, and data length are inconsistent".into(),
            ));
        }
        if self.data.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidMeasurements(
                "measurements contain non-finite values".into(),
            ));
        }
        if self.frame_metadata.len() != shape.0 {
            return Err(Error::InvalidMeasurements(format!(
                "{} metadata entries for {} frames",
                self.frame_metadata.len(),
                shape.0
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
        let image_shape = self.image_shape();
        if self
            .dark_frame
            .as_ref()
            .is_some_and(|values| values.dim() != image_shape || !all_finite(values.as_slice()))
        {
            return Err(Error::InvalidMeasurements(
                "dark frame must be finite and match the image shape".into(),
            ));
        }
        if self.flat_field.as_ref().is_some_and(|values| {
            values.dim() != image_shape
                || values
                    .as_slice()
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(Error::InvalidMeasurements(
                "flat field must be positive, finite, and match the image shape".into(),
            ));
        }
        self.validate_frame_array(self.background.as_ref(), "background")?;
        self.validate_frame_array(self.masks.as_ref(), "mask")?;
        if self
            .masks
            .as_ref()
            .is_some_and(|values| values.as_slice().iter().any(|&value| value > 1))
        {
            return Err(Error::InvalidMeasurements(
                "mask values must be exactly zero or one".into(),
            ));
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
        self.check_frame(index)?;
        let frame_len = self.frame_len();
        let start = index * frame_len;
        Ok(&self.data.as_slice()[start..start + frame_len])
    }

    pub fn frame_mut(&mut self, index: usize) -> Result<&mut [f64]> {
        self.check_frame(index)?;
        let frame_len = self.frame_len();
        let start = index * frame_len;
        Ok(&mut self.data.as_slice_mut()[start..start + frame_len])
    }

    pub fn frame_weight(&self, index: usize) -> Result<f64> {
        self.frame_metadata
            .get(index)
            .map(|metadata| metadata.weight)
            .ok_or(Error::FrameOutOfRange {
                index,
                frames: self.frame_count(),
            })
    }

    pub fn frame_mask(&self, index: usize) -> Result<Option<&[u8]>> {
        self.check_frame(index)?;
        Ok(self
            .masks
            .as_ref()
            .map(|values| values.frame(index, self.frame_len())))
    }

    pub fn with_dark_frame(mut self, dark: Array2<f64>) -> Result<Self> {
        let dark = StandardArray2::try_from(dark)?;
        if dark.dim() != self.image_shape() || !all_finite(dark.as_slice()) {
            return Err(Error::InvalidMeasurements(
                "dark frame must be finite and match the image shape".into(),
            ));
        }
        self.dark_frame = Some(dark);
        self.preprocessing.subtract_dark = true;
        Ok(self)
    }

    pub fn with_flat_field(mut self, flat: Array2<f64>) -> Result<Self> {
        let flat = StandardArray2::try_from(flat)?;
        if flat.dim() != self.image_shape()
            || flat
                .as_slice()
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::InvalidMeasurements(
                "flat field must be positive, finite, and match the image shape".into(),
            ));
        }
        self.flat_field = Some(flat);
        self.preprocessing.divide_flat_field = true;
        Ok(self)
    }

    pub fn with_background(mut self, background: Array2<f64>) -> Result<Self> {
        let values = StandardArray2::try_from(background)?;
        if values.dim() != self.image_shape() || !all_finite(values.as_slice()) {
            return Err(Error::InvalidMeasurements(
                "background must be finite and match the image shape".into(),
            ));
        }
        self.background = Some(FrameArray::Shared(values));
        self.preprocessing.subtract_background = true;
        Ok(self)
    }

    pub fn with_per_frame_background(mut self, background: Array3<f64>) -> Result<Self> {
        let values = StandardArray3::try_from(background)?;
        if values.dim() != self.data.dim() || !all_finite(values.as_slice()) {
            return Err(Error::InvalidMeasurements(
                "per-frame background must be finite and match the measurement stack".into(),
            ));
        }
        self.background = Some(FrameArray::PerFrame(values));
        self.preprocessing.subtract_background = true;
        Ok(self)
    }

    pub fn with_masks(mut self, masks: Array2<u8>) -> Result<Self> {
        let values = StandardArray2::try_from(masks)?;
        if values.dim() != self.image_shape() || values.as_slice().iter().any(|&value| value > 1) {
            return Err(Error::InvalidMeasurements(
                "mask must be binary and match the image shape".into(),
            ));
        }
        self.masks = Some(FrameArray::Shared(values));
        Ok(self)
    }

    pub fn with_per_frame_masks(mut self, masks: Array3<u8>) -> Result<Self> {
        let values = StandardArray3::try_from(masks)?;
        if values.dim() != self.data.dim() || values.as_slice().iter().any(|&value| value > 1) {
            return Err(Error::InvalidMeasurements(
                "per-frame masks must be binary and match the measurement stack".into(),
            ));
        }
        self.masks = Some(FrameArray::PerFrame(values));
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

    pub fn with_preprocessing(mut self, preprocessing: PreprocessingConfig) -> Result<Self> {
        self.preprocessing = preprocessing;
        self.validate()?;
        Ok(self)
    }

    pub fn apply_preprocessing(mut self) -> Result<Self> {
        self.validate()?;
        let frame_len = self.frame_len();
        let dark = self.dark_frame.as_ref().map(StandardArray2::as_slice);
        let flat = self.flat_field.as_ref().map(StandardArray2::as_slice);
        let preprocessing = self.preprocessing.clone();
        let metadata = &self.frame_metadata;
        let background = self.background.as_ref();
        let data = self.data.as_slice_mut();
        for (frame_index, frame) in data.chunks_exact_mut(frame_len).enumerate() {
            let exposure = metadata[frame_index].exposure_time;
            let frame_background = background.map(|values| values.frame(frame_index, frame_len));
            for (pixel, value) in frame.iter_mut().enumerate() {
                if preprocessing.subtract_dark
                    && let Some(dark) = dark
                {
                    *value -= dark[pixel];
                }
                if preprocessing.subtract_background
                    && let Some(background) = frame_background
                {
                    *value -= background[pixel];
                }
                if preprocessing.divide_flat_field
                    && let Some(flat) = flat
                {
                    *value /= flat[pixel];
                }
                if preprocessing.normalize_exposure {
                    *value /= exposure;
                }
                if preprocessing.clamp_negative {
                    *value = value.max(0.0);
                }
            }
        }
        Ok(self)
    }

    pub(crate) fn dark_frame_slice(&self) -> Option<&[f64]> {
        self.dark_frame.as_ref().map(StandardArray2::as_slice)
    }

    pub(crate) fn flat_field_slice(&self) -> Option<&[f64]> {
        self.flat_field.as_ref().map(StandardArray2::as_slice)
    }

    pub(crate) fn background_slice(&self) -> Option<&[f64]> {
        self.background.as_ref().map(FrameArray::as_slice)
    }

    pub(crate) fn masks_slice(&self) -> Option<&[u8]> {
        self.masks.as_ref().map(FrameArray::as_slice)
    }

    fn check_frame(&self, index: usize) -> Result<()> {
        if index >= self.frame_count() {
            Err(Error::FrameOutOfRange {
                index,
                frames: self.frame_count(),
            })
        } else {
            Ok(())
        }
    }

    fn validate_frame_array<T>(&self, values: Option<&FrameArray<T>>, name: &str) -> Result<()> {
        let Some(values) = values else {
            return Ok(());
        };
        let valid = match values {
            FrameArray::Shared(values) => values.dim() == self.image_shape(),
            FrameArray::PerFrame(values) => values.dim() == self.data.dim(),
        };
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidMeasurements(format!(
                "{name} dimensions do not match the measurement stack"
            )))
        }
    }

    fn frame_array_from_flat<T>(&self, values: Vec<T>, name: &str) -> Result<FrameArray<T>> {
        if values.len() == self.frame_len() {
            Ok(FrameArray::Shared(StandardArray2::from_shape_vec(
                self.image_shape(),
                values,
            )?))
        } else if values.len() == self.data.len() {
            Ok(FrameArray::PerFrame(StandardArray3::from_shape_vec(
                self.data.dim(),
                values,
            )?))
        } else {
            Err(Error::InvalidMeasurements(format!(
                "{name} length must match one image or the complete frame stack"
            )))
        }
    }
}

impl Serialize for MeasurementStack {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Representation<'a> {
            data: &'a [f64],
            image_shape: (usize, usize),
            frames: usize,
            frame_metadata: &'a [FrameMetadata],
            dark_frame: Option<&'a [f64]>,
            flat_field: Option<&'a [f64]>,
            background: Option<&'a [f64]>,
            masks: Option<&'a [u8]>,
            preprocessing: &'a PreprocessingConfig,
        }

        Representation {
            data: self.data.as_slice(),
            image_shape: self.image_shape(),
            frames: self.frame_count(),
            frame_metadata: &self.frame_metadata,
            dark_frame: self.dark_frame_slice(),
            flat_field: self.flat_field_slice(),
            background: self.background_slice(),
            masks: self.masks_slice(),
            preprocessing: &self.preprocessing,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MeasurementStack {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
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
        let expected = checked_len_3d((
            representation.frames,
            representation.image_shape.0,
            representation.image_shape.1,
        ))
        .map_err(D::Error::custom)?;
        if representation.data.len() != expected {
            return Err(D::Error::custom(
                "measurement data length does not match shape",
            ));
        }
        let mut stack = Self::from_vec(
            representation.data,
            representation.image_shape,
            representation.frame_metadata,
        )
        .map_err(D::Error::custom)?;
        if stack.frame_count() != representation.frames {
            return Err(D::Error::custom("measurement frame count is inconsistent"));
        }
        stack.dark_frame = representation
            .dark_frame
            .map(|values| StandardArray2::from_shape_vec(stack.image_shape(), values))
            .transpose()
            .map_err(D::Error::custom)?;
        stack.flat_field = representation
            .flat_field
            .map(|values| StandardArray2::from_shape_vec(stack.image_shape(), values))
            .transpose()
            .map_err(D::Error::custom)?;
        stack.background = representation
            .background
            .map(|values| stack.frame_array_from_flat(values, "background"))
            .transpose()
            .map_err(D::Error::custom)?;
        stack.masks = representation
            .masks
            .map(|values| stack.frame_array_from_flat(values, "mask"))
            .transpose()
            .map_err(D::Error::custom)?;
        stack.preprocessing = representation.preprocessing;
        stack.validate().map_err(D::Error::custom)?;
        Ok(stack)
    }
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
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
    if image.dim() != expected_shape {
        return Err(Error::InvalidMeasurements(format!(
            "image {} has shape {:?}, expected {expected_shape:?}",
            resolved.display(),
            image.dim()
        )));
    }
    Ok(image.into_iter().collect())
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
            let frame_len = checked_len_2d(expected_shape)?;
            let total = frame_len
                .checked_mul(frame_count)
                .ok_or_else(|| Error::ShapeOverflow {
                    shape: vec![frame_count, expected_shape.0, expected_shape.1],
                })?;
            let mut values = Vec::with_capacity(total);
            for path in paths {
                values.extend(load_manifest_image(base_directory, path, expected_shape)?);
            }
            Ok(values)
        }
    }
}
