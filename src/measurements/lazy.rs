use std::{
    collections::{HashMap, VecDeque},
    mem::size_of,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use crate::{
    Result,
    error::Error,
    image_io::{
        GrayscaleScaling, grayscale_image_shape, grayscale_tiff_page_shapes, load_grayscale,
        load_grayscale_tiff_page,
    },
};

use super::{
    FrameMetadata, ImagePreprocessingConfig, MeasurementManifest, MeasurementStack,
    stack::{load_manifest_image, load_manifest_image_set, resolve_path},
};

/// File-backed measurement frames decoded, preprocessed, and cached on access.
///
/// Construction reads image headers to validate grayscale type and dimensions,
/// but does not decode measurement pixel buffers. A bounded thread-safe LRU
/// cache keeps the working set out of core while returned shared frame handles
/// remain stable. Cache byte accounting covers frame pixel buffers retained by
/// the cache, not handles that callers keep alive after eviction.
#[derive(Debug)]
pub struct LazyMeasurementStack {
    paths: Vec<PathBuf>,
    page_indices: Vec<Option<usize>>,
    image_shape: (usize, usize),
    frame_metadata: Vec<FrameMetadata>,
    dark_frame: Option<Vec<f64>>,
    flat_field: Option<Vec<f64>>,
    background: Option<Vec<f64>>,
    masks: Option<Vec<u8>>,
    preprocessing: ImagePreprocessingConfig,
    cache_capacity: usize,
    cache_byte_capacity: Option<usize>,
    cache: Mutex<FrameCache>,
}

#[derive(Debug, Default)]
struct FrameCache {
    frames: HashMap<usize, Arc<Vec<f64>>>,
    least_recently_used: VecDeque<usize>,
    bytes: usize,
}

impl FrameCache {
    fn touch(&mut self, index: usize) {
        self.least_recently_used.retain(|&cached| cached != index);
        self.least_recently_used.push_back(index);
    }

    fn evict_least_recently_used(&mut self) -> bool {
        let Some(index) = self.least_recently_used.pop_front() else {
            return false;
        };
        if let Some(frame) = self.frames.remove(&index) {
            self.bytes = self
                .bytes
                .saturating_sub(frame.len().saturating_mul(size_of::<f64>()));
        }
        true
    }
}

impl Clone for LazyMeasurementStack {
    fn clone(&self) -> Self {
        Self {
            paths: self.paths.clone(),
            page_indices: self.page_indices.clone(),
            image_shape: self.image_shape,
            frame_metadata: self.frame_metadata.clone(),
            dark_frame: self.dark_frame.clone(),
            flat_field: self.flat_field.clone(),
            background: self.background.clone(),
            masks: self.masks.clone(),
            preprocessing: self.preprocessing.clone(),
            cache_capacity: self.cache_capacity,
            cache_byte_capacity: self.cache_byte_capacity,
            // Clones intentionally start empty so decoded frames are not
            // duplicated implicitly.
            cache: Mutex::new(FrameCache::default()),
        }
    }
}

impl LazyMeasurementStack {
    pub fn from_image_files<P: AsRef<Path>>(
        paths: &[P],
        frame_metadata: Vec<FrameMetadata>,
    ) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::InvalidMeasurements(
                "at least one image path is required".into(),
            ));
        }
        let paths: Vec<PathBuf> = paths.iter().map(|path| path.as_ref().to_owned()).collect();
        let image_shape = grayscale_image_shape(&paths[0])?;
        for path in &paths[1..] {
            let shape = grayscale_image_shape(path)?;
            if shape != image_shape {
                return Err(Error::InvalidMeasurements(format!(
                    "image {} has shape {shape:?}, expected {image_shape:?}",
                    path.display()
                )));
            }
        }
        let frame_metadata =
            metadata_or_labels(&paths, frame_metadata, |path, _| path.display().to_string());
        let frame_count = paths.len();
        Self::new(paths, vec![None; frame_count], image_shape, frame_metadata)
    }

    /// Opens a multipage TIFF lazily. Page headers are validated up front, but
    /// each page's pixels are decoded only when that frame is requested.
    pub fn from_tiff_stack(
        path: impl AsRef<Path>,
        frame_metadata: Vec<FrameMetadata>,
    ) -> Result<Self> {
        let path = path.as_ref().to_owned();
        let shapes = grayscale_tiff_page_shapes(&path)?;
        let image_shape = *shapes.first().ok_or_else(|| {
            Error::InvalidMeasurements("TIFF stack does not contain an image".into())
        })?;
        if shapes.iter().any(|&shape| shape != image_shape) {
            return Err(Error::InvalidMeasurements(
                "all TIFF pages must have the same dimensions".into(),
            ));
        }
        let paths = vec![path; shapes.len()];
        let frame_metadata = metadata_or_labels(&paths, frame_metadata, |path, index| {
            format!("{}#page={index}", path.display())
        });
        let page_indices = (0..shapes.len()).map(Some).collect();
        Self::new(paths, page_indices, image_shape, frame_metadata)
    }

    /// Loads a manifest while leaving measurement frames file-backed. Small
    /// shared correction images and optional per-frame backgrounds/masks are
    /// loaded once; configured corrections are applied during frame decoding.
    pub fn from_manifest(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let manifest = MeasurementManifest::load(path)?;
        let base_directory = path.parent().unwrap_or_else(|| Path::new("."));
        Self::from_manifest_definition(manifest, base_directory)
    }

    pub fn from_manifest_definition(
        manifest: MeasurementManifest,
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
                stack.frame_count(),
            )?);
        }
        if let Some(mask) = &manifest.mask {
            stack.masks = Some(
                load_manifest_image_set(
                    base_directory,
                    mask,
                    stack.image_shape,
                    stack.frame_count(),
                )?
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect(),
            );
        }
        stack.preprocessing = manifest.preprocessing;
        stack.validate()?;
        Ok(stack)
    }

    fn new(
        paths: Vec<PathBuf>,
        page_indices: Vec<Option<usize>>,
        image_shape: (usize, usize),
        frame_metadata: Vec<FrameMetadata>,
    ) -> Result<Self> {
        let stack = Self {
            paths,
            page_indices,
            image_shape,
            frame_metadata,
            dark_frame: None,
            flat_field: None,
            background: None,
            masks: None,
            preprocessing: ImagePreprocessingConfig::default(),
            cache_capacity: 1,
            cache_byte_capacity: None,
            cache: Mutex::new(FrameCache::default()),
        };
        stack.validate()?;
        Ok(stack)
    }

    pub fn with_dark_frame(mut self, dark: Vec<f64>) -> Result<Self> {
        self.validate_correction("dark frame", &dark, false)?;
        self.dark_frame = Some(dark);
        self.preprocessing.subtract_dark = true;
        self.reset_cache();
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
        self.reset_cache();
        Ok(self)
    }

    pub fn with_background(mut self, background: Vec<f64>) -> Result<Self> {
        self.validate_correction("background", &background, true)?;
        self.background = Some(background);
        self.preprocessing.subtract_background = true;
        self.reset_cache();
        Ok(self)
    }

    pub fn with_masks(mut self, masks: Vec<u8>) -> Result<Self> {
        let frame_len = self.frame_len();
        let stack_len = self.stack_len()?;
        if masks.len() != frame_len && masks.len() != stack_len {
            return Err(Error::InvalidMeasurements(format!(
                "mask length {} must be {frame_len} or {stack_len}",
                masks.len()
            )));
        }
        self.masks = Some(masks);
        Ok(self)
    }

    pub fn normalize_exposure(mut self) -> Self {
        self.preprocessing.normalize_exposure = true;
        self.reset_cache();
        self
    }

    pub fn clamp_negative(mut self) -> Self {
        self.preprocessing.clamp_negative = true;
        self.reset_cache();
        self
    }

    /// Replaces the preprocessing flags. Required correction arrays must have
    /// already been supplied through the corresponding builder.
    pub fn with_preprocessing(mut self, preprocessing: ImagePreprocessingConfig) -> Result<Self> {
        self.preprocessing = preprocessing;
        self.reset_cache();
        self.validate()?;
        Ok(self)
    }

    /// Sets the maximum decoded frames retained in memory. The default is one.
    pub fn with_cache_capacity(mut self, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::InvalidParameter {
                name: "cache_capacity",
                reason: "must be greater than zero".into(),
            });
        }
        self.cache_capacity = capacity;
        self.reset_cache();
        Ok(self)
    }

    /// Sets a second cache limit in bytes. It must fit at least one decoded
    /// `f64` frame. Count and byte limits are enforced simultaneously.
    pub fn with_cache_byte_capacity(mut self, capacity: usize) -> Result<Self> {
        let frame_bytes = self.frame_bytes()?;
        if capacity < frame_bytes {
            return Err(Error::InvalidParameter {
                name: "cache_byte_capacity",
                reason: format!("must be at least one decoded frame ({frame_bytes} bytes)"),
            });
        }
        self.cache_byte_capacity = Some(capacity);
        self.reset_cache();
        Ok(self)
    }

    pub fn frame_count(&self) -> usize {
        self.paths.len()
    }

    pub fn image_shape(&self) -> (usize, usize) {
        self.image_shape
    }

    pub fn frame_len(&self) -> usize {
        self.image_shape.0 * self.image_shape.1
    }

    pub fn frame_metadata(&self) -> &[FrameMetadata] {
        &self.frame_metadata
    }

    pub fn preprocessing(&self) -> &ImagePreprocessingConfig {
        &self.preprocessing
    }

    /// Returns one path per frame. Multipage TIFF frames therefore repeat the
    /// stack path and can be distinguished by their metadata labels.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn cached_frame_count(&self) -> usize {
        self.cache.lock().map_or(0, |cache| cache.frames.len())
    }

    pub fn cached_byte_count(&self) -> usize {
        self.cache.lock().map_or(0, |cache| cache.bytes)
    }

    pub fn frame(&self, index: usize) -> Result<Arc<Vec<f64>>> {
        if index >= self.frame_count() {
            return Err(Error::FrameOutOfRange {
                index,
                frames: self.frame_count(),
            });
        }
        {
            let mut cache = self.lock_cache()?;
            if let Some(frame) = cache.frames.get(&index).cloned() {
                cache.touch(index);
                return Ok(frame);
            }
        }

        // Decoding occurs outside the cache lock so unrelated frame reads can
        // proceed concurrently. The second lookup below resolves races safely.
        let path = &self.paths[index];
        let mut values = match self.page_indices[index] {
            Some(page_index) => {
                load_grayscale_tiff_page(path, page_index, GrayscaleScaling::NativeCounts)?
            }
            None => load_grayscale(path, GrayscaleScaling::NativeCounts)?,
        };
        if values.shape() != self.image_shape {
            return Err(Error::InvalidMeasurements(format!(
                "image {} changed shape to {:?}, expected {:?}",
                path.display(),
                values.shape(),
                self.image_shape
            )));
        }
        self.preprocess_frame(index, values.as_mut_slice())?;
        let loaded = Arc::new(values.into_vec());
        let loaded_bytes = loaded.len().checked_mul(size_of::<f64>()).ok_or_else(|| {
            Error::InvalidMeasurements("decoded lazy frame byte length overflows".into())
        })?;

        let mut cache = self.lock_cache()?;
        if let Some(existing) = cache.frames.get(&index).cloned() {
            cache.touch(index);
            return Ok(existing);
        }
        while cache.frames.len() >= self.cache_capacity
            || self
                .cache_byte_capacity
                .is_some_and(|capacity| cache.bytes.saturating_add(loaded_bytes) > capacity)
        {
            if !cache.evict_least_recently_used() {
                break;
            }
        }
        cache.frames.insert(index, loaded.clone());
        cache.bytes = cache.bytes.checked_add(loaded_bytes).ok_or_else(|| {
            Error::InvalidMeasurements("lazy cache byte accounting overflows".into())
        })?;
        cache.touch(index);
        Ok(loaded)
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
        if index >= self.frame_count() {
            return Err(Error::FrameOutOfRange {
                index,
                frames: self.frame_count(),
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

    pub fn materialize(&self) -> Result<MeasurementStack> {
        let mut data = Vec::with_capacity(self.stack_len()?);
        for frame in 0..self.frame_count() {
            data.extend_from_slice(self.frame(frame)?.as_slice());
        }
        let mut stack =
            MeasurementStack::from_vec(data, self.image_shape, self.frame_metadata.clone())?;
        if let Some(masks) = &self.masks {
            stack = stack.with_masks(masks.clone())?;
        }
        Ok(stack)
    }

    pub fn validate(&self) -> Result<()> {
        if self.paths.is_empty()
            || self.frame_metadata.len() != self.paths.len()
            || self.page_indices.len() != self.paths.len()
            || self.cache_capacity == 0
        {
            return Err(Error::InvalidMeasurements(
                "lazy paths, page indices, and metadata must have equal non-zero counts".into(),
            ));
        }
        let frame_bytes = self.frame_bytes()?;
        for (index, metadata) in self.frame_metadata.iter().enumerate() {
            if metadata.frame_index != index
                || !metadata.exposure_time.is_finite()
                || metadata.exposure_time <= 0.0
                || !metadata.weight.is_finite()
                || metadata.weight < 0.0
            {
                return Err(Error::InvalidMeasurements(format!(
                    "lazy frame metadata {index} is invalid"
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
        if let Some(masks) = &self.masks {
            let frame_len = self.frame_len();
            let stack_len = self.stack_len()?;
            if masks.len() != frame_len && masks.len() != stack_len {
                return Err(Error::InvalidMeasurements(
                    "lazy mask dimensions are invalid".into(),
                ));
            }
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
        if self
            .cache_byte_capacity
            .is_some_and(|capacity| capacity < frame_bytes)
        {
            return Err(Error::InvalidMeasurements(
                "lazy byte cache must fit at least one decoded frame".into(),
            ));
        }
        Ok(())
    }

    fn preprocess_frame(&self, frame_index: usize, values: &mut [f64]) -> Result<()> {
        let frame_len = self.frame_len();
        let exposure = self.frame_metadata[frame_index].exposure_time;
        let dark = if self.preprocessing.subtract_dark {
            Some(self.dark_frame.as_deref().ok_or_else(|| {
                Error::InvalidMeasurements(
                    "dark subtraction requested without a dark frame".into(),
                )
            })?)
        } else {
            None
        };
        let background = if self.preprocessing.subtract_background {
            Some(self.background.as_deref().ok_or_else(|| {
                Error::InvalidMeasurements(
                    "background subtraction requested without a background".into(),
                )
            })?)
        } else {
            None
        };
        let flat = if self.preprocessing.divide_flat_field {
            Some(self.flat_field.as_deref().ok_or_else(|| {
                Error::InvalidMeasurements(
                    "flat-field correction requested without a flat field".into(),
                )
            })?)
        } else {
            None
        };
        for (pixel, value) in values.iter_mut().enumerate() {
            if let Some(dark) = dark {
                *value -= dark[pixel];
            }
            if let Some(background) = background {
                *value -= background[if background.len() == frame_len {
                    pixel
                } else {
                    frame_index * frame_len + pixel
                }];
            }
            if let Some(flat) = flat {
                *value /= flat[pixel];
            }
            if self.preprocessing.normalize_exposure {
                *value /= exposure;
            }
            if self.preprocessing.clamp_negative {
                *value = value.max(0.0);
            }
            if !value.is_finite() {
                return Err(Error::InvalidMeasurements(format!(
                    "preprocessing frame {frame_index} produced a non-finite value at pixel {pixel}"
                )));
            }
        }
        Ok(())
    }

    fn validate_correction(&self, name: &str, values: &[f64], per_frame: bool) -> Result<()> {
        let frame_len = self.frame_len();
        let stack_len = self.stack_len()?;
        let valid_length = values.len() == frame_len || (per_frame && values.len() == stack_len);
        if !valid_length || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidMeasurements(format!(
                "{name} must contain finite values and have length {frame_len}{}",
                if per_frame {
                    format!(" or {stack_len}")
                } else {
                    String::new()
                }
            )));
        }
        Ok(())
    }

    fn frame_bytes(&self) -> Result<usize> {
        self.image_shape
            .0
            .checked_mul(self.image_shape.1)
            .filter(|&length| length > 0)
            .and_then(|length| length.checked_mul(size_of::<f64>()))
            .ok_or_else(|| Error::InvalidShape("lazy measurement shape is invalid".into()))
    }

    fn stack_len(&self) -> Result<usize> {
        self.frame_len()
            .checked_mul(self.frame_count())
            .ok_or_else(|| {
                Error::InvalidMeasurements("lazy measurement stack length overflows".into())
            })
    }

    fn reset_cache(&mut self) {
        self.cache = Mutex::new(FrameCache::default());
    }

    fn lock_cache(&self) -> Result<std::sync::MutexGuard<'_, FrameCache>> {
        self.cache
            .lock()
            .map_err(|_| Error::Numerical("lazy measurement cache lock was poisoned".into()))
    }
}

fn metadata_or_labels(
    paths: &[PathBuf],
    frame_metadata: Vec<FrameMetadata>,
    label: impl Fn(&Path, usize) -> String,
) -> Vec<FrameMetadata> {
    if frame_metadata.is_empty() {
        paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let mut metadata = FrameMetadata::new(index);
                metadata.label = Some(label(path, index));
                metadata
            })
            .collect()
    } else {
        frame_metadata
    }
}
