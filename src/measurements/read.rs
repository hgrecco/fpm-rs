use std::{ops::Deref, sync::Arc};

use crate::Result;

use super::{FrameMetadata, LazyMeasurementStack, MeasurementStack};

/// Read-only measurement access used by reconstruction algorithms.
///
/// Implementations retain responsibility for storage, caching, preprocessing,
/// and mutation. The opaque frame view may borrow directly from the
/// implementation or own a shared cache handle.
pub trait MeasurementRead: Sync {
    /// Returns the number of acquisition frames.
    fn frame_count(&self) -> usize;

    /// Returns low-resolution image shape as `(height, width)`.
    fn image_shape(&self) -> (usize, usize);

    /// Returns `height * width`, the row-major element count of one frame.
    fn frame_len(&self) -> usize;

    /// Borrows or shares a row-major intensity frame by zero-based acquisition index.
    fn frame(&self, index: usize) -> Result<impl Deref<Target = [f64]> + '_>;

    /// Returns a frame's non-negative reconstruction weight.
    fn frame_weight(&self, index: usize) -> Result<f64>;

    /// Borrows an optional row-major binary validity mask for one frame.
    fn frame_mask(&self, index: usize) -> Result<Option<&[u8]>>;

    /// Borrows metadata in acquisition-frame order.
    fn frame_metadata(&self) -> &[FrameMetadata];

    /// Checks shapes, counts, finite intensities, metadata, weights, and masks.
    fn validate(&self) -> Result<()>;
}

impl MeasurementRead for MeasurementStack {
    fn frame_count(&self) -> usize {
        MeasurementStack::frame_count(self)
    }

    fn image_shape(&self) -> (usize, usize) {
        MeasurementStack::image_shape(self)
    }

    fn frame_len(&self) -> usize {
        MeasurementStack::frame_len(self)
    }

    fn frame(&self, index: usize) -> Result<impl Deref<Target = [f64]> + '_> {
        MeasurementStack::frame(self, index)
    }

    fn frame_weight(&self, index: usize) -> Result<f64> {
        MeasurementStack::frame_weight(self, index)
    }

    fn frame_mask(&self, index: usize) -> Result<Option<&[u8]>> {
        MeasurementStack::frame_mask(self, index)
    }

    fn frame_metadata(&self) -> &[FrameMetadata] {
        MeasurementStack::frame_metadata(self)
    }

    fn validate(&self) -> Result<()> {
        MeasurementStack::validate(self)
    }
}

struct SharedFrame(Arc<Vec<f64>>);

impl Deref for SharedFrame {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl MeasurementRead for LazyMeasurementStack {
    fn frame_count(&self) -> usize {
        LazyMeasurementStack::frame_count(self)
    }

    fn image_shape(&self) -> (usize, usize) {
        LazyMeasurementStack::image_shape(self)
    }

    fn frame_len(&self) -> usize {
        LazyMeasurementStack::frame_len(self)
    }

    fn frame(&self, index: usize) -> Result<impl Deref<Target = [f64]> + '_> {
        LazyMeasurementStack::frame(self, index).map(SharedFrame)
    }

    fn frame_weight(&self, index: usize) -> Result<f64> {
        LazyMeasurementStack::frame_weight(self, index)
    }

    fn frame_mask(&self, index: usize) -> Result<Option<&[u8]>> {
        LazyMeasurementStack::frame_mask(self, index)
    }

    fn frame_metadata(&self) -> &[FrameMetadata] {
        LazyMeasurementStack::frame_metadata(self)
    }

    fn validate(&self) -> Result<()> {
        LazyMeasurementStack::validate(self)
    }
}
