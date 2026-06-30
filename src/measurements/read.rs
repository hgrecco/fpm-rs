use std::{ops::Deref, sync::Arc};

use crate::Result;

use super::{FrameMetadata, LazyMeasurementStack, MeasurementStack};

/// Read-only measurement access used by reconstruction algorithms.
///
/// Implementations retain responsibility for storage, caching, preprocessing,
/// and mutation. The opaque frame view may borrow directly from the
/// implementation or own a shared cache handle.
pub trait MeasurementRead: Sync {
    fn frame_count(&self) -> usize;

    fn image_shape(&self) -> (usize, usize);

    fn frame_len(&self) -> usize;

    fn frame(&self, index: usize) -> Result<impl Deref<Target = [f64]> + '_>;

    fn frame_weight(&self, index: usize) -> Result<f64>;

    fn frame_mask(&self, index: usize) -> Result<Option<&[u8]>>;

    fn frame_metadata(&self) -> &[FrameMetadata];

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
        &self.frame_metadata
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
