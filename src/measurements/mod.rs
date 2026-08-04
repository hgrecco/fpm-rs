//! Low-resolution measured-intensity stacks and acquisition metadata.
//!
//! Use [`crate::measurements::MeasurementStack`] for resident `(frame, row, column)` data
//! or [`crate::measurements::LazyMeasurementStack`] to load frames on demand. Algorithms
//! depend only on the [`crate::measurements::MeasurementRead`] trait, and
//! [`crate::measurements::MeasurementSpec`] describes file-backed inputs.

mod lazy;
mod manifest;
mod metadata;
mod preprocessing;
mod read;
mod stack;

pub use lazy::LazyMeasurementStack;
pub use manifest::{FrameSpec, ImageSet, MeasurementSpec};
pub use metadata::FrameMetadata;
pub use preprocessing::PreprocessingConfig;
pub use read::MeasurementRead;
pub use stack::MeasurementStack;
