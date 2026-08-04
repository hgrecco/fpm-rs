//! Compiled image-plane forward model used by simulation and reconstruction.
//!
//! [`crate::model::ImagePlaneModel`] stores Fourier crop geometry, sampled
//! [`crate::model::Pupil`] transfer, frame gains, and optional multiplexing.
//! [`crate::model::ForwardModel`] applies it to an object spectrum, while
//! [`crate::model::Sampling`] records the real- and Fourier-space conventions.

mod crop;
mod forward;
mod image_plane_fpm;
mod pupil;
mod sampling;

pub use crop::{CropIndices, FourierCrop, FourierOffset};
pub use forward::{ForwardModel, ForwardWorkspace};
pub(crate) use forward::{fftshift_copy, ifftshift_copy};
pub use image_plane_fpm::{ImagePlaneModel, ReconstructionShape};
pub use pupil::Pupil;
pub use sampling::{CoordinateConvention, Sampling};
