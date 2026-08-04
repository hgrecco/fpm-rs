//! Physical optics and illumination descriptions for an image-plane experiment.
//!
//! Configure [`crate::experiment::Optics`] and a source geometry such as
//! [`crate::experiment::LEDArray`], [`crate::experiment::LEDSphere`], or
//! [`crate::experiment::AngleList`]. Implementations of
//! [`crate::experiment::IlluminationSource`] compile physical coordinates into transverse
//! [`crate::experiment::KVector`] values consumed by
//! [`crate::model::ImagePlaneModel`].

mod illumination;
mod led_array;
mod optics;
mod spherical;

pub use illumination::{
    AngleList, CodedIllumination, Illumination, IlluminationSource, KVector, MultiplexingMatrix,
    SourceWeight,
};
pub use led_array::LEDArray;
pub use optics::{Optics, PupilAberration};
pub use spherical::{LEDSphere, RotatingLEDArc, SphericalLEDArm};
