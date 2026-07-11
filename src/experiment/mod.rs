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
