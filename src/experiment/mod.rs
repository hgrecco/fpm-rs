mod illumination;
mod led_array;
mod optics;

pub use illumination::{
    AngleList, CodedIllumination, Illumination, IlluminationSource, KVector, MultiplexingMatrix,
    SourceWeight,
};
pub use led_array::LEDArray;
pub use optics::Optics;
mod spherical;
pub use spherical::{LEDSphere, RotatingLEDArc, SphericalLEDArm};
