//! Physical optics and illumination descriptions for an image-plane experiment.
//!
//! [`crate::experiment::SourceGeometry`] describes physical sources,
//! [`crate::experiment::SourceCalibration`] stores stable source power, and
//! [`crate::experiment::AcquisitionPlan`] describes sparse source-to-frame
//! acquisition. A complete [`crate::experiment::Illumination`] resolves atomically with
//! [`crate::experiment::Optics`]
//! before compilation into the algorithm-facing [`crate::model::ImagePlaneModel`].

mod illumination;
mod led_array;
mod optics;
mod spectral;
mod spherical;

pub use illumination::{
    AcquisitionPlan, DirectionList, Illumination, IlluminationFrame, KVector, KVectorList,
    MultiplexingMatrix, ResolvedFrame, ResolvedIllumination, ResolvedSources, SourceCalibration,
    SourceContribution, SourceGeometry, SourcePositionList, SourceWeight,
};
pub use led_array::{ArrayPose, PlanarLedArray};
pub use optics::{Optics, PupilAberration};
pub use spectral::{
    SpectralAcquisitionPlan, SpectralChannel, SpectralContribution, SpectralFrame, SpectralGeometry,
};
pub use spherical::{RotatingLedArc, SphericalLedArm, SphericalLedArray};
