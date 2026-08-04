//! Controlled synthetic acquisition for image-plane Fourier ptychography.
//!
//! Construct a [`crate::simulation::SyntheticObject`], configure a
//! [`crate::simulation::Simulator`] with a compiled
//! [`crate::model::ImagePlaneModel`], and optionally add
//! [`crate::simulation::CameraModel`] response or
//! [`crate::simulation::IlluminationAcquisitionErrors`]. Simulation produces a
//! [`crate::simulation::SimulationResult`] with measured intensities and retained ground
//! truth.

mod camera;
mod illumination_acquisition_errors;
pub mod presets;
mod result;
mod simulator;
mod synthetic_object;

pub use camera::CameraModel;
pub use illumination_acquisition_errors::IlluminationAcquisitionErrors;
pub use result::SimulationResult;
pub use simulator::Simulator;
pub use synthetic_object::SyntheticObject;
