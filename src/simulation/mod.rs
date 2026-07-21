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
