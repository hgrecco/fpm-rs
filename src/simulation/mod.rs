mod camera;
mod illumination_acquisition_errors;
mod metrics;
pub mod presets;
mod result;
mod simulator;
mod synthetic_object;

pub use camera::CameraModel;
pub use illumination_acquisition_errors::IlluminationAcquisitionErrors;
pub use metrics::{
    GroundTruthMetrics, compare_to_ground_truth, compare_with_problem, compare_with_true_model,
};
pub use result::SimulationResult;
pub use simulator::Simulator;
pub use synthetic_object::SyntheticObject;
