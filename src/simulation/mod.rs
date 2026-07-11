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
    GroundTruthMetrics, compare_to_ground_truth, compare_to_ground_truth_masked,
    compare_with_problem, compare_with_problem_masked, compare_with_true_model,
    compare_with_true_model_masked, per_frame_residuals,
};
pub use result::SimulationResult;
pub use simulator::Simulator;
pub use synthetic_object::SyntheticObject;
