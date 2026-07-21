use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{Complex64, measurements::MeasurementStack, model::ImagePlaneModel};

use super::{CameraModel, IlluminationAcquisitionErrors};

#[derive(Clone, Debug)]
pub struct SimulationResult {
    pub measurements: MeasurementStack,
    pub ground_truth_object: Array2<Complex64>,
    pub true_model: ImagePlaneModel,
    pub reconstruction_model: ImagePlaneModel,
    pub camera: Option<CameraModel>,
    pub illumination_acquisition_errors: Option<IlluminationAcquisitionErrors>,
    pub parameters: SimulationParameters,
    pub random_seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationParameters {
    pub ideal: bool,
    pub frame_count: usize,
    pub image_shape: (usize, usize),
    pub missing_frames: Vec<usize>,
}
