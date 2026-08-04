use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{Complex64, measurements::MeasurementStack, model::ImagePlaneModel};

use super::{CameraModel, IlluminationAcquisitionErrors};

/// Owned simulated measurements with true and assumed models and retained ground truth.
#[derive(Clone, Debug)]
pub struct SimulationResult {
    /// Simulated low-resolution detector intensities or camera counts in frame order.
    pub measurements: MeasurementStack,
    /// True high-resolution complex sample field shaped `(height, width)`.
    pub ground_truth_object: Array2<Complex64>,
    /// Compiled optical model used to generate measurements.
    pub true_model: ImagePlaneModel,
    /// Assumed model supplied to reconstruction, including known linear camera response.
    pub reconstruction_model: ImagePlaneModel,
    /// Detector pipeline, or `None` for direct ideal optical intensities.
    pub camera: Option<CameraModel>,
    /// Injected non-geometric source errors, or `None` when absent.
    pub illumination_acquisition_errors: Option<IlluminationAcquisitionErrors>,
    /// Compact record of the realized simulation dimensions and missing frames.
    pub parameters: SimulationParameters,
    /// Seed used to initialize deterministic pseudorandom effects.
    pub random_seed: u64,
}

/// Serializable summary of a completed simulated acquisition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationParameters {
    /// Whether no camera or acquisition-error effects were requested.
    pub ideal: bool,
    /// Number of acquisition frames in [`SimulationResult::measurements`].
    pub frame_count: usize,
    /// Low-resolution frame shape as `(height, width)`.
    pub image_shape: (usize, usize),
    /// Zero-based acquisition-frame indices whose illumination was forced to zero.
    pub missing_frames: Vec<usize>,
}
