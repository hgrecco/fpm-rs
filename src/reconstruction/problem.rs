use crate::{Result, error::Error, measurements::MeasurementRead, model::ImagePlaneModel};

/// Validated pairing of measurements with a compiled image-plane model.
///
/// The owned measurement provider and model must agree on acquisition-frame count and
/// low-resolution `(height, width)` shape. Algorithms borrow this value for a run.
#[derive(Clone, Debug)]
pub struct ReconstructionProblem<M> {
    /// Resident or lazy measured-intensity provider.
    pub measurements: M,
    /// Compiled algorithm-facing model with matching frame count and image shape.
    pub model: ImagePlaneModel,
    /// Optional user-facing dataset or run label.
    pub name: Option<String>,
}

impl<M: MeasurementRead> ReconstructionProblem<M> {
    /// Owns and validates `measurements` and `model` as one reconstruction input.
    pub fn new(measurements: M, model: ImagePlaneModel) -> Result<Self> {
        let problem = Self {
            measurements,
            model,
            name: None,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Assigns a descriptive problem name without changing numerical inputs.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Validates the model and measurements, equal frame counts, and equal image shapes.
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.measurements.validate()?;
        if self.measurements.frame_count() != self.model.frame_count() {
            return Err(Error::InvalidModel(format!(
                "{} measurement frames but {} model frames",
                self.measurements.frame_count(),
                self.model.frame_count()
            )));
        }
        if self.measurements.image_shape() != self.model.image_shape {
            return Err(Error::InvalidShape(format!(
                "measurement shape {:?} differs from model image shape {:?}",
                self.measurements.image_shape(),
                self.model.image_shape
            )));
        }
        let mut positive_weight_frames = 0;
        for frame in 0..self.measurements.frame_count() {
            if self.measurements.frame_weight(frame)? > 0.0 {
                positive_weight_frames += 1;
            }
            if self.measurements.frame_weight(frame)? > 0.0
                && self
                    .measurements
                    .frame_mask(frame)?
                    .is_some_and(|mask| mask.iter().all(|&value| value == 0))
            {
                return Err(Error::InvalidMeasurements(format!(
                    "positive-weight frame {frame} has no unmasked pixels"
                )));
            }
        }
        if positive_weight_frames == 0 {
            return Err(Error::InvalidMeasurements(
                "at least one frame must have positive weight".into(),
            ));
        }
        Ok(())
    }
}
