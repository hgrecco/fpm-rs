use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::{IlluminationSource, KVector, Optics};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LEDArray {
    pub grid_shape: (usize, usize),
    pub pitch: f64,
    pub distance: f64,
    /// LED-grid coordinate that lies on the optical axis, `(column, row)`.
    pub center: (f64, f64),
    pub wavelength_override: Option<f64>,
    pub illumination_order: Option<Vec<usize>>,
    pub intensity_weights: Option<Vec<f64>>,
    pub rotation_radians: f64,
}

impl Default for LEDArray {
    fn default() -> Self {
        Self {
            grid_shape: (1, 1),
            pitch: 4e-3,
            distance: 90e-3,
            center: (0.0, 0.0),
            wavelength_override: None,
            illumination_order: None,
            intensity_weights: None,
            rotation_radians: 0.0,
        }
    }
}

impl LEDArray {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn grid_shape(mut self, shape: (usize, usize)) -> Self {
        self.grid_shape = shape;
        self
    }

    pub fn pitch(mut self, pitch: f64) -> Self {
        self.pitch = pitch;
        self
    }

    pub fn distance(mut self, distance: f64) -> Self {
        self.distance = distance;
        self
    }

    pub fn center(mut self, center: (f64, f64)) -> Self {
        self.center = center;
        self
    }

    pub fn wavelength_override(mut self, wavelength: f64) -> Self {
        self.wavelength_override = Some(wavelength);
        self
    }

    pub fn illumination_order(mut self, order: Vec<usize>) -> Self {
        self.illumination_order = Some(order);
        self
    }

    pub fn intensity_weights(mut self, weights: Vec<f64>) -> Self {
        self.intensity_weights = Some(weights);
        self
    }

    pub fn rotation_deg(mut self, degrees: f64) -> Self {
        self.rotation_radians = degrees.to_radians();
        self
    }

    pub fn validate(&self) -> Result<()> {
        let count = self
            .grid_shape
            .0
            .checked_mul(self.grid_shape.1)
            .ok_or_else(|| Error::InvalidParameter {
                name: "grid_shape",
                reason: "LED count overflows".into(),
            })?;
        if count == 0 {
            return Err(Error::InvalidParameter {
                name: "grid_shape",
                reason: "dimensions must be non-zero".into(),
            });
        }
        for (name, value) in [("pitch", self.pitch), ("distance", self.distance)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidParameter {
                    name,
                    reason: "must be finite and positive".into(),
                });
            }
        }
        if !self.center.0.is_finite()
            || !self.center.1.is_finite()
            || !self.rotation_radians.is_finite()
        {
            return Err(Error::InvalidParameter {
                name: "LED geometry",
                reason: "center and rotation must be finite".into(),
            });
        }
        if let Some(order) = &self.illumination_order {
            if order.len() != count || order.iter().any(|&index| index >= count) {
                return Err(Error::InvalidParameter {
                    name: "illumination_order",
                    reason: format!("must contain {count} valid indices"),
                });
            }
            let mut sorted = order.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != count {
                return Err(Error::InvalidParameter {
                    name: "illumination_order",
                    reason: "indices must form a permutation".into(),
                });
            }
        }
        if self.intensity_weights.as_ref().is_some_and(|weights| {
            weights.len() != count
                || weights
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
        }) {
            return Err(Error::InvalidParameter {
                name: "intensity_weights",
                reason: format!("must contain {count} finite positive values"),
            });
        }
        Ok(())
    }
}

impl IlluminationSource for LEDArray {
    fn k_vectors(&self, optics: &Optics) -> Result<Vec<KVector>> {
        self.validate()?;
        optics.validate()?;
        let wavelength = self.wavelength_override.unwrap_or(optics.wavelength);
        if !wavelength.is_finite() || wavelength <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "wavelength_override",
                reason: "must be finite and positive".into(),
            });
        }
        let wavenumber = std::f64::consts::TAU * optics.medium_index / wavelength;
        let (sin_rotation, cos_rotation) = self.rotation_radians.sin_cos();
        let count = self
            .grid_shape
            .0
            .checked_mul(self.grid_shape.1)
            .ok_or_else(|| Error::InvalidParameter {
                name: "grid_shape",
                reason: "LED count overflows".into(),
            })?;
        let mut natural = Vec::with_capacity(count);
        for row in 0..self.grid_shape.0 {
            for column in 0..self.grid_shape.1 {
                let x = (column as f64 - self.center.0) * self.pitch;
                let y = (row as f64 - self.center.1) * self.pitch;
                let rotated_x = cos_rotation * x - sin_rotation * y;
                let rotated_y = sin_rotation * x + cos_rotation * y;
                let distance =
                    (rotated_x * rotated_x + rotated_y * rotated_y + self.distance * self.distance)
                        .sqrt();
                natural.push(KVector::new(
                    wavenumber * rotated_x / distance,
                    wavenumber * rotated_y / distance,
                ));
            }
        }
        Ok(if let Some(order) = &self.illumination_order {
            order.iter().map(|&index| natural[index]).collect()
        } else {
            natural
        })
    }

    fn frame_gains(&self) -> Result<Option<Vec<f64>>> {
        self.validate()?;
        Ok(self.intensity_weights.as_ref().map(|weights| {
            self.illumination_order.as_ref().map_or_else(
                || weights.clone(),
                |order| order.iter().map(|&index| weights[index]).collect(),
            )
        }))
    }
}
