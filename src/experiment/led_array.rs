use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::{IlluminationSource, KVector, Optics};

/// Regular planar LED grid centered above the sample.
///
/// The grid shape is `(rows, columns)`, while [`Self::center`] is `(column, row)` in
/// grid-index coordinates. Natural source order is row-major; an optional permutation
/// changes the compiled source order to match acquisition-frame order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LEDArray {
    /// Number of LEDs as `(rows, columns)`; both dimensions must be non-zero.
    pub grid_shape: (usize, usize),
    /// Centre-to-centre spacing between adjacent LEDs, in metres.
    pub pitch: f64,
    /// Positive perpendicular distance from the LED plane to the sample, in metres.
    pub distance: f64,
    /// LED-grid coordinate that lies on the optical axis, `(column, row)`.
    pub center: (f64, f64),
    /// Optional vacuum wavelength in metres, replacing [`Optics::wavelength`] for this array.
    pub wavelength_override: Option<f64>,
    /// Optional permutation from acquisition order to natural row-major LED indices.
    pub illumination_order: Option<Vec<usize>>,
    /// Optional positive multiplicative intensity gain per natural-order LED.
    pub intensity_weights: Option<Vec<f64>>,
    /// Counter-clockwise in-plane grid rotation about its centre, in radians.
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
    /// Creates the default one-LED grid with 4 mm pitch and 90 mm distance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `(rows, columns)`; validation later rejects zero dimensions or count overflow.
    pub fn grid_shape(mut self, shape: (usize, usize)) -> Self {
        self.grid_shape = shape;
        self
    }

    /// Sets the positive LED pitch in metres.
    pub fn pitch(mut self, pitch: f64) -> Self {
        self.pitch = pitch;
        self
    }

    /// Sets the positive LED-plane-to-sample distance in metres.
    pub fn distance(mut self, distance: f64) -> Self {
        self.distance = distance;
        self
    }

    /// Sets the optical-axis grid coordinate as `(column, row)`, including fractional indices.
    pub fn center(mut self, center: (f64, f64)) -> Self {
        self.center = center;
        self
    }

    /// Overrides the vacuum illumination wavelength with a positive value in metres.
    pub fn wavelength_override(mut self, wavelength: f64) -> Self {
        self.wavelength_override = Some(wavelength);
        self
    }

    /// Sets the acquisition-to-natural source permutation.
    pub fn illumination_order(mut self, order: Vec<usize>) -> Self {
        self.illumination_order = Some(order);
        self
    }

    /// Sets one finite positive intensity gain for each natural-order LED.
    pub fn intensity_weights(mut self, weights: Vec<f64>) -> Self {
        self.intensity_weights = Some(weights);
        self
    }

    /// Sets the counter-clockwise in-plane rotation in degrees, stored internally in radians.
    pub fn rotation_deg(mut self, degrees: f64) -> Self {
        self.rotation_radians = degrees.to_radians();
        self
    }

    /// Checks non-zero dimensions, positive finite geometry and wavelength, finite pose,
    /// a complete source permutation, and one finite positive weight per LED.
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
