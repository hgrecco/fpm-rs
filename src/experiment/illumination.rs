use serde::{Deserialize, Serialize};

use crate::{Result, error::Error};

use super::{LEDArray, LEDSphere, Optics, RotatingLEDArc, SphericalLEDArm};

pub type SourceWeight = (usize, f64);
pub type MultiplexingMatrix = Vec<Vec<SourceWeight>>;

/// Transverse illumination wave vector in radians per metre.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KVector {
    pub kx: f64,
    pub ky: f64,
}

impl KVector {
    pub fn new(kx: f64, ky: f64) -> Self {
        Self { kx, ky }
    }
}

pub trait IlluminationSource {
    fn k_vectors(&self, optics: &Optics) -> Result<Vec<KVector>>;

    /// Optional multiplicative intensity gain for each compiled frame.
    fn frame_gains(&self) -> Result<Option<Vec<f64>>> {
        Ok(None)
    }

    /// Optional incoherent source weights for each measured frame.
    fn multiplexing_matrix(&self) -> Result<Option<MultiplexingMatrix>> {
        Ok(None)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AngleList {
    /// Illumination angles `(theta_x, theta_y)` in radians.
    pub angles: Vec<(f64, f64)>,
}

impl AngleList {
    pub fn new(angles: Vec<(f64, f64)>) -> Self {
        Self { angles }
    }
}

impl IlluminationSource for AngleList {
    fn k_vectors(&self, optics: &Optics) -> Result<Vec<KVector>> {
        optics.validate()?;
        let k = optics.medium_wavenumber();
        let vectors = self
            .angles
            .iter()
            .map(|&(theta_x, theta_y)| {
                if !theta_x.is_finite()
                    || !theta_y.is_finite()
                    || theta_x.abs() > std::f64::consts::FRAC_PI_2
                    || theta_y.abs() > std::f64::consts::FRAC_PI_2
                {
                    return Err(Error::InvalidParameter {
                        name: "illumination angles",
                        reason: "angles must be finite and between -pi/2 and pi/2".into(),
                    });
                }
                Ok(KVector::new(k * theta_x.sin(), k * theta_y.sin()))
            })
            .collect::<Result<Vec<_>>>()?;
        validate_k_vectors(&vectors, optics)?;
        Ok(vectors)
    }
}

/// Reserved representation for known linear combinations of source frames.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CodedIllumination {
    pub source_k_vectors: Vec<KVector>,
    pub frame_weights: MultiplexingMatrix,
}

impl CodedIllumination {
    pub fn validate(&self, optics: &Optics) -> Result<()> {
        validate_k_vectors(&self.source_k_vectors, optics)?;
        validate_multiplexing_matrix(&self.frame_weights, self.source_k_vectors.len())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Illumination {
    KVectors(Vec<KVector>),
    Angles(AngleList),
    LEDArray(LEDArray),
    LEDSphere(LEDSphere),
    SphericalLEDArm(SphericalLEDArm),
    RotatingLEDArc(RotatingLEDArc),
    Coded(CodedIllumination),
    /// Imported/calibrated source vectors with optional per-frame gains and
    /// optional incoherent multiplexing rows.
    Calibrated {
        k_vectors: Vec<KVector>,
        frame_gains: Option<Vec<f64>>,
        frame_weights: Option<MultiplexingMatrix>,
    },
}

impl IlluminationSource for Illumination {
    fn k_vectors(&self, optics: &Optics) -> Result<Vec<KVector>> {
        match self {
            Self::KVectors(vectors) => {
                validate_k_vectors(vectors, optics)?;
                Ok(vectors.clone())
            }
            Self::Angles(angles) => angles.k_vectors(optics),
            Self::LEDArray(array) => array.k_vectors(optics),
            Self::LEDSphere(sphere) => sphere.k_vectors(optics),
            Self::SphericalLEDArm(arm) => arm.k_vectors(optics),
            Self::RotatingLEDArc(arc) => arc.k_vectors(optics),
            Self::Coded(coded) => coded.k_vectors(optics),
            Self::Calibrated {
                k_vectors,
                frame_gains,
                frame_weights,
            } => {
                validate_calibrated(
                    k_vectors,
                    frame_gains.as_deref(),
                    frame_weights.as_deref(),
                    optics,
                )?;
                Ok(k_vectors.clone())
            }
        }
    }

    fn frame_gains(&self) -> Result<Option<Vec<f64>>> {
        match self {
            Self::LEDArray(array) => array.frame_gains(),
            Self::LEDSphere(sphere) => sphere.frame_gains(),
            Self::SphericalLEDArm(arm) => arm.frame_gains(),
            Self::RotatingLEDArc(arc) => arc.frame_gains(),
            Self::Calibrated {
                k_vectors,
                frame_gains,
                frame_weights,
            } => {
                let frame_count = frame_weights.as_ref().map_or(k_vectors.len(), Vec::len);
                validate_frame_gains(frame_gains.as_deref(), frame_count)?;
                Ok(frame_gains.clone())
            }
            Self::KVectors(_) | Self::Angles(_) | Self::Coded(_) => Ok(None),
        }
    }

    fn multiplexing_matrix(&self) -> Result<Option<MultiplexingMatrix>> {
        match self {
            Self::Coded(coded) => coded.multiplexing_matrix(),
            Self::Calibrated {
                k_vectors,
                frame_gains,
                frame_weights,
            } => {
                let frame_count = frame_weights.as_ref().map_or(k_vectors.len(), Vec::len);
                validate_frame_gains(frame_gains.as_deref(), frame_count)?;
                if let Some(matrix) = frame_weights {
                    validate_multiplexing_matrix(matrix, k_vectors.len())?;
                }
                Ok(frame_weights.clone())
            }
            Self::KVectors(_)
            | Self::Angles(_)
            | Self::LEDArray(_)
            | Self::LEDSphere(_)
            | Self::SphericalLEDArm(_)
            | Self::RotatingLEDArc(_) => Ok(None),
        }
    }
}

impl IlluminationSource for CodedIllumination {
    fn k_vectors(&self, optics: &Optics) -> Result<Vec<KVector>> {
        self.validate(optics)?;
        Ok(self.source_k_vectors.clone())
    }

    fn multiplexing_matrix(&self) -> Result<Option<MultiplexingMatrix>> {
        validate_multiplexing_matrix(&self.frame_weights, self.source_k_vectors.len())?;
        Ok(Some(self.frame_weights.clone()))
    }
}

impl IlluminationSource for Vec<KVector> {
    fn k_vectors(&self, optics: &Optics) -> Result<Vec<KVector>> {
        validate_k_vectors(self, optics)?;
        Ok(self.clone())
    }
}

fn validate_k_vectors(vectors: &[KVector], optics: &Optics) -> Result<()> {
    optics.validate()?;
    if vectors.is_empty() {
        return Err(Error::InvalidParameter {
            name: "k_vectors",
            reason: "at least one illumination source is required".into(),
        });
    }
    let maximum_transverse = optics.medium_wavenumber() * (1.0 + 128.0 * f64::EPSILON);
    if vectors.iter().any(|vector| {
        !vector.kx.is_finite()
            || !vector.ky.is_finite()
            || vector.kx.hypot(vector.ky) > maximum_transverse
    }) {
        return Err(Error::InvalidParameter {
            name: "k_vectors",
            reason: "vectors must be finite propagating transverse wave vectors with magnitude no greater than the medium wavenumber"
                .into(),
        });
    }
    Ok(())
}

fn validate_multiplexing_matrix(matrix: &[Vec<SourceWeight>], source_count: usize) -> Result<()> {
    if matrix.is_empty() {
        return Err(Error::InvalidParameter {
            name: "frame_weights",
            reason: "at least one measured frame is required".into(),
        });
    }
    for (frame, row) in matrix.iter().enumerate() {
        let mut seen = vec![false; source_count];
        if row.is_empty()
            || row.iter().any(|&(source, weight)| {
                let duplicate = source < source_count && seen[source];
                if source < source_count {
                    seen[source] = true;
                }
                source >= source_count || !weight.is_finite() || weight <= 0.0 || duplicate
            })
        {
            return Err(Error::InvalidParameter {
                name: "frame_weights",
                reason: format!(
                    "frame {frame} must contain unique valid source indices with finite positive weights"
                ),
            });
        }
    }
    Ok(())
}

fn validate_calibrated(
    k_vectors: &[KVector],
    frame_gains: Option<&[f64]>,
    frame_weights: Option<&[Vec<SourceWeight>]>,
    optics: &Optics,
) -> Result<()> {
    validate_k_vectors(k_vectors, optics)?;
    if let Some(matrix) = frame_weights {
        validate_multiplexing_matrix(matrix, k_vectors.len())?;
    }
    validate_frame_gains(
        frame_gains,
        frame_weights.map_or(k_vectors.len(), |matrix| matrix.len()),
    )
}

fn validate_frame_gains(frame_gains: Option<&[f64]>, frame_count: usize) -> Result<()> {
    if frame_gains.is_some_and(|gains| {
        gains.len() != frame_count
            || gains
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
    }) {
        return Err(Error::InvalidParameter {
            name: "frame_gains",
            reason: format!("must contain {frame_count} finite positive values"),
        });
    }
    Ok(())
}
