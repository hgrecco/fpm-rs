use serde::{Deserialize, Serialize};

use crate::{Result, error::Error};

use super::{LEDArray, Optics};

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
    Coded(CodedIllumination),
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
            Self::Coded(coded) => coded.k_vectors(optics),
        }
    }

    fn frame_gains(&self) -> Result<Option<Vec<f64>>> {
        match self {
            Self::LEDArray(array) => array.frame_gains(),
            Self::KVectors(_) | Self::Angles(_) | Self::Coded(_) => Ok(None),
        }
    }

    fn multiplexing_matrix(&self) -> Result<Option<MultiplexingMatrix>> {
        match self {
            Self::Coded(coded) => coded.multiplexing_matrix(),
            Self::KVectors(_) | Self::Angles(_) | Self::LEDArray(_) => Ok(None),
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

fn validate_multiplexing_matrix(matrix: &MultiplexingMatrix, source_count: usize) -> Result<()> {
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
