use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{Array2, Result, error::Error, experiment::Optics};

use super::Sampling;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pupil {
    pub values: Array2<Complex64>,
    pub support: Vec<bool>,
}

impl Pupil {
    pub fn new(values: Array2<Complex64>, support: Vec<bool>) -> Result<Self> {
        if support.len() != values.len() {
            return Err(Error::InvalidShape(format!(
                "pupil support length {} does not match pupil length {}",
                support.len(),
                values.len()
            )));
        }
        if values
            .as_slice()
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::InvalidModel(
                "pupil contains non-finite complex values".into(),
            ));
        }
        Ok(Self { values, support })
    }

    pub fn circular(shape: (usize, usize), sampling: &Sampling, optics: &Optics) -> Result<Self> {
        sampling.validate()?;
        optics.validate()?;
        let cutoff = std::f64::consts::TAU * optics.objective_na / optics.wavelength;
        let medium_k = optics.medium_wavenumber();
        let mut values = Vec::with_capacity(shape.0 * shape.1);
        let mut support = Vec::with_capacity(shape.0 * shape.1);
        let aberration = optics.initial_pupil_aberration.as_ref();
        for row in 0..shape.0 {
            let ky = (row as f64 - (shape.0 / 2) as f64) * sampling.dky;
            for column in 0..shape.1 {
                let kx = (column as f64 - (shape.1 / 2) as f64) * sampling.dkx;
                let radius = kx.hypot(ky);
                let inside = radius <= cutoff;
                support.push(inside);
                if !inside {
                    values.push(Complex64::new(0.0, 0.0));
                    continue;
                }
                let rho = if cutoff > 0.0 { radius / cutoff } else { 0.0 };
                let theta = ky.atan2(kx);
                let mut phase = 0.0;
                if let Some(defocus) = optics.defocus {
                    phase -= defocus * (kx * kx + ky * ky) / (2.0 * medium_k);
                }
                if let Some(aberration) = aberration {
                    phase += aberration.defocus * (2.0 * rho * rho - 1.0);
                    phase += aberration.astigmatism * rho * rho * (2.0 * theta).cos();
                    phase += aberration.coma * (3.0 * rho.powi(3) - 2.0 * rho) * theta.cos();
                    phase += aberration.spherical * (6.0 * rho.powi(4) - 6.0 * rho * rho + 1.0);
                }
                values.push(Complex64::from_polar(1.0, phase));
            }
        }
        Self::new(Array2::from_vec(shape, values)?, support)
    }

    pub fn shape(&self) -> (usize, usize) {
        self.values.shape()
    }

    pub fn apply_support(&mut self) {
        for (value, &inside) in self.values.as_mut_slice().iter_mut().zip(&self.support) {
            if !inside {
                *value = Complex64::new(0.0, 0.0);
            }
        }
    }
}
