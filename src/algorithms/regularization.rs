use num_complex::Complex64;

use crate::{Result, error::Error};

pub(crate) fn apply_complex_tv_step(
    values: &mut [Complex64],
    shape: (usize, usize),
    weight: f64,
    epsilon: f64,
    gradient: &mut Vec<Complex64>,
) -> Result<()> {
    validate_inputs(values, shape, weight, gradient)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "object_tv_epsilon",
            reason: "must be finite and positive".into(),
        });
    }
    if weight == 0.0 {
        return Ok(());
    }
    gradient.fill(Complex64::default());
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            let pixel = row * shape.1 + column;
            let horizontal = (column + 1 < shape.1).then(|| values[pixel + 1] - values[pixel]);
            let vertical = (row + 1 < shape.0).then(|| values[pixel + shape.1] - values[pixel]);
            let norm = (horizontal.map_or(0.0, |value| value.norm_sqr())
                + vertical.map_or(0.0, |value| value.norm_sqr())
                + epsilon * epsilon)
                .sqrt();
            if let Some(difference) = horizontal {
                let normalized = difference / norm;
                gradient[pixel] -= normalized;
                gradient[pixel + 1] += normalized;
            }
            if let Some(difference) = vertical {
                let normalized = difference / norm;
                gradient[pixel] -= normalized;
                gradient[pixel + shape.1] += normalized;
            }
        }
    }
    for (value, &derivative) in values.iter_mut().zip(gradient.iter()) {
        *value -= weight * derivative;
    }
    validate_output(values)
}

pub(crate) fn apply_quadratic_smoothing_step(
    values: &mut [Complex64],
    shape: (usize, usize),
    weight: f64,
    gradient: &mut Vec<Complex64>,
) -> Result<()> {
    validate_inputs(values, shape, weight, gradient)?;
    if weight == 0.0 {
        return Ok(());
    }
    gradient.fill(Complex64::default());
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            let pixel = row * shape.1 + column;
            if column + 1 < shape.1 {
                let difference = values[pixel + 1] - values[pixel];
                gradient[pixel] -= 2.0 * difference;
                gradient[pixel + 1] += 2.0 * difference;
            }
            if row + 1 < shape.0 {
                let difference = values[pixel + shape.1] - values[pixel];
                gradient[pixel] -= 2.0 * difference;
                gradient[pixel + shape.1] += 2.0 * difference;
            }
        }
    }
    for (value, &derivative) in values.iter_mut().zip(gradient.iter()) {
        *value -= weight * derivative;
    }
    validate_output(values)
}

fn validate_inputs(
    values: &[Complex64],
    shape: (usize, usize),
    weight: f64,
    gradient: &mut Vec<Complex64>,
) -> Result<()> {
    let expected = shape
        .0
        .checked_mul(shape.1)
        .ok_or_else(|| Error::InvalidShape(format!("shape {shape:?} overflows")))?;
    if shape.0 == 0 || shape.1 == 0 || values.len() != expected {
        return Err(Error::LengthMismatch {
            actual: values.len(),
            expected,
            shape,
        });
    }
    if !weight.is_finite() || weight < 0.0 {
        return Err(Error::InvalidParameter {
            name: "regularization weight",
            reason: "must be finite and non-negative".into(),
        });
    }
    gradient.resize(expected, Complex64::default());
    Ok(())
}

fn validate_output(values: &[Complex64]) -> Result<()> {
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        Err(Error::Numerical(
            "regularization produced a non-finite value".into(),
        ))
    } else {
        Ok(())
    }
}
