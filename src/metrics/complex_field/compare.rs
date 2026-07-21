//! Metrics comparing reference and estimate complex-valued images.
//!
//! Every metric applies the requested alignment to the estimate without
//! allocating an aligned image. A mask value of `true` includes that pixel in
//! both the alignment fit and metric accumulation.

use ndarray::ArrayView2;
use num_complex::{Complex, Complex64};
use num_traits::{Float, ToPrimitive};
use thiserror::Error;

/// Alignment applied to an estimate before evaluating a complex-image metric.
///
/// Every fitted mode minimizes the complex least-squares residual over the
/// selected pixels. Alignment is always explicit at the call site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComplexAlignment {
    /// Compare the estimate directly with the reference.
    None,
    /// Fit a unit-magnitude complex scalar, removing one global phase offset.
    GlobalPhase,
    /// Fit a non-negative real scalar, preserving phase.
    Scale,
    /// Fit an unconstrained complex scalar, removing scale and global phase.
    ComplexGain,
}

/// Errors returned by explicit-alignment complex-image metrics.
#[derive(Debug, Error, PartialEq)]
pub enum ComplexMetricError {
    #[error("reference and estimate shapes differ: {reference:?} and {estimate:?}")]
    ShapeMismatch {
        reference: (usize, usize),
        estimate: (usize, usize),
    },
    #[error("mask shape {actual:?} does not match image shape {expected:?}")]
    MaskShapeMismatch {
        actual: (usize, usize),
        expected: (usize, usize),
    },
    #[error("at least one selected pixel is required")]
    EmptyMask,
    #[error("{input} contains a component that cannot be represented as f64")]
    UnsupportedScalar { input: &'static str },
    #[error("{input} contains a non-finite real or imaginary component")]
    NonFinite { input: &'static str },
    #[error("{metric} is undefined because the reference normalization is zero")]
    ZeroReferenceNormalization { metric: &'static str },
    #[error("{metric} is undefined because the aligned estimate normalization is zero")]
    ZeroEstimateNormalization { metric: &'static str },
    #[error("{alignment:?} alignment is undefined: {reason}")]
    DegenerateAlignment {
        alignment: ComplexAlignment,
        reason: &'static str,
    },
}

/// Return the arithmetic mean of `aligned_estimate - reference`.
///
/// The selected [`ComplexAlignment`] is fitted over exactly the pixels
/// included by `mask`; `true` includes a pixel.
pub fn bias<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<Complex64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let mut residual_sum = Complex64::default();
    let count = for_each_aligned_pair(
        reference,
        estimate,
        mask,
        alignment,
        |reference, estimate| {
            residual_sum += estimate - reference;
        },
    )?;
    Ok(residual_sum / count as f64)
}

/// Return the mean residual magnitude, `mean(|aligned_estimate - reference|)`.
///
/// `aligned_estimate` uses the requested [`ComplexAlignment`] fitted over the
/// selected pixels.
pub fn mae<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<f64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let mut residual_sum = 0.0;
    let count = for_each_aligned_pair(
        reference,
        estimate,
        mask,
        alignment,
        |reference, estimate| {
            residual_sum += (estimate - reference).norm();
        },
    )?;
    Ok(residual_sum / count as f64)
}

/// Return `mean(|aligned_estimate - reference|²)`.
///
/// `aligned_estimate` uses the requested [`ComplexAlignment`] fitted over the
/// selected pixels.
pub fn mse<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<f64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let (sum, count) = squared_error_sum(reference, estimate, mask, alignment)?;
    Ok(sum / count as f64)
}

/// Return `sqrt(mean(|aligned_estimate - reference|²))`.
///
/// `aligned_estimate` uses the requested [`ComplexAlignment`] fitted over the
/// selected pixels.
pub fn rmse<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<f64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    Ok(mse(reference, estimate, mask, alignment)?.sqrt())
}

/// Return `sum(|aligned_estimate - reference|) / sum(|reference|)`.
///
/// `aligned_estimate` uses the requested [`ComplexAlignment`] fitted over the
/// selected pixels.
pub fn relative_l1<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<f64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let mut residual_sum = 0.0;
    let mut reference_sum = 0.0;
    for_each_aligned_pair(
        reference,
        estimate,
        mask,
        alignment,
        |reference, estimate| {
            residual_sum += (estimate - reference).norm();
            reference_sum += reference.norm();
        },
    )?;
    if reference_sum == 0.0 {
        return Err(ComplexMetricError::ZeroReferenceNormalization {
            metric: "relative_l1",
        });
    }
    Ok(residual_sum / reference_sum)
}

/// Return `||aligned_estimate - reference||₂ / ||reference||₂`.
///
/// `aligned_estimate` uses the requested [`ComplexAlignment`] fitted over the
/// selected pixels.
pub fn nrmse<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<f64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let mut residual_squared = 0.0;
    let mut reference_squared = 0.0;
    for_each_aligned_pair(
        reference,
        estimate,
        mask,
        alignment,
        |reference, estimate| {
            residual_squared += (estimate - reference).norm_sqr();
            reference_squared += reference.norm_sqr();
        },
    )?;
    if reference_squared == 0.0 {
        return Err(ComplexMetricError::ZeroReferenceNormalization { metric: "nrmse" });
    }
    Ok((residual_squared / reference_squared).sqrt())
}

/// Compare magnitudes after alignment, normalized by reference magnitude energy.
///
/// This returns `|| |aligned_estimate| - |reference| ||₂ / ||reference||₂`.
/// The requested [`ComplexAlignment`] is fitted over the selected pixels.
pub fn amplitude_nrmse<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<f64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let mut residual_squared = 0.0;
    let mut reference_squared = 0.0;
    for_each_aligned_pair(
        reference,
        estimate,
        mask,
        alignment,
        |reference, estimate| {
            residual_squared += (estimate.norm() - reference.norm()).powi(2);
            reference_squared += reference.norm_sqr();
        },
    )?;
    if reference_squared == 0.0 {
        return Err(ComplexMetricError::ZeroReferenceNormalization {
            metric: "amplitude_nrmse",
        });
    }
    Ok((residual_squared / reference_squared).sqrt())
}

/// Return normalized complex correlation after alignment.
///
/// The conjugation convention is
/// `sum(conj(reference) * aligned_estimate) /
/// sqrt(sum(|reference|²) * sum(|aligned_estimate|²))`. Consequently, without
/// alignment, an estimate equal to `reference * exp(iθ)` has correlation
/// `exp(iθ)`.
pub fn correlation<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<Complex64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let mut product_sum = Complex64::default();
    let mut reference_squared = 0.0;
    let mut estimate_squared = 0.0;
    for_each_aligned_pair(
        reference,
        estimate,
        mask,
        alignment,
        |reference, estimate| {
            product_sum += reference.conj() * estimate;
            reference_squared += reference.norm_sqr();
            estimate_squared += estimate.norm_sqr();
        },
    )?;
    if reference_squared == 0.0 {
        return Err(ComplexMetricError::ZeroReferenceNormalization {
            metric: "correlation",
        });
    }
    if estimate_squared == 0.0 {
        return Err(ComplexMetricError::ZeroEstimateNormalization {
            metric: "correlation",
        });
    }
    Ok(product_sum / (reference_squared.sqrt() * estimate_squared.sqrt()))
}

fn squared_error_sum<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<(f64, usize), ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let mut sum = 0.0;
    let count = for_each_aligned_pair(
        reference,
        estimate,
        mask,
        alignment,
        |reference, estimate| {
            sum += (estimate - reference).norm_sqr();
        },
    )?;
    Ok((sum, count))
}

fn for_each_aligned_pair<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
    mut function: impl FnMut(Complex64, Complex64),
) -> Result<usize, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let (factor, count) = alignment_factor(reference, estimate, mask, alignment)?;
    let shape = reference.dim();
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            if mask.as_ref().is_some_and(|mask| !mask[(row, column)]) {
                continue;
            }
            let reference = value_as_complex64(reference[(row, column)], "reference")?;
            let estimate = value_as_complex64(estimate[(row, column)], "estimate")? * factor;
            function(reference, estimate);
        }
    }
    Ok(count)
}

fn alignment_factor<T>(
    reference: ArrayView2<'_, Complex<T>>,
    estimate: ArrayView2<'_, Complex<T>>,
    mask: Option<ArrayView2<'_, bool>>,
    alignment: ComplexAlignment,
) -> Result<(Complex64, usize), ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let shape = reference.dim();
    validate_shapes(shape, estimate.dim(), mask.as_ref().map(|mask| mask.dim()))?;

    let mut cross = Complex64::default();
    let mut estimate_energy = 0.0;
    let mut count = 0;
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            if mask.as_ref().is_some_and(|mask| !mask[(row, column)]) {
                continue;
            }
            let reference = value_as_complex64(reference[(row, column)], "reference")?;
            let estimate = value_as_complex64(estimate[(row, column)], "estimate")?;
            if alignment != ComplexAlignment::None {
                cross += estimate.conj() * reference;
                estimate_energy += estimate.norm_sqr();
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err(ComplexMetricError::EmptyMask);
    }

    let factor = match alignment {
        ComplexAlignment::None => Complex64::new(1.0, 0.0),
        ComplexAlignment::GlobalPhase => {
            validate_estimate_energy(estimate_energy, alignment)?;
            let magnitude = cross.norm();
            if magnitude == 0.0 {
                return Err(ComplexMetricError::DegenerateAlignment {
                    alignment,
                    reason: "reference/estimate cross-correlation is zero",
                });
            }
            cross / magnitude
        }
        ComplexAlignment::Scale => {
            validate_estimate_energy(estimate_energy, alignment)?;
            Complex64::new((cross.re / estimate_energy).max(0.0), 0.0)
        }
        ComplexAlignment::ComplexGain => {
            validate_estimate_energy(estimate_energy, alignment)?;
            cross / estimate_energy
        }
    };
    Ok((factor, count))
}

fn validate_estimate_energy(
    estimate_energy: f64,
    alignment: ComplexAlignment,
) -> Result<(), ComplexMetricError> {
    if estimate_energy == 0.0 {
        return Err(ComplexMetricError::DegenerateAlignment {
            alignment,
            reason: "estimate energy is zero",
        });
    }
    Ok(())
}

fn validate_shapes(
    reference: (usize, usize),
    estimate: (usize, usize),
    mask: Option<(usize, usize)>,
) -> Result<(), ComplexMetricError> {
    if reference != estimate {
        return Err(ComplexMetricError::ShapeMismatch {
            reference,
            estimate,
        });
    }
    if let Some(actual) = mask
        && actual != reference
    {
        return Err(ComplexMetricError::MaskShapeMismatch {
            actual,
            expected: reference,
        });
    }
    Ok(())
}

fn value_as_complex64<T>(
    value: Complex<T>,
    input: &'static str,
) -> Result<Complex64, ComplexMetricError>
where
    T: Float + ToPrimitive,
{
    let real = value
        .re
        .to_f64()
        .ok_or(ComplexMetricError::UnsupportedScalar { input })?;
    let imaginary = value
        .im
        .to_f64()
        .ok_or(ComplexMetricError::UnsupportedScalar { input })?;
    if !real.is_finite() || !imaginary.is_finite() {
        return Err(ComplexMetricError::NonFinite { input });
    }
    Ok(Complex64::new(real, imaginary))
}
