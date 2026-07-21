//! Atomic, domain-agnostic metrics for comparing scalar intensity images.
//!
//! Inputs use `reference` and `candidate` terminology.  Signed quantities use
//! the residual `candidate - reference`; `valid_mask`, when supplied, includes
//! pixels whose value is `true`.  These functions are for evaluation, not
//! optimization objectives.

use ndarray::ArrayView2;
use num_traits::ToPrimitive;
use thiserror::Error;

const SSIM_WINDOW_SIZE: usize = 11;
const SSIM_GAUSSIAN_SIGMA: f64 = 1.5;
const SSIM_K1: f64 = 0.01;
const SSIM_K2: f64 = 0.03;

/// Errors returned by atomic intensity metrics.
#[derive(Debug, Error, PartialEq)]
pub enum IntensityMetricError {
    #[error("reference and candidate shapes differ: {reference:?} and {candidate:?}")]
    ShapeMismatch {
        reference: (usize, usize),
        candidate: (usize, usize),
    },
    #[error("valid_mask shape {actual:?} does not match image shape {expected:?}")]
    MaskShapeMismatch {
        actual: (usize, usize),
        expected: (usize, usize),
    },
    #[error("at least one valid pixel is required")]
    EmptyValidMask,
    #[error("{input} contains a value that cannot be represented as f64")]
    UnsupportedScalar { input: &'static str },
    #[error("{input} contains a non-finite value")]
    NonFinite { input: &'static str },
    #[error("{metric} is undefined because the reference normalization is zero")]
    ZeroReferenceNormalization { metric: &'static str },
    #[error("correlation is undefined for a constant input")]
    ZeroVariance,
    #[error("{metric} requires non-negative intensities")]
    NegativeIntensity { metric: &'static str },
    #[error("data_range must be finite and strictly positive")]
    InvalidDataRange,
    #[error("epsilon must be finite and strictly positive")]
    InvalidEpsilon,
    #[error(
        "SSIM requires images at least {SSIM_WINDOW_SIZE} by {SSIM_WINDOW_SIZE}, got {shape:?}"
    )]
    SsimImageTooSmall { shape: (usize, usize) },
    #[error(
        "SSIM requires at least one fully valid {SSIM_WINDOW_SIZE} by {SSIM_WINDOW_SIZE} window"
    )]
    NoValidSsimWindow,
}

/// Return the mean signed residual, `candidate - reference`.
pub fn bias<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut sum = 0.0;
    let count = for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        sum += candidate - reference;
    })?;
    Ok(sum / count as f64)
}

/// Return the mean absolute error.
pub fn mae<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut sum = 0.0;
    let count = for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        sum += (candidate - reference).abs();
    })?;
    Ok(sum / count as f64)
}

/// Return the mean squared error.
pub fn mse<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let (sum, count) = squared_error_sum(reference, candidate, valid_mask)?;
    Ok(sum / count as f64)
}

/// Return the root mean squared error.
pub fn rmse<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    Ok(mse(reference, candidate, valid_mask)?.sqrt())
}

/// Return `sum(abs(candidate - reference)) / sum(abs(reference))`.
pub fn relative_l1<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut residual_sum = 0.0;
    let mut reference_sum = 0.0;
    for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        residual_sum += (candidate - reference).abs();
        reference_sum += reference.abs();
    })?;
    if reference_sum == 0.0 {
        return Err(IntensityMetricError::ZeroReferenceNormalization {
            metric: "relative_l1",
        });
    }
    Ok(residual_sum / reference_sum)
}

/// Return `||candidate - reference||_2 / ||reference||_2`.
pub fn nrmse<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut residual_squared = 0.0;
    let mut reference_squared = 0.0;
    for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        residual_squared += (candidate - reference).powi(2);
        reference_squared += reference.powi(2);
    })?;
    if reference_squared == 0.0 {
        return Err(IntensityMetricError::ZeroReferenceNormalization { metric: "nrmse" });
    }
    Ok(residual_squared.sqrt() / reference_squared.sqrt())
}

/// Compare square-root intensities, normalized by the reference amplitude L2 norm.
pub fn amplitude_nrmse<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_non_negative(reference, candidate, valid_mask.clone(), "amplitude_nrmse")?;
    let mut residual_squared = 0.0;
    let mut reference_squared = 0.0;
    for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        let reference_amplitude = reference.sqrt();
        let candidate_amplitude = candidate.sqrt();
        residual_squared += (candidate_amplitude - reference_amplitude).powi(2);
        reference_squared += reference;
    })?;
    if reference_squared == 0.0 {
        return Err(IntensityMetricError::ZeroReferenceNormalization {
            metric: "amplitude_nrmse",
        });
    }
    Ok(residual_squared.sqrt() / reference_squared.sqrt())
}

/// Return the Pearson correlation coefficient of valid pixels.
pub fn correlation<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut reference_sum = 0.0;
    let mut candidate_sum = 0.0;
    let mut reference_squared = 0.0;
    let mut candidate_squared = 0.0;
    let mut product_sum = 0.0;
    let count = for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        reference_sum += reference;
        candidate_sum += candidate;
        reference_squared += reference * reference;
        candidate_squared += candidate * candidate;
        product_sum += reference * candidate;
    })? as f64;
    let reference_variance = reference_squared - reference_sum * reference_sum / count;
    let candidate_variance = candidate_squared - candidate_sum * candidate_sum / count;
    if reference_variance <= 0.0 || candidate_variance <= 0.0 {
        return Err(IntensityMetricError::ZeroVariance);
    }
    let covariance = product_sum - reference_sum * candidate_sum / count;
    Ok(covariance / (reference_variance * candidate_variance).sqrt())
}

/// Return peak signal-to-noise ratio in dB for an explicit intensity range.
pub fn psnr<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    data_range: f64,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_data_range(data_range)?;
    let value = mse(reference, candidate, valid_mask)?;
    if value == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(10.0 * (data_range * data_range / value).log10())
    }
}

/// Return canonical single-scale SSIM using an 11×11 Gaussian window (σ=1.5).
pub fn ssim<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    data_range: f64,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_data_range(data_range)?;
    let shape = reference.dim();
    validate_shapes(
        shape,
        candidate.dim(),
        valid_mask.as_ref().map(|mask| mask.dim()),
    )?;
    if shape.0 < SSIM_WINDOW_SIZE || shape.1 < SSIM_WINDOW_SIZE {
        return Err(IntensityMetricError::SsimImageTooSmall { shape });
    }
    // Validate every included pixel even if the mask leaves no complete SSIM window.
    for_each_valid_pair(reference, candidate, valid_mask.clone(), |_, _| {})?;

    let weights = gaussian_weights();
    let c1 = (SSIM_K1 * data_range).powi(2);
    let c2 = (SSIM_K2 * data_range).powi(2);
    let mut score_sum = 0.0;
    let mut window_count = 0usize;
    for row in 0..=shape.0 - SSIM_WINDOW_SIZE {
        for column in 0..=shape.1 - SSIM_WINDOW_SIZE {
            if valid_mask.as_ref().is_some_and(|mask| {
                (row..row + SSIM_WINDOW_SIZE).any(|window_row| {
                    (column..column + SSIM_WINDOW_SIZE)
                        .any(|window_column| !mask[(window_row, window_column)])
                })
            }) {
                continue;
            }
            let (mut reference_mean, mut candidate_mean) = (0.0, 0.0);
            for window_row in 0..SSIM_WINDOW_SIZE {
                for window_column in 0..SSIM_WINDOW_SIZE {
                    let weight = weights[window_row * SSIM_WINDOW_SIZE + window_column];
                    reference_mean += weight
                        * value_as_f64(
                            &reference[(row + window_row, column + window_column)],
                            "reference",
                        )?;
                    candidate_mean += weight
                        * value_as_f64(
                            &candidate[(row + window_row, column + window_column)],
                            "candidate",
                        )?;
                }
            }
            let (mut reference_variance, mut candidate_variance, mut covariance) = (0.0, 0.0, 0.0);
            for window_row in 0..SSIM_WINDOW_SIZE {
                for window_column in 0..SSIM_WINDOW_SIZE {
                    let weight = weights[window_row * SSIM_WINDOW_SIZE + window_column];
                    let reference_value = value_as_f64(
                        &reference[(row + window_row, column + window_column)],
                        "reference",
                    )? - reference_mean;
                    let candidate_value = value_as_f64(
                        &candidate[(row + window_row, column + window_column)],
                        "candidate",
                    )? - candidate_mean;
                    reference_variance += weight * reference_value * reference_value;
                    candidate_variance += weight * candidate_value * candidate_value;
                    covariance += weight * reference_value * candidate_value;
                }
            }
            score_sum += ((2.0 * reference_mean * candidate_mean + c1) * (2.0 * covariance + c2))
                / ((reference_mean.powi(2) + candidate_mean.powi(2) + c1)
                    * (reference_variance + candidate_variance + c2));
            window_count += 1;
        }
    }
    if window_count == 0 {
        return Err(IntensityMetricError::NoValidSsimWindow);
    }
    Ok(score_sum / window_count as f64)
}

/// Return summed Poisson deviance, flooring candidate intensities at `epsilon`.
pub fn poisson_deviance<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    epsilon: f64,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_epsilon(epsilon)?;
    validate_non_negative(reference, candidate, valid_mask.clone(), "poisson_deviance")?;
    let mut sum = 0.0;
    for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        if reference == 0.0 && candidate == 0.0 {
            return;
        }
        let candidate = candidate.max(epsilon);
        let log_term = if reference == 0.0 {
            0.0
        } else {
            reference * (reference / candidate).ln()
        };
        sum += 2.0 * (candidate - reference + log_term);
    })?;
    Ok(sum)
}

/// Return mean Poisson deviance, flooring candidate intensities at `epsilon`.
pub fn mean_poisson_deviance<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    epsilon: f64,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_epsilon(epsilon)?;
    validate_non_negative(
        reference,
        candidate,
        valid_mask.clone(),
        "mean_poisson_deviance",
    )?;
    let mut sum = 0.0;
    let count = for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        if reference == 0.0 && candidate == 0.0 {
            return;
        }
        let candidate = candidate.max(epsilon);
        let log_term = if reference == 0.0 {
            0.0
        } else {
            reference * (reference / candidate).ln()
        };
        sum += 2.0 * (candidate - reference + log_term);
    })?;
    Ok(sum / count as f64)
}

/// Fit the least-squares scalar `gain` in `candidate ≈ gain × reference`.
pub fn fitted_gain<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        numerator += reference * candidate;
        denominator += reference * reference;
    })?;
    if denominator == 0.0 {
        return Err(IntensityMetricError::ZeroReferenceNormalization {
            metric: "fitted_gain",
        });
    }
    Ok(numerator / denominator)
}

fn squared_error_sum<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<(f64, usize), IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut sum = 0.0;
    let count = for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        sum += (candidate - reference).powi(2);
    })?;
    Ok((sum, count))
}

fn for_each_valid_pair<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    mut function: impl FnMut(f64, f64),
) -> Result<usize, IntensityMetricError>
where
    T: ToPrimitive,
{
    let shape = reference.dim();
    validate_shapes(
        shape,
        candidate.dim(),
        valid_mask.as_ref().map(|mask| mask.dim()),
    )?;
    let mut count = 0usize;
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            if valid_mask.as_ref().is_some_and(|mask| !mask[(row, column)]) {
                continue;
            }
            let reference = value_as_f64(&reference[(row, column)], "reference")?;
            let candidate = value_as_f64(&candidate[(row, column)], "candidate")?;
            function(reference, candidate);
            count += 1;
        }
    }
    if count == 0 {
        return Err(IntensityMetricError::EmptyValidMask);
    }
    Ok(count)
}

fn validate_non_negative<T>(
    reference: ArrayView2<'_, T>,
    candidate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    metric: &'static str,
) -> Result<(), IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut negative = false;
    for_each_valid_pair(reference, candidate, valid_mask, |reference, candidate| {
        negative |= reference < 0.0 || candidate < 0.0;
    })?;
    if negative {
        return Err(IntensityMetricError::NegativeIntensity { metric });
    }
    Ok(())
}

fn validate_shapes(
    reference: (usize, usize),
    candidate: (usize, usize),
    valid_mask: Option<(usize, usize)>,
) -> Result<(), IntensityMetricError> {
    if reference != candidate {
        return Err(IntensityMetricError::ShapeMismatch {
            reference,
            candidate,
        });
    }
    if let Some(actual) = valid_mask
        && actual != reference
    {
        return Err(IntensityMetricError::MaskShapeMismatch {
            actual,
            expected: reference,
        });
    }
    Ok(())
}

fn value_as_f64<T>(value: &T, input: &'static str) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let value = value
        .to_f64()
        .ok_or(IntensityMetricError::UnsupportedScalar { input })?;
    if !value.is_finite() {
        return Err(IntensityMetricError::NonFinite { input });
    }
    Ok(value)
}

fn validate_data_range(data_range: f64) -> Result<(), IntensityMetricError> {
    if !data_range.is_finite() || data_range <= 0.0 {
        return Err(IntensityMetricError::InvalidDataRange);
    }
    Ok(())
}

fn validate_epsilon(epsilon: f64) -> Result<(), IntensityMetricError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(IntensityMetricError::InvalidEpsilon);
    }
    Ok(())
}

fn gaussian_weights() -> Vec<f64> {
    let center = (SSIM_WINDOW_SIZE - 1) as f64 / 2.0;
    let mut weights = Vec::with_capacity(SSIM_WINDOW_SIZE * SSIM_WINDOW_SIZE);
    for row in 0..SSIM_WINDOW_SIZE {
        for column in 0..SSIM_WINDOW_SIZE {
            let squared_distance = (row as f64 - center).powi(2) + (column as f64 - center).powi(2);
            weights.push((-squared_distance / (2.0 * SSIM_GAUSSIAN_SIGMA.powi(2))).exp());
        }
    }
    let normalization = weights.iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= normalization;
    }
    weights
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn residual_metrics_use_candidate_minus_reference_and_support_integers() {
        let reference = array![[1u8, 2u8]];
        let candidate = array![[2u8, 0u8]];
        assert_relative_eq!(
            bias(reference.view(), candidate.view(), None).unwrap(),
            -0.5
        );
        assert_relative_eq!(mae(reference.view(), candidate.view(), None).unwrap(), 1.5);
        assert_relative_eq!(mse(reference.view(), candidate.view(), None).unwrap(), 2.5);
        assert_relative_eq!(
            rmse(reference.view(), candidate.view(), None).unwrap(),
            2.5f64.sqrt()
        );
        assert_relative_eq!(
            relative_l1(reference.view(), candidate.view(), None).unwrap(),
            1.0
        );
        assert_relative_eq!(
            nrmse(reference.view(), candidate.view(), None).unwrap(),
            1.0
        );
        assert_relative_eq!(
            correlation(reference.view(), candidate.view(), None).unwrap(),
            -1.0
        );
    }

    #[test]
    fn mask_selects_valid_pixels() {
        let reference = array![[1.0, 20.0]];
        let candidate = array![[2.0, 0.0]];
        let mask = array![[true, false]];
        assert_relative_eq!(
            bias(reference.view(), candidate.view(), Some(mask.view())).unwrap(),
            1.0
        );
    }

    #[test]
    fn quality_metrics_have_documented_reference_values() {
        let reference = array![[0.0, 1.0]];
        let candidate = array![[0.0, 0.0]];
        assert_relative_eq!(
            psnr(reference.view(), candidate.view(), None, 1.0).unwrap(),
            10.0 * 2.0f64.log10()
        );

        let image = ndarray::Array2::<f64>::ones((11, 11));
        assert_relative_eq!(ssim(image.view(), image.view(), None, 1.0).unwrap(), 1.0);
    }

    #[test]
    fn poisson_deviance_and_gain_are_well_defined() {
        let reference = array![[0.0, 2.0]];
        let candidate = array![[0.0, 2.0]];
        assert_relative_eq!(
            poisson_deviance(reference.view(), candidate.view(), None, 1e-12).unwrap(),
            0.0
        );
        assert_relative_eq!(
            mean_poisson_deviance(reference.view(), candidate.view(), None, 1e-12).unwrap(),
            0.0
        );

        let scaled = array![[0.0, 4.0]];
        assert_relative_eq!(
            fitted_gain(reference.view(), scaled.view(), None).unwrap(),
            2.0
        );
    }

    #[test]
    fn validation_covers_masks_ranges_and_intensity_domains() {
        let reference = array![[1.0, 2.0]];
        let candidate = array![[2.0, 0.0]];
        let wrong_mask = array![[true], [false]];
        assert!(matches!(
            mae(reference.view(), candidate.view(), Some(wrong_mask.view())),
            Err(IntensityMetricError::MaskShapeMismatch { .. })
        ));
        assert!(matches!(
            psnr(reference.view(), candidate.view(), None, 0.0),
            Err(IntensityMetricError::InvalidDataRange)
        ));
        assert!(matches!(
            ssim(reference.view(), candidate.view(), None, 1.0),
            Err(IntensityMetricError::SsimImageTooSmall { .. })
        ));

        let negative = array![[-1.0, 2.0]];
        assert!(matches!(
            amplitude_nrmse(negative.view(), candidate.view(), None),
            Err(IntensityMetricError::NegativeIntensity { .. })
        ));
        assert!(matches!(
            poisson_deviance(negative.view(), candidate.view(), None, 1e-12),
            Err(IntensityMetricError::NegativeIntensity { .. })
        ));
    }
}
