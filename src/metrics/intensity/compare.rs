//! Domain-agnostic metrics comparing reference and estimate intensity images.
//!
//! Signed quantities use the residual `estimate - reference`; `valid_mask`,
//! when supplied, includes pixels whose value is `true`. These functions are
//! evaluation metrics, not reconstruction optimization objectives.

use ndarray::ArrayView2;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SSIM_WINDOW_SIZE: usize = 11;
const SSIM_GAUSSIAN_SIGMA: f64 = 1.5;
const SSIM_K1: f64 = 0.01;
const SSIM_K2: f64 = 0.03;

/// Errors returned by intensity comparison metrics.
#[derive(Debug, Error, PartialEq)]
pub enum IntensityMetricError {
    #[error("reference and estimate shapes differ: {reference:?} and {estimate:?}")]
    ShapeMismatch {
        reference: (usize, usize),
        estimate: (usize, usize),
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

/// Aggregate residual statistics comparing an estimate with a reference image.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntensityComparisonMetrics {
    pub reference_sum: f64,
    pub estimate_sum: f64,
    pub residual_l1: f64,
    pub residual_l2: f64,
    pub residual_mean: f64,
    pub residual_std: f64,
    pub residual_max_abs: f64,
    pub normalized_l2: f64,
    pub saturated_pixels: Option<usize>,
}

/// Calculate aggregate residual statistics for a reference/estimate pair.
///
/// Signed residuals are `estimate - reference`. `saturation_value`, when
/// present, counts valid reference pixels greater than or equal to that value.
pub fn compare_intensity<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    saturation_value: Option<f64>,
) -> Result<IntensityComparisonMetrics, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut accumulator = ComparisonAccumulator::default();
    let count = for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        accumulator.push(reference, estimate, saturation_value);
    })?;
    Ok(accumulator.finish(count, saturation_value))
}

/// Adapter for the reconstruction diagnostics' flat slices and byte masks.
///
/// This remains crate-private so the public metric API consistently uses
/// two-dimensional ndarray views and Boolean masks.
pub(crate) fn compare_intensity_u8_masked(
    reference: &[f64],
    estimate: &[f64],
    valid_mask: Option<&[u8]>,
    saturation_value: Option<f64>,
) -> crate::Result<IntensityComparisonMetrics> {
    if reference.len() != estimate.len() || reference.is_empty() {
        return Err(crate::Error::InvalidShape(format!(
            "intensity comparison inputs have lengths {} and {}",
            reference.len(),
            estimate.len()
        )));
    }
    if valid_mask.is_some_and(|mask| mask.len() != reference.len()) {
        return Err(crate::Error::LengthMismatch {
            actual: valid_mask.map_or(0, <[u8]>::len),
            expected: reference.len(),
            shape: (1, reference.len()),
        });
    }

    let mut accumulator = ComparisonAccumulator::default();
    let mut count = 0;
    for (index, (&reference, &estimate)) in reference.iter().zip(estimate).enumerate() {
        if valid_mask.is_some_and(|mask| mask[index] == 0) {
            continue;
        }
        if !reference.is_finite() || !estimate.is_finite() {
            return Err(crate::Error::Numerical(
                "intensity comparison contains a non-finite value".into(),
            ));
        }
        accumulator.push(reference, estimate, saturation_value);
        count += 1;
    }
    if count == 0 {
        return Err(crate::Error::InvalidParameter {
            name: "valid_mask",
            reason: "must select at least one pixel".into(),
        });
    }
    Ok(accumulator.finish(count, saturation_value))
}

#[derive(Default)]
struct ComparisonAccumulator {
    reference_sum: f64,
    estimate_sum: f64,
    residual_l1: f64,
    residual_squared: f64,
    residual_sum: f64,
    residual_max_abs: f64,
    reference_squared: f64,
    saturated_pixels: usize,
}

impl ComparisonAccumulator {
    fn push(&mut self, reference: f64, estimate: f64, saturation_value: Option<f64>) {
        let residual = estimate - reference;
        self.reference_sum += reference;
        self.estimate_sum += estimate;
        self.residual_l1 += residual.abs();
        self.residual_squared += residual * residual;
        self.residual_sum += residual;
        self.residual_max_abs = self.residual_max_abs.max(residual.abs());
        self.reference_squared += reference * reference;
        self.saturated_pixels +=
            usize::from(saturation_value.is_some_and(|limit| reference >= limit));
    }

    fn finish(self, count: usize, saturation_value: Option<f64>) -> IntensityComparisonMetrics {
        debug_assert!(count > 0);
        let count = count as f64;
        let residual_mean = self.residual_sum / count;
        IntensityComparisonMetrics {
            reference_sum: self.reference_sum,
            estimate_sum: self.estimate_sum,
            residual_l1: self.residual_l1,
            residual_l2: self.residual_squared.sqrt(),
            residual_mean,
            residual_std: (self.residual_squared / count - residual_mean * residual_mean)
                .max(0.0)
                .sqrt(),
            residual_max_abs: self.residual_max_abs,
            normalized_l2: self.residual_squared.sqrt()
                / (self.reference_squared.sqrt() + f64::EPSILON),
            saturated_pixels: saturation_value.map(|_| self.saturated_pixels),
        }
    }
}

/// Return the mean signed residual, `estimate - reference`.
pub fn bias<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut sum = 0.0;
    let count = for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        sum += estimate - reference;
    })?;
    Ok(sum / count as f64)
}

/// Return the mean absolute error.
pub fn mae<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut sum = 0.0;
    let count = for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        sum += (estimate - reference).abs();
    })?;
    Ok(sum / count as f64)
}

/// Return the mean squared error.
pub fn mse<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let (sum, count) = squared_error_sum(reference, estimate, valid_mask)?;
    Ok(sum / count as f64)
}

/// Return the root mean squared error.
pub fn rmse<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    Ok(mse(reference, estimate, valid_mask)?.sqrt())
}

/// Return `sum(abs(estimate - reference)) / sum(abs(reference))`.
pub fn relative_l1<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut residual_sum = 0.0;
    let mut reference_sum = 0.0;
    for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        residual_sum += (estimate - reference).abs();
        reference_sum += reference.abs();
    })?;
    if reference_sum == 0.0 {
        return Err(IntensityMetricError::ZeroReferenceNormalization {
            metric: "relative_l1",
        });
    }
    Ok(residual_sum / reference_sum)
}

/// Return `||estimate - reference||_2 / ||reference||_2`.
pub fn nrmse<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut residual_squared = 0.0;
    let mut reference_squared = 0.0;
    for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        residual_squared += (estimate - reference).powi(2);
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
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_non_negative(reference, estimate, valid_mask, "amplitude_nrmse")?;
    let mut residual_squared = 0.0;
    let mut reference_squared = 0.0;
    for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        let reference_amplitude = reference.sqrt();
        let estimate_amplitude = estimate.sqrt();
        residual_squared += (estimate_amplitude - reference_amplitude).powi(2);
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
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut reference_sum = 0.0;
    let mut estimate_sum = 0.0;
    let mut reference_squared = 0.0;
    let mut estimate_squared = 0.0;
    let mut product_sum = 0.0;
    let count = for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        reference_sum += reference;
        estimate_sum += estimate;
        reference_squared += reference * reference;
        estimate_squared += estimate * estimate;
        product_sum += reference * estimate;
    })? as f64;
    let reference_variance = reference_squared - reference_sum * reference_sum / count;
    let estimate_variance = estimate_squared - estimate_sum * estimate_sum / count;
    if reference_variance <= 0.0 || estimate_variance <= 0.0 {
        return Err(IntensityMetricError::ZeroVariance);
    }
    let covariance = product_sum - reference_sum * estimate_sum / count;
    Ok(covariance / (reference_variance * estimate_variance).sqrt())
}

/// Return peak signal-to-noise ratio in dB for an explicit intensity range.
pub fn psnr<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    data_range: f64,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_data_range(data_range)?;
    let value = mse(reference, estimate, valid_mask)?;
    if value == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(10.0 * (data_range * data_range / value).log10())
    }
}

/// Return canonical single-scale SSIM using an 11×11 Gaussian window (σ=1.5).
pub fn ssim<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
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
        estimate.dim(),
        valid_mask.as_ref().map(|mask| mask.dim()),
    )?;
    if shape.0 < SSIM_WINDOW_SIZE || shape.1 < SSIM_WINDOW_SIZE {
        return Err(IntensityMetricError::SsimImageTooSmall { shape });
    }
    // Validate every included pixel even if the mask leaves no complete SSIM window.
    for_each_valid_pair(reference, estimate, valid_mask, |_, _| {})?;

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
            let (mut reference_mean, mut estimate_mean) = (0.0, 0.0);
            for window_row in 0..SSIM_WINDOW_SIZE {
                for window_column in 0..SSIM_WINDOW_SIZE {
                    let weight = weights[window_row * SSIM_WINDOW_SIZE + window_column];
                    reference_mean += weight
                        * value_as_f64(
                            &reference[(row + window_row, column + window_column)],
                            "reference",
                        )?;
                    estimate_mean += weight
                        * value_as_f64(
                            &estimate[(row + window_row, column + window_column)],
                            "estimate",
                        )?;
                }
            }
            let (mut reference_variance, mut estimate_variance, mut covariance) = (0.0, 0.0, 0.0);
            for window_row in 0..SSIM_WINDOW_SIZE {
                for window_column in 0..SSIM_WINDOW_SIZE {
                    let weight = weights[window_row * SSIM_WINDOW_SIZE + window_column];
                    let reference_value = value_as_f64(
                        &reference[(row + window_row, column + window_column)],
                        "reference",
                    )? - reference_mean;
                    let estimate_value = value_as_f64(
                        &estimate[(row + window_row, column + window_column)],
                        "estimate",
                    )? - estimate_mean;
                    reference_variance += weight * reference_value * reference_value;
                    estimate_variance += weight * estimate_value * estimate_value;
                    covariance += weight * reference_value * estimate_value;
                }
            }
            score_sum += ((2.0 * reference_mean * estimate_mean + c1) * (2.0 * covariance + c2))
                / ((reference_mean.powi(2) + estimate_mean.powi(2) + c1)
                    * (reference_variance + estimate_variance + c2));
            window_count += 1;
        }
    }
    if window_count == 0 {
        return Err(IntensityMetricError::NoValidSsimWindow);
    }
    Ok(score_sum / window_count as f64)
}

/// Return summed Poisson deviance, flooring estimate intensities at `epsilon`.
pub fn poisson_deviance<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    epsilon: f64,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_epsilon(epsilon)?;
    validate_non_negative(reference, estimate, valid_mask, "poisson_deviance")?;
    let mut sum = 0.0;
    for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        if reference == 0.0 && estimate == 0.0 {
            return;
        }
        let estimate = estimate.max(epsilon);
        let log_term = if reference == 0.0 {
            0.0
        } else {
            reference * (reference / estimate).ln()
        };
        sum += 2.0 * (estimate - reference + log_term);
    })?;
    Ok(sum)
}

/// Return mean Poisson deviance, flooring estimate intensities at `epsilon`.
pub fn mean_poisson_deviance<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    epsilon: f64,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    validate_epsilon(epsilon)?;
    validate_non_negative(reference, estimate, valid_mask, "mean_poisson_deviance")?;
    let mut sum = 0.0;
    let count = for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        if reference == 0.0 && estimate == 0.0 {
            return;
        }
        let estimate = estimate.max(epsilon);
        let log_term = if reference == 0.0 {
            0.0
        } else {
            reference * (reference / estimate).ln()
        };
        sum += 2.0 * (estimate - reference + log_term);
    })?;
    Ok(sum / count as f64)
}

/// Fit the least-squares scalar `gain` in `estimate ≈ gain × reference`.
pub fn fitted_gain<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<f64, IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        numerator += reference * estimate;
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
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
) -> Result<(f64, usize), IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut sum = 0.0;
    let count = for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        sum += (estimate - reference).powi(2);
    })?;
    Ok((sum, count))
}

fn for_each_valid_pair<T>(
    reference: ArrayView2<'_, T>,
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    mut function: impl FnMut(f64, f64),
) -> Result<usize, IntensityMetricError>
where
    T: ToPrimitive,
{
    let shape = reference.dim();
    validate_shapes(
        shape,
        estimate.dim(),
        valid_mask.as_ref().map(|mask| mask.dim()),
    )?;
    let mut count = 0usize;
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            if valid_mask.as_ref().is_some_and(|mask| !mask[(row, column)]) {
                continue;
            }
            let reference = value_as_f64(&reference[(row, column)], "reference")?;
            let estimate = value_as_f64(&estimate[(row, column)], "estimate")?;
            function(reference, estimate);
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
    estimate: ArrayView2<'_, T>,
    valid_mask: Option<ArrayView2<'_, bool>>,
    metric: &'static str,
) -> Result<(), IntensityMetricError>
where
    T: ToPrimitive,
{
    let mut negative = false;
    for_each_valid_pair(reference, estimate, valid_mask, |reference, estimate| {
        negative |= reference < 0.0 || estimate < 0.0;
    })?;
    if negative {
        return Err(IntensityMetricError::NegativeIntensity { metric });
    }
    Ok(())
}

fn validate_shapes(
    reference: (usize, usize),
    estimate: (usize, usize),
    valid_mask: Option<(usize, usize)>,
) -> Result<(), IntensityMetricError> {
    if reference != estimate {
        return Err(IntensityMetricError::ShapeMismatch {
            reference,
            estimate,
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
    fn residual_metrics_use_estimate_minus_reference_and_support_integers() {
        let reference = array![[1u8, 2u8]];
        let estimate = array![[2u8, 0u8]];
        assert_relative_eq!(bias(reference.view(), estimate.view(), None).unwrap(), -0.5);
        assert_relative_eq!(mae(reference.view(), estimate.view(), None).unwrap(), 1.5);
        assert_relative_eq!(mse(reference.view(), estimate.view(), None).unwrap(), 2.5);
        assert_relative_eq!(
            rmse(reference.view(), estimate.view(), None).unwrap(),
            2.5f64.sqrt()
        );
        assert_relative_eq!(
            relative_l1(reference.view(), estimate.view(), None).unwrap(),
            1.0
        );
        assert_relative_eq!(nrmse(reference.view(), estimate.view(), None).unwrap(), 1.0);
        assert_relative_eq!(
            correlation(reference.view(), estimate.view(), None).unwrap(),
            -1.0
        );
    }

    #[test]
    fn mask_selects_valid_pixels() {
        let reference = array![[1.0, 20.0]];
        let estimate = array![[2.0, 0.0]];
        let mask = array![[true, false]];
        assert_relative_eq!(
            bias(reference.view(), estimate.view(), Some(mask.view())).unwrap(),
            1.0
        );
    }

    #[test]
    fn quality_metrics_have_documented_reference_values() {
        let reference = array![[0.0, 1.0]];
        let estimate = array![[0.0, 0.0]];
        assert_relative_eq!(
            psnr(reference.view(), estimate.view(), None, 1.0).unwrap(),
            10.0 * 2.0f64.log10()
        );

        let image = ndarray::Array2::<f64>::ones((11, 11));
        assert_relative_eq!(ssim(image.view(), image.view(), None, 1.0).unwrap(), 1.0);
    }

    #[test]
    fn poisson_deviance_and_gain_are_well_defined() {
        let reference = array![[0.0, 2.0]];
        let estimate = array![[0.0, 2.0]];
        assert_relative_eq!(
            poisson_deviance(reference.view(), estimate.view(), None, 1e-12).unwrap(),
            0.0
        );
        assert_relative_eq!(
            mean_poisson_deviance(reference.view(), estimate.view(), None, 1e-12).unwrap(),
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
        let estimate = array![[2.0, 0.0]];
        let wrong_mask = array![[true], [false]];
        assert!(matches!(
            mae(reference.view(), estimate.view(), Some(wrong_mask.view())),
            Err(IntensityMetricError::MaskShapeMismatch { .. })
        ));
        assert!(matches!(
            psnr(reference.view(), estimate.view(), None, 0.0),
            Err(IntensityMetricError::InvalidDataRange)
        ));
        assert!(matches!(
            ssim(reference.view(), estimate.view(), None, 1.0),
            Err(IntensityMetricError::SsimImageTooSmall { .. })
        ));

        let negative = array![[-1.0, 2.0]];
        assert!(matches!(
            amplitude_nrmse(negative.view(), estimate.view(), None),
            Err(IntensityMetricError::NegativeIntensity { .. })
        ));
        assert!(matches!(
            poisson_deviance(negative.view(), estimate.view(), None, 1e-12),
            Err(IntensityMetricError::NegativeIntensity { .. })
        ));
    }
}
