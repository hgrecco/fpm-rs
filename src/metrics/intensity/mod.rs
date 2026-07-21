//! Metrics for scalar intensity images.
//!
//! [`stats`] summarizes one image. All other functions compare a `reference`
//! image with an `estimate`; signed residuals are `estimate - reference`.
//! Implementation modules are private so callers use this stable facade.

mod compare;
mod single;

pub(crate) use compare::compare_intensity_u8_masked;
pub use compare::{
    IntensityComparisonMetrics, IntensityMetricError, amplitude_nrmse, bias, compare_intensity,
    correlation, fitted_gain, mae, mean_poisson_deviance, mse, nrmse, poisson_deviance, psnr,
    relative_l1, rmse, ssim,
};
pub use single::{IntensityStats, stats};
