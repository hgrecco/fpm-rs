//! Metrics for detector-plane intensity images.

pub mod atomic;
pub mod comparison;
pub mod statistics;

pub use atomic::{
    IntensityMetricError, amplitude_nrmse, bias, correlation, fitted_gain, mae,
    mean_poisson_deviance, mse, nrmse, poisson_deviance, psnr, relative_l1, rmse, ssim,
};
pub use comparison::{IntensityComparisonMetrics, compare_intensity, compare_intensity_masked};
pub use statistics::{IntensityStatistics, intensity_statistics};
