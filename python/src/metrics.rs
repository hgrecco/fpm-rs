use fpm_rs::{
    Complex64,
    algorithms::objective::{LossType, loss},
    metrics::{
        complex_field::{
            compare_complex_fields, compare_complex_fields_masked, radial_fourier_spectrum,
        },
        intensity::{
            IntensityMetricError, amplitude_nrmse, bias, compare_intensity, correlation,
            fitted_gain, mae, mean_poisson_deviance, mse, nrmse, poisson_deviance, psnr,
            relative_l1, rmse, ssim, stats,
        },
    },
};
use numpy::{PyReadonlyArray2, ndarray};
use pyo3::{prelude::*, types::PyDict};

use crate::errors::to_py_err;

#[pyfunction]
fn stats_py(
    py: Python<'_>,
    frame: PyReadonlyArray2<'_, f64>,
    saturation_value: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let frame: Vec<_> = frame.as_array().iter().copied().collect();
    let metrics = py
        .detach(move || stats(&frame, saturation_value))
        .map_err(to_py_err)?;
    intensity_stats_to_py(py, &metrics)
}

#[pyfunction]
#[pyo3(signature = (reference, estimate, *, valid_mask=None, saturation_value=None))]
fn compare_intensity_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    estimate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    saturation_value: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let (reference, estimate, valid_mask) = scalar_metric_inputs(reference, estimate, valid_mask);
    let metrics = py
        .detach(move || {
            compare_intensity(
                reference.view(),
                estimate.view(),
                valid_mask.as_ref().map(|mask| mask.view()),
                saturation_value,
            )
        })
        .map_err(intensity_metric_to_py_err)?;
    intensity_comparison_to_py(py, &metrics)
}

macro_rules! scalar_intensity_metric {
    ($name:ident, $metric:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (reference, estimate, *, valid_mask=None))]
        fn $name(
            py: Python<'_>,
            reference: PyReadonlyArray2<'_, f64>,
            estimate: PyReadonlyArray2<'_, f64>,
            valid_mask: Option<PyReadonlyArray2<'_, bool>>,
        ) -> PyResult<f64> {
            let (reference, estimate, valid_mask) =
                scalar_metric_inputs(reference, estimate, valid_mask);
            py.detach(move || {
                $metric(
                    reference.view(),
                    estimate.view(),
                    valid_mask.as_ref().map(|mask| mask.view()),
                )
            })
            .map_err(intensity_metric_to_py_err)
        }
    };
}

scalar_intensity_metric!(
    bias_py,
    bias,
    "Return the mean signed residual, estimate minus reference."
);
scalar_intensity_metric!(mae_py, mae, "Return mean absolute error.");
scalar_intensity_metric!(mse_py, mse, "Return mean squared error.");
scalar_intensity_metric!(rmse_py, rmse, "Return root mean squared error.");
scalar_intensity_metric!(
    relative_l1_py,
    relative_l1,
    "Return L1 residual normalized by the reference L1 norm."
);
scalar_intensity_metric!(
    nrmse_py,
    nrmse,
    "Return RMSE normalized by the reference L2 norm."
);
scalar_intensity_metric!(
    amplitude_nrmse_py,
    amplitude_nrmse,
    "Compare square-root intensities, normalized by reference amplitude energy."
);
scalar_intensity_metric!(
    correlation_py,
    correlation,
    "Return Pearson correlation over valid pixels."
);
scalar_intensity_metric!(
    fitted_gain_py,
    fitted_gain,
    "Fit gain in estimate approximately equal to gain times reference."
);

#[pyfunction]
#[pyo3(signature = (reference, estimate, *, valid_mask=None, data_range))]
/// Return PSNR in dB using an explicit positive intensity range.
fn psnr_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    estimate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    data_range: f64,
) -> PyResult<f64> {
    let (reference, estimate, valid_mask) = scalar_metric_inputs(reference, estimate, valid_mask);
    py.detach(move || {
        psnr(
            reference.view(),
            estimate.view(),
            valid_mask.as_ref().map(|mask| mask.view()),
            data_range,
        )
    })
    .map_err(intensity_metric_to_py_err)
}

#[pyfunction]
#[pyo3(signature = (reference, estimate, *, valid_mask=None, data_range))]
/// Return canonical single-scale SSIM with an 11 by 11 Gaussian window.
///
/// Reference
/// ---------
/// [Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality
/// assessment: From error visibility to structural similarity"
/// (2004)](https://doi.org/10.1109/TIP.2003.819861), IEEE Transactions on Image
/// Processing 13(4), 600-612.
fn ssim_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    estimate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    data_range: f64,
) -> PyResult<f64> {
    let (reference, estimate, valid_mask) = scalar_metric_inputs(reference, estimate, valid_mask);
    py.detach(move || {
        ssim(
            reference.view(),
            estimate.view(),
            valid_mask.as_ref().map(|mask| mask.view()),
            data_range,
        )
    })
    .map_err(intensity_metric_to_py_err)
}

#[pyfunction]
#[pyo3(signature = (reference, estimate, *, valid_mask=None, epsilon))]
/// Return summed Poisson deviance with a positive estimate-intensity floor.
fn poisson_deviance_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    estimate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    epsilon: f64,
) -> PyResult<f64> {
    let (reference, estimate, valid_mask) = scalar_metric_inputs(reference, estimate, valid_mask);
    py.detach(move || {
        poisson_deviance(
            reference.view(),
            estimate.view(),
            valid_mask.as_ref().map(|mask| mask.view()),
            epsilon,
        )
    })
    .map_err(intensity_metric_to_py_err)
}

#[pyfunction]
#[pyo3(signature = (reference, estimate, *, valid_mask=None, epsilon))]
/// Return mean Poisson deviance with a positive estimate-intensity floor.
fn mean_poisson_deviance_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    estimate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    epsilon: f64,
) -> PyResult<f64> {
    let (reference, estimate, valid_mask) = scalar_metric_inputs(reference, estimate, valid_mask);
    py.detach(move || {
        mean_poisson_deviance(
            reference.view(),
            estimate.view(),
            valid_mask.as_ref().map(|mask| mask.view()),
            epsilon,
        )
    })
    .map_err(intensity_metric_to_py_err)
}

#[pyfunction]
#[pyo3(signature = (reference, candidate, *, valid_mask=None))]
fn compare_complex_fields_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, Complex64>,
    candidate: PyReadonlyArray2<'_, Complex64>,
    valid_mask: Option<PyReadonlyArray2<'_, u8>>,
) -> PyResult<Py<PyDict>> {
    let reference = reference.as_array().to_owned();
    let candidate = candidate.as_array().to_owned();
    let valid_mask = valid_mask.map(|mask| mask.as_array().to_owned());
    let metrics = py
        .detach(move || match valid_mask.as_ref() {
            Some(mask) => {
                compare_complex_fields_masked(reference.view(), candidate.view(), Some(mask.view()))
            }
            None => compare_complex_fields(reference.view(), candidate.view()),
        })
        .map_err(to_py_err)?;
    complex_field_to_py(py, &metrics)
}

#[pyfunction]
fn radial_fourier_spectrum_py(
    py: Python<'_>,
    field: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<Py<PyDict>> {
    let field = field.as_array().to_owned();
    let spectrum = py
        .detach(move || radial_fourier_spectrum(field.view()))
        .map_err(to_py_err)?;
    let output = PyDict::new(py);
    output.set_item("radius_px", spectrum.radius_px)?;
    output.set_item("power", spectrum.power)?;
    output.set_item("sample_count", spectrum.sample_count)?;
    Ok(output.unbind())
}

#[pyfunction]
fn intensity_loss(
    reference: PyReadonlyArray2<'_, f64>,
    candidate: PyReadonlyArray2<'_, f64>,
    loss_type: &str,
) -> PyResult<f64> {
    let reference: Vec<_> = reference.as_array().iter().copied().collect();
    let candidate: Vec<_> = candidate.as_array().iter().copied().collect();
    if reference.len() != candidate.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "reference and candidate must have the same shape",
        ));
    }
    let loss_type = match loss_type {
        "amplitude_mse" => LossType::AmplitudeMse,
        "intensity_mse" => LossType::IntensityMse,
        "poisson_nll" => LossType::PoissonNegativeLogLikelihood,
        "huber_amplitude" => LossType::HuberAmplitude,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("unknown loss_type")),
    };
    loss(&candidate, &reference, loss_type).map_err(to_py_err)
}

fn scalar_metric_inputs(
    reference: PyReadonlyArray2<'_, f64>,
    estimate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    Option<ndarray::Array2<bool>>,
) {
    (
        reference.as_array().to_owned(),
        estimate.as_array().to_owned(),
        valid_mask.map(|mask| mask.as_array().to_owned()),
    )
}

fn intensity_metric_to_py_err(error: IntensityMetricError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

fn intensity_stats_to_py(
    py: Python<'_>,
    value: &fpm_rs::metrics::intensity::IntensityStats,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    output.set_item("mean", value.mean)?;
    output.set_item("std", value.std)?;
    output.set_item("min", value.min)?;
    output.set_item("max", value.max)?;
    output.set_item("sum", value.sum)?;
    output.set_item("saturated_pixels", value.saturated_pixels)?;
    output.set_item("zero_pixels", value.zero_pixels)?;
    Ok(output.unbind())
}

fn intensity_comparison_to_py(
    py: Python<'_>,
    value: &fpm_rs::metrics::intensity::IntensityComparisonMetrics,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    output.set_item("reference_sum", value.reference_sum)?;
    output.set_item("estimate_sum", value.estimate_sum)?;
    output.set_item("residual_l1", value.residual_l1)?;
    output.set_item("residual_l2", value.residual_l2)?;
    output.set_item("residual_mean", value.residual_mean)?;
    output.set_item("residual_std", value.residual_std)?;
    output.set_item("residual_max_abs", value.residual_max_abs)?;
    output.set_item("normalized_l2", value.normalized_l2)?;
    output.set_item("saturated_pixels", value.saturated_pixels)?;
    Ok(output.unbind())
}

fn complex_field_to_py(
    py: Python<'_>,
    value: &fpm_rs::metrics::complex_field::ComplexFieldComparisonMetrics,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    output.set_item("amplitude_rmse", value.amplitude_rmse)?;
    output.set_item("amplitude_nrmse", value.amplitude_nrmse)?;
    output.set_item("complex_rmse", value.complex_rmse)?;
    output.set_item("complex_nrmse", value.complex_nrmse)?;
    output.set_item("phase_rmse", value.phase_rmse)?;
    output.set_item("phase_mae", value.phase_mae)?;
    output.set_item("fourier_nrmse", value.fourier_nrmse)?;
    output.set_item("global_phase_offset", value.global_phase_offset)?;
    Ok(output.unbind())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(stats_py, module)?)?;
    module.add_function(wrap_pyfunction!(compare_intensity_py, module)?)?;
    module.add_function(wrap_pyfunction!(bias_py, module)?)?;
    module.add_function(wrap_pyfunction!(mae_py, module)?)?;
    module.add_function(wrap_pyfunction!(mse_py, module)?)?;
    module.add_function(wrap_pyfunction!(rmse_py, module)?)?;
    module.add_function(wrap_pyfunction!(relative_l1_py, module)?)?;
    module.add_function(wrap_pyfunction!(nrmse_py, module)?)?;
    module.add_function(wrap_pyfunction!(amplitude_nrmse_py, module)?)?;
    module.add_function(wrap_pyfunction!(correlation_py, module)?)?;
    module.add_function(wrap_pyfunction!(psnr_py, module)?)?;
    module.add_function(wrap_pyfunction!(ssim_py, module)?)?;
    module.add_function(wrap_pyfunction!(poisson_deviance_py, module)?)?;
    module.add_function(wrap_pyfunction!(mean_poisson_deviance_py, module)?)?;
    module.add_function(wrap_pyfunction!(fitted_gain_py, module)?)?;
    module.add_function(wrap_pyfunction!(compare_complex_fields_py, module)?)?;
    module.add_function(wrap_pyfunction!(radial_fourier_spectrum_py, module)?)?;
    module.add_function(wrap_pyfunction!(intensity_loss, module)?)?;
    Ok(())
}
