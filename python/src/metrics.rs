use fpm_rs::{
    Complex64,
    algorithms::objective::{LossType, loss},
    metrics::{
        complex_field::{
            compare_complex_fields, compare_complex_fields_masked, radial_fourier_spectrum,
        },
        intensity::{
            IntensityMetricError, amplitude_nrmse, bias, compare_intensity,
            compare_intensity_masked, correlation, fitted_gain, intensity_statistics, mae,
            mean_poisson_deviance, mse, nrmse, poisson_deviance, psnr, relative_l1, rmse, ssim,
        },
    },
};
use numpy::{PyReadonlyArray2, ndarray};
use pyo3::{prelude::*, types::PyDict};

use crate::{arrays::core_array2, errors::to_py_err};

#[pyfunction]
fn intensity_statistics_py(
    py: Python<'_>,
    frame: PyReadonlyArray2<'_, f64>,
    saturation_value: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let frame = core_array2(&frame).map_err(to_py_err)?;
    let metrics = py
        .detach(move || intensity_statistics(frame.as_slice(), saturation_value))
        .map_err(to_py_err)?;
    intensity_statistics_to_py(py, &metrics)
}

#[pyfunction]
#[pyo3(signature = (reference, candidate, *, mask=None, saturation_value=None))]
fn compare_intensity_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    candidate: PyReadonlyArray2<'_, f64>,
    mask: Option<PyReadonlyArray2<'_, u8>>,
    saturation_value: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let reference = core_array2(&reference).map_err(to_py_err)?;
    let candidate = core_array2(&candidate).map_err(to_py_err)?;
    if reference.shape() != candidate.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "reference and candidate must have the same shape",
        ));
    }
    let mask = mask
        .map(|mask| core_array2(&mask))
        .transpose()
        .map_err(to_py_err)?;
    let metrics = py
        .detach(move || match mask.as_ref() {
            Some(mask) => compare_intensity_masked(
                reference.as_slice(),
                candidate.as_slice(),
                Some(mask.as_slice()),
                saturation_value,
            ),
            None => compare_intensity(reference.as_slice(), candidate.as_slice(), saturation_value),
        })
        .map_err(to_py_err)?;
    intensity_comparison_to_py(py, &metrics)
}

macro_rules! scalar_intensity_metric {
    ($name:ident, $metric:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (reference, candidate, *, valid_mask=None))]
        fn $name(
            py: Python<'_>,
            reference: PyReadonlyArray2<'_, f64>,
            candidate: PyReadonlyArray2<'_, f64>,
            valid_mask: Option<PyReadonlyArray2<'_, bool>>,
        ) -> PyResult<f64> {
            let (reference, candidate, valid_mask) =
                scalar_metric_inputs(reference, candidate, valid_mask);
            py.detach(move || {
                $metric(
                    reference.view(),
                    candidate.view(),
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
    "Return the mean signed residual, candidate minus reference."
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
    "Fit gain in candidate approximately equal to gain times reference."
);

#[pyfunction]
#[pyo3(signature = (reference, candidate, *, valid_mask=None, data_range))]
/// Return PSNR in dB using an explicit positive intensity range.
fn psnr_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    candidate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    data_range: f64,
) -> PyResult<f64> {
    let (reference, candidate, valid_mask) = scalar_metric_inputs(reference, candidate, valid_mask);
    py.detach(move || {
        psnr(
            reference.view(),
            candidate.view(),
            valid_mask.as_ref().map(|mask| mask.view()),
            data_range,
        )
    })
    .map_err(intensity_metric_to_py_err)
}

#[pyfunction]
#[pyo3(signature = (reference, candidate, *, valid_mask=None, data_range))]
/// Return canonical single-scale SSIM with an 11 by 11 Gaussian window.
fn ssim_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    candidate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    data_range: f64,
) -> PyResult<f64> {
    let (reference, candidate, valid_mask) = scalar_metric_inputs(reference, candidate, valid_mask);
    py.detach(move || {
        ssim(
            reference.view(),
            candidate.view(),
            valid_mask.as_ref().map(|mask| mask.view()),
            data_range,
        )
    })
    .map_err(intensity_metric_to_py_err)
}

#[pyfunction]
#[pyo3(signature = (reference, candidate, *, valid_mask=None, epsilon))]
/// Return summed Poisson deviance with a positive candidate-intensity floor.
fn poisson_deviance_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    candidate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    epsilon: f64,
) -> PyResult<f64> {
    let (reference, candidate, valid_mask) = scalar_metric_inputs(reference, candidate, valid_mask);
    py.detach(move || {
        poisson_deviance(
            reference.view(),
            candidate.view(),
            valid_mask.as_ref().map(|mask| mask.view()),
            epsilon,
        )
    })
    .map_err(intensity_metric_to_py_err)
}

#[pyfunction]
#[pyo3(signature = (reference, candidate, *, valid_mask=None, epsilon))]
/// Return mean Poisson deviance with a positive candidate-intensity floor.
fn mean_poisson_deviance_py(
    py: Python<'_>,
    reference: PyReadonlyArray2<'_, f64>,
    candidate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
    epsilon: f64,
) -> PyResult<f64> {
    let (reference, candidate, valid_mask) = scalar_metric_inputs(reference, candidate, valid_mask);
    py.detach(move || {
        mean_poisson_deviance(
            reference.view(),
            candidate.view(),
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
    let reference = core_array2(&reference).map_err(to_py_err)?;
    let candidate = core_array2(&candidate).map_err(to_py_err)?;
    let valid_mask = valid_mask
        .map(|mask| core_array2(&mask))
        .transpose()
        .map_err(to_py_err)?;
    let metrics = py
        .detach(move || match valid_mask.as_ref() {
            Some(mask) => compare_complex_fields_masked(&reference, &candidate, Some(mask)),
            None => compare_complex_fields(&reference, &candidate),
        })
        .map_err(to_py_err)?;
    complex_field_to_py(py, &metrics)
}

#[pyfunction]
fn radial_fourier_spectrum_py(
    py: Python<'_>,
    field: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<Py<PyDict>> {
    let field = core_array2(&field).map_err(to_py_err)?;
    let spectrum = py
        .detach(move || radial_fourier_spectrum(&field))
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
    let reference = core_array2(&reference).map_err(to_py_err)?;
    let candidate = core_array2(&candidate).map_err(to_py_err)?;
    if reference.shape() != candidate.shape() {
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
    loss(candidate.as_slice(), reference.as_slice(), loss_type).map_err(to_py_err)
}

fn scalar_metric_inputs(
    reference: PyReadonlyArray2<'_, f64>,
    candidate: PyReadonlyArray2<'_, f64>,
    valid_mask: Option<PyReadonlyArray2<'_, bool>>,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    Option<ndarray::Array2<bool>>,
) {
    (
        reference.as_array().to_owned(),
        candidate.as_array().to_owned(),
        valid_mask.map(|mask| mask.as_array().to_owned()),
    )
}

fn intensity_metric_to_py_err(error: IntensityMetricError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

fn intensity_statistics_to_py(
    py: Python<'_>,
    value: &fpm_rs::metrics::intensity::IntensityStatistics,
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
    output.set_item("candidate_sum", value.candidate_sum)?;
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
    module.add_function(wrap_pyfunction!(intensity_statistics_py, module)?)?;
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
