use fpm_rs::{
    Complex64, metrics::complex_field::radial_fourier_spectrum as core_radial_fourier_spectrum,
};
use numpy::PyReadonlyArray2;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use crate::errors::to_py_err;

/// Calculate azimuthally averaged, normalized Fourier power in integer-radius bins.
///
/// The returned dictionary contains Fourier-grid-pixel ``radius_px``, mean
/// normalized ``power``, and the number of samples in each annulus under
/// ``sample_count``. Physical frequencies require a sampling-pitch conversion
/// by the caller.
#[pyfunction]
fn radial_fourier_spectrum(
    py: Python<'_>,
    field: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<Py<PyDict>> {
    let field = field.as_array().to_owned();
    let spectrum = py
        .detach(move || core_radial_fourier_spectrum(field.view()))
        .map_err(to_py_err)?;
    let output = PyDict::new(py);
    output.set_item("radius_px", PyList::new(py, spectrum.radius_px)?)?;
    output.set_item("power", PyList::new(py, spectrum.power)?)?;
    output.set_item("sample_count", PyList::new(py, spectrum.sample_count)?)?;
    Ok(output.unbind())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(radial_fourier_spectrum, module)?)?;
    Ok(())
}
