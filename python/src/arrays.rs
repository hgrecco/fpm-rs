use fpm_rs::Complex64;
use numpy::{
    Element, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods, ndarray,
};
use pyo3::prelude::*;

pub(crate) fn copy_array2<T: Element + Copy>(array: &PyReadonlyArray2<'_, T>) -> Vec<T> {
    array
        .as_slice()
        .map(<[T]>::to_vec)
        .unwrap_or_else(|_| array.as_array().iter().copied().collect())
}

pub(crate) fn core_array2<T: Element + Copy>(
    array: &PyReadonlyArray2<'_, T>,
) -> fpm_rs::Result<ndarray::Array2<T>> {
    let shape = array.shape();
    let view = array.as_array();
    if !view.is_standard_layout() {
        return Err(fpm_rs::Error::NonStandardLayout {
            context: "NumPy computational input (use numpy.ascontiguousarray explicitly to copy)",
            shape: view.shape().to_vec(),
            strides: view.strides().to_vec(),
        });
    }
    Ok(ndarray::Array2::from_shape_vec(
        (shape[0], shape[1]),
        array
            .as_slice()
            .map_err(|_| fpm_rs::Error::NonStandardLayout {
                context: "NumPy computational input (use numpy.ascontiguousarray explicitly to copy)",
                shape: view.shape().to_vec(),
                strides: view.strides().to_vec(),
            })?
            .to_vec(),
    )?)
}

pub(crate) fn core_array3<T: Element + Copy>(
    array: &PyReadonlyArray3<'_, T>,
) -> fpm_rs::Result<ndarray::Array3<T>> {
    let shape = array.shape();
    let view = array.as_array();
    if !view.is_standard_layout() {
        return Err(fpm_rs::Error::NonStandardLayout {
            context: "NumPy computational input (use numpy.ascontiguousarray explicitly to copy)",
            shape: view.shape().to_vec(),
            strides: view.strides().to_vec(),
        });
    }
    Ok(ndarray::Array3::from_shape_vec(
        (shape[0], shape[1], shape[2]),
        array
            .as_slice()
            .map_err(|_| fpm_rs::Error::NonStandardLayout {
                context: "NumPy computational input (use numpy.ascontiguousarray explicitly to copy)",
                shape: view.shape().to_vec(),
                strides: view.strides().to_vec(),
            })?
            .to_vec(),
    )?)
}

pub(crate) fn array2_to_py<T: Element>(
    py: Python<'_>,
    array: ndarray::Array2<T>,
) -> PyResult<Py<PyArray2<T>>> {
    Ok(PyArray2::from_owned_array(py, array).unbind())
}

pub(crate) fn vec2_to_py<T: Element>(
    py: Python<'_>,
    shape: (usize, usize),
    values: Vec<T>,
) -> PyResult<Py<PyArray2<T>>> {
    let owned = ndarray::Array2::from_shape_vec(shape, values)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, owned).unbind())
}

pub(crate) fn vec3_to_py<T: Element>(
    py: Python<'_>,
    shape: (usize, usize, usize),
    values: Vec<T>,
) -> PyResult<Py<PyArray3<T>>> {
    let owned = ndarray::Array3::from_shape_vec(shape, values)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyArray3::from_owned_array(py, owned).unbind())
}

pub(crate) fn complex_array2_to_py(
    py: Python<'_>,
    array: ndarray::Array2<Complex64>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    array2_to_py(py, array)
}
