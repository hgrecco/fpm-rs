use fpm_rs::model::ImagePlaneModel;
use numpy::{PyArray1, PyArray2, ndarray};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::{
    arrays::{complex_array2_to_py, vec2_to_py},
    config::{PyCameraModel, PyOptics, extract_illumination},
    errors::to_py_err,
};

#[pyclass(
    module = "fpm_rs._core",
    name = "ImagePlaneModel",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyImagePlaneModel {
    pub(crate) inner: Arc<ImagePlaneModel>,
}

#[pymethods]
impl PyImagePlaneModel {
    #[getter]
    fn image_shape(&self) -> (usize, usize) {
        self.inner.image_shape
    }

    #[getter]
    fn reconstruction_shape(&self) -> (usize, usize) {
        self.inner.reconstruction_shape
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    #[getter]
    fn is_multiplexed(&self) -> bool {
        self.inner.is_multiplexed()
    }

    #[getter]
    fn k_vectors(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let values = self
            .inner
            .k_vectors
            .iter()
            .flat_map(|vector| [vector.kx, vector.ky])
            .collect();
        vec2_to_py(py, (self.inner.source_count(), 2), values)
    }

    #[getter]
    fn pupil(&self, py: Python<'_>) -> PyResult<Py<PyArray2<fpm_rs::Complex64>>> {
        complex_array2_to_py(py, self.inner.pupil.values.clone())
    }

    #[getter]
    fn pupil_support(&self, py: Python<'_>) -> PyResult<Py<PyArray2<u8>>> {
        let values = self
            .inner
            .pupil
            .support
            .iter()
            .map(|&value| u8::from(value))
            .collect();
        vec2_to_py(py, self.inner.image_shape, values)
    }

    #[getter]
    fn frame_gains(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.inner.frame_gains.as_ref().map(|values| {
            PyArray1::from_owned_array(py, ndarray::Array1::from_vec(values.clone())).unbind()
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ImagePlaneModel(image_shape={:?}, reconstruction_shape={:?}, sources={}, frames={})",
            self.inner.image_shape,
            self.inner.reconstruction_shape,
            self.inner.source_count(),
            self.inner.frame_count()
        )
    }
}

#[pyfunction]
pub(crate) fn compile_model(
    py: Python<'_>,
    optics: PyRef<'_, PyOptics>,
    illumination: &Bound<'_, PyAny>,
    image_shape: (usize, usize),
    reconstruction_shape: (usize, usize),
) -> PyResult<PyImagePlaneModel> {
    let illumination = extract_illumination(illumination)?;
    let optics = optics.inner.clone();
    // The inputs are all Rust-owned after conversion. Pupil construction and
    // source/crop compilation are independent of the interpreter and can take
    // appreciable time for dense illumination arrays.
    let inner = py
        .detach(move || {
            ImagePlaneModel::from_experiment(
                &optics,
                &illumination,
                image_shape,
                reconstruction_shape,
            )
        })
        .map_err(to_py_err)?;
    Ok(PyImagePlaneModel {
        inner: Arc::new(inner),
    })
}

#[pyfunction]
pub(crate) fn compile_camera_model(
    py: Python<'_>,
    model: PyRef<'_, PyImagePlaneModel>,
    camera: PyRef<'_, PyCameraModel>,
) -> PyResult<PyImagePlaneModel> {
    let model = model.inner.clone();
    let camera = camera.inner.clone();
    let inner = py
        .detach(move || camera.compile_reconstruction_model((*model).clone()))
        .map_err(to_py_err)?;
    Ok(PyImagePlaneModel {
        inner: Arc::new(inner),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyImagePlaneModel>()?;
    module.add_function(wrap_pyfunction!(compile_model, module)?)?;
    module.add_function(wrap_pyfunction!(compile_camera_model, module)?)?;
    Ok(())
}
