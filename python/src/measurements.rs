use fpm_rs::measurements::{FrameMetadata, MeasurementStack};
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods, ndarray};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::{
    arrays::{core_array3, vec3_to_py},
    errors::to_py_err,
};

#[pyclass(
    module = "fpm_rs._core",
    name = "MeasurementStack",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyMeasurementStack {
    pub(crate) inner: Arc<MeasurementStack>,
}

impl PyMeasurementStack {
    pub(crate) fn from_numpy(
        measurements: &PyReadonlyArray3<'_, f64>,
        frame_weights: Option<Vec<f64>>,
        masks: Option<&PyReadonlyArray3<'_, u8>>,
    ) -> PyResult<Self> {
        let shape = measurements.shape();
        if shape.contains(&0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "measurements must have non-zero shape (frames, height, width)",
            ));
        }
        let frames = shape[0];
        let weights = frame_weights.unwrap_or_else(|| vec![1.0; frames]);
        if weights.len() != frames {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "frame_weights must contain {frames} values"
            )));
        }
        let metadata = weights
            .into_iter()
            .enumerate()
            .map(|(frame, weight)| {
                let mut metadata = FrameMetadata::new(frame);
                metadata.weight = weight;
                metadata
            })
            .collect();
        let mut inner =
            MeasurementStack::new(core_array3(measurements).map_err(to_py_err)?, metadata)
                .map_err(to_py_err)?;
        if let Some(masks) = masks {
            if masks.shape() != shape {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "masks must have the same shape as measurements",
                ));
            }
            inner = inner
                .with_per_frame_masks(core_array3(masks).map_err(to_py_err)?)
                .map_err(to_py_err)?;
        }
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

#[pymethods]
impl PyMeasurementStack {
    #[new]
    #[pyo3(signature = (measurements, *, frame_weights=None, masks=None))]
    fn new(
        measurements: PyReadonlyArray3<'_, f64>,
        frame_weights: Option<Vec<f64>>,
        masks: Option<PyReadonlyArray3<'_, u8>>,
    ) -> PyResult<Self> {
        Self::from_numpy(&measurements, frame_weights, masks.as_ref())
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let (height, width) = self.inner.image_shape();
        (self.inner.frame_count(), height, width)
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    #[getter]
    fn image_shape(&self) -> (usize, usize) {
        self.inner.image_shape()
    }

    #[getter]
    fn array(&self, py: Python<'_>) -> PyResult<Py<PyArray3<f64>>> {
        let (height, width) = self.inner.image_shape();
        vec3_to_py(
            py,
            (self.inner.frame_count(), height, width),
            self.inner.as_slice().to_vec(),
        )
    }

    #[getter]
    fn frame_weights(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        let values = self
            .inner
            .frame_metadata()
            .iter()
            .map(|metadata| metadata.weight)
            .collect();
        PyArray1::from_owned_array(py, ndarray::Array1::from_vec(values)).unbind()
    }

    fn __len__(&self) -> usize {
        self.inner.frame_count()
    }

    fn __repr__(&self) -> String {
        format!("MeasurementStack(shape={:?})", self.shape())
    }
}

pub(crate) fn extract_measurements(
    value: &Bound<'_, PyAny>,
    frame_weights: Option<Vec<f64>>,
    masks: Option<&Bound<'_, PyAny>>,
) -> PyResult<Arc<MeasurementStack>> {
    if let Ok(stack) = value.extract::<PyRef<'_, PyMeasurementStack>>() {
        if frame_weights.is_some() || masks.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frame_weights and masks must be configured when constructing MeasurementStack",
            ));
        }
        return Ok(stack.inner.clone());
    }
    let array = value
        .cast::<PyArray3<f64>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "measurements must be a float64 NumPy array or MeasurementStack",
            )
        })?
        .readonly();
    let mask_array = masks
        .map(|value| {
            value
                .cast::<PyArray3<u8>>()
                .map(PyArrayMethods::readonly)
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err("masks must be a uint8 NumPy array")
                })
        })
        .transpose()?;
    Ok(PyMeasurementStack::from_numpy(&array, frame_weights, mask_array.as_ref())?.inner)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMeasurementStack>()?;
    Ok(())
}
