use fpm_rs::{
    Complex64,
    simulation::{SimulationResult, Simulator, SyntheticObject},
};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use std::{path::PathBuf, sync::Arc};

use crate::{
    arrays::{complex_array2_to_py, core_array2},
    config::{PyCameraModel, PyIlluminationAcquisitionErrors},
    errors::to_py_err,
    measurements::PyMeasurementStack,
    model::PyImagePlaneModel,
};

#[pyclass(module = "fpm_rs._core", name = "SyntheticObject", frozen)]
#[derive(Clone)]
pub(crate) struct PySyntheticObject {
    inner: SyntheticObject,
}

#[pymethods]
impl PySyntheticObject {
    #[new]
    #[pyo3(signature = (field))]
    fn new(field: PyReadonlyArray2<'_, Complex64>) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::new(core_array2(&field).map_err(to_py_err)?),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, amplitude=1.0, phase=0.0))]
    fn constant(shape: (usize, usize), amplitude: f64, phase: f64) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::constant(shape, amplitude, phase).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn amplitude_only(amplitude: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::amplitude_only(core_array2(&amplitude).map_err(to_py_err)?)
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn phase_only(phase: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::phase_only(core_array2(&phase).map_err(to_py_err)?)
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn from_amplitude_phase(
        amplitude: PyReadonlyArray2<'_, f64>,
        phase: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let amplitude = core_array2(&amplitude).map_err(to_py_err)?;
        let phase = core_array2(&phase).map_err(to_py_err)?;
        Ok(Self {
            inner: SyntheticObject::from_amplitude_phase(&amplitude, &phase).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn from_amplitude_image(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::from_amplitude_image(path).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn from_amplitude_phase_images(
        amplitude_path: PathBuf,
        phase_path: PathBuf,
        phase_extent: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::from_amplitude_phase_images(
                amplitude_path,
                phase_path,
                phase_extent,
            )
            .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn phase_disk(shape: (usize, usize), radius_pixels: f64, phase_shift: f64) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::phase_disk(shape, radius_pixels, phase_shift)
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn siemens_star(shape: (usize, usize), spokes: usize) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::siemens_star(shape, spokes).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn resolution_target(shape: (usize, usize)) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::resolution_target(shape).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn random_phase(
        shape: (usize, usize),
        standard_deviation: f64,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::random_phase(shape, standard_deviation, seed)
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn particle_field(shape: (usize, usize), particles: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::particle_field(shape, particles, seed).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn mixed_test_pattern(shape: (usize, usize)) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::mixed_test_pattern(shape).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn biological_like(shape: (usize, usize), features: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: SyntheticObject::biological_like(shape, features, seed).map_err(to_py_err)?,
        })
    }

    #[getter]
    fn field(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Complex64>>> {
        complex_array2_to_py(py, self.inner.field.clone())
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.label.clone()
    }

    fn __repr__(&self) -> String {
        format!("SyntheticObject(shape={:?})", self.shape())
    }
}

#[pyclass(module = "fpm_rs._core", name = "SimulationResult", frozen)]
pub(crate) struct PySimulationResult {
    measurements: Arc<fpm_rs::measurements::MeasurementStack>,
    ground_truth_object: Py<PyArray2<Complex64>>,
    true_model: fpm_rs::model::ImagePlaneModel,
    reconstruction_model: fpm_rs::model::ImagePlaneModel,
    ideal: bool,
    missing_frames: Vec<usize>,
    random_seed: u64,
}

impl PySimulationResult {
    fn from_core(py: Python<'_>, result: SimulationResult) -> PyResult<Self> {
        Ok(Self {
            measurements: Arc::new(result.measurements),
            ground_truth_object: complex_array2_to_py(py, result.ground_truth_object)?,
            true_model: result.true_model,
            reconstruction_model: result.reconstruction_model,
            ideal: result.parameters.ideal,
            missing_frames: result.parameters.missing_frames,
            random_seed: result.random_seed,
        })
    }
}

#[pymethods]
impl PySimulationResult {
    #[getter]
    fn measurements(&self) -> PyMeasurementStack {
        PyMeasurementStack {
            inner: self.measurements.clone(),
        }
    }

    #[getter]
    fn ground_truth_object(&self, py: Python<'_>) -> Py<PyArray2<Complex64>> {
        self.ground_truth_object.clone_ref(py)
    }

    #[getter]
    fn true_model(&self) -> PyImagePlaneModel {
        PyImagePlaneModel {
            inner: self.true_model.clone(),
        }
    }

    #[getter]
    fn reconstruction_model(&self) -> PyImagePlaneModel {
        PyImagePlaneModel {
            inner: self.reconstruction_model.clone(),
        }
    }

    #[getter]
    fn ideal(&self) -> bool {
        self.ideal
    }

    #[getter]
    fn missing_frames(&self) -> Vec<usize> {
        self.missing_frames.clone()
    }

    #[getter]
    fn random_seed(&self) -> u64 {
        self.random_seed
    }
}

fn extract_synthetic_object(value: &Bound<'_, PyAny>) -> PyResult<SyntheticObject> {
    if let Ok(object) = value.extract::<PyRef<'_, PySyntheticObject>>() {
        return Ok(object.inner.clone());
    }
    let array = value
        .cast::<PyArray2<Complex64>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "object must be a complex128 NumPy array or SyntheticObject",
            )
        })?
        .readonly();
    Ok(SyntheticObject::new(core_array2(&array).map_err(to_py_err)?))
}

#[pyfunction]
#[pyo3(signature = (true_model, object, *, reconstruction_model=None, camera=None, illumination_errors=None, seed=0))]
pub(crate) fn simulate(
    py: Python<'_>,
    true_model: PyRef<'_, PyImagePlaneModel>,
    object: &Bound<'_, PyAny>,
    reconstruction_model: Option<PyRef<'_, PyImagePlaneModel>>,
    camera: Option<PyRef<'_, PyCameraModel>>,
    illumination_errors: Option<PyRef<'_, PyIlluminationAcquisitionErrors>>,
    seed: u64,
) -> PyResult<PySimulationResult> {
    let object = extract_synthetic_object(object)?;
    let true_model = true_model.inner.clone();
    let assumed_model = reconstruction_model.map(|model| model.inner.clone());
    let camera = camera.map(|camera| camera.inner.clone());
    let illumination_errors = illumination_errors.map(|errors| errors.inner.clone());
    let ideal = camera.is_none() && illumination_errors.is_none();
    let result = py
        .detach(move || {
            let mut simulator = if ideal {
                Simulator::ideal(true_model)
            } else {
                Simulator::new(true_model)
            }
            .object(object)
            .seed(seed);
            if let Some(model) = assumed_model {
                simulator = simulator.reconstruction_model(model);
            }
            if let Some(camera) = camera {
                simulator = simulator.camera(camera);
            }
            if let Some(errors) = illumination_errors {
                simulator = simulator.illumination_acquisition_errors(errors);
            }
            simulator.simulate()
        })
        .map_err(to_py_err)?;
    PySimulationResult::from_core(py, result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySyntheticObject>()?;
    module.add_class::<PySimulationResult>()?;
    module.add_function(wrap_pyfunction!(simulate, module)?)?;
    Ok(())
}
