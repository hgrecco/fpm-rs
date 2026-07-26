use std::{
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

use fpm_rs::{
    Complex64,
    reconstruction::{BundleArtifact, BundleVerificationResult, ResultBundle, read_bundle},
};
use numpy::{PyArray1, PyArray2, ndarray};
use pyo3::{prelude::*, types::PyDict};

use crate::{
    arrays::{array2_to_py, complex_array2_to_py, vec2_to_py},
    errors::to_py_err,
    reconstruction::{PyReconstructionResult, PyRuntimeInfo, diagnostics_to_py, evaluation_to_py},
};

#[derive(Clone, Copy, Debug)]
enum ArrayRole {
    Object,
    ObjectSpectrum,
    Pupil,
    PupilSupport,
    IlluminationCalibration,
    FrameGains,
    Background,
}

#[derive(Default)]
struct PythonBundleCache {
    object: Option<Py<PyArray2<Complex64>>>,
    object_spectrum: Option<Py<PyArray2<Complex64>>>,
    pupil: Option<Py<PyArray2<Complex64>>>,
    pupil_support: Option<Py<PyArray2<u8>>>,
    illumination_calibration: Option<Py<PyArray2<f64>>>,
    frame_gains: Option<Py<PyArray1<f64>>>,
    background: Option<Py<PyArray1<f64>>>,
    amplitude: Option<Py<PyArray2<f64>>>,
    phase: Option<Py<PyArray2<f64>>>,
    result: Option<Py<PyReconstructionResult>>,
}

struct PythonBundleState {
    bundle: ResultBundle,
    cache: Mutex<PythonBundleCache>,
}

impl PythonBundleState {
    fn cache(&self) -> MutexGuard<'_, PythonBundleCache> {
        self.cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "BundleArtifact",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBundleArtifact {
    inner: BundleArtifact,
}

#[pymethods]
impl PyBundleArtifact {
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path.clone()
    }

    #[getter]
    fn media_type(&self) -> &str {
        &self.inner.media_type
    }

    #[getter]
    fn byte_size(&self) -> u64 {
        self.inner.byte_size
    }

    #[getter]
    fn sha256(&self) -> &str {
        &self.inner.sha256
    }

    #[getter]
    fn role(&self) -> &str {
        &self.inner.role
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "BundleArray",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBundleArray {
    artifact: BundleArtifact,
    state: Arc<PythonBundleState>,
    role: ArrayRole,
}

#[pymethods]
impl PyBundleArray {
    #[getter]
    fn path(&self) -> PathBuf {
        self.artifact.path.clone()
    }

    #[getter]
    fn media_type(&self) -> &str {
        &self.artifact.media_type
    }

    #[getter]
    fn byte_size(&self) -> u64 {
        self.artifact.byte_size
    }

    #[getter]
    fn sha256(&self) -> &str {
        &self.artifact.sha256
    }

    #[getter]
    fn role(&self) -> &str {
        &self.artifact.role
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut cache = self.state.cache();
        match self.role {
            ArrayRole::Object => {
                if cache.object.is_none() {
                    let bundle = self.state.bundle.clone();
                    let values = py.detach(move || bundle.object()).map_err(to_py_err)?;
                    let array = complex_array2_to_py(py, (*values).clone())?;
                    make_read_only(py, &array)?;
                    cache.object = Some(array);
                }
                Ok(cache.object.as_ref().unwrap().clone_ref(py).into_any())
            }
            ArrayRole::ObjectSpectrum => {
                if cache.object_spectrum.is_none() {
                    let bundle = self.state.bundle.clone();
                    let values = py
                        .detach(move || bundle.object_spectrum())
                        .map_err(to_py_err)?;
                    let array = complex_array2_to_py(py, (*values).clone())?;
                    make_read_only(py, &array)?;
                    cache.object_spectrum = Some(array);
                }
                Ok(cache
                    .object_spectrum
                    .as_ref()
                    .unwrap()
                    .clone_ref(py)
                    .into_any())
            }
            ArrayRole::Pupil => {
                if cache.pupil.is_none() {
                    let bundle = self.state.bundle.clone();
                    let values = py.detach(move || bundle.pupil()).map_err(to_py_err)?;
                    let array = complex_array2_to_py(py, (*values).clone())?;
                    make_read_only(py, &array)?;
                    cache.pupil = Some(array);
                }
                Ok(cache.pupil.as_ref().unwrap().clone_ref(py).into_any())
            }
            ArrayRole::PupilSupport => {
                if cache.pupil_support.is_none() {
                    let bundle = self.state.bundle.clone();
                    let values = py
                        .detach(move || bundle.pupil_support())
                        .map_err(to_py_err)?;
                    let array = array2_to_py(py, (*values).clone())?;
                    make_read_only(py, &array)?;
                    cache.pupil_support = Some(array);
                }
                Ok(cache
                    .pupil_support
                    .as_ref()
                    .unwrap()
                    .clone_ref(py)
                    .into_any())
            }
            ArrayRole::IlluminationCalibration => {
                ensure_calibration_arrays(py, &self.state, &mut cache)?;
                Ok(cache
                    .illumination_calibration
                    .as_ref()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "bundle illumination calibration artifact has no value",
                        )
                    })?
                    .clone_ref(py)
                    .into_any())
            }
            ArrayRole::FrameGains => {
                ensure_calibration_arrays(py, &self.state, &mut cache)?;
                Ok(cache
                    .frame_gains
                    .as_ref()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "bundle frame-gain artifact has no value",
                        )
                    })?
                    .clone_ref(py)
                    .into_any())
            }
            ArrayRole::Background => {
                ensure_calibration_arrays(py, &self.state, &mut cache)?;
                Ok(cache
                    .background
                    .as_ref()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "bundle background artifact has no value",
                        )
                    })?
                    .clone_ref(py)
                    .into_any())
            }
        }
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "BundleTables",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBundleTables {
    #[pyo3(get)]
    summary: PyBundleArtifact,
    #[pyo3(get)]
    history: PyBundleArtifact,
    #[pyo3(get)]
    algorithm_metrics: Option<PyBundleArtifact>,
    #[pyo3(get)]
    iteration_diagnostics: Option<PyBundleArtifact>,
    #[pyo3(get)]
    frame_diagnostics: Option<PyBundleArtifact>,
    #[pyo3(get)]
    raw_frame_statistics: Option<PyBundleArtifact>,
    #[pyo3(get)]
    frame_evaluation: Option<PyBundleArtifact>,
    #[pyo3(get)]
    illumination_calibration: Option<PyBundleArtifact>,
    #[pyo3(get)]
    frame_calibration: Option<PyBundleArtifact>,
    #[pyo3(get)]
    scalar_diagnostics: Option<PyBundleArtifact>,
    #[pyo3(get)]
    metadata: Option<PyBundleArtifact>,
}

#[pymethods]
impl PyBundleTables {}

#[pyclass(
    module = "fpm_rs._core",
    name = "BundleArrays",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBundleArrays {
    #[pyo3(get)]
    object: PyBundleArray,
    #[pyo3(get)]
    object_spectrum: PyBundleArray,
    #[pyo3(get)]
    pupil: PyBundleArray,
    #[pyo3(get)]
    pupil_support: PyBundleArray,
    #[pyo3(get)]
    illumination_calibration: Option<PyBundleArray>,
    #[pyo3(get)]
    frame_gains: Option<PyBundleArray>,
    #[pyo3(get)]
    background: Option<PyBundleArray>,
}

#[pymethods]
impl PyBundleArrays {}

#[pyclass(
    module = "fpm_rs._core",
    name = "BundlePreviews",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBundlePreviews {
    #[pyo3(get)]
    object_amplitude: Option<PyBundleArtifact>,
    #[pyo3(get)]
    object_phase: Option<PyBundleArtifact>,
    #[pyo3(get)]
    pupil_amplitude: Option<PyBundleArtifact>,
    #[pyo3(get)]
    pupil_phase: Option<PyBundleArtifact>,
    #[pyo3(get)]
    fourier_coverage: Option<PyBundleArtifact>,
}

#[pymethods]
impl PyBundlePreviews {}

#[pyclass(module = "fpm_rs._core", name = "BundleVerificationResult", frozen)]
pub(crate) struct PyBundleVerificationResult {
    #[pyo3(get)]
    artifact_count: usize,
    #[pyo3(get)]
    total_bytes: u64,
}

impl From<BundleVerificationResult> for PyBundleVerificationResult {
    fn from(value: BundleVerificationResult) -> Self {
        Self {
            artifact_count: value.artifact_count,
            total_bytes: value.total_bytes,
        }
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "ResultBundle",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyResultBundle {
    state: Arc<PythonBundleState>,
    tables: PyBundleTables,
    arrays: PyBundleArrays,
    previews: PyBundlePreviews,
}

impl PyResultBundle {
    pub(crate) fn from_core(bundle: ResultBundle) -> Self {
        let state = Arc::new(PythonBundleState {
            bundle: bundle.clone(),
            cache: Mutex::new(PythonBundleCache::default()),
        });
        let tables = PyBundleTables {
            summary: artifact(bundle.tables.summary.clone()),
            history: artifact(bundle.tables.history.clone()),
            algorithm_metrics: bundle.tables.algorithm_metrics.clone().map(artifact),
            iteration_diagnostics: bundle.tables.iteration_diagnostics.clone().map(artifact),
            frame_diagnostics: bundle.tables.frame_diagnostics.clone().map(artifact),
            raw_frame_statistics: bundle.tables.raw_frame_statistics.clone().map(artifact),
            frame_evaluation: bundle.tables.frame_evaluation.clone().map(artifact),
            illumination_calibration: bundle.tables.illumination_calibration.clone().map(artifact),
            frame_calibration: bundle.tables.frame_calibration.clone().map(artifact),
            scalar_diagnostics: bundle.tables.scalar_diagnostics.clone().map(artifact),
            metadata: bundle.tables.metadata.clone().map(artifact),
        };
        let arrays = PyBundleArrays {
            object: array_artifact(
                bundle.arrays.object.clone(),
                state.clone(),
                ArrayRole::Object,
            ),
            object_spectrum: array_artifact(
                bundle.arrays.object_spectrum.clone(),
                state.clone(),
                ArrayRole::ObjectSpectrum,
            ),
            pupil: array_artifact(bundle.arrays.pupil.clone(), state.clone(), ArrayRole::Pupil),
            pupil_support: array_artifact(
                bundle.arrays.pupil_support.clone(),
                state.clone(),
                ArrayRole::PupilSupport,
            ),
            illumination_calibration: bundle.arrays.illumination_calibration.clone().map(|value| {
                array_artifact(value, state.clone(), ArrayRole::IlluminationCalibration)
            }),
            frame_gains: bundle
                .arrays
                .frame_gains
                .clone()
                .map(|value| array_artifact(value, state.clone(), ArrayRole::FrameGains)),
            background: bundle
                .arrays
                .background
                .clone()
                .map(|value| array_artifact(value, state.clone(), ArrayRole::Background)),
        };
        let previews = PyBundlePreviews {
            object_amplitude: bundle.previews.object_amplitude.clone().map(artifact),
            object_phase: bundle.previews.object_phase.clone().map(artifact),
            pupil_amplitude: bundle.previews.pupil_amplitude.clone().map(artifact),
            pupil_phase: bundle.previews.pupil_phase.clone().map(artifact),
            fourier_coverage: bundle.previews.fourier_coverage.clone().map(artifact),
        };
        Self {
            state,
            tables,
            arrays,
            previews,
        }
    }
}

#[pymethods]
impl PyResultBundle {
    #[getter]
    fn path(&self) -> PathBuf {
        self.state.bundle.path.clone()
    }

    #[getter]
    fn manifest_path(&self) -> PathBuf {
        self.state.bundle.manifest_path.clone()
    }

    #[getter]
    fn run_id(&self) -> &str {
        &self.state.bundle.run_id
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.state.bundle.label.as_deref()
    }

    #[getter]
    fn tables(&self) -> PyBundleTables {
        self.tables.clone()
    }

    #[getter]
    fn arrays(&self) -> PyBundleArrays {
        self.arrays.clone()
    }

    #[getter]
    fn previews(&self) -> PyBundlePreviews {
        self.previews.clone()
    }

    #[getter]
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyReconstructionResult>> {
        {
            let cache = self.state.cache();
            if let Some(result) = &cache.result {
                return Ok(result.clone_ref(py));
            }
        }
        let bundle = self.state.bundle.clone();
        let result = py.detach(move || bundle.result()).map_err(to_py_err)?;
        let mut cache = self.state.cache();
        if let Some(value) = &cache.result {
            return Ok(value.clone_ref(py));
        }
        ensure_result_arrays(py, &result, &mut cache)?;
        let value = Py::new(
            py,
            PyReconstructionResult {
                object: cache.object.as_ref().unwrap().clone_ref(py),
                amplitude: cache.amplitude.as_ref().unwrap().clone_ref(py),
                phase: cache.phase.as_ref().unwrap().clone_ref(py),
                object_spectrum: cache.object_spectrum.as_ref().unwrap().clone_ref(py),
                recovered_pupil: cache.pupil.as_ref().unwrap().clone_ref(py),
                pupil_support: cache.pupil_support.as_ref().unwrap().clone_ref(py),
                calibrated_illumination: cache
                    .illumination_calibration
                    .as_ref()
                    .map(|value| value.clone_ref(py)),
                recovered_frame_gains: cache.frame_gains.as_ref().map(|value| value.clone_ref(py)),
                recovered_background: cache.background.as_ref().map(|value| value.clone_ref(py)),
                trace: result
                    .trace
                    .iterations
                    .iter()
                    .map(|record| (record.iteration, record.objective, record.elapsed_seconds))
                    .collect(),
                algorithm_metrics: result
                    .trace
                    .algorithm_metrics
                    .iter()
                    .map(|record| {
                        (
                            record.iteration,
                            record.namespace.clone(),
                            record.metric.clone(),
                            record.value,
                        )
                    })
                    .collect(),
                scalar_diagnostics: result.scalar_diagnostics.clone(),
                runtime: PyRuntimeInfo {
                    elapsed_seconds: result.runtime.elapsed_seconds,
                    completed_iterations: result.runtime.completed_iterations,
                    stopped_early: result.runtime.stopped_early,
                    algorithm: result.runtime.algorithm.clone(),
                },
                metadata: result.metadata.clone(),
            },
        )?;
        cache.result = Some(value.clone_ref(py));
        Ok(value)
    }

    #[getter]
    fn diagnostics(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let bundle = self.state.bundle.clone();
        py.detach(move || bundle.diagnostics())
            .map_err(to_py_err)?
            .as_deref()
            .map(|value| diagnostics_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn evaluation(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let bundle = self.state.bundle.clone();
        py.detach(move || bundle.evaluation())
            .map_err(to_py_err)?
            .as_deref()
            .map(|value| evaluation_to_py(py, value))
            .transpose()
    }

    fn verify(&self, py: Python<'_>) -> PyResult<PyBundleVerificationResult> {
        let bundle = self.state.bundle.clone();
        py.detach(move || bundle.verify())
            .map(Into::into)
            .map_err(to_py_err)
    }

    fn clear_cache(&self) {
        self.state.bundle.clear_cache();
        *self.state.cache() = PythonBundleCache::default();
    }
}

#[pyfunction(name = "read_bundle")]
fn read_bundle_py(py: Python<'_>, path: PathBuf) -> PyResult<PyResultBundle> {
    py.detach(move || read_bundle(path))
        .map(PyResultBundle::from_core)
        .map_err(to_py_err)
}

pub(crate) fn artifact(value: BundleArtifact) -> PyBundleArtifact {
    PyBundleArtifact { inner: value }
}

fn array_artifact(
    artifact: BundleArtifact,
    state: Arc<PythonBundleState>,
    role: ArrayRole,
) -> PyBundleArray {
    PyBundleArray {
        artifact,
        state,
        role,
    }
}

fn make_read_only<T: numpy::Element>(py: Python<'_>, array: &Py<PyArray2<T>>) -> PyResult<()> {
    array.bind(py).call_method1("setflags", (false,))?;
    Ok(())
}

fn make_read_only_1<T: numpy::Element>(py: Python<'_>, array: &Py<PyArray1<T>>) -> PyResult<()> {
    array.bind(py).call_method1("setflags", (false,))?;
    Ok(())
}

fn ensure_calibration_arrays(
    py: Python<'_>,
    state: &PythonBundleState,
    cache: &mut PythonBundleCache,
) -> PyResult<()> {
    let bundle = state.bundle.clone();
    let result = py.detach(move || bundle.result()).map_err(to_py_err)?;
    if let Some(values) = &result.calibrated_illumination
        && cache.illumination_calibration.is_none()
    {
        let flattened = values
            .iter()
            .flat_map(|&(row, column)| [row, column])
            .collect();
        let array = vec2_to_py(py, (values.len(), 2), flattened)?;
        make_read_only(py, &array)?;
        cache.illumination_calibration = Some(array);
    }
    if let Some(values) = &result.recovered_frame_gains
        && cache.frame_gains.is_none()
    {
        let array =
            PyArray1::from_owned_array(py, ndarray::Array1::from_vec(values.clone())).unbind();
        make_read_only_1(py, &array)?;
        cache.frame_gains = Some(array);
    }
    if let Some(values) = &result.recovered_background
        && cache.background.is_none()
    {
        let array =
            PyArray1::from_owned_array(py, ndarray::Array1::from_vec(values.clone())).unbind();
        make_read_only_1(py, &array)?;
        cache.background = Some(array);
    }
    Ok(())
}

fn ensure_result_arrays(
    py: Python<'_>,
    result: &fpm_rs::reconstruction::ReconstructionResult,
    cache: &mut PythonBundleCache,
) -> PyResult<()> {
    if cache.object.is_none() {
        let array = complex_array2_to_py(py, result.object.clone())?;
        make_read_only(py, &array)?;
        cache.object = Some(array);
    }
    if cache.object_spectrum.is_none() {
        let array = complex_array2_to_py(py, result.object_spectrum.clone())?;
        make_read_only(py, &array)?;
        cache.object_spectrum = Some(array);
    }
    if cache.pupil.is_none() {
        let array = complex_array2_to_py(py, result.recovered_pupil.values().to_owned())?;
        make_read_only(py, &array)?;
        cache.pupil = Some(array);
    }
    if cache.pupil_support.is_none() {
        let array = array2_to_py(py, result.recovered_pupil.support().to_owned())?;
        make_read_only(py, &array)?;
        cache.pupil_support = Some(array);
    }
    if cache.amplitude.is_none() {
        let array = array2_to_py(py, result.amplitude.clone())?;
        make_read_only(py, &array)?;
        cache.amplitude = Some(array);
    }
    if cache.phase.is_none() {
        let array = array2_to_py(py, result.phase.clone())?;
        make_read_only(py, &array)?;
        cache.phase = Some(array);
    }
    if let Some(values) = &result.calibrated_illumination
        && cache.illumination_calibration.is_none()
    {
        let flattened = values
            .iter()
            .flat_map(|&(row, column)| [row, column])
            .collect();
        let array = vec2_to_py(py, (values.len(), 2), flattened)?;
        make_read_only(py, &array)?;
        cache.illumination_calibration = Some(array);
    }
    if let Some(values) = &result.recovered_frame_gains
        && cache.frame_gains.is_none()
    {
        let array =
            PyArray1::from_owned_array(py, ndarray::Array1::from_vec(values.clone())).unbind();
        make_read_only_1(py, &array)?;
        cache.frame_gains = Some(array);
    }
    if let Some(values) = &result.recovered_background
        && cache.background.is_none()
    {
        let array =
            PyArray1::from_owned_array(py, ndarray::Array1::from_vec(values.clone())).unbind();
        make_read_only_1(py, &array)?;
        cache.background = Some(array);
    }
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(read_bundle_py, module)?)?;
    module.add_class::<PyBundleArtifact>()?;
    module.add_class::<PyBundleArray>()?;
    module.add_class::<PyBundleTables>()?;
    module.add_class::<PyBundleArrays>()?;
    module.add_class::<PyBundlePreviews>()?;
    module.add_class::<PyBundleVerificationResult>()?;
    module.add_class::<PyResultBundle>()?;
    Ok(())
}
