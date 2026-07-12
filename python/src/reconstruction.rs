use std::{
    collections::BTreeMap,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use fpm_rs::{
    Complex64, Error,
    algorithms::{
        Admm, AlternatingProjection, Epry, Fpie, GradientDescent, ReconstructionAlgorithm,
    },
    callbacks::{
        Callback, CallbackAction, CallbackHook, CheckpointEvery, CsvLogger, ProgressLogger,
        SaveImageEvery, SavePupilEvery, SaveResidualsEvery, StepContext, StopOnPlateau,
    },
    diagnostics::{
        DiagnosticRecorder as CoreDiagnosticRecorder, DiagnosticRecorderConfig, DiagnosticRequest,
        LossType, ReconstructionDiagnostics,
    },
    measurements::{FrameMetadata, MeasurementRead, MeasurementStack},
    reconstruction::{
        FrameSchedule, ReconstructionCheckpoint, ReconstructionProblem, ReconstructionResult,
        RunOptions, Runner,
    },
};
use numpy::{PyArray1, PyArray2, ndarray};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use crate::{
    arrays::{array2_to_py, complex_array2_to_py, vec2_to_py},
    errors::to_py_err,
    measurements::extract_measurements,
    model::PyImagePlaneModel,
};

#[derive(Clone)]
struct SharedMeasurementStack(Arc<MeasurementStack>);

impl MeasurementRead for SharedMeasurementStack {
    fn frame_count(&self) -> usize {
        self.0.frame_count()
    }

    fn image_shape(&self) -> (usize, usize) {
        self.0.image_shape()
    }

    fn frame_len(&self) -> usize {
        self.0.frame_len()
    }

    fn frame(&self, index: usize) -> fpm_rs::Result<impl Deref<Target = [f64]> + '_> {
        self.0.frame(index)
    }

    fn frame_weight(&self, index: usize) -> fpm_rs::Result<f64> {
        self.0.frame_weight(index)
    }

    fn frame_mask(&self, index: usize) -> fpm_rs::Result<Option<&[u8]>> {
        self.0.frame_mask(index)
    }

    fn frame_metadata(&self) -> &[FrameMetadata] {
        &self.0.frame_metadata
    }

    fn validate(&self) -> fpm_rs::Result<()> {
        self.0.validate()
    }
}

#[pyclass(module = "fpm_rs._core", name = "ReconstructionProblem", frozen)]
#[derive(Clone)]
pub(crate) struct PyReconstructionProblem {
    inner: ReconstructionProblem<SharedMeasurementStack>,
}

impl PyReconstructionProblem {
    pub(crate) fn from_parts(
        measurements: MeasurementStack,
        model: fpm_rs::model::ImagePlaneModel,
        name: Option<String>,
    ) -> fpm_rs::Result<Self> {
        let mut inner =
            ReconstructionProblem::new(SharedMeasurementStack(Arc::new(measurements)), model)?;
        inner.name = name;
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyReconstructionProblem {
    #[new]
    #[pyo3(signature = (measurements, model, *, frame_weights=None, masks=None, name=None))]
    fn new(
        py: Python<'_>,
        measurements: &Bound<'_, PyAny>,
        model: PyRef<'_, PyImagePlaneModel>,
        frame_weights: Option<Vec<f64>>,
        masks: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let measurements = extract_measurements(measurements, frame_weights, masks)?;
        let model = model.inner.clone();
        let mut inner = py
            .detach(move || {
                ReconstructionProblem::new(SharedMeasurementStack(measurements), (*model).clone())
            })
            .map_err(to_py_err)?;
        inner.name = name;
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.measurements.frame_count()
    }

    #[getter]
    fn image_shape(&self) -> (usize, usize) {
        self.inner.model.image_shape
    }

    #[getter]
    fn reconstruction_shape(&self) -> (usize, usize) {
        self.inner.model.reconstruction_shape
    }
}

#[pyclass(module = "fpm_rs._core", name = "ReconstructionCheckpoint", frozen)]
#[derive(Clone)]
pub(crate) struct PyReconstructionCheckpoint {
    // Checkpoints can hold large object spectra. Shared immutable ownership
    // keeps saving them from cloning that state while the GIL is held.
    inner: Arc<ReconstructionCheckpoint>,
}

#[pymethods]
impl PyReconstructionCheckpoint {
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                py.detach(move || ReconstructionCheckpoint::load(path))
                    .map_err(to_py_err)?,
            ),
        })
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let checkpoint = self.inner.clone();
        py.detach(move || checkpoint.save(path)).map_err(to_py_err)
    }

    #[getter]
    fn format_version(&self) -> u32 {
        self.inner.format_version
    }

    #[getter]
    fn completed_iterations(&self) -> usize {
        self.inner.completed_iterations
    }
}

#[pyclass(module = "fpm_rs._core", name = "RuntimeInfo", frozen)]
#[derive(Clone)]
pub(crate) struct PyRuntimeInfo {
    #[pyo3(get)]
    elapsed_seconds: f64,
    #[pyo3(get)]
    completed_iterations: usize,
    #[pyo3(get)]
    stopped_early: bool,
    #[pyo3(get)]
    algorithm: String,
}

#[pyclass(module = "fpm_rs._core", name = "ReconstructionResult", frozen)]
pub(crate) struct PyReconstructionResult {
    object: Py<PyArray2<Complex64>>,
    amplitude: Py<PyArray2<f64>>,
    phase: Py<PyArray2<f64>>,
    object_spectrum: Py<PyArray2<Complex64>>,
    recovered_pupil: Py<PyArray2<Complex64>>,
    pupil_support: Py<PyArray2<u8>>,
    calibrated_illumination: Option<Py<PyArray2<f64>>>,
    recovered_frame_gains: Option<Py<PyArray1<f64>>>,
    recovered_background: Option<Py<PyArray1<f64>>>,
    history: Vec<(usize, f64, f64)>,
    admm_residual_history: Vec<(usize, f64, f64)>,
    diagnostics: BTreeMap<String, f64>,
    runtime: PyRuntimeInfo,
    metadata: BTreeMap<String, String>,
}

impl PyReconstructionResult {
    fn from_core(py: Python<'_>, result: ReconstructionResult) -> PyResult<Self> {
        let pupil_shape = result.recovered_pupil.values.shape();
        let pupil_support = result
            .recovered_pupil
            .support
            .iter()
            .map(|&value| u8::from(value))
            .collect();
        let calibrated_illumination = result
            .calibrated_illumination
            .map(|values| {
                let rows = values.len();
                let values = values
                    .into_iter()
                    .flat_map(|(row, column)| [row, column])
                    .collect();
                vec2_to_py(py, (rows, 2), values)
            })
            .transpose()?;
        let recovered_frame_gains = result.recovered_frame_gains.map(|values| {
            PyArray1::from_owned_array(py, ndarray::Array1::from_vec(values)).unbind()
        });
        let recovered_background = result.recovered_background.map(|values| {
            PyArray1::from_owned_array(py, ndarray::Array1::from_vec(values)).unbind()
        });
        let admm_residual_history = result
            .history
            .iterations
            .iter()
            .filter_map(|record| {
                Some((
                    record.iteration,
                    record.admm_primal_residual_rms?,
                    record.admm_dual_residual_rms?,
                ))
            })
            .collect();
        let history = result
            .history
            .iterations
            .into_iter()
            .map(|record| (record.iteration, record.loss, record.elapsed_seconds))
            .collect();
        let runtime = PyRuntimeInfo {
            elapsed_seconds: result.runtime.elapsed_seconds,
            completed_iterations: result.runtime.completed_iterations,
            stopped_early: result.runtime.stopped_early,
            algorithm: result.runtime.algorithm,
        };
        Ok(Self {
            object: complex_array2_to_py(py, result.object)?,
            amplitude: array2_to_py(py, result.amplitude)?,
            phase: array2_to_py(py, result.phase)?,
            object_spectrum: complex_array2_to_py(py, result.object_spectrum)?,
            recovered_pupil: complex_array2_to_py(py, result.recovered_pupil.values)?,
            pupil_support: vec2_to_py(py, pupil_shape, pupil_support)?,
            calibrated_illumination,
            recovered_frame_gains,
            recovered_background,
            history,
            admm_residual_history,
            diagnostics: result.diagnostics,
            runtime,
            metadata: result.metadata,
        })
    }
}

#[pymethods]
impl PyReconstructionResult {
    #[getter]
    fn object(&self, py: Python<'_>) -> Py<PyArray2<Complex64>> {
        self.object.clone_ref(py)
    }

    #[getter]
    fn amplitude(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.amplitude.clone_ref(py)
    }

    #[getter]
    fn phase(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.phase.clone_ref(py)
    }

    #[getter]
    fn object_spectrum(&self, py: Python<'_>) -> Py<PyArray2<Complex64>> {
        self.object_spectrum.clone_ref(py)
    }

    #[getter]
    fn recovered_pupil(&self, py: Python<'_>) -> Py<PyArray2<Complex64>> {
        self.recovered_pupil.clone_ref(py)
    }

    #[getter]
    fn pupil_support(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.pupil_support.clone_ref(py)
    }

    #[getter]
    fn calibrated_illumination(&self, py: Python<'_>) -> Option<Py<PyArray2<f64>>> {
        self.calibrated_illumination
            .as_ref()
            .map(|value| value.clone_ref(py))
    }

    #[getter]
    fn recovered_frame_gains(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.recovered_frame_gains
            .as_ref()
            .map(|value| value.clone_ref(py))
    }

    #[getter]
    fn recovered_background(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.recovered_background
            .as_ref()
            .map(|value| value.clone_ref(py))
    }

    #[getter]
    fn history(&self) -> Vec<(usize, f64, f64)> {
        self.history.clone()
    }

    /// `(iteration, primal_rms, dual_rms)` records for ADMM runs.
    #[getter]
    fn admm_residual_history(&self) -> Vec<(usize, f64, f64)> {
        self.admm_residual_history.clone()
    }

    #[getter]
    fn diagnostics(&self) -> BTreeMap<String, f64> {
        self.diagnostics.clone()
    }

    #[getter]
    fn runtime(&self) -> PyRuntimeInfo {
        self.runtime.clone()
    }

    #[getter]
    fn metadata(&self) -> BTreeMap<String, String> {
        self.metadata.clone()
    }

    #[getter]
    fn final_loss(&self) -> Option<f64> {
        self.history.last().map(|record| record.1)
    }
}

/// Rust-backed reconstruction diagnostic recorder.
///
/// Add this object to an algorithm's ``callbacks`` argument and call
/// ``diagnostics()`` after the run. Recording executes entirely in Rust.
#[pyclass(module = "fpm_rs._core", name = "DiagnosticRecorder", frozen)]
pub(crate) struct PyDiagnosticRecorder {
    inner: CoreDiagnosticRecorder,
    mode: String,
    every: usize,
}

#[pymethods]
impl PyDiagnosticRecorder {
    #[new]
    #[pyo3(signature = (mode="basic", *, every=1))]
    fn new(mode: &str, every: usize) -> PyResult<Self> {
        if every == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "every must be greater than zero",
            ));
        }
        let mut config = DiagnosticRecorderConfig {
            every,
            ..DiagnosticRecorderConfig::default()
        };
        match mode {
            "minimal" => {}
            "basic" => {
                config.record_coverage = true;
            }
            "debug" => {
                config.record_frame_summaries = true;
                config.record_raw_stack_stats = true;
                config.record_coverage = true;
            }
            // ReconstructionProblem does not currently carry ground truth, so
            // simulation mode intentionally remains a cheap basic preset.
            "simulation" => {
                config.record_coverage = true;
            }
            unknown => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown diagnostic recorder mode '{unknown}'; expected one of 'minimal', 'basic', 'debug', 'simulation'"
                )));
            }
        }
        Ok(Self {
            inner: CoreDiagnosticRecorder::new(config),
            mode: mode.to_owned(),
            every,
        })
    }

    #[getter]
    fn mode(&self) -> &str {
        &self.mode
    }

    #[getter]
    fn every(&self) -> usize {
        self.every
    }

    fn diagnostics(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        diagnostics_to_py(py, &self.inner.diagnostics())
    }

    fn to_json(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let recorder = self.inner.clone();
        py.detach(move || recorder.diagnostics().to_json_file(path))
            .map_err(to_py_err)
    }
}

fn diagnostics_to_py(
    py: Python<'_>,
    diagnostics: &ReconstructionDiagnostics,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);

    let iteration_history = PyList::empty(py);
    for entry in &diagnostics.iteration_history {
        let value = PyDict::new(py);
        value.set_item("iteration", entry.iteration)?;
        value.set_item("total_loss", entry.total_loss)?;
        value.set_item("data_loss", entry.data_loss)?;
        value.set_item("regularization_loss", entry.regularization_loss)?;
        value.set_item("object_relative_change", entry.object_relative_change)?;
        value.set_item("pupil_relative_change", entry.pupil_relative_change)?;
        value.set_item("median_frame_loss", entry.median_frame_loss)?;
        value.set_item("worst_frame_loss", entry.worst_frame_loss)?;
        value.set_item("elapsed_ms", entry.elapsed_ms)?;
        iteration_history.append(value)?;
    }
    output.set_item("iteration_history", iteration_history)?;

    let frame_diagnostics = PyList::empty(py);
    for entry in &diagnostics.frame_diagnostics {
        let value = PyDict::new(py);
        value.set_item("iteration", entry.iteration)?;
        value.set_item("frame_index", entry.frame_index)?;
        value.set_item("illumination_index", entry.illumination_index)?;
        value.set_item("measured_sum", entry.measured_sum)?;
        value.set_item("predicted_sum", entry.predicted_sum)?;
        value.set_item("residual_l1", entry.residual_l1)?;
        value.set_item("residual_l2", entry.residual_l2)?;
        value.set_item("residual_mean", entry.residual_mean)?;
        value.set_item("residual_std", entry.residual_std)?;
        value.set_item("residual_max_abs", entry.residual_max_abs)?;
        value.set_item("normalized_l2", entry.normalized_l2)?;
        value.set_item("saturated_pixels", entry.saturated_pixels)?;
        frame_diagnostics.append(value)?;
    }
    output.set_item("frame_diagnostics", frame_diagnostics)?;

    let raw_frame_stats = PyList::empty(py);
    for entry in &diagnostics.raw_frame_stats {
        let value = PyDict::new(py);
        value.set_item("frame_index", entry.frame_index)?;
        value.set_item("mean", entry.mean)?;
        value.set_item("std", entry.std)?;
        value.set_item("min", entry.min)?;
        value.set_item("max", entry.max)?;
        value.set_item("sum", entry.sum)?;
        value.set_item("saturated_pixels", entry.saturated_pixels)?;
        value.set_item("zero_pixels", entry.zero_pixels)?;
        raw_frame_stats.append(value)?;
    }
    output.set_item("raw_frame_stats", raw_frame_stats)?;

    if let Some(coverage) = &diagnostics.coverage {
        let value = PyDict::new(py);
        value.set_item("synthetic_na", coverage.synthetic_na)?;
        value.set_item("pupil_radius_px", coverage.pupil_radius_px)?;
        value.set_item(
            "pupil_centers_px",
            coverage
                .pupil_centers_px
                .iter()
                .map(|center| (center[0], center[1]))
                .collect::<Vec<_>>(),
        )?;
        value.set_item("illumination_na", &coverage.illumination_na)?;
        let crop_indices = PyList::empty(py);
        for crop in &coverage.crop_indices {
            let item = PyDict::new(py);
            item.set_item("illumination_index", crop.illumination_index)?;
            item.set_item("x_start", crop.x_start)?;
            item.set_item("x_end", crop.x_end)?;
            item.set_item("y_start", crop.y_start)?;
            item.set_item("y_end", crop.y_end)?;
            item.set_item("center_x", crop.center_x)?;
            item.set_item("center_y", crop.center_y)?;
            crop_indices.append(item)?;
        }
        value.set_item("crop_indices", crop_indices)?;
        value.set_item("overlap_shape", coverage.overlap_shape)?;
        output.set_item("coverage", value)?;
    } else {
        output.set_item("coverage", py.None())?;
    }

    if let Some(metrics) = &diagnostics.ground_truth_metrics {
        let value = PyDict::new(py);
        value.set_item("amplitude_rmse", metrics.amplitude_rmse)?;
        value.set_item("amplitude_nrmse", metrics.amplitude_nrmse)?;
        value.set_item("complex_rmse", metrics.complex_rmse)?;
        value.set_item("complex_nrmse", metrics.complex_nrmse)?;
        value.set_item("phase_rmse", metrics.phase_rmse)?;
        value.set_item("phase_mae", metrics.phase_mae)?;
        value.set_item("fourier_nrmse", metrics.fourier_nrmse)?;
        output.set_item("ground_truth_metrics", value)?;
    } else {
        output.set_item("ground_truth_metrics", py.None())?;
    }

    Ok(output.unbind())
}

#[pyclass(module = "fpm_rs._core", name = "ProgressLogger", frozen)]
pub(crate) struct PyProgressLogger {
    every: usize,
}

#[pymethods]
impl PyProgressLogger {
    #[new]
    #[pyo3(signature = (*, every=1))]
    fn new(every: usize) -> Self {
        Self {
            every: every.max(1),
        }
    }
}

#[pyclass(module = "fpm_rs._core", name = "CheckpointEvery", frozen)]
pub(crate) struct PyCheckpointEvery {
    every: usize,
    directory: PathBuf,
}

#[pymethods]
impl PyCheckpointEvery {
    #[new]
    fn new(every: usize, directory: PathBuf) -> Self {
        Self {
            every: every.max(1),
            directory,
        }
    }
}

#[pyclass(module = "fpm_rs._core", name = "CsvLogger", frozen)]
pub(crate) struct PyCsvLogger {
    path: PathBuf,
}

#[pymethods]
impl PyCsvLogger {
    #[new]
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

#[pyclass(module = "fpm_rs._core", name = "StopOnPlateau", frozen)]
pub(crate) struct PyStopOnPlateau {
    patience: usize,
    minimum_improvement: f64,
}

#[pymethods]
impl PyStopOnPlateau {
    #[new]
    #[pyo3(signature = (patience, *, minimum_improvement=0.0))]
    fn new(patience: usize, minimum_improvement: f64) -> Self {
        Self {
            patience,
            minimum_improvement,
        }
    }
}

macro_rules! directory_callback {
    ($rust:ident, $python:literal) => {
        #[pyclass(module = "fpm_rs._core", name = $python, frozen)]
        pub(crate) struct $rust {
            every: usize,
            directory: PathBuf,
        }

        #[pymethods]
        impl $rust {
            #[new]
            fn new(every: usize, directory: PathBuf) -> Self {
                Self {
                    every: every.max(1),
                    directory,
                }
            }
        }
    };
}

directory_callback!(PySaveImageEvery, "SaveImageEvery");
directory_callback!(PySavePupilEvery, "SavePupilEvery");
directory_callback!(PySaveResidualsEvery, "SaveResidualsEvery");

#[pyclass(module = "fpm_rs._core", name = "IterationCallback", frozen)]
pub(crate) struct PyIterationCallback {
    callable: Py<PyAny>,
    every: usize,
}

#[pymethods]
impl PyIterationCallback {
    #[new]
    #[pyo3(signature = (callable, *, every=1))]
    fn new(callable: Py<PyAny>, every: usize, py: Python<'_>) -> PyResult<Self> {
        if !callable.bind(py).is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "callable must be callable",
            ));
        }
        Ok(Self {
            callable,
            every: every.max(1),
        })
    }
}

struct PythonIterationCallback {
    callable: Py<PyAny>,
    every: usize,
    error: Arc<Mutex<Option<PyErr>>>,
}

impl Callback for PythonIterationCallback {
    fn requires(&self) -> Vec<DiagnosticRequest> {
        vec![DiagnosticRequest::Loss]
    }

    fn requires_for(&self, hook: CallbackHook, iteration: usize) -> Vec<DiagnosticRequest> {
        if hook == CallbackHook::IterationEnd && iteration.is_multiple_of(self.every) {
            self.requires()
        } else {
            Vec::new()
        }
    }

    fn on_iteration_end(&mut self, context: &StepContext<'_>) -> fpm_rs::Result<CallbackAction> {
        if !context.iteration.is_multiple_of(self.every) {
            return Ok(CallbackAction::Continue);
        }
        Python::attach(|py| {
            let values = PyDict::new(py);
            values.set_item("iteration", context.iteration)?;
            values.set_item("loss", context.diagnostics.loss)?;
            values.set_item("problem_name", context.problem_name)?;
            let response = self.callable.bind(py).call1((values,));
            match response {
                Ok(response) => Ok(if response.is_none() || response.is_truthy()? {
                    CallbackAction::Continue
                } else {
                    CallbackAction::Stop
                }),
                Err(error) => {
                    *lock_callback_error(&self.error) = Some(error);
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Python iteration callback failed",
                    ))
                }
            }
        })
        .map_err(|error| Error::Unsupported(error.to_string()))
    }
}

type CallbackErrors = Vec<Arc<Mutex<Option<PyErr>>>>;

/// Python exceptions are best-effort delivery state. Recover its payload after
/// a panic rather than turning a later exception check into a second panic.
fn lock_callback_error(
    error: &Arc<Mutex<Option<PyErr>>>,
) -> std::sync::MutexGuard<'_, Option<PyErr>> {
    error
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn build_callbacks(
    py: Python<'_>,
    callbacks: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<Box<dyn Callback>>, CallbackErrors)> {
    let mut built: Vec<Box<dyn Callback>> = Vec::new();
    let mut errors = Vec::new();
    let Some(callbacks) = callbacks else {
        return Ok((built, errors));
    };
    for item in callbacks.try_iter()? {
        let item = item?;
        if let Ok(callback) = item.extract::<PyRef<'_, PyProgressLogger>>() {
            built.push(Box::new(ProgressLogger::new(callback.every)));
        } else if let Ok(callback) = item.extract::<PyRef<'_, PyCheckpointEvery>>() {
            built.push(Box::new(CheckpointEvery::new(
                callback.every,
                callback.directory.clone(),
            )));
        } else if let Ok(callback) = item.extract::<PyRef<'_, PyCsvLogger>>() {
            built.push(Box::new(CsvLogger::new(callback.path.clone())));
        } else if let Ok(callback) = item.extract::<PyRef<'_, PyStopOnPlateau>>() {
            built.push(Box::new(StopOnPlateau::new(
                callback.patience,
                callback.minimum_improvement,
            )));
        } else if let Ok(callback) = item.extract::<PyRef<'_, PySaveImageEvery>>() {
            built.push(Box::new(SaveImageEvery::new(
                callback.every,
                callback.directory.clone(),
            )));
        } else if let Ok(callback) = item.extract::<PyRef<'_, PySavePupilEvery>>() {
            built.push(Box::new(SavePupilEvery::new(
                callback.every,
                callback.directory.clone(),
            )));
        } else if let Ok(callback) = item.extract::<PyRef<'_, PySaveResidualsEvery>>() {
            built.push(Box::new(SaveResidualsEvery::new(
                callback.every,
                callback.directory.clone(),
            )));
        } else if let Ok(callback) = item.extract::<PyRef<'_, PyDiagnosticRecorder>>() {
            built.push(Box::new(callback.inner.clone()));
        } else if let Ok(callback) = item.extract::<PyRef<'_, PyIterationCallback>>() {
            let error = Arc::new(Mutex::new(None));
            built.push(Box::new(PythonIterationCallback {
                callable: callback.callable.clone_ref(py),
                every: callback.every,
                error: error.clone(),
            }));
            errors.push(error);
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported callback object",
            ));
        }
    }
    Ok((built, errors))
}

fn parse_loss_type(value: &str) -> PyResult<LossType> {
    match value {
        "amplitude_mse" => Ok(LossType::AmplitudeMse),
        "intensity_mse" => Ok(LossType::IntensityMse),
        "poisson_nll" => Ok(LossType::PoissonNegativeLogLikelihood),
        "huber_amplitude" => Ok(LossType::HuberAmplitude),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "loss_type must be amplitude_mse, intensity_mse, poisson_nll, or huber_amplitude",
        )),
    }
}

fn parse_schedule(value: &str, seed: u64) -> PyResult<FrameSchedule> {
    match value {
        "sequential" => Ok(FrameSchedule::Sequential),
        "brightfield_first" => Ok(FrameSchedule::BrightfieldFirst),
        "spiral_out" => Ok(FrameSchedule::SpiralOut),
        "random" => Ok(FrameSchedule::RandomShuffle { seed }),
        "snr_weighted" => Ok(FrameSchedule::SnrWeighted),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "schedule must be sequential, brightfield_first, spiral_out, random, or snr_weighted",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_algorithm<A>(
    py: Python<'_>,
    algorithm: A,
    problem: &PyReconstructionProblem,
    callbacks: Option<&Bound<'_, PyAny>>,
    resume_from: Option<&PyReconstructionCheckpoint>,
    schedule: &str,
    schedule_seed: u64,
) -> PyResult<PyReconstructionResult>
where
    A: ReconstructionAlgorithm + Send + 'static,
{
    let options = RunOptions {
        max_iterations: algorithm.iterations(),
        batch_size: algorithm.batch_size(),
        schedule: parse_schedule(schedule, schedule_seed)?,
        enable_frame_callbacks: false,
    };
    let problem = problem.inner.clone();
    let checkpoint = resume_from.map(|checkpoint| (*checkpoint.inner).clone());
    let (callbacks, callback_errors) = build_callbacks(py, callbacks)?;
    let result = py.detach(move || {
        let mut runner = Runner::new(algorithm, options).with_callbacks(callbacks);
        if let Some(checkpoint) = checkpoint {
            runner = runner.resume_from(checkpoint);
        }
        runner.run(&problem)
    });
    for error in callback_errors {
        if let Some(error) = lock_callback_error(&error).take() {
            return Err(error);
        }
    }
    PyReconstructionResult::from_core(py, result.map_err(to_py_err)?)
}

/// Alternating-projection reconstruction for Fourier ptychographic microscopy.
///
/// Each frame selects an overlapping patch of the object spectrum. The method
/// propagates that patch through the pupil, replaces its predicted detector
/// amplitude with the measured amplitude while retaining phase, and
/// back-projects the correction into the object spectrum.
///
/// Parameters
/// ----------
/// iterations : int
///     Number of complete passes through the acquisition schedule.
/// object_step : float
///     Relaxation factor applied to each object-spectrum correction.
/// batch_size : int
///     Number of measured frames supplied to each reconstruction step.
/// epsilon : float
///     Positive numerical floor used in divisions and dark-field handling.
/// loss_type : str
///     Loss reported in diagnostics. The projection always enforces amplitude.
///
/// Reference
/// ---------
/// G. Zheng, R. Horstmeyer, and C. Yang, "Wide-field, high-resolution Fourier
/// ptychographic microscopy," Nature Photonics 7, 739-745 (2013).
/// doi:10.1038/nphoton.2013.187.
#[pyclass(module = "fpm_rs._core", name = "AlternatingProjection", frozen)]
pub(crate) struct PyAlternatingProjection {
    inner: AlternatingProjection,
}

#[pymethods]
impl PyAlternatingProjection {
    #[new]
    #[pyo3(signature = (*, iterations=50, object_step=1.0, batch_size=1, epsilon=1e-10, loss_type="amplitude_mse"))]
    fn new(
        iterations: usize,
        object_step: f64,
        batch_size: usize,
        epsilon: f64,
        loss_type: &str,
    ) -> PyResult<Self> {
        let inner = AlternatingProjection {
            iterations,
            object_step,
            batch_size,
            epsilon,
            loss_type: parse_loss_type(loss_type)?,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (problem, *, callbacks=None, resume_from=None, schedule="sequential", schedule_seed=0))]
    fn run(
        &self,
        py: Python<'_>,
        problem: PyRef<'_, PyReconstructionProblem>,
        callbacks: Option<&Bound<'_, PyAny>>,
        resume_from: Option<PyRef<'_, PyReconstructionCheckpoint>>,
        schedule: &str,
        schedule_seed: u64,
    ) -> PyResult<PyReconstructionResult> {
        run_algorithm(
            py,
            self.inner.clone(),
            &problem,
            callbacks,
            resume_from.as_deref(),
            schedule,
            schedule_seed,
        )
    }
}

/// Regularized ptychographic iterative-engine reconstruction adapted to FPM.
///
/// This method uses detector-amplitude projection but preconditions object
/// corrections with a blend of local and maximum pupil power. That rPIE
/// denominator suppresses unstable updates where pupil transfer is weak.
///
/// Parameters
/// ----------
/// iterations : int
///     Number of complete passes through the acquisition schedule.
/// object_step : float
///     Relaxation factor applied to each object-spectrum correction.
/// stability : float
///     Blend from local pupil power (0) to maximum pupil power (1).
/// batch_size : int
///     Number of measured frames supplied to each reconstruction step.
/// epsilon : float
///     Positive floor added to the rPIE denominator.
/// loss_type : str
///     Loss reported in diagnostics. The projection always enforces amplitude.
///
/// Reference
/// ---------
/// A. Maiden, D. Johnson, and P. Li, "Further improvements to the
/// ptychographical iterative engine," Optica 4(7), 736-745 (2017).
/// doi:10.1364/OPTICA.4.000736.
#[pyclass(module = "fpm_rs._core", name = "Fpie", frozen)]
pub(crate) struct PyFpie {
    inner: Fpie,
}

#[pymethods]
impl PyFpie {
    #[new]
    #[pyo3(signature = (*, iterations=50, object_step=0.8, stability=0.1, batch_size=1, epsilon=1e-10, loss_type="amplitude_mse"))]
    fn new(
        iterations: usize,
        object_step: f64,
        stability: f64,
        batch_size: usize,
        epsilon: f64,
        loss_type: &str,
    ) -> PyResult<Self> {
        let inner = Fpie {
            iterations,
            object_step,
            stability,
            batch_size,
            epsilon,
            loss_type: parse_loss_type(loss_type)?,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (problem, *, callbacks=None, resume_from=None, schedule="sequential", schedule_seed=0))]
    fn run(
        &self,
        py: Python<'_>,
        problem: PyRef<'_, PyReconstructionProblem>,
        callbacks: Option<&Bound<'_, PyAny>>,
        resume_from: Option<PyRef<'_, PyReconstructionCheckpoint>>,
        schedule: &str,
        schedule_seed: u64,
    ) -> PyResult<PyReconstructionResult> {
        run_algorithm(
            py,
            self.inner.clone(),
            &problem,
            callbacks,
            resume_from.as_deref(),
            schedule,
            schedule_seed,
        )
    }
}

/// Embedded pupil-recovery reconstruction for Fourier ptychographic microscopy.
///
/// EPRY projects each predicted field onto the measured amplitude and uses the
/// exit-wave error to update both the object spectrum and complex pupil. This
/// separates specimen structure from pupil aberrations. Optional per-frame
/// gain and additive-background recovery are fpm-rs extensions.
///
/// Parameters
/// ----------
/// iterations : int
///     Number of complete passes through the acquisition schedule.
/// object_step, pupil_step : float
///     Relaxation factors for the object and pupil corrections.
/// batch_size : int
///     Number of measured frames supplied to each reconstruction step.
/// recover_pupil : bool
///     Update the complex pupil when true.
/// constrain_pupil_support : bool
///     Zero recovered pupil values outside the compiled aperture.
/// recover_frame_gains : bool
///     Estimate one multiplicative intensity gain per frame.
/// gain_step : float
///     Fraction of each frame-gain estimate applied per update.
/// gain_bounds : tuple[float, float]
///     Inclusive lower and upper bounds for recovered gains.
/// recover_background : bool
///     Estimate one spatially uniform additive background per frame.
/// background_step : float
///     Fraction of the mean frame residual applied per update.
/// background_bounds : tuple[float, float]
///     Inclusive bounds for recovered background intensities.
/// epsilon : float
///     Positive numerical floor used in normalized updates.
/// loss_type : str
///     Loss reported in diagnostics. The projection always enforces amplitude.
///
/// Reference
/// ---------
/// X. Ou, G. Zheng, and C. Yang, "Embedded pupil function recovery for Fourier
/// ptychographic microscopy," Optics Express 22(5), 4960-4972 (2014).
/// doi:10.1364/OE.22.004960.
#[pyclass(module = "fpm_rs._core", name = "Epry", frozen)]
pub(crate) struct PyEpry {
    inner: Epry,
}

#[pymethods]
impl PyEpry {
    #[new]
    #[pyo3(signature = (*, iterations=100, object_step=0.8, pupil_step=0.1, batch_size=1, recover_pupil=true, constrain_pupil_support=true, recover_frame_gains=false, gain_step=0.2, gain_bounds=(1e-6, 1e6), recover_background=false, background_step=0.2, background_bounds=(0.0, 1e12), epsilon=1e-10, loss_type="amplitude_mse"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        iterations: usize,
        object_step: f64,
        pupil_step: f64,
        batch_size: usize,
        recover_pupil: bool,
        constrain_pupil_support: bool,
        recover_frame_gains: bool,
        gain_step: f64,
        gain_bounds: (f64, f64),
        recover_background: bool,
        background_step: f64,
        background_bounds: (f64, f64),
        epsilon: f64,
        loss_type: &str,
    ) -> PyResult<Self> {
        let inner = Epry {
            iterations,
            object_step,
            pupil_step,
            batch_size,
            recover_pupil,
            constrain_pupil_support,
            recover_frame_gains,
            gain_step,
            minimum_gain: gain_bounds.0,
            maximum_gain: gain_bounds.1,
            recover_background,
            background_step,
            minimum_background: background_bounds.0,
            maximum_background: background_bounds.1,
            epsilon,
            loss_type: parse_loss_type(loss_type)?,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (problem, *, callbacks=None, resume_from=None, schedule="sequential", schedule_seed=0))]
    fn run(
        &self,
        py: Python<'_>,
        problem: PyRef<'_, PyReconstructionProblem>,
        callbacks: Option<&Bound<'_, PyAny>>,
        resume_from: Option<PyRef<'_, PyReconstructionCheckpoint>>,
        schedule: &str,
        schedule_seed: u64,
    ) -> PyResult<PyReconstructionResult> {
        run_algorithm(
            py,
            self.inner.clone(),
            &problem,
            callbacks,
            resume_from.as_deref(),
            schedule,
            schedule_seed,
        )
    }
}

/// Linearized ADMM reconstruction for Fourier ptychographic microscopy.
///
/// ADMM separates detector-amplitude fitting from overlapping-patch consensus
/// with per-mode auxiliary and scaled-dual fields. It alternates an amplitude
/// proximal update, a pupil-preconditioned object update, and a dual update.
///
/// Parameters
/// ----------
/// iterations : int
///     Number of complete passes through the acquisition schedule.
/// object_step : float
///     Step size of the linearized object update.
/// penalty : float
///     Positive augmented-Lagrangian consensus penalty.
/// dual_relaxation : float
///     Scaled-dual update relaxation in the range [0, 2].
/// batch_size : int or None
///     Frames per step. None processes all frames together.
/// epsilon : float
///     Positive numerical floor used in normalized updates.
///
/// Reference
/// ---------
/// A. Wang, Z. Zhang, S. Wang, A. Pan, C. Ma, and B. Yao, "Fourier
/// Ptychographic Microscopy via Alternating Direction Method of Multipliers,"
/// Cells 11(9), 1512 (2022). doi:10.3390/cells11091512.
#[pyclass(module = "fpm_rs._core", name = "Admm", frozen)]
pub(crate) struct PyAdmm {
    inner: Admm,
}

#[pymethods]
impl PyAdmm {
    #[new]
    #[pyo3(signature = (*, iterations=100, object_step=0.8, penalty=1.0, dual_relaxation=1.0, batch_size=None, epsilon=1e-10))]
    fn new(
        iterations: usize,
        object_step: f64,
        penalty: f64,
        dual_relaxation: f64,
        batch_size: Option<usize>,
        epsilon: f64,
    ) -> PyResult<Self> {
        let inner = Admm {
            iterations,
            object_step,
            penalty,
            dual_relaxation,
            batch_size: batch_size.unwrap_or(usize::MAX),
            epsilon,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (problem, *, callbacks=None, resume_from=None, schedule="sequential", schedule_seed=0))]
    fn run(
        &self,
        py: Python<'_>,
        problem: PyRef<'_, PyReconstructionProblem>,
        callbacks: Option<&Bound<'_, PyAny>>,
        resume_from: Option<PyRef<'_, PyReconstructionCheckpoint>>,
        schedule: &str,
        schedule_seed: u64,
    ) -> PyResult<PyReconstructionResult> {
        run_algorithm(
            py,
            self.inner.clone(),
            &problem,
            callbacks,
            resume_from.as_deref(),
            schedule,
            schedule_seed,
        )
    }
}

/// Wirtinger-style loss-gradient reconstruction for Fourier ptychography.
///
/// The solver differentiates the selected data loss through the complex FPM
/// forward model, averages mini-batch gradients, and applies a pupil-power-
/// preconditioned object update. It can also recover the pupil and illumination
/// offsets and regularize the complex object or pupil.
///
/// Parameters
/// ----------
/// iterations : int
///     Number of complete passes through the acquisition schedule.
/// object_step : float
///     Step size of the preconditioned object update.
/// batch_size : int
///     Number of frame gradients averaged into one update.
/// epsilon : float
///     Positive numerical floor used by losses and preconditioners.
/// loss_type : str
///     Data loss to optimize and report.
/// recover_illumination : bool
///     Estimate Fourier-grid offsets for illumination sources.
/// illumination_step : float
///     Step size of the diagonally scaled offset update.
/// illumination_finite_difference : float
///     Finite-difference spacing in Fourier-grid pixels.
/// illumination_bounds : float
///     Maximum absolute row or column correction in Fourier-grid pixels.
/// recover_pupil : bool
///     Update the complex pupil when true.
/// pupil_step : float
///     Step size of the normalized pupil update.
/// constrain_pupil_support : bool
///     Zero recovered pupil values outside the compiled aperture.
/// object_tv : float
///     Isotropic complex-object total-variation weight; 0 disables it.
/// object_tv_epsilon : float
///     Positive smoothing constant in the differentiable TV norm.
/// pupil_smoothing : float
///     Quadratic nearest-neighbor pupil penalty; requires pupil recovery.
/// parallel_workers : int
///     Maximum frame-gradient workers; 0 selects available CPU parallelism.
///
/// Reference
/// ---------
/// L. Bian, J. Suo, G. Zheng, K. Guo, F. Chen, and Q. Dai, "Fourier
/// ptychographic reconstruction using Wirtinger flow optimization," Optics
/// Express 23(4), 4856-4866 (2015). doi:10.1364/OE.23.004856.
#[pyclass(module = "fpm_rs._core", name = "GradientDescent", frozen)]
pub(crate) struct PyGradientDescent {
    inner: GradientDescent,
}

#[pymethods]
impl PyGradientDescent {
    #[new]
    #[pyo3(signature = (*, iterations=100, object_step=0.5, batch_size=1, epsilon=1e-10, loss_type="amplitude_mse", recover_illumination=false, illumination_step=0.1, illumination_finite_difference=0.05, illumination_bounds=1.0, recover_pupil=false, pupil_step=0.05, constrain_pupil_support=true, object_tv=0.0, object_tv_epsilon=1e-6, pupil_smoothing=0.0, parallel_workers=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        iterations: usize,
        object_step: f64,
        batch_size: usize,
        epsilon: f64,
        loss_type: &str,
        recover_illumination: bool,
        illumination_step: f64,
        illumination_finite_difference: f64,
        illumination_bounds: f64,
        recover_pupil: bool,
        pupil_step: f64,
        constrain_pupil_support: bool,
        object_tv: f64,
        object_tv_epsilon: f64,
        pupil_smoothing: f64,
        parallel_workers: usize,
    ) -> PyResult<Self> {
        let default_workers = GradientDescent::default().parallel_workers;
        let inner = GradientDescent {
            iterations,
            object_step,
            batch_size,
            epsilon,
            loss_type: parse_loss_type(loss_type)?,
            recover_illumination,
            illumination_step,
            illumination_finite_difference,
            maximum_illumination_correction: illumination_bounds,
            recover_pupil,
            pupil_step,
            constrain_pupil_support,
            object_tv_weight: object_tv,
            object_tv_epsilon,
            pupil_smoothing_weight: pupil_smoothing,
            parallel_workers: if parallel_workers == 0 {
                default_workers
            } else {
                parallel_workers
            },
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (problem, *, callbacks=None, resume_from=None, schedule="sequential", schedule_seed=0))]
    fn run(
        &self,
        py: Python<'_>,
        problem: PyRef<'_, PyReconstructionProblem>,
        callbacks: Option<&Bound<'_, PyAny>>,
        resume_from: Option<PyRef<'_, PyReconstructionCheckpoint>>,
        schedule: &str,
        schedule_seed: u64,
    ) -> PyResult<PyReconstructionResult> {
        run_algorithm(
            py,
            self.inner.clone(),
            &problem,
            callbacks,
            resume_from.as_deref(),
            schedule,
            schedule_seed,
        )
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyReconstructionProblem>()?;
    module.add_class::<PyReconstructionCheckpoint>()?;
    module.add_class::<PyRuntimeInfo>()?;
    module.add_class::<PyReconstructionResult>()?;
    module.add_class::<PyDiagnosticRecorder>()?;
    module.add_class::<PyProgressLogger>()?;
    module.add_class::<PyCheckpointEvery>()?;
    module.add_class::<PyCsvLogger>()?;
    module.add_class::<PyStopOnPlateau>()?;
    module.add_class::<PySaveImageEvery>()?;
    module.add_class::<PySavePupilEvery>()?;
    module.add_class::<PySaveResidualsEvery>()?;
    module.add_class::<PyIterationCallback>()?;
    module.add_class::<PyAlternatingProjection>()?;
    module.add_class::<PyFpie>()?;
    module.add_class::<PyEpry>()?;
    module.add_class::<PyAdmm>()?;
    module.add_class::<PyGradientDescent>()?;
    Ok(())
}
