use std::{
    collections::BTreeMap,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use fpm_rs::{
    Complex64, Error,
    algorithms::objective::LossType,
    algorithms::{
        AdaptiveAlternatingProjection, Admm, AlternatingProjection, Epry, Fpie, GradientDescent,
        JointReconstruction, JointReconstructionResult as CoreJointReconstructionResult, Mpie,
        ReconstructionAlgorithm,
    },
    callbacks::{
        Callback, CallbackAction, CallbackHook, CheckpointEvery, CsvLogger, ProgressLogger,
        SaveImageEvery, SavePupilEvery, SaveResidualsEvery, StepContext, StopOnPlateau,
    },
    diagnostics::{
        DiagnosticRecorder as CoreDiagnosticRecorder, DiagnosticRecorderConfig, DiagnosticRequest,
        ReconstructionDiagnostics,
    },
    illumination_calibration::{
        BoundedFiniteDifferenceOptimizer, CalibrationConditioning, CalibrationConvergenceReason,
        CalibrationLossHistoryEntry, CalibrationParameterHistoryEntry, CalibrationParameterSpec,
        IlluminationCalibration, IlluminationCalibrationState, PlanarArrayCalibrationParameters,
        PlanarArrayParameterValues,
    },
    measurements::{FrameMetadata, MeasurementRead, MeasurementStack},
    model::Pupil,
    reconstruction::{
        AlgorithmMetricRecord, FrameSchedule, IterationRecord, ReconstructionCheckpoint,
        ReconstructionProblem, ReconstructionResult, ReconstructionTrace, RunOptions, Runner,
    },
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, ndarray};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use crate::{
    arrays::{array2_to_py, complex_array2_to_py, copy_array2, vec2_to_py},
    config::{PyIllumination, PyOptics},
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
        self.0.frame_metadata()
    }

    fn validate(&self) -> fpm_rs::Result<()> {
        self.0.validate()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "ReconstructionProblem",
    frozen,
    from_py_object
)]
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
        self.inner.model.image_shape()
    }

    #[getter]
    fn reconstruction_shape(&self) -> (usize, usize) {
        self.inner.model.reconstruction_shape()
    }
}

/// Serializable state for resuming a compatible reconstruction.
///
/// Problem-aware restoration requires the checkpoint pupil support to match
/// the compiled model support exactly. Pupil-recovering algorithms project a
/// restored object/pupil pair into their canonical gauge before start callbacks
/// and continued iterations.
#[pyclass(
    module = "fpm_rs._core",
    name = "ReconstructionCheckpoint",
    frozen,
    from_py_object
)]
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
        self.inner.format_version()
    }

    #[getter]
    fn completed_iterations(&self) -> usize {
        self.inner.completed_iterations()
    }
}

#[pyclass(module = "fpm_rs._core", name = "RuntimeInfo", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyRuntimeInfo {
    #[pyo3(get)]
    pub(crate) elapsed_seconds: f64,
    #[pyo3(get)]
    pub(crate) completed_iterations: usize,
    #[pyo3(get)]
    pub(crate) stopped_early: bool,
    #[pyo3(get)]
    pub(crate) algorithm: String,
}

/// Owned reconstruction products, histories, diagnostics, and metadata.
///
/// For built-in pupil-recovering algorithms, the object and pupil arrays use
/// the compiled pupil's canonical scale and phase gauge.
#[pyclass(module = "fpm_rs._core", name = "ReconstructionResult", frozen)]
pub(crate) struct PyReconstructionResult {
    pub(crate) object: Py<PyArray2<Complex64>>,
    pub(crate) amplitude: Py<PyArray2<f64>>,
    pub(crate) phase: Py<PyArray2<f64>>,
    pub(crate) object_spectrum: Py<PyArray2<Complex64>>,
    pub(crate) recovered_pupil: Py<PyArray2<Complex64>>,
    pub(crate) pupil_support: Py<PyArray2<u8>>,
    pub(crate) calibrated_illumination: Option<Py<PyArray2<f64>>>,
    pub(crate) recovered_frame_gains: Option<Py<PyArray1<f64>>>,
    pub(crate) recovered_background: Option<Py<PyArray1<f64>>>,
    pub(crate) physical_illumination_calibration: Option<IlluminationCalibrationState>,
    pub(crate) calibrated_model: Option<fpm_rs::model::ImagePlaneModel>,
    pub(crate) trace: Vec<(usize, f64, f64)>,
    pub(crate) algorithm_metrics: Vec<(usize, String, String, f64)>,
    pub(crate) scalar_diagnostics: BTreeMap<String, f64>,
    pub(crate) runtime: PyRuntimeInfo,
    pub(crate) metadata: BTreeMap<String, String>,
}

impl PyReconstructionResult {
    pub(crate) fn from_core(py: Python<'_>, result: ReconstructionResult) -> PyResult<Self> {
        let pupil_shape = result.recovered_pupil.shape();
        let physical_illumination_calibration = result.physical_illumination_calibration.clone();
        let calibrated_model = result.calibrated_model.clone();
        let pupil_support = result.recovered_pupil.support().iter().copied().collect();
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
        let algorithm_metrics = result
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
            .collect();
        let trace = result
            .trace
            .iterations
            .into_iter()
            .map(|record| (record.iteration, record.objective, record.elapsed_seconds))
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
            recovered_pupil: complex_array2_to_py(py, result.recovered_pupil.values().to_owned())?,
            pupil_support: vec2_to_py(py, pupil_shape, pupil_support)?,
            calibrated_illumination,
            recovered_frame_gains,
            recovered_background,
            physical_illumination_calibration,
            calibrated_model,
            trace,
            algorithm_metrics,
            scalar_diagnostics: result.scalar_diagnostics,
            runtime,
            metadata: result.metadata,
        })
    }

    pub(crate) fn to_core(&self, py: Python<'_>) -> PyResult<ReconstructionResult> {
        let object =
            crate::arrays::core_array2(&self.object.bind(py).readonly()).map_err(to_py_err)?;
        let amplitude =
            crate::arrays::core_array2(&self.amplitude.bind(py).readonly()).map_err(to_py_err)?;
        let phase =
            crate::arrays::core_array2(&self.phase.bind(py).readonly()).map_err(to_py_err)?;
        let object_spectrum = crate::arrays::core_array2(&self.object_spectrum.bind(py).readonly())
            .map_err(to_py_err)?;
        let pupil_values = crate::arrays::core_array2(&self.recovered_pupil.bind(py).readonly())
            .map_err(to_py_err)?;
        let pupil_support = crate::arrays::core_array2(&self.pupil_support.bind(py).readonly())
            .map_err(to_py_err)?;
        let recovered_pupil = Pupil::new(pupil_values, pupil_support).map_err(to_py_err)?;
        let calibrated_illumination = self
            .calibrated_illumination
            .as_ref()
            .map(|values| {
                let values = copy_array2(&values.bind(py).readonly());
                if values.len() % 2 != 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "calibrated illumination must have two columns",
                    ));
                }
                Ok(values
                    .chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect())
            })
            .transpose()?;
        let recovered_frame_gains = self.recovered_frame_gains.as_ref().map(|values| {
            values
                .bind(py)
                .readonly()
                .as_slice()
                .map(<[f64]>::to_vec)
                .unwrap_or_else(|_| {
                    values
                        .bind(py)
                        .readonly()
                        .as_array()
                        .iter()
                        .copied()
                        .collect()
                })
        });
        let recovered_background = self.recovered_background.as_ref().map(|values| {
            values
                .bind(py)
                .readonly()
                .as_slice()
                .map(<[f64]>::to_vec)
                .unwrap_or_else(|_| {
                    values
                        .bind(py)
                        .readonly()
                        .as_array()
                        .iter()
                        .copied()
                        .collect()
                })
        });
        Ok(ReconstructionResult {
            object,
            amplitude,
            phase,
            object_spectrum,
            recovered_pupil,
            calibrated_illumination,
            recovered_frame_gains,
            recovered_background,
            physical_illumination_calibration: self.physical_illumination_calibration.clone(),
            calibrated_model: self.calibrated_model.clone(),
            trace: ReconstructionTrace {
                iterations: self
                    .trace
                    .iter()
                    .map(|&(iteration, objective, elapsed_seconds)| IterationRecord {
                        iteration,
                        objective,
                        elapsed_seconds,
                    })
                    .collect(),
                algorithm_metrics: self
                    .algorithm_metrics
                    .iter()
                    .map(
                        |(iteration, namespace, metric, value)| AlgorithmMetricRecord {
                            iteration: *iteration,
                            namespace: namespace.clone(),
                            metric: metric.clone(),
                            value: *value,
                        },
                    )
                    .collect(),
            },
            scalar_diagnostics: self.scalar_diagnostics.clone(),
            runtime: fpm_rs::reconstruction::RuntimeInfo {
                elapsed_seconds: self.runtime.elapsed_seconds,
                completed_iterations: self.runtime.completed_iterations,
                stopped_early: self.runtime.stopped_early,
                algorithm: self.runtime.algorithm.clone(),
            },
            metadata: self.metadata.clone(),
        })
    }
}

#[pyfunction]
#[pyo3(signature = (result, truth, *, problem=None, reference_model=None, valid_object_mask=None))]
fn evaluate_reconstruction_py(
    py: Python<'_>,
    result: PyRef<'_, PyReconstructionResult>,
    truth: numpy::PyReadonlyArray2<'_, Complex64>,
    problem: Option<PyRef<'_, PyReconstructionProblem>>,
    reference_model: Option<PyRef<'_, PyImagePlaneModel>>,
    valid_object_mask: Option<numpy::PyReadonlyArray2<'_, u8>>,
) -> PyResult<Py<PyDict>> {
    let result = result.to_core(py)?;
    // Evaluation is layout-independent. Copying here releases the Python
    // borrow before detaching while preserving the input's logical order.
    let truth = truth.as_array().to_owned();
    let mask = valid_object_mask.map(|value| value.as_array().to_owned());
    let reference_model = reference_model.map(|value| (*value.inner).clone());
    let evaluation = if let Some(problem) = problem {
        let problem = problem.inner.clone();
        py.detach(move || {
            fpm_rs::evaluation::evaluate_reconstruction_with_problem(
                &result,
                &problem,
                truth.view(),
                reference_model.as_ref(),
                mask.as_ref().map(|mask| mask.view()),
            )
        })
    } else {
        py.detach(move || {
            fpm_rs::evaluation::evaluate_reconstruction(
                &result,
                truth.view(),
                reference_model.as_ref(),
                mask.as_ref().map(|mask| mask.view()),
            )
        })
    }
    .map_err(to_py_err)?;
    evaluation_to_py(py, &evaluation)
}

pub(crate) fn evaluation_to_py(
    py: Python<'_>,
    evaluation: &fpm_rs::evaluation::ReconstructionEvaluation,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    let object = PyDict::new(py);
    object.set_item("amplitude_rmse", evaluation.object.amplitude_rmse)?;
    object.set_item("amplitude_nrmse", evaluation.object.amplitude_nrmse)?;
    object.set_item("complex_rmse", evaluation.object.complex_rmse)?;
    object.set_item("complex_nrmse", evaluation.object.complex_nrmse)?;
    object.set_item("phase_rmse", evaluation.object.phase_rmse)?;
    object.set_item("phase_mae", evaluation.object.phase_mae)?;
    object.set_item("fourier_nrmse", evaluation.object.fourier_nrmse)?;
    object.set_item("global_phase_offset", evaluation.object.global_phase_offset)?;
    output.set_item("object", object)?;
    if let Some(value) = &evaluation.pupil {
        let section = PyDict::new(py);
        section.set_item("amplitude_rmse", value.amplitude_rmse)?;
        section.set_item("phase_rmse", value.phase_rmse)?;
        output.set_item("pupil", section)?;
    } else {
        output.set_item("pupil", py.None())?;
    }
    if let Some(value) = &evaluation.illumination {
        let section = PyDict::new(py);
        section.set_item("position_rmse", value.position_rmse)?;
        output.set_item("illumination", section)?;
    } else {
        output.set_item("illumination", py.None())?;
    }
    if let Some(value) = &evaluation.frame_gains {
        let section = PyDict::new(py);
        section.set_item("relative_error", value.relative_error)?;
        output.set_item("frame_gains", section)?;
    } else {
        output.set_item("frame_gains", py.None())?;
    }
    if let Some(value) = &evaluation.intensity {
        let section = PyDict::new(py);
        let frames = PyList::empty(py);
        for frame in &value.per_frame {
            let item = PyDict::new(py);
            item.set_item("reference_sum", frame.reference_sum)?;
            item.set_item("estimate_sum", frame.estimate_sum)?;
            item.set_item("residual_l1", frame.residual_l1)?;
            item.set_item("residual_l2", frame.residual_l2)?;
            item.set_item("residual_mean", frame.residual_mean)?;
            item.set_item("residual_std", frame.residual_std)?;
            item.set_item("residual_max_abs", frame.residual_max_abs)?;
            item.set_item("normalized_l2", frame.normalized_l2)?;
            item.set_item("saturated_pixels", frame.saturated_pixels)?;
            frames.append(item)?;
        }
        section.set_item("per_frame", frames)?;
        output.set_item("intensity", section)?;
    } else {
        output.set_item("intensity", py.None())?;
    }
    Ok(output.unbind())
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
    /// Return the canonical sampled complex pupil on the low-resolution grid.
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
    fn physical_illumination_calibration(&self) -> Option<PyIlluminationCalibrationState> {
        self.physical_illumination_calibration
            .clone()
            .map(|inner| PyIlluminationCalibrationState { inner })
    }

    #[getter]
    fn calibrated_model(&self) -> Option<PyImagePlaneModel> {
        self.calibrated_model
            .clone()
            .map(|inner| PyImagePlaneModel {
                inner: Arc::new(inner),
            })
    }

    #[getter]
    fn trace(&self) -> Vec<(usize, f64, f64)> {
        self.trace.clone()
    }

    #[getter]
    fn algorithm_metrics(&self) -> Vec<(usize, String, String, f64)> {
        self.algorithm_metrics.clone()
    }

    #[getter]
    fn scalar_diagnostics(&self) -> BTreeMap<String, f64> {
        self.scalar_diagnostics.clone()
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
    fn final_objective(&self) -> Option<f64> {
        self.trace.last().map(|record| record.1)
    }

    #[pyo3(signature = (path, *, run_id=None, label=None, include_previews=true))]
    fn write_bundle(
        &self,
        py: Python<'_>,
        path: PathBuf,
        run_id: Option<String>,
        label: Option<String>,
        include_previews: bool,
    ) -> PyResult<crate::bundle::PyResultBundle> {
        let result = self.to_core(py)?;
        let bundle = py
            .detach(move || {
                result.write_bundle(
                    path,
                    fpm_rs::reconstruction::BundleExportOptions {
                        run_id,
                        label,
                        include_previews,
                    },
                )
            })
            .map_err(to_py_err)?;
        Ok(crate::bundle::PyResultBundle::from_core(bundle))
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

pub(crate) fn diagnostics_to_py(
    py: Python<'_>,
    diagnostics: &ReconstructionDiagnostics,
) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);

    let iteration_diagnostics = PyList::empty(py);
    for entry in &diagnostics.iteration_diagnostics {
        let value = PyDict::new(py);
        value.set_item("iteration", entry.iteration)?;
        value.set_item("total_objective", entry.total_objective)?;
        value.set_item("data_objective", entry.data_objective)?;
        value.set_item("regularization_objective", entry.regularization_objective)?;
        value.set_item("object_relative_change", entry.object_relative_change)?;
        value.set_item("pupil_relative_change", entry.pupil_relative_change)?;
        value.set_item("median_frame_objective", entry.median_frame_objective)?;
        value.set_item("worst_frame_objective", entry.worst_frame_objective)?;
        value.set_item("elapsed_seconds", entry.elapsed_seconds)?;
        iteration_diagnostics.append(value)?;
    }
    output.set_item("iteration_diagnostics", iteration_diagnostics)?;

    let frame_diagnostics = PyList::empty(py);
    for entry in &diagnostics.frame_diagnostics {
        let value = PyDict::new(py);
        value.set_item("iteration", entry.iteration)?;
        value.set_item("frame_index", entry.frame_index)?;
        value.set_item("illumination_index", entry.illumination_index)?;
        value.set_item("reference_sum", entry.metrics.reference_sum)?;
        value.set_item("estimate_sum", entry.metrics.estimate_sum)?;
        value.set_item("residual_l1", entry.metrics.residual_l1)?;
        value.set_item("residual_l2", entry.metrics.residual_l2)?;
        value.set_item("residual_mean", entry.metrics.residual_mean)?;
        value.set_item("residual_std", entry.metrics.residual_std)?;
        value.set_item("residual_max_abs", entry.metrics.residual_max_abs)?;
        value.set_item("normalized_l2", entry.metrics.normalized_l2)?;
        value.set_item("saturated_pixels", entry.metrics.saturated_pixels)?;
        frame_diagnostics.append(value)?;
    }
    output.set_item("frame_diagnostics", frame_diagnostics)?;

    let raw_frame_statistics = PyList::empty(py);
    for entry in &diagnostics.raw_frame_statistics {
        let value = PyDict::new(py);
        value.set_item("frame_index", entry.frame_index)?;
        value.set_item("mean", entry.metrics.mean)?;
        value.set_item("std", entry.metrics.std)?;
        value.set_item("min", entry.metrics.min)?;
        value.set_item("max", entry.metrics.max)?;
        value.set_item("sum", entry.metrics.sum)?;
        value.set_item("saturated_pixels", entry.metrics.saturated_pixels)?;
        value.set_item("zero_pixels", entry.metrics.zero_pixels)?;
        raw_frame_statistics.append(value)?;
    }
    output.set_item("raw_frame_statistics", raw_frame_statistics)?;

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
        value.set_item("global_phase_offset", metrics.global_phase_offset)?;
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
        vec![DiagnosticRequest::Objective]
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
            values.set_item("objective", context.diagnostics.objective)?;
            values.set_item(
                "algorithm_metrics",
                context
                    .current_algorithm_metrics
                    .iter()
                    .map(|record| {
                        (
                            record.namespace.as_str(),
                            record.metric.as_str(),
                            record.value,
                        )
                    })
                    .collect::<Vec<_>>(),
            )?;
            let physical_calibration = context
                .state
                .physical_illumination_calibration()
                .map(|state| {
                    Py::new(
                        py,
                        PyIlluminationCalibrationState {
                            inner: state.clone(),
                        },
                    )
                })
                .transpose()?;
            values.set_item("physical_illumination_calibration", physical_calibration)?;
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

#[pyclass(
    module = "fpm_rs._core",
    name = "CalibrationParameterSpec",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyCalibrationParameterSpec {
    inner: CalibrationParameterSpec,
}

#[pymethods]
impl PyCalibrationParameterSpec {
    #[new]
    #[pyo3(signature = (lower_bound, upper_bound, *, scale=1.0, finite_difference_step=None, prior_center=None, regularization_strength=0.0))]
    fn new(
        lower_bound: f64,
        upper_bound: f64,
        scale: f64,
        finite_difference_step: Option<f64>,
        prior_center: Option<f64>,
        regularization_strength: f64,
    ) -> PyResult<Self> {
        let mut inner = CalibrationParameterSpec::new(lower_bound, upper_bound, scale);
        if let Some(step) = finite_difference_step {
            inner.finite_difference_step = step;
        }
        inner.prior_center = prior_center;
        inner.regularization_strength = regularization_strength;
        inner.validate("calibration parameter").map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn lower_bound(&self) -> f64 {
        self.inner.lower_bound
    }

    #[getter]
    fn upper_bound(&self) -> f64 {
        self.inner.upper_bound
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    #[getter]
    fn finite_difference_step(&self) -> f64 {
        self.inner.finite_difference_step
    }

    #[getter]
    fn prior_center(&self) -> Option<f64> {
        self.inner.prior_center
    }

    #[getter]
    fn regularization_strength(&self) -> f64 {
        self.inner.regularization_strength
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "PlanarArrayCalibrationParameters",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPlanarArrayCalibrationParameters {
    inner: PlanarArrayCalibrationParameters,
}

#[pymethods]
impl PyPlanarArrayCalibrationParameters {
    #[new]
    #[pyo3(signature = (*, translation=(false, false, false), rotation=(false, false, false), pitch=(false, false), reference_index=(false, false), position_offsets=Vec::new(), relative_source_power=false, frame_gains=false, translation_spec=None, rotation_spec=None, pitch_spec=None, reference_index_spec=None, position_offset_spec=None, relative_source_power_spec=None, frame_gain_spec=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        translation: (bool, bool, bool),
        rotation: (bool, bool, bool),
        pitch: (bool, bool),
        reference_index: (bool, bool),
        position_offsets: Vec<usize>,
        relative_source_power: bool,
        frame_gains: bool,
        translation_spec: Option<PyRef<'_, PyCalibrationParameterSpec>>,
        rotation_spec: Option<PyRef<'_, PyCalibrationParameterSpec>>,
        pitch_spec: Option<PyRef<'_, PyCalibrationParameterSpec>>,
        reference_index_spec: Option<PyRef<'_, PyCalibrationParameterSpec>>,
        position_offset_spec: Option<PyRef<'_, PyCalibrationParameterSpec>>,
        relative_source_power_spec: Option<PyRef<'_, PyCalibrationParameterSpec>>,
        frame_gain_spec: Option<PyRef<'_, PyCalibrationParameterSpec>>,
    ) -> PyResult<Self> {
        if translation.0 && reference_index.0 || translation.1 && reference_index.1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lateral translation and the corresponding reference index are gauge-ambiguous",
            ));
        }
        if relative_source_power && frame_gains {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "relative source power and frame gains cannot be optimized together",
            ));
        }
        for (provided, selected, name) in [
            (
                translation_spec.is_some(),
                translation.0 || translation.1 || translation.2,
                "translation_spec",
            ),
            (
                rotation_spec.is_some(),
                rotation.0 || rotation.1 || rotation.2,
                "rotation_spec",
            ),
            (pitch_spec.is_some(), pitch.0 || pitch.1, "pitch_spec"),
            (
                reference_index_spec.is_some(),
                reference_index.0 || reference_index.1,
                "reference_index_spec",
            ),
            (
                position_offset_spec.is_some(),
                !position_offsets.is_empty(),
                "position_offset_spec",
            ),
        ] {
            if provided && !selected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} requires at least one corresponding active parameter"
                )));
            }
        }
        let mut inner = PlanarArrayCalibrationParameters::builder()
            .translation([translation.0, translation.1, translation.2])
            .rotation([rotation.0, rotation.1, rotation.2])
            .pitch([pitch.0, pitch.1])
            .reference_index([reference_index.0, reference_index.1])
            .position_offsets(position_offsets)
            .relative_source_power(relative_source_power)
            .frame_gains(frame_gains)
            .build()
            .map_err(to_py_err)?;
        replace_active_specs(&mut inner.translation, translation_spec.as_deref());
        replace_active_specs(&mut inner.rotation, rotation_spec.as_deref());
        replace_active_specs(&mut inner.pitch, pitch_spec.as_deref());
        replace_active_specs(&mut inner.reference_index, reference_index_spec.as_deref());
        if let Some(spec) = position_offset_spec {
            for specs in inner.position_offsets.values_mut() {
                specs.fill(spec.inner.clone());
            }
        }
        if relative_source_power_spec.is_some() && !relative_source_power {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "relative_source_power_spec requires relative_source_power=True",
            ));
        }
        if frame_gain_spec.is_some() && !frame_gains {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frame_gain_spec requires frame_gains=True",
            ));
        }
        if let Some(spec) = relative_source_power_spec {
            inner.relative_source_power = Some(spec.inner.clone());
        }
        if let Some(spec) = frame_gain_spec {
            inner.frame_gains = Some(spec.inner.clone());
        }
        Ok(Self { inner })
    }

    #[getter]
    fn translation(&self) -> (bool, bool, bool) {
        (
            self.inner.translation[0].is_some(),
            self.inner.translation[1].is_some(),
            self.inner.translation[2].is_some(),
        )
    }

    #[getter]
    fn rotation(&self) -> (bool, bool, bool) {
        (
            self.inner.rotation[0].is_some(),
            self.inner.rotation[1].is_some(),
            self.inner.rotation[2].is_some(),
        )
    }

    #[getter]
    fn pitch(&self) -> (bool, bool) {
        (self.inner.pitch[0].is_some(), self.inner.pitch[1].is_some())
    }

    #[getter]
    fn reference_index(&self) -> (bool, bool) {
        (
            self.inner.reference_index[0].is_some(),
            self.inner.reference_index[1].is_some(),
        )
    }

    #[getter]
    fn position_offsets(&self) -> Vec<usize> {
        self.inner.position_offsets.keys().copied().collect()
    }

    #[getter]
    fn relative_source_power(&self) -> bool {
        self.inner.relative_source_power.is_some()
    }

    #[getter]
    fn frame_gains(&self) -> bool {
        self.inner.frame_gains.is_some()
    }

    #[getter]
    fn translation_specs(
        &self,
    ) -> (
        Option<PyCalibrationParameterSpec>,
        Option<PyCalibrationParameterSpec>,
        Option<PyCalibrationParameterSpec>,
    ) {
        specification_triple(&self.inner.translation)
    }

    #[getter]
    fn rotation_specs(
        &self,
    ) -> (
        Option<PyCalibrationParameterSpec>,
        Option<PyCalibrationParameterSpec>,
        Option<PyCalibrationParameterSpec>,
    ) {
        specification_triple(&self.inner.rotation)
    }

    #[getter]
    fn pitch_specs(
        &self,
    ) -> (
        Option<PyCalibrationParameterSpec>,
        Option<PyCalibrationParameterSpec>,
    ) {
        specification_pair(&self.inner.pitch)
    }

    #[getter]
    fn reference_index_specs(
        &self,
    ) -> (
        Option<PyCalibrationParameterSpec>,
        Option<PyCalibrationParameterSpec>,
    ) {
        specification_pair(&self.inner.reference_index)
    }

    #[getter]
    fn position_offset_specs(
        &self,
    ) -> BTreeMap<
        usize,
        (
            PyCalibrationParameterSpec,
            PyCalibrationParameterSpec,
            PyCalibrationParameterSpec,
        ),
    > {
        self.inner
            .position_offsets
            .iter()
            .map(|(&source, specs)| {
                (
                    source,
                    (
                        PyCalibrationParameterSpec {
                            inner: specs[0].clone(),
                        },
                        PyCalibrationParameterSpec {
                            inner: specs[1].clone(),
                        },
                        PyCalibrationParameterSpec {
                            inner: specs[2].clone(),
                        },
                    ),
                )
            })
            .collect()
    }

    #[getter]
    fn relative_source_power_spec(&self) -> Option<PyCalibrationParameterSpec> {
        self.inner
            .relative_source_power
            .clone()
            .map(|inner| PyCalibrationParameterSpec { inner })
    }

    #[getter]
    fn frame_gain_spec(&self) -> Option<PyCalibrationParameterSpec> {
        self.inner
            .frame_gains
            .clone()
            .map(|inner| PyCalibrationParameterSpec { inner })
    }
}

fn specification_triple(
    values: &[Option<CalibrationParameterSpec>; 3],
) -> (
    Option<PyCalibrationParameterSpec>,
    Option<PyCalibrationParameterSpec>,
    Option<PyCalibrationParameterSpec>,
) {
    (
        values[0]
            .clone()
            .map(|inner| PyCalibrationParameterSpec { inner }),
        values[1]
            .clone()
            .map(|inner| PyCalibrationParameterSpec { inner }),
        values[2]
            .clone()
            .map(|inner| PyCalibrationParameterSpec { inner }),
    )
}

fn specification_pair(
    values: &[Option<CalibrationParameterSpec>; 2],
) -> (
    Option<PyCalibrationParameterSpec>,
    Option<PyCalibrationParameterSpec>,
) {
    (
        values[0]
            .clone()
            .map(|inner| PyCalibrationParameterSpec { inner }),
        values[1]
            .clone()
            .map(|inner| PyCalibrationParameterSpec { inner }),
    )
}

fn replace_active_specs<const N: usize>(
    values: &mut [Option<CalibrationParameterSpec>; N],
    replacement: Option<&PyCalibrationParameterSpec>,
) {
    if let Some(replacement) = replacement {
        for value in values.iter_mut().flatten() {
            *value = replacement.inner.clone();
        }
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "BoundedFiniteDifferenceOptimizer",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBoundedFiniteDifferenceOptimizer {
    inner: BoundedFiniteDifferenceOptimizer,
}

#[pymethods]
impl PyBoundedFiniteDifferenceOptimizer {
    #[new]
    #[pyo3(signature = (*, max_steps=2, relative_tolerance=1e-6, initial_step_size=0.25, minimum_step_size=1e-6, step_reduction=0.5))]
    fn new(
        max_steps: usize,
        relative_tolerance: f64,
        initial_step_size: f64,
        minimum_step_size: f64,
        step_reduction: f64,
    ) -> PyResult<Self> {
        let inner = BoundedFiniteDifferenceOptimizer {
            max_steps,
            relative_tolerance,
            initial_step_size,
            minimum_step_size,
            step_reduction,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn max_steps(&self) -> usize {
        self.inner.max_steps
    }

    #[getter]
    fn relative_tolerance(&self) -> f64 {
        self.inner.relative_tolerance
    }

    #[getter]
    fn initial_step_size(&self) -> f64 {
        self.inner.initial_step_size
    }

    #[getter]
    fn minimum_step_size(&self) -> f64 {
        self.inner.minimum_step_size
    }

    #[getter]
    fn step_reduction(&self) -> f64 {
        self.inner.step_reduction
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "IlluminationCalibration",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyIlluminationCalibration {
    inner: IlluminationCalibration,
}

#[pymethods]
impl PyIlluminationCalibration {
    #[new]
    #[pyo3(signature = (parameters, *, optimizer=None, loss_type="amplitude_mse"))]
    fn new(
        parameters: PyRef<'_, PyPlanarArrayCalibrationParameters>,
        optimizer: Option<PyRef<'_, PyBoundedFiniteDifferenceOptimizer>>,
        loss_type: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: IlluminationCalibration::new(parameters.inner.clone())
                .optimizer(
                    optimizer
                        .map(|value| value.inner.clone())
                        .unwrap_or_default(),
                )
                .loss_type(parse_loss_type(loss_type)?),
        })
    }

    #[getter]
    fn parameters(&self) -> PyPlanarArrayCalibrationParameters {
        PyPlanarArrayCalibrationParameters {
            inner: self.inner.parameters.clone(),
        }
    }

    #[getter]
    fn optimizer(&self) -> PyBoundedFiniteDifferenceOptimizer {
        PyBoundedFiniteDifferenceOptimizer {
            inner: self.inner.optimizer.clone(),
        }
    }

    #[getter]
    fn loss_type(&self) -> &'static str {
        loss_type_name(self.inner.loss_type)
    }
}

fn loss_type_name(loss_type: LossType) -> &'static str {
    match loss_type {
        LossType::AmplitudeMse => "amplitude_mse",
        LossType::IntensityMse => "intensity_mse",
        LossType::PoissonNegativeLogLikelihood => "poisson_nll",
        LossType::HuberAmplitude => "huber_amplitude",
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "PlanarArrayParameterValues",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPlanarArrayParameterValues {
    inner: PlanarArrayParameterValues,
}

#[pymethods]
impl PyPlanarArrayParameterValues {
    #[getter]
    fn translation_m(&self) -> (f64, f64, f64) {
        self.inner.translation_m.into()
    }

    #[getter]
    fn rotation_rad(&self) -> (f64, f64, f64) {
        self.inner.rotation_rad.into()
    }

    #[getter]
    fn pitch_m(&self) -> (f64, f64) {
        self.inner.pitch_m.into()
    }

    #[getter]
    fn reference_index(&self) -> (f64, f64) {
        self.inner.reference_index.into()
    }

    #[getter]
    fn position_offsets_m(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .position_offsets_m
            .iter()
            .map(|value| (*value).into())
            .collect()
    }

    #[getter]
    fn relative_source_power(&self) -> Vec<f64> {
        self.inner.relative_source_power.clone()
    }

    #[getter]
    fn frame_gains(&self) -> Vec<f64> {
        self.inner.frame_gains.clone()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "CalibrationParameterHistoryEntry",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyCalibrationParameterHistoryEntry {
    inner: CalibrationParameterHistoryEntry,
}

#[pymethods]
impl PyCalibrationParameterHistoryEntry {
    #[getter]
    fn outer_iteration(&self) -> usize {
        self.inner.outer_iteration
    }

    #[getter]
    fn optimizer_step(&self) -> usize {
        self.inner.optimizer_step
    }

    #[getter]
    fn accepted(&self) -> bool {
        self.inner.accepted
    }

    #[getter]
    fn step_size(&self) -> f64 {
        self.inner.step_size
    }

    #[getter]
    fn normalized_values(&self) -> Vec<f64> {
        self.inner.normalized_values.clone()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "CalibrationLossHistoryEntry",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyCalibrationLossHistoryEntry {
    inner: CalibrationLossHistoryEntry,
}

#[pymethods]
impl PyCalibrationLossHistoryEntry {
    #[getter]
    fn outer_iteration(&self) -> usize {
        self.inner.outer_iteration
    }

    #[getter]
    fn optimizer_step(&self) -> usize {
        self.inner.optimizer_step
    }

    #[getter]
    fn total_loss(&self) -> f64 {
        self.inner.total_loss
    }

    #[getter]
    fn data_loss(&self) -> f64 {
        self.inner.data_loss
    }

    #[getter]
    fn regularization_loss(&self) -> f64 {
        self.inner.regularization_loss
    }

    #[getter]
    fn accepted(&self) -> bool {
        self.inner.accepted
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "CalibrationConditioning",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyCalibrationConditioning {
    inner: CalibrationConditioning,
}

#[pymethods]
impl PyCalibrationConditioning {
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names.clone()
    }

    #[getter]
    fn scaled_sensitivities(&self) -> Vec<f64> {
        self.inner.scaled_sensitivities.clone()
    }

    #[getter]
    fn scaled_diagonal_curvature(&self) -> Vec<f64> {
        self.inner.scaled_diagonal_curvature.clone()
    }

    #[getter]
    fn diagonal_condition_estimate(&self) -> Option<f64> {
        self.inner.diagonal_condition_estimate
    }

    #[getter]
    fn parameters_at_bounds(&self) -> Vec<String> {
        self.inner.parameters_at_bounds.clone()
    }

    #[getter]
    fn rejected_steps(&self) -> usize {
        self.inner.rejected_steps
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "IlluminationCalibrationState",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyIlluminationCalibrationState {
    inner: IlluminationCalibrationState,
}

#[pymethods]
impl PyIlluminationCalibrationState {
    #[getter]
    fn initial_illumination(&self) -> PyIllumination {
        PyIllumination {
            inner: self.inner.initial_illumination.clone(),
        }
    }

    #[getter]
    fn current_illumination(&self) -> PyIllumination {
        PyIllumination {
            inner: self.inner.current_illumination.clone(),
        }
    }

    #[getter]
    fn initial_parameters(&self) -> PyPlanarArrayParameterValues {
        PyPlanarArrayParameterValues {
            inner: self.inner.initial_parameters.clone(),
        }
    }

    #[getter]
    fn current_parameters(&self) -> PyPlanarArrayParameterValues {
        PyPlanarArrayParameterValues {
            inner: self.inner.current_parameters.clone(),
        }
    }

    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names.clone()
    }

    #[getter]
    fn normalized_variables(&self) -> Vec<f64> {
        self.inner.normalized_variables.clone()
    }

    #[getter]
    fn applied_constraints(&self) -> Vec<String> {
        self.inner.applied_constraints.clone()
    }

    #[getter]
    fn parameter_history(&self) -> Vec<PyCalibrationParameterHistoryEntry> {
        self.inner
            .parameter_history
            .iter()
            .cloned()
            .map(|inner| PyCalibrationParameterHistoryEntry { inner })
            .collect()
    }

    #[getter]
    fn loss_history(&self) -> Vec<PyCalibrationLossHistoryEntry> {
        self.inner
            .loss_history
            .iter()
            .cloned()
            .map(|inner| PyCalibrationLossHistoryEntry { inner })
            .collect()
    }

    #[getter]
    fn convergence_reason(&self) -> Option<&'static str> {
        self.inner.convergence_reason.map(convergence_reason_name)
    }

    #[getter]
    fn conditioning(&self) -> PyCalibrationConditioning {
        PyCalibrationConditioning {
            inner: self.inner.conditioning.clone(),
        }
    }

    #[getter]
    fn geometry_recompilations(&self) -> usize {
        self.inner.geometry_recompilations
    }

    #[getter]
    fn multiplicative_updates(&self) -> usize {
        self.inner.multiplicative_updates
    }

    #[getter]
    fn rejected_steps(&self) -> usize {
        self.inner.rejected_steps
    }
}

fn convergence_reason_name(reason: CalibrationConvergenceReason) -> &'static str {
    match reason {
        CalibrationConvergenceReason::MaximumSteps => "maximum_steps",
        CalibrationConvergenceReason::RelativeTolerance => "relative_tolerance",
        CalibrationConvergenceReason::NegligibleGradient => "negligible_gradient",
        CalibrationConvergenceReason::LineSearchFailed => "line_search_failed",
    }
}

#[pyclass(module = "fpm_rs._core", name = "JointReconstruction", frozen)]
pub(crate) struct PyJointReconstruction {
    inner: PyJointObjectAlgorithm,
}

#[derive(Clone)]
enum PyJointObjectAlgorithm {
    Fpie(JointReconstruction<Fpie>),
    Epry(JointReconstruction<Epry>),
}

impl PyJointObjectAlgorithm {
    fn outer_iterations(&self) -> usize {
        match self {
            Self::Fpie(value) => value.outer_iterations,
            Self::Epry(value) => value.outer_iterations,
        }
    }

    fn validate(&self) -> fpm_rs::Result<()> {
        match self {
            Self::Fpie(value) => value.validate(),
            Self::Epry(value) => value.validate(),
        }
    }
}

#[pymethods]
impl PyJointReconstruction {
    #[new]
    #[pyo3(signature = (object_algorithm, optics, initial_illumination, illumination_calibration, *, outer_iterations=10, object_iterations_per_outer=1, illumination_steps_per_outer=1))]
    fn new(
        object_algorithm: &Bound<'_, PyAny>,
        optics: PyRef<'_, PyOptics>,
        initial_illumination: PyRef<'_, PyIllumination>,
        illumination_calibration: PyRef<'_, PyIlluminationCalibration>,
        outer_iterations: usize,
        object_iterations_per_outer: usize,
        illumination_steps_per_outer: usize,
    ) -> PyResult<Self> {
        let inner = if let Ok(algorithm) = object_algorithm.extract::<PyRef<'_, PyFpie>>() {
            PyJointObjectAlgorithm::Fpie(
                JointReconstruction::new(
                    algorithm.inner.clone(),
                    optics.inner.clone(),
                    initial_illumination.inner.clone(),
                    illumination_calibration.inner.clone(),
                    outer_iterations,
                )
                .object_iterations_per_outer(object_iterations_per_outer)
                .illumination_steps_per_outer(illumination_steps_per_outer),
            )
        } else if let Ok(algorithm) = object_algorithm.extract::<PyRef<'_, PyEpry>>() {
            PyJointObjectAlgorithm::Epry(
                JointReconstruction::new(
                    algorithm.inner.clone(),
                    optics.inner.clone(),
                    initial_illumination.inner.clone(),
                    illumination_calibration.inner.clone(),
                    outer_iterations,
                )
                .object_iterations_per_outer(object_iterations_per_outer)
                .illumination_steps_per_outer(illumination_steps_per_outer),
            )
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "object_algorithm must be Fpie or Epry",
            ));
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
    ) -> PyResult<PyJointReconstructionResult> {
        let algorithm = self.inner.clone();
        let options = RunOptions {
            max_iterations: algorithm.outer_iterations(),
            batch_size: usize::MAX,
            schedule: parse_schedule(schedule, schedule_seed)?,
            enable_frame_callbacks: false,
        };
        let problem = problem.inner.clone();
        let checkpoint = resume_from.map(|value| (*value.inner).clone());
        let (callbacks, callback_errors) = build_callbacks(py, callbacks)?;
        let result = py.detach(move || {
            let reconstruction = match algorithm {
                PyJointObjectAlgorithm::Fpie(algorithm) => {
                    let mut runner = Runner::new(algorithm, options).with_callbacks(callbacks);
                    if let Some(checkpoint) = checkpoint {
                        runner = runner.resume_from(checkpoint);
                    }
                    runner.run(&problem)
                }
                PyJointObjectAlgorithm::Epry(algorithm) => {
                    let mut runner = Runner::new(algorithm, options).with_callbacks(callbacks);
                    if let Some(checkpoint) = checkpoint {
                        runner = runner.resume_from(checkpoint);
                    }
                    runner.run(&problem)
                }
            }?;
            CoreJointReconstructionResult::from_reconstruction(reconstruction)
        });
        for error in callback_errors {
            if let Some(error) = lock_callback_error(&error).take() {
                return Err(error);
            }
        }
        PyJointReconstructionResult::from_core(py, result.map_err(to_py_err)?)
    }
}

#[pyclass(module = "fpm_rs._core", name = "JointReconstructionResult", frozen)]
pub(crate) struct PyJointReconstructionResult {
    reconstruction: Py<PyReconstructionResult>,
    initial_illumination: PyIllumination,
    calibrated_illumination: PyIllumination,
    calibrated_model: PyImagePlaneModel,
    initial_parameters: PyPlanarArrayParameterValues,
    final_parameters: PyPlanarArrayParameterValues,
    parameter_history: Vec<PyCalibrationParameterHistoryEntry>,
    loss_history: Vec<PyCalibrationLossHistoryEntry>,
    convergence_reason: Option<&'static str>,
    conditioning: PyCalibrationConditioning,
    diagnostics: PyIlluminationCalibrationState,
}

impl PyJointReconstructionResult {
    fn from_core(py: Python<'_>, result: CoreJointReconstructionResult) -> PyResult<Self> {
        let initial_illumination = PyIllumination {
            inner: result.initial_illumination,
        };
        let calibrated_illumination = PyIllumination {
            inner: result.calibrated_illumination,
        };
        let calibrated_model = PyImagePlaneModel {
            inner: Arc::new(result.calibrated_model),
        };
        let initial_parameters = PyPlanarArrayParameterValues {
            inner: result.initial_parameters,
        };
        let final_parameters = PyPlanarArrayParameterValues {
            inner: result.final_parameters,
        };
        let parameter_history = result
            .parameter_history
            .into_iter()
            .map(|inner| PyCalibrationParameterHistoryEntry { inner })
            .collect();
        let loss_history = result
            .loss_history
            .into_iter()
            .map(|inner| PyCalibrationLossHistoryEntry { inner })
            .collect();
        let convergence_reason = result.convergence_reason.map(convergence_reason_name);
        let conditioning = PyCalibrationConditioning {
            inner: result.conditioning,
        };
        let diagnostics = PyIlluminationCalibrationState {
            inner: result.diagnostics,
        };
        let reconstruction = Py::new(
            py,
            PyReconstructionResult::from_core(py, result.reconstruction)?,
        )?;
        Ok(Self {
            reconstruction,
            initial_illumination,
            calibrated_illumination,
            calibrated_model,
            initial_parameters,
            final_parameters,
            parameter_history,
            loss_history,
            convergence_reason,
            conditioning,
            diagnostics,
        })
    }

    fn to_core(&self, py: Python<'_>) -> PyResult<CoreJointReconstructionResult> {
        let reconstruction = self.reconstruction.bind(py).borrow().to_core(py)?;
        CoreJointReconstructionResult::from_reconstruction(reconstruction).map_err(to_py_err)
    }
}

#[pymethods]
impl PyJointReconstructionResult {
    #[getter]
    fn reconstruction(&self, py: Python<'_>) -> Py<PyReconstructionResult> {
        self.reconstruction.clone_ref(py)
    }

    #[getter]
    fn initial_illumination(&self) -> PyIllumination {
        self.initial_illumination.clone()
    }

    #[getter]
    fn calibrated_illumination(&self) -> PyIllumination {
        self.calibrated_illumination.clone()
    }

    #[getter]
    fn calibrated_model(&self) -> PyImagePlaneModel {
        self.calibrated_model.clone()
    }

    #[getter]
    fn initial_parameters(&self) -> PyPlanarArrayParameterValues {
        self.initial_parameters.clone()
    }

    #[getter]
    fn final_parameters(&self) -> PyPlanarArrayParameterValues {
        self.final_parameters.clone()
    }

    #[getter]
    fn parameter_history(&self) -> Vec<PyCalibrationParameterHistoryEntry> {
        self.parameter_history.clone()
    }

    #[getter]
    fn loss_history(&self) -> Vec<PyCalibrationLossHistoryEntry> {
        self.loss_history.clone()
    }

    #[getter]
    fn convergence_reason(&self) -> Option<&'static str> {
        self.convergence_reason
    }

    #[getter]
    fn conditioning(&self) -> PyCalibrationConditioning {
        self.conditioning.clone()
    }

    #[getter]
    fn diagnostics(&self) -> PyIlluminationCalibrationState {
        self.diagnostics.clone()
    }

    fn save_json(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let result = self.to_core(py)?;
        py.detach(move || result.save_json(path)).map_err(to_py_err)
    }

    #[staticmethod]
    fn load_json(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let result = py
            .detach(move || CoreJointReconstructionResult::load_json(path))
            .map_err(to_py_err)?;
        Self::from_core(py, result)
    }

    #[pyo3(signature = (path, *, run_id=None, label=None, include_previews=true))]
    fn write_bundle(
        &self,
        py: Python<'_>,
        path: PathBuf,
        run_id: Option<String>,
        label: Option<String>,
        include_previews: bool,
    ) -> PyResult<crate::bundle::PyResultBundle> {
        let result = self.to_core(py)?;
        let bundle = py
            .detach(move || {
                result.write_bundle(
                    path,
                    fpm_rs::reconstruction::BundleExportOptions {
                        run_id,
                        label,
                        include_previews,
                    },
                )
            })
            .map_err(to_py_err)?;
        Ok(crate::bundle::PyResultBundle::from_core(bundle))
    }
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
/// [G. Zheng, R. Horstmeyer, and C. Yang, "Wide-field, high-resolution Fourier
/// ptychographic microscopy" (2013)](https://doi.org/10.1038/nphoton.2013.187),
/// Nature Photonics 7, 739-745.
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

/// Noise-robust alternating projection with a pass-adaptive object step.
///
/// The method uses the ordinary fixed-pupil amplitude-projection update, but
/// shares one object step across each complete acquisition pass. It retains the
/// step while the accumulated amplitude-MSE objective makes sufficient
/// relative progress and otherwise reduces it to a positive floor. The
/// controller adds no extra forward evaluation.
///
/// Parameters
/// ----------
/// iterations : int
///     Number of complete passes through the acquisition schedule.
/// initial_object_step : float
///     Positive object relaxation used before the first reduction.
/// progress_threshold : float
///     Minimum relative pass-objective decrease required to retain the step.
/// reduction_factor : float
///     Factor in ``(0, 1)`` applied when progress is insufficient.
/// minimum_object_step : float
///     Positive step floor no greater than ``initial_object_step``.
/// batch_size : int
///     Number of measured frames supplied to each reconstruction step. It does
///     not change adaptation cadence or the numerical path.
/// epsilon : float
///     Positive numerical floor used in projection and relative progress.
///
/// Notes
/// -----
/// Feedback always uses the frame-weighted, mask-aware amplitude MSE after
/// applying configured gains and background. The cited convergence proof is
/// for convex component objectives, whereas FPM phase retrieval is non-convex.
/// The implementation uses the paper's inexpensive accumulated-objective
/// approximation and keeps the pupil fixed.
///
/// References
/// ----------
/// [C. Zuo, J. Sun, and Q. Chen, "Adaptive step-size strategy for noise-robust
/// Fourier ptychographic microscopy" (2016)](https://doi.org/10.1364/OE.24.020724),
/// Optics Express 24(18), 20724-20744.
#[pyclass(
    module = "fpm_rs._core",
    name = "AdaptiveAlternatingProjection",
    frozen
)]
pub(crate) struct PyAdaptiveAlternatingProjection {
    inner: AdaptiveAlternatingProjection,
}

#[pymethods]
impl PyAdaptiveAlternatingProjection {
    #[new]
    #[pyo3(signature = (*, iterations=50, initial_object_step=1.0, progress_threshold=0.01, reduction_factor=0.5, minimum_object_step=0.001, batch_size=1, epsilon=1e-10))]
    fn new(
        iterations: usize,
        initial_object_step: f64,
        progress_threshold: f64,
        reduction_factor: f64,
        minimum_object_step: f64,
        batch_size: usize,
        epsilon: f64,
    ) -> PyResult<Self> {
        let inner = AdaptiveAlternatingProjection {
            iterations,
            initial_object_step,
            progress_threshold,
            reduction_factor,
            minimum_object_step,
            batch_size,
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
/// [A. Maiden, D. Johnson, and P. Li, "Further improvements to the
/// ptychographical iterative engine" (2017)](https://doi.org/10.1364/OPTICA.4.000736),
/// Optica 4(7), 736-745.
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

/// Momentum-accelerated regularized PIE adapted to image-plane FPM.
///
/// `Mpie` applies the object-only rPIE projection used by `Fpie`, then updates a
/// centered complex object-spectrum velocity after a fixed number of
/// positive-weight measured frames. Multiplexed frames count once after all
/// source modes are inserted; zero-weight frames do not advance the interval.
/// Momentum state and a partial interval are preserved in checkpoints.
///
/// The cited method was tested for scanned ptychography and accelerates both
/// object and probe. This implementation keeps the FPM pupil fixed. It exposes
/// separate friction and feedback controls; equal values reproduce the paper's
/// single object momentum coefficient.
///
/// Parameters
/// ----------
/// iterations : int
///     Number of complete passes through the acquisition schedule.
/// object_step : float
///     Positive scale applied to each rPIE object-spectrum correction.
/// stability : float
///     Blend from local pupil power (0) to maximum pupil power (1).
/// momentum_interval : int
///     Positive-weight measured-frame updates between momentum events.
/// momentum_friction : float
///     Previous-velocity fraction in the half-open interval ``[0, 1)``.
/// momentum_feedback : float
///     Updated-velocity fraction added to the object, in ``[0, 1]``.
/// batch_size : int
///     Number of measured frames supplied to each reconstruction step. This
///     does not change momentum cadence.
/// epsilon : float
///     Positive floor added to the rPIE denominator.
/// loss_type : str
///     Loss reported in diagnostics. The projection always enforces amplitude.
///
/// Reference
/// ---------
/// [A. Maiden, D. Johnson, and P. Li, "Further improvements to the
/// ptychographical iterative engine" (2017)](https://doi.org/10.1364/OPTICA.4.000736),
/// Optica 4(7), 736-745.
#[pyclass(module = "fpm_rs._core", name = "Mpie", frozen)]
pub(crate) struct PyMpie {
    inner: Mpie,
}

#[pymethods]
impl PyMpie {
    #[new]
    #[pyo3(signature = (*, iterations=50, object_step=0.2, stability=0.05, momentum_interval=30, momentum_friction=0.9, momentum_feedback=0.9, batch_size=1, epsilon=1e-10, loss_type="amplitude_mse"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        iterations: usize,
        object_step: f64,
        stability: f64,
        momentum_interval: usize,
        momentum_friction: f64,
        momentum_feedback: f64,
        batch_size: usize,
        epsilon: f64,
        loss_type: &str,
    ) -> PyResult<Self> {
        let inner = Mpie {
            iterations,
            object_step,
            stability,
            momentum_interval,
            momentum_friction,
            momentum_feedback,
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
/// With pupil recovery enabled, iteration-boundary results use the compiled
/// pupil as a gauge reference. Supported pupil energy is matched to the
/// reference, pupil and object piston are fixed, and affine pupil phase is
/// removed on axes with zero effective subpixel offsets. Fractional axes keep
/// their affine phase because bilinear crop interpolation does not preserve
/// that ambiguity exactly. This projection runs before iteration callbacks,
/// checkpoints, final results, and continued work after checkpoint restoration.
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
/// References
/// ----------
/// [X. Ou, G. Zheng, and C. Yang, "Embedded pupil function recovery for Fourier
/// ptychographic microscopy" (2014)](https://doi.org/10.1364/OE.22.004960),
/// Optics Express 22(5), 4960-4972.
///
/// [A. Fannjiang and P. Chen, "Blind ptychography: uniqueness and ambiguities"
/// (2020)](https://doi.org/10.1088/1361-6420/ab6504), Inverse Problems 36,
/// 045005. fpm-rs uses a Fourier-domain object and retains affine phase on
/// fractionally interpolated axes.
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
/// [A. Wang, Z. Zhang, S. Wang, A. Pan, C. Ma, and B. Yao, "Fourier
/// Ptychographic Microscopy via Alternating Direction Method of Multipliers"
/// (2022)](https://doi.org/10.3390/cells11091512), Cells 11(9), 1512.
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
/// With pupil recovery enabled, iteration-boundary results use the compiled
/// pupil as a gauge reference. Supported pupil energy is matched to the
/// reference, pupil and object piston are fixed, and affine pupil phase is
/// removed on axes with zero effective subpixel offsets. Fractional axes keep
/// their affine phase because bilinear crop interpolation does not preserve
/// that ambiguity exactly. This projection runs before iteration callbacks,
/// checkpoints, final results, and continued work after checkpoint restoration.
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
/// poisson_truncation_threshold : float or None
///     Positive signal-dependent Poisson outlier coefficient. ``None`` keeps
///     the ordinary untruncated gradient; the cited TPWFP work used 25.
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
/// References
/// ----------
/// [L. Bian, J. Suo, G. Zheng, K. Guo, F. Chen, and Q. Dai, "Fourier
/// ptychographic reconstruction using Wirtinger flow optimization"
/// (2015)](https://doi.org/10.1364/OE.23.004856), Optics Express 23(4),
/// 4856-4866.
///
/// [L. Bian, J. Suo, J. Chung, X. Ou, C. Yang, F. Chen, and Q. Dai, "Fourier
/// ptychographic reconstruction using Poisson maximum likelihood and truncated
/// Wirtinger gradient" (2016)](https://doi.org/10.1038/srep27384), Scientific
/// Reports 6, 27384. fpm-rs uses an intrinsic-intensity mini-batch statistic,
/// one gate for all modes of a multiplexed pixel, and a fixed object step; its
/// optional pupil and illumination updates extend the paper's object-only
/// presentation.
///
/// [A. Fannjiang and P. Chen, "Blind ptychography: uniqueness and ambiguities"
/// (2020)](https://doi.org/10.1088/1361-6420/ab6504), Inverse Problems 36,
/// 045005. fpm-rs uses a Fourier-domain object and retains affine phase on
/// fractionally interpolated axes.
#[pyclass(module = "fpm_rs._core", name = "GradientDescent", frozen)]
pub(crate) struct PyGradientDescent {
    inner: GradientDescent,
}

#[pymethods]
impl PyGradientDescent {
    #[new]
    #[pyo3(signature = (*, iterations=100, object_step=0.5, batch_size=1, epsilon=1e-10, loss_type="amplitude_mse", poisson_truncation_threshold=None, recover_illumination=false, illumination_step=0.1, illumination_finite_difference=0.05, illumination_bounds=1.0, recover_pupil=false, pupil_step=0.05, constrain_pupil_support=true, object_tv=0.0, object_tv_epsilon=1e-6, pupil_smoothing=0.0, parallel_workers=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        iterations: usize,
        object_step: f64,
        batch_size: usize,
        epsilon: f64,
        loss_type: &str,
        poisson_truncation_threshold: Option<f64>,
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
            poisson_truncation_threshold,
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
    module.add_function(wrap_pyfunction!(evaluate_reconstruction_py, module)?)?;
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
    module.add_class::<PyCalibrationParameterSpec>()?;
    module.add_class::<PyPlanarArrayCalibrationParameters>()?;
    module.add_class::<PyBoundedFiniteDifferenceOptimizer>()?;
    module.add_class::<PyIlluminationCalibration>()?;
    module.add_class::<PyPlanarArrayParameterValues>()?;
    module.add_class::<PyCalibrationParameterHistoryEntry>()?;
    module.add_class::<PyCalibrationLossHistoryEntry>()?;
    module.add_class::<PyCalibrationConditioning>()?;
    module.add_class::<PyIlluminationCalibrationState>()?;
    module.add_class::<PyJointReconstruction>()?;
    module.add_class::<PyJointReconstructionResult>()?;
    module.add_class::<PyAlternatingProjection>()?;
    module.add_class::<PyAdaptiveAlternatingProjection>()?;
    module.add_class::<PyFpie>()?;
    module.add_class::<PyMpie>()?;
    module.add_class::<PyEpry>()?;
    module.add_class::<PyAdmm>()?;
    module.add_class::<PyGradientDescent>()?;
    Ok(())
}
