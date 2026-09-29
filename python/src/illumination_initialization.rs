use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use fpm_rs::{
    Error,
    backend::CpuBackend,
    illumination_initialization::{
        BrightfieldCircleInitializer, BrightfieldCircleObservation, BrightfieldCircleOptions,
        InitializationBundle, InitializationBundleArtifact, InitializationBundleVerificationResult,
        PlanarArrayInitializationAction, PlanarArrayInitializationCallback,
        PlanarArrayInitializationDiagnostics, PlanarArrayInitializationFitRecord,
        PlanarArrayInitializationProgress, PlanarArrayInitializationResult,
        PlanarArrayInitializationRuntime, PlanarArrayInitializationStage,
        read_initialization_bundle,
    },
};
use numpy::{PyArray1, ndarray::Array1};
use pyo3::{
    prelude::*,
    types::{PyAny, PyDict},
};

use crate::{
    config::{PyIllumination, PyOptics},
    errors::to_py_err,
    measurements::extract_measurements,
    model::PyImagePlaneModel,
    reconstruction::{PyPlanarArrayCalibrationParameters, PyPlanarArrayParameterValues},
};

#[pyclass(
    module = "fpm_rs._core",
    name = "BrightfieldCircleOptions",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBrightfieldCircleOptions {
    inner: BrightfieldCircleOptions,
}

#[pymethods]
impl PyBrightfieldCircleOptions {
    #[new]
    #[pyo3(signature = (*, frame_indices=None, center_search_radius_na=0.02, brightfield_margin_na=0.002, pupil_radius_search_na=0.01, gaussian_sigma_pixels=2.0, angular_samples=180, radial_derivative_step_pixels=1.0, minimum_arc_fraction=0.2, minimum_edge_contrast=0.01, mean_spectrum_floor=1e-8, robust_residual_scale_na=0.002, maximum_fit_steps=100, fit_relative_tolerance=1e-8, fit_initial_step_size=0.5, fit_minimum_step_size=1e-6, fit_step_reduction=0.5, rank_tolerance=1e-8, pupil_radius_tolerance_na=0.02))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        frame_indices: Option<Vec<usize>>,
        center_search_radius_na: f64,
        brightfield_margin_na: f64,
        pupil_radius_search_na: f64,
        gaussian_sigma_pixels: f64,
        angular_samples: usize,
        radial_derivative_step_pixels: f64,
        minimum_arc_fraction: f64,
        minimum_edge_contrast: f64,
        mean_spectrum_floor: f64,
        robust_residual_scale_na: f64,
        maximum_fit_steps: usize,
        fit_relative_tolerance: f64,
        fit_initial_step_size: f64,
        fit_minimum_step_size: f64,
        fit_step_reduction: f64,
        rank_tolerance: f64,
        pupil_radius_tolerance_na: f64,
    ) -> PyResult<Self> {
        let inner = BrightfieldCircleOptions {
            frame_indices,
            center_search_radius_na,
            brightfield_margin_na,
            pupil_radius_search_na,
            gaussian_sigma_pixels,
            angular_samples,
            radial_derivative_step_pixels,
            minimum_arc_fraction,
            minimum_edge_contrast,
            mean_spectrum_floor,
            robust_residual_scale_na,
            maximum_fit_steps,
            fit_relative_tolerance,
            fit_initial_step_size,
            fit_minimum_step_size,
            fit_step_reduction,
            rank_tolerance,
            pupil_radius_tolerance_na,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn frame_indices(&self) -> Option<Vec<usize>> {
        self.inner.frame_indices.clone()
    }

    #[getter]
    fn center_search_radius_na(&self) -> f64 {
        self.inner.center_search_radius_na
    }

    #[getter]
    fn brightfield_margin_na(&self) -> f64 {
        self.inner.brightfield_margin_na
    }

    #[getter]
    fn pupil_radius_search_na(&self) -> f64 {
        self.inner.pupil_radius_search_na
    }

    #[getter]
    fn gaussian_sigma_pixels(&self) -> f64 {
        self.inner.gaussian_sigma_pixels
    }

    #[getter]
    fn angular_samples(&self) -> usize {
        self.inner.angular_samples
    }

    #[getter]
    fn radial_derivative_step_pixels(&self) -> f64 {
        self.inner.radial_derivative_step_pixels
    }

    #[getter]
    fn minimum_arc_fraction(&self) -> f64 {
        self.inner.minimum_arc_fraction
    }

    #[getter]
    fn minimum_edge_contrast(&self) -> f64 {
        self.inner.minimum_edge_contrast
    }

    #[getter]
    fn mean_spectrum_floor(&self) -> f64 {
        self.inner.mean_spectrum_floor
    }

    #[getter]
    fn robust_residual_scale_na(&self) -> f64 {
        self.inner.robust_residual_scale_na
    }

    #[getter]
    fn maximum_fit_steps(&self) -> usize {
        self.inner.maximum_fit_steps
    }

    #[getter]
    fn fit_relative_tolerance(&self) -> f64 {
        self.inner.fit_relative_tolerance
    }

    #[getter]
    fn fit_initial_step_size(&self) -> f64 {
        self.inner.fit_initial_step_size
    }

    #[getter]
    fn fit_minimum_step_size(&self) -> f64 {
        self.inner.fit_minimum_step_size
    }

    #[getter]
    fn fit_step_reduction(&self) -> f64 {
        self.inner.fit_step_reduction
    }

    #[getter]
    fn rank_tolerance(&self) -> f64 {
        self.inner.rank_tolerance
    }

    #[getter]
    fn pupil_radius_tolerance_na(&self) -> f64 {
        self.inner.pupil_radius_tolerance_na
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "BrightfieldCircleObservation",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBrightfieldCircleObservation {
    inner: BrightfieldCircleObservation,
}

#[pymethods]
impl PyBrightfieldCircleObservation {
    #[getter]
    fn frame_index(&self) -> usize {
        self.inner.frame_index
    }

    #[getter]
    fn source_index(&self) -> usize {
        self.inner.source_index
    }

    #[getter]
    fn nominal_k_rad_per_m(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        pair_to_py(py, self.inner.nominal_k_rad_per_m)
    }

    #[getter]
    fn detected_k_rad_per_m(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.inner
            .detected_k_rad_per_m
            .map(|values| pair_to_py(py, values))
    }

    #[getter]
    fn detected_na(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.inner.detected_na.map(|values| pair_to_py(py, values))
    }

    #[getter]
    fn fourier_grid_position(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.inner
            .fourier_grid_position
            .map(|values| pair_to_py(py, values))
    }

    #[getter]
    fn fitted_pupil_radius_na(&self) -> f64 {
        self.inner.fitted_pupil_radius_na
    }

    #[getter]
    fn first_derivative_score(&self) -> f64 {
        self.inner.first_derivative_score
    }

    #[getter]
    fn second_derivative_score(&self) -> f64 {
        self.inner.second_derivative_score
    }

    #[getter]
    fn combined_score(&self) -> f64 {
        self.inner.combined_score
    }

    #[getter]
    fn conjugate_score(&self) -> f64 {
        self.inner.conjugate_score
    }

    #[getter]
    fn usable_arc_fraction(&self) -> f64 {
        self.inner.usable_arc_fraction
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    fn negative_sample_fraction(&self) -> f64 {
        self.inner.negative_sample_fraction
    }

    #[getter]
    fn rejection_reason(&self) -> Option<String> {
        self.inner.rejection_reason.clone()
    }

    #[getter]
    fn accepted(&self) -> bool {
        self.inner.accepted()
    }
}

fn pair_to_py(py: Python<'_>, values: [f64; 2]) -> Py<PyArray1<f64>> {
    PyArray1::from_owned_array(py, Array1::from_vec(values.to_vec())).unbind()
}

#[pyclass(
    module = "fpm_rs._core",
    name = "PlanarArrayInitializationFitRecord",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPlanarArrayInitializationFitRecord {
    inner: PlanarArrayInitializationFitRecord,
}

#[pymethods]
impl PyPlanarArrayInitializationFitRecord {
    #[getter]
    fn step(&self) -> usize {
        self.inner.step
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
    fn data_loss(&self) -> f64 {
        self.inner.data_loss
    }

    #[getter]
    fn regularization_loss(&self) -> f64 {
        self.inner.regularization_loss
    }

    #[getter]
    fn total_loss(&self) -> f64 {
        self.inner.total_loss
    }

    #[getter]
    fn normalized_values(&self) -> Vec<f64> {
        self.inner.normalized_values.clone()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "PlanarArrayInitializationDiagnostics",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPlanarArrayInitializationDiagnostics {
    inner: PlanarArrayInitializationDiagnostics,
}

#[pymethods]
impl PyPlanarArrayInitializationDiagnostics {
    #[getter]
    fn candidate_frames(&self) -> usize {
        self.inner.candidate_frames
    }

    #[getter]
    fn accepted_observations(&self) -> usize {
        self.inner.accepted_observations
    }

    #[getter]
    fn rejected_observations(&self) -> usize {
        self.inner.rejected_observations
    }

    #[getter]
    fn configured_pupil_radius_na(&self) -> f64 {
        self.inner.configured_pupil_radius_na
    }

    #[getter]
    fn fitted_pupil_radius_na(&self) -> f64 {
        self.inner.fitted_pupil_radius_na
    }

    #[getter]
    fn jacobian_rank(&self) -> usize {
        self.inner.jacobian_rank
    }

    #[getter]
    fn active_parameter_count(&self) -> usize {
        self.inner.active_parameter_count
    }

    #[getter]
    fn jacobian_condition_estimate(&self) -> Option<f64> {
        self.inner.jacobian_condition_estimate
    }

    #[getter]
    fn initial_residual_rms_na(&self) -> f64 {
        self.inner.initial_residual_rms_na
    }

    #[getter]
    fn final_residual_rms_na(&self) -> f64 {
        self.inner.final_residual_rms_na
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "PlanarArrayInitializationRuntime",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPlanarArrayInitializationRuntime {
    inner: PlanarArrayInitializationRuntime,
}

#[pymethods]
impl PyPlanarArrayInitializationRuntime {
    #[getter]
    fn elapsed_seconds(&self) -> f64 {
        self.inner.elapsed_seconds
    }

    #[getter]
    fn measurement_passes(&self) -> usize {
        self.inner.measurement_passes
    }

    #[getter]
    fn physical_objective_evaluations(&self) -> usize {
        self.inner.physical_objective_evaluations
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "PlanarArrayInitializationCallback",
    frozen
)]
pub(crate) struct PyPlanarArrayInitializationCallback {
    callable: Py<PyAny>,
}

#[pymethods]
impl PyPlanarArrayInitializationCallback {
    #[new]
    fn new(callable: Py<PyAny>, py: Python<'_>) -> PyResult<Self> {
        if !callable.bind(py).is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "callable must be callable",
            ));
        }
        Ok(Self { callable })
    }
}

struct PythonInitializationCallback {
    callable: Py<PyAny>,
    error: Arc<Mutex<Option<PyErr>>>,
}

impl PlanarArrayInitializationCallback for PythonInitializationCallback {
    fn on_progress(
        &mut self,
        progress: &PlanarArrayInitializationProgress,
    ) -> fpm_rs::Result<PlanarArrayInitializationAction> {
        Python::attach(|py| {
            let values = PyDict::new(py);
            values.set_item("stage", stage_name(progress.stage))?;
            values.set_item("completed", progress.completed)?;
            values.set_item("total", progress.total)?;
            values.set_item("frame_index", progress.frame_index)?;
            match self.callable.bind(py).call1((values,)) {
                Ok(response) => Ok(if response.is_none() || response.is_truthy()? {
                    PlanarArrayInitializationAction::Continue
                } else {
                    PlanarArrayInitializationAction::Cancel
                }),
                Err(error) => {
                    *lock_callback_error(&self.error) = Some(error);
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Python initialization callback failed",
                    ))
                }
            }
        })
        .map_err(|error| Error::Unsupported(error.to_string()))
    }
}

fn stage_name(stage: PlanarArrayInitializationStage) -> &'static str {
    match stage {
        PlanarArrayInitializationStage::MeanSpectrum => "mean_spectrum",
        PlanarArrayInitializationStage::CircleDetection => "circle_detection",
        PlanarArrayInitializationStage::PhysicalFit => "physical_fit",
        PlanarArrayInitializationStage::Complete => "complete",
    }
}

fn lock_callback_error(
    error: &Arc<Mutex<Option<PyErr>>>,
) -> std::sync::MutexGuard<'_, Option<PyErr>> {
    error
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[pyclass(
    module = "fpm_rs._core",
    name = "PlanarArrayInitializationResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPlanarArrayInitializationResult {
    inner: Arc<PlanarArrayInitializationResult>,
}

#[pymethods]
impl PyPlanarArrayInitializationResult {
    #[getter]
    fn format_version(&self) -> u32 {
        self.inner.format_version
    }

    #[getter]
    fn nominal_illumination(&self) -> PyIllumination {
        PyIllumination {
            inner: self.inner.nominal_illumination.clone(),
        }
    }

    #[getter]
    fn initialized_illumination(&self) -> PyIllumination {
        PyIllumination {
            inner: self.inner.initialized_illumination.clone(),
        }
    }

    #[getter]
    fn initialized_model(&self) -> PyImagePlaneModel {
        PyImagePlaneModel {
            inner: Arc::new(self.inner.initialized_model.clone()),
        }
    }

    #[getter]
    fn parameters(&self) -> PyPlanarArrayCalibrationParameters {
        PyPlanarArrayCalibrationParameters {
            inner: self.inner.parameters.clone(),
        }
    }

    #[getter]
    fn options(&self) -> PyBrightfieldCircleOptions {
        PyBrightfieldCircleOptions {
            inner: self.inner.options.clone(),
        }
    }

    #[getter]
    fn initial_parameters(&self) -> PyPlanarArrayParameterValues {
        PyPlanarArrayParameterValues {
            inner: self.inner.initial_parameters.clone(),
        }
    }

    #[getter]
    fn initialized_parameters(&self) -> PyPlanarArrayParameterValues {
        PyPlanarArrayParameterValues {
            inner: self.inner.initialized_parameters.clone(),
        }
    }

    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names.clone()
    }

    #[getter]
    fn observations(&self) -> Vec<PyBrightfieldCircleObservation> {
        self.inner
            .observations
            .iter()
            .cloned()
            .map(|inner| PyBrightfieldCircleObservation { inner })
            .collect()
    }

    #[getter]
    fn fit_history(&self) -> Vec<PyPlanarArrayInitializationFitRecord> {
        self.inner
            .fit_history
            .iter()
            .cloned()
            .map(|inner| PyPlanarArrayInitializationFitRecord { inner })
            .collect()
    }

    #[getter]
    fn diagnostics(&self) -> PyPlanarArrayInitializationDiagnostics {
        PyPlanarArrayInitializationDiagnostics {
            inner: self.inner.diagnostics.clone(),
        }
    }

    #[getter]
    fn runtime(&self) -> PyPlanarArrayInitializationRuntime {
        PyPlanarArrayInitializationRuntime {
            inner: self.inner.runtime.clone(),
        }
    }

    fn save_json(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

    fn write_bundle(&self, path: PathBuf) -> PyResult<PyInitializationBundle> {
        Ok(PyInitializationBundle {
            inner: self.inner.write_bundle(path).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn load_json(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(PlanarArrayInitializationResult::load_json(path).map_err(to_py_err)?),
        })
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "InitializationBundleArtifact",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyInitializationBundleArtifact {
    inner: InitializationBundleArtifact,
}

#[pymethods]
impl PyInitializationBundleArtifact {
    #[getter]
    fn role(&self) -> String {
        self.inner.role.clone()
    }

    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path.clone()
    }

    #[getter]
    fn media_type(&self) -> String {
        self.inner.media_type.clone()
    }

    #[getter]
    fn byte_size(&self) -> u64 {
        self.inner.byte_size
    }

    #[getter]
    fn sha256(&self) -> String {
        self.inner.sha256.clone()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "InitializationBundleVerificationResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyInitializationBundleVerificationResult {
    inner: InitializationBundleVerificationResult,
}

#[pymethods]
impl PyInitializationBundleVerificationResult {
    #[getter]
    fn artifact_count(&self) -> usize {
        self.inner.artifact_count
    }

    #[getter]
    fn total_bytes(&self) -> u64 {
        self.inner.total_bytes
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "InitializationBundle",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyInitializationBundle {
    inner: InitializationBundle,
}

#[pymethods]
impl PyInitializationBundle {
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path.clone()
    }

    #[getter]
    fn manifest_path(&self) -> PathBuf {
        self.inner.manifest_path.clone()
    }

    #[getter]
    fn result_artifact(&self) -> PyInitializationBundleArtifact {
        PyInitializationBundleArtifact {
            inner: self.inner.result_artifact.clone(),
        }
    }

    #[getter]
    fn observations_artifact(&self) -> PyInitializationBundleArtifact {
        PyInitializationBundleArtifact {
            inner: self.inner.observations_artifact.clone(),
        }
    }

    #[getter]
    fn fit_history_artifact(&self) -> PyInitializationBundleArtifact {
        PyInitializationBundleArtifact {
            inner: self.inner.fit_history_artifact.clone(),
        }
    }

    #[getter]
    fn result(&self) -> PyPlanarArrayInitializationResult {
        PyPlanarArrayInitializationResult {
            inner: Arc::new(self.inner.result.clone()),
        }
    }

    fn verify(&self) -> PyResult<PyInitializationBundleVerificationResult> {
        Ok(PyInitializationBundleVerificationResult {
            inner: self.inner.verify().map_err(to_py_err)?,
        })
    }
}

#[pyfunction(name = "read_initialization_bundle")]
fn read_initialization_bundle_py(path: PathBuf) -> PyResult<PyInitializationBundle> {
    Ok(PyInitializationBundle {
        inner: read_initialization_bundle(path).map_err(to_py_err)?,
    })
}

#[pyclass(
    module = "fpm_rs._core",
    name = "BrightfieldCircleInitializer",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBrightfieldCircleInitializer {
    inner: BrightfieldCircleInitializer,
}

#[pymethods]
impl PyBrightfieldCircleInitializer {
    #[new]
    #[pyo3(signature = (parameters, *, options=None))]
    fn new(
        parameters: PyRef<'_, PyPlanarArrayCalibrationParameters>,
        options: Option<PyRef<'_, PyBrightfieldCircleOptions>>,
    ) -> PyResult<Self> {
        let inner = BrightfieldCircleInitializer::new(parameters.inner.clone()).options(
            options
                .as_deref()
                .map_or_else(BrightfieldCircleOptions::default, |value| {
                    value.inner.clone()
                }),
        );
        inner.options.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn parameters(&self) -> PyPlanarArrayCalibrationParameters {
        PyPlanarArrayCalibrationParameters {
            inner: self.inner.parameters.clone(),
        }
    }

    #[getter]
    fn options(&self) -> PyBrightfieldCircleOptions {
        PyBrightfieldCircleOptions {
            inner: self.inner.options.clone(),
        }
    }

    #[pyo3(signature = (measurements, optics, nominal_illumination, model, *, frame_weights=None, masks=None, callback=None))]
    #[allow(clippy::too_many_arguments)]
    fn initialize(
        &self,
        py: Python<'_>,
        measurements: &Bound<'_, PyAny>,
        optics: PyRef<'_, PyOptics>,
        nominal_illumination: PyRef<'_, PyIllumination>,
        model: PyRef<'_, PyImagePlaneModel>,
        frame_weights: Option<Vec<f64>>,
        masks: Option<&Bound<'_, PyAny>>,
        callback: Option<PyRef<'_, PyPlanarArrayInitializationCallback>>,
    ) -> PyResult<PyPlanarArrayInitializationResult> {
        let measurements = extract_measurements(measurements, frame_weights, masks)?;
        let optics = optics.inner.clone();
        let illumination = nominal_illumination.inner.clone();
        let model = model.inner.clone();
        let initializer = self.inner.clone();
        let callback_error = callback.as_ref().map(|_| Arc::new(Mutex::new(None)));
        let callback_callable = callback.map(|value| value.callable.clone_ref(py));
        let shared_error = callback_error.clone();
        let result = py.detach(move || {
            if let (Some(callable), Some(error)) = (callback_callable, shared_error) {
                let backend = Arc::new(CpuBackend::new(
                    model.image_shape(),
                    model.reconstruction_shape(),
                )?);
                let mut callback = PythonInitializationCallback { callable, error };
                initializer.initialize_with_callback(
                    measurements.as_ref(),
                    &optics,
                    &illumination,
                    model.as_ref(),
                    backend,
                    &mut callback,
                )
            } else {
                initializer.initialize(
                    measurements.as_ref(),
                    &optics,
                    &illumination,
                    model.as_ref(),
                )
            }
        });
        if let Some(error) = callback_error
            && let Some(error) = lock_callback_error(&error).take()
        {
            return Err(error);
        }
        Ok(PyPlanarArrayInitializationResult {
            inner: Arc::new(result.map_err(to_py_err)?),
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(read_initialization_bundle_py, module)?)?;
    module.add_class::<PyBrightfieldCircleOptions>()?;
    module.add_class::<PyBrightfieldCircleObservation>()?;
    module.add_class::<PyPlanarArrayInitializationFitRecord>()?;
    module.add_class::<PyPlanarArrayInitializationDiagnostics>()?;
    module.add_class::<PyPlanarArrayInitializationRuntime>()?;
    module.add_class::<PyPlanarArrayInitializationCallback>()?;
    module.add_class::<PyPlanarArrayInitializationResult>()?;
    module.add_class::<PyInitializationBundleArtifact>()?;
    module.add_class::<PyInitializationBundleVerificationResult>()?;
    module.add_class::<PyInitializationBundle>()?;
    module.add_class::<PyBrightfieldCircleInitializer>()?;
    Ok(())
}
