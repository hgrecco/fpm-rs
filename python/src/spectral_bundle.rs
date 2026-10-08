//! Python reopening of separately tagged spectral result bundles.
use crate::{
    bundle::{PyBundleArtifact, PyBundleVerificationResult, artifact},
    errors::to_py_err,
    multi_wavelength::PyMultiWavelengthSolverResult,
    spectral::{PyOpticalPathDifferenceResult, PySpectralReconstructionResult},
    spectral_checkpoint::PySpectralReconstructionCheckpoint,
};
use fpm_rs::tabular::parquet::{SpectralResultBundle, read_spectral_bundle};
use pyo3::prelude::*;
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};
/// Spectral result bundle with eager manifest metadata and verified lazy result loading.
/// Scientific result getters return independent writable NumPy copies. Checkpoint,
/// joint OPD and unwrapping records remain typed; file work releases the GIL.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralResultBundle",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralResultBundle {
    inner: Arc<SpectralResultBundle>,
}
impl PySpectralResultBundle {
    pub(crate) fn from_core(inner: SpectralResultBundle) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}
#[pymethods]
impl PySpectralResultBundle {
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path().to_owned()
    }
    #[getter]
    fn run_id(&self) -> &str {
        self.inner.run_id()
    }
    #[getter]
    fn label(&self) -> Option<&str> {
        self.inner.label()
    }
    #[getter]
    fn artifacts(&self) -> BTreeMap<String, PyBundleArtifact> {
        self.inner
            .artifacts()
            .iter()
            .map(|(k, v)| (k.clone(), artifact(v.clone())))
            .collect()
    }
    #[getter]
    fn result(&self, py: Python<'_>) -> PyResult<PySpectralReconstructionResult> {
        let inner = self.inner.clone();
        Ok(PySpectralReconstructionResult {
            inner: py.detach(move || inner.result()).map_err(to_py_err)?,
        })
    }
    #[getter]
    fn checkpoint(&self, py: Python<'_>) -> PyResult<PySpectralReconstructionCheckpoint> {
        let inner = self.inner.clone();
        Ok(PySpectralReconstructionCheckpoint {
            inner: Arc::new(py.detach(move || inner.checkpoint()).map_err(to_py_err)?),
        })
    }
    #[getter]
    fn joint_result(&self, py: Python<'_>) -> PyResult<Option<PyMultiWavelengthSolverResult>> {
        let inner = self.inner.clone();
        Ok(py
            .detach(move || inner.joint_result())
            .map_err(to_py_err)?
            .map(PyMultiWavelengthSolverResult::from_core))
    }
    #[getter]
    fn unwrapped_opd(&self, py: Python<'_>) -> PyResult<Option<PyOpticalPathDifferenceResult>> {
        let inner = self.inner.clone();
        Ok(py
            .detach(move || inner.unwrapped_opd())
            .map_err(to_py_err)?
            .map(|v| PyOpticalPathDifferenceResult { inner: Arc::new(v) }))
    }
    fn verify(&self, py: Python<'_>) -> PyResult<PyBundleVerificationResult> {
        let inner = self.inner.clone();
        py.detach(move || inner.verify())
            .map(Into::into)
            .map_err(to_py_err)
    }
    fn clear_cache(&self) {
        self.inner.clear_cache();
    }
}
#[pyfunction(name = "read_spectral_bundle")]
#[pyo3(signature=(path))]
fn read_spectral_bundle_py(py: Python<'_>, path: PathBuf) -> PyResult<PySpectralResultBundle> {
    py.detach(move || read_spectral_bundle(path))
        .map(PySpectralResultBundle::from_core)
        .map_err(to_py_err)
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySpectralResultBundle>()?;
    module.add_function(wrap_pyfunction!(read_spectral_bundle_py, module)?)?;
    Ok(())
}
