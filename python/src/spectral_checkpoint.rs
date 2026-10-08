//! Python access to separately versioned spectral iteration state.
use crate::{arrays::array2_to_py, errors::to_py_err, spectral::PySpectralImagePlaneModel};
use fpm_rs::reconstruction::{SpectralCheckpointOptions, SpectralReconstructionCheckpoint};
use numpy::PyArray2;
use pyo3::prelude::*;
use std::{path::PathBuf, sync::Arc};

/// Spectral AP or joint OPD iteration-boundary snapshot, separate from ordinary checkpoints.
/// Saves float-roundtrip JSON, compiled channels, detector-data fingerprint, schedules,
/// options and gauges. Loading validates state; the solver also verifies the problem.
/// File operations release the GIL. NumPy getters return writable copies.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralReconstructionCheckpoint",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralReconstructionCheckpoint {
    pub(crate) inner: Arc<SpectralReconstructionCheckpoint>,
}
#[pymethods]
impl PySpectralReconstructionCheckpoint {
    #[staticmethod]
    #[pyo3(signature=(*,path))]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(move || SpectralReconstructionCheckpoint::load(path))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
    #[pyo3(signature=(*,path))]
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let inner = self.inner.clone();
        py.detach(move || inner.save(path)).map_err(to_py_err)
    }
    #[getter]
    fn format_version(&self) -> u32 {
        self.inner.format_version()
    }
    #[getter]
    fn completed_iterations(&self) -> usize {
        self.inner.completed_iterations()
    }
    #[getter]
    fn algorithm(&self) -> &str {
        self.inner.algorithm()
    }
    #[getter]
    fn model(&self) -> PySpectralImagePlaneModel {
        PySpectralImagePlaneModel {
            inner: Arc::new(self.inner.model().clone()),
        }
    }
    #[getter]
    fn trace(&self) -> Vec<(usize, f64, f64)> {
        self.inner
            .trace()
            .iterations
            .iter()
            .map(|r| (r.iteration, r.objective, r.elapsed_seconds))
            .collect()
    }
    #[getter]
    fn opd_m(&self, py: Python<'_>) -> PyResult<Option<Py<PyArray2<f64>>>> {
        self.inner
            .opd_m()
            .map(|v| array2_to_py(py, v.to_owned()))
            .transpose()
    }
}
pub(crate) fn options(directory: Option<PathBuf>, every: usize) -> SpectralCheckpointOptions {
    SpectralCheckpointOptions { directory, every }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySpectralReconstructionCheckpoint>()
}
