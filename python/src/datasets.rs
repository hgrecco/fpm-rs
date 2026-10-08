use std::{path::PathBuf, sync::Arc};

use fpm_rs::datasets::{
    Dataset, DatasetListing, DatasetRegistry, open_dataset as core_open_dataset,
};
use numpy::PyArray2;
use pyo3::{prelude::*, types::PyDict};

use crate::{
    arrays::{array2_to_py, complex_array2_to_py},
    errors::to_py_err,
    measurements::PyMeasurementStack,
    model::PyImagePlaneModel,
    reconstruction::PyReconstructionProblem,
};

fn to_dataset_py_err(error: fpm_rs::Error) -> PyErr {
    match error {
        error @ fpm_rs::Error::Dataset(_) => to_py_err(error),
        error => to_py_err(fpm_rs::Error::Dataset(error.to_string())),
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "DatasetRegistryEntry",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyDatasetRegistryEntry {
    listing: DatasetListing,
}

#[pymethods]
impl PyDatasetRegistryEntry {
    #[getter]
    fn id(&self) -> &str {
        &self.listing.entry.id
    }

    #[getter]
    fn version(&self) -> &str {
        &self.listing.entry.version
    }

    #[getter]
    fn title(&self) -> &str {
        &self.listing.entry.title
    }

    #[getter]
    fn description(&self) -> &str {
        &self.listing.entry.description
    }

    #[getter]
    fn format_version(&self) -> u32 {
        self.listing.entry.format_version
    }

    #[getter]
    fn archive_url(&self) -> &str {
        &self.listing.entry.archive.url
    }

    #[getter]
    fn archive_sha256(&self) -> &str {
        &self.listing.entry.archive.sha256
    }

    #[getter]
    fn archive_size_bytes(&self) -> u64 {
        self.listing.entry.archive.size_bytes
    }

    #[getter]
    fn license_spdx(&self) -> &str {
        &self.listing.entry.license.spdx
    }

    #[getter]
    fn license_url(&self) -> &str {
        &self.listing.entry.license.url
    }

    #[getter]
    fn citation_doi(&self) -> &str {
        &self.listing.entry.citation.doi
    }

    #[getter]
    fn citation_text(&self) -> &str {
        &self.listing.entry.citation.text
    }

    #[getter]
    fn source_url(&self) -> &str {
        &self.listing.entry.source.url
    }

    #[getter]
    fn source_description(&self) -> &str {
        &self.listing.entry.source.description
    }

    #[getter]
    fn tags(&self) -> Vec<String> {
        self.listing.entry.tags.clone()
    }

    #[getter]
    fn cached(&self) -> bool {
        self.listing.cached
    }

    #[getter]
    fn cache_path(&self) -> Option<PathBuf> {
        self.listing.cache_path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetRegistryEntry(id={:?}, version={:?}, cached={})",
            self.id(),
            self.version(),
            self.cached()
        )
    }
}

#[pyclass(module = "fpm_rs._core", name = "Dataset", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDataset {
    inner: Arc<Dataset>,
}

impl PyDataset {
    fn new(inner: Dataset) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

#[pymethods]
impl PyDataset {
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        self.inner.source_path().map(PathBuf::from)
    }

    #[getter]
    fn measurements(&self) -> PyMeasurementStack {
        PyMeasurementStack {
            inner: Arc::new(self.inner.measurements().clone()),
        }
    }

    #[getter]
    fn true_model(&self) -> PyImagePlaneModel {
        PyImagePlaneModel {
            inner: Arc::new(
                self.inner
                    .configuration()
                    .compiled_models
                    .true_model
                    .clone(),
            ),
        }
    }

    #[getter]
    fn reconstruction_model(&self) -> PyImagePlaneModel {
        PyImagePlaneModel {
            inner: Arc::new(
                self.inner
                    .configuration()
                    .compiled_models
                    .reconstruction_model
                    .clone(),
            ),
        }
    }

    #[getter]
    fn ground_truth_object(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyArray2<fpm_rs::Complex64>>>> {
        self.inner
            .ground_truth_object()
            .cloned()
            .map(|array| complex_array2_to_py(py, array))
            .transpose()
    }

    #[getter]
    fn valid_object_mask(&self, py: Python<'_>) -> PyResult<Option<Py<PyArray2<u8>>>> {
        self.inner
            .valid_object_mask()
            .cloned()
            .map(|array| array2_to_py(py, array))
            .transpose()
    }

    #[getter]
    fn provenance(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dictionary = PyDict::new(py);
        for (key, value) in self.inner.provenance() {
            dictionary.set_item(key, value)?;
        }
        Ok(dictionary.unbind())
    }

    #[getter]
    fn measurement_units(&self) -> Option<&str> {
        self.inner.measurement_units()
    }

    fn reconstruction_problem(&self) -> PyResult<PyReconstructionProblem> {
        PyReconstructionProblem::from_parts(
            self.inner.measurements().clone(),
            self.inner
                .configuration()
                .compiled_models
                .reconstruction_model
                .clone(),
            self.inner.provenance().get("dataset_id").cloned(),
        )
        .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(path={:?}, frames={}, image_shape={:?})",
            self.path(),
            self.inner.measurements().frame_count(),
            self.inner.measurements().image_shape()
        )
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "DatasetRegistry",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyDatasetRegistry {
    inner: DatasetRegistry,
}

#[pymethods]
impl PyDatasetRegistry {
    #[new]
    #[pyo3(signature = (*, registry_url=None, cache_dir=None))]
    fn new(registry_url: Option<String>, cache_dir: Option<PathBuf>) -> PyResult<Self> {
        let inner = match (registry_url, cache_dir) {
            (None, None) => DatasetRegistry::from_defaults(),
            (Some(registry_url), Some(cache_dir)) => DatasetRegistry::new(registry_url, cache_dir),
            (registry_url, cache_dir) => {
                let defaults = DatasetRegistry::from_defaults().map_err(to_dataset_py_err)?;
                DatasetRegistry::new(
                    registry_url.unwrap_or_else(|| defaults.registry_url().to_owned()),
                    cache_dir.unwrap_or_else(|| defaults.cache_dir().to_owned()),
                )
            }
        }
        .map_err(to_dataset_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn registry_url(&self) -> &str {
        self.inner.registry_url()
    }

    #[getter]
    fn cache_dir(&self) -> PathBuf {
        self.inner.cache_dir().to_owned()
    }

    fn list(&self, py: Python<'_>) -> PyResult<Vec<PyDatasetRegistryEntry>> {
        let registry = self.inner.clone();
        py.detach(move || registry.list())
            .map(|listings| {
                listings
                    .into_iter()
                    .map(|listing| PyDatasetRegistryEntry { listing })
                    .collect()
            })
            .map_err(to_dataset_py_err)
    }

    fn download(&self, py: Python<'_>, id: String) -> PyResult<PathBuf> {
        let registry = self.inner.clone();
        py.detach(move || registry.download(&id))
            .map_err(to_dataset_py_err)
    }

    fn download_all(&self, py: Python<'_>) -> PyResult<Vec<PathBuf>> {
        let registry = self.inner.clone();
        py.detach(move || registry.download_all())
            .map_err(to_dataset_py_err)
    }

    fn open(&self, py: Python<'_>, id: String) -> PyResult<PyDataset> {
        let registry = self.inner.clone();
        py.detach(move || registry.open(&id))
            .map(PyDataset::new)
            .map_err(to_dataset_py_err)
    }

    /// Opens a registered version-two spectral ID through verified download/cache operations.
    /// Releases the GIL; version-one data requires open instead.
    fn open_spectral(&self, py: Python<'_>, id: String) -> PyResult<PySpectralDataset> {
        let registry = self.inner.clone();
        py.detach(move || registry.open_spectral(&id))
            .map(PySpectralDataset::new)
            .map_err(to_dataset_py_err)
    }

    fn clean(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        let registry = self.inner.clone();
        py.detach(move || registry.clean(&id))
            .map_err(to_dataset_py_err)
    }

    fn clean_all(&self, py: Python<'_>) -> PyResult<usize> {
        let registry = self.inner.clone();
        py.detach(move || registry.clean_all())
            .map_err(to_dataset_py_err)
    }
}

/// Resident version-two spectral dataset loaded entirely offline from converted files.
/// Channel IDs, vacuum wavelengths and response provenance are explicit; no RGB
/// channel inference or registration occurs. Scientific getters return writable copies.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralDataset",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralDataset {
    inner: Arc<fpm_rs::datasets::SpectralDataset>,
}
impl PySpectralDataset {
    fn new(inner: fpm_rs::datasets::SpectralDataset) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}
#[pymethods]
impl PySpectralDataset {
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.source_path().to_owned()
    }
    #[getter]
    fn measurements(&self) -> PyMeasurementStack {
        PyMeasurementStack {
            inner: Arc::new(self.inner.measurements().clone()),
        }
    }
    #[getter]
    fn model(&self) -> crate::spectral::PySpectralImagePlaneModel {
        crate::spectral::PySpectralImagePlaneModel {
            inner: Arc::new(self.inner.model().clone()),
        }
    }
    #[getter]
    fn response_provenance(&self) -> std::collections::BTreeMap<String, String> {
        self.inner
            .manifest()
            .channels
            .iter()
            .map(|c| (c.channel_id.clone(), c.response_provenance.clone()))
            .collect()
    }
    #[getter]
    fn provenance(&self) -> std::collections::BTreeMap<String, String> {
        self.inner.manifest().provenance.clone()
    }
    #[getter]
    fn measurement_units(&self) -> Option<String> {
        self.inner.manifest().measurement_units.clone()
    }
    #[getter]
    fn ground_truth_objects(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Option<Py<PyArray2<fpm_rs::Complex64>>>>> {
        self.inner
            .ground_truth_objects()
            .iter()
            .map(|v| {
                v.as_ref()
                    .map(|a| complex_array2_to_py(py, a.clone()))
                    .transpose()
            })
            .collect()
    }
    #[getter]
    fn valid_object_masks(&self, py: Python<'_>) -> PyResult<Vec<Option<Py<PyArray2<u8>>>>> {
        self.inner
            .valid_object_masks()
            .iter()
            .map(|v| v.as_ref().map(|a| array2_to_py(py, a.clone())).transpose())
            .collect()
    }
    fn reconstruction_problem(&self) -> crate::spectral::PySpectralReconstructionProblem {
        crate::spectral::PySpectralReconstructionProblem {
            measurements: Arc::new(self.inner.measurements().clone()),
            model: Arc::new(self.inner.model().clone()),
        }
    }
}
/// Loads an explicit version-two spectral profile from a local directory, with no network.
/// Validates compiled kernels, scalar frames, sparse exposure order and contained paths.
/// File work releases the GIL; failures raise DatasetError.
#[pyfunction]
#[pyo3(signature=(*,path))]
fn load_spectral_dataset(py: Python<'_>, path: PathBuf) -> PyResult<PySpectralDataset> {
    py.detach(move || fpm_rs::datasets::DatasetLoader::new(path)?.load_spectral())
        .map(PySpectralDataset::new)
        .map_err(to_dataset_py_err)
}

#[pyfunction]
#[pyo3(signature = (id, *, registry_url=None, cache_dir=None))]
fn open_dataset(
    py: Python<'_>,
    id: String,
    registry_url: Option<String>,
    cache_dir: Option<PathBuf>,
) -> PyResult<PyDataset> {
    if registry_url.is_none() && cache_dir.is_none() {
        return py
            .detach(move || core_open_dataset(&id))
            .map(PyDataset::new)
            .map_err(to_dataset_py_err);
    }
    let registry = PyDatasetRegistry::new(registry_url, cache_dir)?.inner;
    py.detach(move || registry.open(&id))
        .map(PyDataset::new)
        .map_err(to_dataset_py_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDatasetRegistryEntry>()?;
    module.add_class::<PyDataset>()?;
    module.add_class::<PySpectralDataset>()?;
    module.add_function(wrap_pyfunction!(load_spectral_dataset, module)?)?;
    module.add_class::<PyDatasetRegistry>()?;
    module.add_function(wrap_pyfunction!(open_dataset, module)?)?;
    Ok(())
}
