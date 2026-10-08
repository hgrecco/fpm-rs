use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

use fpm_rs::{
    benchmark::BenchmarkRecord,
    benchmark_bundle::{
        BenchmarkBundle, BenchmarkBundleExportOptions, read_benchmark_bundle,
        write_benchmark_bundle,
    },
    reconstruction::ReconstructionResult,
};
use pyo3::{prelude::*, types::PyDict};

use crate::{
    bundle::{PyBundleArtifact, PyResultBundle, artifact},
    datasets::PyDatasetSubset,
    errors::to_py_err,
    reconstruction::PyReconstructionResult,
};

#[pyclass(
    module = "fpm_rs._core",
    name = "BenchmarkBundleTables",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyBenchmarkBundleTables {
    #[pyo3(get)]
    runs: PyBundleArtifact,
    #[pyo3(get)]
    frames: PyBundleArtifact,
    #[pyo3(get)]
    artifacts: PyBundleArtifact,
    #[pyo3(get)]
    metadata: PyBundleArtifact,
}

#[pymethods]
impl PyBenchmarkBundleTables {}

#[pyclass(module = "fpm_rs._core", name = "BenchmarkBundle", frozen)]
pub(crate) struct PyBenchmarkBundle {
    inner: BenchmarkBundle,
    tables: PyBenchmarkBundleTables,
    results: BTreeMap<String, PyResultBundle>,
}

impl PyBenchmarkBundle {
    fn from_core(inner: BenchmarkBundle) -> Self {
        let tables = PyBenchmarkBundleTables {
            runs: artifact(inner.tables.runs.clone()),
            frames: artifact(inner.tables.frames.clone()),
            artifacts: artifact(inner.tables.artifacts.clone()),
            metadata: artifact(inner.tables.metadata.clone()),
        };
        let results = inner
            .results
            .iter()
            .map(|(run_id, result)| (run_id.clone(), PyResultBundle::from_core(result.clone())))
            .collect();
        Self {
            inner,
            tables,
            results,
        }
    }
}

#[pymethods]
impl PyBenchmarkBundle {
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path.clone()
    }

    #[getter]
    fn manifest_path(&self) -> PathBuf {
        self.inner.manifest_path.clone()
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }

    #[getter]
    fn tables(&self) -> PyBenchmarkBundleTables {
        self.tables.clone()
    }

    #[getter]
    fn results(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);
        for (run_id, bundle) in &self.results {
            output.set_item(run_id, Py::new(py, bundle.clone())?)?;
        }
        Ok(output.unbind())
    }
}

struct BenchmarkSuiteEntry {
    record: BenchmarkRecord,
    result: ReconstructionResult,
}

#[pyclass(module = "fpm_rs._core", name = "BenchmarkSuite", frozen)]
pub(crate) struct PyBenchmarkSuite {
    name: String,
    entries: Mutex<Vec<BenchmarkSuiteEntry>>,
}

impl PyBenchmarkSuite {
    fn entries(&self) -> MutexGuard<'_, Vec<BenchmarkSuiteEntry>> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[pymethods]
impl PyBenchmarkSuite {
    #[new]
    fn new(name: String) -> PyResult<Self> {
        if name.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "benchmark suite name must be non-empty",
            ));
        }
        Ok(Self {
            name,
            entries: Mutex::new(Vec::new()),
        })
    }

    /// Add a completed result; an optional resolved dataset subset preserves its
    /// original frame/illumination identities, crop and provenance in the bundle.
    /// Frame count and shapes must match; no residual metrics are recomputed.
    #[pyo3(signature = (result, *, case_id, dataset_name, algorithm_configuration="", dataset_subset=None))]
    fn add_result(
        &self,
        py: Python<'_>,
        result: PyRef<'_, PyReconstructionResult>,
        case_id: String,
        dataset_name: String,
        algorithm_configuration: &str,
        dataset_subset: Option<PyRef<'_, PyDatasetSubset>>,
    ) -> PyResult<String> {
        if case_id.is_empty() || dataset_name.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "case_id and dataset_name must be non-empty",
            ));
        }
        let mut result = result.to_core(py)?;
        result.metadata.insert("case_id".into(), case_id.clone());
        result
            .metadata
            .insert("dataset_name".into(), dataset_name.clone());
        result.metadata.insert(
            "algorithm_configuration".into(),
            algorithm_configuration.into(),
        );
        let record = if let Some(subset) = dataset_subset {
            BenchmarkRecord::from_subset_result(
                case_id,
                dataset_name,
                algorithm_configuration,
                &result,
                &subset.inner,
            )
            .map_err(to_py_err)?
        } else {
            BenchmarkRecord::from_result(case_id, dataset_name, algorithm_configuration, &result)
        };
        result.metadata.extend(record.metadata.clone());
        result
            .metadata
            .insert("case_id".into(), record.case_id.clone());
        result
            .metadata
            .insert("dataset_name".into(), record.dataset_name.clone());
        let run_id = record.run_id.clone();
        self.entries().push(BenchmarkSuiteEntry { record, result });
        Ok(run_id)
    }

    #[pyo3(signature = (path, *, label=None))]
    fn write_bundle(
        &self,
        py: Python<'_>,
        path: PathBuf,
        label: Option<String>,
    ) -> PyResult<PyBenchmarkBundle> {
        let entries = self.entries();
        let records = entries
            .iter()
            .map(|entry| entry.record.clone())
            .collect::<Vec<_>>();
        let results = entries
            .iter()
            .map(|entry| (entry.record.run_id.clone(), entry.result.clone()))
            .collect::<BTreeMap<_, _>>();
        drop(entries);
        let name = self.name.clone();
        py.detach(move || {
            write_benchmark_bundle(
                path,
                name,
                &records,
                &results,
                BenchmarkBundleExportOptions { label },
            )
        })
        .map(PyBenchmarkBundle::from_core)
        .map_err(to_py_err)
    }
}

#[pyfunction(name = "read_benchmark_bundle")]
fn read_benchmark_bundle_py(py: Python<'_>, path: PathBuf) -> PyResult<PyBenchmarkBundle> {
    py.detach(move || read_benchmark_bundle(path))
        .map(PyBenchmarkBundle::from_core)
        .map_err(to_py_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(read_benchmark_bundle_py, module)?)?;
    module.add_class::<PyBenchmarkBundleTables>()?;
    module.add_class::<PyBenchmarkBundle>()?;
    module.add_class::<PyBenchmarkSuite>()?;
    Ok(())
}
