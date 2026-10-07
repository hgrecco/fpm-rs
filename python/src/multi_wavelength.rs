//! Python API for joint shared-OPD optimization against spectral detector data.

use std::sync::Arc;

use fpm_rs::{
    algorithms::MultiWavelengthGradientDescent, reconstruction::SpectralReconstructionProblem,
};
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3, ndarray::Axis};
use pyo3::prelude::*;

use crate::{
    arrays::{array2_to_py, core_array2, core_array3},
    errors::to_py_err,
    spectral::{
        PyOpticalPathDifferenceResult, PySpectralReconstructionProblem,
        PySpectralReconstructionResult, PySyntheticWavelengthUnwrapper, extract_phase_reference,
    },
};

/// Joint nondispersive OPD and wavelength-amplitude optimization of detector intensities.
/// Uses one OPD map for every channel phase, analytic full-data adjoints, and
/// monotone backtracking of a smoothed amplitude objective. The explicit reference
/// region is fixed at its known OPD; piston offsets preserve the initial OPD mean.
/// Automatic initialization runs independent AP then synthetic-wavelength mixing.
/// Pupils/calibration remain fixed; NumPy inputs are copied and computation releases the GIL.
///
/// References
/// ----------
/// L. Bian, J. Suo, G. Zheng, K. Guo, F. Chen, and Q. Dai,
/// [“Fourier ptychographic reconstruction using Wirtinger flow optimization”](https://doi.org/10.1364/OE.23.004856),
/// Optics Express 23(4), 4856–4866 (2015), for FPM loss-gradient optimization.
/// Shared OPD, the smoothed amplitude loss, box/gauge projections, and monotone
/// backtracking are implementation extensions. SyntheticWavelengthUnwrapper
/// documents the initialization method and reference.
#[pyclass(
    module = "fpm_rs._core",
    name = "MultiWavelengthGradientDescent",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyMultiWavelengthGradientDescent {
    inner: MultiWavelengthGradientDescent,
}

#[pymethods]
impl PyMultiWavelengthGradientDescent {
    #[new]
    #[pyo3(signature=(*,iterations=100,initialization_iterations=50,amplitude_step=1.0,opd_step=1.0,max_backtracks=30,epsilon=1e-10))]
    fn new(
        iterations: usize,
        initialization_iterations: usize,
        amplitude_step: f64,
        opd_step: f64,
        max_backtracks: usize,
        epsilon: f64,
    ) -> PyResult<Self> {
        let inner = MultiWavelengthGradientDescent {
            iterations,
            initialization_iterations,
            amplitude_step,
            opd_step,
            max_backtracks,
            epsilon,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Fits all scalar detector frames with one OPD and channel-specific amplitudes.
    /// Exactly one reference choice is required. Optional initial_opd_m and
    /// initial_amplitudes must be supplied together as copied finite C-contiguous
    /// float64 arrays shaped (rows, columns) and (channels, rows, columns).
    /// Explicit starts bypass AP/unwrapping and use unwrapper.opd_range_m as bounds;
    /// otherwise automatic unwrapping must produce an entirely valid OPD map.
    /// Region references fix selected OPDs; offsets retain the initial spatial mean.
    /// Returns copied scientific arrays via owned result getters; releases the GIL.
    /// Invalid layouts/shapes/values raise FpmError; conflicting choices raise ValueError.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(*,problem,unwrapper,phase_offsets_rad=None,reference_mask=None,reference_opd_m=0.0,initial_opd_m=None,initial_amplitudes=None))]
    fn run(
        &self,
        py: Python<'_>,
        problem: PyRef<'_, PySpectralReconstructionProblem>,
        unwrapper: PyRef<'_, PySyntheticWavelengthUnwrapper>,
        phase_offsets_rad: Option<Vec<f64>>,
        reference_mask: Option<PyReadonlyArray2<'_, u8>>,
        reference_opd_m: f64,
        initial_opd_m: Option<PyReadonlyArray2<'_, f64>>,
        initial_amplitudes: Option<PyReadonlyArray3<'_, f64>>,
    ) -> PyResult<PyMultiWavelengthSolverResult> {
        let reference =
            extract_phase_reference(phase_offsets_rad, reference_mask, reference_opd_m)?;
        let initial = match (initial_opd_m, initial_amplitudes) {
            (Some(opd), Some(amplitudes)) => Some((
                core_array2(&opd).map_err(to_py_err)?,
                core_array3(&amplitudes).map_err(to_py_err)?,
            )),
            (None, None) => None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "initial_opd_m and initial_amplitudes must be supplied together",
                ));
            }
        };
        let algorithm = self.inner.clone();
        let unwrapper = unwrapper.inner.clone();
        let measurements = problem.measurements.clone();
        let model = problem.model.clone();
        let result = py
            .detach(move || {
                let problem = SpectralReconstructionProblem::new(&*measurements, (*model).clone())?;
                match initial {
                    Some((opd, amplitudes)) => algorithm.run_from_opd(
                        &problem,
                        unwrapper.opd_range_m,
                        &reference,
                        opd,
                        amplitudes
                            .axis_iter(Axis(0))
                            .map(|a| a.to_owned())
                            .collect(),
                    ),
                    None => algorithm.run(&problem, &unwrapper, &reference),
                }
            })
            .map_err(to_py_err)?;
        Ok(PyMultiWavelengthSolverResult {
            spectral: Arc::new(result.spectral),
            opd_m: Arc::new(result.opd_m),
            initialization_opd: result.initialization_opd.map(Arc::new),
            initialization_trace: result.initialization_trace.map(|t| {
                t.iterations
                    .iter()
                    .map(|r| (r.iteration, r.objective, r.elapsed_seconds))
                    .collect()
            }),
        })
    }
}

/// Jointly fitted OPD map, derived channel fields, and optional initialization records.
/// spectral.trace contains iteration zero's constrained initial loss and subsequent
/// post-update full-data losses. NumPy getters return independent writable copies.
/// initialization_opd and initialization_trace are None for explicit OPD starts.
#[pyclass(
    module = "fpm_rs._core",
    name = "MultiWavelengthSolverResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyMultiWavelengthSolverResult {
    spectral: Arc<fpm_rs::reconstruction::SpectralReconstructionResult>,
    opd_m: Arc<numpy::ndarray::Array2<f64>>,
    initialization_opd: Option<Arc<fpm_rs::reconstruction::OpticalPathDifferenceResult>>,
    initialization_trace: Option<Vec<(usize, f64, f64)>>,
}

#[pymethods]
impl PyMultiWavelengthSolverResult {
    #[getter]
    fn completed_iterations(&self) -> usize {
        self.spectral.runtime.completed_iterations
    }
    #[getter]
    fn stopped_early(&self) -> bool {
        self.spectral.runtime.stopped_early
    }
    #[getter]
    fn opd_m(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        array2_to_py(py, (*self.opd_m).clone())
    }
    #[getter]
    fn spectral(&self) -> PySpectralReconstructionResult {
        PySpectralReconstructionResult {
            inner: self.spectral.clone(),
        }
    }
    #[getter]
    fn initialization_opd(&self) -> Option<PyOpticalPathDifferenceResult> {
        self.initialization_opd
            .as_ref()
            .map(|v| PyOpticalPathDifferenceResult { inner: v.clone() })
    }
    #[getter]
    fn initialization_trace(&self) -> Option<Vec<(usize, f64, f64)>> {
        self.initialization_trace.clone()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMultiWavelengthGradientDescent>()?;
    module.add_class::<PyMultiWavelengthSolverResult>()?;
    Ok(())
}
