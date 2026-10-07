//! Python boundary for narrowband spectral compilation and object reconstruction.

use std::sync::Arc;

use fpm_rs::{
    Complex64,
    algorithms::SpectralAlternatingProjection,
    experiment::{
        SpectralAcquisitionPlan, SpectralChannel, SpectralContribution, SpectralFrame,
        SpectralGeometry,
    },
    measurements::MeasurementStack,
    model::{ObjectCoupling, ReconstructionShape, SpectralImagePlaneModel},
    reconstruction::{
        OpticalPathDifferenceResult, PhaseReference, SpectralFrameSchedule,
        SpectralReconstructionProblem, SpectralReconstructionResult, SpectralRunner,
        SyntheticWavelengthUnwrapper,
    },
};
use numpy::{
    PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3,
    ndarray::{Array2, Axis},
};
use pyo3::prelude::*;

use crate::{
    arrays::{array2_to_py, core_array2, core_array3, vec3_to_py},
    config::{PyAcquisitionPlan, PyOptics, PySourceCalibration, extract_source_geometry},
    errors::to_py_err,
    measurements::extract_measurements,
    model::{PyImagePlaneModel, PyReconstructionShape},
};

/// Fixed narrowband optics, source powers, and local acquisition rows.
///
/// Inputs are copied. Compilation validates a stable unique channel ID, distinct
/// positive vacuum wavelength in metres, and local acquisition coverage.
/// Physical geometry is supplied separately through SpectralGeometry.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralChannel",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralChannel {
    inner: SpectralChannel,
}

#[pymethods]
impl PySpectralChannel {
    #[new]
    #[pyo3(signature = (*, channel_id, optics, calibration, acquisition))]
    fn new(
        channel_id: String,
        optics: PyRef<'_, PyOptics>,
        calibration: PyRef<'_, PySourceCalibration>,
        acquisition: PyRef<'_, PyAcquisitionPlan>,
    ) -> Self {
        Self {
            inner: SpectralChannel {
                channel_id,
                optics: optics.inner.clone(),
                calibration: calibration.inner.clone(),
                acquisition: acquisition.inner.clone(),
            },
        }
    }
    #[getter]
    fn channel_id(&self) -> String {
        self.inner.channel_id.clone()
    }
    #[getter]
    fn wavelength_vacuum_m(&self) -> f64 {
        self.inner.optics.wavelength_vacuum_m
    }
}

/// Explicit shared physical or per-channel geometry choice.
/// Shared physical geometry is resolved separately with every channel's optics.
/// Shared wavelength-specific direct k-vectors are rejected during compilation.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralGeometry",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralGeometry {
    inner: SpectralGeometry,
}

#[pymethods]
impl PySpectralGeometry {
    #[staticmethod]
    #[pyo3(signature = (*, geometry))]
    fn shared(geometry: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: SpectralGeometry::Shared(extract_source_geometry(geometry)?),
        })
    }
    #[staticmethod]
    #[pyo3(signature = (*, geometries))]
    fn per_channel(geometries: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: SpectralGeometry::PerChannel(
                geometries
                    .iter()
                    .map(extract_source_geometry)
                    .collect::<PyResult<_>>()?,
            ),
        })
    }
}

/// One physical detector exposure's sparse intensity composition.
/// Contributions are (channel_index, local_frame_index, spectral_weight) triples.
/// Detector gain and uniform nonnegative background apply after the spectral sum.
/// The plan constructor validates values and canonicalizes duplicate/zero weights.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralFrame",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralFrame {
    inner: SpectralFrame,
}

#[pymethods]
impl PySpectralFrame {
    #[new]
    #[pyo3(signature = (*, contributions, gain=1.0, background=0.0))]
    fn new(contributions: Vec<(usize, usize, f64)>, gain: f64, background: f64) -> Self {
        Self {
            inner: SpectralFrame {
                contributions: contributions
                    .into_iter()
                    .map(
                        |(channel, local_frame, spectral_weight)| SpectralContribution {
                            channel,
                            local_frame,
                            spectral_weight,
                        },
                    )
                    .collect(),
                gain,
                background,
            },
        }
    }
    #[getter]
    fn contributions(&self) -> Vec<(usize, usize, f64)> {
        self.inner
            .contributions
            .iter()
            .map(|c| (c.channel, c.local_frame, c.spectral_weight))
            .collect()
    }
    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain
    }
    #[getter]
    fn background(&self) -> f64 {
        self.inner.background
    }
}

/// Canonical physical detector rows with explicit separate/multiplexed semantics.
/// Duplicate pairs are summed, zero weights removed, and pair order canonicalized.
/// Compilation validates references and requires coverage of every local frame.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralAcquisitionPlan",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralAcquisitionPlan {
    inner: SpectralAcquisitionPlan,
}

#[pymethods]
impl PySpectralAcquisitionPlan {
    #[staticmethod]
    #[pyo3(signature = (*, frame_counts))]
    fn separate(frame_counts: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: SpectralAcquisitionPlan::separate(&frame_counts).map_err(to_py_err)?,
        })
    }
    #[staticmethod]
    #[pyo3(signature = (*, frames))]
    fn multiplexed(py: Python<'_>, frames: Vec<Py<PySpectralFrame>>) -> PyResult<Self> {
        Ok(Self {
            inner: SpectralAcquisitionPlan::multiplexed(
                frames
                    .iter()
                    .map(|frame| frame.borrow(py).inner.clone())
                    .collect(),
            )
            .map_err(to_py_err)?,
        })
    }
    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }
    #[getter]
    fn frames(&self) -> Vec<PySpectralFrame> {
        self.inner
            .frames()
            .iter()
            .map(|frame| PySpectralFrame {
                inner: frame.clone(),
            })
            .collect()
    }
}

fn parse_coupling(value: &str) -> PyResult<ObjectCoupling> {
    match value {
        "independent" => Ok(ObjectCoupling::Independent),
        "shared_complex" => Ok(ObjectCoupling::SharedComplex),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "object_coupling must be 'independent' or 'shared_complex'",
        )),
    }
}
fn coupling_name(value: ObjectCoupling) -> &'static str {
    match value {
        ObjectCoupling::Independent => "independent",
        ObjectCoupling::SharedComplex => "shared_complex",
    }
}

/// Immutable wavelength-specific channel kernels on one common reconstruction grid.
/// Created by compile_spectral_model; contains no physical source geometry.
/// Objects are independent by default; shared_complex explicitly shares one field.
/// Pupils remain wavelength-specific in either object-coupling mode.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralImagePlaneModel",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralImagePlaneModel {
    inner: Arc<SpectralImagePlaneModel>,
}

#[pymethods]
impl PySpectralImagePlaneModel {
    #[getter]
    fn channel_ids(&self) -> Vec<String> {
        self.inner
            .channels()
            .iter()
            .map(|channel| channel.channel_id.clone())
            .collect()
    }
    #[getter]
    fn wavelengths_vacuum_m(&self) -> Vec<f64> {
        self.inner
            .channels()
            .iter()
            .map(|channel| channel.model.sampling().wavelength.unwrap())
            .collect()
    }
    #[getter]
    fn image_shape(&self) -> (usize, usize) {
        self.inner.image_shape()
    }
    #[getter]
    fn reconstruction_shape(&self) -> (usize, usize) {
        self.inner.reconstruction_shape()
    }
    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }
    #[getter]
    fn object_coupling(&self) -> &'static str {
        coupling_name(self.inner.object_coupling())
    }
    #[getter]
    fn channel_models(&self) -> Vec<PyImagePlaneModel> {
        self.inner
            .channels()
            .iter()
            .map(|channel| PyImagePlaneModel {
                inner: Arc::new(channel.model.clone()),
            })
            .collect()
    }
    #[getter]
    fn acquisition(&self) -> PySpectralAcquisitionPlan {
        PySpectralAcquisitionPlan {
            inner: self.inner.acquisition().clone(),
        }
    }

    /// Evaluates scalar detector intensities from centered common-grid spectra.
    /// Copies finite C-contiguous complex128 (objects, height, width) spectra,
    /// normalized by the core's 1/(height * width) forward FFT convention.
    /// Independent coupling needs one spectrum per channel; shared coupling needs one.
    /// Returns a writable float64 (frames, image_height, image_width) copy.
    /// Releases the GIL and raises FpmError on invalid input or numerical failure.
    #[pyo3(signature = (*, object_spectra))]
    fn forward_intensities(
        &self,
        py: Python<'_>,
        object_spectra: PyReadonlyArray3<'_, Complex64>,
    ) -> PyResult<Py<PyArray3<f64>>> {
        let spectra = core_array3(&object_spectra).map_err(to_py_err)?;
        let model = self.inner.clone();
        let shape = (
            model.frame_count(),
            model.image_shape().0,
            model.image_shape().1,
        );
        let values = py
            .detach(move || {
                let views: Vec<_> = spectra.axis_iter(Axis(0)).collect();
                let mut values = Vec::new();
                for frame in 0..model.frame_count() {
                    values.extend(model.forward_intensity(&views, frame)?);
                }
                Ok::<_, fpm_rs::Error>(values)
            })
            .map_err(to_py_err)?;
        vec3_to_py(py, shape, values)
    }
}

/// Compiles narrowband channels using wavelength-specific canonical kernels.
/// Requires matching detector sampling and rejects shared wavelength-specific k-vectors.
#[pyfunction]
#[pyo3(signature = (*, channels, geometry, acquisition, image_shape, reconstruction_shape=PyReconstructionShape(ReconstructionShape::Smooth), object_coupling="independent"), text_signature = "(*, channels, geometry, acquisition, image_shape, reconstruction_shape='smooth', object_coupling='independent')")]
fn compile_spectral_model(
    py: Python<'_>,
    channels: Vec<Py<PySpectralChannel>>,
    geometry: PyRef<'_, PySpectralGeometry>,
    acquisition: PyRef<'_, PySpectralAcquisitionPlan>,
    image_shape: (usize, usize),
    reconstruction_shape: PyReconstructionShape,
    object_coupling: &str,
) -> PyResult<PySpectralImagePlaneModel> {
    let channels: Vec<_> = channels
        .iter()
        .map(|channel| channel.borrow(py).inner.clone())
        .collect();
    let geometry = geometry.inner.clone();
    let acquisition = acquisition.inner.clone();
    let coupling = parse_coupling(object_coupling)?;
    let inner = py
        .detach(move || {
            SpectralImagePlaneModel::from_experiment(
                &channels,
                &geometry,
                acquisition,
                image_shape,
                reconstruction_shape.0,
                coupling,
            )
        })
        .map_err(to_py_err)?;
    Ok(PySpectralImagePlaneModel {
        inner: Arc::new(inner),
    })
}

/// Scalar detector measurements paired with an immutable spectral model.
/// Copies float64 (frames, image_height, image_width) NumPy inputs into Rust storage.
/// Optional masks use matching uint8 shape; zero excludes a pixel.
/// Every local row must participate in a positive-weight, nonempty detector frame.
/// Validation releases the GIL after conversion.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralReconstructionProblem",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralReconstructionProblem {
    pub(crate) measurements: Arc<MeasurementStack>,
    pub(crate) model: Arc<SpectralImagePlaneModel>,
}

#[pymethods]
impl PySpectralReconstructionProblem {
    #[new]
    #[pyo3(signature = (*, measurements, model, frame_weights=None, masks=None))]
    fn new(
        py: Python<'_>,
        measurements: &Bound<'_, PyAny>,
        model: PyRef<'_, PySpectralImagePlaneModel>,
        frame_weights: Option<Vec<f64>>,
        masks: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let measurements = extract_measurements(measurements, frame_weights, masks)?;
        let model = model.inner.clone();
        let validate_measurements = measurements.clone();
        let validate_model = model.clone();
        py.detach(move || {
            SpectralReconstructionProblem::new(&*validate_measurements, (*validate_model).clone())
                .map(|_| ())
        })
        .map_err(to_py_err)?;
        Ok(Self {
            measurements,
            model,
        })
    }
    #[getter]
    fn frame_count(&self) -> usize {
        self.model.frame_count()
    }
    #[getter]
    fn image_shape(&self) -> (usize, usize) {
        self.model.image_shape()
    }
    #[getter]
    fn reconstruction_shape(&self) -> (usize, usize) {
        self.model.reconstruction_shape()
    }
}

/// Channel-ordered fields, fixed pupils, metadata, and a global reconstruction trace.
/// Object arrays return writable copies shaped (channels, height, width).
/// Shared coupling repeats the identical final object for every channel.
/// Unconstrained AP has independent channel pistons; shared-complex coupling has one.
/// Joint OPD descent imposes a further referenced phase constraint; object_coupling
/// describes the compiled model's field storage/reuse rule.
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralReconstructionResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralReconstructionResult {
    pub(crate) inner: Arc<SpectralReconstructionResult>,
}

impl PySpectralReconstructionResult {
    fn stack<T: numpy::Element + Copy>(
        &self,
        py: Python<'_>,
        array: impl Fn(&fpm_rs::reconstruction::SpectralChannelResult) -> &Array2<T>,
    ) -> PyResult<Py<PyArray3<T>>> {
        let first = array(&self.inner.channels[0]);
        vec3_to_py(
            py,
            (self.inner.channels.len(), first.dim().0, first.dim().1),
            self.inner
                .channels
                .iter()
                .flat_map(|channel| array(channel).iter().copied())
                .collect(),
        )
    }
}

#[pymethods]
impl PySpectralReconstructionResult {
    /// Mixes referenced independent channel phases into a nondispersive OPD map.
    /// Exactly one of phase_offsets_rad and reference_mask is required.
    /// reference_mask and mask are copied C-contiguous uint8 common-grid arrays.
    /// reference_opd_m is the constant known OPD of the reference region in metres.
    /// Releases the GIL; rejects shared-complex coupling and malformed inputs.
    #[pyo3(signature = (*, unwrapper, phase_offsets_rad=None, reference_mask=None, reference_opd_m=0.0, mask=None))]
    fn unwrap_opd(
        &self,
        py: Python<'_>,
        unwrapper: PyRef<'_, PySyntheticWavelengthUnwrapper>,
        phase_offsets_rad: Option<Vec<f64>>,
        reference_mask: Option<PyReadonlyArray2<'_, u8>>,
        reference_opd_m: f64,
        mask: Option<PyReadonlyArray2<'_, u8>>,
    ) -> PyResult<PyOpticalPathDifferenceResult> {
        let reference =
            extract_phase_reference(phase_offsets_rad, reference_mask, reference_opd_m)?;
        let mask = mask
            .map(|v| core_array2(&v))
            .transpose()
            .map_err(to_py_err)?;
        let unwrapper = unwrapper.inner.clone();
        let spectral = self.inner.clone();
        let inner = py
            .detach(move || {
                spectral.unwrap_opd(&unwrapper, &reference, mask.as_ref().map(|m| m.view()))
            })
            .map_err(to_py_err)?;
        Ok(PyOpticalPathDifferenceResult {
            inner: Arc::new(inner),
        })
    }

    #[getter]
    fn channel_ids(&self) -> Vec<String> {
        self.inner
            .channels
            .iter()
            .map(|channel| channel.channel_id.clone())
            .collect()
    }
    #[getter]
    fn wavelengths_vacuum_m(&self) -> Vec<f64> {
        self.inner
            .channels
            .iter()
            .map(|channel| channel.wavelength_vacuum_m)
            .collect()
    }
    #[getter]
    fn object_coupling(&self) -> &'static str {
        coupling_name(self.inner.object_coupling)
    }
    #[getter]
    fn object(&self, py: Python<'_>) -> PyResult<Py<PyArray3<Complex64>>> {
        self.stack(py, |channel| &channel.object)
    }
    #[getter]
    fn object_spectrum(&self, py: Python<'_>) -> PyResult<Py<PyArray3<Complex64>>> {
        self.stack(py, |channel| &channel.object_spectrum)
    }
    #[getter]
    fn amplitude(&self, py: Python<'_>) -> PyResult<Py<PyArray3<f64>>> {
        self.stack(py, |channel| &channel.amplitude)
    }
    #[getter]
    fn phase(&self, py: Python<'_>) -> PyResult<Py<PyArray3<f64>>> {
        self.stack(py, |channel| &channel.phase)
    }
    #[getter]
    fn pupils(&self, py: Python<'_>) -> PyResult<Py<PyArray3<Complex64>>> {
        let shape = self.inner.channels[0].pupil.shape();
        vec3_to_py(
            py,
            (self.inner.channels.len(), shape.0, shape.1),
            self.inner
                .channels
                .iter()
                .flat_map(|channel| channel.pupil.values().iter().copied().collect::<Vec<_>>())
                .collect(),
        )
    }
    #[getter]
    fn trace(&self) -> Vec<(usize, f64, f64)> {
        self.inner
            .trace
            .iterations
            .iter()
            .map(|row| (row.iteration, row.objective, row.elapsed_seconds))
            .collect()
    }
    #[getter]
    fn completed_iterations(&self) -> usize {
        self.inner.runtime.completed_iterations
    }
}

/// Object-only narrowband spectral amplitude projection with fixed channel pupils.
/// Uses one amplitude ratio per physical exposure and releases the GIL while running.
///
/// References: S. Dong, R. Shiradkar, P. Nanda, and G. Zheng,
/// [“Spectral multiplexing and coherent-state decomposition in Fourier ptychographic imaging”](https://doi.org/10.1364/BOE.5.001757),
/// Biomedical Optics Express 5(6), 1757–1767 (2014).
#[pyclass(
    module = "fpm_rs._core",
    name = "SpectralAlternatingProjection",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySpectralAlternatingProjection {
    inner: SpectralAlternatingProjection,
}

#[pymethods]
impl PySpectralAlternatingProjection {
    /// Reconstructs wavelength fields and then unwraps a referenced common OPD map.
    /// Performs run followed by SpectralReconstructionResult.unwrap_opd; returns both outputs.
    /// Assumes nondispersive OPD and registered fields with matched spatial resolution.
    /// Requires independent object coupling and exactly one phase reference choice.
    /// NumPy inputs are copied; computation releases the GIL and validation raises FpmError.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, problem, unwrapper, phase_offsets_rad=None, reference_mask=None, reference_opd_m=0.0, mask=None, initial_objects=None, frame_order=None, seed=None))]
    fn run_opd(
        &self,
        py: Python<'_>,
        problem: PyRef<'_, PySpectralReconstructionProblem>,
        unwrapper: PyRef<'_, PySyntheticWavelengthUnwrapper>,
        phase_offsets_rad: Option<Vec<f64>>,
        reference_mask: Option<PyReadonlyArray2<'_, u8>>,
        reference_opd_m: f64,
        mask: Option<PyReadonlyArray2<'_, u8>>,
        initial_objects: Option<PyReadonlyArray3<'_, Complex64>>,
        frame_order: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> PyResult<PyMultiWavelengthReconstructionResult> {
        let reference =
            extract_phase_reference(phase_offsets_rad, reference_mask, reference_opd_m)?;
        let mask = mask
            .map(|v| core_array2(&v))
            .transpose()
            .map_err(to_py_err)?;
        if problem.model.object_coupling() != ObjectCoupling::Independent {
            return Err(to_py_err(fpm_rs::Error::InvalidParameter {
                name: "object_coupling",
                reason: "OPD phase mixing requires independent wavelength fields".into(),
            }));
        }
        let spectral = self
            .run(py, problem, initial_objects, frame_order, seed)?
            .inner;
        let unwrapper = unwrapper.inner.clone();
        let inner = py
            .detach(move || {
                let opd =
                    spectral.unwrap_opd(&unwrapper, &reference, mask.as_ref().map(|m| m.view()))?;
                Ok::<_, fpm_rs::Error>(PyMultiWavelengthReconstructionResult {
                    spectral,
                    opd: Arc::new(opd),
                })
            })
            .map_err(to_py_err)?;
        Ok(inner)
    }

    #[new]
    #[pyo3(signature = (*, iterations=50, object_step=1.0, batch_size=1, epsilon=1e-10))]
    fn new(iterations: usize, object_step: f64, batch_size: usize, epsilon: f64) -> Self {
        Self {
            inner: SpectralAlternatingProjection {
                iterations,
                object_step,
                batch_size,
                epsilon,
                ..SpectralAlternatingProjection::default()
            },
        }
    }

    /// Runs object reconstruction with fixed channel pupils and detector calibration.
    /// Optional finite C-contiguous complex128 initial_objects are copied and shaped
    /// (objects, height, width), with one plane per independent channel or one shared.
    /// Omission initializes from weighted measured amplitudes; mixed signal is split
    /// per unit spectral/local-gain weight as a heuristic, not spectral demixing.
    /// frame_order is a global detector permutation; seed selects a deterministic
    /// per-pass shuffle, and the two options are mutually exclusive.
    /// Returns owned channel records; releases the GIL and raises FpmError on failure.
    #[pyo3(signature = (*, problem, initial_objects=None, frame_order=None, seed=None))]
    fn run(
        &self,
        py: Python<'_>,
        problem: PyRef<'_, PySpectralReconstructionProblem>,
        initial_objects: Option<PyReadonlyArray3<'_, Complex64>>,
        frame_order: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> PyResult<PySpectralReconstructionResult> {
        if frame_order.is_some() && seed.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frame_order and seed are mutually exclusive",
            ));
        }
        let mut runner = SpectralRunner::new(self.inner.clone());
        if let Some(objects) = initial_objects {
            let objects = core_array3(&objects).map_err(to_py_err)?;
            runner = runner.with_initial_objects(
                objects
                    .axis_iter(Axis(0))
                    .map(|object| object.to_owned())
                    .collect(),
            );
        }
        let schedule = match (frame_order, seed) {
            (Some(order), _) => SpectralFrameSchedule::Explicit(order),
            (_, Some(seed)) => SpectralFrameSchedule::RandomShuffle { seed },
            _ => SpectralFrameSchedule::Sequential,
        };
        runner = runner.with_schedule(schedule);
        let measurements = problem.measurements.clone();
        let model = problem.model.clone();
        let inner = py
            .detach(move || {
                runner.run(&SpectralReconstructionProblem::new(
                    &*measurements,
                    (*model).clone(),
                )?)
            })
            .map_err(to_py_err)?;
        Ok(PySpectralReconstructionResult {
            inner: Arc::new(inner),
        })
    }
}

pub(crate) fn extract_phase_reference(
    offsets: Option<Vec<f64>>,
    region: Option<PyReadonlyArray2<'_, u8>>,
    opd_m: f64,
) -> PyResult<PhaseReference> {
    match (offsets, region) {
        (Some(values), None) if opd_m == 0.0 => Ok(PhaseReference::Offsets(values)),
        (None, Some(mask)) => Ok(PhaseReference::Region {
            mask: core_array2(&mask).map_err(to_py_err)?,
            opd_m,
        }),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "supply exactly one of phase_offsets_rad and reference_mask; reference_opd_m applies only to a reference mask",
        )),
    }
}

/// Referenced, nondispersive OPD recovery from registered complex wavelength fields.
/// Starts from the longest phase-difference synthetic period in an explicit half-open
/// OPD interval, refines orders through shorter beat/original periods, and fits all
/// original phases with equal phase weights. Inputs are copied and computation
/// releases the GIL. Noise can cause incorrect fringe orders; inspect validity and residuals.
/// This post-reconstruction method assumes matched spatial registration/resolution.
///
/// References
/// ----------
/// S. K. Mirsky and N. T. Shaked,
/// [“Six-pack holography for dynamic profiling of thick and extended objects by simultaneous three-wavelength phase unwrapping with doubled field of view”](https://doi.org/10.1038/s41598-023-45237-6),
/// Scientific Reports 13, article 19293 (2023). Applies their phase-difference
/// hierarchy to reconstructed FPM fields, using an interval instead of spatial
/// unwrapping of the longest beat phase; omits phase-sum wavelengths and their optics.
#[pyclass(
    module = "fpm_rs._core",
    name = "SyntheticWavelengthUnwrapper",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySyntheticWavelengthUnwrapper {
    pub(crate) inner: SyntheticWavelengthUnwrapper,
}

#[pymethods]
impl PySyntheticWavelengthUnwrapper {
    #[new]
    #[pyo3(signature = (*, opd_range_m, minimum_amplitude=0.0, max_phase_residual_rad=None))]
    fn new(
        opd_range_m: (f64, f64),
        minimum_amplitude: f64,
        max_phase_residual_rad: Option<f64>,
    ) -> PyResult<Self> {
        let inner = SyntheticWavelengthUnwrapper {
            opd_range_m,
            minimum_amplitude,
            max_phase_residual_rad,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }
    #[getter]
    fn opd_range_m(&self) -> (f64, f64) {
        self.inner.opd_range_m
    }
    #[getter]
    fn minimum_amplitude(&self) -> f64 {
        self.inner.minimum_amplitude
    }
    #[getter]
    fn max_phase_residual_rad(&self) -> Option<f64> {
        self.inner.max_phase_residual_rad
    }

    /// Unwraps copied complex128 C-contiguous (channels, rows, columns) fields.
    /// wavelengths_vacuum_m gives distinct positive vacuum wavelengths in metres.
    /// Exactly one of phase_offsets_rad or a uint8 reference_mask is required;
    /// reference_opd_m specifies the region's known constant OPD in metres.
    /// Optional uint8 mask excludes pixels; any zero/below-floor channel amplitude
    /// invalidates that pixel. Returns owned OPD/diagnostics, releases the GIL,
    /// and raises FpmError for invalid scientific values or nonstandard layouts.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, fields, wavelengths_vacuum_m, phase_offsets_rad=None, reference_mask=None, reference_opd_m=0.0, mask=None))]
    fn unwrap_fields(
        &self,
        py: Python<'_>,
        fields: PyReadonlyArray3<'_, Complex64>,
        wavelengths_vacuum_m: Vec<f64>,
        phase_offsets_rad: Option<Vec<f64>>,
        reference_mask: Option<PyReadonlyArray2<'_, u8>>,
        reference_opd_m: f64,
        mask: Option<PyReadonlyArray2<'_, u8>>,
    ) -> PyResult<PyOpticalPathDifferenceResult> {
        let fields = core_array3(&fields).map_err(to_py_err)?;
        let reference =
            extract_phase_reference(phase_offsets_rad, reference_mask, reference_opd_m)?;
        let mask = mask
            .map(|m| core_array2(&m))
            .transpose()
            .map_err(to_py_err)?;
        let unwrapper = self.inner.clone();
        let inner = py
            .detach(move || {
                let views: Vec<_> = fields.axis_iter(Axis(0)).collect();
                unwrapper.unwrap_fields(
                    &views,
                    &wavelengths_vacuum_m,
                    &reference,
                    mask.as_ref().map(|m| m.view()),
                )
            })
            .map_err(to_py_err)?;
        Ok(PyOpticalPathDifferenceResult {
            inner: Arc::new(inner),
        })
    }
}

/// Nondispersive OPD in metres and phase-mixing diagnostics on the common grid.
/// NumPy getters return independent writable copies. Invalid pixels have NaN OPD
/// and residuals, zero fringe orders, and valid_mask=0. Residuals cannot certify
/// fringe uniqueness when noise is large. Channel metadata preserves input order.
#[pyclass(
    module = "fpm_rs._core",
    name = "OpticalPathDifferenceResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyOpticalPathDifferenceResult {
    pub(crate) inner: Arc<OpticalPathDifferenceResult>,
}

#[pymethods]
impl PyOpticalPathDifferenceResult {
    #[getter]
    fn opd_m(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        array2_to_py(py, self.inner.opd_m.clone())
    }
    #[getter]
    fn valid_mask(&self, py: Python<'_>) -> PyResult<Py<PyArray2<u8>>> {
        array2_to_py(py, self.inner.valid_mask.clone())
    }
    #[getter]
    fn phase_residual_rad(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        array2_to_py(py, self.inner.phase_residual_rad.clone())
    }
    #[getter]
    fn phase_offsets_rad(&self) -> Vec<f64> {
        self.inner.phase_offsets_rad.clone()
    }
    #[getter]
    fn wavelengths_vacuum_m(&self) -> Vec<f64> {
        self.inner.wavelengths_vacuum_m.clone()
    }
    #[getter]
    fn wavelength_ladder_m(&self) -> Vec<f64> {
        self.inner.wavelength_ladder_m.clone()
    }
    #[getter]
    fn fringe_orders(&self, py: Python<'_>) -> PyResult<Py<PyArray3<i64>>> {
        let shape = self.inner.opd_m.dim();
        vec3_to_py(
            py,
            (self.inner.fringe_orders.len(), shape.0, shape.1),
            self.inner
                .fringe_orders
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect(),
        )
    }
}

/// Wavelength-field reconstruction and the resulting referenced common OPD map.
/// The spectral property retains channel fields and the intensity trace; opd
/// retains the phase-mixed map and diagnostics. Returned wrappers share immutable
/// Rust ownership, and their scientific NumPy getters copy data.
#[pyclass(
    module = "fpm_rs._core",
    name = "MultiWavelengthReconstructionResult",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyMultiWavelengthReconstructionResult {
    spectral: Arc<SpectralReconstructionResult>,
    opd: Arc<OpticalPathDifferenceResult>,
}

#[pymethods]
impl PyMultiWavelengthReconstructionResult {
    #[getter]
    fn spectral(&self) -> PySpectralReconstructionResult {
        PySpectralReconstructionResult {
            inner: self.spectral.clone(),
        }
    }
    #[getter]
    fn opd(&self) -> PyOpticalPathDifferenceResult {
        PyOpticalPathDifferenceResult {
            inner: self.opd.clone(),
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySyntheticWavelengthUnwrapper>()?;
    module.add_class::<PyOpticalPathDifferenceResult>()?;
    module.add_class::<PyMultiWavelengthReconstructionResult>()?;
    module.add_class::<PySpectralChannel>()?;
    module.add_class::<PySpectralGeometry>()?;
    module.add_class::<PySpectralFrame>()?;
    module.add_class::<PySpectralAcquisitionPlan>()?;
    module.add_class::<PySpectralImagePlaneModel>()?;
    module.add_class::<PySpectralReconstructionProblem>()?;
    module.add_class::<PySpectralReconstructionResult>()?;
    module.add_class::<PySpectralAlternatingProjection>()?;
    module.add_function(wrap_pyfunction!(compile_spectral_model, module)?)?;
    Ok(())
}
