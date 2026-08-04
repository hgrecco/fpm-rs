use fpm_rs::{
    experiment::{
        AcquisitionPlan, ArrayPose, DirectionList, Illumination, IlluminationFrame, KVector,
        KVectorList, Optics, PlanarLedArray, PupilAberration, ResolvedFrame, ResolvedIllumination,
        ResolvedSources, RotatingLedArc, SourceCalibration, SourceContribution, SourceGeometry,
        SourcePositionList, SphericalLedArm, SphericalLedArray,
    },
    simulation::{CameraModel, IlluminationAcquisitionErrors},
};
use numpy::{
    PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods,
    ndarray::{Array1, Array2},
};
use pyo3::prelude::*;

use crate::{arrays::copy_array2, errors::to_py_err};

#[pyclass(
    module = "fpm_rs._core",
    name = "PupilAberration",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPupilAberration {
    pub(crate) inner: PupilAberration,
}

#[pymethods]
impl PyPupilAberration {
    #[new]
    #[pyo3(signature = (*, astigmatism=0.0, coma=0.0, spherical=0.0, edge_apodization=0.0))]
    fn new(astigmatism: f64, coma: f64, spherical: f64, edge_apodization: f64) -> PyResult<Self> {
        let inner = PupilAberration {
            astigmatism,
            coma,
            spherical,
            edge_apodization,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn astigmatism(&self) -> f64 {
        self.inner.astigmatism
    }

    #[getter]
    fn coma(&self) -> f64 {
        self.inner.coma
    }

    #[getter]
    fn spherical(&self) -> f64 {
        self.inner.spherical
    }

    #[getter]
    fn edge_apodization(&self) -> f64 {
        self.inner.edge_apodization
    }
}

#[pyclass(module = "fpm_rs._core", name = "Optics", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyOptics {
    pub(crate) inner: Optics,
}

#[pymethods]
impl PyOptics {
    #[new]
    #[pyo3(signature = (wavelength_vacuum_m, objective_na, magnification, camera_pixel_size, *, illumination_refractive_index=1.0, objective_medium_refractive_index=1.0, defocus_distance=None, pupil_aberration=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        wavelength_vacuum_m: f64,
        objective_na: f64,
        magnification: f64,
        camera_pixel_size: f64,
        illumination_refractive_index: f64,
        objective_medium_refractive_index: f64,
        defocus_distance: Option<f64>,
        pupil_aberration: Option<PyRef<'_, PyPupilAberration>>,
    ) -> PyResult<Self> {
        let inner = Optics {
            wavelength_vacuum_m,
            objective_na,
            magnification,
            camera_pixel_size,
            illumination_refractive_index,
            objective_medium_refractive_index,
            defocus_distance,
            pupil_aberration: pupil_aberration.map(|value| value.inner.clone()),
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn wavelength_vacuum_m(&self) -> f64 {
        self.inner.wavelength_vacuum_m
    }

    #[getter]
    fn objective_na(&self) -> f64 {
        self.inner.objective_na
    }

    #[getter]
    fn magnification(&self) -> f64 {
        self.inner.magnification
    }

    #[getter]
    fn camera_pixel_size(&self) -> f64 {
        self.inner.camera_pixel_size
    }

    #[getter]
    fn illumination_refractive_index(&self) -> f64 {
        self.inner.illumination_refractive_index
    }

    #[getter]
    fn objective_medium_refractive_index(&self) -> f64 {
        self.inner.objective_medium_refractive_index
    }

    #[getter]
    fn defocus_distance(&self) -> Option<f64> {
        self.inner.defocus_distance
    }

    #[getter]
    fn object_pixel_size(&self) -> f64 {
        self.inner.object_pixel_size()
    }
}

#[pyclass(module = "fpm_rs._core", name = "ArrayPose", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyArrayPose {
    inner: ArrayPose,
}

#[pymethods]
impl PyArrayPose {
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: ArrayPose::identity(),
        }
    }

    #[staticmethod]
    fn from_translation(translation_m: [f64; 3]) -> PyResult<Self> {
        validated_pose(ArrayPose::from_translation(translation_m))
    }

    #[staticmethod]
    fn from_translation_and_extrinsic_xyz_radians(
        translation_m: [f64; 3],
        rotation_rad: [f64; 3],
    ) -> PyResult<Self> {
        validated_pose(ArrayPose::from_translation_and_extrinsic_xyz_radians(
            translation_m,
            rotation_rad,
        ))
    }

    #[staticmethod]
    fn from_translation_and_extrinsic_xyz_degrees(
        translation_m: [f64; 3],
        rotation_deg: [f64; 3],
    ) -> PyResult<Self> {
        validated_pose(ArrayPose::from_translation_and_extrinsic_xyz_degrees(
            translation_m,
            rotation_deg,
        ))
    }

    #[getter]
    fn translation_m(&self) -> [f64; 3] {
        self.inner.translation_m
    }

    #[getter]
    fn rotation_rad(&self) -> [f64; 3] {
        self.inner.rotation_rad
    }
}

fn validated_pose(inner: ArrayPose) -> PyResult<PyArrayPose> {
    inner.validate().map_err(to_py_err)?;
    Ok(PyArrayPose { inner })
}

#[pyclass(
    module = "fpm_rs._core",
    name = "PlanarLEDArray",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPlanarLedArray {
    inner: PlanarLedArray,
}

#[pymethods]
impl PyPlanarLedArray {
    #[new]
    #[pyo3(signature = (shape, pitch_m, reference_index, pose, *, position_offsets_m=None))]
    fn new(
        shape: (usize, usize),
        pitch_m: &Bound<'_, PyAny>,
        reference_index: (f64, f64),
        pose: PyRef<'_, PyArrayPose>,
        position_offsets_m: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        let pitch_m = parse_pitch(pitch_m)?;
        let offsets = position_offsets_m
            .as_ref()
            .map(|values| copy_xyz(values, "position_offsets_m", true))
            .transpose()?
            .unwrap_or_default();
        let inner = PlanarLedArray::new(shape, pitch_m, reference_index, pose.inner.clone())
            .with_position_offsets_m(offsets);
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn pitch_m(&self) -> (f64, f64) {
        self.inner.pitch_m()
    }

    #[getter]
    fn reference_index(&self) -> (f64, f64) {
        self.inner.reference_index()
    }

    #[getter]
    fn pose(&self) -> PyArrayPose {
        PyArrayPose {
            inner: self.inner.pose().clone(),
        }
    }

    #[getter]
    fn position_offsets_m(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        xyz_to_py(py, self.inner.position_offsets_m())
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    fn source_index(&self, row: usize, column: usize) -> PyResult<usize> {
        self.inner.source_index(row, column).map_err(to_py_err)
    }

    fn source_row_column(&self, index: usize) -> PyResult<(usize, usize)> {
        self.inner.source_row_column(index).map_err(to_py_err)
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedSources> {
        resolved_sources(self.inner.resolve(&optics.inner))
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "SphericalLEDArray",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySphericalLedArray {
    inner: SphericalLedArray,
}

#[pymethods]
impl PySphericalLedArray {
    #[new]
    #[pyo3(signature = (angles, radius, *, center_offset=(0.0, 0.0, 0.0), orientation_degrees=(0.0, 0.0, 0.0), angular_corrections=None))]
    fn new(
        angles: PyReadonlyArray2<'_, f64>,
        radius: f64,
        center_offset: (f64, f64, f64),
        orientation_degrees: (f64, f64, f64),
        angular_corrections: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        let mut inner = SphericalLedArray::new(copy_pairs(&angles, "angles")?, radius)
            .center_offset(center_offset)
            .orientation_deg(orientation_degrees);
        if let Some(values) = angular_corrections {
            inner = inner.angular_corrections(copy_pairs(&values, "angular_corrections")?);
        }
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedSources> {
        resolved_sources(self.inner.resolve(&optics.inner))
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "SphericalLEDArm",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySphericalLedArm {
    inner: SphericalLedArm,
}

#[pymethods]
impl PySphericalLedArm {
    #[new]
    #[pyo3(signature = (commanded_angles, arm_length, *, pivot_offset=(0.0, 0.0, 0.0), orientation_degrees=(0.0, 0.0, 0.0), theta_zero_degrees=0.0, phi_zero_degrees=0.0, theta_scale=1.0, phi_scale=1.0, elevation_axis_tilt_degrees=0.0, theta_backlash_degrees=0.0, phi_backlash_degrees=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        commanded_angles: PyReadonlyArray2<'_, f64>,
        arm_length: f64,
        pivot_offset: (f64, f64, f64),
        orientation_degrees: (f64, f64, f64),
        theta_zero_degrees: f64,
        phi_zero_degrees: f64,
        theta_scale: f64,
        phi_scale: f64,
        elevation_axis_tilt_degrees: f64,
        theta_backlash_degrees: f64,
        phi_backlash_degrees: f64,
    ) -> PyResult<Self> {
        let inner = SphericalLedArm::new(
            copy_pairs(&commanded_angles, "commanded_angles")?,
            arm_length,
        )
        .pivot_offset(pivot_offset)
        .orientation_deg(orientation_degrees)
        .encoder_zero_deg(theta_zero_degrees, phi_zero_degrees)
        .encoder_scale(theta_scale, phi_scale)
        .elevation_axis_tilt_deg(elevation_axis_tilt_degrees)
        .backlash_deg(theta_backlash_degrees, phi_backlash_degrees);
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedSources> {
        resolved_sources(self.inner.resolve(&optics.inner))
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "RotatingLEDArc",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyRotatingLedArc {
    inner: RotatingLedArc,
}

#[pymethods]
impl PyRotatingLedArc {
    #[new]
    #[pyo3(signature = (led_thetas, rotation_angles, radius, *, axis_origin_offset=(0.0, 0.0, 0.0), axis_tilt_degrees=(0.0, 0.0), led_angular_corrections=None, led_radial_offsets=None, rotation_zero_degrees=0.0, rotation_scale=1.0, rotation_backlash_degrees=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        led_thetas: Vec<f64>,
        rotation_angles: Vec<f64>,
        radius: f64,
        axis_origin_offset: (f64, f64, f64),
        axis_tilt_degrees: (f64, f64),
        led_angular_corrections: Option<PyReadonlyArray2<'_, f64>>,
        led_radial_offsets: Option<Vec<f64>>,
        rotation_zero_degrees: f64,
        rotation_scale: f64,
        rotation_backlash_degrees: f64,
    ) -> PyResult<Self> {
        let mut inner = RotatingLedArc::new(led_thetas, rotation_angles, radius)
            .axis_origin_offset(axis_origin_offset)
            .axis_tilt_deg(axis_tilt_degrees.0, axis_tilt_degrees.1)
            .rotation_encoder_zero_deg(rotation_zero_degrees)
            .rotation_encoder_scale(rotation_scale)
            .rotation_backlash_deg(rotation_backlash_degrees);
        if let Some(values) = led_angular_corrections {
            inner = inner.led_angular_corrections(copy_pairs(&values, "led_angular_corrections")?);
        }
        if let Some(values) = led_radial_offsets {
            inner = inner.led_radial_offsets(values);
        }
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn led_count(&self) -> usize {
        self.inner.led_thetas.len()
    }

    #[getter]
    fn rotation_count(&self) -> usize {
        self.inner.rotation_angles.len()
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedSources> {
        resolved_sources(self.inner.resolve(&optics.inner))
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "SourcePositionList",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySourcePositionList {
    inner: SourcePositionList,
}

#[pymethods]
impl PySourcePositionList {
    #[new]
    fn new(positions_m: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            inner: SourcePositionList::new(copy_xyz(&positions_m, "positions_m", false)?),
        })
    }

    #[getter]
    fn positions_m(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        xyz_to_py(py, self.inner.positions_m())
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedSources> {
        resolved_sources(self.inner.resolve(&optics.inner))
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "DirectionList",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyDirectionList {
    inner: DirectionList,
}

#[pymethods]
impl PyDirectionList {
    #[new]
    fn new(unit_vectors: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Self::from_unit_vectors(unit_vectors)
    }

    #[staticmethod]
    fn from_unit_vectors(values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            inner: DirectionList::from_unit_vectors(copy_xyz(&values, "unit_vectors", false)?)
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (values, *, normalize=true))]
    fn from_vectors(values: PyReadonlyArray2<'_, f64>, normalize: bool) -> PyResult<Self> {
        Ok(Self {
            inner: DirectionList::from_vectors(copy_xyz(&values, "vectors", false)?, normalize)
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn from_direction_cosines(values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        direction_from_pairs(&values, DirectionList::from_direction_cosines)
    }

    #[staticmethod]
    fn from_component_angles_radians(values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        direction_from_pairs(&values, DirectionList::from_component_angles_radians)
    }

    #[staticmethod]
    fn from_component_angles_degrees(values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        direction_from_pairs(&values, DirectionList::from_component_angles_degrees)
    }

    #[staticmethod]
    fn from_polar_angles_radians(values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        direction_from_pairs(&values, DirectionList::from_polar_angles_radians)
    }

    #[staticmethod]
    fn from_polar_angles_degrees(values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        direction_from_pairs(&values, DirectionList::from_polar_angles_degrees)
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    #[getter]
    fn unit_vectors(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        xyz_to_py(py, self.inner.unit_vectors())
    }

    #[getter]
    fn direction_cosines(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        pairs_to_py(py, &self.inner.direction_cosines())
    }

    #[getter]
    fn component_angles_rad(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        pairs_to_py(py, &self.inner.component_angles_rad())
    }

    #[getter]
    fn component_angles_deg(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        pairs_to_py(py, &self.inner.component_angles_deg())
    }

    #[getter]
    fn polar_angles_rad(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        pairs_to_py(py, &self.inner.polar_angles_rad())
    }

    #[getter]
    fn polar_angles_deg(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        pairs_to_py(py, &self.inner.polar_angles_deg())
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedSources> {
        resolved_sources(self.inner.resolve(&optics.inner))
    }
}

fn direction_from_pairs(
    values: &PyReadonlyArray2<'_, f64>,
    constructor: impl FnOnce(Vec<[f64; 2]>) -> fpm_rs::Result<DirectionList>,
) -> PyResult<PyDirectionList> {
    Ok(PyDirectionList {
        inner: constructor(copy_pair_arrays(values, "directions")?).map_err(to_py_err)?,
    })
}

#[pyclass(module = "fpm_rs._core", name = "KVectorList", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyKVectorList {
    inner: KVectorList,
}

#[pymethods]
impl PyKVectorList {
    #[new]
    fn new(k_vectors: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let values = copy_pair_arrays(&k_vectors, "k_vectors")?
            .into_iter()
            .map(|[kx, ky]| KVector::new(kx, ky))
            .collect();
        Ok(Self {
            inner: KVectorList::new(values),
        })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    #[getter]
    fn k_vectors(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        kvectors_to_py(py, self.inner.k_vectors())
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedSources> {
        resolved_sources(self.inner.resolve(&optics.inner))
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "SourceGeometry",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySourceGeometry {
    inner: SourceGeometry,
}

#[pymethods]
impl PySourceGeometry {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: extract_source_geometry(value)?,
        })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            SourceGeometry::PlanarArray(_) => "planar_led_array",
            SourceGeometry::SphericalArray(_) => "spherical_led_array",
            SourceGeometry::SphericalArm(_) => "spherical_led_arm",
            SourceGeometry::RotatingArc(_) => "rotating_led_arc",
            SourceGeometry::Positions(_) => "source_position_list",
            SourceGeometry::Directions(_) => "direction_list",
            SourceGeometry::KVectors(_) => "k_vector_list",
        }
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedSources> {
        resolved_sources(self.inner.resolve(&optics.inner))
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "SourceCalibration",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySourceCalibration {
    inner: SourceCalibration,
}

#[pymethods]
impl PySourceCalibration {
    #[new]
    #[pyo3(signature = (*, relative_power=None))]
    fn new(relative_power: Option<Vec<f64>>) -> Self {
        Self {
            inner: SourceCalibration::new(relative_power),
        }
    }

    #[staticmethod]
    fn unity() -> Self {
        Self {
            inner: SourceCalibration::unity(),
        }
    }

    #[getter]
    fn relative_power(&self) -> Option<Vec<f64>> {
        self.inner.relative_power().map(<[f64]>::to_vec)
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "SourceContribution",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySourceContribution {
    inner: SourceContribution,
}

#[pymethods]
impl PySourceContribution {
    #[new]
    fn new(source: usize, intensity_weight: f64) -> Self {
        Self {
            inner: SourceContribution::new(source, intensity_weight),
        }
    }

    #[getter]
    fn source(&self) -> usize {
        self.inner.source
    }

    #[getter]
    fn intensity_weight(&self) -> f64 {
        self.inner.intensity_weight
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "IlluminationFrame",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyIlluminationFrame {
    inner: IlluminationFrame,
}

#[pymethods]
impl PyIlluminationFrame {
    #[new]
    #[pyo3(signature = (contributions, *, gain=1.0))]
    fn new(contributions: Vec<(usize, f64)>, gain: f64) -> Self {
        Self {
            inner: IlluminationFrame::new(
                contributions
                    .into_iter()
                    .map(|(source, weight)| SourceContribution::new(source, weight))
                    .collect(),
                gain,
            ),
        }
    }

    #[getter]
    fn contributions(&self) -> Vec<PySourceContribution> {
        self.inner
            .contributions
            .iter()
            .copied()
            .map(|inner| PySourceContribution { inner })
            .collect()
    }

    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "AcquisitionPlan",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyAcquisitionPlan {
    inner: AcquisitionPlan,
}

#[pymethods]
impl PyAcquisitionPlan {
    #[staticmethod]
    fn all_sources(source_count: usize) -> PyResult<Self> {
        Ok(Self {
            inner: AcquisitionPlan::all_sources(source_count).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn sequential(order: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: AcquisitionPlan::sequential(order).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn from_sparse(py: Python<'_>, frames: Vec<Py<PyIlluminationFrame>>) -> PyResult<Self> {
        Ok(Self {
            inner: AcquisitionPlan::from_sparse(
                frames
                    .into_iter()
                    .map(|frame| frame.borrow(py).inner.clone())
                    .collect(),
            )
            .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn from_dense(weights: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let shape = weights.shape();
        if shape[0] == 0 || shape[1] == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weights must have shape (frames, sources) with non-zero axes",
            ));
        }
        let dense = copy_array2(&weights)
            .chunks_exact(shape[1])
            .map(<[f64]>::to_vec)
            .collect();
        Ok(Self {
            inner: AcquisitionPlan::from_dense(dense).map_err(to_py_err)?,
        })
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    #[getter]
    fn frames(&self) -> Vec<PyIlluminationFrame> {
        self.inner
            .frames()
            .iter()
            .cloned()
            .map(|inner| PyIlluminationFrame { inner })
            .collect()
    }

    fn dense_weights(&self, py: Python<'_>, source_count: usize) -> PyResult<Py<PyArray2<f64>>> {
        dense_to_py(
            py,
            self.inner.dense_weights(source_count).map_err(to_py_err)?,
        )
    }
}

#[pyclass(module = "fpm_rs._core", name = "Illumination", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyIllumination {
    pub(crate) inner: Illumination,
}

#[pymethods]
impl PyIllumination {
    #[new]
    #[pyo3(signature = (geometry, *, calibration=None, acquisition=None))]
    fn new(
        geometry: &Bound<'_, PyAny>,
        calibration: Option<PyRef<'_, PySourceCalibration>>,
        acquisition: Option<PyRef<'_, PyAcquisitionPlan>>,
    ) -> PyResult<Self> {
        let geometry = extract_source_geometry(geometry)?;
        let calibration = calibration
            .map(|value| value.inner.clone())
            .unwrap_or_else(SourceCalibration::unity);
        let acquisition = match acquisition {
            Some(value) => value.inner.clone(),
            None => AcquisitionPlan::all_sources(geometry.source_count()).map_err(to_py_err)?,
        };
        Ok(Self {
            inner: Illumination::new(geometry, calibration, acquisition),
        })
    }

    #[getter]
    fn geometry(&self) -> PySourceGeometry {
        PySourceGeometry {
            inner: self.inner.geometry().clone(),
        }
    }

    #[getter]
    fn calibration(&self) -> PySourceCalibration {
        PySourceCalibration {
            inner: self.inner.calibration().clone(),
        }
    }

    #[getter]
    fn acquisition(&self) -> PyAcquisitionPlan {
        PyAcquisitionPlan {
            inner: self.inner.acquisition().clone(),
        }
    }

    fn resolve(&self, optics: PyRef<'_, PyOptics>) -> PyResult<PyResolvedIllumination> {
        Ok(PyResolvedIllumination {
            inner: self.inner.resolve(&optics.inner).map_err(to_py_err)?,
        })
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "ResolvedSources",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyResolvedSources {
    inner: ResolvedSources,
}

#[pymethods]
impl PyResolvedSources {
    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    #[getter]
    fn directions(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        xyz_to_py(py, self.inner.directions())
    }

    #[getter]
    fn k_vectors(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        kvectors_to_py(py, self.inner.k_vectors())
    }

    #[getter]
    fn positions_m(&self, py: Python<'_>) -> PyResult<Option<Py<PyArray2<f64>>>> {
        self.inner
            .positions_m()
            .map(|values| xyz_to_py(py, values))
            .transpose()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "ResolvedFrame",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyResolvedFrame {
    inner: ResolvedFrame,
}

#[pymethods]
impl PyResolvedFrame {
    #[getter]
    fn contributions(&self) -> Vec<PySourceContribution> {
        self.inner
            .contributions()
            .iter()
            .copied()
            .map(|inner| PySourceContribution { inner })
            .collect()
    }

    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "ResolvedIllumination",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyResolvedIllumination {
    inner: ResolvedIllumination,
}

#[pymethods]
impl PyResolvedIllumination {
    #[getter]
    fn sources(&self) -> PyResolvedSources {
        PyResolvedSources {
            inner: self.inner.sources().clone(),
        }
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    #[getter]
    fn is_multiplexed(&self) -> bool {
        self.inner.is_multiplexed()
    }

    #[getter]
    fn frames(&self) -> Vec<PyResolvedFrame> {
        self.inner
            .frames()
            .iter()
            .cloned()
            .map(|inner| PyResolvedFrame { inner })
            .collect()
    }

    #[getter]
    fn source_power(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_owned_array(py, Array1::from_vec(self.inner.source_power().to_vec()))
            .unbind()
    }

    #[getter]
    fn frame_gains(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_owned_array(py, Array1::from_vec(self.inner.frame_gains())).unbind()
    }

    #[getter]
    fn directions(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        xyz_to_py(py, self.inner.directions())
    }

    #[getter]
    fn positions_m(&self, py: Python<'_>) -> PyResult<Option<Py<PyArray2<f64>>>> {
        self.inner
            .positions_m()
            .map(|values| xyz_to_py(py, values))
            .transpose()
    }

    #[getter]
    fn k_vectors(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        kvectors_to_py(py, self.inner.k_vectors())
    }

    #[getter]
    fn dense_weights(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        dense_to_py(py, self.inner.dense_weights())
    }
}

pub(crate) fn extract_illumination(value: &Bound<'_, PyAny>) -> PyResult<Illumination> {
    value
        .extract::<PyRef<'_, PyIllumination>>()
        .map(|illumination| illumination.inner.clone())
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("illumination must be an Illumination object")
        })
}

fn extract_source_geometry(value: &Bound<'_, PyAny>) -> PyResult<SourceGeometry> {
    if let Ok(value) = value.extract::<PyRef<'_, PySourceGeometry>>() {
        return Ok(value.inner.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyPlanarLedArray>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PySphericalLedArray>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PySphericalLedArm>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyRotatingLedArc>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PySourcePositionList>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyDirectionList>>() {
        return Ok(value.inner.clone().into());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyKVectorList>>() {
        return Ok(value.inner.clone().into());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "geometry must be PlanarLEDArray, SphericalLEDArray, SphericalLEDArm, RotatingLEDArc, SourcePositionList, DirectionList, KVectorList, or SourceGeometry",
    ))
}

fn resolved_sources(result: fpm_rs::Result<ResolvedSources>) -> PyResult<PyResolvedSources> {
    Ok(PyResolvedSources {
        inner: result.map_err(to_py_err)?,
    })
}

fn parse_pitch(value: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok((scalar, scalar));
    }
    value.extract::<(f64, f64)>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("pitch_m must be a scalar or (pitch_x, pitch_y)")
    })
}

fn copy_pairs(values: &PyReadonlyArray2<'_, f64>, name: &'static str) -> PyResult<Vec<(f64, f64)>> {
    Ok(copy_pair_arrays(values, name)?
        .into_iter()
        .map(|[x, y]| (x, y))
        .collect())
}

fn copy_pair_arrays(
    values: &PyReadonlyArray2<'_, f64>,
    name: &'static str,
) -> PyResult<Vec<[f64; 2]>> {
    let shape = values.shape();
    if shape[1] != 2 || shape[0] == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (sources, 2)"
        )));
    }
    Ok(copy_array2(values)
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect())
}

fn copy_xyz(
    values: &PyReadonlyArray2<'_, f64>,
    name: &'static str,
    allow_empty: bool,
) -> PyResult<Vec<[f64; 3]>> {
    let shape = values.shape();
    if shape[1] != 3 || (!allow_empty && shape[0] == 0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (sources, 3)"
        )));
    }
    Ok(copy_array2(values)
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

fn xyz_to_py(py: Python<'_>, values: &[[f64; 3]]) -> PyResult<Py<PyArray2<f64>>> {
    let flat = values.iter().flatten().copied().collect();
    array2_to_py(py, (values.len(), 3), flat)
}

fn pairs_to_py(py: Python<'_>, values: &[[f64; 2]]) -> PyResult<Py<PyArray2<f64>>> {
    let flat = values.iter().flatten().copied().collect();
    array2_to_py(py, (values.len(), 2), flat)
}

fn kvectors_to_py(py: Python<'_>, values: &[KVector]) -> PyResult<Py<PyArray2<f64>>> {
    let flat = values
        .iter()
        .flat_map(|value| [value.kx, value.ky])
        .collect();
    array2_to_py(py, (values.len(), 2), flat)
}

fn dense_to_py(py: Python<'_>, values: Vec<Vec<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    let rows = values.len();
    let columns = values.first().map_or(0, Vec::len);
    let flat = values.into_iter().flatten().collect();
    array2_to_py(py, (rows, columns), flat)
}

fn array2_to_py(
    py: Python<'_>,
    shape: (usize, usize),
    values: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let array = Array2::from_shape_vec(shape, values)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, array).unbind())
}

#[pyclass(module = "fpm_rs._core", name = "CameraModel", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyCameraModel {
    pub(crate) inner: CameraModel,
}

#[pymethods]
impl PyCameraModel {
    #[new]
    #[pyo3(signature = (*, photons_per_pixel=1000.0, gain_counts_per_electron=1.0, offset_counts=0.0, read_noise_electrons=0.0, dark_current_electrons=0.0, shot_noise=false, pixel_sensitivity=None, bit_depth=Some(16), saturation_counts=None, quantize=true, bad_pixels=Vec::new(), bad_pixel_value_counts=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        photons_per_pixel: f64,
        gain_counts_per_electron: f64,
        offset_counts: f64,
        read_noise_electrons: f64,
        dark_current_electrons: f64,
        shot_noise: bool,
        pixel_sensitivity: Option<PyReadonlyArray2<'_, f64>>,
        bit_depth: Option<u8>,
        saturation_counts: Option<f64>,
        quantize: bool,
        bad_pixels: Vec<usize>,
        bad_pixel_value_counts: Option<f64>,
    ) -> PyResult<Self> {
        let inner = CameraModel {
            photons_per_pixel,
            gain_counts_per_electron,
            offset_counts,
            read_noise_electrons,
            dark_current_electrons,
            shot_noise,
            pixel_sensitivity: pixel_sensitivity.as_ref().map(copy_array2),
            bit_depth,
            saturation_counts,
            quantize,
            bad_pixels,
            bad_pixel_value_counts,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn ideal() -> Self {
        Self {
            inner: CameraModel::ideal(),
        }
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "IlluminationAcquisitionErrors",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyIlluminationAcquisitionErrors {
    pub(crate) inner: IlluminationAcquisitionErrors,
}

#[pymethods]
impl PyIlluminationAcquisitionErrors {
    #[new]
    #[pyo3(signature = (*, frame_gain_relative_std=0.0, missing_frames=Vec::new(), source_permutation=None))]
    fn new(
        frame_gain_relative_std: f64,
        missing_frames: Vec<usize>,
        source_permutation: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        if !frame_gain_relative_std.is_finite() || frame_gain_relative_std < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frame_gain_relative_std must be finite and non-negative",
            ));
        }
        Ok(Self {
            inner: IlluminationAcquisitionErrors {
                frame_gain_relative_std,
                missing_frames,
                source_permutation,
            },
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPupilAberration>()?;
    module.add_class::<PyOptics>()?;
    module.add_class::<PyArrayPose>()?;
    module.add_class::<PyPlanarLedArray>()?;
    module.add_class::<PySphericalLedArray>()?;
    module.add_class::<PySphericalLedArm>()?;
    module.add_class::<PyRotatingLedArc>()?;
    module.add_class::<PySourcePositionList>()?;
    module.add_class::<PyDirectionList>()?;
    module.add_class::<PyKVectorList>()?;
    module.add_class::<PySourceGeometry>()?;
    module.add_class::<PySourceCalibration>()?;
    module.add_class::<PySourceContribution>()?;
    module.add_class::<PyIlluminationFrame>()?;
    module.add_class::<PyAcquisitionPlan>()?;
    module.add_class::<PyIllumination>()?;
    module.add_class::<PyResolvedSources>()?;
    module.add_class::<PyResolvedFrame>()?;
    module.add_class::<PyResolvedIllumination>()?;
    module.add_class::<PyCameraModel>()?;
    module.add_class::<PyIlluminationAcquisitionErrors>()?;
    Ok(())
}
