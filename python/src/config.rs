use fpm_rs::{
    experiment::{
        AngleList, CodedIllumination, Illumination, KVector, LEDArray, LEDSphere, Optics,
        PupilAberration, RotatingLEDArc, SphericalLEDArm,
    },
    simulation::{CameraModel, IlluminationAcquisitionErrors},
};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
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
    #[pyo3(signature = (wavelength, objective_na, magnification, camera_pixel_size, *, medium_index=1.0, defocus_distance=None, pupil_aberration=None))]
    fn new(
        wavelength: f64,
        objective_na: f64,
        magnification: f64,
        camera_pixel_size: f64,
        medium_index: f64,
        defocus_distance: Option<f64>,
        pupil_aberration: Option<PyRef<'_, PyPupilAberration>>,
    ) -> PyResult<Self> {
        let inner = Optics {
            wavelength,
            objective_na,
            magnification,
            camera_pixel_size,
            medium_index,
            defocus_distance,
            pupil_aberration: pupil_aberration.map(|value| value.inner.clone()),
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn wavelength(&self) -> f64 {
        self.inner.wavelength
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
    fn medium_index(&self) -> f64 {
        self.inner.medium_index
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

#[pyclass(module = "fpm_rs._core", name = "LEDArray", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyLedArray {
    pub(crate) inner: LEDArray,
}

#[pymethods]
impl PyLedArray {
    #[new]
    #[pyo3(signature = (grid_shape, pitch, distance, center, *, wavelength_override=None, illumination_order=None, intensity_weights=None, rotation_degrees=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        grid_shape: (usize, usize),
        pitch: f64,
        distance: f64,
        center: (f64, f64),
        wavelength_override: Option<f64>,
        illumination_order: Option<Vec<usize>>,
        intensity_weights: Option<Vec<f64>>,
        rotation_degrees: f64,
    ) -> PyResult<Self> {
        let inner = LEDArray {
            grid_shape,
            pitch,
            distance,
            center,
            wavelength_override,
            illumination_order,
            intensity_weights,
            rotation_radians: rotation_degrees.to_radians(),
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn grid_shape(&self) -> (usize, usize) {
        self.inner.grid_shape
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.grid_shape.0 * self.inner.grid_shape.1
    }
}

#[pyclass(module = "fpm_rs._core", name = "LEDSphere", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyLedSphere {
    pub(crate) inner: LEDSphere,
}

#[pymethods]
impl PyLedSphere {
    #[new]
    #[pyo3(signature = (angles, radius, *, center_offset=(0.0, 0.0, 0.0), orientation_degrees=(0.0, 0.0, 0.0), angular_corrections=None, wavelength_override=None, illumination_order=None, intensity_weights=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        angles: PyReadonlyArray2<'_, f64>,
        radius: f64,
        center_offset: (f64, f64, f64),
        orientation_degrees: (f64, f64, f64),
        angular_corrections: Option<PyReadonlyArray2<'_, f64>>,
        wavelength_override: Option<f64>,
        illumination_order: Option<Vec<usize>>,
        intensity_weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let angles = copy_angle_pairs(&angles, "angles")?;
        let angular_corrections = angular_corrections
            .as_ref()
            .map(|values| copy_angle_pairs(values, "angular_corrections"))
            .transpose()?;
        let inner = LEDSphere {
            angles,
            radius,
            center_offset,
            orientation_radians: (
                orientation_degrees.0.to_radians(),
                orientation_degrees.1.to_radians(),
                orientation_degrees.2.to_radians(),
            ),
            angular_corrections,
            wavelength_override,
            illumination_order,
            intensity_weights,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.angles.len()
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
    pub(crate) inner: SphericalLEDArm,
}

#[pymethods]
impl PySphericalLedArm {
    #[new]
    #[pyo3(signature = (commanded_angles, arm_length, *, pivot_offset=(0.0, 0.0, 0.0), orientation_degrees=(0.0, 0.0, 0.0), theta_zero_degrees=0.0, phi_zero_degrees=0.0, theta_scale=1.0, phi_scale=1.0, elevation_axis_tilt_degrees=0.0, theta_backlash_degrees=0.0, phi_backlash_degrees=0.0, wavelength_override=None, intensity_weights=None))]
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
        wavelength_override: Option<f64>,
        intensity_weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let inner = SphericalLEDArm {
            commanded_angles: copy_angle_pairs(&commanded_angles, "commanded_angles")?,
            arm_length,
            pivot_offset,
            orientation_radians: (
                orientation_degrees.0.to_radians(),
                orientation_degrees.1.to_radians(),
                orientation_degrees.2.to_radians(),
            ),
            theta_zero_radians: theta_zero_degrees.to_radians(),
            phi_zero_radians: phi_zero_degrees.to_radians(),
            theta_scale,
            phi_scale,
            elevation_axis_tilt_radians: elevation_axis_tilt_degrees.to_radians(),
            theta_backlash_radians: theta_backlash_degrees.to_radians(),
            phi_backlash_radians: phi_backlash_degrees.to_radians(),
            wavelength_override,
            intensity_weights,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.commanded_angles.len()
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
    pub(crate) inner: RotatingLEDArc,
}

#[pymethods]
impl PyRotatingLedArc {
    #[new]
    #[pyo3(signature = (led_thetas, rotation_angles, radius, *, axis_origin_offset=(0.0, 0.0, 0.0), axis_tilt_degrees=(0.0, 0.0), led_angular_corrections=None, led_radial_offsets=None, rotation_zero_degrees=0.0, rotation_scale=1.0, rotation_backlash_degrees=0.0, wavelength_override=None, led_intensity_weights=None))]
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
        wavelength_override: Option<f64>,
        led_intensity_weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let led_angular_corrections = led_angular_corrections
            .as_ref()
            .map(|values| copy_angle_pairs(values, "led_angular_corrections"))
            .transpose()?;
        let inner = RotatingLEDArc {
            led_thetas,
            rotation_angles,
            radius,
            axis_origin_offset,
            axis_tilt_radians: (
                axis_tilt_degrees.0.to_radians(),
                axis_tilt_degrees.1.to_radians(),
            ),
            led_angular_corrections,
            led_radial_offsets,
            rotation_zero_radians: rotation_zero_degrees.to_radians(),
            rotation_scale,
            rotation_backlash_radians: rotation_backlash_degrees.to_radians(),
            wavelength_override,
            led_intensity_weights,
        };
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
    fn source_count(&self) -> PyResult<usize> {
        self.inner.source_count().map_err(to_py_err)
    }
}

#[pyclass(module = "fpm_rs._core", name = "AngleList", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyAngleList {
    pub(crate) inner: AngleList,
}

#[pymethods]
impl PyAngleList {
    #[new]
    fn new(angles: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let shape = angles.shape();
        if shape[1] != 2 || shape[0] == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "angles must have shape (sources, 2)",
            ));
        }
        let values = copy_array2(&angles)
            .chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect();
        Ok(Self {
            inner: AngleList::new(values),
        })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.angles.len()
    }
}

fn copy_angle_pairs(
    values: &PyReadonlyArray2<'_, f64>,
    name: &'static str,
) -> PyResult<Vec<(f64, f64)>> {
    let shape = values.shape();
    if shape[1] != 2 || shape[0] == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (sources, 2)"
        )));
    }
    Ok(copy_array2(values)
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect())
}

#[pyclass(module = "fpm_rs._core", name = "KVectorList", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyKVectorList {
    pub(crate) inner: Vec<KVector>,
}

#[pymethods]
impl PyKVectorList {
    #[new]
    fn new(k_vectors: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let shape = k_vectors.shape();
        if shape[1] != 2 || shape[0] == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k_vectors must have shape (sources, 2)",
            ));
        }
        let inner = copy_array2(&k_vectors)
            .chunks_exact(2)
            .map(|pair| KVector::new(pair[0], pair[1]))
            .collect();
        Ok(Self { inner })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(
    module = "fpm_rs._core",
    name = "CodedIllumination",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyCodedIllumination {
    pub(crate) inner: CodedIllumination,
}

#[pymethods]
impl PyCodedIllumination {
    #[new]
    fn new(
        k_vectors: PyReadonlyArray2<'_, f64>,
        frame_weights: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let vector_shape = k_vectors.shape();
        let weight_shape = frame_weights.shape();
        if vector_shape[1] != 2 || vector_shape[0] == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k_vectors must have shape (sources, 2)",
            ));
        }
        if weight_shape[0] == 0 || weight_shape[1] != vector_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frame_weights must have shape (frames, sources)",
            ));
        }
        let vectors = copy_array2(&k_vectors)
            .chunks_exact(2)
            .map(|pair| KVector::new(pair[0], pair[1]))
            .collect();
        let dense = copy_array2(&frame_weights);
        let rows = dense
            .chunks_exact(weight_shape[1])
            .map(|row| {
                row.iter()
                    .copied()
                    .enumerate()
                    .filter(|&(_, weight)| weight != 0.0)
                    .collect()
            })
            .collect();
        Ok(Self {
            inner: CodedIllumination {
                source_k_vectors: vectors,
                frame_weights: rows,
            },
        })
    }

    #[getter]
    fn source_count(&self) -> usize {
        self.inner.source_k_vectors.len()
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_weights.len()
    }
}

pub(crate) fn extract_illumination(value: &Bound<'_, PyAny>) -> PyResult<Illumination> {
    if let Ok(source) = value.extract::<PyRef<'_, PyLedArray>>() {
        return Ok(Illumination::LEDArray(source.inner.clone()));
    }
    if let Ok(source) = value.extract::<PyRef<'_, PyLedSphere>>() {
        return Ok(Illumination::LEDSphere(source.inner.clone()));
    }
    if let Ok(source) = value.extract::<PyRef<'_, PySphericalLedArm>>() {
        return Ok(Illumination::SphericalLEDArm(source.inner.clone()));
    }
    if let Ok(source) = value.extract::<PyRef<'_, PyRotatingLedArc>>() {
        return Ok(Illumination::RotatingLEDArc(source.inner.clone()));
    }
    if let Ok(source) = value.extract::<PyRef<'_, PyAngleList>>() {
        return Ok(Illumination::Angles(source.inner.clone()));
    }
    if let Ok(source) = value.extract::<PyRef<'_, PyKVectorList>>() {
        return Ok(Illumination::KVectors(source.inner.clone()));
    }
    if let Ok(source) = value.extract::<PyRef<'_, PyCodedIllumination>>() {
        return Ok(Illumination::Coded(source.inner.clone()));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "illumination must be LEDArray, LEDSphere, SphericalLEDArm, RotatingLEDArc, AngleList, KVectorList, or CodedIllumination",
    ))
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
    module.add_class::<PyLedArray>()?;
    module.add_class::<PyLedSphere>()?;
    module.add_class::<PySphericalLedArm>()?;
    module.add_class::<PyRotatingLedArc>()?;
    module.add_class::<PyAngleList>()?;
    module.add_class::<PyKVectorList>()?;
    module.add_class::<PyCodedIllumination>()?;
    module.add_class::<PyCameraModel>()?;
    module.add_class::<PyIlluminationAcquisitionErrors>()?;
    Ok(())
}
