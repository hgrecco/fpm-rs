use std::{collections::BTreeMap, fs::File, io::BufWriter, path::Path};

use image::{GrayImage, Luma};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    array_layout::StandardArray2,
    array_serde::Array2Data,
    backend::FftDirection,
    complex,
    error::Error,
    illumination_calibration::IlluminationCalibrationState,
    model::{ImagePlaneModel, Pupil, ifftshift_copy},
};

use super::{ReconstructionState, ReconstructionTrace};

/// Execution summary attached to a completed reconstruction.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RuntimeInfo {
    /// Wall-clock seconds spent in this run, including elapsed time restored from a checkpoint.
    pub elapsed_seconds: f64,
    /// Number of complete reconstruction iterations represented by the result.
    pub completed_iterations: usize,
    /// Whether a callback requested termination before the configured iteration limit.
    pub stopped_early: bool,
    /// Stable algorithm type name used for the run.
    pub algorithm: String,
}

/// Owned reconstructed fields, calibration values, trace, and runtime metadata.
///
/// Object-domain arrays and the centered object spectrum all have high-resolution
/// `(height, width)` shape and standard row-major storage. `object` is complex field,
/// `amplitude` is its magnitude, and `phase` is wrapped in radians in `[-π, π]`.
#[derive(Clone, Debug)]
pub struct ReconstructionResult {
    /// Reconstructed high-resolution complex sample transmission field.
    pub object: Array2<Complex64>,
    /// Non-negative magnitude of [`Self::object`].
    pub amplitude: Array2<f64>,
    /// Wrapped argument of [`Self::object`], in radians in `[-π, π]`.
    pub phase: Array2<f64>,
    /// Centered Fourier spectrum corresponding to [`Self::object`].
    pub object_spectrum: Array2<Complex64>,
    /// Recovered low-resolution complex pupil and binary aperture support.
    pub recovered_pupil: Pupil,
    /// Per-source `(row, column)` corrections in Fourier-grid pixels.
    pub calibrated_illumination: Option<Vec<(f64, f64)>>,
    /// Optional positive multiplicative gains in acquisition-frame order.
    pub recovered_frame_gains: Option<Vec<f64>>,
    /// Optional additive intensity background in acquisition-frame order.
    pub recovered_background: Option<Vec<f64>>,
    /// Complete physical planar-array calibration state for joint runs.
    pub physical_illumination_calibration: Option<IlluminationCalibrationState>,
    /// Reusable model compiled from the final physical illumination.
    pub calibrated_model: Option<ImagePlaneModel>,
    /// Universal and algorithm-specific iteration history.
    pub trace: ReconstructionTrace,
    /// Final named scalar diagnostics not represented by the trace.
    pub scalar_diagnostics: BTreeMap<String, f64>,
    /// Timing, iteration count, early-stop status, and algorithm name.
    pub runtime: RuntimeInfo,
    /// User- and runner-supplied string metadata.
    pub metadata: BTreeMap<String, String>,
}

impl ReconstructionResult {
    pub(crate) fn from_state(
        state: &mut ReconstructionState,
        trace: ReconstructionTrace,
        runtime: RuntimeInfo,
    ) -> Result<Self> {
        let object = state_object(state)?;
        let amplitude = complex::amplitude(object.view());
        let phase = complex::phase(object.view());
        let mut scalar_diagnostics = BTreeMap::new();
        if let Some(objective) = trace.final_objective() {
            scalar_diagnostics.insert("final_objective".into(), objective);
        }
        Ok(Self {
            object,
            amplitude,
            phase,
            object_spectrum: state.object_spectrum.clone().into_inner(),
            recovered_pupil: state.pupil.clone(),
            calibrated_illumination: state.illumination_corrections.clone(),
            recovered_frame_gains: state.frame_gains.clone(),
            recovered_background: state.background.clone(),
            physical_illumination_calibration: state.physical_illumination_calibration.clone(),
            calibrated_model: state.calibrated_model.clone(),
            trace,
            scalar_diagnostics,
            runtime,
            metadata: BTreeMap::new(),
        })
    }

    /// Writes object amplitude as a linearly normalized 8-bit grayscale image.
    pub fn save_amplitude(&self, path: impl AsRef<Path>) -> Result<()> {
        save_grayscale(self.amplitude.view(), path, false)
    }

    /// Writes wrapped object phase as an 8-bit grayscale image mapping `[-π, π]` to `[0, 255]`.
    pub fn save_phase(&self, path: impl AsRef<Path>) -> Result<()> {
        save_grayscale(self.phase.view(), path, true)
    }

    /// Serializes the complex object and its `(height, width)` shape as JSON.
    pub fn save_complex_object(&self, path: impl AsRef<Path>) -> Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(writer, &Array2Data::from_view(self.object.view()))?;
        Ok(())
    }

    /// Serializes recovered complex pupil values and binary support as JSON.
    pub fn save_pupil(&self, path: impl AsRef<Path>) -> Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(writer, &self.recovered_pupil)?;
        Ok(())
    }

    /// Writes one CSV row per iteration with objective and elapsed seconds.
    pub fn save_trace_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record(["iteration", "objective", "elapsed_seconds"])?;
        for record in &self.trace.iterations {
            writer.serialize((record.iteration, record.objective, record.elapsed_seconds))?;
        }
        writer.flush()?;
        Ok(())
    }

    #[cfg(feature = "parquet")]
    /// Writes a self-describing Parquet/NPY result bundle and reopens it lazily.
    ///
    /// The returned [`crate::reconstruction::ResultBundle`] reads its manifest
    /// immediately but does not load scientific arrays until an accessor is called.
    /// Repeated access returns the same cached [`std::sync::Arc`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use fpm_rs::{
    ///     reconstruction::{BundleExportOptions, ReconstructionResult, read_bundle},
    ///     Result,
    /// };
    /// use std::sync::Arc;
    ///
    /// # fn completed_reconstruction() -> Result<ReconstructionResult> { unimplemented!() }
    /// # fn main() -> Result<()> {
    /// let result = completed_reconstruction()?;
    /// result.write_bundle("result-bundle", BundleExportOptions::default())?;
    ///
    /// let bundle = read_bundle("result-bundle")?;
    /// let first = bundle.object()?;
    /// let second = bundle.object()?;
    /// assert!(Arc::ptr_eq(&first, &second));
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_bundle(
        &self,
        path: impl AsRef<Path>,
        options: crate::reconstruction::BundleExportOptions,
    ) -> Result<crate::reconstruction::ResultBundle> {
        crate::tabular::parquet::write_result_bundle(self, path.as_ref(), options, None, None)
    }

    /// Writes a bundle including optional callback diagnostics and reference
    /// evaluation records.
    #[cfg(feature = "parquet")]
    pub fn write_bundle_with_context(
        &self,
        path: impl AsRef<Path>,
        options: crate::reconstruction::BundleExportOptions,
        diagnostics: Option<&crate::diagnostics::ReconstructionDiagnostics>,
        evaluation: Option<&crate::evaluation::ReconstructionEvaluation>,
    ) -> Result<crate::reconstruction::ResultBundle> {
        crate::tabular::parquet::write_result_bundle(
            self,
            path.as_ref(),
            options,
            diagnostics,
            evaluation,
        )
    }

    /// Checks matching non-empty standard-layout arrays, pupil shape, finite values,
    /// calibration lengths and ranges, and consistency with optional frame-count metadata.
    pub fn validate(&self) -> Result<()> {
        let shape = self.object.dim();
        if shape.0 == 0 || shape.1 == 0 {
            return Err(Error::InvalidShape(
                "result reconstruction arrays must be non-empty".into(),
            ));
        }
        for (context, array_shape, strides, is_standard) in [
            (
                "reconstruction result object",
                self.object.shape(),
                self.object.strides(),
                self.object.is_standard_layout(),
            ),
            (
                "reconstruction result amplitude",
                self.amplitude.shape(),
                self.amplitude.strides(),
                self.amplitude.is_standard_layout(),
            ),
            (
                "reconstruction result phase",
                self.phase.shape(),
                self.phase.strides(),
                self.phase.is_standard_layout(),
            ),
            (
                "reconstruction result object spectrum",
                self.object_spectrum.shape(),
                self.object_spectrum.strides(),
                self.object_spectrum.is_standard_layout(),
            ),
        ] {
            if !is_standard {
                return Err(Error::NonStandardLayout {
                    context,
                    shape: array_shape.to_vec(),
                    strides: strides.to_vec(),
                });
            }
        }
        if self.amplitude.dim() != shape
            || self.phase.dim() != shape
            || self.object_spectrum.dim() != shape
        {
            return Err(Error::InvalidShape(
                "result object, amplitude, phase, and spectrum shapes must match".into(),
            ));
        }
        if self.recovered_pupil.support.len() != self.recovered_pupil.values.len() {
            return Err(Error::InvalidShape(
                "result pupil support and values have different lengths".into(),
            ));
        }
        if self
            .calibrated_illumination
            .as_ref()
            .is_some_and(Vec::is_empty)
            || self
                .recovered_frame_gains
                .as_ref()
                .is_some_and(Vec::is_empty)
            || self
                .recovered_background
                .as_ref()
                .is_some_and(Vec::is_empty)
        {
            return Err(Error::InvalidShape(
                "present result calibration arrays must be non-empty".into(),
            ));
        }
        if let (Some(gains), Some(background)) =
            (&self.recovered_frame_gains, &self.recovered_background)
            && gains.len() != background.len()
        {
            return Err(Error::InvalidShape(
                "result frame gains and background lengths must match".into(),
            ));
        }
        if let Some(frame_count) = self
            .metadata
            .get("frame_count")
            .and_then(|value| value.parse::<usize>().ok())
            && self
                .recovered_frame_gains
                .as_ref()
                .into_iter()
                .chain(self.recovered_background.as_ref())
                .any(|values| values.len() != frame_count)
        {
            return Err(Error::InvalidShape(
                "result frame calibration length must match metadata frame_count".into(),
            ));
        }
        if self
            .object
            .iter()
            .chain(self.object_spectrum.iter())
            .chain(self.recovered_pupil.values.as_slice())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
            || self
                .amplitude
                .iter()
                .chain(self.phase.iter())
                .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidModel(
                "result arrays contain non-finite values".into(),
            ));
        }
        if self.calibrated_illumination.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|&(row, column)| !row.is_finite() || !column.is_finite())
        }) || self.recovered_frame_gains.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        }) || self
            .recovered_background
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::InvalidModel(
                "result calibration values are invalid".into(),
            ));
        }
        if self.physical_illumination_calibration.is_some() != self.calibrated_model.is_some() {
            return Err(Error::InvalidModel(
                "result physical calibration and calibrated model must be present together".into(),
            ));
        }
        if let Some(model) = &self.calibrated_model {
            model.validate()?;
            if model.reconstruction_shape() != shape
                || model.pupil().shape() != self.recovered_pupil.shape()
            {
                return Err(Error::InvalidModel(
                    "result calibrated model shapes do not match reconstructed fields".into(),
                ));
            }
        }
        if let Some(calibration) = &self.physical_illumination_calibration {
            calibration.validate()?;
        }
        if self
            .scalar_diagnostics
            .values()
            .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidModel(
                "result diagnostics contain non-finite values".into(),
            ));
        }
        if !self.runtime.elapsed_seconds.is_finite()
            || self.runtime.elapsed_seconds < 0.0
            || self.runtime.algorithm.is_empty()
            || self.runtime.completed_iterations != self.trace.iterations.len()
            || self
                .trace
                .iterations
                .iter()
                .enumerate()
                .any(|(index, record)| {
                    record.iteration != index + 1
                        || !record.objective.is_finite()
                        || !record.elapsed_seconds.is_finite()
                        || record.elapsed_seconds < 0.0
                })
            || self
                .trace
                .iterations
                .windows(2)
                .any(|pair| pair[1].elapsed_seconds < pair[0].elapsed_seconds)
            || self
                .trace
                .iterations
                .last()
                .is_some_and(|record| record.elapsed_seconds > self.runtime.elapsed_seconds)
        {
            return Err(Error::InvalidModel(
                "result runtime and trace are inconsistent".into(),
            ));
        }
        if self.trace.algorithm_metrics.iter().any(|record| {
            record.iteration == 0
                || record.iteration > self.runtime.completed_iterations
                || record.namespace.is_empty()
                || record.metric.is_empty()
                || !record.value.is_finite()
        }) {
            return Err(Error::InvalidModel(
                "result algorithm metrics are invalid".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn state_object(state: &mut ReconstructionState) -> Result<Array2<Complex64>> {
    if let Some(cached) = &state.object_real_space_cache {
        return Ok(cached.clone().into_inner());
    }
    let shape = state.object_spectrum.dim();
    let mut unshifted = vec![Complex64::default(); state.object_spectrum.len()];
    ifftshift_copy(state.object_spectrum.as_slice(), &mut unshifted, shape);
    state.backend.fft2(
        &mut unshifted,
        shape,
        FftDirection::Inverse,
        &mut state.scratch.column,
    )?;
    let object = StandardArray2::from_shape_vec(shape, unshifted)?;
    state.object_real_space_cache = Some(object.clone());
    Ok(object.into_inner())
}

pub(crate) fn save_grayscale(
    values: ArrayView2<'_, f64>,
    path: impl AsRef<Path>,
    phase: bool,
) -> Result<()> {
    let range = if phase {
        (-std::f64::consts::PI, std::f64::consts::PI)
    } else {
        let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (minimum, maximum)
    };
    save_grayscale_with_range(values, path, range)
}

pub(crate) fn save_signed_grayscale(
    values: ArrayView2<'_, f64>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let maximum_absolute = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(f64::EPSILON);
    save_grayscale_with_range(values, path, (-maximum_absolute, maximum_absolute))
}

fn save_grayscale_with_range(
    values: ArrayView2<'_, f64>,
    path: impl AsRef<Path>,
    (minimum, maximum): (f64, f64),
) -> Result<()> {
    let width = u32::try_from(values.ncols()).map_err(|_| {
        Error::InvalidShape("image width does not fit the PNG dimension type".into())
    })?;
    let height = u32::try_from(values.nrows()).map_err(|_| {
        Error::InvalidShape("image height does not fit the PNG dimension type".into())
    })?;
    let range = (maximum - minimum).max(f64::EPSILON);
    let mut image = GrayImage::new(width, height);
    for row in 0..values.nrows() {
        for column in 0..values.ncols() {
            let normalized = ((values[(row, column)] - minimum) / range).clamp(0.0, 1.0);
            image.put_pixel(
                column as u32,
                row as u32,
                Luma([(normalized * 255.0).round() as u8]),
            );
        }
    }
    image.save(path)?;
    Ok(())
}

impl Serialize for ReconstructionResult {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Representation<'a> {
            object: Array2Data<Complex64>,
            amplitude: Array2Data<f64>,
            phase: Array2Data<f64>,
            object_spectrum: Array2Data<Complex64>,
            recovered_pupil: &'a Pupil,
            calibrated_illumination: &'a Option<Vec<(f64, f64)>>,
            recovered_frame_gains: &'a Option<Vec<f64>>,
            recovered_background: &'a Option<Vec<f64>>,
            physical_illumination_calibration: &'a Option<IlluminationCalibrationState>,
            calibrated_model: &'a Option<ImagePlaneModel>,
            trace: &'a ReconstructionTrace,
            scalar_diagnostics: &'a BTreeMap<String, f64>,
            runtime: &'a RuntimeInfo,
            metadata: &'a BTreeMap<String, String>,
        }

        Representation {
            object: Array2Data::from_view(self.object.view()),
            amplitude: Array2Data::from_view(self.amplitude.view()),
            phase: Array2Data::from_view(self.phase.view()),
            object_spectrum: Array2Data::from_view(self.object_spectrum.view()),
            recovered_pupil: &self.recovered_pupil,
            calibrated_illumination: &self.calibrated_illumination,
            recovered_frame_gains: &self.recovered_frame_gains,
            recovered_background: &self.recovered_background,
            physical_illumination_calibration: &self.physical_illumination_calibration,
            calibrated_model: &self.calibrated_model,
            trace: &self.trace,
            scalar_diagnostics: &self.scalar_diagnostics,
            runtime: &self.runtime,
            metadata: &self.metadata,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ReconstructionResult {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Representation {
            object: Array2Data<Complex64>,
            amplitude: Array2Data<f64>,
            phase: Array2Data<f64>,
            object_spectrum: Array2Data<Complex64>,
            recovered_pupil: Pupil,
            calibrated_illumination: Option<Vec<(f64, f64)>>,
            recovered_frame_gains: Option<Vec<f64>>,
            recovered_background: Option<Vec<f64>>,
            physical_illumination_calibration: Option<IlluminationCalibrationState>,
            calibrated_model: Option<ImagePlaneModel>,
            trace: ReconstructionTrace,
            scalar_diagnostics: BTreeMap<String, f64>,
            runtime: RuntimeInfo,
            metadata: BTreeMap<String, String>,
        }

        let representation = Representation::deserialize(deserializer)?;
        let result = Self {
            object: representation
                .object
                .into_array()
                .map_err(D::Error::custom)?,
            amplitude: representation
                .amplitude
                .into_array()
                .map_err(D::Error::custom)?,
            phase: representation
                .phase
                .into_array()
                .map_err(D::Error::custom)?,
            object_spectrum: representation
                .object_spectrum
                .into_array()
                .map_err(D::Error::custom)?,
            recovered_pupil: representation.recovered_pupil,
            calibrated_illumination: representation.calibrated_illumination,
            recovered_frame_gains: representation.recovered_frame_gains,
            recovered_background: representation.recovered_background,
            physical_illumination_calibration: representation.physical_illumination_calibration,
            calibrated_model: representation.calibrated_model,
            trace: representation.trace,
            scalar_diagnostics: representation.scalar_diagnostics,
            runtime: representation.runtime,
            metadata: representation.metadata,
        };
        result.validate().map_err(D::Error::custom)?;
        Ok(result)
    }
}
