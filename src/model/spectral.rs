//! Compiled narrowband kernels with a common grid and sparse detector composition.

use std::{collections::BTreeSet, sync::Arc};

use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    backend::{Backend, CpuBackend},
    error::Error,
    experiment::{
        Illumination, SourceGeometry, SpectralAcquisitionPlan, SpectralChannel, SpectralGeometry,
    },
};

use super::{ForwardModel, ImagePlaneModel, ReconstructionShape};

/// Explicit sharing of the thin sample's complex transmission between wavelengths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectCoupling {
    /// One independently reconstructed complex transmission per channel (the default).
    #[default]
    Independent,
    /// A single complex transmission used by all channels; appropriate only when
    /// wavelength-independent transmission is an acceptable sample approximation.
    SharedComplex,
}

/// A channel ID paired with its wavelength-specific compiled numerical kernel.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralModelChannel {
    /// Stable channel identifier, preserved in result order.
    pub channel_id: String,
    /// Ordinary compiled kernel, retaining wavelength, pupil, and local source weights.
    pub model: ImagePlaneModel,
}

/// Algorithm-facing narrowband spectral model, containing no physical geometry.
///
/// All channels share detector shape and object-plane pixel pitch. Each keeps
/// its own crops, subpixel offsets, pupil, and wavelength. For detector row `f`,
/// prediction is `gain[f] * sum(weight * local_intensity) + background[f]`.
/// Local source powers, source weights, and illumination-frame gains are
/// included in `local_intensity`. Detector calibration is applied once.
#[derive(Clone, Debug, Serialize)]
pub struct SpectralImagePlaneModel {
    channels: Vec<SpectralModelChannel>,
    acquisition: SpectralAcquisitionPlan,
    object_coupling: ObjectCoupling,
}

impl<'de> Deserialize<'de> for SpectralImagePlaneModel {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        use serde::de::Error as _;
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Representation {
            channels: Vec<SpectralModelChannel>,
            acquisition: SpectralAcquisitionPlan,
            object_coupling: ObjectCoupling,
        }
        let repr = Representation::deserialize(deserializer)?;
        let model = Self {
            channels: repr.channels,
            acquisition: repr.acquisition,
            object_coupling: repr.object_coupling,
        };
        model.validate().map_err(D::Error::custom)?;
        Ok(model)
    }
}

impl SpectralImagePlaneModel {
    /// Resolves every channel atomically and selects a grid from the union of crop bounds.
    ///
    /// IDs and wavelengths must be unique, wavelengths finite and positive,
    /// and all detector pitches equal. Shared direct k-vectors and configurations
    /// requiring spatial registration or resampling are rejected. An explicit
    /// reconstruction shape must contain all channels' interpolation stencils.
    ///
    /// # Example
    ///
    /// ```
    /// use fpm_rs::{
    ///     algorithms::{SpectralAlternatingProjection, SpectralReconstructionAlgorithm},
    ///     experiment::{AcquisitionPlan, DirectionList, Optics, SourceCalibration,
    ///         SpectralAcquisitionPlan, SpectralChannel, SpectralGeometry},
    ///     measurements::MeasurementStack,
    ///     model::{ObjectCoupling, ReconstructionShape, SpectralImagePlaneModel},
    ///     reconstruction::SpectralReconstructionProblem,
    /// };
    /// # fn main() -> fpm_rs::Result<()> {
    /// let channels: Vec<_> = [450e-9, 630e-9].into_iter().enumerate().map(|(index, wavelength)| {
    ///     Ok(SpectralChannel {
    ///         channel_id: format!("channel-{index}"),
    ///         optics: Optics {
    ///             wavelength_vacuum_m: wavelength, objective_na: 0.1,
    ///             magnification: 4.0, camera_pixel_size: 6.5e-6,
    ///             illumination_refractive_index: 1.0,
    ///             objective_medium_refractive_index: 1.0,
    ///             defocus_distance: None, pupil_aberration: None,
    ///         },
    ///         calibration: SourceCalibration::unity(),
    ///         acquisition: AcquisitionPlan::all_sources(1)?,
    ///     })
    /// }).collect::<fpm_rs::Result<_>>()?;
    /// let geometry = SpectralGeometry::Shared(
    ///     DirectionList::from_direction_cosines(vec![[0.0, 0.0]])?.into(),
    /// );
    /// let model = SpectralImagePlaneModel::from_experiment(
    ///     &channels, &geometry, SpectralAcquisitionPlan::separate(&[1, 1])?,
    ///     (4, 4), ReconstructionShape::Smooth, ObjectCoupling::Independent,
    /// )?;
    /// let measurements = MeasurementStack::from_vec(vec![1.0; 2 * 4 * 4], (4, 4), vec![])?;
    /// let problem = SpectralReconstructionProblem::new(measurements, model)?;
    /// let result = SpectralAlternatingProjection::default().iterations(2).run(&problem)?;
    /// assert_eq!(result.channels.len(), 2);
    /// let continued = fpm_rs::reconstruction::SpectralRunner::new(
    ///     SpectralAlternatingProjection::default().iterations(3),
    /// ).with_checkpoint(result.checkpoint.unwrap()).run(&problem)?;
    /// assert_eq!(continued.runtime.completed_iterations, 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_experiment(
        channels: &[SpectralChannel],
        geometry: &SpectralGeometry,
        acquisition: SpectralAcquisitionPlan,
        image_shape: (usize, usize),
        reconstruction_shape: ReconstructionShape,
        object_coupling: ObjectCoupling,
    ) -> Result<Self> {
        if channels.is_empty() {
            return Err(invalid(
                "spectral_channels",
                "at least one channel is required",
            ));
        }
        let geometries = match geometry {
            SpectralGeometry::Shared(SourceGeometry::KVectors(_)) => {
                return Err(invalid(
                    "spectral_geometry",
                    "wavelength-specific KVectorList cannot be shared physical geometry",
                ));
            }
            SpectralGeometry::Shared(value) => vec![value; channels.len()],
            SpectralGeometry::PerChannel(values) if values.len() == channels.len() => {
                values.iter().collect()
            }
            SpectralGeometry::PerChannel(_) => {
                return Err(invalid(
                    "spectral_geometry",
                    "requires one geometry per channel",
                ));
            }
        };
        let mut resolved = Vec::with_capacity(channels.len());
        let mut bounds = Vec::with_capacity(channels.len());
        let pitch = channels[0].optics.object_pixel_size();
        for (channel, geometry) in channels.iter().zip(geometries) {
            channel.optics.validate()?;
            if !same_pitch(pitch, channel.optics.object_pixel_size()) {
                return Err(invalid(
                    "object_plane_pixel_pitch_m",
                    "spectral channels require the same detector sampling; resampling is not implemented",
                ));
            }
            let illumination = Illumination::new(
                geometry.clone(),
                channel.calibration.clone(),
                channel.acquisition.clone(),
            )
            .resolve(&channel.optics)?;
            bounds.push(ImagePlaneModel::crop_displacement_bounds(
                &channel.optics,
                image_shape,
                illumination.k_vectors(),
            )?);
            resolved.push(illumination);
        }
        let common_shape = ImagePlaneModel::resolve_reconstruction_shape(
            image_shape,
            reconstruction_shape,
            &bounds,
        )?;
        let kernels = channels
            .iter()
            .zip(&resolved)
            .map(|(channel, illumination)| {
                Ok(SpectralModelChannel {
                    channel_id: channel.channel_id.clone(),
                    model: ImagePlaneModel::compile(
                        &channel.optics,
                        illumination,
                        image_shape,
                        ReconstructionShape::Exact(common_shape),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let model = Self {
            channels: kernels,
            acquisition,
            object_coupling,
        };
        model.validate()?;
        Ok(model)
    }

    /// Assembles validated wavelength kernels already compiled on one common grid.
    /// Used by external dataset converters; rejects missing wavelength metadata,
    /// duplicate IDs/wavelengths, incompatible sampling, and uncovered local frames.
    pub fn from_compiled_channels(
        channels: Vec<SpectralModelChannel>,
        acquisition: SpectralAcquisitionPlan,
        object_coupling: ObjectCoupling,
    ) -> Result<Self> {
        let model = Self {
            channels,
            acquisition,
            object_coupling,
        };
        model.validate()?;
        Ok(model)
    }

    /// Reports the collapsed detector-exposure/channel mode-weight matrix.
    /// Each entry sums detector gain × spectral weight × local frame gain ×
    /// local incoherent source-weight sum. Background and measurement masks/weights
    /// are excluded. Different crops and pupils remain separate forward operators;
    /// matrix rank alone does not establish nonlinear reconstruction identifiability.
    ///
    /// `relative_tolerance` must be finite and nonnegative; omission uses
    /// `max(exposures, channels) * f64::EPSILON`. Rank counts singular values
    /// strictly above `largest * relative_tolerance`. Singular values use a scaled
    /// cyclic one-sided Jacobi SVD, transposing wide matrices. Nonconvergence fails.
    ///
    /// # References
    /// NumPy, [matrix_rank documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html),
    /// for the default numerical-rank threshold. LAPACK 3.12.1,
    /// [DGESVJ documentation](https://www.netlib.org/lapack/explore-html/d9/deb/group__gesvj_ga7aec05d2a1523bbeee77ece21b12187c.html),
    /// for one-sided Jacobi SVD; this crate implements its own cyclic rotations.
    pub fn mixing_diagnostics(
        &self,
        relative_tolerance: Option<f64>,
    ) -> Result<super::SpectralMixingDiagnostics> {
        self.validate()?;
        let mut matrix = Array2::zeros((self.frame_count(), self.channels.len()));
        for (row_index, row) in self.acquisition.frames().iter().enumerate() {
            for contribution in &row.contributions {
                let kernel = &self.channels[contribution.channel].model;
                matrix[[row_index, contribution.channel]] += row.gain
                    * contribution.spectral_weight
                    * kernel.frame_gain(contribution.local_frame)?
                    * local_weight_sum(kernel, contribution.local_frame);
            }
        }
        super::mixing::diagnose(
            self.channels.iter().map(|c| c.channel_id.clone()).collect(),
            matrix,
            relative_tolerance,
        )
    }

    /// Borrows the ordered channel metadata and ordinary numerical kernels.
    pub fn channels(&self) -> &[SpectralModelChannel] {
        &self.channels
    }

    /// Borrows the canonical sparse physical detector plan.
    pub fn acquisition(&self) -> &SpectralAcquisitionPlan {
        &self.acquisition
    }

    /// Returns the explicitly chosen complex-object sharing rule.
    pub fn object_coupling(&self) -> ObjectCoupling {
        self.object_coupling
    }

    /// Returns the number of independently stored object spectra (one when shared).
    pub fn object_count(&self) -> usize {
        match self.object_coupling {
            ObjectCoupling::Independent => self.channels.len(),
            ObjectCoupling::SharedComplex => 1,
        }
    }

    /// Returns the stored object index used by a given channel.
    pub fn object_index(&self, channel: usize) -> Result<usize> {
        if channel >= self.channels.len() {
            return Err(invalid("channel", "channel index is out of range"));
        }
        Ok(match self.object_coupling {
            ObjectCoupling::Independent => channel,
            ObjectCoupling::SharedComplex => 0,
        })
    }

    /// Returns the common low-resolution detector `(height, width)`.
    pub fn image_shape(&self) -> (usize, usize) {
        self.channels[0].model.image_shape()
    }

    /// Returns the common high-resolution object `(height, width)`.
    pub fn reconstruction_shape(&self) -> (usize, usize) {
        self.channels[0].model.reconstruction_shape()
    }

    /// Returns the physical detector frame count.
    pub fn frame_count(&self) -> usize {
        self.acquisition.frame_count()
    }

    /// Validates common sampling, unique IDs/wavelengths, references, and local coverage.
    pub fn validate(&self) -> Result<()> {
        let first = self
            .channels
            .first()
            .ok_or_else(|| invalid("spectral_channels", "at least one channel is required"))?;
        let mut ids = BTreeSet::new();
        let mut wavelengths = Vec::new();
        let mut covered = Vec::new();
        for channel in &self.channels {
            channel.model.validate()?;
            if channel.channel_id.trim().is_empty() || !ids.insert(&channel.channel_id) {
                return Err(invalid(
                    "channel_id",
                    "channel IDs must be non-empty and unique",
                ));
            }
            let wavelength = channel.model.sampling().wavelength.ok_or_else(|| {
                invalid(
                    "wavelength_vacuum_m",
                    "every channel must retain wavelength metadata",
                )
            })?;
            if !wavelength.is_finite() || wavelength <= 0.0 || wavelengths.contains(&wavelength) {
                return Err(invalid(
                    "wavelength_vacuum_m",
                    "wavelengths must be finite, positive, and distinct",
                ));
            }
            wavelengths.push(wavelength);
            if channel.model.image_shape() != first.model.image_shape()
                || channel.model.reconstruction_shape() != first.model.reconstruction_shape()
                || !same_pitch(
                    channel.model.sampling().low_res_pixel_size,
                    first.model.sampling().low_res_pixel_size,
                )
            {
                return Err(invalid(
                    "spectral_sampling",
                    "all channel kernels must share detector shape, reconstruction grid, and object-plane pixel pitch",
                ));
            }
            if channel.model.background().is_some() {
                return Err(invalid(
                    "spectral_background",
                    "background belongs to physical detector rows, not channel kernels",
                ));
            }
            covered.push(vec![false; channel.model.frame_count()]);
        }
        for row in self.acquisition.frames() {
            let mut mode_weight_sum = 0.0;
            for contribution in &row.contributions {
                let local = covered
                    .get_mut(contribution.channel)
                    .and_then(|frames| frames.get_mut(contribution.local_frame))
                    .ok_or_else(|| {
                        invalid(
                            "spectral_acquisition",
                            "channel/local_frame reference is out of range",
                        )
                    })?;
                *local = true;
                let kernel = &self.channels[contribution.channel].model;
                mode_weight_sum += contribution.spectral_weight
                    * kernel.frame_gain(contribution.local_frame)?
                    * local_weight_sum(kernel, contribution.local_frame);
            }
            if !mode_weight_sum.is_finite()
                || mode_weight_sum <= 0.0
                || !(row.gain * mode_weight_sum).is_finite()
                || row.gain * mode_weight_sum <= 0.0
            {
                return Err(invalid(
                    "spectral_weight",
                    "effective source-mode weight sum must be finite and positive",
                ));
            }
        }
        if covered.iter().flatten().any(|value| !value) {
            return Err(invalid(
                "spectral_acquisition",
                "every channel and local frame must participate in the detector plan",
            ));
        }
        Ok(())
    }

    /// Evaluates a detector frame by calling each canonical single-channel forward model.
    ///
    /// `spectra` contains centered, standard-layout common-grid spectra in stored
    /// object order, with the core's `1/(height * width)` forward FFT normalization:
    /// one per channel for independent objects, one for a shared
    /// object. Returns an owned standard-layout intensity image. Geometry, pupil
    /// sampling, signs, and subpixel interpolation use the ordinary kernels.
    pub fn forward_intensity(
        &self,
        spectra: &[ArrayView2<'_, Complex64>],
        frame: usize,
    ) -> Result<Array2<f64>> {
        self.validate_spectra(spectra)?;
        let row = self
            .acquisition
            .frames()
            .get(frame)
            .ok_or(Error::FrameOutOfRange {
                index: frame,
                frames: self.frame_count(),
            })?;
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(
            self.image_shape(),
            self.reconstruction_shape(),
        )?);
        let mut result = Array2::zeros(self.image_shape());
        for contribution in &row.contributions {
            let kernel = &self.channels[contribution.channel].model;
            let forward = ForwardModel::with_backend(kernel, backend.clone())?;
            let local = forward.forward_intensity(
                spectra[self.object_index(contribution.channel)?],
                kernel.pupil(),
                contribution.local_frame,
            )?;
            result.scaled_add(contribution.spectral_weight, &local);
        }
        result.mapv_inplace(|intensity| row.gain * intensity + row.background);
        if result.iter().any(|value| !value.is_finite()) {
            return Err(Error::Numerical(
                "spectral prediction contains non-finite intensities".into(),
            ));
        }
        Ok(result)
    }

    pub(crate) fn validate_spectra(&self, spectra: &[ArrayView2<'_, Complex64>]) -> Result<()> {
        if spectra.len() != self.object_count() {
            return Err(invalid(
                "object_spectra",
                "spectrum count differs from object coupling",
            ));
        }
        for spectrum in spectra {
            crate::array_layout::StandardView2::try_from(*spectrum)?;
            if spectrum.dim() != self.reconstruction_shape() {
                return Err(Error::InvalidShape(
                    "spectral object spectrum differs from common grid".into(),
                ));
            }
            if spectrum
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(invalid("object_spectra", "complex values must be finite"));
            }
        }
        Ok(())
    }
}

pub(crate) fn local_weight_sum(model: &ImagePlaneModel, frame: usize) -> f64 {
    model.multiplexing_matrix().map_or(1.0, |matrix| {
        matrix[frame].iter().map(|(_, weight)| *weight).sum()
    })
}

fn same_pitch(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * left.abs().max(right.abs())
}

fn invalid(name: &'static str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name,
        reason: reason.into(),
    }
}
