use ndarray::{ArrayView2, ArrayViewMut2};
use num_complex::Complex64;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    Result,
    array_layout::{StandardView2, checked_len_2d},
    error::Error,
    experiment::{
        AcquisitionPlan, Illumination, KVector, MultiplexingMatrix, Optics, ResolvedIllumination,
    },
};

use super::{CropIndices, FourierCrop, FourierOffset, Pupil, Sampling};

/// Selects an explicit reconstruction grid or an automatic sizing strategy.
///
/// Automatic shapes preserve the low-resolution aspect ratio, so the recovered
/// object has the same pixel size in both axes. [`Self::Smooth`] and
/// [`Self::PowerOfTwo`] round the shared reduced-aspect-ratio multiplier rather
/// than each dimension independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconstructionShape {
    /// Use the supplied concrete `(height, width)` after validating it.
    Exact((usize, usize)),
    /// Use the smallest grid containing every Fourier crop and interpolation
    /// stencil.
    Minimum,
    /// Round the minimum shared multiplier up to a value whose prime factors
    /// are limited to 2, 3, 5, and 7.
    Smooth,
    /// Round the minimum shared multiplier up to a power of two.
    PowerOfTwo,
}

#[derive(Clone, Copy, Debug)]
struct GridShift {
    integer: isize,
    offset: f64,
    lower: isize,
    upper: isize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CropDisplacementBounds {
    minimum_row: isize,
    maximum_row: isize,
    minimum_column: isize,
    maximum_column: isize,
}

/// Algorithm-facing image-plane FPM model. It contains no LED or camera geometry.
///
/// Source-indexed vectors, crops, and optional subpixel offsets describe individual
/// illuminations. Acquisition-frame-indexed gains, background, and multiplexing describe
/// measured frames. Shapes are `(height, width)` and stored arrays own their data.
#[derive(Clone, Debug, Serialize)]
pub struct ImagePlaneModel {
    /// One transverse wave vector per illumination source.
    pub(crate) k_vectors: Vec<KVector>,
    pub(crate) pupil: Pupil,
    pub(crate) crop_indices: CropIndices,
    /// Fractional `(row, column)` Fourier-grid offsets relative to each crop.
    #[serde(default)]
    pub(crate) subpixel_offsets: Option<Vec<FourierOffset>>,
    pub(crate) sampling: Sampling,
    pub(crate) image_shape: (usize, usize),
    pub(crate) reconstruction_shape: (usize, usize),
    pub(crate) frame_gains: Option<Vec<f64>>,
    pub(crate) background: Option<Vec<f64>>,
    /// Optional measured-frame rows of `(source_index, incoherent_weight)`.
    pub(crate) multiplexing_matrix: Option<MultiplexingMatrix>,
}

impl<'de> Deserialize<'de> for ImagePlaneModel {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Representation {
            k_vectors: Vec<KVector>,
            pupil: Pupil,
            crop_indices: CropIndices,
            #[serde(default)]
            subpixel_offsets: Option<Vec<FourierOffset>>,
            sampling: Sampling,
            image_shape: (usize, usize),
            reconstruction_shape: (usize, usize),
            frame_gains: Option<Vec<f64>>,
            background: Option<Vec<f64>>,
            multiplexing_matrix: Option<MultiplexingMatrix>,
        }

        let representation = Representation::deserialize(deserializer)?;
        let model = Self {
            k_vectors: representation.k_vectors,
            pupil: representation.pupil,
            crop_indices: representation.crop_indices,
            subpixel_offsets: representation.subpixel_offsets,
            sampling: representation.sampling,
            image_shape: representation.image_shape,
            reconstruction_shape: representation.reconstruction_shape,
            frame_gains: representation.frame_gains,
            background: representation.background,
            multiplexing_matrix: representation.multiplexing_matrix,
        };
        model.validate().map_err(D::Error::custom)?;
        Ok(model)
    }
}

impl ImagePlaneModel {
    /// Saves this validated compiled kernel as float-roundtrip JSON.
    /// Intended for external dataset converters; overwrites the destination and
    /// synchronizes file contents. Does not create parent directories or access the network.
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.validate()?;
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(&file, self)?;
        file.sync_all()?;
        Ok(())
    }
    /// Loads and validates a compiled kernel JSON entirely from local files.
    /// Preserves explicit sampling metadata, source weights, crops and fixed pupil.
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let model: Self =
            serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))?;
        model.validate()?;
        Ok(model)
    }

    /// Builds and validates a non-multiplexed model from explicitly compiled components.
    ///
    /// `k_vectors` and `crop_indices` must have equal non-zero source counts. `pupil`
    /// and each crop must have `image_shape`; `sampling` and `reconstruction_shape`
    /// must be mutually consistent.
    pub fn new(
        k_vectors: Vec<KVector>,
        pupil: Pupil,
        crop_indices: CropIndices,
        sampling: Sampling,
        image_shape: (usize, usize),
        reconstruction_shape: (usize, usize),
    ) -> Result<Self> {
        let model = Self {
            k_vectors,
            pupil,
            crop_indices,
            subpixel_offsets: None,
            sampling,
            image_shape,
            reconstruction_shape,
            frame_gains: None,
            background: None,
            multiplexing_matrix: None,
        };
        model.validate()?;
        Ok(model)
    }

    /// Compiles an experiment using an explicit or automatically selected
    /// reconstruction shape.
    pub fn from_experiment(
        optics: &Optics,
        illumination: &Illumination,
        image_shape: (usize, usize),
        reconstruction_shape: ReconstructionShape,
    ) -> Result<Self> {
        optics.validate()?;
        let resolved = illumination.resolve(optics)?;
        Self::compile(optics, &resolved, image_shape, reconstruction_shape)
    }

    /// Resolves an explicit shape or suggests an automatic reconstruction grid
    /// from the actual illumination wave vectors.
    pub fn suggest_reconstruction_shape(
        optics: &Optics,
        illumination: &Illumination,
        image_shape: (usize, usize),
        reconstruction_shape: ReconstructionShape,
    ) -> Result<(usize, usize)> {
        optics.validate()?;
        let resolved = illumination.resolve(optics)?;
        let bounds = Self::crop_displacement_bounds(optics, image_shape, resolved.k_vectors())?;
        Self::resolve_reconstruction_shape(image_shape, reconstruction_shape, &[bounds])
    }

    /// Compiles already resolved illumination into algorithm-facing numerical state.
    ///
    /// Geometry positions and poses are intentionally not retained. Sparse frame
    /// weights are multiplied by stable source powers during compilation.
    pub fn compile(
        optics: &Optics,
        illumination: &ResolvedIllumination,
        image_shape: (usize, usize),
        reconstruction_shape: ReconstructionShape,
    ) -> Result<Self> {
        optics.validate()?;
        let k_vectors = illumination.k_vectors().to_vec();
        let bounds = Self::crop_displacement_bounds(optics, image_shape, &k_vectors)?;
        let reconstruction_shape =
            Self::resolve_reconstruction_shape(image_shape, reconstruction_shape, &[bounds])?;
        let scale = reconstruction_shape.1 as f64 / image_shape.1 as f64;
        let low_res_pixel_size = optics.object_pixel_size();
        let mut sampling = Sampling::new(
            low_res_pixel_size,
            low_res_pixel_size / scale,
            std::f64::consts::TAU / (image_shape.1 as f64 * low_res_pixel_size),
            std::f64::consts::TAU / (image_shape.0 as f64 * low_res_pixel_size),
        )?;
        sampling.wavelength = Some(optics.wavelength_vacuum_m);
        let maximum_illumination_na = k_vectors
            .iter()
            .map(|vector| {
                vector.kx.hypot(vector.ky) * optics.wavelength_vacuum_m / std::f64::consts::TAU
            })
            .fold(0.0, f64::max);
        sampling.synthetic_na = Some(optics.objective_na + maximum_illumination_na);
        let pupil = Pupil::circular(image_shape, &sampling, optics)?;
        let mut offsets = Vec::with_capacity(k_vectors.len());
        let crops = k_vectors
            .iter()
            .map(|vector| {
                let (crop, offset) =
                    Self::crop_for_k_vector(vector, &sampling, image_shape, reconstruction_shape)?;
                offsets.push(offset);
                Ok(crop)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut model = Self::new(
            k_vectors,
            pupil,
            CropIndices::new(crops),
            sampling,
            image_shape,
            reconstruction_shape,
        )?;
        model.frame_gains = Some(illumination.frame_gains());
        let matrix = illumination.compiled_multiplexing_matrix();
        let identity = matrix.len() == model.source_count()
            && matrix
                .iter()
                .enumerate()
                .all(|(source, row)| row.as_slice() == [(source, 1.0)]);
        model.multiplexing_matrix = (!identity).then_some(matrix);
        model.subpixel_offsets = Some(offsets);
        model.validate()?;
        Ok(model)
    }

    pub(crate) fn crop_displacement_bounds(
        optics: &Optics,
        image_shape: (usize, usize),
        k_vectors: &[KVector],
    ) -> Result<CropDisplacementBounds> {
        optics.validate()?;
        validate_image_shape(image_shape)?;
        if k_vectors.is_empty() {
            return Err(Error::InvalidModel(
                "illumination must contain at least one frame".into(),
            ));
        }
        let low_res_pixel_size = optics.object_pixel_size();
        let dkx = std::f64::consts::TAU / (image_shape.1 as f64 * low_res_pixel_size);
        let dky = std::f64::consts::TAU / (image_shape.0 as f64 * low_res_pixel_size);
        let mut bounds = CropDisplacementBounds {
            minimum_row: isize::MAX,
            maximum_row: isize::MIN,
            minimum_column: isize::MAX,
            maximum_column: isize::MIN,
        };
        for vector in k_vectors {
            let row = checked_grid_shift(vector.ky / dky)?;
            let column = checked_grid_shift(vector.kx / dkx)?;
            bounds.minimum_row = bounds.minimum_row.min(row.lower);
            bounds.maximum_row = bounds.maximum_row.max(row.upper);
            bounds.minimum_column = bounds.minimum_column.min(column.lower);
            bounds.maximum_column = bounds.maximum_column.max(column.upper);
        }
        Ok(bounds)
    }

    pub(crate) fn resolve_reconstruction_shape(
        image_shape: (usize, usize),
        reconstruction_shape: ReconstructionShape,
        bounds: &[CropDisplacementBounds],
    ) -> Result<(usize, usize)> {
        validate_image_shape(image_shape)?;
        if bounds.is_empty() {
            return Err(Error::InvalidModel(
                "at least one set of illumination crop bounds is required".into(),
            ));
        }
        match reconstruction_shape {
            ReconstructionShape::Exact(shape) => {
                validate_reconstruction_aspect(image_shape, shape)?;
                if !shape_contains_bounds(image_shape, shape, bounds)? {
                    return Err(Error::InvalidShape(format!(
                        "reconstruction shape {shape:?} does not contain every illumination crop and subpixel interpolation stencil"
                    )));
                }
                Ok(shape)
            }
            policy => {
                let divisor = greatest_common_divisor(image_shape.0, image_shape.1);
                let aspect_height = image_shape.0 / divisor;
                let aspect_width = image_shape.1 / divisor;
                let maximum_multiplier =
                    (isize::MAX as usize / aspect_height).min(isize::MAX as usize / aspect_width);
                let minimum_multiplier = minimum_fitting_multiplier(
                    image_shape,
                    (aspect_height, aspect_width),
                    divisor,
                    maximum_multiplier,
                    bounds,
                )?;
                let multiplier = match policy {
                    ReconstructionShape::Minimum => minimum_multiplier,
                    ReconstructionShape::Smooth => next_smooth_multiplier(minimum_multiplier)
                        .filter(|&value| value <= maximum_multiplier)
                        .ok_or_else(|| {
                            Error::InvalidShape(
                                "no supported 2/3/5/7-smooth reconstruction multiplier exists"
                                    .into(),
                            )
                        })?,
                    ReconstructionShape::PowerOfTwo => minimum_multiplier
                        .checked_next_power_of_two()
                        .filter(|&value| value <= maximum_multiplier)
                        .ok_or_else(|| {
                            Error::InvalidShape(
                                "no supported power-of-two reconstruction multiplier exists".into(),
                            )
                        })?,
                    ReconstructionShape::Exact(shape) => {
                        return Err(Error::InvalidShape(format!(
                            "unexpected exact reconstruction shape {shape:?} during automatic sizing"
                        )));
                    }
                };
                candidate_shape((aspect_height, aspect_width), multiplier)
            }
        }
    }

    /// Returns the number of individual illumination sources.
    pub fn source_count(&self) -> usize {
        self.k_vectors.len()
    }

    /// Returns the acquisition-frame count, which can differ when sources are multiplexed.
    pub fn frame_count(&self) -> usize {
        self.multiplexing_matrix
            .as_ref()
            .map_or_else(|| self.source_count(), Vec::len)
    }

    /// Returns whether acquisition frames contain incoherent combinations of sources.
    pub fn is_multiplexed(&self) -> bool {
        self.multiplexing_matrix
            .as_ref()
            .is_some_and(|matrix| matrix.iter().any(|row| row.len() > 1))
    }

    /// Borrows transverse wave vectors in source order, in radians per metre.
    pub fn k_vectors(&self) -> &[KVector] {
        &self.k_vectors
    }

    /// Borrows the low-resolution complex pupil and binary support.
    pub fn pupil(&self) -> &Pupil {
        &self.pupil
    }

    /// Mutably borrows the pupil; callers must preserve its shape and support invariants.
    pub fn pupil_mut(&mut self) -> &mut Pupil {
        &mut self.pupil
    }

    /// Borrows integer Fourier crops in individual source order.
    pub fn crop_indices(&self) -> &CropIndices {
        &self.crop_indices
    }

    /// Borrows optional fractional `(row, column)` offsets in source order.
    pub fn subpixel_offsets(&self) -> Option<&[FourierOffset]> {
        self.subpixel_offsets.as_deref()
    }

    /// Borrows real- and Fourier-space sampling metadata.
    pub const fn sampling(&self) -> &Sampling {
        &self.sampling
    }

    /// Returns low-resolution detector shape as `(height, width)`.
    pub const fn image_shape(&self) -> (usize, usize) {
        self.image_shape
    }

    /// Returns high-resolution object shape as `(height, width)`.
    pub const fn reconstruction_shape(&self) -> (usize, usize) {
        self.reconstruction_shape
    }

    /// Borrows optional non-negative multiplicative gains in acquisition-frame order.
    pub fn frame_gains(&self) -> Option<&[f64]> {
        self.frame_gains.as_deref()
    }

    /// Borrows optional non-negative optical intensity background.
    ///
    /// Length is one low-resolution frame (broadcast) or a complete row-major
    /// `(frame, row, column)` stack.
    pub fn background(&self) -> Option<&[f64]> {
        self.background.as_deref()
    }

    /// Borrows optional acquisition-frame rows of non-negative source weights.
    pub fn multiplexing_matrix(&self) -> Option<&MultiplexingMatrix> {
        self.multiplexing_matrix.as_ref()
    }

    /// Replaces gains, requiring one finite non-negative value per acquisition frame.
    pub fn with_frame_gains(mut self, values: Option<Vec<f64>>) -> Result<Self> {
        self.frame_gains = values;
        self.validate()?;
        Ok(self)
    }

    /// Refreshes only source powers, sparse acquisition weights, and frame gains.
    ///
    /// Physical positions, propagation vectors, Fourier crops, subpixel offsets,
    /// pupil samples, sampling metadata, and reconstruction shapes are retained.
    /// This is the intensity-only update boundary used by physical illumination
    /// calibration.
    pub fn update_intensity_calibration(
        &mut self,
        source_power: &[f64],
        acquisition: &AcquisitionPlan,
    ) -> Result<()> {
        if source_power.len() != self.source_count()
            || source_power
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "source_power",
                reason: format!(
                    "must contain {} finite non-negative values",
                    self.source_count()
                ),
            });
        }
        if acquisition.frame_count() != self.frame_count() {
            return Err(Error::InvalidParameter {
                name: "acquisition",
                reason: format!(
                    "frame count {} differs from compiled frame count {}",
                    acquisition.frame_count(),
                    self.frame_count()
                ),
            });
        }
        let mut matrix = Vec::with_capacity(acquisition.frame_count());
        let mut gains = Vec::with_capacity(acquisition.frame_count());
        for frame in acquisition.frames() {
            let mut row = Vec::with_capacity(frame.contributions.len());
            for contribution in &frame.contributions {
                let power = source_power.get(contribution.source).ok_or_else(|| {
                    Error::InvalidParameter {
                        name: "acquisition",
                        reason: format!(
                            "source index {} is outside {} compiled sources",
                            contribution.source,
                            self.source_count()
                        ),
                    }
                })?;
                let weight = contribution.intensity_weight * power;
                if !weight.is_finite() || weight < 0.0 {
                    return Err(Error::InvalidParameter {
                        name: "acquisition",
                        reason: "compiled source weights must be finite and non-negative".into(),
                    });
                }
                if weight != 0.0 {
                    row.push((contribution.source, weight));
                }
            }
            if row.is_empty() {
                return Err(Error::InvalidParameter {
                    name: "acquisition",
                    reason: "every calibrated frame must retain positive source weight".into(),
                });
            }
            matrix.push(row);
            gains.push(frame.gain);
        }
        let identity = matrix.len() == self.source_count()
            && matrix
                .iter()
                .enumerate()
                .all(|(source, row)| row.as_slice() == [(source, 1.0)]);
        let previous_matrix =
            std::mem::replace(&mut self.multiplexing_matrix, (!identity).then_some(matrix));
        let previous_gains = self.frame_gains.replace(gains);
        if let Err(error) = self.validate() {
            self.multiplexing_matrix = previous_matrix;
            self.frame_gains = previous_gains;
            return Err(error);
        }
        Ok(())
    }

    /// Recompiles illumination-dependent geometry into the existing numerical grid.
    ///
    /// The low- and high-resolution shapes, optical sampling, pupil values,
    /// background, and other static model state are retained. A geometry that
    /// would move a crop outside the existing reconstruction grid is rejected.
    pub fn update_illumination_geometry(
        &mut self,
        optics: &Optics,
        illumination: &Illumination,
    ) -> Result<()> {
        let resolved = illumination.resolve(optics)?;
        if resolved.source_count() != self.source_count()
            || resolved.frame_count() != self.frame_count()
        {
            return Err(Error::InvalidModel(
                "calibrated illumination must preserve source and frame counts".into(),
            ));
        }
        let k_vectors = resolved.k_vectors().to_vec();
        let mut offsets = Vec::with_capacity(k_vectors.len());
        let crops = k_vectors
            .iter()
            .map(|vector| {
                let (crop, offset) = Self::crop_for_k_vector(
                    vector,
                    &self.sampling,
                    self.image_shape,
                    self.reconstruction_shape,
                )?;
                offsets.push(offset);
                Ok(crop)
            })
            .collect::<Result<Vec<_>>>()?;

        let previous_vectors = std::mem::replace(&mut self.k_vectors, k_vectors);
        let previous_crops = std::mem::replace(&mut self.crop_indices, CropIndices::new(crops));
        let previous_offsets = self.subpixel_offsets.replace(offsets);
        let previous_synthetic_na = self.sampling.synthetic_na;
        self.sampling.synthetic_na = Some(
            optics.objective_na
                + self
                    .k_vectors
                    .iter()
                    .map(|vector| {
                        vector.kx.hypot(vector.ky) * optics.wavelength_vacuum_m
                            / std::f64::consts::TAU
                    })
                    .fold(0.0, f64::max),
        );
        if let Err(error) =
            self.update_intensity_calibration(resolved.source_power(), illumination.acquisition())
        {
            self.k_vectors = previous_vectors;
            self.crop_indices = previous_crops;
            self.subpixel_offsets = previous_offsets;
            self.sampling.synthetic_na = previous_synthetic_na;
            return Err(error);
        }
        self.validate()
    }

    /// Replaces the source wave vectors while preserving the compiled crops.
    ///
    /// This is intended for calibrated models whose replacement vectors use
    /// the same Fourier sampling and source ordering. The full model invariant
    /// set is revalidated before the replacement is committed.
    pub fn replace_k_vectors(&mut self, values: Vec<KVector>) -> Result<()> {
        let previous = std::mem::replace(&mut self.k_vectors, values);
        if let Err(error) = self.validate() {
            self.k_vectors = previous;
            return Err(error);
        }
        Ok(())
    }

    /// Replaces optical intensity background with a broadcast frame, full stack, or `None`.
    pub fn with_background(mut self, values: Option<Vec<f64>>) -> Result<Self> {
        self.background = values;
        self.validate()?;
        Ok(self)
    }

    /// Sets compiled sparse acquisition with one non-empty non-negative row per frame.
    pub fn with_multiplexing(mut self, matrix: MultiplexingMatrix) -> Result<Self> {
        self.multiplexing_matrix = Some(matrix);
        self.validate()?;
        Ok(self)
    }

    /// Sets one finite fractional Fourier-grid offset per individual source.
    pub fn with_subpixel_offsets(mut self, offsets: Vec<FourierOffset>) -> Result<Self> {
        self.subpixel_offsets = Some(offsets);
        self.validate()?;
        Ok(self)
    }

    /// Returns a source's fractional Fourier-grid offset, or zero when absent.
    pub fn source_offset(&self, source: usize) -> Result<FourierOffset> {
        self.crop_indices.get(source)?;
        match &self.subpixel_offsets {
            None => Ok(FourierOffset::default()),
            Some(offsets) => offsets.get(source).copied().ok_or_else(|| {
                Error::InvalidModel("subpixel offset count does not match source count".into())
            }),
        }
    }

    /// Extracts one source patch from a standard-layout high-resolution object spectrum.
    ///
    /// `destination` has `image_shape.0 * image_shape.1` row-major complex values.
    /// Fractional offsets use bilinear Fourier-grid interpolation.
    pub fn extract_patch(
        &self,
        object_spectrum: ArrayView2<'_, Complex64>,
        source: usize,
        destination: &mut [Complex64],
    ) -> Result<()> {
        let object_spectrum = StandardView2::try_from(object_spectrum)?;
        self.extract_patch_standard(object_spectrum, source, destination)
    }

    pub(crate) fn extract_patch_standard(
        &self,
        object_spectrum: StandardView2<'_, Complex64>,
        source: usize,
        destination: &mut [Complex64],
    ) -> Result<()> {
        self.extract_patch_at_offset(
            object_spectrum,
            source,
            self.source_offset(source)?,
            destination,
        )
    }

    pub(crate) fn extract_patch_at_offset(
        &self,
        object_spectrum: StandardView2<'_, Complex64>,
        source: usize,
        offset: FourierOffset,
        destination: &mut [Complex64],
    ) -> Result<()> {
        if object_spectrum.dim() != self.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "object spectrum shape {:?} does not match {:?}",
                object_spectrum.dim(),
                self.reconstruction_shape
            )));
        }
        let crop = self.crop_indices.get(source)?;
        crop.extract_subpixel_standard(object_spectrum, destination, offset)
    }

    /// Adds an update into a high-resolution spectrum through the adjoint crop operator.
    pub fn insert_patch_adjoint(
        &self,
        destination: ArrayViewMut2<'_, Complex64>,
        source: usize,
        update: &[Complex64],
        scale: f64,
    ) -> Result<()> {
        let destination = crate::array_layout::StandardViewMut2::try_from(destination)?;
        self.insert_patch_adjoint_standard(destination, source, update, scale)
    }

    pub(crate) fn insert_patch_adjoint_standard(
        &self,
        destination: crate::array_layout::StandardViewMut2<'_, Complex64>,
        source: usize,
        update: &[Complex64],
        scale: f64,
    ) -> Result<()> {
        self.insert_patch_adjoint_at_offset(
            destination,
            source,
            update,
            scale,
            self.source_offset(source)?,
        )
    }

    pub(crate) fn insert_patch_adjoint_at_offset(
        &self,
        mut destination: crate::array_layout::StandardViewMut2<'_, Complex64>,
        source: usize,
        update: &[Complex64],
        scale: f64,
        offset: FourierOffset,
    ) -> Result<()> {
        if destination.dim() != self.reconstruction_shape {
            return Err(Error::InvalidShape(format!(
                "object spectrum shape {:?} does not match {:?}",
                destination.dim(),
                self.reconstruction_shape
            )));
        }
        let crop = self.crop_indices.get(source)?;
        crop.insert_subpixel_adjoint_slice(
            destination.as_slice_mut(),
            self.reconstruction_shape,
            update,
            scale,
            offset,
        )
    }

    pub(crate) fn insert_patch_adjoint_slice_at_offset(
        &self,
        destination: &mut [Complex64],
        source: usize,
        update: &[Complex64],
        scale: f64,
        offset: FourierOffset,
    ) -> Result<()> {
        let crop = self.crop_indices.get(source)?;
        crop.insert_subpixel_adjoint_slice(
            destination,
            self.reconstruction_shape,
            update,
            scale,
            offset,
        )
    }

    /// Checks that `source` exists and the interpolation stencil for `offset` is in bounds.
    pub fn validate_source_offset(&self, source: usize, offset: FourierOffset) -> Result<()> {
        self.crop_indices
            .get(source)?
            .validate_subpixel_inside(self.reconstruction_shape, offset)
    }

    pub(crate) fn crop_for_k_vector(
        vector: &KVector,
        sampling: &Sampling,
        image_shape: (usize, usize),
        reconstruction_shape: (usize, usize),
    ) -> Result<(FourierCrop, FourierOffset)> {
        let continuous_row = vector.ky / sampling.dky;
        let continuous_column = vector.kx / sampling.dkx;
        let row_shift = checked_grid_shift(continuous_row)?;
        let column_shift = checked_grid_shift(continuous_column)?;
        let reconstruction_center_row = isize::try_from(reconstruction_shape.0 / 2)
            .map_err(|_| Error::InvalidShape("reconstruction height is too large".into()))?;
        let reconstruction_center_column = isize::try_from(reconstruction_shape.1 / 2)
            .map_err(|_| Error::InvalidShape("reconstruction width is too large".into()))?;
        let image_half_height = isize::try_from(image_shape.0 / 2)
            .map_err(|_| Error::InvalidShape("image height is too large".into()))?;
        let image_half_width = isize::try_from(image_shape.1 / 2)
            .map_err(|_| Error::InvalidShape("image width is too large".into()))?;
        let start_row = reconstruction_center_row
            .checked_sub(image_half_height)
            .and_then(|value| value.checked_add(row_shift.integer));
        let start_column = reconstruction_center_column
            .checked_sub(image_half_width)
            .and_then(|value| value.checked_add(column_shift.integer));
        let (Some(start_row), Some(start_column)) = (start_row, start_column) else {
            return Err(Error::InvalidModel(format!(
                "illumination vector {vector:?} overflows the reconstruction grid"
            )));
        };
        if start_row < 0 || start_column < 0 {
            return Err(Error::InvalidModel(format!(
                "illumination vector {vector:?} produces a crop outside the reconstruction grid"
            )));
        }
        let crop = FourierCrop::new(
            usize::try_from(start_row).map_err(|_| {
                Error::InvalidModel("crop row cannot be represented as an index".into())
            })?,
            usize::try_from(start_column).map_err(|_| {
                Error::InvalidModel("crop column cannot be represented as an index".into())
            })?,
            image_shape.0,
            image_shape.1,
        );
        let offset = FourierOffset::new(row_shift.offset, column_shift.offset);
        crop.validate_subpixel_inside(reconstruction_shape, offset)?;
        Ok((crop, offset))
    }

    /// Returns a frame's positive gain, defaulting to `1.0` when gains are absent.
    pub fn frame_gain(&self, frame: usize) -> Result<f64> {
        if frame >= self.frame_count() {
            return Err(Error::FrameOutOfRange {
                index: frame,
                frames: self.frame_count(),
            });
        }
        Ok(self
            .frame_gains
            .as_ref()
            .map_or(1.0, |values| values[frame]))
    }

    /// Returns background intensity for a frame and row-major pixel index.
    ///
    /// Returns zero when background is absent and handles broadcast backgrounds.
    pub fn background_value(&self, frame: usize, pixel: usize) -> Result<f64> {
        if frame >= self.frame_count() {
            return Err(Error::FrameOutOfRange {
                index: frame,
                frames: self.frame_count(),
            });
        }
        let image_len = checked_len_2d(self.image_shape)?;
        if pixel >= image_len {
            return Err(Error::InvalidParameter {
                name: "pixel",
                reason: format!("index {pixel} is outside an image with {image_len} pixels"),
            });
        }
        Ok(self.background.as_ref().map_or(0.0, |values| {
            values[if values.len() == image_len {
                pixel
            } else {
                frame * image_len + pixel
            }]
        }))
    }

    /// Validates source/crop counts, shapes, pupil, sampling, vectors, interpolation
    /// bounds, gains, background, and optional multiplexing rows.
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        if self.k_vectors.is_empty() || self.source_count() != self.crop_indices.len() {
            return Err(Error::InvalidModel(format!(
                "{} source k-vectors and {} crops; counts must be equal and non-zero",
                self.source_count(),
                self.crop_indices.len()
            )));
        }
        if self
            .k_vectors
            .iter()
            .any(|vector| !vector.kx.is_finite() || !vector.ky.is_finite())
        {
            return Err(Error::InvalidModel(
                "k-vectors must contain finite values".into(),
            ));
        }
        if self.pupil.shape() != self.image_shape {
            return Err(Error::InvalidModel(format!(
                "pupil shape {:?} does not match image shape {:?}",
                self.pupil.shape(),
                self.image_shape
            )));
        }
        if self.pupil.support.len() != self.pupil.values.len()
            || self
                .pupil
                .values
                .as_slice()
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::InvalidModel(
                "pupil support or numeric values are invalid".into(),
            ));
        }
        if self.reconstruction_shape.0 < self.image_shape.0
            || self.reconstruction_shape.1 < self.image_shape.1
        {
            return Err(Error::InvalidModel(
                "reconstruction shape must contain a low-resolution crop".into(),
            ));
        }
        for crop in &self.crop_indices.crops {
            if (crop.height, crop.width) != self.image_shape {
                return Err(Error::InvalidModel(format!(
                    "crop shape {:?} does not match image shape {:?}",
                    (crop.height, crop.width),
                    self.image_shape
                )));
            }
            crop.validate_inside(self.reconstruction_shape)?;
        }
        if let Some(offsets) = &self.subpixel_offsets {
            if offsets.len() != self.source_count() {
                return Err(Error::InvalidModel(
                    "subpixel offset count does not match source count".into(),
                ));
            }
            for (crop, &offset) in self.crop_indices.crops.iter().zip(offsets) {
                crop.validate_subpixel_inside(self.reconstruction_shape, offset)?;
            }
        }
        if let Some(values) = &self.frame_gains {
            if values.len() != self.frame_count() {
                return Err(Error::InvalidModel(
                    "frame gain count does not match frame count".into(),
                ));
            }
            if values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(Error::InvalidModel(
                    "frame gains must be finite and non-negative".into(),
                ));
            }
        }
        let image_len = checked_len_2d(self.image_shape)?;
        let stack_len =
            image_len
                .checked_mul(self.frame_count())
                .ok_or_else(|| Error::ShapeOverflow {
                    shape: vec![self.frame_count(), self.image_shape.0, self.image_shape.1],
                })?;
        if self
            .background
            .as_ref()
            .is_some_and(|values| values.len() != image_len && values.len() != stack_len)
        {
            return Err(Error::InvalidModel(
                "background must be one image or one image per frame".into(),
            ));
        }
        if self
            .background
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::InvalidModel(
                "background values must be finite".into(),
            ));
        }
        if let Some(matrix) = &self.multiplexing_matrix {
            if matrix.is_empty() {
                return Err(Error::InvalidModel(
                    "multiplexing matrix must contain at least one measured frame".into(),
                ));
            }
            for (row_index, row) in matrix.iter().enumerate() {
                let mut seen = vec![false; self.source_count()];
                let invalid = row.is_empty()
                    || row.iter().any(|&(source, weight)| {
                        let duplicate = source < self.source_count() && seen[source];
                        if source < self.source_count() {
                            seen[source] = true;
                        }
                        source >= self.source_count()
                            || !weight.is_finite()
                            || weight < 0.0
                            || duplicate
                    });
                if invalid {
                    return Err(Error::InvalidModel(format!(
                        "multiplexing row {row_index} is empty or contains a duplicate/invalid source or weight"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn checked_grid_shift(value: f64) -> Result<GridShift> {
    if !value.is_finite() {
        return Err(Error::InvalidModel(
            "illumination shift must be finite".into(),
        ));
    }
    let rounded = value.round();
    if rounded < isize::MIN as f64 || rounded > isize::MAX as f64 {
        return Err(Error::InvalidModel(
            "illumination shift is outside the supported index range".into(),
        ));
    }
    let integer = rounded as isize;
    let offset = value - rounded;
    let nearest_offset = offset.round();
    let (lower, upper) = if (offset - nearest_offset).abs() <= 1e-12 {
        let displacement = integer
            .checked_add(nearest_offset as isize)
            .ok_or_else(|| {
                Error::InvalidModel(
                    "illumination shift is outside the supported index range".into(),
                )
            })?;
        (displacement, displacement)
    } else if offset > 0.0 {
        (
            integer,
            integer.checked_add(1).ok_or_else(|| {
                Error::InvalidModel(
                    "illumination shift is outside the supported index range".into(),
                )
            })?,
        )
    } else {
        (
            integer.checked_sub(1).ok_or_else(|| {
                Error::InvalidModel(
                    "illumination shift is outside the supported index range".into(),
                )
            })?,
            integer,
        )
    };
    Ok(GridShift {
        integer,
        offset,
        lower,
        upper,
    })
}

fn validate_image_shape(image_shape: (usize, usize)) -> Result<()> {
    if image_shape.0 == 0 || image_shape.1 == 0 {
        return Err(Error::InvalidShape(format!(
            "image shape {image_shape:?} must be non-zero"
        )));
    }
    if image_shape.0 > isize::MAX as usize || image_shape.1 > isize::MAX as usize {
        return Err(Error::InvalidShape(
            "image shape is outside the supported index range".into(),
        ));
    }
    Ok(())
}

fn validate_reconstruction_aspect(
    image_shape: (usize, usize),
    reconstruction_shape: (usize, usize),
) -> Result<()> {
    if reconstruction_shape.0 < image_shape.0 || reconstruction_shape.1 < image_shape.1 {
        return Err(Error::InvalidShape(format!(
            "image shape {image_shape:?} must fit reconstruction shape {reconstruction_shape:?}"
        )));
    }
    if reconstruction_shape.0 > isize::MAX as usize || reconstruction_shape.1 > isize::MAX as usize
    {
        return Err(Error::InvalidShape(
            "reconstruction shape is outside the supported index range".into(),
        ));
    }
    let left = reconstruction_shape.0 as u128 * image_shape.1 as u128;
    let right = reconstruction_shape.1 as u128 * image_shape.0 as u128;
    if left != right {
        return Err(Error::InvalidShape(
            "reconstruction must use the same scale factor in both dimensions".into(),
        ));
    }
    Ok(())
}

fn shape_contains_bounds(
    image_shape: (usize, usize),
    reconstruction_shape: (usize, usize),
    bounds: &[CropDisplacementBounds],
) -> Result<bool> {
    validate_reconstruction_aspect(image_shape, reconstruction_shape)?;
    for bound in bounds {
        if !axis_contains_bounds(
            image_shape.0,
            reconstruction_shape.0,
            bound.minimum_row,
            bound.maximum_row,
        )? || !axis_contains_bounds(
            image_shape.1,
            reconstruction_shape.1,
            bound.minimum_column,
            bound.maximum_column,
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn axis_contains_bounds(
    image_length: usize,
    reconstruction_length: usize,
    minimum_displacement: isize,
    maximum_displacement: isize,
) -> Result<bool> {
    let centre = i128::try_from(reconstruction_length / 2)
        .map_err(|_| Error::InvalidShape("reconstruction dimension is too large".into()))?;
    let image_half = i128::try_from(image_length / 2)
        .map_err(|_| Error::InvalidShape("image dimension is too large".into()))?;
    let base = centre.checked_sub(image_half).ok_or_else(|| {
        Error::InvalidShape("reconstruction crop origin overflows the index range".into())
    })?;
    let lower = base
        .checked_add(minimum_displacement as i128)
        .ok_or_else(|| {
            Error::InvalidShape("reconstruction crop origin overflows the index range".into())
        })?;
    let upper = base
        .checked_add((image_length - 1) as i128)
        .and_then(|value| value.checked_add(maximum_displacement as i128))
        .ok_or_else(|| {
            Error::InvalidShape("reconstruction crop extent overflows the index range".into())
        })?;
    Ok(lower >= 0 && upper < reconstruction_length as i128)
}

fn minimum_fitting_multiplier(
    image_shape: (usize, usize),
    aspect: (usize, usize),
    initial_multiplier: usize,
    maximum_multiplier: usize,
    bounds: &[CropDisplacementBounds],
) -> Result<usize> {
    let fits = |multiplier| -> Result<bool> {
        let shape = candidate_shape(aspect, multiplier)?;
        shape_contains_bounds(image_shape, shape, bounds)
    };
    if fits(initial_multiplier)? {
        return Ok(initial_multiplier);
    }
    let mut lower = initial_multiplier;
    let mut upper = initial_multiplier;
    loop {
        let doubled = upper.checked_mul(2).unwrap_or(maximum_multiplier);
        upper = doubled.min(maximum_multiplier);
        if upper == lower {
            return Err(Error::InvalidShape(
                "illumination crops require a reconstruction shape outside the supported index range"
                    .into(),
            ));
        }
        if fits(upper)? {
            break;
        }
        lower = upper;
    }
    while lower + 1 < upper {
        let middle = lower + (upper - lower) / 2;
        if fits(middle)? {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    Ok(upper)
}

fn candidate_shape(aspect: (usize, usize), multiplier: usize) -> Result<(usize, usize)> {
    let height = aspect.0.checked_mul(multiplier).ok_or_else(|| {
        Error::InvalidShape("reconstruction height overflows the supported range".into())
    })?;
    let width = aspect.1.checked_mul(multiplier).ok_or_else(|| {
        Error::InvalidShape("reconstruction width overflows the supported range".into())
    })?;
    Ok((height, width))
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn next_smooth_multiplier(minimum: usize) -> Option<usize> {
    fn visit(value: usize, prime_index: usize, minimum: usize, best: &mut Option<usize>) {
        if value >= minimum {
            if best.is_none_or(|current| value < current) {
                *best = Some(value);
            }
            return;
        }
        const PRIMES: [usize; 4] = [2, 3, 5, 7];
        for (index, &prime) in PRIMES.iter().enumerate().skip(prime_index) {
            let Some(next) = value.checked_mul(prime) else {
                continue;
            };
            if best.is_none_or(|current| next < current) {
                visit(next, index, minimum, best);
            }
        }
    }

    let mut best = None;
    visit(1, 0, minimum, &mut best);
    best
}
