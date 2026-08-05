//! Illumination geometry, stable source calibration, and acquisition structure.
//!
//! Sources normally lie at negative sample `z`, the objective lies at positive
//! `z`, and incident light propagates approximately along `+z`. Physical source
//! positions are converted to propagation directions with
//! `normalize([0, 0, 0] - source_position)`. The transverse propagation-vector
//! components `(kx, ky)` are also the values used by the image-plane model: a
//! positive component displaces the Fourier crop toward the corresponding
//! positive Fourier-grid axis.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Result, error::Error};

use super::{Optics, PlanarLedArray, RotatingLedArc, SphericalLedArm, SphericalLedArray};

/// `(source_index, intensity_weight)` for one incoherent illumination component.
pub type SourceWeight = (usize, f64);
/// Sparse acquisition-frame rows of source-indexed intensity weights.
pub type MultiplexingMatrix = Vec<Vec<SourceWeight>>;

/// Transverse propagation-vector components in radians per metre.
///
/// These are physical incident-wave components. Model compilation maps positive
/// `kx` and `ky` to positive Fourier column and row displacement, respectively;
/// no additional geometry-dependent sign change is applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KVector {
    /// Transverse angular spatial frequency along sample `x`, in radians per metre.
    pub kx: f64,
    /// Transverse angular spatial frequency along sample `y`, in radians per metre.
    pub ky: f64,
}

impl KVector {
    /// Creates a transverse vector `(kx, ky)` in radians per metre.
    pub const fn new(kx: f64, ky: f64) -> Self {
        Self { kx, ky }
    }
}

/// Arbitrary physical source positions in sample coordinates, in metres.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourcePositionList {
    positions_m: Vec<[f64; 3]>,
}

impl SourcePositionList {
    /// Stores source positions in sample coordinates.
    pub fn new(positions_m: Vec<[f64; 3]>) -> Self {
        Self { positions_m }
    }

    /// Returns the number of physical sources.
    pub fn source_count(&self) -> usize {
        self.positions_m.len()
    }

    /// Borrows source positions as `(x, y, z)` values in metres.
    pub fn positions_m(&self) -> &[[f64; 3]] {
        &self.positions_m
    }

    /// Resolves positions to source-to-sample directions and transverse vectors.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedSources> {
        ResolvedSources::from_positions(self.positions_m.clone(), optics)
    }
}

/// Wavelength-independent incident propagation directions in sample coordinates.
///
/// Unit vectors are the canonical stored and serialized form. Directions must
/// have non-negative `z`; the on-axis direction is `(0, 0, 1)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionList {
    unit_vectors: Vec<[f64; 3]>,
}

impl DirectionList {
    /// Validates and stores already normalized propagation directions.
    pub fn from_unit_vectors(unit_vectors: Vec<[f64; 3]>) -> Result<Self> {
        Self::from_vectors(unit_vectors, false)
    }

    /// Stores propagation vectors, optionally normalizing each input vector.
    pub fn from_vectors(mut vectors: Vec<[f64; 3]>, normalize: bool) -> Result<Self> {
        if vectors.is_empty() {
            return Err(invalid(
                "unit_vectors",
                "at least one direction is required",
            ));
        }
        for vector in &mut vectors {
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(invalid("unit_vectors", "components must be finite"));
            }
            let norm = vector[0].hypot(vector[1]).hypot(vector[2]);
            if !norm.is_finite() || norm <= 0.0 {
                return Err(invalid(
                    "unit_vectors",
                    "vectors must have non-zero finite norm",
                ));
            }
            if normalize {
                for component in vector.iter_mut() {
                    *component /= norm;
                }
            } else if (norm - 1.0).abs() > 128.0 * f64::EPSILON {
                return Err(invalid("unit_vectors", "vectors must be normalized"));
            }
            if vector[2] < -128.0 * f64::EPSILON {
                return Err(invalid(
                    "unit_vectors",
                    "directions must lie in the positive-z illumination hemisphere",
                ));
            }
            vector[2] = vector[2].max(0.0);
        }
        Ok(Self {
            unit_vectors: vectors,
        })
    }

    /// Constructs directions from transverse direction cosines `(dx, dy)`.
    pub fn from_direction_cosines(values: Vec<[f64; 2]>) -> Result<Self> {
        Self::from_transverse_components(values, |value| value)
    }

    /// Constructs directions from independent component angles `(theta_x, theta_y)` in radians.
    pub fn from_component_angles_radians(values: Vec<[f64; 2]>) -> Result<Self> {
        Self::from_transverse_components(values, f64::sin)
    }

    /// Constructs directions from independent component angles in degrees.
    pub fn from_component_angles_degrees(values: Vec<[f64; 2]>) -> Result<Self> {
        Self::from_component_angles_radians(
            values
                .into_iter()
                .map(|[x, y]| [x.to_radians(), y.to_radians()])
                .collect(),
        )
    }

    /// Constructs directions from polar angle `theta` and azimuth `phi`, in radians.
    pub fn from_polar_angles_radians(values: Vec<[f64; 2]>) -> Result<Self> {
        let vectors = values
            .into_iter()
            .map(|[theta, phi]| {
                let (sin_theta, cos_theta) = theta.sin_cos();
                let (sin_phi, cos_phi) = phi.sin_cos();
                [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
            })
            .collect();
        Self::from_unit_vectors(vectors)
    }

    /// Constructs directions from polar angle `theta` and azimuth `phi`, in degrees.
    pub fn from_polar_angles_degrees(values: Vec<[f64; 2]>) -> Result<Self> {
        Self::from_polar_angles_radians(
            values
                .into_iter()
                .map(|[theta, phi]| [theta.to_radians(), phi.to_radians()])
                .collect(),
        )
    }

    fn from_transverse_components(values: Vec<[f64; 2]>, map: impl Fn(f64) -> f64) -> Result<Self> {
        let mut vectors = Vec::with_capacity(values.len());
        for [x, y] in values {
            if !x.is_finite() || !y.is_finite() {
                return Err(invalid("directions", "components must be finite"));
            }
            let dx = map(x);
            let dy = map(y);
            let transverse_squared = dx.mul_add(dx, dy * dy);
            if transverse_squared > 1.0 + 128.0 * f64::EPSILON {
                return Err(invalid(
                    "directions",
                    "squared transverse direction magnitude cannot exceed one",
                ));
            }
            vectors.push([dx, dy, (1.0 - transverse_squared).max(0.0).sqrt()]);
        }
        Self::from_unit_vectors(vectors)
    }

    /// Returns the number of directions.
    pub fn source_count(&self) -> usize {
        self.unit_vectors.len()
    }

    /// Borrows the canonical propagation unit vectors.
    pub fn unit_vectors(&self) -> &[[f64; 3]] {
        &self.unit_vectors
    }

    /// Allocates transverse direction cosines `(dx, dy)`.
    pub fn direction_cosines(&self) -> Vec<[f64; 2]> {
        self.unit_vectors.iter().map(|v| [v[0], v[1]]).collect()
    }

    /// Allocates independent component angles `(asin(dx), asin(dy))` in radians.
    pub fn component_angles_rad(&self) -> Vec<[f64; 2]> {
        self.unit_vectors
            .iter()
            .map(|v| [v[0].asin(), v[1].asin()])
            .collect()
    }

    /// Allocates independent component angles in degrees.
    pub fn component_angles_deg(&self) -> Vec<[f64; 2]> {
        self.component_angles_rad()
            .into_iter()
            .map(|[x, y]| [x.to_degrees(), y.to_degrees()])
            .collect()
    }

    /// Allocates polar `(theta, phi)` angles in radians.
    pub fn polar_angles_rad(&self) -> Vec<[f64; 2]> {
        self.unit_vectors
            .iter()
            .map(|v| [v[2].clamp(-1.0, 1.0).acos(), v[1].atan2(v[0])])
            .collect()
    }

    /// Allocates polar `(theta, phi)` angles in degrees.
    pub fn polar_angles_deg(&self) -> Vec<[f64; 2]> {
        self.polar_angles_rad()
            .into_iter()
            .map(|[theta, phi]| [theta.to_degrees(), phi.to_degrees()])
            .collect()
    }

    /// Resolves directions with the illumination wavenumber from `optics`.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedSources> {
        optics.validate()?;
        let k = optics.illumination_wavenumber();
        let k_vectors = self
            .unit_vectors
            .iter()
            .map(|direction| KVector::new(k * direction[0], k * direction[1]))
            .collect();
        Ok(ResolvedSources {
            directions: self.unit_vectors.clone(),
            k_vectors,
            positions_m: None,
        })
    }
}

/// Explicit wavelength-dependent transverse vectors in radians per metre.
///
/// Resolving with different optics preserves `(kx, ky)`, not illumination angle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KVectorList {
    k_vectors: Vec<KVector>,
}

impl KVectorList {
    /// Stores source-order transverse vectors for validation during resolution.
    pub fn new(k_vectors: Vec<KVector>) -> Self {
        Self { k_vectors }
    }

    /// Returns the number of vectors.
    pub fn source_count(&self) -> usize {
        self.k_vectors.len()
    }

    /// Borrows transverse vectors in radians per metre.
    pub fn k_vectors(&self) -> &[KVector] {
        &self.k_vectors
    }

    /// Validates propagating compatibility and resolves positive-z directions.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedSources> {
        validate_k_vectors(&self.k_vectors, optics)?;
        let k = optics.illumination_wavenumber();
        let directions = self
            .k_vectors
            .iter()
            .map(|vector| {
                let dx = vector.kx / k;
                let dy = vector.ky / k;
                [dx, dy, (1.0 - dx.mul_add(dx, dy * dy)).max(0.0).sqrt()]
            })
            .collect();
        Ok(ResolvedSources {
            directions,
            k_vectors: self.k_vectors.clone(),
            positions_m: None,
        })
    }
}

/// Serializable physical or direct source geometry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceGeometry {
    /// Regular planar LED array.
    #[serde(rename = "planar_led_array")]
    PlanarArray(PlanarLedArray),
    /// Fixed spherical LED array.
    #[serde(rename = "spherical_led_array")]
    SphericalArray(SphericalLedArray),
    /// Mechanically moved spherical LED arm.
    #[serde(rename = "spherical_led_arm")]
    SphericalArm(SphericalLedArm),
    /// LED arc sampled over commanded rotations.
    #[serde(rename = "rotating_led_arc")]
    RotatingArc(RotatingLedArc),
    /// Arbitrary physical positions.
    #[serde(rename = "source_position_list")]
    Positions(SourcePositionList),
    /// Canonical propagation directions.
    #[serde(rename = "direction_list")]
    Directions(DirectionList),
    /// Direct transverse vectors.
    #[serde(rename = "k_vector_list")]
    KVectors(KVectorList),
}

impl SourceGeometry {
    /// Returns the number of physical or direct sources before acquisition planning.
    pub fn source_count(&self) -> usize {
        match self {
            Self::PlanarArray(value) => value.source_count(),
            Self::SphericalArray(value) => value.source_count(),
            Self::SphericalArm(value) => value.source_count(),
            Self::RotatingArc(value) => value.source_count(),
            Self::Positions(value) => value.source_count(),
            Self::Directions(value) => value.source_count(),
            Self::KVectors(value) => value.source_count(),
        }
    }

    /// Atomically resolves geometry using the wavelength and media in `optics`.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedSources> {
        match self {
            Self::PlanarArray(value) => value.resolve(optics),
            Self::SphericalArray(value) => value.resolve(optics),
            Self::SphericalArm(value) => value.resolve(optics),
            Self::RotatingArc(value) => value.resolve(optics),
            Self::Positions(value) => value.resolve(optics),
            Self::Directions(value) => value.resolve(optics),
            Self::KVectors(value) => value.resolve(optics),
        }
    }
}

macro_rules! geometry_from {
    ($type:ty, $variant:ident) => {
        impl From<$type> for SourceGeometry {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}

geometry_from!(PlanarLedArray, PlanarArray);
geometry_from!(SphericalLedArray, SphericalArray);
geometry_from!(SphericalLedArm, SphericalArm);
geometry_from!(RotatingLedArc, RotatingArc);
geometry_from!(SourcePositionList, Positions);
geometry_from!(DirectionList, Directions);
geometry_from!(KVectorList, KVectors);

/// Inspectable geometry resolved with a particular [`Optics`] configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedSources {
    directions: Vec<[f64; 3]>,
    k_vectors: Vec<KVector>,
    positions_m: Option<Vec<[f64; 3]>>,
}

impl ResolvedSources {
    pub(crate) fn from_positions(positions_m: Vec<[f64; 3]>, optics: &Optics) -> Result<Self> {
        optics.validate()?;
        if positions_m.is_empty() {
            return Err(invalid(
                "positions_m",
                "at least one source position is required",
            ));
        }
        let mut directions = Vec::with_capacity(positions_m.len());
        for position in &positions_m {
            if position.iter().any(|value| !value.is_finite()) {
                return Err(invalid("positions_m", "position components must be finite"));
            }
            let norm = position[0].hypot(position[1]).hypot(position[2]);
            if !norm.is_finite() || norm <= 0.0 {
                return Err(invalid(
                    "positions_m",
                    "no source may coincide with the sample origin",
                ));
            }
            directions.push([
                -position[0] / norm,
                -position[1] / norm,
                -position[2] / norm,
            ]);
        }
        let k = optics.illumination_wavenumber();
        let k_vectors = directions
            .iter()
            .map(|direction| KVector::new(k * direction[0], k * direction[1]))
            .collect();
        Ok(Self {
            directions,
            k_vectors,
            positions_m: Some(positions_m),
        })
    }

    /// Returns the source count.
    pub fn source_count(&self) -> usize {
        self.k_vectors.len()
    }

    /// Borrows source-order propagation unit vectors.
    pub fn directions(&self) -> &[[f64; 3]] {
        &self.directions
    }

    /// Borrows source-order transverse vectors in radians per metre.
    pub fn k_vectors(&self) -> &[KVector] {
        &self.k_vectors
    }

    /// Borrows physical positions in metres when the geometry defines them.
    pub fn positions_m(&self) -> Option<&[[f64; 3]]> {
        self.positions_m.as_deref()
    }
}

/// Stable, source-indexed optical calibration independent of acquisition order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCalibration {
    relative_power: Option<Vec<f64>>,
}

impl SourceCalibration {
    /// Creates a calibration with optional dimensionless relative source powers.
    pub fn new(relative_power: Option<Vec<f64>>) -> Self {
        Self { relative_power }
    }

    /// Creates the default unit-power calibration.
    pub const fn unity() -> Self {
        Self {
            relative_power: None,
        }
    }

    /// Borrows explicitly configured relative powers, if present.
    pub fn relative_power(&self) -> Option<&[f64]> {
        self.relative_power.as_deref()
    }

    fn resolve(&self, source_count: usize) -> Result<Vec<f64>> {
        match &self.relative_power {
            Some(values)
                if values.len() != source_count
                    || values
                        .iter()
                        .any(|value| !value.is_finite() || *value < 0.0) =>
            {
                Err(invalid(
                    "relative_power",
                    format!("must contain {source_count} finite non-negative values"),
                ))
            }
            Some(values) => Ok(values.clone()),
            None => Ok(vec![1.0; source_count]),
        }
    }
}

/// One source contribution to an acquired intensity frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceContribution {
    /// Physical source index.
    pub source: usize,
    /// Dimensionless, non-negative intensity weight.
    pub intensity_weight: f64,
}

impl SourceContribution {
    /// Creates a source-indexed intensity contribution.
    pub const fn new(source: usize, intensity_weight: f64) -> Self {
        Self {
            source,
            intensity_weight,
        }
    }
}

/// Sparse mutually incoherent source contributions and gain for one frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IlluminationFrame {
    /// Sparse source contributions. Acquisition construction canonicalizes them.
    pub contributions: Vec<SourceContribution>,
    /// Dimensionless, non-negative intensity gain applied to the summed frame.
    pub gain: f64,
}

impl IlluminationFrame {
    /// Creates a frame for validation and canonicalization by [`AcquisitionPlan`].
    pub fn new(contributions: Vec<SourceContribution>, gain: f64) -> Self {
        Self {
            contributions,
            gain,
        }
    }
}

/// Sparse source-to-frame acquisition structure.
///
/// Predicted intensity is `gain[f] * sum_s(weight[f,s] * power[s] * I_s)`.
/// Source contributions are mutually incoherent; every multiplier acts on
/// intensity rather than field amplitude and is not automatically normalized.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionPlan {
    frames: Vec<IlluminationFrame>,
}

impl AcquisitionPlan {
    /// Creates one unit-gain frame per source in natural order.
    pub fn all_sources(source_count: usize) -> Result<Self> {
        Self::sequential((0..source_count).collect())
    }

    /// Creates one unit-gain frame per listed source.
    ///
    /// The order may omit or repeat sources. Index range is checked when the
    /// complete [`Illumination`] is resolved.
    pub fn sequential(order: Vec<usize>) -> Result<Self> {
        let frames = order
            .into_iter()
            .map(|source| IlluminationFrame::new(vec![SourceContribution::new(source, 1.0)], 1.0))
            .collect();
        Self::from_sparse(frames)
    }

    /// Canonicalizes sparse frames by merging duplicates and removing zero weights.
    pub fn from_sparse(frames: Vec<IlluminationFrame>) -> Result<Self> {
        if frames.is_empty() {
            return Err(invalid(
                "frames",
                "at least one acquisition frame is required",
            ));
        }
        let mut canonical = Vec::with_capacity(frames.len());
        for (frame_index, frame) in frames.into_iter().enumerate() {
            if !frame.gain.is_finite() || frame.gain < 0.0 {
                return Err(invalid(
                    "gain",
                    "frame gains must be finite and non-negative",
                ));
            }
            let mut merged = BTreeMap::<usize, f64>::new();
            for contribution in frame.contributions {
                if !contribution.intensity_weight.is_finite() || contribution.intensity_weight < 0.0
                {
                    return Err(invalid(
                        "intensity_weight",
                        "source weights must be finite and non-negative",
                    ));
                }
                if contribution.intensity_weight != 0.0 {
                    let value = merged.entry(contribution.source).or_insert(0.0);
                    *value += contribution.intensity_weight;
                    if !value.is_finite() {
                        return Err(invalid("intensity_weight", "merged weight is not finite"));
                    }
                }
            }
            if merged.is_empty() {
                return Err(invalid(
                    "contributions",
                    format!("frame {frame_index} is empty after canonicalization"),
                ));
            }
            canonical.push(IlluminationFrame {
                contributions: merged
                    .into_iter()
                    .map(|(source, intensity_weight)| {
                        SourceContribution::new(source, intensity_weight)
                    })
                    .collect(),
                gain: frame.gain,
            });
        }
        Ok(Self { frames: canonical })
    }

    /// Converts a dense `(frames, sources)` matrix to canonical sparse storage.
    pub fn from_dense(weights: Vec<Vec<f64>>) -> Result<Self> {
        if weights.is_empty() {
            return Err(invalid("weights", "at least one dense frame is required"));
        }
        let source_count = weights[0].len();
        if source_count == 0 || weights.iter().any(|row| row.len() != source_count) {
            return Err(invalid(
                "weights",
                "dense rows must have one common non-zero source count",
            ));
        }
        Self::from_sparse(
            weights
                .into_iter()
                .map(|row| {
                    IlluminationFrame::new(
                        row.into_iter()
                            .enumerate()
                            .map(|(source, weight)| SourceContribution::new(source, weight))
                            .collect(),
                        1.0,
                    )
                })
                .collect(),
        )
    }

    /// Returns the number of acquired frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Borrows canonical sparse frames.
    pub fn frames(&self) -> &[IlluminationFrame] {
        &self.frames
    }

    /// Allocates dense `(frames, sources)` row-major weights.
    pub fn dense_weights(&self, source_count: usize) -> Result<Vec<Vec<f64>>> {
        self.validate_indices(source_count)?;
        let mut dense = vec![vec![0.0; source_count]; self.frames.len()];
        for (frame_index, frame) in self.frames.iter().enumerate() {
            for contribution in &frame.contributions {
                dense[frame_index][contribution.source] = contribution.intensity_weight;
            }
        }
        Ok(dense)
    }

    fn validate_indices(&self, source_count: usize) -> Result<()> {
        if self
            .frames
            .iter()
            .flat_map(|frame| &frame.contributions)
            .any(|contribution| contribution.source >= source_count)
        {
            return Err(invalid(
                "source",
                format!("source indices must be less than {source_count}"),
            ));
        }
        Ok(())
    }
}

/// Complete serializable illumination configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Illumination {
    geometry: SourceGeometry,
    calibration: SourceCalibration,
    acquisition: AcquisitionPlan,
}

impl Illumination {
    /// Creates a complete configuration from separate source concepts.
    pub const fn new(
        geometry: SourceGeometry,
        calibration: SourceCalibration,
        acquisition: AcquisitionPlan,
    ) -> Self {
        Self {
            geometry,
            calibration,
            acquisition,
        }
    }

    /// Creates unit-power sequential acquisition for every geometry source.
    pub fn from_geometry(geometry: impl Into<SourceGeometry>) -> Result<Self> {
        let geometry = geometry.into();
        let acquisition = AcquisitionPlan::all_sources(geometry.source_count())?;
        Ok(Self::new(geometry, SourceCalibration::unity(), acquisition))
    }

    /// Borrows the physical or direct source geometry.
    pub const fn geometry(&self) -> &SourceGeometry {
        &self.geometry
    }

    /// Borrows the stable source calibration.
    pub const fn calibration(&self) -> &SourceCalibration {
        &self.calibration
    }

    /// Borrows the acquisition structure.
    pub const fn acquisition(&self) -> &AcquisitionPlan {
        &self.acquisition
    }

    /// Replaces the source geometry while retaining calibration and acquisition state.
    pub fn with_geometry(mut self, geometry: impl Into<SourceGeometry>) -> Self {
        self.geometry = geometry.into();
        self
    }

    /// Replaces stable source-power calibration while retaining geometry and acquisition.
    pub fn with_calibration(mut self, calibration: SourceCalibration) -> Self {
        self.calibration = calibration;
        self
    }

    /// Replaces sparse acquisition weights and frame gains while retaining source state.
    pub fn with_acquisition(mut self, acquisition: AcquisitionPlan) -> Self {
        self.acquisition = acquisition;
        self
    }

    /// Atomically resolves geometry, calibration, acquisition weights, and gains.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedIllumination> {
        let sources = self.geometry.resolve(optics)?;
        let source_count = sources.source_count();
        self.acquisition.validate_indices(source_count)?;
        let source_power = self.calibration.resolve(source_count)?;
        let frames = self
            .acquisition
            .frames
            .iter()
            .map(|frame| ResolvedFrame {
                contributions: frame.contributions.clone(),
                gain: frame.gain,
            })
            .collect();
        Ok(ResolvedIllumination {
            sources,
            frames,
            source_power,
        })
    }
}

/// Canonical sparse acquisition frame after complete illumination validation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedFrame {
    contributions: Vec<SourceContribution>,
    gain: f64,
}

impl ResolvedFrame {
    /// Borrows source-indexed intensity contributions.
    pub fn contributions(&self) -> &[SourceContribution] {
        &self.contributions
    }

    /// Returns the explicit dimensionless frame gain.
    pub const fn gain(&self) -> f64 {
        self.gain
    }
}

/// Complete illumination state resolved for one optical configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedIllumination {
    sources: ResolvedSources,
    frames: Vec<ResolvedFrame>,
    source_power: Vec<f64>,
}

impl ResolvedIllumination {
    /// Borrows resolved source geometry.
    pub const fn sources(&self) -> &ResolvedSources {
        &self.sources
    }

    /// Returns the number of physical sources.
    pub fn source_count(&self) -> usize {
        self.sources.source_count()
    }

    /// Returns the number of acquired frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns whether any frame combines more than one source.
    pub fn is_multiplexed(&self) -> bool {
        self.frames
            .iter()
            .any(|frame| frame.contributions.len() > 1)
    }

    /// Borrows canonical sparse frames.
    pub fn frames(&self) -> &[ResolvedFrame] {
        &self.frames
    }

    /// Borrows explicit source powers, including default unit values.
    pub fn source_power(&self) -> &[f64] {
        &self.source_power
    }

    /// Returns explicit frame gains, including default unit values.
    pub fn frame_gains(&self) -> Vec<f64> {
        self.frames.iter().map(|frame| frame.gain).collect()
    }

    /// Borrows resolved incident directions.
    pub fn directions(&self) -> &[[f64; 3]] {
        self.sources.directions()
    }

    /// Borrows physical source positions when available.
    pub fn positions_m(&self) -> Option<&[[f64; 3]]> {
        self.sources.positions_m()
    }

    /// Borrows transverse propagation vectors in radians per metre.
    pub fn k_vectors(&self) -> &[KVector] {
        self.sources.k_vectors()
    }

    /// Allocates acquisition weights shaped `(frame_count, source_count)`.
    pub fn dense_weights(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.source_count()]; self.frame_count()];
        for (frame_index, frame) in self.frames.iter().enumerate() {
            for contribution in &frame.contributions {
                dense[frame_index][contribution.source] = contribution.intensity_weight;
            }
        }
        dense
    }

    pub(crate) fn compiled_multiplexing_matrix(&self) -> MultiplexingMatrix {
        self.frames
            .iter()
            .map(|frame| {
                frame
                    .contributions
                    .iter()
                    .map(|contribution| {
                        (
                            contribution.source,
                            contribution.intensity_weight * self.source_power[contribution.source],
                        )
                    })
                    .collect()
            })
            .collect()
    }
}

fn validate_k_vectors(vectors: &[KVector], optics: &Optics) -> Result<()> {
    optics.validate()?;
    if vectors.is_empty() {
        return Err(invalid("k_vectors", "at least one source is required"));
    }
    let maximum_transverse = optics.illumination_wavenumber() * (1.0 + 128.0 * f64::EPSILON);
    if vectors.iter().any(|vector| {
        !vector.kx.is_finite()
            || !vector.ky.is_finite()
            || vector.kx.hypot(vector.ky) > maximum_transverse
    }) {
        return Err(invalid(
            "k_vectors",
            "vectors must be finite and propagating at the configured vacuum wavelength and illumination refractive index",
        ));
    }
    Ok(())
}

fn invalid(name: &'static str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name,
        reason: reason.into(),
    }
}
