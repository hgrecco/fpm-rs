//! Referenced, nondispersive OPD recovery by synthetic-wavelength phase mixing.

use std::f64::consts::TAU;

use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

use crate::{Error, Result, array_layout::StandardView2, model::ObjectCoupling};

use super::SpectralReconstructionResult;

/// Fixes each channel's unobservable constant phase before wavelength mixing.
#[derive(Clone, Debug)]
pub enum PhaseReference {
    /// Channel-ordered phase pistons in radians, subtracted from field phases.
    /// Use zero offsets only for fields that already share a calibrated zero-OPD reference.
    Offsets(Vec<f64>),
    /// Estimates pistons by an equal-weight circular phase mean on a constant-OPD region.
    Region {
        /// Common-grid mask: nonzero selects reference pixels, intersected with valid data.
        mask: Array2<u8>,
        /// Known nondispersive OPD of the reference region, in metres.
        opd_m: f64,
    },
}

/// Owned common-grid OPD and diagnostics in input-channel order.
///
/// Invalid pixels have NaN OPD/residual and zero fringe orders; always inspect
/// `valid_mask`. Residuals test phase consistency, not uniqueness of the recovered
/// branch: sufficiently large phase noise can still cause an undetected order error.
#[derive(Clone, Debug)]
pub struct OpticalPathDifferenceResult {
    /// Recovered nondispersive transmission OPD in metres, `(row, column)`.
    pub opd_m: Array2<f64>,
    /// One for usable pixels inside the requested interval and residual tolerance; zero otherwise.
    pub valid_mask: Array2<u8>,
    /// RMS principal wrapped phase mismatch over original channels, in radians.
    pub phase_residual_rad: Array2<f64>,
    /// Integer cycles added to each referenced principal phase, one array per channel.
    pub fringe_orders: Vec<Array2<i64>>,
    /// Applied constant phase pistons in radians, one per input channel.
    pub phase_offsets_rad: Vec<f64>,
    /// Vacuum wavelengths in metres, in input-channel order.
    pub wavelengths_vacuum_m: Vec<f64>,
    /// Descending synthetic-difference/original periods used for coarse-to-fine recovery.
    pub wavelength_ladder_m: Vec<f64>,
}

/// Combines registered complex fields into one nondispersive OPD map.
///
/// For transmission phase `φ_i = wrap(2π d / λ_i)`, subtracting the phase of
/// the longer wavelength from that of the shorter produces period
/// `Λ = λ_short λ_long / (λ_long - λ_short)`. The longest such period selects
/// the coarse branch in an explicit half-open OPD interval. All remaining
/// shorter difference periods and original wavelengths refine integer orders
/// by rounding; a final equal-phase-weight least-squares fit uses every original
/// channel. This is post-reconstruction phase mixing, not optical interference
/// between colors or a joint intensity/OPD optimizer.
///
/// Inputs must have matching spatial registration, resolution, phase reference,
/// and negligible OPD dispersion. No spatial continuity is required, so jumps
/// exceeding an original wavelength can be recovered. Noise in a coarse OPD
/// estimate must remain below half the next ladder period for correct rounding.
/// This implementation uses an explicit interval instead of spatially unwrapping
/// the longest synthetic phase and does not use phase-sum synthetic wavelengths.
///
/// # Example
/// ```
/// use fpm_rs::{Complex64, reconstruction::{PhaseReference, SyntheticWavelengthUnwrapper}};
/// use ndarray::Array2;
/// let wavelengths = [500e-9, 550e-9];
/// let fields: Vec<_> = wavelengths.iter().map(|&wavelength| {
///     Array2::from_elem((2, 3), Complex64::from_polar(1.0,
///         std::f64::consts::TAU * 2.1e-6 / wavelength))
/// }).collect();
/// let views: Vec<_> = fields.iter().map(|field| field.view()).collect();
/// let result = SyntheticWavelengthUnwrapper::new((0.0, 4e-6))?
///     .unwrap_fields(&views, &wavelengths, &PhaseReference::Offsets(vec![0.0; 2]), None)?;
/// assert!((result.opd_m[(0, 0)] - 2.1e-6).abs() < 1e-18);
/// # Ok::<(), fpm_rs::Error>(())
/// ```
///
/// # References
/// S. K. Mirsky and N. T. Shaked,
/// [“Six-pack holography for dynamic profiling of thick and extended objects by simultaneous three-wavelength phase unwrapping with doubled field of view”](https://doi.org/10.1038/s41598-023-45237-6),
/// Scientific Reports **13**, article 19293 (2023), synthetic-wavelength and
/// hierarchical phase-unwrapping sections. Here their phase-difference hierarchy
/// is applied to already reconstructed FPM fields, without their holographic optics.
#[derive(Clone, Debug)]
pub struct SyntheticWavelengthUnwrapper {
    /// Half-open allowed absolute OPD interval `[minimum, maximum)` in metres.
    /// Its width must not exceed the longest difference synthetic wavelength.
    pub opd_range_m: (f64, f64),
    /// All channel amplitudes must strictly exceed this finite nonnegative value.
    pub minimum_amplitude: f64,
    /// Optional maximum RMS phase residual in radians, in `(0, π]`.
    /// `None` retains all otherwise valid pixels and still reports residuals.
    pub max_phase_residual_rad: Option<f64>,
}

#[derive(Clone, Copy)]
struct Period {
    wavelength: f64,
    first: usize,
    second: Option<usize>,
}

fn parameter(name: &'static str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name,
        reason: reason.into(),
    }
}

fn wrap(phase: f64) -> f64 {
    phase.sin().atan2(phase.cos())
}

impl SyntheticWavelengthUnwrapper {
    /// Creates a validated interval configuration with no residual cutoff and zero amplitude floor.
    pub fn new(opd_range_m: (f64, f64)) -> Result<Self> {
        let value = Self {
            opd_range_m,
            minimum_amplitude: 0.0,
            max_phase_residual_rad: None,
        };
        value.validate()?;
        Ok(value)
    }

    /// Checks finite ordered bounds, amplitude floor, and optional residual tolerance.
    pub fn validate(&self) -> Result<()> {
        let (lower, upper) = self.opd_range_m;
        if !lower.is_finite()
            || !upper.is_finite()
            || upper <= lower
            || !(upper - lower).is_finite()
        {
            return Err(parameter(
                "opd_range_m",
                "requires finite increasing bounds with finite width",
            ));
        }
        if !self.minimum_amplitude.is_finite() || self.minimum_amplitude < 0.0 {
            return Err(parameter(
                "minimum_amplitude",
                "must be finite and nonnegative",
            ));
        }
        if self
            .max_phase_residual_rad
            .is_some_and(|value| !value.is_finite() || value <= 0.0 || value > std::f64::consts::PI)
        {
            return Err(parameter(
                "max_phase_residual_rad",
                "must be finite in (0, pi]",
            ));
        }
        Ok(())
    }

    fn periods(&self, wavelengths: &[f64]) -> Result<Vec<Period>> {
        self.validate()?;
        if wavelengths.len() < 2 || wavelengths.iter().any(|&w| !w.is_finite() || w <= 0.0) {
            return Err(parameter(
                "wavelengths_vacuum_m",
                "requires at least two distinct finite positive wavelengths",
            ));
        }
        let mut indices: Vec<_> = (0..wavelengths.len()).collect();
        indices.sort_by(|&a, &b| wavelengths[a].total_cmp(&wavelengths[b]));
        let mut periods = Vec::new();
        for (position, &first) in indices.iter().enumerate() {
            for &second in &indices[position + 1..] {
                let difference = wavelengths[second] - wavelengths[first];
                if difference == 0.0 {
                    return Err(parameter(
                        "wavelengths_vacuum_m",
                        "duplicate wavelengths have no finite beat period",
                    ));
                }
                let wavelength = wavelengths[first] * (wavelengths[second] / difference);
                if !wavelength.is_finite() || wavelength <= 0.0 {
                    return Err(parameter(
                        "wavelengths_vacuum_m",
                        "synthetic periods must be finite and positive",
                    ));
                }
                periods.push(Period {
                    wavelength,
                    first,
                    second: Some(second),
                });
            }
        }
        periods.sort_by(|a, b| b.wavelength.total_cmp(&a.wavelength));
        let longest = periods[0].wavelength;
        if 64.0 * f64::EPSILON * longest >= wavelengths[indices[0]] / 2.0 {
            return Err(parameter(
                "wavelengths_vacuum_m",
                "longest beat is too ill-conditioned for floating-point fringe-order recovery",
            ));
        }
        let (lower, upper) = self.opd_range_m;
        if upper - lower > longest {
            return Err(parameter(
                "opd_range_m",
                "width exceeds the longest synthetic wavelength; specify a narrower known branch",
            ));
        }
        // Keep integer orders exactly representable by f64 before converting to i64.
        if lower.abs().max(upper.abs()) / wavelengths[indices[0]] >= (1_u64 << 52) as f64 {
            return Err(parameter(
                "opd_range_m",
                "bounds exceed exact floating-point fringe-order precision",
            ));
        }
        for &first in &indices {
            if wavelengths[first] <= longest {
                periods.push(Period {
                    wavelength: wavelengths[first],
                    first,
                    second: None,
                });
            }
        }
        periods.sort_by(|a, b| b.wavelength.total_cmp(&a.wavelength));
        periods.dedup_by(|a, b| a.wavelength == b.wavelength);
        Ok(periods)
    }

    /// Recovers OPD from at least two finite C-contiguous fields on one nonempty grid.
    ///
    /// `reference` explicitly supplies pistons or a constant-known-OPD reference
    /// mask. `mask`, when present, is a common-grid uint8 validity mask (zero
    /// excludes a pixel). Zero/below-floor amplitudes in any channel invalidate
    /// that pixel. Reference means use only pixels valid in every channel.
    /// Undefined reference means, numerically ill-conditioned beat periods, and
    /// malformed shapes/layout/values return errors;
    /// individual range/residual failures are represented in the returned mask.
    /// Wavelengths are vacuum wavelengths in metres in the same order as fields.
    pub fn unwrap_fields(
        &self,
        fields: &[ArrayView2<'_, Complex64>],
        wavelengths_vacuum_m: &[f64],
        reference: &PhaseReference,
        mask: Option<ArrayView2<'_, u8>>,
    ) -> Result<OpticalPathDifferenceResult> {
        let periods = self.periods(wavelengths_vacuum_m)?;
        if fields.len() != wavelengths_vacuum_m.len() {
            return Err(parameter(
                "fields",
                "field count must equal wavelength count",
            ));
        }
        let shape = fields[0].dim();
        if shape.0 == 0 || shape.1 == 0 {
            return Err(Error::InvalidShape(
                "OPD field grid must be nonempty".into(),
            ));
        }
        let fields = fields
            .iter()
            .map(|field| {
                if field.dim() != shape {
                    return Err(Error::InvalidShape(
                        "all OPD fields must have the same shape".into(),
                    ));
                }
                let view = StandardView2::try_from(*field)?;
                if view
                    .as_slice()
                    .iter()
                    .any(|v| !v.re.is_finite() || !v.im.is_finite() || !v.norm().is_finite())
                {
                    return Err(parameter(
                        "fields",
                        "complex fields and amplitudes must be finite",
                    ));
                }
                Ok(view)
            })
            .collect::<Result<Vec<_>>>()?;
        let mask = mask
            .map(|view| {
                if view.dim() != shape {
                    return Err(Error::InvalidShape(
                        "OPD mask must match the field grid".into(),
                    ));
                }
                StandardView2::try_from(view)
            })
            .transpose()?;
        let count = fields[0].as_slice().len();
        let usable: Vec<_> = (0..count)
            .map(|pixel| {
                mask.as_ref().is_none_or(|v| v.as_slice()[pixel] != 0)
                    && fields
                        .iter()
                        .all(|field| field.as_slice()[pixel].norm() > self.minimum_amplitude)
            })
            .collect();
        if !usable.iter().any(|&v| v) {
            return Err(parameter(
                "fields/mask",
                "no pixels have usable amplitudes in all channels",
            ));
        }
        let offsets = match reference {
            PhaseReference::Offsets(values) => {
                if values.len() != fields.len() || values.iter().any(|v| !v.is_finite()) {
                    return Err(parameter(
                        "phase_offsets_rad",
                        "requires one finite piston per channel",
                    ));
                }
                values.iter().map(|&v| wrap(v)).collect::<Vec<_>>()
            }
            PhaseReference::Region { mask, opd_m } => {
                if mask.dim() != shape {
                    return Err(Error::InvalidShape(
                        "OPD reference mask must match the field grid".into(),
                    ));
                }
                if !opd_m.is_finite() {
                    return Err(parameter("reference", "requires a finite reference OPD"));
                }
                let mask = StandardView2::try_from(mask.view())?;
                let selected: Vec<_> = (0..count)
                    .filter(|&p| usable[p] && mask.as_slice()[p] != 0)
                    .collect();
                if selected.is_empty() {
                    return Err(parameter(
                        "reference_mask",
                        "reference region has no valid pixels",
                    ));
                }
                fields
                    .iter()
                    .zip(wavelengths_vacuum_m)
                    .map(|(field, &wavelength)| {
                        let sum: Complex64 = selected
                            .iter()
                            .map(|&p| {
                                let value = field.as_slice()[p];
                                value / value.norm()
                            })
                            .sum();
                        if sum.norm() <= 1e-12 * selected.len() as f64 {
                            return Err(parameter(
                                "reference_mask",
                                "circular reference phase mean is undefined",
                            ));
                        }
                        let reference_cycles = opd_m / wavelength;
                        if !reference_cycles.is_finite()
                            || reference_cycles.abs() >= (1_u64 << 52) as f64
                        {
                            return Err(parameter(
                                "reference_opd_m",
                                "exceeds exact floating-point fringe-order precision",
                            ));
                        }
                        Ok(wrap(sum.arg() - TAU * reference_cycles.rem_euclid(1.0)))
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };
        let mut result = OpticalPathDifferenceResult {
            opd_m: Array2::from_elem(shape, f64::NAN),
            valid_mask: Array2::zeros(shape),
            phase_residual_rad: Array2::from_elem(shape, f64::NAN),
            fringe_orders: (0..fields.len()).map(|_| Array2::zeros(shape)).collect(),
            phase_offsets_rad: offsets.clone(),
            wavelengths_vacuum_m: wavelengths_vacuum_m.to_vec(),
            wavelength_ladder_m: periods.iter().map(|p| p.wavelength).collect(),
        };
        let shortest = wavelengths_vacuum_m
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let fit_weights: Vec<_> = wavelengths_vacuum_m.iter().map(|w| shortest / w).collect();
        let denominator: f64 = fit_weights.iter().map(|w| w * w).sum();
        let (lower, upper) = self.opd_range_m;
        let coarse_boundary_tolerance =
            64.0 * f64::EPSILON * periods[0].wavelength.max(lower.abs()).max(upper.abs());
        let boundary_tolerance = 64.0 * f64::EPSILON * shortest.max(lower.abs()).max(upper.abs());
        let mut phases = vec![0.0; fields.len()];
        let mut orders = vec![0_i64; fields.len()];
        for (pixel, &valid) in usable.iter().enumerate() {
            if !valid {
                continue;
            }
            for ((phase, field), &offset) in phases.iter_mut().zip(&fields).zip(&offsets) {
                *phase = wrap(field.as_slice()[pixel].arg() - offset);
            }
            let phase_for = |period: &Period| {
                wrap(phases[period.first] - period.second.map_or(0.0, |second| phases[second]))
            };
            let longest = periods[0].wavelength;
            let coarse = phase_for(&periods[0]) / TAU * longest;
            let remainder = (coarse - lower).rem_euclid(longest);
            // Roundoff at the lower branch boundary must not send an exact
            // zero/reference pixel to the next synthetic period.
            let mut opd = if remainder <= coarse_boundary_tolerance
                || longest - remainder <= coarse_boundary_tolerance
            {
                lower
            } else {
                lower + remainder
            };
            if opd >= upper {
                continue;
            }
            // Mirsky & Shaked (2023): round the coarse/fine OPD difference
            // divided by the next period to recover its integer fringe order.
            for period in &periods[1..] {
                let fraction = phase_for(period) / TAU;
                opd = ((opd / period.wavelength - fraction).round() + fraction) * period.wavelength;
            }
            for ((order, &phase), &wavelength) in
                orders.iter_mut().zip(&phases).zip(wavelengths_vacuum_m)
            {
                *order = (opd / wavelength - phase / TAU).round() as i64;
            }
            opd = shortest
                * phases
                    .iter()
                    .zip(&orders)
                    .zip(&fit_weights)
                    .map(|((&phase, &order), &weight)| weight * (phase / TAU + order as f64))
                    .sum::<f64>()
                / denominator;
            if (opd - lower).abs() <= boundary_tolerance {
                opd = lower;
            }
            if (opd - upper).abs() <= boundary_tolerance {
                opd = upper;
            }
            let residual = (phases
                .iter()
                .zip(wavelengths_vacuum_m)
                .map(|(&phase, &wavelength)| {
                    let value = wrap(TAU * (opd / wavelength).rem_euclid(1.0) - phase);
                    value * value
                })
                .sum::<f64>()
                / fields.len() as f64)
                .sqrt();
            if !opd.is_finite()
                || opd < lower
                || opd >= upper
                || self
                    .max_phase_residual_rad
                    .is_some_and(|limit| residual > limit)
            {
                continue;
            }
            let index = (pixel / shape.1, pixel % shape.1);
            result.opd_m[index] = opd;
            result.valid_mask[index] = 1;
            result.phase_residual_rad[index] = residual;
            for (array, &order) in result.fringe_orders.iter_mut().zip(&orders) {
                array[index] = order;
            }
        }
        Ok(result)
    }
}

impl SpectralReconstructionResult {
    /// Mixes independent channel phases into a referenced nondispersive OPD map.
    /// Uses channel metadata for wavelength ordering. Shared-complex coupling is
    /// rejected because identical complex phases do not express `2π OPD / λ`.
    pub fn unwrap_opd(
        &self,
        unwrapper: &SyntheticWavelengthUnwrapper,
        reference: &PhaseReference,
        mask: Option<ArrayView2<'_, u8>>,
    ) -> Result<OpticalPathDifferenceResult> {
        if self.object_coupling != ObjectCoupling::Independent {
            return Err(parameter(
                "object_coupling",
                "OPD phase mixing requires independent wavelength fields",
            ));
        }
        let fields: Vec<_> = self.channels.iter().map(|c| c.object.view()).collect();
        let wavelengths: Vec<_> = self
            .channels
            .iter()
            .map(|c| c.wavelength_vacuum_m)
            .collect();
        unwrapper.unwrap_fields(&fields, &wavelengths, reference, mask)
    }
}

/// Independent wavelength reconstruction together with referenced common OPD recovery.
#[derive(Clone, Debug)]
pub struct MultiWavelengthReconstructionResult {
    /// Reconstructed channel fields, pupils, metadata, and intensity-fit trace.
    pub spectral: SpectralReconstructionResult,
    /// Phase-mixed nondispersive OPD map and diagnostic arrays.
    pub opd: OpticalPathDifferenceResult,
}
