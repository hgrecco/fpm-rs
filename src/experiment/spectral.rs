//! Narrowband channels and sparse composition of physical detector exposures.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Result, error::Error};

use super::{AcquisitionPlan, Optics, SourceCalibration, SourceGeometry};

/// One narrowband channel's optics, source powers, and local acquisition rows.
///
/// The channel ID is stable metadata; its position in the input list is used by
/// [`SpectralContribution`]. Geometry is supplied separately by [`SpectralGeometry`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralChannel {
    /// Non-empty, unique channel identifier.
    pub channel_id: String,
    /// Channel-specific optics with an explicit vacuum wavelength in metres.
    pub optics: Optics,
    /// Fixed relative source powers, including known spectral response if desired.
    pub calibration: SourceCalibration,
    /// Ordinary local acquisition rows; their gains precede the spectral sum.
    pub acquisition: AcquisitionPlan,
}

/// Declared geometry sharing across narrowband channels.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "geometry", rename_all = "snake_case")]
pub enum SpectralGeometry {
    /// Resolve this physical geometry independently with each channel's optics.
    /// Direct wavelength-specific k-vectors are rejected in this variant.
    Shared(SourceGeometry),
    /// One geometry per channel, in channel order; direct k-vectors are allowed.
    PerChannel(Vec<SourceGeometry>),
}

/// A non-negative intensity contribution to a detector exposure.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralContribution {
    /// Zero-based index in the ordered spectral channel list.
    pub channel: usize,
    /// Zero-based acquisition row in that channel's ordinary model.
    pub local_frame: usize,
    /// Fixed non-negative intensity multiplier; this is not a field amplitude.
    pub spectral_weight: f64,
}

/// One physical detector frame, including fixed calibration after the spectral sum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralFrame {
    /// Sparse channel/local-frame contributions, canonicalized by the plan constructor.
    pub contributions: Vec<SpectralContribution>,
    /// Finite positive detector gain applied once after summing intensities.
    pub gain: f64,
    /// Finite non-negative uniform detector background, in intensity units.
    pub background: f64,
}

/// Canonical sparse global acquisition plan for separate or multiplexed exposures.
///
/// Rows are in physical acquisition order. Duplicate `(channel, local_frame)`
/// pairs are summed, zero weights removed, and pairs sorted. Every row must
/// retain a positive contribution. References are checked during compilation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpectralAcquisitionPlan {
    frames: Vec<SpectralFrame>,
}

impl<'de> Deserialize<'de> for SpectralAcquisitionPlan {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        use serde::de::Error as _;
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Representation {
            frames: Vec<SpectralFrame>,
        }
        Self::multiplexed(Representation::deserialize(deserializer)?.frames)
            .map_err(D::Error::custom)
    }
}

impl SpectralAcquisitionPlan {
    /// Creates channel-major separate exposures with unit weights/gains and zero background.
    /// `frame_counts` must contain a positive local-frame count for every channel.
    pub fn separate(frame_counts: &[usize]) -> Result<Self> {
        if frame_counts.is_empty() || frame_counts.contains(&0) {
            return Err(invalid(
                "frame_counts",
                "every channel must have at least one local frame",
            ));
        }
        Self::multiplexed(
            frame_counts
                .iter()
                .enumerate()
                .flat_map(|(channel, &count)| {
                    (0..count).map(move |local_frame| SpectralFrame {
                        contributions: vec![SpectralContribution {
                            channel,
                            local_frame,
                            spectral_weight: 1.0,
                        }],
                        gain: 1.0,
                        background: 0.0,
                    })
                })
                .collect(),
        )
    }

    /// Canonicalizes explicitly supplied sparse exposures, preserving global row order.
    /// Non-finite or negative weights/background and non-positive gains are rejected.
    pub fn multiplexed(mut frames: Vec<SpectralFrame>) -> Result<Self> {
        if frames.is_empty() {
            return Err(invalid(
                "spectral_acquisition",
                "at least one detector frame is required",
            ));
        }
        for frame in &mut frames {
            if !frame.gain.is_finite() || frame.gain <= 0.0 {
                return Err(invalid(
                    "spectral_frame_gain",
                    "must be finite and positive",
                ));
            }
            if !frame.background.is_finite() || frame.background < 0.0 {
                return Err(invalid(
                    "spectral_frame_background",
                    "must be finite and non-negative",
                ));
            }
            let mut pairs = BTreeMap::<(usize, usize), f64>::new();
            for contribution in &frame.contributions {
                if !contribution.spectral_weight.is_finite() || contribution.spectral_weight < 0.0 {
                    return Err(invalid(
                        "spectral_weight",
                        "must be finite and non-negative",
                    ));
                }
                *pairs
                    .entry((contribution.channel, contribution.local_frame))
                    .or_default() += contribution.spectral_weight;
            }
            if pairs.values().any(|weight| !weight.is_finite()) {
                return Err(invalid(
                    "spectral_weight",
                    "summed duplicate weights must be finite",
                ));
            }
            frame.contributions = pairs
                .into_iter()
                .filter(|(_, weight)| *weight > 0.0)
                .map(
                    |((channel, local_frame), spectral_weight)| SpectralContribution {
                        channel,
                        local_frame,
                        spectral_weight,
                    },
                )
                .collect();
            if frame.contributions.is_empty() {
                return Err(invalid(
                    "spectral_acquisition",
                    "each detector frame must have a positive contribution",
                ));
            }
        }
        Ok(Self { frames })
    }

    /// Borrows canonical detector rows in acquisition order.
    pub fn frames(&self) -> &[SpectralFrame] {
        &self.frames
    }

    /// Returns the number of physical detector exposures.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

pub(crate) fn invalid(name: &'static str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        name,
        reason: reason.into(),
    }
}
