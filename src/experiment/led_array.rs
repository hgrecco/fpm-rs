use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::{Optics, ResolvedSources};

/// Rigid pose mapping array-local coordinates into sample coordinates.
///
/// Rotations are active, right-handed, extrinsic rotations about the fixed
/// sample `x`, `y`, then `z` axes. A column vector is transformed with
/// `Rz(rz) * Ry(ry) * Rx(rx)` before translation is added.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrayPose {
    /// Translation `(x, y, z)` in metres.
    pub translation_m: [f64; 3],
    /// Extrinsic fixed-axis `(rx, ry, rz)` rotation in radians.
    pub rotation_rad: [f64; 3],
    rotation_convention: ArrayRotationConvention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ArrayRotationConvention {
    #[serde(rename = "active_extrinsic_xyz")]
    ActiveExtrinsicXyz,
}

impl ArrayPose {
    /// Returns the identity rigid transform.
    pub const fn identity() -> Self {
        Self {
            translation_m: [0.0; 3],
            rotation_rad: [0.0; 3],
            rotation_convention: ArrayRotationConvention::ActiveExtrinsicXyz,
        }
    }

    /// Creates a pure translation in metres.
    pub const fn from_translation(translation_m: [f64; 3]) -> Self {
        Self {
            translation_m,
            rotation_rad: [0.0; 3],
            rotation_convention: ArrayRotationConvention::ActiveExtrinsicXyz,
        }
    }

    /// Creates a translation and active extrinsic XYZ rotation in radians.
    pub const fn from_translation_and_extrinsic_xyz_radians(
        translation_m: [f64; 3],
        rotation_rad: [f64; 3],
    ) -> Self {
        Self {
            translation_m,
            rotation_rad,
            rotation_convention: ArrayRotationConvention::ActiveExtrinsicXyz,
        }
    }

    /// Creates a translation and active extrinsic XYZ rotation in degrees.
    pub fn from_translation_and_extrinsic_xyz_degrees(
        translation_m: [f64; 3],
        rotation_deg: [f64; 3],
    ) -> Self {
        Self {
            translation_m,
            rotation_rad: rotation_deg.map(f64::to_radians),
            rotation_convention: ArrayRotationConvention::ActiveExtrinsicXyz,
        }
    }

    /// Validates finite translation and rotation components.
    pub fn validate(&self) -> Result<()> {
        if self
            .translation_m
            .iter()
            .chain(&self.rotation_rad)
            .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidParameter {
                name: "pose",
                reason: "translation and rotation components must be finite".into(),
            });
        }
        Ok(())
    }

    pub(crate) fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        let [rx, ry, rz] = self.rotation_rad;
        let (sin_x, cos_x) = rx.sin_cos();
        let after_x = [
            point[0],
            cos_x * point[1] - sin_x * point[2],
            sin_x * point[1] + cos_x * point[2],
        ];
        let (sin_y, cos_y) = ry.sin_cos();
        let after_y = [
            cos_y * after_x[0] + sin_y * after_x[2],
            after_x[1],
            -sin_y * after_x[0] + cos_y * after_x[2],
        ];
        let (sin_z, cos_z) = rz.sin_cos();
        let rotated = [
            cos_z * after_y[0] - sin_z * after_y[1],
            sin_z * after_y[0] + cos_z * after_y[1],
            after_y[2],
        ];
        [
            rotated[0] + self.translation_m[0],
            rotated[1] + self.translation_m[1],
            rotated[2] + self.translation_m[2],
        ]
    }
}

impl Default for ArrayPose {
    fn default() -> Self {
        Self::identity()
    }
}

/// Regular planar LED grid with row-major physical source indexing.
///
/// `shape` is `(rows, columns)`, `pitch_m` is `(pitch_x, pitch_y)`, and
/// `reference_index` is the fractional `(column, row)` lattice coordinate at
/// the pose origin. Source index is `row * columns + column`. Per-source local
/// Cartesian corrections are applied before the global [`ArrayPose`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarLedArray {
    shape: (usize, usize),
    pitch_m: (f64, f64),
    reference_index: (f64, f64),
    pose: ArrayPose,
    position_offsets_m: Vec<[f64; 3]>,
}

impl PlanarLedArray {
    /// Creates a planar array from its canonical physical fields.
    pub fn new(
        shape: (usize, usize),
        pitch_m: (f64, f64),
        reference_index: (f64, f64),
        pose: ArrayPose,
    ) -> Self {
        Self {
            shape,
            pitch_m,
            reference_index,
            pose,
            position_offsets_m: Vec::new(),
        }
    }

    /// Sets array-local per-source Cartesian corrections in metres.
    ///
    /// An empty vector means all-zero corrections; otherwise its length must
    /// equal [`Self::source_count`].
    pub fn with_position_offsets_m(mut self, offsets: Vec<[f64; 3]>) -> Self {
        self.position_offsets_m = offsets;
        self
    }

    /// Replaces the physical `(pitch_x, pitch_y)` lattice spacing in metres.
    pub fn with_pitch_m(mut self, pitch_m: (f64, f64)) -> Self {
        self.pitch_m = pitch_m;
        self
    }

    /// Replaces the fractional `(column, row)` lattice coordinate at the pose origin.
    pub fn with_reference_index(mut self, reference_index: (f64, f64)) -> Self {
        self.reference_index = reference_index;
        self
    }

    /// Replaces the active-extrinsic-XYZ rigid pose of the array.
    pub fn with_pose(mut self, pose: ArrayPose) -> Self {
        self.pose = pose;
        self
    }

    /// Returns array shape as `(rows, columns)`.
    pub const fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Returns pitch as `(pitch_x, pitch_y)` in metres.
    pub const fn pitch_m(&self) -> (f64, f64) {
        self.pitch_m
    }

    /// Returns the fractional reference lattice coordinate `(column, row)`.
    pub const fn reference_index(&self) -> (f64, f64) {
        self.reference_index
    }

    /// Borrows the array-local to sample-coordinate rigid pose.
    pub const fn pose(&self) -> &ArrayPose {
        &self.pose
    }

    /// Borrows canonical local position offsets in metres.
    pub fn position_offsets_m(&self) -> &[[f64; 3]] {
        &self.position_offsets_m
    }

    /// Returns `rows * columns`, or zero if an impossible platform overflow occurs.
    pub fn source_count(&self) -> usize {
        self.shape.0.checked_mul(self.shape.1).unwrap_or(0)
    }

    /// Converts `(row, column)` to its row-major source index.
    pub fn source_index(&self, row: usize, column: usize) -> Result<usize> {
        if row >= self.shape.0 || column >= self.shape.1 {
            return Err(Error::InvalidParameter {
                name: "source_index",
                reason: format!("({row}, {column}) lies outside shape {:?}", self.shape),
            });
        }
        row.checked_mul(self.shape.1)
            .and_then(|value| value.checked_add(column))
            .ok_or_else(|| Error::InvalidParameter {
                name: "shape",
                reason: "source index overflows".into(),
            })
    }

    /// Converts a row-major source index to `(row, column)`.
    pub fn source_row_column(&self, index: usize) -> Result<(usize, usize)> {
        let count = self.validated_source_count()?;
        if index >= count {
            return Err(Error::InvalidParameter {
                name: "source_index",
                reason: format!("index {index} must be less than {count}"),
            });
        }
        Ok((index / self.shape.1, index % self.shape.1))
    }

    /// Validates dimensions, pitches, reference coordinate, pose, and offsets.
    pub fn validate(&self) -> Result<()> {
        let count = self.validated_source_count()?;
        if [self.pitch_m.0, self.pitch_m.1]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "pitch_m",
                reason: "pitch_x and pitch_y must be finite and positive".into(),
            });
        }
        if !self.reference_index.0.is_finite() || !self.reference_index.1.is_finite() {
            return Err(Error::InvalidParameter {
                name: "reference_index",
                reason: "column and row coordinates must be finite".into(),
            });
        }
        self.pose.validate()?;
        if (!self.position_offsets_m.is_empty() && self.position_offsets_m.len() != count)
            || self
                .position_offsets_m
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidParameter {
                name: "position_offsets_m",
                reason: format!("must be empty or contain {count} finite XYZ offsets"),
            });
        }
        Ok(())
    }

    fn validated_source_count(&self) -> Result<usize> {
        let count =
            self.shape
                .0
                .checked_mul(self.shape.1)
                .ok_or_else(|| Error::InvalidParameter {
                    name: "shape",
                    reason: "source count overflows".into(),
                })?;
        if count == 0 {
            return Err(Error::InvalidParameter {
                name: "shape",
                reason: "rows and columns must be non-zero".into(),
            });
        }
        Ok(count)
    }

    /// Resolves physical positions, directions, and transverse propagation vectors.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedSources> {
        self.validate()?;
        let count = self.validated_source_count()?;
        let mut positions = Vec::with_capacity(count);
        for row in 0..self.shape.0 {
            for column in 0..self.shape.1 {
                let index = row * self.shape.1 + column;
                let offset = self
                    .position_offsets_m
                    .get(index)
                    .copied()
                    .unwrap_or([0.0; 3]);
                let local = [
                    (column as f64 - self.reference_index.0) * self.pitch_m.0 + offset[0],
                    (row as f64 - self.reference_index.1) * self.pitch_m.1 + offset[1],
                    offset[2],
                ];
                positions.push(self.pose.transform_point(local));
            }
        }
        ResolvedSources::from_positions(positions, optics)
    }
}
