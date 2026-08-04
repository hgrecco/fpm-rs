use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::{Optics, ResolvedSources};

/// Fixed LEDs mounted on a spherical surface on the negative-`z` source side.
///
/// Each angle is `(theta, phi)` in radians. `theta` is the polar angle from
/// the positive propagation (`z`) axis and `phi` is the propagation azimuth
/// from positive `x` toward positive `y`. Physical source positions are the
/// opposite of those nominal propagation directions. The sphere pose is
/// applied as `Rz * Ry * Rx`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SphericalLedArray {
    /// Nominal `(theta, phi)` LED positions in radians and natural source order.
    #[serde(rename = "polar_angles_rad")]
    pub angles: Vec<(f64, f64)>,
    /// Positive nominal sphere radius in metres.
    #[serde(rename = "radius_m")]
    pub radius: f64,
    /// Sphere-centre displacement `(x, y, z)` from the sample, in metres.
    #[serde(rename = "center_offset_m")]
    pub center_offset: (f64, f64, f64),
    /// Rigid mount rotation `(rx, ry, rz)` in radians.
    #[serde(rename = "rotation_rad")]
    pub orientation_radians: (f64, f64, f64),
    /// Optional per-LED `(delta_theta, delta_phi)` placement corrections.
    #[serde(rename = "angular_corrections_rad")]
    pub angular_corrections: Option<Vec<(f64, f64)>>,
}

impl SphericalLedArray {
    /// Creates a sphere from natural-order `(theta, phi)` radians and a radius in metres.
    pub fn new(angles: Vec<(f64, f64)>, radius: f64) -> Self {
        Self {
            angles,
            radius,
            center_offset: (0.0, 0.0, 0.0),
            orientation_radians: (0.0, 0.0, 0.0),
            angular_corrections: None,
        }
    }

    /// Sets the sphere-centre displacement `(x, y, z)` from the sample, in metres.
    pub fn center_offset(mut self, offset: (f64, f64, f64)) -> Self {
        self.center_offset = offset;
        self
    }

    /// Sets extrinsic mount rotations `(rx, ry, rz)` in degrees, applied as `Rz * Ry * Rx`.
    pub fn orientation_deg(mut self, degrees: (f64, f64, f64)) -> Self {
        self.orientation_radians = (
            degrees.0.to_radians(),
            degrees.1.to_radians(),
            degrees.2.to_radians(),
        );
        self
    }

    /// Sets one `(delta_theta, delta_phi)` correction in radians per natural-order LED.
    pub fn angular_corrections(mut self, corrections: Vec<(f64, f64)>) -> Self {
        self.angular_corrections = Some(corrections);
        self
    }

    /// Returns the number of fixed physical sources.
    pub fn source_count(&self) -> usize {
        self.angles.len()
    }

    /// Validates angles, positive radius, finite pose, and placement corrections.
    pub fn validate(&self) -> Result<()> {
        validate_angle_list(&self.angles, "LED sphere angles")?;
        validate_length(self.radius, "radius")?;
        validate_pose(self.center_offset, self.orientation_radians)?;
        if let Some(corrections) = &self.angular_corrections
            && (corrections.len() != self.angles.len()
                || corrections
                    .iter()
                    .any(|&(theta, phi)| !theta.is_finite() || !phi.is_finite()))
        {
            return Err(Error::InvalidParameter {
                name: "angular_corrections",
                reason: format!(
                    "must contain {} finite (delta_theta, delta_phi) pairs",
                    self.angles.len()
                ),
            });
        }
        Ok(())
    }

    fn natural_positions(&self) -> Result<Vec<[f64; 3]>> {
        self.angles
            .iter()
            .enumerate()
            .map(|(index, &(theta, phi))| {
                let (delta_theta, delta_phi) = self
                    .angular_corrections
                    .as_ref()
                    .map_or((0.0, 0.0), |values| values[index]);
                let direction = spherical_direction(theta + delta_theta, phi + delta_phi);
                source_position(
                    scale(direction, -self.radius),
                    self.center_offset,
                    self.orientation_radians,
                    "LED sphere",
                )
            })
            .collect()
    }
    /// Resolves physical positions to source-to-sample directions and transverse vectors.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedSources> {
        self.validate()?;
        ResolvedSources::from_positions(self.natural_positions()?, optics)
    }
}

/// A single LED moved over a spherical trajectory by an azimuth/elevation arm.
///
/// `commanded_angles` are `(theta, phi)` encoder commands in acquisition order.
/// The kinematic model includes encoder affine errors, direction-dependent
/// backlash, inner-axis non-orthogonality, arm-pivot displacement, and a rigid
/// mount rotation. Azimuth commands should be unwrapped when crossing `+-pi` so
/// that backlash direction is unambiguous.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SphericalLedArm {
    /// Movement-order `(theta, phi)` encoder commands in radians.
    #[serde(rename = "commanded_angles_rad")]
    pub commanded_angles: Vec<(f64, f64)>,
    /// Positive distance from pivot to LED, in metres.
    #[serde(rename = "arm_length_m")]
    pub arm_length: f64,
    /// Arm-pivot displacement `(x, y, z)` from the sample, in metres.
    #[serde(rename = "pivot_offset_m")]
    pub pivot_offset: (f64, f64, f64),
    /// Rigid mount rotation `(rx, ry, rz)` in radians.
    #[serde(rename = "rotation_rad")]
    pub orientation_radians: (f64, f64, f64),
    /// Additive elevation-encoder zero offset, in radians.
    #[serde(rename = "theta_zero_rad")]
    pub theta_zero_radians: f64,
    /// Additive azimuth-encoder zero offset, in radians.
    #[serde(rename = "phi_zero_rad")]
    pub phi_zero_radians: f64,
    /// Positive dimensionless elevation-encoder scale.
    pub theta_scale: f64,
    /// Positive dimensionless azimuth-encoder scale.
    pub phi_scale: f64,
    /// Tilt of the elevation axis toward the azimuth axis, in radians.
    #[serde(rename = "elevation_axis_tilt_rad")]
    pub elevation_axis_tilt_radians: f64,
    /// Total separation between increasing and decreasing encoder branches.
    #[serde(rename = "theta_backlash_rad")]
    pub theta_backlash_radians: f64,
    /// Total separation between increasing and decreasing encoder branches.
    #[serde(rename = "phi_backlash_rad")]
    pub phi_backlash_radians: f64,
}

impl SphericalLedArm {
    /// Creates an arm from movement-order commands in radians and length in metres.
    pub fn new(commanded_angles: Vec<(f64, f64)>, arm_length: f64) -> Self {
        Self {
            commanded_angles,
            arm_length,
            pivot_offset: (0.0, 0.0, 0.0),
            orientation_radians: (0.0, 0.0, 0.0),
            theta_zero_radians: 0.0,
            phi_zero_radians: 0.0,
            theta_scale: 1.0,
            phi_scale: 1.0,
            elevation_axis_tilt_radians: 0.0,
            theta_backlash_radians: 0.0,
            phi_backlash_radians: 0.0,
        }
    }

    /// Sets pivot displacement `(x, y, z)` from the sample in metres.
    pub fn pivot_offset(mut self, offset: (f64, f64, f64)) -> Self {
        self.pivot_offset = offset;
        self
    }

    /// Sets extrinsic mount rotations `(rx, ry, rz)` in degrees, applied as `Rz * Ry * Rx`.
    pub fn orientation_deg(mut self, degrees: (f64, f64, f64)) -> Self {
        self.orientation_radians = (
            degrees.0.to_radians(),
            degrees.1.to_radians(),
            degrees.2.to_radians(),
        );
        self
    }

    /// Sets additive elevation and azimuth encoder-zero offsets in degrees.
    pub fn encoder_zero_deg(mut self, theta: f64, phi: f64) -> Self {
        self.theta_zero_radians = theta.to_radians();
        self.phi_zero_radians = phi.to_radians();
        self
    }

    /// Sets positive dimensionless elevation and azimuth encoder scales.
    pub fn encoder_scale(mut self, theta: f64, phi: f64) -> Self {
        self.theta_scale = theta;
        self.phi_scale = phi;
        self
    }

    /// Sets the elevation-axis non-orthogonality tilt in degrees.
    pub fn elevation_axis_tilt_deg(mut self, degrees: f64) -> Self {
        self.elevation_axis_tilt_radians = degrees.to_radians();
        self
    }

    /// Sets total elevation and azimuth backlash branch separations in degrees.
    pub fn backlash_deg(mut self, theta: f64, phi: f64) -> Self {
        self.theta_backlash_radians = theta.to_radians();
        self.phi_backlash_radians = phi.to_radians();
        self
    }

    /// Returns the number of commanded physical source positions.
    pub fn source_count(&self) -> usize {
        self.commanded_angles.len()
    }

    /// Checks non-empty finite commands, positive geometry and encoder scales, and
    /// finite pose and mechanical calibration values.
    pub fn validate(&self) -> Result<()> {
        validate_angle_list(&self.commanded_angles, "arm commanded_angles")?;
        validate_length(self.arm_length, "arm_length")?;
        validate_pose(self.pivot_offset, self.orientation_radians)?;
        if !self.theta_zero_radians.is_finite()
            || !self.phi_zero_radians.is_finite()
            || !self.theta_scale.is_finite()
            || self.theta_scale <= 0.0
            || !self.phi_scale.is_finite()
            || self.phi_scale <= 0.0
        {
            return Err(Error::InvalidParameter {
                name: "arm encoder calibration",
                reason: "zero offsets must be finite and scales must be finite and positive".into(),
            });
        }
        if !self.elevation_axis_tilt_radians.is_finite()
            || self.elevation_axis_tilt_radians.abs() >= std::f64::consts::FRAC_PI_2
        {
            return Err(Error::InvalidParameter {
                name: "elevation_axis_tilt_radians",
                reason: "must be finite with magnitude less than pi/2".into(),
            });
        }
        if !self.theta_backlash_radians.is_finite()
            || self.theta_backlash_radians < 0.0
            || !self.phi_backlash_radians.is_finite()
            || self.phi_backlash_radians < 0.0
        {
            return Err(Error::InvalidParameter {
                name: "arm backlash",
                reason: "backlash widths must be finite and non-negative".into(),
            });
        }
        Ok(())
    }

    fn positions(&self) -> Result<Vec<[f64; 3]>> {
        let theta_branches = backlash_branches(&self.commanded_angles, 0);
        let phi_branches = backlash_branches(&self.commanded_angles, 1);
        let (axis_sin, axis_cos) = self.elevation_axis_tilt_radians.sin_cos();
        let elevation_axis = [0.0, axis_cos, axis_sin];

        self.commanded_angles
            .iter()
            .enumerate()
            .map(|(index, &(theta_command, phi_command))| {
                let theta = self.theta_scale * theta_command
                    + self.theta_zero_radians
                    + 0.5 * self.theta_backlash_radians * theta_branches[index];
                let phi = self.phi_scale * phi_command
                    + self.phi_zero_radians
                    + 0.5 * self.phi_backlash_radians * phi_branches[index];
                let home = [0.0, 0.0, -self.arm_length];
                let elevated = rotate_axis_angle(home, elevation_axis, theta);
                let arm_position = rotate_z(elevated, phi);
                source_position(
                    arm_position,
                    self.pivot_offset,
                    self.orientation_radians,
                    "spherical LED arm",
                )
            })
            .collect()
    }
    /// Resolves physical positions to source-to-sample directions and transverse vectors.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedSources> {
        self.validate()?;
        ResolvedSources::from_positions(self.positions()?, optics)
    }
}

/// A quarter-circle LED arc rotated about an axis nominally collinear with the
/// optical axis.
///
/// Every LED is compiled at every commanded rotation angle. Sources are
/// ordered rotation-major and then by `led_thetas`. The nominal arm lies in the
/// `x-z` plane, with LED polar positions measured from positive `z`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotatingLedArc {
    /// Nominal LED polar positions along the quarter-circle arm, in radians.
    #[serde(rename = "led_polar_angles_rad")]
    pub led_thetas: Vec<f64>,
    /// Commanded arm azimuths in physical movement order, in radians.
    #[serde(rename = "rotation_angles_rad")]
    pub rotation_angles: Vec<f64>,
    /// Positive nominal arc radius in metres.
    #[serde(rename = "radius_m")]
    pub radius: f64,
    /// Point on the rotation axis relative to the sample, in metres.
    #[serde(rename = "axis_origin_offset_m")]
    pub axis_origin_offset: (f64, f64, f64),
    /// Rotation-axis tilt `(rx, ry)` away from the optical axis, in radians.
    #[serde(rename = "axis_tilt_rad")]
    pub axis_tilt_radians: (f64, f64),
    /// Per-LED `(delta_theta, delta_phi)` mounting corrections.
    #[serde(rename = "led_angular_corrections_rad")]
    pub led_angular_corrections: Option<Vec<(f64, f64)>>,
    /// Per-LED radial corrections to the nominal arm radius, in metres.
    #[serde(rename = "led_radial_offsets_m")]
    pub led_radial_offsets: Option<Vec<f64>>,
    /// Additive rotation-encoder zero offset in radians.
    #[serde(rename = "rotation_zero_rad")]
    pub rotation_zero_radians: f64,
    /// Positive dimensionless rotation-encoder scale.
    pub rotation_scale: f64,
    /// Total separation between increasing and decreasing rotation branches.
    #[serde(rename = "rotation_backlash_rad")]
    pub rotation_backlash_radians: f64,
}

impl RotatingLedArc {
    /// Creates an arc from LED polar angles and movement-order rotations in radians,
    /// with a positive radius in metres.
    pub fn new(led_thetas: Vec<f64>, rotation_angles: Vec<f64>, radius: f64) -> Self {
        Self {
            led_thetas,
            rotation_angles,
            radius,
            axis_origin_offset: (0.0, 0.0, 0.0),
            axis_tilt_radians: (0.0, 0.0),
            led_angular_corrections: None,
            led_radial_offsets: None,
            rotation_zero_radians: 0.0,
            rotation_scale: 1.0,
            rotation_backlash_radians: 0.0,
        }
    }

    /// Sets a point on the rotation axis `(x, y, z)` relative to the sample, in metres.
    pub fn axis_origin_offset(mut self, offset: (f64, f64, f64)) -> Self {
        self.axis_origin_offset = offset;
        self
    }

    /// Sets extrinsic `x` and `y` tilts of the rotation axis in degrees.
    pub fn axis_tilt_deg(mut self, x: f64, y: f64) -> Self {
        self.axis_tilt_radians = (x.to_radians(), y.to_radians());
        self
    }

    /// Sets per-LED `(delta_theta, delta_phi)` mounting corrections in radians.
    pub fn led_angular_corrections(mut self, corrections: Vec<(f64, f64)>) -> Self {
        self.led_angular_corrections = Some(corrections);
        self
    }

    /// Sets per-LED radial corrections in metres.
    pub fn led_radial_offsets(mut self, offsets: Vec<f64>) -> Self {
        self.led_radial_offsets = Some(offsets);
        self
    }

    /// Sets the additive rotation-encoder zero offset in degrees.
    pub fn rotation_encoder_zero_deg(mut self, degrees: f64) -> Self {
        self.rotation_zero_radians = degrees.to_radians();
        self
    }

    /// Sets the positive dimensionless rotation-encoder scale.
    pub fn rotation_encoder_scale(mut self, scale: f64) -> Self {
        self.rotation_scale = scale;
        self
    }

    /// Sets total separation between increasing and decreasing rotation branches in degrees.
    pub fn rotation_backlash_deg(mut self, degrees: f64) -> Self {
        self.rotation_backlash_radians = degrees.to_radians();
        self
    }

    /// Returns the rotation-major physical source count.
    pub fn source_count(&self) -> usize {
        self.led_thetas
            .len()
            .checked_mul(self.rotation_angles.len())
            .unwrap_or(0)
    }

    /// Validates angular domains, positive radius and scale, finite pose and encoder
    /// calibration, per-LED correction lengths, optional wavelength, and gains.
    pub fn validate(&self) -> Result<()> {
        if self.led_thetas.is_empty()
            || self.led_thetas.iter().any(|&theta| {
                !theta.is_finite() || !(0.0..std::f64::consts::FRAC_PI_2).contains(&theta)
            })
        {
            return Err(Error::InvalidParameter {
                name: "led_thetas",
                reason: "must contain at least one finite angle with 0 <= theta < pi/2".into(),
            });
        }
        if self.rotation_angles.is_empty()
            || self.rotation_angles.iter().any(|value| !value.is_finite())
        {
            return Err(Error::InvalidParameter {
                name: "rotation_angles",
                reason: "must contain at least one finite commanded azimuth".into(),
            });
        }
        self.led_thetas
            .len()
            .checked_mul(self.rotation_angles.len())
            .ok_or_else(|| Error::InvalidParameter {
                name: "rotating LED arc",
                reason: "source count overflows".into(),
            })?;
        validate_length(self.radius, "radius")?;
        validate_pose(
            self.axis_origin_offset,
            (self.axis_tilt_radians.0, self.axis_tilt_radians.1, 0.0),
        )?;
        if self.axis_tilt_radians.0.abs() >= std::f64::consts::FRAC_PI_2
            || self.axis_tilt_radians.1.abs() >= std::f64::consts::FRAC_PI_2
        {
            return Err(Error::InvalidParameter {
                name: "axis_tilt_radians",
                reason: "components must have magnitude less than pi/2".into(),
            });
        }
        if let Some(corrections) = &self.led_angular_corrections
            && (corrections.len() != self.led_thetas.len()
                || corrections
                    .iter()
                    .enumerate()
                    .any(|(index, &(theta, phi))| {
                        !theta.is_finite()
                            || !phi.is_finite()
                            || !(0.0..std::f64::consts::FRAC_PI_2)
                                .contains(&(self.led_thetas[index] + theta))
                    }))
        {
            return Err(Error::InvalidParameter {
                name: "led_angular_corrections",
                reason: format!(
                    "must contain {} finite corrections that keep each LED on the quarter-circle cap",
                    self.led_thetas.len()
                ),
            });
        }
        if let Some(offsets) = &self.led_radial_offsets
            && (offsets.len() != self.led_thetas.len()
                || offsets
                    .iter()
                    .any(|offset| !offset.is_finite() || self.radius + offset <= 0.0))
        {
            return Err(Error::InvalidParameter {
                name: "led_radial_offsets",
                reason: format!(
                    "must contain {} finite offsets with positive corrected radii",
                    self.led_thetas.len()
                ),
            });
        }
        if !self.rotation_zero_radians.is_finite()
            || !self.rotation_scale.is_finite()
            || self.rotation_scale <= 0.0
        {
            return Err(Error::InvalidParameter {
                name: "rotation encoder calibration",
                reason: "zero must be finite and scale must be finite and positive".into(),
            });
        }
        if !self.rotation_backlash_radians.is_finite() || self.rotation_backlash_radians < 0.0 {
            return Err(Error::InvalidParameter {
                name: "rotation_backlash_radians",
                reason: "must be finite and non-negative".into(),
            });
        }
        Ok(())
    }

    fn positions(&self) -> Result<Vec<[f64; 3]>> {
        let branches = scalar_backlash_branches(&self.rotation_angles);
        let mut positions = Vec::with_capacity(self.source_count());
        for (rotation_index, &command) in self.rotation_angles.iter().enumerate() {
            let rotation = self.rotation_scale * command
                + self.rotation_zero_radians
                + 0.5 * self.rotation_backlash_radians * branches[rotation_index];
            for (led_index, &nominal_theta) in self.led_thetas.iter().enumerate() {
                let (delta_theta, delta_phi) = self
                    .led_angular_corrections
                    .as_ref()
                    .map_or((0.0, 0.0), |values| values[led_index]);
                let radius = self.radius
                    + self
                        .led_radial_offsets
                        .as_ref()
                        .map_or(0.0, |values| values[led_index]);
                let local = scale(
                    spherical_direction(nominal_theta + delta_theta, delta_phi),
                    -radius,
                );
                let rotated = rotate_z(local, rotation);
                positions.push(source_position(
                    rotated,
                    self.axis_origin_offset,
                    (self.axis_tilt_radians.0, self.axis_tilt_radians.1, 0.0),
                    "rotating LED arc",
                )?);
            }
        }
        Ok(positions)
    }
    /// Resolves rotation-major physical positions and transverse vectors.
    pub fn resolve(&self, optics: &Optics) -> Result<ResolvedSources> {
        self.validate()?;
        ResolvedSources::from_positions(self.positions()?, optics)
    }
}

fn validate_angle_list(angles: &[(f64, f64)], name: &'static str) -> Result<()> {
    if angles.is_empty()
        || angles.iter().any(|&(theta, phi)| {
            !theta.is_finite()
                || !(0.0..std::f64::consts::FRAC_PI_2).contains(&theta)
                || !phi.is_finite()
        })
    {
        return Err(Error::InvalidParameter {
            name,
            reason: "must contain at least one finite (theta, phi) pair with 0 <= theta < pi/2"
                .into(),
        });
    }
    Ok(())
}

fn validate_length(value: f64, name: &'static str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::InvalidParameter {
            name,
            reason: "must be finite and positive".into(),
        });
    }
    Ok(())
}

fn validate_pose(offset: (f64, f64, f64), orientation: (f64, f64, f64)) -> Result<()> {
    if [
        offset.0,
        offset.1,
        offset.2,
        orientation.0,
        orientation.1,
        orientation.2,
    ]
    .into_iter()
    .any(|value| !value.is_finite())
    {
        return Err(Error::InvalidParameter {
            name: "spherical geometry pose",
            reason: "offset and orientation components must be finite".into(),
        });
    }
    Ok(())
}

fn spherical_direction(theta: f64, phi: f64) -> [f64; 3] {
    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();
    [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
}

fn source_position(
    local: [f64; 3],
    offset: (f64, f64, f64),
    orientation: (f64, f64, f64),
    geometry: &'static str,
) -> Result<[f64; 3]> {
    let rotated = rotate_pose(local, orientation);
    let position = [
        rotated[0] + offset.0,
        rotated[1] + offset.1,
        rotated[2] + offset.2,
    ];
    let norm =
        (position[0] * position[0] + position[1] * position[1] + position[2] * position[2]).sqrt();
    if !norm.is_finite() || norm <= 0.0 || position[2] >= 0.0 {
        return Err(Error::InvalidParameter {
            name: geometry,
            reason: "every transformed source must be finite, distinct from the sample, and on the negative-z source hemisphere"
                .into(),
        });
    }
    Ok(position)
}

fn backlash_branches(commands: &[(f64, f64)], component: usize) -> Vec<f64> {
    let mut branches = Vec::with_capacity(commands.len());
    let mut branch = 0.0;
    branches.push(branch);
    for pair in commands.windows(2) {
        let previous = if component == 0 { pair[0].0 } else { pair[0].1 };
        let current = if component == 0 { pair[1].0 } else { pair[1].1 };
        let delta = current - previous;
        if delta != 0.0 {
            branch = delta.signum();
        }
        branches.push(branch);
    }
    branches
}

fn scalar_backlash_branches(commands: &[f64]) -> Vec<f64> {
    let mut branches = Vec::with_capacity(commands.len());
    let mut branch = 0.0;
    branches.push(branch);
    for pair in commands.windows(2) {
        let delta = pair[1] - pair[0];
        if delta != 0.0 {
            branch = delta.signum();
        }
        branches.push(branch);
    }
    branches
}

fn scale(vector: [f64; 3], scalar: f64) -> [f64; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

fn rotate_pose(vector: [f64; 3], radians: (f64, f64, f64)) -> [f64; 3] {
    let (sin_x, cos_x) = radians.0.sin_cos();
    let after_x = [
        vector[0],
        cos_x * vector[1] - sin_x * vector[2],
        sin_x * vector[1] + cos_x * vector[2],
    ];
    let (sin_y, cos_y) = radians.1.sin_cos();
    let after_y = [
        cos_y * after_x[0] + sin_y * after_x[2],
        after_x[1],
        -sin_y * after_x[0] + cos_y * after_x[2],
    ];
    rotate_z(after_y, radians.2)
}

fn rotate_z(vector: [f64; 3], radians: f64) -> [f64; 3] {
    let (sin, cos) = radians.sin_cos();
    [
        cos * vector[0] - sin * vector[1],
        sin * vector[0] + cos * vector[1],
        vector[2],
    ]
}

fn rotate_axis_angle(vector: [f64; 3], axis: [f64; 3], radians: f64) -> [f64; 3] {
    let (sin, cos) = radians.sin_cos();
    let dot = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2];
    let cross = [
        axis[1] * vector[2] - axis[2] * vector[1],
        axis[2] * vector[0] - axis[0] * vector[2],
        axis[0] * vector[1] - axis[1] * vector[0],
    ];
    [
        vector[0] * cos + cross[0] * sin + axis[0] * dot * (1.0 - cos),
        vector[1] * cos + cross[1] * sin + axis[1] * dot * (1.0 - cos),
        vector[2] * cos + cross[2] * sin + axis[2] * dot * (1.0 - cos),
    ]
}
