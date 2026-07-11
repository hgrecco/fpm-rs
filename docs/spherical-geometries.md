# Spherical illumination geometries

`LEDSphere` describes fixed LEDs on a spherical cap. `SphericalLEDArm`
describes one LED moved through a sequence of commanded polar/azimuth angles.
Both compile to the same transverse `KVector` representation used by the
reconstruction algorithms.

Angles are `(theta, phi)` in radians. `theta` is measured from the positive
optical (`z`) axis and must satisfy `0 <= theta < pi/2`; `phi` is measured from
positive `x` toward positive `y`. Distances are metres. Rigid orientations are
stored as `(rx, ry, rz)` and applied in the order `Rz * Ry * Rx`.

## Fixed LED sphere

For LED `i`, the model is

```text
u_i = [sin(theta_i + dtheta_i) cos(phi_i + dphi_i),
       sin(theta_i + dtheta_i) sin(phi_i + dphi_i),
       cos(theta_i + dtheta_i)]
p_i = c + R r u_i
k_i = (2 pi n / lambda) [p_i.x, p_i.y] / |p_i|
```

where `r` is `radius`, `c` is `center_offset`, `R` is
`orientation_radians`, and `(dtheta_i, dphi_i)` comes from
`angular_corrections`.

The proposed geometrical error model separates:

- rigid assembly errors: sphere decentring and three-axis mounting rotation;
- construction errors: an independent angular placement correction for every
  LED;
- radius error: relevant only when the sphere is not centred on the sample.

That last qualification is important: in the plane-wave model, radius cancels
exactly for a perfectly centred sphere. Radius cannot be estimated from
illumination directions alone, and radial LED placement errors have no effect
unless combined with decentring or a finite-distance wavefront model. Likewise,
a global azimuth shift and a mount rotation about `z` are partly redundant; a
calibration should fix one of them.

```rust
use fpm_rs::experiment::LEDSphere;

let sphere = LEDSphere::new(
    vec![(0.0, 0.0), (0.25, 0.0), (0.25, 1.57)],
    90e-3,
)
.center_offset((0.4e-3, -0.2e-3, 0.0))
.orientation_deg((0.1, -0.2, 0.3))
.angular_corrections(vec![(0.0, 0.0), (0.001, -0.002), (-0.001, 0.001)]);
```

## Moving spherical arm

The arm treats angle errors as shared kinematic errors rather than independent
errors at every acquisition point:

```text
theta_i' = theta_scale theta_i + theta_zero + direction_i B_theta / 2
phi_i'   = phi_scale phi_i     + phi_zero   + direction_i B_phi / 2
p_i = c + R Rz(phi_i') Rot(a, theta_i') [0, 0, L]
a = [0, cos(epsilon), sin(epsilon)]
```

Here `L` is `arm_length`, `c` is `pivot_offset`, `epsilon` is
`elevation_axis_tilt_radians`, and `B_theta`/`B_phi` are total backlash dead-band
widths. A positive/negative move selects the positive/negative half of the dead
band. The first command has no inferred approach direction, and a repeated
command retains the previous branch. Azimuth commands should therefore be
unwrapped before crossing `+-pi`.

The proposed arm error model covers:

- pivot-to-sample displacement and rigid mount rotation;
- encoder zero and scale errors on both axes;
- non-orthogonality between the elevation and azimuth axes;
- deterministic direction-dependent backlash;
- arm-length error, which—like sphere radius—only affects directions when the
  pivot is displaced from the sample.

This is a compact systematic model suitable for calibration. It deliberately
does not pretend that random repeatability, elastic sag, bearing runout, or
thermal drift are independent acquisition noise. If measurements show those
effects, use repeated-position residuals to choose an extension: a repeatability
distribution, an angle-dependent sag term, or a periodic runout term. Adding all
of them before inspecting residual structure would make the calibration poorly
identifiable.

## Rotating quarter-circle LED arc

`RotatingLEDArc` represents several LEDs fixed to a meridional quarter-circle
arm. The complete arm rotates around an axis nominally collinear with the
optical axis. Each LED is compiled at every commanded rotation; the source order
is rotation-major and then LED-position order.

For LED `j` at rotation `m`, the model is

```text
theta_j' = theta_j + dtheta_j
r_j      = radius + dr_j
phi_m'   = rotation_scale phi_m + rotation_zero + direction_m B / 2
q_mj = Rz(phi_m') r_j u(theta_j', dphi_j)
p_mj = c + Ry(beta_y) Rx(beta_x) q_mj
k_mj = (2 pi n / lambda) [p_mj.x, p_mj.y] / |p_mj|
```

The proposed error model covers:

- rotation-axis decentring through `axis_origin_offset`, including axial
  displacement of the arc centre from the sample;
- loss of collinearity with the optical axis through the two-component
  `axis_tilt_radians`;
- rotation encoder zero, scale, and direction-dependent backlash;
- per-LED position along the arc (`delta_theta`), out-of-plane mounting or arm
  twist (`delta_phi`), and radial/shape error (`led_radial_offsets`).

```rust
use fpm_rs::experiment::RotatingLEDArc;

let arc = RotatingLEDArc::new(
    vec![0.0, 0.10, 0.20, 0.30],
    vec![0.0, 0.5, 1.0, 0.5],
    90e-3,
)
.axis_origin_offset((0.3e-3, -0.2e-3, 0.1e-3))
.axis_tilt_deg(0.1, -0.2)
.rotation_encoder_zero_deg(0.3)
.rotation_backlash_deg(0.05)
.led_angular_corrections(vec![(0.0, 0.0); 4])
.led_radial_offsets(vec![0.0; 4]);
```

There are several important identifiability limits. A uniform per-LED
`delta_phi` is indistinguishable from rotation encoder zero. Backlash cannot be
separated from zero offset during a purely monotonic sweep; at least one
direction reversal is needed. Radial errors do not affect the plane-wave
directions when the arc centre is exactly at the sample, and the azimuth of an
LED at `theta = 0` is undefined. Axis offset and tilt also become strongly
correlated when only a narrow azimuth range is measured. Calibration should fix
one parameter in each degenerate pair or use independent mechanical metrology.

## Simulating mismatch

Geometric errors are deterministic source geometry in this crate. Compile the
measured or perturbed parameters into the simulator's true model and compile the
nominal parameters into its reconstruction model. `IlluminationAcquisitionErrors`
remains reserved for non-geometric effects such as failed frames, gain variation,
and source-order mistakes.
