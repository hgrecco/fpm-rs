# Experiment and simulation configuration schema

`SimulationConfiguration` version 2 is the reproducible JSON boundary for
experiment descriptions and their compiled models. It contains separate true
and assumed experiment descriptions, shared image/reconstruction shapes, the
two derived `ImagePlaneModel` values, optional camera and acquisition-error
models, and a deterministic random seed. Version 1 illumination objects are not
accepted.

Each experiment contains `Optics`, one complete `Illumination`, and optional
optical background. Illumination geometry, stable source calibration, and
acquisition structure are distinct:

```json
{
  "optics": {
    "wavelength_vacuum_m": 5.32e-7,
    "objective_na": 0.1,
    "magnification": 4.0,
    "camera_pixel_size": 6.5e-6,
    "illumination_refractive_index": 1.0,
    "objective_medium_refractive_index": 1.0,
    "defocus_distance": null,
    "pupil_aberration": null
  },
  "illumination": {
    "geometry": {
      "kind": "planar_led_array",
      "shape": [3, 3],
      "pitch_m": [0.004, 0.004],
      "reference_index": [1.0, 1.0],
      "pose": {
        "translation_m": [0.0, 0.0, -0.09],
        "rotation_rad": [0.0, 0.0, 0.0],
        "rotation_convention": "active_extrinsic_xyz"
      },
      "position_offsets_m": []
    },
    "calibration": {
      "relative_power": null
    },
    "acquisition": {
      "frames": [
        {
          "contributions": [
            {"source": 0, "intensity_weight": 1.0}
          ],
          "gain": 1.0
        }
      ]
    }
  },
  "optical_background": null
}
```

Geometry `kind` is one of `planar_led_array`, `spherical_led_array`,
`spherical_led_arm`, `rotating_led_arc`, `source_position_list`,
`direction_list`, or `k_vector_list`. Direction lists serialize canonical unit
vectors. K-vector lists serialize source-order `{kx, ky}` values in
radians/metre. Physical positions and all distance fields use explicit metre
suffixes; angular serialization uses radian suffixes.

Acquisition is always canonical sparse frame storage. Duplicate source entries
are merged before serialization, zero weights are absent, and each frame is
nonempty. Source weights, relative powers, and gains are finite non-negative
intensity multipliers and are not normalized. Defaults are expanded only in
`ResolvedIllumination`; optional unit source power remains `null` in the
configuration.

`wavelength_vacuum_m` is the vacuum wavelength. Source propagation uses
`illumination_refractive_index`; pupil propagation uses
`objective_medium_refractive_index`. No geometry carries wavelength state.
The sample-plane detector pitch must satisfy
`camera_pixel_size / magnification < wavelength_vacuum_m / (2 * objective_na)`.
Equality is rejected because the coherent pupil cutoff would lie on the
one-sided discrete Nyquist boundary.

Use `ExperimentDescription::compile` for one model or
`SimulationConfiguration::new` for a validated true/reconstruction pair. Use
`save` and `load` for persistence: they validate the format version and ensure
the stored compiled pupil, crops, source vectors, weights, gains, background,
and sampling still agree with the descriptions. Automatic reconstruction-shape
selection covers the union of true and assumed source vectors; the selected
concrete shape is serialized.

Known uniform camera response is not baked into the serialized reconstruction
model. `reconstruction_model_for_counts()` applies it when constructing a
problem from detector counts. `Simulator::simulate` returns an already adjusted
reconstruction model for its simulated count data.

Physical planar-array calibration configuration is serialized separately from
the experiment description. `PlanarArrayCalibrationParameters` stores one
optional `CalibrationParameterSpec` per active pose, pitch, or reference
component; explicit source-indexed XYZ specs; and optional common power or gain
specs. Each spec stores finite lower/upper bounds, physical finite-difference
step, optimizer scale, optional prior center, and regularization strength.
`IlluminationCalibration` adds the bounded optimizer settings and loss type.

Checkpoints use format version 2 and include `physical_illumination_calibration`
and `calibrated_model` together. The physical state contains the initial and
current normal `Illumination`, absolute and normalized parameters, applied
gauge constraints, histories, convergence reason, conditioning indicators, and
partial-update counters. Result bundles use format version 2 and persist the
same pair as the verified `domain.physical_illumination` JSON artifact; generic
per-source Fourier-grid corrections remain in the separate illumination
calibration table/array artifacts.
