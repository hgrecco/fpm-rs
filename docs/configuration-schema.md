# Experiment and simulation configuration schema

`SimulationConfiguration` is the versioned JSON schema for reproducible model
compilation. Version 1 contains:

- `true_experiment`: concrete `Optics`, an `Illumination` variant, and optional
  optical background;
- `reconstruction_experiment`: the separately assumed experiment description;
- `image_shape` and `reconstruction_shape`, always `(height, width)`;
- `compiled_models`: both derived `ImagePlaneModel` values;
- optional concrete `CameraModel` and `IlluminationAcquisitionErrors`;
- deterministic `random_seed`.

All physical distances are metres, wave vectors are radians/metre, array data is
row-major, and source-weight values are intensity weights. `Illumination` uses
Serde's externally tagged representation. For example:

```json
{
  "illumination": {
    "LEDArray": {
      "grid_shape": [3, 3],
      "pitch": 0.004,
      "distance": 0.09,
      "center": [1.0, 1.0],
      "wavelength_override": null,
      "illumination_order": null,
      "intensity_weights": null,
      "rotation_radians": 0.0
    }
  }
}
```

The additional externally tagged variants are `LEDSphere`, `SphericalLEDArm`,
and `RotatingLEDArc`. Their coordinate conventions, fields, equations, and
identifiability constraints are documented in
[Spherical illumination geometries](spherical-geometries.md).

Serialized `PupilAberration` values are direct coefficients used by
`Pupil::circular`: astigmatism, coma, and spherical terms are radian weights for
the sampled radial polynomials documented on `Optics`, and
`edge_apodization` sets the radial amplitude decay. They are not normalized
Zernike coefficients.

Use `ExperimentDescription::compile` for one model or
`SimulationConfiguration::new` for a validated true/reconstruction pair. Use
`save` and `load` instead of calling `serde_json` directly: these methods enforce
the format version and validate that serialized compiled geometry, pupil,
sampling, gains, background, and multiplexing still agree with their concrete
descriptions.

Both constructors take `ReconstructionShape`. `Exact((height, width))` validates
a prescribed grid, while `Minimum`, `Smooth`, and `PowerOfTwo` resolve a concrete
aspect-preserving grid from the compiled illumination geometry. For paired
simulation configurations, automatic sizing covers the union of the true and
assumed geometries. Only the resolved tuple is serialized; the selection variant
does not alter the versioned schema.

Known uniform camera response is deliberately not baked into
`compiled_models.reconstruction_model`. Call
`reconstruction_model_for_counts()` when constructing a problem directly from
detector counts loaded from a serialized configuration. In contrast,
`Simulator::simulate` returns a `SimulationResult.reconstruction_model` that has
already been adjusted for the known linear camera response, so simulated detector
counts can be used directly.
