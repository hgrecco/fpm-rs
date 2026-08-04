# Run a simulation

Compile an `ImagePlaneModel`, create either a complex128 object array or a
`SyntheticObject`, and call `simulate`:

```python
truth = fpm.SyntheticObject.mixed_test_pattern(model.reconstruction_shape)
simulation = fpm.simulate(model, truth, seed=17)
```

With no camera or acquisition errors this is an ideal, deterministic simulation.
The result contains detector `measurements`, the `ground_truth_object`, the
`true_model`, and the `reconstruction_model` that should be paired with the
returned detector values.

Pass a `CameraModel` to introduce detector gain, offset, read/shot noise,
quantization, saturation, sensitivity variations, or bad pixels. Pass
`IlluminationAcquisitionErrors` for gain variation, missing frames, or source
permutation. Set a fixed seed whenever the workflow must be reproducible.

The Rust simulator also provides named deterministic benchmark presets and
ground-truth metrics. Their precise definitions and validation policy are in
[Reconstruction benchmarks](../benchmarks.md).

## Model mismatch and detector counts

Represent geometric mismatch with separate concrete source descriptions:
compile the actual geometry into the model given to `Simulator::new`, and the
assumed geometry into the reconstruction model. `IlluminationAcquisitionErrors`
is limited to acquisition effects—relative frame-gain variation, failed frames,
and source permutations. Failed frames remain as dark, zero-weight frames so
frame/model indexing is retained. Fixed-sphere placement/pose and moving-arm
kinematic errors are detailed in [Spherical illumination geometries](../spherical-geometries.md).

`CameraModel` owns pixel sensitivity, photon/electron conversion, shot and read
noise, dark current, gain, offset, quantization, saturation, and deterministic
bad pixels. Defocus, pupil aberration, and edge apodization compile through
`Optics` into the model pupil; represent aberration mismatch with separate true
and reconstruction models. Stable illumination-angle transmission belongs in
`SourceCalibration.relative_power`; acquisition-specific source weights and
frame gains belong in `AcquisitionPlan`, not geometry or the pupil.

`SimulationResult` retains both compiled models and acquisition configuration.
Use `evaluation::evaluate_reconstruction_with_problem` in Rust, or
`fpm.evaluation.evaluate_reconstruction(..., problem=..., reference_model=...)`
in Python, to combine masked per-frame intensity residuals with object, pupil,
and calibration comparisons. Pupil metrics remove the best global complex
scale, because object and pupil share that ambiguity.

`Simulator::simulate` compiles known linear camera response into the returned
reconstruction model: gains include photon conversion and electronic gain, while
background includes dark current and offset. Digitized counts can therefore be
passed directly to `ReconstructionProblem`; clipping, quantization, noise,
pixel-response variation, and bad pixels remain deliberate nonlinear effects or
model mismatch. A serialized `SimulationConfiguration` keeps
`compiled_models.reconstruction_model` as the strict optical model; use
`reconstruction_model_for_counts()` when making a problem from its saved detector
counts. Optical background belongs in both optical models; detector dark current
and electronic offset belong in `CameraModel`.

Synthetic objects include amplitude-only and phase-only fields, mixed patterns,
resolution targets, particles, random phase, Siemens stars, and seeded
biological-like phase/absorption fields. Image-backed constructors normalize
grayscale values to physical amplitude and a selected phase range; empty or
colour images, mismatched dimensions, and invalid phase ranges are rejected.
