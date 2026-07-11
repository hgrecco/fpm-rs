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
