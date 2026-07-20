# fpm-rs

Reconstruct and simulate image-plane Fourier ptychographic microscopy (FPM)
acquisitions from Rust or Python.

`fpm-rs` gives microscopy researchers and algorithm developers a CPU-based core
for compiling illumination geometry, simulating measurements, and recovering a
complex object. It supports image-plane FPM only: diffraction-plane
ptychography, multislice propagation, and GPU execution are not implemented.

## What it can do for you

- **Compile an optical model.** Describe planar, spherical, calibrated, or
  coded illumination and turn it into Fourier-space sampling and a pupil.
- **Simulate an acquisition.** Generate ideal or camera-affected intensity
  frames from synthetic or supplied complex objects.
- **Reconstruct the object.** Start with alternating projection, or use FPIE,
  EPRY, ADMM, or gradient descent when their calibration and regularization
  options fit the experiment.
- **Keep a run inspectable.** Record diagnostics, write checkpoints, and
  compare simulations with known ground truth.

## A typical workflow

1. Compile the optics and illumination into an `ImagePlaneModel`.
1. Load measured intensity frames, or simulate them from a known object.
1. Build a `ReconstructionProblem`, run an algorithm, and inspect amplitude,
   phase, diagnostics, or checkpoints.

The same forward model is used for simulation and reconstruction, while
algorithms consume the compiled model rather than experimental geometry.

## Start here

Install the Python package:

```console
python -m pip install fpm-rs
```

Then compile a small model, simulate it, and reconstruct it:

```python
import numpy as np
import fpm_rs as fpm

optics = fpm.Optics(532e-9, 0.10, 4.0, 6.5e-6)
leds = fpm.LEDArray((3, 3), 4e-3, 90e-3, (1.0, 1.0))
model = fpm.compile_model(optics, leds, (32, 32))

simulation = fpm.simulate(
    model,
    np.ones(model.reconstruction_shape, dtype=np.complex128),
    seed=1234,
)
problem = fpm.ReconstructionProblem(
    simulation.measurements,
    simulation.reconstruction_model,
)
result = fpm.AlternatingProjection(iterations=20).run(problem)
print(result.amplitude.shape)
```

For the Rust equivalent, run:

```console
cargo run --example simulate_and_reconstruct
```

## Go deeper

- [Documentation](https://hgrecco.github.io/fpm-rs/): installation, tutorials,
  guides, concepts, and API references.
- [Quickstart](docs/getting-started/quickstart.md): a narrated Python workflow.
- [Reconstruction guide](docs/guides/reconstruction.md): model sizing,
  algorithms, callbacks, and checkpoints.
- [Measurements](docs/guides/measurements.md), [simulation](docs/guides/simulation.md),
  and [datasets](docs/datasets.md): prepare real data and create test cases.
- [Contributor guide](docs/development/index.md): build, test, and document a
  checkout.

## License

fpm-rs is available under the [MIT License](LICENSE-MIT).
