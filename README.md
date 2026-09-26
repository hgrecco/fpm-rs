# fpm-rs

[![CI](https://github.com/hgrecco/fpm-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/hgrecco/fpm-rs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fpm-rs.svg)](https://pypi.org/project/fpm-rs/)

Reconstruct and simulate image-plane Fourier ptychographic microscopy (FPM)
acquisitions from Rust or Python.

`fpm-rs` gives microscopy researchers and algorithm developers a CPU-based core
for compiling illumination geometry, simulating measurements, and recovering a
complex object. It supports image-plane FPM only: diffraction-plane
ptychography, multislice propagation, and GPU execution are not implemented.

## What it can do for you

- **Compile an optical model.** Resolve source geometry, stable calibration,
  and sparse acquisition structure into Fourier-space sampling and a pupil.
- **Simulate an acquisition.** Generate ideal or camera-affected intensity
  frames from synthetic or supplied complex objects.
- **Reconstruct the object.** Start with alternating projection, or use FPIE,
  EPRY, ADMM, or gradient descent when their calibration and regularization
  options fit the experiment.
- **Calibrate a planar LED array physically.** Alternate analytic object or
  object/pupil updates with bounded pose, pitch, reference-index,
  selected-offset, source-power, or frame-gain updates and reuse the returned
  `Illumination`.
- **Keep a run inspectable.** Record diagnostics, write checkpoints, and
  compare simulations with known ground truth. Optional Parquet support writes
  self-describing result and benchmark bundles for downstream analysis.
- **Measure at the right layer.** Reusable reference/estimate metrics,
  optimization objectives, reconstruction evaluation, and recorder-driven
  diagnostics are separate APIs.

## A typical workflow

1. Compile the optics and illumination into an `ImagePlaneModel`.
1. Load measured intensity frames, or simulate them from a known object.
1. Build a `ReconstructionProblem`, run an algorithm, and inspect amplitude,
   phase, diagnostics, or checkpoints.

The same forward model is used for simulation and reconstruction, while
algorithms consume the compiled model rather than experimental geometry.

## Array layout contract

The Rust API uses native `ndarray` arrays and views. Pointwise utilities and
metrics accept arbitrary logical layouts, including transposed and stepped
views. Computational inputs whose kernels use flat Fourier-grid offsets—such
as pupils, synthetic objects, measurement stacks, crop operations, and direct
forward-model spectra—require standard C-style row-major layout. Those APIs
validate the layout without copying and return a `NonStandardLayout` error for
strided input. Owned result and diagnostic arrays are ordinary `ndarray`
arrays. Result bundle export is a strict persistence boundary and rejects
nonstandard result layouts rather than silently materializing a copy.

Python follows the same distinction. Metrics accept strided NumPy arrays.
Reconstruction, simulation, pupil, and measurement inputs require
C-contiguous arrays and raise an error suggesting `numpy.ascontiguousarray`;
the caller therefore decides whether to pay for that copy. Accepted NumPy
inputs are copied into Rust-owned storage at the binding boundary. Arrays
opened through a result bundle are loaded on first access, cached, and exposed
as immutable NumPy arrays.

## Start here

Install the Python package:

```console
python -m pip install fpm-rs
```

Then compile a small model, simulate it, and reconstruct it:

```python
import numpy as np
import fpm_rs as fpm

optics = fpm.Optics(
    wavelength_vacuum_m=532e-9,
    objective_na=0.10,
    magnification=4.0,
    camera_pixel_size=6.5e-6,
)
geometry = fpm.PlanarLEDArray(
    shape=(3, 3),
    pitch_m=4e-3,
    reference_index=(1.0, 1.0),
    pose=fpm.ArrayPose.from_translation(
        translation_m=(0.0, 0.0, -90e-3),
    ),
)
illumination = fpm.Illumination(geometry=geometry)
model = fpm.compile_model(optics, illumination, (32, 32))

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
  algorithms, physical illumination calibration, callbacks, and checkpoints.
- [Measurements](docs/guides/measurements.md), [simulation](docs/guides/simulation.md),
  and [datasets](docs/datasets.md): prepare real data and create test cases.
- [Contributor guide](docs/development/index.md): build, test, and document a
  checkout.

## License

fpm-rs is available under the [MIT License](LICENSE-MIT).
