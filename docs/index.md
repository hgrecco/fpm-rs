# fpm-rs

`fpm-rs` is a CPU library for reconstructing and simulating **image-plane
Fourier ptychographic microscopy (FPM)** acquisitions. It combines a Rust core
for physical modelling and iterative algorithms with a typed Python interface
for NumPy-based scientific workflows.

Use it to compile illumination geometry into a Fourier-space model, simulate
detector measurements, reconstruct a complex object with AP, FPIE, mPIE, EPRY,
ADMM, or gradient descent, and record diagnostics or checkpoints. It is intended for
microscopy researchers, algorithm developers, and applications that need a
reusable FPM core. Diffraction-plane ptychography, multislice propagation, and
GPU execution are outside the current scope.

## Install from PyPI

With Python 3.12 or newer:

```sh
python -m pip install fpm-rs
```

For notebook and plotting support, install `"fpm-rs[notebook]"`. See
[Installation](getting-started/installation.md) for supported platforms, source
builds, Pixi setup, and troubleshooting.

## Minimal example

```python
import numpy as np
import fpm_rs as fpm

optics = fpm.Optics(532e-9, 0.10, 4.0, 6.5e-6)
geometry = fpm.PlanarLEDArray(
    (3, 3), 4e-3, (1.0, 1.0),
    fpm.ArrayPose.from_translation((0.0, 0.0, -90e-3)),
)
illumination = fpm.Illumination(geometry)
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
result = fpm.AlternatingProjection(iterations=10).run(problem)
print(result.amplitude.shape)  # (42, 42)
```

Continue through the [Quickstart](getting-started/quickstart.md), run the
[first reconstruction tutorial](tutorials/notebooks/quickstart.ipynb), or go
straight to the [Python API](reference/python/index.md).

## Library components

- `experiment` separates source geometry, stable calibration, sparse acquisition,
  and resolved illumination state.
- `model` compiles sampling, pupil, Fourier crops, and the shared forward model.
- `measurements` provides resident and lazy intensity stacks plus preprocessing.
- `algorithms` implements AP, FPIE, mPIE, EPRY, linearized ADMM, and gradient descent.
- `reconstruction` provides problems, state, results, schedules, batches, and
  runner orchestration.
- `callbacks` records images, CSV history, checkpoints, progress, and early
  stopping.
- `simulation` supplies synthetic objects, acquisition effects, camera response,
  and ground-truth metrics.
- `datasets` validates local bundles, discovers and downloads registered bundles,
  manages cache, and selects deterministic subsets.
- `backend` provides the CPU backend and future resident-buffer boundary.
- `benchmark` runs single cases and writes versioned benchmark records and
  artifacts.
- `tabular` optionally converts results, diagnostics, evaluation, and
  benchmarks to Polars tables and writes Parquet/NumPy result bundles.

## Documentation map

- [Tutorials](tutorials/index.md) provide executable, public-API workflows.
- [Guides](guides/reconstruction.md) answer task-oriented questions.
- [Concepts and conventions](concepts/core-concepts.md) explain the physical
  and computational model.
- [Python API](reference/python/index.md) is generated from the typed package
  surface and docstrings.
- [Rust API](reference/rust.md) links to native rustdoc for lower-level use.
- [Contributor guide](development/index.md) covers builds, tests, notebooks,
  and documentation.
- [Datasets](datasets.md) documents the generic converted-bundle workflow.
- [Project information](project.md) covers citation, licensing, and the roadmap.
