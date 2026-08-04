# Changes

This file records notable user-facing changes. The newest release is listed
first.

## 0.2.0 (unreleased)

### Breaking changes

- Replaced the combined illumination variants with `SourceGeometry`,
  `SourceCalibration`, `AcquisitionPlan`, and atomic `Illumination::resolve`.
  Planar geometry is now `PlanarLedArray`/`PlanarLEDArray`; physical positions,
  propagation directions, sparse frame contributions, explicit source powers,
  and illumination/objective refractive indices have unambiguous ownership.
  Configuration format version 2 contains only the decomposed schema.
- Replaced the crate's custom `Array2` type with native `ndarray` arrays and
  views throughout the public Rust numerical API. FFT, model, measurement,
  reconstruction, and persistence boundaries now require standard row-major
  storage and return an explicit layout error for strided inputs. Python uses
  the corresponding C-contiguous input contract; metric functions continue to
  accept strided NumPy arrays.
- Replaced `ReconstructionResult.history`, `admm_residual_history`, and the
  undifferentiated diagnostics map with a universal reconstruction `trace`,
  long-form `algorithm_metrics`, and `scalar_diagnostics`. Trace and diagnostic
  fields now use `objective` rather than `loss`, and `save_loss_csv` is now
  `save_trace_csv`.
- Replaced the Rust JSON `save_bundle`/`load_bundle` result format with the
  optional Parquet/NumPy `write_bundle`/`read_bundle` interface. The new bundle
  format is a validated directory artifact with stable tables, authoritative
  arrays, optional previews, and a manifest.

### Added

- Added automatic reconstruction-grid sizing. Callers can request the exact
  minimum geometry-valid shape, a smooth FFT-friendly shape, or a power-of-two
  shape, and can inspect the choice with `suggest_reconstruction_shape`.
- Added focused intensity and complex-field metrics in Rust and Python,
  including error, correlation, PSNR, SSIM, Poisson-deviance, fitted-gain, and
  complex-alignment comparisons. Reconstruction evaluation remains separate
  from optimization objectives.
- Added self-describing result bundles and normalized benchmark suites backed
  by Parquet tables and `.npy` arrays, with lazy, cached, read-only Python array
  access and artifact verification.
- Added an algorithm-neutral reconstruction trace to every result, plus typed
  per-algorithm iteration metrics and richer benchmark records.
- Added Python 3.12 support alongside Python 3.13 and 3.14.

### Changed

- Reworked diagnostic records, reports, and plots around objective histories,
  per-frame summaries, Fourier coverage, raw-stack statistics, and explicit
  evaluation data.
- Expanded and reorganized the documentation for array-layout contracts,
  reconstruction sizing, metrics, diagnostics, result bundles, benchmarks,
  measurements, and simulation.
- Rebuilt continuous integration and release automation around independent
  Rust, Python, documentation, dependency, compatibility, and hardening gates.
  PyPI releases now build and test wheels for CPython 3.12--3.14 on Linux
  x86-64/ARM64, macOS Intel/Apple Silicon, and Windows x64, plus a tested source
  distribution.

### Fixed

- Validate all image-plane model invariants when deserializing configurations,
  including array layouts and geometry-dependent fields.
- Isolate Python source-release tests in a clean virtual environment so they do
  not accidentally import or install against the checkout environment.

## 0.1.0 (2026-07-17)

First public release.

### Added

- Added a CPU implementation of image-plane Fourier ptychographic microscopy
  simulation and reconstruction with a shared forward model, cached FFT plans,
  reusable workspaces, subpixel Fourier crops, and aberrated pupil sampling.
- Added planar LED arrays, fixed and moving spherical illumination geometries,
  calibrated wave vectors, per-frame gains, and incoherently multiplexed coded
  illumination.
- Added resident and lazy measurement stacks, masks, metadata, preprocessing,
  manifest loading, and concurrent lazy-frame decoding.
- Added alternating projection, FPIE, EPRY, linearized ADMM, and gradient-descent
  reconstruction with schedules, batches, checkpoints, callbacks,
  regularization, pupil recovery, illumination recovery, and deterministic
  parallel gradient updates.
- Added ideal and camera-affected simulation, deterministic presets, synthetic
  objects, acquisition mismatch controls, and masked ground-truth metrics.
- Added structured diagnostics, convergence history, Fourier coverage,
  per-frame and raw-stack summaries, image/CSV/checkpoint callbacks, benchmark
  records, and early stopping.
- Defined the portable dataset bundle specification and added offline loading,
  deterministic subsets, registry discovery, verified downloads, managed
  caching, and the `fpm-datasets` command-line interface.
- Added typed PyO3 bindings for experiment configuration, models,
  measurements, simulation, reconstruction, callbacks, checkpoints,
  diagnostics, datasets, plots, and reports. Blocking native work releases the
  Python GIL.
- Added an MkDocs documentation site, Python tutorials and notebooks, generated
  Rust and Python API references, reproducible Pixi environments, validation
  hooks, and CI for Rust, Python, documentation, notebooks, and security.
- Published the MIT-licensed Python package with CPython 3.13 wheels for Linux
  x86-64/ARM64, macOS Intel/Apple Silicon, and Windows x64, together with a
  source distribution; Python 3.14 was covered by source tests.
