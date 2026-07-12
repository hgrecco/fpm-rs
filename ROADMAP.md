# Roadmap

This file contains the project’s unfinished, in-scope work. Completed work and
historical implementation notes are kept in version control history.

## Core reconstruction

- [ ] Implement a CUDA backend with cuFFT and resident kernels for Fourier
  crops, pupil operations, intensity formation, projection, and reductions.
- [ ] Keep reconstruction state and scratch buffers device-resident across AP,
  FPIE, EPRY, ADMM, and gradient-descent updates.
- [ ] Add CPU/GPU numerical-parity, unsupported-device, performance, and memory
  tests.

## Dataset format and loading

- [ ] Add a lazy `dataset_spec` loading path for frame folders and large TIFF
  stacks without weakening bundle validation.
- [ ] Expand generic bundle tests for malformed manifests, inconsistent frame
  counts and shapes, invalid illumination metadata, and multiplexed datasets.

## Simulation and validation

- [ ] Record stable metric bounds for the named deterministic simulation
  presets.
- [ ] Add deterministic presets and robustness thresholds for illumination,
  frame-gain, background, saturation, bad-pixel, and physically documented
  vignetting mismatch.
- [ ] Add a smooth phase-only synthetic object with an explicit spatial
  bandwidth.
- [ ] Add a compact table-driven convention suite covering representative
  wavelengths, pixel sizes, magnifications, and LED layouts.

## Diagnostics and benchmarks

- [ ] Record resolved source frame indices, illumination associations, and
  spatial crops in benchmark records produced from dataset subsets.

## Documentation

- [ ] Add a focused guide for implementing and validating a non-CPU resident
  backend when the first such backend is available.
