# Repository guide for agents

## Scope

`fpm-rs` is a Rust library with Python bindings for image-plane Fourier
ptychographic microscopy. Diffraction-plane ptychography, multislice models, and
GPU execution are not implemented today.

## Structure

- `src/experiment`: physical optics and illumination sources. Geometry compiles
  to transverse `KVector` values and frame/source weights.
- `src/model`: algorithm-facing `ImagePlaneModel`, Fourier crops, pupil
  sampling, FFT-shift helpers, and the shared forward model.
- `src/measurements`: resident and lazy measurement stacks, masks, metadata,
  preprocessing, and JSON measurement manifests.
- `src/algorithms`: AP, FPIE, EPRY, ADMM, gradient descent, and regularization
  updates. Algorithms consume `ReconstructionProblem<MeasurementRead>` and
  `ImagePlaneModel`, not LED geometry.
- `src/reconstruction`: problem/state/result types, runner, schedules, batches,
  checkpoints, and run options.
- `src/simulation`: synthetic objects, simulator, camera response, acquisition
  effects, deterministic presets, and ground-truth metrics.
- `src/datasets`: `dataset_spec` manifest validation, registry discovery,
  verified downloads, managed caching, generic loading, and deterministic
  subsets. Source-specific acquisition and conversion are external.
- `src/diagnostics` and `src/callbacks`: convergence/history recording,
  per-frame diagnostics, reports, CSV/images/checkpoints, and early stopping.
- `src/backend`: CPU backend and resident-buffer interface for future device
  backends.
- `src/benchmark.rs`: reusable single-case benchmark records and artifact
  writers.
- `python/`: PyO3 bindings, typed Python package surface, tests, diagnostics
  CLI/reporting, and focused example notebooks.
- `docs/`: authored MkDocs documentation and curated tutorial notebooks.
- `examples/`: runnable Rust workflows.
- `tests/`: Rust integration tests and regression fixtures.

## Documentation map

- Start with `README.md` for architecture, conventions, modules, and common
  workflows.
- Use `docs/spherical-geometries.md` for spherical source equations and
  identifiability limits.
- Use `dataset_spec.md` and `docs/datasets.md` for the dataset contract,
  provenance, and loader behavior.
- Use `docs/benchmarks.md` for benchmark records and presets.
- Use `docs/reference/` for generated API entry points.
- Keep all pending work in the root `ROADMAP.md`; do not create parallel
  roadmaps, backlogs, or implementation plans.

## Working rules

- Prefer improving existing docs over creating new ones. Create a new document
  only when no existing document has the right purpose.
- Treat `README.md` and rustdoc as public API documentation. Keep links to
  focused docs instead of copying long sections.
- Do not add network access to builds or tests. Registry operations may access
  the network explicitly or while opening a registered ID; local
  `DatasetLoader` remains offline. Source-specific conversion stays external.
- Preserve the architecture boundary: algorithms use compiled models and
  measurement traits, while experiment geometry stays in `experiment` and
  configuration/loading code.
- Keep illumination geometry, stable `SourceCalibration`, and sparse canonical
  `AcquisitionPlan` state separate. Resolve them atomically through `Illumination`;
  ordinary algorithms consume only `ImagePlaneModel`. The physical planar-array
  calibrator may own the corresponding `Optics` and `Illumination`, but its object
  phase must still call the canonical compiled forward model and measurement loss.
- Keep physical `PlanarLedArray` calibration distinct from generic independent
  k-vector/Fourier-grid correction. Physical parameters retain SI units and the
  active, right-handed, extrinsic XYZ rotation convention; never label arbitrary
  source shifts as realizable apparatus calibration.
- Preserve physical-calibration identifiability checks: reject lateral translation
  with its corresponding reference index, constrain selected offset means when
  translation is active, normalize source powers and frame gains to mean one, and
  do not silently enable source offsets or an unconstrained power/gain combination.
- Update physical calibration results, callbacks, checkpoints, bundles,
  serialization, Rust/Python APIs, authoritative stubs, and synthetic recovery
  tests together. New physical parameters require bounded deterministic recovery
  tests and partial-update regression coverage.
- Preserve Rust/Python illumination parity, explicit SI-unit field names, sources
  normally at negative sample `z`, positive-`z` incident propagation, and the
  separately documented propagation-vector/Fourier-crop sign convention.
- Do not add geometry-level wavelength overrides, acquisition ordering, powers,
  gains, angle-list or coded-geometry types, or pre-release compatibility layers.
- When documenting a scientific method, state implementation location, public
  API, assumptions, approximations, and references. Do not invent citations.
- If docs and code disagree, report the discrepancy and update the most suitable
  existing document.

## Public API documentation policy

- Documentation is part of every intentionally public Rust or Python API
  change. Update affected reference prose, examples, bindings, stubs, and tests
  in the same change whenever behavior, units, shapes, ordering, ownership,
  validation, errors, or return values change. Bare signatures, placeholders,
  and prose that only repeats an identifier or type are not acceptable.
- Rust public crates, modules, types, traits, associated items, fields,
  variants, constants, functions, and methods require semantic rustdoc; new
  modules require useful `//!` landing documentation. Keep
  `#![deny(missing_docs)]` enabled and add small compiling examples for principal
  workflows.
- The checked-in public stubs under `python/fpm_rs/` are authoritative for
  generated Python signatures and long-form API prose. Keep them synchronized
  with `python/src/` bindings and public Python wrappers. Use NumPy-style
  docstrings and document scientific array shape, dtype, axis order, copying,
  mutability, units, optional values, failures, and blocking behavior where
  relevant. Do not expose private `_core` implementation details as public API.
- Cite publication-derived methods and scientific claims near their use with
  complete, verified bibliographic metadata and a clickable canonical
  `https://doi.org/...` link when available. Prefer primary sources, document
  material implementation differences, and never leave bare DOI strings or
  guess citation metadata.
- Evaluate API changes in generated Rustdoc and MkDocs pages. Keep Python API
  domains split by purpose, source implementations collapsed, stable symbol
  links discoverable, and generated `site/` or `target/` output uncommitted.

Run `pixi run rust-doc`, `pixi run rust-doc-test`, `pixi run python-api-docs`,
`pixi run docs-build`, and relevant examples/notebook tests for documentation
changes; `pixi run ci` is the canonical complete source suite. Report public
items and examples changed, semantic convention changes, citations and their
verification sources, checks run, and any unresolved API or attribution
inconsistency.

## Useful commands

```sh
cargo test
cargo test --all-targets
pixi run rust-doc
pixi run python-api-docs
pixi run docs-build
cargo run --example simulate_and_reconstruct
cargo run --example benchmark_algorithms
pixi run -e py312 python-test
pixi run ci
```
