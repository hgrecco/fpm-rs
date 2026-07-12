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
- When documenting a scientific method, state implementation location, public
  API, assumptions, approximations, and references. Do not invent citations.
- If docs and code disagree, report the discrepancy and update the most suitable
  existing document.

## Useful commands

```sh
cargo test
cargo test --all-targets
cargo doc --workspace --no-deps --all-features
cargo run --example simulate_and_reconstruct
cargo run --example benchmark_algorithms
pixi run -e py313 python-test
```
