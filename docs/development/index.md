# Contributor and development guide

## Repository layout

- The workspace has two members. The root `fpm-rs` crate is the public Rust
  library. The private `python/` member builds `fpm_rs._core` and depends on the
  root crate with Parquet support; the Rust library never depends on Python.
- `src/experiment` compiles physical optics and illumination geometry into the
  transverse wave vectors consumed by `src/model`. Algorithms depend on
  `ImagePlaneModel` and `MeasurementRead`, not experiment geometry.
- `src/reconstruction` owns problems, standard-layout numerical state,
  algorithm-neutral traces, checkpoints, results, schedules, and the runner.
  `src/algorithms` supplies typed step metrics; `src/callbacks` and
  `src/diagnostics` request and record optional derived data.
- `src/tabular` converts domain objects into Polars `DataFrame` values behind
  the `tabular` feature. Its `parquet` submodule writes versioned result
  bundles. `src/benchmark_bundle.rs` composes normalized comparison tables with
  nested result bundles.
- `python/src/` is the PyO3 binding implementation. `python/fpm_rs/` is the
  typed public package plus plotting and reporting helpers. NumPy is the array
  boundary; Parquet paths are exposed as ordinary artifact handles so users can
  choose Polars, PyArrow, pandas, or DuckDB themselves.
- `docs/` contains this authored site and its curated tutorial notebooks.
- `examples/` and `python/examples/` contain Rust workflows and focused Python
  notebook examples not all intended for publication.
- `tests/` and `python/tests/` contain integration and API tests.

## Array and ownership map

The public Rust numerical API uses `ndarray::Array2`, `Array3`, and their view
types. Metrics and pointwise utilities accept compatible strided views. Strict
FFT, backend, measurement, pupil, reconstruction-state, and persistence
boundaries require standard row-major storage and return `NonStandardLayout`
instead of copying. Private `StandardArray2` and `StandardArray3` wrappers keep
that invariant for long-lived core state. Flat `Vec` buffers remain appropriate
for FFT scratch, backend workspaces, ragged records, and serialized rows.

Python inputs are NumPy arrays. General metrics accept strided views, while
model and reconstruction inputs require C-contiguous arrays. Reconstruction
results returned directly by an algorithm own normal NumPy arrays. Arrays
loaded from a `ResultBundle` are lazy, cached, shared with
`bundle.result`, and read-only.

## Execution and persistence map

Every run records a `ReconstructionTrace` independently of diagnostic
callbacks. Its one-based iteration rows contain `objective` and canonical
`elapsed_seconds`; algorithm-specific scalars are separate long-form
`AlgorithmMetricRecord` rows. Checkpoints serialize the trace together with
the object spectrum, pupil, calibration, and algorithm auxiliary state so a
resume continues iteration numbering and elapsed time.

Callbacks may write checkpoint JSON, objective CSV, PNG snapshots, or residual
images. The diagnostic recorder optionally retains iteration, frame, raw-stack,
coverage, and snapshot data. A final result bundle instead writes stable
Parquet tables, authoritative `.npy` arrays, optional PNG previews, and a
manifest written last. Benchmark bundles contain normalized runs, frames,
artifacts, and metadata tables plus one nested result bundle for each successful
run. Local dataset manifests and diagnostic JSON remain separate formats with
different purposes.

## Build and test

```sh
cargo test
cargo test --all-targets
cargo doc --workspace --no-deps --all-features
pixi run -e py312 python-test
pixi run lint
```

Formatting tasks are `pixi run format-rust`, `pixi run format-python`, and
`pixi run format-toml`. `pixi run lint` runs the configured pre-commit checks.
Dataset tests use generated local bundles and never require network access.

## Python extension

`pyproject.toml` uses maturin with `python/Cargo.toml`, module name
`fpm_rs._core`, and Python sources under `python/`. The build-aware development
task is:

```sh
pixi run -e py312 python-develop
```

Keep Python-visible signatures synchronized with
`python/fpm_rs/__init__.pyi`. Document array shapes, dtypes, physical units,
ownership, blocking/GIL behavior, return values, and typed failures when adding
public calls.

## Notebook maintenance

Published notebooks live in `docs/tutorials/notebooks/`; keep them small,
self-contained, deterministic, free of machine paths, and based on public APIs.
Do not commit large outputs. Their code cells are validated by
`python/tests/test_diagnostic_notebooks.py`; MkDocs renders them without
execution. Advanced exploratory examples can remain under `python/examples/`
without entering site navigation.

## Documentation

```sh
pixi run docs-serve
pixi run docs-build
pixi run docs-preview
```

`docs-serve` provides live reload at `http://127.0.0.1:8000/` for Markdown,
notebooks, and Python API pages. It does not continuously rebuild rustdoc.
`docs-build` creates the strict combined site in `site/`. `docs-preview` first
builds that exact output, including rustdoc, then serves it at the same address.

The Pages workflow invokes `pixi run docs-build`, so local and deployed builds
share one implementation. Generated `site/` and isolated rustdoc output under
`target/docs-rust/` are ignored.

## Release checks

Before pushing a release tag, run the Rust and Python matrices, notebook
validation, strict documentation build, package build, and the manual hardening
workflow. A push of `v<package-version>` starts the Python workflow, checks that
the tag matches `pyproject.toml`, builds wheels plus an sdist, smoke-tests the
wheels, and then publishes the verified artifacts to PyPI.

The upload uses PyPI Trusted Publishing rather than a stored token. Before the
first release, configure PyPI to trust the `hgrecco/fpm-rs` repository's
`.github/workflows/python.yml` workflow and its `pypi` environment; PyPI supports
a pending publisher for a project that does not yet exist. See the
[PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
for that one-time configuration.
