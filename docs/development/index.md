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
pixi run ci
```

This is the canonical local source-validation command. It composes strict Rust
formatting and Clippy checks, all-feature Rust tests, doctests and examples,
and the complete Python suite with Polars, Matplotlib, and IPython installed.
The heavy Rust test task serializes linker work. Individual tasks include
`rust-format`, `rust-clippy`, `rust-test`, `rust-msrv`, `python-test`, and
`python-test-full`. Formatting tasks that modify sources remain `format-rust`,
`format-python`, and `format-toml`; `pixi run lint` runs the configured
pre-commit checks. Dataset tests use generated local bundles and never require
network access.

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

The workflows have separate costs and responsibilities:

- `ci.yml` validates pull requests and `main` with comprehensive Linux Rust
  checks, representative macOS/Windows checks, a Python boundary matrix, and
  one optimized installed-wheel smoke test.
- `compatibility.yml` runs weekly or manually across Python 3.12--3.14 on
  Linux, macOS, and Windows, selected Rust feature combinations, and coverage.
- `docs.yml` validates path-filtered documentation changes and deploys Pages
  only from `main`.
- `hardening.yml` checks dependency policy when Cargo inputs change and runs
  pinned-nightly Miri and AddressSanitizer jobs weekly or manually.
- `release.yml` is the only artifact publication pipeline. Release tags run
  independent Rust, Python, documentation, and dependency gates before PyPI.

The extension does not enable a PyO3 `abi3` feature, so releases build a wheel
for every CPython ABI (3.12, 3.13, and 3.14) on Linux x86-64 and ARM64, macOS
Intel and ARM64, and Windows x64. Linux ARM64 wheels execute on native ARM64
runners. The sdist is installed and tested on the minimum and maximum Python
versions. Every distribution is built once, hashed, installed in a clean
environment, exercised, hash-verified, collected into the single
`verified-distributions` artifact, and published without rebuilding.

The MSRV and release compiler are separate policy choices even though both are
currently pinned to Rust 1.97.0. Change them independently when the project
raises compatibility requirements or adopts a newer release compiler.

To make an optimized wheel for the current machine, run:

```sh
pixi run python-wheel
```

Maturin remains the package builder and Cargo remains authoritative for Rust
resolution and compilation. A local native wheel is behaviorally comparable,
but is not expected to be byte-identical to CI's manylinux, macOS, or Windows
artifacts because their toolchains and target environments differ.

The upload uses PyPI Trusted Publishing rather than a stored token. Before the
first release, configure PyPI to trust the `hgrecco/fpm-rs` repository's
`.github/workflows/release.yml` workflow and its `pypi` environment; PyPI supports
a pending publisher for a project that does not yet exist. See the
[PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
for that one-time configuration.
