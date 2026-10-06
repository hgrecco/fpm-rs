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
formatting and Clippy checks, warning-free all-feature Rustdoc, all-feature Rust
tests, doctests and examples, Python API documentation and runtime/stub checks,
and the complete Python suite with Polars, Matplotlib, and IPython installed.
The heavy Rust test task serializes linker work. Individual tasks include
`rust-format`, `rust-clippy`, `rust-test`, `rust-doc`, `rust-msrv`,
`python-api-docs`, `python-test`, and `python-test-full`. Formatting tasks that
modify sources remain `format-rust`, `format-python`, and `format-toml`; `pixi
run lint` runs the configured pre-commit checks. Dataset tests use generated
local bundles and never require network access.

Choose validation in proportion to the change. Focused implementation changes
start with their affected Rust or Python tests. Public Rust API and rustdoc
changes require `pixi run rust-doc` and `pixi run rust-doc-test`; public Python
API, signature, stub, or docstring changes require `pixi run python-api-docs`.
Authored-site, navigation, and notebook changes require `pixi run docs-build`,
while citation changes additionally require `pixi run citation-check`. Use
`pixi run ci` for cross-cutting changes or when complete source validation is
warranted; a narrow prose correction does not require every unrelated suite.

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

### Python API documentation sources

The checked-in `.pyi` files are the authoritative generated-reference source:
`python/fpm_rs/__init__.pyi` describes the root extension API, while
`metrics.pyi` and `evaluation.pyi` describe those public modules. Public Python
wrappers retain their own docstrings. `scripts/mkdocs_hooks.py` stages those
sources under `target/docs-python-api/` so mkdocstrings renders Python
signatures and terminology without exposing the private PyO3 implementation.
Edit the checked-in source, never the staged copy.

`python/src/` remains authoritative for extension behavior and runtime
signatures. When an API changes, update the binding and stub together; keep a
publication citation in the public stub and use the same publication in a
runtime docstring that also discusses the method. `scripts/check_python_api.py`
compares exports, public class members, and callable parameter shapes against an
installed local extension. `scripts/check_python_docs.py` checks every routed
public stub and wrapper item for meaningful prose without a broad exclusion
list. Both run through `pixi run python-api-docs` after `maturin develop`.
That task also checks keyword-oriented calls across `README.md`, the site landing
page, the quickstart, and its tutorial notebook so their recommended calling
convention cannot drift silently.

## Physical calibration changes

Keep physical `PlanarLedArray` calibration distinct from generic independent
k-vector or Fourier-grid correction. Physical parameters retain SI units and
the active, right-handed, extrinsic XYZ rotation convention; arbitrary source
shifts are not realizable apparatus calibration.

Keep illumination geometry, stable `SourceCalibration`, and sparse canonical
`AcquisitionPlan` state separate and resolve them atomically through
`Illumination`. The planar-array calibrator may own the corresponding `Optics`
and `Illumination`, but its object phase must call the canonical compiled
forward model and measurement loss. Preserve Rust/Python illumination parity,
sources normally at negative sample `z`, positive-`z` incident propagation, and
the separately documented propagation-vector/Fourier-crop sign convention.

Preserve the identifiability checks: reject lateral translation with its
corresponding reference index, constrain selected offset means when translation
is active, normalize source powers and frame gains to mean one, and never
silently enable source offsets or an unconstrained source-power/frame-gain
combination. Do not add geometry-level wavelength overrides, acquisition
ordering, powers, gains, angle-list or coded-geometry types, or pre-release
compatibility layers.

Update physical calibration results, callbacks, checkpoints, bundles,
serialization, Rust and Python APIs, authoritative stubs, and synthetic recovery
tests together. New physical parameters require bounded deterministic recovery
tests and partial-update regression coverage.

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

The documentation build first checks Python documentation coverage,
runtime/stub synchronization, and citation syntax. It then builds MkDocs in
strict mode and workspace Rustdoc with all features and `-D warnings`.
`#![deny(missing_docs)]` in the library crate makes missing public Rust
documentation a compiler error. The same gates run in pull-request CI; the
fast citation check also runs in pre-commit.

### Scientific citations

Cite a source at the closest durable explanation of every
publication-derived algorithm, formula, threshold, convention, dataset, or
scientific claim. A citation does not replace an explanation of what this
implementation does, its assumptions, or its material differences from the
cited work.

State the implementation location, public API, assumptions, approximations,
and references when documenting a scientific method.

Use the primary source when available. Verify bibliographic metadata against
the canonical DOI resolver and an authoritative publisher or archive; never
infer authors, titles, venues, dates, pages, or article numbers. Give a complete
reference: authors, a linked title or author-year label, venue, volume and issue
when applicable, page range or article number, and year. Link to the canonical
`https://doi.org/...` URL rather than displaying a bare DOI or using another
resolver.

Place the full reference according to the public surface:

- In Rust, use a `# References` section in the rustdoc for the public item that
  exposes the method or behavior. Add an implementation comment only when a
  formula, sign, constant, or translation would otherwise be hard to audit;
  identify the source there by author and year.
- In Python, use a NumPy-style `References` section in the authoritative
  checked-in stub. Keep corresponding wrapper and PyO3 runtime docstrings
  semantically synchronized when they expose the same method or claim.
- In Markdown and notebooks, put the full reference in the paragraph making
  the claim or in a clearly linked `References` section on the same page.
- In tests, examples, and fixtures, cite a publication only when an expected
  value, convention, or dataset comes from it. Explain the derivation or
  transformation nearby and point to the public documentation containing the
  full reference.

`CITATION.cff` describes how to cite `fpm-rs`; it is not the bibliography for
scientific methods. Update it only for software authorship, release version,
title, repository, or preferred software-citation changes, and keep its version
synchronized with `Cargo.toml` and `pyproject.toml` at every release. Treat it
as part of the release checklist alongside `CHANGES.md`.

Run `pixi run citation-check` after adding or changing citations, then render
the affected Rustdoc, Python API documentation, or MkDocs page. Live publisher
checks are intentionally not a merge gate because publisher outages and bot
blocking are nondeterministic; report the DOI resolver and publisher or archive
used to verify changed metadata.

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
