# Contributor and development guide

## Repository layout

- `src/` is the public Rust `fpm-rs` library. Experiment geometry compiles into
  the model consumed by algorithms, reconstruction, and simulation.
- `python/src/` is the private PyO3 extension crate; `python/fpm_rs/` is the
  typed Python package and plotting/report layer.
- `docs/` contains this authored site and its curated tutorial notebooks.
- `examples/` and `python/examples/` contain Rust workflows and focused Python
  notebook examples not all intended for publication.
- `tests/` and `python/tests/` contain integration and API tests.

## Build and test

```sh
cargo test
cargo test --all-targets
cargo doc --workspace --no-deps --all-features
pixi run -e py313 python-test
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
pixi run -e py313 python-develop
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
