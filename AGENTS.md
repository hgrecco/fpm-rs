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
- Follow the citation policy below for publication-derived methods and
  scientific claims.
- Evaluate API changes in generated Rustdoc and MkDocs pages. Keep Python API
  domains split by purpose, source implementations collapsed, stable symbol
  links discoverable, and generated `site/` or `target/` output uncommitted.

Run `pixi run rust-doc`, `pixi run rust-doc-test`, `pixi run python-api-docs`,
`pixi run docs-build`, and relevant examples/notebook tests for documentation
changes; `pixi run ci` is the canonical complete source suite. Report public
items and examples changed, semantic convention changes, citations and their
verification sources, checks run, and any unresolved API or attribution
inconsistency.

## Citation policy

- Cite a source at the closest durable explanation of every
  publication-derived algorithm, formula, threshold, convention, dataset, or
  scientific claim. A citation does not replace an explanation of what this
  implementation does, its assumptions, or its material differences from the
  cited work.
- Use the primary source when available. Verify the bibliographic metadata
  against the canonical DOI resolver and an authoritative publisher or archive;
  never infer or invent authors, titles, venues, dates, pages, or article
  numbers.
- Give a complete reference: authors, linked title, venue, volume and issue when
  applicable, page range or article number, and year. Link the title or an
  author-year label to the canonical `https://doi.org/...` URL. Do not use a
  bare DOI, a DOI as the link label, or a noncanonical DOI resolver.
- In Rust, put the full reference in the Rustdoc for the public item that exposes
  the method or behavior, normally under `# References`. Use an implementation
  comment only when a formula, sign, constant, or translation from the paper is
  otherwise hard to audit; identify the source there by author and year and
  keep the full reference in the enclosing item's Rustdoc.
- In Python, put the full reference in the authoritative checked-in stub
  docstring under a NumPy-style `References` section. Keep any corresponding
  public wrapper docstring and PyO3 runtime docstring semantically synchronized
  when they expose the same method or claim.
- In Markdown and notebooks, place the full reference in the paragraph that
  makes the claim or in a clearly linked `References` section on the same page.
  Do not rely on a bibliography on another page to identify the source.
- In tests, examples, and fixtures, cite a publication only when an expected
  value, convention, or dataset comes from it. Explain the derivation or
  transformation in a nearby comment and point to the public documentation that
  contains the full reference.
- `CITATION.cff` describes how to cite `fpm-rs`; it is not the bibliography for
  scientific methods used by the code. Update it for software authorship,
  release version, title, repository, or preferred software-citation changes.
  Do not add a paper to it solely because Rustdoc, a docstring, or narrative
  documentation cites that paper.
- Run `pixi run citation-check` after adding or changing citations, then render
  the affected Rustdoc, Python API documentation, or MkDocs pages to verify that
  the reference is readable and its link is clickable.

## Narrative documentation and onboarding policy

- Keep `README.md`, `docs/getting-started/quickstart.md`, and
  `docs/tutorials/notebooks/quickstart.ipynb` on the same calling convention,
  using keyword arguments for multi-parameter constructors. Update all three
  together whenever a constructor signature or recommended idiom changes; do
  not let them drift.
- Keep `docs/getting-started/quickstart.md` to the shortest path to one
  successful reconstruction. Put sizing strategies, batching, regularization,
  calibration, and other secondary configuration in the relevant page under
  `docs/guides/`, and cross-link it instead of inlining it in the quickstart.
- Give every reconstruction algorithm an entry in the algorithm-selection
  guidance in `docs/guides/reconstruction.md`, in addition to its rustdoc and
  citation. Describe when to prefer it over the alternatives in the same
  change that adds the algorithm.
- Decompose every end-user one-call convenience or demo entry point into the
  primitives it wraps immediately in the same document. Present it as an
  installation sanity check and a worked example to modify, never as an
  unexplained replacement for the underlying walkthrough.
- Define domain terms introduced for non-specialists in
  `docs/concepts/glossary.md`, including numerical aperture, synthetic NA,
  pupil, k-vector, dark-field and bright-field illumination, pitch, defocus,
  and similar terms. Add a glossary entry when a guide or tutorial introduces
  a term not already defined there.
- Keep `CITATION.cff` synchronized with the version in `Cargo.toml` and
  `pyproject.toml` at every release. Treat it as part of the release checklist
  alongside `CHANGES.md`.
- Keep root `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` as thin,
  current pointers to canonical documentation such as
  `docs/development/index.md`. Do not duplicate canonical prose; improve the
  existing document instead.

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
