# Repository guide for agents

## Scope

`fpm-rs` is a Rust library with Python bindings for image-plane Fourier
ptychographic microscopy. Do not expand it into diffraction-plane ptychography,
multislice models, or GPU execution unless the task explicitly changes project
scope.

## Always-on rules

- Before editing inside a subtree, read and follow the nearest scoped
  `AGENTS.md`; scoped instructions add to or override this file for that subtree.
- Preserve the architecture boundary: experiment geometry compiles into
  `ImagePlaneModel`; reconstruction algorithms consume the compiled model and
  `MeasurementRead`, not LED geometry.
- Builds and tests must remain offline. Registry operations may use the network
  only when explicitly requested or while opening a registered dataset ID.
- Preserve explicit SI-unit names, the documented illumination propagation and
  Fourier-crop sign conventions, and Rust/Python behavior parity.
- Prefer improving an existing document to creating a new one. Keep persistent
  project work only in `ROADMAP.md`; transient conversational or tool plans are
  allowed, but do not add competing roadmap, backlog, or plan files.
- Keep `CONTRIBUTING.md` as a thin pointer to contributor documentation. Treat
  `SECURITY.md` and `CODE_OF_CONDUCT.md` as standalone canonical policies; do
  not replace them with pointers into the documentation site.
- If code and documentation disagree, establish the intended behavior before
  changing either side and report any unresolved discrepancy.

## Read when relevant

- Use `README.md` for the public architecture and `docs/development/index.md`
  for the detailed repository map, source-of-truth rules, and validation tasks.
- Before changing physical illumination calibration, read
  `docs/development/index.md#physical-calibration-changes`.
- Before adding or changing a publication-derived method, formula, threshold,
  convention, dataset, or scientific claim, read
  `docs/development/index.md#scientific-citations`.
- Dataset work also follows `dataset_spec.md`, `docs/datasets.md`, and
  `src/datasets/AGENTS.md`.
- Before changing a first-touch Python example or recommended calling
  convention, read `docs/AGENTS.md` and keep `README.md`, `docs/index.md`,
  `docs/getting-started/quickstart.md`, and
  `docs/tutorials/notebooks/quickstart.ipynb` synchronized.
- Keep all public Rust and Python behavior, documentation, bindings, stubs,
  examples, serialization, and tests synchronized where the affected API is
  exposed.

## Validation and handoff

- Run the smallest relevant checks first, then broaden them in proportion to
  the change. `pixi run ci` is the canonical complete source suite.
- Do not add network access to validation. Do not commit generated `site/` or
  `target/` output.
- Report the public items and examples changed, semantic convention changes,
  checks run, and unresolved inconsistencies. Report citation verification only
  when citations changed.
