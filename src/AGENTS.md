# Rust source instructions

These instructions apply to Rust sources under `src/` in addition to the root
repository guide.

## Architecture and API

- Keep experiment geometry in `experiment` and configuration/loading code.
  Algorithms operate on `ReconstructionProblem<MeasurementRead>` and
  `ImagePlaneModel`.
- Public crates, modules, types, traits, associated items, fields, variants,
  constants, functions, and methods require semantic rustdoc. New modules need
  useful `//!` landing documentation; keep `#![deny(missing_docs)]` enabled.
- Add small compiling examples for principal public workflows. Explain behavior,
  units, shapes, ownership, validation, errors, and return values rather than
  restating identifiers.
- When Rust changes affect Python-visible behavior, update the PyO3 binding,
  authoritative stubs, public wrappers, examples, and tests in the same change.
- For publication-derived behavior, follow the scientific citation policy in
  `docs/development/index.md#scientific-citations`.
- For physical calibration work, follow
  `docs/development/index.md#physical-calibration-changes`.

## Validation

- Run focused Rust tests while iterating.
- For public rustdoc changes, run `pixi run rust-doc` and
  `pixi run rust-doc-test`.
- For cross-cutting Rust changes, use the relevant `pixi` tasks documented in
  `docs/development/index.md`; run `pixi run ci` when complete-suite validation
  is warranted.
