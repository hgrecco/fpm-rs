# Python instructions

These instructions apply to the PyO3 bindings, typed package, Python tests, and
Python examples.

## Public API

- `python/src/` is authoritative for extension behavior and runtime signatures.
  The checked-in stubs under `python/fpm_rs/` are authoritative for generated
  Python signatures and long-form API prose; never edit staged copies under
  `target/`.
- Keep PyO3 bindings, authoritative stubs, public wrapper docstrings, exports,
  examples, and tests synchronized for every Python-visible change.
- Use NumPy-style docstrings. Document scientific array shape, dtype, axis order,
  copying, mutability, units, optional values, failures, and blocking/GIL
  behavior where relevant.
- Do not expose private `_core` implementation details as public API.
- Keep generated Python API domains split by purpose, source implementations
  collapsed, and stable public symbol links discoverable.
- Put publication references in the authoritative stub and keep corresponding
  wrapper and PyO3 runtime prose semantically synchronized. Follow
  `docs/development/index.md#scientific-citations`.

## Validation

- Run focused tests under `python/tests/` while iterating.
- Run `pixi run python-api-docs` for public API, signature, stub, or docstring
  changes.
- Run `pixi run -e py312 python-test` for broader Python behavior changes; use
  `python-test-full` or `pixi run ci` when the change warrants complete coverage.
