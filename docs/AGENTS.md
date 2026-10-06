# Documentation instructions

These instructions apply to authored MkDocs pages and curated tutorial
notebooks. Generated `site/` and `target/` content is never a source file.

## Content

- Improve an existing page before creating a new one. Keep `README.md` and
  rustdoc as public API entry points that link to focused explanations rather
  than duplicating them.
- Keep `README.md`, `docs/index.md`, `docs/getting-started/quickstart.md`, and
  `docs/tutorials/notebooks/quickstart.ipynb` on the same calling convention.
  Use keyword arguments for multi-parameter public calls.
- Keep the quickstart to the shortest path to one successful reconstruction.
  Put sizing, batching, regularization, calibration, and other secondary topics
  in the relevant guide and cross-link them.
- Give every reconstruction algorithm an entry in
  `docs/guides/reconstruction.md`, including when to prefer it over
  alternatives.
- Explain the primitives behind every end-user one-call convenience or demo in
  the same document; present it as a sanity check and modifiable example.
- Define new non-specialist domain terms in `docs/concepts/glossary.md`.
- Follow `docs/development/index.md#scientific-citations` for scientific claims
  and references.
- Read only the sections relevant to the task instead of preloading every linked
  guide.

## Validation

- Run the narrowest relevant check while iterating.
- Run `pixi run citation-check` when citations change and render the affected
  documentation surface.
- Run `pixi run docs-build` for authored-site, navigation, notebook, or
  cross-surface documentation changes. Use `pixi run rust-doc`,
  `pixi run rust-doc-test`, or `pixi run python-api-docs` separately when only
  that public API surface changed.
