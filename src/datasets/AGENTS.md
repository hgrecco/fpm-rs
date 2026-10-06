# Dataset instructions

These instructions apply to dataset registry, manifest, cache, and loading code.

- Treat `dataset_spec.md` as the dataset contract and `docs/datasets.md` as the
  public loader and provenance guide; update them with behavior changes.
- Keep local `DatasetLoader` operations offline. Registry discovery, verified
  downloads, and opening a registered ID may access the network only through
  explicit registry operations.
- Keep source-specific acquisition and conversion external to this library.
- Preserve verified downloads, managed caching, generic loading, deterministic
  subsets, and actionable validation errors.
- Dataset tests must use local or generated fixtures and must not require the
  network.
- If an expected value, convention, or dataset comes from a publication, follow
  `docs/development/index.md#scientific-citations` and explain any derivation or
  transformation near the fixture or test.
