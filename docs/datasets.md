# Datasets

`fpm-rs` uses explicit ordinary (version 1) and spectral (version 2) profiles
with a local loader, regardless of where data came from:

```text
dataset_spec + dataset files -> DatasetLoader -> Dataset -> reconstruction
```

The complete language-neutral contract is
[`dataset_spec.md`](https://github.com/hgrecco/fpm-rs/blob/main/dataset_spec.md).
Source-specific conversion remains outside this repository. This repository's
registry distributes already-converted bundles that use the same local loader.

## Open a local bundle

The standard entry point is `dataset.json` at the bundle root:

```text
dataset-root/
  dataset.json
  measurements.json
  configuration.json
  frames/
  corrections/             # optional
  ground-truth.json         # optional
  valid-mask.json           # optional
```

Rust loading is explicit and offline:

```rust
use fpm_rs::{Result, datasets::DatasetLoader};

fn main() -> Result<()> {
    let dataset = DatasetLoader::new("/data/converted-fpm")?.load()?;
    let problem = dataset.reconstruction_problem()?;
    println!("{} frames", problem.measurements.frame_count());
    Ok(())
}
```

`DatasetLoader` validates safe paths, measurements, configuration and compiled
models, optional truth and masks, provenance, and units. It never accesses the
network.

## Select frames and crop an ordinary dataset

Rust's `Dataset::subset` builder accepts `FrameSelector::All`,
`EveryNth(step)`, or ordered unique `Indices`, plus an optional detector `Rect`.
Python resolves the same selection in one call:

```python
subset = dataset.subset(frames=[2, 0], crop=(16, 32, 64, 64))
problem = subset.reconstruction_problem()
print(subset.spatial_crop, subset.measurements.shape)
# Alternatively select 0, 2, 4, ... and retain the full image:
frame_subset = dataset.subset(every_nth_frame=2)
```

Indices refer to the loaded acquisition, with the explicit supplied order
preserved. `frames` and `every_nth_frame` are mutually exclusive. Crop order is
`(row, column, height, width)` in original detector pixels, with positive sizes
and containment checked before construction. Its object-space mapping must
land on integer reconstruction pixels. Cropping owns new measurements and
correction arrays, recompiles selected models, and crops optional truth/masks;
it does not resample data. Python releases the GIL during subset construction.

`DatasetSubset::spatial_crop` / `subset.spatial_crop` returns the resolved
rectangle, including the full image when no spatial crop was requested. Frame
metadata preserves original acquisition indices and optional illumination
identifiers separately from reindexed model sources. Dataset provenance and
measurement units are retained. Use the
[subset-aware benchmark workflow](benchmarks.md) to export these selections to
JSON, CSV, and normalized Parquet tables. This is an in-memory selection API;
the on-disk dataset contract is unchanged.

## Open an explicit spectral bundle

Version 2 stores stable channel IDs, positive distinct vacuum wavelengths,
response/weight provenance, per-channel compiled kernels on a common grid, and
an explicit sparse detector exposure plan. Frames remain grayscale. External
converters produce this metadata; RGB image planes are not inferred channels.
The [dataset specification](https://github.com/hgrecco/fpm-rs/blob/main/dataset_spec.md#spectral-dataset-manifest-version-2)
defines the strict profile. Channel kernels can be emitted by
`ImagePlaneModel.save_json(path=...)` from Python.

```python
spectral = fpm.load_spectral_dataset(path="/data/converted-spectral-fpm")
problem = spectral.reconstruction_problem()
print(spectral.model.channel_ids, spectral.response_provenance)
result = fpm.SpectralAlternatingProjection(iterations=50).run(problem=problem)
```

In Rust use `DatasetLoader::new(path)?.load_spectral()?`, then
`reconstruction_problem()`. Local loading stays offline and rejects missing
response provenance, wavelength/kernel mismatches, incompatible common grids,
invalid sparse references, unsafe paths (including escaping symlinks), and
non-grayscale frames. Optional truth/masks remain channel ordered. Ordinary
`load` and spectral `load_spectral` reject the other profile explicitly.

For a registered spectral ID, use `registry.open_spectral(id)` in either
language, or `fpm-datasets open ID --spectral`. Registry profile versions 1 and
2 share verified downloads, staging, cache repair and cleanup; the registry
schema itself remains version 1.

## Discover and open registered datasets

The committed `dataset_registry.json` is an initially empty strict version-1
registry. By default installed clients read:

```text
https://raw.githubusercontent.com/hgrecco/fpm-rs/main/dataset_registry.json
```

In Rust, use `DatasetRegistry` when an identifier should be downloaded on
demand:

```rust
use fpm_rs::{Result, datasets::DatasetRegistry};

fn main() -> Result<()> {
    let registry = DatasetRegistry::from_defaults()?;
    for item in registry.list()? {
        println!("{} {} cached={}", item.entry.id, item.entry.version, item.cached);
    }
    let dataset = registry.open("example-led-array-dataset")?;
    dataset.reconstruction_problem()?;
    Ok(())
}
```

`open` reuses a valid current cache entry. Otherwise it downloads the immutable
tar.zst archive, enforces its declared byte size, verifies SHA-256, extracts it
into staging, validates it with `DatasetLoader`, and atomically installs it.
The downloaded archive is discarded after installation.

Python exposes the same operations:

```python
import fpm_rs as fpm

registry = fpm.DatasetRegistry()
for item in registry.list():
    print(item.id, item.version, item.cached)

dataset = registry.open("example-led-array-dataset")
problem = dataset.reconstruction_problem()
```

For a one-off default open, use `datasets::open_dataset(id)` in Rust or
`fpm_rs.open_dataset(id)` in Python. Registry-backed operations may access the
network; Python releases the GIL while they run.

## Configuration and cache

Explicit constructor or CLI values take precedence over environment variables,
which take precedence over defaults.

| Setting | Environment variable | Default |
| --- | --- | --- |
| Registry | `FPM_RS_DATASET_REGISTRY_URL` | Repository raw `dataset_registry.json` |
| Cache | `FPM_RS_DATASET_CACHE_DIR` | Platform cache directory under `fpm-rs/datasets` |

Registry sources may be HTTP(S), `file://` URLs, or filesystem paths. Successful
registry responses are cached by source URL. Clients try the configured source
first and use its matching snapshot only when retrieval fails.

The managed layout is:

```text
<cache>/
  .fpm-rs-dataset-cache
  registries/<source-hash>.json
  datasets/<id>/<version>/
    dataset.json
    .fpm-rs-install.json
    ...
  partial/                       # transient only
```

Cache cleanup refuses unmarked roots and symbolic links. `clean(id)` removes
all cached versions of one identifier; `clean_all()` removes the complete
managed cache. A corrupt installed bundle is removed and downloaded once more
when opened.

## Command line

The Cargo package and Python wheel both install `fpm-datasets`:

```sh
fpm-datasets list
fpm-datasets download example-led-array-dataset
fpm-datasets download --all
fpm-datasets open example-led-array-dataset
fpm-datasets clean example-led-array-dataset
fpm-datasets clean --all
```

Every command accepts `--registry-url URL` and `--cache-dir PATH` before the
subcommand. `list` reports ID, version, cache status, archive size, and title.
`open` ensures the dataset is present, validates it, and prints its path and
shapes.

## Producing bundles

Conversion pipelines emit the files and metadata required by
`dataset_spec.md`. Registry archives must contain `dataset.json` at archive
root and must have immutable URLs, exact compressed sizes, and SHA-256 values.
The registry carries discovery, license, citation, and source metadata; optical
and frame-level metadata remain authoritative inside the bundle.
