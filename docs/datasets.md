# Datasets

`fpm-rs` uses one bundle format and one loader regardless of where data came
from:

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

```rust,no_run
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

## Discover and open registered datasets

The committed `dataset_registry.json` is an initially empty strict version-1
registry. By default installed clients read:

```text
https://raw.githubusercontent.com/hgrecco/fpm-rs/main/dataset_registry.json
```

In Rust, use `DatasetRegistry` when an identifier should be downloaded on
demand:

```rust,no_run
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
