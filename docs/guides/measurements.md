# Load and prepare an acquisition

The Python boundary accepts an in-memory NumPy stack. Load TIFF or another
instrument format with the tool appropriate to that format, convert detector
intensities to `float64`, and keep frame order aligned with the illumination
description.

```python
import numpy as np
import fpm_rs as fpm

frames = np.ascontiguousarray(raw_frames, dtype=np.float64)  # (frames, height, width)
measurements = fpm.MeasurementStack(frames)
problem = fpm.ReconstructionProblem(measurements, model)
```

Input arrays must be C-contiguous; transposed, reversed, and stepped views are
rejected without an implicit repair copy. Call `np.ascontiguousarray` explicitly
when that copy is appropriate. Accepted arrays are copied once into Rust-owned
storage. Every frame must have the same `(height, width)` as
`model.image_shape`. Values represent intensity, not amplitude. Masks follow
the same contiguity rule. Use `frame_weights` to down-weight or disable complete
frames and `masks` to exclude detector pixels.

The Rust API additionally provides resident and lazy stacks, image-stack
loading, measurement manifests, metadata, dark/background correction, masks,
and explicit preprocessing. See [Datasets](../datasets.md) for the native
bundle workflow and the [Rust API](../reference/rust.md) for `measurements`
types. Source-specific conversion is performed outside this repository; the
generic loader only consumes converted bundles and never accesses the network.

## Rust image loading and lazy stacks

`MeasurementStack::from_image_files` loads an ordered list of single-channel
8-bit or 16-bit PNG/TIFF frames while preserving native detector counts. Each
path becomes one frame and receives generated path metadata. Use
`from_tiff_stack` when each page in a multipage TIFF is a measurement frame.

`LazyMeasurementStack::from_image_files` validates headers up front and keeps a
bounded, thread-safe LRU of decoded frames (one by default). Its
`from_tiff_stack` treats TIFF pages as lazy frames, and `from_manifest` decodes
and preprocesses manifest frames on demand. A lazy stack can be passed directly
to `ReconstructionProblem::new` or converted to processed resident storage with
`materialize`. `with_cache_capacity` limits retained frame count;
`with_cache_byte_capacity` limits decoded-pixel bytes. `cached_frame_count` and
`cached_byte_count` report the retained cache. Correction images, masks, and
metadata stay resident; byte accounting excludes frame handles held by callers
after eviction.

## Measurement manifests

`MeasurementSpec` is the strict JSON manifest type. Ordered `FrameSpec` entries
provide paths, illumination indices, exposures, weights, and labels. Optional
dark, flat, background, and mask images use `ImageSet` where applicable, while
`PreprocessingConfig` holds preprocessing flags.

`MeasurementStack::from_manifest` resolves paths relative to the manifest.
Backgrounds and masks can be one broadcast path or one path per frame. The
configuration describes preprocessing without applying it; call
`apply_preprocessing()` explicitly to transform detector counts. The corresponding
lazy constructor applies configured dark, background, flat-field, exposure, and
negative-clamping operations as a frame is decoded.

For a minimal native workflow, run:

```sh
cargo run --example load_measurement_manifest -- measurements.json
```
