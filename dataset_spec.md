# Local Fourier-Ptychography Dataset Format, Version 1

This is a language-neutral specification for a reader or writer. It defines
the files, JSON values, image encodings, coordinate conventions, and the
portable direct-vector configuration profile. An implementation can be written
in Python, C++, JavaScript, Julia, or another language; it does not require a
particular library or runtime.

A conforming writer emits UTF-8 JSON, lossless grayscale images, and only
finite JSON numbers. A conforming reader treats all JSON objects described as
strict: writers must not add fields not listed by this specification, and
readers may reject them. JSON uses ordinary strings, booleans, arrays, objects,
and numbers. There are no comments, `NaN`, or `Infinity` values.

## Terms and conventions

- A **shape** is `[height, width]`, never `[width, height]`.
- Flattened array data is row-major: element `(row, column)` has index
  `row * width + column`.
- All physical lengths are metres; angles are radians; transverse wave vectors
  are radians per metre.
- All real-valued calculations use IEEE-754 binary64 (`float64`) semantics.
- `tau` means `2 * pi`.
- An **image frame** is one measured intensity image. Frame order is semantic.

## Bundle layout and paths

The standard entry point is `dataset.json` in the dataset root:

```text
<dataset-root>/
  dataset.json
  measurements.json
  configuration.json
  frames/
    frame-0000.tiff
    frame-0001.tiff
  corrections/                 # optional
    dark.tiff
    flat.tiff
    background.tiff
    mask.tiff
  ground-truth.json            # optional
  valid-mask.json              # optional; requires ground truth
```

The names other than `dataset.json` are conventions. References in
`dataset.json` are resolved relative to the directory containing
`dataset.json`; references in `measurements.json` are resolved relative to the
directory containing `measurements.json`.

Every manifest path must be a non-empty, safe relative path. It must not be
absolute and must not contain `.` or `..` components. For example,
`frames/frame-0000.tiff` is valid, while `/data/frame.tiff`,
`../frame.tiff`, and `./frame.tiff` are invalid. A portable bundle should not
depend on symbolic links.

## Dataset manifest: `dataset.json`

```json
{
  "format_version": 1,
  "measurement_manifest": "measurements.json",
  "configuration": "configuration.json",
  "ground_truth_object": "ground-truth.json",
  "valid_object_mask": "valid-mask.json",
  "provenance": {
    "source": "laboratory acquisition",
    "license": "CC-BY-4.0"
  },
  "measurement_units": "camera counts"
}
```

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `format_version` | unsigned integer | yes | Must be `1`. |
| `measurement_manifest` | string | yes | Safe relative path to the measurement manifest. |
| `configuration` | string | yes | Safe relative path to the optical configuration. |
| `ground_truth_object` | string or `null` | no | Safe relative path to a complex array. Omit or use `null` when unavailable. |
| `valid_object_mask` | string or `null` | no | Safe relative path to a binary array. It requires `ground_truth_object`. |
| `provenance` | object mapping strings to strings | no | Defaults to `{}`. No key or value may be empty or whitespace-only. |
| `measurement_units` | string or `null` | no | A non-empty description such as `"camera counts"`. |

## Measurement manifest: `measurements.json`

```json
{
  "frames": [
    {
      "path": "frames/frame-0000.tiff",
      "illumination_index": 0,
      "exposure_time": 0.02,
      "weight": 1.0,
      "label": "LED row 0 column 0"
    },
    {
      "path": "frames/frame-0001.tiff",
      "illumination_index": 1,
      "exposure_time": 0.02,
      "weight": 1.0,
      "label": "LED row 0 column 1"
    }
  ],
  "dark_frame": "corrections/dark.tiff",
  "flat_field": "corrections/flat.tiff",
  "background": "corrections/background.tiff",
  "mask": "corrections/mask.tiff",
  "preprocessing": {
    "subtract_dark": true,
    "divide_flat_field": true,
    "normalize_exposure": true,
    "subtract_background": false,
    "clamp_negative": true
  }
}
```

### Frame records

`frames` is a non-empty ordered array. The frame at array index `i` is paired
with configuration-model frame `i`. `illumination_index` is descriptive
metadata only; it does not reorder frames.

| Field | Type | Default when absent | Rules |
| --- | --- | --- | --- |
| `path` | string | none | Required safe relative path to one image. |
| `illumination_index` | unsigned integer or `null` | `null` | Optional acquisition identifier. |
| `exposure_time` | number | `1.0` | Finite and greater than zero. |
| `weight` | number | `1.0` | Finite and non-negative. At least one frame in the bundle must have positive weight. |
| `label` | string or `null` | `null` | Optional descriptive label. |

The configuration's model frame count must equal `frames.length`, and every
frame must have the configuration's `image_shape`.

### Image encoding

Each frame and correction image must be a single-channel, unsigned 8-bit or
16-bit grayscale PNG or TIFF. Each manifest entry names one single-page image.
Do not use RGB, RGBA, palette, signed, floating-point, or 32-bit images.

Pixels are read as native detector counts and converted to binary64 without
normalization: `0..255` for 8-bit data and `0..65535` for 16-bit data. All
frame images must have identical nonzero dimensions. The first frame determines
the measurement image shape. Every correction image must have that same shape.

`dark_frame` and `flat_field` each name one shared correction image. Every
flat-field pixel must be strictly positive. `background` and `mask` are either
one shared image path or an array of exactly one path per frame:

```json
"background": "corrections/shared-background.tiff"
```

```json
"background": [
  "corrections/background-0000.tiff",
  "corrections/background-0001.tiff"
]
```

The same rule applies to `mask`. A mask is binarized on read: zero excludes a
pixel and every nonzero value includes it. Every positive-weight frame must
retain at least one included pixel.

### Preprocessing

Omit `preprocessing` to disable all processing. When present, it must contain
all five boolean fields shown in the manifest example. Enabling a correction
requires its matching correction image.

For each pixel in each frame, apply enabled operations in this exact order:

1. subtract the shared dark frame;
2. subtract the shared or per-frame background;
3. divide by the shared flat field;
4. divide by that frame's `exposure_time`;
5. clamp negative results to zero.

For uncorrected detector-count data, omit all correction fields and omit
`preprocessing` or set all five flags to `false`.

## Common array values

Arrays use an object with explicit dimensions and a flat row-major `data`
array. This is used for ground truth and for complex pupil samples:

```json
{
  "height": 2,
  "width": 3,
  "data": [1, 2, 3, 4, 5, 6]
}
```

`height` and `width` are positive unsigned integers, and `data.length` must
equal `height * width`. A complex number is an object with finite `re` and
`im` numbers:

```json
{"re": 1.0, "im": -0.25}
```

## Optical configuration: `configuration.json`

The configuration contains both human-supplied experiment descriptions and a
fully compiled numerical forward model. The compiled model is intentionally
stored in the file so a reader in any language can reconstruct without
reimplementing LED geometry. A writer must populate it; it cannot be omitted.

This specification defines the **direct-vector profile**. It is sufficient for
non-multiplexed image-plane data with one calibrated transverse wave vector per
measured frame. A writer should use this profile unless it has an independent
implementation of another illumination geometry.

### Top-level configuration

Every top-level field is required, including fields whose value is `null`:

```json
{
  "format_version": 1,
  "true_experiment": {},
  "reconstruction_experiment": {},
  "image_shape": [256, 256],
  "reconstruction_shape": [512, 512],
  "compiled_models": {
    "true_model": {},
    "reconstruction_model": {}
  },
  "camera": null,
  "illumination_acquisition_errors": null,
  "random_seed": 0
}
```

| Field | Type and rules |
| --- | --- |
| `format_version` | Unsigned integer, exactly `1`. |
| `true_experiment` | An experiment object. For measured data, this often equals `reconstruction_experiment`. |
| `reconstruction_experiment` | An experiment object used to derive `reconstruction_model`. |
| `image_shape` | `[height, width]`; positive and equal to every measurement image shape. |
| `reconstruction_shape` | `[height, width]`; each dimension is at least the image dimension. The height and width scale factors must be equal. |
| `compiled_models` | Object containing `true_model` and `reconstruction_model`, each a compiled-model object. |
| `camera` | `null` for a measured dataset, or a camera object described below. |
| `illumination_acquisition_errors` | `null` for a measured dataset, or an acquisition-errors object described below. |
| `random_seed` | Unsigned 64-bit integer. It is provenance for stochastic simulation, not image data. |

### Experiment object

```json
{
  "optics": {
    "wavelength": 5.32e-7,
    "objective_na": 0.1,
    "magnification": 4.0,
    "camera_pixel_size": 6.5e-6,
    "medium_index": 1.0,
    "defocus_distance": null,
    "pupil_aberration": null
  },
  "illumination": {
    "KVectors": [
      {"kx": 0.0, "ky": 0.0}
    ]
  },
  "optical_background": null
}
```

All three fields are required. The `KVectors` object is a tagged union whose
single key identifies the illumination form. The direct-vector profile uses
only `KVectors`, whose value is a non-empty ordered array of `{ "kx": number,
"ky": number }` objects.

The vector magnitude must not exceed `tau * medium_index / wavelength`. The
vector order is the model source order and, for this profile, the measurement
frame order.

#### Optics object

| Field | Type and rules |
| --- | --- |
| `wavelength` | Finite positive number in metres. |
| `objective_na` | Finite positive number no greater than `medium_index`. |
| `magnification` | Finite positive number. |
| `camera_pixel_size` | Finite positive number in metres. |
| `medium_index` | Finite positive refractive index. |
| `defocus_distance` | `null` or finite axial displacement in metres. |
| `pupil_aberration` | `null` or the object below. |

`pupil_aberration`, when present, has four required finite fields:

```json
{
  "astigmatism": 0.0,
  "coma": 0.0,
  "spherical": 0.0,
  "edge_apodization": 0.0
}
```

`edge_apodization` must be non-negative. The three phase coefficients are
radians; they are direct radial-polynomial weights, not normalized Zernike
coefficients.

`optical_background` is `null` or a flat array of finite non-negative numbers.
When non-null, it must contain either one image (`height * width` values) or
one image per model frame (`frame_count * height * width` values).

### Compiled-model object

Each of `compiled_models.true_model` and
`compiled_models.reconstruction_model` has exactly these fields:

```json
{
  "k_vectors": [
    {"kx": 0.0, "ky": 0.0}
  ],
  "pupil": {
    "values": {
      "height": 2,
      "width": 2,
      "data": [
        {"re": 0.0, "im": 0.0},
        {"re": 0.0, "im": 0.0},
        {"re": 0.0, "im": 0.0},
        {"re": 1.0, "im": 0.0}
      ]
    },
    "support": [false, false, false, true]
  },
  "crop_indices": {
    "crops": [
      {"start_row": 1, "start_col": 1, "height": 2, "width": 2}
    ]
  },
  "subpixel_offsets": [
    {"row": 0.0, "column": 0.0}
  ],
  "sampling": {
    "low_res_pixel_size": 1.625e-6,
    "high_res_pixel_size": 8.125e-7,
    "dkx": 1933287.786824488,
    "dky": 1933287.786824488,
    "wavelength": 5.32e-7,
    "synthetic_na": 0.1,
    "coordinate_convention": "CenteredPositiveK"
  },
  "image_shape": [2, 2],
  "reconstruction_shape": [4, 4],
  "frame_gains": null,
  "background": null,
  "multiplexing_matrix": null
}
```

The JSON above illustrates field shapes only. A writer must calculate all
derived numerical values from its experiment; it must not reuse the example
values.

| Field | Rules |
| --- | --- |
| `k_vectors` | Exact ordered vectors derived from the experiment. For `KVectors`, copy the experiment values. |
| `pupil.values` | Complex array with shape `image_shape`, calculated below. |
| `pupil.support` | Boolean array of length `height * width`, calculated below. |
| `crop_indices.crops` | One crop per source, in source order. Each crop has unsigned `start_row`, `start_col`, `height`, and `width`. |
| `subpixel_offsets` | For the direct-vector profile, an array of one `{row, column}` object per source. It must not be `null`. |
| `sampling` | Numerical sampling record, calculated below. `coordinate_convention` is exactly `"CenteredPositiveK"`. |
| `image_shape` | Exact copy of top-level `image_shape`. |
| `reconstruction_shape` | Exact copy of top-level `reconstruction_shape`. |
| `frame_gains` | `null` in the basic direct-vector profile. |
| `background` | Exact copy of the experiment's `optical_background`. |
| `multiplexing_matrix` | `null` in the basic direct-vector profile. |

For a direct-vector experiment, the compiled model has one source and one
frame per vector. Therefore `k_vectors.length`, `crops.length`,
`subpixel_offsets.length`, and `frames.length` must all be equal.

### Direct-vector model compilation

The formulas in this section produce both compiled models. Compile the true
model from `true_experiment` and the reconstruction model from
`reconstruction_experiment` independently. If the two experiment objects are
identical, their compiled models must be identical.

Let measurement shape be `[h, w]`, reconstruction shape be `[hr, wr]`, and
optics be the selected experiment's optics. Require `hr >= h`, `wr >= w`, and
`abs(hr / h - wr / w) <= 1e-9 * max(hr / h, wr / w)`.

1. Compute basic scales:

   ```text
   low_res_pixel_size  = camera_pixel_size / magnification
   scale               = hr / h
   high_res_pixel_size = low_res_pixel_size / scale
   dkx                 = tau / (w * low_res_pixel_size)
   dky                 = tau / (h * low_res_pixel_size)
   medium_wavenumber   = tau * medium_index / wavelength
   synthetic_na        = objective_na + max(norm(k) * wavelength / tau)
   ```

   `max(...)` ranges over every source vector. Store these values in
   `sampling`, with `wavelength` equal to the optics wavelength and
   `coordinate_convention` equal to `"CenteredPositiveK"`.

2. Build the pupil with shape `[h, w]`. For every zero-based `(row, column)`,
   calculate:

   ```text
   ky       = (row    - floor(h / 2)) * dky
   kx       = (column - floor(w / 2)) * dkx
   radius   = sqrt(kx*kx + ky*ky)
   cutoff   = tau * objective_na / wavelength
   support  = radius <= cutoff
   ```

   If `support` is false, write `false` and complex `{ "re": 0.0,
   "im": 0.0 }`. If it is true, start with `amplitude = 1.0` and `phase = 0.0`.
   If `defocus_distance` is non-null, add:

   ```text
   phase -= defocus_distance * (kx*kx + ky*ky) / (2 * medium_wavenumber)
   ```

   If `pupil_aberration` is non-null, set `rho = radius / cutoff` and
   `theta = atan2(ky, kx)`, then add:

   ```text
   phase += astigmatism * rho*rho * cos(2*theta)
   phase += coma * (3*rho*rho*rho - 2*rho) * cos(theta)
   phase += spherical * (6*rho^4 - 6*rho*rho + 1)
   amplitude = exp(-edge_apodization * rho*rho)
   ```

   Store the complex pupil value as:

   ```text
   re = amplitude * cos(phase)
   im = amplitude * sin(phase)
   ```

3. Build one crop and one offset for each source vector `{kx, ky}`. Let
   `round_away(x)` round to the nearest integer with exact half-way cases away
   from zero. Compute:

   ```text
   continuous_row    = ky / dky
   continuous_column = kx / dkx
   shift_row         = round_away(continuous_row)
   shift_column      = round_away(continuous_column)
   offset_row        = continuous_row - shift_row
   offset_column     = continuous_column - shift_column

   start_row = floor(hr / 2) - floor(h / 2) + shift_row
   start_col = floor(wr / 2) - floor(w / 2) + shift_column
   ```

   Write a crop with `height = h`, `width = w`, and the calculated starts.
   Write `{ "row": offset_row, "column": offset_column }`. The crop and its
   bilinear interpolation stencil must lie inside `[hr, wr]`:

   - If an offset component is exactly integral within `1e-12`, use that
     integral shift with no interpolation.
   - Otherwise, use `floor(offset)` as the lower shift and interpolate between
     it and the next larger index. Both indices must be inside the grid.

4. Set `frame_gains` and `multiplexing_matrix` to `null`; set `background` to
   the experiment's `optical_background`; and copy both shapes exactly.

The generated compiled model is consistent with the `CenteredPositiveK`
convention: positive `kx` moves a crop toward increasing column indices, and
positive `ky` moves it toward increasing row indices.

### Optional simulation metadata

The direct-vector profile normally writes both fields as `null`. A reader that
preserves richer simulation metadata should support these exact object forms.

`camera` is `null` or:

```json
{
  "photons_per_pixel": 1000.0,
  "gain_counts_per_electron": 1.0,
  "offset_counts": 0.0,
  "read_noise_electrons": 0.0,
  "dark_current_electrons": 0.0,
  "shot_noise": false,
  "pixel_sensitivity": null,
  "bit_depth": 16,
  "saturation_counts": null,
  "quantize": true,
  "bad_pixels": [],
  "bad_pixel_value_counts": null
}
```

`photons_per_pixel` and `gain_counts_per_electron` are finite and positive.
`offset_counts` is finite; `read_noise_electrons` and
`dark_current_electrons` are finite and non-negative. `pixel_sensitivity` is
`null` or one finite non-negative value per image pixel. `bit_depth` is `null`
or an integer from 1 through 32. `saturation_counts` is `null` or finite and
positive. `bad_pixels` contains unique flattened pixel indices below
`height * width`; `bad_pixel_value_counts` is `null` or finite and
non-negative. If `bad_pixels` is non-empty and `bad_pixel_value_counts` is
`null`, at least one of `bit_depth` or `saturation_counts` must be non-null so
that the replacement value has a finite upper bound.

`illumination_acquisition_errors` is `null` or:

```json
{
  "frame_gain_relative_std": 0.0,
  "missing_frames": [],
  "source_permutation": null
}
```

`frame_gain_relative_std` is finite and non-negative. `missing_frames` contains
unique frame indices below the model frame count. `source_permutation` is
`null` or a permutation of `0..source_count-1`.

## Optional ground truth and valid-object mask

`ground-truth.json` is a complex array with shape exactly equal to
`reconstruction_shape`:

```json
{
  "height": 2,
  "width": 3,
  "data": [
    {"re": 1.0, "im": 0.0},
    {"re": 0.9, "im": 0.1},
    {"re": 1.1, "im": -0.1},
    {"re": 1.0, "im": 0.2},
    {"re": 0.8, "im": 0.0},
    {"re": 1.2, "im": -0.2}
  ]
}
```

Every component must be finite. `valid-mask.json`, if present, is an unsigned
integer array of the same shape:

```json
{
  "height": 2,
  "width": 3,
  "data": [1, 1, 0, 1, 1, 0]
}
```

Mask values must be exactly `0` or `1`, and the mask must contain at least one
`1`. A valid-object mask is invalid without ground truth.

## Independent writer algorithm

An implementation in any language can generate a valid direct-vector bundle as
follows:

1. Convert each source measurement to a lossless L8 or L16 grayscale PNG/TIFF
   without accidental `[0, 1]` normalization. Ensure a common nonzero shape.
2. Emit `measurements.json` in acquisition/model order, along with optional
   correction images and preprocessing fields.
3. Choose calibrated optical values and one `{kx, ky}` vector per frame.
4. Create identical true and reconstruction experiment objects unless a known
   model mismatch is intentional.
5. Compile both models using the formulas above and emit every field of
   `configuration.json`.
6. Emit optional ground truth and mask arrays, then `dataset.json` with only
   safe relative paths.
7. Independently validate: frame count and shape agreement; finite numerical
   fields; crop/stencil containment; positive-weight frames with unmasked
   pixels; and array lengths.

## Common errors

- Writing `[width, height]` instead of `[height, width]`.
- Writing nested JSON arrays for numerical arrays instead of `{height, width,
  data}`.
- Writing RGB or floating-point TIFF/PNG images.
- Normalizing detector counts unintentionally.
- Omitting `compiled_models` or copying derived values from a different optics
  configuration.
- Reordering frames based on `illumination_index`.
- Using a different crop sign convention from `CenteredPositiveK`.
- Using absolute paths, `./`, or `../` in either manifest.

## Dataset registry

A dataset registry is a separate UTF-8 JSON document used to discover and
distribute versioned archives of bundles conforming to this specification. It
is not part of a bundle, and it does not replace any metadata in `dataset.json`,
`measurements.json`, or `configuration.json`.

Registry JSON objects are strict: readers reject unknown fields. Version 1 has
this shape:

```json
{
  "registry_version": 1,
  "datasets": [
    {
      "id": "example-led-array-dataset",
      "version": "1.0.0",
      "title": "Example LED-array Fourier-ptychography dataset",
      "description": "Experimental non-multiplexed image-plane acquisition.",
      "format_version": 1,
      "archive": {
        "url": "https://example.org/example-led-array-dataset-1.0.0.tar.zst",
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "size_bytes": 123456789
      },
      "license": {
        "spdx": "CC-BY-4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/"
      },
      "citation": {
        "doi": "10.xxxx/example",
        "text": "Author et al., Title, Journal, Year"
      },
      "source": {
        "url": "https://example.org/original-dataset",
        "description": "Original laboratory dataset"
      },
      "tags": ["experimental", "led-array", "classical-fpm"]
    }
  ]
}
```

| Field | Rules |
| --- | --- |
| `registry_version` | Required unsigned integer, exactly `1`. |
| `datasets` | Required array. Dataset IDs must be unique within the registry. |
| `id` | Required non-empty stable identifier containing only ASCII letters, digits, `.`, `_`, or `-`. It must not be `.` or `..`. |
| `version` | Required non-empty immutable version using the same character set as `id`. One registry contains at most one current version of an ID. |
| `title`, `description` | Required non-empty human-readable strings. |
| `format_version` | Required unsigned integer identifying the bundle format. It must be `1` for this specification. |
| `archive.url` | Required non-empty URL identifying immutable `.tar.zst` content. |
| `archive.sha256` | Required SHA-256 of the archive exactly as downloaded, encoded as 64 hexadecimal characters. |
| `archive.size_bytes` | Required positive compressed archive size. |
| `license.spdx`, `license.url` | Required non-empty license identifier and reference URL. |
| `citation.doi`, `citation.text` | Required non-empty preferred citation fields. |
| `source.url`, `source.description` | Required non-empty original-source provenance. |
| `tags` | Required non-empty array of unique, non-empty strings describing intended use or acquisition type. |

The archive contains the bundle contents directly: `dataset.json` is at the
archive root, not beneath an additional wrapper directory. Every archive entry
must be a safe relative path. Symbolic links, hard links, devices, absolute
paths, `.`, `..`, and duplicate destinations are invalid.

The archive URL should be immutable, such as a versioned release, fixed
repository snapshot, or permanent research-data deposit. A downloader verifies
`size_bytes` and `sha256` before extraction, validates the extracted bundle
through the ordinary local loader, and only then makes it available in a cache.
A downloaded dataset has no source-specific runtime path.

A bundle may record its registry identity as ordinary provenance:

```json
{
  "provenance": {
    "dataset_id": "example-led-array-dataset",
    "dataset_version": "1.0.0",
    "source_url": "https://example.org/original-dataset",
    "license": "CC-BY-4.0",
    "citation_doi": "10.xxxx/example"
  }
}
```

These fields provide traceability but do not affect loading or numerical
interpretation. The registry, source-specific preparation tools, and immutable
archives may be maintained independently.
