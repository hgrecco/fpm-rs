# Intensity metrics

`metrics::intensity` contains pure, domain-agnostic calculations for comparing
two scalar two-dimensional images. It does not contain losses used to optimize
a reconstruction: those remain in `algorithms::objective`.

Every two-image metric calls its inputs `reference` and `candidate`. A signed
residual is always `candidate - reference`. An optional `valid_mask` has the
same shape as both images; `true` includes a pixel and `false` excludes it.
All metrics reject mismatched shapes, non-finite included values, and an empty
valid mask.

## Rust API

All atomic functions accept `ndarray::ArrayView2<'_, T>` with
`T: num_traits::ToPrimitive`. They convert individual included samples to
`f64` for evaluation and return `f64`; the image data is borrowed, never
copied. This supports ordinary `u8`, `u16`, `u32`, `f32`, and `f64` images.

```rust
use fpm_rs::metrics::intensity::{nrmse, psnr};

let relative_error = nrmse(reference.view(), candidate.view(), None)?;
let quality_db = psnr(reference.view(), candidate.view(), None, 65_535.0)?;
```

The atomic API is intentionally not an optimization primitive. Reconstruction
objectives operate on the algorithm's floating scalar and have gradient and
stability contracts that differ from reporting metrics.

## Metric definitions

| Function | Definition or convention |
| --- | --- |
| `bias` | Mean signed residual. |
| `mae`, `mse`, `rmse` | Mean absolute, squared, and root mean squared residual. |
| `relative_l1` | `sum(abs(candidate - reference)) / sum(abs(reference))`. |
| `nrmse` | `L2(candidate - reference) / L2(reference)`. |
| `amplitude_nrmse` | NRMSE after applying `sqrt` to both intensities; requires non-negative values. |
| `correlation` | Pearson correlation; undefined for a constant valid image. |
| `psnr` | Peak signal-to-noise ratio in dB. `data_range` is required, finite, and positive; identical inputs produce `+∞`. |
| `ssim` | Single-scale SSIM, higher is more similar. Uses an 11×11 Gaussian window with σ=1.5, `K1=0.01`, and `K2=0.03`. |
| `poisson_deviance` | Summed Poisson deviance for non-negative intensities. A positive `epsilon` floors candidate intensity. |
| `mean_poisson_deviance` | Poisson deviance divided by valid-pixel count. |
| `fitted_gain` | Least-squares gain in `candidate ≈ gain × reference`. |

`relative_l1`, `nrmse`, and `fitted_gain` are undefined for a zero reference
normalization and return an error rather than silently choosing a scale.

`data_range` is explicit for PSNR and SSIM because deriving it separately from
each image makes results incomparable between frames and runs. Use a documented
normalization such as `1.0`, or the camera's calibrated full-scale value.

For SSIM, a masked local window is included only when all of its 11×11 pixels
are valid. Images smaller than the window, or masks without a fully valid
window, return an error.

## Python API

The same functions are available under `fpm_rs.metrics`:

```python
import fpm_rs as fpm

score = fpm.metrics.ssim(
    reference,
    candidate,
    valid_mask=valid_pixels,
    data_range=65535.0,
)
gain = fpm.metrics.fitted_gain(reference, candidate)
```

Python accepts any real, two-dimensional NumPy-compatible numeric array,
including `uint8` and `uint16`, and evaluates it as `float64`. `valid_mask`
must be a two-dimensional Boolean array. The conversion is part of the Python
adapter; the Rust `ArrayView2` API itself remains zero-copy.
