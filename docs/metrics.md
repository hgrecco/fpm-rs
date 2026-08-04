# Image metrics

## Intensity metrics

`metrics::intensity` contains pure, domain-agnostic calculations over scalar
intensity images. Internally, `single.rs` holds single-image calculations and
`compare.rs` holds reference/estimate comparisons. Those implementation
modules are private: every function and result type is available directly from
`metrics::intensity`.

Every two-image metric calls its inputs `reference` and `estimate`. A signed
residual is always `estimate - reference`. An optional `valid_mask` has the
same shape as both images; `true` includes a pixel and `false` excludes it.
All metrics reject mismatched shapes, non-finite included values, and an empty
valid mask.

Evaluation metrics do not contain losses used to optimize a reconstruction:
optimization objectives remain separate in `algorithms::objective` because
their gradient and numerical-stability contracts differ from reporting
metrics.

### Rust API

All comparison functions accept `ndarray::ArrayView2<'_, T>` with
`T: num_traits::ToPrimitive`. They convert individual included samples to
`f64` for evaluation and return `f64`; the image data is borrowed, never
copied. This supports ordinary `u8`, `u16`, `u32`, `f32`, and `f64` images.
The `stats` function summarizes one non-empty `&[f64]` intensity image.

```rust
use fpm_rs::metrics::intensity::{nrmse, psnr, stats};

let image_stats = stats(reference.as_slice().unwrap(), None)?;
let relative_error = nrmse(reference.view(), estimate.view(), None)?;
let quality_db = psnr(reference.view(), estimate.view(), None, 65_535.0)?;
```

### Metric definitions

| Function | Definition or convention |
| --- | --- |
| `stats` | Mean, population standard deviation, extrema, sum, zeros, and optional saturation count for one image. |
| `compare_intensity` | Aggregate sums and residual statistics for a reference/estimate pair. |
| `bias` | Mean signed residual. |
| `mae`, `mse`, `rmse` | Mean absolute, squared, and root mean squared residual. |
| `relative_l1` | `sum(abs(estimate - reference)) / sum(abs(reference))`. |
| `nrmse` | `L2(estimate - reference) / L2(reference)`. |
| `amplitude_nrmse` | NRMSE after applying `sqrt` to both intensities; requires non-negative values. |
| `correlation` | Pearson correlation; undefined for a constant valid image. |
| `psnr` | Peak signal-to-noise ratio in dB. `data_range` is required, finite, and positive; identical inputs produce `+∞`. |
| `ssim` | Single-scale SSIM, higher is more similar. Uses an 11×11 Gaussian window with σ=1.5, `K1=0.01`, and `K2=0.03`, following [Wang, Bovik, Sheikh, and Simoncelli, “Image quality assessment: From error visibility to structural similarity” (2004)](https://doi.org/10.1109/TIP.2003.819861). |
| `poisson_deviance` | Summed Poisson deviance for non-negative intensities. A positive `epsilon` floors estimate intensity. |
| `mean_poisson_deviance` | Poisson deviance divided by valid-pixel count. |
| `fitted_gain` | Least-squares gain in `estimate ≈ gain × reference`. |

`relative_l1`, `nrmse`, and `fitted_gain` are undefined for a zero reference
normalization and return an error rather than silently choosing a scale.

`data_range` is explicit for PSNR and SSIM because deriving it separately from
each image makes results incomparable between frames and runs. Use a documented
normalization such as `1.0`, or the camera's calibrated full-scale value.

For SSIM, a masked local window is included only when all of its 11×11 pixels
are valid. Images smaller than the window, or masks without a fully valid
window, return an error.

### Python API

The same functions are available under `fpm_rs.metrics`:

```python
import fpm_rs as fpm

image_stats = fpm.metrics.stats(reference, saturation_value=65535.0)
summary = fpm.metrics.compare_intensity(reference, estimate, valid_mask=valid_pixels)
score = fpm.metrics.ssim(
    reference,
    estimate,
    valid_mask=valid_pixels,
    data_range=65535.0,
)
gain = fpm.metrics.fitted_gain(reference, estimate)
```

Python accepts any real, two-dimensional NumPy-compatible numeric array,
including transposed and stepped arrays and `uint8`/`uint16` data, and evaluates
it as `float64`. Existing `float64` strides are preserved; dtype conversion may
allocate. `valid_mask` may also be strided and must be a two-dimensional Boolean
array. The Python bridge copies logical values into Rust-owned arrays before
releasing the GIL; this is not a contiguity-repair copy and applies to both
standard and strided inputs. The Rust `ArrayView2` metric API itself borrows
arbitrary layouts without a copy; FFT-based complex metrics additionally
allocate their required transform workspace.

## Complex-field metrics

`metrics::complex_field` provides eight focused comparison metrics for complex
images. Each function borrows `ndarray::ArrayView2<Complex<T>>` inputs, supports
`f32` and `f64`, accumulates in `f64`, and requires an explicit
`ComplexAlignment`. An optional Boolean mask uses `true` to include a pixel;
the same pixels determine both the fitted alignment and the reported metric.

```rust
use fpm_rs::metrics::complex_field::{ComplexAlignment, nrmse};

let error = nrmse(
    reference.view(),
    estimate.view(),
    Some(valid_mask.view()),
    ComplexAlignment::GlobalPhase,
)?;
```

For selected reference values `r` and estimate values `e`, alignment multiplies
the estimate by a scalar `a`. All fitted modes minimize
`sum(|a e - r|²)`. Defining `c = sum(conj(e) r)` and
`E = sum(|e|²)`, the modes are:

| Alignment | Fitted scalar |
| --- | --- |
| `None` | `a = 1` |
| `GlobalPhase` | `a = c / |c|`, constrained to unit magnitude |
| `Scale` | `a = max(real(c) / E, 0)`, constrained to a non-negative real value |
| `ComplexGain` | `a = c / E`, unconstrained complex value |

After alignment, the residual is always `a e - r`.

| Function | Definition or convention |
| --- | --- |
| `bias` | Arithmetic mean of the complex residual. |
| `mae` | Mean residual magnitude. |
| `mse`, `rmse` | Mean squared residual magnitude and its square root. |
| `relative_l1` | `sum(|a e - r|) / sum(|r|)`. |
| `nrmse` | `L2(a e - r) / L2(r)`. |
| `amplitude_nrmse` | `L2(|a e| - |r|) / L2(|r|)`. |
| `correlation` | `sum(conj(r) a e) / sqrt(sum(|r|²) sum(|a e|²))`. |

The correlation convention means that with no alignment, an estimate equal to
`reference * exp(i theta)` has correlation `exp(i theta)`. Undefined zero
normalizations, degenerate fitted alignments, shape mismatches, empty masks,
and non-finite selected components return `ComplexMetricError`; denominators
are never adjusted with an arbitrary epsilon. Alignment is applied during
accumulation, so no aligned image is allocated.

The aggregate `compare_complex_fields` report used by reconstruction evaluation
remains separate from this explicit-alignment API. Evaluation metrics also
remain separate from the optimization losses in `algorithms::objective`.
