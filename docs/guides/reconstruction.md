# Configure and run a reconstruction

## Compile the model

Create `Optics` and one illumination description, then call `compile_model`.
The measured `image_shape` and recovered `reconstruction_shape` are both
`(height, width)`. Physical distances are metres and angles are radians unless
a parameter explicitly says degrees.

## Choose the reconstruction shape

The measured intensity frames have `image_shape`; the reconstructed complex
object, including its amplitude and phase, has `reconstruction_shape`. Both
cover the same field of view. If the low-resolution object-plane pixel size is
`p_low`, aspect-preserving sizing gives `p_high = p_low * H_low / H_high =
p_low * W_low / W_high`.

Consequently, a synthetic-to-objective NA ratio is a useful estimate of the
linear increase in pixels. It is not the sizing rule used by the software. For
a fixed aspect ratio, a linear scale of `s` produces about `s²` as many total
pixels.

Python and Rust expose the same reconstruction-shape choices:

| Python value | Rust variant | Selected grid |
| --- | --- | --- |
| `(height, width)` | `ReconstructionShape::Exact((height, width))` | This exact grid, after validation. |
| `"minimum"` | `ReconstructionShape::Minimum` | The smallest geometry-valid grid. |
| `"smooth"` | `ReconstructionShape::Smooth` | The smallest valid grid whose shared multiplier has only 2, 3, 5, and 7 as prime factors. This is the Python default. |
| `"power_of_two"` | `ReconstructionShape::PowerOfTwo` | The smallest valid grid whose shared multiplier is a power of two. |

Inspect a choice without compiling a pupil by calling
`suggest_reconstruction_shape`:

```python
image_shape = (32, 32)
minimum = fpm.suggest_reconstruction_shape(
    optics,
    illumination,
    image_shape,
    reconstruction_shape="minimum",
)
smooth = fpm.suggest_reconstruction_shape(optics, illumination, image_shape)
radix2 = fpm.suggest_reconstruction_shape(
    optics,
    illumination,
    image_shape,
    reconstruction_shape="power_of_two",
)

model = fpm.compile_model(
    optics,
    illumination,
    image_shape,
    reconstruction_shape="smooth",
)
assert model.reconstruction_shape == smooth
```

Omitting the Python argument is equivalent to passing `"smooth"`. `None` is not
an automatic-mode sentinel and raises `TypeError`; use a tuple or one of the
three strings.

The equivalent Rust API uses the unified `ReconstructionShape` enum:

```rust
use fpm_rs::model::{ImagePlaneModel, ReconstructionShape};

# fn example(
#     optics: &fpm_rs::experiment::Optics,
#     illumination: &fpm_rs::experiment::LEDArray,
# ) -> fpm_rs::Result<()> {
let image_shape = (32, 32);
let suggested = ImagePlaneModel::suggest_reconstruction_shape(
    optics,
    illumination,
    image_shape,
    ReconstructionShape::Smooth,
)?;
let model = ImagePlaneModel::from_experiment(
    optics,
    illumination,
    image_shape,
    ReconstructionShape::Smooth,
)?;
assert_eq!(model.reconstruction_shape, suggested);
# Ok(())
# }
```

### How automatic sizing is calculated

The implementation compiles the illumination into transverse wave vectors and
maps them onto the low-resolution Fourier spacing. It then finds a grid that
contains every full low-resolution crop, including the extra neighbor required
by a fractional bilinear interpolation stencil. Asymmetric and one-sided
illumination are therefore sized from their actual directional extents, not
from a symmetric NA estimate.

For a low-resolution shape reduced to the aspect ratio `(a, b)`, every candidate
has the form `(a*t, b*t)`. `"minimum"` selects the smallest fitting integer `t`;
`"smooth"` selects the smallest fitting 2/3/5/7-smooth `t`; and
`"power_of_two"` selects the smallest fitting power-of-two `t`. The dimensions
can still contain prime factors inherited from `(a, b)`. For example, a `2:3`
aspect ratio always produces `(2*t, 3*t)`, so `"power_of_two"` does not imply
that both final dimensions are powers of two.

An exact tuple must preserve the low-resolution aspect ratio and contain every
crop and interpolation stencil. Invalid or undersized tuples fail during
suggestion and compilation. Every positive dimension is supported by RustFFT;
the automatic modes are memory/performance choices, not FFT compatibility
requirements. They also do not guarantee recoverable information or uniform
Fourier coverage.

Choose `LEDArray` for a planar grid, `AngleList` or `KVectorList` for calibrated
directions, and `CodedIllumination` for multiplexed frames. Spherical source
classes have additional identifiability constraints documented in
[Spherical illumination geometries](../spherical-geometries.md).

## Build the problem

```python
problem = fpm.ReconstructionProblem(
    measurements,
    model,
    frame_weights=frame_weights,
    masks=masks,
    name="sample-a",
)
```

Measurements must be a float64 array shaped `(frames, height, width)` or a
`MeasurementStack`. Masks use the same stack shape (or the supported broadcast
shape) and zero-valued pixels are excluded. The frame count and image dimensions
must agree with the compiled model.

## Select and run an algorithm

`AlternatingProjection` is the simplest starting point. `Fpie` adds regularized
object updates, `Epry` can recover the pupil and frame response, `Admm` separates
data fitting from overlap consensus, and `GradientDescent` supports calibrated
illumination/pupil recovery and regularization.

```python
algorithm = fpm.AlternatingProjection(iterations=20, object_step=1.0)
result = algorithm.run(problem, schedule="sequential")
```

`run` blocks until completion but releases the GIL while Rust reconstruction
work executes. A Python `IterationCallback` reacquires the GIL only when called.
Result array properties are NumPy arrays backed by Python-owned result storage;
treat them as outputs rather than mutable reconstruction state.

Use callbacks to add progress, CSV history, checkpoints, image snapshots, or
early stopping. See [Diagnostics and callbacks](../diagnostics.md) and the
[generated reconstruction API](../reference/python/reconstruction.md).

## Algorithm options and calibration

`Admm` uses an amplitude proximal operator, a preconditioned linearized object
consensus update, and scaled dual variables. It honors masks, frame weights,
known gains and backgrounds, schedules, and batches. `penalty`, `object_step`,
and `dual_relaxation` control those updates. The default batch spans all frames.
For multiplexed data, its joint amplitude proximal operates across all source
modes, so checkpointed auxiliary and dual fields contain two complex values per
frame-source-mode pixel.

`GradientDescent` defaults to an image-amplitude residual. Its `loss_type` can
select amplitude MSE, intensity MSE, Poisson negative log likelihood, or robust
Huber amplitude loss. Losses are evaluated in intrinsic intensity units after
removing known linear gain and background, keeping the trajectory independent
of camera-count scaling. It supports masks, frame weights, schedules, and true
mini-batches: frame gradients accumulate in reusable storage and one averaged
object update is applied per batch. `object_step` controls that update.

The gradient implementation evaluates independent frame chunks on up to the
available CPU workers; `parallel_workers` limits the count. Each worker reuses
local state, and the main thread reduces chunks in deterministic batch order,
including multiplexed shared-source curvature. Memory therefore grows with
active workers rather than batch length. TV and pupil smoothing run once after
the reduction for each batch.

`recover_pupil(true)` enables mini-batch pupil updates for ordinary and
multiplexed frames. `pupil_step` controls the normalized update and
`constrain_pupil_support` projects it onto the compiled support. It can be used
with illumination recovery. `object_tv(weight)` applies isotropic TV to both
components of the complex object; `object_tv_epsilon` controls its smooth
near-zero approximation. `pupil_smoothing(weight)` applies a quadratic
nearest-neighbour penalty and requires pupil recovery. Regularization weights
are scaled by the fraction of all frames in the batch, while history continues
to report data loss rather than the regularized objective.

Enable per-source position recovery with
`GradientDescent::recover_illumination(true)`. Corrections are returned as
`(row, column)` Fourier-grid offsets from the compiled model: `(dr, dc)` means
`dky = dr * sampling.dky` and `dkx = dc * sampling.dkx`. The implementation
uses finite differences of the shared subpixel forward operator with diagonal
Gauss–Newton/Fisher scaling. `illumination_step`,
`illumination_finite_difference`, and `illumination_bounds` control damping,
derivative spacing, and maximum correction. It works per source in multiplexed
frames and costs four additional forward-field evaluations per calibrated source.

EPRY can recover relative frame gains with `recover_frame_gains(true)`. Its
bounded, damped least-squares update is controlled by `gain_step` and
`gain_bounds`; evaluation removes the global object/gain scale ambiguity.
`recover_background(true)` estimates additive per-frame offsets, with
`background_step` and `background_bounds` controlling the residual-mean update.
It preserves supplied spatial background maps, but a common absolute background
is ambiguous with DC object intensity and should be referenced to a frame or
dark measurement.

## Checkpoints, results, callbacks, and schedules

Checkpoints contain spectrum, pupil, calibration variables, and full history.
Load one with `ReconstructionCheckpoint::load` and pass it to
`ReconstructionAlgorithm::run_from_checkpoint`; `iterations` remains the target
total, not an additional number of iterations. Save/load validates format
version, finite state, auxiliary consistency, and monotonic history;
`load_for_problem` additionally validates dimensions and calibration against a
specific problem before reconstruction begins.

`ReconstructionResult::save_bundle` and `load_bundle` persist a validated,
versioned JSON result bundle containing the complex object and spectrum,
amplitude, phase, pupil, calibration, diagnostics, history, runtime, and
metadata. Component-level PNG, JSON, and CSV writers remain available.

Periodic callbacks request work only at active hook points. For example,
`SaveImageEvery::new(10, ...)` performs the object inverse FFT on iterations
10, 20, and so on. Custom callbacks can implement `requires_for` for the same
behaviour while retaining `requires` for capability inspection.
`SaveResidualsEvery` writes a mask-aware, zero-centred residual image per frame
at its configured cadence. `on_frame_end` runs once per completed frame; for a
multi-frame batch it receives the shared post-batch state, frame and batch
indices, and the individual frame loss.

Schedules include sequential, brightfield-first, spiral-out, seeded random,
and measurement-aware SNR ordering. `FrameSchedule::SnrWeighted` processes the
highest empirical shot-noise-SNR frame first, accounting for weights, masks,
and known background. Its score is a measured-intensity proxy, not a calibrated
camera-noise model. `Runner` supplies measurements automatically; model-only
`FrameSchedule::order` calls use sequential order for this mode.
