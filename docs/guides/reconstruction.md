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
#     illumination: &fpm_rs::experiment::Illumination,
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

Choose `PlanarLEDArray` for a planar grid, `DirectionList` for wavelength-independent
directions, and `KVectorList` for wavelength-dependent calibrated vectors. Put
subsets, repetitions, or multiplexing in `AcquisitionPlan`, then combine it with
geometry and `SourceCalibration` in `Illumination`. Spherical source classes have
additional identifiability constraints documented in
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

## Choose an algorithm

Choose for the dominant data or model mismatch. These are starting points for
the implementations in fpm-rs, not a claim that one method dominates across
experiments.

| Condition or priority | Reach for | Decision boundary |
| --- | --- | --- |
| Clean data and a trusted pupil, illumination, gain, and background model | `AlternatingProjection` | Use the smallest baseline first. It has few controls and exposes whether the acquisition and compiled model are internally consistent. |
| Object-only recovery with weak pupil transfer or mild noise | `Fpie` | Prefer its stabilized object update when plain projection is too sensitive in weak-transfer regions. It does not estimate the pupil or source geometry. |
| Noise statistics, outliers, or an object prior must enter the update | `GradientDescent` | Select Poisson, Huber-amplitude, intensity, or amplitude loss as appropriate; use total variation only when that prior is defensible. This is the most configurable route, with more tuning and compute. |
| Pupil aberration or defocus is suspected | `Epry` | Recover the complex pupil with the object. It can also estimate per-frame gain or uniform background. If a selectable data loss or pupil regularization is essential, use pupil-recovering `GradientDescent` instead. |
| Updates need full- or multi-frame consensus rather than sequential frame corrections | `Admm` | Its auxiliary fields and dual variables make cross-frame agreement explicit. The default full-frame batch costs more memory and introduces penalty and relaxation controls. |
| Independent illumination vectors may be wrong | `GradientDescent(recover_illumination=True)` | Use this for generic per-source Fourier-grid corrections. The result is not necessarily a realizable apparatus geometry. |
| A planar LED array's pose, pitch, reference index, selected offsets, source powers, or frame gains must be self-calibrated | `JointReconstruction` around `Fpie` or `Epry` | Use the physical workflow when the desired result must remain a bounded, serializable `PlanarLEDArray`. Its identifiability constraints are part of the model, not optional tuning. |

When several rows apply, establish an object-only baseline before enabling the
smallest set of recovery variables that explains the residuals. In particular,
do not use generic source correction as a substitute for physical planar-array
calibration. The generated [algorithm API](../reference/python/algorithms.md)
documents each implementation and its cited method; the physical calibration
assumptions and gauges are detailed below.

## Select and run an algorithm

`AlternatingProjection` is the simplest starting point. `Fpie` adds regularized
object updates, `Epry` can recover the pupil and frame response, `Admm` separates
data fitting from overlap consensus, and `GradientDescent` supports generic
Fourier-grid source correction, pupil recovery, and regularization. Physical
planar-array calibration is the separate `JointReconstruction` workflow below.

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
[generated algorithm API](../reference/python/algorithms.md) and
[results API](../reference/python/results-and-bundles.md).

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
are scaled by the fraction of all frames in the batch. The universal trace
reports the algorithm's data objective; recorder-only fields can separately
report data and regularization components when an algorithm supplies them.

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

## Physical planar LED-array calibration

Physical calibration and generic k-vector correction solve different problems.
`GradientDescent(recover_illumination=True)` estimates independent source shifts
in Fourier-grid pixels for any compiled source geometry. Those shifts need not
describe a realizable apparatus. `JointReconstruction` instead accepts only
`PlanarLEDArray`, optimizes its pose and lattice in physical coordinates, and
returns a normal serializable `Illumination`. `DirectionList`, `KVectorList`, and
spherical geometries continue to use generic correction and are rejected by the
physical calibrator.

The forward model is the same `ImagePlaneModel`/`ForwardModel` implementation
used by simulation and reconstruction. Each outer iteration performs complete
passes of the wrapped analytic object algorithm, bounded physical updates, and
an illumination-only model refresh. Rust accepts any compatible reconstruction
algorithm; Python accepts `Fpie` or pupil-recovering `Epry`. The default
calibration objective is amplitude MSE. Intensity MSE,
Poisson negative log likelihood, and Huber amplitude loss are also available;
all honor measurement masks and frame weights. A parameter prior contributes
`0.5 * strength * ((value - center) / scale)²`.

### Parameters, units, and gauges

| Group | Absolute values | Unit and convention |
| --- | --- | --- |
| translation | `tx`, `ty`, `tz` | metres in sample coordinates |
| rotation | `rx`, `ry`, `rz` | radians; active, right-handed, extrinsic fixed sample axes, x then y then z (`Rz * Ry * Rx`) |
| pitch | `pitch_x`, `pitch_y` | metres |
| reference index | `reference_column`, `reference_row` | fractional lattice indices |
| selected offsets | `offset_x`, `offset_y`, `offset_z` | local array coordinates in metres, for explicitly named source indices only |
| source power | one value per stable source | dimensionless relative intensity |
| frame gain | one value per acquisition frame | dimensionless intensity gain |

Every active parameter has finite inclusive bounds, a physical finite-difference
step, an optimizer scale, and an optional quadratic prior. Unspecified
parameters stay fixed. Perturbations use central differences when both sides
are valid and a one-sided derivative at bounds or beside invalid geometries.
The normalized variables exposed in results are `(current - initial) / scale`.
Backtracking rejects non-finite positions, non-positive pitch or multipliers,
sources on the sample plane, and crops outside the fixed reconstruction grid.

The following gauges are enforced:

- `tx` with `reference_column`, and `ty` with `reference_row`, are rejected.
- Source power and frame gains are each normalized to mean one. Optimizing both
  groups together is rejected because their product retains a scale ambiguity.
- When translation and selected source offsets are active together, at least
  two sources must be selected and their mean XYZ offset is constrained to zero.
- Pitch and axial translation require explicit finite bounds. They may remain
  strongly correlated, so the conditioning result emits a warning and reports
  scaled sensitivity, approximate diagonal curvature, a diagonal condition
  estimate, bound activity, and rejected steps. These are numerical
  identifiability indicators, not statistical uncertainty.

### Python workflow

This example calibrates translation and rotation, jointly reconstructs the
object, and then reuses the calibrated illumination for a continuation run:

```python
parameters = fpm.PlanarArrayCalibrationParameters(
    translation=(True, True, True),
    rotation=(True, True, True),
    translation_spec=fpm.CalibrationParameterSpec(
        -0.1,
        0.1,
        scale=1e-3,
        finite_difference_step=1e-5,
    ),
    rotation_spec=fpm.CalibrationParameterSpec(
        -0.25,
        0.25,
        scale=1e-2,
        finite_difference_step=1e-4,
    ),
)
calibration = fpm.IlluminationCalibration(
    parameters,
    optimizer=fpm.BoundedFiniteDifferenceOptimizer(
        max_steps=2,
        relative_tolerance=1e-6,
    ),
)
joint = fpm.JointReconstruction(
    fpm.Fpie(iterations=1, object_step=0.8),
    optics,
    illumination,
    calibration,
    outer_iterations=10,
    object_iterations_per_outer=10,
)
joint_result = joint.run(problem)

continued_model = fpm.compile_model(
    optics,
    joint_result.calibrated_illumination,
    problem.image_shape,
    problem.reconstruction_shape,
)
continued = fpm.Fpie(iterations=20).run(
    fpm.ReconstructionProblem(measurements, continued_model)
)
```

Use group-specific bounded configurations for the other supported cases:

```python
# Pitch plus axial distance: finite physical bounds are mandatory; inspect the warning.
pitch_distance = fpm.PlanarArrayCalibrationParameters(
    translation=(False, False, True),
    pitch=(True, True),
    translation_spec=fpm.CalibrationParameterSpec(
        -0.12, -0.04, scale=1e-3, finite_difference_step=1e-5
    ),
    pitch_spec=fpm.CalibrationParameterSpec(
        3.5e-3, 4.5e-3, scale=1e-4, finite_difference_step=1e-6
    ),
)

# Only these stable source indices receive local XYZ variables.
selected_offsets = fpm.PlanarArrayCalibrationParameters(
    position_offsets=[12, 24, 103],
    position_offset_spec=fpm.CalibrationParameterSpec(
        -0.5e-3, 0.5e-3, scale=50e-6, finite_difference_step=1e-6,
        prior_center=0.0, regularization_strength=1e-4,
    ),
)

# Source powers use the intensity-only update path and are normalized to mean one.
source_power = fpm.PlanarArrayCalibrationParameters(relative_source_power=True)

# Frame gains are an alternative mean-one group, not simultaneous with source power.
frame_gain = fpm.PlanarArrayCalibrationParameters(frame_gains=True)
```

For deterministic true-versus-assumed simulation, compile measurements with a
deliberately translated/tilted/pitch-perturbed `true_illumination`, pass the
nominal model as `reconstruction_model`, and build the reconstruction problem
from `simulation.reconstruction_model`. The calibrated illumination can then be
resolved, serialized, plotted, simulated, or compiled exactly like the nominal
one. `result.parameter_history`, `loss_history`, `conditioning`, and
`diagnostics` retain accepted/rejected steps and units through stable parameter
names.

### Rust workflow

The equivalent Rust selection and joint run use the same canonical units:

```rust,no_run
use fpm_rs::{
    Result,
    algorithms::{Fpie, JointReconstruction},
    experiment::{Illumination, Optics},
    illumination_calibration::{
        BoundedFiniteDifferenceOptimizer, CalibrationParameterSpec,
        IlluminationCalibration, PlanarArrayCalibrationParameters,
    },
    measurements::MeasurementStack,
    model::{ImagePlaneModel, ReconstructionShape},
    reconstruction::{ReconstructionProblem},
};

# fn run(
#   optics: Optics,
#   illumination: Illumination,
#   measurements: MeasurementStack,
# ) -> Result<()> {
let translation = CalibrationParameterSpec::new(-0.1, 0.1, 1e-3)
    .finite_difference_step(1e-5);
let rotation = CalibrationParameterSpec::new(-0.25, 0.25, 1e-2)
    .finite_difference_step(1e-4);
let parameters = PlanarArrayCalibrationParameters::builder()
    .translation_specs(std::array::from_fn(|_| Some(translation.clone())))
    .rotation_specs(std::array::from_fn(|_| Some(rotation.clone())))
    .build()?;
let calibration = IlluminationCalibration::new(parameters).optimizer(
    BoundedFiniteDifferenceOptimizer {
        max_steps: 2,
        relative_tolerance: 1e-6,
        ..Default::default()
    },
);
let model = ImagePlaneModel::from_experiment(
    &optics,
    &illumination,
    measurements.image_shape(),
    ReconstructionShape::Smooth,
)?;
let problem = ReconstructionProblem::new(measurements, model)?;
let result = JointReconstruction::new(
    Fpie::default().iterations(1),
    optics.clone(),
    illumination,
    calibration,
    10,
)
.object_iterations_per_outer(10)
.run(&problem)?;

let reusable: &Illumination = &result.calibrated_illumination;
let continued_model = ImagePlaneModel::from_experiment(
    &optics,
    reusable,
    problem.model.image_shape(),
    ReconstructionShape::Exact(problem.model.reconstruction_shape()),
)?;
# let _ = continued_model;
# Ok(())
# }
```

Rust selects pitch/distance, offsets, powers, and gains with
`pitch_specs`, `position_offset_specs` or `position_offsets`,
`relative_source_power_spec`, and `frame_gain_spec`. A fixed reconstructed
object can be calibrated without alternating updates through
`IlluminationCalibration::calibrate`.

### Partial updates, persistence, and limitations

Geometry changes recompute source positions, directions, k-vectors, crops, and
subpixel offsets inside the existing grid. They retain pupil samples, optical
sampling, background, shapes, and backend FFT plans. Source powers and frame
gains update only compiled incoherent weights and gains; tests assert that
k-vectors, crops, offsets, and pupils remain byte-for-byte unchanged. A global
geometry finite difference requires one forward evaluation and one
illumination-geometry refresh per valid side, while a power or gain difference
uses a model clone and the intensity-only update boundary. The
`geometry_recompilations` and `multiplicative_updates` counters include accepted
and rejected finite-difference/line-search trial evaluations, making the partial
update cost directly observable.

Ordinary callbacks run at outer-iteration boundaries and receive
`physical_illumination` metrics for the object and illumination phases.
Checkpoints contain absolute/normalized physical values, histories, the final
illumination, and the refreshed model, so resumed trajectories use the same
state. Result bundles store these in the verified
`domain.physical_illumination` artifact; histories do not inflate string
metadata. Rust `JointReconstructionResult::save_json`/`load_json` and the
matching Python methods preserve the complete structured result; `write_bundle`
uses the normal verified bundle layout.

The formulation follows the joint-estimation motivation of [J. Sun, Q. Chen,
Y. Zhang, and C. Zuo, “Efficient positional misalignment correction method for
Fourier ptychographic microscopy,” *Biomedical Optics Express* **7**(4),
1336–1350 (2016)](https://doi.org/10.1364/BOE.7.001336) and the illumination
self-calibration context of [R. Eckert, Z. F. Phillips, and L. Waller,
“Efficient illumination angle self-calibration in Fourier ptychography,”
*Applied Optics* **57**(19), 5434–5442
(2018)](https://doi.org/10.1364/AO.57.005434). Unlike Sun et al., fpm-rs uses
deterministic bounded, scaled finite differences rather than simulated
annealing and nonlinear regression; unlike Eckert et al., it does not perform
brightfield circle detection or spectral correlation. The shared thin-sample
FPM forward model originates with [G. Zheng, R. Horstmeyer, and C. Yang,
“Wide-field, high-resolution Fourier ptychographic microscopy,” *Nature
Photonics* **7**, 739–745
(2013)](https://doi.org/10.1038/nphoton.2013.187).

## Checkpoints, results, callbacks, and schedules

Checkpoints contain spectrum, pupil, calibration variables, algorithm
auxiliary state, and the full `ReconstructionTrace`.
Load one with `ReconstructionCheckpoint::load` and pass it to
`ReconstructionAlgorithm::run_from_checkpoint`; `iterations` remains the target
total, not an additional number of iterations. Save/load validates format
version, finite state, auxiliary consistency, one-based trace rows, and
monotonic elapsed seconds;
`load_for_problem` additionally validates dimensions and calibration against a
specific problem before reconstruction begins.

Every result owns a trace, even when no diagnostic callback is installed.
Universal iteration rows are `(iteration, objective, elapsed_seconds)`.
Algorithm-specific values such as ADMM primal and dual residuals are separate
long-form records with `iteration`, `namespace`, `metric`, and `value`.
`elapsed_seconds` includes any elapsed time restored from a checkpoint.

With the Rust `parquet` feature, write a final result bundle with:

```rust,no_run
# use fpm_rs::{Result, reconstruction::{BundleExportOptions, ReconstructionResult}};
# fn save(result: &ReconstructionResult) -> Result<()> {
let bundle = result.write_bundle(
    "output/reconstruction",
    BundleExportOptions {
        run_id: Some("experiment-42".into()),
        label: Some("baseline AP".into()),
        include_previews: true,
    },
)?;
let reopened = fpm_rs::read_bundle(&bundle.path)?;
let object = reopened.object()?; // hash-checked and cached on first access
# let _ = object;
# Ok(())
# }
```

The bundle is a directory containing stable Parquet tables, authoritative
`.npy` arrays, optional PNG previews, and `manifest.json`. Export uses a unique
final directory, works in a sibling `.inprogress` directory, removes transient
run-state/checkpoint files, writes the manifest last, and then renames the
directory atomically. Existing outputs are never silently replaced. Failed
workspaces are retained for inspection.

Python exposes the same structure without importing Polars:

```python
bundle = result.write_bundle(
    "output/reconstruction",
    run_id="experiment-42",
    include_previews=True,
)
reopened = fpm.read_bundle(bundle.path)
object_array = reopened.arrays.object.value
assert reopened.result.object is object_array
```

Manifest and artifact handles are eager. Scientific arrays, reconstructed
domain objects, optional diagnostics, and evaluation are loaded and cached on
first access. Cached NumPy arrays are read-only; `clear_cache()` drops only the
bundle's references, so arrays already held by user code remain valid.
`verify()` hashes every artifact and validates table and array structure.
Use `bundle.tables.history.path` with `polars.scan_parquet`, PyArrow, pandas, or
DuckDB. Install `fpm-rs[polars]` only when the Python Polars package is wanted.
Component-level PNG, JSON, checkpoint, and objective-CSV writers remain
available for their focused workflows.

### Query bundle tables

Artifact properties are ordinary paths, so analysis libraries remain optional:

```python
import polars as pl

history = pl.scan_parquet(bundle.tables.history.path)
print(history.select("iteration", "objective", "elapsed_seconds").collect())

if bundle.tables.algorithm_metrics is not None:
    algorithm_metrics = pl.scan_parquet(bundle.tables.algorithm_metrics.path)
    print(algorithm_metrics.filter(pl.col("namespace") == "admm").collect())

if bundle.tables.iteration_diagnostics is not None:
    iteration_diagnostics = pl.scan_parquet(
        bundle.tables.iteration_diagnostics.path
    )
    convergence = history.join(
        iteration_diagnostics,
        on=["run_id", "iteration"],
        how="left",
    ).collect()
```

Dynamic scalar diagnostics and string metadata use long-form key/value rows.
Pivot only when a wide report is useful:

```python
if bundle.tables.scalar_diagnostics is not None:
    scalar_rows = pl.read_parquet(bundle.tables.scalar_diagnostics.path)
    scalar_wide = scalar_rows.pivot(
        on="key",
        index="run_id",
        values="value",
        aggregate_function="first",
    )

if bundle.tables.metadata is not None:
    metadata_rows = pl.read_parquet(bundle.tables.metadata.path)
    metadata_wide = metadata_rows.pivot(
        on="key",
        index="run_id",
        values="value",
        aggregate_function="first",
    )
```

Equivalent readers require no fpm-rs adapter:

```python
import pandas as pd
import pyarrow.parquet as pq
import duckdb

history_pandas = pd.read_parquet(bundle.tables.history.path)
history_arrow = pq.read_table(bundle.tables.history.path)
history_duckdb = duckdb.read_parquet(str(bundle.tables.history.path))
```

`pandas.read_parquet` requires an installed Parquet engine such as PyArrow.
Use bundle properties for authoritative arrays and preview paths, or open the
standard `.npy` artifact directly:

```python
import numpy as np

object_field = bundle.arrays.object.value       # cached, read-only NumPy
object_npy_path = bundle.arrays.object.path     # usable with numpy.load
object_from_file = np.load(object_npy_path)
amplitude_preview = bundle.previews.object_amplitude
if amplitude_preview is not None:
    print(amplitude_preview.path)
```

Preview artifacts are display-oriented PNG files. Pillow remains optional:

```python
from PIL import Image

preview = bundle.previews.object_amplitude
if preview is not None:
    image = Image.open(preview.path)
```

Periodic callbacks request work only at active hook points. For example,
`SaveImageEvery::new(10, ...)` performs the object inverse FFT on iterations
10, 20, and so on. Custom callbacks can implement `requires_for` for the same
behaviour while retaining `requires` for capability inspection.
`SaveResidualsEvery` writes a mask-aware, zero-centred residual image per frame
at its configured cadence. `on_frame_end` runs once per completed frame; for a
multi-frame batch it receives the shared post-batch state, frame and batch
indices, and the individual frame objective.

Schedules include sequential, brightfield-first, spiral-out, seeded random,
and measurement-aware SNR ordering. `FrameSchedule::SnrWeighted` processes the
highest empirical shot-noise-SNR frame first, accounting for weights, masks,
and known background. Its score is a measured-intensity proxy, not a calibrated
camera-noise model. `Runner` supplies measurements automatically; model-only
`FrameSchedule::order` calls use sequential order for this mode.
