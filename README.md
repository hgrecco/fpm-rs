# fpm-rs

`fpm-rs` is a CPU reconstruction and simulation library for **image-plane Fourier
ptychographic microscopy**. Measurements are real-space intensity images acquired
under different illumination angles; diffraction-plane ptychography is outside
the crate's scope.

**[Read the documentation](docs/index.md)** for installation, a Python-first
quickstart, maintained tutorials, task guides, concepts, and generated Python
and Rust API references.

The main architectural boundary is:

```text
Optics + illumination geometry
        ↓ compile
k-vectors + pupil + Fourier crops + sampling
        ↓
ImagePlaneModel + (MeasurementStack or LazyMeasurementStack)
        ↓
ReconstructionProblem<M: MeasurementRead>
        ↓
Algorithm update rules + Runner orchestration
        ↓
ReconstructionResult
```

Algorithms consume `ImagePlaneModel`, never `Optics` or `LEDArray`. The same
`ForwardModel` is used by reconstruction and `Simulator`, preventing the two
models from drifting apart.

Reconstruction is generic over the read-only `MeasurementRead` trait.
`MeasurementStack` and `LazyMeasurementStack` implement it directly, so
algorithms are statically dispatched without an intermediate dataset enum or
measurement trait objects. Construction, mutation, preprocessing,
materialization, and cache controls remain methods of the concrete storage
types.

LED intensity weights are reordered with the acquisition order and compiled into
`ImagePlaneModel::frame_gains`; they are therefore available without retaining
the original `LEDArray` metadata.

For coded illumination, `source_count()` is distinct from `frame_count()` and the
multiplexing matrix maps each measured frame to incoherently weighted sources.
The shared forward model, simulator, AP, Fpie, Epry, ADMM, and
`GradientDescent` solver all support these intensity sums. Projection methods
apply one measured/predicted amplitude ratio to every incoherent source mode and
back-project each mode using its normalized multiplex weight.

## Quick start

```rust
use fpm_rs::{
    algorithms::{AlternatingProjection, ReconstructionAlgorithm},
    experiment::{LEDArray, Optics},
    model::ImagePlaneModel,
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject},
};

# fn main() -> fpm_rs::Result<()> {
let optics = Optics {
    wavelength: 532e-9,
    objective_na: 0.10,
    magnification: 4.0,
    camera_pixel_size: 6.5e-6,
    medium_index: 1.0,
    defocus_distance: None,
    pupil_aberration: None,
};
let leds = LEDArray::new()
    .grid_shape((3, 3))
    .pitch(4e-3)
    .distance(90e-3)
    .center((1.0, 1.0));
let model = ImagePlaneModel::from_experiment(
    &optics, &leds, (32, 32), (64, 64),
)?;
let simulation = Simulator::ideal(model)
    .object(SyntheticObject::resolution_target((64, 64))?)
    .seed(1234)
    .simulate()?;
let problem = ReconstructionProblem::new(
    simulation.measurements,
    simulation.reconstruction_model,
)?;
let result = AlternatingProjection::default()
    .iterations(20)
    .run(&problem)?;
assert_eq!(result.amplitude.shape(), (64, 64));
# Ok(())
# }
```

Run the complete example with:

```sh
cargo run --example simulate_and_reconstruct
```

### Python diagnostics

Python diagnostics use the same Rust callback system as the reconstruction
runner. The recorder remains accessible after the run because the Python object
and runner callback share the underlying Rust state:

```python
recorder = fpm_rs.DiagnosticRecorder("basic")
result = algorithm.run(problem, callbacks=[recorder])

diagnostics = recorder.diagnostics()
fpm_rs.diagnostics.make_diagnostic_report(
    diagnostics,
    output_dir="diagnostic_report",
)
```

`minimal` records convergence history, `basic` adds Fourier coverage, and
`debug` additionally records raw-stack and per-frame summaries. `simulation`
currently matches `basic` because reconstruction problems do not retain ground
truth. Plot rendering lives under `fpm_rs.plot`, while `fpm_rs.diagnostics`
handles loading and report generation. JSON persistence is optional through
`recorder.to_json(path)`. See the [diagnostics workflow](docs/diagnostics.md).
[`diagnostics_quickstart.ipynb`](docs/tutorials/notebooks/diagnostics_quickstart.ipynb)
teaches the lightweight `basic` recorder and the two overview plots.
[`diagnostics_debug.ipynb`](python/examples/diagnostics_debug.ipynb) focuses on
debug-only frame and raw-stack inspection. [`diagnostics_plots.ipynb`](python/examples/diagnostics_plots.ipynb)
demonstrates calling the plotting helpers directly and customizing the returned
Matplotlib axes. The notebooks show figures inline by default and include
commented `savefig(...)` examples when you want to export them.

## Coordinate and normalization conventions

- Arrays are row-major and shapes are `(height, width)`.
- Spectra and pupils are stored FFT-shifted, with zero frequency at the array
  centre.
- Each source stores an integer Fourier crop plus an optional fractional
  `(row, column)` offset in grid pixels. Experiment compilation uses
  `row = ky/dky` and `column = kx/dkx`, rounds the integer crop origin, and
  retains the signed fractional remainder.
- Fractional crops use bilinear sampling. Reconstruction inserts updates with
  the exact adjoint interpolation weights; it does not round or treat the
  interpolation as an invertible copy. Zero-offset crops retain the direct-copy
  fast path.
- `KVector` values are transverse angular spatial frequencies in radians/metre.
  Positive `kx`/`ky` move the Fourier crop toward increasing column/row indices.
- High-level direct-k and angle descriptions must represent propagating waves:
  the transverse magnitude cannot exceed `2πn/λ`. Component angles are limited
  to `[-π/2, π/2]`, and their combined transverse direction is validated.
- An LED at lateral position `(x, y)` and axial distance `z` produces
  `k0 * (x, y) / sqrt(x² + y² + z²)`, where `k0 = 2πn/λ`.
- Spherical coordinates use polar angle `theta` from positive `z` and azimuth
  `phi` from positive `x` toward positive `y`.
- `dkx = 2π / (width * object-plane pixel size)` and similarly for `dky`.
- CPU FFTs normalize the forward transform by `1/N`; inverse transforms are
  unnormalized. This preserves constant-object amplitude when a high-resolution
  spectrum is cropped and inverse-transformed on a smaller grid.
- The ideal pupil includes samples with transverse frequency
  `sqrt(kx² + ky²) <= 2π NA / λ`.
- `PupilAberration` coefficients are direct radian weights for the sampled
  radial-polynomial terms documented on `Optics` and `Pupil::circular`; they
  are not normalized Zernike coefficients.

`CoordinateConvention::CenteredPositiveK` records these choices in compiled
models. Low-level model construction remains available for imported k-vectors,
simulations, and algorithm tests. `ImagePlaneModel::new` defaults to integer
crops; use `with_subpixel_offsets` when low-level inputs include calibrated
fractional source positions.

`Illumination::Calibrated` represents imported k-vectors with optional
per-frame gains and optional multiplexing rows. Dataset subsets use this variant
to preserve calibrated model effects without inventing LED geometry.

## Modules

- `experiment`: optional `Optics`, planar `LEDArray`, fixed `LEDSphere`, moving
  `SphericalLEDArm`, rotating `RotatingLEDArc`, angle, and direct-k descriptions.
- `model`: sampling, pupil, Fourier crops, compiled model, shared forward model.
- `measurements`: image-plane intensity stacks and preprocessing.
- `algorithms`: alternating projection, Fpie, Epry, linearized ADMM, and
  loss-gradient reconstruction.
- `reconstruction`: problem/state/result, schedules, batches, and runner.
- `callbacks`: image output, CSV logging, checkpoints, progress, early stopping.
- `simulation`: synthetic objects, source mismatch, concrete camera response,
  and ground-truth metrics.
- `datasets`: strict local bundle loading, registry discovery, verified
  downloads, managed caching, and deterministic frame/pixel subsets.
- `backend`: a small backend boundary and cached-plan `rustfft` CPU backend.
- `benchmark`: generic single-case execution plus versioned CSV/JSON benchmark
  records and reconstruction artifacts.

The flat-buffer core and backend boundary allow later GPU-resident state and PyO3
bindings without putting experimental metadata into algorithm update rules.

Source-specific conversion lives outside this repository. The dataset registry
can discover and download already-converted archives into a managed cache;
local and registered acquisitions then use the same `DatasetLoader`. The
format, registry, CLI, cache, and subset APIs are documented in
[Datasets](docs/datasets.md) and specified by [dataset_spec.md](dataset_spec.md).

Deterministic simulator presets and named benchmark profiles are documented in
[Reconstruction benchmarks](docs/benchmarks.md). Run
`cargo run --example benchmark_algorithms` for the offline smoke profile.
Native manifests may include optional ground truth, valid-object masks,
provenance, and measurement units.

`Admm` uses an amplitude proximal operator, a preconditioned linearized object
consensus update, and scaled dual variables. It honors masks, frame weights, known
gains/backgrounds, schedules, and batches. `penalty`, `object_step`, and
`dual_relaxation` control its updates. The default batch spans all frames. For
multiplexed data, its joint amplitude proximal operates across all source modes;
auxiliary and dual fields therefore require two complex values per
frame-source-mode pixel. They are included in resumable checkpoints.

`GradientDescent` defaults to an image-amplitude residual and accepts both one
source per frame and incoherently multiplexed frames. `loss_type` selects amplitude
MSE, intensity MSE, Poisson negative log likelihood, or robust Huber amplitude
loss. Losses are evaluated in intrinsic intensity units after removing known
linear gain and background, so camera count scaling does not change the update
trajectory. The solver also honors masks, frame weights, acquisition schedules,
and true mini-batches: frame gradients are accumulated in reusable storage and an
averaged object update is applied once per batch. Use `object_step` to control the
update size. By default the solver updates only the object spectrum.
Independent frame gradients run on up to the machine's available CPU workers and
are reduced in deterministic batch order; `parallel_workers` controls this limit.
Each worker reuses one local state and accumulates object, pupil, and
illumination-position contributions across a contiguous frame chunk. The main
thread reduces those chunks in batch order, including shared-source curvature
from multiplexed frames, before applying one update. Memory therefore scales with
the active worker count rather than the batch length. TV and pupil smoothing
remain one post-reduction operation per batch.

`recover_pupil(true)` enables mini-batch pupil updates for ordinary or
multiplexed frames. `pupil_step` controls the normalized update and
`constrain_pupil_support` projects the result back onto the compiled support.
Pupil and illumination recovery can be enabled together; both gradients are
computed from the same pre-update object state.

`object_tv(weight)` applies an isotropic total-variation step jointly to the real
and imaginary object components; `object_tv_epsilon` controls its differentiable
near-zero approximation. TV requires one inverse/forward high-resolution FFT pair
per batch. `pupil_smoothing(weight)` applies a quadratic nearest-neighbor penalty
and requires `recover_pupil(true)`. Regularization weights are multiplied by the
batch's fraction of all frames, keeping their per-iteration strength approximately
stable across batch sizes. Reconstruction history continues to report data loss,
not data loss plus the regularization penalty.

Enable per-source position recovery with
`GradientDescent::recover_illumination(true)`. Corrections are stored and
returned as `(row, column)` offsets in Fourier-grid pixels, relative to the
compiled model. Thus a correction `(dr, dc)` corresponds to
`dky = dr * sampling.dky` and `dkx = dc * sampling.dkx`. The implementation uses
finite-difference derivatives of the shared subpixel forward operator with a
diagonal Gauss–Newton/Fisher scaling. `illumination_step`,
`illumination_finite_difference`, and `illumination_bounds` control damping,
derivative spacing, and the maximum absolute correction. Position recovery works
per source even when measured frames multiplex several sources, and corrections
are included in checkpoints and results. It is disabled by default because each
calibrated source requires four additional forward-field evaluations.

Backends are thread-safe trait objects and can be injected with
`Runner::with_backend`, `ReconstructionState::initialize_with_backend`, or
`ForwardModel::with_backend`. Checkpoint resume preserves the injected backend.
Callback-requested forward diagnostics use the same injected backend rather than
silently constructing a separate CPU implementation.
`BackendCapabilities`, typed `ComplexBuffer`/`RealBuffer` objects, and the
`ResidentBackend` extension expose preferred memory location, transfers, and
resident FFT dispatch. `CpuBackend` implements this contract with host buffers;
a CUDA implementation can use device buffers without changing
`ImagePlaneModel` or `ReconstructionProblem`. Reconstruction algorithms still
use host-owned state today, so this is the device-residency seam rather than a
claim of current GPU execution.

## CPU performance

`ForwardModel::forward_intensity` is the allocation-owning convenience API.
Repeated simulation, metrics, or parameter-search code can create one
`ForwardWorkspace` with `ForwardModel::workspace` and call
`forward_intensity_into` or `forward_source_field_into`; the crate's simulator and
diagnostic loops use this path. A workspace is mutable scratch and must not be
shared concurrently—create one per worker when evaluating independent frames in
parallel. FFT plans and backend objects remain shareable.

For complete stacks, `ForwardModel::forward_intensity_stack_into` evaluates
frames with scoped CPU workers while preserving `[frame][row][column]` output
order and using one workspace per worker. `Simulator` uses the machine's
available parallelism for optical prediction, then asks `CameraModel` to convert
frames to counts serially so seeded detector noise is independent of worker count.

Run the dependency-free forward benchmark with:

```sh
cargo bench --bench forward_model
```

Set `FPM_BENCH_ITERATIONS` to change its duration. It reports both the allocating
and workspace-reuse paths but applies no machine-specific pass/fail threshold.

Run the gradient scaling and memory benchmark with:

```sh
cargo bench --bench gradient_parallel
```

It covers ordinary object updates plus multiplexed object, pupil, and illumination
updates. For each worker count it reports milliseconds per batch step, speedup
relative to one worker, and peak incremental heap measured during the step.
`FPM_GRADIENT_BENCH_LOW_SIZE`, `FPM_GRADIENT_BENCH_HIGH_SIZE`,
`FPM_GRADIENT_BENCH_ITERATIONS`, and `FPM_GRADIENT_BENCH_MAX_WORKERS` configure
the workload. The heap figure includes worker-local state and reduction buffers,
but excludes pre-existing reconstruction state, native thread stacks, and memory
owned internally by system FFT or allocator implementations.

Use one worker for single-frame or very small batches. For larger CPU batches,
start with two to four workers and benchmark the actual image and multiplexing
sizes before increasing the limit: worker-local high-resolution accumulators make
memory grow approximately linearly, and thread overhead can outweigh additional
parallelism. In a 20-sample default 32×32/64×64 development run, four workers
gave 1.89× ordinary-update speedup with 1.49 MiB peak incremental heap versus
0.06 MiB serial; eight workers improved only to 1.91× while using 2.26 MiB.
These values are illustrative rather than portable performance guarantees.

Checkpoints contain the spectrum, pupil, calibration variables, and full history.
Load one with `ReconstructionCheckpoint::load` and pass it to
`ReconstructionAlgorithm::run_from_checkpoint`; `iterations` remains the target
total iteration count rather than an additional count.
Checkpoint save/load validates format version, finite state, auxiliary-state
consistency, and monotonic history before touching reconstruction state. Use
`ReconstructionCheckpoint::load_for_problem` to additionally verify all array and
calibration dimensions against the intended problem at file-loading time.

`ReconstructionResult::save_bundle` and `load_bundle` persist the complete final
result—including complex object and spectrum, amplitude, phase, pupil,
calibration, diagnostics, history, runtime, and metadata—in a validated,
versioned JSON file. Component-level PNG, JSON, and CSV writers remain available.

Periodic callbacks request diagnostics only at active hook points. For example,
`SaveImageEvery::new(10, ...)` performs the object inverse FFT only on iterations
10, 20, and so on. Custom callbacks can implement `requires_for` to use the same
behavior while retaining `requires` for general capability introspection.
`SaveResidualsEvery` similarly performs calibrated forward predictions only at
its configured frequency and writes one mask-aware, zero-centered signed PNG per
measurement frame.
When frame callbacks are enabled, `on_frame_end` runs exactly once per completed
frame. Multi-frame batches expose their shared post-batch state, the individual
frame index and loss, and the containing batch index.

Frame schedules include sequential, brightfield-first, spiral-out, seeded random,
and measurement-aware SNR ordering. `FrameSchedule::SnrWeighted` processes the
highest empirical shot-noise SNR first, using frame weights, masks, and known
background. Its score is a measured-intensity proxy rather than a calibrated
camera-noise model. `Runner` supplies the required measurements automatically;
model-only calls to `FrameSchedule::order` retain sequential order for this mode.

Geometric simulation mismatch is represented with separate concrete source
descriptions. Compile the actual source geometry into the model passed to
`Simulator::new`, compile the assumed geometry into the model passed to
`reconstruction_model`, and let each `IlluminationSource` apply its own physical
parameters. `IlluminationAcquisitionErrors` is limited to acquisition effects:
relative frame-gain variation, failed frames, and source permutations. Failed
frames remain in the stack as dark frames with zero reconstruction weight,
preserving frame/model indexing. `CameraModel` owns pixel sensitivity, photon and
electron conversion, shot and read noise, dark current, gain, offset, quantization,
saturation, and deterministic bad pixels. Defocus distance, pupil aberration, and
edge apodization are compiled through `Optics` into the model pupil. Aberration
mismatch is represented by compiling separate true and reconstruction models.
Illumination-angle transmission belongs in source intensity weights, ordinary
frame gains, or multiplexed source weights rather than in the pupil model.
The fixed-sphere placement/pose model and moving-arm kinematic error model are
documented in [Spherical illumination geometries](docs/spherical-geometries.md).
`SimulationResult` retains both compiled optical models and the acquisition
configuration. Use `compare_with_problem` to add masked, normalized per-frame
intensity residuals to the amplitude, phase, complex-field, Fourier, and pupil
ground-truth metrics. When a true model is supplied, it also reports
source-position RMSE in Fourier-grid pixels using the recovered illumination
corrections.

`Simulator::simulate` compiles known linear camera response into the returned
reconstruction model: frame gains include photon conversion and electronic gain,
while background includes dark current and offset. Digitized simulated counts can
therefore be passed directly to `ReconstructionProblem`; clipping, quantization,
noise, pixel-response variation, and bad pixels remain deliberate
non-linear/model-mismatch effects. Serialized `SimulationConfiguration` keeps
`compiled_models.reconstruction_model` as the strict optical model; use
`reconstruction_model_for_counts()` when constructing a problem from detector
counts loaded through that configuration path.
Optical background belongs in the true and reconstruction models; detector dark
current and electronic offset belong in `CameraModel`.

Pupil metrics remove the best global complex scale because object and pupil share
that ambiguity. Synthetic objects include amplitude-only and phase-only arrays,
mixed test patterns, resolution targets, particles, random phase, Siemens stars,
and seeded biological-like phase/absorption fields.

Epry can optionally recover relative per-frame gains with
`recover_frame_gains(true)`. Updates use a bounded, damped least-squares estimate
over unmasked pixels; `gain_step` and `gain_bounds` control stability. Ground-truth
gain error removes the global object/gain scale ambiguity, and recovered values are
included in results and resumable checkpoints.

Epry also supports `recover_background(true)` for additive per-frame offsets.
`background_step` damps the residual-mean update and `background_bounds` prevents
unstable excursions. Existing spatial background maps are preserved and expanded
per frame only when corrections are applied. Absolute common background remains
ambiguous with the object's DC intensity, so calibration should be interpreted
relative to a reference frame or constrained with known dark measurements.

`MeasurementStack::from_image_files` loads ordered single-channel 8-bit or 16-bit
PNG/TIFF frames and preserves native detector counts; each listed file contributes
one frame and autogenerated metadata records its path. Use `from_tiff_stack` when
each page of one multipage TIFF is a measurement frame.

`LazyMeasurementStack::from_image_files` validates headers up front and uses a
bounded, thread-safe LRU of decoded frames (one frame by default).
`from_tiff_stack` treats each multipage TIFF page as a lazy frame, and
`from_manifest` keeps the manifest's measurement frames lazy while applying
configured dark, background, flat-field, exposure, and negative-clamping
corrections as each frame is decoded. It can be passed directly to
`ReconstructionProblem::new` or converted to already-processed resident storage
with `materialize`. `with_cache_capacity` limits retained frame count;
`with_cache_byte_capacity` imposes a simultaneous decoded-pixel byte limit.
`cached_frame_count` and `cached_byte_count` report the retained cache contents.
Correction images, masks, and metadata remain resident, and byte accounting does
not include frame handles retained by callers after cache eviction.

`MeasurementSpec` represents the strict JSON manifest. Its ordered `FrameSpec`
entries contain paths, illumination indices, exposures, weights, and labels;
optional dark, flat, background, and mask images use `ImageSet` where applicable.
`PreprocessingConfig` holds the preprocessing flags. `MeasurementStack::from_manifest`
loads the specification, resolving relative paths against the manifest file.
Backgrounds and masks may be a single broadcast path or an array containing one
path per frame. The preprocessing flags configure the returned stack;
call `apply_preprocessing()` explicitly to transform detector counts.
See `cargo run --example load_measurement_manifest -- measurements.json` for the
minimal loading workflow.

`SyntheticObject::from_amplitude_image` and
`from_amplitude_phase_images` instead normalize grayscale values to physical
amplitude and a user-selected phase range. Empty stacks, color images, mismatched
dimensions, and invalid phase ranges return typed errors.

## Supported toolchains

The MSRV is Rust 1.97. Stable Rust is tested on Linux, macOS, and Windows;
Python bindings support CPython 3.13 and 3.14. See the
[configuration schema](docs/configuration-schema.md) and
[API reference](docs/reference/index.md).

## Python bindings

The `fpm_rs` Python package requires Python 3.13 or newer and is built with
PyO3/maturin. The Python layer exposes concrete configuration classes and NumPy
arrays; simulation and reconstruction continue to execute in the Rust core and
release the GIL. Image-backed synthetic-object loading, model compilation,
checkpoint loading/saving, and diagnostic JSON export also detach while their
Rust-owned work runs. NumPy measurement input is copied once while holding the
GIL because the source buffer remains Python-owned; result arrays are likewise
created with the GIL held, then reused by subsequent property access. Python
iteration callbacks reacquire it only for the callback invocation.

Install the latest released package from [PyPI](https://pypi.org/project/fpm-rs/):

```sh
python -m pip install fpm-rs
```

For inline plotting and the tutorial notebooks, install the extra instead:

```sh
python -m pip install "fpm-rs[notebook]"
```

The PyPI wheels include the Rust extension, so a Rust toolchain is not needed
when a compatible wheel is available. See the [installation guide](docs/getting-started/installation.md)
for source builds and platform details.

### Development from a checkout

To contribute or run the in-tree test suite, create the Python 3.13 development
environment and run its tests with:

```sh
pixi install -e py313
pixi run -e py313 python-test
```

Launch Python or run a script through the build-aware task:

```sh
pixi run -e py313 python
pixi run -e py313 python path/to/script.py
```

The task builds the native extension in place before starting Python. The pixi
activation environment also places the in-tree `python/` package on
`PYTHONPATH`, including for Jupyter.

Python 3.14 has a separate validation environment:

```sh
pixi install -e py314
pixi run -e py314 python-test
```

A minimal API flow is:

```python
import numpy as np
import fpm_rs as fpm

optics = fpm.Optics(532e-9, 0.10, 4.0, 6.5e-6)
leds = fpm.LEDArray((3, 3), 4e-3, 90e-3, (1.0, 1.0))
model = fpm.compile_model(optics, leds, (32, 32), (64, 64))
simulation = fpm.simulate(model, np.ones((64, 64), dtype=np.complex128))
problem = fpm.ReconstructionProblem(
    simulation.measurements,
    simulation.reconstruction_model,
)
result = fpm.AlternatingProjection(iterations=20).run(problem)
```

See `docs/tutorials/notebooks/quickstart.ipynb` for a notebook workflow,
`docs/tutorials/notebooks/synthetic_objects_quickstart.ipynb` for a gallery of the
Rust-backed synthetic object constructors, and `python/fpm_rs/__init__.pyi`
for the typed public surface. The notebooks use the released `fpm-rs` PyPI
package; install the `notebook` extra before opening them outside a checkout.

## License

Licensed under the [MIT License](LICENSE-MIT).
