# fpm-rs

`fpm-rs` is a CPU reconstruction and simulation library for **image-plane Fourier
ptychographic microscopy**. Measurements are real-space intensity images acquired
under different illumination angles; diffraction-plane ptychography is outside
the crate's scope.

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
    defocus: None,
    initial_pupil_aberration: None,
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
- `dkx = 2π / (width * object-plane pixel size)` and similarly for `dky`.
- CPU FFTs normalize the forward transform by `1/N`; inverse transforms are
  unnormalized. This preserves constant-object amplitude when a high-resolution
  spectrum is cropped and inverse-transformed on a smaller grid.
- The ideal pupil includes samples with transverse frequency
  `sqrt(kx² + ky²) <= 2π NA / λ`.

`CoordinateConvention::CenteredPositiveK` records these choices in compiled
models. Low-level model construction remains available for imported k-vectors,
simulations, and algorithm tests. `ImagePlaneModel::new` defaults to integer
crops; use `with_subpixel_offsets` when low-level inputs include calibrated
fractional source positions.

## Modules

- `experiment`: optional `Optics`, `LEDArray`, angle and direct-k descriptions.
- `model`: sampling, pupil, Fourier crops, compiled model, shared forward model.
- `measurements`: image-plane intensity stacks and preprocessing.
- `algorithms`: alternating projection, Fpie, Epry, linearized ADMM, and
  loss-gradient reconstruction.
- `reconstruction`: problem/state/result, schedules, batches, and runner.
- `callbacks`: image output, CSV logging, checkpoints, progress, early stopping.
- `simulation`: synthetic objects, aberrations, source mismatch, camera/noise,
  and ground-truth metrics.
- `backend`: a small backend boundary and cached-plan `rustfft` CPU backend.

The flat-buffer core and backend boundary allow later GPU-resident state and PyO3
bindings without putting experimental metadata into algorithm update rules.

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
available parallelism for optical prediction, then applies camera effects and
seeded noise serially so reproducibility does not depend on worker count.

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

Simulation mismatch options include array shift/rotation/scale, per-source jitter,
gain variation, missing sources, and source-order permutations. Missing sources
remain in the stack as dark frames with zero reconstruction weight, preserving
frame/model indexing. Camera simulation supports photon conversion, read noise,
dark current, gain, offset, quantization, saturation, and deterministic bad pixels.
`AberrationModel` supports defocus, astigmatism, coma, spherical aberration,
pupil-edge apodization, and illumination-angle vignetting. Vignetting attenuates
ordinary frame gains or individual multiplexed source weights, leaving the
reconstruction model unchanged for calibration experiments.
`SimulationResult` retains all of these configuration objects. Use
`compare_with_problem` to add masked, normalized per-frame intensity residuals to
the amplitude, phase, complex-field, Fourier, and pupil ground-truth metrics. When
a true model is supplied, it also reports source-position RMSE in Fourier-grid
pixels using the recovered illumination corrections.

Known linear camera response is compiled into the returned reconstruction model:
frame gains include photon conversion and electronic gain, while background
includes dark current and offset. Digitized counts can therefore be passed
directly to `ReconstructionProblem`; clipping, quantization, noise, flat-field
errors, and bad pixels remain deliberate non-linear/model-mismatch effects.

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

`MeasurementStack::from_manifest` loads a strict JSON manifest with ordered frame
paths, illumination indices, exposures, weights, labels, and optional dark, flat,
background, and mask images. Relative paths resolve against the manifest file.
Backgrounds and masks may be a single broadcast path or an array containing one
path per frame. The manifest's preprocessing flags configure the returned stack;
call `apply_preprocessing()` explicitly to transform detector counts.
See `cargo run --example load_measurement_manifest -- measurements.json` for the
minimal loading workflow.

`SyntheticObject::from_amplitude_image` and
`from_amplitude_phase_images` instead normalize grayscale values to physical
amplitude and a user-selected phase range. Empty stacks, color images, mismatched
dimensions, and invalid phase ranges return typed errors.
