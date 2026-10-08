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

## Reconstruct multiple wavelengths

`SpectralImagePlaneModel` composes ordinary wavelength-specific kernels into
one sparse detector acquisition. Use `SpectralAlternatingProjection` when the
object varies by wavelength or a detector exposure combines multiple narrowband
channels. `SpectralReconstructionProblem` accepts the same scalar grayscale
measurement shape as ordinary reconstruction; the spectral plan specifies
channel membership explicitly.

This example is a small synthetic sanity check. `SpectralChannel` supplies each
channel's optics, source powers, and ordinary local acquisition rows.
`SpectralGeometry.shared` resolves physical directions separately at each
wavelength. `SpectralAcquisitionPlan.separate` places those local frames into
channel-major detector order. Replace the generated measurements with your
calibrated stack to reconstruct an experiment.

```python
import numpy as np
import fpm_rs as fpm

geometry = fpm.DirectionList.from_direction_cosines(
    values=np.array([[0.0, 0.0], [0.04, 0.0], [0.0, 0.04]], dtype=np.float64),
)
channels = [
    fpm.SpectralChannel(
        channel_id=channel_id,
        optics=fpm.Optics(
            wavelength_vacuum_m=wavelength,
            objective_na=0.10,
            magnification=4.0,
            camera_pixel_size=6.5e-6,
        ),
        calibration=fpm.SourceCalibration.unity(),
        acquisition=fpm.AcquisitionPlan.all_sources(source_count=3),
    )
    for channel_id, wavelength in [("blue", 450e-9), ("red", 630e-9)]
]
model = fpm.compile_spectral_model(
    channels=channels,
    geometry=fpm.SpectralGeometry.shared(geometry=geometry),
    acquisition=fpm.SpectralAcquisitionPlan.separate(frame_counts=[3, 3]),
    image_shape=(16, 16),
    object_coupling="independent",
)
objects = np.ones((2, *model.reconstruction_shape), dtype=np.complex128)
# Match the core's normalized forward FFT and centered spectrum convention.
spectra = np.ascontiguousarray(
    np.fft.fftshift(np.fft.fft2(objects), axes=(-2, -1))
    / np.prod(model.reconstruction_shape)
)
measurements = model.forward_intensities(object_spectra=spectra)
problem = fpm.SpectralReconstructionProblem(measurements=measurements, model=model)
result = fpm.SpectralAlternatingProjection(iterations=10).run(problem=problem)
print(result.channel_ids, result.amplitude.shape)
```

For multiplexed exposures, replace the plan with explicit sparse rows:

```python
acquisition = fpm.SpectralAcquisitionPlan.multiplexed(
    frames=[
        fpm.SpectralFrame(
            contributions=[(0, local_frame, 1.0), (1, local_frame, 0.7)],
            gain=1.2,
            background=0.01,
        )
        for local_frame in range(3)
    ],
)
```

Each triple is `(channel_index, local_frame_index, spectral_weight)`, with an
intensity weight. Recompile the model with this plan and use measurements from
the corresponding mixed exposures. The prediction is
`gain * sum(spectral_weight * channel_local_intensity) + background`. Local
intensity already includes source powers, local source weights, and local
frame gain. Detector gain and uniform background apply once after summing.

Independent complex objects are the default. `object_coupling="shared_complex"`
explicitly assumes one wavelength-independent complex transmission, stores one
object, and accumulates every channel correction into it. Its `initial_objects`
input has shape `(1, height, width)`; independent inputs have one plane per
channel. Returned arrays always have channel first, so shared results repeat
the identical final object in every plane. Returned pupils remain distinct.

All channels require the same detector shape and object-plane pixel pitch;
automatic sizing uses the union of every channel's Fourier crop bounds. Shared
direct `KVectorList` geometry, registration/resampling, finite spectral
bandwidth, pupil recovery, and physical calibration are excluded. The current
spectral workflow uses separately tagged checkpoints/result bundles and the
explicit version-2 spectral dataset profile described below.

The solver streams physical exposures and evaluates all their coherent modes
before any update, applies one amplitude ratio, then uses the canonical crop
adjoints with normalized mode weights. Batching preserves exact exposure
order. `frame_order` accepts a detector permutation; `seed` selects a
deterministic per-pass shuffle. Default separate independent runs match
ordinary per-channel alternating projection under equivalent schedules. Align
each independent channel's global phase piston separately when comparing
ground truth; a shared object has one piston.

Implementation: `src/experiment/spectral.rs`, `src/model/spectral.rs`,
`src/reconstruction/spectral.rs`, and `src/algorithms/spectral.rs`. This typed
narrowband composition generalizes the state-decomposition idea of S. Dong,
R. Shiradkar, P. Nanda, and G. Zheng,
[“Spectral multiplexing and coherent-state decomposition in Fourier ptychographic imaging”](https://doi.org/10.1364/BOE.5.001757),
*Biomedical Optics Express* **5**(6), 1757–1767 (2014). See the
[spectral API](../reference/python/algorithms.md#narrowband-spectral-reconstruction)
for complete signatures and array contracts.

### Unwrap OPD across wavelengths

Use `SyntheticWavelengthUnwrapper` when the wavelength phases describe one
nondispersive optical path difference (OPD) `d`, with transmission phase
`wrap(2π d / λ)`. The shared quantity is OPD; select independent complex objects
for reconstruction. `shared_complex` imposes the same phase at every wavelength
and is rejected by the OPD workflow.

Subtracting the longer-wavelength phase from the shorter-wavelength phase gives
a synthetic period `Λ = λ_short λ_long / (λ_long - λ_short)`. The longest beat
selects the coarse OPD branch, then all shorter difference periods and original
wavelengths refine fringe orders by rounding. The final map fits the unwrapped
original-channel phases by least squares with equal phase weights. This adapts
the phase-difference hierarchy of S. K. Mirsky and N. T. Shaked,
[“Six-pack holography for dynamic profiling of thick and extended objects by simultaneous three-wavelength phase unwrapping with doubled field of view”](https://doi.org/10.1038/s41598-023-45237-6),
*Scientific Reports* **13**, article 19293 (2023), to reconstructed FPM fields.
Implementation: `src/reconstruction/opd.rs`; public APIs:
`SyntheticWavelengthUnwrapper.unwrap_fields`,
`SpectralReconstructionResult.unwrap_opd`, and
`SpectralAlternatingProjection.run_opd` (Rust: `SpectralRunner::run_opd`).
We use an explicit OPD interval instead of spatially unwrapping the longest
synthetic phase and omit phase-sum synthetic wavelengths and holographic optics.

The following analytic sanity check recovers discontinuous signed OPDs exceeding
either original wavelength. The two known channel pistons are removed before
mixing; explicit zero offsets are appropriate only for already referenced fields.

```python
wavelengths = np.array([500e-9, 550e-9])
expected_opd_m = np.array([[-1.6e-6, 2.8e-6, 0.0], [1.2e-6, -0.4e-6, 0.71e-6]])
pistons_rad = np.array([1.1, -2.3])
fields = np.ascontiguousarray(
    np.exp(1j * (
        2 * np.pi * expected_opd_m[None] / wavelengths[:, None, None]
        + pistons_rad[:, None, None]
    ))
)
unwrapper = fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-2e-6, 3e-6))
opd = unwrapper.unwrap_fields(
    fields=fields,
    wavelengths_vacuum_m=wavelengths,
    phase_offsets_rad=pistons_rad,
)
assert np.all(opd.valid_mask)
np.testing.assert_allclose(opd.opd_m, expected_opd_m, atol=1e-18)
```

For FPM data, the phase pistons are generally unknown. Select a spatial region
whose OPD is constant and known, using a C-contiguous uint8 `reference_mask`.
Each channel's piston is estimated from the circular mean of unit phasors on
the reference pixels valid in every channel. Supply either this mask or
`phase_offsets_rad`, never both. The interval is half-open, must contain the
sample's OPDs, and its width must not exceed the longest synthetic period.
Without a calibrated reference and a known branch, intensities cannot determine
absolute OPD.

This one-call example uses the **separate, uniform-object problem above**, where
the whole field is a known zero-OPD reference. For a real specimen, replace the
reference mask with its constant-known-OPD region and choose justified bounds.

```python
reference_mask = np.ones(model.reconstruction_shape, dtype=np.uint8)
unwrapper = fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-0.5e-6, 1e-6))
combined = fpm.SpectralAlternatingProjection(iterations=10).run_opd(
    problem=problem,
    unwrapper=unwrapper,
    reference_mask=reference_mask,
    reference_opd_m=0.0,
)
opd_m = combined.opd.opd_m
valid = combined.opd.valid_mask != 0
print(opd_m[valid].mean(), combined.spectral.completed_iterations)
```

The primitive operations are `algorithm.run(problem=problem)` followed by
`spectral_result.unwrap_opd(unwrapper=unwrapper, reference_mask=reference_mask)`.
`run_opd` returns both the original spectral result and the phase-mixed OPD
diagnostics; it does not jointly optimize an OPD field against intensities.

All fields must already be spatially registered with matched effective
resolution. A common array grid alone does not establish equal resolution.
Registration, resolution matching, dispersion correction, and automatic branch
selection are not performed. The nondispersive assumption is about OPD, not
thickness; converting OPD to thickness requires a refractive-index contrast.

Long beat periods amplify OPD noise. At each refinement, coarse OPD error must
remain below half the next period for correct fringe rounding; additional
wavelengths can supply intermediate periods. The result reports `fringe_orders`,
applied `phase_offsets_rad`, the `wavelength_ladder_m`, and RMS original-channel
wrapped `phase_residual_rad`. An optional explicit `max_phase_residual_rad`
rejects inconsistent fits, but a small residual cannot certify the correct
fringe branch under large noise. `minimum_amplitude` excludes weak/zero fields;
an optional common-grid uint8 `mask` excludes pixels before reference estimation
and mixing. Invalid pixels have zero `valid_mask`, NaN OPD/residuals, and zero
orders. Inspect that mask when using or plotting the map.

### Jointly fit OPD to detector intensities

`MultiWavelengthGradientDescent` directly optimizes a shared nondispersive OPD
map `d` and a separate positive amplitude map `A_c` for every channel. Each
forward evaluation builds `O_c = A_c exp(2π i d / λ_c)` and uses the compiled
spectral kernels to fit **all** scalar detector exposures, including source and
wavelength intensity mixtures. Channel phases are constrained throughout the
fit. Use it when the common-OPD assumption is justified and intensity data should
refine the phase-mixed estimate; retain `SpectralAlternatingProjection` for
wavelength fields whose phases need not describe one nondispersive OPD.

By default, `.run` reconstructs independent fields with spectral AP, unwraps
them with `SyntheticWavelengthUnwrapper`, then jointly refines amplitudes and
OPD. Every initial unwrapped pixel must be valid; an incomplete initialization
returns an error. Pass both `initial_opd_m` and `initial_amplitudes` to bypass
AP/unwrapping with an externally known branch (Rust: `run_from_opd`). Those
arrays must be finite, C-contiguous float64 on the common grid, with amplitudes
stacked in model channel order. In this explicit-start path only the unwrapper's
OPD bounds apply; its phase/residual/amplitude thresholds do not run, and the
interval can exceed the longest synthetic period.

The following modifiable sanity check uses the two-channel model constructed
above. It synthesizes wavelength fields from an OPD spanning several original
phase cycles, then deliberately perturbs its OPD and amplitudes before fitting.
The reference pixel has a known OPD. Replace the synthetic data and starting
arrays with your calibrated measurements and estimated branch; omit both
starting arrays to use automatic spectral AP and phase mixing instead.

```python
rows, columns = np.indices(model.reconstruction_shape)
x = 2 * np.pi * columns / model.reconstruction_shape[1]
y = 2 * np.pi * rows / model.reconstruction_shape[0]
expected_opd_m = 1.2e-6 + 10e-9 * np.sin(x) + 5e-9 * np.sin(y)
amplitudes = np.ascontiguousarray([0.8 + 0.03 * np.cos(x + y)] * 2)
wavelengths = np.array(model.wavelengths_vacuum_m)
objects = amplitudes * np.exp(
    2j * np.pi * expected_opd_m[None] / wavelengths[:, None, None]
)
spectra = np.ascontiguousarray(
    np.fft.fftshift(np.fft.fft2(objects), axes=(-2, -1))
    / np.prod(model.reconstruction_shape)
)
problem = fpm.SpectralReconstructionProblem(
    measurements=model.forward_intensities(object_spectra=spectra), model=model,
)
reference_mask = np.zeros(model.reconstruction_shape, dtype=np.uint8)
reference_mask[0, 0] = 1
joint = fpm.MultiWavelengthGradientDescent(iterations=100).run(
    problem=problem,
    unwrapper=fpm.SyntheticWavelengthUnwrapper(opd_range_m=(0.0, 1.5e-6)),
    reference_mask=reference_mask,
    reference_opd_m=float(expected_opd_m[0, 0]),
    initial_opd_m=np.ascontiguousarray(expected_opd_m + 8e-9 * np.sin(x + y)),
    initial_amplitudes=np.ascontiguousarray(1.05 * amplitudes),
)
assert joint.spectral.trace[-1][1] < joint.spectral.trace[0][1]
print(joint.completed_iterations, joint.stopped_early, joint.opd_m.mean())
```

`MultiWavelengthSolverResult.opd_m` is the optimized shared parameter.
`.spectral` contains its derived channel fields, amplitudes, fixed pupils, and
the joint objective trace. Iteration zero records the constrained starting
objective; subsequent entries record post-update full-data objectives.
`initialization_opd` and `initialization_trace` retain pre-fit phase mixing and
AP diagnostics for automatic starts, and are `None` for explicit starts.
Runtime seconds cover joint refinement; the initialization trace times the AP
stage separately. Returned scientific arrays are writable copies.

The objective is the frame-weighted mean of valid detector-pixel
`(sqrt(prediction + epsilon) - sqrt(max(measurement, 0) + epsilon))**2`, including
known gain/background. Analytic adjoints use the canonical Fourier crop and
both FFT normalization factors. The real chain rule is
`g_Ac = 2 Re(exp(-i φ_c) g_Oc)` and
`g_q = sum_c 2 (λ_min / λ_c) Im(conj(O_c) g_Oc)`, where
`q = 2π d / λ_min` and `g_Oc` is the complex Wirtinger gradient. Both real
gradients are multiplied by the common-grid pixel count, with separate
`amplitude_step` and `opd_step` multipliers. A common backtracking factor is
halved until the full objective does not increase. This is a full-data solver;
ordinary frame schedules and batch sizes do not apply.

Amplitudes are projected to at least `sqrt(epsilon)` and OPD to `[lower, upper)`.
A constant-known-OPD reference mask fixes those pixels throughout fitting.
With explicit channel pistons, the initialized spatial mean OPD fixes the gauge
instead; an explicit unwrapped start is already referenced, so the supplied
pistons validate that choice without subtracting another phase. Projection onto
the bounds and fixed mean uses a common shift followed by clipping. When no
acceptable backtracking step is found, the solver retains the last accepted
state and returns `stopped_early=True`. Always inspect the objective and this
flag. Initialization remains critical for this nonconvex problem; an incorrect
fringe branch can survive a good intensity fit.

Implementation: `src/algorithms/multi_wavelength.rs` and
`python/src/multi_wavelength.rs`. The loss-gradient viewpoint follows
L. Bian, J. Suo, G. Zheng, K. Guo, F. Chen, and Q. Dai,
[“Fourier ptychographic reconstruction using Wirtinger flow optimization”](https://doi.org/10.1364/OE.23.004856),
*Optics Express* **23**(4), 4856–4866 (2015). The shared-OPD chain rule, smoothed
amplitude loss, box/gauge projections, and monotone full-data backtracking are
implementation extensions; this does not reproduce that publication's solver.
The initialization hierarchy uses the Mirsky–Shaked reference above. Nondispersive
OPD, registered common grids, matched effective resolution, and a calibrated
reference remain assumptions. Pupil/calibration/dispersion recovery and ordinary
checkpoints use a separate spectral format described below.

### Persist and resume spectral solvers

Both solvers return a final `SpectralReconstructionCheckpoint`. Creating it
requires one additional streaming pass over detector values, masks and weights
for a SHA-256 fingerprint; no full data copy is needed. The snapshot stores the
ordered compiled kernels, sparse detector plan, fixed pupils/calibration,
stepping options and trace. Spectral AP retains centered spectra and schedule;
joint descent retains authoritative OPD, amplitudes, bounds, fixed gauge and
automatic initialization records. Ordinary checkpoint format 2 is independent
of spectral checkpoint format 1.

```python
partial = fpm.SpectralAlternatingProjection(iterations=20).run(
    problem=problem, seed=7,
    checkpoint_directory="spectral-checkpoints", checkpoint_every=5,
)
checkpoint = partial.checkpoint
checkpoint.save(path="spectral-final.json")
loaded = fpm.SpectralReconstructionCheckpoint.load(path="spectral-final.json")
continued = fpm.SpectralAlternatingProjection(iterations=50).run(
    problem=problem, resume=loaded,
)

joint.checkpoint.save(path="joint-final.json")
continued_joint = fpm.MultiWavelengthGradientDescent(iterations=150).run_from_checkpoint(
    problem=problem, checkpoint=joint.checkpoint,
)
```

The iteration target includes saved iterations and may increase; every other
stepping option must match. Omitted AP ordering inherits the saved schedule.
Explicit ordering must match. Changed channel order, wavelengths, grid,
coupling, pupil, calibration, detector data, masks or weights fail validation.
AP restores centered spectra directly; joint resume does not rerun AP/unwrapping
or project the saved gauge. Scientific arrays/objectives match an uninterrupted
run exactly; accumulated wall times naturally differ. Checkpoints are written
at complete AP passes or accepted joint updates, periodically and at the final
boundary. Initial guesses cannot accompany an AP resume. In Rust use
`SpectralRunner::with_checkpoint`/`with_checkpoint_options`, or joint
`run_from_checkpoint`/`run_with_options`.

```python
bundle = continued.write_bundle(path="spectral-result", label="three channels")
# For joint OPD, write the enclosing result to preserve its authoritative map.
joint_bundle = continued_joint.write_bundle(path="joint-result")
reopened = fpm.read_spectral_bundle(joint_bundle.path)
reopened.verify()
restored_joint = reopened.joint_result
print(restored_joint.opd_m.mean(), reopened.checkpoint.completed_iterations)
```

Spectral bundle format 1 has its own manifest kind. It stores lossless
per-channel object, spectrum, amplitude, phase and pupil/support NPY files,
Parquet channel/history/metric tables, and the full resumable state. Joint
bundles retain OPD and initialization records; writing a post-AP
`MultiWavelengthReconstructionResult` also retains unwrapping diagnostics,
including invalid NaN pixels. `read_spectral_bundle` rejects ordinary bundles,
and `read_bundle` rejects spectral bundles. Artifact hashes, sizes, safe paths,
dtypes/shapes and state consistency are checked when loaded; `verify` checks
every artifact and table. Metadata loads eagerly; channel arrays load together
on first result access and are cached in Rust. Python result getters return
independent writable copies. Spectral bundles currently omit PNG previews.

### Inspect declared mixing and benchmark recovery

```python
diagnostic = model.mixing_diagnostics()
print(diagnostic.rank, diagnostic.condition_number, diagnostic.channel_ids)
```

The matrix is shaped `(detector exposures, channels)`. Each entry sums detector
gain × spectral weight × local gain × local incoherent source-weight sum.
Background and measurement masks/weights are excluded. Rank counts singular
values strictly above the largest value times the relative tolerance, default
`max(exposures, channels) * float64 epsilon`, following NumPy's
[`matrix_rank` documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html).
`src/model/mixing.rs` implements a scaled cyclic one-sided Jacobi SVD, following
the column-orthogonalization formulation in LAPACK 3.12.1's
[DGESVJ documentation](https://www.netlib.org/lapack/explore-html/d9/deb/group__gesvj_ga7aec05d2a1523bbeee77ece21b12187c.html).
Condition is `None` for deficient column rank or an overflowing ratio. These
weights do not include differing Fourier crops/pupils, so linear rank does not
prove nonlinear recovery. Exposure noise and spatial calibration still matter.

Run the offline benchmark with a pass sweep and a common runtime ceiling:

```sh
pixi run python python/examples/benchmark_multi_wavelength.py \
  --passes 10 40 160 --repeats 3 --runtime-budget-s 2 \
  --output spectral-benchmark.json
```

The fixture has registered nondispersive OPD, three calibrated wavelengths,
27 separate or coded exposures, and unit total spectral weight per exposure.
AP and joint descent receive identical perturbed OPD/amplitude fields. It
reports per-channel piston-aligned relative field error, referenced OPD RMSE,
valid fraction, noisy/clean smoothed amplitude MSE, runtime, peak/baseline
process RSS, and failure rates across repeated noise seeds. Separate AP is also
compared against ordinary per-channel AP with both using the same default
weighted measured-amplitude initialization. Peak RSS is
measured in fresh processes at solver completion, before ordinary parity runs.
It includes the interpreter, fixture and result/checkpoint storage; baseline
RSS is reported rather than claiming solver-only memory.

Cross-talk is defined here as the largest absolute projection coefficient of
one channel's amplitude error onto another channel's mean-subtracted true
amplitude contrast. Noise is additive Gaussian detector noise with sigma
relative to the global mean clean intensity, followed by nonnegative clipping.
Failure means an exception, invalid OPD pixels, or OPD RMSE above the configurable
`--failure-opd-nm` threshold (100 nm is a fixture criterion). Fixed-pass results
report actual completed passes and contribution evaluations. Runtime comparisons
select the best sampled intensity fit under the same ceiling; they are discrete
budget comparisons, not identical wall times. These small synthetic cases do
not establish performance on experimental data, registration mismatch,
dispersion or blind pupil/calibration recovery.

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
| Separate or multiplexed narrowband wavelengths with trusted fixed channel pupils | `SpectralAlternatingProjection` | Use a compiled `SpectralImagePlaneModel` and scalar detector frames. Choose independent transmission fields or explicitly shared complex transmission; channel calibration stays fixed. |
| Registered wavelength phases need a larger unambiguous OPD interval | `SyntheticWavelengthUnwrapper`, or `SpectralAlternatingProjection.run_opd` | Require nondispersive OPD, matched resolution, an explicit phase reference, and known OPD bounds. Mix independent channel phases through a synthetic-wavelength hierarchy. |
| All wavelength measurements should refine one shared OPD map | `MultiWavelengthGradientDescent` | Require nondispersive OPD, trusted fixed channel pupils/calibration, matched resolution, a reference, and a valid initial fringe branch. Jointly fit channel amplitudes and common OPD with full-data backtracking. |
| Noisy data with a trusted fixed pupil and measurement-response model | `AdaptiveAlternatingProjection` | Retain AP's fast initial unit-step progress, then reduce the object step when pass-to-pass amplitude loss stops improving materially. Use fixed-step AP when exact manual control of every pass is preferable. |
| Object-only recovery with weak pupil transfer or mild noise | `Fpie` | Prefer its stabilized object update when plain projection is too sensitive in weak-transfer regions. It does not estimate the pupil or source geometry. |
| A fixed-pupil `Fpie` trajectory converges slowly or stagnates | `Mpie` | Add periodic object-spectrum momentum after establishing a stable rPIE baseline. It adds interval, friction, and feedback tuning and cannot be wrapped by `JointReconstruction`. |
| Noise statistics, outliers, or an object prior must enter the update | `GradientDescent` | Select Poisson, Huber-amplitude, intensity, or amplitude loss as appropriate. For sparse gross outliers with a Poisson model, enable `poisson_truncation_threshold`; use total variation only when that prior is defensible. This is the most configurable route, with more tuning and compute. |
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

`AlternatingProjection` is the simplest starting point.
`AdaptiveAlternatingProjection` adds pass-level noise-robust step feedback,
`Fpie` adds regularized object updates, `Mpie` adds periodic object-spectrum
momentum to that fixed-pupil update, `Epry` can recover the pupil and frame
response, `Admm` separates data fitting from overlap consensus,
`GradientDescent` supports generic Fourier-grid source correction, pupil
recovery, and regularization. Physical planar-array calibration is the separate
`JointReconstruction` workflow below.

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

`AdaptiveAlternatingProjection` uses the same object-only, fixed-pupil
amplitude projection as `AlternatingProjection`, with one step shared by a
complete scheduled pass. The first pass establishes an objective baseline.
After two pass objectives are available, the next pass retains the current step
when the relative amplitude-MSE decrease is greater than `progress_threshold`
(default `0.01`); otherwise it multiplies the step by `reduction_factor`
(default `0.5`) down to `minimum_object_step` (default `0.001`). This is
feedback, not backtracking: an unsuccessful pass is neither retried nor rolled
back.

The feedback objective is the existing frame-weighted mean of mask-aware,
per-frame amplitude MSE after known gain and background. It is accumulated
during projection, so the implementation uses the paper's inexpensive
incremental approximation rather than an extra exact full-data evaluation.
Batch boundaries do not affect the controller or numerical path. A different
sequential, reverse, or seeded shuffled order can affect both intentionally.
Zero-weight frames are visited but contribute neither an update nor feedback.
The method keeps the pupil fixed and is rejected by `JointReconstruction`
because physical model recompilation would invalidate its objective history.

This rule follows C. Zuo, J. Sun, and Q. Chen,
[“Adaptive step-size strategy for noise-robust Fourier ptychographic
microscopy,”](https://doi.org/10.1364/OE.24.020724) *Optics Express* **24**(18),
20724–20744 (2016). Their convergence proof assumes convex component
objectives, whereas FPM phase retrieval is non-convex; fpm-rs therefore treats
the rule as an empirically motivated noise-robustness strategy, not a global
convergence guarantee. The paper also explores a pupil-recovery extension;
this API implements only its main fixed-pupil, object-only method.

`Admm` uses an amplitude proximal operator, a preconditioned linearized object
consensus update, and scaled dual variables. It honors masks, frame weights,
known gains and backgrounds, schedules, and batches. `penalty`, `object_step`,
and `dual_relaxation` control those updates. The default batch spans all frames.
For multiplexed data, its joint amplitude proximal operates across all source
modes, so checkpointed auxiliary and dual fields contain two complex values per
frame-source-mode pixel.

`Mpie` starts from the `Fpie` rPIE update and applies momentum after a configured
number of positive-weight measured frames. If `O_rpie` is the spectrum after
the current frame, `O_anchor` is the spectrum after the preceding momentum
event, and `V` is velocity, an event applies
`V = friction * V + (O_rpie - O_anchor)` followed by
`O = O_rpie + feedback * V`. A multiplexed frame counts once after all source
modes are inserted, zero-weight frames do not count, and a partial interval
crosses batch and iteration boundaries. `batch_size` therefore does not change
the numerical path. Defaults use an interval of 30, friction and feedback of
0.9, an object step of 0.2, and rPIE stability of 0.05; the deterministic CPU
benchmark also records a tuned interval-10, coefficient-0.7 case. Establish a
stable `Fpie` result before tuning these controls.

This is an object-only FPM adaptation of [A. Maiden, D. Johnson, and P. Li,
“Further improvements to the ptychographical iterative engine,” *Optica*
**4**(7), 736–745
(2017)](https://doi.org/10.1364/OPTICA.4.000736). Their work tested scanned
ptychography, applied momentum to both object and probe, and left Fourier
ptychography testing for future work. Equal friction and feedback reproduce
their object recurrence; fpm-rs exposes them separately and keeps the compiled
pupil fixed.

`GradientDescent` defaults to an image-amplitude residual. Its `loss_type` can
select amplitude MSE, intensity MSE, Poisson negative log likelihood, or robust
Huber amplitude loss. Losses are evaluated in intrinsic intensity units after
removing known linear gain and background, keeping the trajectory independent
of camera-count scaling. It supports masks, frame weights, schedules, and true
mini-batches: frame gradients accumulate in reusable storage and one averaged
object update is applied per batch. `object_step` controls that update.

With `loss_type="poisson_nll"`, setting
`poisson_truncation_threshold=25.0` enables the signal-dependent rejection rule
of L. Bian, J. Suo, J. Chung, X. Ou, C. Yang, F. Chen, and Q. Dai, [“Fourier
ptychographic reconstruction using Poisson maximum likelihood and truncated
Wirtinger gradient,” *Scientific Reports* **6**, 27384
(2016)](https://doi.org/10.1038/srep27384). For each mini-batch, fpm-rs forms a
frame-weighted mean absolute residual from positive-weight, unmasked pixels in
intrinsic intensity units after removing known gain and background. A pixel's
residual is retained according to that statistic, its predicted amplitude, and
the object's RMS amplitude. One decision applies to every mode of an
incoherently multiplexed detector pixel and to object, pupil, and illumination
updates. The trace still reports the full untruncated Poisson objective, while
`gradient_descent/retained_pixel_fraction` reports the selected fraction.

The cited method uses a full-data statistic, object-only recovery, and a
scheduled step. This implementation deliberately uses the current mini-batch,
the existing fixed `object_step`, and supports the solver's optional pupil and
illumination extensions. Batch size and schedule can therefore change both the
gate and the trajectory. Leave the threshold as `None` for the existing
untruncated Poisson gradient; values much below 25 can discard useful data,
while very large values approach the untruncated path.

The gradient implementation evaluates independent frame chunks on up to the
available CPU workers; `parallel_workers` limits the count. A truncating run
computes its batch statistic once before workers split the frames. Each worker
reuses local state, and the main thread reduces chunks in deterministic batch
order, including multiplexed shared-source curvature. Memory therefore grows
with active workers rather than batch length. TV and pupil smoothing run once
after the reduction for each batch.

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

### Blind pupil gauge

Pupil-recovering EPRY and gradient descent return one stable representative of
the coupled object/pupil solution. The compiled pupil supplies the reference
support, energy, and phase. After each iteration, the solver applies reciprocal
object/pupil corrections that preserve every modeled intensity, then fixes the
remaining object global phase. This happens after regularization and before
iteration callbacks, checkpoint capture, and result construction. A compatible
checkpoint is canonicalized before resumed work, so saved and uninterrupted
runs use the same convention.

The solver removes affine pupil phase separately on axes whose effective
subpixel offsets are zero. It retains the slope on an axis with any nonzero
offset because the bilinear fractional-crop operator does not preserve that
ambiguity exactly. The convention introduces no constructor option. See
[Blind object/pupil gauge](../concepts/core-concepts.md#blind-objectpupil-gauge)
for the transformations, failure behavior, and primary reference.

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

## Bright-field planar-array initialization

`BrightfieldCircleInitializer` is an optional physical warm start for a
`PlanarLEDArray`. It is separate from reconstruction: the initializer makes two
streaming passes over eligible bright-field frames, detects circular pupil
edges in their centered intensity spectra, fits the accepted `(NA_x, NA_y)`
centers directly to the canonical planar-array geometry, and returns both a
normal `Illumination` and an atomically refreshed `ImagePlaneModel`. Use that
model for reconstruction as-is or pass the returned physical illumination to
`JointReconstruction` for measurement-loss refinement.

The method assumes monochromatic coherent image-plane imaging, a thin specimen,
a shift-invariant circular pupil, and enough specimen texture and unscattered
reference interference to expose the pupil edges. Automatic selection keeps
the complete center-search region strictly inside `objective_na`. Explicitly
selected frames must meet the same rule. A usable frame has one positive source
contribution, positive gain, source power and measurement weight, and no masked
pixels. The implementation subtracts only the model's known background and
uses its known multiplicative factors; it never estimates or silently clamps
them.

Only global translation, rotation, pitch, and reference-index components may
be selected. Per-source offsets, source powers, and frame gains are rejected
because circle centers do not identify them. The initializer applies the same
translation/reference-index gauge checks as physical calibration and requires
the accepted-center data Jacobian to have full column rank before priors are
considered. Pitch and axial distance often form a scale gauge and must not be
fitted together unless the selected observations actually make the requested
combination full rank.

```python
parameters = fpm.PlanarArrayCalibrationParameters(
    translation=(True, True, False),
    translation_spec=fpm.CalibrationParameterSpec(
        -1e-3, 1e-3, scale=0.2e-3, finite_difference_step=1e-6
    ),
)
initializer = fpm.BrightfieldCircleInitializer(
    parameters,
    options=fpm.BrightfieldCircleOptions(
        center_search_radius_na=0.012,
        pupil_radius_search_na=0.012,
    ),
)
initialized = initializer.initialize(
    measurements,
    optics,
    nominal_illumination,
    nominal_model,
)
initialized.save_json("planar-array-initialization.json")

problem = fpm.ReconstructionProblem(measurements, initialized.initialized_model)
result = fpm.Fpie(iterations=20).run(problem)
```

`observations` retains the acquisition frame and stable source index, nominal
and detected wave vectors, dimensionless NA center, floating-point
`(row, column)` Fourier-grid position, fitted radius, both derivative scores,
conjugate score, confidence, negative corrected-sample fraction, and rejection
reason. `diagnostics` records acceptance counts, pupil-radius agreement,
data-Jacobian rank and conditioning, and initial/final center residuals. JSON
round trips preserve the complete physical result, options, model, and fit
history. `write_bundle` adds a hash-verified manifest plus normalized
observation and fit-history CSV tables; `read_initialization_bundle` verifies
all three artifacts before loading the authoritative result. The call blocks in
Python; native processing releases the GIL and reacquires it only for a
`PlanarArrayInitializationCallback`.

### Evaluate a bright-field warm start

The offline example
[`benchmark_brightfield_initialization.py`](https://github.com/hgrecco/fpm-rs/blob/main/python/examples/benchmark_brightfield_initialization.py)
compares four pipelines on the same detector data: nominal geometry with FPIE,
cold `JointReconstruction`, circle initialization followed by FPIE, and circle
initialization followed by joint refinement. The circle-only pipeline still
runs FPIE to measure downstream object error; it performs no subsequent
geometry refinement. Cold and warm joint runs use the same object-pass and
physical-step budgets. Each pipeline runs in a fresh process, and warm-run
timing includes circle detection and fitting.

```console
pixi run python python/examples/benchmark_brightfield_initialization.py \
  --scenarios all --captures 0.2 0.6 --passes 1 4 \
  --output /tmp/brightfield-comparison.json
```

Fixtures cover amplitude, phase, mixed, weak-texture and textureless specimens;
additive detector noise; near-cutoff and bright-field-poor acquisitions;
rotation, pitch, distance and reference-index errors; objective NA; known pupil
aberration; spatial vignetting; known source power, frame gain and background;
and pupil-radius mismatch. `capture` scales the named physical perturbation in
each fixture; the report retains SI parameter errors and dimensionless source
NA errors. The bright-field-poor fixture needs a larger Fourier grid to include
its dark-field sources; all four methods in a fixture share that grid.

The report includes source-vector NA RMSE, translation/pitch errors in metres,
rotation error in radians, reference-index error, detector amplitude MSE,
complex-field relative error, circle counts, rank/conditioning, fit evaluations,
runtime and process peak RSS. Complex-field comparison removes periodic integer
translation and global phase, retaining amplitude-scale error. Evaluation
after reconstruction is excluded from runtime and peak-memory sampling.
The two circle-detection measurement passes are counted separately from
reconstruction forward passes. `forward_model_pass_upper_bound` counts object
passes, each joint phase's base objective and geometry trial recompilations;
a failed recompile can consume no forward pass, so this is an upper bound.

Detection recall counts accepted centers within `--center-tolerance-na` of
geometrically eligible truth centers. Eligibility requires a safe bright-field
margin, a nominal center separated from its conjugate search region, and truth
inside the configured center search. Textureless and radius-mismatch fixtures
have no eligible truth edges. False acceptance is the fraction of accepted
centers that fail these labels or the localization tolerance; it is not a
population false-positive rate. Rejections and execution errors remain in the
report. The example uses an explicit 0.012 NA center search, 0.03 NA radius
search and 0.015 NA radius-agreement tolerance; these are benchmark controls.

`sampled_capture` reports the largest tested perturbation meeting both source
and object-error tolerances. `time_to_target` selects the fastest successful
sampled pass budget, including initialization time. Neither establishes a
continuous capture range or a convergence-time guarantee. The initializer
remains optional: inaccurate individual circles and unsuitable acquisitions
must be considered even when a global fit reduces the source-vector error.

J. Sun, Q. Chen, Y. Zhang, and C. Zuo, [“Efficient positional misalignment
correction method for Fourier ptychographic microscopy,” *Biomedical Optics
Express* **7**(4), 1336–1350
(2016)](https://doi.org/10.1364/BOE.7.001336), search independent apertures with
simulated annealing during reconstruction and then regress a four-parameter
planar model. fpm-rs instead performs no reconstructed-object search here and
fits detected centers directly to bounded physical geometry. R. Eckert,
Z. F. Phillips, and L. Waller, [“Efficient illumination angle self-calibration
in Fourier ptychography,” *Applied Optics* **57**(19), 5434–5442
(2018)](https://doi.org/10.1364/AO.57.005434), combine bright-field
preprocessing with iterative spectral correlation and cover additional
illuminator and three-dimensional settings. This implementation adopts only
their circular-edge initialization concept for two-dimensional planar arrays;
iterative physical refinement remains the existing calibrator's job. It does
not implement spectral correlation or label independent Fourier shifts as
apparatus calibration.

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
an illumination-only model refresh. Rust accepts compatible reconstruction
algorithms; Python accepts `Fpie` or pupil-recovering `Epry`. `Mpie` is rejected
because its velocity has no defined reset or
transport across physical model recompilation. The default
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
(2018)](https://doi.org/10.1364/AO.57.005434). The joint calibrator uses
deterministic bounded, scaled finite differences on the canonical measurement
loss rather than simulated annealing or spectral correlation; the separate
initializer above provides circle-based warm starts. The shared thin-sample FPM
forward model originates with [G. Zheng, R. Horstmeyer, and C. Yang,
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
`load_for_problem` additionally validates dimensions, exact binary pupil
support, and calibration against a specific problem before reconstruction
begins. Pupil-recovering algorithms store iteration-boundary canonical arrays;
an older valid checkpoint is projected into the current convention before start
callbacks and resumed work without changing the checkpoint format version.
`Mpie` stores its centered velocity and anchor spectra, effective-frame counter,
and defining update parameters in `algorithm_auxiliary`. A matching checkpoint
continues a partial interval; an auxiliary-free checkpoint is a warm start, and
a different solver's auxiliary variant is rejected.
`AdaptiveAlternatingProjection` similarly stores its current step, preceding
pass objective, active-pass sums and frame count, and defining controller
parameters. A matching checkpoint continues the feedback sequence; an
auxiliary-free checkpoint starts a new baseline, and other auxiliary variants
are rejected. The format remains version 2 because `algorithm_auxiliary` is the
existing solver-state extension point, though readers that predate the `Mpie`
or `AdaptiveAlternatingProjection` enum variant cannot load checkpoints
containing those variants.
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
