# Core concepts and conventions

## Computational boundary

Experiment geometry (`Optics` plus illumination) compiles once into an
`ImagePlaneModel`: transverse wave vectors, pupil samples, Fourier crop
locations, subpixel sampling offsets, multiplex weights, and optional frame
gains. Algorithms consume that compiled model and measurements; they do not
depend on the original LED geometry. Simulation and reconstruction share the
same forward model.

Reconstruction is generic over the read-only `MeasurementRead` trait.
`MeasurementStack` and `LazyMeasurementStack` implement it directly, so the
algorithm path is statically dispatched without a dataset enum or measurement
trait objects. Construction, preprocessing, materialization, and cache controls
remain operations on the concrete storage types.

## Object, pupil, and Fourier sampling

The object is a complex field. Its magnitude is amplitude and its angle is
phase. Spectra and pupils are FFT-shifted so zero frequency is at the array
centre. Each illumination source selects an overlapping low-resolution crop of
the high-resolution object spectrum and propagates it through the pupil.
Fractional crop offsets use bilinear sampling and adjoint-weighted insertion.
The zero-offset path remains a direct copy; interpolation is not treated as an
invertible copy. `ImagePlaneModel::new` uses integer crops by default; use
`with_subpixel_offsets` when calibrated low-level inputs include fractional
source positions.

Arrays are row-major. Shapes use `(height, width)`, indices use `(row, column)`,
and k-vectors store `(kx, ky)` transverse angular spatial frequencies in
radians/metre. Positive `kx` moves toward increasing columns and positive `ky`
toward increasing rows.

`KVector` values are transverse angular spatial frequencies. Direct vectors
must represent propagating waves: their transverse magnitude cannot exceed
`2π n_illumination / wavelength_vacuum`. Direction lists store normalized
wave-propagation vectors instead of angles. Component-angle construction uses
`dx = sin(theta_x)` and `dy = sin(theta_y)`; polar construction uses `theta`
from positive `z` and azimuth `phi` from positive `x` toward positive `y`.
The corresponding illumination numerical aperture is
`wavelength_vacuum * hypot(kx, ky) / (2π)`. A source is bright-field when this
value does not exceed `objective_na`, so its unscattered wave lies inside the
objective passband; larger values are dark-field.

The sample plane is `z = 0`, illumination sources normally have `z < 0`, and
the objective is on the `z > 0` side. The right-handed on-axis incident
direction is `(0, 0, 1)`. A physical source position `p` resolves to
`normalize(-p)`, then to `k0 * (direction.x, direction.y)`. This propagation
sign is a physical convention. Separately, the centered FFT implementation
uses positive `kx` for increasing Fourier columns and positive `ky` for
increasing Fourier rows. Model compilation applies no extra sign reversal.

<figure>
  <svg viewBox="0 0 900 330" role="img" aria-labelledby="coordinate-diagram-title coordinate-diagram-desc" style="width: 100%; height: auto;" xmlns="http://www.w3.org/2000/svg">
    <title id="coordinate-diagram-title">FPM axial and Fourier-crop conventions</title>
    <desc id="coordinate-diagram-desc">A source below the sample propagates toward positive z and the objective. Its positive transverse wave vector selects an offset crop of the centered high-resolution object spectrum.</desc>
    <defs>
      <marker id="coordinate-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L8,4 L0,8 Z" fill="currentColor"/>
      </marker>
      <marker id="coordinate-accent-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L8,4 L0,8 Z" fill="#d97706"/>
      </marker>
    </defs>
    <g fill="none" stroke="currentColor" stroke-width="2">
      <line x1="55" y1="285" x2="55" y2="35" marker-end="url(#coordinate-arrow)"/>
      <line x1="75" y1="160" x2="390" y2="160"/>
      <path d="M220,55 Q260,25 300,55 Q260,85 220,55 Z"/>
      <circle cx="115" cy="275" r="13" fill="#4f7cac" stroke="#4f7cac"/>
      <line x1="127" y1="266" x2="253" y2="166" stroke="#d97706" stroke-width="3" marker-end="url(#coordinate-accent-arrow)"/>
      <line x1="260" y1="153" x2="260" y2="82" stroke="#d97706" stroke-dasharray="7 5" marker-end="url(#coordinate-accent-arrow)"/>
      <rect x="510" y="45" width="300" height="240" rx="4"/>
      <line x1="660" y1="170" x2="790" y2="170" marker-end="url(#coordinate-arrow)"/>
      <line x1="660" y1="170" x2="660" y2="270" marker-end="url(#coordinate-arrow)"/>
      <circle cx="660" cy="170" r="4" fill="currentColor"/>
      <rect x="605" y="125" width="110" height="90" stroke-dasharray="6 5" opacity="0.55"/>
      <line x1="660" y1="170" x2="735" y2="220" stroke="#d97706" stroke-width="3" marker-end="url(#coordinate-accent-arrow)"/>
      <rect x="680" y="175" width="110" height="90" fill="#4f7cac" fill-opacity="0.18" stroke="#4f7cac" stroke-width="3"/>
    </g>
    <g fill="currentColor" font-family="sans-serif" font-size="16">
      <text x="38" y="28">+z</text>
      <text x="80" y="150">sample, z = 0</text>
      <text x="215" y="25">objective, z &gt; 0</text>
      <text x="74" y="310">source, z &lt; 0</text>
      <text x="150" y="240" fill="#d97706">incident k, propagation toward +z</text>
      <text x="510" y="28">centered high-resolution object spectrum</text>
      <text x="795" y="164">+kx</text>
      <text x="670" y="285">+ky (rows)</text>
      <text x="643" y="160">0</text>
      <text x="700" y="168" fill="#d97706">(kx, ky)</text>
      <text x="690" y="250">selected low-resolution crop</text>
    </g>
  </svg>
  <figcaption>A source below the sample produces a positive-z incident wave. Its transverse <code>(kx, ky)</code> shifts the crop toward increasing Fourier columns and rows; no additional sign reversal is applied.</figcaption>
</figure>

The low-resolution Fourier spacings are `dkx = 2π / (width * object-plane
pixel size)` and equivalently for `dky`. CPU FFTs normalize the forward
transform by `1/N` and leave the inverse unnormalized, preserving
constant-object amplitude through a high-resolution crop and low-resolution
inverse transform. The ideal pupil includes samples whose transverse frequency
is at most `2π NA / λ`. `PupilAberration` coefficients are direct radian
weights for the documented sampled radial-polynomial terms, not normalized
Zernike coefficients. A `defocus_distance` of `d` metres adds the paraxial
pupil phase `-d * (kx² + ky²) / (2 * k_medium)` inside the support.

The low-resolution FFT grid must contain that coherent pupil. With object-plane
detector pitch `Δx = camera_pixel_size / magnification`, pupil cutoff
`k_cutoff = 2π objective_na / wavelength_vacuum`, and Nyquist angular frequency
`k_Nyquist = π / Δx`, model validation requires

```text
k_cutoff < k_Nyquist
Δx < wavelength_vacuum / (2 * objective_na)
```

Equality is rejected because an even discrete grid represents only one side of
the Nyquist boundary. The objective numerical aperture already includes the
objective medium, so the expression uses the vacuum wavelength without another
refractive-index factor.

The continuous intensity `|field|²` can have twice the coherent-field
bandwidth. Strict point sampling of every such intensity component would use
the stronger condition `Δx < wavelength_vacuum / (4 * objective_na)`. The
current forward model instead evaluates `|field|²` at the coherent field's
sample locations; it does not integrate intensity over detector-pixel area or
implement a sub-sampled sensor model. The stronger condition is therefore
acquisition guidance, not a second validation error.

These conventions follow [G. Zheng, R. Horstmeyer, and C. Yang, “Wide-field,
high-resolution Fourier ptychographic microscopy”
(2013)](https://doi.org/10.1038/nphoton.2013.187), *Nature Photonics* **7**,
739–745, and the practical sampling discussion in [S. Jiang, P. Song, T. Wang,
L. Yang, R. Wang, C. Guo, B. Feng, A. Maiden, and G. Zheng, “Spatial- and
Fourier-domain ptychography for high-throughput bio-imaging”
(2023)](https://doi.org/10.1038/s41596-023-00829-4), *Nature Protocols* **18**,
2051–2083.

`CoordinateConvention::CenteredPositiveK` records these choices in compiled
models.

### Blind object/pupil gauge

Joint object and pupil recovery does not determine a unique pair of complex
arrays. For an integer source displacement `d_s`, the exit spectrum is

```text
E_s(u) = O(u + d_s) P(u).
```

Reciprocal object/pupil scale, independent constant phase factors, and coupled
affine phase ramps can therefore describe the same measured intensities. These
blind-ptychography ambiguities are characterized by [A. Fannjiang and P. Chen,
“Blind ptychography: uniqueness and ambiguities” (2020), *Inverse Problems*
**36**, 045005](https://doi.org/10.1088/1361-6420/ab6504). The paper treats the
real-space blind-ptychography model; fpm-rs adapts the ambiguity to its centered
Fourier-domain object and common sampled pupil.

Built-in EPRY and gradient descent use the immutable compiled pupil as their
gauge reference whenever pupil recovery is enabled. At initialization or
checkpoint restoration and after every complete iteration, they:

1. remove a relative pupil phase ramp on each axis whose effective subpixel
   offsets are all zero within `1e-12` Fourier-grid pixels;
2. match the recovered pupil's supported energy to the compiled pupil and make
   their supported complex overlap positive real, applying the reciprocal
   constant to the object spectrum; and
3. rotate the object spectrum so its centered DC coefficient is non-negative
   real, using its strongest coefficient if DC is numerically zero.

These transformations preserve the modeled intensities. They run after pupil
support projection and regularization, and before iteration diagnostics,
iteration callbacks, checkpoints, and final result construction. A restored
checkpoint is canonicalized before its start callback. Checkpoint compatibility
also requires its binary pupil support to match the compiled support exactly.

Fractional crop offsets use bilinear interpolation, which does not commute
exactly with a discrete affine phase ramp. The affine correction is therefore
skipped independently on each axis containing any fractional effective source
offset; that axis's slope remains in the reported pupil. Reciprocal scale and
constant-phase normalization remain exact and are always applied. A collapsed
or non-finite supported pupil cannot be normalized and ends the reconstruction
with a numerical error. Corrections within `64 * machine epsilon * |S|`, where
`|S|` is the supported-sample count, are treated as the identity so repeated
canonicalization does not perturb an already canonical state.

## Low-resolution frames and the reconstruction grid

Each measured intensity frame is a low-resolution image with `image_shape`.
The recovered amplitude (or modulation) and phase are components of one complex
field with the larger `reconstruction_shape`. The grids cover the same field of
view, so increasing each linear dimension by a factor `s` decreases the
reconstructed pixel pitch by the same factor and increases the total pixel
count by approximately `s²`.

The frequently used estimate `s ≈ synthetic_na / objective_na` describes an
ideal bandwidth ratio. Model compilation does not turn that ratio directly into
an array size. It resolves the supplied illumination into k-vectors, maps their
shifts onto the discrete Fourier grid, and finds enough room for all complete
low-resolution crops and any fractional interpolation neighbors. This handles
asymmetric, calibrated, and one-sided illumination without assuming symmetric
coverage. Automatic choices preserve the low-resolution aspect ratio and
isotropic high-resolution pixel sampling.

The compiled `sampling.synthetic_na` diagnostic is `objective_na` plus the
largest illumination numerical aperture. It summarizes the outermost ideal
radial extent; it does not assert complete, isotropic, or recoverable Fourier
coverage.

See [Configure and run a reconstruction](../guides/reconstruction.md#choose-the-reconstruction-shape)
for the exact, minimum, smooth, and power-of-two selection modes.

## Illumination architecture, units, and acquisition

`SourceGeometry` owns only source placement or direct directions/vectors.
`SourceCalibration` owns stable per-source relative optical power.
`AcquisitionPlan` owns sparse source-to-frame contributions and frame gains.
`Illumination` resolves all three atomically with `Optics` into inspectable
`ResolvedSources` and `ResolvedIllumination`; only the compiled
`ImagePlaneModel` crosses into simulation and reconstruction algorithms.

`Optics.wavelength_vacuum_m` is vacuum wavelength. Illumination propagation
uses `illumination_refractive_index`, while pupil propagation uses the distinct
`objective_medium_refractive_index`. Geometry has no wavelength override.
Distances and positions are metres, k-vectors are radians/metre, refractive
indices and powers are dimensionless, and angle units appear in constructor and
field names.

`PlanarLEDArray.shape` is `(rows, columns)`, `pitch_m` is `(pitch_x, pitch_y)`,
and `reference_index` is fractional `(column, row)`. Indexing is row-major:
`source = row * columns + column`. Local position offsets are applied before
`ArrayPose`. Pose rotations are active, right-handed, extrinsic about fixed
sample x, y, then z axes; the matrix is `Rz @ Ry @ Rx`.

Acquisition storage is sparse and canonical. Duplicate sources in a frame are
merged, zero weights are removed, and empty frames are rejected. Subsets and
repeated sources are valid. All weights, source powers, and gains are finite,
non-negative intensity multipliers and are not normalized automatically:

```text
I_frame[f] = gain[f] * sum_s(
    frame_weight[f, s] * relative_power[s] * I_source[s]
)
```

Contributions are mutually incoherent. Source count and frame count are
therefore independent. Dense `(frames, sources)` accessors allocate from sparse
canonical storage.

### Python illumination patterns

```python
import numpy as np
import fpm_rs as fpm

optics = fpm.Optics(532e-9, 0.1, 4.0, 6.5e-6)
pose = fpm.ArrayPose.from_translation((0.0, 0.0, -90e-3))

# 1–4: regular/tilted, unequal-pitch, position-corrected planar arrays.
regular = fpm.PlanarLEDArray((3, 3), 4e-3, (1.0, 1.0), pose)
tilted = fpm.PlanarLEDArray(
    (3, 3), (4e-3, 5e-3), (1.0, 1.0),
    fpm.ArrayPose.from_translation_and_extrinsic_xyz_degrees(
        (1e-3, 0.0, -90e-3), (2.0, -1.0, 0.5)
    ),
    position_offsets_m=np.zeros((9, 3)),
)
# 5–7: positions, directions, and direct wavelength-dependent k-vectors.
positions = fpm.SourcePositionList(np.array([[0.0, 0.0, -0.09]]))
directions = fpm.DirectionList.from_polar_angles_degrees(
    np.array([[0.0, 0.0], [15.0, 45.0]])
)
vectors = fpm.KVectorList(np.array([[0.0, 0.0], [1e4, -2e4]]))
# 8–10: subset, repeated, and multiplexed acquisition.
subset = fpm.AcquisitionPlan.sequential([4, 0, 8])
repeated = fpm.AcquisitionPlan.sequential([0, 1, 0])
multiplexed = fpm.AcquisitionPlan.from_dense(
    np.array([[1.0, 0.0], [0.25, 0.75]])
)
illumination = fpm.Illumination(
    vectors,
    calibration=fpm.SourceCalibration(relative_power=[0.8, 1.2]),
    acquisition=multiplexed,
)
resolved = illumination.resolve(optics)
model = fpm.compile_model(optics, illumination, (32, 32))
# 11–12: deterministic true/assumed geometry mismatch and reconstruction.
true_model = fpm.compile_model(optics, fpm.Illumination(tilted), (32, 32))
assumed_model = fpm.compile_model(optics, fpm.Illumination(regular), (32, 32))
simulation = fpm.simulate(
    true_model,
    np.ones(true_model.reconstruction_shape, dtype=np.complex128),
    reconstruction_model=assumed_model,
)
problem = fpm.ReconstructionProblem(simulation.measurements, assumed_model)
result = fpm.AlternatingProjection(iterations=10).run(problem)
```

### Equivalent Rust illumination patterns

```rust
use fpm_rs::{
    algorithms::{AlternatingProjection, ReconstructionAlgorithm},
    experiment::{
        AcquisitionPlan, ArrayPose, DirectionList, Illumination, KVector,
        KVectorList, Optics, PlanarLedArray, SourceCalibration,
        SourcePositionList,
    },
    model::{ImagePlaneModel, ReconstructionShape},
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject},
};

# fn example() -> fpm_rs::Result<()> {
let optics = Optics {
    wavelength_vacuum_m: 532e-9,
    objective_na: 0.1,
    magnification: 4.0,
    camera_pixel_size: 6.5e-6,
    illumination_refractive_index: 1.0,
    objective_medium_refractive_index: 1.0,
    defocus_distance: None,
    pupil_aberration: None,
};
let pose = ArrayPose::from_translation([0.0, 0.0, -90e-3]);
let regular = PlanarLedArray::new((3, 3), (4e-3, 4e-3), (1.0, 1.0), pose);
let tilted = PlanarLedArray::new(
    (3, 3),
    (4e-3, 5e-3),
    (1.0, 1.0),
    ArrayPose::from_translation_and_extrinsic_xyz_degrees(
        [1e-3, 0.0, -90e-3],
        [2.0, -1.0, 0.5],
    ),
)
.with_position_offsets_m(vec![[0.0; 3]; 9]);
let _positions = SourcePositionList::new(vec![[0.0, 0.0, -0.09]]);
let _directions = DirectionList::from_polar_angles_degrees(vec![[0.0, 0.0], [15.0, 45.0]])?;
let vectors = KVectorList::new(vec![KVector::new(0.0, 0.0), KVector::new(1e4, -2e4)]);
let _subset = AcquisitionPlan::sequential(vec![4, 0, 8])?;
let _repeated = AcquisitionPlan::sequential(vec![0, 1, 0])?;
let multiplexed = AcquisitionPlan::from_dense(vec![vec![1.0, 0.0], vec![0.25, 0.75]])?;
let illumination = Illumination::new(
    vectors.into(),
    SourceCalibration::new(Some(vec![0.8, 1.2])),
    multiplexed,
);
let resolved = illumination.resolve(&optics)?;
let _model = ImagePlaneModel::compile(
    &optics,
    &resolved,
    (32, 32),
    ReconstructionShape::Smooth,
)?;
let true_model = ImagePlaneModel::from_experiment(
    &optics,
    &Illumination::from_geometry(tilted)?,
    (32, 32),
    ReconstructionShape::Smooth,
)?;
let assumed_model = ImagePlaneModel::from_experiment(
    &optics,
    &Illumination::from_geometry(regular)?,
    (32, 32),
    ReconstructionShape::Smooth,
)?;
let simulation = Simulator::new(true_model)
    .object(SyntheticObject::constant(assumed_model.reconstruction_shape(), 1.0, 0.0)?)
    .reconstruction_model(assumed_model.clone())
    .simulate()?;
let problem = ReconstructionProblem::new(simulation.measurements, assumed_model)?;
let _result = AlternatingProjection::default().iterations(10).run(&problem)?;
# Ok(())
# }
```

Generic reconstruction-time k-vector corrections remain compiled-state offsets.
They need not correspond to any physical geometry and are intentionally distinct
from `PlanarLedArray.position_offsets_m`.

## Callbacks and execution boundary

The runner calls callbacks at lifecycle points around reconstruction and
iterations. Built-in Rust callbacks do not cross into Python. Python simulation,
reconstruction, model compilation, checkpoint I/O, image-backed object loading,
and diagnostic serialization release the GIL around Rust-owned work. A Python
iteration callback necessarily reacquires it for the callable and therefore can
affect throughput. Reconstruction calls block the invoking Python thread.

## Backend scope

The current backend is CPU-only. The Rust backend traits define an integration
seam for future resident/device execution, but selecting a GPU backend is not a
supported task today. Diffraction-plane and multislice forward models are also
not implemented.

Backends are thread-safe trait objects. Rust callers can inject one through
`Runner::with_backend`, `ReconstructionState::initialize_with_backend`, or
`ForwardModel::with_backend`; checkpoint resume and callback-requested forward
diagnostics retain that choice. `BackendCapabilities`, typed
`ComplexBuffer`/`RealBuffer`, and `ResidentBackend` describe preferred memory
location, transfers, and resident FFT dispatch. `CpuBackend` uses host buffers.
Reconstruction state is host-owned today, so this is an extension seam rather
than GPU execution support.

Algorithm rustdoc and the focused guides document equations, approximations,
implementation details, and literature references at their point of use.
