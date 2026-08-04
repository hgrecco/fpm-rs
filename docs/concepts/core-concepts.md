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

The sample plane is `z = 0`, illumination sources normally have `z < 0`, and
the objective is on the `z > 0` side. The right-handed on-axis incident
direction is `(0, 0, 1)`. A physical source position `p` resolves to
`normalize(-p)`, then to `k0 * (direction.x, direction.y)`. This propagation
sign is a physical convention. Separately, the centered FFT implementation
uses positive `kx` for increasing Fourier columns and positive `ky` for
increasing Fourier rows. Model compilation applies no extra sign reversal.

The low-resolution Fourier spacings are `dkx = 2π / (width * object-plane
pixel size)` and equivalently for `dky`. CPU FFTs normalize the forward
transform by `1/N` and leave the inverse unnormalized, preserving
constant-object amplitude through a high-resolution crop and low-resolution
inverse transform. The ideal pupil includes samples whose transverse frequency
is at most `2π NA / λ`. `PupilAberration` coefficients are direct radian
weights for the documented sampled radial-polynomial terms, not normalized
Zernike coefficients.

`CoordinateConvention::CenteredPositiveK` records these choices in compiled
models.

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
