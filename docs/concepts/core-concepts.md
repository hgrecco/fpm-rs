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

`KVector` values are transverse angular spatial frequencies. Direct-k and
angle-based descriptions must represent propagating waves: the transverse
magnitude cannot exceed `2πn/λ`; component angles lie in `[-π/2, π/2]`; and the
combined transverse direction is validated. For a planar LED at lateral
position `(x, y)` and axial distance `z`, the compiled vector is
`k0 * (x, y) / sqrt(x² + y² + z²)`, where `k0 = 2πn/λ`. Spherical coordinates
use polar angle `theta` from positive `z` and azimuth `phi` from positive `x`
toward positive `y`.

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

## Units and illumination ordering

Distances are metres, wavelength is metres, numerical aperture and refractive
index are dimensionless, and component angles are radians unless their name
explicitly includes `degrees`. Source order determines measurement frame order.
An explicit acquisition order reorders source weights with the wave vectors.
For coded illumination, source count and measured frame count differ because a
frame is an incoherent intensity sum of source modes.

LED intensity weights are reordered with acquisition order and compiled into
`ImagePlaneModel::frame_gains`, so reconstruction does not need to retain the
original `LEDArray`. The multiplexing matrix maps each measured frame to its
incoherently weighted sources. The shared forward model, simulator, and all
reconstruction algorithms support those sums: projection methods apply the
same measured/predicted amplitude ratio to every source mode and back-project
with its normalized multiplex weight.

`Illumination::Calibrated` represents imported k-vectors with optional
per-frame gains and multiplexing rows. Dataset subsets use it to preserve
calibrated model effects without inventing LED geometry.

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
