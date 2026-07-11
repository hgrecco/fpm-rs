# Core concepts and conventions

## Computational boundary

Experiment geometry (`Optics` plus illumination) compiles once into an
`ImagePlaneModel`: transverse wave vectors, pupil samples, Fourier crop
locations, subpixel sampling offsets, multiplex weights, and optional frame
gains. Algorithms consume that compiled model and measurements; they do not
depend on the original LED geometry. Simulation and reconstruction share the
same forward model.

## Object, pupil, and Fourier sampling

The object is a complex field. Its magnitude is amplitude and its angle is
phase. Spectra and pupils are FFT-shifted so zero frequency is at the array
centre. Each illumination source selects an overlapping low-resolution crop of
the high-resolution object spectrum and propagates it through the pupil.
Fractional crop offsets use bilinear sampling and adjoint-weighted insertion.

Arrays are row-major. Shapes use `(height, width)`, indices use `(row, column)`,
and k-vectors store `(kx, ky)` transverse angular spatial frequencies in
radians/metre. Positive `kx` moves toward increasing columns and positive `ky`
toward increasing rows.

## Units and illumination ordering

Distances are metres, wavelength is metres, numerical aperture and refractive
index are dimensionless, and component angles are radians unless their name
explicitly includes `degrees`. Source order determines measurement frame order.
An explicit acquisition order reorders source weights with the wave vectors.
For coded illumination, source count and measured frame count differ because a
frame is an incoherent intensity sum of source modes.

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

Algorithm rustdoc and the focused guides document equations, approximations,
implementation details, and literature references at their point of use.
