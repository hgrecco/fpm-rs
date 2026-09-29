# FPM glossary

These short definitions are for readers who are comfortable with Python and
NumPy but new to physical optics. Follow each link for the implementation's
units, signs, and discrete-array conventions.

Numerical aperture (NA)
: A dimensionless measure of the angular range admitted by an optical system.
  In fpm-rs, `objective_na` sets the radius of the objective's circular Fourier
  passband. A larger value passes higher transverse spatial frequencies. See
  [Object, pupil, and Fourier sampling](core-concepts.md#object-pupil-and-fourier-sampling).

Synthetic NA
: A summary of the largest Fourier radius reached by combining the objective
  passband with shifted illumination. fpm-rs records it as `objective_na` plus
  the largest illumination NA. It is useful for estimating resolution, but it
  does not prove that the enclosed Fourier region is filled or recoverable.
  See [Low-resolution frames and the reconstruction grid](core-concepts.md#low-resolution-frames-and-the-reconstruction-grid).

Pupil function
: A complex-valued Fourier-space transfer function for the objective. Its
  magnitude describes transmitted spatial frequencies; its phase carries
  defocus and other modeled aberrations. Multiplying an object-spectrum crop by
  the pupil produces the field that is propagated to a detector frame. See
  [Object, pupil, and Fourier sampling](core-concepts.md#object-pupil-and-fourier-sampling).

k-vector
: A wave-propagation vector expressed as angular spatial frequency. fpm-rs
  stores the transverse pair `(kx, ky)` in radians per metre. During model
  compilation, that pair selects the location of a low-resolution crop in the
  centered high-resolution object spectrum. See
  [Object, pupil, and Fourier sampling](core-concepts.md#object-pupil-and-fourier-sampling).

Bright-field and dark-field illumination
: A source is bright-field when its unscattered illumination lies inside the
  objective passband; its illumination NA is no greater than the objective NA.
  A source outside that passband is dark-field, so an unscattered field would
  not reach the detector through the ideal pupil. Dark-field frames can still
  carry specimen-scattered information. See
  [Object, pupil, and Fourier sampling](core-concepts.md#object-pupil-and-fourier-sampling).

Bright-field circle initialization
: A pre-reconstruction estimate of planar-array geometry from the circular
  pupil edges visible in Fourier transforms of suitable bright-field intensity
  images. In fpm-rs, detected centers are fitted to physical pose, pitch, or
  reference-index variables; they are not stored as arbitrary per-source
  shifts. The method needs textured, single-source, all-valid frames and is not
  interchangeable with measurement-loss calibration. See
  [Bright-field planar-array initialization](../guides/reconstruction.md#bright-field-planar-array-initialization).

LED pitch
: The centre-to-centre spacing of neighboring LEDs in a planar array. fpm-rs
  stores `pitch_m` in metres as `(pitch_x, pitch_y)` after accepting either a
  scalar equal pitch or a two-value input. Pitch is separate from the array's
  pose and reference index. See
  [Illumination architecture, units, and acquisition](core-concepts.md#illumination-architecture-units-and-acquisition).

Defocus
: Axial displacement from the modeled focus. fpm-rs expresses
  `defocus_distance` in metres and compiles it as a spatial-frequency-dependent
  phase in the pupil, rather than shifting image pixels. See
  [Object, pupil, and Fourier sampling](core-concepts.md#object-pupil-and-fourier-sampling).

The Fourier-crop interpretation follows G. Zheng, R. Horstmeyer, and C. Yang,
[“Wide-field, high-resolution Fourier ptychographic microscopy,” *Nature
Photonics* **7**, 739–745
(2013)](https://doi.org/10.1038/nphoton.2013.187). The joint object/pupil
interpretation follows X. Ou, G. Zheng, and C. Yang, [“Embedded pupil function
recovery for Fourier ptychographic microscopy,” *Optics Express* **22**(5),
4960–4972 (2014)](https://doi.org/10.1364/OE.22.004960).
