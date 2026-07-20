# Quickstart

This workflow generates a tiny deterministic FPM acquisition in memory and
reconstructs it. It needs no downloaded data or machine-specific path.

```python
import numpy as np
import fpm_rs as fpm

# All distances are metres. Array shapes are (height, width).
optics = fpm.Optics(
    wavelength=532e-9,
    objective_na=0.10,
    magnification=4.0,
    camera_pixel_size=6.5e-6,
)
illumination = fpm.LEDArray(
    grid_shape=(3, 3),
    pitch=4e-3,
    distance=90e-3,
    center=(1.0, 1.0),
)
model = fpm.compile_model(optics, illumination, image_shape=(32, 32))

row, column = np.indices(model.reconstruction_shape)
amplitude = 0.7 + 0.3 * ((row // 8 + column // 8) % 2)
phase = 0.4 * np.sin(row / 7.0) * np.cos(column / 9.0)
object_field = np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex128)

simulation = fpm.simulate(model, object_field, seed=1234)
problem = fpm.ReconstructionProblem(
    simulation.measurements,
    simulation.reconstruction_model,
    name="quickstart",
)
result = fpm.AlternatingProjection(iterations=20).run(problem)

print(result.amplitude.shape)
print(result.runtime.completed_iterations)
print(result.final_loss)
```

Expected output includes an amplitude shape of `(42, 42)`, 20 completed
iterations, and a finite final loss. Exact floating-point loss values may vary
slightly by platform.

The model separates the measured low-resolution frame shape from the recovered
object shape. `simulate` returns both detector intensities and the reconstruction
model appropriate for those intensities. The algorithm returns NumPy amplitude,
phase, spectrum, and pupil arrays.

## Choose the reconstruction grid

Omitting `reconstruction_shape` selects the default `"smooth"` grid. You can
inspect every automatic choice before compiling the model:

```python
image_shape = (32, 32)

minimum_shape = fpm.suggest_reconstruction_shape(
    optics, illumination, image_shape, "minimum"
)
# Omitting the fourth argument selects "smooth".
smooth_shape = fpm.suggest_reconstruction_shape(optics, illumination, image_shape)
power_of_two_shape = fpm.suggest_reconstruction_shape(
    optics, illumination, image_shape, "power_of_two"
)

print(minimum_shape, smooth_shape, power_of_two_shape)
# (42, 42) (42, 42) (64, 64)

radix2_model = fpm.compile_model(
    optics,
    illumination,
    image_shape,
    reconstruction_shape="power_of_two",
)
exact_model = fpm.compile_model(
    optics,
    illumination,
    image_shape,
    reconstruction_shape=(64, 64),
)
```

`"minimum"` uses the smallest grid that contains every shifted
low-resolution Fourier crop. `"smooth"` rounds the shared aspect-ratio factor
to a number composed of 2, 3, 5, and 7, while `"power_of_two"` rounds that
factor to a power of two. An exact tuple is validated against the same crop
bounds. Pass one of these values explicitly or omit the argument; `None` is not
accepted.

The linear increase in pixels is often estimated by the ratio of synthetic NA
to objective NA, but that is only a heuristic. The software sizes from the
actual illumination wave vectors and includes the interpolation margin needed
for fractional Fourier shifts. See [Configure and run a reconstruction](../guides/reconstruction.md#choose-the-reconstruction-shape)
for the complete sizing rules.

Next, follow the rendered [first reconstruction tutorial](../tutorials/notebooks/quickstart.ipynb),
learn how to [record diagnostics](../diagnostics.md), or consult the
[Python API](../reference/python/index.md). For acquired arrays rather than a
simulation, see [Measurements and acquisitions](../guides/measurements.md).
