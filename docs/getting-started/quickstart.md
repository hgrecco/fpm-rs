# Quickstart

This workflow generates a tiny deterministic FPM acquisition in memory and
reconstructs it. It needs no downloaded data or machine-specific path.
For unfamiliar optics terms, keep the [FPM glossary](../concepts/glossary.md)
open alongside the example.

```python
import numpy as np
import fpm_rs as fpm

# All distances are metres. Array shapes are (height, width).
optics = fpm.Optics(
    wavelength_vacuum_m=532e-9,
    objective_na=0.10,
    magnification=4.0,
    camera_pixel_size=6.5e-6,
)
geometry = fpm.PlanarLEDArray(
    shape=(3, 3),
    pitch_m=4e-3,
    reference_index=(1.0, 1.0),
    pose=fpm.ArrayPose.from_translation(
        translation_m=(0.0, 0.0, -90e-3),
    ),
)
illumination = fpm.Illumination(geometry=geometry)
model = fpm.compile_model(
    optics=optics,
    illumination=illumination,
    image_shape=(32, 32),
)

row, column = np.indices(model.reconstruction_shape)
amplitude = 0.7 + 0.3 * ((row // 8 + column // 8) % 2)
phase = 0.4 * np.sin(row / 7.0) * np.cos(column / 9.0)
object_field = np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex128)

simulation = fpm.simulate(
    true_model=model,
    object=object_field,
    seed=1234,
)
problem = fpm.ReconstructionProblem(
    measurements=simulation.measurements,
    model=simulation.reconstruction_model,
    name="quickstart",
)
result = fpm.AlternatingProjection(iterations=20).run(problem)

print(result.amplitude.shape)
print(result.runtime.completed_iterations)
print(result.final_objective)
```

Expected output includes an amplitude shape of `(42, 42)`, 20 completed
iterations, and a finite final objective. Exact floating-point objective values may vary
slightly by platform.

The model separates the measured low-resolution frame shape from the recovered
object shape. `simulate` returns both detector intensities and the reconstruction
model appropriate for those intensities. The algorithm returns NumPy amplitude,
phase, spectrum, and pupil arrays.

The default reconstruction grid works for this example; see
[Choose the reconstruction shape](../guides/reconstruction.md#choose-the-reconstruction-shape)
for exact sizing, the three automatic strategies, and the synthetic-NA
heuristic's limits.

Next, follow the rendered [first reconstruction tutorial](../tutorials/notebooks/quickstart.ipynb),
learn how to [record diagnostics](../diagnostics.md), or consult the
[Python API](../reference/python/index.md). For acquired arrays rather than a
simulation, see [Measurements and acquisitions](../guides/measurements.md).
