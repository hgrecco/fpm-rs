# Configure and run a reconstruction

## Compile the model

Create `Optics` and one illumination description, then call `compile_model`.
The measured `image_shape` and recovered `reconstruction_shape` are both
`(height, width)`. Physical distances are metres and angles are radians unless
a parameter explicitly says degrees.

When `reconstruction_shape` is omitted, compilation selects `"smooth"`: the
smallest fitting aspect-preserving grid whose shared size multiplier contains
only factors 2, 3, 5, and 7. Other accepted values are an exact `(height,
width)`, `"minimum"`, and `"power_of_two"`. Inspect the choice without compiling
a pupil with `suggest_reconstruction_shape`:

```python
minimum = fpm.suggest_reconstruction_shape(
    optics, illumination, image_shape, "minimum"
)
model = fpm.compile_model(optics, illumination, image_shape)
print(minimum, model.reconstruction_shape)
```

Every positive dimension is supported by RustFFT. These modes choose a
memory/performance tradeoff; they do not guarantee recoverable information or
uniform Fourier coverage. Automatic sizing uses the actual illumination wave
vectors and includes fractional interpolation stencils, so it need not equal a
simple synthetic-NA/objective-NA estimate.

Choose `LEDArray` for a planar grid, `AngleList` or `KVectorList` for calibrated
directions, and `CodedIllumination` for multiplexed frames. Spherical source
classes have additional identifiability constraints documented in
[Spherical illumination geometries](../spherical-geometries.md).

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

## Select and run an algorithm

`AlternatingProjection` is the simplest starting point. `Fpie` adds regularized
object updates, `Epry` can recover the pupil and frame response, `Admm` separates
data fitting from overlap consensus, and `GradientDescent` supports calibrated
illumination/pupil recovery and regularization.

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
[generated reconstruction API](../reference/python/reconstruction.md).
