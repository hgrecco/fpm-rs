# Load and prepare an acquisition

The Python boundary accepts an in-memory NumPy stack. Load TIFF or another
instrument format with the tool appropriate to that format, convert detector
intensities to `float64`, and keep frame order aligned with the illumination
description.

```python
import numpy as np
import fpm_rs as fpm

frames = np.asarray(raw_frames, dtype=np.float64)  # (frames, height, width)
measurements = fpm.MeasurementStack(frames)
problem = fpm.ReconstructionProblem(measurements, model)
```

Input arrays are copied once because their storage is Python-owned. Every frame
must have the same `(height, width)` as `model.image_shape`. Values represent
intensity, not amplitude. Use `frame_weights` to down-weight or disable complete
frames and `masks` to exclude detector pixels.

The Rust API additionally provides resident and lazy stacks, image-stack
loading, measurement manifests, metadata, dark/background correction, masks,
and explicit preprocessing. See [Datasets](../datasets.md) for the native
bundle workflow and the [Rust API](../reference/rust.md) for `measurements`
types. Source-specific conversion is performed outside this repository; the
generic loader only consumes converted bundles and never accesses the network.
