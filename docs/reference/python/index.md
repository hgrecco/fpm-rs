# Python API

The public package is `fpm_rs`. Configuration objects and NumPy arrays cross the
Python boundary while model compilation, simulation, reconstruction, and bundle
I/O run in Rust. Private `_core` implementation names are intentionally omitted.

Start with the domain that owns the concept you need:

- [Datasets](datasets.md) for registry discovery, caching, and loaded data.
- [Experiment and illumination](experiment.md) for physical microscope inputs.
- [Compiled models and measurements](model-and-measurements.md) for the
  algorithm-facing forward model and measured stacks.
- [Simulation](simulation.md) for synthetic objects, detector effects, and
  deterministic acquisitions.
- [Reconstruction algorithms](algorithms.md) for solver selection and execution.
- [Results and bundles](results-and-bundles.md) for outputs, checkpoints, and
  durable or benchmark artifacts.
- [Callbacks and diagnostics](diagnostics.md) for run hooks, recording, plotting,
  and reports.
- [Evaluation and metrics](evaluation-and-metrics.md) for comparison with
  reference data.

The checked-in `python/fpm_rs/__init__.pyi`, `metrics.pyi`, and `evaluation.pyi`
files are the authoritative signatures and API prose used by this reference.

## Shared types and exceptions

All package-specific failures derive from `FpmError`. Shape, parameter, model,
measurement, length, frame-range, numerical, unsupported-operation, dataset,
I/O, and serialization failures have distinct subclasses.

::: fpm_rs
    options:
      members:
        - __version__
        - Shape2D
        - ReconstructionShapeSpec
        - FloatArray
        - ComplexArray
        - MaskArray
        - Path
        - Illumination
        - Callback
        - FpmError
        - InvalidShapeError
        - InvalidParameterError
        - InvalidModelError
        - InvalidMeasurementsError
        - LengthMismatchError
        - FrameOutOfRangeError
        - NumericalError
        - UnsupportedError
        - DatasetError
        - FpmIoError
        - SerializationError
      show_root_toc_entry: false
