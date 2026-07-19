# Python API

The public package is `fpm_rs`. Configuration objects and arrays cross the
Python boundary while model compilation, simulation, and reconstruction run in
the Rust extension. The checked-in `python/fpm_rs/__init__.pyi` supplies the
typed public surface; private `_core` implementation objects are intentionally
not listed here.

- [Models, measurements, and simulation](model-and-simulation.md)
- [Reconstruction and callbacks](reconstruction.md)
- [Plotting and reports](diagnostics.md)

::: fpm_rs
    options:
      members:
        - open_dataset
        - suggest_reconstruction_shape
        - compile_model
        - compile_camera_model
        - simulate
      show_root_toc_entry: false

## Exceptions

All package-specific failures derive from `FpmError`. Shape, parameter, model,
measurement, length, frame-range, numerical, unsupported-operation, dataset,
I/O, and serialization failures have distinct subclasses.

::: fpm_rs
    options:
      members:
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
