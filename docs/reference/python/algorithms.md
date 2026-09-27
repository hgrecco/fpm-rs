# Reconstruction algorithms

Every solver exposes the same `run` method and releases the Python GIL while
executing Rust reconstruction code. The choice of solver controls its update
rule, recoverable quantities, and algorithm-specific parameters.

::: fpm_rs.AlternatingProjection

::: fpm_rs.AdaptiveAlternatingProjection

::: fpm_rs.Fpie

::: fpm_rs.Mpie

::: fpm_rs.Epry

::: fpm_rs.Admm

::: fpm_rs.GradientDescent

## Physical planar-array calibration

::: fpm_rs.CalibrationParameterSpec

::: fpm_rs.PlanarArrayCalibrationParameters

::: fpm_rs.BoundedFiniteDifferenceOptimizer

::: fpm_rs.IlluminationCalibration

::: fpm_rs.JointReconstruction
