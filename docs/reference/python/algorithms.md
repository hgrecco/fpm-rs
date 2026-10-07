# Reconstruction algorithms

Ordinary solvers share a `run` interface; the spectral solver accepts a separate
spectral problem with keyword-only arguments. Both release the Python GIL while
executing Rust reconstruction code. The choice of solver controls its update
rule, recoverable quantities, and algorithm-specific parameters.

::: fpm_rs.AlternatingProjection

::: fpm_rs.AdaptiveAlternatingProjection

::: fpm_rs.Fpie

::: fpm_rs.Mpie

::: fpm_rs.Epry

::: fpm_rs.Admm

::: fpm_rs.GradientDescent

## Narrowband spectral reconstruction

::: fpm_rs.SpectralAlternatingProjection

::: fpm_rs.SyntheticWavelengthUnwrapper

::: fpm_rs.MultiWavelengthGradientDescent

## Physical planar-array calibration

### Bright-field initialization

::: fpm_rs.BrightfieldCircleOptions

::: fpm_rs.PlanarArrayInitializationCallback

::: fpm_rs.BrightfieldCircleInitializer

### Measurement-loss calibration

::: fpm_rs.CalibrationParameterSpec

::: fpm_rs.PlanarArrayCalibrationParameters

::: fpm_rs.BoundedFiniteDifferenceOptimizer

::: fpm_rs.IlluminationCalibration

::: fpm_rs.JointReconstruction
