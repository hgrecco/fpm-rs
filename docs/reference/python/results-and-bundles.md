# Results and bundles

Reconstruction results live in memory. Result bundles persist numerical arrays
as NPY files, structured records as Parquet, and provenance in a JSON manifest.
Opened bundles load large arrays on first access and cache them until cleared.

## Results and checkpoints

::: fpm_rs.ReconstructionCheckpoint

::: fpm_rs.RuntimeInfo

::: fpm_rs.ReconstructionResult

::: fpm_rs.PlanarArrayParameterValues

::: fpm_rs.CalibrationParameterHistoryEntry

::: fpm_rs.CalibrationLossHistoryEntry

::: fpm_rs.CalibrationConditioning

::: fpm_rs.IlluminationCalibrationState

::: fpm_rs.JointReconstructionResult

## Result bundles

::: fpm_rs.BundleArtifact

::: fpm_rs.BundleArray

::: fpm_rs.BundleTables

::: fpm_rs.BundleArrays

::: fpm_rs.BundlePreviews

::: fpm_rs.BundleVerificationResult

::: fpm_rs.ResultBundle

::: fpm_rs.read_bundle

## Benchmark bundles

::: fpm_rs.BenchmarkBundleTables

::: fpm_rs.BenchmarkBundle

::: fpm_rs.BenchmarkSuite

::: fpm_rs.read_benchmark_bundle
