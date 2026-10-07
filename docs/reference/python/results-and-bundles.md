# Results and bundles

Reconstruction results live in memory. Result bundles persist numerical arrays
as NPY files, structured records as Parquet, and provenance in a JSON manifest.
Opened bundles load large arrays on first access and cache them until cleared.
Planar-array initialization bundles are smaller verified directories: their
authoritative result is JSON and their per-frame observations and physical-fit
history are normalized CSV tables.

## Results and checkpoints

::: fpm_rs.ReconstructionCheckpoint

::: fpm_rs.RuntimeInfo

::: fpm_rs.ReconstructionResult

::: fpm_rs.SpectralReconstructionResult

::: fpm_rs.OpticalPathDifferenceResult

::: fpm_rs.MultiWavelengthReconstructionResult

::: fpm_rs.MultiWavelengthSolverResult

::: fpm_rs.PlanarArrayParameterValues

::: fpm_rs.CalibrationParameterHistoryEntry

::: fpm_rs.CalibrationLossHistoryEntry

::: fpm_rs.CalibrationConditioning

::: fpm_rs.IlluminationCalibrationState

::: fpm_rs.BrightfieldCircleObservation

::: fpm_rs.PlanarArrayInitializationFitRecord

::: fpm_rs.PlanarArrayInitializationDiagnostics

::: fpm_rs.PlanarArrayInitializationRuntime

::: fpm_rs.PlanarArrayInitializationResult

::: fpm_rs.JointReconstructionResult

## Result bundles

::: fpm_rs.BundleArtifact

::: fpm_rs.BundleArray

::: fpm_rs.BundleTables

::: fpm_rs.BundleArrays

::: fpm_rs.BundlePreviews

::: fpm_rs.BundleVerificationResult

::: fpm_rs.ResultBundle

::: fpm_rs.InitializationBundleArtifact

::: fpm_rs.InitializationBundleVerificationResult

::: fpm_rs.InitializationBundle

::: fpm_rs.read_initialization_bundle

::: fpm_rs.read_bundle

## Benchmark bundles

::: fpm_rs.BenchmarkBundleTables

::: fpm_rs.BenchmarkBundle

::: fpm_rs.BenchmarkSuite

::: fpm_rs.read_benchmark_bundle
