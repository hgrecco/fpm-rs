# Reconstruction benchmarks

The benchmark framework runs existing reconstruction algorithms on an immutable
`ReconstructionProblem`. It does not add an algorithm registry or a second
synthetic-dataset abstraction.

Three deterministic simulator presets are available:

- `noiseless_mixed_fpm`: ideal mixed amplitude/phase acquisition;
- `aberrated_pupil_fpm`: known defocus and pupil-aberration mismatch;
- `poisson_gaussian_fpm`: shot noise, Gaussian read noise, detector offset,
  quantization, and a finite detector range.

`CameraModel` is separately validated with seeded distributional tests rather
than exact sample values: un-clipped Poisson draws match their expected mean and
variance, read-noise variance scales with the square of detector gain, and the
combined variance is `gain^2 * (mean_photoelectrons + read_noise_electrons^2)`.
These tests use 65,536 draws at signals far above the lower clamp. Clipping and
quantization are tested deterministically as a distinct detector behavior.

The normal regression suite also runs the diagnostics recorder against the
versioned noiseless preset. It checks cadence, Fourier coverage dimensions,
finite per-frame residuals, and decreasing recorded loss without treating
runtime or elapsed-time values as stable regression data.

Each returns the normal `SimulationResult`. Preset names end in a schema version
such as `_v1`; changing the physical definition requires a new preset version.

`run_benchmark_case` is generic over `ReconstructionAlgorithm` and
`MeasurementRead`. It returns a `BenchmarkRecord` plus the optional successful
`ReconstructionResult`. The record includes dimensions, original frame/source
indices, algorithm configuration, runtime, loss ratio, object/pupil errors when
ground truth exists, and per-frame residual summaries. Reconstruction and metric
failures are recorded rather than discarded.

The final argument to `run_benchmark_case` is an optional reconstruction-space
valid-object mask. When present, amplitude, phase, complex-field, and Fourier
metrics ignore pixels outside the mask. Public data without ground truth passes
`None`; per-frame intensity residuals remain available.

```rust,no_run
use fpm_rs::{
    Result,
    algorithms::AlternatingProjection,
    benchmark::run_benchmark_case,
    reconstruction::ReconstructionProblem,
    simulation::presets::{NOISELESS_MIXED_PRESET, noiseless_mixed_fpm},
};

fn main() -> Result<()> {
    let simulation = noiseless_mixed_fpm(123)?;
    let truth = simulation.ground_truth_object;
    let true_model = simulation.true_model;
    let problem = ReconstructionProblem::new(
        simulation.measurements,
        simulation.reconstruction_model,
    )?;
    let (mut record, result) = run_benchmark_case(
        "synthetic",
        "iterations=10,object_step=1.0",
        AlternatingProjection::default().iterations(10),
        &problem,
        Some(&truth),
        Some(&true_model),
        None,
    );
    record.preset_name = Some(NOISELESS_MIXED_PRESET.into());
    record.random_seed = Some(123);
    assert!(record.success && result.is_some());
    Ok(())
}
```

Use `write_benchmark_csv` and `write_benchmark_json` for summaries.
`save_benchmark_outputs` writes amplitude, phase, result bundle, and loss
history. Output stems contain a stable case hash so different algorithm
configurations do not overwrite one another.

Named profiles are metadata only; examples still choose concrete algorithms
directly instead of using an algorithm registry.

| Profile | Command | Algorithms | Expected runtime | Output |
|---|---|---|---|---|
| `smoke` | `cargo run --example benchmark_algorithms -- smoke` | AP, Fpie, Epry, ADMM, GradientDescent | Under 1 minute on a typical laptop CPU | `target/benchmark-results/smoke` |
| `cpu` | `cargo run --example benchmark_algorithms -- cpu` | AP, Fpie, Epry, ADMM, GradientDescent | 1-5 minutes on a typical laptop CPU | `target/benchmark-results/cpu` |

Run the default offline smoke profile with:

```sh
cargo run --example benchmark_algorithms
```

Converted-dataset benchmarks use the same API. When ground truth is unavailable,
pass `ground_truth: None`; normalized frame residuals remain available without
ground truth. The `load_local_dataset` example demonstrates this path.
