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

## Forward-model and gradient scaling benchmarks

`ForwardModel::forward_intensity` is the allocation-owning convenience API.
Repeated simulation, metrics, or parameter searches can reuse mutable scratch
from `ForwardModel::workspace` through `forward_intensity_into` or
`forward_source_field_into`. A workspace must not be shared concurrently;
create one per worker. FFT plans and backend objects remain shareable.

`forward_intensity_stack_into` evaluates complete stacks with scoped CPU workers
and preserves `[frame][row][column]` order. `Simulator` parallelizes optical
prediction, then applies `CameraModel` serially so seeded detector noise is
independent of worker count.

Run the dependency-free forward benchmark with:

```sh
cargo bench --bench forward_model
```

Set `FPM_BENCH_ITERATIONS` to change its duration. It compares allocation with
workspace reuse but has no machine-specific pass/fail threshold.

The gradient scaling and memory benchmark is:

```sh
cargo bench --bench gradient_parallel
```

It covers ordinary and multiplexed object, pupil, and illumination updates.
For each worker count it reports milliseconds per batch step, speedup relative
to one worker, and peak incremental heap for the step. Configure it with
`FPM_GRADIENT_BENCH_LOW_SIZE`, `FPM_GRADIENT_BENCH_HIGH_SIZE`,
`FPM_GRADIENT_BENCH_ITERATIONS`, and `FPM_GRADIENT_BENCH_MAX_WORKERS`. The heap
measure includes worker-local state and reduction buffers, but excludes existing
reconstruction state, native thread stacks, and system FFT/allocator memory.

Use one worker for single-frame or tiny batches. For larger CPU batches, start
with two to four workers and benchmark the actual image and multiplexing sizes:
worker-local high-resolution accumulators make memory grow roughly linearly and
thread overhead can dominate. A 20-sample 32×32/64×64 development run measured
1.89× ordinary-update speedup with four workers (1.49 MiB incremental heap,
versus 0.06 MiB serial); eight workers reached 1.91× with 2.26 MiB. These are
illustrative measurements, not portable guarantees.
