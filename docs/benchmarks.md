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
finite per-frame residuals, and a decreasing recorded objective without treating
runtime or elapsed-time values as stable regression data.

Each returns the normal `SimulationResult`. Preset names end in a schema version
such as `_v1`; changing the physical definition requires a new preset version.

`run_benchmark_case` is generic over `ReconstructionAlgorithm` and
`MeasurementRead`. It returns a `BenchmarkRecord` plus the optional successful
`ReconstructionResult`. The record includes dimensions, original frame/source
indices, algorithm configuration, elapsed seconds, objective ratio, object/pupil errors when
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
`save_benchmark_outputs` writes amplitude, phase, a result bundle, and objective
history CSV. Output stems contain a stable case hash so different algorithm
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

## Normalized benchmark bundles

A `BenchmarkRecord` has two identities: `case_id` identifies an immutable case
configuration, while `run_id` is unique for each execution. Repetitions of one
case share `case_id` and have distinct `run_id` values. Frame metrics are
normalized as one row per `(run_id, frame_index)` rather than stored as
parallel lists.

With the `parquet` feature, `write_benchmark_bundle` writes four stable tables:
`runs`, `frames`, `artifacts`, and long-form `metadata`. Every successful run
also has one nested `ResultBundle` under `results/<run_id>`; arrays are not
duplicated in benchmark-level storage. Shared run columns have the same names
and dtypes as result summary tables, so joins are direct.

Python can build a comparison from existing results:

```python
suite = fpm.BenchmarkSuite("algorithm-comparison")
run_id = suite.add_result(
    result,
    case_id="synthetic-v1-seed-17",
    dataset_name="synthetic",
    algorithm_configuration="iterations=20,object_step=1.0",
)
benchmark = suite.write_bundle("output/comparison", label="AP repeats")
reopened = fpm.read_benchmark_bundle(benchmark.path)
result_bundle = reopened.results[run_id]
```

Install `fpm-rs[polars]` to query the ordinary Parquet paths:

```python
import polars as pl

runs = pl.scan_parquet(benchmark.tables.runs.path)
frames = pl.scan_parquet(benchmark.tables.frames.path)

comparison = (
    runs.filter(pl.col("case_id") == "synthetic-v1-seed-17")
    .select("run_id", "algorithm", "elapsed_seconds", "final_objective")
    .collect()
)

per_frame = (
    frames.join(runs.select("run_id", "algorithm"), on="run_id")
    .group_by("algorithm")
    .agg(pl.col("normalized_l2").mean())
    .collect()
)
```

Result summaries can be joined through the artifacts table or read from the
nested bundles:

```python
runs_eager = pl.read_parquet(benchmark.tables.runs.path)
summaries = pl.concat(
    [
        pl.read_parquet(
            benchmark.results[run_id].tables.summary.path
        )
        for run_id in runs_eager["run_id"]
    ]
)
run_summaries = runs_eager.join(
    summaries,
    on="run_id",
    how="left",
    suffix="_result",
)

one_case = (
    runs_eager.filter(
        (pl.col("case_id") == "synthetic-v1-seed-17")
        & pl.col("success")
    )
    .select(
        "case_id",
        "algorithm",
        "elapsed_seconds",
        "completed_iterations",
        "final_objective",
    )
    .sort(["case_id", "final_objective"])
)

repeats = (
    runs_eager.filter(pl.col("success"))
    .group_by(["case_id", "algorithm"])
    .agg(
        pl.len().alias("run_count"),
        pl.col("elapsed_seconds").mean().alias("mean_elapsed_seconds"),
        pl.col("elapsed_seconds").std().alias("std_elapsed_seconds"),
        pl.col("final_objective").mean().alias("mean_final_objective"),
    )
)
```

For frame-level analysis, join `frames` to `runs` on `run_id`, then retain
`case_id`, `algorithm`, and `dataset_name` from the run table. The Python
extension does not import Polars and does not define a second DataFrame wrapper.

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

### Array/trace refactor validation snapshot

The ndarray, typed-trace, and bundle refactor was compared with its clean
pre-refactor `HEAD` on the same machine and toolchain. Debug forward-model
measurements over 500 evaluations changed from 2265.149 to 2244.738 µs/frame
for the allocating path, 2140.048 to 2135.985 µs/frame with a reused workspace,
and 600.479 to 607.455 µs/frame for the eight-worker stack. The largest absolute
change was 1.2%.

Ten-sample 32×32/64×64 single-worker gradient steps changed as follows:

| Case | Before (ms/step) | After (ms/step) | Change |
|---|---:|---:|---:|
| Ordinary object | 8.847 | 8.882 | +0.4% |
| Multiplexed object | 11.491 | 11.615 | +1.1% |
| Multiplexed object and pupil | 12.386 | 12.922 | +4.3% |
| Multiplexed object and illumination | 29.072 | 29.041 | -0.1% |

Peak incremental heap values were unchanged in every worker configuration. The
validation threshold was a repeatable 10% regression in representative serial
paths or an unexplained increase in incremental heap; neither occurred.
Multi-worker timings remain scheduler-sensitive and are retained as descriptive
output rather than a release gate.
