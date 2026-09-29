# Python diagnostics workflow

Reconstruction diagnostics are collected by a Rust callback and returned to
Python as a dictionary shaped like the Rust `ReconstructionDiagnostics` model.
Recording does not invoke Python or acquire the GIL inside the reconstruction
loop.

Reusable calculations are split by responsibility. Rust `metrics` contains
domain-agnostic reference/estimate comparisons; `algorithms::objective`
contains objective functions optimized by reconstruction algorithms; and
`evaluation` combines a reconstruction with optional reference model and
measurement stack.
Python exposes the direct equivalents under `fpm.metrics` and
`fpm.evaluation`. This module focuses only on run-scoped diagnostic requests,
Fourier coverage, convergence records, recorder cadence, and serialization.
See [intensity metrics](metrics.md) for standalone reference/estimate image
calculations such as NRMSE, PSNR, SSIM, and Poisson deviance.

Every reconstruction records `result.trace` independently of this optional
recorder. Its one-based tuples contain `(iteration, objective,
elapsed_seconds)`, where seconds are measured from the start of the run and
continue across checkpoint resume. Recorder iteration diagnostics are optional
derived values at the configured cadence and are not the execution trace.

Reusable metrics, recorder dictionaries, diagnostic JSON, and tabular exports
all use `reference_sum` and `estimate_sum`.

## End-to-end report

```python
import fpm_rs as fpm

recorder = fpm.DiagnosticRecorder("basic")
result = algorithm.run(problem, callbacks=[recorder])

diagnostics = recorder.diagnostics()
fpm.diagnostics.make_diagnostic_report(
    diagnostics,
    output_dir="diagnostic_report",
)
```

`result.scalar_diagnostics` is the reconstruction result's small, dynamic
scalar summary.
The structured recorder output is retrieved with `recorder.diagnostics()`.

## Radial Fourier power spectrum

`fpm.radial_fourier_spectrum(field)` computes a forward two-dimensional FFT
of a complex field and averages its normalized power in integer-radius annuli.
It returns a dictionary containing `radius_px`, `power`, and `sample_count`.
The radii are Fourier-grid pixels; convert them to physical spatial frequency
only when the input sampling pitch is known.

Algorithms can additionally expose long-form `result.algorithm_metrics` tuples:
`(iteration, namespace, metric, value)`. ADMM uses the `admm` namespace for
`primal_residual_rms` and `dual_residual_rms`. Global Gauss–Newton uses
`global_gauss_newton` for conjugate-gradient iterations and residual ratio,
line-search evaluations and accepted scale, and the pre-update gradient norm.
These values do not appear in universal iteration rows or scalar diagnostics.
Algorithms without stable algorithm-specific metrics return an empty list.

Joint physical illumination runs emit `data_loss`, `regularization_loss`,
`total_loss`, geometry-refresh count, intensity-only-update count, and rejected
step count in the `physical_illumination` namespace. Callback contexts use the
refreshed calibrated model. Rust callbacks can inspect absolute parameters and
conditioning warnings through
`context.state.physical_illumination_calibration()`; Python iteration callbacks
receive the scalar phase records in `algorithm_metrics` and the complete typed
state snapshot in `physical_illumination_calibration`.

The report writer creates plots only for sections present in the diagnostics.
It always writes `summary.txt`. No intermediate JSON file is required.

## Recorder modes

| Mode | Iteration diagnostics | Fourier coverage | Frame summaries | Raw-stack statistics |
| --- | --- | --- | --- | --- |
| `minimal` | yes | no | no | no |
| `basic` | yes | yes | no | no |
| `debug` | yes | yes | yes | yes |
| `simulation` | yes | yes | no | no |

`basic` is the default. Frame summaries require an additional calibrated
forward pass over the measurement stack at each recording interval, so they
are restricted to `debug`. Object and pupil snapshots are disabled in every
Python preset.

Use `every` to reduce the recording cadence:

```python
recorder = fpm.DiagnosticRecorder("debug", every=5)
```

The recorder clears its previous values when a new reconstruction starts, so
reusing an object produces diagnostics for the latest run rather than combining
unrelated runs. Clones share that state: retain one clone for reading while one
runner owns another, but do not install clones in simultaneous runs. The
recorder deliberately recovers a poisoned internal lock as best-effort
diagnostic state; the next run resets it before recording.

## Individual plots

The dictionary returned by the recorder can be passed directly to every plot.
All plotting entry points now live under `fpm.plot`. Like
`plot_reconstruction`, each function returns its Matplotlib figure and a
dictionary of named axes:

```python
diagnostics = recorder.diagnostics()

figure, axes = fpm.plot.plot_convergence(diagnostics)
axes["objective"].set_title("Experiment convergence")
```

Reconstruction plots live in the same namespace:

```python
figure, axes = fpm.plot.plot_fourier_coverage(diagnostics)
figure, axes = fpm.plot.plot_reconstruction(truth, result)
```

The plotting helpers only build figures. Save them explicitly when needed:

```python
figure, axes = fpm.plot.plot_frame_residuals(diagnostics)
figure.savefig("diagnostic_report/frame_residuals.png", dpi=180, bbox_inches="tight")
```

Unavailable sections are skipped without error. Matplotlib is imported only
when plotting is requested; install the `plot` optional dependency when using
these functions. Direct callers own returned figures and should close them when
finished. `make_diagnostic_report` saves and closes its figures automatically.

## Optional JSON persistence

JSON is optional and contains the same structured diagnostic model:

```python
recorder.to_json("diagnostics.json")

diagnostics = fpm.diagnostics.load_diagnostics("diagnostics.json")
fpm.diagnostics.make_diagnostic_report(
    diagnostics,
    output_dir="diagnostic_report_from_json",
)
```

The report function also accepts the JSON path directly. Object and pupil
snapshots are not exported.

One maintained site tutorial and two focused repository notebooks provide the
same workflow with inline plot display:

- [`diagnostics_quickstart.ipynb`](tutorials/notebooks/diagnostics_quickstart.ipynb)
  is the published tutorial for the lightweight `basic` recorder and overview plots.
- `python/examples/diagnostics_debug.ipynb` focuses on debug-only frame and
  raw-stack inspection.
- `python/examples/diagnostics_plots.ipynb` demonstrates the individual
  plotting helpers and axes customization.

Each notebook displays figures inline and includes commented `savefig(...)`
calls that can be uncommented when file export is desired.

Launch the focused examples from the repository environment with
`pixi run jupyter lab`.
