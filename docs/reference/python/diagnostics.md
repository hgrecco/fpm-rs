# Callbacks and diagnostics

## Run callbacks

::: fpm_rs.ProgressLogger

::: fpm_rs.CheckpointEvery

::: fpm_rs.CsvLogger

::: fpm_rs.StopOnPlateau

::: fpm_rs.SaveImageEvery

::: fpm_rs.SavePupilEvery

::: fpm_rs.SaveResidualsEvery

::: fpm_rs.IterationCallback

::: fpm_rs.DiagnosticRecorder

## Diagnostic I/O and reports

::: fpm_rs.diagnostics
    options:
      members:
        - ensure_output_dir
        - coerce_diagnostics
        - load_diagnostics
        - savefig
        - make_diagnostic_report
        - write_ground_truth_metrics
        - write_summary
      show_root_toc_entry: false

Plotting functions import Matplotlib lazily and return `(figure, axes)`. Install
the `plot` extra before using them. Diagnostic I/O accepts either a recorder
dictionary or a JSON path where documented.

::: fpm_rs.plot
    options:
      members:
        - PlotResult
        - plot_reconstruction
        - plot_convergence
        - plot_frame_residuals
        - plot_raw_stack_stats
        - plot_fourier_coverage
        - plot_crop_indices
        - plot_residuals_on_fourier_centers
      show_root_toc_entry: false
