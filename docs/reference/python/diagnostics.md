# Plotting and diagnostic reports

Plotting functions import Matplotlib lazily and return `(figure, axes)`. Install
the `plot` extra before using them. Diagnostic I/O accepts either a recorder
dictionary or a JSON path where documented.

::: fpm_rs.plot
    options:
      members:
        - plot_reconstruction
        - plot_convergence
        - plot_frame_residuals
        - plot_raw_stack_stats
        - plot_fourier_coverage
        - plot_crop_indices
        - plot_residuals_on_fourier_centers
      show_root_toc_entry: false
      show_source: true

::: fpm_rs.diagnostics
    options:
      members: true
      show_root_toc_entry: false
      show_source: true
