"""Matplotlib rendering helpers for reconstruction results and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .diagnostics.io import latest_frame_diagnostics

if TYPE_CHECKING:
    from ._core import ReconstructionResult


PlotResult = tuple[Any, dict[str, Any]]
"""Matplotlib figure and mapping of stable axis names to axes."""
DiagnosticsData = Mapping[str, Any]
"""Dictionary-like diagnostics payload returned by ``DiagnosticRecorder``."""

_RECONSTRUCTION_LAYOUT = (
    ("true_amplitude", "true_phase", "objective"),
    ("reconstructed_amplitude", "reconstructed_phase", "objective"),
)
_RECONSTRUCTION_IMAGE_AXES = (
    "true_amplitude",
    "true_phase",
    "reconstructed_amplitude",
    "reconstructed_phase",
)
_RECONSTRUCTION_REQUIRED_AXES = frozenset((*_RECONSTRUCTION_IMAGE_AXES, "objective"))


def plot_reconstruction(
    truth: np.ndarray,
    result: ReconstructionResult,
    *,
    layout: Sequence[Sequence[str]] | str | None = None,
    figsize: tuple[float, float] = (13.0, 7.0),
) -> PlotResult:
    """Plot ground truth, reconstruction, and objective history.

    ``truth`` is a complex 2D field matching ``result.object``. ``layout`` may
    be a Matplotlib subplot-mosaic specification containing the required named
    axes. Returns the figure and those axes; invalid shapes or layouts raise
    ``ValueError``.
    """
    plt = _import_pyplot()

    truth_array = np.asarray(truth)
    amplitude = np.asarray(result.amplitude)
    phase = np.asarray(result.phase)
    if truth_array.ndim != 2:
        raise ValueError(
            f"truth must be two-dimensional, got shape {truth_array.shape}"
        )
    if truth_array.size == 0:
        raise ValueError("truth must not be empty")
    if amplitude.shape != truth_array.shape or phase.shape != truth_array.shape:
        raise ValueError(
            "truth, reconstructed amplitude, and reconstructed phase must have "
            f"the same shape; got {truth_array.shape}, {amplitude.shape}, and "
            f"{phase.shape}"
        )

    figure, axes = plt.subplot_mosaic(
        _RECONSTRUCTION_LAYOUT if layout is None else layout,
        figsize=figsize,
        constrained_layout=True,
    )
    missing = _RECONSTRUCTION_REQUIRED_AXES.difference(axes)
    if missing:
        plt.close(figure)
        raise ValueError(
            f"layout is missing required axes: {', '.join(sorted(missing))}"
        )

    amplitude_max = max(float(np.abs(truth_array).max()), float(amplitude.max()))
    axes["true_amplitude"].imshow(
        np.abs(truth_array), cmap="gray", vmin=0.0, vmax=amplitude_max
    )
    axes["true_amplitude"].set_title("Original amplitude")
    axes["reconstructed_amplitude"].imshow(
        amplitude, cmap="gray", vmin=0.0, vmax=amplitude_max
    )
    axes["reconstructed_amplitude"].set_title("Reconstructed amplitude")

    axes["true_phase"].imshow(
        np.angle(truth_array), cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    axes["true_phase"].set_title("Original phase")
    axes["reconstructed_phase"].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes["reconstructed_phase"].set_title("Reconstructed phase")

    iterations = [record[0] for record in result.trace]
    objectives = [record[1] for record in result.trace]
    axes["objective"].semilogy(iterations, objectives, marker="o")
    axes["objective"].set(
        title="Reconstruction objective", xlabel="Iteration", ylabel="Objective"
    )
    axes["objective"].grid(True, which="both", alpha=0.3)

    for name in _RECONSTRUCTION_IMAGE_AXES:
        axes[name].set_axis_off()

    return figure, axes


def plot_convergence(
    diagnostics: DiagnosticsData,
    *,
    figsize: tuple[float, float] = (13.0, 9.0),
) -> PlotResult | None:
    """Plot objective, relative-change, frame, and timing histories.

    Returns the figure and named axes, or ``None`` when
    ``iteration_diagnostics`` is absent or empty.
    """
    history = diagnostics.get("iteration_diagnostics")
    if not isinstance(history, list) or not history:
        return None

    iterations = _series(history, "iteration")
    if np.isnan(iterations).all():
        iterations = np.arange(len(history), dtype=float)
    total_objective = _series(history, "total_objective")
    data_objective = _series(history, "data_objective")
    regularization_objective = _series(history, "regularization_objective")
    object_change = _series(history, "object_relative_change")
    pupil_change = _series(history, "pupil_relative_change")
    median_frame_objective = _series(history, "median_frame_objective")
    worst_frame_objective = _series(history, "worst_frame_objective")
    elapsed_seconds = _series(history, "elapsed_seconds")

    plt = _import_pyplot()
    figure, raw_axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    raw_axes = raw_axes.ravel()
    axes = {
        "objective": raw_axes[0],
        "relative_change": raw_axes[1],
        "frame_objective": raw_axes[2],
        "elapsed_time": raw_axes[3],
    }

    _plot_lines(
        axes["objective"],
        iterations,
        [
            ("total objective", total_objective),
            ("data objective", data_objective),
            ("regularization objective", regularization_objective),
        ],
        title="Objective history",
        xlabel="Iteration",
        ylabel="Objective",
        log_scale=True,
    )
    _plot_lines(
        axes["relative_change"],
        iterations,
        [("object change", object_change), ("pupil change", pupil_change)],
        title="Relative change",
        xlabel="Iteration",
        ylabel="Relative change",
        log_scale=True,
    )
    _plot_lines(
        axes["frame_objective"],
        iterations,
        [
            ("median frame objective", median_frame_objective),
            ("worst frame objective", worst_frame_objective),
        ],
        title="Per-frame objective",
        xlabel="Iteration",
        ylabel="Objective",
        log_scale=True,
    )
    _plot_lines(
        axes["elapsed_time"],
        iterations,
        [("elapsed seconds", elapsed_seconds)],
        title="Elapsed time",
        xlabel="Iteration",
        ylabel="Seconds",
        log_scale=False,
    )
    return figure, axes


def plot_frame_residuals(
    diagnostics: DiagnosticsData,
    *,
    figsize: tuple[float, float] = (13.0, 9.0),
) -> PlotResult | None:
    """Plot residual summaries from the latest recorded iteration.

    Returns the figure and named axes, or ``None`` when no frame diagnostics
    are available.
    """
    frame_diagnostics = latest_frame_diagnostics(diagnostics.get("frame_diagnostics"))
    if not frame_diagnostics:
        return None

    illumination_index = _series(frame_diagnostics, "illumination_index")
    if np.isnan(illumination_index).all():
        illumination_index = _series(frame_diagnostics, "frame_index")
    residual_l2 = _series(frame_diagnostics, "residual_l2")
    normalized_l2 = _series(frame_diagnostics, "normalized_l2")
    residual_mean = _series(frame_diagnostics, "residual_mean")
    residual_std = _series(frame_diagnostics, "residual_std")
    residual_max_abs = _series(frame_diagnostics, "residual_max_abs")
    reference_sum = _series(frame_diagnostics, "reference_sum")
    estimate_sum = _series(frame_diagnostics, "estimate_sum")

    plt = _import_pyplot()
    figure, raw_axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    raw_axes = raw_axes.ravel()
    axes = {
        "residual_magnitude": raw_axes[0],
        "residual_moments": raw_axes[1],
        "frame_sums": raw_axes[2],
        "reference_vs_estimate": raw_axes[3],
    }

    _plot_lines(
        axes["residual_magnitude"],
        illumination_index,
        [("residual L2", residual_l2), ("normalized L2", normalized_l2)],
        title="Residual magnitude",
        xlabel="Illumination index",
        ylabel="Residual",
        log_scale=True,
    )
    _plot_lines(
        axes["residual_moments"],
        illumination_index,
        [
            ("residual mean", residual_mean),
            ("residual std", residual_std),
            ("residual max abs", residual_max_abs),
        ],
        title="Residual moments",
        xlabel="Illumination index",
        ylabel="Value",
        log_scale=False,
    )
    _plot_lines(
        axes["frame_sums"],
        illumination_index,
        [("reference sum", reference_sum), ("estimate sum", estimate_sum)],
        title="Frame sums",
        xlabel="Illumination index",
        ylabel="Sum",
        log_scale=False,
    )
    finite_mask = np.isfinite(reference_sum) & np.isfinite(estimate_sum)
    if np.any(finite_mask):
        scatter = axes["reference_vs_estimate"].scatter(
            reference_sum[finite_mask],
            estimate_sum[finite_mask],
            c=normalized_l2[finite_mask],
            cmap="viridis",
            edgecolor="none",
        )
        minimum = float(
            np.nanmin(
                [reference_sum[finite_mask].min(), estimate_sum[finite_mask].min()]
            )
        )
        maximum = float(
            np.nanmax(
                [reference_sum[finite_mask].max(), estimate_sum[finite_mask].max()]
            )
        )
        axes["reference_vs_estimate"].plot(
            [minimum, maximum],
            [minimum, maximum],
            color="0.4",
            linestyle="--",
            linewidth=1.0,
        )
        figure.colorbar(
            scatter,
            ax=axes["reference_vs_estimate"],
            label="Normalized L2",
        )
    axes["reference_vs_estimate"].set(
        title="Reference vs estimate sums",
        xlabel="Reference sum",
        ylabel="Estimate sum",
    )
    axes["reference_vs_estimate"].grid(True, alpha=0.25)
    return figure, axes


def plot_raw_stack_stats(
    diagnostics: DiagnosticsData,
    *,
    figsize: tuple[float, float] = (13.0, 9.0),
) -> PlotResult | None:
    """Plot per-frame mean, spread, extrema, saturation, and zero counts.

    Returns the figure and named axes, or ``None`` when
    ``raw_frame_statistics`` is absent or empty.
    """
    raw_stats = diagnostics.get("raw_frame_statistics")
    if not isinstance(raw_stats, list) or not raw_stats:
        return None

    frame_index = _series(raw_stats, "frame_index")
    if np.isnan(frame_index).all():
        frame_index = np.arange(len(raw_stats), dtype=float)
    mean = _series(raw_stats, "mean")
    std = _series(raw_stats, "std")
    minimum = _series(raw_stats, "min")
    maximum = _series(raw_stats, "max")
    saturated = _series(raw_stats, "saturated_pixels")
    zero_pixels = _series(raw_stats, "zero_pixels")

    plt = _import_pyplot()
    figure, raw_axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    raw_axes = raw_axes.ravel()
    axes = {
        "mean_std": raw_axes[0],
        "min_max": raw_axes[1],
        "saturated_pixels": raw_axes[2],
        "zero_pixels": raw_axes[3],
    }

    _plot_lines(
        axes["mean_std"],
        frame_index,
        [("mean", mean), ("std", std)],
        title="Raw-frame mean and std",
        xlabel="Frame index",
        ylabel="Value",
        log_scale=False,
    )
    _plot_lines(
        axes["min_max"],
        frame_index,
        [("min", minimum), ("max", maximum)],
        title="Raw-frame min and max",
        xlabel="Frame index",
        ylabel="Value",
        log_scale=False,
    )
    _plot_lines(
        axes["saturated_pixels"],
        frame_index,
        [("saturated pixels", saturated)],
        title="Saturated pixels",
        xlabel="Frame index",
        ylabel="Pixels",
        log_scale=False,
    )
    _plot_lines(
        axes["zero_pixels"],
        frame_index,
        [("zero pixels", zero_pixels)],
        title="Zero pixels",
        xlabel="Frame index",
        ylabel="Pixels",
        log_scale=False,
    )
    return figure, axes


def plot_fourier_coverage(
    diagnostics: DiagnosticsData,
    *,
    figsize: tuple[float, float] = (8.5, 8.5),
) -> PlotResult | None:
    """Plot shifted pupil disks and illumination NA in Fourier-pixel coordinates.

    Returns the figure and a ``{"coverage": axis}`` mapping, or ``None`` when
    coverage centers or the positive pupil radius are unavailable.
    """
    coverage = diagnostics.get("coverage")
    if not isinstance(coverage, dict):
        return None
    centers = _array2d(coverage.get("pupil_centers_px"))
    radius = _float_or_nan(coverage.get("pupil_radius_px"))
    if centers.size == 0 or not np.isfinite(radius) or radius <= 0.0:
        return None

    illumination_na = _series_from_sequence(coverage.get("illumination_na"))
    synthetic_na = _float_or_nan(coverage.get("synthetic_na"))

    plt = _import_pyplot()
    from matplotlib.patches import Circle

    figure, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    centers_x = centers[:, 0]
    centers_y = centers[:, 1]
    if illumination_na.size == centers.shape[0] and np.isfinite(illumination_na).any():
        scatter = ax.scatter(
            centers_x,
            centers_y,
            c=illumination_na,
            cmap="viridis",
            s=30,
            zorder=3,
        )
        figure.colorbar(scatter, ax=ax, label="Illumination NA")
    else:
        ax.scatter(centers_x, centers_y, color="C0", s=30, zorder=3)

    for center_x, center_y in centers:
        ax.add_patch(
            Circle((center_x, center_y), radius, fill=False, edgecolor="C1", alpha=0.35)
        )

    if np.isfinite(synthetic_na):
        ax.set_title(f"Fourier coverage (synthetic NA: {synthetic_na:.4g})")
    else:
        ax.set_title("Fourier coverage")
    ax.set_xlabel("Fourier x (px)")
    ax.set_ylabel("Fourier y (px)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    _set_limits_from_centers(ax, centers, radius)
    return figure, {"coverage": ax}


def plot_crop_indices(
    diagnostics: DiagnosticsData,
    *,
    figsize: tuple[float, float] = (8.5, 8.5),
) -> PlotResult | None:
    """Plot compiled Fourier crop boxes and centers in pixel coordinates.

    Returns the figure and a ``{"crop_indices": axis}`` mapping, or ``None``
    when no valid crop records are available.
    """
    coverage = diagnostics.get("coverage")
    if not isinstance(coverage, dict):
        return None
    crop_indices = coverage.get("crop_indices")
    if not isinstance(crop_indices, list) or not crop_indices:
        return None

    plt = _import_pyplot()
    from matplotlib.patches import Rectangle

    figure, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    centers_x: list[float] = []
    centers_y: list[float] = []
    x_bounds: list[float] = []
    y_bounds: list[float] = []
    for crop in crop_indices:
        if not isinstance(crop, dict):
            continue
        x_start = _float_or_nan(crop.get("x_start"))
        y_start = _float_or_nan(crop.get("y_start"))
        x_end = _float_or_nan(crop.get("x_end"))
        y_end = _float_or_nan(crop.get("y_end"))
        center_x = _float_or_nan(crop.get("center_x"))
        center_y = _float_or_nan(crop.get("center_y"))
        if not np.isfinite([x_start, y_start, x_end, y_end, center_x, center_y]).all():
            continue
        ax.add_patch(
            Rectangle(
                (x_start, y_start),
                x_end - x_start,
                y_end - y_start,
                fill=False,
                edgecolor="C0",
                alpha=0.5,
            )
        )
        centers_x.append(center_x)
        centers_y.append(center_y)
        x_bounds.extend([x_start, x_end])
        y_bounds.extend([y_start, y_end])
    if not centers_x:
        plt.close(figure)
        return None
    ax.scatter(centers_x, centers_y, color="C1", s=25, zorder=3)
    ax.set_title("Crop indices")
    ax.set_xlabel("Fourier x (px)")
    ax.set_ylabel("Fourier y (px)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    _set_limits_from_bounds(ax, np.asarray(x_bounds), np.asarray(y_bounds))
    return figure, {"crop_indices": ax}


def plot_residuals_on_fourier_centers(
    diagnostics: DiagnosticsData,
    *,
    figsize: tuple[float, float] = (8.5, 8.5),
) -> PlotResult | None:
    """Map latest normalized frame residuals onto illumination Fourier centers.

    Returns the figure and a named axis, or ``None`` when coverage or matching
    per-frame residuals are unavailable.
    """
    coverage = diagnostics.get("coverage")
    frame_diagnostics = latest_frame_diagnostics(diagnostics.get("frame_diagnostics"))
    if not isinstance(coverage, dict) or not frame_diagnostics:
        return None
    centers = _array2d(coverage.get("pupil_centers_px"))
    if centers.size == 0:
        return None
    residual_by_illumination: dict[int, float] = {}
    for entry in frame_diagnostics:
        if not isinstance(entry, dict):
            continue
        illumination_index = entry.get("illumination_index")
        residual = _float_or_nan(entry.get("normalized_l2"))
        if isinstance(illumination_index, int) and np.isfinite(residual):
            residual_by_illumination[illumination_index] = residual
    if not residual_by_illumination:
        return None

    xs = []
    ys = []
    colors = []
    for illumination_index, residual in residual_by_illumination.items():
        if 0 <= illumination_index < centers.shape[0]:
            xs.append(float(centers[illumination_index, 0]))
            ys.append(float(centers[illumination_index, 1]))
            colors.append(float(residual))
    if not xs:
        return None

    plt = _import_pyplot()
    figure, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    scatter = ax.scatter(xs, ys, c=colors, cmap="magma", s=35)
    figure.colorbar(scatter, ax=ax, label="Normalized L2 residual")
    ax.set_title("Residuals on Fourier centers")
    ax.set_xlabel("Fourier x (px)")
    ax.set_ylabel("Fourier y (px)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    _set_limits_from_points(ax, np.asarray(xs), np.asarray(ys))
    return figure, {"residuals_on_fourier_centers": ax}


def _import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name == "matplotlib":
            raise ModuleNotFoundError(
                "plotting helpers require Matplotlib; install "
                "fpm-rs[plot] or fpm-rs[notebook]"
            ) from error
        raise
    return plt


def _plot_lines(
    ax: Any,
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    log_scale: bool,
) -> None:
    finite_series = False
    for label, values in series:
        if values.size == 0:
            continue
        ax.plot(x, values, marker="o", linewidth=1.2, markersize=3.5, label=label)
        if np.isfinite(values).any():
            finite_series = True
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if log_scale and finite_series and _all_positive(series):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    if len(series) > 1:
        ax.legend(frameon=False)


def _all_positive(series: list[tuple[str, np.ndarray]]) -> bool:
    finite_values: list[np.ndarray] = []
    for _, values in series:
        if values.size:
            filtered = values[np.isfinite(values)]
            if filtered.size:
                finite_values.append(filtered)
    if not finite_values:
        return False
    concatenated = np.concatenate(finite_values)
    return bool(np.all(concatenated > 0.0))


def _series(entries: list[Any], key: str) -> np.ndarray:
    return np.asarray(
        [
            _float_or_nan(entry.get(key) if isinstance(entry, dict) else None)
            for entry in entries
        ],
        dtype=float,
    )


def _series_from_sequence(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        return np.asarray([], dtype=float)
    return np.asarray([_float_or_nan(value) for value in values], dtype=float)


def _array2d(values: Any) -> np.ndarray:
    if not isinstance(values, list) or not values:
        return np.asarray([], dtype=float).reshape(0, 2)
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        return np.asarray([], dtype=float).reshape(0, 2)
    return array


def _float_or_nan(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _set_limits_from_centers(ax: Any, centers: np.ndarray, radius: float) -> None:
    xs = centers[:, 0]
    ys = centers[:, 1]
    x_min = float(np.nanmin(xs) - radius)
    x_max = float(np.nanmax(xs) + radius)
    y_min = float(np.nanmin(ys) - radius)
    y_max = float(np.nanmax(ys) + radius)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _set_limits_from_points(ax: Any, xs: np.ndarray, ys: np.ndarray) -> None:
    x_min = float(np.nanmin(xs))
    x_max = float(np.nanmax(xs))
    y_min = float(np.nanmin(ys))
    y_max = float(np.nanmax(ys))
    pad_x = 0.05 * max(x_max - x_min, 1.0)
    pad_y = 0.05 * max(y_max - y_min, 1.0)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)


def _set_limits_from_bounds(ax: Any, xs: np.ndarray, ys: np.ndarray) -> None:
    if xs.size == 0 or ys.size == 0:
        return
    x_min = float(np.nanmin(xs))
    x_max = float(np.nanmax(xs))
    y_min = float(np.nanmin(ys))
    y_max = float(np.nanmax(ys))
    pad_x = 0.05 * max(x_max - x_min, 1.0)
    pad_y = 0.05 * max(y_max - y_min, 1.0)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)


__all__ = [
    "PlotResult",
    "plot_convergence",
    "plot_crop_indices",
    "plot_frame_residuals",
    "plot_fourier_coverage",
    "plot_raw_stack_stats",
    "plot_reconstruction",
    "plot_residuals_on_fourier_centers",
]
