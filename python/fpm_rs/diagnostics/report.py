from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..plot import (
    plot_convergence,
    plot_crop_indices,
    plot_frame_residuals,
    plot_fourier_coverage,
    plot_raw_stack_stats,
    plot_residuals_on_fourier_centers,
)
from .io import coerce_diagnostics, ensure_output_dir, latest_frame_diagnostics


def write_ground_truth_metrics(diag: dict[str, Any], output_dir: str | Path) -> None:
    metrics = diag.get("ground_truth_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return
    output_path = ensure_output_dir(output_dir) / "ground_truth_metrics.txt"
    lines = []
    for key in (
        "amplitude_rmse",
        "amplitude_nrmse",
        "complex_rmse",
        "complex_nrmse",
        "phase_rmse",
        "phase_mae",
        "fourier_nrmse",
    ):
        value = metrics.get(key)
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_summary(diag: dict[str, Any], output_dir: str | Path) -> None:
    output_path = ensure_output_dir(output_dir) / "summary.txt"
    iteration_diagnostics = diag.get("iteration_diagnostics")
    all_frame_diagnostics = diag.get("frame_diagnostics")
    frame_diagnostics = latest_frame_diagnostics(all_frame_diagnostics)
    raw_frame_statistics = diag.get("raw_frame_statistics")
    coverage = diag.get("coverage")
    ground_truth_metrics = diag.get("ground_truth_metrics")

    lines = [
        f"recorded_iterations: {len(iteration_diagnostics) if isinstance(iteration_diagnostics, list) else 0}",
        f"frame_diagnostics: {len(all_frame_diagnostics) if isinstance(all_frame_diagnostics, list) else 0}",
        f"raw_frame_statistics: {len(raw_frame_statistics) if isinstance(raw_frame_statistics, list) else 0}",
        f"coverage_available: {bool(isinstance(coverage, dict) and coverage)}",
        f"ground_truth_metrics_available: {bool(isinstance(ground_truth_metrics, dict) and ground_truth_metrics)}",
    ]

    if isinstance(iteration_diagnostics, list) and iteration_diagnostics:
        last = iteration_diagnostics[-1]
        if isinstance(last, dict):
            lines.append("last_iteration:")
            for key in (
                "iteration",
                "total_objective",
                "data_objective",
                "regularization_objective",
                "object_relative_change",
                "pupil_relative_change",
                "median_frame_objective",
                "worst_frame_objective",
                "elapsed_seconds",
            ):
                value = last.get(key)
                if value is not None:
                    lines.append(f"  {key}: {value}")

    worst_frame = _worst_frame(frame_diagnostics)
    if worst_frame is not None:
        frame_index, illumination_index, value = worst_frame
        lines.append("worst_frame:")
        lines.append(f"  frame_index: {frame_index}")
        lines.append(f"  illumination_index: {illumination_index}")
        lines.append(f"  normalized_l2: {value}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_diagnostic_report(
    diagnostics: Mapping[str, Any] | str | Path,
    output_dir: str | Path = "diagnostic_report",
) -> None:
    diagnostics = coerce_diagnostics(diagnostics)
    output_path = ensure_output_dir(output_dir)
    for filename, plotter in (
        ("convergence.png", plot_convergence),
        ("frame_residuals.png", plot_frame_residuals),
        ("raw_stack_stats.png", plot_raw_stack_stats),
        ("fourier_coverage.png", plot_fourier_coverage),
        ("crop_indices.png", plot_crop_indices),
        ("residuals_on_fourier_centers.png", plot_residuals_on_fourier_centers),
    ):
        result = plotter(diagnostics)
        if result is not None:
            result[0].savefig(output_path / filename, dpi=180, bbox_inches="tight")
            _close_figure(result[0])
    write_ground_truth_metrics(diagnostics, output_dir)
    write_summary(diagnostics, output_dir)


def _worst_frame(frame_diagnostics: Any) -> tuple[int, int, float] | None:
    if not isinstance(frame_diagnostics, list):
        return None
    worst: tuple[int, int, float] | None = None
    for entry in frame_diagnostics:
        if not isinstance(entry, dict):
            continue
        value = entry.get("normalized_l2")
        frame_index = entry.get("frame_index")
        illumination_index = entry.get("illumination_index")
        if value is None or frame_index is None or illumination_index is None:
            continue
        try:
            normalized_l2 = float(value)
            frame_index = int(frame_index)
            illumination_index = int(illumination_index)
        except (TypeError, ValueError):
            continue
        if worst is None or normalized_l2 > worst[2]:
            worst = (frame_index, illumination_index, normalized_l2)
    return worst


def _close_figure(figure: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)
