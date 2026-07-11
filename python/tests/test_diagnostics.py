from __future__ import annotations

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

import fpm_rs as fpm
from fpm_rs.diagnostics import load_diagnostics, make_diagnostic_report, write_summary


def test_load_diagnostics_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.json"
    path.write_text(
        json.dumps({"iteration_history": [{"iteration": 1, "total_loss": 1.0}]}),
        encoding="utf-8",
    )

    diagnostics = load_diagnostics(path)

    assert diagnostics["iteration_history"][0]["iteration"] == 1


def test_make_diagnostic_report_handles_minimal_iteration_history(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "iteration_history": [
                    {
                        "iteration": 1,
                        "total_loss": 1.0,
                        "data_loss": 1.0,
                        "regularization_loss": None,
                        "object_relative_change": None,
                        "pupil_relative_change": None,
                        "median_frame_loss": None,
                        "worst_frame_loss": None,
                        "elapsed_ms": 1.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "report"
    make_diagnostic_report(diagnostics_path, output_dir)

    assert (output_dir / "convergence.png").is_file()
    assert (output_dir / "summary.txt").is_file()
    assert not (output_dir / "frame_residuals.png").exists()


def test_summary_uses_latest_frame_diagnostic_iteration(tmp_path: Path) -> None:
    diagnostics = {
        "frame_diagnostics": [
            {
                "iteration": 1,
                "frame_index": 0,
                "illumination_index": 0,
                "normalized_l2": 100.0,
            },
            {
                "iteration": 2,
                "frame_index": 0,
                "illumination_index": 0,
                "normalized_l2": 1.0,
            },
        ]
    }

    write_summary(diagnostics, tmp_path)

    summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "frame_diagnostics: 2" in summary
    assert "normalized_l2: 1.0" in summary
    assert "normalized_l2: 100.0" not in summary


def test_diagnostic_recorder_presets_validate_inputs() -> None:
    default = fpm.DiagnosticRecorder()
    assert default.mode == "basic"
    assert default.every == 1
    assert fpm.DiagnosticRecorder("minimal").mode == "minimal"
    assert fpm.DiagnosticRecorder("basic", every=5).every == 5
    assert fpm.DiagnosticRecorder("debug").mode == "debug"
    assert fpm.DiagnosticRecorder("simulation").mode == "simulation"

    with pytest.raises(ValueError, match="unknown diagnostic recorder mode"):
        fpm.DiagnosticRecorder("verbose")
    with pytest.raises(ValueError, match="every must be greater than zero"):
        fpm.DiagnosticRecorder(every=0)


def test_rust_backed_recorder_integrates_with_existing_callback_api(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    recorder = fpm.DiagnosticRecorder("basic")
    result = fpm.AlternatingProjection(iterations=2).run(
        problem,
        callbacks=[recorder],
    )

    assert result.runtime.completed_iterations == 2
    assert isinstance(result.diagnostics, dict)
    assert "final_loss" in result.diagnostics

    diagnostics = recorder.diagnostics()
    assert set(diagnostics) == {
        "iteration_history",
        "frame_diagnostics",
        "raw_frame_stats",
        "coverage",
        "ground_truth_metrics",
    }
    assert len(diagnostics["iteration_history"]) == 2
    assert diagnostics["frame_diagnostics"] == []
    assert diagnostics["raw_frame_stats"] == []
    assert isinstance(diagnostics["coverage"], dict)
    assert diagnostics["ground_truth_metrics"] is None

    make_diagnostic_report(diagnostics, tmp_path / "direct-report")
    assert (tmp_path / "direct-report" / "convergence.png").is_file()
    assert (tmp_path / "direct-report" / "fourier_coverage.png").is_file()
    assert (tmp_path / "direct-report" / "summary.txt").is_file()

    json_path = tmp_path / "diagnostics.json"
    recorder.to_json(json_path)
    loaded = load_diagnostics(json_path)
    assert loaded["iteration_history"] == diagnostics["iteration_history"]
    make_diagnostic_report(json_path, tmp_path / "json-report")
    assert (tmp_path / "json-report" / "summary.txt").is_file()


def test_debug_recorder_collects_frame_and_raw_summaries(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    recorder = fpm.DiagnosticRecorder("debug")
    fpm.AlternatingProjection(iterations=1).run(problem, callbacks=[recorder])

    diagnostics = recorder.diagnostics()
    assert len(diagnostics["frame_diagnostics"]) == problem.frame_count
    assert len(diagnostics["raw_frame_stats"]) == problem.frame_count
    assert all("iteration" in entry for entry in diagnostics["frame_diagnostics"])
    assert "object_snapshots" not in diagnostics
    assert "pupil_snapshots" not in diagnostics

    assert not hasattr(fpm.diagnostics, "plot_frame_residuals")
    assert not hasattr(fpm.diagnostics, "plots")

    plot = fpm.plot.plot_frame_residuals(diagnostics)
    assert plot is not None
    figure, axes = plot
    assert set(axes) == {
        "residual_magnitude",
        "residual_moments",
        "frame_sums",
        "measured_vs_predicted",
    }
    plt.close(figure)

    make_diagnostic_report(diagnostics, tmp_path)
    assert (tmp_path / "frame_residuals.png").is_file()
    assert (tmp_path / "raw_stack_stats.png").is_file()
    assert (tmp_path / "residuals_on_fourier_centers.png").is_file()


def test_reusing_recorder_starts_a_fresh_recording(
    problem: fpm.ReconstructionProblem,
) -> None:
    recorder = fpm.DiagnosticRecorder("minimal")
    fpm.AlternatingProjection(iterations=1).run(problem, callbacks=[recorder])
    assert len(recorder.diagnostics()["iteration_history"]) == 1

    fpm.AlternatingProjection(iterations=2).run(problem, callbacks=[recorder])
    assert len(recorder.diagnostics()["iteration_history"]) == 2
