from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

import fpm_rs as fpm


def test_plot_reconstruction_builds_expected_mosaic(
    simulation: fpm.SimulationResult,
    problem: fpm.ReconstructionProblem,
) -> None:
    result = fpm.AlternatingProjection(iterations=2).run(problem)

    assert not hasattr(fpm, "plot_reconstruction")
    figure, axes = fpm.plot.plot_reconstruction(simulation.ground_truth_object, result)

    assert set(axes) == {
        "true_amplitude",
        "true_phase",
        "reconstructed_amplitude",
        "reconstructed_phase",
        "loss",
    }
    assert axes["loss"].get_yscale() == "log"
    assert axes["loss"].get_title() == "Reconstruction error (loss)"
    assert len(axes["loss"].lines[0].get_xdata()) == len(result.history)
    assert all(not axes[name].axison for name in set(axes) - {"loss"})
    plt.close(figure)


def test_plot_reconstruction_validates_truth_and_layout(
    simulation: fpm.SimulationResult,
    problem: fpm.ReconstructionProblem,
) -> None:
    result = fpm.AlternatingProjection(iterations=1).run(problem)

    with pytest.raises(ValueError, match="two-dimensional"):
        fpm.plot.plot_reconstruction(np.ones(3, dtype=np.complex128), result)
    with pytest.raises(ValueError, match="missing required axes"):
        fpm.plot.plot_reconstruction(
            simulation.ground_truth_object,
            result,
            layout=[["true_amplitude"]],
        )
