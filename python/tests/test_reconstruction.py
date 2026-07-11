from __future__ import annotations

import math

import numpy as np
import pytest

import fpm_rs as fpm


@pytest.mark.parametrize(
    "algorithm",
    [
        fpm.AlternatingProjection(iterations=1),
        fpm.Fpie(iterations=1),
        fpm.Epry(iterations=1),
        fpm.Admm(iterations=1, batch_size=1),
        fpm.GradientDescent(iterations=1, parallel_workers=1),
    ],
    ids=["ap", "fpie", "epry", "admm", "gradient"],
)
def test_all_algorithms_return_numpy_results(
    problem: fpm.ReconstructionProblem,
    algorithm: object,
) -> None:
    result = algorithm.run(problem)

    assert result.object.shape == (16, 16)
    assert result.object.dtype == np.complex128
    assert result.amplitude.shape == (16, 16)
    assert result.amplitude.dtype == np.float64
    assert result.phase.shape == (16, 16)
    assert result.object_spectrum.dtype == np.complex128
    assert result.recovered_pupil.shape == (8, 8)
    assert result.runtime.completed_iterations == 1
    assert len(result.history) == 1
    assert math.isfinite(result.final_loss)
    assert result.amplitude is result.amplitude

    if isinstance(algorithm, fpm.Admm):
        assert len(result.admm_residual_history) == 1
        _, primal, dual = result.admm_residual_history[0]
        assert math.isfinite(primal) and primal >= 0.0
        assert math.isfinite(dual) and dual >= 0.0
    else:
        assert result.admm_residual_history == []


def test_schedules_are_exposed(problem: fpm.ReconstructionProblem) -> None:
    for schedule in (
        "sequential",
        "brightfield_first",
        "spiral_out",
        "random",
        "snr_weighted",
    ):
        result = fpm.AlternatingProjection(iterations=1).run(
            problem,
            schedule=schedule,
            schedule_seed=5,
        )
        assert result.runtime.completed_iterations == 1

    with pytest.raises(ValueError, match="schedule"):
        fpm.AlternatingProjection(iterations=1).run(problem, schedule="unknown")


def test_algorithm_parameter_errors_remain_typed() -> None:
    with pytest.raises(fpm.InvalidParameterError, match="object_step"):
        fpm.AlternatingProjection(object_step=0.0)

    with pytest.raises(ValueError, match="loss_type"):
        fpm.GradientDescent(loss_type="unknown")
