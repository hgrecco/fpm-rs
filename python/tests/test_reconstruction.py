from __future__ import annotations

import inspect
import math

import fpm_rs as fpm
import numpy as np
import pytest


def assert_canonical_blind_result(
    result: fpm.ReconstructionResult,
    model: fpm.ImagePlaneModel,
) -> None:
    support = model.pupil_support.astype(bool)
    reference = model.pupil[support]
    recovered = result.recovered_pupil[support]
    assert np.sum(np.abs(recovered) ** 2) == pytest.approx(
        np.sum(np.abs(reference) ** 2), rel=1e-12, abs=1e-12
    )
    overlap = np.vdot(reference, recovered)
    assert overlap.real > 0.0
    assert abs(overlap.imag) <= 1e-11 * max(overlap.real, 1.0)
    center = tuple(length // 2 for length in result.object_spectrum.shape)
    dc = result.object_spectrum[center]
    assert dc.real >= 0.0
    assert abs(dc.imag) <= 1e-11 * max(abs(dc), 1.0)


@pytest.mark.parametrize(
    "algorithm",
    [
        fpm.AlternatingProjection(iterations=1),
        fpm.AdaptiveAlternatingProjection(iterations=1),
        fpm.Fpie(iterations=1),
        fpm.Mpie(iterations=1),
        fpm.Epry(iterations=1),
        fpm.Admm(iterations=1, batch_size=1),
        fpm.GradientDescent(iterations=1, parallel_workers=1),
    ],
    ids=[
        "ap",
        "adaptive-ap",
        "fpie",
        "mpie",
        "epry",
        "admm",
        "gradient",
    ],
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
    assert len(result.trace) == 1
    assert math.isfinite(result.final_objective)
    assert result.amplitude is result.amplitude

    if isinstance(algorithm, fpm.Admm):
        assert len(result.algorithm_metrics) == 2
        values = {
            (namespace, metric): value
            for _, namespace, metric, value in result.algorithm_metrics
        }
        assert set(values) == {
            ("admm", "primal_residual_rms"),
            ("admm", "dual_residual_rms"),
        }
        assert all(math.isfinite(value) and value >= 0.0 for value in values.values())
    elif isinstance(algorithm, fpm.AdaptiveAlternatingProjection):
        assert result.algorithm_metrics == [
            (1, "adaptive_alternating_projection", "object_step", 1.0)
        ]
    else:
        assert result.algorithm_metrics == []


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

    with pytest.raises(fpm.InvalidParameterError, match="progress_threshold"):
        fpm.AdaptiveAlternatingProjection(progress_threshold=1.0)

    with pytest.raises(fpm.InvalidParameterError, match="minimum_object_step"):
        fpm.AdaptiveAlternatingProjection(
            initial_object_step=0.5,
            minimum_object_step=0.6,
        )

    with pytest.raises(fpm.InvalidParameterError, match="momentum_interval"):
        fpm.Mpie(momentum_interval=0)

    with pytest.raises(fpm.InvalidParameterError, match="momentum_friction"):
        fpm.Mpie(momentum_friction=1.0)

    with pytest.raises(ValueError, match="loss_type"):
        fpm.GradientDescent(loss_type="unknown")

    with pytest.raises(fpm.InvalidParameterError, match="poisson_truncation_threshold"):
        fpm.GradientDescent(poisson_truncation_threshold=25.0)

    with pytest.raises(fpm.InvalidParameterError, match="poisson_truncation_threshold"):
        fpm.GradientDescent(
            loss_type="poisson_nll",
            poisson_truncation_threshold=0.0,
        )


def test_truncated_poisson_gradient_reports_retained_fraction(
    problem: fpm.ReconstructionProblem,
) -> None:
    result = fpm.GradientDescent(
        iterations=2,
        batch_size=5,
        loss_type="poisson_nll",
        poisson_truncation_threshold=25.0,
        parallel_workers=1,
    ).run(problem)

    assert len(result.trace) == 2
    assert len(result.algorithm_metrics) == 2
    for iteration, namespace, metric, value in result.algorithm_metrics:
        assert iteration in (1, 2)
        assert namespace == "gradient_descent"
        assert metric == "retained_pixel_fraction"
        assert 0.0 <= value <= 1.0


@pytest.mark.parametrize(
    "algorithm",
    [
        fpm.Epry(iterations=1),
        fpm.GradientDescent(
            iterations=1,
            recover_pupil=True,
            parallel_workers=1,
        ),
    ],
    ids=["epry", "gradient"],
)
def test_pupil_recovery_results_use_the_canonical_gauge(
    problem: fpm.ReconstructionProblem,
    model: fpm.ImagePlaneModel,
    algorithm: object,
) -> None:
    assert_canonical_blind_result(algorithm.run(problem), model)


def test_complex_algorithm_constructor_signatures_are_explicit() -> None:
    expected = {
        fpm.AdaptiveAlternatingProjection: [
            "iterations",
            "initial_object_step",
            "progress_threshold",
            "reduction_factor",
            "minimum_object_step",
            "batch_size",
            "epsilon",
        ],
        fpm.Mpie: [
            "iterations",
            "object_step",
            "stability",
            "momentum_interval",
            "momentum_friction",
            "momentum_feedback",
            "batch_size",
            "epsilon",
            "loss_type",
        ],
        fpm.Epry: [
            "iterations",
            "object_step",
            "pupil_step",
            "batch_size",
            "recover_pupil",
            "constrain_pupil_support",
            "recover_frame_gains",
            "gain_step",
            "gain_bounds",
            "recover_background",
            "background_step",
            "background_bounds",
            "epsilon",
            "loss_type",
        ],
        fpm.GradientDescent: [
            "iterations",
            "object_step",
            "batch_size",
            "epsilon",
            "loss_type",
            "poisson_truncation_threshold",
            "recover_illumination",
            "illumination_step",
            "illumination_finite_difference",
            "illumination_bounds",
            "recover_pupil",
            "pupil_step",
            "constrain_pupil_support",
            "object_tv",
            "object_tv_epsilon",
            "pupil_smoothing",
            "parallel_workers",
        ],
    }

    for cls, names in expected.items():
        signature = inspect.signature(cls)
        assert list(signature.parameters) == names
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )
