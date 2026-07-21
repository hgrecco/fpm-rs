from __future__ import annotations

import numpy as np
import pytest

import fpm_rs as fpm


def test_direct_intensity_metrics_are_role_agnostic() -> None:
    statistics = fpm.metrics.intensity_statistics(
        np.array([[0.0, 1.0], [2.0, 3.0]]), saturation_value=2.0
    )
    comparison = fpm.metrics.compare_intensity(
        np.array([[1.0, 2.0]]), np.array([[2.0, 0.0]])
    )

    assert statistics["mean"] == pytest.approx(1.5)
    assert statistics["saturated_pixels"] == 2
    assert comparison["reference_sum"] == pytest.approx(3.0)
    assert comparison["candidate_sum"] == pytest.approx(2.0)
    assert comparison["residual_mean"] == pytest.approx(-0.5)


def test_atomic_intensity_metrics_support_integer_images_and_valid_masks() -> None:
    reference = np.array([[1, 2]], dtype=np.uint8)
    candidate = np.array([[2, 0]], dtype=np.uint8)

    assert fpm.metrics.bias(reference, candidate) == pytest.approx(-0.5)
    assert fpm.metrics.mae(reference, candidate) == pytest.approx(1.5)
    assert fpm.metrics.mse(reference, candidate) == pytest.approx(2.5)
    assert fpm.metrics.rmse(reference, candidate) == pytest.approx(np.sqrt(2.5))
    assert fpm.metrics.relative_l1(reference, candidate) == pytest.approx(1.0)
    assert fpm.metrics.nrmse(reference, candidate) == pytest.approx(1.0)
    assert fpm.metrics.correlation(reference, candidate) == pytest.approx(-1.0)
    assert fpm.metrics.bias(reference, candidate, valid_mask=np.array([[True, False]])) == 1.0
    assert fpm.metrics.fitted_gain(reference, candidate) == pytest.approx(0.4)


def test_atomic_intensity_quality_metrics() -> None:
    reference = np.array([[0.0, 1.0]])
    candidate = np.array([[0.0, 0.0]])

    assert fpm.metrics.psnr(reference, candidate, data_range=1.0) == pytest.approx(
        10.0 * np.log10(2.0)
    )
    assert fpm.metrics.poisson_deviance(reference, reference, epsilon=1e-12) == 0.0
    assert fpm.metrics.mean_poisson_deviance(reference, reference, epsilon=1e-12) == 0.0

    image = np.ones((11, 11))
    assert fpm.metrics.ssim(image, image, data_range=1.0) == pytest.approx(1.0)


def test_complex_metric_aligns_candidate_phase_and_aliases_radial_spectrum() -> None:
    reference = np.array([[1.0 + 0.0j, 2.0 + 0.0j]])
    candidate = reference * np.exp(0.7j)

    metrics = fpm.metrics.compare_complex_fields(reference, candidate)

    assert metrics["complex_nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert fpm.radial_fourier_spectrum(reference) == fpm.metrics.radial_fourier_spectrum(reference)


def test_reconstruction_evaluation_is_nested(
    simulation: fpm.SimulationResult, problem: fpm.ReconstructionProblem
) -> None:
    result = fpm.AlternatingProjection(iterations=1).run(problem)

    evaluation = fpm.evaluation.evaluate_reconstruction(
        result,
        simulation.ground_truth_object,
        problem=problem,
        reference_model=simulation.true_model,
    )

    assert set(evaluation) == {
        "object", "pupil", "illumination", "frame_gains", "intensity"
    }
    assert evaluation["object"]["complex_nrmse"] >= 0.0
    assert len(evaluation["intensity"]["per_frame"]) == problem.frame_count
