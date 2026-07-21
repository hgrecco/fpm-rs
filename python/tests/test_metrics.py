from __future__ import annotations

import numpy as np
import pytest

import fpm_rs as fpm


def test_direct_intensity_metrics_are_role_agnostic() -> None:
    statistics = fpm.metrics.stats(
        np.array([[0, 1], [2, 3]], dtype=np.uint16), saturation_value=2.0
    )
    comparison = fpm.metrics.compare_intensity(
        reference=np.array([[1.0, 2.0]]),
        estimate=np.array([[2.0, 0.0]]),
    )

    assert statistics["mean"] == pytest.approx(1.5)
    assert statistics["saturated_pixels"] == 2
    assert comparison["reference_sum"] == pytest.approx(3.0)
    assert comparison["estimate_sum"] == pytest.approx(2.0)
    assert comparison["residual_mean"] == pytest.approx(-0.5)
    assert not hasattr(fpm.metrics, "intensity_statistics")


def test_intensity_metrics_support_integer_images_and_valid_masks() -> None:
    reference = np.array([[1, 2]], dtype=np.uint8)
    estimate = np.array([[2, 0]], dtype=np.uint8)

    assert fpm.metrics.bias(reference, estimate=estimate) == pytest.approx(-0.5)
    assert fpm.metrics.mae(reference, estimate) == pytest.approx(1.5)
    assert fpm.metrics.mse(reference, estimate) == pytest.approx(2.5)
    assert fpm.metrics.rmse(reference, estimate) == pytest.approx(np.sqrt(2.5))
    assert fpm.metrics.relative_l1(reference, estimate) == pytest.approx(1.0)
    assert fpm.metrics.nrmse(reference, estimate) == pytest.approx(1.0)
    assert fpm.metrics.correlation(reference, estimate) == pytest.approx(-1.0)
    assert (
        fpm.metrics.bias(reference, estimate, valid_mask=np.array([[True, False]]))
        == 1.0
    )
    assert fpm.metrics.fitted_gain(reference, estimate) == pytest.approx(0.4)


def test_intensity_quality_metrics() -> None:
    reference = np.array([[0.0, 1.0]])
    estimate = np.array([[0.0, 0.0]])

    assert fpm.metrics.psnr(reference, estimate, data_range=1.0) == pytest.approx(
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
    assert fpm.radial_fourier_spectrum(
        reference
    ) == fpm.metrics.radial_fourier_spectrum(reference)


def test_metrics_accept_transposed_stepped_inputs_and_masks() -> None:
    reference = np.arange(1.0, 49.0).reshape(6, 8)
    estimate = reference + 0.5
    valid = (reference.astype(np.int64) % 3) != 0
    strided_reference = reference.T[::2, ::2]
    strided_estimate = estimate.T[::2, ::2]
    strided_valid = valid.T[::2, ::2]

    actual = fpm.metrics.mse(
        strided_reference,
        strided_estimate,
        valid_mask=strided_valid,
    )
    expected = fpm.metrics.mse(
        strided_reference.copy(),
        strided_estimate.copy(),
        valid_mask=strided_valid.copy(),
    )
    assert actual == pytest.approx(expected)

    complex_reference = reference.astype(np.complex128).T
    complex_estimate = complex_reference * np.exp(0.2j)
    comparison = fpm.metrics.compare_complex_fields(
        complex_reference[::2, ::2],
        complex_estimate[::2, ::2],
    )
    assert comparison["complex_nrmse"] == pytest.approx(0.0, abs=1e-12)


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
        "object",
        "pupil",
        "illumination",
        "frame_gains",
        "intensity",
    }
    assert evaluation["object"]["complex_nrmse"] >= 0.0
    assert len(evaluation["intensity"]["per_frame"]) == problem.frame_count
    frame = evaluation["intensity"]["per_frame"][0]
    assert "reference_sum" in frame
    assert "estimate_sum" in frame
    assert "candidate_sum" not in frame


def test_reconstruction_evaluation_accepts_strided_truth_and_mask(
    simulation: fpm.SimulationResult, problem: fpm.ReconstructionProblem
) -> None:
    result = fpm.AlternatingProjection(iterations=1).run(problem)
    truth = simulation.ground_truth_object.T
    mask = np.ones_like(truth, dtype=np.uint8).T

    evaluation = fpm.evaluation.evaluate_reconstruction(
        result,
        truth,
        valid_object_mask=mask,
    )
    assert evaluation["object"]["complex_nrmse"] >= 0.0
