from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

import fpm_rs as fpm


def planar_illumination(translation_m: tuple[float, float, float]) -> fpm.Illumination:
    return fpm.Illumination(
        fpm.PlanarLEDArray(
            (5, 5),
            4e-3,
            (2.0, 2.0),
            fpm.ArrayPose.from_translation(translation_m),
        )
    )


def mean_source_error(
    optics: fpm.Optics,
    actual: fpm.Illumination,
    expected: fpm.Illumination,
) -> float:
    difference = actual.resolve(optics).k_vectors - expected.resolve(optics).k_vectors
    return float(np.mean(np.linalg.norm(difference[:, :2], axis=1)))


def test_brightfield_initializer_returns_reusable_physical_result(
    tmp_path: Path,
) -> None:
    optics = fpm.Optics(532e-9, 0.1, 4.0, 6.5e-6)
    nominal = planar_illumination((0.0, 0.0, -80e-3))
    truth = planar_illumination((0.4e-3, -0.3e-3, -80e-3))
    image_shape = (48, 56)
    reconstruction_shape = (96, 112)
    true_model = fpm.compile_model(optics, truth, image_shape, reconstruction_shape)
    nominal_model = fpm.compile_model(
        optics, nominal, image_shape, reconstruction_shape
    )
    simulation = fpm.simulate(
        true_model,
        fpm.SyntheticObject.mixed_test_pattern(reconstruction_shape),
        reconstruction_model=nominal_model,
    )
    translation = fpm.CalibrationParameterSpec(
        -1e-3,
        1e-3,
        scale=0.2e-3,
        finite_difference_step=1e-6,
    )
    parameters = fpm.PlanarArrayCalibrationParameters(
        translation=(True, True, False),
        translation_spec=translation,
    )
    options = fpm.BrightfieldCircleOptions(
        center_search_radius_na=0.012,
        pupil_radius_search_na=0.012,
        gaussian_sigma_pixels=1.5,
        minimum_edge_contrast=1e-4,
        maximum_fit_steps=150,
        fit_initial_step_size=0.25,
    )
    events: list[dict[str, object]] = []
    result = fpm.BrightfieldCircleInitializer(
        parameters,
        options=options,
    ).initialize(
        simulation.measurements,
        optics,
        nominal,
        nominal_model,
        callback=fpm.PlanarArrayInitializationCallback(
            lambda progress: events.append(progress) or True
        ),
    )

    assert result.diagnostics.accepted_observations > 0
    assert result.diagnostics.final_residual_rms_na < (
        result.diagnostics.initial_residual_rms_na
    )
    assert mean_source_error(optics, result.initialized_illumination, truth) < (
        mean_source_error(optics, nominal, truth)
    )
    assert result.initialized_model.pupil.shape == image_shape
    assert result.runtime.measurement_passes == 2
    assert events[-1]["stage"] == "complete"
    accepted = next(
        observation for observation in result.observations if observation.accepted
    )
    assert accepted.nominal_k_rad_per_m.shape == (2,)
    assert accepted.nominal_k_rad_per_m.dtype == np.float64
    assert accepted.detected_na is not None
    assert accepted.detected_na.shape == (2,)
    assert accepted.fourier_grid_position.shape == (2,)

    json_path = tmp_path / "initialization.json"
    result.save_json(json_path)
    restored = fpm.PlanarArrayInitializationResult.load_json(json_path)
    assert restored.parameter_names == ["tx_m", "ty_m"]

    bundle = result.write_bundle(tmp_path / "initialization-bundle")
    assert bundle.verify().artifact_count == 3
    reopened = fpm.read_initialization_bundle(bundle.path)
    assert reopened.result.parameter_names == result.parameter_names
    assert reopened.observations_artifact.media_type == "text/csv"


def test_brightfield_options_and_parameter_scope_are_validated(
    optics: fpm.Optics,
) -> None:
    with pytest.raises(fpm.InvalidParameterError, match="angular_samples"):
        fpm.BrightfieldCircleOptions(angular_samples=8)

    illumination = planar_illumination((0.0, 0.0, -80e-3))
    model = fpm.compile_model(optics, illumination, (16, 16), (32, 32))
    measurements = fpm.MeasurementStack(np.ones((25, 16, 16), dtype=np.float64))
    initializer = fpm.BrightfieldCircleInitializer(
        fpm.PlanarArrayCalibrationParameters(relative_source_power=True)
    )
    with pytest.raises(fpm.UnsupportedError, match="source powers"):
        initializer.initialize(measurements, optics, illumination, model)


@pytest.mark.parametrize("excluded", ["weight", "mask", "cutoff"])
def test_explicit_brightfield_frames_are_validated_before_detection(excluded):
    optics = fpm.Optics(532e-9, 0.1, 4.0, 6.5e-6)
    illumination = planar_illumination((0.0, 0.0, -80e-3))
    model = fpm.compile_model(optics, illumination, (48, 56), (96, 112))
    weights = np.ones(25)
    masks = np.ones((25, 48, 56), dtype=np.uint8)
    frame = 0 if excluded == "cutoff" else 11
    if excluded == "weight":
        weights[frame] = 0
    if excluded == "mask":
        masks[frame, 0, 0] = 0
    measurements = fpm.MeasurementStack(
        measurements=np.ones((25, 48, 56)), frame_weights=weights.tolist(), masks=masks
    )
    initializer = fpm.BrightfieldCircleInitializer(
        parameters=fpm.PlanarArrayCalibrationParameters(
            translation=(True, True, False)
        ),
        options=fpm.BrightfieldCircleOptions(
            frame_indices=[frame], center_search_radius_na=0.012
        ),
    )
    message = {
        "weight": "positive measurement weight",
        "mask": "invalid pixels",
        "cutoff": "bright-field boundary",
    }[excluded]
    with pytest.raises(fpm.InvalidMeasurementsError, match=message):
        initializer.initialize(
            measurements=measurements,
            optics=optics,
            nominal_illumination=illumination,
            model=model,
        )


def benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "benchmark_brightfield_initialization.py"
    )
    spec = importlib.util.spec_from_file_location("brightfield_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_brightfield_benchmark_compares_actual_pipeline_errors_and_budgets():
    benchmark = benchmark_module()
    args = SimpleNamespace(
        scenario="clean",
        capture=0.4,
        seed=0,
        rows=48,
        columns=56,
        iterations=1,
        center_tolerance_na=0.004,
    )
    records = []
    for method in benchmark.METHODS:
        args.method = method
        records.append(benchmark.trial(args))
    assert all(record["status"] == "ok" for record in records)
    assert all(record["elapsed_seconds"] > 0 for record in records)
    assert all(
        record["peak_rss_mib"] >= record["baseline_rss_mib"] > 0 for record in records
    )
    cold, warm = records[1], records[3]
    assert warm["source_na_rmse"] < cold["source_na_rmse"]
    assert warm["initialization_measurement_passes"] == 2
    assert warm["fit_rank"] == 2
    assert warm["geometry_recompilations"] > 0
    summary = benchmark.comparisons(records, 0.004, 0.4)
    assert len(summary["matched_budget"]) == 1
    assert summary["matched_budget"][0]["warm_source_error_ratio"] < 1
    # The reported alignment must remove the documented integer translation
    # and piston without silently fitting an amplitude scale.
    rng = np.random.default_rng(19)
    field = rng.standard_normal((8, 12)) + 1j * rng.standard_normal((8, 12))
    shifted = np.roll(field, (2, -3), axis=(0, 1)) * np.exp(0.7j)
    assert benchmark.aligned_error(field, shifted) < 1e-12
    assert benchmark.aligned_error(field, 2 * shifted) == pytest.approx(1.0)
