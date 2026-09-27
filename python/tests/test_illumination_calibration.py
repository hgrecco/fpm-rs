from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fpm_rs as fpm


def calibration_problem() -> tuple[
    fpm.Optics,
    fpm.Illumination,
    fpm.ReconstructionProblem,
]:
    optics = fpm.Optics(532e-9, 0.1, 4.0, 6.5e-6)
    nominal = fpm.Illumination(
        fpm.PlanarLEDArray(
            (3, 3),
            4e-3,
            (1.0, 1.0),
            fpm.ArrayPose.from_translation((0.0, 0.0, -80e-3)),
        )
    )
    true_illumination = fpm.Illumination(
        fpm.PlanarLEDArray(
            (3, 3),
            4e-3,
            (1.0, 1.0),
            fpm.ArrayPose.from_translation((0.25e-3, 0.0, -80e-3)),
        )
    )
    true_model = fpm.compile_model(optics, true_illumination, (12, 12), (36, 36))
    nominal_model = fpm.compile_model(optics, nominal, (12, 12), (36, 36))
    object_field = np.ones((36, 36), dtype=np.complex128)
    object_field[9:27, 12:24] *= np.exp(0.8j)
    simulation = fpm.simulate(
        true_model,
        object_field,
        reconstruction_model=nominal_model,
        seed=7,
    )
    problem = fpm.ReconstructionProblem(simulation.measurements, nominal_model)
    return optics, nominal, problem


def test_typed_parameter_selection_and_gauges() -> None:
    spec = fpm.CalibrationParameterSpec(
        -2e-3,
        2e-3,
        scale=5e-4,
        finite_difference_step=2e-5,
        prior_center=0.0,
        regularization_strength=1e-5,
    )
    parameters = fpm.PlanarArrayCalibrationParameters(
        translation=(True, False, False),
        rotation=(False, False, True),
        position_offsets=[1, 7],
        translation_spec=spec,
        rotation_spec=spec,
    )
    assert parameters.translation == (True, False, False)
    assert parameters.rotation == (False, False, True)
    assert parameters.position_offsets == [1, 7]
    assert spec.finite_difference_step == 2e-5
    assert parameters.translation_specs[0].scale == 5e-4
    assert parameters.translation_specs[1] is None
    assert parameters.rotation_specs[2].lower_bound == -2e-3
    assert parameters.position_offset_specs[1][0].scale > 0.0

    with pytest.raises(ValueError, match="gauge"):
        fpm.PlanarArrayCalibrationParameters(
            translation=(True, False, False),
            reference_index=(True, False),
        )
    with pytest.raises(ValueError, match="cannot be optimized together"):
        fpm.PlanarArrayCalibrationParameters(
            relative_source_power=True,
            frame_gains=True,
        )
    with pytest.raises(fpm.InvalidParameterError, match="strictly ordered"):
        fpm.CalibrationParameterSpec(1.0, 0.0)
    with pytest.raises(ValueError, match="requires at least one"):
        fpm.PlanarArrayCalibrationParameters(translation_spec=spec)


def test_joint_reconstruction_exposes_reusable_physical_result(tmp_path: Path) -> None:
    optics, nominal, problem = calibration_problem()
    parameters = fpm.PlanarArrayCalibrationParameters(
        translation=(True, False, False),
        translation_spec=fpm.CalibrationParameterSpec(
            -1e-3,
            1e-3,
            scale=2.5e-4,
            finite_difference_step=2e-5,
        ),
    )
    calibration = fpm.IlluminationCalibration(
        parameters,
        optimizer=fpm.BoundedFiniteDifferenceOptimizer(
            max_steps=2,
            initial_step_size=0.5,
        ),
    )
    algorithm = fpm.JointReconstruction(
        fpm.Fpie(iterations=1),
        optics,
        nominal,
        calibration,
        outer_iterations=1,
    )
    callbacks: list[dict[str, object]] = []
    result = algorithm.run(
        problem,
        callbacks=[fpm.IterationCallback(lambda context: callbacks.append(context) or True)],
    )

    assert result.reconstruction.runtime.completed_iterations == 1
    assert result.calibrated_model.source_count == 9
    assert result.calibrated_illumination.geometry.kind == "planar_led_array"
    assert result.initial_parameters.translation_m == (0.0, 0.0, -80e-3)
    assert len(result.parameter_history) >= 1
    assert len(result.loss_history) >= 1
    assert result.diagnostics.parameter_names == ["tx_m"]
    assert result.reconstruction.physical_illumination_calibration is not None
    assert result.reconstruction.calibrated_model is not None
    assert callbacks
    callback_state = callbacks[0]["physical_illumination_calibration"]
    assert isinstance(callback_state, fpm.IlluminationCalibrationState)
    assert callback_state.current_parameters.translation_m == (
        result.final_parameters.translation_m
    )
    metric_names = {
        (namespace, metric)
        for namespace, metric, _ in callbacks[0]["algorithm_metrics"]  # type: ignore[index]
    }
    assert ("physical_illumination", "data_loss") in metric_names

    json_path = tmp_path / "joint_result.json"
    result.save_json(json_path)
    restored = fpm.JointReconstructionResult.load_json(json_path)
    assert restored.final_parameters.translation_m == result.final_parameters.translation_m
    bundle = result.write_bundle(tmp_path / "joint_bundle", include_previews=False)
    assert bundle.result.physical_illumination_calibration is not None


def test_joint_reconstruction_accepts_pupil_recovery_algorithm() -> None:
    optics, nominal, problem = calibration_problem()
    parameters = fpm.PlanarArrayCalibrationParameters(
        translation=(True, False, False),
        translation_spec=fpm.CalibrationParameterSpec(
            -1e-3,
            1e-3,
            scale=2.5e-4,
            finite_difference_step=2e-5,
        ),
    )
    result = fpm.JointReconstruction(
        fpm.Epry(iterations=1, recover_pupil=True),
        optics,
        nominal,
        fpm.IlluminationCalibration(parameters),
        outer_iterations=1,
    ).run(problem)
    assert result.reconstruction.runtime.completed_iterations == 1
    assert result.calibrated_model.source_count == 9


def test_joint_reconstruction_rejects_mpie() -> None:
    optics, nominal, _problem = calibration_problem()
    calibration = fpm.IlluminationCalibration(
        fpm.PlanarArrayCalibrationParameters(translation=(True, False, False))
    )
    with pytest.raises(TypeError, match="Fpie or Epry"):
        fpm.JointReconstruction(
            fpm.Mpie(iterations=1),
            optics,
            nominal,
            calibration,
            outer_iterations=1,
        )

    with pytest.raises(TypeError, match="Fpie or Epry"):
        fpm.JointReconstruction(
            fpm.AdaptiveAlternatingProjection(iterations=1),
            optics,
            nominal,
            calibration,
            outer_iterations=1,
        )


def test_unsupported_geometry_fails_clearly(optics: fpm.Optics) -> None:
    illumination = fpm.Illumination(
        fpm.DirectionList.from_unit_vectors(np.array([[0.0, 0.0, 1.0]]))
    )
    model = fpm.compile_model(optics, illumination, (8, 8), (16, 16))
    problem = fpm.ReconstructionProblem(np.ones((1, 8, 8)), model)
    calibration = fpm.IlluminationCalibration(
        fpm.PlanarArrayCalibrationParameters(translation=(True, False, False))
    )
    with pytest.raises(fpm.UnsupportedError, match="PlanarLedArray"):
        fpm.JointReconstruction(
            fpm.Fpie(iterations=1),
            optics,
            illumination,
            calibration,
            outer_iterations=1,
        ).run(problem)
