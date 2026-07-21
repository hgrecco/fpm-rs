from __future__ import annotations

import numpy as np
import pytest

import fpm_rs as fpm


def test_simulation_uses_distinct_true_and_reconstruction_models(
    optics: fpm.Optics,
    model: fpm.ImagePlaneModel,
) -> None:
    assumed_optics = fpm.Optics(
        optics.wavelength,
        optics.objective_na,
        optics.magnification,
        optics.camera_pixel_size,
        defocus_distance=2e-6,
    )
    illumination = fpm.LEDArray((1, 1), 4e-3, 90e-3, (0.0, 0.0))
    assumed = fpm.compile_model(assumed_optics, illumination, (8, 8), (16, 16))
    camera = fpm.CameraModel(
        photons_per_pixel=10.0,
        gain_counts_per_electron=2.0,
        offset_counts=4.0,
        shot_noise=True,
        quantize=True,
    )
    object_field = np.ones((16, 16), dtype=np.complex128)

    result = fpm.simulate(
        model,
        object_field,
        reconstruction_model=assumed,
        camera=camera,
        seed=99,
    )

    assert not result.ideal
    assert result.random_seed == 99
    assert result.measurements.array.shape == (1, 8, 8)
    assert result.measurements.array.dtype == np.float64
    np.testing.assert_allclose(result.reconstruction_model.frame_gains, [20.0])


def test_acquisition_errors_are_concrete_configuration(
    model: fpm.ImagePlaneModel,
) -> None:
    errors = fpm.IlluminationAcquisitionErrors(missing_frames=[0])
    result = fpm.simulate(
        model,
        np.ones((16, 16), dtype=np.complex128),
        illumination_errors=errors,
    )
    assert result.missing_frames == [0]
    np.testing.assert_array_equal(result.measurements.frame_weights, [0.0])


def test_measurement_stack_rejects_noncontiguous_arrays_and_masks(
    model: fpm.ImagePlaneModel,
) -> None:
    contiguous = np.ones((1, 8, 8), dtype=np.float64)
    measurements = contiguous[:, ::-1, :]
    with pytest.raises(fpm.InvalidShapeError, match="ascontiguousarray"):
        fpm.MeasurementStack(measurements, frame_weights=[0.5])

    masks = np.ones((1, 8, 8), dtype=np.uint8)[:, :, ::-1]
    with pytest.raises(fpm.InvalidShapeError, match="ascontiguousarray"):
        fpm.MeasurementStack(contiguous, frame_weights=[0.5], masks=masks)

    stack = fpm.MeasurementStack(
        contiguous,
        frame_weights=[0.5],
        masks=np.ascontiguousarray(masks),
    )
    problem = fpm.ReconstructionProblem(stack, model, name="array-problem")

    assert problem.name == "array-problem"
    assert problem.frame_count == 1
    assert stack.shape == (1, 8, 8)
    np.testing.assert_allclose(stack.frame_weights, [0.5])


def test_synthetic_object_rejects_noncontiguous_fields() -> None:
    field = np.ones((16, 16), dtype=np.complex128).T
    with pytest.raises(fpm.InvalidShapeError, match="ascontiguousarray"):
        fpm.SyntheticObject(field)


def test_problem_accepts_a_raw_numpy_stack(model: fpm.ImagePlaneModel) -> None:
    measurements = np.ones((1, 8, 8), dtype=np.float64)
    problem = fpm.ReconstructionProblem(measurements, model)
    assert problem.image_shape == (8, 8)


def test_problem_rejects_wrong_numpy_dtype(model: fpm.ImagePlaneModel) -> None:
    with pytest.raises(TypeError, match="float64"):
        fpm.ReconstructionProblem(np.ones((1, 8, 8), dtype=np.float32), model)


def test_synthetic_object_helpers_are_available_from_python(
    model: fpm.ImagePlaneModel,
) -> None:
    object_field = fpm.SyntheticObject.mixed_test_pattern((16, 16))
    assert object_field.shape == (16, 16)
    assert object_field.label is None
    assert object_field.field.dtype == np.complex128

    simulation = fpm.simulate(model, object_field, seed=5)
    assert simulation.ground_truth_object.shape == (16, 16)

    phase_disk = fpm.SyntheticObject.phase_disk((16, 16), 4.0, 0.5)
    assert phase_disk.field.dtype == np.complex128
    np.testing.assert_allclose(
        fpm.simulate(model, phase_disk, seed=5).ground_truth_object,
        phase_disk.field,
    )
