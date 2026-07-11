from __future__ import annotations

import numpy as np
import pytest

import fpm_rs as fpm


def test_python_313_is_the_minimum_runtime() -> None:
    import sys

    assert sys.version_info >= (3, 13)
    assert fpm.__version__ == "0.1.0"


def test_compile_led_model_and_numpy_properties(optics: fpm.Optics) -> None:
    aberration = fpm.PupilAberration(astigmatism=0.1, edge_apodization=0.2)
    configured = fpm.Optics(
        optics.wavelength,
        optics.objective_na,
        optics.magnification,
        optics.camera_pixel_size,
        pupil_aberration=aberration,
    )
    leds = fpm.LEDArray((3, 3), 4e-3, 90e-3, (1.0, 1.0))
    model = fpm.compile_model(configured, leds, (8, 8), (16, 16))

    assert model.image_shape == (8, 8)
    assert model.reconstruction_shape == (16, 16)
    assert model.source_count == model.frame_count == 9
    assert model.k_vectors.shape == (9, 2)
    assert model.k_vectors.dtype == np.float64
    assert model.pupil.shape == (8, 8)
    assert model.pupil.dtype == np.complex128
    assert model.pupil_support.dtype == np.uint8


@pytest.mark.parametrize(
    "illumination, expected_sources, expected_frames",
    [
        (fpm.AngleList(np.array([[0.0, 0.0], [0.01, -0.01]])), 2, 2),
        (fpm.LEDSphere(np.array([[0.0, 0.0], [0.02, 0.3]]), 0.09), 2, 2),
        (
            fpm.SphericalLEDArm(
                np.array([[0.0, 0.0], [0.02, 0.3]]),
                0.09,
                theta_zero_degrees=0.1,
                theta_backlash_degrees=0.05,
            ),
            2,
            2,
        ),
        (
            fpm.RotatingLEDArc(
                [0.0, 0.02],
                [0.0, 0.3],
                0.09,
                rotation_backlash_degrees=0.05,
            ),
            4,
            4,
        ),
        (fpm.KVectorList(np.array([[0.0, 0.0], [1.0e4, 0.0]])), 2, 2),
        (
            fpm.CodedIllumination(
                np.array([[0.0, 0.0], [1.0e4, 0.0]]),
                np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            ),
            2,
            3,
        ),
    ],
)
def test_concrete_illumination_sources(
    optics: fpm.Optics,
    illumination: object,
    expected_sources: int,
    expected_frames: int,
) -> None:
    model = fpm.compile_model(optics, illumination, (8, 8), (16, 16))
    assert model.source_count == expected_sources
    assert model.frame_count == expected_frames


def test_camera_response_is_compiled_explicitly(model: fpm.ImagePlaneModel) -> None:
    camera = fpm.CameraModel(
        photons_per_pixel=20.0,
        gain_counts_per_electron=2.0,
        offset_counts=5.0,
        bit_depth=None,
        quantize=False,
    )
    count_model = fpm.compile_camera_model(model, camera)
    np.testing.assert_allclose(count_model.frame_gains, [40.0])


def test_typed_configuration_errors() -> None:
    with pytest.raises(fpm.InvalidParameterError, match="objective_na"):
        fpm.Optics(532e-9, -0.1, 4.0, 6.5e-6)

    with pytest.raises(ValueError, match="shape"):
        fpm.AngleList(np.zeros((2, 3), dtype=np.float64))
