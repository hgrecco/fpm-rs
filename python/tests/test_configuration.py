from __future__ import annotations

import numpy as np
import pytest

import fpm_rs as fpm


def test_python_312_is_the_minimum_runtime() -> None:
    import sys

    assert sys.version_info >= (3, 12)
    assert fpm.__version__ == "0.2.0-beta.2"


def test_compile_led_model_and_numpy_properties(optics: fpm.Optics) -> None:
    aberration = fpm.PupilAberration(astigmatism=0.1, edge_apodization=0.2)
    configured = fpm.Optics(
        optics.wavelength_vacuum_m,
        optics.objective_na,
        optics.magnification,
        optics.camera_pixel_size,
        pupil_aberration=aberration,
    )
    leds = fpm.PlanarLEDArray(
        (3, 3),
        4e-3,
        (1.0, 1.0),
        fpm.ArrayPose.from_translation((0.0, 0.0, -90e-3)),
    )
    illumination = fpm.Illumination(leds)
    model = fpm.compile_model(configured, illumination, (8, 8), (16, 16))

    assert model.image_shape == (8, 8)
    assert model.reconstruction_shape == (16, 16)
    assert model.source_count == model.frame_count == 9
    assert model.k_vectors.shape == (9, 2)
    assert model.k_vectors.dtype == np.float64
    assert model.pupil.shape == (8, 8)
    assert model.pupil.dtype == np.complex128
    assert model.pupil_support.dtype == np.uint8


def test_reconstruction_shape_suggestion_and_automatic_compilation(
    optics: fpm.Optics,
) -> None:
    image_shape = (8, 8)
    dk = 2.0 * np.pi / (image_shape[1] * optics.object_pixel_size)
    illumination = fpm.Illumination(fpm.KVectorList(np.array([[2.25 * dk, -1.4 * dk]])))

    assert fpm.suggest_reconstruction_shape(
        optics, illumination, image_shape, "minimum"
    ) == (13, 13)
    assert fpm.suggest_reconstruction_shape(optics, illumination, image_shape) == (
        14,
        14,
    )
    assert fpm.suggest_reconstruction_shape(
        optics, illumination, image_shape, "power_of_two"
    ) == (16, 16)
    assert fpm.suggest_reconstruction_shape(
        optics, illumination, image_shape, (16, 16)
    ) == (16, 16)

    assert fpm.compile_model(
        optics, illumination, image_shape
    ).reconstruction_shape == (
        14,
        14,
    )
    assert fpm.compile_model(
        optics, illumination, image_shape, "minimum"
    ).reconstruction_shape == (13, 13)
    assert fpm.compile_model(
        optics, illumination, image_shape, (16, 16)
    ).reconstruction_shape == (16, 16)


def test_reconstruction_shape_argument_rejects_none_and_unknown_modes(
    optics: fpm.Optics,
) -> None:
    illumination = fpm.Illumination(fpm.KVectorList(np.array([[0.0, 0.0]])))

    with pytest.raises(TypeError, match="reconstruction_shape must be"):
        fpm.compile_model(optics, illumination, (8, 8), None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reconstruction_shape must be"):
        fpm.suggest_reconstruction_shape(  # type: ignore[arg-type]
            optics, illumination, (8, 8), None
        )
    with pytest.raises(ValueError, match="reconstruction_shape must be"):
        fpm.compile_model(optics, illumination, (8, 8), "fast")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "illumination, expected_sources, expected_frames",
    [
        (
            fpm.Illumination(
                fpm.DirectionList.from_component_angles_radians(
                    np.array([[0.0, 0.0], [0.01, -0.01]])
                )
            ),
            2,
            2,
        ),
        (
            fpm.Illumination(
                fpm.SphericalLEDArray(np.array([[0.0, 0.0], [0.02, 0.3]]), 0.09)
            ),
            2,
            2,
        ),
        (
            fpm.Illumination(
                fpm.SphericalLEDArm(
                    np.array([[0.0, 0.0], [0.02, 0.3]]),
                    0.09,
                    theta_zero_degrees=0.1,
                    theta_backlash_degrees=0.05,
                )
            ),
            2,
            2,
        ),
        (
            fpm.Illumination(
                fpm.RotatingLEDArc(
                    [0.0, 0.02],
                    [0.0, 0.3],
                    0.09,
                    rotation_backlash_degrees=0.05,
                )
            ),
            4,
            4,
        ),
        (
            fpm.Illumination(
                fpm.SourcePositionList(
                    np.array([[0.0, 0.0, -0.09], [0.001, 0.0, -0.09]])
                )
            ),
            2,
            2,
        ),
        (
            fpm.Illumination(fpm.KVectorList(np.array([[0.0, 0.0], [1.0e4, 0.0]]))),
            2,
            2,
        ),
        (
            fpm.Illumination(
                fpm.KVectorList(np.array([[0.0, 0.0], [1.0e4, 0.0]])),
                acquisition=fpm.AcquisitionPlan.from_dense(
                    np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
                ),
            ),
            2,
            3,
        ),
    ],
)
def test_concrete_illumination_sources(
    optics: fpm.Optics,
    illumination: fpm.Illumination,
    expected_sources: int,
    expected_frames: int,
) -> None:
    model = fpm.compile_model(optics, illumination, (8, 8), (16, 16))
    assert model.source_count == expected_sources
    assert model.frame_count == expected_frames


def test_resolved_illumination_exposes_geometry_calibration_and_acquisition(
    optics: fpm.Optics,
) -> None:
    geometry = fpm.PlanarLEDArray(
        (1, 2),
        (4e-3, 5e-3),
        (0.5, 0.0),
        fpm.ArrayPose.from_translation_and_extrinsic_xyz_degrees(
            (1e-3, -2e-3, -90e-3), (1.0, -2.0, 3.0)
        ),
        position_offsets_m=np.array([[0.0, 0.0, 0.0], [1e-4, 0.0, 0.0]]),
    )
    acquisition = fpm.AcquisitionPlan.from_sparse(
        [
            fpm.IlluminationFrame([(1, 1.0)], gain=0.8),
            fpm.IlluminationFrame([(0, 0.25), (1, 0.75)], gain=1.2),
        ]
    )
    illumination = fpm.Illumination(
        geometry,
        calibration=fpm.SourceCalibration(relative_power=[0.5, 2.0]),
        acquisition=acquisition,
    )
    resolved = illumination.resolve(optics)

    assert illumination.geometry.kind == "planar_led_array"
    assert resolved.source_count == 2
    assert resolved.frame_count == 2
    assert resolved.is_multiplexed
    assert resolved.positions_m is not None
    assert resolved.positions_m.shape == (2, 3)
    assert resolved.directions.shape == (2, 3)
    assert resolved.k_vectors.shape == (2, 2)
    np.testing.assert_allclose(resolved.source_power, [0.5, 2.0])
    np.testing.assert_allclose(resolved.frame_gains, [0.8, 1.2])
    np.testing.assert_allclose(resolved.dense_weights, [[0.0, 1.0], [0.25, 0.75]])


def test_direction_list_angular_round_trips_and_sequential_repetition() -> None:
    directions = fpm.DirectionList.from_polar_angles_degrees(
        np.array([[0.0, 0.0], [30.0, 120.0]])
    )
    np.testing.assert_allclose(directions.polar_angles_deg, [[0.0, 0.0], [30.0, 120.0]])
    plan = fpm.AcquisitionPlan.sequential([1, 0, 1])
    np.testing.assert_allclose(
        plan.dense_weights(2), [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )


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

    boundary = 4.0
    with pytest.raises(
        fpm.InvalidParameterError,
        match="camera_pixel_size.*strictly less.*coherent-field sampling",
    ):
        fpm.Optics(
            wavelength_vacuum_m=1.0,
            objective_na=0.25,
            magnification=2.0,
            camera_pixel_size=boundary,
        )
    fpm.Optics(
        wavelength_vacuum_m=1.0,
        objective_na=0.25,
        magnification=2.0,
        camera_pixel_size=np.nextafter(boundary, 0.0),
    )
    with pytest.raises(fpm.InvalidParameterError, match="camera_pixel_size"):
        fpm.Optics(
            wavelength_vacuum_m=1.0,
            objective_na=0.25,
            magnification=2.0,
            camera_pixel_size=np.nextafter(boundary, np.inf),
        )

    with pytest.raises(ValueError, match="shape"):
        fpm.DirectionList(np.zeros((2, 2), dtype=np.float64))
