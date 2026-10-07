"""Verify referenced synthetic-wavelength recovery and the spectral convenience."""

import numpy as np
import pytest

import fpm_rs as fpm
from test_spectral import setup_model, truth_and_measurements


def test_discontinuous_opd_reference_offsets_and_copy_semantics():
    wavelengths = np.array([500e-9, 550e-9])
    opd = np.array([[-1.6e-6, 2.8e-6, 0.0], [1.2e-6, -0.4e-6, 0.71e-6]])
    offsets = np.array([1.1, -2.3])
    fields = np.ascontiguousarray(
        np.exp(
            1j
            * (
                2 * np.pi * opd[None] / wavelengths[:, None, None]
                + offsets[:, None, None]
            )
        )
    )
    unwrapper = fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-2e-6, 3e-6))
    result = unwrapper.unwrap_fields(
        fields=fields, wavelengths_vacuum_m=wavelengths, phase_offsets_rad=offsets
    )
    np.testing.assert_allclose(result.opd_m, opd, atol=1e-18)
    assert np.all(result.valid_mask == 1)
    assert result.fringe_orders.shape == fields.shape
    assert result.fringe_orders.dtype == np.int64
    assert result.phase_residual_rad.max() < 1e-12
    copy = result.opd_m
    copy[:] = 0.0
    np.testing.assert_allclose(result.opd_m, opd, atol=1e-18)
    assert unwrapper.opd_range_m == (-2e-6, 3e-6)


def test_reference_region_masks_and_channel_permutation():
    wavelengths = np.array([490e-9, 532e-9, 692e-9])
    opd = np.array([[0.4e-6, 0.4e-6, 2.4e-6], [0.4e-6, -0.3e-6, 3.7e-6]])
    pistons = np.array([2.4, -1.7, 0.6])
    fields = np.ascontiguousarray(
        np.exp(
            1j
            * (
                2 * np.pi * opd[None] / wavelengths[:, None, None]
                + pistons[:, None, None]
            )
        )
    )
    fields[0, 0, 1] = 0
    reference = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    mask = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.uint8)
    unwrap = fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-1e-6, 4e-6))
    result = unwrap.unwrap_fields(
        fields=fields,
        wavelengths_vacuum_m=wavelengths,
        reference_mask=reference,
        reference_opd_m=0.4e-6,
        mask=mask,
    )
    expected_valid = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(result.valid_mask, expected_valid)
    np.testing.assert_allclose(
        result.opd_m[expected_valid != 0], opd[expected_valid != 0], atol=1e-18
    )
    assert np.all(np.isnan(result.opd_m[expected_valid == 0]))
    order = [2, 0, 1]
    permuted = unwrap.unwrap_fields(
        fields=np.ascontiguousarray(fields[order]),
        wavelengths_vacuum_m=wavelengths[order],
        reference_mask=reference,
        reference_opd_m=0.4e-6,
        mask=mask,
    )
    np.testing.assert_allclose(result.opd_m, permuted.opd_m, atol=1e-18)
    np.testing.assert_array_equal(result.fringe_orders[order], permuted.fringe_orders)


def test_noise_hierarchy_and_residual_rejection():
    wavelengths = np.array([500e-9, 501e-9, 550e-9])
    opd = np.linspace(2e-6, 24e-6, 20).reshape(4, 5)
    fields = np.ascontiguousarray(
        np.exp(
            1j
            * (
                2 * np.pi * opd[None] / wavelengths[:, None, None]
                + np.array([0.01, -0.01, 0.0])[:, None, None]
            )
        )
    )
    result = fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-1e-6, 30e-6)).unwrap_fields(
        fields=fields, wavelengths_vacuum_m=wavelengths, phase_offsets_rad=[0.0] * 3
    )
    assert result.wavelength_ladder_m[0] > 250e-6
    np.testing.assert_allclose(result.opd_m, opd, atol=2e-9)
    result = fpm.SyntheticWavelengthUnwrapper(
        opd_range_m=(-1e-6, 30e-6), max_phase_residual_rad=0.001
    ).unwrap_fields(
        fields=fields, wavelengths_vacuum_m=wavelengths, phase_offsets_rad=[0.0] * 3
    )
    assert np.all(result.valid_mask == 0)


def test_reconstruction_convenience_matches_explicit_operations():
    model = setup_model()
    objects, _, measurements = truth_and_measurements(model)
    # This fixture verifies exact composition of the two public operations;
    # noiseless physical OPD recovery from intensity data is tested in Rust.
    problem = fpm.SpectralReconstructionProblem(measurements=measurements, model=model)
    algorithm = fpm.SpectralAlternatingProjection(iterations=3)
    unwrap = fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-1e-6, 1e-6))
    reference = np.zeros(model.reconstruction_shape, dtype=np.uint8)
    reference[0, 0] = 1
    joint = algorithm.run_opd(
        problem=problem,
        unwrapper=unwrap,
        reference_mask=reference,
        initial_objects=objects,
        seed=42,
    )
    explicit = algorithm.run(problem=problem, initial_objects=objects, seed=42)
    opd = explicit.unwrap_opd(unwrapper=unwrap, reference_mask=reference)
    np.testing.assert_array_equal(joint.spectral.object, explicit.object)
    np.testing.assert_array_equal(joint.opd.opd_m, opd.opd_m)
    np.testing.assert_array_equal(joint.opd.fringe_orders, opd.fringe_orders)


def test_validation_and_shared_complex_rejection():
    fields = np.ones((2, 2, 3), dtype=np.complex128)
    wavelengths = [500e-9, 550e-9]
    unwrap = fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-1e-6, 4e-6))
    with pytest.raises(ValueError, match="exactly one"):
        unwrap.unwrap_fields(fields=fields, wavelengths_vacuum_m=wavelengths)
    with pytest.raises(ValueError, match="exactly one"):
        unwrap.unwrap_fields(
            fields=fields,
            wavelengths_vacuum_m=wavelengths,
            phase_offsets_rad=[0.0] * 2,
            reference_mask=np.ones((2, 3), dtype=np.uint8),
        )
    with pytest.raises(fpm.InvalidParameterError):
        unwrap.unwrap_fields(
            fields=fields,
            wavelengths_vacuum_m=[500e-9] * 2,
            phase_offsets_rad=[0.0] * 2,
        )
    with pytest.raises(fpm.FpmError, match="C-contiguous"):
        unwrap.unwrap_fields(
            fields=fields[:, :, ::-1],
            wavelengths_vacuum_m=wavelengths,
            phase_offsets_rad=[0.0] * 2,
        )
    with pytest.raises(fpm.InvalidShapeError):
        unwrap.unwrap_fields(
            fields=fields,
            wavelengths_vacuum_m=wavelengths,
            reference_mask=np.ones((3, 2), dtype=np.uint8),
        )
    with pytest.raises(fpm.InvalidParameterError, match="width"):
        fpm.SyntheticWavelengthUnwrapper(opd_range_m=(0.0, 6e-6)).unwrap_fields(
            fields=fields, wavelengths_vacuum_m=wavelengths, phase_offsets_rad=[0.0] * 2
        )
    model = setup_model(coupling="shared_complex", count=2)
    _, _, measurements = truth_and_measurements(model)
    problem = fpm.SpectralReconstructionProblem(measurements=measurements, model=model)
    with pytest.raises(fpm.InvalidParameterError, match="independent"):
        fpm.SpectralAlternatingProjection(iterations=1).run_opd(
            problem=problem, unwrapper=unwrap, phase_offsets_rad=[0.0] * 2
        )
    result = fpm.SpectralAlternatingProjection(iterations=1).run(problem=problem)
    with pytest.raises(fpm.InvalidParameterError, match="independent"):
        result.unwrap_opd(unwrapper=unwrap, phase_offsets_rad=[0.0] * 2)
