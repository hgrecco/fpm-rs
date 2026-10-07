"""Exercise real shared-OPD optimization and its initialization/gauge contracts."""

import numpy as np
import pytest

import fpm_rs as fpm
from test_spectral import setup_model, truth_and_measurements


def physical_problem(*, mixed=False):
    directions = np.array(
        [(x, y) for y in [-0.12, 0.0, 0.12] for x in [-0.12, 0.0, 0.12]]
    )
    wavelengths = [500e-9, 532e-9, 550e-9]
    channels = [
        fpm.SpectralChannel(
            channel_id=f"channel{index}",
            optics=fpm.Optics(
                wavelength_vacuum_m=w,
                objective_na=0.12,
                magnification=4.0,
                camera_pixel_size=6.5e-6,
            ),
            calibration=fpm.SourceCalibration.unity(),
            acquisition=fpm.AcquisitionPlan.all_sources(source_count=9),
        )
        for index, w in enumerate(wavelengths)
    ]
    plan = (
        fpm.SpectralAcquisitionPlan.multiplexed(
            frames=[
                fpm.SpectralFrame(
                    contributions=[
                        (c, local, 1.0 if c == dominant else 0.1) for c in range(3)
                    ],
                    gain=1.4,
                    background=0.03,
                )
                for local in range(9)
                for dominant in range(3)
            ]
        )
        if mixed
        else fpm.SpectralAcquisitionPlan.separate(frame_counts=[9] * 3)
    )
    model = fpm.compile_spectral_model(
        channels=channels,
        geometry=fpm.SpectralGeometry.shared(
            geometry=fpm.DirectionList.from_direction_cosines(values=directions)
        ),
        acquisition=plan,
        image_shape=(8, 12),
        object_coupling="independent",
    )
    rows, cols = np.indices(model.reconstruction_shape)
    x = 2 * np.pi * cols / model.reconstruction_shape[1]
    y = 2 * np.pi * rows / model.reconstruction_shape[0]
    opd = 1.8e-6 + 0.03e-6 * np.cos(x) + 0.02e-6 * np.sin(2 * y)
    amplitudes = np.ascontiguousarray(
        [
            0.7
            + 0.1 * c
            + 0.04 * np.cos(2 * np.pi * (rows + cols) / model.reconstruction_shape[1])
            for c in range(3)
        ]
    )
    fields = amplitudes * np.exp(
        2j * np.pi * opd[None] / np.array(wavelengths)[:, None, None]
    )
    spectra = np.ascontiguousarray(
        np.fft.fftshift(np.fft.fft2(fields), axes=(-2, -1))
        / np.prod(model.reconstruction_shape)
    )
    measurements = model.forward_intensities(object_spectra=spectra)
    problem = fpm.SpectralReconstructionProblem(measurements=measurements, model=model)
    return problem, model, opd, amplitudes, measurements


@pytest.mark.parametrize("mixed", [False, True])
def test_joint_refinement_improves_intensity_and_opd_and_keeps_one_phase_map(mixed):
    problem, model, truth, amplitudes, _ = physical_problem(mixed=mixed)
    rows, cols = np.indices(truth.shape)
    initial = np.ascontiguousarray(
        truth + 12e-9 * np.sin(2 * np.pi * (rows + cols) / truth.shape[1])
    )
    reference = np.zeros(truth.shape, dtype=np.uint8)
    reference[0, 0] = 1
    result = fpm.MultiWavelengthGradientDescent(iterations=150).run(
        problem=problem,
        unwrapper=fpm.SyntheticWavelengthUnwrapper(opd_range_m=(0.0, 4e-6)),
        reference_mask=reference,
        reference_opd_m=float(truth[0, 0]),
        initial_opd_m=initial,
        initial_amplitudes=np.ascontiguousarray(amplitudes * 1.06),
    )
    assert result.initialization_opd is None and result.initialization_trace is None
    losses = np.array([r[1] for r in result.spectral.trace])
    assert result.spectral.trace[0][0] == 0
    assert losses[-1] < losses[0] * 0.05
    assert np.all(np.diff(losses) <= 0)
    assert np.mean((result.opd_m - truth) ** 2) < 0.4 * np.mean((initial - truth) ** 2)
    assert result.opd_m[0, 0] == truth[0, 0]
    assert result.completed_iterations == 150
    assert not result.stopped_early
    expected = result.spectral.amplitude * np.exp(
        2j
        * np.pi
        * result.opd_m[None]
        / np.array(model.wavelengths_vacuum_m)[:, None, None]
    )
    np.testing.assert_allclose(result.spectral.object, expected, atol=2e-14)
    copy = result.opd_m
    copy[:] = 0.0
    assert np.mean(result.opd_m) > 1e-6


def test_automatic_initialization_retains_phase_fusion_and_ap_diagnostics():
    problem, _, truth, _, _ = physical_problem()
    reference = np.zeros(truth.shape, dtype=np.uint8)
    reference[0, 0] = 1
    result = fpm.MultiWavelengthGradientDescent(
        iterations=80, initialization_iterations=200
    ).run(
        problem=problem,
        unwrapper=fpm.SyntheticWavelengthUnwrapper(opd_range_m=(0.0, 4e-6)),
        reference_mask=reference,
        reference_opd_m=float(truth[0, 0]),
    )
    assert len(result.initialization_trace) == 200
    assert np.all(result.initialization_opd.valid_mask)
    assert result.spectral.trace[-1][1] < result.spectral.trace[0][1]
    np.testing.assert_allclose(result.opd_m, truth, atol=20e-9)


def test_masks_frame_weights_mean_gauge_and_bounded_updates():
    problem, model, truth, amplitudes, measurements = physical_problem(mixed=True)
    masks = np.ones(measurements.shape, dtype=np.uint8)
    masks[:, :, 0] = 0
    weights = np.ones(model.frame_count)
    weights[0] = 0.0
    modified = measurements.copy()
    modified[:, :, 0] += 1e8
    modified[0] += 3e7
    first = fpm.SpectralReconstructionProblem(
        measurements=measurements, model=model, masks=masks, frame_weights=weights
    )
    second = fpm.SpectralReconstructionProblem(
        measurements=modified, model=model, masks=masks, frame_weights=weights
    )
    initial = np.ascontiguousarray(truth + 6e-9)
    solver = fpm.MultiWavelengthGradientDescent(
        iterations=8, opd_step=100.0, amplitude_step=100.0
    )
    kwargs = dict(
        unwrapper=fpm.SyntheticWavelengthUnwrapper(opd_range_m=(1.7e-6, 1.9e-6)),
        phase_offsets_rad=[0.0] * 3,
        initial_opd_m=initial,
        initial_amplitudes=amplitudes,
    )
    a = solver.run(problem=first, **kwargs)
    b = solver.run(problem=second, **kwargs)
    np.testing.assert_array_equal(a.opd_m, b.opd_m)
    np.testing.assert_array_equal(a.spectral.amplitude, b.spectral.amplitude)
    assert abs(a.opd_m.mean() - initial.mean()) < 1e-19
    assert np.all((a.opd_m >= 1.7e-6) & (a.opd_m < 1.9e-6))
    assert np.all(a.spectral.amplitude > 0.0)


def test_exhausted_backtracking_retains_the_last_accepted_state():
    problem, _, truth, amplitudes, _ = physical_problem()
    initial = np.ascontiguousarray(truth)
    result = fpm.MultiWavelengthGradientDescent(
        iterations=4, max_backtracks=1, amplitude_step=1e8, opd_step=1e8
    ).run(
        problem=problem,
        unwrapper=fpm.SyntheticWavelengthUnwrapper(opd_range_m=(0.0, 4e-6)),
        phase_offsets_rad=[0.0] * 3,
        initial_opd_m=initial,
        initial_amplitudes=np.ascontiguousarray(amplitudes * 1.06),
    )
    assert result.stopped_early
    assert result.completed_iterations == 0
    assert len(result.spectral.trace) == 1
    np.testing.assert_allclose(result.opd_m, initial, atol=1e-20)
    np.testing.assert_array_equal(result.spectral.amplitude, amplitudes * 1.06)


def test_validation_rejects_incompatible_models_references_and_starts():
    model = setup_model()
    objects, _, measurements = truth_and_measurements(model)
    problem = fpm.SpectralReconstructionProblem(measurements=measurements, model=model)
    shape = model.reconstruction_shape
    solver = fpm.MultiWavelengthGradientDescent(iterations=1)
    kwargs = dict(
        problem=problem,
        unwrapper=fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-1e-6, 1e-6)),
        phase_offsets_rad=[0.0] * 3,
    )
    with pytest.raises(ValueError, match="together"):
        solver.run(**kwargs, initial_opd_m=np.zeros(shape))
    with pytest.raises(fpm.InvalidParameterError):
        solver.run(
            **kwargs,
            initial_opd_m=np.full(shape, 1e-6),
            initial_amplitudes=np.abs(objects),
        )
    with pytest.raises(fpm.FpmError, match="C-contiguous"):
        solver.run(
            **kwargs,
            initial_opd_m=np.zeros(shape)[:, ::-1],
            initial_amplitudes=np.abs(objects),
        )
    with pytest.raises(fpm.InvalidShapeError):
        solver.run(
            **kwargs,
            initial_opd_m=np.zeros(shape),
            initial_amplitudes=np.ones((2, *shape)),
        )
    with pytest.raises(fpm.InvalidParameterError):
        fpm.MultiWavelengthGradientDescent(iterations=0)
    with pytest.raises(fpm.InvalidParameterError, match="initialization_iterations"):
        fpm.MultiWavelengthGradientDescent(initialization_iterations=0).run(**kwargs)
    shared = setup_model(coupling="shared_complex")
    _, _, measured = truth_and_measurements(shared)
    with pytest.raises(fpm.InvalidParameterError, match="independent"):
        solver.run(
            problem=fpm.SpectralReconstructionProblem(
                measurements=measured, model=shared
            ),
            unwrapper=kwargs["unwrapper"],
            phase_offsets_rad=[0.0] * 3,
        )
    with pytest.raises(fpm.InvalidParameterError, match="no pixels|invalid pixels"):
        solver.run(
            problem=problem,
            unwrapper=fpm.SyntheticWavelengthUnwrapper(
                opd_range_m=(-1e-6, 1e-6), minimum_amplitude=1e8
            ),
            phase_offsets_rad=[0.0] * 3,
        )
