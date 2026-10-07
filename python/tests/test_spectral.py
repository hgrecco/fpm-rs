"""Exercise spectral composition and reduction to ordinary reconstruction."""

import numpy as np
import pytest

import fpm_rs as fpm


def setup_model(*, coupling="independent", mixed=False, count=3):
    geometry = fpm.DirectionList.from_direction_cosines(
        values=np.array([(0.0, 0.0), (0.08, 0.03), (-0.05, 0.07)], dtype=np.float64)
    )
    channels = [
        fpm.SpectralChannel(
            channel_id=name,
            optics=fpm.Optics(
                wavelength_vacuum_m=wavelength,
                objective_na=0.12,
                magnification=4.0,
                camera_pixel_size=6.5e-6,
                defocus_distance=1e-6,
            ),
            calibration=fpm.SourceCalibration(relative_power=[1.0, 0.8, 1.2]),
            acquisition=fpm.AcquisitionPlan.all_sources(source_count=3),
        )
        for name, wavelength in list(
            zip(["blue", "green", "red"], [450e-9, 532e-9, 630e-9], strict=True)
        )[:count]
    ]
    acquisition = (
        fpm.SpectralAcquisitionPlan.multiplexed(
            frames=[
                fpm.SpectralFrame(
                    contributions=[
                        (channel, local, 0.4 + 0.2 * channel)
                        for channel in range(count)
                    ],
                    gain=1.7,
                    background=0.2,
                )
                for local in range(3)
            ]
        )
        if mixed
        else fpm.SpectralAcquisitionPlan.separate(frame_counts=[3] * count)
    )
    return fpm.compile_spectral_model(
        channels=channels,
        geometry=fpm.SpectralGeometry.shared(geometry=geometry),
        acquisition=acquisition,
        image_shape=(8, 12),
        object_coupling=coupling,
    )


def truth_and_measurements(model):
    rows, cols = np.indices(model.reconstruction_shape)
    count = 1 if model.object_coupling == "shared_complex" else len(model.channel_ids)
    objects = np.ascontiguousarray(
        [
            (0.8 + 0.1 * np.sin(rows + cols + channel))
            * np.exp(0.1j * np.cos(rows * 0.4 + cols * 0.7 + channel))
            for channel in range(count)
        ],
        dtype=np.complex128,
    )
    # The canonical CPU forward FFT is normalized by the high-resolution area.
    spectra = np.ascontiguousarray(
        np.fft.fftshift(np.fft.fft2(objects), axes=(-2, -1))
        / np.prod(model.reconstruction_shape)
    )
    return objects, spectra, model.forward_intensities(object_spectra=spectra)


def test_separate_channels_match_ordinary_runs():
    model = setup_model()
    _, _, measurements = truth_and_measurements(model)
    problem = fpm.SpectralReconstructionProblem(measurements=measurements, model=model)
    result = fpm.SpectralAlternatingProjection(iterations=4, batch_size=4).run(
        problem=problem
    )
    assert result.channel_ids == ["blue", "green", "red"]
    assert result.wavelengths_vacuum_m == [450e-9, 532e-9, 630e-9]
    assert result.object.shape == (3, *model.reconstruction_shape)
    assert result.pupils.shape == (3, 8, 12)
    assert result.completed_iterations == 4
    assert [row[0] for row in result.trace] == [1, 2, 3, 4]
    for channel, kernel in enumerate(model.channel_models):
        ordinary = fpm.ReconstructionProblem(
            measurements=np.ascontiguousarray(
                measurements[channel * 3 : (channel + 1) * 3]
            ),
            model=kernel,
        )
        expected = fpm.AlternatingProjection(iterations=4).run(ordinary)
        np.testing.assert_array_equal(result.object[channel], expected.object)
        np.testing.assert_array_equal(
            result.object_spectrum[channel], expected.object_spectrum
        )
    copy = result.object
    copy[:] = 0.0
    assert np.any(result.object != 0.0)


def test_single_channel_trace_matches_ordinary_ap():
    model = setup_model(count=1)
    _, _, measurements = truth_and_measurements(model)
    spectral = fpm.SpectralAlternatingProjection(iterations=3).run(
        problem=fpm.SpectralReconstructionProblem(
            measurements=measurements, model=model
        )
    )
    ordinary = fpm.AlternatingProjection(iterations=3).run(
        fpm.ReconstructionProblem(
            measurements=measurements, model=model.channel_models[0]
        )
    )
    np.testing.assert_array_equal(spectral.object[0], ordinary.object)
    assert [row[1] for row in spectral.trace] == [row[1] for row in ordinary.trace]


def test_shared_fields_and_fixed_channel_pupils():
    model = setup_model(coupling="shared_complex", mixed=True)
    objects, spectra, measurements = truth_and_measurements(model)
    result = fpm.SpectralAlternatingProjection(iterations=2).run(
        problem=fpm.SpectralReconstructionProblem(
            measurements=measurements, model=model
        ),
        initial_objects=objects,
    )
    assert result.object_coupling == "shared_complex"
    np.testing.assert_array_equal(result.object[0], result.object[1])
    np.testing.assert_array_equal(result.object[1], result.object[2])
    assert not np.array_equal(result.pupils[0], result.pupils[1])
    assert result.trace[-1][1] < 1e-16
    with pytest.raises(fpm.FpmError):
        model.forward_intensities(object_spectra=np.repeat(spectra, 3, axis=0))


def test_weights_canonicalization_and_validation():
    plan = fpm.SpectralAcquisitionPlan.multiplexed(
        frames=[
            fpm.SpectralFrame(contributions=[(0, 0, 0.2), (0, 0, 0.3), (4, 9, 0.0)])
        ]
    )
    assert plan.frames[0].contributions == [(0, 0, 0.5)]
    for weight in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(fpm.FpmError):
            fpm.SpectralAcquisitionPlan.multiplexed(
                frames=[fpm.SpectralFrame(contributions=[(0, 0, weight)])]
            )
    model = setup_model(mixed=True)
    objects, spectra, measurements = truth_and_measurements(model)
    with pytest.raises(fpm.FpmError, match="positive-weight"):
        fpm.SpectralReconstructionProblem(
            measurements=measurements, model=model, frame_weights=[1.0, 0.0, 1.0]
        )
    problem = fpm.SpectralReconstructionProblem(measurements=measurements, model=model)
    solver = fpm.SpectralAlternatingProjection(iterations=1)
    with pytest.raises(fpm.FpmError, match="contiguous"):
        solver.run(problem=problem, initial_objects=objects[:, :, ::-1])
    with pytest.raises(fpm.FpmError, match="permutation"):
        solver.run(problem=problem, frame_order=[0, 0, 2])
    with pytest.raises(ValueError, match="mutually exclusive"):
        solver.run(problem=problem, frame_order=[0, 1, 2], seed=1)
    with pytest.raises(fpm.FpmError, match="contiguous"):
        model.forward_intensities(object_spectra=spectra[:, :, ::-1])


def test_shared_direct_kvectors_are_rejected():
    channel = fpm.SpectralChannel(
        channel_id="blue",
        optics=fpm.Optics(
            wavelength_vacuum_m=450e-9,
            objective_na=0.1,
            magnification=4.0,
            camera_pixel_size=6.5e-6,
        ),
        calibration=fpm.SourceCalibration.unity(),
        acquisition=fpm.AcquisitionPlan.all_sources(source_count=1),
    )
    with pytest.raises(fpm.FpmError, match="cannot be shared"):
        fpm.compile_spectral_model(
            channels=[channel],
            geometry=fpm.SpectralGeometry.shared(
                geometry=fpm.KVectorList(
                    k_vectors=np.array([(0.0, 0.0)], dtype=np.float64)
                )
            ),
            acquisition=fpm.SpectralAcquisitionPlan.separate(frame_counts=[1]),
            image_shape=(8, 8),
        )
