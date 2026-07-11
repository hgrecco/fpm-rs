from __future__ import annotations

import numpy as np
import pytest

import fpm_rs as fpm


@pytest.fixture
def optics() -> fpm.Optics:
    return fpm.Optics(
        wavelength=532e-9,
        objective_na=0.10,
        magnification=4.0,
        camera_pixel_size=6.5e-6,
    )


@pytest.fixture
def model(optics: fpm.Optics) -> fpm.ImagePlaneModel:
    illumination = fpm.LEDArray(
        grid_shape=(1, 1),
        pitch=4e-3,
        distance=90e-3,
        center=(0.0, 0.0),
    )
    return fpm.compile_model(optics, illumination, (8, 8), (16, 16))


@pytest.fixture
def simulation(model: fpm.ImagePlaneModel) -> fpm.SimulationResult:
    rows, columns = np.indices(model.reconstruction_shape)
    amplitude = 0.75 + 0.25 * ((rows + columns) % 2)
    phase = 0.15 * np.sin(rows / 3.0)
    object_field = np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex128)
    return fpm.simulate(model, object_field, seed=7)


@pytest.fixture
def problem(simulation: fpm.SimulationResult) -> fpm.ReconstructionProblem:
    return fpm.ReconstructionProblem(
        simulation.measurements,
        simulation.reconstruction_model,
        name="test-problem",
    )
