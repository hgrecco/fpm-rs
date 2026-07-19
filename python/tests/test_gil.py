from __future__ import annotations

from collections.abc import Callable
import threading
import time

import numpy as np

import fpm_rs as fpm


def _assert_worker_runs_before_return(
    operation: Callable[[], object], *, delay: float
) -> None:
    ran_during_operation = threading.Event()

    def worker() -> None:
        time.sleep(delay)
        ran_during_operation.set()

    thread = threading.Thread(target=worker)
    thread.start()
    operation()
    observed_before_return = ran_during_operation.is_set()
    thread.join()

    assert observed_before_return


def test_simulation_releases_the_gil(optics: fpm.Optics) -> None:
    leds = fpm.LEDArray((7, 7), 4e-3, 90e-3, (3.0, 3.0))
    model = fpm.compile_model(optics, leds, (64, 64), (128, 128))
    _assert_worker_runs_before_return(
        lambda: fpm.simulate(model, np.ones((128, 128), dtype=np.complex128)),
        delay=0.005,
    )


def test_reconstruction_releases_the_gil(optics: fpm.Optics) -> None:
    leds = fpm.LEDArray((5, 5), 4e-3, 90e-3, (2.0, 2.0))
    model = fpm.compile_model(optics, leds, (32, 32), (64, 64))
    simulation = fpm.simulate(model, np.ones((64, 64), dtype=np.complex128))
    problem = fpm.ReconstructionProblem(simulation.measurements, model)
    _assert_worker_runs_before_return(
        lambda: fpm.AlternatingProjection(iterations=20).run(problem),
        delay=0.01,
    )


def test_model_compilation_releases_the_gil(optics: fpm.Optics) -> None:
    # A large pupil makes this operation outlive the worker's delay. Observing
    # the event before the call returns therefore distinguishes detachment from
    # a worker merely running after the Python call completes.
    leds = fpm.LEDArray((1, 1), 4e-3, 90e-3, (0.0, 0.0))
    _assert_worker_runs_before_return(
        lambda: fpm.compile_model(optics, leds, (1024, 1024), (2048, 2048)),
        delay=0.005,
    )


def test_reconstruction_shape_suggestion_releases_the_gil(optics: fpm.Optics) -> None:
    illumination = fpm.KVectorList(np.zeros((250_000, 2), dtype=np.float64))
    _assert_worker_runs_before_return(
        lambda: fpm.suggest_reconstruction_shape(optics, illumination, (64, 64)),
        delay=0.005,
    )


def test_camera_model_compilation_releases_the_gil(optics: fpm.Optics) -> None:
    leds = fpm.LEDArray((1, 1), 4e-3, 90e-3, (0.0, 0.0))
    model = fpm.compile_model(optics, leds, (1024, 1024), (2048, 2048))
    camera = fpm.CameraModel(offset_counts=1.0)
    _assert_worker_runs_before_return(
        lambda: fpm.compile_camera_model(model, camera),
        delay=0.005,
    )
