from __future__ import annotations

from collections.abc import Callable
import sys
import threading

import numpy as np

import fpm_rs as fpm


def _assert_python_runs_while_operation_is_active(
    operation: Callable[[], object],
) -> None:
    operation_started = threading.Event()
    operation_finished = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        operation_started.set()
        try:
            operation()
        except BaseException as error:  # propagate failures on the test thread
            errors.append(error)
        finally:
            operation_finished.set()

    # Prevent a GIL-enabled interpreter from switching between the native call
    # returning and the worker setting ``operation_finished``. A detached call
    # still lets this test thread run immediately. Free-threaded builds do not
    # need the setting, but exercise the same concurrency contract.
    previous_switch_interval = sys.getswitchinterval()
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    gil_enabled = True if gil_probe is None else gil_probe()
    thread = threading.Thread(target=worker, daemon=True)
    try:
        sys.setswitchinterval(max(previous_switch_interval, 1.0))
        thread.start()
        assert operation_started.wait(timeout=5), (
            "native-operation worker did not start"
        )

        active_on_wakeup = not operation_finished.is_set()
        progress = 0
        while progress < 1_000 and not operation_finished.is_set():
            progress += 1
    finally:
        sys.setswitchinterval(previous_switch_interval)
        thread.join(timeout=60)

    assert not thread.is_alive(), "native operation exceeded the bounded test workload"
    if errors:
        raise errors[0]

    interpreter = "GIL-enabled" if gil_enabled else "free-threaded"
    assert active_on_wakeup and progress > 0, (
        f"Python made no progress during the native operation on {interpreter} Python"
    )


def test_simulation_releases_the_gil(optics: fpm.Optics) -> None:
    leds = fpm.LEDArray((7, 7), 4e-3, 90e-3, (3.0, 3.0))
    model = fpm.compile_model(optics, leds, (64, 64), (128, 128))
    _assert_python_runs_while_operation_is_active(
        lambda: fpm.simulate(model, np.ones((128, 128), dtype=np.complex128))
    )


def test_reconstruction_releases_the_gil(optics: fpm.Optics) -> None:
    leds = fpm.LEDArray((5, 5), 4e-3, 90e-3, (2.0, 2.0))
    model = fpm.compile_model(optics, leds, (32, 32), (64, 64))
    simulation = fpm.simulate(model, np.ones((64, 64), dtype=np.complex128))
    problem = fpm.ReconstructionProblem(simulation.measurements, model)
    _assert_python_runs_while_operation_is_active(
        lambda: fpm.AlternatingProjection(iterations=20).run(problem)
    )


def test_model_compilation_releases_the_gil(optics: fpm.Optics) -> None:
    # A large pupil keeps the bounded native operation observable even in an
    # optimized wheel.
    leds = fpm.LEDArray((1, 1), 4e-3, 90e-3, (0.0, 0.0))
    _assert_python_runs_while_operation_is_active(
        lambda: fpm.compile_model(optics, leds, (1024, 1024), (2048, 2048))
    )


def test_reconstruction_shape_suggestion_releases_the_gil(optics: fpm.Optics) -> None:
    illumination = fpm.KVectorList(np.zeros((250_000, 2), dtype=np.float64))
    _assert_python_runs_while_operation_is_active(
        lambda: fpm.suggest_reconstruction_shape(optics, illumination, (64, 64))
    )


def test_camera_model_compilation_releases_the_gil(optics: fpm.Optics) -> None:
    leds = fpm.LEDArray((1, 1), 4e-3, 90e-3, (0.0, 0.0))
    model = fpm.compile_model(optics, leds, (1024, 1024), (2048, 2048))
    camera = fpm.CameraModel(offset_counts=1.0)
    _assert_python_runs_while_operation_is_active(
        lambda: fpm.compile_camera_model(model, camera)
    )
