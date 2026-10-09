"""Offline capture-range and downstream bright-field warm-start comparison.

Run with `pixi run python python/examples/benchmark_brightfield_initialization.py`.
Each method runs in a fresh process; initialization time is included for warm
methods. Fixtures, detector controls and acceptance tolerances are benchmark
choices. See the reconstruction guide for definitions and limitations.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import fpm_rs as fpm

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    import resource


if sys.platform == "win32":

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _get_process_memory_info = _psapi.GetProcessMemoryInfo
    _get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    _get_process_memory_info.restype = wintypes.BOOL


SCENARIOS = (
    "clean",
    "amplitude",
    "phase",
    "weak",
    "textureless",
    "noisy",
    "near_cutoff",
    "poor",
    "rotation",
    "pitch",
    "distance",
    "reference",
    "low_na",
    "aberration",
    "vignetting",
    "known_response",
    "radius_mismatch",
)
METHODS = ("nominal", "cold_joint", "circles", "warm_joint")


def peak_mib():
    """Return fresh-process peak resident memory in MiB on every supported OS."""
    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not _get_process_memory_info(
            _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.peak_working_set_size / 1024**2
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def aligned_error(reference, candidate):
    """Relative field error after periodic integer translation and phase piston."""
    correlation = np.fft.ifft2(np.fft.fft2(reference) * np.conj(np.fft.fft2(candidate)))
    shift = np.unravel_index(np.argmax(np.abs(correlation)), reference.shape)
    shifted = np.roll(candidate, shift, axis=(0, 1))
    phase = np.angle(np.vdot(shifted, reference))
    return float(
        np.linalg.norm(shifted * np.exp(1j * phase) - reference)
        / np.linalg.norm(reference)
    )


def fixture(scenario, capture, seed, shape):
    """Build matched detector data, physical truths, and parameter selections."""
    na = 0.08 if scenario == "low_na" else 0.1
    aberration = (
        fpm.PupilAberration(astigmatism=0.5, coma=0.3)
        if scenario == "aberration"
        else None
    )
    optics = fpm.Optics(
        wavelength_vacuum_m=532e-9,
        objective_na=na,
        magnification=4,
        camera_pixel_size=6.5e-6,
        pupil_aberration=aberration,
    )
    pitch = 8e-3 if scenario == "poor" else 4e-3
    if scenario == "near_cutoff":
        pitch = 5e-3
    translation = (capture * 1e-3, -0.75 * capture * 1e-3, -80e-3)
    rotation = (0.0, 0.0, 0.0)
    true_pitch = pitch
    reference = (2.0, 2.0)
    kwargs = dict(
        translation=(True, True, False),
        translation_spec=fpm.CalibrationParameterSpec(
            -2e-3, 2e-3, scale=0.2e-3, finite_difference_step=1e-6
        ),
    )
    if scenario in ("rotation", "pitch", "distance", "reference"):
        translation = (0.0, 0.0, -80e-3)
        if scenario == "rotation":
            rotation = (0.0, 0.0, capture * 0.1)
            kwargs = dict(
                rotation=(False, False, True),
                rotation_spec=fpm.CalibrationParameterSpec(
                    -0.2, 0.2, scale=0.05, finite_difference_step=1e-5
                ),
            )
        elif scenario == "pitch":
            true_pitch += capture * 0.5e-3
            kwargs = dict(
                pitch=(True, True),
                pitch_spec=fpm.CalibrationParameterSpec(
                    3e-3, 5e-3, scale=0.2e-3, finite_difference_step=1e-6
                ),
            )
        elif scenario == "distance":
            translation = (0.0, 0.0, -80e-3 - capture * 10e-3)
            kwargs = dict(
                translation=(False, False, True),
                translation_spec=fpm.CalibrationParameterSpec(
                    -0.1, -0.06, scale=2e-3, finite_difference_step=1e-5
                ),
            )
        else:
            reference = (2 + capture * 0.2, 2 - capture * 0.15)
            kwargs = dict(
                reference_index=(True, True),
                reference_index_spec=fpm.CalibrationParameterSpec(
                    1, 3, scale=0.1, finite_difference_step=1e-5
                ),
            )
    parameters = fpm.PlanarArrayCalibrationParameters(**kwargs)
    gains = np.linspace(0.6, 1.4, 25) if scenario == "known_response" else np.ones(25)
    powers = np.linspace(0.7, 1.3, 25) if scenario == "known_response" else np.ones(25)
    acquisition = fpm.AcquisitionPlan.from_sparse(
        frames=[
            fpm.IlluminationFrame(contributions=[(i, 1.0)], gain=float(gains[i]))
            for i in range(25)
        ]
    )
    calibration = fpm.SourceCalibration(relative_power=powers.tolist())

    def illumination(p, ref, t, r):
        return fpm.Illumination(
            geometry=fpm.PlanarLEDArray(
                shape=(5, 5),
                pitch_m=p,
                reference_index=ref,
                pose=fpm.ArrayPose.from_translation_and_extrinsic_xyz_radians(
                    translation_m=t, rotation_rad=r
                ),
            ),
            calibration=calibration,
            acquisition=acquisition,
        )

    nominal = illumination(pitch, (2, 2), (0, 0, -80e-3), (0, 0, 0))
    truth = illumination(true_pitch, reference, translation, rotation)
    # The sparse bright-field fixture still contains all dark-field sources;
    # its larger Fourier extent needs a larger reconstruction grid.
    factor = 3 if scenario == "poor" else 2
    high_shape = (shape[0] * factor, shape[1] * factor)
    model = fpm.compile_model(
        optics=optics,
        illumination=nominal,
        image_shape=shape,
        reconstruction_shape=high_shape,
    )
    true_optics = optics
    if scenario == "radius_mismatch":
        true_optics = fpm.Optics(
            wavelength_vacuum_m=532e-9,
            objective_na=0.075,
            magnification=4,
            camera_pixel_size=6.5e-6,
        )
    true_model = fpm.compile_model(
        optics=true_optics,
        illumination=truth,
        image_shape=shape,
        reconstruction_shape=high_shape,
    )
    field = fpm.SyntheticObject.mixed_test_pattern(shape=high_shape).field
    if scenario == "amplitude":
        field = np.abs(field).astype(np.complex128)
    elif scenario == "phase":
        field = np.exp(1j * np.angle(field))
    elif scenario == "weak":
        field = 1 + 0.02 * (field - 1)
    elif scenario == "textureless":
        field = np.ones(high_shape, dtype=np.complex128)
    field = np.ascontiguousarray(field)
    camera = None
    if scenario == "known_response":
        camera = fpm.CameraModel(
            photons_per_pixel=1, offset_counts=0.03, bit_depth=None, quantize=False
        )
        model = fpm.compile_camera_model(model=model, camera=camera)
    simulation = fpm.simulate(
        true_model=true_model, object=field, camera=camera, reconstruction_model=model
    )
    data = simulation.measurements.array
    if scenario == "noisy":
        rng = np.random.default_rng(seed)
        data = np.maximum(
            0, data + 0.02 * data.mean() * rng.standard_normal(data.shape)
        )
    if scenario == "vignetting":
        rows, cols = np.indices(shape)
        radius = ((rows - shape[0] / 2) / shape[0]) ** 2 + (
            (cols - shape[1] / 2) / shape[1]
        ) ** 2
        data *= np.exp(-2 * radius)[None]
    measurements = fpm.MeasurementStack(measurements=np.ascontiguousarray(data))
    expected = dict(
        translation_m=translation,
        rotation_rad=rotation,
        pitch_m=(true_pitch, true_pitch),
        reference_index=reference,
    )
    initial = dict(
        translation_m=(0, 0, -80e-3),
        rotation_rad=(0, 0, 0),
        pitch_m=(pitch, pitch),
        reference_index=(2, 2),
    )
    return (
        optics,
        nominal,
        truth,
        model,
        measurements,
        field,
        parameters,
        expected,
        initial,
    )


def circle_metrics(result, optics, truth, scenario, tolerance):
    """Geometric recall and false acceptance with explicit synthetic labels."""
    true_na = truth.resolve(optics).k_vectors * optics.wavelength_vacuum_m / (2 * np.pi)
    options = result.options
    nominal_na = result.nominal_illumination.resolve(optics).k_vectors * (
        optics.wavelength_vacuum_m / (2 * np.pi)
    )
    pixel_na = optics.wavelength_vacuum_m / (
        min(result.initialized_model.image_shape)
        * optics.camera_pixel_size
        / optics.magnification
    )
    eligible = (
        (
            np.linalg.norm(true_na, axis=1)
            + options.center_search_radius_na
            + options.brightfield_margin_na
            < optics.objective_na
        )
        & (
            np.linalg.norm(nominal_na, axis=1)
            > max(pixel_na, options.center_search_radius_na)
        )
        & (
            np.linalg.norm(true_na - nominal_na, axis=1)
            <= options.center_search_radius_na
        )
    )
    if scenario in ("textureless", "radius_mismatch"):
        eligible[:] = False
    accepted = [o for o in result.observations if o.accepted]
    correct = sum(
        eligible[o.source_index]
        and np.linalg.norm(o.detected_na - true_na[o.source_index]) <= tolerance
        for o in accepted
    )
    return dict(
        candidate_frames=result.diagnostics.candidate_frames,
        accepted_centers=len(accepted),
        expected_centers=int(eligible.sum()),
        detection_recall=float(correct / eligible.sum()) if eligible.any() else None,
        false_acceptance_fraction=float((len(accepted) - correct) / len(accepted))
        if accepted
        else 0.0,
        fit_rank=result.diagnostics.jacobian_rank,
        fit_condition=result.diagnostics.jacobian_condition_estimate,
        initialization_measurement_passes=result.runtime.measurement_passes,
        geometry_fit_evaluations=result.runtime.physical_objective_evaluations,
    )


def trial(args):
    """One pipeline, counting initialization in runtime and retaining failures."""
    optics, nominal, truth, model, measurements, field, parameters, expected, fitted = (
        fixture(args.scenario, args.capture, args.seed, (args.rows, args.columns))
    )
    baseline = peak_mib()
    started = time.perf_counter()
    initialized = None
    active = nominal
    record = dict(
        method=args.method,
        scenario=args.scenario,
        capture=args.capture,
        seed=args.seed,
        refinement_passes=args.iterations,
        image_shape=list(model.image_shape),
        reconstruction_shape=list(model.reconstruction_shape),
        nominal_physical_parameters=fitted.copy(),
        truth_physical_parameters=expected,
    )
    if args.method in ("circles", "warm_joint"):
        options = fpm.BrightfieldCircleOptions(
            center_search_radius_na=0.012,
            pupil_radius_search_na=0.03,
            gaussian_sigma_pixels=1.5,
            minimum_edge_contrast=0.01,
            maximum_fit_steps=150,
            fit_initial_step_size=0.25,
            pupil_radius_tolerance_na=0.015,
        )
        record["detector_options"] = {
            name: getattr(options, name)
            for name in (
                "center_search_radius_na",
                "pupil_radius_search_na",
                "gaussian_sigma_pixels",
                "minimum_edge_contrast",
                "maximum_fit_steps",
                "fit_initial_step_size",
                "pupil_radius_tolerance_na",
            )
        }
        try:
            initialized = fpm.BrightfieldCircleInitializer(
                parameters=parameters, options=options
            ).initialize(
                measurements=measurements,
                optics=optics,
                nominal_illumination=nominal,
                model=model,
            )
        except fpm.FpmError as error:
            return record | dict(
                status="initialization_rejected",
                error=str(error),
                elapsed_seconds=time.perf_counter() - started,
                peak_rss_mib=peak_mib(),
                baseline_rss_mib=baseline,
            )
        active = initialized.initialized_illumination
        model = initialized.initialized_model
        fitted = {
            key: getattr(initialized.initialized_parameters, key) for key in fitted
        }
        record.update(
            circle_metrics(
                initialized, optics, truth, args.scenario, args.center_tolerance_na
            )
        )
    problem = fpm.ReconstructionProblem(measurements=measurements, model=model)
    if args.method in ("cold_joint", "warm_joint"):
        result = fpm.JointReconstruction(
            object_algorithm=fpm.Fpie(iterations=1),
            optics=optics,
            initial_illumination=active,
            illumination_calibration=fpm.IlluminationCalibration(parameters=parameters),
            outer_iterations=args.iterations,
            object_iterations_per_outer=1,
            illumination_steps_per_outer=1,
        ).run(problem=problem)
        active, model = result.calibrated_illumination, result.calibrated_model
        reconstructed = result.reconstruction.object
        fitted = {key: getattr(result.final_parameters, key) for key in fitted}
        record["geometry_recompilations"] = result.diagnostics.geometry_recompilations
        record["calibration_rejected_steps"] = result.diagnostics.rejected_steps
        # Each trial recompile can perform at most one full-data objective,
        # plus a base objective and an object projection per outer iteration.
        # Failed recompiles do not perform a forward pass, hence an upper bound.
        record["forward_model_pass_upper_bound"] = (
            result.diagnostics.geometry_recompilations + 2 * args.iterations
        )
    else:
        reconstructed = fpm.Fpie(iterations=args.iterations).run(problem=problem).object
        record["geometry_recompilations"] = 0
        record["forward_model_pass_upper_bound"] = args.iterations
    elapsed = time.perf_counter() - started
    peak = peak_mib()
    # Evaluation is excluded from timed work. Use the calibrated model's known
    # camera response too; simulate compiles its supplied camera only once.
    prediction = fpm.simulate(true_model=model, object=reconstructed).measurements.array
    a = active.resolve(optics).k_vectors
    b = truth.resolve(optics).k_vectors
    na_error = np.linalg.norm(a - b, axis=1) * optics.wavelength_vacuum_m / (2 * np.pi)
    return record | dict(
        status="ok",
        elapsed_seconds=elapsed,
        peak_rss_mib=peak,
        baseline_rss_mib=baseline,
        source_na_rmse=float(np.sqrt(np.mean(na_error**2))),
        translation_error_m=float(
            np.linalg.norm(
                np.subtract(fitted["translation_m"], expected["translation_m"])
            )
        ),
        rotation_error_rad=float(
            np.linalg.norm(
                np.subtract(fitted["rotation_rad"], expected["rotation_rad"])
            )
        ),
        pitch_error_m=float(
            np.linalg.norm(np.subtract(fitted["pitch_m"], expected["pitch_m"]))
        ),
        reference_index_error=float(
            np.linalg.norm(
                np.subtract(fitted["reference_index"], expected["reference_index"])
            )
        ),
        amplitude_mse=float(
            np.mean(
                (np.sqrt(prediction + 1e-10) - np.sqrt(measurements.array + 1e-10)) ** 2
            )
        ),
        aligned_complex_relative_error=aligned_error(field, reconstructed),
    )


def comparisons(trials, source_tolerance, field_tolerance):
    """Report matched-budget gains and largest sampled successful capture."""
    groups = {}
    for trial in trials:
        key = (
            trial["scenario"],
            trial["capture"],
            trial["seed"],
            trial["refinement_passes"],
        )
        groups.setdefault(key, {})[trial["method"]] = trial
    matched = []
    for key, methods in groups.items():
        cold, warm = methods.get("cold_joint"), methods.get("warm_joint")
        if cold and warm and cold["status"] == warm["status"] == "ok":
            matched.append(
                dict(
                    scenario=key[0],
                    capture=key[1],
                    seed=key[2],
                    passes=key[3],
                    warm_source_error_ratio=warm["source_na_rmse"]
                    / max(cold["source_na_rmse"], 1e-16),
                    warm_field_error_ratio=warm["aligned_complex_relative_error"]
                    / max(cold["aligned_complex_relative_error"], 1e-16),
                    warm_runtime_ratio=warm["elapsed_seconds"]
                    / cold["elapsed_seconds"],
                )
            )
    capture = []
    for scenario in sorted({t["scenario"] for t in trials}):
        for method in METHODS:
            samples = [
                t for t in trials if t["scenario"] == scenario and t["method"] == method
            ]
            successful = [
                t["capture"]
                for t in samples
                if t["status"] == "ok"
                and t["source_na_rmse"] <= source_tolerance
                and t["aligned_complex_relative_error"] <= field_tolerance
            ]
            capture.append(
                dict(
                    scenario=scenario,
                    method=method,
                    largest_sampled_success=max(successful) if successful else None,
                    successful_trials=len(successful),
                    total_trials=len(samples),
                )
            )
    time_groups = {}
    for trial in trials:
        key = (trial["scenario"], trial["capture"], trial["seed"], trial["method"])
        time_groups.setdefault(key, []).append(trial)
    times = []
    for key, samples in time_groups.items():
        reached = [
            t
            for t in samples
            if t["status"] == "ok"
            and t["source_na_rmse"] <= source_tolerance
            and t["aligned_complex_relative_error"] <= field_tolerance
        ]
        best = min(reached, key=lambda t: t["elapsed_seconds"]) if reached else None
        times.append(
            dict(
                scenario=key[0],
                capture=key[1],
                seed=key[2],
                method=key[3],
                fastest_sampled_seconds=best["elapsed_seconds"] if best else None,
                sampled_passes=best["refinement_passes"] if best else None,
            )
        )
    return dict(matched_budget=matched, sampled_capture=capture, time_to_target=times)


def main():
    """Collect independent-process trials with explicit configurable budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=(*SCENARIOS, "all"),
        default=["clean", "phase", "noisy", "poor"],
    )
    parser.add_argument("--captures", nargs="+", type=float, default=[0.2, 0.6])
    parser.add_argument("--passes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--columns", type=int, default=56)
    parser.add_argument("--center-tolerance-na", type=float, default=0.004)
    parser.add_argument("--field-tolerance", type=float, default=0.4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scenario", choices=SCENARIOS, default="clean", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--method", choices=METHODS, default="circles", help=argparse.SUPPRESS
    )
    parser.add_argument("--capture", type=float, default=0.4, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.passes + [args.repeats, args.rows, args.columns, args.iterations]) <= 0
        or not all(np.isfinite(v) and v >= 0 for v in args.captures + [args.capture])
        or not all(
            np.isfinite(v) and v > 0
            for v in [args.center_tolerance_na, args.field_tolerance]
        )
    ):
        parser.error(
            "sizes and tolerances must be positive; capture values finite and nonnegative"
        )
    if args.worker:
        print(json.dumps(trial(args), allow_nan=False))
        return
    scenarios = (
        SCENARIOS if "all" in args.scenarios else list(dict.fromkeys(args.scenarios))
    )
    trials = []
    for scenario, capture, seed, passes, method in itertools.product(
        scenarios,
        sorted(set(args.captures)),
        range(args.repeats),
        sorted(set(args.passes)),
        METHODS,
    ):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--scenario",
            scenario,
            "--capture",
            str(capture),
            "--seed",
            str(seed),
            "--iterations",
            str(passes),
            "--method",
            method,
            "--rows",
            str(args.rows),
            "--columns",
            str(args.columns),
            "--center-tolerance-na",
            str(args.center_tolerance_na),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            trials.append(
                dict(
                    scenario=scenario,
                    capture=capture,
                    seed=seed,
                    refinement_passes=passes,
                    method=method,
                    status="error",
                    error=completed.stderr.strip(),
                )
            )
        else:
            trials.append(json.loads(completed.stdout))
    report = dict(
        format_version=1,
        trials=trials,
        source_success_tolerance_na=args.center_tolerance_na,
        field_success_tolerance=args.field_tolerance,
        **comparisons(trials, args.center_tolerance_na, args.field_tolerance),
    )
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
