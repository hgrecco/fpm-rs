"""Offline solver comparison with matched starts, exposure counts and pass budgets.

Run with `pixi run python python/examples/benchmark_multi_wavelength.py`.
Each trial runs in a fresh process so peak RSS is independent of previous trials.
The generated fixture and failure threshold are benchmark choices, not published
method parameters. The reconstruction guide explains metrics and limitations.
"""

from __future__ import annotations

import argparse
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


def fixture(mixed: bool, noise: float, seed: int, shape: tuple[int, int]):
    """Build registered nondispersive truth and equal-count separate/coded exposures."""
    directions = np.array([(x, y) for y in [-0.12, 0, 0.12] for x in [-0.12, 0, 0.12]])
    wavelengths = np.array([500e-9, 532e-9, 550e-9])
    channels = [
        fpm.SpectralChannel(
            channel_id=f"channel{c}",
            optics=fpm.Optics(
                wavelength_vacuum_m=float(w),
                objective_na=0.12,
                magnification=4,
                camera_pixel_size=6.5e-6,
            ),
            calibration=fpm.SourceCalibration.unity(),
            acquisition=fpm.AcquisitionPlan.all_sources(source_count=9),
        )
        for c, w in enumerate(wavelengths)
    ]
    # Each row's weights sum to one; both plans have 27 scalar exposures.
    plan = (
        fpm.SpectralAcquisitionPlan.multiplexed(
            frames=[
                fpm.SpectralFrame(
                    contributions=[
                        (c, local, 0.8 if c == dominant else 0.1) for c in range(3)
                    ]
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
        image_shape=shape,
    )
    rows, cols = np.indices(model.reconstruction_shape)
    x = 2 * np.pi * cols / cols.shape[1]
    y = 2 * np.pi * rows / rows.shape[0]
    opd = 1.8e-6 + 35e-9 * np.cos(x) + 25e-9 * np.sin(2 * y)
    amplitudes = np.ascontiguousarray(
        [0.7 + 0.1 * c + 0.04 * np.cos((c + 1) * x + y) for c in range(3)]
    )
    fields = amplitudes * np.exp(2j * np.pi * opd[None] / wavelengths[:, None, None])
    spectra = np.ascontiguousarray(
        np.fft.fftshift(np.fft.fft2(fields), axes=(-2, -1)) / np.prod(opd.shape)
    )
    clean = model.forward_intensities(object_spectra=spectra)
    rng = np.random.default_rng(seed)
    # Independent additive detector noise, sigma relative to global mean intensity.
    data = np.ascontiguousarray(
        np.maximum(0, clean + noise * clean.mean() * rng.standard_normal(clean.shape))
    )
    initial_opd = np.ascontiguousarray(opd + 12e-9 * np.sin(x + y))
    initial_amp = np.ascontiguousarray(amplitudes * 1.06)
    initial_fields = np.ascontiguousarray(
        initial_amp
        * np.exp(2j * np.pi * initial_opd[None] / wavelengths[:, None, None])
    )
    mask = np.zeros(opd.shape, dtype=np.uint8)
    mask[0, 0] = 1
    problem = fpm.SpectralReconstructionProblem(measurements=data, model=model)
    return (
        problem,
        model,
        opd,
        amplitudes,
        fields,
        data,
        clean,
        initial_opd,
        initial_amp,
        initial_fields,
        mask,
    )


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
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def trial(args):
    """Return one measured trial with explicit metric definitions and raw budgets."""
    values = fixture(
        args.acquisition == "mixed", args.noise, args.seed, (args.rows, args.columns)
    )
    (
        problem,
        model,
        truth,
        amplitudes,
        fields,
        data,
        clean,
        opd0,
        amp0,
        fields0,
        mask,
    ) = values
    baseline = peak_mib()
    start = time.perf_counter()
    unwrapper = fpm.SyntheticWavelengthUnwrapper(opd_range_m=(0.0, 4e-6))
    if args.solver == "joint":
        joint = fpm.MultiWavelengthGradientDescent(iterations=args.iterations).run(
            problem=problem,
            unwrapper=unwrapper,
            reference_mask=mask,
            reference_opd_m=float(truth[0, 0]),
            initial_opd_m=opd0,
            initial_amplitudes=amp0,
        )
        result = joint.spectral
        opd = joint.opd_m
        valid = np.ones(truth.shape, dtype=bool)
    else:
        result = fpm.SpectralAlternatingProjection(iterations=args.iterations).run(
            problem=problem, initial_objects=fields0
        )
        fusion = result.unwrap_opd(
            unwrapper=unwrapper, reference_mask=mask, reference_opd_m=float(truth[0, 0])
        )
        opd = fusion.opd_m
        valid = fusion.valid_mask.astype(bool)
    seconds = time.perf_counter() - start
    peak = peak_mib()
    recovered = result.object
    gauge_errors = []
    for a, b in zip(recovered, fields, strict=True):
        piston = np.angle(np.vdot(a, b))
        gauge_errors.append(
            float(np.linalg.norm(a * np.exp(1j * piston) - b) / np.linalg.norm(b))
        )
    opd_rmse = (
        float(np.sqrt(np.mean((opd[valid] - truth[valid]) ** 2)) * 1e9)
        if np.any(valid)
        else None
    )
    prediction = model.forward_intensities(object_spectra=result.object_spectrum)
    error = recovered.__abs__() - amplitudes
    # Off-channel projections of amplitude error onto other true amplitude contrasts.
    cross_talk = []
    for c in range(3):
        for other in range(3):
            if c != other:
                contrast = amplitudes[other] - amplitudes[other].mean()
                cross_talk.append(
                    float(
                        abs(np.vdot(error[c], contrast)) / np.vdot(contrast, contrast)
                    )
                )
    parity = None
    ordinary_seconds = None
    if args.solver == "ap" and args.acquisition == "separate":
        start = time.perf_counter()
        ordinary = []
        for c, kernel in enumerate(model.channel_models):
            p = fpm.ReconstructionProblem(
                measurements=np.ascontiguousarray(data[c * 9 : (c + 1) * 9]),
                model=kernel,
            )
            ordinary.append(
                fpm.AlternatingProjection(iterations=args.iterations)
                .run(problem=p)
                .object
            )
        ordinary_seconds = time.perf_counter() - start
        parity_spectral = fpm.SpectralAlternatingProjection(
            iterations=args.iterations
        ).run(problem=problem)
        parity = float(np.max(np.abs(np.stack(ordinary) - parity_spectral.object)))
    diagnostic = model.mixing_diagnostics()
    return dict(
        solver=args.solver,
        acquisition=args.acquisition,
        noise_sigma_relative_mean=args.noise,
        seed=args.seed,
        requested_passes=args.iterations,
        completed_passes=result.completed_iterations,
        detector_frames=model.frame_count,
        contribution_evaluations_per_pass=sum(
            len(f.contributions) for f in model.acquisition.frames
        ),
        elapsed_seconds=seconds,
        peak_rss_mib=peak,
        baseline_rss_mib=baseline,
        field_relative_error_by_channel=gauge_errors,
        opd_rmse_nm=opd_rmse,
        valid_fraction=float(valid.mean()),
        amplitude_cross_talk_max=max(cross_talk),
        noisy_amplitude_mse=float(
            np.mean((np.sqrt(prediction + 1e-10) - np.sqrt(data + 1e-10)) ** 2)
        ),
        clean_amplitude_mse=float(
            np.mean((np.sqrt(prediction + 1e-10) - np.sqrt(clean + 1e-10)) ** 2)
        ),
        mixing_rank=diagnostic.rank,
        mixing_condition=diagnostic.condition_number,
        ordinary_parity_max_abs=parity,
        ordinary_elapsed_seconds=ordinary_seconds,
        ordinary_parity_initialization="weighted measured amplitude"
        if parity is not None
        else None,
        failed=not np.all(valid) or opd_rmse is None or opd_rmse > args.failure_opd_nm,
        failure_opd_threshold_nm=args.failure_opd_nm,
    )


def main():
    """Run independent trials and report both fixed-pass and common-runtime comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=int, nargs="+", default=[10, 40, 160])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.02])
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--columns", type=int, default=24)
    parser.add_argument("--runtime-budget-s", type=float, default=None)
    parser.add_argument("--failure-opd-nm", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--solver", choices=["ap", "joint"], default="ap", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--acquisition",
        choices=["separate", "mixed"],
        default="separate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--iterations", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--noise", type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.passes + [args.repeats, args.rows, args.columns, args.iterations]) <= 0
        or min(args.noise_levels + [args.noise]) < 0
        or args.failure_opd_nm <= 0
        or (args.runtime_budget_s is not None and args.runtime_budget_s <= 0)
    ):
        parser.error(
            "sizes, repeats and budgets must be positive; noise must be nonnegative"
        )
    if args.worker:
        print(json.dumps(trial(args), allow_nan=False))
        return
    trials = []
    for acquisition in ["separate", "mixed"]:
        for noise in args.noise_levels:
            for seed in range(args.repeats):
                for passes in sorted(set(args.passes)):
                    for solver in ["ap", "joint"]:
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--solver",
                            solver,
                            "--acquisition",
                            acquisition,
                            "--iterations",
                            str(passes),
                            "--noise",
                            str(noise),
                            "--seed",
                            str(seed),
                            "--rows",
                            str(args.rows),
                            "--columns",
                            str(args.columns),
                            "--failure-opd-nm",
                            str(args.failure_opd_nm),
                        ]
                        completed = subprocess.run(
                            command, text=True, capture_output=True, check=False
                        )
                        if completed.returncode:
                            trials.append(
                                dict(
                                    solver=solver,
                                    acquisition=acquisition,
                                    noise_sigma_relative_mean=noise,
                                    seed=seed,
                                    requested_passes=passes,
                                    failed=True,
                                    error=completed.stderr.strip(),
                                )
                            )
                        else:
                            trials.append(json.loads(completed.stdout))
    runtime_comparisons = []
    # Compare the best sampled point under one declared runtime ceiling. This is
    # a discrete budget comparison, not a claim of identical measured wall times.
    if args.runtime_budget_s is not None:
        for acquisition in ["separate", "mixed"]:
            for noise in args.noise_levels:
                for seed in range(args.repeats):
                    for solver in ["ap", "joint"]:
                        eligible = [
                            t
                            for t in trials
                            if t["acquisition"] == acquisition
                            and t["noise_sigma_relative_mean"] == noise
                            and t["seed"] == seed
                            and t["solver"] == solver
                            and t.get("elapsed_seconds", float("inf"))
                            <= args.runtime_budget_s
                            and "noisy_amplitude_mse" in t
                        ]
                        best = (
                            min(eligible, key=lambda t: t["noisy_amplitude_mse"])
                            if eligible
                            else None
                        )
                        runtime_comparisons.append(
                            dict(
                                acquisition=acquisition,
                                noise_sigma_relative_mean=noise,
                                seed=seed,
                                solver=solver,
                                runtime_budget_s=args.runtime_budget_s,
                                best_sample=best,
                            )
                        )
    groups = []
    for acquisition in ["separate", "mixed"]:
        for noise in args.noise_levels:
            for passes in sorted(set(args.passes)):
                for solver in ["ap", "joint"]:
                    records = [
                        t
                        for t in trials
                        if (
                            t["acquisition"],
                            t["noise_sigma_relative_mean"],
                            t["requested_passes"],
                            t["solver"],
                        )
                        == (acquisition, noise, passes, solver)
                    ]
                    groups.append(
                        dict(
                            acquisition=acquisition,
                            noise_sigma_relative_mean=noise,
                            requested_passes=passes,
                            solver=solver,
                            failure_rate=sum(t["failed"] for t in records)
                            / len(records),
                            median_elapsed_seconds=float(
                                np.median(
                                    [
                                        t["elapsed_seconds"]
                                        for t in records
                                        if "elapsed_seconds" in t
                                    ]
                                )
                            )
                            if any("elapsed_seconds" in t for t in records)
                            else None,
                        )
                    )
    report = dict(
        protocol="registered nondispersive explicit starts; 27 exposures; complete-pass sweep; independent-process peak RSS",
        trials=trials,
        groups=groups,
        runtime_comparisons=runtime_comparisons,
    )
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
