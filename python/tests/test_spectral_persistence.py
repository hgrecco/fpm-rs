"""Verify typed spectral state across Rust/Python persistence and local datasets."""

import json
import struct
import zlib

import numpy as np
import pytest

import fpm_rs as fpm
from test_spectral import setup_model, truth_and_measurements
from test_multi_wavelength import physical_problem


@pytest.mark.parametrize(
    "mixed,coupling",
    [(False, "independent"), (True, "independent"), (True, "shared_complex")],
)
def test_ap_exact_resume_and_bundle(mixed, coupling, tmp_path):
    model = setup_model(mixed=mixed, coupling=coupling)
    _, _, data = truth_and_measurements(model)
    problem = fpm.SpectralReconstructionProblem(measurements=data, model=model)
    first = fpm.SpectralAlternatingProjection(iterations=3, batch_size=2).run(
        problem=problem, seed=17, checkpoint_directory=tmp_path, checkpoint_every=2
    )
    assert first.checkpoint.completed_iterations == 3
    checkpoint = fpm.SpectralReconstructionCheckpoint.load(
        path=tmp_path / "spectral_checkpoint_00003.json"
    )
    assert checkpoint.opd_m is None
    assert checkpoint.completed_iterations == 3
    resumed = fpm.SpectralAlternatingProjection(iterations=7, batch_size=2).run(
        problem=problem, resume=checkpoint
    )
    full = fpm.SpectralAlternatingProjection(iterations=7, batch_size=2).run(
        problem=problem, seed=17
    )
    np.testing.assert_array_equal(resumed.object_spectrum, full.object_spectrum)
    assert [r[:2] for r in resumed.trace] == [r[:2] for r in full.trace]
    with pytest.raises(fpm.FpmError):
        fpm.SpectralAlternatingProjection(iterations=7, batch_size=3).run(
            problem=problem, resume=checkpoint
        )
    bundle = resumed.write_bundle(
        path=tmp_path / "bundle", run_id="spectral-fixture", label="exact resume"
    )
    reopened = fpm.read_spectral_bundle(bundle.path)
    assert reopened.verify().artifact_count == len(reopened.artifacts)
    np.testing.assert_array_equal(reopened.result.object, resumed.object)
    assert reopened.result.channel_ids == model.channel_ids
    assert reopened.joint_result is None
    copy = reopened.result.object
    copy[:] = 0
    np.testing.assert_array_equal(reopened.result.object, resumed.object)
    with pytest.raises(fpm.FpmError):
        fpm.read_bundle(bundle.path)
    artifact = reopened.artifacts["channels.0.object"]
    data = bytearray(artifact.path.read_bytes())
    data[-1] ^= 1
    artifact.path.write_bytes(data)
    reopened.clear_cache()
    with pytest.raises(fpm.FpmError):
        reopened.verify()
    with pytest.raises(fpm.FpmError):
        _ = reopened.result


@pytest.mark.parametrize("mixed", [False, True])
def test_joint_exact_resume_bundle_and_fixed_reference(mixed, tmp_path):
    problem, model, opd, amplitudes, _ = physical_problem(mixed=mixed)
    mask = np.zeros(opd.shape, dtype=np.uint8)
    mask[0, 0] = 1
    kwargs = dict(
        problem=problem,
        unwrapper=fpm.SyntheticWavelengthUnwrapper(opd_range_m=(0, 4e-6)),
        reference_mask=mask,
        reference_opd_m=float(opd[0, 0]),
        initial_opd_m=np.ascontiguousarray(
            opd + 10e-9 * np.sin(np.indices(opd.shape)[1])
        ),
        initial_amplitudes=np.ascontiguousarray(amplitudes * 1.05),
    )
    first = fpm.MultiWavelengthGradientDescent(iterations=3).run(
        **kwargs, checkpoint_directory=tmp_path
    )
    assert first.checkpoint.completed_iterations == 3
    checkpoint = fpm.SpectralReconstructionCheckpoint.load(
        path=tmp_path / "joint_opd_checkpoint_00003.json"
    )
    resumed = fpm.MultiWavelengthGradientDescent(iterations=7).run_from_checkpoint(
        problem=problem, checkpoint=checkpoint
    )
    full = fpm.MultiWavelengthGradientDescent(iterations=7).run(**kwargs)
    np.testing.assert_array_equal(resumed.opd_m, full.opd_m)
    np.testing.assert_array_equal(resumed.spectral.amplitude, full.spectral.amplitude)
    assert [r[:2] for r in resumed.spectral.trace] == [
        r[:2] for r in full.spectral.trace
    ]
    assert resumed.opd_m[0, 0] == opd[0, 0]
    bundle = resumed.write_bundle(path=tmp_path / "joint")
    bundle.verify()
    loaded = fpm.read_spectral_bundle(bundle.path).joint_result
    np.testing.assert_array_equal(loaded.opd_m, full.opd_m)
    assert loaded.initialization_opd is None
    with pytest.raises(fpm.FpmError):
        fpm.SpectralAlternatingProjection(iterations=8).run(
            problem=problem, resume=checkpoint
        )
    with pytest.raises(fpm.FpmError):
        fpm.MultiWavelengthGradientDescent(
            iterations=8, opd_step=0.5
        ).run_from_checkpoint(problem=problem, checkpoint=checkpoint)


@pytest.mark.parametrize("mixed", [False, True])
def test_mixing_svd_matches_numpy(mixed):
    model = setup_model(mixed=mixed)
    d = model.mixing_diagnostics()
    singular = np.linalg.svd(d.matrix, compute_uv=False)
    np.testing.assert_allclose(d.singular_values, singular, rtol=2e-14, atol=2e-15)
    assert d.rank == np.linalg.matrix_rank(d.matrix)
    assert d.channel_ids == model.channel_ids
    if mixed:
        assert d.condition_number is None
    else:
        assert d.condition_number == pytest.approx(np.linalg.cond(d.matrix))
    copy = d.matrix
    copy[:] = 0
    assert np.any(d.matrix)
    assert model.mixing_diagnostics(relative_tolerance=2).rank == 0
    with pytest.raises(fpm.FpmError):
        model.mixing_diagnostics(relative_tolerance=-1)


def _png(path, array):
    """Write a tiny lossless grayscale fixture using only the standard library."""

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    h, w = array.shape
    payload = b"".join(b"\0" + row.tobytes() for row in array)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(payload))
        + chunk(b"IEND", b"")
    )


def test_offline_spectral_dataset_and_external_compiled_kernels(tmp_path):
    model = setup_model(mixed=True)
    frames = []
    for i in range(model.frame_count):
        name = f"frame-{i}.png"
        _png(tmp_path / name, np.full(model.image_shape, i + 1, dtype=np.uint8))
        frames.append({"path": name})
    (tmp_path / "measurements.json").write_text(json.dumps({"frames": frames}))
    channels = []
    for i, kernel in enumerate(model.channel_models):
        name = f"kernel-{i}.json"
        kernel.save_json(path=tmp_path / name)
        loaded = fpm.ImagePlaneModel.load_json(path=tmp_path / name)
        np.testing.assert_array_equal(loaded.pupil, kernel.pupil)
        channels.append(
            dict(
                channel_id=model.channel_ids[i],
                wavelength_vacuum_m=model.wavelengths_vacuum_m[i],
                compiled_model=name,
                response_provenance="Generated explicit spectral weights and unity calibration.",
            )
        )
    acquisition = dict(
        frames=[
            dict(
                contributions=[
                    dict(channel=c, local_frame=local_frame, spectral_weight=w)
                    for c, local_frame, w in f.contributions
                ],
                gain=f.gain,
                background=f.background,
            )
            for f in model.acquisition.frames
        ]
    )
    manifest = dict(
        format_version=2,
        measurement_manifest="measurements.json",
        channels=channels,
        object_coupling="independent",
        acquisition=acquisition,
        provenance={"source": "generated"},
        measurement_units="camera counts",
    )
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(manifest))
    dataset = fpm.load_spectral_dataset(path=tmp_path)
    assert dataset.model.channel_ids == model.channel_ids
    assert dataset.response_provenance["blue"] == channels[0]["response_provenance"]
    assert dataset.ground_truth_objects == [None] * 3
    assert dataset.provenance == {"source": "generated"}
    assert dataset.measurement_units == "camera counts"
    assert dataset.measurements.array[1, 0, 0] == 2
    fpm.SpectralAlternatingProjection(iterations=1).run(
        problem=dataset.reconstruction_problem()
    )
    rebuilt = fpm.SpectralImagePlaneModel.from_compiled_channels(
        channel_ids=model.channel_ids,
        models=model.channel_models,
        acquisition=model.acquisition,
    )
    np.testing.assert_array_equal(
        rebuilt.mixing_diagnostics().matrix, model.mixing_diagnostics().matrix
    )
    manifest["channels"][0]["wavelength_vacuum_m"] = 400e-9
    path.write_text(json.dumps(manifest))
    with pytest.raises(fpm.DatasetError):
        fpm.load_spectral_dataset(path=tmp_path)


def test_phase_mixing_bundle_preserves_invalid_nan_pixels(tmp_path):
    model = setup_model()
    fields, _, data = truth_and_measurements(model)
    mask = np.ones(model.reconstruction_shape, dtype=np.uint8)
    mask[0, 0] = 0
    result = fpm.SpectralAlternatingProjection(iterations=2).run_opd(
        problem=fpm.SpectralReconstructionProblem(measurements=data, model=model),
        unwrapper=fpm.SyntheticWavelengthUnwrapper(opd_range_m=(-0.5e-6, 0.5e-6)),
        phase_offsets_rad=[0.0] * 3,
        mask=mask,
        initial_objects=fields,
    )
    assert np.isnan(result.opd.opd_m[0, 0])
    bundle = result.write_bundle(path=tmp_path / "phase-mixing")
    bundle.verify()
    loaded = fpm.read_spectral_bundle(bundle.path).unwrapped_opd
    np.testing.assert_array_equal(loaded.opd_m, result.opd.opd_m)
    np.testing.assert_array_equal(loaded.valid_mask, result.opd.valid_mask)
    np.testing.assert_array_equal(
        loaded.phase_residual_rad, result.opd.phase_residual_rad
    )
    np.testing.assert_array_equal(loaded.fringe_orders, result.opd.fringe_orders)


def test_benchmark_reports_matched_starts_parity_noise_and_process_memory(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "benchmark_multi_wavelength.py"
    )
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--passes",
            "4",
            "--repeats",
            "1",
            "--rows",
            "8",
            "--columns",
            "12",
            "--runtime-budget-s",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert len(report["trials"]) == 8
    assert len(report["runtime_comparisons"]) == 8
    for trial in report["trials"]:
        assert "error" not in trial
        assert not trial["failed"]
        assert trial["detector_frames"] == 27
        assert trial["peak_rss_mib"] >= trial["baseline_rss_mib"] > 0
        assert trial["elapsed_seconds"] > 0
        assert trial["mixing_rank"] == 3
        assert trial["opd_rmse_nm"] < 10
        assert np.all(np.isfinite(trial["field_relative_error_by_channel"]))
        if trial["ordinary_parity_max_abs"] is not None:
            assert trial["ordinary_parity_max_abs"] == 0
