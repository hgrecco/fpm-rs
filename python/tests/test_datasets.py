from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

import fpm_rs as fpm
from fpm_rs.datasets.cli import build_parser, main


@pytest.fixture
def ordinary_dataset(tmp_path):
    """Generate a local installed bundle; its archive URL is a nonexistent local file."""
    cache = tmp_path / "cache"
    root = cache / "datasets" / "fixture" / "1"
    root.mkdir(parents=True)
    (cache / ".fpm-rs-dataset-cache").write_bytes(b"fpm-rs managed dataset cache v1\n")
    (root / ".fpm-rs-install.json").write_text(
        json.dumps(
            {
                "cache_format_version": 1,
                "id": "fixture",
                "version": "1",
                "archive_sha256": "0" * 64,
            }
        )
    )
    vectors = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]])
    optics = fpm.Optics(
        wavelength_vacuum_m=532e-9,
        objective_na=0.1,
        magnification=4.0,
        camera_pixel_size=6.5e-6,
    )
    illumination = fpm.Illumination(geometry=fpm.KVectorList(k_vectors=vectors))
    model = fpm.compile_model(
        optics=optics,
        illumination=illumination,
        image_shape=(8, 8),
        reconstruction_shape=(16, 16),
    )
    model.save_json(path=root / "kernel.json")
    kernel = json.loads((root / "kernel.json").read_text())
    experiment = {
        "optics": {
            "wavelength_vacuum_m": 532e-9,
            "objective_na": 0.1,
            "magnification": 4.0,
            "camera_pixel_size": 6.5e-6,
            "illumination_refractive_index": 1.0,
            "objective_medium_refractive_index": 1.0,
            "defocus_distance": None,
            "pupil_aberration": None,
        },
        "illumination": {
            "geometry": {
                "kind": "k_vector_list",
                "k_vectors": [{"kx": x, "ky": y} for x, y in vectors],
            },
            "calibration": {"relative_power": None},
            "acquisition": {
                "frames": [
                    {
                        "contributions": [{"source": i, "intensity_weight": 1.0}],
                        "gain": 1.0,
                    }
                    for i in range(3)
                ]
            },
        },
        "optical_background": None,
    }
    (root / "configuration.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "true_experiment": experiment,
                "reconstruction_experiment": experiment,
                "image_shape": [8, 8],
                "reconstruction_shape": [16, 16],
                "compiled_models": {
                    "true_model": kernel,
                    "reconstruction_model": kernel,
                },
                "camera": None,
                "illumination_acquisition_errors": None,
                "random_seed": 17,
            }
        )
    )

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    frames = []
    for i, source in enumerate([91, None, 73]):
        name = f"frame-{i}.png"
        pixels = np.arange(64, dtype=np.uint8).reshape(8, 8) + i
        payload = b"".join(b"\0" + row.tobytes() for row in pixels)
        (root / name).write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(payload))
            + chunk(b"IEND", b"")
        )
        frames.append({"path": name, "illumination_index": source})
    (root / "measurements.json").write_text(json.dumps({"frames": frames}))
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "measurement_manifest": "measurements.json",
                "configuration": "configuration.json",
                "provenance": {"source": "generated", "dataset_version": "1"},
                "measurement_units": "camera counts",
            }
        )
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "registry_version": 1,
                "datasets": [
                    {
                        "id": "fixture",
                        "version": "1",
                        "title": "Fixture",
                        "description": "Generated fixture",
                        "format_version": 1,
                        "archive": {
                            "url": (tmp_path / "missing.tar.zst").as_uri(),
                            "sha256": "0" * 64,
                            "size_bytes": 1,
                        },
                        "license": {
                            "spdx": "CC0",
                            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                        },
                        "citation": {
                            "doi": "fixture",
                            "text": "Generated test fixture",
                        },
                        "source": {
                            "url": root.as_uri(),
                            "description": "Generated offline",
                        },
                        "tags": ["test"],
                    }
                ],
            }
        )
    )
    return fpm.DatasetRegistry(registry_url=str(registry), cache_dir=cache).open(
        "fixture"
    )


def test_dataset_subset_records_resolved_provenance_in_benchmark(
    ordinary_dataset, tmp_path
):
    pl = pytest.importorskip("polars")
    dataset = ordinary_dataset
    subset = dataset.subset(frames=[2, 1, 0], crop=(1, 2, 4, 4))
    assert subset.spatial_crop == (1, 2, 4, 4)
    assert subset.measurements.shape == (3, 4, 4)
    assert subset.reconstruction_model.reconstruction_shape == (8, 8)
    assert subset.true_model.image_shape == (4, 4)
    np.testing.assert_array_equal(
        subset.measurements.array, dataset.measurements.array[[2, 1, 0], 1:5, 2:6]
    )
    assert subset.ground_truth_object is None and subset.valid_object_mask is None
    assert subset.measurement_units == "camera counts"
    assert dataset.subset(every_nth_frame=2).spatial_crop == (0, 0, 8, 8)
    result = fpm.AlternatingProjection(iterations=1).run(
        problem=subset.reconstruction_problem()
    )
    suite = fpm.BenchmarkSuite("subset-suite")
    run_id = suite.add_result(
        result=result,
        case_id="resolved-subset",
        dataset_name="fixture",
        dataset_subset=subset,
    )
    bundle = suite.write_bundle(path=tmp_path / "benchmark")
    assert (
        json.loads(bundle.manifest_path.read_text())["benchmark_bundle_format_version"]
        == 2
    )
    runs = pl.read_parquet(bundle.tables.runs.path)
    frames = pl.read_parquet(bundle.tables.frames.path)
    assert runs.select(
        "crop_row", "crop_column", "crop_height", "crop_width"
    ).rows() == [(1, 2, 4, 4)]
    assert runs["dataset_version"].to_list() == ["1"]
    assert frames["original_frame_index"].to_list() == [2, 1, 0]
    assert frames["original_illumination_index"].to_list() == [73, None, 91]
    assert frames["normalized_l2"].null_count() == 3
    assert (
        bundle.results[run_id].result.metadata["measurement_units"] == "camera counts"
    )
    assert fpm.read_benchmark_bundle(path=bundle.path).results[run_id].run_id == run_id
    with pytest.raises(fpm.InvalidParameterError, match="frame count and shapes"):
        suite.add_result(
            result=result,
            case_id="bad",
            dataset_name="fixture",
            dataset_subset=dataset.subset(),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"frames": []},
        {"frames": [0, 0]},
        {"frames": [3]},
        {"every_nth_frame": 0},
        {"crop": (0, 0, 0, 4)},
        {"crop": (6, 6, 4, 4)},
    ],
)
def test_dataset_subset_rejects_invalid_selection(ordinary_dataset, arguments):
    with pytest.raises(fpm.DatasetError):
        ordinary_dataset.subset(**arguments)


def test_dataset_subset_rejects_conflicting_selectors(ordinary_dataset):
    with pytest.raises(ValueError, match="mutually exclusive"):
        ordinary_dataset.subset(frames=[0], every_nth_frame=1)


def test_registry_lists_an_empty_local_document_and_exposes_configuration(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"registry_version": 1, "datasets": []}), encoding="utf-8"
    )
    cache = tmp_path / "cache"
    registry = fpm.DatasetRegistry(registry_url=str(registry_path), cache_dir=cache)

    assert registry.registry_url == str(registry_path)
    assert registry.cache_dir == cache
    assert registry.list() == []
    assert registry.download_all() == []
    assert registry.clean_all() == 0


def test_registry_errors_remain_typed_dataset_errors(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"registry_version": 1, "datasets": []}), encoding="utf-8"
    )
    registry = fpm.DatasetRegistry(
        registry_url=str(registry_path), cache_dir=tmp_path / "cache"
    )

    with pytest.raises(fpm.DatasetError, match="no entry 'missing'"):
        registry.open("missing")
    with pytest.raises(fpm.DatasetError, match="no entry 'missing'"):
        fpm.open_dataset(
            "missing", registry_url=str(registry_path), cache_dir=tmp_path / "cache"
        )


def test_python_cli_supports_every_command_and_lists_cache_state(tmp_path, capsys):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"registry_version": 1, "datasets": []}), encoding="utf-8"
    )
    common = [
        "--registry-url",
        str(registry_path),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]

    assert main([*common, "list"]) == 0
    assert capsys.readouterr().out == "ID\tVERSION\tCACHED\tSIZE_BYTES\tTITLE\n"
    assert main([*common, "download", "--all"]) == 0
    assert main([*common, "clean", "fixture"]) == 0
    assert "removed=false" in capsys.readouterr().out
    assert main([*common, "clean", "--all"]) == 0
    assert "removed=0" in capsys.readouterr().out

    for arguments in [
        ["list"],
        ["download", "fixture"],
        ["download", "--all"],
        ["open", "fixture"],
        ["clean", "fixture"],
        ["clean", "--all"],
    ]:
        build_parser().parse_args(arguments)
