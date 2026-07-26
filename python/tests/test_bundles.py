from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fpm_rs as fpm


def test_result_bundle_handles_lazy_arrays_cache_and_clear(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    result = fpm.Admm(iterations=2, batch_size=1).run(problem)
    bundle = result.write_bundle(
        tmp_path / "result",
        run_id="python-run",
        label="Python round trip",
        include_previews=True,
    )

    assert bundle.run_id == "python-run"
    assert bundle.label == "Python round trip"
    assert bundle.manifest_path.is_file()
    assert bundle.tables.summary.path.is_file()
    assert bundle.tables.history.path.is_file()
    assert bundle.tables.algorithm_metrics is not None
    assert bundle.tables.algorithm_metrics.path.is_file()
    assert bundle.previews.object_amplitude is not None
    assert bundle.previews.object_amplitude.path.is_file()
    assert bundle.diagnostics is None
    assert bundle.evaluation is None

    object_array = bundle.arrays.object.value
    assert object_array.dtype == np.complex128
    assert object_array.flags.c_contiguous
    assert not object_array.flags["W"]
    assert bundle.arrays.object.value is object_array
    assert bundle.result.object is object_array
    with pytest.raises(ValueError, match="read-only"):
        object_array[0, 0] = 0.0

    verification = bundle.verify()
    assert verification.artifact_count >= 10
    assert verification.total_bytes > 0

    reopened = fpm.read_bundle(bundle.path)
    np.testing.assert_array_equal(reopened.result.object, result.object)
    bundle.clear_cache()
    reloaded = bundle.arrays.object.value
    assert reloaded is not object_array
    np.testing.assert_array_equal(reloaded, object_array)
    assert object_array.shape == reloaded.shape


def test_result_bundle_chooses_unique_final_paths(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    result = fpm.AlternatingProjection(iterations=1).run(problem)
    first = result.write_bundle(tmp_path / "result", include_previews=False)
    second = result.write_bundle(tmp_path / "result", include_previews=False)

    assert first.path != second.path
    assert first.path.is_dir()
    assert second.path.is_dir()
    assert not (tmp_path / "result.inprogress").exists()
    assert first.previews.object_amplitude is None


def test_result_bundle_detects_corruption_on_first_array_access(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    result = fpm.AlternatingProjection(iterations=1).run(problem)
    bundle = result.write_bundle(tmp_path / "result", include_previews=False)

    with bundle.arrays.object.path.open("r+b") as stream:
        stream.seek(0)
        stream.write(b"X")
        stream.flush()

    reopened = fpm.read_bundle(bundle.path)
    with pytest.raises(fpm.SerializationError, match="SHA-256"):
        _ = reopened.arrays.object.value


def test_benchmark_suite_writes_normalized_tables_and_nested_results(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    result = fpm.AlternatingProjection(iterations=1).run(problem)
    suite = fpm.BenchmarkSuite("python-comparison")
    first = suite.add_result(
        result,
        case_id="same-case",
        dataset_name="synthetic",
        algorithm_configuration="iterations=1",
    )
    second = suite.add_result(
        result,
        case_id="same-case",
        dataset_name="synthetic",
        algorithm_configuration="iterations=1",
    )
    assert first != second

    bundle = suite.write_bundle(tmp_path / "benchmark", label="repeats")
    assert bundle.name == "python-comparison"
    assert bundle.label == "repeats"
    assert bundle.tables.runs.path.is_file()
    assert bundle.tables.frames.path.is_file()
    assert bundle.tables.artifacts.path.is_file()
    assert bundle.tables.metadata.path.is_file()
    assert set(bundle.results) == {first, second}
    assert bundle.results[first].run_id == first
    assert bundle.results[first].result.runtime.completed_iterations == 1

    reopened = fpm.read_benchmark_bundle(bundle.path)
    assert set(reopened.results) == {first, second}


def test_bundle_tables_are_directly_scannable_when_polars_is_installed(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    pl = pytest.importorskip("polars")
    result = fpm.AlternatingProjection(iterations=2).run(problem)
    bundle = result.write_bundle(tmp_path / "result", run_id="polars-run")

    history = pl.scan_parquet(bundle.tables.history.path).collect()
    assert history.columns == ["run_id", "iteration", "objective", "elapsed_seconds"]
    assert history["run_id"].to_list() == ["polars-run", "polars-run"]
