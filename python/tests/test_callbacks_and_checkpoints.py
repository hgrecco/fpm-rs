from __future__ import annotations

from pathlib import Path

import pytest

import fpm_rs as fpm


def test_built_in_and_python_callbacks(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    events: list[dict[str, object]] = []
    result = fpm.AlternatingProjection(iterations=2).run(
        problem,
        callbacks=[
            fpm.IterationCallback(events.append),
            fpm.CsvLogger(tmp_path / "loss.csv"),
            fpm.CheckpointEvery(1, tmp_path / "checkpoints"),
            fpm.SaveImageEvery(2, tmp_path / "images"),
        ],
    )

    assert result.runtime.completed_iterations == 2
    assert [event["iteration"] for event in events] == [1, 2]
    assert (tmp_path / "loss.csv").is_file()
    assert (tmp_path / "checkpoints" / "checkpoint_00001.json").is_file()
    assert (tmp_path / "checkpoints" / "checkpoint_00002.json").is_file()
    assert (tmp_path / "images" / "amplitude_00002.png").is_file()


def test_checkpoint_load_save_and_resume(
    problem: fpm.ReconstructionProblem,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "checkpoints"
    fpm.AlternatingProjection(iterations=1).run(
        problem,
        callbacks=[fpm.CheckpointEvery(1, directory)],
    )
    checkpoint = fpm.ReconstructionCheckpoint.load(directory / "checkpoint_00001.json")
    copied = tmp_path / "copied.json"
    checkpoint.save(copied)
    restored = fpm.ReconstructionCheckpoint.load(copied)

    result = fpm.AlternatingProjection(iterations=2).run(
        problem,
        resume_from=restored,
    )
    assert checkpoint.format_version == 1
    assert checkpoint.completed_iterations == 1
    assert result.runtime.completed_iterations == 2
    assert len(result.history) == 2


def test_python_callback_can_stop_early(problem: fpm.ReconstructionProblem) -> None:
    result = fpm.AlternatingProjection(iterations=10).run(
        problem,
        callbacks=[fpm.IterationCallback(lambda _context: False)],
    )
    assert result.runtime.stopped_early
    assert result.runtime.completed_iterations == 1


def test_python_callback_exception_propagates(
    problem: fpm.ReconstructionProblem,
) -> None:
    def fail(_context: object) -> None:
        raise LookupError("callback failed")

    with pytest.raises(LookupError, match="callback failed"):
        fpm.AlternatingProjection(iterations=2).run(
            problem,
            callbacks=[fpm.IterationCallback(fail)],
        )


def test_python_callback_can_be_reused_after_an_exception(
    problem: fpm.ReconstructionProblem,
) -> None:
    calls = 0

    def fail_once(_context: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LookupError("first callback failure")

    callback = fpm.IterationCallback(fail_once)
    with pytest.raises(LookupError, match="first callback failure"):
        fpm.AlternatingProjection(iterations=1).run(problem, callbacks=[callback])

    result = fpm.AlternatingProjection(iterations=1).run(problem, callbacks=[callback])
    assert result.runtime.completed_iterations == 1
