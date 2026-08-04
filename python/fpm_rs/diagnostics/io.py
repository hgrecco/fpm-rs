"""Diagnostics JSON, output-directory, and Matplotlib file helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_diagnostics(path: str | Path) -> dict[str, Any]:
    """Load a diagnostics JSON object from ``path``.

    Raises ``FileNotFoundError`` for a missing path and ``ValueError`` when the
    file is invalid JSON or its top level is not an object.
    """
    diagnostics_path = Path(path)
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"diagnostics file not found: {diagnostics_path}")
    try:
        with diagnostics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid diagnostics JSON in {diagnostics_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"diagnostics JSON must contain an object at the top level, got {type(payload).__name__}"
        )
    return payload


def coerce_diagnostics(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Copy a diagnostics mapping or load one from a JSON filesystem path."""
    if isinstance(value, (str, Path)):
        return load_diagnostics(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "diagnostics must be a mapping or a path to a diagnostics JSON file"
    )


def ensure_output_dir(path: str | Path) -> Path:
    """Create ``path`` and missing parents, then return it as a ``Path``."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def latest_frame_diagnostics(value: Any) -> list[Any]:
    """Select entries from the greatest tagged iteration.

    A non-list returns an empty list. If no numeric ``iteration`` tags exist,
    the original list is returned unchanged.
    """
    if not isinstance(value, list):
        return []
    tagged_iterations = [
        entry.get("iteration")
        for entry in value
        if isinstance(entry, dict)
        and isinstance(entry.get("iteration"), (int, float))
        and not isinstance(entry.get("iteration"), bool)
    ]
    if not tagged_iterations:
        return value
    latest_iteration = max(tagged_iterations)
    return [
        entry
        for entry in value
        if isinstance(entry, dict) and entry.get("iteration") == latest_iteration
    ]


def savefig(path: str | Path, dpi: int = 180) -> None:
    """Save Matplotlib's current figure with tight bounds at ``dpi`` resolution."""
    import matplotlib.pyplot as plt

    figure = plt.gcf()
    figure.savefig(Path(path), dpi=dpi, bbox_inches="tight")
