"""Validate keyword-oriented calls across first-touch Python examples."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import re
import sys

import fpm_rs as fpm


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_EXAMPLES = (
    Path("README.md"),
    Path("docs/index.md"),
    Path("docs/getting-started/quickstart.md"),
)
NOTEBOOK_EXAMPLE = Path("docs/tutorials/notebooks/quickstart.ipynb")
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
REQUIRED_KEYWORDS = {
    "fpm.Optics": {
        "wavelength_vacuum_m",
        "objective_na",
        "magnification",
        "camera_pixel_size",
    },
    "fpm.PlanarLEDArray": {"shape", "pitch_m", "reference_index", "pose"},
    "fpm.ArrayPose.from_translation": {"translation_m"},
    "fpm.Illumination": {"geometry"},
    "fpm.compile_model": {"optics", "illumination", "image_shape"},
    "fpm.simulate": {"true_model", "object", "seed"},
    "fpm.ReconstructionProblem": {"measurements", "model"},
    "fpm.AlternatingProjection": {"iterations"},
}
PUBLIC_CALLS = {
    "fpm.Optics": fpm.Optics,
    "fpm.PlanarLEDArray": fpm.PlanarLEDArray,
    "fpm.ArrayPose.from_translation": fpm.ArrayPose.from_translation,
    "fpm.Illumination": fpm.Illumination,
    "fpm.compile_model": fpm.compile_model,
    "fpm.simulate": fpm.simulate,
    "fpm.ReconstructionProblem": fpm.ReconstructionProblem,
    "fpm.AlternatingProjection": fpm.AlternatingProjection,
}
VALID_KEYWORDS = {
    name: set(inspect.signature(call).parameters) for name, call in PUBLIC_CALLS.items()
}


def dotted_name(node: ast.expr) -> str | None:
    """Return a dotted call name for simple names and attributes."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def markdown_python(path: Path) -> str:
    """Return the concatenated Python fences from a Markdown document."""
    source = path.read_text(encoding="utf-8")
    blocks = PYTHON_FENCE.findall(source)
    if not blocks:
        raise RuntimeError(f"{path}: no Python code fence found")
    return "\n\n".join(blocks)


def notebook_python(path: Path) -> str:
    """Return the concatenated code cells from a notebook document."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def validate(path: Path, source: str) -> list[str]:
    """Return calling-convention failures for one onboarding surface."""
    module = ast.parse(source, filename=str(path))
    seen: set[str] = set()
    failures: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        name = dotted_name(node.func)
        if name not in REQUIRED_KEYWORDS:
            continue
        seen.add(name)
        if node.args:
            failures.append(f"{path}:{node.lineno}: {name} uses positional arguments")
        supplied = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
        unknown = sorted(supplied - VALID_KEYWORDS[name])
        if unknown:
            failures.append(
                f"{path}:{node.lineno}: {name} uses unknown keyword arguments: "
                + ", ".join(unknown)
            )
        missing = sorted(REQUIRED_KEYWORDS[name] - supplied)
        if missing:
            failures.append(
                f"{path}:{node.lineno}: {name} is missing keyword arguments: "
                + ", ".join(missing)
            )
    for name in sorted(REQUIRED_KEYWORDS.keys() - seen):
        failures.append(f"{path}: required onboarding call is missing: {name}")
    return failures


def main() -> None:
    """Fail when an onboarding example drifts from keyword-oriented calls."""
    invalid_requirements = [
        f"{name}: checker requires unknown keyword arguments: "
        + ", ".join(sorted(required - VALID_KEYWORDS[name]))
        for name, required in REQUIRED_KEYWORDS.items()
        if required - VALID_KEYWORDS[name]
    ]
    if invalid_requirements:
        print("Quickstart checker configuration is invalid:", file=sys.stderr)
        for failure in invalid_requirements:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    surfaces = {path: markdown_python(ROOT / path) for path in MARKDOWN_EXAMPLES}
    surfaces[NOTEBOOK_EXAMPLE] = notebook_python(ROOT / NOTEBOOK_EXAMPLE)
    failures = [
        failure
        for path, source in surfaces.items()
        for failure in validate(path, source)
    ]
    if failures:
        print("Quickstart calling-convention check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated keyword-oriented calls in {len(surfaces)} onboarding surfaces")


if __name__ == "__main__":
    main()
