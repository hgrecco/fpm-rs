"""Ensure every public reconstruction algorithm appears in selection guidance."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "python" / "fpm_rs" / "__init__.pyi"
GUIDE = ROOT / "docs" / "guides" / "reconstruction.md"
SECTION_START = "## Choose an algorithm"
SECTION_END = "## Select and run an algorithm"
TYPE_IN_CODE = re.compile(r"`([A-Za-z][A-Za-z0-9]*)`")


def public_algorithm_types() -> set[str]:
    """Return public stub classes that inherit the common algorithm surface."""
    module = ast.parse(STUB.read_text(encoding="utf-8"), filename=str(STUB))
    algorithms = set()
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        base_names = {base.id for base in node.bases if isinstance(base, ast.Name)}
        if "_Algorithm" in base_names:
            algorithms.add(node.name)
    return algorithms


def guided_algorithm_types() -> set[str]:
    """Return backticked types in the table's recommendation column."""
    document = GUIDE.read_text(encoding="utf-8")
    try:
        section = document.split(SECTION_START, 1)[1].split(SECTION_END, 1)[0]
    except IndexError as error:
        raise RuntimeError(
            "algorithm-selection section boundaries are missing"
        ) from error

    guided = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"Condition or priority", "---"}:
            continue
        guided.update(TYPE_IN_CODE.findall(cells[1]))
    return guided


def main() -> None:
    """Fail when a public algorithm lacks a recommendation-table entry."""
    algorithms = public_algorithm_types()
    guided = guided_algorithm_types()
    missing = sorted(algorithms - guided)
    if missing:
        print(
            "Reconstruction guidance is missing public algorithm types: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"Validated reconstruction guidance for {len(algorithms)} algorithms")


if __name__ == "__main__":
    main()
