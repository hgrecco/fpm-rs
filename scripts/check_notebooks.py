"""Execute the explicit documentation-notebook allowlist in isolated copies."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "target" / "docs-notebooks-check"
NOTEBOOKS = (
    Path("docs/tutorials/notebooks/quickstart.ipynb"),
    Path("docs/tutorials/notebooks/synthetic_objects_quickstart.ipynb"),
    Path("docs/tutorials/notebooks/diagnostics_quickstart.ipynb"),
)


def run(*command: str) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def execute_notebook(relative_path: Path) -> None:
    source = ROOT / relative_path
    destination = OUTPUT_ROOT / relative_path.name
    work_dir = OUTPUT_ROOT / relative_path.stem
    work_dir.mkdir(parents=True)

    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(work_dir)}},
    )
    print(f"Executing {relative_path}", flush=True)
    try:
        client.execute()
    except CellExecutionError as error:
        print(f"Notebook failed: {relative_path}", file=sys.stderr)
        raise SystemExit(1) from error
    nbformat.write(notebook, destination)
    print(f"Validated copy: {destination.relative_to(ROOT)}", flush=True)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    run("maturin", "develop", "--skip-install", "--locked")
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    OUTPUT_ROOT.mkdir(parents=True)
    for notebook in NOTEBOOKS:
        execute_notebook(notebook)
    print(f"Validated {len(NOTEBOOKS)} curated notebooks", flush=True)


if __name__ == "__main__":
    main()
