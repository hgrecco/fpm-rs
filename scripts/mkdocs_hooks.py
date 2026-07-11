"""MkDocs hooks for preparing a static view of the typed Python API."""

from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python" / "fpm_rs"
API_ROOT = ROOT / "target" / "docs-python-api"


def on_config(config, **kwargs):  # noqa: ANN001, ANN003, ANN201
    """Stage the canonical stub and pure-Python modules for mkdocstrings."""
    package = API_ROOT / "fpm_rs"
    shutil.rmtree(API_ROOT, ignore_errors=True)
    package.mkdir(parents=True)
    shutil.copy2(PYTHON_SOURCE / "__init__.pyi", package / "__init__.py")
    shutil.copy2(PYTHON_SOURCE / "plot.py", package / "plot.py")
    shutil.copytree(
        PYTHON_SOURCE / "diagnostics",
        package / "diagnostics",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return config
