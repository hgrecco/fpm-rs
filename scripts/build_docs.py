"""Build the combined MkDocs, Python API, notebook, and rustdoc site."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
RUSTDOC_TARGET = ROOT / "target" / "docs-rust"


def run(*command: str, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    run("maturin", "develop", "--skip-install", "--locked")
    package = importlib.import_module("fpm_rs")
    print(f"Documenting fpm_rs {package.__version__}", flush=True)
    run(sys.executable, "scripts/check_api_docs.py")

    run(sys.executable, "-m", "mkdocs", "build", "--strict", "--clean")

    rustdoc_env = os.environ.copy()
    rustdoc_env["CARGO_TARGET_DIR"] = str(RUSTDOC_TARGET)
    rustdoc_env["RUSTDOCFLAGS"] = "-D warnings"
    run(
        "cargo",
        "doc",
        "--workspace",
        "--all-features",
        "--no-deps",
        "--locked",
        env=rustdoc_env,
    )

    rustdoc_source = RUSTDOC_TARGET / "doc"
    rustdoc_destination = SITE_DIR / "rust-api"
    shutil.rmtree(rustdoc_destination, ignore_errors=True)
    shutil.copytree(rustdoc_source, rustdoc_destination)

    rustdoc_index = rustdoc_destination / "index.html"
    if not rustdoc_index.exists():
        rustdoc_index.write_text(
            '<!doctype html><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="0; url=fpm_rs/index.html">'
            "<title>fpm-rs Rust API</title>"
            '<p><a href="fpm_rs/index.html">Open the fpm-rs Rust API.</a></p>\n',
            encoding="utf-8",
        )

    (SITE_DIR / ".nojekyll").touch()
    print(f"Combined documentation written to {SITE_DIR}", flush=True)


if __name__ == "__main__":
    main()
