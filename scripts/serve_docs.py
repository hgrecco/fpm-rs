"""Build the local extension and run MkDocs' live authoring server."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        ["maturin", "develop", "--skip-install", "--locked"],
        cwd=ROOT,
        check=True,
    )
    try:
        subprocess.run(
            ["mkdocs", "serve", "--strict", "--dev-addr", "127.0.0.1:8000"],
            cwd=ROOT,
            check=True,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
