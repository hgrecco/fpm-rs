"""Build workspace rustdoc with complete features and warnings denied."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the canonical strict rustdoc build."""
    environment = os.environ.copy()
    environment["RUSTDOCFLAGS"] = "-D warnings"
    command = (
        "cargo",
        "doc",
        "--workspace",
        "--all-features",
        "--no-deps",
        "--locked",
    )
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
