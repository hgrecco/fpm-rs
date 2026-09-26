"""Run the Python API coverage, synchronization, and citation checks."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    "check_python_docs.py",
    "check_python_api.py",
    "check_algorithm_guidance.py",
    "check_citations.py",
)


def main() -> None:
    """Run every documentation-source validation with the active interpreter."""
    for script in CHECKS:
        command = (sys.executable, str(ROOT / "scripts" / script))
        print(f"+ {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
