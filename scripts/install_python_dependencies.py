"""Install centralized Python dependency groups without installing fpm-rs."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) == 1:
        raise SystemExit(
            "usage: install_python_dependencies.py [build|runtime|OPTIONAL-EXTRA] [...]"
        )
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    optional = project.get("optional-dependencies", {})
    requirements: list[str] = []
    for group in sys.argv[1:]:
        if group == "build":
            requirements.extend(metadata["build-system"]["requires"])
        elif group == "runtime":
            requirements.extend(project.get("dependencies", []))
        elif group in optional:
            requirements.extend(optional[group])
        else:
            choices = ", ".join(["build", "runtime", *sorted(optional)])
            raise SystemExit(
                f"unknown dependency group {group!r}; choose from {choices}"
            )

    unique_requirements = list(dict.fromkeys(requirements))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *unique_requirements],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
