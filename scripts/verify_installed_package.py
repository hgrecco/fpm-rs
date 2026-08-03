"""Confirm that a clean environment imports the expected installed package."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version")
    arguments = parser.parse_args()

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = arguments.expected_version or metadata["project"]["version"]

    import fpm_rs

    location = Path(fpm_rs.__file__).resolve()
    source_package = (ROOT / "python" / "fpm_rs").resolve()
    if location == source_package or source_package in location.parents:
        raise SystemExit(f"import resolved to the source checkout: {location}")
    if fpm_rs.__version__ != expected:
        raise SystemExit(
            f"installed package reports {fpm_rs.__version__!r}, expected {expected!r}"
        )
    print(f"Imported fpm_rs {fpm_rs.__version__} from {location}")


if __name__ == "__main__":
    main()
