"""Validate release version metadata and tag provenance before artifact builds."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def output(*command: str) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def main() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = pyproject["project"]["version"]
    metadata = json.loads(
        output("cargo", "metadata", "--no-deps", "--locked", "--format-version", "1")
    )
    inconsistent = {
        package["name"]: package["version"]
        for package in metadata["packages"]
        if package["version"] != expected
    }
    if inconsistent:
        raise SystemExit(
            f"workspace package versions do not match {expected}: {inconsistent}"
        )

    ref_type = os.environ.get("GITHUB_REF_TYPE")
    ref_name = os.environ.get("GITHUB_REF_NAME")
    if ref_type == "tag":
        expected_tag = f"v{expected}"
        if ref_name != expected_tag:
            raise SystemExit(f"release tag {ref_name!r} must be {expected_tag!r}")
        head = output("git", "rev-parse", "HEAD")
        tagged = output("git", "rev-list", "-n", "1", ref_name)
        if head != tagged:
            raise SystemExit(f"checked-out commit {head} is not tagged commit {tagged}")

    print(f"Validated fpm-rs {expected} at {output('git', 'rev-parse', 'HEAD')}")


if __name__ == "__main__":
    main()
