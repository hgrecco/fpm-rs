"""Record, inspect, collect, and verify immutable release distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz")
REQUIRED_SDIST_PATHS = (
    "benches/forward_model.rs",
    "benches/gradient_parallel.rs",
    "Cargo.lock",
    "Cargo.toml",
    "pyproject.toml",
    "python/Cargo.toml",
    "python/fpm_rs/__init__.py",
    "python/src/lib.rs",
    "src/lib.rs",
)
PYTHON_TAGS = {"cp312": "3.12", "cp313": "3.13", "cp314": "3.14"}
PLATFORMS = {
    "linux-x86_64",
    "linux-aarch64",
    "macos-x86_64",
    "macos-aarch64",
    "windows-x86_64",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distributions(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name.endswith(DISTRIBUTION_SUFFIXES)
    )


def write_record(directory: Path, output: Path) -> None:
    files = distributions(directory)
    if len(files) != 1:
        raise SystemExit(
            f"expected one distribution in {directory}, found {len(files)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"files": {files[0].name: sha256(files[0])}}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {files[0].name} in {output}")


def verify_record(directory: Path, record: Path) -> None:
    expected = json.loads(record.read_text(encoding="utf-8"))["files"]
    actual = {path.name: sha256(path) for path in distributions(directory)}
    if actual != expected:
        raise SystemExit(f"artifact hash mismatch: expected {expected}, found {actual}")
    print(f"Verified {len(actual)} immutable distribution(s) against {record}")


def classify_platform(platform_tags: set[str]) -> str:
    joined = " ".join(platform_tags)
    if "manylinux" in joined and "x86_64" in joined:
        return "linux-x86_64"
    if "manylinux" in joined and "aarch64" in joined:
        return "linux-aarch64"
    if "macosx" in joined and "x86_64" in joined:
        return "macos-x86_64"
    if "macosx" in joined and "arm64" in joined:
        return "macos-aarch64"
    if "win_amd64" in joined:
        return "windows-x86_64"
    raise SystemExit(f"unsupported wheel platform tags: {sorted(platform_tags)}")


def inspect_distributions(directory: Path, version: str) -> None:
    from packaging.utils import parse_sdist_filename, parse_wheel_filename
    from packaging.version import Version

    files = distributions(directory)
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(wheels) != 15 or len(sdists) != 1:
        raise SystemExit(
            f"expected 15 wheels and one sdist, found {len(wheels)} and {len(sdists)}"
        )

    expected_version = Version(version)
    observed: set[tuple[str, str]] = set()
    for wheel in wheels:
        name, wheel_version, _, tags = parse_wheel_filename(wheel.name)
        if name != "fpm-rs" or wheel_version != expected_version:
            raise SystemExit(f"unexpected wheel identity: {wheel.name}")
        interpreters = {tag.interpreter for tag in tags}
        abis = {tag.abi for tag in tags}
        if len(interpreters) != 1 or interpreters != abis:
            raise SystemExit(f"wheel must use a matching CPython ABI: {wheel.name}")
        interpreter = interpreters.pop()
        if interpreter not in PYTHON_TAGS:
            raise SystemExit(f"unsupported Python tag {interpreter}: {wheel.name}")
        platform = classify_platform({tag.platform for tag in tags})
        key = (PYTHON_TAGS[interpreter], platform)
        if key in observed:
            raise SystemExit(f"duplicate wheel coverage for {key}: {wheel.name}")
        observed.add(key)

    expected = {
        (python, platform) for python in PYTHON_TAGS.values() for platform in PLATFORMS
    }
    if observed != expected:
        raise SystemExit(
            f"wheel matrix mismatch; missing={expected - observed}, extra={observed - expected}"
        )

    sdist_name, sdist_version = parse_sdist_filename(sdists[0].name)
    if sdist_name != "fpm-rs" or sdist_version != expected_version:
        raise SystemExit(f"unexpected sdist identity: {sdists[0].name}")
    inspect_sdist(sdists[0])
    print(f"Validated release matrix for fpm-rs {version}")


def inspect_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    missing = [
        required
        for required in REQUIRED_SDIST_PATHS
        if not any(name.endswith(f"/{required}") for name in members)
    ]
    if missing:
        raise SystemExit(
            f"sdist {path.name} is missing required workspace files: {missing}"
        )
    print(f"Validated required workspace contents in {path.name}")


def collect(dist: Path, records: Path, output: Path) -> None:
    files = distributions(dist)
    approved: dict[str, set[str]] = {}
    for record in records.rglob("*.json"):
        data = json.loads(record.read_text(encoding="utf-8"))
        for name, digest in data.get("files", {}).items():
            approved.setdefault(name, set()).add(digest)

    actual = {path.name: sha256(path) for path in files}
    missing = set(actual) - set(approved)
    disagreements = {
        name: sorted(approved[name])
        for name, digest in actual.items()
        if name in approved and approved[name] != {digest}
    }
    if missing or disagreements:
        raise SystemExit(
            f"unapproved distributions={sorted(missing)}, hash disagreements={disagreements}"
        )

    package_dir = output / "distributions"
    package_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, package_dir / path.name)
    manifest = {"algorithm": "sha256", "files": actual}
    (output / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_manifest(output)


def verify_manifest(directory: Path) -> None:
    manifest_path = directory / "RELEASE_MANIFEST.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    actual = {
        path.name: sha256(path) for path in distributions(directory / "distributions")
    }
    if actual != expected:
        raise SystemExit(
            f"release manifest mismatch: expected {expected}, found {actual}"
        )
    print(f"Verified release manifest for {len(actual)} distributions")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record")
    record.add_argument("directory", type=Path)
    record.add_argument("output", type=Path)

    verify = commands.add_parser("verify-record")
    verify.add_argument("directory", type=Path)
    verify.add_argument("record", type=Path)

    inspect = commands.add_parser("inspect")
    inspect.add_argument("directory", type=Path)
    inspect.add_argument("version")

    sdist = commands.add_parser("inspect-sdist")
    sdist.add_argument("path", type=Path)

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("dist", type=Path)
    collect_parser.add_argument("records", type=Path)
    collect_parser.add_argument("output", type=Path)

    manifest = commands.add_parser("verify-manifest")
    manifest.add_argument("directory", type=Path)

    arguments = parser.parse_args()
    if arguments.command == "record":
        write_record(arguments.directory, arguments.output)
    elif arguments.command == "verify-record":
        verify_record(arguments.directory, arguments.record)
    elif arguments.command == "inspect":
        inspect_distributions(arguments.directory, arguments.version)
    elif arguments.command == "inspect-sdist":
        inspect_sdist(arguments.path)
    elif arguments.command == "collect":
        collect(arguments.dist, arguments.records, arguments.output)
    elif arguments.command == "verify-manifest":
        verify_manifest(arguments.directory)


if __name__ == "__main__":
    main()
