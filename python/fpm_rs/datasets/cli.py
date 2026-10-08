from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fpm_rs import DatasetError, DatasetRegistry


def _selection(subparsers: argparse._SubParsersAction, name: str, help: str) -> None:
    parser = subparsers.add_parser(name, help=help)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("id", nargs="?", help="dataset identifier")
    selection.add_argument("--all", action="store_true", help="apply to every dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpm-datasets", description="Discover and manage fpm-rs datasets"
    )
    parser.add_argument("--registry-url", help="registry URL, file URL, or JSON path")
    parser.add_argument(
        "--cache-dir", type=Path, help="managed dataset cache directory"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list registry entries and cache status")
    _selection(commands, "download", "download one dataset or every registry entry")
    opened = commands.add_parser(
        "open", help="download if needed, then open one dataset"
    )
    opened.add_argument("id", help="dataset identifier")
    opened.add_argument(
        "--spectral",
        action="store_true",
        help="open an explicit version-two spectral profile",
    )
    _selection(commands, "clean", "remove one dataset or the complete managed cache")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        registry = DatasetRegistry(
            registry_url=args.registry_url,
            cache_dir=args.cache_dir,
        )
        if args.command == "list":
            print("ID\tVERSION\tCACHED\tSIZE_BYTES\tTITLE")
            for entry in registry.list():
                print(
                    f"{entry.id}\t{entry.version}\t"
                    f"{'yes' if entry.cached else 'no'}\t"
                    f"{entry.archive_size_bytes}\t{entry.title}"
                )
        elif args.command == "download":
            paths = (
                registry.download_all() if args.all else [registry.download(args.id)]
            )
            for path in paths:
                print(path)
        elif args.command == "open":
            dataset = (
                registry.open_spectral(args.id)
                if args.spectral
                else registry.open(args.id)
            )
            print(f"id={args.id}")
            print(f"path={dataset.path}")
            print(f"frames={dataset.measurements.frame_count}")
            print(f"image_shape={dataset.measurements.image_shape}")
            model = dataset.model if args.spectral else dataset.reconstruction_model
            print(f"reconstruction_shape={model.reconstruction_shape}")
            if args.spectral:
                print(f"channels={len(model.channel_ids)}")
        elif args.command == "clean":
            removed = registry.clean_all() if args.all else registry.clean(args.id)
            rendered = str(removed).lower() if isinstance(removed, bool) else removed
            print(f"removed={rendered}")
    except (DatasetError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0
