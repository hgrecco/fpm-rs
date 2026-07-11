from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .report import make_diagnostic_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fpm-diagnostics")
    parser.add_argument("diagnostics_json", type=Path, help="path to diagnostics.json")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("diagnostic_report"),
        help="output directory",
    )
    args = parser.parse_args(argv)
    try:
        make_diagnostic_report(args.diagnostics_json, args.output_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
