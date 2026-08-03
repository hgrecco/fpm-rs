"""Fail when a supposedly complete pytest run contains skipped tests."""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def main() -> None:
    report = Path(sys.argv[1])
    root = ET.parse(report).getroot()
    skipped = []
    for case in root.iter("testcase"):
        skip = case.find("skipped")
        if skip is not None:
            skipped.append(
                f"{case.get('classname', '')}::{case.get('name', '')}: "
                f"{skip.get('message', 'skipped')}"
            )
    if skipped:
        details = "\n".join(f"- {item}" for item in skipped)
        raise SystemExit(f"complete Python test run contained skips:\n{details}")
    print(f"No skipped tests in {report}")


if __name__ == "__main__":
    main()
