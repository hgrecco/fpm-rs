"""Validate DOI citation syntax in authored documentation sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DOI = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
DOI_RE = re.compile(DOI, re.IGNORECASE)
CANONICAL_LINK_RE = re.compile(
    rf"\[(?P<label>[^]]+)]\(https://doi\.org/(?P<doi>{DOI})\)",
    re.IGNORECASE,
)
NONCANONICAL_RESOLVER_RE = re.compile(
    r"https?://(?:dx\.)?doi\.org/|https://dx\.doi\.org/", re.IGNORECASE
)
BARE_DOI_PREFIX_RE = re.compile(r"\bdoi\s*:\s*10\.", re.IGNORECASE)
TRAILING_PROSE = ".,;:"


@dataclass(frozen=True)
class Finding:
    """One source location that violates the citation-link policy."""

    path: Path
    offset: int
    message: str

    def render(self, text: str) -> str:
        """Render the finding with a repository-relative path and line number."""
        line = text.count("\n", 0, self.offset) + 1
        return f"{self.path.relative_to(ROOT)}:{line}: {self.message}"


def source_paths() -> list[Path]:
    """Return authored sources whose prose can enter public documentation."""
    paths = [ROOT / "README.md"]
    patterns = (
        (ROOT / "docs", ("*.md", "*.ipynb")),
        (ROOT / "src", ("*.rs",)),
        (ROOT / "python" / "src", ("*.rs",)),
        (ROOT / "python" / "fpm_rs", ("*.py", "*.pyi")),
        (ROOT / "examples", ("*.rs",)),
    )
    for directory, globs in patterns:
        for glob in globs:
            paths.extend(directory.rglob(glob))
    return sorted(set(path for path in paths if path.exists()))


def citation_findings(path: Path, text: str) -> list[Finding]:
    """Find malformed resolver URLs and DOI identifiers outside canonical links."""
    findings: list[Finding] = []
    canonical_doi_spans: set[tuple[int, int]] = set()
    for match in CANONICAL_LINK_RE.finditer(text):
        label = match.group("label")
        doi_span = match.span("doi")
        canonical_doi_spans.add(doi_span)
        if DOI_RE.search(label):
            findings.append(
                Finding(
                    path,
                    match.start("label"),
                    "use a title or author-year link label, not a DOI",
                )
            )

    for match in NONCANONICAL_RESOLVER_RE.finditer(text):
        if match.group(0).lower() != "https://doi.org/":
            findings.append(
                Finding(
                    path,
                    match.start(),
                    "DOI links must use the https://doi.org/ resolver",
                )
            )

    for match in BARE_DOI_PREFIX_RE.finditer(text):
        findings.append(
            Finding(
                path,
                match.start(),
                "bare DOI identifiers must be canonical Markdown links",
            )
        )

    for match in DOI_RE.finditer(text):
        end = match.end()
        while end > match.start() and text[end - 1] in TRAILING_PROSE:
            end -= 1
        if (
            end > match.start()
            and text[end - 1] == ")"
            and (match.start(), end) not in canonical_doi_spans
        ):
            end -= 1
        span = (match.start(), end)
        if span not in canonical_doi_spans:
            findings.append(
                Finding(
                    path,
                    match.start(),
                    "DOI identifier is not the target of a canonical Markdown citation link",
                )
            )
    return findings


def main() -> int:
    """Run citation syntax validation over all authored documentation sources."""
    sources = source_paths()
    rendered: list[str] = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        rendered.extend(
            finding.render(text) for finding in citation_findings(path, text)
        )
    if rendered:
        print("Citation syntax check failed:", file=sys.stderr)
        for finding in rendered:
            print(f"  {finding}", file=sys.stderr)
        return 1
    print(f"Citation syntax check passed ({len(sources)} sources).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
