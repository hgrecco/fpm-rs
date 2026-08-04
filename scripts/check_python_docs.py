"""Enforce semantic documentation for the authoritative Python API sources."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python" / "fpm_rs"
API_SOURCES = (
    ROOT / "python" / "fpm_rs" / "__init__.pyi",
    ROOT / "python" / "fpm_rs" / "metrics.pyi",
    ROOT / "python" / "fpm_rs" / "evaluation.pyi",
    ROOT / "python" / "fpm_rs" / "plot.py",
    ROOT / "python" / "fpm_rs" / "datasets" / "__init__.py",
    ROOT / "python" / "fpm_rs" / "diagnostics" / "__init__.py",
    ROOT / "python" / "fpm_rs" / "diagnostics" / "io.py",
    ROOT / "python" / "fpm_rs" / "diagnostics" / "report.py",
)
REFERENCE_ROOT = ROOT / "docs" / "reference" / "python"
PLACEHOLDERS = re.compile(
    r"\b(?:todo|tbd|fixme|undocumented|documentation pending)\b", re.I
)
WORDS = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    name: str
    reason: str

    def render(self) -> str:
        relative = self.path.relative_to(ROOT)
        return f"{relative}:{self.line}: {self.name}: {self.reason}"


def is_public(name: str) -> bool:
    """Return whether a source-level name belongs to the documented surface."""
    return not name.startswith("_")


def semantic_problem(docstring: str | None) -> str | None:
    """Return a failure reason for missing or placeholder prose."""
    if docstring is None or not docstring.strip():
        return "missing docstring"
    if PLACEHOLDERS.search(docstring):
        return "docstring contains placeholder text"
    if len(WORDS.findall(docstring)) < 4:
        return "docstring is too short to convey semantics"
    return None


def adjacent_attribute_doc(body: list[ast.stmt], index: int) -> str | None:
    """Return a PEP 258 attribute docstring following ``body[index]``."""
    if index + 1 >= len(body):
        return None
    following = body[index + 1]
    if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
        if isinstance(following.value.value, str):
            return following.value.value
    return None


def assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """Return simple names assigned by a statement."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]


def inspect_body(
    path: Path,
    body: list[ast.stmt],
    prefix: str,
    findings: list[Finding],
    *,
    check_attributes: bool,
) -> None:
    """Inspect public definitions and documented attributes in one AST body."""
    for index, node in enumerate(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not is_public(node.name):
                continue
            qualified = f"{prefix}.{node.name}" if prefix else node.name
            if reason := semantic_problem(ast.get_docstring(node, clean=False)):
                findings.append(Finding(path, node.lineno, qualified, reason))
            continue

        if isinstance(node, ast.ClassDef):
            if not is_public(node.name):
                continue
            qualified = f"{prefix}.{node.name}" if prefix else node.name
            if reason := semantic_problem(ast.get_docstring(node, clean=False)):
                findings.append(Finding(path, node.lineno, qualified, reason))
            inspect_body(
                path,
                node.body,
                qualified,
                findings,
                check_attributes=True,
            )
            continue

        if check_attributes and isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in assignment_names(node):
                if not is_public(name):
                    continue
                qualified = f"{prefix}.{name}" if prefix else name
                if reason := semantic_problem(adjacent_attribute_doc(body, index)):
                    findings.append(Finding(path, node.lineno, qualified, reason))


def inspect_source(path: Path) -> list[Finding]:
    """Inspect one Python source or stub file."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[Finding] = []
    module_name = path.relative_to(ROOT / "python").with_suffix("").as_posix()
    module_name = module_name.replace("/__init__", "").replace("/", ".")
    if reason := semantic_problem(ast.get_docstring(module, clean=False)):
        findings.append(Finding(path, 1, module_name, reason))
    inspect_body(
        path,
        module.body,
        module_name,
        findings,
        check_attributes=path.suffix == ".pyi" or path.name == "plot.py",
    )
    return findings


def root_reference_findings() -> list[Finding]:
    """Ensure every public root class, function, and attribute is routed to a page."""
    stub_path = ROOT / "python" / "fpm_rs" / "__init__.pyi"
    module = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    names: list[tuple[str, int]] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (
            is_public(node.name) or node.name == "__version__"
        ):
            names.append((node.name, node.lineno))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in assignment_names(node):
                if is_public(name) or name == "__version__":
                    names.append((name, node.lineno))

    reference = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(REFERENCE_ROOT.glob("*.md"))
    )
    findings = []
    for name, line in names:
        patterns = (f"fpm_rs.{name}", f"- {name}\n")
        if not any(pattern in reference for pattern in patterns):
            findings.append(
                Finding(
                    stub_path,
                    line,
                    f"fpm_rs.{name}",
                    "not routed to a Python reference page",
                )
            )
    return findings


def exported_names(path: Path) -> list[tuple[str, int]]:
    """Return literal names in a module's ``__all__`` declaration."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            return [
                (item.value, item.lineno)
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
    return []


def module_directive_covers(reference: str, module_name: str, name: str) -> bool:
    """Return whether a module directive renders one exported member."""
    lines = reference.splitlines()
    marker = f"::: fpm_rs.{module_name}"
    for index, line in enumerate(lines):
        if line != marker:
            continue
        option_lines = []
        for following in lines[index + 1 :]:
            if following and not following.startswith(" "):
                break
            option_lines.append(following)
        options = "\n".join(option_lines)
        if "members:" not in options or "members: true" in options:
            return True
        if f"- {name}" in options:
            return True
    return False


def module_reference_findings() -> list[Finding]:
    """Ensure explicit public-module exports are routed to reference pages."""
    modules = {
        "datasets": PACKAGE_ROOT / "datasets" / "__init__.py",
        "diagnostics": PACKAGE_ROOT / "diagnostics" / "__init__.py",
        "evaluation": PACKAGE_ROOT / "evaluation.py",
        "metrics": PACKAGE_ROOT / "metrics.py",
        "plot": PACKAGE_ROOT / "plot.py",
    }
    reference = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(REFERENCE_ROOT.glob("*.md"))
    )
    findings = []
    for module_name, path in modules.items():
        for name, line in exported_names(path):
            direct_routes = (
                f"fpm_rs.{module_name}.{name}",
                f"fpm_rs.{name}",
            )
            if not any(
                route in reference for route in direct_routes
            ) and not module_directive_covers(reference, module_name, name):
                findings.append(
                    Finding(
                        path,
                        line,
                        f"fpm_rs.{module_name}.{name}",
                        "explicit module export is not routed to a Python reference page",
                    )
                )
    return findings


def main() -> int:
    """Run Python documentation coverage checks."""
    findings = [finding for path in API_SOURCES for finding in inspect_source(path)]
    findings.extend(root_reference_findings())
    findings.extend(module_reference_findings())
    if findings:
        print("Python API documentation check failed:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding.render()}", file=sys.stderr)
        return 1
    print(f"Python API documentation check passed ({len(API_SOURCES)} sources).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
