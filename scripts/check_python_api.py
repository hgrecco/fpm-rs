"""Check authoritative Python stubs against the installed runtime surface."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python" / "fpm_rs"


@dataclass(frozen=True)
class ParameterShape:
    """Parameter name, calling convention, and optionality."""

    name: str
    kind: str
    required: bool


def public(name: str) -> bool:
    """Return whether ``name`` is public for surface comparison."""
    return not name.startswith("_")


def annotation_name(annotation: ast.expr | None) -> str | None:
    """Return the final identifier in a simple annotation expression."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    return None


def parse_stub(path: Path) -> ast.Module:
    """Parse a checked-in stub file."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def root_stub_exports(module: ast.Module) -> set[str]:
    """Collect runtime-bearing root names declared by the package stub."""
    exports: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and public(node.name):
            exports.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if annotation_name(node.annotation) != "TypeAlias":
                exports.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fpm_rs."):
                    exports.add(alias.asname or alias.name.rsplit(".", 1)[-1])
    return exports


def module_stub_functions(module: ast.Module) -> set[str]:
    """Collect public functions defined in a module stub."""
    return {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and public(node.name)
    }


def format_set_difference(label: str, names: set[str]) -> str | None:
    """Format a deterministic set-difference diagnostic."""
    if not names:
        return None
    return f"{label}: {', '.join(sorted(names))}"


def stub_classes(module: ast.Module) -> dict[str, ast.ClassDef]:
    """Index every stub class, including private bases used for inheritance."""
    return {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}


def simple_base_names(node: ast.ClassDef) -> list[str]:
    """Return simple class base names used by local stub inheritance."""
    return [base.id for base in node.bases if isinstance(base, ast.Name)]


def declared_members(node: ast.ClassDef) -> set[str]:
    """Collect public methods, properties, and annotated attributes."""
    members: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if public(item.name):
                members.add(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if public(item.target.id):
                members.add(item.target.id)
    return members


def effective_stub_members(
    name: str,
    classes: dict[str, ast.ClassDef],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    """Collect members declared by a stub class and its local stub bases."""
    if name in seen:
        return set()
    node = classes[name]
    members = declared_members(node)
    for base in simple_base_names(node):
        if base in classes:
            members.update(effective_stub_members(base, classes, seen | {name}))
    return members


def runtime_members(value: type[Any]) -> set[str]:
    """Collect members implemented directly on a runtime extension class."""
    return {name for name in vars(value) if public(name)}


def ast_parameter_shapes(
    node: ast.FunctionDef, *, drop_self: bool
) -> list[ParameterShape]:
    """Convert an AST function signature to comparison records."""
    shapes: list[ParameterShape] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    posonly_count = len(node.args.posonlyargs)
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        if drop_self and index == 0 and argument.arg in {"self", "cls"}:
            continue
        kind = "positional-only" if index < posonly_count else "positional-or-keyword"
        shapes.append(ParameterShape(argument.arg, kind, default is None))
    if node.args.vararg is not None:
        shapes.append(ParameterShape(node.args.vararg.arg, "var-positional", False))
    for argument, default in zip(
        node.args.kwonlyargs, node.args.kw_defaults, strict=True
    ):
        shapes.append(ParameterShape(argument.arg, "keyword-only", default is None))
    if node.args.kwarg is not None:
        shapes.append(ParameterShape(node.args.kwarg.arg, "var-keyword", False))
    return shapes


def runtime_parameter_shapes(value: Any) -> list[ParameterShape]:
    """Convert an inspectable runtime signature to comparison records."""
    kind_names = {
        inspect.Parameter.POSITIONAL_ONLY: "positional-only",
        inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional-or-keyword",
        inspect.Parameter.VAR_POSITIONAL: "var-positional",
        inspect.Parameter.KEYWORD_ONLY: "keyword-only",
        inspect.Parameter.VAR_KEYWORD: "var-keyword",
    }
    return [
        ParameterShape(
            parameter.name,
            kind_names[parameter.kind],
            parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD},
        )
        for parameter in inspect.signature(value).parameters.values()
    ]


def render_parameters(parameters: list[ParameterShape]) -> str:
    """Render concise parameter-shape diagnostics."""
    return ", ".join(
        f"{item.name}:{item.kind}{'' if item.required else '=default'}"
        for item in parameters
    )


def signature_mismatch(
    label: str,
    stub_node: ast.FunctionDef,
    runtime_value: Any,
    *,
    drop_self: bool,
) -> str | None:
    """Return a signature mismatch, or ``None`` when parameter shapes agree."""
    expected = ast_parameter_shapes(stub_node, drop_self=drop_self)
    try:
        actual = runtime_parameter_shapes(runtime_value)
    except (TypeError, ValueError) as error:
        return f"{label}: runtime signature is unavailable: {error}"
    if (
        expected
        and actual
        and expected[0].name in {"self", "cls"}
        and actual[0].name == expected[0].name
    ):
        actual[0] = ParameterShape(actual[0].name, expected[0].kind, actual[0].required)
    if expected == actual:
        return None
    return (
        f"{label}: signature mismatch\n"
        f"    stub:    {render_parameters(expected)}\n"
        f"    runtime: {render_parameters(actual)}"
    )


def find_method(node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    """Find a named method declared directly in one stub class."""
    return next(
        (
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name == name
        ),
        None,
    )


def compare_root(package: ModuleType, module: ast.Module) -> list[str]:
    """Compare root exports, class members, constructors, and call signatures."""
    errors: list[str] = []
    expected_exports = root_stub_exports(module)
    actual_exports = set(package.__all__) | {"__version__"}
    for diagnostic in (
        format_set_difference(
            "runtime exports missing from stub", actual_exports - expected_exports
        ),
        format_set_difference(
            "stub exports missing at runtime", expected_exports - actual_exports
        ),
    ):
        if diagnostic:
            errors.append(diagnostic)

    classes = stub_classes(module)
    for name, node in classes.items():
        if not public(name) or not hasattr(package, name):
            continue
        runtime_class = getattr(package, name)
        if not isinstance(runtime_class, type):
            errors.append(f"fpm_rs.{name}: stub class is not a runtime class")
            continue
        expected_members = effective_stub_members(name, classes)
        actual_members = runtime_members(runtime_class)
        for diagnostic in (
            format_set_difference(
                f"fpm_rs.{name}: runtime members missing from stub",
                actual_members - expected_members,
            ),
            format_set_difference(
                f"fpm_rs.{name}: stub members missing at runtime",
                expected_members - actual_members,
            ),
        ):
            if diagnostic:
                errors.append(diagnostic)

        constructor = find_method(node, "__init__")
        if constructor is not None:
            if mismatch := signature_mismatch(
                f"fpm_rs.{name}", constructor, runtime_class, drop_self=True
            ):
                errors.append(mismatch)

        for member in sorted(expected_members & actual_members):
            stub_method = find_method(node, member)
            if stub_method is None:
                for base in simple_base_names(node):
                    if base in classes:
                        stub_method = find_method(classes[base], member)
                        if stub_method is not None:
                            break
            if stub_method is None:
                continue
            runtime_member = getattr(runtime_class, member)
            if not callable(runtime_member):
                continue
            if mismatch := signature_mismatch(
                f"fpm_rs.{name}.{member}",
                stub_method,
                runtime_member,
                drop_self=False,
            ):
                errors.append(mismatch)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and public(node.name):
            if not hasattr(package, node.name):
                continue
            if mismatch := signature_mismatch(
                f"fpm_rs.{node.name}",
                node,
                getattr(package, node.name),
                drop_self=False,
            ):
                errors.append(mismatch)
    return errors


def compare_stub_module(module_name: str, stub_name: str) -> list[str]:
    """Compare a public module's ``__all__`` functions with its stub."""
    runtime = importlib.import_module(module_name)
    stub = parse_stub(PACKAGE_ROOT / stub_name)
    expected = module_stub_functions(stub)
    actual = set(runtime.__all__)
    errors = []
    for diagnostic in (
        format_set_difference(
            f"{module_name}: runtime exports missing from stub", actual - expected
        ),
        format_set_difference(
            f"{module_name}: stub exports missing at runtime", expected - actual
        ),
    ):
        if diagnostic:
            errors.append(diagnostic)
    for node in stub.body:
        if isinstance(node, ast.FunctionDef) and node.name in actual:
            if mismatch := signature_mismatch(
                f"{module_name}.{node.name}",
                node,
                getattr(runtime, node.name),
                drop_self=False,
            ):
                errors.append(mismatch)
    return errors


def main() -> int:
    """Run runtime/stub synchronization checks."""
    try:
        package = importlib.import_module("fpm_rs")
    except ImportError as error:
        print(
            "Python API sync check requires the local extension; run `maturin develop --skip-install --locked` first.",
            file=sys.stderr,
        )
        print(f"Import failed: {error}", file=sys.stderr)
        return 2

    errors = compare_root(package, parse_stub(PACKAGE_ROOT / "__init__.pyi"))
    errors.extend(compare_stub_module("fpm_rs.metrics", "metrics.pyi"))
    errors.extend(compare_stub_module("fpm_rs.evaluation", "evaluation.pyi"))
    if errors:
        print("Python runtime/stub API sync check failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("Python runtime/stub API sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
