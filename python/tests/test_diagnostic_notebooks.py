from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("IPython")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.mark.parametrize(
    ("notebook_name", "expected_plot_keys"),
    [
        (
            "diagnostics_quickstart.ipynb",
            {"convergence", "fourier_coverage"},
        ),
        (
            "diagnostics_debug.ipynb",
            {
                "frame_residuals",
                "raw_stack_stats",
                "crop_indices",
                "residuals_on_fourier_centers",
            },
        ),
        (
            "diagnostics_plots.ipynb",
            {
                "convergence",
                "frame_residuals",
                "raw_stack_stats",
                "fourier_coverage",
                "crop_indices",
                "residuals_on_fourier_centers",
            },
        ),
    ],
)
def test_diagnostic_notebook_executes(
    notebook_name: str,
    expected_plot_keys: set[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook_path = Path(__file__).parents[1] / "examples" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__diagnostic_notebook_test__"}
    monkeypatch.chdir(tmp_path)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        exec(compile(code, f"{notebook_path}#cell-{index}", "exec"), namespace)

    plot_results = namespace["plot_results"]
    assert expected_plot_keys <= set(plot_results)
    assert all(plot_results[name] is not None for name in expected_plot_keys)
    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("summary.txt"))


@pytest.mark.parametrize(
    ("notebook_name", "expected_variables"),
    [
        ("quickstart.ipynb", {"result", "figure", "axes"}),
        ("synthetic_objects_quickstart.ipynb", {"result", "figure", "axes"}),
    ],
)
def test_general_notebook_executes(
    notebook_name: str,
    expected_variables: set[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook_path = Path(__file__).parents[1] / "examples" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__general_notebook_test__"}
    monkeypatch.chdir(tmp_path)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        exec(compile(code, f"{notebook_path}#cell-{index}", "exec"), namespace)

    assert expected_variables <= set(namespace)
