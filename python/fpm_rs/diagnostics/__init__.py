"""Diagnostics loading, reporting, and lightweight text summaries."""

from .io import coerce_diagnostics, ensure_output_dir, load_diagnostics, savefig
from .report import make_diagnostic_report, write_ground_truth_metrics, write_summary

__all__ = [
    "ensure_output_dir",
    "coerce_diagnostics",
    "load_diagnostics",
    "savefig",
    "make_diagnostic_report",
    "write_ground_truth_metrics",
    "write_summary",
]
