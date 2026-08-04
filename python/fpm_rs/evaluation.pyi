"""Composed reconstruction evaluation against reference data."""

from typing import Any

from . import (
    ComplexArray,
    ImagePlaneModel,
    MaskArray,
    ReconstructionProblem,
    ReconstructionResult,
)

def evaluate_reconstruction(
    result: ReconstructionResult,
    truth: ComplexArray,
    *,
    problem: ReconstructionProblem | None = None,
    reference_model: ImagePlaneModel | None = None,
    valid_object_mask: MaskArray | None = None,
) -> dict[str, Any]:
    """Evaluate a reconstruction against a complex ground-truth field.

    Parameters
    ----------
    result
        Completed reconstruction to evaluate.
    truth
        Complex128 reference field shaped like ``result.object``.
    problem
        Optional reconstruction problem. When supplied, predicted intensities
        are compared with its measured frames.
    reference_model
        Optional model containing reference pupil and illumination calibration.
    valid_object_mask
        Optional uint8 object-space mask; zero excludes a pixel.

    Returns
    -------
    dict
        Nested ``object``, ``pupil``, ``illumination``, ``frame_gains``, and
        ``intensity`` sections. Optional sections are ``None`` when their
        required inputs or recovered quantities are unavailable.

    Notes
    -----
    Complex-field and phase metrics remove the best global phase offset before
    comparison. Object and pupil arrays use their respective grid shapes.
    """
