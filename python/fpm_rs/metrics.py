"""Direct, domain-agnostic metric calculations.

All two-image functions use ``reference`` and ``candidate`` terminology.
Signed residuals are ``candidate - reference``.  These reporting metrics are
separate from reconstruction objectives in Rust's ``algorithms::objective``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import (
    amplitude_nrmse_py as _amplitude_nrmse,
    bias_py as _bias,
    compare_complex_fields_py as compare_complex_fields,
    compare_intensity_py as compare_intensity,
    correlation_py as _correlation,
    fitted_gain_py as _fitted_gain,
    intensity_statistics_py as intensity_statistics,
    mae_py as _mae,
    mean_poisson_deviance_py as _mean_poisson_deviance,
    mse_py as _mse,
    nrmse_py as _nrmse,
    poisson_deviance_py as _poisson_deviance,
    psnr_py as _psnr,
    radial_fourier_spectrum_py as radial_fourier_spectrum,
    relative_l1_py as _relative_l1,
    rmse_py as _rmse,
    ssim_py as _ssim,
)

__all__ = [
    "amplitude_nrmse",
    "bias",
    "compare_complex_fields",
    "compare_intensity",
    "correlation",
    "fitted_gain",
    "intensity_statistics",
    "mae",
    "mean_poisson_deviance",
    "mse",
    "nrmse",
    "poisson_deviance",
    "psnr",
    "radial_fourier_spectrum",
    "relative_l1",
    "rmse",
    "ssim",
]


def bias(reference: Any, candidate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean signed residual, ``candidate - reference``."""
    return _scalar_metric(_bias, reference, candidate, valid_mask)


def mae(reference: Any, candidate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean absolute error over valid pixels."""
    return _scalar_metric(_mae, reference, candidate, valid_mask)


def mse(reference: Any, candidate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean squared error over valid pixels."""
    return _scalar_metric(_mse, reference, candidate, valid_mask)


def rmse(reference: Any, candidate: Any, *, valid_mask: Any | None = None) -> float:
    """Return root mean squared error over valid pixels."""
    return _scalar_metric(_rmse, reference, candidate, valid_mask)


def relative_l1(reference: Any, candidate: Any, *, valid_mask: Any | None = None) -> float:
    """Return L1 residual divided by the reference L1 norm."""
    return _scalar_metric(_relative_l1, reference, candidate, valid_mask)


def nrmse(reference: Any, candidate: Any, *, valid_mask: Any | None = None) -> float:
    """Return residual L2 norm divided by the reference L2 norm."""
    return _scalar_metric(_nrmse, reference, candidate, valid_mask)


def amplitude_nrmse(
    reference: Any, candidate: Any, *, valid_mask: Any | None = None
) -> float:
    """Compare square-root intensities, normalized by reference amplitude energy."""
    return _scalar_metric(_amplitude_nrmse, reference, candidate, valid_mask)


def correlation(reference: Any, candidate: Any, *, valid_mask: Any | None = None) -> float:
    """Return Pearson correlation over valid pixels."""
    return _scalar_metric(_correlation, reference, candidate, valid_mask)


def psnr(
    reference: Any,
    candidate: Any,
    *,
    valid_mask: Any | None = None,
    data_range: float,
) -> float:
    """Return PSNR in dB for an explicit finite, positive ``data_range``."""
    reference, candidate, valid_mask = _comparison_inputs(reference, candidate, valid_mask)
    return _psnr(reference, candidate, valid_mask=valid_mask, data_range=data_range)


def ssim(
    reference: Any,
    candidate: Any,
    *,
    valid_mask: Any | None = None,
    data_range: float,
) -> float:
    """Return canonical single-scale SSIM using an 11×11 Gaussian window (σ=1.5)."""
    reference, candidate, valid_mask = _comparison_inputs(reference, candidate, valid_mask)
    return _ssim(reference, candidate, valid_mask=valid_mask, data_range=data_range)


def poisson_deviance(
    reference: Any,
    candidate: Any,
    *,
    valid_mask: Any | None = None,
    epsilon: float,
) -> float:
    """Return summed Poisson deviance; ``epsilon`` floors candidate intensity."""
    reference, candidate, valid_mask = _comparison_inputs(reference, candidate, valid_mask)
    return _poisson_deviance(
        reference, candidate, valid_mask=valid_mask, epsilon=epsilon
    )


def mean_poisson_deviance(
    reference: Any,
    candidate: Any,
    *,
    valid_mask: Any | None = None,
    epsilon: float,
) -> float:
    """Return Poisson deviance averaged over valid pixels."""
    reference, candidate, valid_mask = _comparison_inputs(reference, candidate, valid_mask)
    return _mean_poisson_deviance(
        reference, candidate, valid_mask=valid_mask, epsilon=epsilon
    )


def fitted_gain(reference: Any, candidate: Any, *, valid_mask: Any | None = None) -> float:
    """Fit the least-squares gain in ``candidate ≈ gain × reference``."""
    return _scalar_metric(_fitted_gain, reference, candidate, valid_mask)


def _comparison_inputs(
    reference: Any, candidate: Any, valid_mask: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    return (
        _intensity_image(reference, "reference"),
        _intensity_image(candidate, "candidate"),
        _valid_mask(valid_mask),
    )


def _scalar_metric(function: Any, reference: Any, candidate: Any, valid_mask: Any | None) -> float:
    reference, candidate, valid_mask = _comparison_inputs(reference, candidate, valid_mask)
    return function(reference, candidate, valid_mask=valid_mask)


def _intensity_image(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional real numeric array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a two-dimensional real numeric array")
    return np.ascontiguousarray(array, dtype=np.float64)


def _valid_mask(values: Any | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise ValueError("valid_mask must be a two-dimensional boolean array")
    return np.ascontiguousarray(array)
