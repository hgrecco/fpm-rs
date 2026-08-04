"""Direct, domain-agnostic metric calculations.

All two-image functions use ``reference`` and ``estimate`` terminology.
Signed residuals are ``estimate - reference``.  These reporting metrics are
separate from reconstruction objectives in Rust's ``algorithms::objective``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import (
    amplitude_nrmse_py as _amplitude_nrmse,
    bias_py as _bias,
    compare_complex_fields_py as compare_complex_fields,
    compare_intensity_py as _compare_intensity,
    correlation_py as _correlation,
    fitted_gain_py as _fitted_gain,
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
    stats_py as _stats,
)

__all__ = [
    "amplitude_nrmse",
    "bias",
    "compare_complex_fields",
    "compare_intensity",
    "correlation",
    "fitted_gain",
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
    "stats",
]


def stats(image: Any, saturation_value: float | None = None) -> dict[str, float | int]:
    """Return summary statistics for one non-empty intensity image."""
    return _stats(_intensity_image(image, "image"), saturation_value)


def compare_intensity(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    saturation_value: float | None = None,
) -> dict[str, float | int | None]:
    """Return aggregate residual statistics for a reference/estimate pair."""
    reference, estimate, valid_mask = _comparison_inputs(
        reference, estimate, valid_mask
    )
    return _compare_intensity(
        reference,
        estimate,
        valid_mask=valid_mask,
        saturation_value=saturation_value,
    )


def bias(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean signed residual, ``estimate - reference``."""
    return _scalar_metric(_bias, reference, estimate, valid_mask)


def mae(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean absolute error over valid pixels."""
    return _scalar_metric(_mae, reference, estimate, valid_mask)


def mse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean squared error over valid pixels."""
    return _scalar_metric(_mse, reference, estimate, valid_mask)


def rmse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return root mean squared error over valid pixels."""
    return _scalar_metric(_rmse, reference, estimate, valid_mask)


def relative_l1(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float:
    """Return L1 residual divided by the reference L1 norm."""
    return _scalar_metric(_relative_l1, reference, estimate, valid_mask)


def nrmse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return residual L2 norm divided by the reference L2 norm."""
    return _scalar_metric(_nrmse, reference, estimate, valid_mask)


def amplitude_nrmse(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float:
    """Compare square-root intensities, normalized by reference amplitude energy."""
    return _scalar_metric(_amplitude_nrmse, reference, estimate, valid_mask)


def correlation(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float:
    """Return Pearson correlation over valid pixels."""
    return _scalar_metric(_correlation, reference, estimate, valid_mask)


def psnr(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    data_range: float,
) -> float:
    """Return PSNR in dB for an explicit finite, positive ``data_range``."""
    reference, estimate, valid_mask = _comparison_inputs(
        reference, estimate, valid_mask
    )
    return _psnr(reference, estimate, valid_mask=valid_mask, data_range=data_range)


def ssim(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    data_range: float,
) -> float:
    """Return canonical single-scale SSIM using an 11×11 Gaussian window (σ=1.5).

    References
    ----------
    [Wang, Bovik, Sheikh, and Simoncelli, *Image quality assessment: From error
    visibility to structural similarity*
    (2004)](https://doi.org/10.1109/TIP.2003.819861), IEEE Transactions on Image
    Processing 13(4), 600–612.
    """
    reference, estimate, valid_mask = _comparison_inputs(
        reference, estimate, valid_mask
    )
    return _ssim(reference, estimate, valid_mask=valid_mask, data_range=data_range)


def poisson_deviance(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    epsilon: float,
) -> float:
    """Return summed Poisson deviance; ``epsilon`` floors estimate intensity."""
    reference, estimate, valid_mask = _comparison_inputs(
        reference, estimate, valid_mask
    )
    return _poisson_deviance(
        reference, estimate, valid_mask=valid_mask, epsilon=epsilon
    )


def mean_poisson_deviance(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    epsilon: float,
) -> float:
    """Return Poisson deviance averaged over valid pixels."""
    reference, estimate, valid_mask = _comparison_inputs(
        reference, estimate, valid_mask
    )
    return _mean_poisson_deviance(
        reference, estimate, valid_mask=valid_mask, epsilon=epsilon
    )


def fitted_gain(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float:
    """Fit the least-squares gain in ``estimate ≈ gain × reference``."""
    return _scalar_metric(_fitted_gain, reference, estimate, valid_mask)


def _comparison_inputs(
    reference: Any, estimate: Any, valid_mask: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    return (
        _intensity_image(reference, "reference"),
        _intensity_image(estimate, "estimate"),
        _valid_mask(valid_mask),
    )


def _scalar_metric(
    function: Any, reference: Any, estimate: Any, valid_mask: Any | None
) -> float:
    reference, estimate, valid_mask = _comparison_inputs(
        reference, estimate, valid_mask
    )
    return function(reference, estimate, valid_mask=valid_mask)


def _intensity_image(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional real numeric array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must be a two-dimensional real numeric array")
    # Preserve arbitrary strides when the dtype already matches. Integer and
    # lower-precision inputs still require the documented dtype conversion.
    return np.asarray(array, dtype=np.float64)


def _valid_mask(values: Any | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise ValueError("valid_mask must be a two-dimensional boolean array")
    return array
