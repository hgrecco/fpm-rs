"""Direct, domain-agnostic image and complex-field metrics.

Two-image intensity functions accept matching nonempty two-dimensional real
numeric arrays and convert them to float64. ``valid_mask``, when supplied, is a
matching boolean array where ``True`` includes a pixel. Signed residuals are
``estimate - reference``. These reporting metrics are distinct from iterative
reconstruction objectives.
"""

from typing import Any, TypedDict

class IntensityStats(TypedDict):
    """Summary statistics for one intensity image."""

    mean: float
    """Arithmetic mean over all pixels."""
    std: float
    """Population standard deviation over all pixels."""
    min: float
    """Minimum intensity across all pixels."""
    max: float
    """Maximum intensity across all pixels."""
    sum: float
    """Sum of all pixel intensities."""
    saturated_pixels: int
    """Pixels at or above the requested saturation value, or zero if omitted."""
    zero_pixels: int
    """Pixels whose intensity is exactly zero."""

class IntensityComparison(TypedDict):
    """Aggregate residual statistics for matching intensity images."""

    reference_sum: float
    """Sum of valid reference intensities."""
    estimate_sum: float
    """Sum of valid estimate intensities."""
    residual_l1: float
    """Sum of absolute valid residuals."""
    residual_l2: float
    """Euclidean norm of valid residuals."""
    residual_mean: float
    """Mean signed valid residual."""
    residual_std: float
    """Population standard deviation of valid residuals."""
    residual_max_abs: float
    """Maximum absolute valid residual."""
    normalized_l2: float
    """Residual L2 norm divided by reference L2 norm."""
    saturated_pixels: int | None
    """Valid estimate pixels at saturation, or ``None`` when no threshold is given."""

def stats(image: Any, saturation_value: float | None = None) -> IntensityStats:
    """Summarize one nonnegative finite intensity image.

    ``saturation_value`` must be finite and nonnegative when supplied.
    """

def compare_intensity(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    saturation_value: float | None = None,
) -> IntensityComparison:
    """Return aggregate signed and absolute residual statistics.

    ``saturation_value`` counts valid estimate pixels at or above the supplied
    finite nonnegative threshold.
    """

def bias(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean signed residual, ``estimate - reference``."""

def mae(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean absolute error over valid pixels."""

def mse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return mean squared error over valid pixels."""

def rmse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return root mean squared error over valid pixels."""

def relative_l1(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float:
    """Return residual L1 norm divided by the reference L1 norm."""

def nrmse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float:
    """Return residual L2 norm divided by the reference L2 norm."""

def amplitude_nrmse(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float:
    """Compare square-root intensities, normalized by reference amplitude energy."""

def correlation(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float:
    """Return Pearson correlation over valid pixels."""

def psnr(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    data_range: float,
) -> float:
    """Return peak signal-to-noise ratio in dB for a positive ``data_range``."""

def ssim(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    data_range: float,
) -> float:
    """Return single-scale SSIM using an 11×11 Gaussian window with sigma 1.5.

    ``data_range`` is the positive intensity range. Masked pixels are excluded
    from the final average, but the local-window calculation is not renormalized
    around mask boundaries.

    References
    ----------
    [Wang, Bovik, Sheikh, and Simoncelli, *Image quality assessment: From error
    visibility to structural similarity*
    (2004)](https://doi.org/10.1109/TIP.2003.819861), IEEE Transactions on Image
    Processing 13(4), 600–612.
    """

def poisson_deviance(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    epsilon: float,
) -> float:
    """Return summed Poisson deviance with estimate intensity floored by ``epsilon``."""

def mean_poisson_deviance(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    epsilon: float,
) -> float:
    """Return Poisson deviance averaged over valid pixels."""

def fitted_gain(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float:
    """Fit the least-squares gain in ``estimate ≈ gain × reference``."""

def compare_complex_fields(
    reference: Any, candidate: Any, *, valid_mask: Any | None = None
) -> dict[str, float]:
    """Compare two complex128 fields after removing their global phase offset.

    Returns amplitude, complex-field, phase, and Fourier error metrics plus the
    fitted global phase in radians. Arrays and the optional boolean mask must
    have matching nonempty two-dimensional shapes.
    """

def radial_fourier_spectrum(field: Any) -> dict[str, list[float] | list[int]]:
    """Radially bin normalized centered-Fourier power for a complex 2D field.

    The returned mapping contains radial bin indices, normalized power, and
    the number of Fourier samples in each bin.
    """
