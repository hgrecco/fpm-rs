"""Direct, domain-agnostic metric calculations."""

from typing import Any, TypedDict

class IntensityStats(TypedDict):
    mean: float
    std: float
    min: float
    max: float
    sum: float
    saturated_pixels: int
    zero_pixels: int

class IntensityComparison(TypedDict):
    reference_sum: float
    estimate_sum: float
    residual_l1: float
    residual_l2: float
    residual_mean: float
    residual_std: float
    residual_max_abs: float
    normalized_l2: float
    saturated_pixels: int | None

def stats(image: Any, saturation_value: float | None = None) -> IntensityStats: ...
def compare_intensity(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    saturation_value: float | None = None,
) -> IntensityComparison: ...
def bias(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float: ...
def mae(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float: ...
def mse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float: ...
def rmse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float: ...
def relative_l1(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float: ...
def nrmse(reference: Any, estimate: Any, *, valid_mask: Any | None = None) -> float: ...
def amplitude_nrmse(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float: ...
def correlation(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float: ...
def psnr(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    data_range: float,
) -> float: ...
def ssim(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    data_range: float,
) -> float: ...
def poisson_deviance(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    epsilon: float,
) -> float: ...
def mean_poisson_deviance(
    reference: Any,
    estimate: Any,
    *,
    valid_mask: Any | None = None,
    epsilon: float,
) -> float: ...
def fitted_gain(
    reference: Any, estimate: Any, *, valid_mask: Any | None = None
) -> float: ...
def compare_complex_fields(
    reference: Any, candidate: Any, *, valid_mask: Any | None = None
) -> dict[str, float]: ...
def radial_fourier_spectrum(field: Any) -> dict[str, list[float] | list[int]]: ...
