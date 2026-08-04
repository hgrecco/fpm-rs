"""Typed Python interface for image-plane Fourier ptychographic microscopy.

The package configures experiments, compiles forward models, simulates measured
intensities, reconstructs complex objects, and reads or writes result bundles.
Array shapes use ``(height, width)`` unless a frame axis is stated explicitly.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from os import PathLike
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
import fpm_rs.datasets as datasets
import fpm_rs.diagnostics as diagnostics
import fpm_rs.evaluation as evaluation
import fpm_rs.metrics as metrics
import fpm_rs.plot as plot

def radial_fourier_spectrum(field: ComplexArray) -> dict[str, list[float] | list[int]]:
    """Return radial bins, normalized Fourier power, and sample counts.

    This is the package-level alias of
    [``fpm_rs.metrics.radial_fourier_spectrum``][fpm_rs.metrics.radial_fourier_spectrum].
    ``field`` must be a nonempty two-dimensional complex array.
    """
    ...

Shape2D: TypeAlias = tuple[int, int]
"""Two-dimensional ``(height, width)`` array shape."""
ReconstructionShapeSpec: TypeAlias = (
    Shape2D | Literal["minimum", "smooth", "power_of_two"]
)
"""Explicit reconstruction shape or an automatic size-selection strategy."""
FloatArray: TypeAlias = NDArray[np.float64]
"""NumPy array whose elements are double-precision real values."""
ComplexArray: TypeAlias = NDArray[np.complex128]
"""NumPy array whose elements are double-precision complex values."""
MaskArray: TypeAlias = NDArray[np.uint8]
"""NumPy mask array with zero for excluded pixels and nonzero for valid pixels."""
Path: TypeAlias = str | PathLike[str]
"""Text or path-like filesystem path accepted by the public API."""

__version__: str
"""Installed package version."""

class FpmError(Exception):
    """Base class for errors raised by the Rust extension."""

class InvalidShapeError(FpmError):
    """An array or image shape is empty, inconsistent, or unsupported."""

class InvalidParameterError(FpmError):
    """A configuration parameter is non-finite or outside its valid range."""

class InvalidModelError(FpmError):
    """A compiled model is internally inconsistent or unsuitable for an operation."""

class InvalidMeasurementsError(FpmError):
    """A measurement stack contains invalid dimensions or intensity values."""

class LengthMismatchError(FpmError):
    """Related sequences or arrays do not have the required matching lengths."""

class FrameOutOfRangeError(FpmError):
    """A requested acquisition-frame index lies outside the measurement stack."""

class NumericalError(FpmError):
    """A numerical operation produced an invalid or non-finite result."""

class UnsupportedError(FpmError):
    """The requested feature or combination of options is not implemented."""

class DatasetError(FpmError):
    """Dataset discovery, validation, download, or loading failed."""

class FpmIoError(FpmError):
    """A filesystem operation failed."""

class SerializationError(FpmError):
    """Structured data could not be serialized or deserialized."""

class DatasetRegistryEntry:
    """Metadata and local-cache state for one registered dataset version."""

    @property
    def id(self) -> str:
        """Return the stable registry identifier."""
    @property
    def version(self) -> str:
        """Return the source-controlled dataset version."""
    @property
    def title(self) -> str:
        """Return the human-readable dataset title."""
    @property
    def description(self) -> str:
        """Return the registry description of the data and acquisition."""
    @property
    def format_version(self) -> int:
        """Return the measurement-manifest format version."""
    @property
    def archive_url(self) -> str:
        """Return the URL of the downloadable archive."""
    @property
    def archive_sha256(self) -> str:
        """Return the expected lowercase SHA-256 archive digest."""
    @property
    def archive_size_bytes(self) -> int:
        """Return the expected archive size in bytes."""
    @property
    def license_spdx(self) -> str:
        """Return the dataset license as an SPDX identifier."""
    @property
    def license_url(self) -> str:
        """Return the URL of the dataset license text."""
    @property
    def citation_doi(self) -> str:
        """Return the dataset citation DOI recorded by the registry."""
    @property
    def citation_text(self) -> str:
        """Return the preferred human-readable citation."""
    @property
    def source_url(self) -> str:
        """Return the authoritative source or landing-page URL."""
    @property
    def source_description(self) -> str:
        """Return provenance details supplied by the dataset publisher."""
    @property
    def tags(self) -> list[str]:
        """Return searchable registry tags."""
    @property
    def cached(self) -> bool:
        """Return whether a verified local copy is available."""
    @property
    def cache_path(self) -> Path | None:
        """Return the verified local dataset directory, if cached."""

class Dataset:
    """Loaded measurements, models, optional truth, and provenance.

    Dataset loading validates the manifest and array shapes before returning
    this object. Arrays are exposed using the same conventions as the core API.
    """

    @property
    def path(self) -> Path | None:
        """Return the local dataset directory, if it originated on disk."""
    @property
    def measurements(self) -> MeasurementStack:
        """Return the resident measured-intensity stack."""
    @property
    def true_model(self) -> ImagePlaneModel:
        """Return the acquisition model, including known simulated errors."""
    @property
    def reconstruction_model(self) -> ImagePlaneModel:
        """Return the model intended for reconstruction."""
    @property
    def ground_truth_object(self) -> ComplexArray | None:
        """Return the complex ground-truth field when supplied by the dataset."""
    @property
    def valid_object_mask(self) -> MaskArray | None:
        """Return the optional valid-pixel mask for object-space evaluation."""
    @property
    def provenance(self) -> Mapping[str, str]:
        """Return immutable acquisition and source provenance fields."""
    @property
    def measurement_units(self) -> str | None:
        """Return the declared units of measured intensities, if present."""
    def reconstruction_problem(self) -> ReconstructionProblem:
        """Build a validated problem from the dataset measurements and model."""

class DatasetRegistry:
    """Discover, verify, cache, and open datasets from a JSON registry.

    Parameters
    ----------
    registry_url
        Registry JSON URL. ``None`` selects the packaged default registry.
    cache_dir
        Managed cache directory. ``None`` selects the platform-specific default.

    Notes
    -----
    Listing may fetch registry metadata. Download and ``open`` may access the
    network; an already cached dataset is verified before use.
    """

    def __init__(
        self,
        *,
        registry_url: str | None = None,
        cache_dir: Path | None = None,
    ) -> None: ...
    @property
    def registry_url(self) -> str:
        """Return the configured registry URL."""
    @property
    def cache_dir(self) -> Path:
        """Return the managed local cache directory."""
    def list(self) -> list[DatasetRegistryEntry]:
        """Return registry entries annotated with their local cache state."""
    def download(self, id: str) -> Path:
        """Download, checksum-verify, extract, and return one dataset directory."""
    def download_all(self) -> list[Path]:
        """Download and verify every registry entry, returning local directories."""
    def open(self, id: str) -> Dataset:
        """Open a cached dataset, downloading it first when necessary."""
    def clean(self, id: str) -> bool:
        """Remove one cached dataset and return whether it existed."""
    def clean_all(self) -> int:
        """Remove all managed datasets and return the number removed."""

def open_dataset(
    id: str,
    *,
    registry_url: str | None = None,
    cache_dir: Path | None = None,
) -> Dataset:
    """Open a registered dataset using a temporary
    [``DatasetRegistry``][fpm_rs.DatasetRegistry].

    Parameters are equivalent to [``DatasetRegistry``][fpm_rs.DatasetRegistry];
    ``id`` is the exact
    registry identifier. The dataset is downloaded if no verified cached copy
    is available.
    """

class PupilAberration:
    """Direct radian weights for sampled pupil-polynomial terms.

    All four coefficients default to zero. The astigmatism, coma, and spherical
    values are direct phase weights in radians. ``edge_apodization`` controls
    radial amplitude decay and is not a normalized Zernike coefficient.
    """
    def __init__(
        self,
        *,
        astigmatism: float = 0.0,
        coma: float = 0.0,
        spherical: float = 0.0,
        edge_apodization: float = 0.0,
    ) -> None: ...
    @property
    def astigmatism(self) -> float:
        """Return the astigmatism phase weight in radians."""
    @property
    def coma(self) -> float:
        """Return the coma phase weight in radians."""
    @property
    def spherical(self) -> float:
        """Return the spherical-aberration phase weight in radians."""
    @property
    def edge_apodization(self) -> float:
        """Return the nonnegative radial pupil-amplitude decay coefficient."""

class Optics:
    """Physical microscope parameters used to compile an image-plane model.

    Parameters
    ----------
    wavelength
        Positive vacuum illumination wavelength in metres.
    objective_na
        Positive, dimensionless objective numerical aperture, no greater than
        ``medium_index``.
    magnification
        Positive, dimensionless microscope magnification.
    camera_pixel_size
        Positive detector-plane pixel pitch in metres.
    medium_index
        Positive refractive index of the immersion medium.
    defocus_distance
        Optional signed propagation distance in metres.
    pupil_aberration
        Optional sampled pupil-aberration coefficients.
    """
    def __init__(
        self,
        wavelength: float,
        objective_na: float,
        magnification: float,
        camera_pixel_size: float,
        *,
        medium_index: float = 1.0,
        defocus_distance: float | None = None,
        pupil_aberration: PupilAberration | None = None,
    ) -> None: ...
    @property
    def wavelength(self) -> float:
        """Return the vacuum illumination wavelength in metres."""
    @property
    def objective_na(self) -> float:
        """Return the objective numerical aperture."""
    @property
    def magnification(self) -> float:
        """Return the microscope magnification."""
    @property
    def camera_pixel_size(self) -> float:
        """Return the detector-plane pixel pitch in metres."""
    @property
    def medium_index(self) -> float:
        """Return the immersion-medium refractive index."""
    @property
    def defocus_distance(self) -> float | None:
        """Return the optional signed propagation distance in metres."""
    @property
    def object_pixel_size(self) -> float:
        """Return ``camera_pixel_size / magnification`` in metres."""

class LEDArray:
    """Planar LED grid geometry.

    Parameters
    ----------
    grid_shape
        Number of LEDs as ``(rows, columns)``.
    pitch
        Positive centre-to-centre LED spacing in metres.
    distance
        Positive perpendicular LED-plane-to-sample distance in metres.
    center
        Optical-axis grid coordinate as ``(column, row)``; fractional indices
        are allowed.
    wavelength_override
        Optional positive vacuum wavelength in metres for this source.
    illumination_order
        Optional acquisition-to-natural row-major LED permutation.
    intensity_weights
        Optional positive gain per natural-order LED.
    rotation_degrees
        Counter-clockwise in-plane rotation about ``center`` in degrees.
    """
    def __init__(
        self,
        grid_shape: Shape2D,
        pitch: float,
        distance: float,
        center: tuple[float, float],
        *,
        wavelength_override: float | None = None,
        illumination_order: Sequence[int] | None = None,
        intensity_weights: Sequence[float] | None = None,
        rotation_degrees: float = 0.0,
    ) -> None: ...
    @property
    def grid_shape(self) -> Shape2D:
        """Return the LED-grid shape as ``(rows, columns)``."""
    @property
    def source_count(self) -> int:
        """Return the number of LEDs in the grid."""

class LEDSphere:
    """Fixed LEDs at arbitrary polar and azimuthal positions on a sphere.

    Parameters
    ----------
    angles
        Float64 array shaped ``(sources, 2)`` containing natural-order
        ``(theta, phi)`` radians. ``theta`` is polar angle from positive ``z``;
        ``phi`` is azimuth from positive ``x`` toward positive ``y``.
    radius
        Positive nominal sphere radius in metres.
    center_offset
        Sphere-centre displacement ``(x, y, z)`` from the sample, in metres.
    orientation_degrees
        Extrinsic mount rotations ``(rx, ry, rz)`` in degrees, applied as
        ``Rz * Ry * Rx``.
    angular_corrections
        Optional float64 ``(sources, 2)`` array of ``(delta_theta, delta_phi)``
        placement corrections in radians.
    wavelength_override
        Optional positive vacuum wavelength in metres for this source.
    illumination_order
        Optional acquisition-to-natural LED permutation.
    intensity_weights
        Optional positive gain per natural-order LED.
    """

    def __init__(
        self,
        angles: FloatArray,
        radius: float,
        *,
        center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_corrections: FloatArray | None = None,
        wavelength_override: float | None = None,
        illumination_order: Sequence[int] | None = None,
        intensity_weights: Sequence[float] | None = None,
    ) -> None: ...
    @property
    def source_count(self) -> int:
        """Return the number of fixed LEDs."""

class SphericalLEDArm:
    """A single LED moved along a calibrated spherical-arm trajectory.

    Parameters
    ----------
    commanded_angles
        Float64 array shaped ``(sources, 2)`` of movement-order ``(theta, phi)``
        encoder commands in radians. Unwrap azimuth across branch cuts when
        backlash direction must remain unambiguous.
    arm_length
        Positive pivot-to-LED distance in metres.
    pivot_offset
        Pivot displacement ``(x, y, z)`` from the sample, in metres.
    orientation_degrees
        Extrinsic mount rotations ``(rx, ry, rz)`` in degrees.
    theta_zero_degrees, phi_zero_degrees
        Additive encoder-zero offsets in degrees.
    theta_scale, phi_scale
        Positive dimensionless encoder scales.
    elevation_axis_tilt_degrees
        Elevation-axis non-orthogonality in degrees.
    theta_backlash_degrees, phi_backlash_degrees
        Total separation of increasing and decreasing branches in degrees.
    wavelength_override
        Optional positive vacuum wavelength in metres.
    intensity_weights
        Optional positive gain per movement-order source position.
    """

    def __init__(
        self,
        commanded_angles: FloatArray,
        arm_length: float,
        *,
        pivot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
        theta_zero_degrees: float = 0.0,
        phi_zero_degrees: float = 0.0,
        theta_scale: float = 1.0,
        phi_scale: float = 1.0,
        elevation_axis_tilt_degrees: float = 0.0,
        theta_backlash_degrees: float = 0.0,
        phi_backlash_degrees: float = 0.0,
        wavelength_override: float | None = None,
        intensity_weights: Sequence[float] | None = None,
    ) -> None: ...
    @property
    def source_count(self) -> int:
        """Return the number of commanded source positions."""

class RotatingLEDArc:
    """A quarter-circle LED arc sampled at commanded axial rotations.

    Every LED is compiled at every rotation. Sources are ordered rotation-major,
    then in ``led_thetas`` order.

    Parameters
    ----------
    led_thetas
        Natural-order LED polar angles from positive ``z``, in radians.
    rotation_angles
        Movement-order arm azimuth commands in radians.
    radius
        Positive nominal arc radius in metres.
    axis_origin_offset
        Point on the rotation axis relative to the sample, in metres.
    axis_tilt_degrees
        Extrinsic ``(x, y)`` rotation-axis tilts in degrees.
    led_angular_corrections
        Optional ``(leds, 2)`` array of ``(delta_theta, delta_phi)`` radians.
    led_radial_offsets
        Optional per-LED radial corrections in metres.
    rotation_zero_degrees
        Additive rotation-encoder zero offset in degrees.
    rotation_scale
        Positive dimensionless rotation-encoder scale.
    rotation_backlash_degrees
        Total separation of increasing and decreasing branches in degrees.
    wavelength_override
        Optional positive vacuum wavelength in metres.
    led_intensity_weights
        Optional positive intrinsic gain per LED, repeated at every rotation.
    """

    def __init__(
        self,
        led_thetas: Sequence[float],
        rotation_angles: Sequence[float],
        radius: float,
        *,
        axis_origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_tilt_degrees: tuple[float, float] = (0.0, 0.0),
        led_angular_corrections: FloatArray | None = None,
        led_radial_offsets: Sequence[float] | None = None,
        rotation_zero_degrees: float = 0.0,
        rotation_scale: float = 1.0,
        rotation_backlash_degrees: float = 0.0,
        wavelength_override: float | None = None,
        led_intensity_weights: Sequence[float] | None = None,
    ) -> None: ...
    @property
    def led_count(self) -> int:
        """Return the number of LEDs along the arc."""
    @property
    def rotation_count(self) -> int:
        """Return the number of commanded arc rotations."""
    @property
    def source_count(self) -> int:
        """Return ``led_count * rotation_count`` compiled source positions."""

class AngleList:
    """Explicit independent illumination tilts in acquisition order.

    ``angles`` is a nonempty float64 array shaped ``(sources, 2)`` containing
    ``(theta_x, theta_y)`` radians. Each component is converted independently
    to transverse wave-vector coordinates during model compilation.
    """

    def __init__(self, angles: FloatArray) -> None: ...
    @property
    def source_count(self) -> int:
        """Return the number of angle pairs."""

class KVectorList:
    """Calibrated transverse illumination vectors in acquisition order.

    ``k_vectors`` must have float64 shape ``(sources, 2)`` and units of radians
    per metre. Compilation validates that each vector represents a propagating
    wave for the supplied optics.
    """

    def __init__(self, k_vectors: FloatArray) -> None: ...
    @property
    def source_count(self) -> int:
        """Return the number of calibrated source vectors."""

class CodedIllumination:
    """Known incoherent combinations of calibrated source vectors.

    Parameters
    ----------
    k_vectors
        Float64 array shaped ``(sources, 2)`` in radians per metre.
    frame_weights
        Float64 array shaped ``(frames, sources)``. Nonzero entries select the
        positive incoherent source contributions to each measured frame.
    """

    def __init__(self, k_vectors: FloatArray, frame_weights: FloatArray) -> None: ...
    @property
    def source_count(self) -> int:
        """Return the number of individual calibrated sources."""
    @property
    def frame_count(self) -> int:
        """Return the number of coded acquisition frames."""

class CameraModel:
    """Detector response and acquisition-noise model.

    Parameters
    ----------
    photons_per_pixel
        Positive expected photoelectrons at unit optical intensity.
    gain_counts_per_electron
        Positive linear conversion gain in digital counts per electron.
    offset_counts
        Finite additive electronic bias in digital counts.
    read_noise_electrons
        Nonnegative Gaussian read-noise standard deviation per pixel.
    dark_current_electrons
        Nonnegative expected dark-current electrons per pixel and exposure.
    shot_noise
        Whether to Poisson-sample photoelectrons and dark current.
    pixel_sensitivity
        Optional nonnegative float64 sensitivity map shaped like one detector
        frame, ``(height, width)``.
    bit_depth
        Optional digitizer bit depth in ``[1, 53]``; ``None`` disables the
        corresponding ``2**bits - 1`` clipping limit.
    saturation_counts
        Optional nonnegative clipping threshold in digital counts.
    quantize
        Whether to round final counts to integer-valued float64 values.
    bad_pixels
        Unique row-major detector indices replaced after other effects.
    bad_pixel_value_counts
        Replacement value for ``bad_pixels`` in digital counts.
    """
    def __init__(
        self,
        *,
        photons_per_pixel: float = 1000.0,
        gain_counts_per_electron: float = 1.0,
        offset_counts: float = 0.0,
        read_noise_electrons: float = 0.0,
        dark_current_electrons: float = 0.0,
        shot_noise: bool = False,
        pixel_sensitivity: FloatArray | None = None,
        bit_depth: int | None = 16,
        saturation_counts: float | None = None,
        quantize: bool = True,
        bad_pixels: Sequence[int] = (),
        bad_pixel_value_counts: float | None = None,
    ) -> None: ...
    @staticmethod
    def ideal() -> CameraModel:
        """Return a unit-response detector without noise, clipping, or quantization."""

class IlluminationAcquisitionErrors:
    """Non-geometric illumination errors injected during simulation.

    Parameters
    ----------
    frame_gain_relative_std
        Nonnegative one-sigma Gaussian frame-gain variation relative to each
        frame's compiled gain.
    missing_frames
        Unique zero-based acquisition frames whose illumination is forced to zero.
    source_permutation
        Optional complete permutation assigning a true source to each compiled
        source slot.
    """

    def __init__(
        self,
        *,
        frame_gain_relative_std: float = 0.0,
        missing_frames: Sequence[int] = (),
        source_permutation: Sequence[int] | None = None,
    ) -> None: ...

class SyntheticObject:
    """Rust-owned high-resolution complex sample field.

    The complex128 field is shaped ``(height, width)`` and indexed
    ``(row, column)``. Magnitude represents amplitude transmission and argument
    represents phase delay in radians. Construction copies Python-owned arrays.
    """
    def __init__(self, field: ComplexArray) -> None: ...
    @staticmethod
    def constant(
        shape: Shape2D, amplitude: float = 1.0, phase: float = 0.0
    ) -> SyntheticObject:
        """Create a constant field with nonnegative amplitude and radian phase."""
    @staticmethod
    def amplitude_only(amplitude: FloatArray) -> SyntheticObject:
        """Create a zero-phase object from a finite nonnegative 2D amplitude array."""
    @staticmethod
    def phase_only(phase: FloatArray) -> SyntheticObject:
        """Create a unit-amplitude object from a finite 2D radian phase array."""
    @staticmethod
    def from_amplitude_phase(
        amplitude: FloatArray, phase: FloatArray
    ) -> SyntheticObject:
        """Combine matching 2D amplitude and radian-phase arrays into a field."""
    @staticmethod
    def from_amplitude_image(path: Path) -> SyntheticObject:
        """Load a grayscale image normalized to ``[0, 1]`` as amplitude."""
    @staticmethod
    def from_amplitude_phase_images(
        amplitude_path: Path, phase_path: Path, phase_extent: float
    ) -> SyntheticObject:
        """Load matching grayscale amplitude and phase images.

        Black and white phase pixels map linearly to ``-phase_extent`` and
        ``+phase_extent`` radians. Image decoding runs without the Python GIL.
        """
    @staticmethod
    def phase_disk(
        shape: Shape2D, radius_pixels: float, phase_shift: float
    ) -> SyntheticObject:
        """Create a centered unit-amplitude disk with a radian phase shift."""
    @staticmethod
    def siemens_star(shape: Shape2D, spokes: int) -> SyntheticObject:
        """Create a centered binary-amplitude Siemens star with at least two spokes."""
    @staticmethod
    def resolution_target(shape: Shape2D) -> SyntheticObject:
        """Create deterministic horizontal and vertical binary bar groups."""
    @staticmethod
    def random_phase(
        shape: Shape2D, standard_deviation: float, seed: int
    ) -> SyntheticObject:
        """Create unit amplitude with seeded zero-mean Gaussian phase in radians."""
    @staticmethod
    def particle_field(shape: Shape2D, particles: int, seed: int) -> SyntheticObject:
        """Create a seeded unit-amplitude field with dark single-pixel particles."""
    @staticmethod
    def mixed_test_pattern(shape: Shape2D) -> SyntheticObject:
        """Create a deterministic mixed amplitude-and-phase smoke-test target."""
    @staticmethod
    def biological_like(shape: Shape2D, features: int, seed: int) -> SyntheticObject:
        """Create seeded smooth absorption and phase blobs for testing.

        This convenient target is not a tissue-specific physical model.
        """
    @property
    def field(self) -> ComplexArray:
        """Return a copy of the complex128 field shaped ``(height, width)``."""
    @property
    def shape(self) -> Shape2D:
        """Return the object shape as ``(height, width)``."""
    @property
    def label(self) -> str | None:
        """Return the optional generator label."""

class ImagePlaneModel:
    """Compiled pupil, Fourier crops, sampling, and frame/source weights.

    Models are constructed by [``compile_model``][fpm_rs.compile_model], not
    directly. Returned arrays copy immutable compiled data. Shape tuples use
    ``(height, width)``; wave vectors use radians per metre.
    """
    @property
    def image_shape(self) -> Shape2D:
        """Return the low-resolution detector-frame shape."""
    @property
    def reconstruction_shape(self) -> Shape2D:
        """Return the high-resolution complex-object shape."""
    @property
    def source_count(self) -> int:
        """Return the number of individual illumination sources."""
    @property
    def frame_count(self) -> int:
        """Return the number of acquisition frames."""
    @property
    def is_multiplexed(self) -> bool:
        """Return whether any frame incoherently combines multiple sources."""
    @property
    def k_vectors(self) -> FloatArray:
        """Return a float64 ``(sources, 2)`` array of ``(kx, ky)`` rad/m."""
    @property
    def pupil(self) -> ComplexArray:
        """Return the sampled complex pupil on the low-resolution Fourier grid."""
    @property
    def pupil_support(self) -> MaskArray:
        """Return the uint8 aperture-support mask shaped ``image_shape``."""
    @property
    def frame_gains(self) -> FloatArray | None:
        """Return optional positive acquisition-frame intensity multipliers."""

Illumination: TypeAlias = (
    LEDArray
    | LEDSphere
    | SphericalLEDArm
    | RotatingLEDArc
    | AngleList
    | KVectorList
    | CodedIllumination
)
"""Physical, angular, calibrated, or coded illumination accepted for compilation."""

def compile_model(
    optics: Optics,
    illumination: Illumination,
    image_shape: Shape2D,
    reconstruction_shape: ReconstructionShapeSpec = "smooth",
) -> ImagePlaneModel:
    """Compile experiment geometry into an immutable image-plane model.

    Parameters
    ----------
    optics
        Valid physical microscope parameters.
    illumination
        Source geometry or calibrated transverse wave vectors.
    image_shape
        Low-resolution frame shape as ``(height, width)``.
    reconstruction_shape
        Exact high-resolution shape, or ``"minimum"``, ``"smooth"``, or
        ``"power_of_two"`` automatic sizing. The default chooses efficient
        small-prime FFT dimensions no smaller than the required Fourier extent.

    Returns
    -------
    ImagePlaneModel
        Pupil, Fourier crops, sampling, source weights, and frame gains used by
        simulation and reconstruction.
    """

def suggest_reconstruction_shape(
    optics: Optics,
    illumination: Illumination,
    image_shape: Shape2D,
    reconstruction_shape: ReconstructionShapeSpec = "smooth",
) -> Shape2D:
    """Resolve a reconstruction-size choice without building pupil or crop arrays."""

def compile_camera_model(
    model: ImagePlaneModel, camera: CameraModel
) -> ImagePlaneModel:
    """Compile the known uniform linear detector response into a model.

    Pixel sensitivity, stochastic noise, clipping, quantization, and bad pixels
    remain detector effects and are not represented in the returned model.
    """

class MeasurementStack:
    """Resident float64 intensity stack shaped ``(frames, height, width)``.

    Parameters
    ----------
    measurements
        Nonnegative finite float64 array with shape ``(frames, height, width)``.
        Python-owned data is copied during construction.
    frame_weights
        Optional finite nonnegative weight per frame; omitted weights are one.
    masks
        Optional uint8 array matching ``measurements``. Zero excludes a pixel
        and nonzero includes it in reconstruction objectives.
    """
    def __init__(
        self,
        measurements: FloatArray,
        *,
        frame_weights: Sequence[float] | None = None,
        masks: MaskArray | None = None,
    ) -> None: ...
    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(frames, height, width)``."""
    @property
    def frame_count(self) -> int:
        """Return the number of measured acquisition frames."""
    @property
    def image_shape(self) -> Shape2D:
        """Return the detector-frame shape as ``(height, width)``."""
    @property
    def array(self) -> FloatArray:
        """Return a float64 copy shaped ``(frames, height, width)``."""
    @property
    def frame_weights(self) -> FloatArray:
        """Return a float64 copy containing one objective weight per frame."""

class SimulationResult:
    """Measurements and retained ground truth from one simulated acquisition."""

    @property
    def measurements(self) -> MeasurementStack:
        """Return simulated detector intensities or camera counts in frame order."""
    @property
    def ground_truth_object(self) -> ComplexArray:
        """Return the true high-resolution complex field."""
    @property
    def true_model(self) -> ImagePlaneModel:
        """Return the optical model used to generate measurements."""
    @property
    def reconstruction_model(self) -> ImagePlaneModel:
        """Return the assumed model intended for reconstruction."""
    @property
    def ideal(self) -> bool:
        """Return whether camera and acquisition-error effects were absent."""
    @property
    def missing_frames(self) -> list[int]:
        """Return zero-based frames whose simulated illumination was forced to zero."""
    @property
    def random_seed(self) -> int:
        """Return the deterministic seed used for pseudorandom effects."""

def simulate(
    true_model: ImagePlaneModel,
    object: ComplexArray | SyntheticObject,
    *,
    reconstruction_model: ImagePlaneModel | None = None,
    camera: CameraModel | None = None,
    illumination_errors: IlluminationAcquisitionErrors | None = None,
    seed: int = 0,
) -> SimulationResult:
    """Simulate an image-plane FPM acquisition without holding the Python GIL.

    Parameters
    ----------
    true_model
        Compiled physical model used by the forward simulation.
    object
        Complex128 field or validated
        [``SyntheticObject``][fpm_rs.SyntheticObject] matching the model's
        reconstruction shape.
    reconstruction_model
        Optional separately compiled assumed model. Omission reuses
        ``true_model`` before known linear camera response is applied.
    camera
        Optional detector pipeline producing digital counts.
    illumination_errors
        Optional gain variation, missing frames, or source permutation.
    seed
        Deterministic unsigned random seed.

    Returns
    -------
    SimulationResult
        Measurements, true and assumed models, object truth, and realized
        acquisition metadata.
    """

class ReconstructionProblem:
    """Validated pairing of measurements and a compiled image-plane model.

    Parameters
    ----------
    measurements
        Float64 ``(frames, height, width)`` array or an existing
        [``MeasurementStack``][fpm_rs.MeasurementStack].
    model
        Compiled model with matching frame count and low-resolution shape.
    frame_weights, masks
        Optional metadata used only when ``measurements`` is a NumPy array.
    name
        Optional human-readable problem identifier propagated to metadata.
    """
    def __init__(
        self,
        measurements: FloatArray | MeasurementStack,
        model: ImagePlaneModel,
        *,
        frame_weights: Sequence[float] | None = None,
        masks: MaskArray | None = None,
        name: str | None = None,
    ) -> None: ...
    @property
    def name(self) -> str | None:
        """Return the optional problem identifier."""
    @property
    def frame_count(self) -> int:
        """Return the validated acquisition-frame count."""
    @property
    def image_shape(self) -> Shape2D:
        """Return the low-resolution measurement shape."""
    @property
    def reconstruction_shape(self) -> Shape2D:
        """Return the high-resolution object shape."""

class ReconstructionCheckpoint:
    """Serializable algorithm state for resuming a compatible run."""

    @staticmethod
    def load(path: Path) -> ReconstructionCheckpoint:
        """Read and validate a checkpoint file without holding the Python GIL."""
    def save(self, path: Path) -> None:
        """Atomically serialize this checkpoint without holding the Python GIL."""
    @property
    def format_version(self) -> int:
        """Return the checkpoint format version."""
    @property
    def completed_iterations(self) -> int:
        """Return the number of complete iterations represented by the state."""

class RuntimeInfo:
    """Execution summary attached to a reconstruction result."""

    elapsed_seconds: float
    """Wall-clock run duration in seconds."""
    completed_iterations: int
    """Number of complete iterations executed, including resumed progress."""
    stopped_early: bool
    """Whether a callback requested termination before the configured limit."""
    algorithm: str
    """Stable algorithm name recorded by the runner."""

class ReconstructionResult:
    """Owned reconstruction products, histories, diagnostics, and metadata.

    Object-space arrays use ``reconstruction_shape``. Pupil arrays use the
    model's low-resolution ``image_shape``. Calibration arrays are present only
    when the selected algorithm recovered those quantities.
    """

    object: ComplexArray
    """Reconstructed complex128 object field."""
    amplitude: FloatArray
    """Magnitude of the reconstructed complex object field."""
    phase: FloatArray
    """Wrapped argument of ``object`` in radians."""
    object_spectrum: ComplexArray
    """Centered complex Fourier spectrum corresponding to ``object``."""
    recovered_pupil: ComplexArray
    """Final sampled complex pupil on the low-resolution Fourier grid."""
    pupil_support: MaskArray
    """Uint8 aperture-support mask for ``recovered_pupil``."""
    calibrated_illumination: FloatArray | None
    """Optional ``(sources, 2)`` recovered Fourier-grid offsets in pixels."""
    recovered_frame_gains: FloatArray | None
    """Optional recovered positive multiplicative gain per frame."""
    recovered_background: FloatArray | None
    """Optional recovered nonnegative uniform background per frame."""
    trace: list[tuple[int, float, float]]
    """``(iteration, objective, elapsed_seconds)`` records."""
    algorithm_metrics: list[tuple[int, str, str, float]]
    """``(iteration, namespace, metric, value)`` algorithm-specific records."""
    scalar_diagnostics: Mapping[str, float]
    """Named final scalar diagnostics."""
    runtime: RuntimeInfo
    """Execution duration, progress, stopping state, and algorithm name."""
    metadata: Mapping[str, str]
    """Stable string metadata carried by the result."""
    final_objective: float | None
    """Objective from the last trace record, or ``None`` for an empty trace."""
    def write_bundle(
        self,
        path: Path,
        *,
        run_id: str | None = None,
        label: str | None = None,
        include_previews: bool = True,
    ) -> ResultBundle:
        """Write a self-describing Parquet/NPY result bundle and reopen it.

        ``path`` is the destination directory. ``run_id`` defaults to a generated
        identifier, ``label`` is optional display metadata, and disabling
        ``include_previews`` omits derived PNG images. Existing nonempty targets
        are rejected.
        """

class BundleArtifact:
    """Manifest metadata for one file in a result or benchmark bundle."""

    path: Path
    """Absolute path to the artifact in the opened bundle."""
    media_type: str
    """Declared MIME media type."""
    byte_size: int
    """Expected file size in bytes."""
    sha256: str
    """Expected lowercase SHA-256 digest."""
    role: str
    """Stable semantic role recorded in the manifest."""

class BundleArray:
    """Lazily loaded NPY array artifact.

    Accessing [``value``][fpm_rs.BundleArray.value] validates and caches a
    read-only NumPy array.
    """

    path: Path
    """Absolute path to the NPY file."""
    media_type: str
    """Declared MIME media type."""
    byte_size: int
    """Expected file size in bytes."""
    sha256: str
    """Expected lowercase SHA-256 digest."""
    role: str
    """Stable semantic array role."""
    value: ComplexArray | FloatArray | MaskArray
    """Read-only array, loaded and cached on first access."""

class BundleTables:
    """Parquet table artifacts present in a result bundle."""

    summary: BundleArtifact
    """One-row run summary table."""
    history: BundleArtifact
    """Per-iteration objective and timing history."""
    algorithm_metrics: BundleArtifact | None
    """Optional algorithm-specific metric records."""
    iteration_diagnostics: BundleArtifact | None
    """Optional per-iteration diagnostic records."""
    frame_diagnostics: BundleArtifact | None
    """Optional per-frame residual summaries."""
    raw_frame_statistics: BundleArtifact | None
    """Optional raw measurement-frame statistics."""
    frame_evaluation: BundleArtifact | None
    """Optional per-frame ground-truth evaluation records."""
    illumination_calibration: BundleArtifact | None
    """Optional recovered illumination-calibration table."""
    frame_calibration: BundleArtifact | None
    """Optional recovered frame-gain and background table."""
    scalar_diagnostics: BundleArtifact | None
    """Optional named scalar-diagnostic table."""
    metadata: BundleArtifact | None
    """Optional key-value result metadata table."""

class BundleArrays:
    """Lazy numerical-array artifacts in a result bundle."""

    object: BundleArray
    """Reconstructed complex object field."""
    object_spectrum: BundleArray
    """Centered complex object spectrum."""
    pupil: BundleArray
    """Final recovered complex pupil array."""
    pupil_support: BundleArray
    """Uint8 pupil-support mask."""
    illumination_calibration: BundleArray | None
    """Optional recovered source-offset array."""
    frame_gains: BundleArray | None
    """Optional recovered frame-gain array."""
    background: BundleArray | None
    """Optional recovered frame-background array."""

class BundlePreviews:
    """Optional PNG preview artifacts derived during bundle export."""

    object_amplitude: BundleArtifact | None
    """Object-amplitude preview, if requested."""
    object_phase: BundleArtifact | None
    """Object-phase preview, if requested."""
    pupil_amplitude: BundleArtifact | None
    """Pupil-amplitude preview, if requested."""
    pupil_phase: BundleArtifact | None
    """Pupil-phase preview, if requested."""
    fourier_coverage: BundleArtifact | None
    """Fourier-coverage preview when coverage diagnostics are available."""

class BundleVerificationResult:
    """Summary returned after all manifest artifacts pass verification."""

    artifact_count: int
    """Number of files whose size and digest were verified."""
    total_bytes: int
    """Total verified payload size in bytes."""

class ResultBundle:
    """Opened result bundle with eager metadata and lazy arrays.

    [``read_bundle``][fpm_rs.read_bundle] validates the manifest and file
    layout. Individual file sizes and hashes are checked when artifacts are
    loaded, or all at once by [``verify``][fpm_rs.ResultBundle.verify].
    """

    path: Path
    """Absolute path to the opened bundle directory."""
    manifest_path: Path
    """Absolute JSON manifest path."""
    run_id: str
    """Stable run identifier from the manifest."""
    label: str | None
    """Optional human-readable bundle label."""
    tables: BundleTables
    """Parquet table artifact descriptors in this bundle."""
    arrays: BundleArrays
    """Lazy NPY artifact descriptors."""
    previews: BundlePreviews
    """Optional image-preview descriptors."""
    result: ReconstructionResult
    """Lazily reconstructed result backed by cached read-only arrays."""
    diagnostics: dict[str, Any] | None
    """Structured reconstruction diagnostics, if stored."""
    evaluation: dict[str, Any] | None
    """Structured ground-truth evaluation, if stored."""
    def verify(self) -> BundleVerificationResult:
        """Verify sizes and SHA-256 digests for every manifest artifact."""
    def clear_cache(self) -> None:
        """Release cached arrays and reconstructed result values."""

def read_bundle(path: Path) -> ResultBundle:
    """Validate a result-bundle manifest and open its artifacts lazily."""

class BenchmarkBundleTables:
    """Parquet artifact descriptors in a benchmark bundle."""

    runs: BundleArtifact
    """One summary row per benchmark run."""
    frames: BundleArtifact
    """Per-frame benchmark records."""
    artifacts: BundleArtifact
    """Inventory linking run IDs to nested result bundles."""
    metadata: BundleArtifact
    """Benchmark-suite key-value metadata."""

class BenchmarkBundle:
    """Opened benchmark suite and its nested result bundles."""

    path: Path
    """Absolute benchmark-bundle directory."""
    manifest_path: Path
    """Absolute JSON manifest path."""
    name: str
    """Stable benchmark-suite name from the manifest."""
    label: str | None
    """Optional human-readable suite label."""
    tables: BenchmarkBundleTables
    """Benchmark-level Parquet artifacts."""
    results: Mapping[str, ResultBundle]
    """Nested result bundles keyed by run ID."""

class BenchmarkSuite:
    """Mutable collection used to export comparable reconstruction runs.

    ``name`` must be nonempty and identifies the suite in the bundle manifest.
    """

    def __init__(self, name: str) -> None: ...
    def add_result(
        self,
        result: ReconstructionResult,
        *,
        case_id: str,
        dataset_name: str,
        algorithm_configuration: str = "",
    ) -> str:
        """Add a completed result and return its deterministic run ID.

        ``case_id`` and ``dataset_name`` must be nonempty. The optional
        configuration string distinguishes algorithm settings for reporting.
        """
    def write_bundle(
        self,
        path: Path,
        *,
        label: str | None = None,
    ) -> BenchmarkBundle:
        """Write the suite and nested result bundles, then reopen it lazily."""

def read_benchmark_bundle(path: Path) -> BenchmarkBundle:
    """Validate and lazily open a benchmark bundle directory."""

class ProgressLogger:
    """Print iteration progress every ``every`` completed iterations."""

    def __init__(self, *, every: int = 1) -> None: ...

class CheckpointEvery:
    """Write a resumable checkpoint to ``directory`` every ``every`` iterations."""

    def __init__(self, every: int, directory: Path) -> None: ...

class CsvLogger:
    """Append iteration, objective, and elapsed time records to a CSV file."""

    def __init__(self, path: Path) -> None: ...

class StopOnPlateau:
    """Stop after objective improvement remains too small for ``patience`` records.

    ``minimum_improvement`` is the nonnegative absolute objective decrease
    required to reset the patience counter.
    """

    def __init__(self, patience: int, *, minimum_improvement: float = 0.0) -> None: ...

class SaveImageEvery:
    """Save object amplitude and phase PNG files at an iteration interval."""

    def __init__(self, every: int, directory: Path) -> None: ...

class SavePupilEvery:
    """Save pupil amplitude and phase PNG files at an iteration interval."""

    def __init__(self, every: int, directory: Path) -> None: ...

class SaveResidualsEvery:
    """Save the current per-frame residual stack at an iteration interval."""

    def __init__(self, every: int, directory: Path) -> None: ...

class IterationCallback:
    """Call Python with a read-only step mapping every ``every`` iterations.

    The callable may return ``False`` to stop or any other object to continue.
    Unlike Rust-backed callbacks, this callback reacquires the GIL for each
    invocation and therefore has interpreter-crossing overhead.
    """
    def __init__(
        self, callable: Callable[[Mapping[str, Any]], object], *, every: int = 1
    ) -> None: ...

class DiagnosticRecorder:
    """Collect structured reconstruction diagnostics entirely in Rust.

    ``mode`` is one of ``"minimal"``, ``"basic"``, ``"debug"``, or
    ``"simulation"``. ``every`` is the positive iteration-recording interval.
    Add the recorder to ``callbacks`` and query it after the run.
    """

    def __init__(self, mode: str = "basic", *, every: int = 1) -> None: ...
    @property
    def mode(self) -> str:
        """Return the configured diagnostic preset name."""
    @property
    def every(self) -> int:
        """Return the positive iteration-recording interval."""
    def diagnostics(self) -> dict[str, Any]:
        """Return a new dictionary containing all records gathered so far."""
    def to_json(self, path: Path) -> None:
        """Serialize current diagnostics as JSON without holding the Python GIL."""

Callback: TypeAlias = (
    ProgressLogger
    | CheckpointEvery
    | CsvLogger
    | StopOnPlateau
    | SaveImageEvery
    | SavePupilEvery
    | SaveResidualsEvery
    | IterationCallback
    | DiagnosticRecorder
)
"""Rust-backed or Python callback accepted by reconstruction algorithms."""

class _Algorithm:
    def run(
        self,
        problem: ReconstructionProblem,
        *,
        callbacks: Iterable[Callback] | None = None,
        resume_from: ReconstructionCheckpoint | None = None,
        schedule: str = "sequential",
        schedule_seed: int = 0,
    ) -> ReconstructionResult:
        """Run reconstruction without holding the Python GIL.

        Parameters
        ----------
        problem
            Validated measurements and compiled model.
        callbacks
            Optional callbacks invoked in their listed order.
        resume_from
            Compatible checkpoint whose algorithm state and completed progress
            should be restored.
        schedule
            Frame order: ``"sequential"``, ``"reverse"``, or ``"shuffled"``.
        schedule_seed
            Deterministic seed used only by the shuffled schedule.
        """

class AlternatingProjection(_Algorithm):
    """Alternating-projection FPM reconstruction.

    Each frame selects an overlapping object-spectrum patch, propagates it
    through the pupil, replaces the predicted detector amplitude with the
    measured amplitude while retaining phase, and back-projects the correction.

    Parameters
    ----------
    iterations
        Complete passes through the acquisition schedule.
    object_step
        Relaxation factor for object-spectrum corrections.
    batch_size
        Measured frames supplied to each reconstruction step.
    epsilon
        Positive numerical floor for divisions and dark fields.
    loss_type
        Diagnostic loss; the projection itself always enforces amplitude.

    References
    ----------
    [Zheng, Horstmeyer, and Yang, *Wide-field, high-resolution Fourier
    ptychographic microscopy* (2013)](https://doi.org/10.1038/nphoton.2013.187).
    """
    def __init__(
        self,
        *,
        iterations: int = 50,
        object_step: float = 1.0,
        batch_size: int = 1,
        epsilon: float = 1e-10,
        loss_type: str = "amplitude_mse",
    ) -> None: ...

class Fpie(_Algorithm):
    """Regularized ptychographic iterative engine adapted to FPM.

    The amplitude-projection correction is preconditioned by a blend of local
    and maximum pupil power, suppressing unstable updates in weak-transfer
    regions.

    Parameters
    ----------
    iterations
        Complete passes through the acquisition schedule.
    object_step
        Relaxation factor for object-spectrum corrections.
    stability
        Blend from local (0) to maximum (1) pupil power in the denominator.
    batch_size
        Measured frames supplied to each reconstruction step.
    epsilon
        Positive floor added to the rPIE denominator.
    loss_type
        Diagnostic loss; the projection itself always enforces amplitude.

    References
    ----------
    [Maiden, Johnson, and Li, *Further improvements to the ptychographical
    iterative engine* (2017)](https://doi.org/10.1364/OPTICA.4.000736).
    """
    def __init__(
        self,
        *,
        iterations: int = 50,
        object_step: float = 0.8,
        stability: float = 0.1,
        batch_size: int = 1,
        epsilon: float = 1e-10,
        loss_type: str = "amplitude_mse",
    ) -> None: ...

class Epry(_Algorithm):
    """Embedded pupil-recovery reconstruction for FPM.

    EPRY uses projected exit-wave errors to update the object spectrum and
    complex pupil together, separating specimen structure from aberrations.
    Per-frame gain and background recovery are fpm-rs extensions.

    Parameters
    ----------
    iterations
        Complete passes through the acquisition schedule.
    object_step, pupil_step
        Relaxation factors for object and pupil corrections.
    batch_size
        Measured frames supplied to each step.
    recover_pupil
        Whether to update the complex pupil.
    constrain_pupil_support
        Whether to zero pupil values outside the compiled aperture.
    recover_frame_gains, recover_background
        Whether to estimate multiplicative gains or uniform additive backgrounds.
    gain_step, background_step
        Fractions of each calibration estimate applied per update.
    gain_bounds, background_bounds
        Inclusive lower and upper limits for recovered calibration values.
    epsilon
        Positive numerical floor for normalized updates.
    loss_type
        Diagnostic loss; the projection itself always enforces amplitude.

    References
    ----------
    [Ou, Zheng, and Yang, *Embedded pupil function recovery for Fourier
    ptychographic microscopy* (2014)](https://doi.org/10.1364/OE.22.004960).
    """
    def __init__(
        self,
        *,
        iterations: int = 100,
        object_step: float = 0.8,
        pupil_step: float = 0.1,
        batch_size: int = 1,
        recover_pupil: bool = True,
        constrain_pupil_support: bool = True,
        recover_frame_gains: bool = False,
        gain_step: float = 0.2,
        gain_bounds: tuple[float, float] = (1e-6, 1e6),
        recover_background: bool = False,
        background_step: float = 0.2,
        background_bounds: tuple[float, float] = (0.0, 1e12),
        epsilon: float = 1e-10,
        loss_type: str = "amplitude_mse",
    ) -> None: ...

class Admm(_Algorithm):
    """Linearized ADMM reconstruction for FPM.

    Per-mode auxiliary and dual fields separate detector-amplitude fitting from
    object consensus. Steps alternate an amplitude proximal operation, a
    pupil-preconditioned object update, and a scaled-dual update.

    Parameters
    ----------
    iterations
        Complete passes through the acquisition schedule.
    object_step
        Step size of the linearized object update.
    penalty
        Positive augmented-Lagrangian consensus penalty.
    dual_relaxation
        Scaled-dual relaxation in the inclusive range ``[0, 2]``.
    batch_size
        Frames per step; ``None`` processes all frames together.
    epsilon
        Positive numerical floor for normalized updates.

    References
    ----------
    [Wang et al., *Fourier Ptychographic Microscopy via Alternating Direction
    Method of Multipliers* (2022)](https://doi.org/10.3390/cells11091512).
    """
    def __init__(
        self,
        *,
        iterations: int = 100,
        object_step: float = 0.8,
        penalty: float = 1.0,
        dual_relaxation: float = 1.0,
        batch_size: int | None = None,
        epsilon: float = 1e-10,
    ) -> None: ...

class GradientDescent(_Algorithm):
    """Wirtinger-style loss-gradient reconstruction for Fourier ptychography.

    The solver differentiates a selected data loss through the FPM forward
    model, averages mini-batch gradients, and applies a pupil-power-
    preconditioned object update. Pupil and illumination recovery and object or
    pupil regularization are optional.

    Parameters
    ----------
    iterations
        Complete passes through the acquisition schedule.
    object_step
        Step size of the preconditioned object update.
    batch_size
        Frame gradients averaged into one update.
    epsilon
        Positive floor used by losses and preconditioners.
    loss_type
        Data loss to optimize and report.
    recover_illumination
        Whether to estimate source offsets on the Fourier grid.
    illumination_step, illumination_finite_difference, illumination_bounds
        Offset-update step, finite-difference spacing, and maximum absolute
        correction, all expressed in Fourier-grid pixels where applicable.
    recover_pupil, pupil_step, constrain_pupil_support
        Whether and how to update the complex pupil within its aperture.
    object_tv, object_tv_epsilon
        Complex-object total-variation weight and smoothing constant; a zero
        weight disables the regularizer.
    pupil_smoothing
        Nonnegative quadratic pupil-smoothing weight.
    parallel_workers
        Worker limit; zero selects available CPU parallelism.

    References
    ----------
    [Bian et al., *Fourier ptychographic reconstruction using Wirtinger flow
    optimization* (2015)](https://doi.org/10.1364/OE.23.004856).
    """
    def __init__(
        self,
        *,
        iterations: int = 100,
        object_step: float = 0.5,
        batch_size: int = 1,
        epsilon: float = 1e-10,
        loss_type: str = "amplitude_mse",
        recover_illumination: bool = False,
        illumination_step: float = 0.1,
        illumination_finite_difference: float = 0.05,
        illumination_bounds: float = 1.0,
        recover_pupil: bool = False,
        pupil_step: float = 0.05,
        constrain_pupil_support: bool = True,
        object_tv: float = 0.0,
        object_tv_epsilon: float = 1e-6,
        pupil_smoothing: float = 0.0,
        parallel_workers: int = 0,
    ) -> None: ...
