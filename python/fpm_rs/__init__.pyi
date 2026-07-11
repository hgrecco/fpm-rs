from collections.abc import Callable, Iterable, Mapping, Sequence
from os import PathLike
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
import fpm_rs.diagnostics as diagnostics
import fpm_rs.plot as plot

Shape2D: TypeAlias = tuple[int, int]
FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
MaskArray: TypeAlias = NDArray[np.uint8]
Path: TypeAlias = str | PathLike[str]

__version__: str

class FpmError(Exception): ...
class InvalidShapeError(FpmError): ...
class InvalidParameterError(FpmError): ...
class InvalidModelError(FpmError): ...
class InvalidMeasurementsError(FpmError): ...
class LengthMismatchError(FpmError): ...
class FrameOutOfRangeError(FpmError): ...
class NumericalError(FpmError): ...
class UnsupportedError(FpmError): ...
class DatasetError(FpmError): ...
class FpmIoError(FpmError): ...
class SerializationError(FpmError): ...

class PupilAberration:
    def __init__(self, *, astigmatism: float = ..., coma: float = ..., spherical: float = ..., edge_apodization: float = ...) -> None: ...
    @property
    def astigmatism(self) -> float: ...
    @property
    def coma(self) -> float: ...
    @property
    def spherical(self) -> float: ...
    @property
    def edge_apodization(self) -> float: ...

class Optics:
    def __init__(self, wavelength: float, objective_na: float, magnification: float, camera_pixel_size: float, *, medium_index: float = ..., defocus_distance: float | None = ..., pupil_aberration: PupilAberration | None = ...) -> None: ...
    @property
    def wavelength(self) -> float: ...
    @property
    def objective_na(self) -> float: ...
    @property
    def magnification(self) -> float: ...
    @property
    def camera_pixel_size(self) -> float: ...
    @property
    def medium_index(self) -> float: ...
    @property
    def defocus_distance(self) -> float | None: ...
    @property
    def object_pixel_size(self) -> float: ...

class LEDArray:
    def __init__(self, grid_shape: Shape2D, pitch: float, distance: float, center: tuple[float, float], *, wavelength_override: float | None = ..., illumination_order: Sequence[int] | None = ..., intensity_weights: Sequence[float] | None = ..., rotation_degrees: float = ...) -> None: ...
    @property
    def grid_shape(self) -> Shape2D: ...
    @property
    def source_count(self) -> int: ...

class LEDSphere:
    def __init__(self, angles: FloatArray, radius: float, *, center_offset: tuple[float, float, float] = ..., orientation_degrees: tuple[float, float, float] = ..., angular_corrections: FloatArray | None = ..., wavelength_override: float | None = ..., illumination_order: Sequence[int] | None = ..., intensity_weights: Sequence[float] | None = ...) -> None: ...
    @property
    def source_count(self) -> int: ...

class SphericalLEDArm:
    def __init__(self, commanded_angles: FloatArray, arm_length: float, *, pivot_offset: tuple[float, float, float] = ..., orientation_degrees: tuple[float, float, float] = ..., theta_zero_degrees: float = ..., phi_zero_degrees: float = ..., theta_scale: float = ..., phi_scale: float = ..., elevation_axis_tilt_degrees: float = ..., theta_backlash_degrees: float = ..., phi_backlash_degrees: float = ..., wavelength_override: float | None = ..., intensity_weights: Sequence[float] | None = ...) -> None: ...
    @property
    def source_count(self) -> int: ...

class RotatingLEDArc:
    def __init__(self, led_thetas: Sequence[float], rotation_angles: Sequence[float], radius: float, *, axis_origin_offset: tuple[float, float, float] = ..., axis_tilt_degrees: tuple[float, float] = ..., led_angular_corrections: FloatArray | None = ..., led_radial_offsets: Sequence[float] | None = ..., rotation_zero_degrees: float = ..., rotation_scale: float = ..., rotation_backlash_degrees: float = ..., wavelength_override: float | None = ..., led_intensity_weights: Sequence[float] | None = ...) -> None: ...
    @property
    def led_count(self) -> int: ...
    @property
    def rotation_count(self) -> int: ...
    @property
    def source_count(self) -> int: ...

class AngleList:
    def __init__(self, angles: FloatArray) -> None: ...
    @property
    def source_count(self) -> int: ...

class KVectorList:
    def __init__(self, k_vectors: FloatArray) -> None: ...
    @property
    def source_count(self) -> int: ...

class CodedIllumination:
    def __init__(self, k_vectors: FloatArray, frame_weights: FloatArray) -> None: ...
    @property
    def source_count(self) -> int: ...
    @property
    def frame_count(self) -> int: ...

class CameraModel:
    def __init__(self, *, photons_per_pixel: float = ..., gain_counts_per_electron: float = ..., offset_counts: float = ..., read_noise_electrons: float = ..., dark_current_electrons: float = ..., shot_noise: bool = ..., pixel_sensitivity: FloatArray | None = ..., bit_depth: int | None = ..., saturation_counts: float | None = ..., quantize: bool = ..., bad_pixels: Sequence[int] = ..., bad_pixel_value_counts: float | None = ...) -> None: ...
    @staticmethod
    def ideal() -> CameraModel: ...

class IlluminationAcquisitionErrors:
    def __init__(self, *, frame_gain_relative_std: float = ..., missing_frames: Sequence[int] = ..., source_permutation: Sequence[int] | None = ...) -> None: ...

class SyntheticObject:
    def __init__(self, field: ComplexArray) -> None: ...
    @staticmethod
    def constant(shape: Shape2D, amplitude: float = ..., phase: float = ...) -> SyntheticObject: ...
    @staticmethod
    def amplitude_only(amplitude: FloatArray) -> SyntheticObject: ...
    @staticmethod
    def phase_only(phase: FloatArray) -> SyntheticObject: ...
    @staticmethod
    def from_amplitude_phase(amplitude: FloatArray, phase: FloatArray) -> SyntheticObject: ...
    @staticmethod
    def from_amplitude_image(path: Path) -> SyntheticObject: ...
    @staticmethod
    def from_amplitude_phase_images(amplitude_path: Path, phase_path: Path, phase_extent: float) -> SyntheticObject: ...
    @staticmethod
    def phase_disk(shape: Shape2D, radius_pixels: float, phase_shift: float) -> SyntheticObject: ...
    @staticmethod
    def siemens_star(shape: Shape2D, spokes: int) -> SyntheticObject: ...
    @staticmethod
    def resolution_target(shape: Shape2D) -> SyntheticObject: ...
    @staticmethod
    def random_phase(shape: Shape2D, standard_deviation: float, seed: int) -> SyntheticObject: ...
    @staticmethod
    def particle_field(shape: Shape2D, particles: int, seed: int) -> SyntheticObject: ...
    @staticmethod
    def mixed_test_pattern(shape: Shape2D) -> SyntheticObject: ...
    @staticmethod
    def biological_like(shape: Shape2D, features: int, seed: int) -> SyntheticObject: ...
    @property
    def field(self) -> ComplexArray: ...
    @property
    def shape(self) -> Shape2D: ...
    @property
    def label(self) -> str | None: ...

class ImagePlaneModel:
    @property
    def image_shape(self) -> Shape2D: ...
    @property
    def reconstruction_shape(self) -> Shape2D: ...
    @property
    def source_count(self) -> int: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def is_multiplexed(self) -> bool: ...
    @property
    def k_vectors(self) -> FloatArray: ...
    @property
    def pupil(self) -> ComplexArray: ...
    @property
    def pupil_support(self) -> MaskArray: ...
    @property
    def frame_gains(self) -> FloatArray | None: ...

Illumination: TypeAlias = LEDArray | LEDSphere | SphericalLEDArm | RotatingLEDArc | AngleList | KVectorList | CodedIllumination
def compile_model(optics: Optics, illumination: Illumination, image_shape: Shape2D, reconstruction_shape: Shape2D) -> ImagePlaneModel: ...
def compile_camera_model(model: ImagePlaneModel, camera: CameraModel) -> ImagePlaneModel: ...

class MeasurementStack:
    def __init__(self, measurements: FloatArray, *, frame_weights: Sequence[float] | None = ..., masks: MaskArray | None = ...) -> None: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def image_shape(self) -> Shape2D: ...
    @property
    def array(self) -> FloatArray: ...
    @property
    def frame_weights(self) -> FloatArray: ...

class SimulationResult:
    @property
    def measurements(self) -> MeasurementStack: ...
    @property
    def ground_truth_object(self) -> ComplexArray: ...
    @property
    def true_model(self) -> ImagePlaneModel: ...
    @property
    def reconstruction_model(self) -> ImagePlaneModel: ...
    @property
    def ideal(self) -> bool: ...
    @property
    def missing_frames(self) -> list[int]: ...
    @property
    def random_seed(self) -> int: ...

def simulate(true_model: ImagePlaneModel, object: ComplexArray | SyntheticObject, *, reconstruction_model: ImagePlaneModel | None = ..., camera: CameraModel | None = ..., illumination_errors: IlluminationAcquisitionErrors | None = ..., seed: int = ...) -> SimulationResult: ...

class ReconstructionProblem:
    def __init__(self, measurements: FloatArray | MeasurementStack, model: ImagePlaneModel, *, frame_weights: Sequence[float] | None = ..., masks: MaskArray | None = ..., name: str | None = ...) -> None: ...
    @property
    def name(self) -> str | None: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def image_shape(self) -> Shape2D: ...
    @property
    def reconstruction_shape(self) -> Shape2D: ...

class ReconstructionCheckpoint:
    @staticmethod
    def load(path: Path) -> ReconstructionCheckpoint: ...
    def save(self, path: Path) -> None: ...
    @property
    def format_version(self) -> int: ...
    @property
    def completed_iterations(self) -> int: ...

class RuntimeInfo:
    elapsed_seconds: float
    completed_iterations: int
    stopped_early: bool
    algorithm: str

class ReconstructionResult:
    object: ComplexArray
    amplitude: FloatArray
    phase: FloatArray
    object_spectrum: ComplexArray
    recovered_pupil: ComplexArray
    pupil_support: MaskArray
    calibrated_illumination: FloatArray | None
    recovered_frame_gains: FloatArray | None
    recovered_background: FloatArray | None
    history: list[tuple[int, float, float]]
    admm_residual_history: list[tuple[int, float, float]]
    diagnostics: Mapping[str, float]
    runtime: RuntimeInfo
    metadata: Mapping[str, str]
    final_loss: float | None

class ProgressLogger:
    def __init__(self, *, every: int = ...) -> None: ...
class CheckpointEvery:
    def __init__(self, every: int, directory: Path) -> None: ...
class CsvLogger:
    def __init__(self, path: Path) -> None: ...
class StopOnPlateau:
    def __init__(self, patience: int, *, minimum_improvement: float = ...) -> None: ...
class SaveImageEvery:
    def __init__(self, every: int, directory: Path) -> None: ...
class SavePupilEvery:
    def __init__(self, every: int, directory: Path) -> None: ...
class SaveResidualsEvery:
    def __init__(self, every: int, directory: Path) -> None: ...
class IterationCallback:
    def __init__(self, callable: Callable[[Mapping[str, Any]], object], *, every: int = ...) -> None: ...

class DiagnosticRecorder:
    def __init__(self, mode: str = ..., *, every: int = ...) -> None: ...
    @property
    def mode(self) -> str: ...
    @property
    def every(self) -> int: ...
    def diagnostics(self) -> dict[str, Any]: ...
    def to_json(self, path: Path) -> None: ...

Callback: TypeAlias = ProgressLogger | CheckpointEvery | CsvLogger | StopOnPlateau | SaveImageEvery | SavePupilEvery | SaveResidualsEvery | IterationCallback | DiagnosticRecorder

class _Algorithm:
    def run(self, problem: ReconstructionProblem, *, callbacks: Iterable[Callback] | None = ..., resume_from: ReconstructionCheckpoint | None = ..., schedule: str = ..., schedule_seed: int = ...) -> ReconstructionResult: ...

class AlternatingProjection(_Algorithm):
    """Alternating-projection FPM reconstruction.

    Each frame selects an overlapping object-spectrum patch, propagates it
    through the pupil, replaces the predicted detector amplitude with the
    measured amplitude while retaining phase, and back-projects the correction.

    Parameters:
        iterations: Complete passes through the acquisition schedule.
        object_step: Relaxation factor for object-spectrum corrections.
        batch_size: Measured frames supplied to each reconstruction step.
        epsilon: Positive numerical floor for divisions and dark fields.
        loss_type: Diagnostic loss; the projection always enforces amplitude.

    Reference: G. Zheng, R. Horstmeyer, and C. Yang, "Wide-field,
    high-resolution Fourier ptychographic microscopy," Nature Photonics 7,
    739-745 (2013), doi:10.1038/nphoton.2013.187.
    """
    def __init__(self, *, iterations: int = ..., object_step: float = ..., batch_size: int = ..., epsilon: float = ..., loss_type: str = ...) -> None: ...
class Fpie(_Algorithm):
    """Regularized ptychographic iterative engine adapted to FPM.

    The amplitude-projection correction is preconditioned by a blend of local
    and maximum pupil power, suppressing unstable updates in weak-transfer
    regions.

    Parameters:
        iterations: Complete passes through the acquisition schedule.
        object_step: Relaxation factor for object-spectrum corrections.
        stability: Blend from local (0) to maximum (1) pupil power.
        batch_size: Measured frames supplied to each reconstruction step.
        epsilon: Positive floor added to the rPIE denominator.
        loss_type: Diagnostic loss; the projection always enforces amplitude.

    Reference: A. Maiden, D. Johnson, and P. Li, "Further improvements to the
    ptychographical iterative engine," Optica 4(7), 736-745 (2017),
    doi:10.1364/OPTICA.4.000736.
    """
    def __init__(self, *, iterations: int = ..., object_step: float = ..., stability: float = ..., batch_size: int = ..., epsilon: float = ..., loss_type: str = ...) -> None: ...
class Epry(_Algorithm):
    """Embedded pupil-recovery reconstruction for FPM.

    EPRY uses projected exit-wave errors to update the object spectrum and
    complex pupil together, separating specimen structure from aberrations.
    Per-frame gain and background recovery are fpm-rs extensions.

    Parameters:
        iterations: Complete passes through the acquisition schedule.
        object_step: Relaxation factor for object corrections.
        pupil_step: Relaxation factor for pupil corrections.
        batch_size: Measured frames supplied to each step.
        recover_pupil: Whether to update the complex pupil.
        constrain_pupil_support: Zero pupil values outside the aperture.
        recover_frame_gains: Estimate a multiplicative gain per frame.
        gain_step: Fraction of each gain estimate applied per update.
        gain_bounds: Lower and upper bounds for recovered gains.
        recover_background: Estimate a uniform additive background per frame.
        background_step: Fraction of the mean residual applied per update.
        background_bounds: Bounds for recovered background intensities.
        epsilon: Positive numerical floor for normalized updates.
        loss_type: Diagnostic loss; the projection always enforces amplitude.

    Reference: X. Ou, G. Zheng, and C. Yang, "Embedded pupil function recovery
    for Fourier ptychographic microscopy," Optics Express 22(5), 4960-4972
    (2014), doi:10.1364/OE.22.004960.
    """
    def __init__(self, **kwargs: Any) -> None: ...
class Admm(_Algorithm):
    """Linearized ADMM reconstruction for FPM.

    Per-mode auxiliary and dual fields separate detector-amplitude fitting from
    object consensus. Steps alternate an amplitude proximal operation, a
    pupil-preconditioned object update, and a scaled-dual update.

    Parameters:
        iterations: Complete passes through the acquisition schedule.
        object_step: Step size of the linearized object update.
        penalty: Positive augmented-Lagrangian consensus penalty.
        dual_relaxation: Scaled-dual relaxation in the range [0, 2].
        batch_size: Frames per step; None processes all frames together.
        epsilon: Positive numerical floor for normalized updates.

    Reference: A. Wang, Z. Zhang, S. Wang, A. Pan, C. Ma, and B. Yao,
    "Fourier Ptychographic Microscopy via Alternating Direction Method of
    Multipliers," Cells 11(9), 1512 (2022), doi:10.3390/cells11091512.
    """
    def __init__(self, *, iterations: int = ..., object_step: float = ..., penalty: float = ..., dual_relaxation: float = ..., batch_size: int | None = ..., epsilon: float = ...) -> None: ...
class GradientDescent(_Algorithm):
    """Wirtinger-style loss-gradient reconstruction for Fourier ptychography.

    The solver differentiates a selected data loss through the FPM forward
    model, averages mini-batch gradients, and applies a pupil-power-
    preconditioned object update. Pupil and illumination recovery and object or
    pupil regularization are optional.

    Parameters:
        iterations: Complete passes through the acquisition schedule.
        object_step: Step size of the preconditioned object update.
        batch_size: Frame gradients averaged into one update.
        epsilon: Positive floor used by losses and preconditioners.
        loss_type: Data loss to optimize and report.
        recover_illumination: Estimate Fourier-grid illumination offsets.
        illumination_step: Step size of the scaled offset update.
        illumination_finite_difference: Difference spacing in grid pixels.
        illumination_bounds: Maximum absolute correction in grid pixels.
        recover_pupil: Whether to update the complex pupil.
        pupil_step: Step size of the normalized pupil update.
        constrain_pupil_support: Zero pupil values outside the aperture.
        object_tv: Complex-object total-variation weight; 0 disables it.
        object_tv_epsilon: Smoothing constant in the TV norm.
        pupil_smoothing: Quadratic pupil-smoothing weight.
        parallel_workers: Worker limit; 0 selects available CPU parallelism.

    Reference: L. Bian, J. Suo, G. Zheng, K. Guo, F. Chen, and Q. Dai,
    "Fourier ptychographic reconstruction using Wirtinger flow optimization,"
    Optics Express 23(4), 4856-4866 (2015), doi:10.1364/OE.23.004856.
    """
    def __init__(self, **kwargs: Any) -> None: ...
