#![deny(missing_docs)]
//! Image-plane Fourier ptychographic microscopy (FPM) reconstruction and simulation.
//!
//! The crate separates experimental geometry from numerical reconstruction. A typical
//! workflow is:
//!
//! 1. describe the microscope with [`experiment::Optics`] and an illumination source
//!    such as [`experiment::PlanarLedArray`], plus calibration and acquisition;
//! 2. compile that description into an algorithm-facing [`model::ImagePlaneModel`];
//! 3. pair the model with resident or lazy [`measurements`] in a
//!    [`reconstruction::ReconstructionProblem`];
//! 4. select an [`algorithms`] implementation and run it directly or through a
//!    [`reconstruction::Runner`];
//! 5. inspect the complex object, amplitude, phase, pupil, trace, and [`diagnostics`]
//!    in the resulting [`reconstruction::ReconstructionResult`]; and
//! 6. optionally create controlled data with [`simulation`], calculate [`metrics`],
//!    or persist result and benchmark bundles through [`reconstruction::ResultBundle`]
//!    and [`benchmark`].
//!
//! Experimental coordinates and detector arrays use explicit conventions documented on
//! their types. Two-dimensional arrays are generally indexed `(row, column)` and shaped
//! `(height, width)`; transverse wave vectors are `(kx, ky)` in radians per metre.
//!
//! # Example
//!
//! ```
//! use fpm_rs::{
//!     algorithms::{AlternatingProjection, ReconstructionAlgorithm},
//!     experiment::{ArrayPose, Illumination, Optics, PlanarLedArray},
//!     measurements::{FrameMetadata, MeasurementStack},
//!     model::{ImagePlaneModel, ReconstructionShape},
//!     reconstruction::ReconstructionProblem,
//! };
//! use ndarray::Array3;
//!
//! # fn main() -> fpm_rs::Result<()> {
//! let optics = Optics {
//!     wavelength_vacuum_m: 532e-9,
//!     objective_na: 0.1,
//!     magnification: 4.0,
//!     camera_pixel_size: 6.5e-6,
//!     illumination_refractive_index: 1.0,
//!     objective_medium_refractive_index: 1.0,
//!     defocus_distance: None,
//!     pupil_aberration: None,
//! };
//! let illumination = Illumination::from_geometry(PlanarLedArray::new(
//!     (1, 1),
//!     (4e-3, 4e-3),
//!     (0.0, 0.0),
//!     ArrayPose::from_translation([0.0, 0.0, -90e-3]),
//! ))?;
//! let model = ImagePlaneModel::from_experiment(
//!     &optics,
//!     &illumination,
//!     (4, 4),
//!     ReconstructionShape::Exact((4, 4)),
//! )?;
//! let measurements = MeasurementStack::new(
//!     Array3::from_elem((1, 4, 4), 1.0),
//!     vec![FrameMetadata::new(0)],
//! )?;
//! let problem = ReconstructionProblem::new(measurements, model)?;
//! let result = AlternatingProjection::default().iterations(1).run(&problem)?;
//! assert_eq!(result.object.dim(), (4, 4));
//! # Ok(())
//! # }
//! ```

/// Reconstruction algorithms and their shared execution contract.
pub mod algorithms;
mod array_layout;
mod array_serde;
/// CPU execution and the interfaces used by future resident-buffer backends.
pub mod backend;
pub mod benchmark;
#[cfg(feature = "parquet")]
pub mod benchmark_bundle;
/// Iteration callbacks for progress, checkpoints, diagnostics, and image output.
pub mod callbacks;
/// Utilities for complex fields, amplitude, and wrapped phase.
pub mod complex;
pub mod configuration;
pub mod datasets;
/// Structured iteration and per-frame reconstruction diagnostics.
pub mod diagnostics;
/// Error categories and the crate-wide [`error::Result`] alias.
pub mod error;
pub mod evaluation;
/// Physical optics and illumination geometry compiled into numerical models.
pub mod experiment;
mod image_io;
/// Resident and lazy low-resolution intensity measurement stacks.
pub mod measurements;
pub mod metrics;
/// Compiled image-plane models, Fourier crops, pupils, and forward propagation.
pub mod model;
/// Reconstruction problems, state, schedules, runners, results, and checkpoints.
pub mod reconstruction;
/// Synthetic objects, camera response, acquisition effects, and simulation.
pub mod simulation;
#[cfg(feature = "tabular")]
pub mod tabular;

#[cfg(feature = "parquet")]
pub use benchmark_bundle::{BenchmarkBundle, read_benchmark_bundle};
pub use error::{Error, Result};
pub use num_complex::Complex64;
#[cfg(feature = "parquet")]
pub use reconstruction::{ResultBundle, read_bundle};
