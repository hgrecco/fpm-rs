//! Reusable reconstruction and simulation for image-plane Fourier ptychography.
//!
//! The central boundary is [`model::ImagePlaneModel`]: experimental descriptions
//! compile into this computational model, and algorithms only consume the model.

pub mod algorithms;
mod array_layout;
mod array_serde;
pub mod backend;
pub mod benchmark;
#[cfg(feature = "parquet")]
pub mod benchmark_bundle;
pub mod callbacks;
pub mod complex;
pub mod configuration;
pub mod datasets;
pub mod diagnostics;
pub mod error;
pub mod evaluation;
pub mod experiment;
mod image_io;
pub mod measurements;
pub mod metrics;
pub mod model;
pub mod reconstruction;
pub mod simulation;
#[cfg(feature = "tabular")]
pub mod tabular;

#[cfg(feature = "parquet")]
pub use benchmark_bundle::{BenchmarkBundle, read_benchmark_bundle};
pub use error::{Error, Result};
pub use num_complex::Complex64;
#[cfg(feature = "parquet")]
pub use reconstruction::{ResultBundle, read_bundle};
