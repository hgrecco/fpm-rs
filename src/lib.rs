//! Reusable reconstruction and simulation for image-plane Fourier ptychography.
//!
//! The central boundary is [`model::ImagePlaneModel`]: experimental descriptions
//! compile into this computational model, and algorithms only consume the model.

pub mod algorithms;
pub mod array;
pub mod backend;
pub mod benchmark;
pub mod callbacks;
pub mod complex;
pub mod configuration;
pub mod diagnostics;
pub mod error;
pub mod experiment;
mod image_io;
pub mod measurements;
pub mod model;
pub mod reconstruction;
pub mod simulation;

pub use array::Array2;
pub use error::{Error, Result};
pub use num_complex::Complex64;
