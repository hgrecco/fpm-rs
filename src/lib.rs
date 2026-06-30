//! Reusable reconstruction and simulation for image-plane Fourier ptychography.
//!
//! The central boundary is [`model::ImagePlaneModel`]: experimental descriptions
//! compile into this computational model, and algorithms only consume the model.

pub mod array;
pub mod complex;
pub mod diagnostics;
pub mod error;
pub mod experiment;

pub use array::Array2;
pub use error::{Error, Result};
pub use num_complex::Complex64;
