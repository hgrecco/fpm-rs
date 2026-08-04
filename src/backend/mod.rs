//! Execution backends and resident-buffer abstractions.
//!
//! [`crate::backend::CpuBackend`] is the implemented backend. Backend authors implement
//! [`crate::backend::Backend`] and, when buffers can remain outside ordinary host arrays,
//! [`crate::backend::ResidentBackend`].
//! Reconstruction code accesses these interfaces through
//! [`crate::reconstruction::ReconstructionState`].

mod cpu;
mod traits;

pub use cpu::CpuBackend;
pub use traits::{
    Backend, BackendCapabilities, ComplexBuffer, FftDirection, MemoryLocation, RealBuffer,
    ResidentBackend,
};
