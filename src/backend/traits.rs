use std::any::Any;

use num_complex::Complex64;

use crate::Result;

/// Direction of an unnormalized two-dimensional complex FFT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftDirection {
    /// Spatial domain to centered or uncentered spectrum as managed by the caller.
    Forward,
    /// Spectrum to spatial domain, normalized by the backend to invert [`Self::Forward`].
    Inverse,
}

/// Memory domain in which a backend buffer resides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Ordinary CPU-addressable host memory.
    Host,
    /// Accelerator/device memory not directly represented by a host slice.
    Device,
}

/// Feature declaration returned by an execution backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Memory domain preferred for persistent reconstruction state.
    pub preferred_memory: MemoryLocation,
    /// Whether [`Backend::resident_backend`] exposes typed persistent buffers.
    pub resident_buffers: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            preferred_memory: MemoryLocation::Host,
            resident_buffers: false,
        }
    }
}

/// Type-erased contiguous buffer of [`Complex64`] values owned by a resident backend.
pub trait ComplexBuffer: Send + Sync {
    /// Returns the number of complex elements.
    fn len(&self) -> usize;
    /// Returns whether the buffer contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the buffer's memory domain.
    fn location(&self) -> MemoryLocation;
    /// Exposes immutable type erasure for backend-specific downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Exposes mutable type erasure for backend-specific downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Type-erased contiguous buffer of `f64` values owned by a resident backend.
pub trait RealBuffer: Send + Sync {
    /// Returns the number of real elements.
    fn len(&self) -> usize;
    /// Returns whether the buffer contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the buffer's memory domain.
    fn location(&self) -> MemoryLocation;
    /// Exposes immutable type erasure for backend-specific downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Exposes mutable type erasure for backend-specific downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Optional typed-buffer operations for backends that can keep reconstruction
/// state resident in their preferred memory domain.
pub trait ResidentBackend: Send + Sync {
    /// Allocates `len` uninitialized or zeroed complex elements according to backend policy.
    fn allocate_complex(&self, len: usize) -> Result<Box<dyn ComplexBuffer>>;
    /// Allocates `len` uninitialized or zeroed real elements according to backend policy.
    fn allocate_real(&self, len: usize) -> Result<Box<dyn RealBuffer>>;
    /// Copies a host complex slice into an equal-length resident buffer.
    fn upload_complex(
        &self,
        destination: &mut dyn ComplexBuffer,
        source: &[Complex64],
    ) -> Result<()>;
    /// Copies an equal-length resident complex buffer into a host slice.
    fn download_complex(
        &self,
        source: &dyn ComplexBuffer,
        destination: &mut [Complex64],
    ) -> Result<()>;
    /// Copies a host real slice into an equal-length resident buffer.
    fn upload_real(&self, destination: &mut dyn RealBuffer, source: &[f64]) -> Result<()>;
    /// Copies an equal-length resident real buffer into a host slice.
    fn download_real(&self, source: &dyn RealBuffer, destination: &mut [f64]) -> Result<()>;
    /// Applies an in-place two-dimensional FFT to `values` shaped `(height, width)`.
    fn fft2_resident(
        &self,
        values: &mut dyn ComplexBuffer,
        shape: (usize, usize),
        direction: FftDirection,
    ) -> Result<()>;
}

/// Minimal backend boundary. Buffers stay owned by reconstruction state.
pub trait Backend: Send + Sync {
    /// Reports memory and resident-buffer capabilities.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }

    /// Returns optional resident-buffer operations when declared by [`Self::capabilities`].
    fn resident_backend(&self) -> Option<&dyn ResidentBackend> {
        None
    }

    /// Applies an in-place normalized 2-D FFT to a row-major host buffer.
    ///
    /// `values` must contain `height * width` elements and `column_scratch` must fit the
    /// larger axis required by the implementation.
    fn fft2(
        &self,
        values: &mut [Complex64],
        shape: (usize, usize),
        direction: FftDirection,
        column_scratch: &mut [Complex64],
    ) -> Result<()>;
}
