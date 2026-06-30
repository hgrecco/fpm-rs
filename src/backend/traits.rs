use std::any::Any;

use num_complex::Complex64;

use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLocation {
    Host,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub preferred_memory: MemoryLocation,
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

pub trait ComplexBuffer: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn location(&self) -> MemoryLocation;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait RealBuffer: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn location(&self) -> MemoryLocation;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Optional typed-buffer operations for backends that can keep reconstruction
/// state resident in their preferred memory domain.
pub trait ResidentBackend: Send + Sync {
    fn allocate_complex(&self, len: usize) -> Result<Box<dyn ComplexBuffer>>;
    fn allocate_real(&self, len: usize) -> Result<Box<dyn RealBuffer>>;
    fn upload_complex(
        &self,
        destination: &mut dyn ComplexBuffer,
        source: &[Complex64],
    ) -> Result<()>;
    fn download_complex(
        &self,
        source: &dyn ComplexBuffer,
        destination: &mut [Complex64],
    ) -> Result<()>;
    fn upload_real(&self, destination: &mut dyn RealBuffer, source: &[f64]) -> Result<()>;
    fn download_real(&self, source: &dyn RealBuffer, destination: &mut [f64]) -> Result<()>;
    fn fft2_resident(
        &self,
        values: &mut dyn ComplexBuffer,
        shape: (usize, usize),
        direction: FftDirection,
    ) -> Result<()>;
}

/// Minimal backend boundary. Buffers stay owned by reconstruction state.
pub trait Backend: Send + Sync {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }

    fn resident_backend(&self) -> Option<&dyn ResidentBackend> {
        None
    }

    fn fft2(
        &self,
        values: &mut [Complex64],
        shape: (usize, usize),
        direction: FftDirection,
        column_scratch: &mut [Complex64],
    ) -> Result<()>;
}
