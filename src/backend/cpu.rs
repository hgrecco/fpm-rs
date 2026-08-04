use std::sync::Arc;

use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use crate::{Result, array_layout::checked_len_2d, error::Error};

use super::{
    Backend, BackendCapabilities, ComplexBuffer, FftDirection, MemoryLocation, RealBuffer,
    ResidentBackend,
};

#[derive(Debug)]
struct CpuComplexBuffer {
    values: Vec<Complex64>,
}

impl ComplexBuffer for CpuComplexBuffer {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn location(&self) -> MemoryLocation {
        MemoryLocation::Host
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
struct CpuRealBuffer {
    values: Vec<f64>,
}

impl RealBuffer for CpuRealBuffer {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn location(&self) -> MemoryLocation {
        MemoryLocation::Host
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
struct CpuFftPlan {
    shape: (usize, usize),
    row_forward: Arc<dyn Fft<f64>>,
    row_inverse: Arc<dyn Fft<f64>>,
    column_forward: Arc<dyn Fft<f64>>,
    column_inverse: Arc<dyn Fft<f64>>,
}

impl CpuFftPlan {
    fn new(shape: (usize, usize)) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            shape,
            row_forward: planner.plan_fft_forward(shape.1),
            row_inverse: planner.plan_fft_inverse(shape.1),
            column_forward: planner.plan_fft_forward(shape.0),
            column_inverse: planner.plan_fft_inverse(shape.0),
        }
    }
}

/// rustfft-based CPU backend with plans cached for low- and high-resolution grids.
#[derive(Clone)]
pub struct CpuBackend {
    low: CpuFftPlan,
    high: CpuFftPlan,
}

impl CpuBackend {
    /// Creates cached FFT plans for non-zero low- and high-resolution `(height, width)` grids.
    pub fn new(low_shape: (usize, usize), high_shape: (usize, usize)) -> Result<Self> {
        if low_shape.0 == 0 || low_shape.1 == 0 || high_shape.0 == 0 || high_shape.1 == 0 {
            return Err(Error::InvalidShape(
                "FFT dimensions must be non-zero".into(),
            ));
        }
        checked_len_2d(low_shape)?;
        checked_len_2d(high_shape)?;
        Ok(Self {
            low: CpuFftPlan::new(low_shape),
            high: CpuFftPlan::new(high_shape),
        })
    }

    fn plan(&self, shape: (usize, usize)) -> Result<&CpuFftPlan> {
        if shape == self.low.shape {
            Ok(&self.low)
        } else if shape == self.high.shape {
            Ok(&self.high)
        } else {
            Err(Error::InvalidShape(format!(
                "FFT shape {shape:?} is neither configured shape {:?} nor {:?}",
                self.low.shape, self.high.shape
            )))
        }
    }
}

impl Backend for CpuBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            preferred_memory: MemoryLocation::Host,
            resident_buffers: true,
        }
    }

    fn resident_backend(&self) -> Option<&dyn ResidentBackend> {
        Some(self)
    }

    fn fft2(
        &self,
        values: &mut [Complex64],
        shape: (usize, usize),
        direction: FftDirection,
        column_scratch: &mut [Complex64],
    ) -> Result<()> {
        let expected = checked_len_2d(shape)?;
        if values.len() != expected || column_scratch.len() < shape.0 {
            return Err(Error::InvalidShape(format!(
                "FFT {:?} requires {expected} values and {} column scratch values",
                shape, shape.0
            )));
        }
        let plan = self.plan(shape)?;
        let (row_fft, column_fft) = match direction {
            FftDirection::Forward => (&plan.row_forward, &plan.column_forward),
            FftDirection::Inverse => (&plan.row_inverse, &plan.column_inverse),
        };
        for row in values.chunks_exact_mut(shape.1) {
            row_fft.process(row);
        }
        let column = &mut column_scratch[..shape.0];
        for column_index in 0..shape.1 {
            for row_index in 0..shape.0 {
                column[row_index] = values[row_index * shape.1 + column_index];
            }
            column_fft.process(column);
            for row_index in 0..shape.0 {
                values[row_index * shape.1 + column_index] = column[row_index];
            }
        }
        // The forward transform is normalized by 1/N and the inverse is
        // unnormalized. This pair is exactly invertible and, unlike the common
        // inverse-normalized convention, preserves the amplitude of a constant
        // object when a high-resolution spectrum is cropped and inverse-
        // transformed on a smaller grid.
        if direction == FftDirection::Forward {
            let normalization = expected as f64;
            for value in values {
                *value /= normalization;
            }
        }
        Ok(())
    }
}

impl ResidentBackend for CpuBackend {
    fn allocate_complex(&self, len: usize) -> Result<Box<dyn ComplexBuffer>> {
        Ok(Box::new(CpuComplexBuffer {
            values: vec![Complex64::default(); len],
        }))
    }

    fn allocate_real(&self, len: usize) -> Result<Box<dyn RealBuffer>> {
        Ok(Box::new(CpuRealBuffer {
            values: vec![0.0; len],
        }))
    }

    fn upload_complex(
        &self,
        destination: &mut dyn ComplexBuffer,
        source: &[Complex64],
    ) -> Result<()> {
        let destination = cpu_complex_mut(destination)?;
        validate_transfer_length(destination.values.len(), source.len())?;
        destination.values.copy_from_slice(source);
        Ok(())
    }

    fn download_complex(
        &self,
        source: &dyn ComplexBuffer,
        destination: &mut [Complex64],
    ) -> Result<()> {
        let source = cpu_complex(source)?;
        validate_transfer_length(destination.len(), source.values.len())?;
        destination.copy_from_slice(&source.values);
        Ok(())
    }

    fn upload_real(&self, destination: &mut dyn RealBuffer, source: &[f64]) -> Result<()> {
        let destination = cpu_real_mut(destination)?;
        validate_transfer_length(destination.values.len(), source.len())?;
        destination.values.copy_from_slice(source);
        Ok(())
    }

    fn download_real(&self, source: &dyn RealBuffer, destination: &mut [f64]) -> Result<()> {
        let source = cpu_real(source)?;
        validate_transfer_length(destination.len(), source.values.len())?;
        destination.copy_from_slice(&source.values);
        Ok(())
    }

    fn fft2_resident(
        &self,
        values: &mut dyn ComplexBuffer,
        shape: (usize, usize),
        direction: FftDirection,
    ) -> Result<()> {
        let values = cpu_complex_mut(values)?;
        let mut column = vec![Complex64::default(); shape.0];
        self.fft2(&mut values.values, shape, direction, &mut column)
    }
}

fn cpu_complex(buffer: &dyn ComplexBuffer) -> Result<&CpuComplexBuffer> {
    buffer
        .as_any()
        .downcast_ref()
        .ok_or_else(|| Error::InvalidParameter {
            name: "complex buffer",
            reason: "buffer was not allocated by CpuBackend".into(),
        })
}

fn cpu_complex_mut(buffer: &mut dyn ComplexBuffer) -> Result<&mut CpuComplexBuffer> {
    buffer
        .as_any_mut()
        .downcast_mut()
        .ok_or_else(|| Error::InvalidParameter {
            name: "complex buffer",
            reason: "buffer was not allocated by CpuBackend".into(),
        })
}

fn cpu_real(buffer: &dyn RealBuffer) -> Result<&CpuRealBuffer> {
    buffer
        .as_any()
        .downcast_ref()
        .ok_or_else(|| Error::InvalidParameter {
            name: "real buffer",
            reason: "buffer was not allocated by CpuBackend".into(),
        })
}

fn cpu_real_mut(buffer: &mut dyn RealBuffer) -> Result<&mut CpuRealBuffer> {
    buffer
        .as_any_mut()
        .downcast_mut()
        .ok_or_else(|| Error::InvalidParameter {
            name: "real buffer",
            reason: "buffer was not allocated by CpuBackend".into(),
        })
}

fn validate_transfer_length(destination: usize, source: usize) -> Result<()> {
    if destination != source {
        return Err(Error::InvalidShape(format!(
            "buffer transfer length {source} does not match destination length {destination}"
        )));
    }
    Ok(())
}
