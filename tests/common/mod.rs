#![allow(dead_code)]

use fpm_rs::{
    Complex64, Result,
    backend::{Backend, CpuBackend, FftDirection},
    experiment::KVector,
    model::{CropIndices, FourierCrop, ImagePlaneModel, Pupil, Sampling},
};
use ndarray::Array2;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub struct CountingBackend {
    inner: CpuBackend,
    calls: Arc<AtomicUsize>,
}

impl CountingBackend {
    pub fn new(
        low_shape: (usize, usize),
        high_shape: (usize, usize),
    ) -> Result<(Arc<Self>, Arc<AtomicUsize>)> {
        let calls = Arc::new(AtomicUsize::new(0));
        Ok((
            Arc::new(Self {
                inner: CpuBackend::new(low_shape, high_shape)?,
                calls: calls.clone(),
            }),
            calls,
        ))
    }
}

impl Backend for CountingBackend {
    fn fft2(
        &self,
        values: &mut [Complex64],
        shape: (usize, usize),
        direction: FftDirection,
        column_scratch: &mut [Complex64],
    ) -> Result<()> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.fft2(values, shape, direction, column_scratch)
    }
}

pub fn direct_model() -> Result<ImagePlaneModel> {
    let image_shape = (8, 8);
    let reconstruction_shape = (16, 16);
    let sampling = Sampling::new(1.0, 0.5, 1.0, 1.0)?;
    let pupil = Pupil::new(
        Array2::from_elem(image_shape, Complex64::new(1.0, 0.0)),
        Array2::from_elem(image_shape, 1_u8),
    )?;
    let shifts = [(-3, 0), (0, -3), (0, 0), (0, 3), (3, 0)];
    let k_vectors = shifts
        .iter()
        .map(|&(column, row)| KVector::new(column as f64, row as f64))
        .collect();
    let crops = shifts
        .iter()
        .map(|&(column, row)| {
            FourierCrop::new(
                (4_i32 + row) as usize,
                (4_i32 + column) as usize,
                image_shape.0,
                image_shape.1,
            )
        })
        .collect();
    ImagePlaneModel::new(
        k_vectors,
        pupil,
        CropIndices::new(crops),
        sampling,
        image_shape,
        reconstruction_shape,
    )
}
