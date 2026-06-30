use num_complex::Complex64;
use std::{sync::Arc, thread};

use crate::{
    Array2, Result,
    backend::{Backend, CpuBackend, FftDirection},
    diagnostics::LossType,
    error::Error,
};

use super::{ImagePlaneModel, Pupil};

/// Reusable host buffers for allocation-free repeated forward evaluations.
#[derive(Clone, Debug)]
pub struct ForwardWorkspace {
    patch: Vec<Complex64>,
    centered_exit: Vec<Complex64>,
    field: Vec<Complex64>,
    column: Vec<Complex64>,
}

impl ForwardWorkspace {
    fn new(model: &ImagePlaneModel) -> Self {
        let image_len = model.image_shape.0 * model.image_shape.1;
        Self {
            patch: vec![Complex64::default(); image_len],
            centered_exit: vec![Complex64::default(); image_len],
            field: vec![Complex64::default(); image_len],
            column: vec![
                Complex64::default();
                model.image_shape.0.max(model.reconstruction_shape.0)
            ],
        }
    }

    pub fn field(&self) -> &[Complex64] {
        &self.field
    }
}

/// Allocation-friendly CPU implementation of the image-plane FPM forward model.
pub struct ForwardModel<'a> {
    model: &'a ImagePlaneModel,
    backend: Arc<dyn Backend>,
}

impl<'a> ForwardModel<'a> {
    pub fn new(model: &'a ImagePlaneModel) -> Result<Self> {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(
            model.image_shape,
            model.reconstruction_shape,
        )?);
        Self::with_backend(model, backend)
    }

    pub fn with_backend(model: &'a ImagePlaneModel, backend: Arc<dyn Backend>) -> Result<Self> {
        model.validate()?;
        Ok(Self { model, backend })
    }

    pub fn model(&self) -> &ImagePlaneModel {
        self.model
    }

    pub fn workspace(&self) -> ForwardWorkspace {
        ForwardWorkspace::new(self.model)
    }

    pub fn extract_patch(
        &self,
        object_spectrum: &Array2<Complex64>,
        source: usize,
    ) -> Result<Array2<Complex64>> {
        let mut values =
            vec![Complex64::default(); self.model.image_shape.0 * self.model.image_shape.1];
        self.model
            .extract_patch(object_spectrum, source, &mut values)?;
        Array2::from_vec(self.model.image_shape, values)
    }

    pub fn insert_patch_update(
        &self,
        object_spectrum: &mut Array2<Complex64>,
        source: usize,
        update: &[Complex64],
        scale: f64,
    ) -> Result<()> {
        self.model
            .insert_patch_adjoint(object_spectrum, source, update, scale)
    }

    pub fn apply_pupil(
        &self,
        patch: &[Complex64],
        pupil: &Pupil,
        destination: &mut [Complex64],
    ) -> Result<()> {
        let len = self.model.image_shape.0 * self.model.image_shape.1;
        if patch.len() != len || destination.len() != len || pupil.values.len() != len {
            return Err(Error::InvalidShape(
                "patch, pupil, and destination lengths must match image shape".into(),
            ));
        }
        for ((destination, &patch), &pupil) in destination
            .iter_mut()
            .zip(patch)
            .zip(pupil.values.as_slice())
        {
            *destination = patch * pupil;
        }
        Ok(())
    }

    /// Coherent low-resolution field for one illumination source.
    pub fn forward_source_field(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        source: usize,
    ) -> Result<Array2<Complex64>> {
        let mut workspace = self.workspace();
        self.forward_source_field_into(object_spectrum, pupil, source, &mut workspace)?;
        Array2::from_vec(self.model.image_shape, workspace.field)
    }

    pub fn forward_source_field_into<'b>(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        source: usize,
        workspace: &'b mut ForwardWorkspace,
    ) -> Result<&'b [Complex64]> {
        self.validate_workspace(workspace)?;
        self.model
            .extract_patch(object_spectrum, source, &mut workspace.patch)?;
        self.apply_pupil(&workspace.patch, pupil, &mut workspace.centered_exit)?;
        ifftshift_copy(
            &workspace.centered_exit,
            &mut workspace.field,
            self.model.image_shape,
        );
        self.backend.fft2(
            &mut workspace.field,
            self.model.image_shape,
            FftDirection::Inverse,
            &mut workspace.column,
        )?;
        Ok(&workspace.field)
    }

    /// Coherent field for a non-multiplexed frame. A coded frame with exactly
    /// one source is represented by a field scaled by the square-root weight.
    pub fn forward_field(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        frame: usize,
    ) -> Result<Array2<Complex64>> {
        if frame >= self.model.frame_count() {
            return Err(Error::FrameOutOfRange {
                index: frame,
                frames: self.model.frame_count(),
            });
        }
        if let Some(matrix) = &self.model.multiplexing_matrix {
            let row = &matrix[frame];
            if row.len() != 1 {
                return Err(Error::Unsupported(
                    "a multiplexed frame has no single coherent field".into(),
                ));
            }
            let (source, weight) = row[0];
            let mut field = self.forward_source_field(object_spectrum, pupil, source)?;
            let field_scale = weight.sqrt();
            for value in field.as_mut_slice() {
                *value *= field_scale;
            }
            Ok(field)
        } else {
            self.forward_source_field(object_spectrum, pupil, frame)
        }
    }

    pub fn forward_intensity(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        frame: usize,
    ) -> Result<Array2<f64>> {
        let mut workspace = self.workspace();
        let mut values = vec![0.0; self.model.image_shape.0 * self.model.image_shape.1];
        self.forward_intensity_into(object_spectrum, pupil, frame, &mut workspace, &mut values)?;
        Array2::from_vec(self.model.image_shape, values)
    }

    pub fn forward_intensity_into(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        frame: usize,
        workspace: &mut ForwardWorkspace,
        destination: &mut [f64],
    ) -> Result<()> {
        if frame >= self.model.frame_count() {
            return Err(Error::FrameOutOfRange {
                index: frame,
                frames: self.model.frame_count(),
            });
        }
        let gain = self.frame_gain(frame)?;
        let image_len = self.model.image_shape.0 * self.model.image_shape.1;
        if destination.len() != image_len {
            return Err(Error::LengthMismatch {
                actual: destination.len(),
                expected: image_len,
                shape: self.model.image_shape,
            });
        }
        destination.fill(0.0);
        if let Some(matrix) = &self.model.multiplexing_matrix {
            for &(source, weight) in &matrix[frame] {
                let field =
                    self.forward_source_field_into(object_spectrum, pupil, source, workspace)?;
                for (intensity, value) in destination.iter_mut().zip(field) {
                    *intensity += weight * value.norm_sqr();
                }
            }
        } else {
            let field = self.forward_source_field_into(object_spectrum, pupil, frame, workspace)?;
            for (intensity, value) in destination.iter_mut().zip(field) {
                *intensity = value.norm_sqr();
            }
        }
        for (pixel, value) in destination.iter_mut().enumerate() {
            *value = gain * *value + self.background(frame, pixel, image_len);
        }
        Ok(())
    }

    /// Predicts every measured frame into a contiguous
    /// `[frame][row][column]` destination using independent worker scratch.
    ///
    /// Frame order and floating-point results within each frame are unchanged
    /// by `worker_count`. Values larger than the frame count are capped; zero is
    /// rejected. A worker count of one executes directly without spawning.
    pub fn forward_intensity_stack_into(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        destination: &mut [f64],
        worker_count: usize,
    ) -> Result<()> {
        if worker_count == 0 {
            return Err(Error::InvalidParameter {
                name: "worker_count",
                reason: "must be greater than zero".into(),
            });
        }
        let image_len = self.model.image_shape.0 * self.model.image_shape.1;
        let expected = image_len
            .checked_mul(self.model.frame_count())
            .ok_or_else(|| Error::InvalidShape("forward stack length overflows".into()))?;
        if destination.len() != expected {
            return Err(Error::LengthMismatch {
                actual: destination.len(),
                expected,
                shape: (self.model.frame_count(), image_len),
            });
        }
        let workers = worker_count.min(self.model.frame_count());
        if workers == 1 {
            let mut workspace = self.workspace();
            for (frame, frame_destination) in destination.chunks_exact_mut(image_len).enumerate() {
                self.forward_intensity_into(
                    object_spectrum,
                    pupil,
                    frame,
                    &mut workspace,
                    frame_destination,
                )?;
            }
            return Ok(());
        }

        let frames_per_worker = self.model.frame_count().div_ceil(workers);
        let values_per_worker = frames_per_worker * image_len;
        thread::scope(|scope| {
            let handles: Vec<_> = destination
                .chunks_mut(values_per_worker)
                .enumerate()
                .map(|(chunk_index, output)| {
                    let first_frame = chunk_index * frames_per_worker;
                    scope.spawn(move || -> Result<()> {
                        let mut workspace = self.workspace();
                        for (local_frame, frame_destination) in
                            output.chunks_exact_mut(image_len).enumerate()
                        {
                            self.forward_intensity_into(
                                object_spectrum,
                                pupil,
                                first_frame + local_frame,
                                &mut workspace,
                                frame_destination,
                            )?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for handle in handles {
                handle.join().map_err(|_| {
                    Error::Numerical("parallel forward worker panicked".into())
                })??;
            }
            Ok(())
        })
    }

    /// Allocation-owning convenience wrapper for [`Self::forward_intensity_stack_into`].
    pub fn forward_intensity_stack(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        worker_count: usize,
    ) -> Result<Vec<f64>> {
        let image_len = self.model.image_shape.0 * self.model.image_shape.1;
        let mut destination = vec![0.0; image_len * self.model.frame_count()];
        self.forward_intensity_stack_into(object_spectrum, pupil, &mut destination, worker_count)?;
        Ok(destination)
    }

    pub fn residual(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        frame: usize,
        measured: &[f64],
    ) -> Result<Vec<f64>> {
        let predicted = self.forward_intensity(object_spectrum, pupil, frame)?;
        if measured.len() != predicted.len() {
            return Err(Error::InvalidShape(
                "measurement does not match predicted frame".into(),
            ));
        }
        Ok(predicted
            .as_slice()
            .iter()
            .zip(measured)
            .map(|(&predicted, &measured)| predicted - measured)
            .collect())
    }

    pub fn amplitude_projection(
        &self,
        field: &[Complex64],
        measured_intensity: &[f64],
        frame: usize,
        epsilon: f64,
    ) -> Result<Vec<Complex64>> {
        if field.len() != measured_intensity.len() {
            return Err(Error::InvalidShape(
                "field and measured image lengths differ".into(),
            ));
        }
        if frame >= self.model.frame_count() {
            return Err(Error::FrameOutOfRange {
                index: frame,
                frames: self.model.frame_count(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                reason: "must be finite and positive".into(),
            });
        }
        if self
            .model
            .multiplexing_matrix
            .as_ref()
            .is_some_and(|matrix| matrix[frame].len() != 1)
        {
            return Err(Error::Unsupported(
                "amplitude projection is not defined for an incoherent multiplexed field".into(),
            ));
        }
        let gain = self.frame_gain(frame)?;
        let image_len = field.len();
        Ok(field
            .iter()
            .zip(measured_intensity)
            .enumerate()
            .map(|(pixel, (&value, &measurement))| {
                let corrected =
                    ((measurement - self.background(frame, pixel, image_len)) / gain).max(0.0);
                let target = corrected.sqrt();
                let magnitude = value.norm();
                if magnitude > epsilon {
                    value * (target / magnitude)
                } else {
                    Complex64::new(target, 0.0)
                }
            })
            .collect())
    }

    pub fn frame_loss(
        &self,
        object_spectrum: &Array2<Complex64>,
        pupil: &Pupil,
        frame: usize,
        measured: &[f64],
        loss_type: LossType,
    ) -> Result<f64> {
        let predicted = self.forward_intensity(object_spectrum, pupil, frame)?;
        crate::diagnostics::loss(predicted.as_slice(), measured, loss_type)
    }

    fn frame_gain(&self, frame: usize) -> Result<f64> {
        let gain = self
            .model
            .frame_gains
            .as_ref()
            .map_or(1.0, |gains| gains[frame]);
        if !gain.is_finite() || gain <= 0.0 {
            return Err(Error::InvalidModel(format!(
                "frame {frame} has invalid gain {gain}"
            )));
        }
        Ok(gain)
    }

    fn background(&self, frame: usize, pixel: usize, image_len: usize) -> f64 {
        self.model.background.as_ref().map_or(0.0, |values| {
            values[if values.len() == image_len {
                pixel
            } else {
                frame * image_len + pixel
            }]
        })
    }

    fn validate_workspace(&self, workspace: &ForwardWorkspace) -> Result<()> {
        let image_len = self.model.image_shape.0 * self.model.image_shape.1;
        if workspace.patch.len() != image_len
            || workspace.centered_exit.len() != image_len
            || workspace.field.len() != image_len
            || workspace.column.len() < self.model.image_shape.0
        {
            return Err(Error::InvalidShape(
                "forward workspace does not match the model image shape".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn fftshift_copy(
    source: &[Complex64],
    destination: &mut [Complex64],
    shape: (usize, usize),
) {
    shift_copy(source, destination, shape, shape.0 / 2, shape.1 / 2);
}

pub(crate) fn ifftshift_copy(
    source: &[Complex64],
    destination: &mut [Complex64],
    shape: (usize, usize),
) {
    shift_copy(
        source,
        destination,
        shape,
        shape.0.div_ceil(2),
        shape.1.div_ceil(2),
    );
}

fn shift_copy(
    source: &[Complex64],
    destination: &mut [Complex64],
    shape: (usize, usize),
    row_shift: usize,
    column_shift: usize,
) {
    debug_assert_eq!(source.len(), shape.0 * shape.1);
    debug_assert_eq!(destination.len(), source.len());
    for row in 0..shape.0 {
        for column in 0..shape.1 {
            let destination_row = (row + row_shift) % shape.0;
            let destination_column = (column + column_shift) % shape.1;
            destination[destination_row * shape.1 + destination_column] =
                source[row * shape.1 + column];
        }
    }
}
