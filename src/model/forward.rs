use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_complex::Complex64;
use std::{sync::Arc, thread};

use crate::{
    Result,
    algorithms::objective::LossType,
    array_layout::{StandardView2, checked_len_2d},
    backend::{Backend, CpuBackend, FftDirection},
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
    fn new(model: &ImagePlaneModel) -> Result<Self> {
        let image_len = checked_len_2d(model.image_shape)?;
        Ok(Self {
            patch: vec![Complex64::default(); image_len],
            centered_exit: vec![Complex64::default(); image_len],
            field: vec![Complex64::default(); image_len],
            column: vec![
                Complex64::default();
                model.image_shape.0.max(model.reconstruction_shape.0)
            ],
        })
    }

    /// Borrows the most recently computed coherent detector field in row-major order.
    pub fn field(&self) -> &[Complex64] {
        &self.field
    }
}

/// Allocation-friendly CPU implementation of the image-plane FPM forward model.
///
/// Public spectrum and two-dimensional destination views must have standard
/// C-style row-major layout. The layout is validated once at each public
/// computational boundary and nonstandard views are rejected without copying.
/// Methods returning an [`Array2`] allocate their result; `_into` methods borrow
/// caller-provided workspace and destinations.
pub struct ForwardModel<'a> {
    model: &'a ImagePlaneModel,
    backend: Arc<dyn Backend>,
}

impl<'a> ForwardModel<'a> {
    /// Validates `model` and creates a CPU-backed evaluator borrowing it.
    pub fn new(model: &'a ImagePlaneModel) -> Result<Self> {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(
            model.image_shape,
            model.reconstruction_shape,
        )?);
        Self::with_backend(model, backend)
    }

    /// Validates `model` and creates an evaluator using the shared execution `backend`.
    pub fn with_backend(model: &'a ImagePlaneModel, backend: Arc<dyn Backend>) -> Result<Self> {
        model.validate()?;
        Ok(Self { model, backend })
    }

    /// Returns the compiled model borrowed by this evaluator.
    pub fn model(&self) -> &ImagePlaneModel {
        self.model
    }

    /// Allocates reusable buffers sized for the model's low-resolution grid.
    pub fn workspace(&self) -> Result<ForwardWorkspace> {
        ForwardWorkspace::new(self.model)
    }

    /// Allocates and returns one source's low-resolution complex Fourier patch.
    pub fn extract_patch(
        &self,
        object_spectrum: ArrayView2<'_, Complex64>,
        source: usize,
    ) -> Result<Array2<Complex64>> {
        let image_len = checked_len_2d(self.model.image_shape)?;
        let mut values = vec![Complex64::default(); image_len];
        self.model
            .extract_patch(object_spectrum, source, &mut values)?;
        Ok(Array2::from_shape_vec(self.model.image_shape, values)?)
    }

    /// Adds `scale * update` through the adjoint crop operator into `object_spectrum`.
    pub fn insert_patch_update(
        &self,
        object_spectrum: ArrayViewMut2<'_, Complex64>,
        source: usize,
        update: &[Complex64],
        scale: f64,
    ) -> Result<()> {
        self.model
            .insert_patch_adjoint(object_spectrum, source, update, scale)
    }

    /// Multiplies a row-major complex patch by the same-shaped sampled `pupil`.
    pub fn apply_pupil(
        &self,
        patch: &[Complex64],
        pupil: &Pupil,
        destination: &mut [Complex64],
    ) -> Result<()> {
        let len = checked_len_2d(self.model.image_shape)?;
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
        object_spectrum: ArrayView2<'_, Complex64>,
        pupil: &Pupil,
        source: usize,
    ) -> Result<Array2<Complex64>> {
        let object_spectrum = StandardView2::try_from(object_spectrum)?;
        let mut workspace = self.workspace()?;
        self.forward_source_field_standard_into(object_spectrum, pupil, source, &mut workspace)?;
        Ok(Array2::from_shape_vec(
            self.model.image_shape,
            workspace.field,
        )?)
    }

    pub(crate) fn forward_source_field_standard_into<'b>(
        &self,
        object_spectrum: StandardView2<'_, Complex64>,
        pupil: &Pupil,
        source: usize,
        workspace: &'b mut ForwardWorkspace,
    ) -> Result<&'b [Complex64]> {
        self.validate_workspace(workspace)?;
        self.model
            .extract_patch_standard(object_spectrum, source, &mut workspace.patch)?;
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
        object_spectrum: ArrayView2<'_, Complex64>,
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
            for value in &mut field {
                *value *= field_scale;
            }
            Ok(field)
        } else {
            self.forward_source_field(object_spectrum, pupil, frame)
        }
    }

    /// Allocates the predicted `(row, column)` intensity array for one acquisition frame.
    ///
    /// Coded frames are incoherent weighted sums of source intensities. Optional frame
    /// gain and optical background are applied to the result.
    pub fn forward_intensity(
        &self,
        object_spectrum: ArrayView2<'_, Complex64>,
        pupil: &Pupil,
        frame: usize,
    ) -> Result<Array2<f64>> {
        let object_spectrum = StandardView2::try_from(object_spectrum)?;
        let mut workspace = self.workspace()?;
        let mut values = vec![0.0; checked_len_2d(self.model.image_shape)?];
        self.forward_intensity_standard_into(
            object_spectrum,
            pupil,
            frame,
            &mut workspace,
            &mut values,
        )?;
        Ok(Array2::from_shape_vec(self.model.image_shape, values)?)
    }

    /// Writes one predicted frame to row-major `destination`, reusing `workspace`.
    pub fn forward_intensity_into(
        &self,
        object_spectrum: ArrayView2<'_, Complex64>,
        pupil: &Pupil,
        frame: usize,
        workspace: &mut ForwardWorkspace,
        destination: &mut [f64],
    ) -> Result<()> {
        self.forward_intensity_standard_into(
            StandardView2::try_from(object_spectrum)?,
            pupil,
            frame,
            workspace,
            destination,
        )
    }

    pub(crate) fn forward_intensity_standard_into(
        &self,
        object_spectrum: StandardView2<'_, Complex64>,
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
        let image_len = checked_len_2d(self.model.image_shape)?;
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
                let field = self.forward_source_field_standard_into(
                    object_spectrum,
                    pupil,
                    source,
                    workspace,
                )?;
                for (intensity, value) in destination.iter_mut().zip(field) {
                    *intensity += weight * value.norm_sqr();
                }
            }
        } else {
            let field =
                self.forward_source_field_standard_into(object_spectrum, pupil, frame, workspace)?;
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
        object_spectrum: ArrayView2<'_, Complex64>,
        pupil: &Pupil,
        destination: &mut [f64],
        worker_count: usize,
    ) -> Result<()> {
        let object_spectrum = StandardView2::try_from(object_spectrum)?;
        if worker_count == 0 {
            return Err(Error::InvalidParameter {
                name: "worker_count",
                reason: "must be greater than zero".into(),
            });
        }
        let image_len = checked_len_2d(self.model.image_shape)?;
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
            let mut workspace = self.workspace()?;
            for (frame, frame_destination) in destination.chunks_exact_mut(image_len).enumerate() {
                self.forward_intensity_standard_into(
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
                        let mut workspace = self.workspace()?;
                        for (local_frame, frame_destination) in
                            output.chunks_exact_mut(image_len).enumerate()
                        {
                            self.forward_intensity_standard_into(
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
                handle
                    .join()
                    .map_err(|_| Error::Numerical("parallel forward worker panicked".into()))??;
            }
            Ok(())
        })
    }

    /// Allocation-owning convenience wrapper for [`Self::forward_intensity_stack_into`].
    pub fn forward_intensity_stack(
        &self,
        object_spectrum: ArrayView2<'_, Complex64>,
        pupil: &Pupil,
        worker_count: usize,
    ) -> Result<Vec<f64>> {
        let image_len = checked_len_2d(self.model.image_shape)?;
        let stack_len = image_len
            .checked_mul(self.model.frame_count())
            .ok_or_else(|| Error::ShapeOverflow {
                shape: vec![
                    self.model.frame_count(),
                    self.model.image_shape.0,
                    self.model.image_shape.1,
                ],
            })?;
        let mut destination = vec![0.0; stack_len];
        self.forward_intensity_stack_into(object_spectrum, pupil, &mut destination, worker_count)?;
        Ok(destination)
    }

    /// Returns row-major `predicted - measured` intensity residuals for one frame.
    pub fn residual(
        &self,
        object_spectrum: ArrayView2<'_, Complex64>,
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
            .iter()
            .zip(measured)
            .map(|(&predicted, &measured)| predicted - measured)
            .collect())
    }

    /// Replaces coherent-field amplitude with measured amplitude while retaining phase.
    ///
    /// Frame gain and background are inverted before taking the square root of measured
    /// intensity. `epsilon` is a positive floor for dark predicted amplitudes.
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

    /// Evaluates `loss_type` between one predicted and measured intensity frame.
    pub fn frame_loss(
        &self,
        object_spectrum: ArrayView2<'_, Complex64>,
        pupil: &Pupil,
        frame: usize,
        measured: &[f64],
        loss_type: LossType,
    ) -> Result<f64> {
        let predicted = self.forward_intensity(object_spectrum, pupil, frame)?;
        let predicted = predicted.as_slice().ok_or_else(|| {
            Error::InvalidModel("internally generated intensity was not standard layout".into())
        })?;
        crate::algorithms::objective::loss(predicted, measured, loss_type)
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
        let image_len = checked_len_2d(self.model.image_shape)?;
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
