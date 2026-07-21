mod common;

use approx::assert_abs_diff_eq;
use fpm_rs::{
    Complex64, Error,
    backend::{Backend, CpuBackend, FftDirection, MemoryLocation},
    model::{ForwardModel, FourierCrop, FourierOffset},
};
use ndarray::{Array2, s};
use std::sync::atomic::Ordering;

#[test]
fn fft_forward_inverse_is_consistent() {
    let shape = (7, 9);
    let backend = CpuBackend::new(shape, shape).unwrap();
    let original: Vec<_> = (0..shape.0 * shape.1)
        .map(|index| Complex64::new(index as f64 / 13.0, (index % 5) as f64))
        .collect();
    let mut values = original.clone();
    let mut column = vec![Complex64::default(); shape.0];
    backend
        .fft2(&mut values, shape, FftDirection::Forward, &mut column)
        .unwrap();
    backend
        .fft2(&mut values, shape, FftDirection::Inverse, &mut column)
        .unwrap();
    for (actual, expected) in values.iter().zip(original) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-10);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-10);
    }
}

#[test]
fn cpu_backend_exposes_typed_resident_buffer_capabilities() {
    let shape = (7, 9);
    let backend = CpuBackend::new(shape, shape).unwrap();
    let capabilities = backend.capabilities();
    assert_eq!(capabilities.preferred_memory, MemoryLocation::Host);
    assert!(capabilities.resident_buffers);
    let resident = backend.resident_backend().unwrap();

    let original: Vec<_> = (0..shape.0 * shape.1)
        .map(|index| Complex64::new(index as f64 / 17.0, (index % 3) as f64))
        .collect();
    let mut complex = resident.allocate_complex(original.len()).unwrap();
    assert_eq!(complex.location(), MemoryLocation::Host);
    resident
        .upload_complex(complex.as_mut(), &original)
        .unwrap();
    resident
        .fft2_resident(complex.as_mut(), shape, FftDirection::Forward)
        .unwrap();
    resident
        .fft2_resident(complex.as_mut(), shape, FftDirection::Inverse)
        .unwrap();
    let mut recovered = vec![Complex64::default(); original.len()];
    resident
        .download_complex(complex.as_ref(), &mut recovered)
        .unwrap();
    for (&actual, &expected) in recovered.iter().zip(&original) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-10);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-10);
    }

    let real_values = vec![1.0, 2.0, 3.0];
    let mut real = resident.allocate_real(real_values.len()).unwrap();
    resident.upload_real(real.as_mut(), &real_values).unwrap();
    let mut real_copy = vec![0.0; real_values.len()];
    resident
        .download_real(real.as_ref(), &mut real_copy)
        .unwrap();
    assert_eq!(real_copy, real_values);
}

#[test]
fn constant_object_has_constant_central_frame() {
    let model = common::direct_model().unwrap();
    let backend = CpuBackend::new(model.image_shape(), model.reconstruction_shape()).unwrap();
    let mut spatial = vec![Complex64::new(2.0, 0.0); 16 * 16];
    let mut column = vec![Complex64::default(); 16];
    backend
        .fft2(
            &mut spatial,
            model.reconstruction_shape(),
            FftDirection::Forward,
            &mut column,
        )
        .unwrap();
    let mut centered = vec![Complex64::default(); spatial.len()];
    for row in 0..16 {
        for column in 0..16 {
            centered[((row + 8) % 16) * 16 + (column + 8) % 16] = spatial[row * 16 + column];
        }
    }
    let spectrum = Array2::from_shape_vec((16, 16), centered).unwrap();
    let intensity = ForwardModel::new(&model)
        .unwrap()
        .forward_intensity(spectrum.view(), model.pupil(), 2)
        .unwrap();
    assert!(intensity.iter().all(|value| (*value - 4.0).abs() < 1e-10));
}

#[test]
fn patch_extraction_and_update_are_consistent() {
    let model = common::direct_model().unwrap();
    let forward = ForwardModel::new(&model).unwrap();
    let mut spectrum = Array2::from_elem((16, 16), Complex64::new(0.0, 0.0));
    let update = vec![Complex64::new(1.0, -0.5); 64];
    forward
        .insert_patch_update(spectrum.view_mut(), 2, &update, 1.0)
        .unwrap();
    let patch = forward.extract_patch(spectrum.view(), 2).unwrap();
    assert_eq!(patch.iter().copied().collect::<Vec<_>>(), update);
}

#[test]
fn subpixel_patch_uses_bilinear_fourier_sampling() {
    let mut offsets = vec![FourierOffset::default(); 5];
    offsets[2] = FourierOffset::new(0.25, 0.5);
    let model = common::direct_model()
        .unwrap()
        .with_subpixel_offsets(offsets)
        .unwrap();
    let values = (0..16)
        .flat_map(|row| {
            (0..16).map(move |column| {
                Complex64::new(
                    1.0 + 2.0 * row as f64 + 3.0 * column as f64,
                    -row as f64 + 0.5 * column as f64,
                )
            })
        })
        .collect();
    let spectrum = Array2::from_shape_vec((16, 16), values).unwrap();
    let patch = ForwardModel::new(&model)
        .unwrap()
        .extract_patch(spectrum.view(), 2)
        .unwrap();

    for row in 0..8 {
        for column in 0..8 {
            let sampled_row = 4.25 + row as f64;
            let sampled_column = 4.5 + column as f64;
            let expected = Complex64::new(
                1.0 + 2.0 * sampled_row + 3.0 * sampled_column,
                -sampled_row + 0.5 * sampled_column,
            );
            assert_abs_diff_eq!(patch[(row, column)].re, expected.re, epsilon = 1e-12);
            assert_abs_diff_eq!(patch[(row, column)].im, expected.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn bilinear_subpixel_sampling_error_increases_with_fourier_grid_bandwidth() {
    let shape = (32, 32);
    let crop = FourierCrop::new(10, 10, 8, 8);
    let offset = FourierOffset::new(0.5, 0.0);
    let low_band_error = subpixel_sinusoid_relative_rms_error(shape, crop, offset, 1);
    let mid_band_error = subpixel_sinusoid_relative_rms_error(shape, crop, offset, 4);
    let high_band_error = subpixel_sinusoid_relative_rms_error(shape, crop, offset, 8);

    // For a half-pixel shift of exp(i 2 pi nu r / H), linear interpolation
    // attenuates the field by cos(pi nu / H). The continuous sinusoid below is
    // the bandlimited-shift reference, independent of crop implementation.
    for (cycles, error) in [
        (1, low_band_error),
        (4, mid_band_error),
        (8, high_band_error),
    ] {
        let expected = 1.0 - (std::f64::consts::PI * cycles as f64 / shape.0 as f64).cos();
        assert_abs_diff_eq!(error, expected, epsilon = 1e-12);
    }
    assert!(low_band_error < 0.005);
    assert!(mid_band_error > low_band_error);
    assert!(high_band_error > mid_band_error);
    assert!(high_band_error > 0.29);
}

fn subpixel_sinusoid_relative_rms_error(
    shape: (usize, usize),
    crop: FourierCrop,
    offset: FourierOffset,
    cycles: usize,
) -> f64 {
    let spectrum = Array2::from_shape_vec(
        shape,
        (0..shape.0 * shape.1)
            .map(|index| {
                let row = index / shape.1;
                Complex64::from_polar(
                    1.0,
                    std::f64::consts::TAU * cycles as f64 * row as f64 / shape.0 as f64,
                )
            })
            .collect(),
    )
    .unwrap();
    let mut sampled = vec![Complex64::default(); crop.height * crop.width];
    crop.extract_subpixel(spectrum.view(), &mut sampled, offset)
        .unwrap();
    let squared_error: f64 = sampled
        .iter()
        .enumerate()
        .map(|(index, &sampled)| {
            let row = index / crop.width;
            let reference_row = crop.start_row as f64 + row as f64 + offset.row;
            let reference = Complex64::from_polar(
                1.0,
                std::f64::consts::TAU * cycles as f64 * reference_row / shape.0 as f64,
            );
            (sampled - reference).norm_sqr()
        })
        .sum();
    (squared_error / sampled.len() as f64).sqrt()
}

#[test]
fn subpixel_patch_insertion_is_the_exact_adjoint() {
    let mut offsets = vec![FourierOffset::default(); 5];
    offsets[2] = FourierOffset::new(-0.3, 0.4);
    let model = common::direct_model()
        .unwrap()
        .with_subpixel_offsets(offsets)
        .unwrap();
    let spectrum = Array2::from_shape_vec(
        (16, 16),
        (0..256)
            .map(|index| {
                Complex64::new((index % 13) as f64 / 7.0, (index % 17) as f64 / 11.0 - 0.5)
            })
            .collect(),
    )
    .unwrap();
    let update: Vec<_> = (0..64)
        .map(|index| Complex64::new((index % 7) as f64 / 5.0 - 0.4, (index % 11) as f64 / 9.0))
        .collect();
    let extracted = ForwardModel::new(&model)
        .unwrap()
        .extract_patch(spectrum.view(), 2)
        .unwrap();
    let mut adjoint = Array2::from_elem((16, 16), Complex64::default());
    model
        .insert_patch_adjoint(adjoint.view_mut(), 2, &update, 1.0)
        .unwrap();
    let left: Complex64 = extracted
        .iter()
        .zip(&update)
        .map(|(&sample, &value)| sample.conj() * value)
        .sum();
    let right: Complex64 = spectrum
        .iter()
        .zip(adjoint.iter())
        .map(|(&value, &backprojected)| value.conj() * backprojected)
        .sum();
    assert_abs_diff_eq!(left.re, right.re, epsilon = 1e-12);
    assert_abs_diff_eq!(left.im, right.im, epsilon = 1e-12);
}

#[test]
fn forward_model_uses_injected_backend() {
    let model = common::direct_model().unwrap();
    let (backend, calls) =
        common::CountingBackend::new(model.image_shape(), model.reconstruction_shape()).unwrap();
    let forward = ForwardModel::with_backend(&model, backend).unwrap();
    let spectrum = Array2::from_elem(model.reconstruction_shape(), Complex64::default());
    forward
        .forward_intensity(spectrum.view(), model.pupil(), 0)
        .unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn reusable_forward_workspace_matches_allocating_api() {
    for model in [
        common::direct_model().unwrap(),
        common::direct_model()
            .unwrap()
            .with_multiplexing(vec![
                vec![(0, 0.7), (1, 0.3)],
                vec![(2, 0.4), (3, 0.6)],
                vec![(4, 1.0)],
            ])
            .unwrap(),
    ] {
        let spectrum = Array2::from_shape_vec(
            model.reconstruction_shape(),
            (0..model.reconstruction_shape().0 * model.reconstruction_shape().1)
                .map(|index| Complex64::new((index % 17) as f64 / 13.0, (index % 11) as f64 / 9.0))
                .collect(),
        )
        .unwrap();
        let forward = ForwardModel::new(&model).unwrap();
        let mut workspace = forward.workspace().unwrap();
        let mut reused = vec![-1.0; model.image_shape().0 * model.image_shape().1];
        let mut serial_stack = Vec::new();
        for frame in 0..model.frame_count() {
            let allocated = forward
                .forward_intensity(spectrum.view(), model.pupil(), frame)
                .unwrap();
            serial_stack.extend(allocated.iter().copied());
            forward
                .forward_intensity_into(
                    spectrum.view(),
                    model.pupil(),
                    frame,
                    &mut workspace,
                    &mut reused,
                )
                .unwrap();
            for (&allocated, &reused) in allocated.iter().zip(&reused) {
                assert_abs_diff_eq!(allocated, reused, epsilon = 1e-14);
            }
        }
        let parallel_stack = forward
            .forward_intensity_stack(spectrum.view(), model.pupil(), 3)
            .unwrap();
        assert_eq!(parallel_stack, serial_stack);

        let mut single_worker_stack = vec![f64::NAN; serial_stack.len()];
        forward
            .forward_intensity_stack_into(
                spectrum.view(),
                model.pupil(),
                &mut single_worker_stack,
                1,
            )
            .unwrap();
        assert_eq!(single_worker_stack, serial_stack);
    }
}

#[test]
fn forward_stack_validates_worker_count_and_destination_length() {
    let model = common::direct_model().unwrap();
    let forward = ForwardModel::new(&model).unwrap();
    let spectrum = Array2::from_elem(model.reconstruction_shape(), Complex64::default());
    let mut destination = vec![0.0; model.frame_count() * model.pupil().values().len()];
    assert!(matches!(
        forward.forward_intensity_stack_into(spectrum.view(), model.pupil(), &mut destination, 0),
        Err(Error::InvalidParameter {
            name: "worker_count",
            ..
        })
    ));
    let short_length = destination.len() - 1;
    assert!(matches!(
        forward.forward_intensity_stack_into(
            spectrum.view(),
            model.pupil(),
            &mut destination[..short_length],
            2
        ),
        Err(Error::LengthMismatch { .. })
    ));
}

#[test]
fn forward_model_rejects_nonstandard_spectra_without_copying() {
    let model = common::direct_model().unwrap();
    let forward = ForwardModel::new(&model).unwrap();
    let spectrum = Array2::from_elem(model.reconstruction_shape(), Complex64::default());

    assert!(matches!(
        forward.forward_intensity(spectrum.t(), model.pupil(), 0),
        Err(Error::NonStandardLayout { .. })
    ));
    assert!(matches!(
        forward.forward_intensity(spectrum.slice(s![.., ..;2]), model.pupil(), 0),
        Err(Error::NonStandardLayout { .. })
    ));
}
