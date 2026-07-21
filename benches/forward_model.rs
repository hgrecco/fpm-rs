use std::{env, hint::black_box, time::Instant};

use fpm_rs::{
    Complex64, Error, Result,
    experiment::KVector,
    model::{CropIndices, ForwardModel, FourierCrop, ImagePlaneModel, Pupil, Sampling},
};
use ndarray::Array2;

fn main() -> Result<()> {
    let model = benchmark_model()?;
    let reconstruction_shape = model.reconstruction_shape();
    let reconstruction_len = reconstruction_shape
        .0
        .checked_mul(reconstruction_shape.1)
        .ok_or_else(|| Error::ShapeOverflow {
            shape: vec![reconstruction_shape.0, reconstruction_shape.1],
        })?;
    let spectrum = Array2::from_shape_vec(
        reconstruction_shape,
        (0..reconstruction_len)
            .map(|index| Complex64::new((index % 101) as f64 / 101.0, (index % 67) as f64 / 67.0))
            .collect(),
    )?;
    let forward = ForwardModel::new(&model)?;
    let iterations = env::var("FPM_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);

    let allocating_started = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        for frame in 0..model.frame_count() {
            let intensity = forward.forward_intensity(spectrum.view(), model.pupil(), frame)?;
            checksum += intensity
                .iter()
                .nth(frame % intensity.len())
                .copied()
                .unwrap_or(0.0);
        }
    }
    let allocating = allocating_started.elapsed();
    black_box(checksum);

    let mut workspace = forward.workspace()?;
    let image_shape = model.image_shape();
    let image_len =
        image_shape
            .0
            .checked_mul(image_shape.1)
            .ok_or_else(|| Error::ShapeOverflow {
                shape: vec![image_shape.0, image_shape.1],
            })?;
    let mut intensity = vec![0.0; image_len];
    let reused_started = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        for frame in 0..model.frame_count() {
            forward.forward_intensity_into(
                spectrum.view(),
                model.pupil(),
                frame,
                &mut workspace,
                &mut intensity,
            )?;
            checksum += intensity[frame % intensity.len()];
        }
    }
    let reused = reused_started.elapsed();
    black_box(checksum);

    let workers = std::thread::available_parallelism().map_or(1, |count| count.get());
    let stack_len = model
        .frame_count()
        .checked_mul(intensity.len())
        .ok_or_else(|| Error::ShapeOverflow {
            shape: vec![model.frame_count(), image_shape.0, image_shape.1],
        })?;
    let mut stack = vec![0.0; stack_len];
    let parallel_started = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        forward.forward_intensity_stack_into(
            spectrum.view(),
            model.pupil(),
            &mut stack,
            workers,
        )?;
        for frame in 0..model.frame_count() {
            checksum += stack[frame * intensity.len() + frame % intensity.len()];
        }
    }
    let parallel = parallel_started.elapsed();
    black_box(checksum);

    let evaluations = iterations * model.frame_count();
    println!("forward evaluations: {evaluations}");
    println!(
        "allocating: {:.3} us/frame",
        allocating.as_secs_f64() * 1e6 / evaluations as f64
    );
    println!(
        "workspace:  {:.3} us/frame",
        reused.as_secs_f64() * 1e6 / evaluations as f64
    );
    println!(
        "stack ({workers} workers): {:.3} us/frame",
        parallel.as_secs_f64() * 1e6 / evaluations as f64
    );
    Ok(())
}

fn benchmark_model() -> Result<ImagePlaneModel> {
    let image_shape = (64, 64);
    let reconstruction_shape = (128, 128);
    let sampling = Sampling::new(1.0, 0.5, 1.0, 1.0)?;
    let pupil = Pupil::new(
        Array2::from_elem(image_shape, Complex64::new(1.0, 0.0)),
        Array2::from_elem(image_shape, 1_u8),
    )?;
    let shifts: Vec<_> = (-2_isize..=2)
        .flat_map(|row| (-2_isize..=2).map(move |column| (8 * column, 8 * row)))
        .collect();
    let vectors = shifts
        .iter()
        .map(|&(column, row)| KVector::new(column as f64, row as f64))
        .collect();
    let crops = shifts
        .iter()
        .map(|&(column, row)| {
            FourierCrop::new(
                (32 + row) as usize,
                (32 + column) as usize,
                image_shape.0,
                image_shape.1,
            )
        })
        .collect();
    ImagePlaneModel::new(
        vectors,
        pupil,
        CropIndices::new(crops),
        sampling,
        image_shape,
        reconstruction_shape,
    )
}
