use std::{
    alloc::{GlobalAlloc, Layout, System},
    env,
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use fpm_rs::{
    Array2, Complex64, Error, Result,
    algorithms::{GradientDescent, ReconstructionAlgorithm},
    experiment::KVector,
    measurements::MeasurementStack,
    model::{CropIndices, FourierCrop, ImagePlaneModel, Pupil, Sampling},
    reconstruction::{Batch, ReconstructionProblem, ReconstructionState},
    simulation::{Simulator, SyntheticObject},
};

struct TrackingAllocator;

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates the allocation request unchanged to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates the allocation request unchanged to the system allocator.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_deallocation(layout.size());
        // SAFETY: `pointer` and `layout` are the pair supplied by the caller.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Delegates the reallocation request unchanged to the system allocator.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            if new_size >= layout.size() {
                record_allocation(new_size - layout.size());
            } else {
                record_deallocation(layout.size() - new_size);
            }
        }
        replacement
    }
}

fn record_allocation(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn record_deallocation(bytes: usize) {
    let _ = LIVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
        Some(live.saturating_sub(bytes))
    });
}

fn main() -> Result<()> {
    let low_size = environment_usize("FPM_GRADIENT_BENCH_LOW_SIZE", 32);
    let high_size = environment_usize("FPM_GRADIENT_BENCH_HIGH_SIZE", low_size * 2);
    let iterations = environment_usize(
        "FPM_GRADIENT_BENCH_ITERATIONS",
        if cfg!(debug_assertions) { 1 } else { 20 },
    );
    let available_workers = std::thread::available_parallelism().map_or(1, |count| count.get());
    let maximum_workers =
        environment_usize("FPM_GRADIENT_BENCH_MAX_WORKERS", available_workers).max(1);

    let ordinary = benchmark_problem(benchmark_model(low_size, high_size)?)?;
    let multiplexed_model = benchmark_model(low_size, high_size)?.with_multiplexing(vec![
        vec![(0, 0.6), (1, 0.4)],
        vec![(1, 0.5), (2, 0.5)],
        vec![(3, 0.7), (4, 0.3)],
        vec![(4, 0.4), (5, 0.6)],
        vec![(6, 0.5), (7, 0.5)],
        vec![(8, 0.6), (0, 0.4)],
    ])?;
    let multiplexed = benchmark_problem(multiplexed_model)?;

    println!(
        "gradient benchmark: low={low_size}x{low_size}, high={high_size}x{high_size}, samples={iterations}"
    );
    benchmark_case(
        "ordinary object",
        &ordinary,
        false,
        false,
        maximum_workers,
        iterations,
    )?;
    benchmark_case(
        "multiplexed object",
        &multiplexed,
        false,
        false,
        maximum_workers,
        iterations,
    )?;
    benchmark_case(
        "multiplexed object+pupil",
        &multiplexed,
        true,
        false,
        maximum_workers,
        iterations,
    )?;
    benchmark_case(
        "multiplexed object+illumination",
        &multiplexed,
        false,
        true,
        maximum_workers,
        iterations,
    )?;
    Ok(())
}

fn benchmark_case(
    name: &str,
    problem: &ReconstructionProblem<MeasurementStack>,
    recover_pupil: bool,
    recover_illumination: bool,
    maximum_workers: usize,
    iterations: usize,
) -> Result<()> {
    let initial = ReconstructionState::initialize(problem)?;
    let batch = Batch::new((0..problem.model.frame_count()).collect(), 0);
    let workers = worker_counts(maximum_workers.min(batch.indices.len()));
    let mut serial_seconds = None;
    println!("\n{name} ({} measured frames)", batch.indices.len());
    println!("workers  ms/step  speedup  peak incremental heap MiB");
    for worker_count in workers {
        let mut algorithm = GradientDescent::default()
            .object_step(0.2)
            .recover_pupil(recover_pupil)
            .pupil_step(0.02)
            .recover_illumination(recover_illumination)
            .illumination_step(0.1)
            .parallel_workers(worker_count);
        let mut warmup = initial.clone();
        algorithm.step(problem, &mut warmup, &batch, 0)?;
        let mut elapsed = Duration::ZERO;
        let mut maximum_peak = 0;
        let mut checksum = 0.0;
        for sample in 0..iterations {
            let mut state = initial.clone();
            let baseline = LIVE_BYTES.load(Ordering::Relaxed);
            PEAK_BYTES.store(baseline, Ordering::Relaxed);
            let started = Instant::now();
            let diagnostics = algorithm.step(problem, &mut state, &batch, sample)?;
            elapsed += started.elapsed();
            let peak = PEAK_BYTES.load(Ordering::Relaxed).saturating_sub(baseline);
            maximum_peak = maximum_peak.max(peak);
            checksum += diagnostics.mean_loss().unwrap_or_default()
                + state.object_spectrum.as_slice()[sample % state.object_spectrum.len()].norm();
            black_box(&state);
        }
        black_box(checksum);
        let seconds = elapsed.as_secs_f64() / iterations as f64;
        let serial = *serial_seconds.get_or_insert(seconds);
        println!(
            "{worker_count:>7}  {:>7.3}  {:>7.2}x  {:>25.3}",
            seconds * 1e3,
            serial / seconds,
            maximum_peak as f64 / (1024.0 * 1024.0),
        );
    }
    Ok(())
}

fn benchmark_problem(model: ImagePlaneModel) -> Result<ReconstructionProblem<MeasurementStack>> {
    let object = SyntheticObject::mixed_test_pattern(model.reconstruction_shape)?;
    let simulation = Simulator::ideal(model).object(object).simulate()?;
    ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)
}

fn benchmark_model(low_size: usize, high_size: usize) -> Result<ImagePlaneModel> {
    if low_size < 8 || high_size < low_size + low_size / 2 {
        return Err(Error::InvalidParameter {
            name: "benchmark size",
            reason: "high size must be at least 1.5 times a low size of at least 8".into(),
        });
    }
    let image_shape = (low_size, low_size);
    let reconstruction_shape = (high_size, high_size);
    let sampling = Sampling::new(1.0, 0.5, 1.0, 1.0)?;
    let pupil = Pupil::new(
        Array2::filled(image_shape, Complex64::new(1.0, 0.0))?,
        vec![true; low_size * low_size],
    )?;
    let distance = (low_size / 8).max(1) as isize;
    let origin = ((high_size - low_size) / 2) as isize;
    let shifts: Vec<_> = [-distance, 0, distance]
        .into_iter()
        .flat_map(|row| {
            [-distance, 0, distance]
                .into_iter()
                .map(move |column| (row, column))
        })
        .collect();
    let vectors = shifts
        .iter()
        .map(|&(row, column)| KVector::new(column as f64, row as f64))
        .collect();
    let crops = shifts
        .iter()
        .map(|&(row, column)| {
            FourierCrop::new(
                (origin + row) as usize,
                (origin + column) as usize,
                low_size,
                low_size,
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

fn worker_counts(maximum: usize) -> Vec<usize> {
    let mut workers = vec![1];
    let mut candidate = 2;
    while candidate < maximum {
        workers.push(candidate);
        candidate *= 2;
    }
    if maximum > 1 && workers.last().copied() != Some(maximum) {
        workers.push(maximum);
    }
    workers
}

fn environment_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}
