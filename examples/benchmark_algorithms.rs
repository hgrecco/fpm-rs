use std::{env, path::Path};

use fpm_rs::{
    Array2, Complex64, Error, Result,
    algorithms::{
        Admm, AlternatingProjection, Epry, Fpie, GradientDescent, ReconstructionAlgorithm,
    },
    benchmark::{
        BENCHMARK_PROFILES, BenchmarkProfile, BenchmarkRecord, CPU_BENCHMARK_PROFILE,
        SMOKE_BENCHMARK_PROFILE, annotate_benchmark_profile, benchmark_profile, run_benchmark_case,
        save_benchmark_outputs, write_benchmark_csv, write_benchmark_json,
    },
    measurements::MeasurementRead,
    model::ImagePlaneModel,
    reconstruction::ReconstructionProblem,
    simulation::presets::{NOISELESS_MIXED_PRESET, noiseless_mixed_fpm},
};

fn main() -> Result<()> {
    let profile_name = env::args()
        .nth(1)
        .unwrap_or_else(|| SMOKE_BENCHMARK_PROFILE.into());
    let profile = benchmark_profile(&profile_name).ok_or_else(|| unknown_profile(&profile_name))?;
    let iterations = match profile.name {
        SMOKE_BENCHMARK_PROFILE => 2,
        CPU_BENCHMARK_PROFILE => 20,
        _ => unreachable!("benchmark profiles are exhaustively matched"),
    };

    let simulation = noiseless_mixed_fpm(2026)?;
    let truth = simulation.ground_truth_object;
    let true_model = simulation.true_model;
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?
            .named(NOISELESS_MIXED_PRESET);
    let output = profile.output_path();
    let case = SyntheticCase {
        profile,
        problem: &problem,
        truth: &truth,
        true_model: &true_model,
        seed: 2026,
        output: &output,
    };

    let mut records = Vec::new();
    run_synthetic_case(
        &mut records,
        &case,
        format!("iterations={iterations},object_step=1.0"),
        AlternatingProjection::default().iterations(iterations),
    )?;
    run_synthetic_case(
        &mut records,
        &case,
        format!("iterations={iterations},object_step=0.8,stability=0.1"),
        Fpie::default().iterations(iterations),
    )?;
    run_synthetic_case(
        &mut records,
        &case,
        format!("iterations={iterations},recover_pupil=true,pupil_step=0.1"),
        Epry::default().iterations(iterations).recover_pupil(true),
    )?;
    run_synthetic_case(
        &mut records,
        &case,
        format!("iterations={iterations},penalty=1.0,batch_size=all"),
        Admm::default().iterations(iterations),
    )?;
    run_synthetic_case(
        &mut records,
        &case,
        format!("iterations={iterations},batch_size=3,parallel_workers=1"),
        GradientDescent::default()
            .iterations(iterations)
            .batch_size(3)
            .parallel_workers(1),
    )?;

    std::fs::create_dir_all(&output)?;
    write_benchmark_csv(&records, output.join("summary.csv"))?;
    write_benchmark_json(&records, output.join("summary.json"))?;
    for record in &records {
        println!(
            "{}: success={} runtime={:.3}s final_loss={:.6e} amplitude_rmse={:.6e}",
            record.algorithm_name,
            record.success,
            record.runtime_seconds,
            record.final_loss.unwrap_or(f64::NAN),
            record.amplitude_rmse.unwrap_or(f64::NAN),
        );
    }
    println!(
        "profile={} expected_runtime={} wrote {}",
        profile.name,
        profile.expected_runtime,
        output.display()
    );
    Ok(())
}

struct SyntheticCase<'a, M> {
    profile: &'a BenchmarkProfile,
    problem: &'a ReconstructionProblem<M>,
    truth: &'a Array2<Complex64>,
    true_model: &'a ImagePlaneModel,
    seed: u64,
    output: &'a Path,
}

fn run_synthetic_case<A, M>(
    records: &mut Vec<BenchmarkRecord>,
    case: &SyntheticCase<'_, M>,
    algorithm_configuration: String,
    algorithm: A,
) -> Result<()>
where
    A: ReconstructionAlgorithm,
    M: MeasurementRead,
{
    let (mut record, result) = run_benchmark_case(
        "synthetic",
        algorithm_configuration,
        algorithm,
        case.problem,
        Some(case.truth),
        Some(case.true_model),
        None,
    );
    record.preset_name = Some(NOISELESS_MIXED_PRESET.into());
    record.random_seed = Some(case.seed);
    annotate_benchmark_profile(&mut record, case.profile);
    if let Some(result) = &result {
        save_benchmark_outputs(&mut record, result, case.output)?;
    }
    records.push(record);
    Ok(())
}

fn unknown_profile(profile_name: &str) -> Error {
    let known = BENCHMARK_PROFILES
        .iter()
        .map(|profile| profile.name)
        .collect::<Vec<_>>()
        .join(", ");
    Error::InvalidParameter {
        name: "profile",
        reason: format!("expected one of {known}, got {profile_name:?}"),
    }
}
