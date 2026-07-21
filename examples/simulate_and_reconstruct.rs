mod support;

use fpm_rs::{
    Result,
    algorithms::{AlternatingProjection, ReconstructionAlgorithm},
    callbacks::{CsvLogger, SaveImageEvery},
    evaluation::evaluate_reconstruction,
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject},
};

fn main() -> Result<()> {
    let model = support::experimental_model()?;
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((64, 64))?)
        .seed(1234)
        .simulate()?;
    let truth = simulation.ground_truth_object.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let result = AlternatingProjection::default()
        .iterations(20)
        .run_with_callbacks(
            &problem,
            vec![
                Box::new(SaveImageEvery::new(5, "output/iterations")),
                Box::new(CsvLogger::new("output/loss.csv")),
            ],
        )?;
    result.save_amplitude("output/amplitude.png")?;
    result.save_phase("output/phase.png")?;
    let metrics = evaluate_reconstruction(&result, &truth, None, None)?;
    println!(
        "final loss {:.4e}, amplitude RMSE {:.4e}",
        result.history.final_loss().unwrap_or(f64::NAN),
        metrics.object.amplitude_rmse
    );
    Ok(())
}
