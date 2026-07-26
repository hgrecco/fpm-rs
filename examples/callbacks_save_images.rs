mod support;

use fpm_rs::{
    Result,
    algorithms::{Epry, ReconstructionAlgorithm},
    callbacks::{CheckpointEvery, CsvLogger, SaveImageEvery, SaveResidualsEvery, StopOnPlateau},
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject},
};

fn main() -> Result<()> {
    let model = support::experimental_model()?;
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::siemens_star((64, 64), 24)?)
        .simulate()?;
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let result = Epry::default().iterations(50).run_with_callbacks(
        &problem,
        vec![
            Box::new(SaveImageEvery::new(10, "output/callback_frames")),
            Box::new(SaveResidualsEvery::new(10, "output/callback_residuals")),
            Box::new(CsvLogger::new("output/callback_objective.csv")),
            Box::new(CheckpointEvery::new(25, "output/checkpoints")),
            Box::new(StopOnPlateau::new(10, 1e-7)),
        ],
    )?;
    println!(
        "completed {} iterations (stopped early: {})",
        result.runtime.completed_iterations, result.runtime.stopped_early
    );
    Ok(())
}
