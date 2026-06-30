mod support;

use fpm_rs::{
    Result,
    algorithms::{Epry, ReconstructionAlgorithm},
    reconstruction::ReconstructionProblem,
    simulation::{AberrationModel, Simulator, SyntheticObject, compare_with_true_model},
};

fn main() -> Result<()> {
    let initial_model = support::experimental_model()?;
    let simulation = Simulator::new(initial_model.clone())
        .object(SyntheticObject::phase_disk((64, 64), 16.0, 0.9)?)
        .aberration(AberrationModel::new().defocus(0.7).astigmatism(0.2))
        .reconstruction_model(initial_model)
        .simulate()?;
    let truth = simulation.ground_truth_object.clone();
    let true_model = simulation.true_model.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let result = Epry::default()
        .iterations(30)
        .pupil_step(0.05)
        .recover_pupil(true)
        .run(&problem)?;
    let metrics = compare_with_true_model(&result, &truth, &true_model)?;
    println!("pupil phase RMSE: {:?}", metrics.pupil_phase_error);
    Ok(())
}
