mod support;

use fpm_rs::{
    Result,
    algorithms::{Admm, AlternatingProjection, Epry, ReconstructionAlgorithm},
    evaluation::evaluate_reconstruction,
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject},
};

fn main() -> Result<()> {
    let model = support::experimental_model()?;
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::resolution_target((64, 64))?)
        .simulate()?;
    let truth = simulation.ground_truth_object.clone();
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let ap = AlternatingProjection::default()
        .iterations(20)
        .run(&problem)?;
    let epry = Epry::default().iterations(20).run(&problem)?;
    let admm = Admm::default().iterations(20).run(&problem)?;
    let ap_metrics = evaluate_reconstruction(&ap, &truth, None, None)?;
    let epry_metrics = evaluate_reconstruction(&epry, &truth, None, None)?;
    let admm_metrics = evaluate_reconstruction(&admm, &truth, None, None)?;
    println!(
        "AP complex error:   {:.4e}",
        ap_metrics.object.complex_nrmse
    );
    println!(
        "Epry complex error: {:.4e}",
        epry_metrics.object.complex_nrmse
    );
    println!(
        "ADMM complex error: {:.4e}",
        admm_metrics.object.complex_nrmse
    );
    Ok(())
}
