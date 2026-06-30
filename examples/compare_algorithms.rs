mod support;

use fpm_rs::{
    Result,
    algorithms::{Admm, AlternatingProjection, Epry, ReconstructionAlgorithm},
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject, compare_to_ground_truth},
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
    let ap_metrics = compare_to_ground_truth(&ap, &truth)?;
    let epry_metrics = compare_to_ground_truth(&epry, &truth)?;
    let admm_metrics = compare_to_ground_truth(&admm, &truth)?;
    println!("AP complex error:   {:.4e}", ap_metrics.complex_field_error);
    println!(
        "Epry complex error: {:.4e}",
        epry_metrics.complex_field_error
    );
    println!(
        "ADMM complex error: {:.4e}",
        admm_metrics.complex_field_error
    );
    Ok(())
}
