mod support;

use fpm_rs::{
    Result,
    algorithms::{Epry, ReconstructionAlgorithm},
    experiment::PupilAberration,
    model::{ImagePlaneModel, ReconstructionShape},
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject, compare_with_true_model},
};

fn main() -> Result<()> {
    let (assumed_optics, illumination) = support::experimental_setup();
    let true_optics = fpm_rs::experiment::Optics {
        defocus_distance: Some(-24e-6),
        pupil_aberration: Some(PupilAberration {
            astigmatism: 0.2,
            ..PupilAberration::default()
        }),
        ..assumed_optics.clone()
    };
    let true_model = ImagePlaneModel::from_experiment(
        &true_optics,
        &illumination,
        (32, 32),
        ReconstructionShape::Exact((64, 64)),
    )?;
    let reconstruction_model = support::experimental_model()?;
    let simulation = Simulator::new(true_model)
        .object(SyntheticObject::phase_disk((64, 64), 16.0, 0.9)?)
        .reconstruction_model(reconstruction_model)
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
