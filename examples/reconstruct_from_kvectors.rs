use fpm_rs::{
    Array2, Complex64, Result,
    algorithms::{Fpie, ReconstructionAlgorithm},
    experiment::KVector,
    model::{CropIndices, FourierCrop, ImagePlaneModel, Pupil, Sampling},
    reconstruction::ReconstructionProblem,
    simulation::{Simulator, SyntheticObject},
};

fn main() -> Result<()> {
    let low_shape = (32, 32);
    let high_shape = (64, 64);
    let sampling = Sampling::new(1.0e-6, 0.5e-6, 1.0, 1.0)?;
    let shifts = [(-8, 0), (0, -8), (0, 0), (0, 8), (8, 0)];
    let k_vectors = shifts
        .iter()
        .map(|&(column, row)| KVector::new(column as f64, row as f64))
        .collect();
    let crops = shifts
        .iter()
        .map(|&(column, row)| {
            FourierCrop::new(
                (16_i32 + row) as usize,
                (16_i32 + column) as usize,
                low_shape.0,
                low_shape.1,
            )
        })
        .collect();
    let pupil = Pupil::new(
        Array2::filled(low_shape, Complex64::new(1.0, 0.0))?,
        vec![true; low_shape.0 * low_shape.1],
    )?;
    let model = ImagePlaneModel::new(
        k_vectors,
        pupil,
        CropIndices::new(crops),
        sampling,
        low_shape,
        high_shape,
    )?;
    let simulation = Simulator::ideal(model)
        .object(SyntheticObject::phase_disk(high_shape, 14.0, 0.8)?)
        .simulate()?;
    let problem =
        ReconstructionProblem::new(simulation.measurements, simulation.reconstruction_model)?;
    let result = Fpie::default().iterations(15).run(&problem)?;
    println!("reconstructed {} pixels", result.object.len());
    Ok(())
}
