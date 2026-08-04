use fpm_rs::{
    Result,
    experiment::{ArrayPose, Illumination, Optics, PlanarLedArray},
    model::{ImagePlaneModel, ReconstructionShape},
};

pub fn experimental_setup() -> Result<(Optics, Illumination)> {
    let optics = Optics {
        wavelength_vacuum_m: 532e-9,
        objective_na: 0.10,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        illumination_refractive_index: 1.0,
        objective_medium_refractive_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let illumination = Illumination::from_geometry(PlanarLedArray::new(
        (3, 3),
        (4e-3, 4e-3),
        (1.0, 1.0),
        ArrayPose::from_translation([0.0, 0.0, -90e-3]),
    ))?;
    Ok((optics, illumination))
}

pub fn experimental_model() -> Result<ImagePlaneModel> {
    let (optics, illumination) = experimental_setup()?;
    ImagePlaneModel::from_experiment(
        &optics,
        &illumination,
        (32, 32),
        ReconstructionShape::Exact((64, 64)),
    )
}
