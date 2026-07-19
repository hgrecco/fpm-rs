use fpm_rs::{
    Result,
    experiment::{LEDArray, Optics},
    model::{ImagePlaneModel, ReconstructionShape},
};

pub fn experimental_setup() -> (Optics, LEDArray) {
    let optics = Optics {
        wavelength: 532e-9,
        objective_na: 0.10,
        magnification: 4.0,
        camera_pixel_size: 6.5e-6,
        medium_index: 1.0,
        defocus_distance: None,
        pupil_aberration: None,
    };
    let illumination = LEDArray::new()
        .grid_shape((3, 3))
        .pitch(4e-3)
        .distance(90e-3)
        .center((1.0, 1.0));
    (optics, illumination)
}

pub fn experimental_model() -> Result<ImagePlaneModel> {
    let (optics, illumination) = experimental_setup();
    ImagePlaneModel::from_experiment(
        &optics,
        &illumination,
        (32, 32),
        ReconstructionShape::Exact((64, 64)),
    )
}
