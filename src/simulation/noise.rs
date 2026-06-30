use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseModel {
    #[default]
    None,
    Poisson,
    Gaussian,
    PoissonGaussian,
}
