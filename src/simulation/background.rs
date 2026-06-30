use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BackgroundModel {
    Constant(f64),
    PerPixel(Vec<f64>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlatFieldModel {
    pub values: Vec<f64>,
}

impl FlatFieldModel {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }
}
