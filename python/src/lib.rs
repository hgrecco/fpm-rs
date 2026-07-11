mod arrays;
mod config;
mod errors;
mod measurements;
mod model;
mod simulation;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    errors::register(module)?;
    config::register(module)?;
    model::register(module)?;
    measurements::register(module)?;
    simulation::register(module)?;
    Ok(())
}
