mod arrays;
mod config;
mod datasets;
mod errors;
mod measurements;
mod model;
mod reconstruction;
mod simulation;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    errors::register(module)?;
    config::register(module)?;
    datasets::register(module)?;
    model::register(module)?;
    measurements::register(module)?;
    simulation::register(module)?;
    reconstruction::register(module)?;
    Ok(())
}
