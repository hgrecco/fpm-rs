mod arrays;
mod benchmark_bundle;
mod bundle;
mod config;
mod datasets;
mod diagnostics;
mod errors;
mod measurements;
mod metrics;
mod model;
mod reconstruction;
mod simulation;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    errors::register(module)?;
    bundle::register(module)?;
    benchmark_bundle::register(module)?;
    config::register(module)?;
    datasets::register(module)?;
    diagnostics::register(module)?;
    metrics::register(module)?;
    model::register(module)?;
    measurements::register(module)?;
    simulation::register(module)?;
    reconstruction::register(module)?;
    Ok(())
}
