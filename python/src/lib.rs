mod arrays;
mod benchmark_bundle;
mod bundle;
mod config;
mod datasets;
mod diagnostics;
mod errors;
mod illumination_initialization;
mod measurements;
mod metrics;
mod model;
mod multi_wavelength;
mod reconstruction;
mod simulation;
mod spectral;

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
    illumination_initialization::register(module)?;
    metrics::register(module)?;
    model::register(module)?;
    measurements::register(module)?;
    simulation::register(module)?;
    reconstruction::register(module)?;
    spectral::register(module)?;
    multi_wavelength::register(module)?;
    Ok(())
}
