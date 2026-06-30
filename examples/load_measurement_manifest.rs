use std::{env, path::PathBuf};

use fpm_rs::{Error, Result, measurements::MeasurementStack};

fn main() -> Result<()> {
    let manifest =
        env::args_os()
            .nth(1)
            .map(PathBuf::from)
            .ok_or_else(|| Error::InvalidParameter {
                name: "manifest",
                reason: "usage: cargo run --example load_measurement_manifest -- measurements.json"
                    .into(),
            })?;
    let stack = MeasurementStack::from_manifest(manifest)?;
    println!(
        "loaded {} frames with shape {:?}",
        stack.frame_count(),
        stack.image_shape()
    );
    let processed = stack.apply_preprocessing()?;
    println!("first processed detector count: {}", processed.frame(0)?[0]);
    Ok(())
}
