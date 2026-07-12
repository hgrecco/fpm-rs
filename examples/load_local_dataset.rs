use std::{env, path::PathBuf};

use fpm_rs::{
    Error, Result,
    algorithms::AlternatingProjection,
    benchmark::{
        run_benchmark_case, save_benchmark_outputs, write_benchmark_csv, write_benchmark_json,
    },
    datasets::DatasetLoader,
};

fn main() -> Result<()> {
    let root = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| Error::Dataset("usage: load_local_dataset <dataset-directory>".into()))?;
    let dataset = DatasetLoader::new(&root)?.load()?;
    println!(
        "loaded {} frames {:?}, object {:?}, units={:?}, provenance={:?}",
        dataset.measurements().frame_count(),
        dataset.measurements().image_shape(),
        dataset.configuration().reconstruction_shape,
        dataset.measurement_units(),
        dataset.provenance(),
    );

    let problem = dataset.reconstruction_problem()?;
    let true_model = &dataset.configuration().compiled_models.true_model;
    let (mut record, result) = run_benchmark_case(
        root.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("dataset")
            .to_owned(),
        "iterations=10,object_step=1.0",
        AlternatingProjection::default().iterations(10),
        &problem,
        dataset.ground_truth_object(),
        Some(true_model),
        dataset.valid_object_mask(),
    );
    record.metadata.extend(dataset.provenance().clone());
    if let Some(units) = dataset.measurement_units() {
        record
            .metadata
            .insert("measurement_units".into(), units.into());
    }

    let output = PathBuf::from("target/local-dataset-results");
    if let Some(result) = &result {
        save_benchmark_outputs(&mut record, result, &output)?;
    }
    std::fs::create_dir_all(&output)?;
    write_benchmark_csv(std::slice::from_ref(&record), output.join("summary.csv"))?;
    write_benchmark_json(std::slice::from_ref(&record), output.join("summary.json"))?;
    println!(
        "benchmark success={} final_loss={:?}; wrote {}",
        record.success,
        record.final_loss,
        output.display()
    );
    Ok(())
}
