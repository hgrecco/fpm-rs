use std::{path::PathBuf, process::ExitCode};

use clap::{Args, Parser, Subcommand};
use fpm_rs::{Result, datasets::DatasetRegistry};

#[derive(Debug, Parser)]
#[command(name = "fpm-datasets", about = "Discover and manage fpm-rs datasets")]
struct Cli {
    /// Registry URL, file URL, or local JSON path.
    #[arg(long, global = true)]
    registry_url: Option<String>,
    /// Managed dataset cache directory.
    #[arg(long, global = true)]
    cache_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List registry entries and their cache status.
    List,
    /// Download and validate one dataset or every registry entry.
    Download(Selection),
    /// Download if necessary, open, and validate one dataset.
    Open { id: String },
    /// Remove one dataset or the complete managed cache.
    Clean(Selection),
}

#[derive(Debug, Args)]
struct Selection {
    /// Dataset identifier.
    id: Option<String>,
    /// Apply the command to every dataset.
    #[arg(long, conflicts_with = "id")]
    all: bool,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    let registry = match (cli.registry_url, cli.cache_dir) {
        (None, None) => DatasetRegistry::from_defaults()?,
        (Some(registry_url), Some(cache_dir)) => DatasetRegistry::new(registry_url, cache_dir)?,
        (registry_url, cache_dir) => {
            let defaults = DatasetRegistry::from_defaults()?;
            DatasetRegistry::new(
                registry_url.unwrap_or_else(|| defaults.registry_url().to_owned()),
                cache_dir.unwrap_or_else(|| defaults.cache_dir().to_owned()),
            )?
        }
    };
    match cli.command {
        Command::List => {
            println!("ID\tVERSION\tCACHED\tSIZE_BYTES\tTITLE");
            for listing in registry.list()? {
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    listing.entry.id,
                    listing.entry.version,
                    if listing.cached { "yes" } else { "no" },
                    listing.entry.archive.size_bytes,
                    listing.entry.title
                );
            }
        }
        Command::Download(selection) => {
            let id = selected_id(selection, "download")?;
            if let Some(id) = id {
                println!("{}", registry.download(&id)?.display());
            } else {
                for path in registry.download_all()? {
                    println!("{}", path.display());
                }
            }
        }
        Command::Open { id } => {
            let dataset = registry.open(&id)?;
            let path = dataset.source_path().map_or_else(
                || "<programmatic>".into(),
                |path| path.display().to_string(),
            );
            println!("id={id}");
            println!("path={path}");
            println!("frames={}", dataset.measurements().frame_count());
            println!("image_shape={:?}", dataset.measurements().image_shape());
            println!(
                "reconstruction_shape={:?}",
                dataset.configuration().reconstruction_shape
            );
        }
        Command::Clean(selection) => {
            let id = selected_id(selection, "clean")?;
            if let Some(id) = id {
                println!("removed={}", registry.clean(&id)?);
            } else {
                println!("removed={}", registry.clean_all()?);
            }
        }
    }
    Ok(())
}

fn selected_id(selection: Selection, command: &str) -> Result<Option<String>> {
    match (selection.id, selection.all) {
        (Some(id), false) => Ok(Some(id)),
        (None, true) => Ok(None),
        _ => Err(fpm_rs::Error::Dataset(format!(
            "{command} requires a dataset ID or --all"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_grammar_accepts_every_supported_operation() {
        for arguments in [
            vec!["fpm-datasets", "list"],
            vec!["fpm-datasets", "download", "fixture"],
            vec!["fpm-datasets", "download", "--all"],
            vec!["fpm-datasets", "open", "fixture"],
            vec!["fpm-datasets", "clean", "fixture"],
            vec!["fpm-datasets", "clean", "--all"],
        ] {
            Cli::try_parse_from(arguments).unwrap();
        }
    }

    #[test]
    fn selection_requires_exactly_one_target() {
        assert!(
            selected_id(
                Selection {
                    id: None,
                    all: false
                },
                "clean"
            )
            .is_err()
        );
        assert_eq!(
            selected_id(
                Selection {
                    id: Some("fixture".into()),
                    all: false,
                },
                "clean",
            )
            .unwrap(),
            Some("fixture".into())
        );
    }
}
