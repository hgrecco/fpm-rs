//! Hooks that observe or control iterative reconstruction.
//!
//! Implement [`crate::callbacks::Callback`] for custom behavior, or use callbacks such as
//! [`crate::callbacks::CheckpointEvery`],
//! [`DiagnosticRecorder`](crate::diagnostics::DiagnosticRecorder), and
//! [`crate::callbacks::StopOnPlateau`]. Callbacks receive a borrowed
//! [`crate::callbacks::StepContext`] and return a [`crate::callbacks::CallbackAction`] to
//! continue or stop a run.

mod checkpoint;
mod context;
mod csv_logger;
mod early_stop;
mod progress;
mod save_image;

pub use checkpoint::CheckpointEvery;
pub use context::{Callback, CallbackAction, CallbackHook, StepContext};
pub use csv_logger::CsvLogger;
pub use early_stop::StopOnPlateau;
pub use progress::ProgressLogger;
pub use save_image::{SaveImageEvery, SavePupilEvery, SaveResidualsEvery};
