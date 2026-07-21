use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: &'static str, reason: String },
    #[error("model validation failed: {0}")]
    InvalidModel(String),
    #[error("measurement validation failed: {0}")]
    InvalidMeasurements(String),
    #[error("array length {actual} does not match shape {shape:?} (expected {expected})")]
    LengthMismatch {
        actual: usize,
        expected: usize,
        shape: (usize, usize),
    },
    #[error("array length {actual} does not match shape {shape:?} (expected {expected})")]
    ArrayLengthMismatch {
        actual: usize,
        expected: usize,
        shape: Vec<usize>,
    },
    #[error("array shape {shape:?} overflows addressable storage")]
    ShapeOverflow { shape: Vec<usize> },
    #[error(
        "{context} with shape {shape:?} and strides {strides:?} is not C-contiguous standard row-major layout"
    )]
    NonStandardLayout {
        context: &'static str,
        shape: Vec<usize>,
        strides: Vec<isize>,
    },
    #[error("ndarray shape construction failed: {0}")]
    NdarrayShape(#[from] ndarray::ShapeError),
    #[error("frame index {index} is out of range for {frames} frames")]
    FrameOutOfRange { index: usize, frames: usize },
    #[error("numerical error: {0}")]
    Numerical(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("dataset error: {0}")]
    Dataset(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("TIFF error: {0}")]
    Tiff(#[from] tiff::TiffError),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
