use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
/// Preprocessing operations configured for a measurement stack.
pub struct PreprocessingConfig {
    /// Subtract the shared detector dark frame before other corrections.
    pub subtract_dark: bool,
    /// Divide by the shared positive flat-field response after dark subtraction.
    pub divide_flat_field: bool,
    /// Divide each acquisition frame by its positive exposure time.
    pub normalize_exposure: bool,
    /// Subtract the shared or per-frame additive background.
    pub subtract_background: bool,
    /// Replace negative corrected intensities with zero after all other operations.
    pub clamp_negative: bool,
}
