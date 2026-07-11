use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
/// Preprocessing operations configured for a measurement stack.
pub struct PreprocessingConfig {
    pub subtract_dark: bool,
    pub divide_flat_field: bool,
    pub normalize_exposure: bool,
    pub subtract_background: bool,
    pub clamp_negative: bool,
}
