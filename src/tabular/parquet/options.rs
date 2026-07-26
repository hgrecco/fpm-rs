use polars::prelude::{ParquetCompression, StatisticsOptions};

#[derive(Clone, Debug)]
pub(crate) struct ParquetWriteOptions {
    pub(crate) compression: ParquetCompression,
    pub(crate) statistics: StatisticsOptions,
    pub(crate) row_group_size: Option<usize>,
    pub(crate) data_page_size: Option<usize>,
    pub(crate) parallel: bool,
}

impl Default for ParquetWriteOptions {
    fn default() -> Self {
        Self {
            compression: ParquetCompression::Zstd(None),
            statistics: StatisticsOptions::default(),
            row_group_size: None,
            data_page_size: None,
            parallel: true,
        }
    }
}
