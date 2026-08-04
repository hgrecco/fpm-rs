/// One scheduled group of acquisition frames supplied to an algorithm step.
#[derive(Clone, Debug)]
pub struct Batch {
    /// Zero-based acquisition-frame indices in processing order.
    pub indices: Vec<usize>,
    /// Zero-based position of this batch within the current iteration.
    pub batch_index: usize,
}

impl Batch {
    /// Creates a batch from acquisition-frame `indices` and its iteration-local index.
    pub fn new(indices: Vec<usize>, batch_index: usize) -> Self {
        Self {
            indices,
            batch_index,
        }
    }

    /// Creates the first batch containing a single acquisition frame.
    pub fn single(frame: usize) -> Self {
        Self::new(vec![frame], 0)
    }
}
