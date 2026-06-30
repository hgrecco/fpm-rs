#[derive(Clone, Debug)]
pub struct Batch {
    pub indices: Vec<usize>,
    pub batch_index: usize,
}

impl Batch {
    pub fn new(indices: Vec<usize>, batch_index: usize) -> Self {
        Self {
            indices,
            batch_index,
        }
    }

    pub fn single(frame: usize) -> Self {
        Self::new(vec![frame], 0)
    }
}
