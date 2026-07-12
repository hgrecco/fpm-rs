use crate::{
    Array2, Complex64, Result,
    configuration::{ExperimentDescription, SimulationConfiguration},
    error::Error,
    experiment::Illumination,
    measurements::MeasurementStack,
    model::ImagePlaneModel,
    reconstruction::ReconstructionProblem,
};

use super::Dataset;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub row: usize,
    pub column: usize,
    pub height: usize,
    pub width: usize,
}

impl Rect {
    pub fn new(row: usize, column: usize, height: usize, width: usize) -> Result<Self> {
        if height == 0 || width == 0 {
            return Err(Error::Dataset(
                "dataset crop height and width must be non-zero".into(),
            ));
        }
        Ok(Self {
            row,
            column,
            height,
            width,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameSelector {
    All,
    EveryNth(usize),
    Indices(Vec<usize>),
}

#[derive(Clone, Debug)]
pub struct DatasetSubset {
    measurements: MeasurementStack,
    configuration: SimulationConfiguration,
    ground_truth_object: Option<Array2<Complex64>>,
    valid_object_mask: Option<Array2<u8>>,
    provenance: std::collections::BTreeMap<String, String>,
    measurement_units: Option<String>,
}

impl DatasetSubset {
    pub fn measurements(&self) -> &MeasurementStack {
        &self.measurements
    }

    pub fn configuration(&self) -> &SimulationConfiguration {
        &self.configuration
    }

    pub fn ground_truth_object(&self) -> Option<&Array2<Complex64>> {
        self.ground_truth_object.as_ref()
    }

    pub fn valid_object_mask(&self) -> Option<&Array2<u8>> {
        self.valid_object_mask.as_ref()
    }

    pub fn provenance(&self) -> &std::collections::BTreeMap<String, String> {
        &self.provenance
    }

    pub fn measurement_units(&self) -> Option<&str> {
        self.measurement_units.as_deref()
    }

    pub fn reconstruction_problem(&self) -> Result<ReconstructionProblem<MeasurementStack>> {
        ReconstructionProblem::new(
            self.measurements.clone(),
            self.configuration
                .compiled_models
                .reconstruction_model
                .clone(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct DatasetSubsetBuilder<'a> {
    dataset: &'a Dataset,
    frames: FrameSelector,
    crop: Option<Rect>,
}

impl<'a> DatasetSubsetBuilder<'a> {
    pub(crate) fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            frames: FrameSelector::All,
            crop: None,
        }
    }

    pub fn frames(mut self, selector: FrameSelector) -> Self {
        self.frames = selector;
        self
    }

    pub fn every_nth_frame(self, step: usize) -> Self {
        self.frames(FrameSelector::EveryNth(step))
    }

    pub fn crop(mut self, crop: Rect) -> Self {
        self.crop = Some(crop);
        self
    }

    pub fn crop_pixels(
        self,
        row: usize,
        column: usize,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        Ok(self.crop(Rect::new(row, column, height, width)?))
    }

    pub fn build(self) -> Result<DatasetSubset> {
        let source = self.dataset.measurements();
        let model = &self
            .dataset
            .configuration()
            .compiled_models
            .reconstruction_model;
        let indices = selected_indices(&self.frames, source.frame_count())?;
        let source_shape = source.image_shape();
        let crop = self.crop.unwrap_or(Rect {
            row: 0,
            column: 0,
            height: source_shape.0,
            width: source_shape.1,
        });
        validate_crop(crop, source_shape)?;

        let mut data = Vec::with_capacity(indices.len() * crop.height * crop.width);
        let mut metadata = Vec::with_capacity(indices.len());
        for (new_index, &source_index) in indices.iter().enumerate() {
            crop_frame(source.frame(source_index)?, source_shape, crop, &mut data);
            let mut frame_metadata = source.frame_metadata[source_index].clone();
            frame_metadata.original_frame_index =
                Some(frame_metadata.original_frame_index.unwrap_or(source_index));
            frame_metadata.original_illumination_index = frame_metadata
                .original_illumination_index
                .or(frame_metadata.illumination_index);
            frame_metadata.frame_index = new_index;
            frame_metadata.illumination_index = (!model.is_multiplexed()).then_some(new_index);
            metadata.push(frame_metadata);
        }
        let mut measurements =
            MeasurementStack::from_vec(data, (crop.height, crop.width), metadata)?;
        measurements.dark_frame =
            crop_optional_shared(source.dark_frame.as_deref(), source_shape, crop);
        measurements.flat_field =
            crop_optional_shared(source.flat_field.as_deref(), source_shape, crop);
        measurements.background = crop_optional_frames(
            source.background.as_deref(),
            source_shape,
            source.frame_count(),
            &indices,
            crop,
        )?;
        measurements.masks = crop_optional_masks(
            source.masks.as_deref(),
            source_shape,
            source.frame_count(),
            &indices,
            crop,
        )?;
        measurements.preprocessing = source.preprocessing.clone();
        measurements.validate()?;

        let scale_y = model.reconstruction_shape.0 as f64 / model.image_shape.0 as f64;
        let scale_x = model.reconstruction_shape.1 as f64 / model.image_shape.1 as f64;
        let reconstruction_shape = (
            (crop.height as f64 * scale_y).round() as usize,
            (crop.width as f64 * scale_x).round() as usize,
        );
        let object_crop = scaled_object_crop(crop, scale_y, scale_x)?;
        let ground_truth_object = self
            .dataset
            .ground_truth_object()
            .map(|truth| crop_array(truth, object_crop))
            .transpose()?;
        let valid_object_mask = self
            .dataset
            .valid_object_mask()
            .map(|mask| crop_array(mask, object_crop))
            .transpose()?;
        let source_configuration = self.dataset.configuration();
        let true_experiment = subset_experiment(
            &source_configuration.true_experiment,
            &source_configuration.compiled_models.true_model,
            &indices,
            crop,
        )?;
        let reconstruction_experiment = subset_experiment(
            &source_configuration.reconstruction_experiment,
            &source_configuration.compiled_models.reconstruction_model,
            &indices,
            crop,
        )?;
        let configuration = SimulationConfiguration::new(
            true_experiment,
            reconstruction_experiment,
            (crop.height, crop.width),
            reconstruction_shape,
        )?
        .with_random_seed(source_configuration.random_seed);
        Ok(DatasetSubset {
            measurements,
            configuration,
            ground_truth_object,
            valid_object_mask,
            provenance: self.dataset.provenance().clone(),
            measurement_units: self.dataset.measurement_units().map(str::to_owned),
        })
    }
}

fn subset_experiment(
    description: &ExperimentDescription,
    model: &ImagePlaneModel,
    indices: &[usize],
    crop: Rect,
) -> Result<ExperimentDescription> {
    let (k_vectors, frame_weights) = match &model.multiplexing_matrix {
        Some(matrix) => (
            model.k_vectors.clone(),
            Some(indices.iter().map(|&index| matrix[index].clone()).collect()),
        ),
        None => (
            indices
                .iter()
                .map(|&index| model.k_vectors[index])
                .collect(),
            None,
        ),
    };
    let frame_gains = model
        .frame_gains
        .as_ref()
        .map(|gains| indices.iter().map(|&index| gains[index]).collect());
    let illumination = Illumination::Calibrated {
        k_vectors,
        frame_gains,
        frame_weights,
    };
    let mut subset = ExperimentDescription::new(description.optics.clone(), illumination);
    subset.optical_background = crop_optional_frames(
        model.background.as_deref(),
        model.image_shape,
        model.frame_count(),
        indices,
        crop,
    )?;
    subset.validate()?;
    Ok(subset)
}

fn selected_indices(selector: &FrameSelector, frame_count: usize) -> Result<Vec<usize>> {
    let indices = match selector {
        FrameSelector::All => (0..frame_count).collect(),
        FrameSelector::EveryNth(0) => {
            return Err(Error::Dataset(
                "frame subset step must be greater than zero".into(),
            ));
        }
        FrameSelector::EveryNth(step) => (0..frame_count).step_by(*step).collect(),
        FrameSelector::Indices(indices) => indices.clone(),
    };
    if indices.is_empty() {
        return Err(Error::Dataset(
            "dataset frame subset must contain at least one frame".into(),
        ));
    }
    let mut seen = vec![false; frame_count];
    for &index in &indices {
        if index >= frame_count {
            return Err(Error::FrameOutOfRange {
                index,
                frames: frame_count,
            });
        }
        if seen[index] {
            return Err(Error::Dataset(format!(
                "dataset frame subset contains duplicate index {index}"
            )));
        }
        seen[index] = true;
    }
    Ok(indices)
}

fn validate_crop(crop: Rect, shape: (usize, usize)) -> Result<()> {
    if crop
        .row
        .checked_add(crop.height)
        .is_none_or(|end| end > shape.0)
        || crop
            .column
            .checked_add(crop.width)
            .is_none_or(|end| end > shape.1)
    {
        return Err(Error::Dataset(format!(
            "crop {crop:?} is outside measurement shape {shape:?}"
        )));
    }
    Ok(())
}

fn crop_frame<T: Copy>(source: &[T], shape: (usize, usize), crop: Rect, output: &mut Vec<T>) {
    for row in crop.row..crop.row + crop.height {
        let start = row * shape.1 + crop.column;
        output.extend_from_slice(&source[start..start + crop.width]);
    }
}

fn scaled_object_crop(crop: Rect, scale_y: f64, scale_x: f64) -> Result<Rect> {
    let values = [
        crop.row as f64 * scale_y,
        crop.column as f64 * scale_x,
        crop.height as f64 * scale_y,
        crop.width as f64 * scale_x,
    ];
    if values
        .iter()
        .any(|value| (value - value.round()).abs() > 1e-9)
    {
        return Err(Error::Dataset(
            "measurement crop does not map to integer reconstruction pixels".into(),
        ));
    }
    Rect::new(
        values[0].round() as usize,
        values[1].round() as usize,
        values[2].round() as usize,
        values[3].round() as usize,
    )
}

fn crop_array<T: Copy>(source: &Array2<T>, crop: Rect) -> Result<Array2<T>> {
    validate_crop(crop, source.shape())?;
    let mut output = Vec::with_capacity(crop.height * crop.width);
    crop_frame(source.as_slice(), source.shape(), crop, &mut output);
    Array2::from_vec((crop.height, crop.width), output)
}

fn crop_optional_shared(
    source: Option<&[f64]>,
    shape: (usize, usize),
    crop: Rect,
) -> Option<Vec<f64>> {
    source.map(|source| {
        let mut output = Vec::with_capacity(crop.height * crop.width);
        crop_frame(source, shape, crop, &mut output);
        output
    })
}

fn crop_optional_frames(
    source: Option<&[f64]>,
    shape: (usize, usize),
    frame_count: usize,
    indices: &[usize],
    crop: Rect,
) -> Result<Option<Vec<f64>>> {
    let Some(source) = source else {
        return Ok(None);
    };
    let frame_len = shape.0 * shape.1;
    if source.len() == frame_len {
        return Ok(crop_optional_shared(Some(source), shape, crop));
    }
    if source.len() != frame_count * frame_len {
        return Err(Error::InvalidMeasurements(
            "source background length is inconsistent".into(),
        ));
    }
    let mut output = Vec::with_capacity(indices.len() * crop.height * crop.width);
    for &index in indices {
        crop_frame(
            &source[index * frame_len..(index + 1) * frame_len],
            shape,
            crop,
            &mut output,
        );
    }
    Ok(Some(output))
}

fn crop_optional_masks(
    source: Option<&[u8]>,
    shape: (usize, usize),
    frame_count: usize,
    indices: &[usize],
    crop: Rect,
) -> Result<Option<Vec<u8>>> {
    let Some(source) = source else {
        return Ok(None);
    };
    let frame_len = shape.0 * shape.1;
    if source.len() == frame_len {
        let mut output = Vec::with_capacity(crop.height * crop.width);
        crop_frame(source, shape, crop, &mut output);
        return Ok(Some(output));
    }
    if source.len() != frame_count * frame_len {
        return Err(Error::InvalidMeasurements(
            "source mask length is inconsistent".into(),
        ));
    }
    let mut output = Vec::with_capacity(indices.len() * crop.height * crop.width);
    for &index in indices {
        crop_frame(
            &source[index * frame_len..(index + 1) * frame_len],
            shape,
            crop,
            &mut output,
        );
    }
    Ok(Some(output))
}
