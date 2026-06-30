use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{
    Result, experiment::KVector, measurements::MeasurementRead, model::ImagePlaneModel,
};

use super::ReconstructionProblem;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum FrameSchedule {
    #[default]
    Sequential,
    BrightfieldFirst,
    SpiralOut,
    RandomShuffle {
        seed: u64,
    },
    /// Highest empirical shot-noise SNR first. This requires measurements and
    /// is applied by [`Self::order_for_problem`]; [`Self::order`] retains
    /// sequential order because a compiled model alone contains no signal data.
    SnrWeighted,
}

impl FrameSchedule {
    /// Returns a model-only frame order.
    ///
    /// [`Self::SnrWeighted`] cannot be estimated from model geometry, so this
    /// method returns sequential order for that variant. Reconstruction runners
    /// use [`Self::order_for_problem`] and therefore have access to measurements.
    pub fn order(&self, model: &ImagePlaneModel, iteration: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..model.frame_count()).collect();
        match self {
            Self::Sequential | Self::SnrWeighted => {}
            Self::BrightfieldFirst => order.sort_by(|&left, &right| {
                let left_vector = frame_vector(model, left);
                let right_vector = frame_vector(model, right);
                left_vector
                    .kx
                    .hypot(left_vector.ky)
                    .total_cmp(&right_vector.kx.hypot(right_vector.ky))
            }),
            Self::SpiralOut => order.sort_by(|&left, &right| {
                let left_vector = frame_vector(model, left);
                let right_vector = frame_vector(model, right);
                left_vector
                    .kx
                    .hypot(left_vector.ky)
                    .total_cmp(&right_vector.kx.hypot(right_vector.ky))
                    .then_with(|| {
                        left_vector
                            .ky
                            .atan2(left_vector.kx)
                            .rem_euclid(std::f64::consts::TAU)
                            .total_cmp(
                                &right_vector
                                    .ky
                                    .atan2(right_vector.kx)
                                    .rem_euclid(std::f64::consts::TAU),
                            )
                    })
            }),
            Self::RandomShuffle { seed } => {
                let mut rng = StdRng::seed_from_u64(
                    seed.wrapping_add((iteration as u64).wrapping_mul(0x9e3779b97f4a7c15)),
                );
                order.shuffle(&mut rng);
            }
        }
        order
    }

    /// Returns a frame order using both the model and measured intensities.
    ///
    /// The SNR-weighted schedule ranks frames by
    /// `frame_weight * sqrt(mean(max(measured - known_background, 0)))` over
    /// unmasked pixels. This is an empirical Poisson shot-noise proxy, not a
    /// camera-noise calibration. Zero-weight or fully masked frames are placed
    /// last, and equal scores retain ascending frame-index order.
    pub fn order_for_problem<M: MeasurementRead>(
        &self,
        problem: &ReconstructionProblem<M>,
        iteration: usize,
    ) -> Result<Vec<usize>> {
        if !matches!(self, Self::SnrWeighted) {
            return Ok(self.order(&problem.model, iteration));
        }
        let mut scored = Vec::with_capacity(problem.model.frame_count());
        for frame in 0..problem.model.frame_count() {
            scored.push((frame, frame_snr_score(problem, frame)?));
        }
        scored.sort_by(|&(left_frame, left_score), &(right_frame, right_score)| {
            right_score
                .total_cmp(&left_score)
                .then_with(|| left_frame.cmp(&right_frame))
        });
        Ok(scored.into_iter().map(|(frame, _)| frame).collect())
    }
}

fn frame_snr_score<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    frame: usize,
) -> Result<f64> {
    let weight = problem.measurements.frame_weight(frame)?;
    if weight == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    let measured = problem.measurements.frame(frame)?;
    let mask = problem.measurements.frame_mask(frame)?;
    let mut signal_sum = 0.0;
    let mut valid_pixels = 0;
    for (pixel, &value) in measured.iter().enumerate() {
        if mask.is_none_or(|values| values[pixel] != 0) {
            let background = problem.model.background_value(frame, pixel)?;
            signal_sum += (value - background).max(0.0);
            valid_pixels += 1;
        }
    }
    if valid_pixels == 0 {
        Ok(f64::NEG_INFINITY)
    } else {
        Ok(weight * (signal_sum / valid_pixels as f64).sqrt())
    }
}

fn frame_vector(model: &ImagePlaneModel, frame: usize) -> KVector {
    if let Some(matrix) = &model.multiplexing_matrix {
        let mut vector = KVector::default();
        let mut total_weight = 0.0;
        for &(source, weight) in &matrix[frame] {
            vector.kx += weight * model.k_vectors[source].kx;
            vector.ky += weight * model.k_vectors[source].ky;
            total_weight += weight;
        }
        vector.kx /= total_weight;
        vector.ky /= total_weight;
        vector
    } else {
        model.k_vectors[frame]
    }
}
