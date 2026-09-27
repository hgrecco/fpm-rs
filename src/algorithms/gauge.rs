//! Internal object/pupil gauge projection for blind reconstruction.

use num_complex::Complex64;

use crate::{
    Result,
    error::Error,
    measurements::MeasurementRead,
    model::FourierOffset,
    reconstruction::{ReconstructionProblem, ReconstructionState},
};

#[derive(Clone, Copy)]
enum Axis {
    Row,
    Column,
}

/// Canonicalizes the exact object/pupil gauges of the current numerical model.
pub(crate) fn canonicalize_object_pupil<M: MeasurementRead>(
    problem: &ReconstructionProblem<M>,
    state: &mut ReconstructionState,
) -> Result<()> {
    let reference = problem.model.pupil();
    if state.pupil.shape() != reference.shape()
        || state.pupil.support.as_slice() != reference.support.as_slice()
    {
        return Err(Error::InvalidModel(
            "recovered pupil shape or support differs from the compiled reference pupil".into(),
        ));
    }

    let support_count = reference
        .support
        .as_slice()
        .iter()
        .filter(|&&inside| inside != 0)
        .count();
    if support_count == 0 {
        return Err(Error::InvalidModel(
            "compiled reference pupil support is empty".into(),
        ));
    }
    let tolerance = 64.0 * f64::EPSILON * support_count as f64;

    let mut row_ramp_is_exact = true;
    let mut column_ramp_is_exact = true;
    for source in 0..problem.model.source_count() {
        let offset = state.effective_source_offset(&problem.model, source)?;
        row_ramp_is_exact &= FourierOffset::new(offset.row, 0.0).is_zero();
        column_ramp_is_exact &= FourierOffset::new(0.0, offset.column).is_zero();
    }

    let shape = state.pupil.shape();
    let row_slope = if row_ramp_is_exact {
        relative_phase_slope(state, reference, shape, Axis::Row, tolerance)
    } else {
        0.0
    };
    let column_slope = if column_ramp_is_exact {
        relative_phase_slope(state, reference, shape, Axis::Column, tolerance)
    } else {
        0.0
    };
    if row_slope.abs() > tolerance || column_slope.abs() > tolerance {
        apply_affine_correction(state, row_slope, column_slope);
    }

    normalize_pupil_scale_and_phase(state, reference.values.as_slice(), tolerance)?;
    normalize_object_phase(state, tolerance)?;
    state.object_real_space_cache = None;
    Ok(())
}

fn relative_phase_slope(
    state: &ReconstructionState,
    reference: &crate::model::Pupil,
    shape: (usize, usize),
    axis: Axis,
    tolerance: f64,
) -> f64 {
    let values = state.pupil.values.as_slice();
    let reference_values = reference.values.as_slice();
    let support = reference.support.as_slice();
    let mut resultant = Complex64::default();
    let mut total_weight = 0.0;
    let mut strongest = Complex64::default();
    let mut strongest_weight = 0.0;

    let (row_limit, column_limit) = match axis {
        Axis::Row => (shape.0.saturating_sub(1), shape.1),
        Axis::Column => (shape.0, shape.1.saturating_sub(1)),
    };
    for row in 0..row_limit {
        for column in 0..column_limit {
            let current = row * shape.1 + column;
            let next = match axis {
                Axis::Row => (row + 1) * shape.1 + column,
                Axis::Column => row * shape.1 + column + 1,
            };
            if support[current] == 0 || support[next] == 0 {
                continue;
            }
            let current_edge = values[next] * values[current].conj();
            let reference_edge = reference_values[next] * reference_values[current].conj();
            let contribution = current_edge * reference_edge.conj();
            let weight = contribution.norm();
            if !weight.is_finite() {
                continue;
            }
            resultant += contribution;
            total_weight += weight;
            if weight > strongest_weight {
                strongest = contribution;
                strongest_weight = weight;
            }
        }
    }

    if total_weight == 0.0 {
        return 0.0;
    }
    if resultant.norm() > tolerance * total_weight {
        resultant.arg()
    } else if strongest_weight > 0.0 {
        strongest.arg()
    } else {
        0.0
    }
}

fn apply_affine_correction(state: &mut ReconstructionState, row_slope: f64, column_slope: f64) {
    let pupil_shape = state.pupil.shape();
    let pupil_center = (pupil_shape.0 / 2, pupil_shape.1 / 2);
    for row in 0..pupil_shape.0 {
        let y = row as f64 - pupil_center.0 as f64;
        for column in 0..pupil_shape.1 {
            let x = column as f64 - pupil_center.1 as f64;
            let phase = -(row_slope * y + column_slope * x);
            state.pupil.values.as_slice_mut()[row * pupil_shape.1 + column] *=
                Complex64::from_polar(1.0, phase);
        }
    }

    let object_shape = state.object_spectrum.dim();
    let object_center = (object_shape.0 / 2, object_shape.1 / 2);
    for row in 0..object_shape.0 {
        let y = row as f64 - object_center.0 as f64;
        for column in 0..object_shape.1 {
            let x = column as f64 - object_center.1 as f64;
            let phase = row_slope * y + column_slope * x;
            state.object_spectrum.as_slice_mut()[row * object_shape.1 + column] *=
                Complex64::from_polar(1.0, phase);
        }
    }
}

fn normalize_pupil_scale_and_phase(
    state: &mut ReconstructionState,
    reference_values: &[Complex64],
    tolerance: f64,
) -> Result<()> {
    let support = state.pupil.support.as_slice();
    let values = state.pupil.values.as_slice();
    let mut reference_energy = 0.0;
    let mut current_energy = 0.0;
    for ((&reference, &current), &inside) in reference_values.iter().zip(values).zip(support) {
        if inside != 0 {
            reference_energy += reference.norm_sqr();
            current_energy += current.norm_sqr();
        }
    }
    if !reference_energy.is_finite() || reference_energy <= 0.0 {
        return Err(Error::InvalidModel(
            "compiled reference pupil has invalid supported energy".into(),
        ));
    }
    if !current_energy.is_finite() || current_energy <= 0.0 {
        return Err(Error::Numerical(
            "object/pupil gauge normalization requires positive finite recovered pupil energy"
                .into(),
        ));
    }

    let computed_scale = (reference_energy / current_energy).sqrt();
    if !computed_scale.is_finite() || computed_scale <= 0.0 {
        return Err(Error::Numerical(
            "object/pupil gauge normalization produced an invalid pupil scale".into(),
        ));
    }
    let scale = if (computed_scale - 1.0).abs() <= tolerance {
        1.0
    } else {
        computed_scale
    };

    let mut overlap = Complex64::default();
    let mut overlap_weight = 0.0;
    let mut strongest = Complex64::default();
    let mut strongest_weight = 0.0;
    for ((&reference, &current), &inside) in reference_values.iter().zip(values).zip(support) {
        if inside == 0 {
            continue;
        }
        let contribution = reference.conj() * current;
        let weight = contribution.norm();
        overlap += contribution;
        overlap_weight += weight;
        if weight > strongest_weight {
            strongest = contribution;
            strongest_weight = weight;
        }
    }
    let phase = if overlap_weight > 0.0 && overlap.norm() > tolerance * overlap_weight {
        overlap.arg()
    } else if strongest_weight > 0.0 {
        strongest.arg()
    } else {
        return Err(Error::Numerical(
            "object/pupil gauge normalization could not anchor recovered pupil phase".into(),
        ));
    };
    if !phase.is_finite() {
        return Err(Error::Numerical(
            "object/pupil gauge normalization produced a non-finite pupil phase".into(),
        ));
    }
    let phase = if phase.abs() <= tolerance { 0.0 } else { phase };
    let pupil_factor = Complex64::from_polar(scale, -phase);
    let object_factor = Complex64::new(1.0, 0.0) / pupil_factor;
    if pupil_factor != Complex64::new(1.0, 0.0) {
        for value in state.pupil.values.as_slice_mut() {
            *value *= pupil_factor;
        }
        for value in state.object_spectrum.as_slice_mut() {
            *value *= object_factor;
        }
    }
    Ok(())
}

fn normalize_object_phase(state: &mut ReconstructionState, tolerance: f64) -> Result<()> {
    let shape = state.object_spectrum.dim();
    let values = state.object_spectrum.as_slice();
    let mut strongest = Complex64::default();
    let mut strongest_magnitude = 0.0;
    for &value in values {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(Error::Numerical(
                "object/pupil gauge normalization requires a finite object spectrum".into(),
            ));
        }
        let magnitude = value.norm();
        if magnitude > strongest_magnitude {
            strongest = value;
            strongest_magnitude = magnitude;
        }
    }
    if strongest_magnitude == 0.0 {
        return Ok(());
    }
    let dc = values[(shape.0 / 2) * shape.1 + shape.1 / 2];
    let anchor = if dc.norm() > tolerance * strongest_magnitude {
        dc
    } else {
        strongest
    };
    let phase = anchor.arg();
    if !phase.is_finite() {
        return Err(Error::Numerical(
            "object/pupil gauge normalization produced a non-finite object phase".into(),
        ));
    }
    if phase.abs() <= tolerance {
        return Ok(());
    }
    let correction = Complex64::from_polar(1.0, -phase);
    for value in state.object_spectrum.as_slice_mut() {
        *value *= correction;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::{
        Complex64,
        experiment::KVector,
        measurements::MeasurementStack,
        model::{
            CropIndices, ForwardModel, FourierCrop, FourierOffset, ImagePlaneModel, Pupil, Sampling,
        },
        reconstruction::{ReconstructionProblem, ReconstructionState},
    };

    use super::canonicalize_object_pupil;

    fn problem(
        offset: FourierOffset,
        multiplexed: bool,
    ) -> ReconstructionProblem<MeasurementStack> {
        let low_shape = (5, 7);
        let high_shape = (10, 14);
        let pupil = Pupil::new(
            Array2::from_elem(low_shape, Complex64::new(1.0, 0.0)),
            Array2::from_elem(low_shape, 1_u8),
        )
        .unwrap();
        let mut model = ImagePlaneModel::new(
            vec![KVector::new(0.0, 0.0), KVector::new(1.0, 1.0)],
            pupil,
            CropIndices::new(vec![
                FourierCrop::new(3, 4, low_shape.0, low_shape.1),
                FourierCrop::new(4, 5, low_shape.0, low_shape.1),
            ]),
            Sampling::new(1.0, 0.5, 1.0, 1.0).unwrap(),
            low_shape,
            high_shape,
        )
        .unwrap()
        .with_subpixel_offsets(vec![offset; 2])
        .unwrap();
        if multiplexed {
            model = model
                .with_multiplexing(vec![vec![(0, 0.4), (1, 0.6)], vec![(0, 0.8), (1, 0.2)]])
                .unwrap();
        }
        let measurements = MeasurementStack::from_vec(
            vec![0.0; model.frame_count() * low_shape.0 * low_shape.1],
            low_shape,
            Vec::new(),
        )
        .unwrap();
        ReconstructionProblem::new(measurements, model).unwrap()
    }

    fn perturbed_state(
        problem: &ReconstructionProblem<MeasurementStack>,
        row_slope: f64,
        column_slope: f64,
    ) -> ReconstructionState {
        let mut state = ReconstructionState::initialize(problem).unwrap();
        let object_shape = state.object_spectrum.dim();
        let object_center = (object_shape.0 / 2, object_shape.1 / 2);
        for row in 0..object_shape.0 {
            let y = row as f64 - object_center.0 as f64;
            for column in 0..object_shape.1 {
                let x = column as f64 - object_center.1 as f64;
                let baseline = Complex64::new(
                    1.0 + 0.01 * (row * object_shape.1 + column) as f64,
                    0.002 * (row as f64 - column as f64),
                );
                let ramp = Complex64::from_polar(1.0, -(row_slope * y + column_slope * x));
                state.object_spectrum.as_slice_mut()[row * object_shape.1 + column] =
                    baseline * Complex64::from_polar(0.4, 0.37) * ramp;
            }
        }
        let dc = (object_shape.0 / 2) * object_shape.1 + object_shape.1 / 2;
        state.object_spectrum.as_slice_mut()[dc] =
            Complex64::new(2.0, 0.0) * Complex64::from_polar(0.4, 0.37);

        let pupil_shape = state.pupil.shape();
        let pupil_center = (pupil_shape.0 / 2, pupil_shape.1 / 2);
        for row in 0..pupil_shape.0 {
            let y = row as f64 - pupil_center.0 as f64;
            for column in 0..pupil_shape.1 {
                let x = column as f64 - pupil_center.1 as f64;
                state.pupil.values.as_slice_mut()[row * pupil_shape.1 + column] =
                    Complex64::from_polar(2.5, -0.21 + row_slope * y + column_slope * x);
            }
        }
        state
    }

    fn predictions(
        problem: &ReconstructionProblem<MeasurementStack>,
        state: &ReconstructionState,
    ) -> Vec<f64> {
        ForwardModel::new(&problem.model)
            .unwrap()
            .forward_intensity_stack(state.object_spectrum.ndarray_view(), &state.pupil, 1)
            .unwrap()
    }

    fn source_fields(
        problem: &ReconstructionProblem<MeasurementStack>,
        state: &ReconstructionState,
    ) -> Vec<Vec<Complex64>> {
        let forward = ForwardModel::new(&problem.model).unwrap();
        (0..problem.model.source_count())
            .map(|source| {
                forward
                    .forward_source_field(
                        state.object_spectrum.ndarray_view(),
                        &state.pupil,
                        source,
                    )
                    .unwrap()
                    .into_raw_vec_and_offset()
                    .0
            })
            .collect()
    }

    fn assert_fields_equal_up_to_unit_phase(before: &[Vec<Complex64>], after: &[Vec<Complex64>]) {
        for (before, after) in before.iter().zip(after) {
            let overlap = before
                .iter()
                .zip(after)
                .fold(Complex64::default(), |sum, (&before, &after)| {
                    sum + before.conj() * after
                });
            assert!(overlap.norm() > 0.0);
            let phase = overlap / overlap.norm();
            for (&before, &after) in before.iter().zip(after) {
                let scale = before.norm().max(after.norm()).max(1.0);
                assert!((after - phase * before).norm() <= 2e-12 * scale);
            }
        }
    }

    fn assert_predictions_equal(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            let scale = left.abs().max(right.abs()).max(1.0);
            assert!(
                (left - right).abs() <= 2e-12 * scale,
                "prediction changed from {left} to {right}"
            );
        }
    }

    #[test]
    fn integer_crops_canonicalize_all_gauges_without_changing_intensity() {
        let problem = problem(FourierOffset::default(), true);
        let mut state = perturbed_state(&problem, 0.17, -0.23);
        let before = predictions(&problem, &state);
        let before_fields = source_fields(&problem, &state);

        canonicalize_object_pupil(&problem, &mut state).unwrap();
        let after = predictions(&problem, &state);
        let after_fields = source_fields(&problem, &state);
        assert_predictions_equal(&before, &after);
        assert_fields_equal_up_to_unit_phase(&before_fields, &after_fields);

        for &value in state.pupil.values.as_slice() {
            assert!((value - Complex64::new(1.0, 0.0)).norm() < 2e-13);
        }
        let object_shape = state.object_spectrum.dim();
        let dc = state.object_spectrum.as_slice()
            [(object_shape.0 / 2) * object_shape.1 + object_shape.1 / 2];
        assert!(dc.re > 0.0);
        assert!(dc.im.abs() < 2e-13);

        let canonical_object = state.object_spectrum.clone();
        let canonical_pupil = state.pupil.clone();
        canonicalize_object_pupil(&problem, &mut state).unwrap();
        assert_eq!(state.object_spectrum, canonical_object);
        assert_eq!(state.pupil, canonical_pupil);
    }

    #[test]
    fn fractional_row_crops_preserve_row_ramp_and_remove_exact_column_ramp() {
        let problem = problem(FourierOffset::new(0.25, 0.0), false);
        let row_slope = 0.17;
        let mut state = perturbed_state(&problem, row_slope, -0.23);
        let before = predictions(&problem, &state);

        canonicalize_object_pupil(&problem, &mut state).unwrap();
        let after = predictions(&problem, &state);
        assert_predictions_equal(&before, &after);

        let shape = state.pupil.shape();
        let center = (shape.0 / 2, shape.1 / 2);
        let center_value = state.pupil.values.as_slice()[center.0 * shape.1 + center.1];
        let row_value = state.pupil.values.as_slice()[(center.0 + 1) * shape.1 + center.1];
        let column_value = state.pupil.values.as_slice()[center.0 * shape.1 + center.1 + 1];
        assert!(((row_value * center_value.conj()).arg() - row_slope).abs() < 2e-13);
        assert!((column_value * center_value.conj()).arg().abs() < 2e-13);
    }

    #[test]
    fn invalid_recovered_fields_fail_gauge_normalization() {
        let problem = problem(FourierOffset::default(), false);
        let mut state = perturbed_state(&problem, 0.0, 0.0);
        state.pupil.values.as_slice_mut().fill(Complex64::default());
        let pupil_error = canonicalize_object_pupil(&problem, &mut state).unwrap_err();
        assert!(matches!(pupil_error, crate::Error::Numerical(_)));

        let mut state = perturbed_state(&problem, 0.0, 0.0);
        state.object_spectrum.as_slice_mut()[0] = Complex64::new(f64::NAN, 0.0);
        let object_error = canonicalize_object_pupil(&problem, &mut state).unwrap_err();
        assert!(matches!(object_error, crate::Error::Numerical(_)));
    }
}
