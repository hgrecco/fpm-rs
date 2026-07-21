use approx::assert_abs_diff_eq;
use fpm_rs::metrics::complex_field::{
    ComplexAlignment, ComplexMetricError, amplitude_nrmse, bias, correlation, mae, mse, nrmse,
    relative_l1, rmse,
};
use ndarray::{Array2, array, s};
use num_complex::{Complex32, Complex64};

const TOLERANCE: f64 = 1e-12;

fn assert_complex_close(actual: Complex64, expected: Complex64, epsilon: f64) {
    assert_abs_diff_eq!(actual.re, expected.re, epsilon = epsilon);
    assert_abs_diff_eq!(actual.im, expected.im, epsilon = epsilon);
}

fn reference() -> Array2<Complex64> {
    array![[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]]
}

#[test]
fn identical_arrays_produce_zero_error() {
    let reference = reference();

    assert_complex_close(
        bias(
            reference.view(),
            reference.view(),
            None,
            ComplexAlignment::None,
        )
        .unwrap(),
        Complex64::default(),
        TOLERANCE,
    );
    assert_abs_diff_eq!(
        mae(
            reference.view(),
            reference.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        mse(
            reference.view(),
            reference.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        rmse(
            reference.view(),
            reference.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        relative_l1(
            reference.view(),
            reference.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            reference.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        amplitude_nrmse(
            reference.view(),
            reference.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
    assert_complex_close(
        correlation(
            reference.view(),
            reference.view(),
            None,
            ComplexAlignment::None,
        )
        .unwrap(),
        Complex64::new(1.0, 0.0),
        TOLERANCE,
    );
}

#[test]
fn known_complex_offset_has_expected_bias_and_errors() {
    let reference = array![[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]];
    let offset = Complex64::new(1.0, 2.0);
    let estimate = reference.mapv(|value| value + offset);

    assert_complex_close(
        bias(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::None,
        )
        .unwrap(),
        offset,
        TOLERANCE,
    );
    assert_abs_diff_eq!(
        mae(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        5.0_f64.sqrt(),
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        mse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        5.0,
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        rmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        5.0_f64.sqrt(),
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        relative_l1(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        2.0 * 5.0_f64.sqrt() / 3.0,
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        2.0_f64.sqrt(),
        epsilon = TOLERANCE
    );
}

#[test]
fn global_phase_removes_a_constant_phase_rotation() {
    let reference = reference();
    let phase = Complex64::from_polar(1.0, 0.7);
    let estimate = reference.mapv(|value| value * phase);

    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::GlobalPhase
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
}

#[test]
fn global_phase_preserves_amplitude_scaling() {
    let reference = reference();
    let phase = Complex64::from_polar(1.0, 0.7);
    let estimate = reference.mapv(|value| 2.0 * phase * value);

    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::GlobalPhase
        )
        .unwrap(),
        1.0,
        epsilon = TOLERANCE
    );
    assert_abs_diff_eq!(
        amplitude_nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::GlobalPhase
        )
        .unwrap(),
        1.0,
        epsilon = TOLERANCE
    );
}

#[test]
fn scale_removes_a_positive_scale_factor() {
    let reference = reference();
    let estimate = reference.mapv(|value| 2.0 * value);

    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::Scale
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
}

#[test]
fn scale_does_not_remove_phase_rotation() {
    let reference = reference();
    let estimate = reference.mapv(|value| Complex64::i() * value);

    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::Scale
        )
        .unwrap(),
        1.0,
        epsilon = TOLERANCE
    );
}

#[test]
fn complex_gain_removes_scale_and_phase() {
    let reference = reference();
    let estimate = reference.mapv(|value| 2.0 * Complex64::i() * value);

    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::ComplexGain
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
}

#[test]
fn no_alignment_preserves_scale_and_phase_differences() {
    let reference = reference();
    let estimate = reference.mapv(|value| 2.0 * Complex64::i() * value);

    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::None
        )
        .unwrap(),
        5.0_f64.sqrt(),
        epsilon = TOLERANCE
    );
}

#[test]
fn mask_controls_alignment_fit_and_metric_evaluation() {
    let reference = array![[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]];
    let estimate = array![[Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)]];
    let mask = array![[true, false]];

    assert_abs_diff_eq!(
        nrmse(
            reference.view(),
            estimate.view(),
            Some(mask.view()),
            ComplexAlignment::Scale
        )
        .unwrap(),
        0.0,
        epsilon = TOLERANCE
    );
    assert!(
        nrmse(
            reference.view(),
            estimate.view(),
            None,
            ComplexAlignment::Scale
        )
        .unwrap()
            > 0.0
    );
}

#[test]
fn f32_and_f64_inputs_are_consistent() {
    let reference32 = array![[Complex32::new(1.0, 2.0), Complex32::new(-0.5, 0.25)]];
    let estimate32 = array![[Complex32::new(0.75, 2.5), Complex32::new(-0.25, 0.5)]];
    let reference64 = reference32.mapv(|value| Complex64::new(value.re as f64, value.im as f64));
    let estimate64 = estimate32.mapv(|value| Complex64::new(value.re as f64, value.im as f64));

    let result32 = nrmse(
        reference32.view(),
        estimate32.view(),
        None,
        ComplexAlignment::GlobalPhase,
    )
    .unwrap();
    let result64 = nrmse(
        reference64.view(),
        estimate64.view(),
        None,
        ComplexAlignment::GlobalPhase,
    )
    .unwrap();
    assert_abs_diff_eq!(result32, result64, epsilon = 1e-7);
}

#[test]
fn correlation_uses_conjugated_reference_convention() {
    let reference = reference();
    let estimate = reference.mapv(|value| Complex64::i() * value);

    let value = correlation(
        reference.view(),
        estimate.view(),
        None,
        ComplexAlignment::None,
    )
    .unwrap();
    assert_complex_close(value, Complex64::i(), TOLERANCE);
}

#[test]
fn zero_normalization_denominators_are_errors() {
    let zero = Array2::from_elem((1, 2), Complex64::default());
    let nonzero = Array2::from_elem((1, 2), Complex64::new(1.0, 0.0));

    assert!(matches!(
        nrmse(zero.view(), nonzero.view(), None, ComplexAlignment::None),
        Err(ComplexMetricError::ZeroReferenceNormalization { metric: "nrmse" })
    ));
    assert!(matches!(
        correlation(nonzero.view(), zero.view(), None, ComplexAlignment::None),
        Err(ComplexMetricError::ZeroEstimateNormalization {
            metric: "correlation"
        })
    ));
}

#[test]
fn degenerate_alignment_inputs_are_errors() {
    let reference = Array2::from_elem((1, 2), Complex64::new(1.0, 0.0));
    let zero_estimate = Array2::from_elem((1, 2), Complex64::default());
    for alignment in [
        ComplexAlignment::GlobalPhase,
        ComplexAlignment::Scale,
        ComplexAlignment::ComplexGain,
    ] {
        assert!(matches!(
            mae(reference.view(), zero_estimate.view(), None, alignment),
            Err(ComplexMetricError::DegenerateAlignment { .. })
        ));
    }

    let orthogonal_estimate = array![[Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]];
    assert!(matches!(
        mae(
            reference.view(),
            orthogonal_estimate.view(),
            None,
            ComplexAlignment::GlobalPhase
        ),
        Err(ComplexMetricError::DegenerateAlignment { .. })
    ));
}

#[test]
fn shape_mask_and_empty_mask_validation_return_errors() {
    let reference = Array2::from_elem((1, 2), Complex64::new(1.0, 0.0));
    let wrong_shape = Array2::from_elem((2, 1), Complex64::new(1.0, 0.0));
    assert!(matches!(
        mae(
            reference.view(),
            wrong_shape.view(),
            None,
            ComplexAlignment::None
        ),
        Err(ComplexMetricError::ShapeMismatch { .. })
    ));

    let wrong_mask = Array2::from_elem((2, 1), true);
    assert!(matches!(
        mae(
            reference.view(),
            reference.view(),
            Some(wrong_mask.view()),
            ComplexAlignment::None
        ),
        Err(ComplexMetricError::MaskShapeMismatch { .. })
    ));

    let empty_mask = Array2::from_elem((1, 2), false);
    assert!(matches!(
        mae(
            reference.view(),
            reference.view(),
            Some(empty_mask.view()),
            ComplexAlignment::None
        ),
        Err(ComplexMetricError::EmptyMask)
    ));
}

#[test]
fn non_finite_complex_components_are_errors() {
    let finite = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));
    let non_finite_reference = array![[Complex64::new(f64::NAN, 0.0)]];
    assert!(matches!(
        mae(
            non_finite_reference.view(),
            finite.view(),
            None,
            ComplexAlignment::None
        ),
        Err(ComplexMetricError::NonFinite { input: "reference" })
    ));

    let non_finite_estimate = array![[Complex64::new(0.0, f64::INFINITY)]];
    assert!(matches!(
        mae(
            finite.view(),
            non_finite_estimate.view(),
            None,
            ComplexAlignment::None
        ),
        Err(ComplexMetricError::NonFinite { input: "estimate" })
    ));
}

#[test]
fn complex_metrics_accept_matching_transposed_and_stepped_layouts() {
    let reference = Array2::from_shape_fn((4, 6), |(row, column)| {
        Complex64::new((row * 6 + column + 1) as f64, row as f64 - column as f64)
    });
    let estimate = reference.mapv(|value| value + Complex64::new(0.25, -0.5));
    let mask = Array2::from_shape_fn((4, 6), |(row, column)| (row + column) % 3 != 0);
    let stepped_reference = reference.slice(s![..;2, ..;2]);
    let stepped_estimate = estimate.slice(s![..;2, ..;2]);
    let stepped_mask = mask.slice(s![..;2, ..;2]);
    let expected = mse(
        stepped_reference.to_owned().view(),
        stepped_estimate.to_owned().view(),
        Some(stepped_mask.to_owned().view()),
        ComplexAlignment::None,
    )
    .unwrap();
    assert_abs_diff_eq!(
        mse(
            stepped_reference,
            stepped_estimate,
            Some(stepped_mask),
            ComplexAlignment::None,
        )
        .unwrap(),
        expected,
        epsilon = 1e-14
    );
    assert_abs_diff_eq!(
        mse(
            reference.t(),
            estimate.t(),
            Some(mask.t()),
            ComplexAlignment::None,
        )
        .unwrap(),
        mse(
            reference.view(),
            estimate.view(),
            Some(mask.view()),
            ComplexAlignment::None,
        )
        .unwrap(),
        epsilon = 1e-14
    );
}
