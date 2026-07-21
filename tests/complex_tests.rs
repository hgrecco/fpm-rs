use approx::assert_abs_diff_eq;
use fpm_rs::{Complex64, complex};
use ndarray::{Array2, s};

#[test]
fn pointwise_complex_utilities_accept_arbitrary_matching_layouts() {
    let field = Array2::from_shape_fn((4, 6), |(row, column)| {
        Complex64::from_polar(1.0 + row as f64, 0.1 * column as f64)
    });
    let strided = field.slice(s![..;2, ..;2]);
    let amplitude = complex::amplitude(strided);
    let phase = complex::phase(strided);
    let rebuilt = complex::from_amplitude_phase(amplitude.t(), phase.t()).unwrap();

    assert_eq!(rebuilt.dim(), (3, 2));
    for (&actual, &expected) in rebuilt.iter().zip(strided.t().iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-14);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-14);
    }
}

#[test]
fn amplitude_phase_construction_rejects_mismatched_shapes() {
    let amplitude = Array2::ones((2, 3));
    let phase = Array2::zeros((3, 2));
    assert!(complex::from_amplitude_phase(amplitude.view(), phase.view()).is_err());
}
