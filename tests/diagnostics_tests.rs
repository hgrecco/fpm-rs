use approx::assert_abs_diff_eq;
use fpm_rs::diagnostics::{LossType, loss};

#[test]
fn loss_modes_match_known_scalar_values() {
    let predicted = [4.0];
    let measured = [1.0];
    assert_abs_diff_eq!(
        loss(&predicted, &measured, LossType::AmplitudeMse).unwrap(),
        1.0
    );
    assert_abs_diff_eq!(
        loss(&predicted, &measured, LossType::IntensityMse).unwrap(),
        9.0
    );
    assert_abs_diff_eq!(
        loss(&predicted, &measured, LossType::HuberAmplitude).unwrap(),
        0.5
    );
    assert_abs_diff_eq!(
        loss(
            &predicted,
            &measured,
            LossType::PoissonNegativeLogLikelihood
        )
        .unwrap(),
        4.0 - 4.0_f64.ln()
    );
}

#[test]
fn loss_rejects_empty_mismatched_and_non_finite_inputs() {
    assert!(loss(&[], &[], LossType::AmplitudeMse).is_err());
    assert!(loss(&[1.0], &[1.0, 2.0], LossType::AmplitudeMse).is_err());
    assert!(loss(&[f64::NAN], &[1.0], LossType::AmplitudeMse).is_err());
}
