use approx::assert_abs_diff_eq;
use fpm_rs::metrics::intensity::{
    amplitude_nrmse, bias, compare_intensity, correlation, fitted_gain, mae, mean_poisson_deviance,
    mse, nrmse, poisson_deviance, psnr, relative_l1, rmse, ssim, stats,
};
use ndarray::{Array2, s};

#[test]
fn intensity_facade_exposes_single_and_comparison_metrics() {
    let reference = Array2::from_shape_fn((11, 11), |(row, column)| (row * 11 + column + 1) as f64);
    let estimate = reference.mapv(|value| 2.0 * value);
    let valid_mask = Array2::from_elem(reference.dim(), true);

    let image_stats = stats(reference.as_slice().unwrap(), None).unwrap();
    assert_abs_diff_eq!(image_stats.sum, reference.sum(), epsilon = 1e-12);

    let comparison = compare_intensity(
        reference.view(),
        estimate.view(),
        Some(valid_mask.view()),
        None,
    )
    .unwrap();
    assert_abs_diff_eq!(comparison.reference_sum, reference.sum(), epsilon = 1e-12);
    assert_abs_diff_eq!(comparison.estimate_sum, estimate.sum(), epsilon = 1e-12);

    assert!(bias(reference.view(), estimate.view(), None).unwrap() > 0.0);
    assert!(mae(reference.view(), estimate.view(), None).unwrap() > 0.0);
    assert!(mse(reference.view(), estimate.view(), None).unwrap() > 0.0);
    assert!(rmse(reference.view(), estimate.view(), None).unwrap() > 0.0);
    assert_abs_diff_eq!(
        relative_l1(reference.view(), estimate.view(), None).unwrap(),
        1.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        nrmse(reference.view(), estimate.view(), None).unwrap(),
        1.0,
        epsilon = 1e-12
    );
    assert!(amplitude_nrmse(reference.view(), estimate.view(), None).unwrap() > 0.0);
    assert_abs_diff_eq!(
        correlation(reference.view(), estimate.view(), None).unwrap(),
        1.0,
        epsilon = 1e-12
    );
    assert!(
        psnr(reference.view(), estimate.view(), None, 255.0)
            .unwrap()
            .is_finite()
    );
    assert!(
        ssim(reference.view(), estimate.view(), None, 255.0)
            .unwrap()
            .is_finite()
    );
    assert!(poisson_deviance(reference.view(), estimate.view(), None, 1e-12).unwrap() > 0.0);
    assert!(mean_poisson_deviance(reference.view(), estimate.view(), None, 1e-12).unwrap() > 0.0);
    assert_abs_diff_eq!(
        fitted_gain(reference.view(), estimate.view(), None).unwrap(),
        2.0,
        epsilon = 1e-12
    );
}

#[test]
fn pure_comparison_serialization_uses_reference_and_estimate_keys() {
    let reference = Array2::from_elem((1, 2), 1.0);
    let estimate = Array2::from_elem((1, 2), 2.0);
    let metrics = compare_intensity(reference.view(), estimate.view(), None, None).unwrap();

    let value = serde_json::to_value(metrics).unwrap();
    assert_eq!(value["reference_sum"], 2.0);
    assert_eq!(value["estimate_sum"], 4.0);
    assert!(value.get("measured_sum").is_none());
    assert!(value.get("predicted_sum").is_none());
}

#[test]
fn intensity_metrics_accept_matching_transposed_and_stepped_layouts() {
    let reference = Array2::from_shape_fn((4, 6), |(row, column)| (row * 6 + column + 1) as f64);
    let estimate = reference.mapv(|value| value + 0.5);
    let mask = Array2::from_shape_fn((4, 6), |(row, column)| (row + column) % 3 != 0);

    let stepped_reference = reference.slice(s![..;2, ..;2]);
    let stepped_estimate = estimate.slice(s![..;2, ..;2]);
    let stepped_mask = mask.slice(s![..;2, ..;2]);
    let expected = mse(
        stepped_reference.to_owned().view(),
        stepped_estimate.to_owned().view(),
        Some(stepped_mask.to_owned().view()),
    )
    .unwrap();
    assert_abs_diff_eq!(
        mse(stepped_reference, stepped_estimate, Some(stepped_mask)).unwrap(),
        expected,
        epsilon = 1e-14
    );
    assert_abs_diff_eq!(
        mse(reference.t(), estimate.t(), Some(mask.t())).unwrap(),
        mse(reference.view(), estimate.view(), Some(mask.view())).unwrap(),
        epsilon = 1e-14
    );
}
