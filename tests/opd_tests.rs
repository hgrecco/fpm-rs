use std::f64::consts::TAU;

use fpm_rs::{
    Complex64,
    reconstruction::{PhaseReference, SyntheticWavelengthUnwrapper},
};
use ndarray::{Array2, s};

fn fields(opd: &Array2<f64>, wavelengths: &[f64], pistons: &[f64]) -> Vec<Array2<Complex64>> {
    wavelengths
        .iter()
        .zip(pistons)
        .map(|(&wavelength, &piston)| {
            opd.mapv(|d| Complex64::from_polar(0.8, TAU * d / wavelength + piston))
        })
        .collect()
}

#[test]
fn beat_phases_recover_discontinuous_signed_opd_and_orders() {
    let wavelengths = [500e-9, 550e-9];
    let pistons = [1.1, -2.3];
    let expected = Array2::from_shape_vec(
        (2, 4),
        vec![
            -1.6e-6, 0.0, 2.8e-6, 1.2e-6, -0.4e-6, 2.3e-6, 0.71e-6, -1.4e-6,
        ],
    )
    .unwrap();
    let objects = fields(&expected, &wavelengths, &pistons);
    let views: Vec<_> = objects.iter().map(|v| v.view()).collect();
    let result = SyntheticWavelengthUnwrapper::new((-2e-6, 3e-6))
        .unwrap()
        .unwrap_fields(
            &views,
            &wavelengths,
            &PhaseReference::Offsets(pistons.to_vec()),
            None,
        )
        .unwrap();
    assert!((result.wavelength_ladder_m[0] - 5.5e-6).abs() < 1e-19);
    assert!(result.valid_mask.iter().all(|&v| v == 1));
    for (&truth, &actual) in expected.iter().zip(&result.opd_m) {
        assert!((truth - actual).abs() < 1e-18);
    }
    for (channel, &wavelength) in wavelengths.iter().enumerate() {
        for ((&opd, &order), &field) in result
            .opd_m
            .iter()
            .zip(&result.fringe_orders[channel])
            .zip(&objects[channel])
        {
            let phase = (field * Complex64::from_polar(1.0, -pistons[channel])).arg();
            assert!((opd - wavelength * (phase / TAU + order as f64)).abs() < 1e-18);
        }
    }
    assert!(result.phase_residual_rad.iter().all(|&v| v < 1e-12));
}

#[test]
fn constant_known_reference_resolves_independent_pistons() {
    let wavelengths = [490e-9, 532e-9, 692e-9];
    let pistons = [2.4, -1.7, 0.6];
    let expected = Array2::from_shape_vec(
        (2, 3),
        vec![0.4e-6, 0.4e-6, 2.4e-6, 0.4e-6, -0.3e-6, 3.7e-6],
    )
    .unwrap();
    let objects = fields(&expected, &wavelengths, &pistons);
    let views: Vec<_> = objects.iter().map(|v| v.view()).collect();
    let reference = PhaseReference::Region {
        mask: Array2::from_shape_vec((2, 3), vec![1, 1, 0, 1, 0, 0]).unwrap(),
        opd_m: 0.4e-6,
    };
    let result = SyntheticWavelengthUnwrapper::new((-1e-6, 4e-6))
        .unwrap()
        .unwrap_fields(&views, &wavelengths, &reference, None)
        .unwrap();
    for (&truth, &actual) in expected.iter().zip(&result.opd_m) {
        assert!((truth - actual).abs() < 1e-18);
    }
    for (&expected, &actual) in pistons.iter().zip(&result.phase_offsets_rad) {
        assert!((expected - actual).abs() < 1e-13);
    }
}

#[test]
fn intermediate_beat_periods_refine_noisy_coarse_opd() {
    // Implementation-derived fixture: the close pair alone magnifies opposite
    // 0.01-radian phase errors. A third channel provides intermediate beat periods.
    let wavelengths = [500e-9, 501e-9, 550e-9];
    let noise = [0.01, -0.01, 0.0];
    let expected = Array2::from_shape_fn((4, 5), |(r, c)| 2e-6 + (r * 5 + c) as f64 * 1.17e-6);
    let objects = fields(&expected, &wavelengths, &noise);
    let views: Vec<_> = objects.iter().map(|v| v.view()).collect();
    let result = SyntheticWavelengthUnwrapper::new((-1e-6, 30e-6))
        .unwrap()
        .unwrap_fields(
            &views,
            &wavelengths,
            &PhaseReference::Offsets(vec![0.0; 3]),
            None,
        )
        .unwrap();
    assert!(result.wavelength_ladder_m[0] > 250e-6);
    assert!(result.valid_mask.iter().all(|&v| v == 1));
    for (&truth, &actual) in expected.iter().zip(&result.opd_m) {
        assert!((truth - actual).abs() < 2e-9, "{truth} {actual}");
    }
    assert!(result.phase_residual_rad.iter().all(|&v| v < 0.02));
}

#[test]
fn permutation_preserves_opd_and_reorders_channel_diagnostics() {
    let wavelengths = [490e-9, 532e-9, 692e-9];
    let pistons = [1.3, -0.5, 2.0];
    let expected = Array2::from_shape_fn((3, 4), |(r, c)| -0.3e-6 + (r * 4 + c) as f64 * 0.32e-6);
    let objects = fields(&expected, &wavelengths, &pistons);
    let unwrap = SyntheticWavelengthUnwrapper::new((-1e-6, 4e-6)).unwrap();
    let first = unwrap
        .unwrap_fields(
            &objects.iter().map(|v| v.view()).collect::<Vec<_>>(),
            &wavelengths,
            &PhaseReference::Offsets(pistons.to_vec()),
            None,
        )
        .unwrap();
    let order = [2, 0, 1];
    let second = unwrap
        .unwrap_fields(
            &order.iter().map(|&i| objects[i].view()).collect::<Vec<_>>(),
            &order.map(|i| wavelengths[i]),
            &PhaseReference::Offsets(order.map(|i| pistons[i]).to_vec()),
            None,
        )
        .unwrap();
    for (&a, &b) in first.opd_m.iter().zip(&second.opd_m) {
        assert!((a - b).abs() < 1e-18);
    }
    for (i, &j) in order.iter().enumerate() {
        assert_eq!(first.fringe_orders[j], second.fringe_orders[i]);
    }
}

#[test]
fn invalid_pixels_remain_masked_and_reference_uses_only_usable_pixels() {
    let wavelengths = [500e-9, 550e-9];
    let expected = Array2::from_elem((2, 3), 1.2e-6);
    let mut objects = fields(&expected, &wavelengths, &[0.2, -0.3]);
    objects[0][(0, 1)] = Complex64::new(0.0, 0.0);
    objects[1][(1, 1)] *= 0.01;
    let mask = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 1, 1, 1]).unwrap();
    let mut unwrap = SyntheticWavelengthUnwrapper::new((-1e-6, 4e-6)).unwrap();
    unwrap.minimum_amplitude = 0.1;
    let result = unwrap
        .unwrap_fields(
            &objects.iter().map(|v| v.view()).collect::<Vec<_>>(),
            &wavelengths,
            &PhaseReference::Region {
                mask: Array2::ones((2, 3)),
                opd_m: 1.2e-6,
            },
            Some(mask.view()),
        )
        .unwrap();
    assert_eq!(
        result.valid_mask,
        Array2::from_shape_vec((2, 3), vec![1, 0, 0, 1, 0, 1]).unwrap()
    );
    for (&valid, &value) in result.valid_mask.iter().zip(&result.opd_m) {
        if valid == 0 {
            assert!(value.is_nan());
        } else {
            assert!((value - 1.2e-6).abs() < 1e-18);
        }
    }
}

#[test]
fn interval_selects_branch_and_outside_pixels_are_invalid() {
    let wavelengths = [500e-9, 550e-9];
    let expected = Array2::from_shape_vec((1, 2), vec![7.2e-6, 9.0e-6]).unwrap();
    let objects = fields(&expected, &wavelengths, &[0.0; 2]);
    let result = SyntheticWavelengthUnwrapper::new((6e-6, 8e-6))
        .unwrap()
        .unwrap_fields(
            &objects.iter().map(|v| v.view()).collect::<Vec<_>>(),
            &wavelengths,
            &PhaseReference::Offsets(vec![0.0; 2]),
            None,
        )
        .unwrap();
    assert!((result.opd_m[(0, 0)] - 7.2e-6).abs() < 1e-18);
    assert_eq!(result.valid_mask[(0, 1)], 0);
}

#[test]
fn roundoff_preserves_half_open_interval_boundaries() {
    let wavelengths = [500e-9, 550e-9];
    let expected = Array2::from_shape_vec((1, 3), vec![0.0, 4e-6, 3.9e-6]).unwrap();
    let objects = fields(&expected, &wavelengths, &[1.1, -2.3]);
    let reference = PhaseReference::Region {
        mask: Array2::from_shape_vec((1, 3), vec![1, 0, 0]).unwrap(),
        opd_m: 0.0,
    };
    let result = SyntheticWavelengthUnwrapper::new((0.0, 4e-6))
        .unwrap()
        .unwrap_fields(
            &objects.iter().map(|v| v.view()).collect::<Vec<_>>(),
            &wavelengths,
            &reference,
            None,
        )
        .unwrap();
    assert_eq!(
        result.valid_mask,
        Array2::from_shape_vec((1, 3), vec![1, 0, 1]).unwrap()
    );
    assert_eq!(result.opd_m[(0, 0)], 0.0);
}

#[test]
fn inconsistent_phases_can_be_rejected_by_explicit_residual_limit() {
    let wavelengths = [490e-9, 532e-9, 692e-9];
    let objects = fields(
        &Array2::from_elem((1, 1), 1.2e-6),
        &wavelengths,
        &[0.0, 0.0, 0.4],
    );
    let views: Vec<_> = objects.iter().map(|v| v.view()).collect();
    let mut unwrap = SyntheticWavelengthUnwrapper::new((-1e-6, 4e-6)).unwrap();
    let reference = PhaseReference::Offsets(vec![0.0; 3]);
    assert_eq!(
        unwrap
            .unwrap_fields(&views, &wavelengths, &reference, None)
            .unwrap()
            .valid_mask[(0, 0)],
        1
    );
    unwrap.max_phase_residual_rad = Some(0.01);
    assert_eq!(
        unwrap
            .unwrap_fields(&views, &wavelengths, &reference, None)
            .unwrap()
            .valid_mask[(0, 0)],
        0
    );
}

#[test]
fn invalid_wavelengths_shapes_references_and_layouts_return_errors() {
    let wavelengths = [500e-9, 550e-9];
    let objects = fields(&Array2::from_elem((2, 3), 1.2e-6), &wavelengths, &[0.0; 2]);
    let views: Vec<_> = objects.iter().map(|v| v.view()).collect();
    let reference = PhaseReference::Offsets(vec![0.0; 2]);
    let unwrap = SyntheticWavelengthUnwrapper::new((-1e-6, 4e-6)).unwrap();
    let near_duplicate = [wavelengths[0], f64::from_bits(wavelengths[0].to_bits() + 1)];
    assert!(
        unwrap
            .unwrap_fields(&views, &near_duplicate, &reference, None)
            .is_err()
    );
    for invalid in [[500e-9, 500e-9], [0.0, 550e-9], [f64::NAN, 550e-9]] {
        assert!(
            unwrap
                .unwrap_fields(&views, &invalid, &reference, None)
                .is_err()
        );
    }
    assert!(
        SyntheticWavelengthUnwrapper::new((0.0, 6e-6))
            .unwrap()
            .unwrap_fields(&views, &wavelengths, &reference, None)
            .is_err()
    );
    assert!(
        unwrap
            .unwrap_fields(&views[..1], &wavelengths, &reference, None)
            .is_err()
    );
    assert!(
        unwrap
            .unwrap_fields(
                &views,
                &wavelengths,
                &PhaseReference::Offsets(vec![0.0]),
                None
            )
            .is_err()
    );
    assert!(
        unwrap
            .unwrap_fields(
                &views,
                &wavelengths,
                &PhaseReference::Region {
                    mask: Array2::zeros((2, 3)),
                    opd_m: 0.0
                },
                None
            )
            .is_err()
    );
    assert!(
        unwrap
            .unwrap_fields(
                &views,
                &wavelengths,
                &reference,
                Some(Array2::<u8>::ones((3, 2)).view())
            )
            .is_err()
    );
    assert!(
        unwrap
            .unwrap_fields(
                &[objects[0].slice(s![..,..;-1]), objects[1].view()],
                &wavelengths,
                &reference,
                None
            )
            .is_err()
    );
    assert!(
        unwrap
            .unwrap_fields(
                &[Array2::<Complex64>::zeros((0, 3)).view(); 2],
                &wavelengths,
                &reference,
                None
            )
            .is_err()
    );
    let mut nonfinite = objects.clone();
    nonfinite[0][(0, 0)].re = f64::NAN;
    assert!(
        unwrap
            .unwrap_fields(
                &nonfinite.iter().map(|v| v.view()).collect::<Vec<_>>(),
                &wavelengths,
                &reference,
                None
            )
            .is_err()
    );
    assert!(SyntheticWavelengthUnwrapper::new((1.0, 0.0)).is_err());
}
