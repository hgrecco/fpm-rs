//! Numerical diagnostics of the declared spectral exposure weights.
use crate::{Error, Result};
use ndarray::Array2;

/// Linear weight diagnostics in detector-row and stable channel order.
/// These omit spatial forward operators and do not prove nonlinear recoverability.
#[derive(Clone, Debug)]
pub struct SpectralMixingDiagnostics {
    /// Stable IDs corresponding to matrix columns.
    pub channel_ids: Vec<String>,
    /// Effective mode weights, shaped `(detector frames, channels)`.
    pub matrix: Array2<f64>,
    /// Descending singular values; length `min(frames, channels)`.
    pub singular_values: Vec<f64>,
    /// Number of singular values strictly above the absolute tolerance.
    pub rank: usize,
    /// Dimensionless threshold relative to the largest singular value.
    pub relative_tolerance: f64,
    /// Singular-value threshold in matrix-weight units.
    pub absolute_tolerance: f64,
    /// Largest/smallest singular value when columns have full numerical rank.
    /// None for deficient column rank or an unrepresentable binary64 ratio.
    pub condition_number: Option<f64>,
}

pub(super) fn diagnose(
    channel_ids: Vec<String>,
    matrix: Array2<f64>,
    tolerance: Option<f64>,
) -> Result<SpectralMixingDiagnostics> {
    let relative_tolerance =
        tolerance.unwrap_or(matrix.nrows().max(matrix.ncols()) as f64 * f64::EPSILON);
    if !relative_tolerance.is_finite()
        || relative_tolerance < 0.0
        || matrix.iter().any(|v| !v.is_finite())
    {
        return Err(Error::InvalidParameter {
            name: "mixing_tolerance",
            reason: "matrix and nonnegative relative tolerance must be finite".into(),
        });
    }
    let singular_values = singular_values(&matrix)?;
    let largest = singular_values.first().copied().unwrap_or(0.0);
    let absolute_tolerance = largest * relative_tolerance;
    if !absolute_tolerance.is_finite() {
        return Err(Error::InvalidParameter {
            name: "mixing_tolerance",
            reason: "absolute tolerance overflows binary64".into(),
        });
    }
    let rank = singular_values
        .iter()
        .filter(|&&s| s > absolute_tolerance)
        .count();
    let condition_number = if rank == matrix.ncols() {
        singular_values
            .last()
            .map(|s| largest / s)
            .filter(|v| v.is_finite())
    } else {
        None
    };
    Ok(SpectralMixingDiagnostics {
        channel_ids,
        matrix,
        singular_values,
        rank,
        relative_tolerance,
        absolute_tolerance,
        condition_number,
    })
}

fn singular_values(matrix: &Array2<f64>) -> Result<Vec<f64>> {
    let scale = matrix.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let count = matrix.nrows().min(matrix.ncols());
    if scale == 0.0 {
        return Ok(vec![0.0; count]);
    }
    let mut a = if matrix.nrows() >= matrix.ncols() {
        matrix.clone()
    } else {
        matrix.t().to_owned()
    };
    a.mapv_inplace(|v| v / scale);
    // Cyclic one-sided Jacobi, following the column-orthogonalization formulation
    // documented by LAPACK DGESVJ. Scaling avoids forming a poorly conditioned Gram matrix.
    let mut converged = false;
    for _ in 0..100 {
        let mut changed = false;
        for p in 0..count {
            for q in p + 1..count {
                let norm_p = a.column(p).iter().fold(0.0_f64, |n, &v| n.hypot(v));
                let norm_q = a.column(q).iter().fold(0.0_f64, |n, &v| n.hypot(v));
                if norm_p == 0.0 || norm_q == 0.0 {
                    continue;
                }
                let correlation = a
                    .column(p)
                    .iter()
                    .zip(a.column(q))
                    .map(|(&x, &y)| (x / norm_p) * (y / norm_q))
                    .sum::<f64>();
                if correlation.abs() <= 8.0 * f64::EPSILON {
                    continue;
                }
                let pair_scale = norm_p.max(norm_q);
                let p_scaled = norm_p / pair_scale;
                let q_scaled = norm_q / pair_scale;
                let delta = (q_scaled * q_scaled - p_scaled * p_scaled) / 2.0;
                let cross = correlation * p_scaled * q_scaled;
                // This form avoids dividing by a tiny off-diagonal entry and
                // overflowing the tangent parameter for strongly unequal columns.
                let t = if delta == 0.0 {
                    1.0
                } else {
                    cross / (delta + delta.hypot(cross).copysign(delta))
                };
                let c = 1.0 / t.hypot(1.0);
                let s = c * t;
                for row in 0..a.nrows() {
                    let x = a[[row, p]];
                    let y = a[[row, q]];
                    a[[row, p]] = c * x - s * y;
                    a[[row, q]] = s * x + c * y;
                }
                changed = true;
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(Error::InvalidParameter {
            name: "mixing_matrix",
            reason: "Jacobi singular values did not converge within 100 sweeps".into(),
        });
    }
    let mut values: Vec<f64> = a
        .columns()
        .into_iter()
        .map(|column| column.iter().fold(0.0_f64, |norm, &v| norm.hypot(v)) * scale)
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "mixing_matrix",
            reason: "singular values overflow binary64".into(),
        });
    }
    values.sort_by(|a, b| b.total_cmp(a));
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn rank_condition_scale_and_wide_matrix() {
        for scale in [1e-200, 1.0, 1e200] {
            let d = diagnose(
                vec![],
                array![[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]] * scale,
                None,
            )
            .unwrap();
            assert_eq!(d.rank, 2);
            assert!((d.condition_number.unwrap() - 3.0).abs() < 1e-13);
            let d = diagnose(
                vec![],
                array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]] * scale,
                None,
            )
            .unwrap();
            assert_eq!(d.rank, 1);
            assert_eq!(d.condition_number, None);
        }
        let tiny = diagnose(vec![], array![[1.0, 1e-200], [0.0, 1e-200]], None).unwrap();
        assert_eq!(tiny.rank, 1);
        assert!((tiny.singular_values[1] / 1e-200 - 1.0).abs() < 1e-14);
        let a = array![[1.0, 1.0], [0.0, 1e-10]];
        let d = diagnose(vec![], a.clone(), None).unwrap();
        assert_eq!(d.rank, 2);
        assert!((d.condition_number.unwrap() / 2e10 - 1.0).abs() < 1e-10);
        assert_eq!(diagnose(vec![], a, Some(1e-8)).unwrap().rank, 1);
        assert!(diagnose(vec![], array![[1.0]], Some(-1.0)).is_err());
    }
}
