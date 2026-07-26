use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use ndarray::Array2;
use num_complex::Complex64;

use crate::{Error, Result, array_layout::checked_len_2d};

const MAGIC: &[u8; 6] = b"\x93NUMPY";

pub(crate) fn write_complex2(
    path: &Path,
    values: ndarray::ArrayView2<'_, Complex64>,
) -> Result<()> {
    let mut file = File::create(path)?;
    write_header(&mut file, "<c16", &[values.nrows(), values.ncols()])?;
    for value in values {
        file.write_all(&value.re.to_le_bytes())?;
        file.write_all(&value.im.to_le_bytes())?;
    }
    file.sync_all()?;
    Ok(())
}

pub(crate) fn write_f64_2(path: &Path, values: ndarray::ArrayView2<'_, f64>) -> Result<()> {
    let mut file = File::create(path)?;
    write_header(&mut file, "<f8", &[values.nrows(), values.ncols()])?;
    for &value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.sync_all()?;
    Ok(())
}

pub(crate) fn write_f64_1(path: &Path, values: &[f64]) -> Result<()> {
    let mut file = File::create(path)?;
    write_header(&mut file, "<f8", &[values.len()])?;
    for &value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.sync_all()?;
    Ok(())
}

pub(crate) fn write_u8_2(path: &Path, values: ndarray::ArrayView2<'_, u8>) -> Result<()> {
    let mut file = File::create(path)?;
    write_header(&mut file, "|u1", &[values.nrows(), values.ncols()])?;
    for &value in values {
        file.write_all(&[value])?;
    }
    file.sync_all()?;
    Ok(())
}

pub(crate) fn read_complex2(
    path: &Path,
    role: &str,
    expected_shape: (usize, usize),
) -> Result<Array2<Complex64>> {
    let (header, bytes) = read(path, role)?;
    validate_header(&header, role, "<c16", &[expected_shape.0, expected_shape.1])?;
    let length = checked_len_2d(expected_shape)?;
    let expected_bytes = length.checked_mul(16).ok_or_else(|| Error::ShapeOverflow {
        shape: vec![expected_shape.0, expected_shape.1, 16],
    })?;
    if bytes.len() != expected_bytes {
        return Err(Error::InvalidArrayShape {
            role: role.into(),
            reason: format!(
                "payload has {} bytes, expected {expected_bytes}",
                bytes.len()
            ),
        });
    }
    let mut values = Vec::with_capacity(length);
    for value in bytes.chunks_exact(16) {
        let real =
            f64::from_le_bytes(
                value[..8]
                    .try_into()
                    .map_err(|_| Error::InvalidArrayShape {
                        role: role.into(),
                        reason: "truncated complex real component".into(),
                    })?,
            );
        let imaginary =
            f64::from_le_bytes(
                value[8..]
                    .try_into()
                    .map_err(|_| Error::InvalidArrayShape {
                        role: role.into(),
                        reason: "truncated complex imaginary component".into(),
                    })?,
            );
        values.push(Complex64::new(real, imaginary));
    }
    Ok(Array2::from_shape_vec(expected_shape, values)?)
}

pub(crate) fn read_u8_2(
    path: &Path,
    role: &str,
    expected_shape: (usize, usize),
) -> Result<Array2<u8>> {
    let (header, bytes) = read(path, role)?;
    validate_header(&header, role, "|u1", &[expected_shape.0, expected_shape.1])?;
    let expected = checked_len_2d(expected_shape)?;
    if bytes.len() != expected {
        return Err(Error::InvalidArrayShape {
            role: role.into(),
            reason: format!("payload has {} bytes, expected {expected}", bytes.len()),
        });
    }
    Ok(Array2::from_shape_vec(expected_shape, bytes)?)
}

pub(crate) fn read_f64(path: &Path, role: &str, expected_shape: &[usize]) -> Result<Vec<f64>> {
    let (header, bytes) = read(path, role)?;
    validate_header(&header, role, "<f8", expected_shape)?;
    let length = expected_shape
        .iter()
        .try_fold(1_usize, |value, &dimension| value.checked_mul(dimension));
    let length = length.ok_or_else(|| Error::ShapeOverflow {
        shape: expected_shape.to_vec(),
    })?;
    let expected_bytes = length.checked_mul(8).ok_or_else(|| Error::ShapeOverflow {
        shape: expected_shape.to_vec(),
    })?;
    if bytes.len() != expected_bytes {
        return Err(Error::InvalidArrayShape {
            role: role.into(),
            reason: format!(
                "payload has {} bytes, expected {expected_bytes}",
                bytes.len()
            ),
        });
    }
    bytes
        .chunks_exact(8)
        .map(|value| {
            Ok(f64::from_le_bytes(value.try_into().map_err(|_| {
                Error::InvalidArrayShape {
                    role: role.into(),
                    reason: "truncated f64 element".into(),
                }
            })?))
        })
        .collect()
}

fn write_header(writer: &mut File, dtype: &str, shape: &[usize]) -> Result<()> {
    let shape = match shape {
        [length] => format!("({length},)"),
        values => format!(
            "({})",
            values
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}");
    let prefix = MAGIC.len() + 2 + 2;
    let padding = (16 - ((prefix + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let header_length = u16::try_from(header.len()).map_err(|_| Error::InvalidArrayShape {
        role: "npy header".into(),
        reason: "header is too long for NumPy format version 1".into(),
    })?;
    writer.write_all(MAGIC)?;
    writer.write_all(&[1, 0])?;
    writer.write_all(&header_length.to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    Ok(())
}

struct NpyHeader {
    dtype: String,
    shape: Vec<usize>,
    fortran_order: bool,
}

fn read(path: &Path, role: &str) -> Result<(NpyHeader, Vec<u8>)> {
    let mut file = File::open(path)?;
    let mut magic = [0_u8; 6];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::InvalidArrayDtype {
            role: role.into(),
            actual: "not a NumPy .npy file".into(),
            expected: "NumPy .npy".into(),
        });
    }
    let mut version = [0_u8; 2];
    file.read_exact(&mut version)?;
    let header_length = match version {
        [1, 0] => {
            let mut length = [0_u8; 2];
            file.read_exact(&mut length)?;
            usize::from(u16::from_le_bytes(length))
        }
        [2 | 3, 0] => {
            let mut length = [0_u8; 4];
            file.read_exact(&mut length)?;
            usize::try_from(u32::from_le_bytes(length)).map_err(|_| Error::InvalidArrayShape {
                role: role.into(),
                reason: "NumPy header length is not addressable".into(),
            })?
        }
        _ => {
            return Err(Error::InvalidArrayDtype {
                role: role.into(),
                actual: format!("unsupported NumPy format {}.{}", version[0], version[1]),
                expected: "NumPy format 1.0, 2.0, or 3.0".into(),
            });
        }
    };
    if header_length > 1_048_576 {
        return Err(Error::InvalidArrayShape {
            role: role.into(),
            reason: "NumPy header exceeds 1 MiB".into(),
        });
    }
    let mut header = vec![0_u8; header_length];
    file.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header).map_err(|_| Error::InvalidArrayDtype {
        role: role.into(),
        actual: "non-UTF-8 NumPy header".into(),
        expected: "valid NumPy header".into(),
    })?;
    let header = parse_header(header, role)?;
    let mut payload = Vec::new();
    file.read_to_end(&mut payload)?;
    Ok((header, payload))
}

fn parse_header(header: &str, role: &str) -> Result<NpyHeader> {
    let dtype = quoted_value(header, "'descr':")
        .ok_or_else(|| Error::InvalidArrayDtype {
            role: role.into(),
            actual: "missing descr".into(),
            expected: "NumPy dtype descriptor".into(),
        })?
        .to_owned();
    let fortran_order = if header.contains("'fortran_order': False") {
        false
    } else if header.contains("'fortran_order': True") {
        true
    } else {
        return Err(Error::InvalidArrayShape {
            role: role.into(),
            reason: "missing fortran_order".into(),
        });
    };
    let shape_start = header
        .find("'shape':")
        .and_then(|index| header[index..].find('(').map(|offset| index + offset))
        .ok_or_else(|| Error::InvalidArrayShape {
            role: role.into(),
            reason: "missing shape tuple".into(),
        })?;
    let shape_end = header[shape_start..]
        .find(')')
        .map(|offset| shape_start + offset)
        .ok_or_else(|| Error::InvalidArrayShape {
            role: role.into(),
            reason: "unterminated shape tuple".into(),
        })?;
    let mut shape = Vec::new();
    for value in header[shape_start + 1..shape_end].split(',') {
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        shape.push(value.parse().map_err(|_| Error::InvalidArrayShape {
            role: role.into(),
            reason: format!("invalid dimension `{value}`"),
        })?);
    }
    Ok(NpyHeader {
        dtype,
        shape,
        fortran_order,
    })
}

fn quoted_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let remainder = &header[header.find(key)? + key.len()..];
    let start = remainder.find('\'')? + 1;
    let end = remainder[start..].find('\'')? + start;
    Some(&remainder[start..end])
}

fn validate_header(
    header: &NpyHeader,
    role: &str,
    expected_dtype: &str,
    expected_shape: &[usize],
) -> Result<()> {
    if header.dtype != expected_dtype {
        return Err(Error::InvalidArrayDtype {
            role: role.into(),
            actual: header.dtype.clone(),
            expected: expected_dtype.into(),
        });
    }
    if header.fortran_order {
        return Err(Error::InvalidArrayShape {
            role: role.into(),
            reason: "Fortran-order arrays are unsupported at this strict boundary".into(),
        });
    }
    if header.shape != expected_shape {
        return Err(Error::InvalidArrayShape {
            role: role.into(),
            reason: format!(
                "file shape {:?}, manifest shape {:?}",
                header.shape, expected_shape
            ),
        });
    }
    Ok(())
}
